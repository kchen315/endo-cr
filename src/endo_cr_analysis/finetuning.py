"""Development-only comparison of frozen, partially unfrozen and fully unfrozen encoders."""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import inner_splits, new_output, validate_patients
from .images import load_encoder, read_manifest
from .evaluation import metric_values
from .delong import auc_summary


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class PatientModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, pooling: str, attention_dim: int = 64):
        super().__init__()
        if pooling not in {"mean", "gated", "probability"}:
            raise ValueError(pooling)
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = nn.Linear(feature_dim, 1)
        if pooling == "gated":
            self.attention_v = nn.Linear(feature_dim, attention_dim)
            self.attention_u = nn.Linear(feature_dim, attention_dim)
            self.attention_w = nn.Linear(attention_dim, 1, bias=False)

    def aggregate(self, features: torch.Tensor):
        # Pooling and BCE stay float32 even when encoder matmuls use bf16.
        features = features.float()
        mean = features.mean(0)
        if self.pooling == "gated":
            raw = self.attention_w(torch.tanh(self.attention_v(features)) *
                                   torch.sigmoid(self.attention_u(features))).squeeze(-1)
            attention = raw.softmax(0)
            pooled = (attention[:, None] * features).sum(0)
        else:
            attention = features.new_full((len(features),), 1 / len(features))
            pooled = mean
        if self.pooling == "probability":
            probability = torch.sigmoid(self.classifier(features).squeeze(-1)).mean()
            # Stable patient-level mean-probability BCE without per-image labels.
            probability = probability.clamp(1e-7, 1 - 1e-7)
            logit = torch.logit(probability)
        else:
            logit = self.classifier(pooled).squeeze(-1)
            probability = torch.sigmoid(logit)
        return logit, probability, mean, pooled, attention


def stage_for_epoch(job: dict, epoch: int) -> str:
    if job["regime"] == "frozen" or epoch < int(job.get("warmup", 5)):
        return "frozen"
    if job["regime"] == "full" and epoch >= int(job.get("warmup", 5)) + int(job.get("partial_epochs", 10)):
        return "full"
    return "partial"


def configure_stage(model: PatientModel, architecture: str, stage: str) -> list[str]:
    model.train()
    model.encoder.eval()  # Also disables dropout/stochastic depth in frozen layers.
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    base = model.encoder
    train_modules = []
    if stage == "full":
        train_modules = [base]
    elif stage == "partial":
        train_modules = ([*list(base.blocks)[-2:], base.norm] if architecture == "vit" else [base.layer4])
    for module in train_modules:
        # Keep stochastic operations off for a controlled freezing comparison.
        for param in module.parameters():
            param.requires_grad_(True)
    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return [n for n, p in model.named_parameters() if p.requires_grad]


def build_optimizer(model: PatientModel, stage: str, multiplier: float):
    buckets = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        lr = ((1e-6 if stage == "full" else 1e-5) if is_encoder
              else (1e-3 if stage == "frozen" else 1e-4)) * multiplier
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") else 1e-4
        buckets.setdefault((lr, decay), []).append(param)
    return torch.optim.AdamW([{"params": p, "lr": lr, "weight_decay": decay}
                              for (lr, decay), p in buckets.items()])


class ImageCache:
    """Bounded resized uint8 cache; no decoded original-size images are retained."""
    def __init__(self, frame: pd.DataFrame, size: int, mean, std, device: torch.device):
        self.bags = {}
        self.labels = {}
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        for pid, group in frame.groupby("patient_id", sort=False):
            frames = []
            for path in group.sort_values("path").path:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    width, height = im.size
                    scale = size / max(width, height)
                    im = im.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
                    square = Image.new("RGB", (size, size), (0, 0, 0))
                    square.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
                    frames.append(torch.from_numpy(np.asarray(square).copy()).permute(2, 0, 1))
            self.bags[pid] = torch.stack(frames)
            self.labels[pid] = int(group.label.iloc[0])

    def tensor(self, patient_id: str, augment: bool):
        x = self.bags[patient_id].to(self.device, dtype=torch.float32).div_(255)
        if augment:
            # Independent, fixed-range mild factors per frame. No geometry search.
            brightness = x.new_tensor([random.uniform(.95, 1.05) for _ in x]).view(-1, 1, 1, 1)
            contrast = x.new_tensor([random.uniform(.95, 1.05) for _ in x]).view(-1, 1, 1, 1)
            x = (x * brightness).clamp_(0, 1)
            gray = (x[:, 0] * .2989 + x[:, 1] * .587 + x[:, 2] * .114).mean((1, 2)).view(-1, 1, 1, 1)
            x = (x * contrast + gray * (1 - contrast)).clamp_(0, 1)
        return (x - self.mean) / self.std


def encoder_features(model: PatientModel, images, microbatch: int, amp: bool):
    pieces = []
    needs_grad = torch.is_grad_enabled() and any(p.requires_grad for p in model.encoder.parameters())
    for chunk in images.split(microbatch):
        with torch.set_grad_enabled(needs_grad), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=amp):
            pieces.append(model.encoder(chunk).float())
    return torch.cat(pieces)


def fixed_lr_readout(train_x, train_y, eval_x, seed, n_components=16, c=.1):
    components = min(int(n_components), train_x.shape[0] - 1, train_x.shape[1])
    if components < 1 or not np.isfinite(train_x).all() or not np.isfinite(eval_x).all():
        raise ValueError("Invalid representation for LR probe")
    estimator = Pipeline([("scale", StandardScaler()),
                          ("pca", PCA(n_components=components, svd_solver="full")),
                          ("lr", LogisticRegression(C=float(c), max_iter=5000, solver="lbfgs", random_state=seed))])
    estimator.fit(train_x, train_y)
    return estimator.predict_proba(eval_x)[:, 1], estimator


@torch.no_grad()
def extract(model, cache, ids, microbatch, amp):
    model.eval()
    means, pooled, probabilities, entropy = [], [], [], []
    for pid in ids:
        features = encoder_features(model, cache.tensor(pid, False), microbatch, amp)
        _, p, m, z, attention = model.aggregate(features)
        means.append(m.cpu().numpy())
        pooled.append(z.cpu().numpy())
        probabilities.append(float(p))
        entropy.append(float(-(attention * attention.clamp_min(1e-12).log()).sum()))
    return np.stack(means), np.stack(pooled), np.asarray(probabilities), np.asarray(entropy)


def fit_partition(name, regime, pooling, checkpoint, cache, train_ids, eval_ids, seed, device):
    seed_everything(seed)
    encoder, _ = load_encoder(name, checkpoint, device)
    model = PatientModel(encoder, 768 if name == 'dino' else 2048, pooling).to(device)
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    previous = None
    for epoch in range(25):
        stage = stage_for_epoch({'regime': regime}, epoch)
        configure_stage(model, 'vit' if name == 'dino' else 'rn50', stage)
        if stage != previous:
            optimizer = build_optimizer(model, stage, 1.)
            previous = stage
        order = np.random.permutation(len(train_ids))
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order):
            pid = train_ids[index]
            features = encoder_features(model, cache.tensor(pid, True), 4, amp)
            logit = model.aggregate(features)[0]
            loss = nn.functional.binary_cross_entropy_with_logits(logit, logit.new_tensor(float(cache.labels[pid])))
            divisor = min(8, len(order) - (step // 8)*8)
            (loss / divisor).backward()
            if (step+1) % 8 == 0 or step+1 == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    train_mean, train_pooled, _, _ = extract(model, cache, train_ids, 4, amp)
    test_mean, test_pooled, direct, _ = extract(model, cache, eval_ids, 4, amp)
    labels = np.array([cache.labels[k] for k in train_ids])
    probs = {'p_direct': direct, 'p_lr_mean': fixed_lr_readout(train_mean, labels, test_mean, seed)[0]}
    if pooling == 'gated':
        probs['p_lr_pooled'] = fixed_lr_readout(train_pooled, labels, test_pooled, seed)[0]
    return pd.DataFrame(probs, index=eval_ids)


def run(args):
    patients = pd.read_csv(args.patients, dtype={'patient_id': str})
    validate_patients(patients, [], development=True)
    if 'cohort' in patients and not patients.cohort.eq('development').all():
        raise ValueError("Fine-tuning uses development patients only")
    bags = read_manifest(args.manifest)
    if {p['patient_id'] for p in bags} != set(patients.patient_id):
        raise ValueError("The image manifest must cover exactly the development patients")
    labels = patients.set_index('patient_id').outcome
    frames = pd.DataFrame([(p['patient_id'], str(path), labels[p['patient_id']])
                           for p in bags for path in p['frames']], columns=['patient_id', 'path', 'label'])
    splits = pd.read_csv(args.inner_folds, dtype={'patient_id': str})
    torch.set_num_threads(4)
    seed_everything(args.seed)
    device = torch.device(args.device)
    out = new_output(args.output)
    predictions, selections = [], []
    for name in ['dino', 'rn50']:
        checkpoint = getattr(args, name)
        base_encoder, spec = load_encoder(name, checkpoint, 'cpu')
        del base_encoder
        cache = ImageCache(frames, *spec, device)
        for fold in range(1, 6):
            train = patients[patients.outer_fold.ne(fold)].reset_index(drop=True)
            test = patients.loc[patients.outer_fold.eq(fold), 'patient_id'].tolist()
            inner = inner_splits(train, splits, fold, 'clinical')
            ranking, refits = [], {}
            for regime in ['frozen', 'partial', 'full']:
                for pooling in ['mean', 'gated', 'probability']:
                    inner_predictions = []
                    for fitting, valid in inner:
                        inner_predictions.append(fit_partition(name, regime, pooling, checkpoint, cache,
                            train.iloc[fitting].patient_id.tolist(), train.iloc[valid].patient_id.tolist(),
                            args.seed, device))
                    scores = pd.concat(inner_predictions).loc[train.patient_id]
                    held = fit_partition(name, regime, pooling, checkpoint, cache, train.patient_id.tolist(),
                                         test, args.seed, device)
                    for readout in scores:
                        recipe = (regime, pooling, readout)
                        p = scores[readout]
                        ranking.append((brier_score_loss(train.outcome, p),
                                        log_loss(train.outcome, p, labels=[0, 1]), *recipe))
                        refits[recipe] = held[readout]
                        predictions.append(held[[readout]].rename(columns={readout: 'score'}).assign(
                            backbone=name, recipe='_'.join(recipe), outer_fold=fold))
            brier, loss, regime, pooling, readout = min(ranking)
            choice = (regime, pooling, readout)
            predictions.append(refits[choice].to_frame('score').assign(backbone=name, recipe='inner_selected',
                                                                       outer_fold=fold))
            selections.append(dict(backbone=name, outer_fold=fold, recipe='_'.join(choice), brier=brier, log_loss=loss))
            print(f"Finished {name}, outer fold {fold}", flush=True)
    held = pd.concat(predictions).rename_axis('patient_id').reset_index()
    paired = held.pivot(index=['patient_id', 'outer_fold', 'recipe'], columns='backbone', values='score')
    if paired.isna().any().any():
        raise ValueError("Ensemble requires both held-out scores")
    ensemble = paired.mean(axis=1).rename('score').reset_index().assign(backbone='equal_ensemble')
    held = pd.concat([held, ensemble], ignore_index=True)
    rows = []
    for (backbone, recipe), group in held.groupby(['backbone', 'recipe']):
        y = group.patient_id.map(labels)
        ci = auc_summary(y, group.score)
        rows.append(dict(backbone=backbone, recipe=recipe, **metric_values(y, group.score),
                         auroc_ci_low=ci['auroc_ci_low'], auroc_ci_high=ci['auroc_ci_high']))
    held.to_csv(out/'development_predictions.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'inner_selections.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'recipe_performance.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['patients', 'inner-folds', 'manifest', 'dino', 'rn50', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 2026, 2027], default=42)
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
