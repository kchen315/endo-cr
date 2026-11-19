"""Frozen GastroNet-5M encoders for prepared RGB endoscopy frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import timm
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import functional as TF

from .contracts import load_vision, new_output, sha256
from .interpretability import combine_registered_maps, token_attribution_for_view
from .vision import VIEWS, fit_mean_head, mean_features, subsample_mean_features

CHECKPOINTS = {
    'dino': '20ab749ec4bfdd53a79fc82784d04d9480d87d9f122b877d0f20c6f40aabdc44',
    'rn50': '5ae8e09cea634179e8caf95dbead1a86d7ce05d1e934c80c742b9ff5887f44dd',
}


def load_encoder(name, path, device):
    if sha256(path) != CHECKPOINTS[name]:
        raise ValueError(f"Expected the study's GastroNet-5M {name} pretrained checkpoint")
    state = torch.load(path, map_location='cpu', weights_only=True)
    if name == 'dino':
        size = 336
        model = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                  pretrained=False, num_classes=0, img_size=size)
        state = {k.removeprefix('backbone.'): v for k, v in state['teacher'].items()}
        config = timm.data.resolve_model_data_config(model)
        mean, std = config['mean'], config['std']
    else:
        size = 224
        model = resnet50(weights=None)
        model.fc = torch.nn.Identity()
        mean, std = (.485, .456, .406), (.229, .224, .225)
    model.load_state_dict({k: state[k] for k in model.state_dict()}, strict=True)
    return model.requires_grad_(False).eval().to(device), (size, mean, std)


def prepared_view(image, view, spec):
    size, mean, std = spec
    width, height = image.size
    crop = (0, 0, width, height)
    if view == 'central_crop_0p95':
        w, h = max(1, round(.95*width)), max(1, round(.95*height))
        x, y = (width-w)//2, (height-h)//2
        crop = (x, y, x+w, y+h)
        image = image.crop(crop)
    elif view != 'identity':
        op, factor = view.split('_')
        enhance = {'brightness': ImageEnhance.Brightness, 'contrast': ImageEnhance.Contrast}[op]
        image = enhance(image).enhance(float(factor.replace('p', '.')))
    scale = size / max(image.size)
    w, h = [max(1, round(v*scale)) for v in image.size]
    x, y = (size-w)//2, (size-h)//2
    square = Image.new('RGB', (size, size))
    square.paste(image.resize((w, h), Image.Resampling.BILINEAR), (x, y))
    return TF.normalize(TF.to_tensor(square), mean, std), crop, (x, y, x+w, y+h)


def read_manifest(path):
    patients = json.loads(path.read_text())
    seen_ids, seen_frames = set(), set()
    if not isinstance(patients, list) or not patients:
        raise ValueError("Supply one nonempty JSON list of patient bags")
    for patient in patients:
        key = patient['patient_id']
        frames = [(path.parent / s).resolve() for s in patient['frames']]
        if not isinstance(key, str) or not key.strip() or key in seen_ids:
            raise ValueError("Patient pseudonyms must be unique nonempty strings")
        if not frames or len(set(frames)) != len(frames) or seen_frames.intersection(frames):
            raise ValueError("Every prepared frame belongs to exactly one location in one bag")
        patient['frames'] = frames
        patient['subsets'] = np.asarray(patient['subsets'])
        seen_ids.add(key)
        seen_frames.update(frames)
    return patients


def frame_features(paths, model, spec, device, batch_size=None):
    banks = []
    with torch.inference_mode():
        for view in VIEWS:
            width = batch_size or (32 if view == 'identity' else 8)
            batches = []
            for start in range(0, len(paths), width):
                tensors = []
                for path in paths[start:start + width]:
                    with Image.open(path) as image:
                        tensors.append(prepared_view(image.convert('RGB'), view, spec)[0])
                batch = torch.stack(tensors).to(device)
                batches.append(model(batch).cpu().numpy())
            banks.append(np.concatenate(batches))
    return np.stack(banks)


def encode(args, patients):
    arrays = {'patient_ids': np.array([p['patient_id'] for p in patients])}
    # Bag order determines the subsamples. Extraction order determines GPU batch shapes.
    paths = [path for patient in patients for path in patient['frames']]
    if any('extraction_indices' in p for p in patients):
        indices = np.concatenate([np.asarray(p['extraction_indices']) for p in patients])
        if (any(len(p['frames']) != len(p['extraction_indices']) for p in patients)
                or indices.dtype.kind not in 'iu' or set(indices) != set(range(len(paths)))):
            raise ValueError("Extraction indices must be a permutation of the cohort frames")
    else:
        indices = np.arange(len(paths))
    paths = [paths[i] for i in np.argsort(indices)]
    boundaries = np.cumsum([len(p['frames']) for p in patients])[:-1]
    subsets = [p['subsets'] for p in patients]
    for name in CHECKPOINTS:
        model, spec = load_encoder(name, getattr(args, name), args.device)
        bank = frame_features(paths, model, spec, args.device)
        views = [np.split(view[indices], boundaries) for view in bank]
        arrays[name+'_identity'] = mean_features(views[0])
        arrays[name+'_application'] = np.stack([subsample_mean_features(bags, subsets) for bags in views])
    np.savez(new_output(args.output) / 'embeddings.npz', **arrays)


def source_map(grid, size, raw_size, crop, padding):
    def resize(a, shape):
        return F.interpolate(a, size=shape, mode='bilinear', align_corners=False)
    square = resize(torch.from_numpy(grid)[None, None], (size, size))
    x, y, right, bottom = padding
    x0, y0, x1, y1 = crop
    result = np.zeros(raw_size[::-1], dtype=np.float32)
    result[y0:y1, x0:x1] = resize(square[:, :, y:bottom, x:right], (y1-y0, x1-x0))[0, 0].numpy()
    return result


def sensitivity_maps(args, patients):
    dev = pd.read_csv(args.patients, dtype={'patient_id': str})
    if set(dev.outer_fold) != set(range(1, 6)) or not dev.outcome.isin([0, 1]).all():
        raise ValueError("Supply development patients with their original outer folds and outcomes")
    means, _ = load_vision(args.vision, dev.patient_id)
    cases = pd.read_csv(args.panels, dtype={'patient_id': str}).sort_values('panel')
    if (cases.panel.tolist() != list(range(1, len(cases)+1)) or cases.patient_id.duplicated().any()
            or not set(cases.patient_id) <= set(dev.patient_id)):
        raise ValueError("Supply unique selected development patients in numbered panel order")
    bags = {p['patient_id']: p for p in patients}
    positions = dev.reset_index(drop=True).set_index('patient_id')
    vit, spec = load_encoder('dino', args.dino, args.device)
    directory = new_output(args.output)
    for case in cases.itertuples():
        bag = bags[case.patient_id]
        selected = positions.loc[case.patient_id]
        training = dev.outer_fold.ne(selected.outer_fold)
        head = fit_mean_head(means['dino'][training], dev.loc[training, 'outcome'])
        bank = frame_features(bag['frames'], vit, spec, args.device, batch_size=1)
        expected = means['dino'][dev.patient_id.eq(case.patient_id)][0]
        if not np.allclose(expected, bank[0].mean(axis=0), rtol=1e-4, atol=2e-4):
            raise ValueError("Selected bag does not match its development embeddings")
        if not float(case.frame_number).is_integer() or not 1 <= case.frame_number <= bank.shape[1]:
            raise ValueError("Panel frame_number must be a one-based index within the selected bag")
        frame = int(case.frame_number) - 1
        maps = []
        with Image.open(bag['frames'][frame]) as source:
            image = source.convert('RGB')
        for view, stored in zip(VIEWS, bank):
            tensor, crop, padding = prepared_view(image, view, spec)
            result = token_attribution_for_view(encoder=vit, target_layer=vit.blocks[-1].norm1,
                prepared_tensor=tensor[None].to(args.device), head=head, stored_embeddings=stored,
                selected_frame=frame, subsets=bag['subsets'], grid_size=tuple(vit.patch_embed.grid_size),
                target_label=selected.outcome)
            maps.append(source_map(result['signed_token_map'], spec[0], image.size, crop, padding))
        np.savez(directory / f'panel_{case.panel:02d}.npz', frame_index=frame,
                 **combine_registered_maps(np.stack(maps)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest='command', required=True)
    enc = cmds.add_parser('encode')
    enc.add_argument('--rn50', type=Path, required=True)
    maps = cmds.add_parser('maps')
    for name in ['patients', 'vision', 'panels']:
        maps.add_argument('--'+name, type=Path, required=True)
    for command in [enc, maps]:
        for name in ['manifest', 'dino', 'output']:
            command.add_argument('--'+name, type=Path, required=True)
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    patients = read_manifest(args.manifest)
    if args.command == 'encode':
        encode(args, patients)
    else:
        sensitivity_maps(args, patients)


if __name__ == '__main__':
    main()
