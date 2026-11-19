"""Pure analytic core of the DINO vision attribution sensitivity analysis.

The target is the deployed DINO probability, averaged over six fixed views and
sixteen fixed frame subsamples, or its negative for an observed nonresponse.
This is activation-times-gradient sensitivity, not attention, a causal effect,
or an explanation of the full clinical-vision fusion model.

Inputs are private prepared tensors/embeddings and a caller-owned local encoder.
There is no image I/O, image processing, case selection, encoder download, or
privacy-redaction implementation here. The image command handles view registration using the supplied panel manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .vision import SUBSAMPLES, SUBSAMPLE_FRACTION, VIEWS


def _head_parts(head: Any) -> tuple[Any, Any, Any]:
    parts = tuple(head[key] if isinstance(head, Mapping) else getattr(head, key)
                  for key in ("scaler", "pca", "classifier"))
    if getattr(parts[1], "whiten", False):
        raise ValueError("The locked PCA does not use whitening")
    if not np.array_equal(parts[2].classes_, [0, 1]):
        raise ValueError("The locked head requires class order [0, 1]")
    return parts


def effective_raw_weight(head: Any) -> np.ndarray:
    """Weight of each unstandardized embedding dimension in the head logit."""
    scaler, pca, classifier = _head_parts(head)
    return ((classifier.coef_[0] @ pca.components_) / scaler.scale_).astype(np.float64)


def _probabilities(head: Any, means: np.ndarray) -> np.ndarray:
    scaler, pca, classifier = _head_parts(head)
    return classifier.predict_proba(pca.transform(scaler.transform(np.asarray(means, dtype=float))))[:, 1]


def _subsets(subsets: np.ndarray, frames: int) -> np.ndarray:
    indices = np.asarray(subsets)
    count = max(1, int(np.ceil(SUBSAMPLE_FRACTION * frames)))
    if indices.shape != (SUBSAMPLES, count) or indices.dtype.kind not in "iu":
        raise ValueError("Supply the 16 locked, 75%-of-frames subsamples")
    if np.any(indices < 0) or np.any(indices >= frames):
        raise ValueError("Subsample indices fall outside this patient bag")
    if count > 1 and np.any(np.diff(indices, axis=1) <= 0):
        raise ValueError("Subsample indices must be sorted and unique within each draw")
    return indices


def _label_sign(target_label: int) -> float:
    if target_label not in (0, 1):
        raise ValueError("The observed-class target must be 0 or 1")
    return 1.0 if target_label == 1 else -1.0


def patient_embedding_gradients(head: Any, embeddings_by_view: np.ndarray,
                                subsets: np.ndarray, target_label: int = 1) -> dict[str, Any]:
    """Analytic observed-class probability derivative for every frame/view.

    ``embeddings_by_view`` has shape ``(6, frames, embedding_dimensions)`` in
    the fixed view order. ``subsets`` has shape ``(16, ceil(.75 * frames))``.
    The returned gradient includes both averaging denominators. Do not average
    those gradients a second time when combining registered spatial maps.
    """
    embeddings = np.asarray(embeddings_by_view)
    if (embeddings.ndim != 3 or embeddings.shape[0] != len(VIEWS)
            or min(embeddings.shape[1:]) < 1 or not np.isfinite(embeddings).all()):
        raise ValueError("Expected finite six-view patient frame embeddings")
    indices = _subsets(subsets, embeddings.shape[1])
    means = embeddings[:, indices].mean(axis=2)
    probabilities = np.stack([_probabilities(head, values) for values in means])
    weights = effective_raw_weight(head)
    if weights.shape != (embeddings.shape[2],):
        raise ValueError("Embedding dimensions differ from the fitted head")
    gradient = np.zeros(embeddings.shape, dtype=float)
    sign = _label_sign(target_label)
    denominator = float(len(VIEWS) * SUBSAMPLES * indices.shape[1])
    for view in range(len(VIEWS)):
        for draw, selected in enumerate(indices):
            p = probabilities[view, draw]
            gradient[view, selected] += sign * p * (1 - p) * weights / denominator
    return {"dino_probability": float(probabilities.mean()),
            "view_subsample_probabilities": probabilities,
            "embedding_gradients": gradient,
            "embedding_times_gradient": gradient * embeddings}


def differentiable_logit(head: Any, feature: Any) -> Any:
    """Torch version of scaler → PCA → logistic head, preserving autograd."""
    import torch

    scaler, pca, classifier = _head_parts(head)
    def tensor(value):
        return torch.as_tensor(value, dtype=feature.dtype, device=feature.device)
    standardized = (feature - tensor(scaler.mean_)) / tensor(scaler.scale_)
    reduced = (standardized - tensor(pca.mean_)) @ tensor(pca.components_).T
    return reduced @ tensor(classifier.coef_[0]) + tensor(classifier.intercept_[0])


def differentiable_view_probability(head: Any, embedding: Any, stored_embeddings: np.ndarray,
                                    selected_frame: int, subsets: np.ndarray) -> Any:
    """Replace one cached embedding differentiably; other frames stay fixed.

    This reproduces the original constant-mean plus single-frame correction,
    including the cached mean arithmetic. The caller verifies encoder/cache
    agreement before interpreting token maps.
    """
    import torch

    stored = np.asarray(stored_embeddings)
    if stored.ndim != 2 or not np.isfinite(stored).all() or min(stored.shape) < 1:
        raise ValueError("Expected finite cached frame embeddings for one view")
    if selected_frame < 0 or selected_frame >= len(stored) or tuple(embedding.shape) != (stored.shape[1],):
        raise ValueError("Selected frame or differentiable embedding shape is invalid")
    indices = _subsets(subsets, len(stored))
    def tensor(value):
        return torch.as_tensor(value, dtype=embedding.dtype, device=embedding.device)
    expected = tensor(stored[selected_frame])
    probabilities = []
    for selected in indices:
        mean = tensor(stored[selected].mean(axis=0))
        if selected_frame in selected:
            mean = mean + (embedding - expected) / float(len(selected))
        probabilities.append(torch.sigmoid(differentiable_logit(head, mean)))
    return torch.stack(probabilities).mean()


def token_attribution_for_view(*, encoder: Any, target_layer: Any, prepared_tensor: Any,
                               head: Any, stored_embeddings: np.ndarray, selected_frame: int,
                               subsets: np.ndarray, grid_size: tuple[int, int],
                               target_label: int = 1, embedding_tolerance: float = 2e-4) -> dict[str, Any]:
    """Signed patch activation×gradient for one already-prepared image view.

    Supply the frozen encoder in evaluation mode and its last-block ``norm1``
    target layer. The locked DINO model has one prefix token and no register
    tokens. A single-image tensor and explicit patch grid are required. The
    returned token grid is not warped, resized, normalized, or composited.

    Autograd computes only the requested activation derivative and does not
    accumulate gradients on encoder weights. Torch is an optional dependency.
    """
    import torch

    if getattr(encoder, "training", False):
        raise ValueError("The frozen local encoder must be in evaluation mode")
    if prepared_tensor.ndim < 2 or prepared_tensor.shape[0] != 1:
        raise ValueError("Supply one already-prepared image tensor with a batch dimension")
    if (len(grid_size) != 2 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in grid_size)
            or not np.isfinite(embedding_tolerance) or embedding_tolerance < 0):
        raise ValueError("Invalid token grid or encoder agreement tolerance")
    sign = _label_sign(target_label)
    captured = []
    handle = target_layer.register_forward_hook(lambda _layer, _args, value: captured.append(value))
    try:
        with torch.enable_grad():
            model_input = prepared_tensor.detach().clone().requires_grad_(True)
            embedding = encoder(model_input).squeeze(0)
            if len(captured) != 1 or captured[0].ndim != 3 or captured[0].shape[0] != 1:
                raise ValueError("Expected exactly one batch of token activations at the target layer")
            view_probability = differentiable_view_probability(head, embedding, stored_embeddings,
                                                                selected_frame, subsets)
            expected = torch.as_tensor(stored_embeddings[selected_frame], dtype=embedding.dtype,
                                       device=embedding.device)
            difference = torch.abs(embedding.detach() - expected)
            error = float(difference.max().cpu())
            if not np.isfinite(error) or error > embedding_tolerance:
                raise ValueError("Local encoder output does not reproduce the cached selected-frame embedding")
            target = sign * view_probability / float(len(VIEWS))
            if target.requires_grad:
                gradient = torch.autograd.grad(target, captured[0], allow_unused=False)[0]
            else:
                gradient = torch.zeros_like(captured[0])
            score = (captured[0].detach()[:, 1:] * gradient.detach()[:, 1:]).sum(dim=-1)
            if score.numel() != int(np.prod(grid_size)):
                raise ValueError("Token count differs from the supplied grid plus one prefix token")
            token_map = score.reshape(grid_size).cpu().numpy()
            if not np.isfinite(token_map).all():
                raise ValueError("Nonfinite token attribution")
            return {"signed_token_map": token_map, "view_probability": float(view_probability.detach().cpu()),
                    "embedding_maximum_absolute_error": error,
                    "embedding_mean_absolute_error": float(difference.mean().cpu()),
                    "view_gradient_denominator": len(VIEWS)}
    finally:
        handle.remove()


def combine_registered_maps(registered_view_maps: np.ndarray) -> dict[str, np.ndarray]:
    """Sum six already-registered signed maps, then normalize positive evidence.

    Registration and zero filling outside view support are caller responsibilities.
    There is intentionally no position-dependent averaging or geometric editing.
    """
    maps = np.asarray(registered_view_maps)
    if maps.ndim != 3 or maps.shape[0] != len(VIEWS) or min(maps.shape[1:]) < 1 or not np.isfinite(maps).all():
        raise ValueError("Supply six finite signed maps registered to the same spatial grid")
    signed = maps.sum(axis=0).astype(np.float32)
    positive = np.maximum(signed, 0).astype(np.float32)
    if positive.max() <= 1e-12:
        raise ValueError("The selected case contains no positive observed-class sensitivity")
    return {"signed_sum": signed, "positive_normalized": positive / positive.max()}
