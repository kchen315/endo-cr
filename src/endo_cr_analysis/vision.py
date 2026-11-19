"""Frozen patient-level vision analysis from private, analysis-ready embeddings.

This module deliberately does not load, curate, redact, crop, or encode images.
Six view-specific embedding banks and the locked frame-subsample indices are
inputs. Keep those arrays and indices private, like other patient-level data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BACKBONES = ("dino", "rn50")
VIEWS = (
    "identity",
    "brightness_0p95",
    "brightness_1p05",
    "contrast_0p95",
    "contrast_1p05",
    "central_crop_0p95",
)
SUBSAMPLES = 16
SUBSAMPLE_FRACTION = 0.75
PCA_COMPONENTS = 16
LOGISTIC_C = 0.1


def _matrix(values: np.ndarray, context: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{context} must be a finite, nonempty two-dimensional array")
    return array


@dataclass
class MeanHead:
    """Training-contained scaler, PCA, and fixed ridge-logistic classifier."""

    scaler: StandardScaler
    pca: PCA
    classifier: LogisticRegression

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return the positive-class probability, one value per patient."""
        matrix = _matrix(features, "Patient mean embeddings")
        reduced = self.pca.transform(self.scaler.transform(matrix))
        positive_column = int(np.flatnonzero(self.classifier.classes_ == 1)[0])
        return self.classifier.predict_proba(reduced)[:, positive_column].astype(float)


def fit_mean_head(identity_means: np.ndarray, labels: np.ndarray) -> MeanHead:
    """Fit the frozen head using identity/full-bag training features only.

    Call separately inside every relevant outer- or inner-training partition.
    Augmented/subsampled features are application inputs, not extra training
    observations. The patient remains the unit of fitting and prediction.
    """
    features = _matrix(identity_means, "Training mean embeddings")
    labels = np.asarray(labels)
    if labels.shape != (len(features),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must be aligned binary patient outcomes containing both classes")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    components = min(PCA_COMPONENTS, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=components, random_state=42)
    reduced = pca.fit_transform(scaled)
    classifier = LogisticRegression(
        C=LOGISTIC_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    classifier.fit(reduced, labels.astype(int))
    return MeanHead(scaler, pca, classifier)


def fit_vision_heads(
    identity_means: Mapping[str, np.ndarray], labels: np.ndarray
) -> dict[str, MeanHead]:
    """Fit one independent frozen head per backbone; do not pool embeddings."""
    if set(identity_means) != set(BACKBONES):
        raise ValueError(f"Expected exactly these backbones: {BACKBONES}")
    return {name: fit_mean_head(identity_means[name], labels) for name in BACKBONES}


def mean_features(bags: Sequence[np.ndarray]) -> np.ndarray:
    """Order-invariant full-bag mean, retaining the original float32 arithmetic."""
    if not len(bags):
        raise ValueError("At least one patient bag is required")
    checked = [_matrix(bag, "Patient frame embeddings").astype(np.float32) for bag in bags]
    if len({array.shape[1] for array in checked}) != 1:
        raise ValueError("Embedding dimensions must agree within a backbone")
    return np.stack([array.mean(axis=0) for array in checked]).astype(np.float32)


def subsample_mean_features(
    bags: Sequence[np.ndarray], subsample_indices: Sequence[np.ndarray]
) -> np.ndarray:
    """Return a 16 x patient x dimension tensor using supplied private indices.

    Each patient's index array has shape (16, ceil(0.75 * frame_count)).
    Rows must contain sorted, unique zero-based indices. The same draws must
    be used across both backbones and all views. Supplying the original draws
    avoids changing predictions when research identifiers are deidentified.
    """
    if not len(bags) or len(bags) != len(subsample_indices):
        raise ValueError("One index array is required for each nonempty patient bag")
    patient_values = []
    dimensions = set()
    for bag, indices in zip(bags, subsample_indices):
        array = _matrix(bag, "Patient frame embeddings").astype(np.float32)
        dimensions.add(array.shape[1])
        selected = np.asarray(indices)
        keep = max(1, int(np.ceil(SUBSAMPLE_FRACTION * len(array))))
        if selected.shape != (SUBSAMPLES, keep) or selected.dtype.kind not in "iu":
            raise ValueError("Each subsample must contain exactly ceil(0.75 * frames) integer indices")
        if np.any(selected < 0) or np.any(selected >= len(array)):
            raise ValueError("Subsample index is outside its patient bag")
        if keep > 1 and np.any(np.diff(selected, axis=1) <= 0):
            raise ValueError("Subsample indices must be sorted and unique within each draw")
        patient_values.append(np.stack([array[draw].mean(axis=0) for draw in selected]))
    if len(dimensions) != 1:
        raise ValueError("Embedding dimensions must agree within a backbone")
    return np.stack(patient_values, axis=1).astype(np.float32)


def prepare_vision_features(
    banks: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    subsample_indices: Sequence[np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Pool prepared banks into full-bag fit features and 6x16xNxD test features.

    The caller must align patient and frame order across every bank before
    calling this routine; dimensions/counts alone cannot establish identity.
    """
    if set(banks) != set(BACKBONES):
        raise ValueError(f"Expected exactly these backbones: {BACKBONES}")
    identity_means: dict[str, np.ndarray] = {}
    application_features: dict[str, np.ndarray] = {}
    reference_counts: tuple[int, ...] | None = None
    for backbone in BACKBONES:
        if set(banks[backbone]) != set(VIEWS):
            raise ValueError(f"Backbone {backbone} requires all six locked views")
        view_features = []
        for view in VIEWS:
            bags = banks[backbone][view]
            counts = tuple(len(bag) for bag in bags)
            if reference_counts is None:
                reference_counts = counts
            elif counts != reference_counts:
                raise ValueError("Patient frame counts must agree across all banks and views")
            if view == "identity":
                identity_means[backbone] = mean_features(bags)
            view_features.append(subsample_mean_features(bags, subsample_indices))
        application_features[backbone] = np.stack(view_features)
    return identity_means, application_features


def summarize_prediction_tensor(predictions: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize 2-backbone x 6-view x 16-subsample x patient probabilities.

    Variances are descriptive population variances (ddof=0), not confidence
    intervals. Augmentation uncertainty averages over backbone/subsample;
    subsampling uncertainty averages over backbone/view. The final fusion
    uses p_dino only, whereas standalone vision uses the equal average.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 4 or values.shape[:3] != (2, len(VIEWS), SUBSAMPLES):
        raise ValueError("Expected a 2 x 6 x 16 x patient probability tensor")
    if values.shape[3] < 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Predictions must be nonempty, finite probabilities")
    backbone_means = values.mean(axis=(1, 2))
    augmentation_draws = values.mean(axis=(0, 2))
    subsample_draws = values.mean(axis=(0, 1))
    disagreement = np.abs(backbone_means[0] - backbone_means[1])
    return {
        "p_dino": backbone_means[0],
        "p_rn50": backbone_means[1],
        "p_vision": subsample_draws.mean(axis=0),
        "augmentation_variance": augmentation_draws.var(axis=0, ddof=0),
        "subsample_variance": subsample_draws.var(axis=0, ddof=0),
        "backbone_disagreement": disagreement,
        "backbone_prediction_variance": disagreement**2 / 4.0,
    }


def predict_vision_ensemble(
    heads: Mapping[str, MeanHead], application_features: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Apply fixed heads and average probabilities, not embeddings or logits."""
    if set(heads) != set(BACKBONES) or set(application_features) != set(BACKBONES):
        raise ValueError(f"Expected exactly these backbones: {BACKBONES}")
    values = []
    patient_count = None
    for backbone in BACKBONES:
        features = np.asarray(application_features[backbone])
        if features.ndim != 4 or features.shape[:2] != (len(VIEWS), SUBSAMPLES):
            raise ValueError("Application features must have shape 6 x 16 x patient x dimension")
        if min(features.shape[2:]) < 1 or not np.isfinite(features).all():
            raise ValueError("Application features must be finite and nonempty")
        if patient_count is None:
            patient_count = features.shape[2]
        elif features.shape[2] != patient_count:
            raise ValueError("Backbone patient counts do not agree")
        # Preserve the original application batch and summation order.
        values.append(np.stack([
            np.stack([heads[backbone].predict_proba(draw) for draw in view])
            for view in features
        ]))
    return summarize_prediction_tensor(np.stack(values))
