"""Frozen single-CatBoost clinical branch, fitted only on analysis-ready tables.

This module deliberately contains no cohort construction, data extraction, or
image-curation code. Callers supply a fixed feature contract and patient-level
inner splits. The locked preprocessing and calibration rules are reproduced;
no model or hyperparameter search is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


CATBOOST_PARAMS = {
    "iterations": 100,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3,
    "thread_count": 2,
}
MISSINGNESS_THRESHOLD = 0.95
NUMERIC_COVERAGE_THRESHOLD = 0.95


def _logit(values: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -50, 50)))


def clean_category_value(value: Any) -> str:
    """Preserve categories, representing absent values by the locked token."""
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    return "Missing" if not text or text.lower() == "nan" else text


def _feature_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("A nonempty, unique feature contract is required")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing {len(missing)} required feature column(s)")
    return frame.loc[:, list(columns)].copy()


@dataclass
class ClinicalPreprocessor:
    """Training-contained feature typing, filtering, and median imputation."""

    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: pd.Series

    @property
    def retained_features(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns

    @property
    def categorical_indices(self) -> list[int]:
        return list(range(len(self.numeric_columns), len(self.retained_features)))

    @classmethod
    def fit(cls, train: pd.DataFrame, feature_columns: Sequence[str]) -> "ClinicalPreprocessor":
        if not len(train):
            raise ValueError("Cannot fit preprocessing on an empty training partition")
        frame = _feature_frame(train, feature_columns)
        numeric, categorical = [], []
        for name in frame.columns:
            series = frame[name]
            if pd.api.types.is_bool_dtype(series):
                frame[name] = series.astype(float)
                numeric.append(name)
            elif pd.api.types.is_numeric_dtype(series):
                frame[name] = pd.to_numeric(series, errors="coerce")
                numeric.append(name)
            elif not len(series.dropna()):
                frame[name] = series.astype(object)
                categorical.append(name)
            else:
                coerced = pd.to_numeric(series, errors="coerce")
                coverage = coerced.notna().sum() / max(1, len(series.dropna()))
                if coverage >= NUMERIC_COVERAGE_THRESHOLD:
                    frame[name] = coerced.astype(float)
                    numeric.append(name)
                else:
                    frame[name] = series.astype(object)
                    categorical.append(name)
        keep = [
            name for name in frame.columns
            if frame[name].isna().mean() <= MISSINGNESS_THRESHOLD
            and frame[name].dropna().nunique() > 1
        ]
        if not keep:
            raise ValueError("No varying clinical features survived training-only filtering")
        numeric = [name for name in numeric if name in keep]
        categorical = [name for name in categorical if name in keep]
        medians = frame.loc[:, numeric].median(axis=0)
        return cls(list(feature_columns), numeric, categorical, medians)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = _feature_frame(frame, self.feature_columns)
        parts = []
        if self.numeric_columns:
            numeric = selected.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.medians)
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError("Numeric features contain nonfinite values after training-only imputation")
            parts.append(numeric)
        if self.categorical_columns:
            categorical = selected.loc[:, self.categorical_columns].copy()
            for name in self.categorical_columns:
                categorical[name] = categorical[name].map(clean_category_value)
            parts.append(categorical)
        return pd.concat(parts, axis=1)


def calibration_affine(calibration: dict[str, Any]) -> tuple[float, float]:
    method = calibration["method"]
    if method == "none":
        return 1.0, 0.0
    if method == "temperature":
        temperature = float(calibration["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        return 1 / temperature, 0.0
    if method == "platt":
        return float(calibration["platt_coef"]), float(calibration["platt_intercept"])
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibration(probabilities: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if calibration["method"] == "none":
        return raw
    slope, intercept = calibration_affine(calibration)
    return _sigmoid(slope * _logit(raw) + intercept)


def choose_calibration(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Select none/temperature/Platt using inner-OOF Brier and ranking guards.

    Calibrators are fitted to inner-OOF predictions within the outer-training
    partition. This is calibration selection, not additional model selection.
    """
    raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels)
    if raw.ndim != 1 or raw.shape != labels.shape or not np.isfinite(raw).all():
        raise ValueError("Calibration requires paired finite one-dimensional inputs")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires both binary outcome classes")
    logits = _logit(raw)

    def objective(log_temperature: np.ndarray) -> float:
        temperature = np.exp(log_temperature[0])
        p = np.clip(_sigmoid(logits / temperature), 1e-6, 1 - 1e-6)
        return float(np.mean(-(labels * np.log(p) + (1 - labels) * np.log(1 - p))))

    result = minimize(objective, x0=np.array([0.0]), method="L-BFGS-B")
    temperature = float(np.exp(result.x[0])) if result.success else 1.0
    platt = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=42)
    platt.fit(logits.reshape(-1, 1), labels)
    candidates = [
        {"method": "none", "temperature": 1.0, "platt_coef": None, "platt_intercept": None},
        {"method": "temperature", "temperature": temperature, "platt_coef": None, "platt_intercept": None},
        {"method": "platt", "temperature": None,
         "platt_coef": float(platt.coef_[0, 0]), "platt_intercept": float(platt.intercept_[0])},
    ]
    for candidate in candidates:
        p = apply_calibration(raw, candidate)
        candidate.update(brier=float(brier_score_loss(labels, p)),
                         auroc=float(roc_auc_score(labels, p)),
                         auprc=float(average_precision_score(labels, p)))
    raw_row = candidates[0]
    admissible = [row for row in candidates
                  if row["auroc"] >= raw_row["auroc"] - 1e-4
                  and row["auprc"] >= raw_row["auprc"] - 1e-4]
    return min(admissible, key=lambda row: (row["brier"], -row["auroc"], -row["auprc"]))


@dataclass
class ClinicalModel:
    estimator: CatBoostClassifier
    preprocessor: ClinicalPreprocessor
    calibration: dict[str, Any] = field(default_factory=lambda: {"method": "none"})
    inner_oof_raw: np.ndarray | None = None
    random_state: int = 42

    def raw_margin(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.preprocessor.transform(frame),
                                                prediction_type="RawFormulaVal"), dtype=float)

    def raw_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.preprocessor.transform(frame))[:, 1].astype(float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return one positive-class probability per patient (one-dimensional)."""
        return apply_calibration(self.raw_probability(frame), self.calibration)

    def shap(self, frame: pd.DataFrame, calibrated: bool = True) -> dict[str, Any]:
        """Held-out native TreeSHAP on raw or affinely calibrated logit scale.

        No outcomes are needed. Raises if probability clipping breaks the affine
        identity, instead of silently claiming an exact attribution.
        """
        transformed = self.preprocessor.transform(frame)
        pool = Pool(transformed, cat_features=self.preprocessor.categorical_indices)
        values = np.asarray(self.estimator.get_feature_importance(pool, type="ShapValues"), dtype=float)
        contributions, baseline = values[:, :-1], values[:, -1]
        margin = self.raw_margin(frame)
        if calibrated:
            slope, intercept = calibration_affine(self.calibration)
            contributions = slope * contributions
            baseline = slope * baseline + intercept
            margin = _logit(self.predict_proba(frame))
        error = float(np.max(np.abs(contributions.sum(axis=1) + baseline - margin)))
        if error > 1e-8:
            raise ValueError("SHAP additivity failed (check saturated probability clipping)")
        return {"feature_names": self.preprocessor.retained_features,
                "contributions": contributions, "baseline": baseline, "margin": margin,
                "reconstruction_error": error}


def _labels(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target not in frame:
        raise ValueError("The training outcome column is missing")
    values = frame[target].to_numpy()
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("Training requires both binary outcome classes without missing labels")
    return values.astype(int)


def fit_raw_clinical(train_df: pd.DataFrame, feature_columns: Sequence[str], *,
                     random_state: int = 42, target: str = "outcome") -> ClinicalModel:
    """Fit the fixed clinical model without calibration (for nested fusion)."""
    if target in feature_columns:
        raise ValueError("Outcome cannot be part of the feature contract")
    labels = _labels(train_df, target)
    preprocessor = ClinicalPreprocessor.fit(train_df, feature_columns)
    class_weights = [1.0, float(np.sum(labels == 0) / np.sum(labels == 1))]
    estimator = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", random_seed=int(random_state),
        verbose=False, allow_writing_files=False, class_weights=class_weights,
        **CATBOOST_PARAMS,
    )
    estimator.fit(preprocessor.transform(train_df), labels,
                  cat_features=preprocessor.categorical_indices)
    return ClinicalModel(estimator, preprocessor, random_state=int(random_state))


def validate_inner_splits(n: int, splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Require nonoverlapping train/validation indices and exact OOF coverage."""
    if len(splits) < 2:
        raise ValueError("At least two supplied inner folds are required")
    coverage = np.zeros(n, dtype=int)
    for train_idx, validation_idx in splits:
        for indices in (train_idx, validation_idx):
            indices = np.asarray(indices)
            if (indices.ndim != 1 or not len(indices)
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= n)
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("Inner splits must contain unique in-range integer indices")
        if np.intersect1d(train_idx, validation_idx).size:
            raise ValueError("Inner training and validation patients overlap")
        if len(train_idx) + len(validation_idx) != n:
            raise ValueError("Each inner split must partition all outer-training patients")
        coverage[validation_idx] += 1
    if not np.all(coverage == 1):
        raise ValueError("Every outer-training patient must receive exactly one inner-OOF prediction")


def crossfit_raw_clinical(train_df: pd.DataFrame, feature_columns: Sequence[str],
                          inner_splits: Sequence[tuple[np.ndarray, np.ndarray]], *,
                          base_seed: int = 242, target: str = "outcome",
                          return_margin: bool = False, split_seed_step: int = 1) -> np.ndarray:
    """Fit each inner model with ``base_seed + split_seed_step * split_number``.

    The manuscript clinical calibration uses step 1; the prespecified ungated
    no-endoscopy sensitivity uses step 100 with vision-aligned inner folds.
    """
    if not isinstance(split_seed_step, (int, np.integer)) or split_seed_step < 1:
        raise ValueError("split_seed_step must be a positive integer")
    validate_inner_splits(len(train_df), inner_splits)
    output = np.full(len(train_df), np.nan)
    for fold, (train_idx, validation_idx) in enumerate(inner_splits, start=1):
        model = fit_raw_clinical(train_df.iloc[train_idx], feature_columns,
                                 random_state=int(base_seed) + int(split_seed_step) * fold,
                                 target=target)
        held_out = train_df.iloc[validation_idx]
        output[validation_idx] = model.raw_margin(held_out) if return_margin else model.raw_probability(held_out)
    if not np.isfinite(output).all():
        raise RuntimeError("Clinical inner-OOF predictions are incomplete")
    return output


def fit_clinical(train_df: pd.DataFrame, feature_columns: Sequence[str],
                 inner_splits: Sequence[tuple[np.ndarray, np.ndarray]], *, seed: int = 42,
                 outer_fold: int = 0, target: str = "outcome") -> ClinicalModel:
    """Fit the manuscript branch with calibration contained in training.

    Use the canonical outer-fold number (1--5) for development replication;
    ``outer_fold=0`` supplies the full-development external-fit seed recipe.
    Supplied inner indices address rows of ``train_df`` in its current order.
    """
    base_seed = int(seed) + int(outer_fold) * 10_000
    raw_oof = crossfit_raw_clinical(train_df, feature_columns, inner_splits,
                                   base_seed=base_seed + 200, target=target)
    calibration = choose_calibration(raw_oof, _labels(train_df, target))
    model = fit_raw_clinical(train_df, feature_columns, random_state=base_seed + 10_002,
                             target=target)
    model.calibration = calibration
    model.inner_oof_raw = raw_oof
    return model
