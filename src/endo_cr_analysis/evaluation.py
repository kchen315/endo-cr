"""Performance metrics and calibration for held-out patient predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss,
    roc_auc_score,
)

EPS = 1e-6
PROBABILITY_MODELS = ("clinical", "fusion", "vision", "dino", "rn50", "human_ridge",
                      "ungated_noendo")
ORDINAL_MODELS = ("human_endoscopy", "human_mri")
ALL_MODELS = PROBABILITY_MODELS + ORDINAL_MODELS
SCOPES = {
    "vision_vs_endoscopy": ("human_endoscopy", "vision", "dino", "rn50"),
    "fusion_vs_human_restaging": ("human_mri", "human_endoscopy", "human_ridge", "fusion"),
}


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate the public input schema; never silently coerce missing labels."""
    required = {"patient_id", "outcome", "cohort"}
    if not required <= set(predictions):
        raise ValueError(f"Missing prediction fields: {sorted(required - set(predictions))}")
    data = predictions.copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Patient identifier, cohort, and outcome must be nonmissing")
    if not data["outcome"].isin([0, 1]).all():
        raise ValueError("outcome must be binary (0 or 1)")
    if not data["cohort"].isin(["development", "external"]).all():
        raise ValueError("cohort must be development or external")
    if data["patient_id"].duplicated().any():
        raise ValueError("Each patient must occur exactly once, in a single cohort")
    for column in ALL_MODELS + ("human_rule",):
        if column not in data:
            continue
        data[column] = pd.to_numeric(data[column], errors="raise")
        values = data[column].dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite predictions in {column}")
        if column in PROBABILITY_MODELS and not values.between(0, 1).all():
            raise ValueError(f"{column} must contain probabilities in [0, 1]")
        if column in ORDINAL_MODELS:
            grid = np.linspace(0, 1, 4 if column == "human_endoscopy" else 5)
            if not np.isclose(values.to_numpy()[:, None], grid).any(axis=1).all():
                raise ValueError(f"Invalid ordinal score for {column}")
        if column == "human_rule" and not values.isin([0, 1]).all():
            raise ValueError("human_rule must be a prespecified binary assessment")
    return data


def paired_data(data: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    """Return the exact shared denominator for all named models."""
    missing = set(models) - set(data)
    if missing:
        raise ValueError(f"Missing requested comparison models: {sorted(missing)}")
    return data.dropna(subset=["outcome", *models]).copy()


def metric_values(y: np.ndarray, score: np.ndarray, *, probabilistic: bool = True) -> dict[str, float]:
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    both_classes = len(np.unique(y)) == 2
    values = {
        "auroc": float(roc_auc_score(y, score)) if both_classes else np.nan,
        "auprc": float(average_precision_score(y, score)) if both_classes else np.nan,
    }
    if probabilistic:
        probability = np.clip(score, EPS, 1 - EPS)
        values.update(brier=float(brier_score_loss(y, probability)),
                      log_loss=float(log_loss(y, probability, labels=[0, 1])))
    return values


def percentile(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def calibration_summary(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Joint diagnostic intercept/slope only; this fit never changes predictions."""
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    logits = np.log(probability / (1 - probability))
    def objective(parameters):
        values = expit(np.clip(parameters[0] + parameters[1] * logits, -40, 40))
        return float(log_loss(y, np.clip(values, EPS, 1 - EPS), labels=[0, 1]))
    result = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    intercept, slope = result.x if np.isfinite(result.x).all() else [np.nan, np.nan]
    return {"calibration_intercept": float(intercept), "calibration_slope": float(slope),
            "optimizer_success": bool(result.success),
            "brier": float(brier_score_loss(y, probability)),
            "mean_probability": float(probability.mean()), "observed_prevalence": float(y.mean())}


def calibration_tables(data: pd.DataFrame, models: Sequence[str], bins: int,
                       samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = paired_data(data, models)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    if any(model not in PROBABILITY_MODELS for model in models):
        raise ValueError("Raw ordinal assessments are not calibrated probabilities")
    cohort = data["cohort"].iloc[0]
    if cohort == "external" and bins > 3:
        raise ValueError("External calibration is limited to three groups")
    bins = min(bins, len(data))
    rng = np.random.default_rng(seed)
    y = data["outcome"].to_numpy(dtype=int)
    rows, summaries = [], []
    for model in models:
        p = data[model].to_numpy(dtype=float)
        for number, indices in enumerate(np.array_split(np.argsort(p, kind="mergesort"), bins), 1):
            draws = [float(y[rng.choice(indices, len(indices), replace=True)].mean()) for _ in range(samples)]
            low, high = percentile(draws)
            rows.append({"cohort": cohort, "model": model, "bin": number, "n": len(indices),
                         "mean_probability": float(p[indices].mean()), "observed_response": float(y[indices].mean()),
                         "ci_low": low, "ci_high": high, "binning": f"{bins}_equal_frequency_groups"})
        summaries.append({"cohort": cohort, "model": model, "n": len(data), "groups": bins,
                          **calibration_summary(y, p)})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


