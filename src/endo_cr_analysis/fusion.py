"""Frozen manuscript fusion and the separate development-only sensitivity.

The fixed gate is an exact logit correction, not a fitted mixture-of-experts.
The ungated model is NOT the promoted manuscript model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-6


def safe_probability(values):
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected finite probabilities between zero and one")
    return np.clip(p, EPS, 1 - EPS)


def safe_logit(values):
    p = safe_probability(values)
    return np.log(p / (1 - p))


def fixed_gate(clinical_probability, dino_probability):
    """Apply the fixed DINO-only gate; RN50 is intentionally not an input."""
    clinical = safe_probability(clinical_probability)
    dino = safe_probability(dino_probability)
    if clinical.shape != dino.shape:
        raise ValueError("Clinical and DINO probabilities must be patient-aligned")
    entropy = np.clip(-(clinical * np.log(clinical) +
                        (1 - clinical) * np.log(1 - clinical)) / np.log(2), 0, 1)
    disagreement = np.clip(2 * np.abs(clinical - dino), .01, 1)
    delta = np.clip(safe_logit(dino) - safe_logit(clinical), -3, 3)
    weight = np.clip(.5 * entropy**2 * disagreement**2, 0, 1.5)
    correction = weight * delta
    probability = safe_probability(expit(np.clip(safe_logit(clinical) + correction, -40, 40)))
    return {"probability": probability, "clinical_entropy": entropy,
            "disagreement": disagreement, "vision_logit_delta": delta,
            "gate_weight": weight, "vision_correction_logit": correction}


def fit_stack(raw, vision, labels, use_vision=True):
    """Fixed nonnegative C=1 ridge logit stack on MATCHED INNER-OOF inputs.

    Clinical input is an uncalibrated CatBoost margin, not an in-sample
    calibrated probability. Vision input is a DINO logit. The caller must
    ensure both branches held out the same inner-validation patients.
    Loss is SUM logistic loss + (clinical_slope² + vision_slope²)/2.
    """
    raw, vision, y = (np.asarray(v, dtype=float) for v in (raw, vision, labels))
    if (raw.ndim != 1 or raw.shape != vision.shape or raw.shape != y.shape
            or not np.isfinite(np.r_[raw, vision, y]).all()
            or set(np.unique(y)) != {0., 1.}):
        raise ValueError("Aligned finite vectors with both outcome classes required")
    x = np.column_stack([raw, vision]) if use_vision else raw[:, None]
    center = x.mean(axis=0)
    x = x - center

    def objective(theta):
        eta = theta[0] + x @ theta[1:]
        loss = np.logaddexp(0, eta).sum() - y @ eta + theta[1:] @ theta[1:] / 2
        residual = expit(eta) - y
        return float(loss), np.r_[residual.sum(), x.T @ residual + theta[1:]]

    initial = np.r_[safe_logit([y.mean()])[0], np.ones(x.shape[1])]
    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(None, None)] + [(0, None)] * x.shape[1],
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    loss, gradient = objective(fitted.x)
    projected = gradient.copy()
    for j in range(1, len(fitted.x)):
        if fitted.x[j] <= 1e-8 and projected[j] > 0:
            projected[j] = 0
    if not fitted.success or np.max(np.abs(projected)) > 1e-4:
        raise RuntimeError(f"Stack optimizer failed: {fitted.message}")
    return {"intercept_centered": float(fitted.x[0]),
            "clinical_coefficient": float(fitted.x[1]),
            "vision_coefficient": float(fitted.x[2]) if use_vision else 0.,
            "clinical_logit_center": float(center[0]),
            "vision_logit_center": float(center[1]) if use_vision else float(vision.mean()),
            "training_n": len(y), "optimizer_success": bool(fitted.success),
            "penalized_loss_sum": loss, "C": 1.,
            "maximum_projected_gradient": float(np.max(np.abs(projected)))}


def stack_margin(fit, raw, vision):
    return (fit["intercept_centered"]
            + fit["clinical_coefficient"] * (np.asarray(raw) - fit["clinical_logit_center"])
            + fit["vision_coefficient"] * (np.asarray(vision) - fit["vision_logit_center"]))


def decompose_stack(fit, raw_shap, raw_baseline, vision_logits):
    """Exact additive logit decomposition; not a causal modality attribution."""
    clinical = fit["clinical_coefficient"] * np.asarray(raw_shap)
    baseline = fit["intercept_centered"] + fit["clinical_coefficient"] * (
        np.asarray(raw_baseline) - fit["clinical_logit_center"])
    vision = fit["vision_coefficient"] * (np.asarray(vision_logits) - fit["vision_logit_center"])
    return {"clinical": clinical, "baseline": baseline, "vision": vision,
            "margin": baseline + clinical.sum(axis=1) + vision}
