"""Patient-level DeLong inference for full AUROC, conditional on fixed scores.

References
----------
DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics. 1988;44:837-845.
doi:10.2307/2531595; https://pubmed.ncbi.nlm.nih.gov/3203132/

For each model, compare every positive score with every negative score using
1 for a win, 0.5 for a tie, and 0 otherwise. The average is the AUROC. Average
the comparisons within each positive and negative patient separately; the
sample covariance of these two sets of placements, divided by the respective
class sizes, gives DeLong's covariance matrix. This transparent O(K*m*n)
implementation is appropriate for this small patient-level study.

Only complete, already aligned patient-level arrays are accepted. Scores must
increase toward outcome 1. No automatic missing-value removal, score reversal,
weighting, multiplicity adjustment, partial-AUROC inference, or model refitting
is performed. DeLong does NOT apply to AUPRC, Brier score, or log loss. Applying
it to OOF predictions is conditional/descriptive inference: it does not model
dependence induced by overlapping training sets, training variability, or prior
model selection. It must not be presented as full model-development uncertainty.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm


def _placements(y_true: Sequence[float], scores: np.ndarray):
    """Return validated labels, AUROCs, positive/negative placement matrices."""
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(scores, dtype=float)
    if y.ndim != 1 or not len(y):
        raise ValueError("y_true must be a nonempty one-dimensional array")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only finite binary 0/1 outcomes")
    if score.ndim == 1:
        score = score[None, :]
    if score.ndim != 2 or not score.shape[0] or score.shape[1] != len(y):
        raise ValueError("scores must have shape (n_models, n_patients)")
    if not np.isfinite(score).all():
        raise ValueError("scores must be finite; align and restrict patients explicitly")
    n_positive = int(np.count_nonzero(y == 1))
    n_negative = len(y) - n_positive
    if min(n_positive, n_negative) < 2:
        raise ValueError("DeLong variance requires at least two patients in each outcome class")
    positive = score[:, y == 1, None]
    negative = score[:, None, y == 0]
    wins = (positive > negative).astype(float) + 0.5 * (positive == negative)
    positive_placements = wins.mean(axis=2)
    negative_placements = wins.mean(axis=1)
    aucs = wins.mean(axis=(1, 2))
    return y, aucs, positive_placements, negative_placements


def auc_covariance(y_true: Sequence[float], scores: np.ndarray):
    """Return (AUROC vector, DeLong covariance), scores = models x patients.

    A one-dimensional score array is treated as one model; covariance always
    has shape (n_models, n_models). Equal-score ties receive exactly half credit.
    """
    _, aucs, positive, negative = _placements(y_true, scores)
    covariance = (np.atleast_2d(np.cov(positive, ddof=1)) / positive.shape[1]
                  + np.atleast_2d(np.cov(negative, ddof=1)) / negative.shape[1])
    return aucs, covariance


def auc_summary(y_true: Sequence[float], scores: Sequence[float], *, confidence: float = 0.95):
    """Return individual AUROC and clipped normal-approximation DeLong CI.

    At zero estimated variance, no nominal CI is reported: a point interval
    at perfect separation or a constant score would misleadingly suggest that
    population AUROC has no uncertainty. These cases have explicit status.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if np.asarray(scores).ndim != 1:
        raise ValueError("auc_summary requires a one-dimensional score array")
    y, aucs, positive, negative = _placements(y_true, scores)
    variance = float(np.var(positive[0], ddof=1) / positive.shape[1]
                     + np.var(negative[0], ddof=1) / negative.shape[1])
    auc = float(aucs[0])
    result = {"n": len(y), "n_positive": int(np.count_nonzero(y)),
              "n_negative": int(np.count_nonzero(y == 0)), "auroc": auc,
              "auroc_variance": variance, "confidence": float(confidence),
              "auroc_ci_low": None, "auroc_ci_high": None,
              "status": "zero_variance_ci_not_estimable"}
    if variance > 0:
        margin = float(norm.ppf((1 + confidence) / 2) * np.sqrt(variance))
        result.update(auroc_ci_low=max(0.0, auc - margin),
                      auroc_ci_high=min(1.0, auc + margin), status="ok")
    return result


def paired_delong(y_true: Sequence[float], scores_a: Sequence[float], scores_b: Sequence[float]):
    """Two-sided paired DeLong test of equality of full AUROC on identical rows.

    ``difference`` is AUROC(a) - AUROC(b), retained for audit but need not be
    displayed in the manuscript. ``p_value`` is unadjusted. Both score vectors
    must correspond to the same patients and outcomes in the same order.

    Equal AUROCs with zero contrast variance return the descriptive convention
    p=1 and ``zero_variance_equal_auc`` status (not evidence of equivalence).
    A nonzero difference with zero variance returns no z or p rather than an
    artificial certainty of p=0.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("paired scores must be equal-length one-dimensional arrays")
    y, aucs, positive, negative = _placements(y_true, np.stack([a, b]))
    # Direct variance of the structural-component differences is equivalent
    # to S_aa + S_bb - 2*S_ab but avoids cancellation for identical rankings.
    variance = float(np.var(positive[0] - positive[1], ddof=1) / positive.shape[1]
                     + np.var(negative[0] - negative[1], ddof=1) / negative.shape[1])
    difference = float(aucs[0] - aucs[1])
    result = {"n": len(y), "n_positive": int(np.count_nonzero(y)),
              "n_negative": int(np.count_nonzero(y == 0)),
              "auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]),
              "difference": difference, "variance_difference": variance,
              "z_statistic": None, "p_value": None, "log10_p_value": None,
              "status": "zero_variance_nonzero_difference_not_estimable"}
    if variance > 0:
        z = float(difference / np.sqrt(variance))
        result.update(z_statistic=z, p_value=float(2 * norm.sf(abs(z))),
                      log10_p_value=float((np.log(2) + norm.logsf(abs(z))) / np.log(10)),
                      status="ok")
    elif difference == 0:
        result.update(z_statistic=0.0, p_value=1.0, log10_p_value=0.0,
                      status="zero_variance_equal_auc")
    return result
