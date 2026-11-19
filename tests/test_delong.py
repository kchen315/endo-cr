"""Synthetic DeLong tests, independently cross-checked using midranks."""


import numpy as np
import pytest
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from endo_cr_analysis import delong as D


def _independent_midrank_reference(y, scores):
    """Independent midrank identity; no pairwise comparisons or production code."""
    ordered = scores[:, np.argsort(-y, kind="stable")]
    m, n = np.sum(y == 1), np.sum(y == 0)
    tx = np.stack([rankdata(row[:m], method="average") for row in ordered])
    ty = np.stack([rankdata(row[m:], method="average") for row in ordered])
    tz = np.stack([rankdata(row, method="average") for row in ordered])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v_positive = (tz[:, :m] - tx) / n
    v_negative = 1 - (tz[:, m:] - ty) / m
    covariance = (np.atleast_2d(np.cov(v_positive, ddof=1)) / m
                  + np.atleast_2d(np.cov(v_negative, ddof=1)) / n)
    return auc, covariance


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ties", [False, True])
def test_auc_covariance_matches_independent_midrank_and_sklearn(seed, ties):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(29), np.ones(17)]
    rng.shuffle(y)
    scores = rng.normal(size=(4, len(y)))
    if ties:
        scores = scores.round(0)
    actual_auc, actual_cov = D.auc_covariance(y, scores)
    reference_auc, reference_cov = _independent_midrank_reference(y, scores)
    np.testing.assert_allclose(actual_auc, reference_auc, atol=1e-15)
    np.testing.assert_allclose(actual_cov, reference_cov, atol=1e-15)
    for k in range(len(scores)):
        assert actual_auc[k] == pytest.approx(roc_auc_score(y, scores[k]))
    pair = D.paired_delong(y, scores[0], scores[1])
    contrast_variance = actual_cov[0, 0] + actual_cov[1, 1] - 2 * actual_cov[0, 1]
    assert pair["variance_difference"] == pytest.approx(contrast_variance)
    z = (actual_auc[0] - actual_auc[1]) / np.sqrt(contrast_variance)
    assert pair["p_value"] == pytest.approx(2 * norm.sf(abs(z)))


def test_hand_computed_ties_covariance_and_normal_interval():
    y = [1, 1, 0, 0]
    # Placements: positive=[.75, 1], negative=[1, .75].
    # AUROC=.875; each placement sample variance=.03125, covariance=.03125.
    result = D.auc_summary(y, [1, 2, 0, 1])
    assert result["auroc"] == 0.875
    assert result["auroc_variance"] == 0.03125
    assert result["auroc_ci_low"] == pytest.approx(.875 - norm.ppf(.975) * np.sqrt(.03125))
    assert result["auroc_ci_high"] == 1
    assert result["status"] == "ok"


def test_single_model_covariance_is_two_dimensional():
    auc, covariance = D.auc_covariance([1, 1, 0, 0], [1, 2, 0, 1])
    assert auc.shape == (1,)
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == .03125


@pytest.mark.parametrize("scores", [[0, 0, 1, 1], [1, 1, 0, 0], [.5, .5, .5, .5]])
def test_degenerate_ci_not_claimed_as_zero_width(scores):
    result = D.auc_summary([0, 0, 1, 1], scores)
    assert result["auroc_variance"] == 0
    assert result["auroc_ci_low"] is None
    assert result["auroc_ci_high"] is None
    assert result["status"] == "zero_variance_ci_not_estimable"


def test_identical_model_test_is_explicit_degenerate_convention():
    scores = [.1, .8, .3, .9]
    result = D.paired_delong([0, 0, 1, 1], scores, scores)
    assert result["difference"] == 0
    assert result["variance_difference"] == 0
    assert result["p_value"] == 1
    assert result["status"] == "zero_variance_equal_auc"


def test_identical_ranking_different_scores_is_degenerate():
    scores = np.array([.1, .8, .3, .9])
    result = D.paired_delong([0, 0, 1, 1], scores, scores * 3 + 1)
    assert result["p_value"] == 1
    assert result["status"] == "zero_variance_equal_auc"


def test_separation_vs_constant_is_not_fabricated_as_zero_p():
    result = D.paired_delong([0, 0, 1, 1], [0, 0, 1, 1], [.5, .5, .5, .5])
    assert result["difference"] == .5
    assert result["variance_difference"] == 0
    assert result["p_value"] is None
    assert result["z_statistic"] is None
    assert result["status"] == "zero_variance_nonzero_difference_not_estimable"


def test_order_invariance_and_test_symmetry():
    y = np.array([1, 0, 1, 0, 0, 1, 0])
    a = np.array([2, 1, 0, 1, 2, 2, 0])
    b = np.array([1, 1, 1, 2, 0, 2, 1])
    original = D.paired_delong(y, a, b)
    permutation = [6, 2, 4, 1, 0, 3, 5]
    permuted = D.paired_delong(y[permutation], a[permutation], b[permutation])
    assert permuted == original
    reversed_pair = D.paired_delong(y, b, a)
    assert reversed_pair["p_value"] == original["p_value"]
    assert reversed_pair["difference"] == -original["difference"]


@pytest.mark.parametrize("y,scores,error", [
    ([], [], "nonempty"),
    ([0, 1, 1], [.1, .3, .8], "at least two"),
    ([0, 0, 0, 0], [.1, .3, .8, .2], "at least two"),
    ([0, 0, 1, 2], [.1, .3, .8, .2], "binary"),
    ([0, 0, 1, np.nan], [.1, .3, .8, .2], "binary"),
    ([0, 0, 1, 1], [.1, .3, np.nan, .2], "finite"),
    ([0, 0, 1, 1], [.1, .3, np.inf, .2], "finite"),
    ([0, 0, 1, 1], [.1, .3, .2], "shape"),
])
def test_invalid_inputs_fail_closed(y, scores, error):
    with pytest.raises(ValueError, match=error):
        D.auc_summary(y, scores)


@pytest.mark.parametrize("confidence", [0, 1, -1, np.nan])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence"):
        D.auc_summary([0, 0, 1, 1], [.1, .8, .3, .9], confidence=confidence)


def test_no_silent_paired_shape_alignment():
    with pytest.raises(ValueError, match="equal-length"):
        D.paired_delong([0, 0, 1, 1], [.1, .2, .3, .4], [.2, .4, .6])
    with pytest.raises(ValueError, match="one-dimensional"):
        D.paired_delong([0, 0, 1, 1], [[.1, .2, .3, .4]], [[.2, .4, .6, .8]])
