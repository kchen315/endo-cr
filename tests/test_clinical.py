"""Clinical-branch tests use synthetic records only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from endo_cr_analysis.clinical import (
    CATBOOST_PARAMS,
    ClinicalPreprocessor,
    apply_calibration,
    choose_calibration,
    crossfit_raw_clinical,
    fit_clinical,
    fit_raw_clinical,
    validate_inner_splits,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(145)
    n = 72
    x = rng.normal(size=n)
    y = (x + rng.normal(size=n) > 0).astype(int)
    frame = pd.DataFrame({"numeric": x,
                          "category": np.where(rng.random(n) > .5, "A", "B"),
                          "missing_numeric": rng.normal(size=n),
                          "constant": 7.0, "outcome": y})
    frame.loc[::5, "missing_numeric"] = np.nan
    frame.loc[::7, "category"] = None
    splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=44).split(frame, y))
    return frame, ["numeric", "category", "missing_numeric", "constant"], splits


def test_preprocessing_uses_training_only():
    train = pd.DataFrame({"number": [1.0, 2.0, 5.0, np.nan],
                          "category": [" A ", "B", None, ""],
                          "numeric_string": ["1", "2", "3", "4"],
                          "constant": [1, 1, 1, 1],
                          "absent": [np.nan] * 4})
    prep = ClinicalPreprocessor.fit(train, list(train.columns))
    other = pd.DataFrame({"number": [np.nan, 10000.0], "category": [None, "new"],
                          "numeric_string": ["bad", "10000"], "constant": [9, 8],
                          "absent": [1, 2]})
    transformed = prep.transform(other)
    assert prep.retained_features == ["number", "numeric_string", "category"]
    assert transformed.loc[0, "number"] == 2
    assert transformed.loc[0, "numeric_string"] == 2.5
    assert transformed.loc[0, "category"] == "Missing"
    assert transformed.loc[1, "category"] == "new"
    assert prep.categorical_indices == [2]


def test_fold_specific_filter_not_heldout_distribution():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "mostly_missing": [np.nan] * 4})
    prep = ClinicalPreprocessor.fit(train, ["x", "mostly_missing"])
    held_out = pd.DataFrame({"x": [99], "mostly_missing": [2]})
    assert list(prep.transform(held_out)) == ["x"]


def test_fit_predict_shap_and_seed_recipe(synthetic):
    frame, features, splits = synthetic
    model = fit_clinical(frame, features, splits, seed=42, outer_fold=2)
    assert model.random_state == 30044
    assert model.estimator.get_params()["iterations"] == CATBOOST_PARAMS["iterations"] == 100
    assert model.inner_oof_raw.shape == (len(frame),)
    unlabeled = frame.drop(columns="outcome").iloc[:8]
    probability = model.predict_proba(unlabeled)
    assert probability.shape == (8,)
    assert np.all((probability > 0) & (probability < 1))
    for calibrated in (True, False):
        attribution = model.shap(unlabeled, calibrated=calibrated)
        assert attribution["reconstruction_error"] < 1e-8
        np.testing.assert_allclose(attribution["contributions"].sum(axis=1) + attribution["baseline"],
                                   attribution["margin"], atol=1e-8)


def test_inner_crossfit_prediction_seed_and_train_containment(synthetic):
    frame, features, splits = synthetic
    oof = crossfit_raw_clinical(frame, features, splits, base_seed=900)
    train_idx, held_out_idx = splits[0]
    expected = fit_raw_clinical(frame.iloc[train_idx], features, random_state=901)
    np.testing.assert_allclose(oof[held_out_idx], expected.raw_probability(frame.iloc[held_out_idx]), atol=1e-12)
    changed = frame.copy()
    changed.loc[held_out_idx, "outcome"] = 1 - changed.loc[held_out_idx, "outcome"]
    changed_oof = crossfit_raw_clinical(changed, features, splits, base_seed=900)
    np.testing.assert_allclose(oof[held_out_idx], changed_oof[held_out_idx], atol=1e-12)


def test_sensitivity_crossfit_returns_raw_margins_with_seed_step(synthetic):
    frame, features, splits = synthetic
    margins = crossfit_raw_clinical(frame, features, splits, base_seed=1000,
                                    split_seed_step=100, return_margin=True)
    train_idx, held_out_idx = splits[0]
    expected = fit_raw_clinical(frame.iloc[train_idx], features, random_state=1100)
    np.testing.assert_allclose(margins[held_out_idx], expected.raw_margin(frame.iloc[held_out_idx]), atol=1e-12)


def test_calibration_ranking_guard_and_brier():
    raw = np.array([.12, .15, .22, .28, .41, .62, .77, .86, .91, .93])
    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    calibration = choose_calibration(raw, y)
    fitted = apply_calibration(raw, calibration)
    assert np.mean((fitted - y) ** 2) <= np.mean((raw - y) ** 2) + 1e-12
    assert np.all(np.diff(fitted) >= 0)


def test_validate_splits_rejects_overlap_and_duplicate_coverage():
    with pytest.raises(ValueError, match="overlap"):
        validate_inner_splits(4, [(np.array([0, 1]), np.array([1, 2])),
                                   (np.array([0, 1]), np.array([2, 3]))])
    with pytest.raises(ValueError, match="exactly one"):
        validate_inner_splits(4, [(np.array([0, 1]), np.array([2, 3])),
                                   (np.array([0, 1]), np.array([2, 3]))])


def test_disallows_outcome_feature_and_missing_contract(synthetic):
    frame, features, _ = synthetic
    with pytest.raises(ValueError, match="Outcome"):
        fit_raw_clinical(frame, features + ["outcome"])
    with pytest.raises(ValueError, match="required feature"):
        ClinicalPreprocessor.fit(frame, ["not_available"])
