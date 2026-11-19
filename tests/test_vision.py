"""Synthetic-only checks of the released frozen vision algorithm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from endo_cr_analysis.vision import (
    BACKBONES,
    VIEWS,
    fit_mean_head,
    fit_vision_heads,
    mean_features,
    predict_vision_ensemble,
    prepare_vision_features,
    subsample_mean_features,
    summarize_prediction_tensor,
)


def test_head_matches_locked_pipeline():
    rng = np.random.default_rng(21)
    train = rng.normal(size=(40, 32)).astype(np.float32)
    test = rng.normal(size=(11, 32)).astype(np.float32)
    labels = np.tile([0, 1], 20)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(train, dtype=float))
    pca = PCA(n_components=16, random_state=42)
    reduced = pca.fit_transform(scaled)
    classifier = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=2000, random_state=42)
    classifier.fit(reduced, labels)
    expected = classifier.predict_proba(pca.transform(scaler.transform(test.astype(float))))[:, 1]
    fitted = fit_mean_head(train, labels)
    assert_allclose(fitted.predict_proba(test), expected, atol=1e-14, rtol=1e-14)
    assert_allclose(fitted.scaler.mean_, train.astype(float).mean(axis=0))


def test_feature_pooling_matches_float32_reference():
    rng = np.random.default_rng(3)
    bags = [rng.normal(size=(count, 19)).astype(np.float32) for count in (1, 3, 5)]
    draws = [np.stack([np.sort(rng.choice(len(bag), int(np.ceil(0.75 * len(bag))), replace=False))
                       for _ in range(16)]) for bag in bags]
    observed = subsample_mean_features(bags, draws)
    expected = np.stack([np.stack([bag[index[replicate]].mean(axis=0)
                                  for bag, index in zip(bags, draws)]) for replicate in range(16)])
    assert_allclose(observed, expected, atol=0, rtol=0)
    assert_allclose(mean_features(bags), np.stack([bag.mean(axis=0) for bag in bags]), atol=0, rtol=0)
    reversed_bags = [bag[::-1] for bag in bags]
    assert_allclose(mean_features(bags), mean_features(reversed_bags), atol=2e-7)


def test_ensemble_and_uncertainty_match_original_axes():
    values = np.random.default_rng(12).uniform(0.05, 0.95, size=(2, 6, 16, 9))
    result = summarize_prediction_tensor(values)
    assert_allclose(result["p_vision"], values.mean(axis=(0, 1)).mean(axis=0), atol=0, rtol=0)
    assert_allclose(result["p_dino"], values.mean(axis=(1, 2))[0], atol=0, rtol=0)
    assert_allclose(result["augmentation_variance"], values.mean(axis=(0, 2)).var(axis=0), atol=0, rtol=0)
    assert_allclose(result["subsample_variance"], values.mean(axis=(0, 1)).var(axis=0), atol=0, rtol=0)
    assert_allclose(result["p_vision"], (result["p_dino"] + result["p_rn50"]) / 2)


def test_prepared_banks_end_to_end():
    rng = np.random.default_rng(8)
    counts = [4] * 24
    draws = [np.tile([0, 1, 3], (16, 1)) for _ in counts]
    banks = {backbone: {view: [rng.normal(size=(count, 20)) for count in counts]
                        for view in VIEWS} for backbone in BACKBONES}
    identity, application = prepare_vision_features(banks, draws)
    heads = fit_vision_heads({key: value[:20] for key, value in identity.items()}, np.tile([0, 1], 10))
    result = predict_vision_ensemble(heads, {key: value[:, :, 20:] for key, value in application.items()})
    assert result["p_vision"].shape == (4,)
    assert np.all((result["p_vision"] > 0) & (result["p_vision"] < 1))
    assert np.all(result["augmentation_variance"] >= 0)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="sorted and unique"):
        subsample_mean_features([np.ones((4, 5))], [np.tile([0, 0, 1], (16, 1))])
    with pytest.raises(ValueError, match="ceil"):
        subsample_mean_features([np.ones((4, 5))], [np.tile([0, 1], (16, 1))])
    with pytest.raises(ValueError, match="finite"):
        fit_mean_head(np.full((4, 3), np.nan), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="both classes"):
        fit_mean_head(np.ones((4, 3)), [0, 0, 0, 0])
    with pytest.raises(ValueError, match="2 x 6 x 16"):
        summarize_prediction_tensor(np.ones((2, 6, 15, 4)))
