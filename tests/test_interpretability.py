"""Synthetic analytic-gradient checks; optional Torch tests never load images."""

import numpy as np
import pytest

from endo_cr_analysis.interpretability import (
    combine_registered_maps,
    differentiable_logit,
    differentiable_view_probability,
    effective_raw_weight,
    patient_embedding_gradients,
    token_attribution_for_view,
)
from endo_cr_analysis.vision import fit_mean_head


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(204)
    training = rng.normal(size=(60, 8))
    labels = (training[:, 0] + rng.normal(size=60) > 0).astype(int)
    head = fit_mean_head(training, labels)
    bags = rng.normal(size=(6, 4, 8))
    subsets = np.tile(np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]), (4, 1))
    return head, bags, subsets


def test_effective_weight_is_embedding_logit_derivative(synthetic):
    head, bags, _ = synthetic
    x = bags[0, 0].copy()
    weight = effective_raw_weight(head)
    epsilon = 1e-5
    for column in (0, 3, 7):
        high, low = x.copy(), x.copy()
        high[column] += epsilon
        low[column] -= epsilon
        p_high, p_low = head.predict_proba(np.stack([high, low]))
        slope = (np.log(p_high / (1 - p_high)) - np.log(p_low / (1 - p_low))) / (2 * epsilon)
        assert slope == pytest.approx(weight[column], abs=1e-10)


def test_patient_gradient_matches_finite_difference(synthetic):
    head, bags, subsets = synthetic
    state = patient_embedding_gradients(head, bags, subsets)
    epsilon = 1e-5
    for index in ((0, 0, 0), (1, 2, 3), (5, 3, 7)):
        high, low = bags.copy(), bags.copy()
        high[index] += epsilon
        low[index] -= epsilon
        derivative = (patient_embedding_gradients(head, high, subsets)["dino_probability"]
                      - patient_embedding_gradients(head, low, subsets)["dino_probability"]) / (2 * epsilon)
        assert derivative == pytest.approx(state["embedding_gradients"][index], abs=1e-10)
    negative = patient_embedding_gradients(head, bags, subsets, target_label=0)
    np.testing.assert_array_equal(negative["embedding_gradients"], -state["embedding_gradients"])
    assert negative["dino_probability"] == state["dino_probability"]


def test_excluded_frame_has_zero_contribution(synthetic):
    head, bags, _ = synthetic
    subsets = np.tile([0, 1, 2], (16, 1))
    state = patient_embedding_gradients(head, bags, subsets)
    np.testing.assert_array_equal(state["embedding_gradients"][:, 3], 0)


def test_differentiable_head_and_fixed_subsample_gradient(synthetic):
    torch = pytest.importorskip("torch")
    head, bags, subsets = synthetic
    frame = torch.tensor(bags[2, 1], dtype=torch.float64, requires_grad=True)
    p = differentiable_view_probability(head, frame, bags[2], 1, subsets)
    gradient = torch.autograd.grad(p / 6, frame)[0]
    expected = patient_embedding_gradients(head, bags, subsets)
    assert float(p.detach()) == pytest.approx(expected["view_subsample_probabilities"][2].mean(), abs=1e-12)
    np.testing.assert_allclose(gradient.detach().numpy(), expected["embedding_gradients"][2, 1], atol=1e-12)
    sigmoid = torch.sigmoid(differentiable_logit(head, frame))
    assert float(sigmoid.detach()) == pytest.approx(head.predict_proba(bags[2, 1:2])[0], abs=1e-12)


def test_token_activation_times_gradient_and_hook_cleanup(synthetic):
    torch = pytest.importorskip("torch")
    head, bags, subsets = synthetic

    class PreparedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target = torch.nn.Identity()

        def forward(self, inputs):
            return self.target(inputs)[:, 1:].mean(dim=1)

    encoder = PreparedEncoder().eval()
    values = np.random.default_rng(19).normal(size=(1, 5, 8))
    prepared = torch.tensor(values, dtype=torch.float64)
    stored = bags[0].copy()
    stored[1] = values[0, 1:].mean(axis=0)
    adjusted = bags.copy()
    adjusted[0] = stored
    expected_gradient = patient_embedding_gradients(head, adjusted, subsets)["embedding_gradients"][0, 1]
    expected_map = (values[0, 1:] * expected_gradient / 4).sum(axis=1).reshape(2, 2)
    kwargs = dict(encoder=encoder, target_layer=encoder.target, prepared_tensor=prepared,
                  head=head, stored_embeddings=stored, selected_frame=1, subsets=subsets,
                  grid_size=(2, 2), embedding_tolerance=1e-12)
    result = token_attribution_for_view(**kwargs)
    np.testing.assert_allclose(result["signed_token_map"], expected_map, atol=1e-12)
    negative = token_attribution_for_view(**kwargs, target_label=0)
    np.testing.assert_allclose(negative["signed_token_map"], -expected_map, atol=1e-12)
    assert not encoder.target._forward_hooks
    assert prepared.grad is None
    wrong = stored.copy()
    wrong[1] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        token_attribution_for_view(**{**kwargs, "stored_embeddings": wrong})
    assert not encoder.target._forward_hooks


def test_registered_maps_sum_before_positive_normalization():
    maps = np.array([[[2, -1], [0, 1]], [[-1, 2], [0, -2]],
                     [[0, 0], [0, 0]], [[0, 0], [0, 0]],
                     [[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=float)
    result = combine_registered_maps(maps)
    np.testing.assert_array_equal(result["signed_sum"], [[1, 1], [0, -1]])
    np.testing.assert_array_equal(result["positive_normalized"], [[1, 1], [0, 0]])
    with pytest.raises(ValueError, match="no positive"):
        combine_registered_maps(-np.ones((6, 2, 2)))


def test_invalid_subsample_and_observed_class_inputs(synthetic):
    head, bags, subsets = synthetic
    with pytest.raises(ValueError, match="75%"):
        patient_embedding_gradients(head, bags, subsets[:, :2])
    with pytest.raises(ValueError, match="target"):
        patient_embedding_gradients(head, bags, subsets, target_label=2)
