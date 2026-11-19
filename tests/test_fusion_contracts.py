import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from endo_cr_analysis.fusion import fixed_gate, fit_stack, stack_margin, decompose_stack, safe_logit
from endo_cr_analysis.contracts import inner_splits, validate_patients, load_vision
from endo_cr_analysis.human import select_thresholds


def test_fixed_gate_identity_direction_and_bound():
    p = np.array([.001, .2, .5, .8, .999])
    np.testing.assert_allclose(fixed_gate(p, p)["probability"], p, atol=1e-15)
    gate = fixed_gate(p, 1-p)
    assert np.max(gate["gate_weight"]) <= .5
    assert np.max(np.abs(gate["vision_correction_logit"])) <= 1.5
    assert np.all((gate["probability"]-p)*(1-2*p) >= 0)
    np.testing.assert_allclose(safe_logit(gate["probability"]), safe_logit(p)+gate["vision_correction_logit"])


def test_gate_invalid_values():
    with pytest.raises(ValueError):
        fixed_gate([.5], [.3, .4])
    with pytest.raises(ValueError):
        fixed_gate([np.nan], [.2])


def test_stack_reconstruction_and_regularization():
    rng = np.random.default_rng(3)
    shap = rng.normal(size=(90, 4))
    baseline = np.full(90, .1)
    raw = baseline + shap.sum(axis=1)
    vision = rng.normal(size=90)
    y = rng.binomial(1, expit(.5*raw+.3*vision))
    fit = fit_stack(raw, vision, y)
    parts = decompose_stack(fit, shap, baseline, vision)
    assert fit["clinical_coefficient"] >= 0 and fit["vision_coefficient"] >= 0
    np.testing.assert_allclose(parts["margin"], stack_margin(fit, raw, vision), atol=1e-12)
    assert fit_stack(raw, vision, y, False)["vision_coefficient"] == 0


def test_inner_partition_rejects_outer_test_leakage():
    train = pd.DataFrame({"patient_id": list("abcdef"), "outcome": [0,1]*3})
    membership = pd.DataFrame({"patient_id": list("abcdef"), "outer_context_fold": 1,
                                "branch": "vision", "inner_fold": [1,1,2,2,3,3]})
    splits = inner_splits(train, membership, 1, "vision")
    assert len(splits) == 3
    membership.loc[0, "patient_id"] = "held_out_patient"
    with pytest.raises(ValueError):
        inner_splits(train, membership, 1, "vision")


def test_no_external_labels_or_metadata_features():
    frame = pd.DataFrame({"patient_id": ["synthetic_a", "synthetic_b"], "feature": [1, 2], "outcome": [0, 1]})
    with pytest.raises(ValueError):
        validate_patients(frame, ["feature"], development=False)
    with pytest.raises(ValueError):
        validate_patients(frame, ["outcome"], development=False)


def test_vision_identity_alignment_and_pickle_disabled(tmp_path):
    path = tmp_path / "vision.npz"
    ids = np.array(["synthetic_b", "synthetic_a"])
    identity = np.array([[2., 3.], [0., 1.]], np.float32)
    np.savez(path, patient_ids=ids, dino_identity=identity, rn50_identity=identity,
             dino_application=np.broadcast_to(identity, (6,16,2,2)),
             rn50_application=np.broadcast_to(identity, (6,16,2,2)))
    means, _ = load_vision(path, ["synthetic_a", "synthetic_b"])
    np.testing.assert_array_equal(means["dino"][0], [0,1])
    with pytest.raises(ValueError):
        load_vision(path, ["synthetic_a", "unmatched"])


def test_threshold_selection_forbids_external_outcomes():
    frame = pd.DataFrame({"cohort": ["external"]})
    with pytest.raises(ValueError, match="development-only"):
        select_thresholds(frame)
