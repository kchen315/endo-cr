"""Synthetic checks of the prediction-lock boundary and attribution padding."""
import numpy as np
import pandas as pd
import pytest

from endo_cr_analysis.clinical import fit_raw_clinical
from endo_cr_analysis.contracts import new_output, sha256, write_json
from endo_cr_analysis.fusion import fixed_gate
from endo_cr_analysis.pipeline import _attribution_rows, evaluate_external


def test_zero_padding_of_fold_filtered_features():
    train = pd.DataFrame({"signal": np.arange(40), "constant": 1., "outcome": [0,1]*20})
    model = fit_raw_clinical(train, ["signal", "constant"])
    test = pd.DataFrame({"signal": [4,20], "constant": [1.,1.], "patient_id": ["SYNTHETIC_A", "SYNTHETIC_B"]})
    gate = fixed_gate(model.predict_proba(test), [.3, .6])
    values = _attribution_rows(model, test, gate["probability"], gate["vision_correction_logit"], "fusion")
    assert len(values.loc[values.feature.eq("constant")]) == 2
    np.testing.assert_array_equal(values.loc[values.feature.eq("constant"), "contribution"], 0.)


def test_external_checksum_verified_before_labels_are_opened(tmp_path):
    pred = tmp_path / "predictions.csv"
    pred.write_text("patient_id,cohort,fusion\nSYNTHETIC_A,external,0.2\n")
    lock = tmp_path / "lock.json"
    write_json(lock, {"external_outcomes_accessed": False, "predictions_sha256": "invalid"})
    with pytest.raises(ValueError, match="lock is invalid"):
        evaluate_external(pred, lock, tmp_path / "labels_that_do_not_exist.csv", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_external_exact_denominator_and_output_no_overwrite(tmp_path):
    pred = tmp_path / "predictions.csv"
    pd.DataFrame({"patient_id": ["SYNTHETIC_A", "SYNTHETIC_B"], "cohort": "external", "fusion": [.2,.8]}).to_csv(pred, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"patient_id": ["SYNTHETIC_B", "SYNTHETIC_A"], "outcome": [1,0]}).to_csv(labels, index=False)
    lock = tmp_path / "lock.json"
    write_json(lock, {"external_outcomes_accessed": False, "predictions_sha256": sha256(pred)})
    output = evaluate_external(pred, lock, labels, tmp_path / "output")
    result = pd.read_csv(output / "external_predictions_evaluated.csv")
    assert result.outcome.tolist() == [0,1]
    with pytest.raises(FileExistsError):
        evaluate_external(pred, lock, labels, tmp_path / "output")
    labels.write_text("patient_id,outcome\nSYNTHETIC_A,0\n")
    with pytest.raises(ValueError, match="match exactly"):
        evaluate_external(pred, lock, labels, tmp_path / "other")


def test_output_cannot_be_source_package():
    from pathlib import Path
    public_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside"):
        new_output(public_root / "should_not_exist")
    assert not (public_root / "should_not_exist").exists()
