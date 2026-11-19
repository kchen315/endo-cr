"""Final-only nested patient-level analysis; no model or preprocessing search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .clinical import fit_clinical, crossfit_raw_clinical
from .contracts import inner_splits, load_feature_columns, load_vision, new_output, sha256, validate_patients, write_json
from .fusion import fixed_gate, safe_logit, fit_stack, stack_margin, decompose_stack
from .human import fit_human, predict_human, fixed_human_rule
from .vision import fit_vision_heads, predict_vision_ensemble


def _load(patients, features, memberships, vision):
    frame = pd.read_csv(patients, dtype={"patient_id": str})
    columns = load_feature_columns(features)
    validate_patients(frame, columns)
    splits = pd.read_csv(memberships, dtype={"patient_id": str})
    means, applications = load_vision(vision, frame.patient_id)
    return frame, columns, splits, means, applications


def _subset(values, index, application=False):
    return {k: v[:, :, index] if application else v[index] for k, v in values.items()}


def _vision_predict(means, applications, train_index, test_index, labels):
    heads = fit_vision_heads(_subset(means, train_index), labels)
    return predict_vision_ensemble(heads, _subset(applications, test_index, True))


def _attribution_rows(model, test, clinical, vision_correction, model_name, fit=None, dino_logit=None):
    raw = model.shap(test, calibrated=fit is None)
    if fit is None:
        contributions = raw["contributions"]
        baseline = raw["baseline"]
        vision = vision_correction
    else:
        parts = decompose_stack(fit, raw["contributions"], raw["baseline"], dino_logit)
        contributions, baseline, vision = parts["clinical"], parts["baseline"], parts["vision"]
    reconstructed = baseline + contributions.sum(axis=1) + vision
    if not np.allclose(expit(reconstructed), clinical, atol=1e-8, rtol=0):
        raise RuntimeError("Fusion attribution failed additive reconstruction")
    # A feature filtered inside one training fold contributes zero there;
    # omitting its patient rows would inflate mean absolute attribution.
    table = pd.DataFrame(0., index=np.arange(len(test)), columns=model.preprocessor.feature_columns)
    table.loc[:, raw["feature_names"]] = contributions
    table["DINO vision contribution"] = vision
    table.insert(0, "patient_id", test.patient_id.to_numpy())
    table = table.melt(id_vars="patient_id", var_name="feature", value_name="contribution")
    table["model"] = model_name
    return table


def run_development(patients, features, memberships, vision, output, noendo_sensitivity=False):
    """Fit only locked recipes; write private OOF outputs to a NEW directory."""
    frame, columns, splits, means, applications = _load(patients, features, memberships, vision)
    for fold in range(1, 6):
        train = frame.loc[frame.outer_fold.ne(fold)].reset_index(drop=True)
        inner_splits(train, splits, fold, "clinical")
        if noendo_sensitivity:
            inner_splits(train, splits, fold, "vision")
    excluded = json.loads(Path(features).read_text())["no_endo_excluded"] if noendo_sensitivity else []
    noendo = [c for c in columns if c not in excluded]
    if noendo_sensitivity and (len(set(excluded)) != 9 or not set(excluded) <= set(columns)):
        raise ValueError("The no-endoscopy sensitivity must exclude exactly nine locked fields")
    out = new_output(output)
    predictions, contributions, calibrations = [], [], []
    for fold in range(1, 6):
        train_index = np.flatnonzero(frame.outer_fold.ne(fold))
        test_index = np.flatnonzero(frame.outer_fold.eq(fold))
        train, test = frame.iloc[train_index].reset_index(drop=True), frame.iloc[test_index].reset_index(drop=True)
        ci = inner_splits(train, splits, fold, "clinical")
        model = fit_clinical(train, columns, ci, seed=42, outer_fold=fold)
        vp = _vision_predict(means, applications, train_index, test_index, train.outcome.to_numpy(int))
        clinical = model.predict_proba(test)
        gate = fixed_gate(clinical, vp["p_dino"])
        row = test[["patient_id", "outcome", "outer_fold"]].copy()
        row["cohort"] = "development"
        row["clinical"], row["fusion"] = clinical, gate["probability"]
        for dest, src in [("dino", "p_dino"), ("rn50", "p_rn50"), ("vision", "p_vision")]:
            row[dest] = vp[src]
        if {"human_endoscopy", "human_mri"}.issubset(frame.columns):
            row[["human_endoscopy", "human_mri"]] = test[["human_endoscopy", "human_mri"]]
            row["human_rule"] = fixed_human_rule(test)
            row["human_ridge"] = predict_human(fit_human(train), test)
        contributions.append(_attribution_rows(model, test, gate["probability"],
                                               gate["vision_correction_logit"], "fusion"))
        calibrations.append({"outer_fold": fold, "model": "clinical", "calibration": model.calibration})
        if noendo_sensitivity:
            vi = inner_splits(train, splits, fold, "vision")
            matched_raw = crossfit_raw_clinical(train, noendo, vi, base_seed=42+fold*10000+200,
                                                split_seed_step=100, return_margin=True)
            dino_inner = np.full(len(train), np.nan)
            for fitting, validation in vi:
                pred = _vision_predict(means, applications, train_index[fitting], train_index[validation],
                                      train.iloc[fitting].outcome.to_numpy(int))
                dino_inner[validation] = pred["p_dino"]
            fit = fit_stack(matched_raw, safe_logit(dino_inner), train.outcome.to_numpy(int))
            noendo_model = fit_clinical(train, noendo, ci, seed=42, outer_fold=fold)
            raw = noendo_model.raw_margin(test)
            dino_logit = safe_logit(vp["p_dino"])
            row["ungated_noendo"] = expit(stack_margin(fit, raw, dino_logit))
            contributions.append(_attribution_rows(noendo_model, test, row.ungated_noendo.to_numpy(),
                                None, "ungated_noendo", fit=fit, dino_logit=dino_logit))
        predictions.append(row)
        print(f"Completed outer fold {fold}/5", flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(out / "development_oof_predictions.csv", index=False)
    pd.concat(contributions, ignore_index=True).to_csv(out / "patient_feature_contributions.csv", index=False)
    write_json(out / "calibration.json", calibrations)
    write_json(out / "run_manifest.json", {"analysis": "development_only", "seed": 42,
               "noendo_ungated_sensitivity": noendo_sensitivity, "external_outcomes_accessed": False,
               "input_sha256": {key: sha256(path) for key, path in
                                  [("patients", patients), ("features", features), ("inner_folds", memberships), ("vision", vision)]}})
    return out


def predict_external(dev_patients, features, memberships, dev_vision, external_patients, external_vision, output):
    """Development refit, then masked external application of FINAL models only."""
    dev, columns, splits, means, _ = _load(dev_patients, features, memberships, dev_vision)
    ci = inner_splits(dev, splits, 0, "clinical")
    model = fit_clinical(dev, columns, ci, seed=42, outer_fold=0)
    heads = fit_vision_heads(means, dev.outcome.to_numpy(int))
    human = fit_human(dev) if {"human_endoscopy", "human_mri"}.issubset(dev.columns) else None
    external = pd.read_csv(external_patients, dtype={"patient_id": str})
    validate_patients(external, columns, development=False)
    if set(dev.patient_id) & set(external.patient_id):
        raise ValueError("Development and external patients overlap")
    _, apply = load_vision(external_vision, external.patient_id)
    vp = predict_vision_ensemble(heads, apply)
    row = external[["patient_id"]].copy()
    row["cohort"] = "external"
    row["clinical"] = model.predict_proba(external)
    row["fusion"] = fixed_gate(row.clinical.to_numpy(), vp["p_dino"])["probability"]
    for dest, src in [("vision", "p_vision"), ("dino", "p_dino"), ("rn50", "p_rn50")]:
        row[dest] = vp[src]
    if human is not None:
        row[["human_endoscopy", "human_mri"]] = external[["human_endoscopy", "human_mri"]]
        row["human_ridge"], row["human_rule"] = predict_human(human, external), fixed_human_rule(external)
    out = new_output(output)
    path = out / "external_predictions_masked.csv"
    row.to_csv(path, index=False)
    write_json(out / "prediction_lock.json", {"external_status": "post_hoc",
               "external_outcomes_accessed": False, "predictions_sha256": sha256(path),
               "clinical_calibration": model.calibration, "final_clinical_seed": 10044,
               "input_sha256": {key: sha256(path) for key, path in [("development_patients", dev_patients),
                  ("feature_contract", features), ("inner_folds", memberships), ("development_vision", dev_vision),
                  ("external_patients_masked", external_patients), ("external_vision", external_vision)]}})
    return out


def evaluate_external(predictions, lock_path, labels, output):
    """Verify the masked prediction checksum BEFORE reading sealed outcomes."""
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("external_outcomes_accessed") is not False or lock.get("predictions_sha256") != sha256(predictions):
        raise ValueError("Masked prediction lock is invalid")
    pred = pd.read_csv(predictions, dtype={"patient_id": str})
    if "outcome" in pred or set(pred.cohort) != {"external"}:
        raise ValueError("Expected masked external predictions only")
    outcomes = pd.read_csv(labels, dtype={"patient_id": str})
    if set(outcomes.columns) != {"patient_id", "outcome"} or set(outcomes.patient_id) != set(pred.patient_id):
        raise ValueError("Sealed labels must match exactly; no patient exclusions at evaluation")
    if outcomes.outcome.isna().any() or set(outcomes.outcome.unique()) != {0, 1}:
        raise ValueError("External outcomes must contain both binary classes")
    merged = pred.merge(outcomes, on="patient_id", validate="one_to_one")
    out = new_output(output)
    merged.to_csv(out / "external_predictions_evaluated.csv", index=False)
    write_json(out / "evaluation_manifest.json", {"status": "post_hoc", "model_selection_performed": False,
               "predictions_sha256": sha256(predictions), "labels_sha256": sha256(labels)})
    return out
