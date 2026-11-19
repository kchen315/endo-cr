"""Validate analysis-ready inputs; no cohort construction or record cleaning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def new_output(path):
    """Never write private analytic products into the public source directory."""
    path = Path(path).resolve()
    module = Path(__file__).resolve().parent
    in_checkout = any((parent / "src/endo_cr_analysis/contracts.py").is_file()
                      and (parent / "config/clinical_features.json").is_file()
                      for parent in (path, *path.parents))
    if path == module or module in path.parents or in_checkout:
        raise ValueError("Inputs and outputs belong outside the public code package")
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_patients(frame, feature_columns, development=True):
    required = ["patient_id", *feature_columns]
    if development:
        required += ["outcome", "outer_fold"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    ids = frame.patient_id
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Exactly one row per nonempty pseudonymous patient_id required")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicated feature names")
    forbidden = {"patient_id", "outcome", "outer_fold", "cohort", "inner_fold", "site", "pathway", "recurrence_within_2y", "human_mri", "human_endoscopy"}
    if forbidden.intersection(feature_columns):
        raise ValueError("Identifiers, outcomes, and split metadata cannot be features")
    if development:
        if frame.outcome.isna().any() or set(frame.outcome.unique()) != {0, 1}:
            raise ValueError("Development outcomes must contain both binary classes")
        if frame.outer_fold.isna().any() or set(frame.outer_fold.unique()) != {1, 2, 3, 4, 5}:
            raise ValueError("Supply the canonical five outer folds; do not regenerate them")
        for fold in range(1, 6):
            if set(frame.loc[frame.outer_fold.eq(fold), "outcome"].unique()) != {0, 1}:
                raise ValueError("Every outer fold must contain both outcome classes")
    elif "outcome" in frame:
        raise ValueError("External prediction input must not contain outcomes")


def load_feature_columns(path):
    value = json.loads(Path(path).read_text())
    return value["features"] if isinstance(value, dict) else value


def inner_splits(train, memberships, context, branch):
    """Use supplied canonical memberships, never re-split from re-identified IDs."""
    required = {"patient_id", "outer_context_fold", "branch", "inner_fold"}
    if not required.issubset(memberships.columns):
        raise ValueError(f"Inner memberships need {sorted(required)}")
    selected = memberships.loc[memberships.outer_context_fold.eq(context)
                               & memberships.branch.eq(branch)]
    if selected.patient_id.duplicated().any() or set(selected.patient_id) != set(train.patient_id):
        raise ValueError("Inner folds must cover exactly the outer-training patients")
    values = selected.set_index("patient_id").loc[train.patient_id, "inner_fold"].to_numpy()
    if set(values) != {1, 2, 3}:
        raise ValueError("Exactly three supplied inner folds required")
    splits = [(np.flatnonzero(values != j), np.flatnonzero(values == j)) for j in (1, 2, 3)]
    for fitting, valid in splits:
        if set(train.iloc[fitting].outcome.unique()) != {0, 1} or len(valid) == 0:
            raise ValueError("Inner training folds require both classes and nonempty validation")
    return splits


def load_vision(path, patient_ids):
    """Load numeric prepared mean embeddings; disallow pickle/object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"patient_ids", "dino_identity", "rn50_identity", "dino_application", "rn50_application"}
        if not required.issubset(archive.files):
            raise ValueError(f"Vision archive must contain {sorted(required)}")
        ids = archive["patient_ids"].astype(str)
        if len(np.unique(ids)) != len(ids) or set(ids) != set(patient_ids):
            raise ValueError("Vision IDs must match patients exactly, without duplicates")
        lookup = {v: i for i, v in enumerate(ids)}
        order = np.array([lookup[str(v)] for v in patient_ids])
        means, applications = {}, {}
        for name in ("dino", "rn50"):
            identity = archive[f"{name}_identity"]
            application = archive[f"{name}_application"]
            if (identity.ndim != 2 or identity.shape[0] != len(ids)
                    or application.shape != (6, 16, len(ids), identity.shape[1])):
                raise ValueError("Expect identity [N,D] and application [6,16,N,D] embeddings")
            if not np.isfinite(identity).all() or not np.isfinite(application).all():
                raise ValueError("Embeddings must be finite")
            means[name] = identity[order]
            applications[name] = application[:, :, order]
    return means, applications
