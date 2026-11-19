"""Fixed human benchmark and development-only operating-point selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["human_endoscopy", "human_mri"]


def fit_human(train):
    eligible = train.dropna(subset=FEATURES)
    if set(eligible.outcome.unique()) != {0, 1}:
        raise ValueError("Evaluable human training patients need both outcome classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., penalty="l2",
                          solver="lbfgs", max_iter=2000, random_state=42))
    model.fit(eligible[FEATURES].to_numpy(float), eligible.outcome.to_numpy(int))
    return model


def predict_human(model, frame):
    output = np.full(len(frame), np.nan)
    mask = frame[FEATURES].notna().all(axis=1).to_numpy()
    if mask.any():
        output[mask] = model.predict_proba(frame.loc[mask, FEATURES].to_numpy(float))[:, 1]
    return output


def fixed_human_rule(frame):
    """Prepared ordinal scores: endoscopy complete=1; mrTRG1/2=1/.75."""
    output = np.full(len(frame), np.nan)
    mask = frame[FEATURES].notna().all(axis=1).to_numpy()
    output[mask] = (frame.loc[mask, "human_endoscopy"].eq(1)
                    & frame.loc[mask, "human_mri"].ge(.75)).astype(int)
    return output



def select_thresholds(development):
    """Maximize Youden J on all development OOF scores, using exact integer ties."""
    if set(development.cohort) != {"development"}:
        raise ValueError("Threshold selection is development-only")
    y = development.outcome.to_numpy()
    scores = development.fusion.to_numpy(float)
    if set(y) != {0, 1} or not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("All development patients need binary outcomes and valid fusion probabilities")
    positive, negative = int(y.sum()), int((y == 0).sum())
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    def rank(threshold):
        decision = scores >= threshold
        tp = int(decision[y == 1].sum())
        fp = int(decision[y == 0].sum())
        return (tp * negative - fp * positive, -fp, threshold)
    return float(max(candidates, key=rank))
