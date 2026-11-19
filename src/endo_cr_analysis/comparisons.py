"""Score saved out-of-fold predictions within each historical comparison cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import new_output
from .evaluation import metric_values


def summarize(predictions):
    ids = ['experiment', 'model', 'patient_id']
    if predictions[ids].isna().any().any() or predictions.duplicated(ids).any():
        raise ValueError("Expected a single held-out score per patient, model and experiment")
    if not predictions.outcome.isin([0, 1]).all() or not predictions.probability.between(0, 1).all():
        raise ValueError("Expected binary outcomes and probabilities in [0, 1]")
    rows = []
    for experiment, group in predictions.groupby('experiment'):
        table = group.pivot(index='patient_id', columns='model', values='probability')
        labels = group.groupby('patient_id').outcome
        if table.isna().any().any() or not labels.nunique().eq(1).all():
            raise ValueError("Models within one comparison must share patients and outcome labels")
        y = labels.first().loc[table.index]
        if len(np.unique(y)) != 2:
            raise ValueError("Comparison cohort requires both outcome classes")
        for model, p in table.items():
            rows.append(dict(experiment=experiment, model=model, n=len(y), **metric_values(y, p)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scores = summarize(pd.read_csv(args.predictions, dtype={'patient_id': str}))
    scores.to_csv(new_output(args.output) / 'model_comparisons.csv', index=False)


if __name__ == '__main__':
    main()
