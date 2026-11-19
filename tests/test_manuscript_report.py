"""Regression checks for the analysis populations and the single decision cutoff."""
import numpy as np
import pandas as pd
import pytest

from endo_cr_analysis.human import select_thresholds
from endo_cr_analysis.report import classification, pathway_tables


def test_balanced_threshold_uses_all_scores_and_prefers_specificity():
    # Two interior optima tie on J. 0.9 is the more specific of the two.
    data = pd.DataFrame(dict(cohort='development', outcome=[1, 0, 1, 0],
                             fusion=[.9, .8, .7, .6], human_mri=[np.nan, 1., .5, .75]))
    assert select_thresholds(data) == .9
    with pytest.raises(ValueError):
        select_thresholds(data.assign(fusion=[np.nan, .8, .7, .6]))


def test_pathway_denominators_and_missing_assessments():
    frame = pd.DataFrame(dict(cohort='development', outcome=[1, 0, 0, 0],
        pathway=['immediate_surgery', 'watch_and_wait', 'watch_and_wait', 'other'],
        recurrence_within_2y=[0, 1, 0, 0], human_endoscopy=[2/3, 0, 0, np.nan],
        human_mri=[np.nan, .75, 1., 0.], human_ridge=[np.nan, .3, .1, np.nan], fusion=[.51, .5, .4, .3]))
    scores, events = pathway_tables(frame, .5)
    assert scores.query("model == 'fusion' and group == 'all'").n.item() == 4
    assert scores.query("model == 'fusion' and group == 'near_complete'").n.item() == 2
    assert scores.query("model == 'human_ridge_at_multimodal_cutoff'").n.item() == 1
    recurrence = events[events.group.str.startswith('WW')].iloc[0]
    assert (recurrence.numerator, recurrence.denominator) == (0, 1)
    metrics = classification([1, 1], [True, True])
    assert round(metrics['sensitivity_low'], 3) == .342
    assert np.isnan(metrics['specificity'])
