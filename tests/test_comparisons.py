import pandas as pd
import pytest

from endo_cr_analysis.comparisons import summarize


def test_historical_models_must_share_patients_within_each_experiment():
    frame = pd.DataFrame(dict(experiment='screen', model=['a', 'a', 'b', 'b'],
        patient_id=['one', 'two']*2, outcome=[0, 1]*2, probability=[.2, .8, .5, .5]))
    table = summarize(frame).set_index('model')
    assert table.loc['a', 'auroc'] == 1.
    assert table.loc['b', 'auroc'] == .5
    with pytest.raises(ValueError, match='share patients'):
        summarize(frame.iloc[:-1])
