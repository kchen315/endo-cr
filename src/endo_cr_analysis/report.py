"""Manuscript estimates from held-out patient predictions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from .contracts import new_output
from .delong import auc_summary, paired_delong
from .evaluation import SCOPES, calibration_tables, metric_values, validate_predictions
from .figures import _curve_figure, make_attribution_figure, plot_pathways
from .human import fixed_human_rule, select_thresholds


def classification(outcome, positive):
    y = np.asarray(outcome, int)
    positive = np.asarray(positive, bool)
    tp = int(positive[y == 1].sum())
    tn = int((~positive[y == 0]).sum())
    cases, controls = int(y.sum()), int((y == 0).sum())
    result = dict(n=len(y), tp=tp, fp=controls - tn, tn=tn, fn=cases - tp)
    for name, count, total in [('accuracy', tp + tn, len(y)), ('sensitivity', tp, cases),
                               ('specificity', tn, controls)]:
        p = count / total if total else np.nan
        z = norm.ppf(.975)
        if total:
            center = (total * p + z*z/2) / (total + z*z)
            half = z * np.sqrt(total*p*(1-p) + z*z/4) / (total + z*z)
        else:
            center = half = np.nan
        result.update({name: p, name+'_low': center-half, name+'_high': center+half})
    return result


def performance(frame):
    rows, tests = [], []
    primary = {'vision_vs_endoscopy': ('vision', 'human_endoscopy'),
               'fusion_vs_human_restaging': ('fusion', 'human_ridge')}
    for scope, (candidate, reference) in primary.items():
        models = SCOPES[scope]
        paired = frame.dropna(subset=list(models))
        y = paired.outcome
        for model in models:
            summary = auc_summary(y, paired[model])
            rows.append(dict(scope=scope, model=model, **summary, **{
                k: v for k, v in metric_values(y, paired[model], probabilistic=model not in
                    ('human_endoscopy', 'human_mri')).items() if k != 'auroc'}))
        tests.append(dict(scope=scope, model=candidate, comparator=reference,
                          **paired_delong(y, paired[candidate], paired[reference])))
    if frame.cohort.iloc[0] == 'development' and 'ungated_noendo' in frame:
        if frame.ungated_noendo.isna().any():
            raise ValueError("Incomplete development sensitivity predictions")
        rows.append(dict(scope='without_endoscopic_assessment', model='ungated_noendo',
                         n=len(frame), **metric_values(frame.outcome, frame.ungated_noendo)))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def pathway_tables(frame, threshold):
    surgery = frame.pathway.eq('immediate_surgery')
    ww = frame.pathway.eq('watch_and_wait')
    near = np.isclose(frame.human_endoscopy, 2/3) | frame.human_mri.eq(.75)
    rows = []
    for group, subset in [('all', frame), ('surgery', frame[surgery]), ('watch_and_wait', frame[ww]),
                          ('near_complete', frame[near])]:
        rows.append(dict(group=group, model='fusion', threshold=threshold,
                         **classification(subset.outcome, subset.fusion >= threshold)))
    # Both candidate interpretations of the manuscript's subgroup comparator are labeled.
    eligible = frame[near].dropna(subset=['human_ridge'])
    for model, decision, cutoff in [
        ('fixed_complete_endoscopy_and_mrTRG_1_or_2', fixed_human_rule(eligible), np.nan),
        ('human_ridge_at_multimodal_cutoff', eligible.human_ridge >= threshold, threshold)
    ]:
        rows.append(dict(group='near_complete', model=model, threshold=cutoff,
                         **classification(eligible.outcome, decision)))
    events = []
    if frame.cohort.iloc[0] == 'development':
        for name, mask, positive in [
            ('Surgical pCR', surgery & frame.outcome.eq(1), True),
            ('WW recurrence within 2 years', ww & frame.recurrence_within_2y.eq(1), False),
            ('Surgical residual disease', surgery & frame.outcome.eq(0), True),
            ('Sustained WW response', ww & frame.outcome.eq(1), False)
        ]:
            events.append(dict(group=name, numerator=int(((frame.loc[mask, 'fusion'] >= threshold)
                                  == positive).sum()), denominator=int(mask.sum()), predicted_response=positive))
    return pd.DataFrame(rows), pd.DataFrame(events)


def attribution_summary(values, features, development, samples):
    mapping = dict(features['feature_groups'])
    mapping['DINO vision contribution'] = 'Endoscopic image model'
    values = values.assign(group=values.feature.map(mapping))
    if values.group.isna().any() or not np.isfinite(values.contribution).all():
        raise ValueError("Each contribution needs a known feature and finite value")
    ordered = development.sort_values('patient_id')
    rng = np.random.default_rng(42)
    groups = ordered.groupby(['outer_fold', 'outcome'], sort=True).indices.values()
    draws = np.concatenate([rng.choice(i, (samples, len(i)), replace=True) for i in groups], axis=1)
    rows = []
    for model, part in values.groupby('model', sort=False):
        expected = set(mapping)
        if model == 'ungated_noendo':
            expected -= set(features['no_endo_excluded'])
        elif model != 'fusion':
            raise ValueError("Unknown attribution model")
        if (set(part.patient_id) != set(ordered.patient_id) or set(part.feature) != expected
                or part.duplicated(['patient_id', 'feature']).any()
                or len(part) != len(expected) * len(ordered)):
            raise ValueError("Incomplete patient-by-feature attribution table")
        wide = part.groupby(['patient_id', 'group']).contribution.sum().unstack().loc[ordered.patient_id]
        for group, signed in wide.items():
            if group == 'Initial diversion status':
                continue
            absolute = np.abs(signed.to_numpy())
            lo, hi = np.quantile(absolute[draws].mean(axis=1), [.025, .975])
            rows.append(dict(model=model, feature=group, mean_abs_contribution=absolute.mean(), ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def make_report(prediction_paths, pathways, output, bootstrap_samples=2000, contributions=None, features=None):
    if bootstrap_samples < 1 or (contributions is not None and features is None):
        raise ValueError("Supply positive bootstrap count and a feature contract for contributions")
    def read(path):
        return pd.read_csv(path, dtype={'patient_id': str}, float_precision='round_trip')
    frame = validate_predictions(pd.concat([read(p) for p in prediction_paths], ignore_index=True))
    paths = read(pathways)
    if set(paths.patient_id) != set(frame.patient_id):
        raise ValueError("Pathway table must cover exactly the evaluated patients")
    frame = frame.merge(paths[['patient_id', 'pathway', 'recurrence_within_2y']], validate='one_to_one')
    if not frame.pathway.isin(['immediate_surgery', 'watch_and_wait', 'other']).all():
        raise ValueError("Unrecognized treatment pathway")
    events = frame.recurrence_within_2y
    if (not events.dropna().isin([0, 1]).all() or
        (events.eq(1) & (~frame.pathway.eq('watch_and_wait') | frame.outcome.ne(0))).any() or
        events[frame.cohort.eq('development') & frame.pathway.eq('watch_and_wait') & frame.outcome.eq(0)].isna().any()):
        raise ValueError("Supply documented two-year event status for each negative development WW endpoint")
    required = ['vision', 'dino', 'rn50', 'clinical', 'fusion']
    if frame[required].isna().any().any():
        raise ValueError("Every patient needs complete model predictions")
    eligible = frame[['human_endoscopy', 'human_mri']].notna().all(axis=1)
    if not eligible.equals(frame.human_ridge.notna()):
        raise ValueError("The restaging benchmark requires both assessments")
    development = frame[frame.cohort == 'development']
    threshold = select_thresholds(development)
    tables = {}
    for cohort, part in frame.sort_values('patient_id').groupby('cohort'):
        scores, tests = performance(part)
        classes, opportunities = pathway_tables(part, threshold)
        bins = 5 if cohort == 'development' else 3
        points, calibration = calibration_tables(part, ['fusion', 'clinical', 'human_ridge'], bins,
                            bootstrap_samples, 40042 if cohort == 'development' else 50042)
        for name, table in [('performance', scores), ('paired_delong', tests), ('classification', classes),
                            ('pathway_counts', opportunities), ('calibration_groups', points),
                            ('calibration', calibration)]:
            if not table.empty:
                tables.setdefault(name, []).append(table.assign(cohort=cohort))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    if contributions is not None:
        tables['attribution'] = attribution_summary(read(contributions), json.loads(Path(features).read_text()),
                                                     development, bootstrap_samples)
    out = new_output(output)
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), index=False)
    for index, scope in enumerate(SCOPES, start=2):
        _curve_figure(frame, scope, out, f'Figure{index}', 200)
    plot_pathways(tables, out)
    if contributions is not None:
        models = ['fusion', 'ungated_noendo'] if 'ungated_noendo' in tables['attribution'].model.values else ['fusion']
        make_attribution_figure(tables['attribution'], out, models=models, stem='Figure5')
    return out
