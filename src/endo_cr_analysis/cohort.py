"""Table 1 from a prepared patient-level descriptive table."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import new_output


CHARACTERISTICS = [
    ("Patient characteristics", "age", "Age at diagnosis, years", None),
    ("Patient characteristics", "sex", "Sex", ["Female", "Male"]),
    ("Patient characteristics", "bmi", "Body mass index, kg/m²", None),
    ("Baseline disease characteristics", "ct", "Clinical T category", ["T1–T2", "T3", "T4"]),
    ("Baseline disease characteristics", "cn", "Clinical N category", ["N0", "N1", "N2"]),
    ("Baseline disease characteristics", "location", "Tumor location", ["Low", "Mid", "High"]),
    ("Baseline disease characteristics", "distance", "Distance from anal verge, cm", None),
    ("Baseline disease characteristics", "mmr", "Mismatch-repair status", ["Proficient", "Deficient"]),
    ("Baseline disease characteristics", "emvi", "Extramural vascular invasion", ["Present", "Absent"]),
    ("Baseline disease characteristics", "cea", "Baseline CEA", None),
    ("Treatment and management", "pathway", "Management pathway", ["Immediate surgery", "Watch-and-wait", "Other, as recorded"]),
    ("Treatment and management", "sequence", "Treatment sequence", ["Consolidation", "Induction"]),
    ("Treatment and management", "cycles", "Recorded systemic-treatment cycles", None),
    ("Treatment and management", "fractions", "Radiotherapy fractions", ["5", "25", "28", "Other, as recorded"]),
    ("Post-treatment restaging", "endo", "Endoscopic assessment", ["Complete", "Near-complete", "Partial/incomplete", "No response/progression"]),
    ("Post-treatment restaging", "mrtrg", "MRI tumor regression grade", ["mrTRG 1", "mrTRG 2", "mrTRG 3", "mrTRG 4", "mrTRG 5"]),
    ("Image and assessment availability", "images", "Images per patient", None),
    ("Image and assessment availability", "days", "Endoscopy after treatment anchor, days", None),
]

def summarize(patients):
    if (patients.patient_id.duplicated().any() or patients.patient_id.isna().any()
            or not patients.cohort.isin(['development', 'external']).all()
            or not patients.outcome.isin([0, 1]).all()):
        raise ValueError("Expected one row per patient with a cohort and binary outcome")
    rows = []
    for (cohort, outcome), group in patients.groupby(['cohort', 'outcome']):
        meta = dict(cohort=cohort, outcome=outcome, n=len(group))
        for _, key, label, levels in CHARACTERISTICS:
            values = group[key]
            if levels is None:
                observed = pd.to_numeric(values.dropna(), errors='raise')
                if not np.isfinite(observed).all():
                    raise ValueError(f"Nonfinite descriptive value in {key}")
                q1, median, q3 = observed.quantile([.25, .5, .75])
                rows.append(dict(**meta, field=key, characteristic=label, observed=len(observed),
                                 q1=q1, median=median, q3=q3))
            else:
                if not values.dropna().isin(levels).all():
                    raise ValueError(f"Unexpected prepared category in {key}")
                for level in [*levels, 'Missing/unknown']:
                    count = values.isna().sum() if level == 'Missing/unknown' else values.eq(level).sum()
                    rows.append(dict(**meta, field=key, characteristic=level, count=count,
                                     percent=100*count/len(group)))
        rows.append(dict(**meta, field='images', characteristic='Total endoscopic images',
                         count=pd.to_numeric(group.images, errors='raise').sum(min_count=len(group))))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.patients, dtype={'patient_id': str, 'fractions': str})
    result = summarize(data)
    result.to_csv(new_output(args.output) / 'table1.csv', index=False)


if __name__ == '__main__':
    main()
