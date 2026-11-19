"""Code-native manuscript plots from private patient-level prediction inputs.

These functions contain no image curation, identifiers, image paths, or image
loading. Patient-level source files and generated figures must stay private
until the study team has reviewed them for release.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .evaluation import SCOPES, metric_values, paired_data

LABELS = {
    "human_endoscopy": "Endoscopic assessment", "human_mri": "mrTRG",
    "human_ridge": "mrTRG + endoscopic assessment", "human_rule": "mrTRG + human endoscopy",
    "vision": "ViT + RN50 vision ensemble", "dino": "GastroNet-5M ViT-B/14",
    "rn50": "GastroNet-5M RN50", "clinical": "Clinical-only model",
    "fusion": "Multimodal model", "all": "Watch-and-wait for all",
    "none": "Watch-and-wait for none",
}
# Blue represents automated models; green represents human assessment.
COLORS = {"human_endoscopy": "#00843D", "human_mri": "#65A57A", "human_ridge": "#005A30",
          "human_rule": "#00843D", "vision": "#003B71", "dino": "#0078BF",
          "rn50": "#73A9D3", "clinical": "#64A2C8", "fusion": "#0072CE",
          "all": "#92999E", "none": "#454B50"}
LINESTYLES = {"human_endoscopy": "--", "human_mri": ":", "human_ridge": "-.",
              "vision": "-", "dino": "--", "rn50": ":", "clinical": "--",
              "fusion": "-", "all": ":", "none": "-"}


def _save(fig, directory: Path, stem: str, dpi: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ["pdf", "svg", "png"]:
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _clean(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def _curve_figure(data: pd.DataFrame, scope: str, directory: Path, stem: str, dpi: int):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
    requested = SCOPES[scope]
    for column, cohort in enumerate(["development", "external"]):
        subset = data.loc[data["cohort"].eq(cohort)]
        models = [model for model in requested if model in subset and subset[model].notna().any()]
        paired = paired_data(subset, models) if models else subset.iloc[:0]
        for row, curve in enumerate(["ROC", "PR"]):
            axis = axes[row, column]
            letter = "ABCD"[row * 2 + column]
            title = "Receiver Operating Characteristic Curves" if curve == "ROC" else "Precision–Recall Curves"
            axis.set_title(f"{letter}   {cohort.title()} {title}", loc="left", fontsize=10, fontweight="bold", pad=13)
            if paired.empty or paired["outcome"].nunique() != 2:
                axis.text(.5, .5, "Not available", ha="center", va="center", transform=axis.transAxes)
                _clean(axis)
                continue
            y = paired["outcome"].to_numpy(dtype=int)
            for model in models:
                score = paired[model].to_numpy(dtype=float)
                metric = metric_values(y, score, probabilistic=False)
                if curve == "ROC":
                    x, value, _ = roc_curve(y, score)
                    label = f"{LABELS[model]}  {metric['auroc']:.3f}"
                else:
                    value, x, _ = precision_recall_curve(y, score)
                    label = f"{LABELS[model]}  {metric['auprc']:.3f}"
                axis.plot(x, value, color=COLORS[model], linestyle=LINESTYLES[model], linewidth=2, label=label)
            if curve == "ROC":
                axis.plot([0, 1], [0, 1], color="#B7BDC1", linestyle=":", linewidth=1)
                axis.set(xlabel="1 − Specificity", ylabel="Sensitivity")
            else:
                axis.axhline(y.mean(), color="#B7BDC1", linestyle=":", linewidth=1)
                axis.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)")
            axis.set(xlim=(0, 1), ylim=(0, 1.02))
            axis.legend(loc="lower right" if curve == "ROC" else "lower left", fontsize=8, frameon=False)
            _clean(axis)
    fig.subplots_adjust(left=.07, right=.99, top=.94, bottom=.07, hspace=.28, wspace=.24)
    return _save(fig, directory, stem, dpi)


def make_attribution_figure(grouped, out_dir, models, stem="Figure5", dpi=200):
    """Plot the selected models' mean absolute grouped contributions and intervals."""
    required = {"model", "feature", "mean_abs_contribution"}
    if not required <= set(grouped) or grouped.empty:
        raise ValueError(f"Attribution table must contain {sorted(required)} and at least one row")
    if models is not None:
        missing = set(models) - set(grouped["model"])
        if missing:
            raise ValueError(f"Attribution model panels not found: {sorted(missing)}")
        grouped = pd.concat([grouped.loc[grouped["model"].eq(model)] for model in models], ignore_index=True)
    if grouped.empty:
        raise ValueError("No attribution groups remain after display filtering")
    models = list(grouped["model"].drop_duplicates())
    panel_labels = {"fusion": "With human endoscopy · fixed gate",
                    "ungated_noendo": "Without human endoscopy · ungated"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, len(models), figsize=(6.3 * len(models), 6.0), squeeze=False)
        maximum = grouped["ci_high"].max() if "ci_high" in grouped else grouped["mean_abs_contribution"].max()
        for index, model in enumerate(models):
            axis = axes[0, index]
            subset = grouped.loc[grouped["model"].eq(model)].sort_values("mean_abs_contribution")
            values = subset["mean_abs_contribution"].to_numpy()
            errors = None
            if {"ci_low", "ci_high"} <= set(subset):
                errors = np.maximum(0, np.vstack([values - subset["ci_low"], subset["ci_high"] - values]))
            colors = ["#0072CE" if any(word in str(feature).lower() for word in ["image", "vision"])
                      else "#00843D" for feature in subset["feature"]]
            axis.barh(subset["feature"], values, xerr=errors, color=colors, capsize=2, error_kw={"linewidth": 1})
            axis.set_title(f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[index]}   {panel_labels.get(model, model)}",
                           loc="left", fontsize=11)
            axis.set(xlabel="Mean absolute contribution (log-odds)", xlim=(0, max(maximum * 1.12, .01)))
            _clean(axis)
        fig.tight_layout(w_pad=3)
        return _save(fig, Path(out_dir), stem, dpi)


def plot_pathways(tables, directory):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    points = tables['calibration_groups']
    for axis, cohort in zip(axes[0], ['development', 'external']):
        for model in ['fusion', 'human_ridge']:
            p = points[(points.cohort == cohort) & (points.model == model)]
            y = p.observed_response.to_numpy()
            if len(p):
                axis.errorbar(p.mean_probability, y, yerr=np.maximum(0, [y-p.ci_low, p.ci_high-y]),
                              marker='o', capsize=3, label=LABELS[model])
        axis.plot([0, 1], [0, 1], ':', color='gray')
        axis.set(title=cohort.title(), xlim=(0, 1), ylim=(-.03, 1.03),
                 xlabel='Mean predicted probability', ylabel='Observed proportion')
        if cohort in points.cohort.values:
            axis.legend(frameon=False, fontsize=8)
    events = tables['pathway_counts']
    axis = axes[1, 0]
    bars = axis.barh(events.group, events.numerator / events.denominator.replace(0, np.nan))
    axis.bar_label(bars, labels=[f'{r.numerator}/{r.denominator}' for r in events.itertuples()], padding=2)
    axis.set(xlim=(0, 1.1), xlabel='Proportion in observed pathway group',
             title='Development pathway classifications')
    axis.invert_yaxis()
    m = tables['classification'].query("cohort == 'development' and group == 'all' and model == 'fusion'").iloc[0]
    matrix = np.array([[m.tp, m.fn], [m.fp, m.tn]], dtype=int)
    axis = axes[1, 1]
    axis.imshow(matrix, cmap='Blues')
    for (i, j), count in np.ndenumerate(matrix):
        axis.text(j, i, str(count), ha='center', va='center',
                  color='white' if count > matrix.max() * .55 else 'black')
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['TR', 'Incomplete'], yticklabels=['TR', 'Incomplete'],
             xlabel='Predicted outcome', ylabel='Observed outcome', title='Development cohort')
    fig.tight_layout()
    _save(fig, directory, 'Figure4', 200)
