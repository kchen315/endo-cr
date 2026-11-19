"""Command-line interface for analysis-ready private data."""
from __future__ import annotations

import argparse
from pathlib import Path



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="endo-cr-analysis 1.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("development", help="Canonical nested development analysis")
    for name in ("patients", "features", "inner-folds", "vision", "output"):
        dev.add_argument(f"--{name}", type=Path, required=True)
    dev.add_argument("--noendo-sensitivity", action="store_true")
    ext = commands.add_parser("predict-external", help="Final development refit; masked external application")
    for name in ("patients", "features", "inner-folds", "vision", "external-patients", "external-vision", "output"):
        ext.add_argument(f"--{name}", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-external", help="Verify prediction lock before joining sealed labels")
    for name in ("predictions", "lock", "labels", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    report = commands.add_parser("report", help="Tables/figures from already-held-out predictions")
    report.add_argument("--predictions", type=Path, nargs="+", required=True)
    report.add_argument("--pathways", type=Path, required=True,
                        help="Prepared evaluation-only pathway and recurrence labels")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--contributions", type=Path)
    report.add_argument("--features", type=Path)
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.command == "development":
        from .pipeline import run_development
        result = run_development(args.patients, args.features, args.inner_folds, args.vision,
                                 args.output, args.noendo_sensitivity)
    elif args.command == "predict-external":
        from .pipeline import predict_external
        result = predict_external(args.patients, args.features, args.inner_folds, args.vision,
                                  args.external_patients, args.external_vision, args.output)
    elif args.command == "evaluate-external":
        from .pipeline import evaluate_external
        result = evaluate_external(args.predictions, args.lock, args.labels, args.output)
    else:
        from .report import make_report
        result = make_report(args.predictions, args.pathways, args.output, args.bootstrap_samples,
                             args.contributions, args.features)
    print(f"Completed: {result}")
