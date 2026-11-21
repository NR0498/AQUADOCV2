"""Command-line interface for the shelved lab-measurement matching workflow.

Run with ``python -m research.lab_matching.cli <command>``. These commands are
not part of the deployed product; they are retained for research reference.
"""

import argparse
import json
import sqlite3
import sys
from typing import Sequence

from aquadoc_core.persistence import DEFAULT_DATABASE_PATH

from .csv_import import import_measurements_csv
from .measurements import MeasurementRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-lab-matching",
        description="Import field/lab measurements, match them to Sentinel-2 scenes, and train baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_measurements = subparsers.add_parser(
        "import-measurements", help="Validate and import field/laboratory measurements from CSV."
    )
    import_measurements.add_argument("csv_path", help="Measurement CSV path.")
    import_measurements.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    measurement_counts = subparsers.add_parser(
        "measurement-counts", help="Count saved sampling events and measurements."
    )
    measurement_counts.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    match_scenes = subparsers.add_parser(
        "match-scenes", help="Match verified measurements to nearby Sentinel-2 scenes."
    )
    match_scenes.add_argument("--parameter", required=True, help="Measurement parameter to match.")
    match_scenes.add_argument("--max-days", type=int, default=3, help="Maximum days before/after sampling.")
    match_scenes.add_argument("--chip-size-m", type=int, default=1000, help="Satellite chip width in metres.")
    match_scenes.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    export_training = subparsers.add_parser(
        "export-training", help="Export matched rows as a deterministic training CSV."
    )
    export_training.add_argument("--parameter", required=True, help="Parameter to export.")
    export_training.add_argument("--output", required=True, help="Output CSV path.")
    export_training.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    review_matches = subparsers.add_parser(
        "review-matches", help="List satellite matches awaiting human review."
    )
    review_matches.add_argument("--parameter", help="Optional parameter filter.")
    review_matches.add_argument(
        "--status", choices=("pending", "approved", "rejected"), default="pending",
        help="Review status to list (default: pending).",
    )
    review_matches.add_argument("--limit", type=int, default=100, help="Maximum matches to return.")
    review_matches.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    review_match = subparsers.add_parser(
        "review-match", help="Approve or reject one satellite/measurement match."
    )
    review_match.add_argument("match_id", help="Match ID returned by review-matches.")
    review_match.add_argument("--status", choices=("approved", "rejected"), required=True)
    review_match.add_argument("--note", help="Optional reason for the review decision.")
    review_match.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    train = subparsers.add_parser(
        "train-baseline", help="Train and evaluate a grouped Random Forest baseline."
    )
    train.add_argument("--dataset", required=True, help="Exported training CSV path.")
    train.add_argument("--model", required=True, help="Output pickle model path.")
    train.add_argument("--report", required=True, help="Output JSON evaluation report path.")
    train.add_argument("--test-size", type=float, default=0.25, help="Fraction of location groups held out.")
    train.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting and training.")

    preprocess = subparsers.add_parser(
        "preprocess-nla2017", help="Normalize and audit local NLA chlorophyll observations."
    )
    preprocess.add_argument("--raw-dir", default="data/raw/nla2017")
    preprocess.add_argument("--output-dir", default="data/processed/nla2017")
    preprocess.add_argument("--manifest", default="data/manifests/nla2017_raw.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "import-measurements":
            payload = import_measurements_csv(args.csv_path, MeasurementRepository(args.db)).to_dict()
        elif args.command == "measurement-counts":
            payload = MeasurementRepository(args.db).counts()
        elif args.command == "match-scenes":
            from .training_pipeline import match_measurements_to_scenes

            payload = match_measurements_to_scenes(
                args.db, args.parameter, max_days=args.max_days, chip_size_m=args.chip_size_m
            ).to_dict()
        elif args.command == "export-training":
            from .training_pipeline import export_training_dataset

            payload = export_training_dataset(args.db, args.parameter, args.output)
        elif args.command == "review-matches":
            from .training_pipeline import TrainingMatchRepository

            payload = {
                "matches": TrainingMatchRepository(args.db).review_matches(
                    parameter=args.parameter, review_status=args.status, limit=args.limit
                )
            }
        elif args.command == "review-match":
            from .training_pipeline import TrainingMatchRepository

            payload = TrainingMatchRepository(args.db).review_match(args.match_id, args.status, args.note)
        elif args.command == "preprocess-nla2017":
            from .nla_preprocessing import preprocess

            payload = preprocess(args.raw_dir, args.output_dir, args.manifest)
        else:
            from .training_pipeline import train_baseline

            payload = train_baseline(
                args.dataset, args.model, args.report, test_size=args.test_size, random_state=args.seed
            )
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
