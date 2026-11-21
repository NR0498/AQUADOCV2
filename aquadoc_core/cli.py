"""Command-line entry point for the AquaDoc backend pipeline."""

import argparse
import json
import sqlite3
import sys
from typing import Callable, Sequence

from .ingestion import fetch_sentinel2_chip
from .pipeline import analyze_multispectral_scene
from .persistence import AnalysisRunRepository, DEFAULT_DATABASE_PATH
from .schemas import AnalysisResult, MultispectralScene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aquadoc_core.cli",
        description="Fetch and analyze multispectral satellite imagery for water-quality signals.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the complete Sentinel-2 analysis path.")
    analyze.add_argument("--lat", type=float, required=True, help="Latitude in decimal degrees.")
    analyze.add_argument("--lon", type=float, required=True, help="Longitude in decimal degrees.")
    analyze.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    analyze.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    analyze.add_argument(
        "--size-m",
        type=int,
        default=3000,
        help="Width and height of the square satellite chip in metres (default: 3000).",
    )
    analyze.add_argument(
        "--db",
        default=str(DEFAULT_DATABASE_PATH),
        help=f"SQLite database path (default: {DEFAULT_DATABASE_PATH}).",
    )
    analyze.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of indented JSON.",
    )
    list_runs = subparsers.add_parser("list-runs", help="List saved analysis runs.")
    list_runs.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")
    list_runs.add_argument("--limit", type=int, default=20, help="Maximum runs to return (default: 20).")

    show_run = subparsers.add_parser("show-run", help="Show one saved analysis run.")
    show_run.add_argument("run_id", help="UUID printed by the analyze command.")
    show_run.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path.")

    return parser


def run_analyze(
    args: argparse.Namespace,
    fetcher: Callable[..., MultispectralScene],
    analyzer: Callable[[MultispectralScene], AnalysisResult],
) -> AnalysisResult:
    scene = fetcher(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=args.end,
        size_m=args.size_m,
    )
    return analyzer(scene)


def main(
    argv: Sequence[str] | None = None,
    fetcher: Callable[..., MultispectralScene] = fetch_sentinel2_chip,
    analyzer: Callable[[MultispectralScene], AnalysisResult] = analyze_multispectral_scene,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "analyze":
            repository = AnalysisRunRepository(args.db)
            result = run_analyze(args, fetcher, analyzer)
            payload = repository.save(result, args.start, args.end, args.size_m).to_dict()
        elif args.command == "list-runs":
            repository = AnalysisRunRepository(args.db)
            payload = {"runs": [run.to_dict() for run in repository.list(args.limit)]}
        elif args.command == "show-run":
            repository = AnalysisRunRepository(args.db)
            run = repository.get(args.run_id)
            if run is None:
                raise RuntimeError(f"Analysis run '{args.run_id}' was not found")
            payload = run.to_dict()
        else:
            raise RuntimeError(f"Unknown command: {args.command}")
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    indent = None if getattr(args, "compact", False) else 2
    print(json.dumps(payload, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
