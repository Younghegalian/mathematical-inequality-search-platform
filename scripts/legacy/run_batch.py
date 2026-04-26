#!/usr/bin/env python3
"""Legacy helper for deterministic bounded AISIS search batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.batch import run_batch
from ns.targets import TARGETS


def parse_ints(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded AISIS batch")
    parser.add_argument("--budget", type=int, default=5, help="Total number of runs to execute")
    parser.add_argument("--targets", nargs="+", choices=sorted(TARGETS), default=None)
    parser.add_argument("--depths", nargs="+", default=["2", "3", "5"])
    parser.add_argument("--widths", nargs="+", default=["20", "50"])
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    depths = parse_ints(args.depths)
    widths = parse_ints(args.widths)
    manifest = run_batch(
        budget=args.budget,
        targets=args.targets,
        depths=depths,
        widths=widths,
        data_dir=Path(args.data_dir),
        dry_run=args.dry_run,
    )

    print(f"Batch: {manifest['batch_id']}")
    print(f"Budget: {manifest['budget']}")
    print(f"Dry run: {manifest['dry_run']}")
    print()
    print("Jobs:")
    for job in manifest["jobs"]:
        print(
            f"  {job['index']:>3}. {job['target_name']} "
            f"depth={job['depth']} width={job['width']}"
        )

    if manifest["runs"]:
        print()
        print("Completed:")
        for run in manifest["runs"]:
            print(
                f"  {run['index']:>3}. {run['run_id']} "
                f"closure={run['best_closure']} score={run['best_score']}"
            )
        print()
        print(f"Manifest: {manifest['manifest_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
