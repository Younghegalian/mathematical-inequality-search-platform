#!/usr/bin/env python3
"""Generate static HTML reports from AISIS traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.reporting import generate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AISIS HTML reports")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "data" / "reports"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index" / "aisis.sqlite"))
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--max-run-pages", type=int, default=250)
    parser.add_argument("--scan-limit", type=int, default=5000)
    parser.add_argument("--promotion-limit", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = None if args.no_index else Path(args.index)
    result = generate_reports(
        Path(args.data_dir),
        Path(args.out),
        index_path=index_path,
        max_run_pages=args.max_run_pages,
        scan_limit=args.scan_limit,
        promotion_limit=args.promotion_limit,
    )
    print(f"Index: {result['index_path']}")
    print(f"Runs indexed: {result['run_report_count']}")
    print(f"Run pages generated: {result['generated_run_pages']}")
    print(f"Run report dir: {result['run_report_dir']}")
    print(f"Index source: {result['index_source']}")
    print(f"Promoted candidates: {result['promoted_candidates']}")
    print(f"Promotion queue: {result['promotion_queue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
