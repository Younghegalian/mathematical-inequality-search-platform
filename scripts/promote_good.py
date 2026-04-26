#!/usr/bin/env python3
"""Export closure-good or nonlinear-term-relevant AISIS candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.promotion import promote_good_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote verification-worthy AISIS candidates")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--include-unknown", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out) if args.out else None
    result = promote_good_candidates(
        Path(args.data_dir),
        out_dir=out_dir,
        limit=args.limit,
        include_unknown=args.include_unknown,
    )
    print(f"Candidates: {result['candidate_count']}")
    print(f"Closures: {result['closure_counts']}")
    print(f"JSONL: {result['jsonl_path']}")
    print(f"Queue: {result['queue_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
