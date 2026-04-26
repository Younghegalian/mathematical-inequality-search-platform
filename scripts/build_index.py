#!/usr/bin/env python3
"""Build or refresh the AISIS SQLite summary index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.indexing import build_sqlite_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AISIS SQLite summary index")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--index", default=str(ROOT / "data" / "index" / "aisis.sqlite"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_sqlite_index(Path(args.data_dir), Path(args.index))
    print(f"Index: {result.index_path}")
    print(f"Summaries scanned: {result.scanned}")
    print(f"Inserted/updated: {result.inserted_or_updated}")
    print(f"Deleted stale rows: {result.deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
