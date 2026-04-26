#!/usr/bin/env python3
"""Legacy helper for summarizing one AISIS JSONL search trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an AISIS run JSONL file")
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.path.read_text(encoding="utf-8").splitlines()]

    events = Counter(row["event"] for row in rows)
    transitions = [row for row in rows if row["event"] == "transition"]
    statuses = Counter(row["status"] for row in transitions)
    actions = Counter(row["action"] for row in transitions)

    closure_statuses: Counter[str] = Counter()
    for row in transitions:
        child = row.get("child_state")
        if child is None or child.get("closure") is None:
            continue
        closure_statuses[child["closure"]["status"]] += 1

    print(f"Run: {rows[0].get('run_id', 'unknown')}")
    print(f"Rows: {len(rows)}")
    print()
    print("Events:")
    for name, count in events.most_common():
        print(f"  {name}: {count}")
    print()
    print("Transition statuses:")
    for name, count in statuses.most_common():
        print(f"  {name}: {count}")
    print()
    print("Actions:")
    for name, count in actions.most_common():
        print(f"  {name}: {count}")
    print()
    print("Child closure statuses:")
    for name, count in closure_statuses.most_common():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
