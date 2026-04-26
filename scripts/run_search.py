#!/usr/bin/env python3
"""Run a small AISIS inequality search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scaling import scaling
from engine.recording import RunRecorder
from engine.runner import SearchRunner
from ns.targets import TARGETS, get_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AISIS v0.1 search")
    parser.add_argument("--target", default="omega_L3")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_targets:
        for name, target in sorted(TARGETS.items()):
            print(f"{name}: {target.description}")
        return 0

    target = get_target(args.target)
    recorder = RunRecorder.create(target.name, Path(args.data_dir)) if args.save else None
    result = SearchRunner(width=args.width, depth=args.depth, recorder=recorder).run(target)
    state = result.most_informative()

    print("Target:")
    print(f"  {target.expr}")
    print()
    print("Derivation:")
    if not state.steps:
        print("  no rule applied")
    for index, step in enumerate(state.steps, start=1):
        print(f"  {index}. {step.rule_name}")
        print(f"     before: {step.before}")
        print(f"     after:  {step.after}")
        print(f"     note:   {step.note}")
    print()
    print("Result:")
    print(f"  {target.expr} <= {state.rhs}")
    print()
    print("Scaling:")
    print(f"  lhs={scaling(target.expr)} rhs={scaling(state.rhs)}")
    print()
    print("Closure:")
    if state.closure is None:
        print("  UNKNOWN")
    else:
        print(f"  {state.closure.status}")
        print(f"  {state.closure.reason}")
    print()
    print(f"Score: {state.score}")
    if result.run_path is not None and result.summary_path is not None:
        print()
        print("Saved:")
        print(f"  transitions: {result.run_path}")
        print(f"  summary:     {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
