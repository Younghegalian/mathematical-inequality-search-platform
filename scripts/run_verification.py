from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifier.pipeline import run_verification_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AISIS promoted-candidate verification gates.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--queue-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = run_verification_pipeline(
        args.data_dir,
        queue_path=args.queue_path,
        out_dir=args.out_dir,
        limit=args.limit,
    )

    print(f"Queue: {result['queue_path']}")
    print(f"Results: {result['results_path']}")
    print(f"Candidates: {result['candidate_count']}")
    print(f"Verified this run: {result['verified_this_run']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
