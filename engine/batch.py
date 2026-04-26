"""Batch execution helpers for bounded AISIS data generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.recording import RunRecorder
from engine.runner import SearchRunner
from ns.targets import TARGETS, TargetSpec


def make_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_batch")


@dataclass(frozen=True)
class BatchJob:
    index: int
    target: TargetSpec
    depth: int
    width: int


@dataclass(frozen=True)
class BatchRunSummary:
    index: int
    target_name: str
    depth: int
    width: int
    run_id: str
    run_path: str
    summary_path: str
    best_closure: str
    best_score: int


def build_jobs(
    *,
    budget: int,
    targets: list[str] | None,
    depths: list[int],
    widths: list[int],
) -> list[BatchJob]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not depths:
        raise ValueError("at least one depth is required")
    if not widths:
        raise ValueError("at least one width is required")

    target_names = targets if targets is not None else sorted(TARGETS)
    unknown = [name for name in target_names if name not in TARGETS]
    if unknown:
        choices = ", ".join(sorted(TARGETS))
        raise ValueError(f"unknown targets {unknown}; choices: {choices}")

    combinations = [
        (TARGETS[target_name], depth, width)
        for target_name in target_names
        for depth in depths
        for width in widths
    ]
    jobs: list[BatchJob] = []
    for index in range(budget):
        target, depth, width = combinations[index % len(combinations)]
        jobs.append(BatchJob(index=index + 1, target=target, depth=depth, width=width))
    return jobs


def run_batch(
    *,
    budget: int,
    targets: list[str] | None,
    depths: list[int],
    widths: list[int],
    data_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    batch_id = make_batch_id()
    jobs = build_jobs(budget=budget, targets=targets, depths=depths, widths=widths)

    manifest: dict[str, Any] = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "dry_run": dry_run,
        "jobs": [],
        "runs": [],
    }

    for job in jobs:
        manifest["jobs"].append(
            {
                "index": job.index,
                "target_name": job.target.name,
                "depth": job.depth,
                "width": job.width,
            }
        )

    if dry_run:
        return manifest

    batch_dir = data_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        metadata = {
            "batch_id": batch_id,
            "batch_index": job.index,
            "batch_budget": budget,
        }
        recorder = RunRecorder.create(job.target.name, data_dir, metadata=metadata)
        result = SearchRunner(depth=job.depth, width=job.width, recorder=recorder).run(job.target)
        best = result.most_informative()
        closure_status = best.closure.status if best.closure is not None else "NONE"
        manifest["runs"].append(
            {
                "index": job.index,
                "target_name": job.target.name,
                "depth": job.depth,
                "width": job.width,
                "run_id": recorder.run_id,
                "run_path": str(result.run_path),
                "summary_path": str(result.summary_path),
                "best_closure": closure_status,
                "best_score": best.score,
            }
        )

    manifest_path = batch_dir / f"{batch_id}.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
