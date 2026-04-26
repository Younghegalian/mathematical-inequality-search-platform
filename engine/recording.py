"""Persistent run recording for ML-friendly search traces."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.serde import closure_to_json, expr_record
from engine.state import ProofStep, SearchState
from ns.targets import TargetSpec


SCHEMA_VERSION = "aisis.trace.v1"


def make_run_id(target_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{target_name}"


def proof_to_json(steps: tuple[ProofStep, ...]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "rule_name": step.rule_name,
            "before": expr_record(step.before),
            "after": expr_record(step.after),
            "note": step.note,
        }
        for index, step in enumerate(steps, start=1)
    ]


def state_to_json(state: SearchState) -> dict[str, Any]:
    return {
        "state_id": state.state_id(),
        "depth": len(state.steps),
        "lhs": expr_record(state.lhs),
        "rhs": expr_record(state.rhs),
        "closure": closure_to_json(state.closure),
        "score": state.score,
        "proof": proof_to_json(state.steps),
    }


@dataclass
class RunRecorder:
    run_id: str
    run_path: Path
    summary_path: Path
    metadata: dict[str, Any] | None = None
    search_context: dict[str, Any] | None = None
    transition_statuses: Counter[str] = field(default_factory=Counter)
    transition_actions: Counter[str] = field(default_factory=Counter)
    transition_count: int = 0

    @classmethod
    def create(cls, target_name: str, root: Path, metadata: dict[str, Any] | None = None) -> "RunRecorder":
        run_id = make_run_id(target_name)
        run_dir = root / "runs"
        result_dir = root / "results"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            run_path=run_dir / f"{run_id}.jsonl",
            summary_path=result_dir / f"summary_{run_id}.json",
            metadata=metadata,
        )

    def record(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "event": event,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        if self.metadata:
            row["metadata"] = self.metadata
        with self.run_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def record_start(self, target: TargetSpec, *, width: int, depth: int, rules: list[str]) -> None:
        self.search_context = {"width": width, "depth": depth, "rules": rules}
        self.record(
            "run_start",
            {
                "target_name": target.name,
                "target_description": target.description,
                "target": expr_record(target.expr),
                "dissipation": expr_record(target.dissipation),
                "controlled": expr_record(target.controlled),
                "search": self.search_context,
            },
        )

    def record_initial_state(self, state: SearchState) -> None:
        self.record("state_initial", {"state": state_to_json(state)})

    def record_transition(
        self,
        *,
        depth: int,
        parent: SearchState,
        rule_name: str,
        note: str,
        rhs_after: Any,
        filter_keep: bool,
        filter_reason: str,
        child: SearchState | None = None,
        status: str,
    ) -> None:
        self.transition_count += 1
        self.transition_statuses[status] += 1
        self.transition_actions[rule_name] += 1
        self.record(
            "transition",
            {
                "depth": depth,
                "parent_id": parent.state_id(),
                "child_id": child.state_id() if child is not None else None,
                "action": rule_name,
                "note": note,
                "status": status,
                "filter": {"keep": filter_keep, "reason": filter_reason},
                "rhs_before": expr_record(parent.rhs),
                "rhs_after": expr_record(rhs_after),
                "child_state": state_to_json(child) if child is not None else None,
            },
        )

    def record_beam(self, *, depth: int, beam: list[SearchState]) -> None:
        self.record(
            "beam_selected",
            {
                "depth": depth,
                "state_ids": [state.state_id() for state in beam],
                "states": [state_to_json(state) for state in beam],
            },
        )

    def write_summary(self, target: TargetSpec, final_state: SearchState, states: list[SearchState]) -> None:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "metadata": self.metadata or {},
            "target_name": target.name,
            "target_description": target.description,
            "target": expr_record(target.expr),
            "search": self.search_context or {},
            "best_state": state_to_json(final_state),
            "num_states": len(states),
            "transition_count": self.transition_count,
            "transition_status_histogram": dict(self.transition_statuses),
            "action_histogram": dict(self.transition_actions),
            "closure_histogram": self._closure_histogram(states),
            "run_path": str(self.run_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.record(
            "run_complete",
            {
                "summary_path": str(self.summary_path),
                "best_state_id": final_state.state_id(),
                "num_states": len(states),
            },
        )

    @staticmethod
    def _closure_histogram(states: list[SearchState]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in states:
            status = state.closure.status if state.closure is not None else "NONE"
            counts[status] = counts.get(status, 0) + 1
        return counts
