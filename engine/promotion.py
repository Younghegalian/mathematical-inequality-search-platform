"""Promotion queue for promising AISIS candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scoring import NONLINEAR_PROMOTION_THRESHOLD, nonlinear_relevance_score
from core.serde import expr_from_json


QUEUE_SCHEMA_VERSION = "aisis.verification_queue.v1"


def promote_good_candidates(
    data_dir: Path,
    *,
    out_dir: Path | None = None,
    limit: int = 1000,
    include_unknown: bool = False,
) -> dict[str, Any]:
    """Export deduplicated verification-worthy candidates into the queue."""

    if out_dir is None:
        out_dir = data_dir / "promotions"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(data_dir, include_unknown=include_unknown)
    candidates = sorted(
        candidates,
        key=lambda item: (-int(item["priority"]), -int(item["score"]), item["candidate_id"]),
    )[:limit]

    jsonl_path = out_dir / "good_candidates.jsonl"
    queue_path = out_dir / "verification_queue.json"
    existing_candidates = read_existing_candidates(queue_path)
    candidates = [merge_existing_verification(candidate, existing_candidates) for candidate in candidates]

    jsonl_path.write_text(
        "".join(json.dumps(candidate, ensure_ascii=True, sort_keys=True) + "\n" for candidate in candidates),
        encoding="utf-8",
    )

    status_counts = Counter(candidate["closure_status"] for candidate in candidates)
    queue = {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "closure_counts": dict(status_counts),
        "source_data_dir": str(data_dir),
        "next_process": [
            "symbolic_replay",
            "scaling_audit",
            "target_relevance_check",
            "numeric_counterexample_search",
            "human_math_review",
        ],
        "candidates": candidates,
    }
    queue_path.write_text(json.dumps(queue, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "candidate_count": len(candidates),
        "closure_counts": dict(status_counts),
        "jsonl_path": str(jsonl_path),
        "queue_path": str(queue_path),
        "candidates": candidates,
    }


def collect_candidates(data_dir: Path, *, include_unknown: bool = False) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for summary_path in sorted((data_dir / "results").glob("summary_*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best_state") or {}
        closure = best.get("closure") or {}
        closure_status = str(closure.get("status", "NONE"))
        nonlinear_relevance = relevance_from_summary(summary)
        is_relevant = nonlinear_relevance >= NONLINEAR_PROMOTION_THRESHOLD
        if closure_status != "GOOD" and not is_relevant and not (include_unknown and closure_status == "UNKNOWN"):
            continue

        candidate = candidate_from_summary(summary, summary_path, nonlinear_relevance=nonlinear_relevance)
        previous = by_id.get(candidate["candidate_id"])
        if previous is None or (int(candidate["priority"]), int(candidate["score"])) > (
            int(previous["priority"]),
            int(previous["score"]),
        ):
            by_id[candidate["candidate_id"]] = candidate
    return list(by_id.values())


def read_existing_candidates(queue_path: Path) -> dict[str, dict[str, Any]]:
    if not queue_path.exists():
        return {}
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    candidates = queue.get("candidates", [])
    if not isinstance(candidates, list):
        return {}
    return {
        str(candidate.get("candidate_id")): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("candidate_id")
    }


def merge_existing_verification(candidate: dict[str, Any], existing: dict[str, dict[str, Any]]) -> dict[str, Any]:
    previous = existing.get(str(candidate.get("candidate_id")))
    if previous is None:
        return candidate

    for key in [
        "queue_status",
        "verification",
        "verification_result_path",
        "verification_updated_at",
    ]:
        if key in previous:
            candidate[key] = previous[key]
    return candidate


def candidate_from_summary(
    summary: dict[str, Any],
    summary_path: Path,
    *,
    nonlinear_relevance: int | None = None,
) -> dict[str, Any]:
    best = summary.get("best_state") or {}
    closure = best.get("closure") or {}
    lhs = best.get("lhs") or summary.get("target") or {}
    rhs = best.get("rhs") or {}
    proof = best.get("proof") or []
    proof_rules = [str(step.get("rule_name", "")) for step in proof if step.get("rule_name")]
    run_id = str(summary.get("run_id", summary_path.stem.removeprefix("summary_")))
    target_name = str(summary.get("target_name", "unknown"))
    score = int(best.get("score", 0))
    if nonlinear_relevance is None:
        nonlinear_relevance = relevance_from_summary(summary)
    closure_status = str(closure.get("status", "NONE"))
    closure_priority = {"GOOD": 70, "BAD": 15, "UNKNOWN": 0}.get(closure_status, 0)
    priority = 1000 + (3 * nonlinear_relevance) + closure_priority + max(score, 0) // 10
    candidate_id = make_candidate_id(target_name, text_of(lhs), text_of(rhs), proof_rules)
    run_path = str(summary.get("run_path", ""))
    search = summary.get("search") if isinstance(summary.get("search"), dict) else {}
    if not search and run_path:
        search = read_search_from_run_start(Path(run_path)) or {}

    return {
        "candidate_id": candidate_id,
        "queue_status": "pending_symbolic",
        "priority": priority,
        "promotion_reason": promotion_reason(closure_status, nonlinear_relevance),
        "nonlinear_relevance_score": nonlinear_relevance,
        "run_id": run_id,
        "target_name": target_name,
        "closure_status": closure_status,
        "closure_reason": str(closure.get("reason", "")),
        "score": score,
        "lhs": lhs,
        "rhs": rhs,
        "lhs_text": text_of(lhs),
        "rhs_text": text_of(rhs),
        "proof_rules": proof_rules,
        "proof": proof,
        "search": search,
        "run_path": run_path,
        "summary_path": str(summary_path),
    }


def relevance_from_summary(summary: dict[str, Any]) -> int:
    best = summary.get("best_state") or {}
    lhs = best.get("lhs") or summary.get("target") or {}
    rhs = best.get("rhs") or {}
    proof = best.get("proof") or []
    proof_rules = [str(step.get("rule_name", "")) for step in proof if isinstance(step, dict) and step.get("rule_name")]
    return nonlinear_relevance_score(
        lhs=expr_of(lhs),
        rhs=expr_of(rhs),
        target_name=str(summary.get("target_name", "")),
        proof_rules=proof_rules,
        lhs_text=text_of(lhs),
        rhs_text=text_of(rhs),
    )


def expr_of(record: Any):
    if not isinstance(record, dict):
        return None
    ast = record.get("ast")
    if not isinstance(ast, dict):
        return None
    try:
        return expr_from_json(ast)
    except (TypeError, ValueError):
        return None


def promotion_reason(closure_status: str, nonlinear_relevance: int) -> str:
    is_relevant = nonlinear_relevance >= NONLINEAR_PROMOTION_THRESHOLD
    if is_relevant and closure_status == "GOOD":
        return "nonlinear_term_relevant_and_closure_good"
    if is_relevant:
        return "nonlinear_term_relevant"
    if closure_status == "GOOD":
        return "closure_good"
    return "manual_or_unknown"


def make_candidate_id(target_name: str, lhs_text: str, rhs_text: str, proof_rules: list[str]) -> str:
    payload = "\n".join([target_name, lhs_text, rhs_text, " -> ".join(proof_rules)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def text_of(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("text", ""))


def read_search_from_run_start(run_path: Path) -> dict[str, Any] | None:
    if not run_path.exists():
        return None
    with run_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "run_start":
                search = row.get("search")
                return search if isinstance(search, dict) else None
    return None
