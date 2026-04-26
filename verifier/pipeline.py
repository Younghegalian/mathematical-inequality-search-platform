"""Verification pipeline for promoted AISIS candidates."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.closure import classify_closure
from core.expr import Derivative, Norm, Power
from core.scaling import scaling
from core.scoring import NONLINEAR_PROMOTION_THRESHOLD, nonlinear_relevance_score
from core.serde import expr_from_json, fraction_to_json
from ns.generated import NONLINEAR_TARGETS
from ns.variables import omega
from verifier.numeric import stress_candidate
from verifier.symbolic import replay_proof


PIPELINE_SCHEMA_VERSION = "aisis.verification_results.v1"


def run_verification_pipeline(
    data_dir: Path,
    *,
    queue_path: Path | None = None,
    out_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if queue_path is None:
        queue_path = data_dir / "promotions" / "verification_queue.json"
    if out_dir is None:
        out_dir = data_dir / "verifications"
    out_dir.mkdir(parents=True, exist_ok=True)

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    candidates = queue.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("verification queue candidates must be a list")

    selected = candidates if limit is None else candidates[:limit]
    results = []
    for candidate in selected:
        if not isinstance(candidate, dict):
            continue
        result = verify_candidate(candidate)
        result_path = out_dir / f"{candidate.get('candidate_id', 'unknown')}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        review_packet_path = write_review_packet(out_dir, candidate, result)
        result["review_packet_path"] = str(review_packet_path)
        result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate["queue_status"] = result["queue_status"]
        candidate["verification"] = result
        candidate["verification_result_path"] = str(result_path)
        candidate["review_packet_path"] = str(review_packet_path)
        candidate["verification_updated_at"] = result["generated_at"]
        results.append(result)

    summary = verification_summary(candidates)
    queue["verification_summary"] = summary
    queue["verification_updated_at"] = datetime.now(timezone.utc).isoformat()
    queue_path.write_text(json.dumps(queue, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results_path = out_dir / "verification_results.json"
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queue_path": str(queue_path),
        "candidate_count": len(candidates),
        "verified_this_run": len(results),
        "summary": summary,
        "results": results,
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "queue_path": str(queue_path),
        "results_path": str(results_path),
        "candidate_count": len(candidates),
        "verified_this_run": len(results),
        "summary": summary,
    }


def verify_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    checks: list[dict[str, Any]] = []

    symbolic = replay_proof(candidate)
    checks.append(gate_result("symbolic_replay", symbolic))
    if not symbolic["passed"]:
        return candidate_result(candidate, "failed_symbolic", checks, generated_at)

    scaling_result = scaling_audit(candidate)
    checks.append(gate_result("scaling_audit", scaling_result))
    if not scaling_result["passed"]:
        return candidate_result(candidate, "failed_scaling", checks, generated_at)

    relevance_result = target_relevance_check(candidate)
    checks.append(gate_result("target_relevance_check", relevance_result))
    if not relevance_result["passed"]:
        return candidate_result(candidate, "failed_relevance", checks, generated_at)

    numeric_result = numeric_counterexample_search(candidate)
    checks.append(gate_result("numeric_counterexample_search", numeric_result))
    if not numeric_result["passed"]:
        return candidate_result(candidate, "failed_numeric", checks, generated_at)

    checks.append(
        gate_result(
            "human_math_review",
            {
                "passed": None,
                "reason": "automated gates passed; awaiting human mathematical review",
                "details": {"review_focus": human_review_focus(candidate)},
            },
        )
    )
    return candidate_result(candidate, "pending_human_review", checks, generated_at)


def scaling_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        lhs = expr_from_json(candidate["lhs"]["ast"])
        rhs = expr_from_json(candidate["rhs"]["ast"])
    except (KeyError, TypeError, ValueError) as error:
        return failed(f"candidate expression could not be decoded: {error}")

    lhs_scaling = scaling(lhs)
    rhs_scaling = scaling(rhs)
    closure = classify_closure(rhs, dissipation=Power(Norm(Derivative(omega, 1), 2), 2), controlled_norm=Norm(omega, 2))
    critical_3d = lhs_scaling == rhs_scaling == 3
    if lhs_scaling != rhs_scaling:
        return failed(
            "lhs/rhs scaling mismatch",
            {"lhs_scaling": fraction_to_json(lhs_scaling), "rhs_scaling": fraction_to_json(rhs_scaling)},
        )
    if not critical_3d:
        return failed(
            "candidate is not 3D Navier-Stokes critical under current scaling model",
            {"lhs_scaling": fraction_to_json(lhs_scaling), "rhs_scaling": fraction_to_json(rhs_scaling)},
        )
    return {
        "passed": True,
        "reason": "lhs/rhs scaling match at 3D critical dimension",
        "details": {
            "lhs_scaling": fraction_to_json(lhs_scaling),
            "rhs_scaling": fraction_to_json(rhs_scaling),
            "closure_status": closure.status,
            "closure_reason": closure.reason,
        },
    }


def target_relevance_check(candidate: dict[str, Any]) -> dict[str, Any]:
    target_name = str(candidate.get("target_name", ""))
    lhs_text = str(candidate.get("lhs_text", ""))
    rhs_text = str(candidate.get("rhs_text", ""))
    proof_rules = [str(rule) for rule in candidate.get("proof_rules", [])]
    nonlinear_targets = {"omega_L3", "omega_cubic_integral", *NONLINEAR_TARGETS.keys()}
    lhs = decode_candidate_expr(candidate, "lhs")
    rhs = decode_candidate_expr(candidate, "rhs")
    relevance = nonlinear_relevance_score(
        lhs=lhs,
        rhs=rhs,
        target_name=target_name,
        proof_rules=proof_rules,
        lhs_text=lhs_text,
        rhs_text=rhs_text,
    )

    if relevance >= NONLINEAR_PROMOTION_THRESHOLD:
        return {
            "passed": True,
            "reason": "candidate is tied to a configured nonlinear-term or enstrophy-production proxy",
            "details": {
                "target_name": target_name,
                "known_nonlinear_target": target_name in nonlinear_targets,
                "nonlinear_relevance_score": relevance,
                "proof_rules": proof_rules,
            },
        }

    return failed(
        "candidate is closure-friendly but currently looks like a pure norm embedding, not a nonlinear-term control",
        {
            "target_name": target_name,
            "lhs_text": lhs_text,
            "proof_rules": proof_rules,
            "nonlinear_relevance_score": relevance,
        },
    )


def decode_candidate_expr(candidate: dict[str, Any], side: str):
    try:
        return expr_from_json(candidate[side]["ast"])
    except (KeyError, TypeError, ValueError):
        return None


def numeric_counterexample_search(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        lhs = expr_from_json(candidate["lhs"]["ast"])
        rhs = expr_from_json(candidate["rhs"]["ast"])
    except (KeyError, TypeError, ValueError) as error:
        return failed(f"candidate expression could not be decoded: {error}")
    return stress_candidate(lhs, rhs)


def human_review_focus(candidate: dict[str, Any]) -> list[str]:
    return [
        "Check every rule-side condition and domain assumption.",
        "Decide whether the target is genuinely tied to the Navier-Stokes nonlinear term.",
        "Look for hidden endpoint or compactness assumptions not represented in the AST.",
        f"Review proof path: {' -> '.join(str(rule) for rule in candidate.get('proof_rules', []))}",
    ]


def write_review_packet(out_dir: Path, candidate: dict[str, Any], result: dict[str, Any]) -> Path:
    packet_dir = out_dir / "review_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = str(candidate.get("candidate_id", "unknown"))
    path = packet_dir / f"{candidate_id}.md"
    path.write_text(review_packet_markdown(candidate, result), encoding="utf-8")
    return path


def review_packet_markdown(candidate: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        f"# AISIS Verification Packet: {candidate.get('target_name', 'unknown')}",
        "",
        f"- Candidate ID: `{candidate.get('candidate_id', '')}`",
        f"- Run ID: `{candidate.get('run_id', '')}`",
        f"- Queue status: `{result.get('queue_status', '')}`",
        f"- Score: `{candidate.get('score', '')}`",
        f"- LHS: `{candidate.get('lhs_text', '')}`",
        f"- RHS: `{candidate.get('rhs_text', '')}`",
        f"- Proof rules: `{' -> '.join(str(rule) for rule in candidate.get('proof_rules', []))}`",
        "",
        "## Gate Results",
        "",
    ]
    for check in result.get("checks", []):
        status = check.get("status", "unknown")
        lines.extend(
            [
                f"### {check.get('gate', 'unknown')} [{status}]",
                "",
                str(check.get("reason", "")),
                "",
                "```json",
                json.dumps(check.get("details", {}), ensure_ascii=True, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )

    if result.get("queue_status") == "pending_human_review":
        lines.extend(
            [
                "## Human Review Checklist",
                "",
                "- Verify all rule side conditions and endpoint assumptions.",
                "- Check whether constants depend only on allowed fixed quantities.",
                "- Confirm the target controls the Navier-Stokes nonlinear term, not just a standalone embedding.",
                "- Try to rewrite the candidate as a paper-style lemma with hypotheses.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Current Verdict",
                "",
                "This candidate is not ready for human theorem review because an automated gate failed.",
                "",
            ]
        )
    return "\n".join(lines)


def gate_result(gate: str, result: dict[str, Any]) -> dict[str, Any]:
    status = "pending" if result.get("passed") is None else "passed" if result.get("passed") else "failed"
    return {
        "gate": gate,
        "status": status,
        "passed": result.get("passed"),
        "reason": str(result.get("reason", "")),
        "details": result.get("details", {}),
    }


def candidate_result(
    candidate: dict[str, Any],
    queue_status: str,
    checks: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "candidate_id": str(candidate.get("candidate_id", "")),
        "target_name": str(candidate.get("target_name", "")),
        "run_id": str(candidate.get("run_id", "")),
        "queue_status": queue_status,
        "checks": checks,
    }


def verification_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(candidate.get("queue_status", "unknown")) for candidate in candidates)
    gate_statuses: Counter[str] = Counter()
    for candidate in candidates:
        verification = candidate.get("verification", {})
        if not isinstance(verification, dict):
            continue
        for check in verification.get("checks", []):
            if not isinstance(check, dict):
                continue
            gate_statuses[f'{check.get("gate", "unknown")}:{check.get("status", "unknown")}'] += 1
    return {"queue_status_counts": dict(statuses), "gate_status_counts": dict(gate_statuses)}


def failed(reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"passed": False, "reason": reason, "details": details or {}}
