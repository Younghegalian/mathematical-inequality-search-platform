"""Symbolic verification utilities for promoted AISIS candidates."""

from __future__ import annotations

from typing import Any

from core.canonical import fingerprint
from core.expr import Expr
from core.rules import default_rules
from core.scaling import scaling_matches
from core.serde import expr_from_json


def verify_scaling(lhs: Expr, rhs: Expr) -> bool:
    return scaling_matches(lhs, rhs)


def replay_proof(candidate: dict[str, Any]) -> dict[str, Any]:
    """Replay a candidate proof path against the current rule library.

    This is intentionally stricter than just trusting the stored proof text:
    every proof edge must chain from the previous expression and must be
    reproduced by the named rule.
    """

    try:
        lhs = expr_from_json(candidate["lhs"]["ast"])
        rhs = expr_from_json(candidate["rhs"]["ast"])
    except (KeyError, TypeError, ValueError) as error:
        return failed(f"candidate expression could not be decoded: {error}")

    proof = candidate.get("proof", [])
    if not proof:
        return failed("candidate has no proof path to replay")

    rules = {rule.name: rule for rule in default_rules()}
    current = lhs
    steps: list[dict[str, Any]] = []

    for index, step in enumerate(proof, start=1):
        rule_name = str(step.get("rule_name", ""))
        rule = rules.get(rule_name)
        if rule is None:
            return failed(f"unknown rule at step {index}: {rule_name}")

        try:
            before = expr_from_json(step["before"]["ast"])
            after = expr_from_json(step["after"]["ast"])
        except (KeyError, TypeError, ValueError) as error:
            return failed(f"step {index} could not be decoded: {error}")

        if fingerprint(before) != fingerprint(current):
            return failed(
                f"step {index} does not chain from previous expression",
                {"expected": str(current), "found": str(before)},
            )

        produced = rule.apply(before)
        produced_fingerprints = {fingerprint(result.expression) for result in produced}
        if fingerprint(after) not in produced_fingerprints:
            return failed(
                f"step {index} is not reproduced by {rule_name}",
                {
                    "before": str(before),
                    "after": str(after),
                    "produced": [str(result.expression) for result in produced],
                },
            )

        steps.append({"index": index, "rule": rule_name, "before": str(before), "after": str(after)})
        current = after

    if fingerprint(current) != fingerprint(rhs):
        return failed("replayed proof does not end at candidate RHS", {"expected": str(rhs), "found": str(current)})

    return {
        "passed": True,
        "reason": f"replayed {len(steps)} proof step(s)",
        "details": {"steps": steps},
    }


def failed(reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"passed": False, "reason": reason, "details": details or {}}
