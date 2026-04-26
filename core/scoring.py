"""Heuristic scores for search states.

The search should prefer candidates that stay connected to the nonlinear
Navier-Stokes term over candidates that merely close as standalone norm
embeddings. Closure is still useful, but it is a secondary signal.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.closure import ClosureResult
from core.expr import Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable
from core.filters import complexity


NONLINEAR_TARGET_NAMES = {
    "vortex_stretching",
    "strain_vorticity",
    "velocity_strain_gradient",
    "strain_strain_vorticity",
}
ENSTROPHY_PROXY_NAMES = {
    "omega_L3",
    "omega_cubic_integral",
}
NONLINEAR_RULE_WEIGHTS = {
    "Biot-Savart": 45,
    "Holder": 30,
    "Integral-to-Norm": 25,
}
ABSORPTION_RULE_WEIGHTS = {
    "Gagliardo-Nirenberg": 15,
    "Young": 10,
    "Sobolev": 5,
}
NONLINEAR_PROMOTION_THRESHOLD = 120


def score(
    rhs: Expr,
    closure: ClosureResult,
    *,
    target: object | None = None,
    lhs: Expr | None = None,
    steps: Iterable[object] = (),
) -> int:
    target_name = target if isinstance(target, str) else str(getattr(target, "name", ""))
    lhs_expr = lhs if lhs is not None else getattr(target, "expr", None)
    relevance = nonlinear_relevance_score(
        lhs=lhs_expr if isinstance(lhs_expr, Expr) else None,
        rhs=rhs,
        target_name=target_name,
        steps=steps,
    )

    value = relevance
    if closure.status == "GOOD":
        value += 70
    elif closure.status == "BAD":
        value += 15
    elif closure.status == "UNKNOWN":
        value += 0

    if isinstance(rhs, Sum):
        value += 8

    if closure.status == "GOOD" and relevance < NONLINEAR_PROMOTION_THRESHOLD and _looks_like_pure_embedding(lhs_expr, rhs):
        value -= 80

    value -= complexity(rhs)
    return value


def nonlinear_relevance_score(
    *,
    lhs: Expr | None = None,
    rhs: Expr | None = None,
    target_name: str = "",
    steps: Iterable[object] = (),
    proof_rules: Iterable[str] = (),
    lhs_text: str = "",
    rhs_text: str = "",
) -> int:
    """Estimate whether a candidate is tied to nonlinear-term control."""

    value = 0
    if target_name in NONLINEAR_TARGET_NAMES:
        value += 160
    elif target_name in ENSTROPHY_PROXY_NAMES:
        value += 120

    expressions = [expr for expr in [lhs, rhs] if isinstance(expr, Expr)]
    if any(_contains_velocity(expr) for expr in expressions):
        value += 45
    if any(_contains_integral_nonlinearity(expr) for expr in expressions):
        value += 55
    if any(_contains_omega_derivative_mix(expr) for expr in expressions):
        value += 25

    rule_names = set(_rule_names(steps))
    rule_names.update(str(rule) for rule in proof_rules if str(rule))
    value += sum(NONLINEAR_RULE_WEIGHTS.get(rule, 0) for rule in rule_names)
    if value > 0:
        value += sum(ABSORPTION_RULE_WEIGHTS.get(rule, 0) for rule in rule_names)

    text_blob = f"{lhs_text} {rhs_text}"
    if text_blob.strip():
        if "int(" in text_blob and "*" in text_blob:
            value += 35
        if "u" in text_blob:
            value += 25
        if "D(" in text_blob and "omega" in text_blob:
            value += 15

    return min(value, 320)


def is_nonlinear_term_relevant(
    *,
    lhs: Expr | None = None,
    rhs: Expr | None = None,
    target_name: str = "",
    steps: Iterable[object] = (),
    proof_rules: Iterable[str] = (),
    lhs_text: str = "",
    rhs_text: str = "",
) -> bool:
    return (
        nonlinear_relevance_score(
            lhs=lhs,
            rhs=rhs,
            target_name=target_name,
            steps=steps,
            proof_rules=proof_rules,
            lhs_text=lhs_text,
            rhs_text=rhs_text,
        )
        >= NONLINEAR_PROMOTION_THRESHOLD
    )


def _rule_names(steps: Iterable[object]) -> list[str]:
    names: list[str] = []
    for step in steps:
        if isinstance(step, str):
            names.append(step)
        elif isinstance(step, dict):
            name = str(step.get("rule_name", ""))
            if name:
                names.append(name)
        else:
            name = str(getattr(step, "rule_name", ""))
            if name:
                names.append(name)
    return names


def _walk(expr: Expr) -> Iterable[Expr]:
    yield expr
    if isinstance(expr, Derivative):
        yield from _walk(expr.expr)
    elif isinstance(expr, Norm):
        yield from _walk(expr.expr)
    elif isinstance(expr, Power):
        yield from _walk(expr.base)
    elif isinstance(expr, Product):
        for term in expr.terms:
            yield from _walk(term)
    elif isinstance(expr, Sum):
        for term in expr.terms:
            yield from _walk(term)
    elif isinstance(expr, Integral):
        yield from _walk(expr.expr)


def _contains_velocity(expr: Expr) -> bool:
    return any(isinstance(node, Variable) and node.name == "u" for node in _walk(expr))


def _contains_integral_nonlinearity(expr: Expr) -> bool:
    return any(isinstance(node, Integral) and _is_nonlinear_integrand(node.expr) for node in _walk(expr))


def _contains_omega_derivative_mix(expr: Expr) -> bool:
    has_omega = any(isinstance(node, Variable) and node.name == "omega" for node in _walk(expr))
    has_derivative = any(isinstance(node, Derivative) for node in _walk(expr))
    return has_omega and has_derivative


def _is_nonlinear_integrand(expr: Expr) -> bool:
    if isinstance(expr, Product):
        return sum(1 for term in expr.terms if _is_field_factor(term)) >= 2
    if isinstance(expr, Power):
        return _is_field_factor(expr.base) and expr.exp > 1
    return False


def _is_field_factor(expr: Expr) -> bool:
    if isinstance(expr, Variable):
        return expr.name in {"u", "omega"}
    if isinstance(expr, Derivative):
        return _is_field_factor(expr.expr)
    if isinstance(expr, Power):
        return _is_field_factor(expr.base)
    return False


def _looks_like_pure_embedding(lhs: object | None, rhs: Expr) -> bool:
    expressions = [expr for expr in [lhs, rhs] if isinstance(expr, Expr)]
    if any(_contains_velocity(expr) or _contains_integral_nonlinearity(expr) for expr in expressions):
        return False
    return any(isinstance(node, Norm) for expr in expressions for node in _walk(expr))
