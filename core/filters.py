"""Early pruning checks for search candidates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable
from core.scaling import scaling_matches


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str = "ok"


def complexity(expr: Expr) -> int:
    if isinstance(expr, (Constant, Variable)):
        return 1
    if isinstance(expr, Derivative):
        return 1 + complexity(expr.expr)
    if isinstance(expr, Norm):
        return 1 + complexity(expr.expr)
    if isinstance(expr, Power):
        return 1 + complexity(expr.base)
    if isinstance(expr, Product):
        return 1 + sum(complexity(term) for term in expr.terms)
    if isinstance(expr, Sum):
        return 1 + sum(complexity(term) for term in expr.terms)
    if isinstance(expr, Integral):
        return 1 + complexity(expr.expr)
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def _walk(expr: Expr) -> list[Expr]:
    children = [expr]
    if isinstance(expr, Derivative):
        children.extend(_walk(expr.expr))
    elif isinstance(expr, Norm):
        children.extend(_walk(expr.expr))
    elif isinstance(expr, Power):
        children.extend(_walk(expr.base))
    elif isinstance(expr, (Product, Sum)):
        for term in expr.terms:
            children.extend(_walk(term))
    elif isinstance(expr, Integral):
        children.extend(_walk(expr.expr))
    return children


def basic_filter(
    lhs: Expr,
    rhs: Expr,
    *,
    allowed_norms: set[Fraction] | None = None,
    max_complexity: int = 40,
    max_power: Fraction = Fraction(8),
) -> FilterDecision:
    if allowed_norms is None:
        allowed_norms = {Fraction(2), Fraction(3), Fraction(4), Fraction(5), Fraction(6)}

    if not scaling_matches(lhs, rhs):
        return FilterDecision(False, "scaling mismatch")

    if complexity(rhs) > max_complexity:
        return FilterDecision(False, "complexity overflow")

    for node in _walk(rhs):
        if isinstance(node, Norm) and node.p not in allowed_norms:
            return FilterDecision(False, f"unknown norm p={node.p}")
        if isinstance(node, Power) and node.exp > max_power:
            return FilterDecision(False, f"power too high: {node.exp}")
        if isinstance(node, Derivative) and node.order > 1:
            return FilterDecision(False, f"derivative order too high: {node.order}")

    return FilterDecision(True)
