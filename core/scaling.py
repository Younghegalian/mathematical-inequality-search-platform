"""Navier-Stokes scaling bookkeeping."""

from __future__ import annotations

from fractions import Fraction

from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable


def scaling(expr: Expr, dimension: int = 3) -> Fraction | None:
    """Return the scaling dimension of an expression.

    For sums, all terms must have the same scaling. Otherwise `None` is
    returned to signal a mismatch.
    """

    if isinstance(expr, Constant):
        return Fraction(0)
    if isinstance(expr, Variable):
        return expr.scaling_dimension
    if isinstance(expr, Derivative):
        inner = scaling(expr.expr, dimension)
        if inner is None:
            return None
        return inner + expr.order
    if isinstance(expr, Norm):
        inner = scaling(expr.expr, dimension)
        if inner is None:
            return None
        return inner - Fraction(dimension, 1) / expr.p
    if isinstance(expr, Power):
        inner = scaling(expr.base, dimension)
        if inner is None:
            return None
        return inner * expr.exp
    if isinstance(expr, Product):
        total = Fraction(0)
        for term in expr.terms:
            value = scaling(term, dimension)
            if value is None:
                return None
            total += value
        return total
    if isinstance(expr, Sum):
        values = [scaling(term, dimension) for term in expr.terms]
        if any(value is None for value in values):
            return None
        first = values[0]
        if all(value == first for value in values):
            return first
        return None
    if isinstance(expr, Integral):
        inner = scaling(expr.expr, dimension)
        if inner is None:
            return None
        return inner - dimension
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def scaling_matches(lhs: Expr, rhs: Expr, dimension: int = 3) -> bool:
    return scaling(lhs, dimension) == scaling(rhs, dimension)
