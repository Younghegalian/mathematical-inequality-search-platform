"""Canonical fingerprints for expression deduplication."""

from __future__ import annotations

from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable, format_fraction


def canonical(expr: Expr) -> Expr:
    return expr


def fingerprint(expr: Expr) -> str:
    expr = canonical(expr)
    if isinstance(expr, Constant):
        return f"Const({expr.name})"
    if isinstance(expr, Variable):
        return f"Var({expr.name};s={format_fraction(expr.scaling_dimension)})"
    if isinstance(expr, Derivative):
        return f"D{expr.order}({fingerprint(expr.expr)})"
    if isinstance(expr, Norm):
        return f"Norm(p={format_fraction(expr.p)};{fingerprint(expr.expr)})"
    if isinstance(expr, Power):
        return f"Pow(exp={format_fraction(expr.exp)};{fingerprint(expr.base)})"
    if isinstance(expr, Product):
        return "Mul(" + ",".join(fingerprint(term) for term in expr.terms) + ")"
    if isinstance(expr, Sum):
        return "Sum(" + ",".join(fingerprint(term) for term in expr.terms) + ")"
    if isinstance(expr, Integral):
        return f"Int({fingerprint(expr.expr)})"
    raise TypeError(f"unsupported expression type: {type(expr)!r}")
