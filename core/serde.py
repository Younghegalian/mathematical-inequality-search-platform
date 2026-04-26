"""Serialization helpers for expressions and search metadata."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from core.canonical import fingerprint
from core.closure import ClosureResult
from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable
from core.scaling import scaling


def fraction_to_json(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def fraction_from_json(value: str | int | None) -> Fraction | None:
    if value is None:
        return None
    return Fraction(str(value))


def expr_to_json(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, Constant):
        return {"type": "Constant", "name": expr.name}
    if isinstance(expr, Variable):
        return {
            "type": "Variable",
            "name": expr.name,
            "scaling_dimension": fraction_to_json(expr.scaling_dimension),
        }
    if isinstance(expr, Derivative):
        return {"type": "Derivative", "order": expr.order, "expr": expr_to_json(expr.expr)}
    if isinstance(expr, Norm):
        return {"type": "Norm", "p": fraction_to_json(expr.p), "expr": expr_to_json(expr.expr)}
    if isinstance(expr, Power):
        return {
            "type": "Power",
            "exp": fraction_to_json(expr.exp),
            "base": expr_to_json(expr.base),
        }
    if isinstance(expr, Product):
        return {"type": "Product", "terms": [expr_to_json(term) for term in expr.terms]}
    if isinstance(expr, Sum):
        return {"type": "Sum", "terms": [expr_to_json(term) for term in expr.terms]}
    if isinstance(expr, Integral):
        return {"type": "Integral", "expr": expr_to_json(expr.expr)}
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def expr_record(expr: Expr) -> dict[str, Any]:
    return {
        "text": str(expr),
        "fingerprint": fingerprint(expr),
        "scaling": fraction_to_json(scaling(expr)),
        "ast": expr_to_json(expr),
    }


def expr_from_json(record: dict[str, Any]) -> Expr:
    expr_type = record.get("type")
    if expr_type == "Constant":
        return Constant(str(record.get("name", "")))
    if expr_type == "Variable":
        scaling_dimension = fraction_from_json(record.get("scaling_dimension")) or Fraction(0)
        return Variable(str(record.get("name", "")), scaling_dimension)
    if expr_type == "Derivative":
        return Derivative(expr_from_json(record["expr"]), int(record.get("order", 1)))
    if expr_type == "Norm":
        p = fraction_from_json(record.get("p"))
        if p is None:
            raise ValueError("Norm record missing p")
        return Norm(expr_from_json(record["expr"]), p)
    if expr_type == "Power":
        exp = fraction_from_json(record.get("exp"))
        if exp is None:
            raise ValueError("Power record missing exp")
        return Power(expr_from_json(record["base"]), exp)
    if expr_type == "Product":
        return Product(expr_from_json(term) for term in record.get("terms", []))
    if expr_type == "Sum":
        return Sum(expr_from_json(term) for term in record.get("terms", []))
    if expr_type == "Integral":
        return Integral(expr_from_json(record["expr"]))
    raise ValueError(f"unsupported expression record type: {expr_type!r}")


def closure_to_json(closure: ClosureResult | None) -> dict[str, Any] | None:
    if closure is None:
        return None
    return {
        "status": closure.status,
        "ode_power": fraction_to_json(closure.ode_power),
        "reason": closure.reason,
    }
