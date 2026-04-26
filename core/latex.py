"""LaTeX formatting for AISIS expressions."""

from __future__ import annotations

from fractions import Fraction

from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable
from core.serde import expr_from_json


def fraction_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def expr_to_latex(expr: Expr) -> str:
    if isinstance(expr, Constant):
        return constant_latex(expr.name)
    if isinstance(expr, Variable):
        return variable_latex(expr.name)
    if isinstance(expr, Derivative):
        inner = grouped_latex(expr.expr)
        if expr.order == 0:
            return expr_to_latex(expr.expr)
        if expr.order == 1:
            return rf"\nabla {inner}"
        return rf"\nabla^{{{expr.order}}} {inner}"
    if isinstance(expr, Norm):
        return rf"\left\|{expr_to_latex(expr.expr)}\right\|_{{{fraction_latex(expr.p)}}}"
    if isinstance(expr, Power):
        base = grouped_latex(expr.base)
        if expr.exp == 1:
            return expr_to_latex(expr.base)
        return rf"{base}^{{{fraction_latex(expr.exp)}}}"
    if isinstance(expr, Product):
        return r"\,".join(grouped_latex(term) for term in expr.terms)
    if isinstance(expr, Sum):
        return " + ".join(expr_to_latex(term) for term in expr.terms)
    if isinstance(expr, Integral):
        return rf"\int {expr_to_latex(expr.expr)}\,dx"
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def inequality_to_latex(lhs: Expr, rhs: Expr) -> str:
    return rf"{expr_to_latex(lhs)} \leq {expr_to_latex(rhs)}"


def candidate_inequality_latex(candidate: dict) -> str:
    lhs = expr_from_json(candidate["lhs"]["ast"])
    rhs = expr_from_json(candidate["rhs"]["ast"])
    return inequality_to_latex(lhs, rhs)


def expr_to_unicode(expr: Expr) -> str:
    if isinstance(expr, Constant):
        return constant_unicode(expr.name)
    if isinstance(expr, Variable):
        return variable_unicode(expr.name)
    if isinstance(expr, Derivative):
        inner = grouped_unicode(expr.expr)
        if expr.order == 0:
            return expr_to_unicode(expr.expr)
        if expr.order == 1:
            return f"∇{inner}"
        return f"∇{superscript_text(str(expr.order))}{inner}"
    if isinstance(expr, Norm):
        return f"‖{expr_to_unicode(expr.expr)}‖{subscript_text(fraction_plain(expr.p))}"
    if isinstance(expr, Power):
        if expr.exp == 1:
            return expr_to_unicode(expr.base)
        return f"{grouped_unicode(expr.base)}{superscript_text(fraction_plain(expr.exp))}"
    if isinstance(expr, Product):
        return " · ".join(grouped_unicode(term) for term in expr.terms)
    if isinstance(expr, Sum):
        return " + ".join(expr_to_unicode(term) for term in expr.terms)
    if isinstance(expr, Integral):
        return f"∫ {expr_to_unicode(expr.expr)} dx"
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def inequality_to_unicode(lhs: Expr, rhs: Expr) -> str:
    return f"{expr_to_unicode(lhs)} ≤ {expr_to_unicode(rhs)}"


def candidate_inequality_unicode(candidate: dict) -> str:
    lhs = expr_from_json(candidate["lhs"]["ast"])
    rhs = expr_from_json(candidate["rhs"]["ast"])
    return inequality_to_unicode(lhs, rhs)


def grouped_latex(expr: Expr) -> str:
    if isinstance(expr, Sum):
        return rf"\left({expr_to_latex(expr)}\right)"
    return expr_to_latex(expr)


def grouped_unicode(expr: Expr) -> str:
    if isinstance(expr, Sum):
        return f"({expr_to_unicode(expr)})"
    return expr_to_unicode(expr)


def constant_latex(name: str) -> str:
    if name == "eps":
        return r"\varepsilon"
    if name == "C_eps":
        return r"C_{\varepsilon}"
    return name


def variable_latex(name: str) -> str:
    mapping = {
        "omega": r"\omega",
        "u": "u",
    }
    return mapping.get(name, name)


def constant_unicode(name: str) -> str:
    if name == "eps":
        return "ε"
    if name == "C_eps":
        return "Cε"
    return name


def variable_unicode(name: str) -> str:
    mapping = {
        "omega": "ω",
        "u": "u",
    }
    return mapping.get(name, name)


def fraction_plain(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def superscript_text(value: str) -> str:
    table = str.maketrans("-0123456789/", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹/")
    return value.translate(table)


def subscript_text(value: str) -> str:
    table = str.maketrans("-0123456789/", "₋₀₁₂₃₄₅₆₇₈₉/")
    return value.translate(table)
