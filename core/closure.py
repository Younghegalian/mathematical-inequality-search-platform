"""Closure classification for candidate inequalities."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from core.canonical import fingerprint
from core.expr import Constant, Expr, Norm, Power, Product, Sum


@dataclass(frozen=True)
class ClosureResult:
    status: str
    ode_power: Fraction | None
    reason: str


def _without_constants(expr: Expr) -> Expr:
    if not isinstance(expr, Product):
        return expr
    terms = [term for term in expr.terms if not isinstance(term, Constant)]
    if len(terms) == 1:
        return terms[0]
    return Product(terms)


def _same(a: Expr, b: Expr) -> bool:
    return fingerprint(a) == fingerprint(b)


def _controlled_ode_power(expr: Expr, controlled_norm: Norm) -> Fraction | None:
    expr = _without_constants(expr)
    if isinstance(expr, Power) and _same(expr.base, controlled_norm):
        return expr.exp / 2
    if _same(expr, Power(controlled_norm, 2)):
        return Fraction(1)
    return None


def classify_closure(rhs: Expr, *, dissipation: Expr, controlled_norm: Norm) -> ClosureResult:
    """Classify whether the RHS closes an enstrophy-type inequality.

    The ODE variable is X = ||omega||_2^2, so ||omega||_2^6 maps to X^3.
    """

    terms = rhs.terms if isinstance(rhs, Sum) else (rhs,)
    powers: list[Fraction] = []
    unknown_terms: list[str] = []
    absorbed = False

    for term in terms:
        bare = _without_constants(term)
        if _same(bare, dissipation):
            absorbed = True
            continue

        power = _controlled_ode_power(term, controlled_norm)
        if power is not None:
            powers.append(power)
            continue

        unknown_terms.append(str(term))

    if unknown_terms:
        return ClosureResult(
            "UNKNOWN",
            None,
            "needs more transformations; unresolved terms: " + ", ".join(unknown_terms),
        )

    if not powers:
        if absorbed:
            return ClosureResult("GOOD", Fraction(0), "only absorbable dissipation remains")
        return ClosureResult("UNKNOWN", None, "no controlled remainder detected")

    ode_power = max(powers)
    if ode_power <= 1:
        return ClosureResult("GOOD", ode_power, f"closes as X' <= C X^{ode_power}")
    return ClosureResult("BAD", ode_power, f"superlinear closure X' <= C X^{ode_power}")
