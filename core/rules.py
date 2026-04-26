"""Inequality transformation rules."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from core.expr import C, C_EPS, EPS, Derivative, Expr, Integral, Norm, Power, Product, Variable, product, summation
from ns.variables import omega, u


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    expression: Expr
    note: str


class Rule(Protocol):
    name: str

    def apply(self, expr: Expr) -> list[RuleResult]:
        ...


def _power_of_norm(expr: Expr) -> tuple[Norm, Fraction] | None:
    if isinstance(expr, Power) and isinstance(expr.base, Norm):
        return expr.base, expr.exp
    if isinstance(expr, Norm):
        return expr, Fraction(1)
    return None


class GagliardoNirenbergRule:
    """3D GN interpolation from L2 and H1.

    ||f||_p^q <= C ||D f||_2^(a q) ||f||_2^((1-a) q)
    where a = 3(1/2 - 1/p), 2 <= p <= 6.
    """

    name = "Gagliardo-Nirenberg"

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    def apply(self, expr: Expr) -> list[RuleResult]:
        parsed = _power_of_norm(expr)
        if parsed is None:
            return []

        norm, outer_power = parsed
        p = norm.p
        if p <= 2 or p > 6:
            return []

        a = Fraction(self.dimension, 1) * (Fraction(1, 2) - Fraction(1, 1) / p)
        if a < 0 or a > 1:
            return []

        grad_norm = Norm(Derivative(norm.expr, 1), 2)
        low_norm = Norm(norm.expr, 2)
        result = product(
            C,
            Power(grad_norm, a * outer_power),
            Power(low_norm, (1 - a) * outer_power),
        )
        note = f"a={a}; interpolates ||f||_{p} between ||f||_2 and ||D f||_2"
        return [RuleResult(self.name, result, note)]


class SobolevRule:
    """3D endpoint Sobolev embedding: ||f||_6 <= C ||D f||_2."""

    name = "Sobolev"

    def apply(self, expr: Expr) -> list[RuleResult]:
        parsed = _power_of_norm(expr)
        if parsed is None:
            return []

        norm, outer_power = parsed
        if norm.p != 6:
            return []

        result = product(C, Power(Norm(Derivative(norm.expr, 1), 2), outer_power))
        return [
            RuleResult(
                self.name,
                result,
                "3D endpoint embedding ||f||_6 <= C ||D f||_2",
            )
        ]


class HolderRule:
    """Small v0.2 Holder library for common Navier-Stokes products."""

    name = "Holder"

    def apply(self, expr: Expr) -> list[RuleResult]:
        if not isinstance(expr, Integral) or not isinstance(expr.expr, Product):
            return []

        terms = expr.expr.terms
        if len(terms) != 3:
            return []

        patterns = [
            (
                "1/2 + 1/4 + 1/4 = 1",
                [
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == u and term.order == 1,
                        2,
                        "Du in L2",
                    ),
                    (lambda term: term == omega, 4, "omega in L4"),
                    (lambda term: term == omega, 4, "omega in L4"),
                ],
            ),
            (
                "1/6 + 1/3 + 1/2 = 1",
                [
                    (lambda term: term == u, 6, "u in L6"),
                    (lambda term: term == omega, 3, "omega in L3"),
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == omega and term.order == 1,
                        2,
                        "Domega in L2",
                    ),
                ],
            ),
            (
                "1/6 + 1/3 + 1/2 = 1",
                [
                    (lambda term: term == u, 6, "u in L6"),
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == u and term.order == 1,
                        3,
                        "Du in L3",
                    ),
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == omega and term.order == 1,
                        2,
                        "Domega in L2",
                    ),
                ],
            ),
            (
                "1/2 + 1/4 + 1/4 = 1",
                [
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == u and term.order == 1,
                        2,
                        "Du in L2",
                    ),
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == u and term.order == 1,
                        4,
                        "Du in L4",
                    ),
                    (lambda term: term == omega, 4, "omega in L4"),
                ],
            ),
            (
                "1/6 + 1/3 + 1/2 = 1",
                [
                    (lambda term: term == u, 6, "u in L6"),
                    (lambda term: term == omega, 3, "omega in L3"),
                    (lambda term: term == omega, 2, "omega in L2"),
                ],
            ),
            (
                "1/3 + 1/6 + 1/2 = 1",
                [
                    (lambda term: term == omega, 3, "omega in L3"),
                    (lambda term: term == omega, 6, "omega in L6"),
                    (
                        lambda term: isinstance(term, Derivative) and term.expr == omega and term.order == 1,
                        2,
                        "Domega in L2",
                    ),
                ],
            ),
        ]

        for prefix, specs in patterns:
            result = self._match_product(terms, prefix, specs)
            if result is not None:
                return [result]

        return []

    def _match_product(
        self,
        terms: tuple[Expr, ...],
        prefix: str,
        specs: list[tuple[object, int, str]],
    ) -> RuleResult | None:
        norm_terms: list[Expr] = []
        used = [False, False, False]
        notes: list[str] = []
        for matcher, p, note in specs:
            matched = False
            for index, term in enumerate(terms):
                if used[index] or not matcher(term):  # type: ignore[operator]
                    continue
                used[index] = True
                norm_terms.append(Norm(term, p))
                notes.append(note)
                matched = True
                break
            if not matched:
                return None

        return RuleResult(
            self.name,
            product(*norm_terms),
            prefix + "; " + ", ".join(notes),
        )


class IntegralPowerToNormRule:
    """Convert int(f^p) into ||f||_p^p for nonnegative tracked quantities."""

    name = "Integral-to-Norm"

    def apply(self, expr: Expr) -> list[RuleResult]:
        if not isinstance(expr, Integral) or not isinstance(expr.expr, Power):
            return []

        power = expr.expr
        if not isinstance(power.base, Variable):
            return []

        return [
            RuleResult(
                self.name,
                Power(Norm(power.base, power.exp), power.exp),
                "identifies int(|f|^p) with ||f||_p^p",
            )
        ]


class BiotSavartRule:
    """Basic vorticity control for velocity norms.

    In 3D, ||u||_6 is controlled by ||D u||_2 and ||D u||_2 is comparable to
    ||omega||_2 for divergence-free velocity fields.
    """

    name = "Biot-Savart"

    def apply(self, expr: Expr) -> list[RuleResult]:
        parsed = _power_of_norm(expr)
        if parsed is None:
            return []

        norm, outer_power = parsed
        if norm.expr == u and norm.p == 6:
            return [
                RuleResult(
                    self.name,
                    product(C, Power(Norm(omega, 2), outer_power)),
                    "controls ||u||_6 by ||omega||_2",
                )
            ]

        if (
            isinstance(norm.expr, Derivative)
            and norm.expr.expr == u
            and norm.expr.order == 1
            and norm.p in {Fraction(2), Fraction(3), Fraction(4), Fraction(6)}
        ):
            return [
                RuleResult(
                    self.name,
                    product(C, Power(Norm(omega, norm.p), outer_power)),
                    f"controls ||D u||_{norm.p} by ||omega||_{norm.p}",
                )
            ]

        return []


class YoungRule:
    """Absorb a fractional dissipation power into eps * dissipation."""

    name = "Young"

    def apply(self, expr: Expr) -> list[RuleResult]:
        if not isinstance(expr, Product):
            return []

        grad_term: Power | None = None
        low_term: Power | None = None

        for term in expr.terms:
            if not isinstance(term, Power) or not isinstance(term.base, Norm):
                continue
            norm = term.base
            if isinstance(norm.expr, Derivative) and norm.expr.order == 1 and norm.p == 2:
                grad_term = term
            elif norm.p == 2:
                low_term = term

        if grad_term is None or low_term is None:
            return []

        alpha = grad_term.exp
        beta = low_term.exp
        if alpha <= 0 or alpha >= 2:
            return []

        conjugate_factor = Fraction(2, 1) / (Fraction(2, 1) - alpha)
        remainder_power = beta * conjugate_factor
        absorbed = product(EPS, Power(grad_term.base, 2))
        remainder = product(C_EPS, Power(low_term.base, remainder_power))
        result = summation(absorbed, remainder)
        note = (
            f"absorbs ||D f||_2^{alpha} into eps ||D f||_2^2; "
            f"remainder exponent={remainder_power}"
        )
        return [RuleResult(self.name, result, note)]


def default_rules() -> list[Rule]:
    return [
        IntegralPowerToNormRule(),
        HolderRule(),
        BiotSavartRule(),
        SobolevRule(),
        GagliardoNirenbergRule(),
        YoungRule(),
    ]
