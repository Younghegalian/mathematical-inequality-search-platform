"""Small expression AST for inequality search.

The AST is deliberately compact in v0.1. It only needs enough structure to
track norms, powers, products, sums, derivatives, and scaling dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Tuple


def frac(value: int | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


class Expr:
    """Base class for all symbolic expressions."""

    def sort_key(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Constant(Expr):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Variable(Expr):
    name: str
    scaling_dimension: Fraction = Fraction(0)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Derivative(Expr):
    expr: Expr
    order: int = 1

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("derivative order must be non-negative")

    def __str__(self) -> str:
        if self.order == 0:
            return str(self.expr)
        if self.order == 1:
            return f"D({self.expr})"
        return f"D^{self.order}({self.expr})"


@dataclass(frozen=True)
class Norm(Expr):
    expr: Expr
    p: Fraction

    def __init__(self, expr: Expr, p: int | str | Fraction) -> None:
        object.__setattr__(self, "expr", expr)
        object.__setattr__(self, "p", frac(p))

    def __str__(self) -> str:
        return f"||{self.expr}||_{format_fraction(self.p)}"


@dataclass(frozen=True)
class Power(Expr):
    base: Expr
    exp: Fraction

    def __init__(self, base: Expr, exp: int | str | Fraction) -> None:
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "exp", frac(exp))

    def __str__(self) -> str:
        if self.exp == 1:
            return str(self.base)
        return f"{self.base}^{format_fraction(self.exp)}"


@dataclass(frozen=True)
class Product(Expr):
    terms: Tuple[Expr, ...]

    def __init__(self, terms: Iterable[Expr]) -> None:
        flat: list[Expr] = []
        for term in terms:
            if isinstance(term, Product):
                flat.extend(term.terms)
            else:
                flat.append(term)

        power_terms: dict[Expr, Fraction] = {}
        clean: list[Expr] = []
        for term in flat:
            if isinstance(term, Constant) and term.name == "1":
                continue
            if isinstance(term, Power) and term.exp == 0:
                continue
            if isinstance(term, Power):
                power_terms[term.base] = power_terms.get(term.base, Fraction(0)) + term.exp
                continue
            if isinstance(term, Norm):
                power_terms[term] = power_terms.get(term, Fraction(0)) + Fraction(1)
                continue
            clean.append(term)

        for base, exponent in power_terms.items():
            if exponent == 0:
                continue
            if exponent == 1:
                clean.append(base)
            else:
                clean.append(Power(base, exponent))

        flat = clean
        flat.sort(key=lambda item: item.sort_key())
        object.__setattr__(self, "terms", tuple(flat))

    def __str__(self) -> str:
        return " * ".join(str(term) for term in self.terms)


@dataclass(frozen=True)
class Sum(Expr):
    terms: Tuple[Expr, ...]

    def __init__(self, terms: Iterable[Expr]) -> None:
        flat: list[Expr] = []
        for term in terms:
            if isinstance(term, Sum):
                flat.extend(term.terms)
            else:
                flat.append(term)
        flat.sort(key=lambda item: item.sort_key())
        object.__setattr__(self, "terms", tuple(flat))

    def __str__(self) -> str:
        return " + ".join(str(term) for term in self.terms)


@dataclass(frozen=True)
class Integral(Expr):
    expr: Expr

    def __str__(self) -> str:
        return f"int({self.expr})"


def product(*terms: Expr) -> Expr:
    clean = [term for term in terms if not (isinstance(term, Constant) and term.name == "1")]
    if not clean:
        return Constant("1")
    if len(clean) == 1:
        return clean[0]
    return Product(clean)


def summation(*terms: Expr) -> Expr:
    if len(terms) == 1:
        return terms[0]
    return Sum(terms)


C = Constant("C")
EPS = Constant("eps")
C_EPS = Constant("C_eps")
