"""Basic Navier-Stokes variables."""

from __future__ import annotations

from fractions import Fraction

from core.expr import Derivative, Expr, Variable


u = Variable("u", Fraction(1))
omega = Variable("omega", Fraction(2))


def grad(expr: Expr) -> Derivative:
    return Derivative(expr, 1)
