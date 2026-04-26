"""Search targets for AISIS."""

from __future__ import annotations

from dataclasses import dataclass

from core.expr import Expr, Integral, Norm, Power, Product
from ns.variables import grad, omega, u


@dataclass(frozen=True)
class TargetSpec:
    name: str
    expr: Expr
    dissipation: Expr
    controlled: Expr
    controlled_norm: Norm
    description: str


omega_l2 = Norm(omega, 2)
omega_h1 = Norm(grad(omega), 2)

OMEGA_L3 = TargetSpec(
    name="omega_L3",
    expr=Power(Norm(omega, 3), 3),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Baseline vorticity target ||omega||_3^3",
)

OMEGA_CUBIC_INTEGRAL = TargetSpec(
    name="omega_cubic_integral",
    expr=Integral(Power(omega, 3)),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Integral form of the baseline target int(|omega|^3)",
)

OMEGA_L6_SQUARED = TargetSpec(
    name="omega_L6_squared",
    expr=Power(Norm(omega, 6), 2),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Endpoint Sobolev sanity target ||omega||_6^2",
)

VORTEX_STRETCHING = TargetSpec(
    name="vortex_stretching",
    expr=Integral(Product([u, omega, grad(omega)])),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Toy enstrophy nonlinear term int(u * omega * Domega)",
)


TARGETS = {
    OMEGA_L3.name: OMEGA_L3,
    OMEGA_CUBIC_INTEGRAL.name: OMEGA_CUBIC_INTEGRAL,
    OMEGA_L6_SQUARED.name: OMEGA_L6_SQUARED,
    VORTEX_STRETCHING.name: VORTEX_STRETCHING,
}


def get_target(name: str) -> TargetSpec:
    try:
        return TARGETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TARGETS))
        raise ValueError(f"unknown target {name!r}; choices: {choices}") from exc
