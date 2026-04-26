"""Generated target families for broader AISIS exploration."""

from __future__ import annotations

import random
from fractions import Fraction

from core.expr import Integral, Norm, Power, Product, format_fraction
from ns.targets import OMEGA_CUBIC_INTEGRAL, OMEGA_L3, OMEGA_L6_SQUARED, VORTEX_STRETCHING, TargetSpec, omega_h1, omega_l2
from ns.variables import grad, omega, u


CRITICAL_P_VALUES = [
    Fraction(3, 1),
    Fraction(10, 3),
    Fraction(4, 1),
    Fraction(9, 2),
    Fraction(5, 1),
    Fraction(6, 1),
]


def rational_p_pool() -> list[Fraction]:
    values = set(CRITICAL_P_VALUES)
    for denominator in range(2, 11):
        for numerator in range(2 * denominator + 1, 6 * denominator + 1):
            value = Fraction(numerator, denominator)
            if Fraction(5, 2) <= value <= Fraction(6, 1):
                values.add(value)
    return sorted(values)


def critical_omega_norm_target(p: Fraction) -> TargetSpec:
    norm_scaling = Fraction(2, 1) - Fraction(3, 1) / p
    exponent = Fraction(3, 1) / norm_scaling
    p_name = format_fraction(p).replace("/", "_")
    q_name = format_fraction(exponent).replace("/", "_")
    return TargetSpec(
        name=f"omega_L{p_name}_crit_q{q_name}",
        expr=Power(Norm(omega, p), exponent),
        dissipation=Power(omega_h1, 2),
        controlled=Power(omega_l2, 2),
        controlled_norm=omega_l2,
        description=f"Critical scaling target ||omega||_{format_fraction(p)}^{format_fraction(exponent)}",
    )


STRAIN_VORTICITY = TargetSpec(
    name="strain_vorticity",
    expr=Integral(Product([grad(u), omega, omega])),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Toy strain-vorticity term int(Du * omega * omega)",
)

VELOCITY_STRAIN_GRADIENT = TargetSpec(
    name="velocity_strain_gradient",
    expr=Integral(Product([u, grad(u), grad(omega)])),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Scaling-critical nonlinear proxy int(u * Du * Domega)",
)

STRAIN_STRAIN_VORTICITY = TargetSpec(
    name="strain_strain_vorticity",
    expr=Integral(Product([grad(u), grad(u), omega])),
    dissipation=Power(omega_h1, 2),
    controlled=Power(omega_l2, 2),
    controlled_norm=omega_l2,
    description="Scaling-critical nonlinear proxy int(Du * Du * omega)",
)

NONLINEAR_TARGETS = {
    VORTEX_STRETCHING.name: VORTEX_STRETCHING,
    STRAIN_VORTICITY.name: STRAIN_VORTICITY,
    VELOCITY_STRAIN_GRADIENT.name: VELOCITY_STRAIN_GRADIENT,
    STRAIN_STRAIN_VORTICITY.name: STRAIN_STRAIN_VORTICITY,
}


GENERATED_TARGETS = {target.name: target for target in [critical_omega_norm_target(p) for p in rational_p_pool()]}
GENERATED_TARGETS.update(NONLINEAR_TARGETS)


BASE_AND_GENERATED_TARGETS = {
    OMEGA_L3.name: OMEGA_L3,
    OMEGA_CUBIC_INTEGRAL.name: OMEGA_CUBIC_INTEGRAL,
    OMEGA_L6_SQUARED.name: OMEGA_L6_SQUARED,
    **NONLINEAR_TARGETS,
    **GENERATED_TARGETS,
}


def random_target(rng: random.Random) -> TargetSpec:
    targets = list(BASE_AND_GENERATED_TARGETS.values())
    return rng.choice(targets)
