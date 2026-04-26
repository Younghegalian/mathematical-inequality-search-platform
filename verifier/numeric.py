"""Numerical stress tests for promoted AISIS candidates.

The constants in AISIS inequalities are symbolic, so this module does not try
to prove ``lhs <= rhs`` with constant one. Instead it searches for structural
instability: non-finite evaluations, vanishing right-hand sides, or ratios
that grow sharply across amplitude, frequency, profile, or random Fourier
families.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light envs.
    np = None  # type: ignore[assignment]

from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable


ArrayLike = Any


@dataclass(frozen=True)
class FieldSample:
    profile: str
    amplitude: float
    frequency: int
    seed: int
    fields: dict[str, np.ndarray]

    @property
    def label(self) -> str:
        return f"{self.profile}:A={self.amplitude}:k={self.frequency}:seed={self.seed}"

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "seed": self.seed,
        }


def available() -> bool:
    return True


def stress_candidate(
    lhs: Expr,
    rhs: Expr,
    *,
    grid_size: int = 16,
    ratio_limit: float = 1.0e8,
    growth_limit: float = 75.0,
) -> dict[str, Any]:
    if np is None:
        return synthetic_stress_candidate(lhs, rhs, ratio_limit=ratio_limit, growth_limit=growth_limit)

    samples = field_samples(grid_size=grid_size)
    records: list[dict[str, Any]] = []

    for sample in samples:
        lhs_value = scalar_value(evaluate(lhs, sample))
        rhs_value = scalar_value(evaluate(rhs, sample))
        if rhs_value <= 1.0e-14 or not isfinite(lhs_value) or not isfinite(rhs_value):
            return {
                "passed": False,
                "reason": "non-finite value or nearly vanishing RHS on a spectral sample",
                "details": {"sample": sample.metadata(), "lhs": lhs_value, "rhs": rhs_value},
            }
        ratio = lhs_value / rhs_value
        records.append({"sample": sample.metadata(), "lhs": lhs_value, "rhs": rhs_value, "ratio": ratio})
        if ratio > ratio_limit:
            return {
                "passed": False,
                "reason": "candidate ratio exceeded hard stress limit",
                "details": {"ratio_limit": ratio_limit, "record": records[-1]},
            }

    family_growth = detect_family_growth(records)
    if family_growth["max_growth"] > growth_limit:
        return {
            "passed": False,
            "reason": "candidate ratio grows too sharply across a stress family",
            "details": {"growth_limit": growth_limit, **family_growth},
        }

    worst = max(records, key=lambda item: float(item["ratio"]))
    return {
        "passed": True,
        "reason": f"bounded ratio over {len(records)} spectral/adversarial samples",
        "details": {
            "grid_size": grid_size,
            "sample_count": len(records),
            "worst_ratio": worst["ratio"],
            "worst_sample": worst["sample"],
            "family_growth": family_growth,
            "profiles": sorted({record["sample"]["profile"] for record in records}),
        },
    }


def field_samples(*, grid_size: int) -> list[FieldSample]:
    if np is None:
        raise RuntimeError("numpy is required for spectral field samples")
    samples: list[FieldSample] = []
    amplitudes = [0.25, 1.0, 4.0]
    max_frequency = max(1, grid_size // 2 - 1)
    frequencies = sorted({value for value in [1, 2, 4, 6] if value <= max_frequency})
    profiles = ["single_mode", "multi_mode", "localized_bump", "two_scale"]
    seeds = [0, 1]
    for profile in profiles:
        for amplitude in amplitudes:
            for frequency in frequencies:
                for seed in seeds:
                    fields = {
                        "omega": make_field(profile, amplitude, frequency, seed, grid_size, phase=0.0),
                        "u": make_field(profile, 0.7 * amplitude, max(1, frequency // 2), seed + 17, grid_size, phase=0.37),
                    }
                    samples.append(FieldSample(profile, amplitude, frequency, seed, fields))
    return samples


def make_field(
    profile: str,
    amplitude: float,
    frequency: int,
    seed: int,
    grid_size: int,
    *,
    phase: float,
) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for spectral field generation")
    coords = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    if profile == "single_mode":
        field = np.sin(frequency * x + phase) * np.cos(frequency * y - 0.5 * phase)
    elif profile == "two_scale":
        field = (
            np.sin(frequency * x + phase)
            + 0.35 * np.sin((frequency + 1) * y - phase)
            + 0.18 * np.cos(max(1, 2 * frequency - 1) * z + phase)
        )
    elif profile == "localized_bump":
        width = max(0.16, 0.9 / max(1, frequency))
        center = np.array([np.pi * (1.0 + 0.07 * seed), np.pi * (0.74 + 0.03 * seed), np.pi * (1.26 - 0.02 * seed)])
        dx = periodic_distance(x, center[0])
        dy = periodic_distance(y, center[1])
        dz = periodic_distance(z, center[2])
        field = np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * width * width))
    elif profile == "multi_mode":
        rng = np.random.default_rng(seed + 1009 * frequency)
        field = np.zeros((grid_size, grid_size, grid_size), dtype=float)
        for _ in range(10):
            kx, ky, kz = rng.integers(1, frequency + 2, size=3)
            coeff = rng.normal()
            shift = rng.uniform(0.0, 2.0 * np.pi)
            field += coeff * np.sin(kx * x + ky * y + kz * z + shift)
    else:
        raise ValueError(f"unknown profile: {profile}")

    field = field - float(np.mean(field))
    rms = float(np.sqrt(np.mean(field * field)))
    if rms <= 1.0e-14:
        return np.zeros_like(field)
    return amplitude * field / rms


def periodic_distance(values: np.ndarray, center: float) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for periodic distance")
    raw = np.abs(values - center)
    return np.minimum(raw, 2.0 * np.pi - raw)


def evaluate(expr: Expr, sample: FieldSample) -> ArrayLike:
    if np is None:
        raise RuntimeError("numpy is required for spectral evaluation")
    if isinstance(expr, Constant):
        return constant_value(expr.name)
    if isinstance(expr, Variable):
        return sample.fields.get(expr.name, sample.fields["omega"])
    if isinstance(expr, Derivative):
        value = evaluate(expr.expr, sample)
        if np.isscalar(value):
            return float(value)
        result = np.asarray(value, dtype=float)
        for _ in range(expr.order):
            result = spectral_gradient_magnitude(result)
        return result
    if isinstance(expr, Norm):
        value = evaluate(expr.expr, sample)
        return lp_norm(np.asarray(value, dtype=float), expr.p)
    if isinstance(expr, Power):
        value = evaluate(expr.base, sample)
        exponent = float(expr.exp)
        if np.isscalar(value):
            return abs(float(value)) ** exponent
        return np.abs(np.asarray(value, dtype=float)) ** exponent
    if isinstance(expr, Product):
        value: ArrayLike = 1.0
        for term in expr.terms:
            value = value * evaluate(term, sample)
        return value
    if isinstance(expr, Sum):
        value: ArrayLike = 0.0
        for term in expr.terms:
            value = value + evaluate(term, sample)
        return value
    if isinstance(expr, Integral):
        value = evaluate(expr.expr, sample)
        if np.isscalar(value):
            return float(value)
        return float(np.mean(np.asarray(value, dtype=float)))
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def spectral_gradient_magnitude(field: np.ndarray) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for spectral derivatives")
    n = field.shape[0]
    freqs = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    fft_field = np.fft.fftn(field)
    factor = 2.0 * np.pi
    gradients = [
        np.fft.ifftn(1j * factor * kx * fft_field).real,
        np.fft.ifftn(1j * factor * ky * fft_field).real,
        np.fft.ifftn(1j * factor * kz * fft_field).real,
    ]
    return np.sqrt(sum(component * component for component in gradients))


def lp_norm(field: np.ndarray, p: Fraction) -> float:
    if p <= 0:
        raise ValueError("Lp exponent must be positive")
    return float(np.mean(np.abs(field) ** float(p)) ** (1.0 / float(p)))


def constant_value(name: str) -> float:
    if name == "eps":
        return 1.0
    return 1.0


def scalar_value(value: ArrayLike) -> float:
    if np is None:
        return float(value)
    if np.isscalar(value):
        return float(value)
    return float(np.mean(np.asarray(value, dtype=float)))


def detect_family_growth(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for record in records:
        sample = record["sample"]
        grouped.setdefault((str(sample["profile"]), int(sample["seed"]), float(sample["amplitude"])), []).append(record)

    worst_growth = 1.0
    worst_family: dict[str, Any] | None = None
    for (profile, seed, amplitude), family in grouped.items():
        ordered = sorted(family, key=lambda record: float(record["sample"]["frequency"]))
        low = max(float(ordered[0]["ratio"]), 1.0e-30)
        high = max(float(ordered[-1]["ratio"]), 1.0e-30)
        growth = high / low
        if growth > worst_growth:
            worst_growth = growth
            worst_family = {
                "profile": profile,
                "seed": seed,
                "amplitude": amplitude,
                "low_frequency": ordered[0]["sample"]["frequency"],
                "high_frequency": ordered[-1]["sample"]["frequency"],
                "low_ratio": low,
                "high_ratio": high,
            }

    return {"max_growth": worst_growth, "worst_family": worst_family}


def synthetic_stress_candidate(
    lhs: Expr,
    rhs: Expr,
    *,
    ratio_limit: float,
    growth_limit: float,
) -> dict[str, Any]:
    records = []
    for sample in synthetic_samples():
        lhs_value = synthetic_evaluate(lhs, sample)
        rhs_value = synthetic_evaluate(rhs, sample)
        if rhs_value <= 1.0e-14 or not isfinite(lhs_value) or not isfinite(rhs_value):
            return {
                "passed": False,
                "reason": "non-finite value or nearly vanishing RHS on a synthetic sample",
                "details": {"sample": sample, "lhs": lhs_value, "rhs": rhs_value},
            }
        ratio = lhs_value / rhs_value
        records.append({"sample": sample, "lhs": lhs_value, "rhs": rhs_value, "ratio": ratio})
        if ratio > ratio_limit:
            return {
                "passed": False,
                "reason": "candidate ratio exceeded hard synthetic stress limit",
                "details": {"ratio_limit": ratio_limit, "record": records[-1]},
            }

    family_growth = detect_family_growth(records)
    if family_growth["max_growth"] > growth_limit:
        return {
            "passed": False,
            "reason": "candidate ratio grows too sharply across a synthetic stress family",
            "details": {"growth_limit": growth_limit, **family_growth},
        }

    worst = max(records, key=lambda item: float(item["ratio"]))
    return {
        "passed": True,
        "reason": f"bounded ratio over {len(records)} synthetic samples; install numpy for spectral-grid stress",
        "details": {
            "sample_count": len(records),
            "worst_ratio": worst["ratio"],
            "worst_sample": worst["sample"],
            "family_growth": family_growth,
            "spectral_grid_enabled": False,
        },
    }


def synthetic_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for profile in ["single_scale", "concentrated", "oscillatory"]:
        for amplitude in [0.25, 1.0, 4.0]:
            for frequency in [1.0, 2.0, 4.0, 8.0, 16.0]:
                samples.append({"profile": profile, "amplitude": amplitude, "frequency": frequency, "seed": 0})
    return samples


def synthetic_evaluate(expr: Expr, sample: dict[str, Any]) -> float:
    if isinstance(expr, Constant):
        return constant_value(expr.name)
    if isinstance(expr, Variable):
        return float(sample["amplitude"])
    if isinstance(expr, Derivative):
        return synthetic_evaluate(expr.expr, sample) * float(sample["frequency"]) ** expr.order
    if isinstance(expr, Norm):
        return synthetic_norm_value(expr, sample)
    if isinstance(expr, Power):
        return abs(synthetic_evaluate(expr.base, sample)) ** float(expr.exp)
    if isinstance(expr, Product):
        value = 1.0
        for term in expr.terms:
            value *= synthetic_evaluate(term, sample)
        return value
    if isinstance(expr, Sum):
        return sum(synthetic_evaluate(term, sample) for term in expr.terms)
    if isinstance(expr, Integral):
        return synthetic_evaluate(expr.expr, sample)
    raise TypeError(f"unsupported expression type: {type(expr)!r}")


def synthetic_norm_value(norm: Norm, sample: dict[str, Any]) -> float:
    amplitude = float(sample["amplitude"])
    frequency = float(sample["frequency"])
    derivative_order = synthetic_derivative_depth(norm.expr)
    p_gain = max(Fraction(0), Fraction(3, 2) - Fraction(3, 1) / norm.p)
    concentration_gain = 1.0
    if sample.get("profile") == "concentrated":
        concentration_gain = frequency ** float(p_gain)
    return amplitude * frequency ** float(derivative_order) * concentration_gain


def synthetic_derivative_depth(expr: Expr) -> Fraction:
    if isinstance(expr, Derivative):
        return Fraction(expr.order) + synthetic_derivative_depth(expr.expr)
    return Fraction(0)
