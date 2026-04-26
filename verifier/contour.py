"""Spatial field diagnostics for candidate inequalities."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from textwrap import shorten
from typing import Any

from core.latex import candidate_inequality_latex
from core.expr import Constant, Derivative, Expr, Integral, Norm, Power, Product, Sum, Variable
from verifier.numeric import FieldSample, evaluate, np, scalar_value, spectral_gradient_magnitude


Array = Any


@dataclass(frozen=True)
class ContourFields:
    lhs_density: Array
    rhs_density: Array
    residual: Array
    log_ratio: Array
    omega: Array
    u: Array
    grad_omega: Array
    aggregates: dict[str, float]


def compute_contour_fields(
    lhs: Expr,
    rhs: Expr,
    sample: FieldSample,
    *,
    constants: dict[str, float] | None = None,
) -> ContourFields:
    """Evaluate an inequality as local diagnostic densities on a field sample.

    Norms are global objects, so the density map is a diagnostic proxy rather
    than a proof object. For ``||f||_p^q`` it distributes the global value over
    ``|f|^p`` so that the spatial mean equals ``||f||_p^q``.
    """

    require_numpy()
    constants = constants or {}
    omega = sample.fields["omega"]
    lhs_raw = contribution_density(lhs, sample, constants=constants)
    rhs_raw = contribution_density(rhs, sample, constants=constants)
    lhs_density = np.abs(as_field(lhs_raw, omega))
    rhs_density = np.abs(as_field(rhs_raw, omega))
    floor = max(1.0e-14, 1.0e-12 * float(np.nanmax(rhs_density + lhs_density) + 1.0))
    residual = lhs_density - rhs_density
    log_ratio = np.log10((lhs_density + floor) / (rhs_density + floor))
    grad_omega = spectral_gradient_magnitude(omega)
    u = sample.fields.get("u", np.zeros_like(omega))
    return ContourFields(
        lhs_density=lhs_density,
        rhs_density=rhs_density,
        residual=residual,
        log_ratio=log_ratio,
        omega=omega,
        u=u,
        grad_omega=grad_omega,
        aggregates={
            "lhs_mean": float(np.mean(lhs_density)),
            "rhs_mean": float(np.mean(rhs_density)),
            "residual_max": float(np.max(residual)),
            "residual_mean": float(np.mean(residual)),
            "log_ratio_max": float(np.max(log_ratio)),
            "log_ratio_mean": float(np.mean(log_ratio)),
            "omega_rms": float(np.sqrt(np.mean(omega * omega))),
            "grad_omega_rms": float(np.sqrt(np.mean(grad_omega * grad_omega))),
        },
    )


def contribution_density(expr: Expr, sample: FieldSample, *, constants: dict[str, float]) -> Array:
    require_numpy()
    reference = sample.fields["omega"]
    if isinstance(expr, Constant):
        return np.full_like(reference, constants.get(expr.name, default_constant(expr.name)), dtype=float)
    if isinstance(expr, Variable):
        return evaluate(expr, sample)
    if isinstance(expr, Derivative):
        return evaluate(expr, sample)
    if isinstance(expr, Integral):
        return contribution_density(expr.expr, sample, constants=constants)
    if isinstance(expr, Norm):
        return norm_power_density(expr, sample, power=1)
    if isinstance(expr, Power):
        if isinstance(expr.base, Norm):
            return norm_power_density(expr.base, sample, power=float(expr.exp))
        base = contribution_density(expr.base, sample, constants=constants)
        return np.abs(as_field(base, reference)) ** float(expr.exp)
    if isinstance(expr, Product):
        result = np.ones_like(reference, dtype=float)
        for term in expr.terms:
            term_density = contribution_density(term, sample, constants=constants)
            result = result * as_field(term_density, reference)
        return result
    if isinstance(expr, Sum):
        result = np.zeros_like(reference, dtype=float)
        for term in expr.terms:
            term_density = contribution_density(term, sample, constants=constants)
            result = result + as_field(term_density, reference)
        return result
    value = evaluate(expr, sample)
    return as_field(value, reference)


def norm_power_density(norm: Norm, sample: FieldSample, *, power: float) -> Array:
    require_numpy()
    value = evaluate(norm.expr, sample)
    reference = sample.fields["omega"]
    field = np.abs(as_field(value, reference))
    p = float(norm.p)
    moment_density = field**p
    moment = float(np.mean(moment_density))
    if moment <= 1.0e-30:
        return np.zeros_like(reference, dtype=float)
    return moment_density * (moment ** (power / p - 1.0))


def as_field(value: Any, reference: Array) -> Array:
    require_numpy()
    if np.isscalar(value):
        return np.full_like(reference, float(value), dtype=float)
    return np.asarray(value, dtype=float)


def slice_array(field: Array, *, axis: str, index: int | None = None) -> Array:
    require_numpy()
    if field.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape {field.shape}")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if index is None:
        index = field.shape[axis_index] // 2
    if index < 0 or index >= field.shape[axis_index]:
        raise ValueError(f"slice index {index} outside axis {axis} with size {field.shape[axis_index]}")
    if axis == "x":
        return field[index, :, :]
    if axis == "y":
        return field[:, index, :]
    return field[:, :, index]


def render_contour_figure(
    fields: ContourFields,
    *,
    output_path: Path,
    title: str,
    subtitle: str,
    axis: str,
    index: int | None,
    dpi: int = 160,
) -> Path:
    require_numpy()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("omega field", fields.omega, "coolwarm", True),
        ("|grad omega|", fields.grad_omega, "viridis", False),
        ("LHS contribution density", fields.lhs_density, "magma", False),
        ("RHS absorption density", fields.rhs_density, "cividis", False),
        ("residual: LHS - RHS", fields.residual, "coolwarm", True),
        ("log10((LHS + delta)/(RHS + delta))", fields.log_ratio, "turbo", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9, color="#4b5563")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.065, top=0.86, hspace=0.34, wspace=0.25)

    extent = [0.0, 6.283185307179586, 0.0, 6.283185307179586]
    for ax, (label, field, cmap, centered) in zip(axes.flat, panels):
        sliced = slice_array(field, axis=axis, index=index)
        norm = None
        if centered:
            vmax = float(np.nanmax(np.abs(sliced)))
            if vmax > 0:
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        image = ax.imshow(sliced.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear")
        draw_contours(ax, sliced, extent=extent)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.82)

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def draw_contours(ax: Any, field: Array, *, extent: list[float]) -> None:
    require_numpy()
    finite = np.asarray(field, dtype=float)
    if not np.isfinite(finite).all():
        return
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= 1.0e-12:
        return
    levels = np.linspace(lo, hi, 9)[1:-1]
    x = np.linspace(extent[0], extent[1], finite.shape[0])
    y = np.linspace(extent[2], extent[3], finite.shape[1])
    ax.contour(x, y, finite.T, levels=levels, colors="white", linewidths=0.55, alpha=0.55)


def title_for_candidate(candidate: dict[str, Any]) -> str:
    target = str(candidate.get("target_name", "candidate"))
    try:
        inequality = f"${candidate_inequality_latex(candidate)}$"
    except (KeyError, TypeError, ValueError):
        lhs = str(candidate.get("lhs_text", "lhs"))
        rhs = str(candidate.get("rhs_text", "rhs"))
        inequality = f"{lhs} <= {rhs}"
    return shorten(f"{target}: {inequality}", width=128, placeholder="...")


def select_frontier_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    """Return candidates that advanced furthest through the verification gates."""

    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=frontier_sort_key):
        key = candidate_signature(candidate)
        if dedupe and key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def frontier_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    progress = verification_progress(candidate)
    score = int(candidate.get("score", 0) or 0)
    priority = int(candidate.get("priority", 0) or 0)
    return (-progress, -score, -priority, str(candidate.get("candidate_id", "")))


def verification_progress(candidate: dict[str, Any]) -> int:
    """Map queue state to a monotone pipeline progress score."""

    status = str(candidate.get("queue_status", "pending_symbolic"))
    status_progress = {
        "pending_symbolic": 0,
        "failed_symbolic": 0,
        "failed_scaling": 1,
        "failed_relevance": 2,
        "failed_numeric": 3,
        "pending_human_review": 4,
    }
    verification = candidate.get("verification")
    checks = verification.get("checks", []) if isinstance(verification, dict) else []
    passed = 0
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            if check.get("status") == "passed":
                passed += 1
                continue
            if check.get("status") in {"failed", "pending"}:
                break
    return max(status_progress.get(status, 0), passed)


def candidate_signature(candidate: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(candidate.get("lhs_text", "")),
        str(candidate.get("rhs_text", "")),
        (),
    )


def default_constant(name: str) -> float:
    if name in {"eps", "C", "C_eps"}:
        return 1.0
    return scalar_value(1.0)


def require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required for contour diagnostics")
