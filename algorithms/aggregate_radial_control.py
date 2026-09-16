"""Causal post-processing controls for a private aggregate.

The objects in this module operate *after* client-side privacy.  They consume
only the current private aggregate and state built from earlier private
aggregates.  Consequently they are post-processing for privacy; they do not
claim that smoothing or projection improves model utility.

Two ellipsoidal controls are deliberately kept separate:

``radial_ellipsoid``
    rescales the innovation on the ray from the predictor.  This has a simple
    closed form and a Mahalanobis non-degradation property when the target is
    inside the ellipsoid, but it is not the Euclidean projection.

``euclidean_ellipsoid_projection``
    is the true Euclidean projection onto a diagonal ellipsoid.  It is solved
    deterministically by bisection and has the usual Euclidean projection
    inequality for every target in the ellipsoid.

The predictable state is updated from the *uncorrected* candidate.  This keeps
all shadow controls on the same causal state, but repeated contaminated
aggregates can therefore contaminate the predictor and covariance persistently.
That limitation is intentional and must be audited rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


CONTROL_MODES = (
    "unchanged",
    "ema",
    "isotropic_clip",
    "radial_ellipsoid",
    "euclidean_ellipsoid_projection",
)


def _finite_vector(value: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{name} must be finite")
    return vector


def _positive_finite(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def _unit_interval(value: float, *, name: str, allow_one: bool = True) -> float:
    resolved = float(value)
    upper_ok = resolved <= 1.0 if allow_one else resolved < 1.0
    if not math.isfinite(resolved) or resolved <= 0.0 or not upper_ok:
        suffix = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{name} must lie in {suffix}")
    return resolved


def stable_l2_norm(vector: torch.Tensor | Any) -> float:
    """Return a finite float64 L2 norm without avoidable square overflow."""

    x = _finite_vector(vector, name="vector")
    scale = float(x.abs().max())
    if scale == 0.0:
        return 0.0
    normalized = x / scale
    norm = scale * math.sqrt(float(torch.dot(normalized, normalized)))
    if not math.isfinite(norm):
        raise ValueError("vector norm overflowed float64")
    return norm


def _variance_vector(variance: torch.Tensor | Any, *, dimension: int) -> torch.Tensor:
    value = _finite_vector(variance, name="variance")
    if value.numel() != dimension:
        raise ValueError("variance and aggregate dimensions differ")
    if bool((value <= 0.0).any()):
        raise ValueError("variance entries must be strictly positive")
    return value


def diagonal_mahalanobis_norm(
    vector: torch.Tensor | Any, variance: torch.Tensor | Any
) -> float:
    """Norm induced by ``diag(variance)^{-1}``, with finite checks."""

    x = _finite_vector(vector, name="vector")
    v = _variance_vector(variance, dimension=x.numel())
    standardized = x / torch.sqrt(v)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("standardization overflowed")
    return stable_l2_norm(standardized)


def ema_smooth_about_predictor(
    candidate: torch.Tensor | Any,
    predictor: torch.Tensor | Any,
    *,
    current_mix: float,
) -> torch.Tensor:
    """Return ``P + current_mix * (A-P)`` for a predictable ``P``."""

    a = _finite_vector(candidate, name="candidate")
    p = _finite_vector(predictor, name="predictor")
    if p.shape != a.shape:
        raise ValueError("candidate and predictor dimensions differ")
    mix = _unit_interval(current_mix, name="current_mix")
    result = p + mix * (a - p)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("EMA output is non-finite")
    return result


def isotropic_clip_about_predictor(
    candidate: torch.Tensor | Any,
    predictor: torch.Tensor | Any,
    *,
    radius: float,
) -> tuple[torch.Tensor, float]:
    """Project onto the Euclidean ball centered at the predictable predictor."""

    a = _finite_vector(candidate, name="candidate")
    p = _finite_vector(predictor, name="predictor")
    if p.shape != a.shape:
        raise ValueError("candidate and predictor dimensions differ")
    resolved_radius = _positive_finite(radius, name="radius")
    innovation = a - p
    norm = stable_l2_norm(innovation)
    if norm <= resolved_radius:
        return a.clone(), 1.0
    gamma = resolved_radius / norm
    result = p + gamma * innovation
    if not bool(torch.isfinite(result).all()):
        raise ValueError("isotropic projection output is non-finite")
    return result, gamma


def radial_ellipsoid_clip(
    candidate: torch.Tensor | Any,
    predictor: torch.Tensor | Any,
    variance: torch.Tensor | Any,
    *,
    radius: float,
) -> tuple[torch.Tensor, float]:
    """Radially clip ``A-P`` in a predictable diagonal Mahalanobis geometry."""

    a = _finite_vector(candidate, name="candidate")
    p = _finite_vector(predictor, name="predictor")
    if p.shape != a.shape:
        raise ValueError("candidate and predictor dimensions differ")
    v = _variance_vector(variance, dimension=a.numel())
    resolved_radius = _positive_finite(radius, name="radius")
    innovation = a - p
    norm = diagonal_mahalanobis_norm(innovation, v)
    if norm <= resolved_radius:
        return a.clone(), 1.0
    gamma = resolved_radius / norm
    result = p + gamma * innovation
    if not bool(torch.isfinite(result).all()):
        raise ValueError("radial ellipsoid output is non-finite")
    return result, gamma


def _projected_mahalanobis_sq(
    innovation: torch.Tensor, variance: torch.Tensor, multiplier: float
) -> float:
    # For y_j = v_j/(v_j+lambda) u_j, y_j^2/v_j is evaluated in a
    # numerically preferable form.  Inputs have already passed finite checks.
    denominator = variance + multiplier
    y = variance / denominator * innovation
    standardized = y / torch.sqrt(variance)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("ellipsoid projection intermediate is non-finite")
    norm = stable_l2_norm(standardized)
    value = norm * norm
    if not math.isfinite(value):
        raise ValueError("ellipsoid projection norm overflowed")
    return value


def euclidean_project_diagonal_ellipsoid(
    candidate: torch.Tensor | Any,
    predictor: torch.Tensor | Any,
    variance: torch.Tensor | Any,
    *,
    radius: float,
    tolerance: float = 1e-10,
    max_iterations: int = 160,
) -> tuple[torch.Tensor, float, int]:
    """Euclidean projection onto ``{P+y: sum(y_j^2/v_j)<=radius^2}``.

    The KKT solution is ``y_j=v_j u_j/(v_j+lambda)``.  A monotone bisection
    finds the unique non-negative multiplier.  ``lambda`` is a numerical
    Lagrange multiplier (a factor of two can be absorbed into its definition).
    """

    a = _finite_vector(candidate, name="candidate")
    p = _finite_vector(predictor, name="predictor")
    if p.shape != a.shape:
        raise ValueError("candidate and predictor dimensions differ")
    v = _variance_vector(variance, dimension=a.numel())
    resolved_radius = _positive_finite(radius, name="radius")
    resolved_tolerance = _positive_finite(tolerance, name="tolerance")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    innovation = a - p
    current = diagonal_mahalanobis_norm(innovation, v)
    if current <= resolved_radius:
        return a.clone(), 0.0, 0

    radius_sq = resolved_radius * resolved_radius
    if not math.isfinite(radius_sq):
        raise ValueError("radius squared overflowed")
    low = 0.0
    # Lambda has the same squared-coordinate units as ``variance``.  Starting
    # at 1.0 (or stopping at an absolute 1e-10) catastrophically overshrinks
    # small-variance ellipsoids, so both the bracket and tolerance are scaled
    # in variance units.
    variance_scale = float(v.max())
    high = variance_scale
    for _ in range(1024):
        if _projected_mahalanobis_sq(innovation, v, high) <= radius_sq:
            break
        high *= 2.0
        if not math.isfinite(high):
            raise ValueError("unable to bracket ellipsoid projection multiplier")
    else:  # pragma: no cover - defensive; finite inputs should bracket
        raise RuntimeError("ellipsoid projection multiplier was not bracketed")

    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        middle = 0.5 * (low + high)
        value = _projected_mahalanobis_sq(innovation, v, middle)
        if value > radius_sq:
            low = middle
        else:
            high = middle
        boundary_residual = abs(value - radius_sq)
        multiplier_tolerance = resolved_tolerance * max(
            variance_scale, high, torch.finfo(torch.float64).tiny
        )
        boundary_tolerance = resolved_tolerance * max(
            radius_sq, torch.finfo(torch.float64).tiny
        )
        if (
            high - low <= multiplier_tolerance
            and boundary_residual <= boundary_tolerance
        ):
            break

    multiplier = high  # feasible side of the bracket
    projected_innovation = v / (v + multiplier) * innovation
    result = p + projected_innovation
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Euclidean ellipsoid projection output is non-finite")
    feasibility = diagonal_mahalanobis_norm(projected_innovation, v)
    allowed = resolved_radius * (1.0 + 10.0 * resolved_tolerance)
    if feasibility > allowed:
        raise RuntimeError("Euclidean ellipsoid projection is infeasible")
    if abs(feasibility - resolved_radius) > 50.0 * resolved_tolerance * max(
        resolved_radius, torch.finfo(torch.float64).tiny
    ):
        raise RuntimeError("Euclidean ellipsoid projection missed the boundary")
    return result, multiplier, iterations


@dataclass(frozen=True)
class AggregateControlConfig:
    """Public configuration of the predictable post-processing state."""

    predictor_rate: float = 0.25
    covariance_rate: float = 0.10
    initial_variance: float = 1.0
    variance_ridge: float = 1e-8
    min_history: int = 1
    projection_tolerance: float = 1e-10
    projection_max_iterations: int = 160

    def validate(self) -> None:
        _unit_interval(self.predictor_rate, name="predictor_rate")
        _unit_interval(self.covariance_rate, name="covariance_rate")
        _positive_finite(self.initial_variance, name="initial_variance")
        _positive_finite(self.variance_ridge, name="variance_ridge")
        _positive_finite(self.projection_tolerance, name="projection_tolerance")
        if int(self.min_history) < 1:
            raise ValueError("min_history must be positive")
        if int(self.projection_max_iterations) < 1:
            raise ValueError("projection_max_iterations must be positive")


@dataclass(frozen=True)
class PredictableSnapshot:
    """State available before the current aggregate is observed."""

    round_num: int
    observations: int
    ready: bool
    predictor: torch.Tensor
    variance: torch.Tensor


class PredictableAggregateState:
    """Causal diagonal predictor/covariance state for aggregate controls."""

    def __init__(self, dimension: int, config: AggregateControlConfig):
        config.validate()
        if int(dimension) < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.config = config
        self._next_round = 0
        self._observations = 0
        self._predictor = torch.zeros(self.dimension, dtype=torch.float64)
        self._variance = torch.full(
            (self.dimension,),
            float(config.initial_variance),
            dtype=torch.float64,
        )

    @property
    def observations(self) -> int:
        return self._observations

    def snapshot(self, round_num: int) -> PredictableSnapshot:
        if int(round_num) != self._next_round:
            raise ValueError(
                f"non-sequential round: expected {self._next_round}, got {round_num}"
            )
        return PredictableSnapshot(
            round_num=int(round_num),
            observations=self._observations,
            ready=self._observations >= int(self.config.min_history),
            predictor=self._predictor.clone(),
            variance=self._variance.clone(),
        )

    def observe_uncorrected(self, candidate: torch.Tensor | Any, round_num: int) -> None:
        """Commit ``A_t`` after all controls used the pre-round snapshot."""

        if int(round_num) != self._next_round:
            raise ValueError(
                f"non-sequential round: expected {self._next_round}, got {round_num}"
            )
        a = _finite_vector(candidate, name="uncorrected candidate")
        if a.numel() != self.dimension:
            raise ValueError("candidate dimension changed")
        if self._observations == 0:
            # A_0 initializes the predictor.  The public initial variance is
            # retained; treating A_0-0 as a stochastic residual would create
            # an arbitrary, origin-dependent covariance spike.
            self._predictor = a.clone()
        else:
            residual = a - self._predictor
            new_variance = (
                (1.0 - self.config.covariance_rate) * self._variance
                + self.config.covariance_rate * residual.square()
            ).clamp_min(self.config.variance_ridge)
            new_predictor = (
                (1.0 - self.config.predictor_rate) * self._predictor
                + self.config.predictor_rate * a
            )
            if not bool(torch.isfinite(new_predictor).all()) or not bool(
                torch.isfinite(new_variance).all()
            ):
                raise ValueError("predictable state update became non-finite")
            self._predictor = new_predictor
            self._variance = new_variance
        self._observations += 1
        self._next_round += 1


def apply_aggregate_control(
    mode: str,
    candidate: torch.Tensor | Any,
    snapshot: PredictableSnapshot,
    *,
    current_mix: float | None = None,
    radius: float | None = None,
    projection_tolerance: float = 1e-10,
    projection_max_iterations: int = 160,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str | None]]:
    """Apply one controller using a fixed pre-current predictable snapshot."""

    resolved = str(mode)
    if resolved not in CONTROL_MODES:
        raise ValueError(f"unknown aggregate control mode: {resolved}")
    a = _finite_vector(candidate, name="candidate")
    p = _finite_vector(snapshot.predictor, name="predictor")
    v = _variance_vector(snapshot.variance, dimension=a.numel())
    if p.shape != a.shape:
        raise ValueError("snapshot dimension changed")
    if not snapshot.ready or resolved == "unchanged":
        output = a.clone()
        gamma: float | None = 1.0
        multiplier: float | None = 0.0
        iterations = 0
    elif resolved == "ema":
        if current_mix is None:
            raise ValueError("EMA requires current_mix")
        output = ema_smooth_about_predictor(a, p, current_mix=current_mix)
        gamma = float(current_mix)
        multiplier = None
        iterations = 0
    elif resolved == "isotropic_clip":
        if radius is None:
            raise ValueError("isotropic clipping requires radius")
        output, gamma = isotropic_clip_about_predictor(a, p, radius=radius)
        multiplier = None
        iterations = 0
    elif resolved == "radial_ellipsoid":
        if radius is None:
            raise ValueError("radial ellipsoid clipping requires radius")
        output, gamma = radial_ellipsoid_clip(a, p, v, radius=radius)
        multiplier = None
        iterations = 0
    else:
        if radius is None:
            raise ValueError("Euclidean ellipsoid projection requires radius")
        output, multiplier, iterations = euclidean_project_diagonal_ellipsoid(
            a,
            p,
            v,
            radius=radius,
            tolerance=projection_tolerance,
            max_iterations=projection_max_iterations,
        )
        gamma = None
    innovation = a - p
    correction = output - a
    variance_condition = float(v.max() / v.min())
    diagnostics: dict[str, float | int | bool | str | None] = {
        "mode": resolved,
        "ready": bool(snapshot.ready),
        "history_observations": int(snapshot.observations),
        "predictor_is_strictly_past_measurable": True,
        "state_update_source": "uncorrected_current_candidate_after_control",
        "persistent_contamination_possible": True,
        "candidate_norm": stable_l2_norm(a),
        "predictor_norm": stable_l2_norm(p),
        "innovation_norm": stable_l2_norm(innovation),
        "innovation_mahalanobis_norm": diagonal_mahalanobis_norm(innovation, v),
        "output_norm": stable_l2_norm(output),
        "correction_norm": stable_l2_norm(correction),
        "correction_applied": not torch.equal(output, a),
        "gamma": gamma,
        "lagrange_multiplier": multiplier,
        "projection_iterations": int(iterations),
        "variance_min": float(v.min()),
        "variance_max": float(v.max()),
        "variance_mean": float(v.mean()),
        "variance_condition": variance_condition,
        "radius": None if radius is None else float(radius),
        "current_mix": None if current_mix is None else float(current_mix),
    }
    if not all(
        value is None
        or isinstance(value, (bool, int, str))
        or math.isfinite(float(value))
        for value in diagnostics.values()
    ):
        raise ValueError("aggregate control diagnostics are non-finite")
    return output, diagnostics


__all__ = [
    "AggregateControlConfig",
    "CONTROL_MODES",
    "PredictableAggregateState",
    "PredictableSnapshot",
    "apply_aggregate_control",
    "diagonal_mahalanobis_norm",
    "ema_smooth_about_predictor",
    "euclidean_project_diagonal_ellipsoid",
    "isotropic_clip_about_predictor",
    "radial_ellipsoid_clip",
    "stable_l2_norm",
]
