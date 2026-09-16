r"""Gaussian-aware robust references for locally private FAR experiments.

This module is intentionally independent of the :class:`FLAlgorithm` registry.
It provides the reference and score primitives needed to evaluate the
``F_{Sigma,Hub}`` proposal before wiring it into an end-to-end algorithm.

The central design choice is to use the public DP covariance to decide *when*
a residual enters the Huber tail, but not to precision-weight the quadratic
centre of the objective.  For client ``i`` and public block ``b`` we use

.. math::

    \ell_{i,b}(r_b)
      = v_{i,b}\,\rho_{c_{i,b}}
        \left(\frac{\|y_{i,b}-r_b\|_2}{\sqrt{v_{i,b}}}\right),

    c_{i,b}=\min\left\{c_{0,b},
        \frac{\rho_b}{\sqrt{v_{i,b}}}\right\}.

Inside the Huber ball this is exactly
``0.5 * ||y[i,b] - r[b]||^2`` for every client.  Thus heterogeneous privacy
noise does not silently replace the equal-client estimand by an
inverse-variance estimand.  Outside the ball, the Euclidean norm of a client's
block gradient is at most ``rho_b``.

All covariance inputs must be public/authenticated mechanism parameters or
pre-registered modelling quantities.  Accepting a variance declared by an
untrusted client would let a Byzantine client manipulate its threshold.

``gaussian_aware_crossfit_bounded_correction`` implements the subsequent G0e
candidate.  It retains a certified robust pilot (normally one-step centered
clipping), calibrates statistical-tail diagnostics with leave-one-out
references, and limits the Gaussian-aware displacement by a public correction
budget.  This keeps robustness, calibration, and stability as separate
auditable obligations.

``gaussian_aware_budget_allocated_correction`` implements G0f.  In contrast to
G0e's identical cap in every block, it allocates one public complete-client
budget ``G`` according to authenticated covariance radii ``a[i,b]``:

.. math::

    A=\max_j\lVert a_j\rVert_2,
    \qquad G_{i,b}=G\,a_{i,b}/A.

Consequently every client satisfies ``||G_i|| <= G`` while the relative
block and client scales remain Gaussian-aware.  Two deliberately weaker
controls are implemented by the same routine: normalization separately for
each client (which is blind to between-client noise scale) and equal block
caps.  Covariance radii are never inferred from the evaluated uploads.

``gaussian_aware_fixed_anchor_gated_reference`` implements the primary G0g-K1
primitive.  It uses authenticated covariance only to decide whether a block
residual is statistically plausible.  Its Huber influence cap is common to
all clients and independent of covariance, so statistical tolerance cannot
grant a noisier identity more influence.  The anchor is fixed (public or
lagged), which gives the exact replace-one certificate ``2*G/n``.

``gaussian_aware_fixed_anchor_scalar_gated_reference`` is the G0g-K2
refinement.  It collapses the blockwise standardized residuals to one public
client-level statistic, then multiplies one *global* L2-clipped contribution
by one scalar gate.  With the gate disabled this is exactly one-step centered
clipping, avoiding the blockwise geometric distortion diagnosed in G0g-K1.

``gaussian_aware_fixed_anchor_dual_gated_reference`` is the G0g-K3
refinement.  It intersects K2's covariance-aware statistical acceptance with
a second, identity-blind acceptance region built from common public radii.
The final gate is the minimum of the two scalar gates.  A noisy identity can
therefore not obtain a larger authorised acceptance region merely because its
authenticated DP variance is larger.  The contribution is still clipped once
by one common global L2 cap and averaged with denominator ``n``.

``gaussian_aware_crossfit_gated_correction`` is the adaptive finite-iteration
G0g ablation.  It freezes one scalar ramp gate per client from leave-one-out
residuals before optimizing a regularized, full-vector radial-Huber objective.
The gates are never normalized by their sum.  This detail preserves both the
meaning of rejection and the stated sensitivity certificate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

TensorOrScalar = torch.Tensor | float | int | Sequence[float]


@torch.no_grad()
def gaussian_radial_thresholds(
    block_sizes: Sequence[int],
    *,
    tail_probability: float = 0.01,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    r"""Return public Gaussian radial thresholds for each block.

    If ``Z_b ~ N(0, v I_{d_b})``, then ``||Z_b||/sqrt(v)`` is the square root
    of a chi-square random variable.  Laurent--Massart gives

    .. math::

        \Pr\{\|Z_b\|/\sqrt v >
        \sqrt{d_b+2\sqrt{d_b x}+2x}\}\le e^{-x}.

    This function sets ``x=log(1/tail_probability)``.  It avoids the common
    dimensional mistake of using a scalar such as ``2.5`` for a 16- or
    60,000-dimensional radial norm.  The bound is conservative by design and
    depends only on public block dimensions and a pre-registered probability.
    """

    blocks = tuple(int(size) for size in block_sizes)
    if not blocks or any(size <= 0 for size in blocks):
        raise ValueError("block_sizes must contain positive integers")
    probability = float(tail_probability)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("tail_probability must lie strictly between zero and one")
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    dimensions = torch.tensor(blocks, device=device, dtype=work_dtype)
    x = math.log(1.0 / probability)
    return (dimensions + 2.0 * torch.sqrt(dimensions * x) + 2.0 * x).sqrt()


def _validate_vectors(vectors: torch.Tensor) -> tuple[int, int, torch.dtype]:
    if not isinstance(vectors, torch.Tensor):
        raise TypeError("vectors must be a torch.Tensor")
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ValueError("vectors must have shape (n,d) with n,d >= 1")
    if not vectors.is_floating_point():
        raise TypeError("vectors must use a floating-point dtype")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")
    work_dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    return int(vectors.shape[0]), int(vectors.shape[1]), work_dtype


def _resolve_blocks(
    dimension: int, block_sizes: Sequence[int] | None
) -> tuple[int, ...]:
    if block_sizes is None:
        return (dimension,)
    blocks = tuple(int(size) for size in block_sizes)
    if not blocks or any(size <= 0 for size in blocks):
        raise ValueError("block_sizes must contain positive integers")
    if sum(blocks) != dimension:
        raise ValueError("block_sizes must sum to the vector dimension")
    return blocks


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    result: list[slice] = []
    start = 0
    for size in block_sizes:
        result.append(slice(start, start + int(size)))
        start += int(size)
    return tuple(result)


def _as_block_matrix(
    value: TensorOrScalar | None,
    *,
    name: str,
    n: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
    strictly_positive: bool,
) -> torch.Tensor:
    """Broadcast a public scalar/block/client quantity to shape ``(n,B)``."""

    if value is None:
        result = torch.full((n, num_blocks), default, device=device, dtype=dtype)
    else:
        result = torch.as_tensor(value, device=device, dtype=dtype)
        if result.ndim == 0:
            result = result.expand(n, num_blocks)
        elif result.shape == (n,) and result.shape == (num_blocks,) and n > 1:
            raise ValueError(
                f"{name} has ambiguous one-dimensional shape ({n},) because "
                "n == B; use (n,1) for per-client values or (1,B) for "
                "per-block values"
            )
        elif result.shape == (n,):
            result = result[:, None].expand(n, num_blocks)
        elif result.shape == (num_blocks,):
            result = result[None, :].expand(n, num_blocks)
        elif result.shape == (n, 1):
            result = result.expand(n, num_blocks)
        elif result.shape == (1, num_blocks):
            result = result.expand(n, num_blocks)
        elif result.shape != (n, num_blocks):
            raise ValueError(
                f"{name} must be scalar, (n,), (B,), (n,1), (1,B), or "
                f"(n,B); "
                f"received {tuple(result.shape)} for n={n}, B={num_blocks}"
            )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if strictly_positive:
        if bool((result <= 0.0).any()):
            raise ValueError(f"{name} must be strictly positive")
    elif bool((result < 0.0).any()):
        raise ValueError(f"{name} must be non-negative")
    return result.clone()


def _as_block_vector(
    value: TensorOrScalar,
    *,
    name: str,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.ndim == 0:
        result = result.expand(num_blocks)
    elif result.shape != (num_blocks,):
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    if not bool(torch.isfinite(result).all()) or bool((result <= 0.0).any()):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result.clone()


def _clip_rows_at_radii(rows: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(rows, dim=1)
    factors = (radii / norms.clamp_min(torch.finfo(rows.dtype).tiny)).clamp(max=1.0)
    return rows * factors[:, None]


def _stable_norm_and_clip_rows(
    rows: torch.Tensor, radii: torch.Tensor, *, quantity_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute row norms without squaring large values and clip safely.

    This helper deliberately stays in the input dtype so it also works on MPS,
    where float64 tensors are unsupported.  It rescales each row before the
    Euclidean norm and constructs a clipped direction from the rescaled row;
    it never multiplies a huge row by an underflowed zero factor.
    """

    norms, scaled, unit_norms = _stable_row_norms(
        rows, quantity_name=quantity_name
    )
    if radii.shape != (rows.shape[0],):
        raise ValueError("radii must contain exactly one value per row")
    if not bool(torch.isfinite(radii).all()) or bool((radii <= 0.0).any()):
        raise ValueError("radii must be finite and strictly positive")
    safe_unit_norms = torch.where(
        unit_norms > 0.0, unit_norms, torch.ones_like(unit_norms)
    )
    clipped_directions = scaled / safe_unit_norms[:, None]
    clipped_candidates = clipped_directions * radii[:, None]
    clipped = torch.where((norms > radii)[:, None], clipped_candidates, rows)
    if not bool(torch.isfinite(clipped).all()):
        raise ValueError(f"{quantity_name} clipped output must be finite")
    return norms, clipped


def _stable_row_norms(
    rows: torch.Tensor, *, quantity_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return finite Euclidean row norms using scale-before-square arithmetic."""

    if rows.ndim != 2:
        raise ValueError(f"{quantity_name} must be a matrix")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError(f"{quantity_name} must be finite before norm evaluation")
    row_scales = rows.abs().amax(dim=1)
    safe_scales = torch.where(row_scales > 0.0, row_scales, torch.ones_like(row_scales))
    scaled = rows / safe_scales[:, None]
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError(f"{quantity_name} rescaling produced a non-finite value")
    unit_norms = torch.sqrt(torch.sum(scaled.square(), dim=1))
    if not bool(torch.isfinite(unit_norms).all()):
        raise ValueError(f"{quantity_name} scaled norms must be finite")
    maximum = torch.full_like(row_scales, torch.finfo(rows.dtype).max)
    overflow = (row_scales > 0.0) & (unit_norms > maximum / safe_scales)
    if bool(overflow.any()):
        raise ValueError(
            f"{quantity_name} row norm exceeds the finite range of {rows.dtype}"
        )
    norms = row_scales * unit_norms
    if not bool(torch.isfinite(norms).all()):
        raise ValueError(f"{quantity_name} raw norms must be finite")
    return norms, scaled, unit_norms


def _clip_vector(vector: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    factor = (float(radius) / norm.clamp_min(torch.finfo(vector.dtype).tiny)).clamp(
        max=1.0
    )
    return vector * factor


def _effective_variances(
    *,
    noise_variances: TensorOrScalar,
    heterogeneity_variances: TensorOrScalar | None,
    reference_variances: TensorOrScalar | None,
    variance_floor: float,
    n: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not math.isfinite(float(variance_floor)) or float(variance_floor) <= 0.0:
        raise ValueError("variance_floor must be finite and strictly positive")
    if float(variance_floor) < torch.finfo(dtype).tiny:
        raise ValueError(
            f"variance_floor underflows in {dtype}; require at least "
            f"{torch.finfo(dtype).tiny:.6g}"
        )
    noise = _as_block_matrix(
        noise_variances,
        name="noise_variances",
        n=n,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
        default=0.0,
        strictly_positive=False,
    )
    heterogeneity = _as_block_matrix(
        heterogeneity_variances,
        name="heterogeneity_variances",
        n=n,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
        default=0.0,
        strictly_positive=False,
    )
    reference = _as_block_matrix(
        reference_variances,
        name="reference_variances",
        n=n,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
        default=0.0,
        strictly_positive=False,
    )
    effective = noise + heterogeneity + reference + float(variance_floor)
    if not bool(torch.isfinite(effective).all()):
        raise ValueError("effective variances overflowed in the working dtype")
    if bool((effective <= 0.0).any()):
        raise ValueError("effective variances must remain strictly positive")
    return effective


def _huber_gradient(
    point: torch.Tensor,
    vectors: torch.Tensor,
    *,
    slices: Sequence[slice],
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean data gradient and client/block tail indicators."""

    gradient = torch.zeros_like(point)
    tail = torch.zeros(
        (vectors.shape[0], len(slices)), dtype=torch.bool, device=vectors.device
    )
    for block_index, block_slice in enumerate(slices):
        residuals = point[None, block_slice] - vectors[:, block_slice]
        norms = torch.linalg.vector_norm(residuals, dim=1)
        block_radii = radii[:, block_index]
        tail[:, block_index] = norms > block_radii
        gradient[block_slice] = _clip_rows_at_radii(residuals, block_radii).mean(dim=0)
    return gradient, tail


def _huber_objective(
    point: torch.Tensor,
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    slices: Sequence[slice],
    radii: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    per_client = torch.zeros(
        vectors.shape[0], device=vectors.device, dtype=vectors.dtype
    )
    for block_index, block_slice in enumerate(slices):
        norms = torch.linalg.vector_norm(
            vectors[:, block_slice] - point[None, block_slice], dim=1
        )
        block_radii = radii[:, block_index]
        quadratic = 0.5 * norms.square()
        linear = block_radii * norms - 0.5 * block_radii.square()
        per_client = per_client + torch.where(norms <= block_radii, quadratic, linear)
    return (
        per_client.mean()
        + 0.5 * float(regularization) * (point - anchor).square().sum()
    )


@torch.no_grad()
def gaussian_aware_huber_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    noise_variances: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    heterogeneity_variances: TensorOrScalar | None = None,
    reference_variances: TensorOrScalar | None = None,
    standardized_threshold: TensorOrScalar | None = None,
    null_tail_probability: float = 0.01,
    influence_cap: TensorOrScalar = 1.0,
    regularization: float = 0.2,
    num_steps: int = 20,
    step_size: float | None = None,
    variance_floor: float = 1e-12,
    output_radius: float | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Compute the finite-step equal-centre Gaussian-aware Huber reference.

    ``noise_variances`` and the optional variance terms are scalar variances
    per coordinate.  They may be scalar, one value per client ``(n,)``, one
    value per block ``(B,)``, or a full public table ``(n,B)``.

    The solver always executes exactly ``num_steps`` iterations.  With the
    default step size ``2/(1+2*regularization)``, its contraction factor is
    ``1/(1+2*regularization)``.  More generally, the accepted public step size
    has contraction factor

    ``max(|1-eta*gamma|, |1-eta*(1+gamma)|) < 1``.

    Conditional on a fixed anchor, fixed covariance table, and replace-one
    cohorts preserving client identities, the returned iterate satisfies the
    finite-solver stability certificate reported in the diagnostics.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    if not math.isfinite(float(regularization)) or float(regularization) <= 0.0:
        raise ValueError("regularization must be finite and strictly positive")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a public integer >= 1")
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)
    work_anchor = torch.as_tensor(anchor, device=device, dtype=work_dtype).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")

    effective_variances = _effective_variances(
        noise_variances=noise_variances,
        heterogeneity_variances=heterogeneity_variances,
        reference_variances=reference_variances,
        variance_floor=variance_floor,
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
    )
    if standardized_threshold is None:
        thresholds = gaussian_radial_thresholds(
            blocks,
            tail_probability=null_tail_probability,
            device=device,
            dtype=work_dtype,
        )
        threshold_source = "laurent_massart_gaussian_radius"
    else:
        thresholds = _as_block_vector(
            standardized_threshold,
            name="standardized_threshold",
            num_blocks=len(blocks),
            device=device,
            dtype=work_dtype,
        )
        threshold_source = "explicit_public_radius"
    caps = _as_block_vector(
        influence_cap,
        name="influence_cap",
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
    )
    # The DP covariance determines the onset of the tail; the Euclidean cap
    # prevents a high-variance or Byzantine client from buying unbounded
    # influence through a large threshold.
    covariance_radii = effective_variances.sqrt() * thresholds[None, :]
    radii = torch.minimum(covariance_radii, caps[None, :])
    # This diagnostic is deliberately computed before the optimisation.  It
    # verifies whether the advertised DP covariance actually changes the
    # Huber transition radius, instead of being everywhere masked by the
    # public Euclidean influence cap.  Equality is counted as cap-limited:
    # only a strict covariance radius below the cap certifies an active
    # covariance branch.
    covariance_limited = covariance_radii < caps[None, :]

    gamma = float(regularization)
    eta = 2.0 / (1.0 + 2.0 * gamma) if step_size is None else float(step_size)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("step_size must be finite and strictly positive")
    contraction = max(abs(1.0 - eta * gamma), abs(1.0 - eta * (1.0 + gamma)))
    if contraction >= 1.0:
        raise ValueError(
            "step_size does not define a contractive public solver; require "
            "max(|1-eta*gamma|,|1-eta*(1+gamma)|) < 1"
        )

    point = work_anchor.clone()
    initial_objective = _huber_objective(
        point,
        work_vectors,
        anchor=work_anchor,
        slices=slices,
        radii=radii,
        regularization=gamma,
    )
    tail = torch.zeros((n, len(blocks)), dtype=torch.bool, device=device)
    for _ in range(num_steps):
        data_gradient, tail = _huber_gradient(
            point, work_vectors, slices=slices, radii=radii
        )
        gradient = data_gradient + gamma * (point - work_anchor)
        point = point - eta * gradient

    data_gradient, tail = _huber_gradient(
        point, work_vectors, slices=slices, radii=radii
    )
    final_gradient = data_gradient + gamma * (point - work_anchor)
    unprojected_point = point
    if output_radius is not None:
        if not math.isfinite(float(output_radius)) or float(output_radius) <= 0.0:
            raise ValueError("output_radius must be finite and positive when provided")
        point = _clip_vector(point, float(output_radius))

    # G is a public cap on the norm of one complete client gradient.  The
    # recurrence delta_{k+1} <= q*delta_k + 2*eta*G/n starts from zero.
    public_gradient_cap = float(torch.linalg.vector_norm(caps).item())
    geometric_sum = (1.0 - contraction**num_steps) / (1.0 - contraction)
    replace_one_bound = 2.0 * eta * public_gradient_cap * geometric_sum / n
    exact_minimizer_bound = 2.0 * public_gradient_cap / (gamma * n)
    finite_solver_error_bound = contraction**num_steps * public_gradient_cap / gamma

    # Do not cast a certified reference back to float16/bfloat16: rounding is
    # not a non-expansive map and can violate a small replace-one bound.  The
    # solver output therefore remains in its float32/float64 working dtype.
    result = point
    if not return_diagnostics:
        return result
    final_objective = _huber_objective(
        unprojected_point,
        work_vectors,
        anchor=work_anchor,
        slices=slices,
        radii=radii,
        regularization=gamma,
    )
    diagnostics: dict[str, Any] = {
        "reference_name": "equal_centre_gaussian_aware_huber",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "num_steps": num_steps,
        "step_size": eta,
        "solver_contraction": contraction,
        "gradient_residual_norm": float(
            torch.linalg.vector_norm(final_gradient).item()
        ),
        "objective_initial": float(initial_objective.item()),
        "objective_final": float(final_objective.item()),
        "tail_fraction_client_blocks": float(tail.to(work_dtype).mean().item()),
        "effective_radius_min": float(radii.min().item()),
        "effective_radius_max": float(radii.max().item()),
        "covariance_radius_min": float(covariance_radii.min().item()),
        "covariance_radius_max": float(covariance_radii.max().item()),
        "fraction_covariance_limited_client_blocks": float(
            covariance_limited.to(work_dtype).mean().item()
        ),
        "fraction_covariance_limited_and_huber_tail_client_blocks": float(
            (covariance_limited & tail).to(work_dtype).mean().item()
        ),
        "fraction_cap_limited_and_huber_tail_client_blocks": float(
            ((~covariance_limited) & tail).to(work_dtype).mean().item()
        ),
        "covariance_branch_active": bool(covariance_limited.any().item()),
        "covariance_branch_changes_influence_at_final_iterate": bool(
            (covariance_limited & tail).any().item()
        ),
        "standardized_radial_thresholds": [
            float(value) for value in thresholds.detach().cpu().tolist()
        ],
        "standardized_threshold_source": threshold_source,
        "standardized_threshold_semantics": (
            "threshold_on_block_l2_norm_divided_by_sqrt_variance"
        ),
        "null_tail_probability_per_block": (
            float(null_tail_probability) if standardized_threshold is None else None
        ),
        "null_tail_probability_union_bound": (
            min(1.0, len(blocks) * float(null_tail_probability))
            if standardized_threshold is None
            else None
        ),
        "euclidean_influence_caps": [
            float(value) for value in caps.detach().cpu().tolist()
        ],
        "public_client_gradient_cap": public_gradient_cap,
        "influence_cap_is_public_hyperparameter_not_data_calibrated": True,
        "finite_solver_replace_one_bound": replace_one_bound,
        "exact_minimizer_replace_one_bound": exact_minimizer_bound,
        "finite_solver_error_bound_before_output_projection": (
            finite_solver_error_bound
        ),
        "certificate_adjacency": "replace_one_fixed_client_identity",
        "certificate_requires_fixed_public_covariances": True,
        "certificate_requires_fixed_anchor": True,
        "central_zone_preserves_equal_client_curvature": True,
        "output_radius": float(output_radius) if output_radius is not None else None,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(result.dtype),
        "certificate_arithmetic": "exact_real_bound_reported_float_runtime",
        "solver_diagnostics_point": "unprojected_fixed_iteration_output",
    }
    return result, diagnostics


@torch.no_grad()
def gaussian_aware_huber_leave_one_out(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    noise_variances: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    heterogeneity_variances: TensorOrScalar | None = None,
    reference_variances: TensorOrScalar | None = None,
    standardized_threshold: TensorOrScalar | None = None,
    null_tail_probability: float = 0.01,
    influence_cap: TensorOrScalar = 1.0,
    regularization: float = 0.2,
    num_steps: int = 20,
    step_size: float | None = None,
    variance_floor: float = 1e-12,
    output_radius: float | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Compute one exact finite-step reference excluding each scored client.

    This deliberately recomputes the fixed-iteration solver ``n`` times.  It
    is a transparent research implementation for ``n`` in the tens, not yet
    an optimized production approximation.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    if n < 2:
        raise ValueError("leave-one-out references require at least two clients")
    blocks = _resolve_blocks(dimension, block_sizes)
    noise_matrix = _as_block_matrix(
        noise_variances,
        name="noise_variances",
        n=n,
        num_blocks=len(blocks),
        device=vectors.device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=False,
    )
    heterogeneity_matrix = _as_block_matrix(
        heterogeneity_variances,
        name="heterogeneity_variances",
        n=n,
        num_blocks=len(blocks),
        device=vectors.device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=False,
    )
    reference_matrix = _as_block_matrix(
        reference_variances,
        name="reference_variances",
        n=n,
        num_blocks=len(blocks),
        device=vectors.device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=False,
    )
    references: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    client_indices = torch.arange(n, device=vectors.device)
    for excluded in range(n):
        keep = client_indices != excluded
        result = gaussian_aware_huber_reference(
            vectors[keep],
            anchor=anchor,
            noise_variances=noise_matrix[keep],
            block_sizes=blocks,
            heterogeneity_variances=heterogeneity_matrix[keep],
            reference_variances=reference_matrix[keep],
            standardized_threshold=standardized_threshold,
            null_tail_probability=null_tail_probability,
            influence_cap=influence_cap,
            regularization=regularization,
            num_steps=num_steps,
            step_size=step_size,
            variance_floor=variance_floor,
            output_radius=output_radius,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            reference, run_diagnostics = result
            references.append(reference)
            diagnostics.append(run_diagnostics)
        else:
            references.append(result)
    stacked = torch.stack(references)
    if not return_diagnostics:
        return stacked
    return stacked, {
        "reference_name": "equal_centre_gaussian_aware_huber_leave_one_out",
        "num_references": n,
        "cohort_size_per_reference": n - 1,
        "per_reference": diagnostics,
        "reference_norm_min": float(
            torch.linalg.vector_norm(stacked, dim=1).min().item()
        ),
        "reference_norm_max": float(
            torch.linalg.vector_norm(stacked, dim=1).max().item()
        ),
    }


@torch.no_grad()
def gaussian_aware_crossfit_bounded_correction(
    vectors: torch.Tensor,
    *,
    pilot: torch.Tensor,
    crossfit_references: torch.Tensor,
    statistical_radii: TensorOrScalar,
    deployed_radii: TensorOrScalar,
    pilot_replace_one_bound: float,
    block_sizes: Sequence[int] | None = None,
    influence_cap: TensorOrScalar = 1.0,
    regularization: float = 0.2,
    correction_budget: float = 0.1,
    num_steps: int = 20,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the G0e correction bounded around a robust pilot.

    G0e deliberately separates three roles which were entangled in G0d:

    * ``pilot`` is the robust centre, intended to be the one-step
      :math:`F_{\mathrm{CC}}` computed from the current cohort;
    * ``crossfit_references[i]`` is a leave-one-client-out centre used only
      to audit the pre-calibrated statistical radius of client ``i``;
    * ``statistical_radii`` are the frozen cross-fitted calibration radii;
    * ``deployed_radii`` add the pre-registered crossfit-to-full-reference
      margin before the public ``influence_cap`` is applied.

    The finite-step Gaussian-aware pilot is

    .. math::

        p_K = \operatorname{GD}_K\!\left[
          \frac1n\sum_{i,b}\rho_{R_{i,b}}(p_b-x_{i,b})
          + \frac\gamma2\lVert p-a\rVert_2^2
        \right],
        \qquad p_0=a,

    where ``a=pilot`` and
    :math:`R_{i,b}=\min\{R^{\rm deploy}_{i,b},G_b\}`.  The returned reference is

    .. math::

        F_{\mathrm{G0e}}=(1-\beta)a+\beta p_K,
        \qquad
        \beta=\min\left\{1,
          \frac{\gamma B_{\rm corr}}{G(1-q^K)}\right\},
        \qquad G=\sqrt{\sum_b G_b^2}.

    Here ``q=1/(1+2*regularization)``.  The Huber data gradient of one
    complete client has norm at most ``G``.  Consequently
    ``||p_K-a|| <= G/gamma * (1-q**K)`` and the correction obeys
    ``||F_G0e-a|| <= correction_budget``.  The fixed public step size is
    ``2/(1+2*regularization)``; fixing it rather than accepting a tuned step
    size keeps the correction and stability certificates unambiguous.  The
    simpler denominator ``G`` (without ``1-q**K``) is a valid but more
    conservative asymptotic choice.

    Conditional replace-one certificate
    -----------------------------------
    Assume that client identities, block structure, statistical/deployed
    radii, caps, regularization, iteration count, and correction budget are
    identical on neighbouring cohorts, and that the supplied pilot satisfies

    ``||a(U)-a(U')|| <= pilot_replace_one_bound``.

    Replacing one upload changes the averaged, clipped data gradient at a
    common point by at most ``2*G/n``.  With
    ``q=1/(1+2*regularization)`` and ``eta=2/(1+2*regularization)``, the
    actually returned finite iterate therefore satisfies

    ``Delta(F_G0e) <= Delta(pilot)``
    ``+ beta * 2*eta*G/n * (1-q**K)/(1-q)``.

    The exact-minimizer analogue replaces the last factor by
    ``2*beta*G/(regularization*n)``.  The cross-fitted references are absent
    from these bounds because they affect diagnostics only, not the returned
    reference.  If they are released, their own privacy/stability accounting
    must be handled separately.

    Parameters
    ----------
    vectors:
        Matrix of current client uploads with shape ``(n,d)``.
    pilot:
        Finite robust pilot of shape ``(d,)``.  Its data dependence is
        explicitly represented by ``pilot_replace_one_bound``.
    crossfit_references:
        Frozen/evaluated leave-one-out references with shape ``(n,d)`` used
        to measure the statistical-tail calibration.  They do not enter the
        optimization objective.
    statistical_radii:
        Positive split-calibration Euclidean radii, broadcastable to
        ``(n,B)``.  The cross-fitted residual is compared to this radius,
        without the crossfit-to-full-reference margin.
    deployed_radii:
        Positive solver radii, broadcastable to ``(n,B)``.  They must equal
        the statistical radii plus a non-negative, pre-registered
        crossfit-to-full-reference margin.  This routine verifies the
        necessary inequality ``deployed_radii >= statistical_radii``; the
        caller records how the margin itself was derived.
    pilot_replace_one_bound:
        Non-negative conditional replace-one bound for the pilot.
    influence_cap:
        Positive public Euclidean cap per block.  Its blockwise L2 norm is
        the complete-client influence cap ``G``.
    correction_budget:
        Public non-negative radius around the pilot.  Zero returns the pilot.

    Notes
    -----
    This routine certifies bounded influence and conditional stability.  It
    does not claim that the statistical calibration is correct, nor that the
    estimator is universally Byzantine robust; those are separate empirical
    and statistical obligations for the G0e campaign.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    if n < 2:
        raise ValueError("G0e cross-fitted diagnostics require at least two clients")
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)

    work_pilot = torch.as_tensor(pilot, device=device, dtype=work_dtype).reshape(-1)
    if work_pilot.shape != (dimension,) or not bool(torch.isfinite(work_pilot).all()):
        raise ValueError("pilot must be a finite vector of dimension d")

    work_crossfit = torch.as_tensor(
        crossfit_references, device=device, dtype=work_dtype
    )
    if work_crossfit.shape != (n, dimension) or not bool(
        torch.isfinite(work_crossfit).all()
    ):
        raise ValueError("crossfit_references must be a finite matrix of shape (n,d)")

    delta_pilot = float(pilot_replace_one_bound)
    if not math.isfinite(delta_pilot) or delta_pilot < 0.0:
        raise ValueError("pilot_replace_one_bound must be finite and non-negative")
    gamma = float(regularization)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("regularization must be finite and strictly positive")
    budget = float(correction_budget)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("correction_budget must be finite and non-negative")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a public integer >= 1")

    public_statistical_radii = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    public_deployed_radii = _as_block_matrix(
        deployed_radii,
        name="deployed_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    if bool((public_deployed_radii < public_statistical_radii).any()):
        raise ValueError(
            "deployed_radii must be at least statistical_radii elementwise"
        )
    deployment_margins = public_deployed_radii - public_statistical_radii
    caps = _as_block_vector(
        influence_cap,
        name="influence_cap",
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
    )
    effective_radii = torch.minimum(public_deployed_radii, caps[None, :])
    cap_limited_radii = public_deployed_radii > caps[None, :]

    # Cross-fitting is used only for the calibration audit.  In particular,
    # changing a diagnostic LOO centre cannot silently change the estimator.
    crossfit_norms = torch.empty((n, len(blocks)), device=device, dtype=work_dtype)
    for block_index, block_slice in enumerate(slices):
        crossfit_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - work_crossfit[:, block_slice], dim=1
        )
    statistical_tail = crossfit_norms > public_statistical_radii

    pilot_norms = torch.empty_like(crossfit_norms)
    for block_index, block_slice in enumerate(slices):
        pilot_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - work_pilot[None, block_slice], dim=1
        )
    online_statistical_tail = pilot_norms > public_deployed_radii

    eta = 2.0 / (1.0 + 2.0 * gamma)
    contraction = 1.0 / (1.0 + 2.0 * gamma)
    contraction_remainder = 1.0 - contraction**num_steps
    complete_client_cap = float(torch.linalg.vector_norm(caps).item())
    beta = min(
        1.0,
        gamma * budget / (complete_client_cap * contraction_remainder),
    )

    point = work_pilot.clone()
    initial_objective = _huber_objective(
        point,
        work_vectors,
        anchor=work_pilot,
        slices=slices,
        radii=effective_radii,
        regularization=gamma,
    )
    for _ in range(num_steps):
        data_gradient, _ = _huber_gradient(
            point, work_vectors, slices=slices, radii=effective_radii
        )
        gradient = data_gradient + gamma * (point - work_pilot)
        point = point - eta * gradient

    data_gradient, effective_tail = _huber_gradient(
        point, work_vectors, slices=slices, radii=effective_radii
    )
    final_gradient = data_gradient + gamma * (point - work_pilot)
    raw_displacement = point - work_pilot
    scaled_correction = beta * raw_displacement
    result = work_pilot + scaled_correction

    # A complete-client Huber gradient is bounded by G.  Starting from the
    # pilot, the contractive recurrence gives the following finite-K radius.
    geometric_sum = (1.0 - contraction**num_steps) / (1.0 - contraction)
    finite_raw_correction_bound = eta * complete_client_cap * geometric_sum
    exact_raw_correction_bound = complete_client_cap / gamma
    finite_correction_bound = beta * finite_raw_correction_bound
    exact_correction_bound = beta * exact_raw_correction_bound

    direct_finite_term = (
        beta * 2.0 * eta * complete_client_cap * geometric_sum / float(n)
    )
    direct_exact_term = beta * 2.0 * complete_client_cap / (gamma * float(n))
    finite_replace_one_bound = delta_pilot + direct_finite_term
    exact_replace_one_bound = delta_pilot + direct_exact_term
    finite_solver_error_bound = contraction**num_steps * complete_client_cap / gamma

    final_norms = torch.empty_like(crossfit_norms)
    for block_index, block_slice in enumerate(slices):
        final_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - point[None, block_slice], dim=1
        )
    # This is deliberately distinct from ``statistical_tail``.  A cap is
    # active only where it is the limiting radius and the final residual
    # actually crosses it.
    cap_active = cap_limited_radii & (final_norms > caps[None, :])

    if not return_diagnostics:
        return result
    final_objective = _huber_objective(
        point,
        work_vectors,
        anchor=work_pilot,
        slices=slices,
        radii=effective_radii,
        regularization=gamma,
    )
    observed_correction_norm = float(torch.linalg.vector_norm(scaled_correction).item())
    diagnostics: dict[str, Any] = {
        "reference_name": "g0e_crossfit_calibrated_bounded_correction",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "num_steps": num_steps,
        "step_size": eta,
        "solver_contraction": contraction,
        "finite_iteration_fraction": contraction_remainder,
        "regularization": gamma,
        "beta": beta,
        "beta_source": (
            "min(1, regularization*correction_budget/"
            "(client_cap*(1-contraction**num_steps)))"
        ),
        "beta_conservative_asymptotic": min(1.0, gamma * budget / complete_client_cap),
        "correction_budget": budget,
        "complete_client_influence_cap": complete_client_cap,
        "raw_displacement_norm": float(
            torch.linalg.vector_norm(raw_displacement).item()
        ),
        "observed_correction_norm": observed_correction_norm,
        "correction_budget_respected": bool(
            observed_correction_norm <= budget + 32.0 * torch.finfo(work_dtype).eps
        ),
        "finite_raw_correction_norm_bound": finite_raw_correction_bound,
        "exact_raw_correction_norm_bound": exact_raw_correction_bound,
        "finite_correction_norm_bound": finite_correction_bound,
        "exact_correction_norm_bound": exact_correction_bound,
        "gradient_residual_norm": float(
            torch.linalg.vector_norm(final_gradient).item()
        ),
        "finite_solver_error_bound": finite_solver_error_bound,
        "finite_solver_distance_to_exact_bound_before_blend": (
            finite_solver_error_bound
        ),
        "finite_solver_distance_to_exact_bound_after_blend": (
            beta * finite_solver_error_bound
        ),
        "finite_solver_is_the_released_mechanism": True,
        "solver_error_is_not_added_to_finite_stability_bound": True,
        "objective_initial": float(initial_objective.item()),
        "objective_final": float(final_objective.item()),
        "statistical_tail_fraction_crossfit": float(
            statistical_tail.to(work_dtype).mean().item()
        ),
        "statistical_tail_client_blocks": (statistical_tail.detach().cpu().tolist()),
        "statistical_tail_definition": (
            "norm(upload_i-crossfit_reference_i) > frozen_statistical_radius_i"
        ),
        "crossfit_residual_norms_by_client_block": (
            crossfit_norms.detach().cpu().tolist()
        ),
        "online_statistical_tail_fraction_at_pilot": float(
            online_statistical_tail.to(work_dtype).mean().item()
        ),
        "online_statistical_tail_client_blocks": (
            online_statistical_tail.detach().cpu().tolist()
        ),
        "online_statistical_tail_definition": (
            "norm(upload_i-full_pilot) > frozen_deployed_radius_i_before_cap"
        ),
        "pilot_residual_norms_by_client_block": (pilot_norms.detach().cpu().tolist()),
        "cap_limited_radius_fraction": float(
            cap_limited_radii.to(work_dtype).mean().item()
        ),
        "cap_limited_radius_client_blocks": (cap_limited_radii.detach().cpu().tolist()),
        "influence_cap_active_fraction_final": float(
            cap_active.to(work_dtype).mean().item()
        ),
        "influence_cap_active_client_blocks": cap_active.detach().cpu().tolist(),
        "influence_cap_active_clients": (cap_active.any(dim=1).detach().cpu().tolist()),
        "effective_huber_tail_fraction_final": float(
            effective_tail.to(work_dtype).mean().item()
        ),
        "effective_huber_tail_client_blocks": (effective_tail.detach().cpu().tolist()),
        "final_residual_norms_by_client_block": (final_norms.detach().cpu().tolist()),
        "pre_blend_reference": point.detach().cpu().tolist(),
        "scaled_correction_vector": scaled_correction.detach().cpu().tolist(),
        "statistical_radius_min": float(public_statistical_radii.min().item()),
        "statistical_radius_max": float(public_statistical_radii.max().item()),
        "deployed_radius_min_before_cap": float(public_deployed_radii.min().item()),
        "deployed_radius_max_before_cap": float(public_deployed_radii.max().item()),
        "deployment_margin_min": float(deployment_margins.min().item()),
        "deployment_margin_max": float(deployment_margins.max().item()),
        "effective_radius_min": float(effective_radii.min().item()),
        "effective_radius_max": float(effective_radii.max().item()),
        "pilot_replace_one_bound": delta_pilot,
        "pilot_bound_is_a_caller_supplied_certificate": True,
        "pilot_role": "caller_supplied_robust_pilot_intended_fcc",
        "direct_finite_replace_one_term": direct_finite_term,
        "direct_exact_replace_one_term": direct_exact_term,
        "direct_finite_term_per_replaced_client": direct_finite_term,
        "direct_exact_term_per_replaced_client": direct_exact_term,
        "finite_solver_replace_one_bound": finite_replace_one_bound,
        "exact_minimizer_replace_one_bound": exact_replace_one_bound,
        "certificate_adjacency": "replace_one_fixed_client_identity",
        "replace_one_certificate_is_not_a_b_client_robust_error_bound": True,
        "b_client_bound_requires_a_separate_pilot_b_replacement_bound": True,
        "correction_budget_only_bounds_displacement_from_pilot": True,
        "correction_budget_does_not_bound_total_robust_error": True,
        "certificate_includes_data_dependent_pilot": True,
        "certificate_requires_fixed_public_statistical_radii": True,
        "certificate_requires_fixed_public_deployed_radii": True,
        "certificate_requires_fixed_public_caps": True,
        "certificate_requires_same_public_solver": True,
        "crossfit_references_affect_returned_reference": False,
        "crossfit_diagnostics_require_separate_release_accounting": True,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(result.dtype),
        "certificate_arithmetic": "exact_real_bound_reported_float_runtime",
    }
    return result, diagnostics


_G0F_ALLOCATION_POLICIES = frozenset(
    {"global_covariance", "per_client_scale_blind", "equal_cap"}
)


@torch.no_grad()
def allocate_gaussian_aware_block_budgets(
    public_radii: torch.Tensor,
    *,
    total_influence_budget: float,
    policy: str = "global_covariance",
) -> tuple[torch.Tensor, dict[str, Any]]:
    r"""Allocate a certified complete-client influence budget across blocks.

    Parameters
    ----------
    public_radii:
        Positive authenticated radii ``a`` with shape ``(n,B)``.  These must
        be fixed before observing the evaluated client uploads.
    total_influence_budget:
        Public common upper bound ``G > 0`` on each complete-client influence.
        Under global allocation, clients below the maximum public radius may
        realize a strictly smaller norm; this is not an equal realized budget.
    policy:
        ``"global_covariance"`` implements the G0f candidate
        ``G*a/max_j||a_j||``.  ``"per_client_scale_blind"`` normalizes each
        row independently and is an ablation that removes between-client
        scale.  ``"equal_cap"`` assigns ``G/sqrt(B)`` to every block.

    Returns
    -------
    budgets, diagnostics:
        The ``(n,B)`` block budgets and JSON-serializable certificate data.

    Notes
    -----
    The construction does not estimate a covariance from ``vectors`` and
    does not accept client-declared scales.  Its certificate is conditional
    on the supplied table being a public/authenticated mechanism parameter.
    """

    if not isinstance(public_radii, torch.Tensor):
        raise TypeError("public_radii must be a torch.Tensor")
    if public_radii.ndim != 2 or min(public_radii.shape) < 1:
        raise ValueError("public_radii must have shape (n,B) with n,B >= 1")
    if not public_radii.is_floating_point():
        raise TypeError("public_radii must use a floating-point dtype")
    work_dtype = torch.float64 if public_radii.dtype == torch.float64 else torch.float32
    radii = public_radii.to(dtype=work_dtype)
    if not bool(torch.isfinite(radii).all()) or bool((radii <= 0.0).any()):
        raise ValueError("public_radii must be finite and strictly positive")
    budget = float(total_influence_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("total_influence_budget must be finite and positive")
    if policy not in _G0F_ALLOCATION_POLICIES:
        allowed = ", ".join(sorted(_G0F_ALLOCATION_POLICIES))
        raise ValueError(f"Unknown allocation policy {policy!r}; expected {allowed}")

    row_norms = torch.linalg.vector_norm(radii, dim=1)
    global_radius = row_norms.max()
    # Reserve a tiny inward floating-point margin.  The real-valued public
    # budget G remains the certificate bound, while every runtime row norm is
    # strictly below it even after float32 norm rounding.
    runtime_target = budget * (1.0 - 8.0 * torch.finfo(work_dtype).eps)
    if policy == "global_covariance":
        budgets = runtime_target * radii / global_radius
        denominator: float | list[float] = float(global_radius.item())
        preserves_between_client_scale = True
        preserves_within_client_block_shape = True
    elif policy == "per_client_scale_blind":
        budgets = runtime_target * radii / row_norms[:, None]
        denominator = [float(value) for value in row_norms.cpu().tolist()]
        preserves_between_client_scale = False
        preserves_within_client_block_shape = True
    else:
        block_budget = runtime_target / math.sqrt(float(radii.shape[1]))
        budgets = torch.full_like(radii, block_budget)
        denominator = math.sqrt(float(radii.shape[1]))
        preserves_between_client_scale = False
        preserves_within_client_block_shape = False

    allocated_norms = torch.linalg.vector_norm(budgets, dim=1)
    certificate_respected = bool((allocated_norms <= budget).all())
    if not certificate_respected:
        raise RuntimeError("Allocated block budgets violate the public L2 budget")
    ratios = budgets / radii
    return budgets, {
        "allocation_policy": policy,
        "total_influence_budget": budget,
        "runtime_inward_target": runtime_target,
        "runtime_inward_rounding_margin": budget - runtime_target,
        "public_radius_row_norm_min": float(row_norms.min().item()),
        "public_radius_row_norm_max_A": float(global_radius.item()),
        "nominal_global_allocation_scale_G_over_A": (
            budget / float(global_radius.item())
        ),
        "runtime_global_allocation_scale_G_runtime_over_A": (
            runtime_target / float(global_radius.item())
        ),
        "ideal_global_effective_fraction_min_1_G_over_A": min(
            1.0, budget / float(global_radius.item())
        ),
        "runtime_global_effective_fraction_min_1_G_runtime_over_A": min(
            1.0, runtime_target / float(global_radius.item())
        ),
        "allocation_denominator": denominator,
        "allocated_client_norm_min": float(allocated_norms.min().item()),
        "allocated_client_norm_max": float(allocated_norms.max().item()),
        "allocated_client_norms": [
            float(value) for value in allocated_norms.cpu().tolist()
        ],
        "allocated_block_budgets": budgets.cpu().tolist(),
        "budget_to_radius_ratio_min": float(ratios.min().item()),
        "budget_to_radius_ratio_max": float(ratios.max().item()),
        "complete_client_budget_respected": certificate_respected,
        "preserves_between_client_covariance_scale": (preserves_between_client_scale),
        "preserves_within_client_block_shape": preserves_within_client_block_shape,
        "allocation_source": "caller_supplied_public_allocation_radii",
        "per_client_policy_is_dp_scale_blind_only_under_separable_radii": True,
        "public_covariance_required": True,
        "client_declared_covariance_forbidden": True,
        "certificate": "max_i_l2_norm_of_allocated_block_budget_leq_G",
    }


@torch.no_grad()
def gaussian_aware_budget_allocated_correction(
    vectors: torch.Tensor,
    *,
    pilot: torch.Tensor,
    crossfit_references: torch.Tensor,
    statistical_radii: TensorOrScalar,
    deployed_radii: TensorOrScalar,
    pilot_replace_one_bound: float,
    total_influence_budget: float,
    block_sizes: Sequence[int] | None = None,
    allocation_radii: TensorOrScalar | None = None,
    allocation_policy: str = "global_covariance",
    allocation_radius_provenance: str = "caller_supplied_public_allocation_radii",
    covariance_provenance: str = "public_authenticated",
    regularization: float = 0.2,
    correction_budget: float = 0.1,
    num_steps: int = 20,
    num_replacements_for_diagnostics: int = 1,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the G0f covariance-budgeted correction around a robust pilot.

    The candidate differs from G0e only in the *allocation* of its total
    influence budget.  Let ``a[i,b]`` be the fixed public radius supplied by
    ``allocation_radii`` (or ``deployed_radii`` by default).  Under the main
    policy,

    ``A=max_j ||a[j]||_2`` and ``G[i,b]=G*a[i,b]/A``.

    Thus ``||G[i]||_2 <= G`` for every client.  ``G[i,b]`` is the allocated
    influence radius; the Huber transition actually used by the solver is
    ``R[i,b]=min(deployed_radii[i,b], G[i,b])``.  This retains
    the covariance geometry even when every radius is cap-limited, unlike a
    common scalar block cap.

    ``allocation_radius_provenance`` records (but does not estimate) how the
    caller constructed the authenticated radii.  The G0f experiment uses
    ``"effective_null_radius"`` for its fixed sum of DP variance,
    heterogeneity, cross-fitted reference variance and numerical floor.

    ``covariance_provenance`` is deliberately restrictive.  Privacy-mechanism
    covariance must be authenticated/public (for example, derived from the
    public clipping norm, batch rule, and noise multiplier), never declared
    by a potentially Byzantine client.

    Conditional on fixed public radii, solver parameters and client identity,
    replacing one upload changes the averaged clipped data gradient by at
    most ``2G/n``.  With ``q=1/(1+2*gamma)`` and
    ``eta=2/(1+2*gamma)``, the released finite-iteration reference satisfies

    ``Delta <= Delta_pilot + beta*(2*eta*G/n)*(1-q**K)/(1-q)``.

    The blend ``beta`` is derived from ``correction_budget`` exactly as in
    G0e, so the displacement from the pilot is also certified.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    if n < 2:
        raise ValueError("G0f cross-fitted diagnostics require at least two clients")
    if covariance_provenance != "public_authenticated":
        raise ValueError(
            "covariance_provenance must be 'public_authenticated'; "
            "client-declared covariance is forbidden"
        )
    allowed_radius_provenance = {
        "caller_supplied_public_allocation_radii",
        "effective_null_radius",
    }
    if allocation_radius_provenance not in allowed_radius_provenance:
        raise ValueError(
            "allocation_radius_provenance must be one of "
            f"{sorted(allowed_radius_provenance)}"
        )
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)

    work_pilot = torch.as_tensor(pilot, device=device, dtype=work_dtype).reshape(-1)
    if work_pilot.shape != (dimension,) or not bool(torch.isfinite(work_pilot).all()):
        raise ValueError("pilot must be a finite vector of dimension d")
    work_crossfit = torch.as_tensor(
        crossfit_references, device=device, dtype=work_dtype
    )
    if work_crossfit.shape != (n, dimension) or not bool(
        torch.isfinite(work_crossfit).all()
    ):
        raise ValueError("crossfit_references must be a finite matrix of shape (n,d)")

    delta_pilot = float(pilot_replace_one_bound)
    if not math.isfinite(delta_pilot) or delta_pilot < 0.0:
        raise ValueError("pilot_replace_one_bound must be finite and non-negative")
    gamma = float(regularization)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("regularization must be finite and strictly positive")
    correction_radius = float(correction_budget)
    if not math.isfinite(correction_radius) or correction_radius < 0.0:
        raise ValueError("correction_budget must be finite and non-negative")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
        raise ValueError("num_steps must be a public integer >= 1")
    if (
        not isinstance(num_replacements_for_diagnostics, int)
        or isinstance(num_replacements_for_diagnostics, bool)
        or not 1 <= num_replacements_for_diagnostics <= n
    ):
        raise ValueError("num_replacements_for_diagnostics must lie in [1,n]")

    statistical = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    deployed = _as_block_matrix(
        deployed_radii,
        name="deployed_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    if bool((deployed < statistical).any()):
        raise ValueError(
            "deployed_radii must be at least statistical_radii elementwise"
        )
    allocation = _as_block_matrix(
        deployed if allocation_radii is None else allocation_radii,
        name="allocation_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    block_budgets, allocation_diagnostics = allocate_gaussian_aware_block_budgets(
        allocation,
        total_influence_budget=total_influence_budget,
        policy=allocation_policy,
    )
    allocation_diagnostics = dict(allocation_diagnostics)
    allocation_diagnostics["allocation_source"] = allocation_radius_provenance
    allocation_diagnostics["effective_null_radius_components"] = (
        "dp_variance_plus_heterogeneity_plus_reference_variance_plus_floor"
        if allocation_radius_provenance == "effective_null_radius"
        else None
    )
    complete_client_cap = float(total_influence_budget)
    effective_radii = torch.minimum(deployed, block_budgets)
    cap_limited = deployed > block_budgets
    allocation_matches_deployed = bool(torch.equal(allocation, deployed))
    effective_to_deployed = effective_radii / deployed
    effective_to_deployed_min = float(effective_to_deployed.min().item())
    effective_to_deployed_max = float(effective_to_deployed.max().item())
    effective_to_deployed_spread = effective_to_deployed_max - effective_to_deployed_min
    global_effective_fraction_theorem_applicable = bool(
        allocation_policy == "global_covariance" and allocation_matches_deployed
    )
    if global_effective_fraction_theorem_applicable:
        global_radius = float(torch.linalg.vector_norm(allocation, dim=1).max().item())
        ideal_effective_fraction: float | None = min(
            1.0, complete_client_cap / global_radius
        )
    else:
        ideal_effective_fraction = None

    crossfit_norms = torch.empty((n, len(blocks)), device=device, dtype=work_dtype)
    pilot_norms = torch.empty_like(crossfit_norms)
    for block_index, block_slice in enumerate(slices):
        crossfit_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - work_crossfit[:, block_slice], dim=1
        )
        pilot_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - work_pilot[None, block_slice], dim=1
        )
    statistical_tail = crossfit_norms > statistical
    online_statistical_tail = pilot_norms > deployed

    eta = 2.0 / (1.0 + 2.0 * gamma)
    contraction = 1.0 / (1.0 + 2.0 * gamma)
    if not 0.0 < contraction < 1.0 or not 1.0 - contraction > 0.0:
        raise ValueError(
            "regularization is not numerically resolvable in the public solver"
        )
    # -expm1(K*log(q)) is accurate when q is close to one.
    finite_fraction = -math.expm1(num_steps * math.log(contraction))
    if not math.isfinite(finite_fraction) or finite_fraction <= 0.0:
        raise ValueError("finite solver fraction is not numerically positive")
    if correction_radius == 0.0:
        beta = 0.0
    else:
        beta = min(
            1.0,
            gamma * correction_radius / (complete_client_cap * finite_fraction),
        )

    point = work_pilot.clone()
    initial_objective = _huber_objective(
        point,
        work_vectors,
        anchor=work_pilot,
        slices=slices,
        radii=effective_radii,
        regularization=gamma,
    )
    for _ in range(num_steps):
        data_gradient, _ = _huber_gradient(
            point, work_vectors, slices=slices, radii=effective_radii
        )
        point = point - eta * (data_gradient + gamma * (point - work_pilot))

    data_gradient, effective_tail = _huber_gradient(
        point, work_vectors, slices=slices, radii=effective_radii
    )
    final_gradient = data_gradient + gamma * (point - work_pilot)
    raw_displacement = point - work_pilot
    scaled_correction = beta * raw_displacement
    result = work_pilot + scaled_correction

    geometric_sum = finite_fraction / (1.0 - contraction)
    finite_raw_bound = eta * complete_client_cap * geometric_sum
    exact_raw_bound = complete_client_cap / gamma
    direct_finite = beta * 2.0 * eta * complete_client_cap * geometric_sum / float(n)
    direct_exact = beta * 2.0 * complete_client_cap / (gamma * float(n))
    finite_replace_one_bound = delta_pilot + direct_finite
    exact_replace_one_bound = delta_pilot + direct_exact
    solver_error_bound = contraction**num_steps * complete_client_cap / gamma
    solver_error_bound_after_blend = beta * solver_error_bound
    b_replacements = int(num_replacements_for_diagnostics)
    finite_b_replacement_bound = b_replacements * finite_replace_one_bound
    exact_b_replacement_bound = b_replacements * exact_replace_one_bound

    final_norms = torch.empty_like(crossfit_norms)
    for block_index, block_slice in enumerate(slices):
        final_norms[:, block_index] = torch.linalg.vector_norm(
            work_vectors[:, block_slice] - point[None, block_slice], dim=1
        )
    cap_active = cap_limited & (final_norms > block_budgets)

    if not return_diagnostics:
        return result
    final_objective = _huber_objective(
        point,
        work_vectors,
        anchor=work_pilot,
        slices=slices,
        radii=effective_radii,
        regularization=gamma,
    )
    observed_correction = float(torch.linalg.vector_norm(scaled_correction).item())
    correction_tolerance = (
        32.0 * torch.finfo(work_dtype).eps * max(1.0, correction_radius)
    )
    diagnostics: dict[str, Any] = {
        "reference_name": "g0f_gaussian_aware_budget_allocated_correction",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "allocation_policy": allocation_policy,
        "allocation_source": allocation_radius_provenance,
        "allocation_uses_deployed_public_pre_cap_radii": (allocation_matches_deployed),
        "allocation_numerically_equals_deployed_radii": allocation_matches_deployed,
        "effective_null_radius_components": (
            "dp_variance_plus_heterogeneity_plus_reference_variance_plus_floor"
            if allocation_radius_provenance == "effective_null_radius"
            else None
        ),
        "covariance_provenance": covariance_provenance,
        "covariance_is_public_authenticated_not_client_declared": True,
        "num_steps": num_steps,
        "step_size": eta,
        "solver_contraction": contraction,
        "finite_iteration_fraction": finite_fraction,
        "regularization": gamma,
        "beta": beta,
        "correction_budget": correction_radius,
        "complete_client_influence_cap": complete_client_cap,
        "complete_client_influence_cap_semantics": (
            "common_upper_bound_not_identical_realized_client_budget"
        ),
        "allocated_block_budgets": block_budgets.cpu().tolist(),
        "effective_solver_radii": effective_radii.cpu().tolist(),
        "effective_solver_radius_definition": (
            "min(deployed_public_radius,allocated_block_budget)"
        ),
        "effective_radius_to_deployed_ratio_min": effective_to_deployed_min,
        "effective_radius_to_deployed_ratio_max": effective_to_deployed_max,
        "effective_radius_to_deployed_ratio_spread": effective_to_deployed_spread,
        "global_effective_fraction_theorem_applicable": (
            global_effective_fraction_theorem_applicable
        ),
        "ideal_global_effective_fraction_min_1_G_over_A": (ideal_effective_fraction),
        "allocated_client_norms": allocation_diagnostics["allocated_client_norms"],
        "allocated_client_norm_max": allocation_diagnostics[
            "allocated_client_norm_max"
        ],
        "complete_client_budget_respected": allocation_diagnostics[
            "complete_client_budget_respected"
        ],
        "allocation_diagnostics": allocation_diagnostics,
        "raw_displacement_norm": float(
            torch.linalg.vector_norm(raw_displacement).item()
        ),
        "observed_correction_norm": observed_correction,
        "correction_budget_respected": bool(
            observed_correction <= correction_radius + correction_tolerance
        ),
        "finite_raw_correction_norm_bound": finite_raw_bound,
        "exact_raw_correction_norm_bound": exact_raw_bound,
        "finite_correction_norm_bound": beta * finite_raw_bound,
        "exact_correction_norm_bound": beta * exact_raw_bound,
        "gradient_residual_norm": float(
            torch.linalg.vector_norm(final_gradient).item()
        ),
        "finite_solver_error_bound": solver_error_bound,
        "finite_solver_error_bound_after_blend": solver_error_bound_after_blend,
        "objective_initial": float(initial_objective.item()),
        "objective_final": float(final_objective.item()),
        "statistical_tail_fraction_crossfit": float(
            statistical_tail.to(work_dtype).mean().item()
        ),
        "statistical_tail_client_blocks": statistical_tail.cpu().tolist(),
        "online_statistical_tail_fraction_at_pilot": float(
            online_statistical_tail.to(work_dtype).mean().item()
        ),
        "online_statistical_tail_client_blocks": (
            online_statistical_tail.cpu().tolist()
        ),
        "crossfit_residual_norms_by_client_block": crossfit_norms.cpu().tolist(),
        "pilot_residual_norms_by_client_block": pilot_norms.cpu().tolist(),
        "cap_limited_radius_fraction": float(cap_limited.to(work_dtype).mean().item()),
        "cap_limited_radius_client_blocks": cap_limited.cpu().tolist(),
        "influence_cap_active_fraction_final": float(
            cap_active.to(work_dtype).mean().item()
        ),
        "influence_cap_active_client_blocks": cap_active.cpu().tolist(),
        "influence_cap_active_clients": cap_active.any(dim=1).cpu().tolist(),
        "effective_huber_tail_fraction_final": float(
            effective_tail.to(work_dtype).mean().item()
        ),
        "effective_huber_tail_client_blocks": effective_tail.cpu().tolist(),
        "final_residual_norms_by_client_block": final_norms.cpu().tolist(),
        "pre_blend_reference": point.cpu().tolist(),
        "scaled_correction_vector": scaled_correction.cpu().tolist(),
        "statistical_radius_min": float(statistical.min().item()),
        "statistical_radius_max": float(statistical.max().item()),
        "deployed_radius_min_before_cap": float(deployed.min().item()),
        "deployed_radius_max_before_cap": float(deployed.max().item()),
        "allocation_radius_min": float(allocation.min().item()),
        "allocation_radius_max": float(allocation.max().item()),
        "effective_radius_min": float(effective_radii.min().item()),
        "effective_radius_max": float(effective_radii.max().item()),
        "pilot_replace_one_bound": delta_pilot,
        "pilot_bound_is_a_caller_supplied_certificate": True,
        "pilot_role": "caller_supplied_robust_pilot_intended_fcc",
        "direct_finite_replace_one_term": direct_finite,
        "direct_exact_replace_one_term": direct_exact,
        "finite_solver_replace_one_bound": finite_replace_one_bound,
        "exact_minimizer_replace_one_bound": exact_replace_one_bound,
        "num_replacements_for_diagnostics": b_replacements,
        "finite_b_replacement_bound_by_replacement_path": (finite_b_replacement_bound),
        "exact_b_replacement_bound_by_replacement_path": (exact_b_replacement_bound),
        "b_replacement_path_bound_formula": (
            "b*(Delta_pilot+2*beta*G*(1-q**K)/(gamma*n))"
        ),
        "b_replacement_path_requires_uniform_pilot_one_replacement_bound": True,
        "certificate_adjacency": "replace_one_fixed_client_identity",
        "certificate_requires_fixed_public_allocation_radii": True,
        "certificate_requires_authenticated_covariance": True,
        "certificate_requires_same_public_solver": True,
        "certificate_includes_data_dependent_pilot": True,
        "crossfit_references_affect_returned_reference": False,
        "crossfit_diagnostics_require_separate_release_accounting": False,
        "crossfit_privacy_status": (
            "server_postprocessing_of_already_locally_private_uploads"
        ),
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(result.dtype),
        "certificate_arithmetic": "exact_real_bound_reported_float_runtime",
    }
    return result, diagnostics


@torch.no_grad()
def standardized_quadratic_scores(
    vectors: torch.Tensor,
    *,
    references: torch.Tensor,
    noise_variances: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    heterogeneity_variances: TensorOrScalar | None = None,
    reference_variances: TensorOrScalar | None = None,
    scoring_variances: TensorOrScalar | None = None,
    variance_floor: float = 1e-12,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return pivotal quadratic residual scores and their exact null moments.

    For block-isotropic, independent null residuals with variance ``V`` and a
    quadratic form using scalar block variance ``S``, this computes

    ``Q_i = sum_b ||R_i,b||^2 / S_i,b``,
    ``m_i = sum_b d_b V_i,b / S_i,b``, and
    ``v_i = 2 sum_b d_b (V_i,b / S_i,b)^2``.

    The returned score is ``(Q_i-m_i)/sqrt(v_i)``.  When ``S=V``, its null
    mean is zero and null variance is one irrespective of each client's DP
    noise scale.  These moments do not include a non-zero honest shift or
    cross-coordinate/block dependence.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    work_vectors = vectors.to(dtype=work_dtype)
    work_references = torch.as_tensor(
        references, device=vectors.device, dtype=work_dtype
    )
    reference_was_leave_one_out = work_references.ndim == 2
    if work_references.shape == (dimension,):
        work_references = work_references[None, :].expand(n, dimension)
    if work_references.shape != (n, dimension):
        raise ValueError("references must have shape (d,) or (n,d)")
    if not bool(torch.isfinite(work_references).all()):
        raise ValueError("references must be finite")

    null_variances = _effective_variances(
        noise_variances=noise_variances,
        heterogeneity_variances=heterogeneity_variances,
        reference_variances=reference_variances,
        variance_floor=variance_floor,
        n=n,
        num_blocks=len(blocks),
        device=vectors.device,
        dtype=work_dtype,
    )
    if scoring_variances is None:
        score_variances = null_variances
    else:
        score_variances = _as_block_matrix(
            scoring_variances,
            name="scoring_variances",
            n=n,
            num_blocks=len(blocks),
            device=vectors.device,
            dtype=work_dtype,
            default=0.0,
            strictly_positive=True,
        )

    residuals = work_vectors - work_references
    quadratic = torch.zeros(n, device=vectors.device, dtype=work_dtype)
    null_mean = torch.zeros_like(quadratic)
    null_variance = torch.zeros_like(quadratic)
    for block_index, block_slice in enumerate(slices):
        dimension_b = float(blocks[block_index])
        score_variance = score_variances[:, block_index]
        ratio = null_variances[:, block_index] / score_variance
        quadratic = quadratic + residuals[:, block_slice].square().sum(dim=1) / (
            score_variance
        )
        null_mean = null_mean + dimension_b * ratio
        null_variance = null_variance + 2.0 * dimension_b * ratio.square()

    scores = (quadratic - null_mean) / null_variance.sqrt()
    # As for the reference, retain the working dtype so a half-precision cast
    # cannot destroy null calibration near zero.
    result = scores
    if not return_diagnostics:
        return result
    return result, {
        "quadratic_energy": quadratic,
        "null_mean": null_mean,
        "null_variance": null_variance,
        "score_mean": float(scores.mean().item()),
        "score_std_population": float(scores.std(unbiased=False).item()),
        "moment_model": "block_isotropic_independent_zero_mean_residual",
        "per_client_references_supplied": reference_was_leave_one_out,
        "reference_uncertainty_included": reference_variances is not None,
    }


@torch.no_grad()
def gaussian_aware_fixed_anchor_gated_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    statistical_radii: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    influence_cap: TensorOrScalar = 1.0,
    gate_transition_width: float = 1.0,
    num_replacements_for_diagnostics: int = 1,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the G0g-K1 fixed-anchor, noise-tolerant reference.

    This construction deliberately separates *statistical tolerance* from
    *authorised influence*.  For client ``i`` and public block ``b`` it forms

    .. math::

        z_{i,b}=\frac{\|x_{i,b}-a_b\|_2}{r_{i,b}},

    where ``r[i,b]`` is a fixed public radius derived from authenticated DP
    covariance and pre-registered nuisance variances.  The soft acceptance
    gate is

    .. math::

        g_h(z)=\begin{cases}
          1, & z\le 1,\\
          1-(z-1)/h, & 1<z<1+h,\\
          0, & z\ge 1+h.
        \end{cases}

    The returned reference is the single public update

    .. math::

        F(U)=a+\frac1n\sum_i \Psi_i,
        \qquad
        \Psi_{i,b}=g_h(z_{i,b})
        \operatorname{Clip}_{G_b}(x_{i,b}-a_b).

    ``G_b`` is common to every client and cannot depend on ``r[i,b]`` or a
    client-declared covariance.  There is no normalisation by ``sum_i g_i``.
    Consequently, with ``G=sqrt(sum_b G_b**2)``, every complete-client
    contribution has norm at most ``G`` for every covariance table.

    Conditional on the same public anchor, radii, caps, block structure and
    client identities on neighbouring cohorts, replace-one sensitivity is
    exactly bounded by ``2*G/n``.  This is a server-side post-processing
    certificate; it is not needed to obtain local DP, which is already carried
    by the input uploads.  A current, data-dependent anchor requires a separate
    indirect-effect analysis and is intentionally outside this primitive.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)

    work_anchor = torch.as_tensor(
        anchor, device=device, dtype=work_dtype
    ).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(
        torch.isfinite(work_anchor).all()
    ):
        raise ValueError("anchor must be a finite vector of dimension d")

    radii = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    caps = _as_block_vector(
        influence_cap,
        name="influence_cap",
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
    )
    width = float(gate_transition_width)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("gate_transition_width must be finite and positive")
    if (
        not isinstance(num_replacements_for_diagnostics, int)
        or isinstance(num_replacements_for_diagnostics, bool)
        or not 1 <= num_replacements_for_diagnostics <= n
    ):
        raise ValueError("num_replacements_for_diagnostics must lie in [1,n]")

    normalized = torch.empty((n, len(blocks)), device=device, dtype=work_dtype)
    gates = torch.empty_like(normalized)
    cap_active = torch.zeros(
        (n, len(blocks)), device=device, dtype=torch.bool
    )
    contributions = torch.zeros_like(work_vectors)
    for block_index, block_slice in enumerate(slices):
        residuals = work_vectors[:, block_slice] - work_anchor[None, block_slice]
        norms = torch.linalg.vector_norm(residuals, dim=1)
        normalized[:, block_index] = norms / radii[:, block_index]
        gates[:, block_index] = (
            1.0
            - (normalized[:, block_index] - 1.0) / width
        ).clamp(min=0.0, max=1.0)
        block_caps = caps[block_index].expand(n)
        clipped = _clip_rows_at_radii(residuals, block_caps)
        contributions[:, block_slice] = (
            gates[:, block_index, None] * clipped
        )
        cap_active[:, block_index] = norms > caps[block_index]

    reference = work_anchor + contributions.mean(dim=0)
    complete_cap = float(torch.linalg.vector_norm(caps).item())
    client_norms = torch.linalg.vector_norm(contributions, dim=1)
    displacement = float(torch.linalg.vector_norm(reference - work_anchor).item())
    replacement_bound = 2.0 * complete_cap / float(n)
    b_replacement_bound = (
        2.0
        * float(num_replacements_for_diagnostics)
        * complete_cap
        / float(n)
    )
    tolerance = 64.0 * torch.finfo(work_dtype).eps * max(1.0, complete_cap)
    certificate_respected = bool((client_norms <= complete_cap + tolerance).all())
    if not certificate_respected:
        raise RuntimeError("G0g client contribution exceeded the public cap")

    if not return_diagnostics:
        return reference
    return reference, {
        "reference_name": "g0g_k1_fixed_anchor_noise_tolerance_fixed_influence",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "anchor_role": "fixed_public_or_prior_transcript_anchor",
        "anchor_is_required_fixed_on_neighbouring_cohorts": True,
        "statistical_radii_provenance_required": "public_authenticated",
        "client_declared_covariance_forbidden": True,
        "statistical_radii_affect_only_gate": True,
        "influence_caps_are_client_independent": True,
        "gate_transition_width": width,
        "gate_lipschitz_constant": 1.0 / width,
        "gate_definition": "1_if_z_le_1_linear_to_0_on_1_to_1_plus_h",
        "normalization_by_gate_sum": False,
        "normalized_residuals_by_client_block": normalized.cpu().tolist(),
        "gates_by_client_block": gates.cpu().tolist(),
        "gate_mean": float(gates.mean().item()),
        "gate_mean_by_block": [
            float(value) for value in gates.mean(dim=0).cpu().tolist()
        ],
        "statistical_tail_fraction": float((normalized > 1.0).float().mean().item()),
        "hard_reject_fraction": float(
            (normalized >= 1.0 + width).float().mean().item()
        ),
        "influence_caps_per_block": [float(value) for value in caps.cpu().tolist()],
        "complete_client_influence_cap": complete_cap,
        "client_contribution_norms": [
            float(value) for value in client_norms.cpu().tolist()
        ],
        "client_contribution_cap_respected": certificate_respected,
        "certificate_float_tolerance": tolerance,
        "cap_active_fraction": float(cap_active.float().mean().item()),
        "cap_active_by_client_block": cap_active.cpu().tolist(),
        "observed_reference_displacement": displacement,
        "reference_displacement_bound": complete_cap,
        "replace_one_bound": replacement_bound,
        "replace_one_bound_formula": "2*G/n_conditional_on_fixed_public_inputs",
        "num_replacements_for_diagnostics": num_replacements_for_diagnostics,
        "b_replacement_bound": b_replacement_bound,
        "local_dp_cost_added_by_reference": 0.0,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(reference.dtype),
    }


@torch.no_grad()
def gaussian_aware_fixed_anchor_scalar_gated_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    statistical_radii: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    influence_cap: float = 1.0,
    gate_transition_width: float = 1.0,
    num_replacements_for_diagnostics: int = 1,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the G0g-K2 scalar-gated, globally clipped reference.

    For public blocks ``b`` and fixed public radii ``r[i,b]``, define

    .. math::

        z_i = \left(\frac1B\sum_b
        \frac{\|x_{i,b}-a_b\|_2^2}{r_{i,b}^2}\right)^{1/2}.

    The same ramp ``g_h`` as G0g-K1 is applied once per client and

    .. math::

        F(U)=a+\frac1n\sum_i g_h(z_i)
        \operatorname{Clip}_{G}(x_i-a).

    Hence every complete-client contribution has L2 norm at most the single
    public cap ``G``.  Conditional on identical public inputs on neighbouring
    cohorts, replace-one sensitivity is at most ``2G/n`` and replacement of
    ``b`` messages changes the result by at most ``2bG/n``.  Importantly,
    setting every gate to one recovers global centered clipping exactly; no
    blockwise cap changes its geometry.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)
    work_anchor = torch.as_tensor(anchor, device=device, dtype=work_dtype).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")
    radii = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    cap = float(influence_cap)
    width = float(gate_transition_width)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("gate_transition_width must be finite and positive")
    if (
        not isinstance(num_replacements_for_diagnostics, int)
        or isinstance(num_replacements_for_diagnostics, bool)
        or not 1 <= num_replacements_for_diagnostics <= n
    ):
        raise ValueError("num_replacements_for_diagnostics must lie in [1,n]")

    residuals = work_vectors - work_anchor[None, :]
    normalized_blocks = torch.empty(
        (n, len(blocks)), device=device, dtype=work_dtype
    )
    for block_index, block_slice in enumerate(slices):
        normalized_blocks[:, block_index] = (
            torch.linalg.vector_norm(residuals[:, block_slice], dim=1)
            / radii[:, block_index]
        )
    normalized_clients = normalized_blocks.square().mean(dim=1).sqrt()
    gates = (1.0 - (normalized_clients - 1.0) / width).clamp(0.0, 1.0)
    clipped = _clip_rows_at_radii(
        residuals,
        torch.full((n,), cap, device=device, dtype=work_dtype),
    )
    contributions = gates[:, None] * clipped
    reference = work_anchor + contributions.mean(dim=0)

    contribution_norms = torch.linalg.vector_norm(contributions, dim=1)
    replacement_bound = 2.0 * cap / float(n)
    b_replacement_bound = (
        2.0 * float(num_replacements_for_diagnostics) * cap / float(n)
    )
    tolerance = 64.0 * torch.finfo(work_dtype).eps * max(1.0, cap)
    certificate_respected = bool((contribution_norms <= cap + tolerance).all())
    if not certificate_respected:
        raise RuntimeError("G0g-K2 client contribution exceeded the public cap")
    if not return_diagnostics:
        return reference
    return reference, {
        "reference_name": "g0g_k2_fixed_anchor_scalar_gate_global_l2_cap",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "anchor_role": "fixed_public_or_prior_transcript_anchor",
        "anchor_is_required_fixed_on_neighbouring_cohorts": True,
        "statistical_radii_provenance_required": "public_authenticated",
        "client_declared_covariance_forbidden": True,
        "statistical_radii_affect_only_gate": True,
        "influence_cap_is_global_and_client_independent": True,
        "gate_transition_width": width,
        "gate_lipschitz_constant": 1.0 / width,
        "gate_definition": "scalar_rms_block_score_with_linear_ramp",
        "normalization_by_gate_sum": False,
        "normalized_residuals_by_client_block": normalized_blocks.cpu().tolist(),
        "normalized_residuals_by_client": normalized_clients.cpu().tolist(),
        "gates_by_client": gates.cpu().tolist(),
        "gate_mean": float(gates.mean().item()),
        "statistical_tail_fraction": float(
            (normalized_clients > 1.0).float().mean().item()
        ),
        "hard_reject_fraction": float(
            (normalized_clients >= 1.0 + width).float().mean().item()
        ),
        "complete_client_influence_cap": cap,
        "client_contribution_norms": contribution_norms.cpu().tolist(),
        "client_contribution_cap_respected": certificate_respected,
        "certificate_float_tolerance": tolerance,
        "global_cap_active_fraction": float(
            (torch.linalg.vector_norm(residuals, dim=1) > cap).float().mean().item()
        ),
        "observed_reference_displacement": float(
            torch.linalg.vector_norm(reference - work_anchor).item()
        ),
        "reference_displacement_bound": cap,
        "replace_one_bound": replacement_bound,
        "replace_one_bound_formula": "2*G/n_conditional_on_fixed_public_inputs",
        "num_replacements_for_diagnostics": num_replacements_for_diagnostics,
        "b_replacement_bound": b_replacement_bound,
        "no_gate_reduces_exactly_to_global_centered_clipping": True,
        "local_dp_cost_added_by_reference": 0.0,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(reference.dtype),
    }


@torch.no_grad()
def gaussian_aware_fixed_anchor_dual_gated_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    statistical_radii: TensorOrScalar,
    common_statistical_radii: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    influence_cap: float = 1.0,
    gate_transition_width: float = 1.0,
    num_replacements_for_diagnostics: int = 1,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return the G0g-K3 dual-gated, globally clipped reference.

    K3 evaluates every residual against two fixed public geometries.  With
    ``B`` blocks, define

    .. math::

        z_i^{\rm aware}
        = \left(\frac1B\sum_b
          \frac{\|x_{i,b}-a_b\|_2^2}{r_{i,b}^2}\right)^{1/2},
        \qquad
        z_i^{\rm common}
        = \left(\frac1B\sum_b
          \frac{\|x_{i,b}-a_b\|_2^2}{\bar r_b^2}\right)^{1/2}.

    ``r[i,b]`` are authenticated covariance-aware radii, whereas
    ``common_statistical_radii`` are identity-blind, sigma-blind radii.  The
    same scalar ramp ``g_h`` used by K2 gives

    .. math::

        g_i=\min\{g_h(z_i^{\rm aware}),g_h(z_i^{\rm common})\},
        \qquad
        F(U)=a+\frac1n\sum_i g_i\,
        \operatorname{Clip}_G(x_i-a).

    The common gate is an *acceptance* restriction, not an additional norm
    budget.  Every complete-client contribution therefore remains bounded by
    the one public, client-independent L2 cap ``G``.  There is deliberately no
    division by ``sum_i g_i``.  Conditional on identical anchor, radii, cap,
    block structure and client identities on neighbouring cohorts, replace-one
    sensitivity is at most ``2G/n`` (and ``2bG/n`` for ``b`` replacements).

    If both radius tables coincide, K3 is exactly K2.  If both gates equal one,
    K3 is exactly global one-step centered clipping around ``anchor``.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    slices = _block_slices(blocks)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)
    work_anchor = torch.as_tensor(anchor, device=device, dtype=work_dtype).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")

    aware_radii = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    common_radii = _as_block_matrix(
        common_statistical_radii,
        name="common_statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    cap = float(influence_cap)
    width = float(gate_transition_width)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("gate_transition_width must be finite and positive")
    if (
        not isinstance(num_replacements_for_diagnostics, int)
        or isinstance(num_replacements_for_diagnostics, bool)
        or not 1 <= num_replacements_for_diagnostics <= n
    ):
        raise ValueError("num_replacements_for_diagnostics must lie in [1,n]")

    residuals = work_vectors - work_anchor[None, :]
    aware_normalized_blocks = torch.empty(
        (n, len(blocks)), device=device, dtype=work_dtype
    )
    common_normalized_blocks = torch.empty_like(aware_normalized_blocks)
    for block_index, block_slice in enumerate(slices):
        block_norms = torch.linalg.vector_norm(
            residuals[:, block_slice], dim=1
        )
        aware_normalized_blocks[:, block_index] = (
            block_norms / aware_radii[:, block_index]
        )
        common_normalized_blocks[:, block_index] = (
            block_norms / common_radii[:, block_index]
        )

    aware_normalized_clients = aware_normalized_blocks.square().mean(dim=1).sqrt()
    common_normalized_clients = (
        common_normalized_blocks.square().mean(dim=1).sqrt()
    )
    aware_gates = (
        1.0 - (aware_normalized_clients - 1.0) / width
    ).clamp(0.0, 1.0)
    common_gates = (
        1.0 - (common_normalized_clients - 1.0) / width
    ).clamp(0.0, 1.0)
    gates = torch.minimum(aware_gates, common_gates)

    clipped = _clip_rows_at_radii(
        residuals,
        torch.full((n,), cap, device=device, dtype=work_dtype),
    )
    contributions = gates[:, None] * clipped
    reference = work_anchor + contributions.mean(dim=0)

    contribution_norms = torch.linalg.vector_norm(contributions, dim=1)
    replacement_bound = 2.0 * cap / float(n)
    b_replacement_bound = (
        2.0 * float(num_replacements_for_diagnostics) * cap / float(n)
    )
    tolerance = 64.0 * torch.finfo(work_dtype).eps * max(1.0, cap)
    certificate_respected = bool((contribution_norms <= cap + tolerance).all())
    if not certificate_respected:
        raise RuntimeError("G0g-K3 client contribution exceeded the public cap")
    if not return_diagnostics:
        return reference

    gate_equality_tolerance = 64.0 * torch.finfo(work_dtype).eps
    aware_limiting = aware_gates < common_gates - gate_equality_tolerance
    common_limiting = common_gates < aware_gates - gate_equality_tolerance
    equal_gates = ~(aware_limiting | common_limiting)
    return reference, {
        "reference_name": "g0g_k3_fixed_anchor_dual_gate_global_l2_cap",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "anchor_role": "fixed_public_or_prior_transcript_anchor",
        "anchor_is_required_fixed_on_neighbouring_cohorts": True,
        "statistical_radii_provenance_required": "public_authenticated",
        "common_statistical_radii_provenance_required": (
            "public_identity_blind_sigma_blind"
        ),
        "client_declared_covariance_forbidden": True,
        "statistical_radii_affect_only_covariance_aware_gate": True,
        "common_statistical_radii_affect_only_common_gate": True,
        "influence_cap_is_global_and_client_independent": True,
        "gate_transition_width": width,
        "gate_lipschitz_constant": 1.0 / width,
        "gate_definition": "minimum_of_aware_and_common_scalar_rms_ramp_gates",
        "gate_combination": "elementwise_minimum_no_renormalization",
        "normalization_by_gate_sum": False,
        "aware_normalized_residuals_by_client_block": (
            aware_normalized_blocks.cpu().tolist()
        ),
        "common_normalized_residuals_by_client_block": (
            common_normalized_blocks.cpu().tolist()
        ),
        "aware_normalized_residuals_by_client": (
            aware_normalized_clients.cpu().tolist()
        ),
        "common_normalized_residuals_by_client": (
            common_normalized_clients.cpu().tolist()
        ),
        "aware_gates_by_client": aware_gates.cpu().tolist(),
        "common_gates_by_client": common_gates.cpu().tolist(),
        "gates_by_client": gates.cpu().tolist(),
        "aware_gate_mean": float(aware_gates.mean().item()),
        "common_gate_mean": float(common_gates.mean().item()),
        "gate_mean": float(gates.mean().item()),
        "aware_gate_limiting_fraction": float(aware_limiting.float().mean().item()),
        "common_gate_limiting_fraction": float(
            common_limiting.float().mean().item()
        ),
        "equal_gate_fraction": float(equal_gates.float().mean().item()),
        "aware_statistical_tail_fraction": float(
            (aware_normalized_clients > 1.0).float().mean().item()
        ),
        "common_statistical_tail_fraction": float(
            (common_normalized_clients > 1.0).float().mean().item()
        ),
        "aware_hard_reject_fraction": float(
            (aware_normalized_clients >= 1.0 + width).float().mean().item()
        ),
        "common_hard_reject_fraction": float(
            (common_normalized_clients >= 1.0 + width).float().mean().item()
        ),
        "final_hard_reject_fraction": float((gates <= 0.0).float().mean().item()),
        "complete_client_influence_cap": cap,
        "client_contribution_norms": contribution_norms.cpu().tolist(),
        "client_contribution_cap_respected": certificate_respected,
        "certificate_float_tolerance": tolerance,
        "global_cap_active_fraction": float(
            (torch.linalg.vector_norm(residuals, dim=1) > cap).float().mean().item()
        ),
        "observed_reference_displacement": float(
            torch.linalg.vector_norm(reference - work_anchor).item()
        ),
        "reference_displacement_bound": cap,
        "replace_one_bound": replacement_bound,
        "replace_one_bound_formula": "2*G/n_conditional_on_fixed_public_inputs",
        "num_replacements_for_diagnostics": num_replacements_for_diagnostics,
        "b_replacement_bound": b_replacement_bound,
        "coincident_radii_reduce_exactly_to_k2": True,
        "all_one_gates_reduce_exactly_to_global_centered_clipping": True,
        "local_dp_cost_added_by_reference": 0.0,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(reference.dtype),
    }


@torch.no_grad()
def gaussian_aware_temporal_standardized_messages(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    noise_variances: TensorOrScalar,
    block_sizes: Sequence[int] | None = None,
    variance_floor: float = 1.0e-12,
    standardized_clip_norm: float | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Whiten current residuals with public DP covariance and clip them.

    For authenticated block-diagonal DP covariance
    Sigma_i = diag(v[i,b] I[d_b]), this helper computes

    .. math::

        Z_i = \operatorname{Clip}_{H_z}
        \left((\Sigma_i+\nu I)^{-1/2}(X_i-a)\right).

    The variance floor is only the public numerical ridge \(\nu\). It must not
    absorb empirical inter-client heterogeneity: persistent heterogeneity is
    learned by K4's client-specific enrollment baseline. No statistic is
    estimated from the evaluated cohort.
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)
    work_anchor = torch.as_tensor(anchor, device=device, dtype=work_dtype).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")
    ridge = float(variance_floor)
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("variance_floor must be finite and strictly positive")
    if ridge < torch.finfo(work_dtype).tiny:
        raise ValueError(
            f"variance_floor underflows in {work_dtype}; require at least "
            f"{torch.finfo(work_dtype).tiny:.6g}"
        )
    variances = _as_block_matrix(
        noise_variances,
        name="noise_variances",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    cap = (
        2.0 * math.sqrt(float(dimension))
        if standardized_clip_norm is None
        else float(standardized_clip_norm)
    )
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError(
            "standardized_clip_norm must be finite and strictly positive"
        )
    residuals = work_vectors - work_anchor[None, :]
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError(
            "vectors minus anchor overflowed; temporal residuals must be finite"
        )
    standardized = torch.empty_like(residuals)
    for block_index, block_slice in enumerate(_block_slices(blocks)):
        effective_variance = variances[:, block_index] + ridge
        if not bool(torch.isfinite(effective_variance).all()) or bool(
            (effective_variance <= 0.0).any()
        ):
            raise ValueError(
                "noise variance plus numerical ridge must remain finite and positive"
            )
        scale = effective_variance.sqrt()
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
            raise ValueError("temporal whitening scales must be finite and positive")
        standardized[:, block_slice] = residuals[:, block_slice] / scale[:, None]
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError(
            "temporal whitening overflowed; standardized messages must be finite"
        )
    raw_norms, clipped = _stable_norm_and_clip_rows(
        standardized,
        torch.full((n,), cap, device=device, dtype=work_dtype),
        quantity_name="temporal standardized messages",
    )
    if not bool(torch.isfinite(raw_norms).all()):
        raise ValueError("temporal standardized raw norms must be finite")
    if not bool(torch.isfinite(clipped).all()):
        raise ValueError("temporal standardized clipped output must be finite")
    if not return_diagnostics:
        return clipped
    return clipped, {
        "reference_name": "g0g_k4_public_dp_whitened_temporal_message",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "covariance_role": "public_authenticated_dp_covariance_only",
        "heterogeneity_in_whitener": False,
        "variance_floor_role": "public_numerical_ridge_only",
        "variance_floor": ridge,
        "standardized_clip_norm": cap,
        "raw_standardized_norms": raw_norms.cpu().tolist(),
        "clipped_standardized_norms": (
            torch.linalg.vector_norm(clipped, dim=1).cpu().tolist()
        ),
        "standardized_clip_active_fraction": float(
            (raw_norms > cap).float().mean().item()
        ),
        "stable_scaled_norm_and_clipping": True,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(clipped.dtype),
    }


@torch.no_grad()
def gaussian_aware_fixed_anchor_temporal_gated_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    statistical_radii: TensorOrScalar,
    temporal_standardized_history: torch.Tensor,
    enrollment_standardized_mean: torch.Tensor,
    enrollment_size: int,
    temporal_gate_inner_threshold: float,
    temporal_gate_outer_threshold: float,
    block_sizes: Sequence[int] | None = None,
    influence_cap: float = 1.0,
    current_gate_transition_width: float = 1.0,
    num_replacements_for_diagnostics: int = 1,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    r"""Return G0g-K4 with a causal history gate and a current K2 gate.

    The temporal history has shape \((n,L,d)\) and contains only messages
    available strictly before the current release. With enrollment mean
    \(\mu_i^0\) based on \(W_0\) clean messages, K4 computes

    .. math::

        T_{i,t-1}
        = \frac{\left\|L^{-1}\sum_{r=t-L}^{t-1} Z_{i,r}
          -\mu_i^0\right\|_2}
        {\sqrt{1/L+1/W_0}}.

    The denominator is a nominal variance scaling inherited from independent
    whitened means. Because clipping and temporal drift alter the exact law,
    it is not claimed to make \(T_{i,t-1}\) an exact z-score; clean trajectory
    calibration supplies the operational thresholds.

    The temporal ramp is one below \(c_0\), zero above \(c_1\), and linear
    between them. It is intersected with K2's covariance-aware current gate:

    .. math::

        g_{i,t}=\min\{g^{\rm aware}_{i,t},h_{i,t}\},\qquad
        F_t=a_t+\frac1n\sum_i g_{i,t}
        \operatorname{Clip}_G(X_{i,t}-a_t).

    The current upload is not part of the supplied history, so only the
    historical factor \(h_{i,t}\) is measurable from the past.  The final gate
    is not: it may change with the current upload through
    \(g^{\rm aware}_{i,t}(X_{i,t})\). Contributions are divided by the public
    cohort size \(n\), never by the sum of gates. Conditional on fixed past
    transcript, anchor, radii, thresholds and identities, current-round
    replace-one sensitivity is at most \(2G/n\).
    """

    n, dimension, work_dtype = _validate_vectors(vectors)
    blocks = _resolve_blocks(dimension, block_sizes)
    device = vectors.device
    work_vectors = vectors.to(dtype=work_dtype)
    work_anchor = torch.as_tensor(anchor, device=device, dtype=work_dtype).reshape(-1)
    if work_anchor.shape != (dimension,) or not bool(torch.isfinite(work_anchor).all()):
        raise ValueError("anchor must be a finite vector of dimension d")
    radii = _as_block_matrix(
        statistical_radii,
        name="statistical_radii",
        n=n,
        num_blocks=len(blocks),
        device=device,
        dtype=work_dtype,
        default=0.0,
        strictly_positive=True,
    )
    history = torch.as_tensor(
        temporal_standardized_history, device=device, dtype=work_dtype
    )
    if (
        history.ndim != 3
        or history.shape[0] != n
        or history.shape[2] != dimension
        or history.shape[1] < 1
        or not bool(torch.isfinite(history).all())
    ):
        raise ValueError(
            "temporal_standardized_history must be finite with shape (n,L,d)"
        )
    enrollment = torch.as_tensor(
        enrollment_standardized_mean, device=device, dtype=work_dtype
    )
    if enrollment.shape != (n, dimension) or not bool(torch.isfinite(enrollment).all()):
        raise ValueError(
            "enrollment_standardized_mean must be finite with shape (n,d)"
        )
    if (
        not isinstance(enrollment_size, int)
        or isinstance(enrollment_size, bool)
        or enrollment_size < 1
    ):
        raise ValueError("enrollment_size must be a positive integer")
    c0 = float(temporal_gate_inner_threshold)
    c1 = float(temporal_gate_outer_threshold)
    if not math.isfinite(c0) or not math.isfinite(c1) or not 0.0 <= c0 < c1:
        raise ValueError("temporal thresholds must satisfy 0 <= c0 < c1")
    cap = float(influence_cap)
    width = float(current_gate_transition_width)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("influence_cap must be finite and strictly positive")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(
            "current_gate_transition_width must be finite and positive"
        )
    if (
        not isinstance(num_replacements_for_diagnostics, int)
        or isinstance(num_replacements_for_diagnostics, bool)
        or not 1 <= num_replacements_for_diagnostics <= n
    ):
        raise ValueError("num_replacements_for_diagnostics must lie in [1,n]")

    residuals = work_vectors - work_anchor[None, :]
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("vectors minus anchor overflowed in G0g-K4")
    normalized_blocks = torch.empty(
        (n, len(blocks)), device=device, dtype=work_dtype
    )
    for block_index, block_slice in enumerate(_block_slices(blocks)):
        block_norms, _, _ = _stable_row_norms(
            residuals[:, block_slice],
            quantity_name=f"G0g-K4 residual block {block_index}",
        )
        normalized_blocks[:, block_index] = block_norms / radii[:, block_index]
    if not bool(torch.isfinite(normalized_blocks).all()):
        raise ValueError("G0g-K4 normalized residuals overflowed")
    normalized_norms, _, _ = _stable_row_norms(
        normalized_blocks, quantity_name="G0g-K4 normalized block residuals"
    )
    current_statistics = normalized_norms / math.sqrt(float(len(blocks)))
    if not bool(torch.isfinite(current_statistics).all()):
        raise ValueError("G0g-K4 current statistics must be finite")
    current_gates = (
        1.0 - (current_statistics - 1.0) / width
    ).clamp(0.0, 1.0)
    if not bool(torch.isfinite(current_gates).all()):
        raise ValueError("G0g-K4 current gates must be finite")

    window_size = int(history.shape[1])
    temporal_scale = math.sqrt(
        1.0 / float(window_size) + 1.0 / float(enrollment_size)
    )
    historical_means = history.mean(dim=1)
    if not bool(torch.isfinite(historical_means).all()):
        raise ValueError("G0g-K4 historical means overflowed")
    temporal_differences = historical_means - enrollment
    if not bool(torch.isfinite(temporal_differences).all()):
        raise ValueError("G0g-K4 history-minus-enrollment overflowed")
    temporal_norms, _, _ = _stable_row_norms(
        temporal_differences, quantity_name="G0g-K4 temporal differences"
    )
    temporal_statistics = temporal_norms / temporal_scale
    if not bool(torch.isfinite(temporal_statistics).all()):
        raise ValueError("G0g-K4 temporal statistics must be finite")
    temporal_gates = ((c1 - temporal_statistics) / (c1 - c0)).clamp(0.0, 1.0)
    if not bool(torch.isfinite(temporal_gates).all()):
        raise ValueError("G0g-K4 temporal gates must be finite")
    gates = torch.minimum(current_gates, temporal_gates)
    if not bool(torch.isfinite(gates).all()):
        raise ValueError("G0g-K4 final gates must be finite")

    residual_norms, clipped = _stable_norm_and_clip_rows(
        residuals,
        torch.full((n,), cap, device=device, dtype=work_dtype),
        quantity_name="G0g-K4 globally capped residuals",
    )
    contributions = gates[:, None] * clipped
    if not bool(torch.isfinite(contributions).all()):
        raise ValueError("G0g-K4 gated contributions must be finite")
    reference = work_anchor + contributions.mean(dim=0)
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("G0g-K4 reference must be finite")
    contribution_norms, _, _ = _stable_row_norms(
        contributions, quantity_name="G0g-K4 client contributions"
    )
    reference_displacement, _, _ = _stable_row_norms(
        (reference - work_anchor).reshape(1, -1),
        quantity_name="G0g-K4 reference displacement",
    )
    replacement_bound = 2.0 * cap / float(n)
    b_replacement_bound = (
        2.0 * float(num_replacements_for_diagnostics) * cap / float(n)
    )
    tolerance = 64.0 * torch.finfo(work_dtype).eps * max(1.0, cap)
    certificate_respected = bool((contribution_norms <= cap + tolerance).all())
    if not certificate_respected:
        raise RuntimeError("G0g-K4 client contribution exceeded the public cap")
    if not return_diagnostics:
        return reference
    equality_tolerance = 64.0 * torch.finfo(work_dtype).eps
    current_limiting = current_gates < temporal_gates - equality_tolerance
    temporal_limiting = temporal_gates < current_gates - equality_tolerance
    equal_gates = ~(current_limiting | temporal_limiting)
    return reference, {
        "reference_name": "g0g_k4_temporal_current_gate_global_l2_cap",
        "num_clients": n,
        "dimension": dimension,
        "block_sizes": list(blocks),
        "anchor_role": "fixed_public_or_prior_transcript_anchor",
        "authenticated_persistent_identities_required": True,
        "compromise_before_enrollment_supported": False,
        "temporal_history_is_strictly_prior_to_current_upload": True,
        "history_gate_is_past_measurable": True,
        "final_gate_depends_on_current_upload_via_current_gate": True,
        "final_gate_is_past_measurable": False,
        "temporal_history_window_size": window_size,
        "enrollment_size": enrollment_size,
        "temporal_nominal_scale": temporal_scale,
        "temporal_scale_is_exact_z_score": False,
        "temporal_gate_inner_threshold": c0,
        "temporal_gate_outer_threshold": c1,
        "current_gate_transition_width": width,
        "gate_definition": "minimum_of_k2_aware_current_and_causal_temporal_gate",
        "normalization_by_gate_sum": False,
        "current_normalized_residuals_by_client_block": (
            normalized_blocks.cpu().tolist()
        ),
        "current_normalized_residuals_by_client": current_statistics.cpu().tolist(),
        "temporal_statistics_by_client": temporal_statistics.cpu().tolist(),
        "current_gates_by_client": current_gates.cpu().tolist(),
        "temporal_gates_by_client": temporal_gates.cpu().tolist(),
        "gates_by_client": gates.cpu().tolist(),
        "current_gate_mean": float(current_gates.mean().item()),
        "temporal_gate_mean": float(temporal_gates.mean().item()),
        "gate_mean": float(gates.mean().item()),
        "current_gate_limiting_fraction": float(
            current_limiting.float().mean().item()
        ),
        "temporal_gate_limiting_fraction": float(
            temporal_limiting.float().mean().item()
        ),
        "equal_gate_fraction": float(equal_gates.float().mean().item()),
        "temporal_trigger_fraction": float(
            (temporal_gates < 1.0).float().mean().item()
        ),
        "temporal_detection_fraction": float(
            (temporal_gates <= 0.5).float().mean().item()
        ),
        "complete_client_influence_cap": cap,
        "client_contribution_norms": contribution_norms.cpu().tolist(),
        "client_contribution_cap_respected": certificate_respected,
        "certificate_float_tolerance": tolerance,
        "global_cap_active_fraction": float(
            (residual_norms > cap).float().mean().item()
        ),
        "observed_reference_displacement": float(reference_displacement[0].item()),
        "reference_displacement_bound": cap,
        "replace_one_bound": replacement_bound,
        "replace_one_bound_formula": (
            "2*G/n_conditional_on_fixed_past_transcript_and_public_inputs"
        ),
        "num_replacements_for_diagnostics": num_replacements_for_diagnostics,
        "b_replacement_bound": b_replacement_bound,
        "all_one_temporal_gates_reduce_exactly_to_k2_aware": True,
        "local_dp_cost_added_by_reference": 0.0,
        "input_dtype": str(vectors.dtype),
        "output_dtype": str(reference.dtype),
    }


__all__ = [
    "allocate_gaussian_aware_block_budgets",
    "gaussian_radial_thresholds",
    "gaussian_aware_budget_allocated_correction",
    "gaussian_aware_crossfit_bounded_correction",
    "gaussian_aware_huber_leave_one_out",
    "gaussian_aware_huber_reference",
    "gaussian_aware_fixed_anchor_gated_reference",
    "gaussian_aware_fixed_anchor_scalar_gated_reference",
    "gaussian_aware_fixed_anchor_dual_gated_reference",
    "gaussian_aware_fixed_anchor_temporal_gated_reference",
    "gaussian_aware_temporal_standardized_messages",
    "standardized_quadratic_scores",
]
