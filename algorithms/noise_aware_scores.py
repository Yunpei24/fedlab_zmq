"""Public noise-aware geometry for locally private FAR variants.

The local Gaussian mechanism may deliberately use different public noise
multipliers across clients.  A raw Euclidean residual then mixes two effects:
client update geometry and the scale of the privacy mechanism.  This module
provides deterministic post-processing rules that use only those *public*
noise scales.  It never inspects a realised noise vector or a private loss.

The ``isotropic_dp_covariance_proxy`` rule is intentionally named a proxy.
After local optimisation and server clipping, the exact upload covariance is
not Gaussian in general.  Under the working model

    Cov(noise_i) proportional to a_i^2 I,

and a mean-like reference built from ``n`` independent uploads, the residual
``X_i - F(X)`` has scale approximately

    sqrt(a_i^2 + mean_j(a_j^2) / n).

Dividing by this scale is the isotropic Mahalanobis correction suggested by
that model.  The divisors are normalised to mean one, so homogeneous public
noise scales leave the historical FAR score exactly unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from robustness.aggregators import centered_clipping_leave_one_out

NOISE_SCORE_MODES = {
    "none",
    "client_noise_scale",
    "isotropic_dp_covariance_proxy",
    "isotropic_dp_excess_energy",
    "isotropic_dp_debiased_distance",
}

ENERGY_SCORE_MODE = "isotropic_dp_excess_energy"
DEBIASED_DISTANCE_SCORE_MODE = "isotropic_dp_debiased_distance"
DIRECT_SCORE_MODES = {ENERGY_SCORE_MODE, DEBIASED_DISTANCE_SCORE_MODE}
NULL_MC_ROBUST_SCORE_MODE = (
    "isotropic_dp_null_mc_moment_calibrated_independence_orthogonalized"
)
LOO_EXCESS_ROBUST_SCORE_MODE = (
    "isotropic_dp_loo_excess_calibrated_dual_trust_orthogonalized"
)
NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE = (
    "isotropic_dp_null_mc_moment_theil_sen_separate_trust"
)
NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE = (
    "isotropic_dp_null_mc_moment_tier_robust_separate_trust"
)
TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE = "isotropic_dp_temporal_previous_projection"
TEMPORAL_EMA_PROJECTION_SCORE_MODE = "isotropic_dp_temporal_ema_projection"
EFFECTIVE_NULL_MOMENT_SCORE_MODE = "effective_null_moment_fcc_loo"
EFFECTIVE_NULL_QUANTILE_SCORE_MODE = "effective_null_quantile_fcc_loo"
EFFECTIVE_NULL_SCORE_MODES = {
    EFFECTIVE_NULL_MOMENT_SCORE_MODE,
    EFFECTIVE_NULL_QUANTILE_SCORE_MODE,
}
TEMPORAL_PROJECTION_SCORE_MODES = {
    TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE,
    TEMPORAL_EMA_PROJECTION_SCORE_MODE,
}
ROBUST_DIRECT_SCORE_MODES = {
    NULL_MC_ROBUST_SCORE_MODE,
    LOO_EXCESS_ROBUST_SCORE_MODE,
    NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
    NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
    *TEMPORAL_PROJECTION_SCORE_MODES,
}
NOISE_SCORE_MODES.update(ROBUST_DIRECT_SCORE_MODES)
NOISE_SCORE_MODES.update(EFFECTIVE_NULL_SCORE_MODES)


def public_noise_scales(client_updates, *, device, dtype) -> torch.Tensor:
    """Return positive public noise scales aligned with received client IDs."""

    values = []
    for _, metadata, _ in client_updates:
        value = metadata.get("privacy_noise_multiplier_scale_public")
        if value is None:
            raise ValueError(
                "Noise-aware FAR requires the public per-client DP noise scale"
            )
        values.append(float(value))
    scales = torch.tensor(values, device=device, dtype=dtype)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise ValueError("Public DP noise scales must be finite and strictly positive")
    return scales


def effective_upload_noise_variances(
    client_updates,
    *,
    config: dict,
    server_clip_factors: torch.Tensor | None,
    device,
    dtype,
) -> tuple[torch.Tensor, dict[str, float | str | bool]]:
    """First-order per-coordinate covariance proxy after local DP and clipping.

    With zero momentum, one local Poisson-DP SGD step injects a parameter
    perturbation with per-coordinate standard deviation

    ``lr * C * sigma_i / (q * N_public)``.

    Independent noise over ``K`` steps therefore contributes ``K`` times the
    variance.  The optional server factor is the radial contraction actually
    applied to the already-private upload.  Using that factor is valid local-DP
    post-processing, although it remains a scalar linearisation of the exact
    clipped covariance rather than an identity.
    """

    exact_values = [
        metadata.get("privacy_upload_noise_variance_per_coordinate")
        for _, metadata, _ in client_updates
    ]
    exact_available = all(value is not None for value in exact_values)
    if any(value is not None for value in exact_values) and not exact_available:
        raise ValueError(
            "Exact upload-noise variances must be present for every client or none"
        )

    values = []
    if exact_available:
        for value in exact_values:
            variance = float(value)
            if variance < 0.0 or not math.isfinite(variance):
                raise ValueError("Exact upload-noise variances must be finite")
            values.append(variance)
        covariance_formula = "exact_private_gradient_gaussian_channel"
        covariance_is_proxy = False
    else:
        lr = float(config.get("lr", 0.01))
        clip_norm = float(config.get("clip_norm", 1.0))
        momentum = float(config.get("momentum", 0.0))
        if lr <= 0.0 or clip_norm <= 0.0:
            raise ValueError("lr and clip_norm must be positive for the DP covariance")
        if abs(momentum) > 1e-15:
            raise ValueError(
                "Noise-aware direct scoring currently requires momentum=0 so its "
                "public covariance formula matches the implemented optimiser"
            )
        for _, metadata, _ in client_updates:
            sigma = metadata.get("privacy_noise_multiplier")
            denominator = metadata.get("privacy_normalization_denominator")
            steps = metadata.get("model_steps")
            if sigma is None or denominator is None or steps is None:
                raise ValueError(
                    "DP excess-energy scoring requires public noise multiplier, "
                    "normalisation denominator and step count metadata"
                )
            sigma = float(sigma)
            denominator = float(denominator)
            steps = int(steps)
            if sigma <= 0.0 or denominator <= 0.0 or steps < 1:
                raise ValueError("Invalid public local-DP covariance parameters")
            step_std = lr * clip_norm * sigma / denominator
            values.append(float(steps) * step_std * step_std)
        covariance_formula = "zero_momentum_poisson_dpsgd_first_order"
        covariance_is_proxy = True

    variances = torch.tensor(values, device=device, dtype=dtype)
    include_server = bool(config.get("noise_score_include_server_contraction", True))
    if include_server:
        if server_clip_factors is None or server_clip_factors.shape != variances.shape:
            raise ValueError(
                "Server contraction factors must align with client uploads"
            )
        factors = server_clip_factors.to(device=device, dtype=dtype)
        variances = variances * factors.square()

    ridge = float(config.get("noise_score_variance_ridge", 1e-12))
    if ridge < 0.0:
        raise ValueError("noise_score_variance_ridge must be non-negative")
    variances = variances.clamp_min(ridge)
    diagnostics: dict[str, float | str | bool] = {
        "noise_score_covariance_formula": covariance_formula,
        "noise_score_covariance_is_proxy": covariance_is_proxy,
        "noise_score_uses_public_mechanism_parameters": True,
        "noise_score_uses_private_upload_postprocessing": include_server,
        "noise_score_includes_server_contraction": include_server,
        "noise_score_upload_variance_min": float(variances.min().item()),
        "noise_score_upload_variance_mean": float(variances.mean().item()),
        "noise_score_upload_variance_max": float(variances.max().item()),
    }
    return variances, diagnostics


def mean_reference_residual_variances(
    upload_variances: torch.Tensor,
) -> torch.Tensor:
    r"""Covariance proxy for ``X_i - n^{-1} sum_j X_j``.

    For independent isotropic client noises with variances ``v_i I``, the
    exact per-coordinate variance for a mean reference is

    ``(1 - 1/n)^2 v_i + sum_{j != i} v_j / n^2``.

    Centered clipping and RFA are nonlinear, so using this expression for
    them is a declared first-order reference proxy to be checked by the frozen
    Monte-Carlo experiment.
    """

    if upload_variances.ndim != 1 or upload_variances.numel() < 2:
        raise ValueError("At least two one-dimensional upload variances are required")
    n = upload_variances.numel()
    total = upload_variances.sum()
    return (1.0 - 1.0 / n) ** 2 * upload_variances + (total - upload_variances) / (
        n * n
    )


def leave_one_out_mean_residual_variances(
    upload_variances: torch.Tensor,
) -> torch.Tensor:
    r"""First-order covariance proxy for ``X_i - mean_{j != i} X_j``."""

    if upload_variances.ndim != 1 or upload_variances.numel() < 2:
        raise ValueError("At least two one-dimensional upload variances are required")
    n = upload_variances.numel()
    total = upload_variances.sum()
    return upload_variances + (total - upload_variances) / ((n - 1) ** 2)


def excess_energy_scores(
    distances: torch.Tensor,
    residual_variances: torch.Tensor,
    *,
    score_dimension: int,
    z_clip: float,
    subtract_noise_floor: bool = True,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Return a bounded positive excess-energy score.

    Under the isotropic Gaussian working model and in the absence of signal,
    ``T_i = ||r_i||^2 / v_i`` is approximately chi-square with ``d_s``
    degrees of freedom.  Consequently ``(T_i-d_s)/sqrt(2 d_s)`` is centred at
    zero.  Only its positive part is used by FAR and is divided by the public
    constant ``z_clip`` before clipping to ``[0,1]``.

    ``subtract_noise_floor=False`` defines the corresponding noise-free oracle
    target: the clean noncentral energy contains no Gaussian floor to remove.
    """

    if distances.ndim != 1 or residual_variances.shape != distances.shape:
        raise ValueError("distances and residual_variances must be aligned vectors")
    if score_dimension < 1:
        raise ValueError("score_dimension must be positive")
    if z_clip <= 0.0:
        raise ValueError("noise_score_z_clip must be positive")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")

    energy = distances.square() / residual_variances
    floor = float(score_dimension) if subtract_noise_floor else 0.0
    z = (energy - floor) / math.sqrt(2.0 * float(score_dimension))
    scores = (torch.relu(z) / float(z_clip)).clamp(max=1.0)
    diagnostics: dict[str, float | bool] = {
        "noise_score_noise_floor_subtracted": bool(subtract_noise_floor),
        "noise_score_dimension": float(score_dimension),
        "noise_score_z_clip": float(z_clip),
        "noise_score_energy_min": float(energy.min().item()),
        "noise_score_energy_mean": float(energy.mean().item()),
        "noise_score_energy_max": float(energy.max().item()),
        "noise_score_excess_z_min": float(z.min().item()),
        "noise_score_excess_z_mean": float(z.mean().item()),
        "noise_score_excess_z_max": float(z.max().item()),
        "noise_score_zero_rate": float((scores <= 1e-12).float().mean().item()),
        "noise_score_saturation_rate": float(
            (scores >= 1.0 - 1e-7).float().mean().item()
        ),
    }
    return scores, diagnostics


def debiased_distance_scores(
    distances: torch.Tensor,
    residual_variances: torch.Tensor,
    *,
    score_dimension: int,
    distance_clip: float,
    subtract_noise_floor: bool = True,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Estimate clean residual distance in the original geometric units.

    Under the working model ``r_i = mu_i + eps_i`` with isotropic
    ``eps_i ~ N(0, v_i I_{d_s})``, one has

    ``E ||r_i||^2 = ||mu_i||^2 + d_s v_i``.

    The pre-threshold estimator

    ``||r_i||^2 - d_s v_i``

    is therefore unbiased for the *squared* clean residual norm.  Its positive
    part is used because a squared distance cannot be negative, then converted
    back to distance units and bounded by the public ``distance_clip``:

    ``min(sqrt(max(||r_i||^2 - d_s v_i, 0)) / distance_clip, 1)``.

    The positive-part and square-root transformations introduce finite-sample
    bias; the function does not claim otherwise.  ``subtract_noise_floor=False``
    defines the matching noise-free oracle target and reduces exactly to a
    bounded raw distance.
    """

    if distances.ndim != 1 or residual_variances.shape != distances.shape:
        raise ValueError("distances and residual_variances must be aligned vectors")
    if score_dimension < 1:
        raise ValueError("score_dimension must be positive")
    if distance_clip <= 0.0:
        raise ValueError("distance_clip must be positive")
    if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
        raise ValueError("distances must be finite and non-negative")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")

    noise_floor = (
        float(score_dimension) * residual_variances
        if subtract_noise_floor
        else torch.zeros_like(residual_variances)
    )
    raw_signal_energy = distances.square() - noise_floor
    signal_energy = torch.relu(raw_signal_energy)
    corrected_distances = torch.sqrt(signal_energy)
    scores = (corrected_distances / float(distance_clip)).clamp(max=1.0)
    diagnostics: dict[str, float | bool] = {
        "noise_score_noise_floor_subtracted": bool(subtract_noise_floor),
        "noise_score_dimension": float(score_dimension),
        "noise_score_distance_clip": float(distance_clip),
        "noise_score_floor_min": float(noise_floor.min().item()),
        "noise_score_floor_mean": float(noise_floor.mean().item()),
        "noise_score_floor_max": float(noise_floor.max().item()),
        "noise_score_raw_signal_energy_min": float(raw_signal_energy.min().item()),
        "noise_score_raw_signal_energy_mean": float(raw_signal_energy.mean().item()),
        "noise_score_raw_signal_energy_max": float(raw_signal_energy.max().item()),
        "noise_score_debiased_distance_min": float(corrected_distances.min().item()),
        "noise_score_debiased_distance_mean": float(corrected_distances.mean().item()),
        "noise_score_debiased_distance_max": float(corrected_distances.max().item()),
        "noise_score_zero_rate": float((scores <= 1e-12).float().mean().item()),
        "noise_score_saturation_rate": float(
            (scores >= 1.0 - 1e-7).float().mean().item()
        ),
    }
    return scores, diagnostics


def temporal_projection_scores(
    residuals: torch.Tensor,
    residual_variances: torch.Tensor,
    predictable_vectors: torch.Tensor,
    *,
    lower_z: float = 0.0,
    upper_z: float = 3.0,
    direction_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Score the component of a fresh residual aligned with its public past.

    ``predictable_vectors[i]`` is constructed only from locally-private uploads
    observed before the current round.  After row normalisation it defines the
    predictable direction :math:`h_{i,t-1}`.  The signed projection

    .. math::

        z_{i,t}=\frac{\langle r_{i,t},h_{i,t-1}\rangle}
        {\sqrt{v^{\mathrm{res}}_{i,t}}}

    has unit variance under the working isotropic fresh-noise model, regardless
    of the client's public noise multiplier.  A persistent client-specific
    residual contributes a positive mean when it remains aligned with its past.
    The public affine clip maps ``lower_z`` to zero and ``upper_z`` to one.

    Rows with no usable past direction receive score zero.  This function is
    deterministic post-processing of already locally-private uploads and public
    mechanism parameters; it does not inspect a realised noise vector.
    """

    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError("residuals must have shape (n,d)")
    if predictable_vectors.shape != residuals.shape:
        raise ValueError("predictable_vectors must align with residuals")
    if residual_variances.shape != (residuals.shape[0],):
        raise ValueError("residual_variances must align with residual rows")
    if not math.isfinite(lower_z) or not math.isfinite(upper_z):
        raise ValueError("temporal projection thresholds must be finite")
    if upper_z <= lower_z:
        raise ValueError("upper_z must be strictly greater than lower_z")
    if direction_ridge <= 0.0:
        raise ValueError("direction_ridge must be positive")
    if not bool(torch.isfinite(residuals).all()) or not bool(
        torch.isfinite(predictable_vectors).all()
    ):
        raise ValueError("temporal projection vectors must be finite")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")

    history_norms = torch.linalg.vector_norm(predictable_vectors, dim=1)
    usable = history_norms > float(direction_ridge)
    directions = torch.zeros_like(predictable_vectors)
    directions[usable] = predictable_vectors[usable] / history_norms[usable, None]
    projections = (residuals * directions).sum(dim=1)
    z = projections / residual_variances.sqrt()
    scores = ((z - float(lower_z)) / (float(upper_z) - float(lower_z))).clamp(
        min=0.0, max=1.0
    )
    scores = torch.where(usable, scores, torch.zeros_like(scores))
    usable_z = z[usable]
    diagnostics: dict[str, float | bool] = {
        "noise_score_temporal_projection_applied": True,
        "noise_score_temporal_lower_z": float(lower_z),
        "noise_score_temporal_upper_z": float(upper_z),
        "noise_score_temporal_history_coverage": float(usable.float().mean().item()),
        "noise_score_temporal_history_norm_min": float(history_norms.min().item()),
        "noise_score_temporal_history_norm_mean": float(history_norms.mean().item()),
        "noise_score_temporal_history_norm_max": float(history_norms.max().item()),
        "noise_score_temporal_z_min": (
            float(usable_z.min().item()) if usable_z.numel() else 0.0
        ),
        "noise_score_temporal_z_mean": (
            float(usable_z.mean().item()) if usable_z.numel() else 0.0
        ),
        "noise_score_temporal_z_max": (
            float(usable_z.max().item()) if usable_z.numel() else 0.0
        ),
        "noise_score_temporal_zero_rate": float(
            (scores <= 1e-12).float().mean().item()
        ),
        "noise_score_temporal_saturation_rate": float(
            (scores >= 1.0 - 1e-7).float().mean().item()
        ),
        "noise_score_temporal_is_postprocessing": True,
        "noise_score_temporal_covariance_is_proxy": True,
    }
    return scores, diagnostics


def shrink_residual_variances(
    residual_variances: torch.Tensor,
    *,
    shrinkage: float,
) -> torch.Tensor:
    r"""Shrink client-specific covariance proxies toward a common variance.

    The convention is

    .. math::

        v_i^{(\beta)}=(1-\beta)v_i+\beta\bar v,
        \qquad \beta\in[0,1].

    ``shrinkage=0`` keeps the complete per-client correction, while
    ``shrinkage=1`` uses one pooled variance for every client.  Intermediate
    values deliberately trade noise-scale invariance against the risk of
    over-correcting genuinely heterogeneous client updates.
    """

    if residual_variances.ndim != 1 or residual_variances.numel() < 1:
        raise ValueError("residual_variances must be a non-empty vector")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0,1]")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")
    pooled = residual_variances.mean()
    return (1.0 - float(shrinkage)) * residual_variances + float(shrinkage) * pooled


def mix_isotropic_covariance_variances(
    residual_variances: torch.Tensor,
    *,
    individual_covariance_weight: float,
) -> torch.Tensor:
    r"""Mix a pooled covariance proxy with client-specific covariance proxies.

    This helper uses an explicit convention that is intentionally different
    from the historical ``shrink_residual_variances`` parameterisation:

    .. math::

        v_i^{(w)}=(1-w)\bar v+w v_i,
        \qquad w\in[0,1].

    Consequently ``individual_covariance_weight=0`` gives every client the
    same pooled variance, whereas ``individual_covariance_weight=1`` keeps the
    complete client-specific correction.  The long parameter name prevents
    the endpoint ambiguity that affected earlier ``beta`` experiment labels.
    """

    if residual_variances.ndim != 1 or residual_variances.numel() < 1:
        raise ValueError("residual_variances must be a non-empty vector")
    if not 0.0 <= individual_covariance_weight <= 1.0:
        raise ValueError("individual_covariance_weight must lie in [0,1]")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")
    pooled = residual_variances.mean()
    weight = float(individual_covariance_weight)
    return (1.0 - weight) * pooled + weight * residual_variances


def shrink_null_energy_moments(
    calibration_energies: torch.Tensor,
    *,
    individual_moment_weight: float,
    variance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Interpolate pooled and client-specific null energy moments.

    The raw null energy is :math:`E_i=\lVert r_i\rVert_2^2`.  Shrinking the
    covariance before computing a fully restandardised scalar energy can
    cancel algebraically.  Stage 14 therefore shrinks the *effective null
    moments* themselves:

    .. math::

        m_i^{(w)}=(1-w)\bar m+w m_i,
        \qquad
        v_i^{(w)}=(1-w)\bar v+w v_i.

    ``w=0`` is one pooled calibration and ``w=1`` is a completely
    client-specific calibration.
    """

    if calibration_energies.ndim != 2 or calibration_energies.shape[0] < 2:
        raise ValueError("calibration_energies must have shape (draws, clients)")
    if not 0.0 <= individual_moment_weight <= 1.0:
        raise ValueError("individual_moment_weight must lie in [0,1]")
    if variance_ridge < 0.0:
        raise ValueError("variance_ridge must be non-negative")
    if not bool(torch.isfinite(calibration_energies).all()) or bool(
        (calibration_energies < 0).any()
    ):
        raise ValueError("calibration energies must be finite and non-negative")

    client_mean = calibration_energies.mean(dim=0)
    client_variance = calibration_energies.var(dim=0, unbiased=True)
    pooled_mean = calibration_energies.mean()
    pooled_variance = calibration_energies.reshape(-1).var(unbiased=True)
    weight = float(individual_moment_weight)
    mean = (1.0 - weight) * pooled_mean + weight * client_mean
    variance = (1.0 - weight) * pooled_variance + weight * client_variance
    return mean, variance.clamp_min(float(variance_ridge))


def calibrated_null_energy_scores(
    observed_energies: torch.Tensor,
    calibration_energies: torch.Tensor,
    *,
    mode: str,
    z_clip: float = 4.0,
    tail_probability: float = 0.90,
    individual_calibration_weight: float = 1.0,
    variance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | str | bool]]:
    r"""Map residual energies to bounded scores using an empirical public null.

    ``calibration_energies`` has shape ``(B,n)``: ``B`` artificial cohorts
    generated only from public mechanism parameters, with the same server
    clipping, public score subspace and reference rule as the observed cohort.

    ``individual_calibration_weight`` fixes the public interpolation between
    one pooled null (zero) and client-specific nulls (one).

    ``mode='moment'`` estimates both null moments and returns

    .. math::

        s_i=\operatorname{clip}_{[0,1]}
        \left(\frac{[T_i-\widehat m_i]_+}
        {z_{\max}\sqrt{\widehat v_i+\lambda}}\right).

    ``mode='quantile'`` is the paired distribution-free control.  It replaces
    the standardized excess by the empirical CDF value and activates only the
    public upper-tail fraction above ``tail_probability``.
    """

    if observed_energies.ndim != 1:
        raise ValueError("observed_energies must be a vector")
    if calibration_energies.ndim != 2 or calibration_energies.shape[1:] != (
        observed_energies.numel(),
    ):
        raise ValueError("calibration_energies must have shape (draws, n_clients)")
    if calibration_energies.shape[0] < 20:
        raise ValueError("at least 20 public null draws are required")
    if not bool(torch.isfinite(observed_energies).all()) or not bool(
        torch.isfinite(calibration_energies).all()
    ):
        raise ValueError("null-calibrated energies must be finite")
    if bool((observed_energies < 0).any()) or bool((calibration_energies < 0).any()):
        raise ValueError("null-calibrated energies must be non-negative")
    if variance_ridge < 0.0:
        raise ValueError("variance_ridge must be non-negative")
    if not 0.0 <= individual_calibration_weight <= 1.0:
        raise ValueError("individual_calibration_weight must lie in [0,1]")

    resolved = str(mode).strip().lower()
    diagnostics: dict[str, float | str | bool] = {
        "noise_score_effective_null_mode": resolved,
        "noise_score_effective_null_draws": float(calibration_energies.shape[0]),
        "noise_score_effective_null_is_public_postprocessing": True,
        "noise_score_individual_calibration_weight": float(
            individual_calibration_weight
        ),
    }
    if resolved == "moment":
        if z_clip <= 0.0:
            raise ValueError("z_clip must be positive")
        null_mean, null_variance = shrink_null_energy_moments(
            calibration_energies,
            individual_moment_weight=individual_calibration_weight,
            variance_ridge=variance_ridge,
        )
        null_std = torch.sqrt(null_variance)
        z = (observed_energies - null_mean) / null_std
        scores = (torch.relu(z) / float(z_clip)).clamp(max=1.0)
        diagnostics.update(
            {
                "noise_score_effective_null_z_clip": float(z_clip),
                "noise_score_effective_null_mean_min": float(null_mean.min()),
                "noise_score_effective_null_mean_max": float(null_mean.max()),
                "noise_score_effective_null_variance_min": float(null_variance.min()),
                "noise_score_effective_null_variance_max": float(null_variance.max()),
            }
        )
    elif resolved == "quantile":
        if not 0.0 <= tail_probability < 1.0:
            raise ValueError("tail_probability must lie in [0,1)")
        individual_cdf = (
            (calibration_energies <= observed_energies[None, :])
            .to(observed_energies.dtype)
            .mean(dim=0)
        )
        pooled_cdf = (
            (calibration_energies.reshape(-1, 1) <= observed_energies[None, :])
            .to(observed_energies.dtype)
            .mean(dim=0)
        )
        weight = float(individual_calibration_weight)
        null_cdf = (1.0 - weight) * pooled_cdf + weight * individual_cdf
        scores = (
            (null_cdf - float(tail_probability)) / (1.0 - float(tail_probability))
        ).clamp(min=0.0, max=1.0)
        diagnostics["noise_score_effective_null_tail_probability"] = float(
            tail_probability
        )
    else:
        raise ValueError("mode must be 'moment' or 'quantile'")

    diagnostics.update(
        {
            "noise_score_effective_null_score_min": float(scores.min()),
            "noise_score_effective_null_score_mean": float(scores.mean()),
            "noise_score_effective_null_score_max": float(scores.max()),
            "noise_score_effective_null_zero_rate": float(
                (scores <= 1e-12).to(torch.float64).mean()
            ),
            "noise_score_effective_null_saturation_rate": float(
                (scores >= 1.0 - 1e-7).to(torch.float64).mean()
            ),
        }
    )
    return scores, diagnostics


def effective_null_fcc_loo_scores(
    vectors: torch.Tensor,
    upload_variances: torch.Tensor,
    *,
    anchor: torch.Tensor,
    reference_radius: float,
    calibration_draws: int,
    calibration_seed: int,
    mode: str,
    reference_time_mode: str = "current_loo",
    z_clip: float = 4.0,
    tail_probability: float = 0.90,
    individual_calibration_weight: float = 1.0,
    variance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | str | bool]]:
    r"""Calibrate an effective-null score in the deployed score subspace.

    Two causal reference conventions are available.

    ``current_loo``
        Each observed residual uses an exact one-step centered-clipping
        leave-one-out reference recomputed from the current private uploads.
        Artificial null cohorts use the same rule.

    ``lagged_anchor``
        Every observed residual uses the common predictable reference
        ``anchor = H_{t-1}``.  The null cohorts are centred on that same
        reference.  This is the strict temporal-reference ablation: the fresh
        upload of client ``i`` cannot move the reference used to score itself.

    The artificial cohorts use the public, first-order post-server-clipping
    covariance proxy in ``upload_variances``.  They do not consume training
    randomness and use no raw client data.  Because multi-step DP-SGD and
    radial server clipping are nonlinear, the resulting calibration is an
    *effective-null proxy*, not an exact distributional identity.

    Returns bounded scores, the observed per-client references, the public
    null energy matrix and diagnostics.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("vectors must have shape (n,d) with n >= 2")
    if upload_variances.shape != (vectors.shape[0],):
        raise ValueError("upload_variances must align with vectors")
    if bool((upload_variances <= 0).any()) or not bool(
        torch.isfinite(upload_variances).all()
    ):
        raise ValueError("upload variances must be finite and positive")
    if reference_radius <= 0.0:
        raise ValueError("reference_radius must be positive")
    if calibration_draws < 20:
        raise ValueError("calibration_draws must be at least 20")
    anchor = anchor.to(device=vectors.device, dtype=vectors.dtype).reshape(-1)
    if anchor.numel() != vectors.shape[1]:
        raise ValueError("anchor dimension must match vectors")

    time_mode = str(reference_time_mode).strip().lower()
    if time_mode == "current_loo":
        observed_references = centered_clipping_leave_one_out(
            vectors, anchor=anchor, tau=float(reference_radius)
        )
    elif time_mode == "lagged_anchor":
        observed_references = anchor[None, :].expand_as(vectors)
    else:
        raise ValueError("reference_time_mode must be current_loo or lagged_anchor")
    observed_energies = torch.linalg.vector_norm(
        vectors - observed_references, dim=1
    ).square()

    generator = torch.Generator(device="cpu").manual_seed(int(calibration_seed))
    standard_deviations = upload_variances.detach().cpu().double().sqrt()
    null_anchor = anchor.detach().cpu().double()
    null_rows = []
    for _ in range(int(calibration_draws)):
        noise = torch.randn(
            vectors.shape,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        null_vectors = null_anchor[None, :] + noise * standard_deviations[:, None]
        if time_mode == "current_loo":
            null_references = centered_clipping_leave_one_out(
                null_vectors,
                anchor=null_anchor,
                tau=float(reference_radius),
            )
        else:
            null_references = null_anchor[None, :].expand_as(null_vectors)
        null_rows.append(
            torch.linalg.vector_norm(null_vectors - null_references, dim=1).square()
        )
    calibration_energies = torch.stack(null_rows).to(observed_energies)
    scores, score_metrics = calibrated_null_energy_scores(
        observed_energies,
        calibration_energies,
        mode=mode,
        z_clip=z_clip,
        tail_probability=tail_probability,
        individual_calibration_weight=individual_calibration_weight,
        variance_ridge=variance_ridge,
    )
    diagnostics: dict[str, float | str | bool] = {
        **score_metrics,
        "noise_score_effective_null_reference_time_mode": time_mode,
        "noise_score_effective_null_reference_radius": float(reference_radius),
        "noise_score_effective_null_calibration_seed": float(calibration_seed),
        "noise_score_effective_null_covariance_is_proxy": True,
        "noise_score_effective_null_training_rng_isolated": True,
        "noise_score_effective_null_observed_energy_min": float(
            observed_energies.min().item()
        ),
        "noise_score_effective_null_observed_energy_mean": float(
            observed_energies.mean().item()
        ),
        "noise_score_effective_null_observed_energy_max": float(
            observed_energies.max().item()
        ),
    }
    return scores, observed_references, calibration_energies, diagnostics


def null_quantile_scores(
    distances: torch.Tensor,
    residual_variances: torch.Tensor,
    *,
    score_dimension: int,
    lower_z: float = 1.2815515655446004,
    upper_z: float = 3.5,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Map excess residual energy to a public approximate null quantile.

    Under the isotropic working null

    .. math::

        r_i\sim\mathcal N(0,v_i I_d),\qquad
        T_i=\lVert r_i\rVert_2^2/v_i\sim\chi_d^2.

    The Wilson--Hilferty transform

    .. math::

        z_i=\frac{(T_i/d)^{1/3}-(1-2/(9d))}{\sqrt{2/(9d)}}

    is approximately standard normal.  The returned score is zero below the
    public threshold ``lower_z``, one above ``upper_z``, and linear between
    them.  Thus the threshold has a dimension-independent interpretation:
    the default ``lower_z`` is approximately the upper 10% null quantile.

    This is a diagnostic score under an isotropic covariance model, not an
    exact finite-sample p-value after nonlinear robust referencing or server
    clipping.  Those departures are audited empirically before promotion.
    """

    if distances.ndim != 1 or residual_variances.shape != distances.shape:
        raise ValueError("distances and residual_variances must be aligned vectors")
    if score_dimension < 1:
        raise ValueError("score_dimension must be positive")
    if not math.isfinite(lower_z) or not math.isfinite(upper_z):
        raise ValueError("null-quantile thresholds must be finite")
    if upper_z <= lower_z:
        raise ValueError("upper_z must be strictly greater than lower_z")
    if bool((residual_variances <= 0).any()) or not bool(
        torch.isfinite(residual_variances).all()
    ):
        raise ValueError("residual variances must be finite and positive")

    dimension = float(score_dimension)
    energy = distances.square() / residual_variances
    transformed = torch.pow((energy / dimension).clamp_min(0.0), 1.0 / 3.0)
    null_mean = 1.0 - 2.0 / (9.0 * dimension)
    null_std = math.sqrt(2.0 / (9.0 * dimension))
    z = (transformed - null_mean) / null_std
    scores = ((z - float(lower_z)) / (float(upper_z) - float(lower_z))).clamp(
        min=0.0, max=1.0
    )
    diagnostics: dict[str, float | bool] = {
        "noise_score_dimension": dimension,
        "noise_score_null_lower_z": float(lower_z),
        "noise_score_null_upper_z": float(upper_z),
        "noise_score_null_z_min": float(z.min().item()),
        "noise_score_null_z_mean": float(z.mean().item()),
        "noise_score_null_z_max": float(z.max().item()),
        "noise_score_zero_rate": float((scores <= 1e-12).float().mean().item()),
        "noise_score_saturation_rate": float(
            (scores >= 1.0 - 1e-7).float().mean().item()
        ),
        "noise_score_null_is_approximation": True,
    }
    return scores, diagnostics


def directional_trust_scores(
    vectors: torch.Tensor,
    references: torch.Tensor,
    *,
    reject_cosine: float = -0.1,
    full_trust_cosine: float = 0.25,
    norm_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Return a bounded directional trust gate relative to robust references.

    ``references`` may be one common vector or one leave-one-out reference per
    client.  The cosine alignment is mapped linearly from zero trust at
    ``reject_cosine`` to full trust at ``full_trust_cosine``.  This gate is
    designed to suppress large anti-aligned IPM/Bit-Flip messages without
    automatically suppressing a large but directionally plausible honest
    update.  ALIE-like attacks can remain aligned, which is why the gate is
    evaluated against several attacks instead of being treated as a proof.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 1:
        raise ValueError("vectors must have shape (n,d)")
    if references.ndim == 1:
        references = references.reshape(1, -1).expand_as(vectors)
    if references.shape != vectors.shape:
        raise ValueError("references must be one vector or align with vectors")
    if not -1.0 <= reject_cosine < full_trust_cosine <= 1.0:
        raise ValueError(
            "cosine thresholds must satisfy -1 <= reject < full_trust <= 1"
        )
    if norm_ridge <= 0.0:
        raise ValueError("norm_ridge must be positive")

    numerator = (vectors * references).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(vectors, dim=1)
        * torch.linalg.vector_norm(references, dim=1)
    ).clamp_min(float(norm_ridge))
    cosines = (numerator / denominator).clamp(min=-1.0, max=1.0)
    trust = (
        (cosines - float(reject_cosine))
        / (float(full_trust_cosine) - float(reject_cosine))
    ).clamp(min=0.0, max=1.0)
    diagnostics = {
        "noise_score_alignment_min": float(cosines.min().item()),
        "noise_score_alignment_mean": float(cosines.mean().item()),
        "noise_score_alignment_max": float(cosines.max().item()),
        "noise_score_trust_min": float(trust.min().item()),
        "noise_score_trust_mean": float(trust.mean().item()),
        "noise_score_trust_max": float(trust.max().item()),
    }
    return trust, diagnostics


def lagged_descent_alignment_scores(
    vectors: torch.Tensor,
    lagged_direction: torch.Tensor,
    *,
    reject_cosine: float = 0.0,
    full_support_cosine: float = 0.50,
    neutral_score: float = 0.50,
    norm_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Score alignment with a robust direction fixed by the past transcript.

    ``lagged_direction`` is the robust reference computed from the preceding
    cohort of already locally-private uploads.  For a current upload ``x_i``,
    the public utility proxy is

    .. math::

        a_i = \left[\frac{\cos(x_i,g_{t-1})-c_0}{c_1-c_0}\right]_{[0,1]}.

    A positive value means that the upload points in a direction compatible
    with the previous robust model displacement.  It does *not* certify a
    decrease of the population loss: non-IID curvature and temporal drift can
    make a useful update poorly aligned with the previous direction.  This is
    precisely the end-to-end hypothesis that must be tested independently.

    The direction is fixed before the current uploads are processed, so a
    client's fresh local-DP noise cannot choose the vector against which its
    own alignment is measured.  All inputs are private releases or public
    constants; the score is therefore deterministic local-DP post-processing.
    If the past direction has negligible norm, the function returns one common
    neutral score and creates no artificial ordering.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 1:
        raise ValueError("vectors must have shape (n,d)")
    if lagged_direction.shape != (vectors.shape[1],):
        raise ValueError("lagged_direction must match the vector dimension")
    if not -1.0 <= reject_cosine < full_support_cosine <= 1.0:
        raise ValueError("invalid lagged-direction cosine thresholds")
    if not 0.0 <= neutral_score <= 1.0:
        raise ValueError("neutral_score must lie in [0,1]")
    if norm_ridge <= 0.0:
        raise ValueError("norm_ridge must be positive")
    if not bool(torch.isfinite(vectors).all()) or not bool(
        torch.isfinite(lagged_direction).all()
    ):
        raise ValueError("vectors and lagged_direction must be finite")

    direction_norm = torch.linalg.vector_norm(lagged_direction)
    if float(direction_norm.item()) <= float(norm_ridge):
        neutral = torch.full(
            (vectors.shape[0],),
            float(neutral_score),
            dtype=vectors.dtype,
            device=vectors.device,
        )
        return neutral, {
            "noise_score_lagged_descent_direction_available": False,
            "noise_score_lagged_descent_direction_norm": float(direction_norm.item()),
            "noise_score_lagged_descent_alignment_min": 0.0,
            "noise_score_lagged_descent_alignment_mean": 0.0,
            "noise_score_lagged_descent_alignment_max": 0.0,
            "noise_score_lagged_descent_trust_min": float(neutral_score),
            "noise_score_lagged_descent_trust_mean": float(neutral_score),
            "noise_score_lagged_descent_trust_max": float(neutral_score),
            "noise_score_lagged_descent_is_loss_decrease_certificate": False,
            "noise_score_lagged_descent_is_private_postprocessing": True,
        }

    vector_norms = torch.linalg.vector_norm(vectors, dim=1)
    denominator = (vector_norms * direction_norm).clamp_min(float(norm_ridge))
    cosines = ((vectors * lagged_direction).sum(dim=1) / denominator).clamp(
        min=-1.0, max=1.0
    )
    trust = (
        (cosines - float(reject_cosine))
        / (float(full_support_cosine) - float(reject_cosine))
    ).clamp(min=0.0, max=1.0)
    diagnostics: dict[str, float | bool] = {
        "noise_score_lagged_descent_direction_available": True,
        "noise_score_lagged_descent_direction_norm": float(direction_norm.item()),
        "noise_score_lagged_descent_alignment_min": float(cosines.min().item()),
        "noise_score_lagged_descent_alignment_mean": float(cosines.mean().item()),
        "noise_score_lagged_descent_alignment_max": float(cosines.max().item()),
        "noise_score_lagged_descent_trust_min": float(trust.min().item()),
        "noise_score_lagged_descent_trust_mean": float(trust.mean().item()),
        "noise_score_lagged_descent_trust_max": float(trust.max().item()),
        "noise_score_lagged_descent_is_loss_decrease_certificate": False,
        "noise_score_lagged_descent_is_private_postprocessing": True,
    }
    return trust.to(dtype=vectors.dtype), diagnostics


def ranked_directional_peer_support_scores(
    vectors: torch.Tensor,
    references: torch.Tensor,
    *,
    assumed_byzantine: int,
    reference_reject_cosine: float = -0.10,
    reference_full_support_cosine: float = 0.25,
    peer_reject_cosine: float = 0.00,
    peer_full_support_cosine: float = 0.20,
    extra_honest_supporters: int = 1,
    norm_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    r"""Require both robust-direction and cross-client directional support.

    Let :math:`c_{ij}` be the cosine between uploads ``i`` and ``j``.  For
    every client, the peer statistic is the ``k``-th largest off-diagonal
    cosine, where

    .. math::

        k = \min\{n-1, f+h\}.

    Here ``f`` is the public assumed Byzantine count and ``h`` is
    ``extra_honest_supporters``.  Consequently, a coalition of at most ``f``
    clients cannot give every member full peer support using only within-
    coalition copies: each member has at most ``f-1`` Byzantine peers.  This
    is a *support condition*, not a Byzantine-identification theorem.  An
    adversary aligned with sufficiently many honest messages can still pass.

    The peer statistic and the cosine between each upload and its robust
    reference are mapped linearly to ``[0,1]``.  Their minimum is returned, so
    a message must satisfy both conditions.  The rule operates only on
    already-private uploads and therefore is deterministic local-DP
    post-processing; its purpose is utility/robustness, not extra privacy.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("vectors must have shape (n,d) with n >= 2")
    if references.ndim == 1:
        references = references.reshape(1, -1).expand_as(vectors)
    if references.shape != vectors.shape:
        raise ValueError("references must be one vector or align with vectors")
    if assumed_byzantine < 0 or assumed_byzantine >= vectors.shape[0]:
        raise ValueError("assumed_byzantine must lie in [0,n)")
    if extra_honest_supporters < 1:
        raise ValueError("extra_honest_supporters must be at least one")
    if not -1.0 <= reference_reject_cosine < reference_full_support_cosine <= 1.0:
        raise ValueError("invalid reference cosine thresholds")
    if not -1.0 <= peer_reject_cosine < peer_full_support_cosine <= 1.0:
        raise ValueError("invalid peer cosine thresholds")
    if norm_ridge <= 0.0:
        raise ValueError("norm_ridge must be positive")
    if not bool(torch.isfinite(vectors).all()) or not bool(
        torch.isfinite(references).all()
    ):
        raise ValueError("vectors and references must be finite")

    reference_support, reference_diagnostics = directional_trust_scores(
        vectors,
        references,
        reject_cosine=reference_reject_cosine,
        full_trust_cosine=reference_full_support_cosine,
        norm_ridge=norm_ridge,
    )

    norms = torch.linalg.vector_norm(vectors, dim=1).clamp_min(float(norm_ridge))
    unit_vectors = vectors / norms[:, None]
    cosines = (unit_vectors @ unit_vectors.T).clamp(min=-1.0, max=1.0)
    diagonal = torch.eye(vectors.shape[0], dtype=torch.bool, device=vectors.device)
    off_diagonal = cosines.masked_fill(diagonal, float("-inf"))
    required_rank = min(
        vectors.shape[0] - 1,
        int(assumed_byzantine) + int(extra_honest_supporters),
    )
    ranked_cosine = off_diagonal.topk(required_rank, dim=1).values[:, -1]
    peer_support = (
        (ranked_cosine - float(peer_reject_cosine))
        / (float(peer_full_support_cosine) - float(peer_reject_cosine))
    ).clamp(min=0.0, max=1.0)
    support = torch.minimum(reference_support, peer_support)
    diagnostics: dict[str, float | int | bool] = {
        **reference_diagnostics,
        "noise_score_peer_support_required_rank": int(required_rank),
        "noise_score_peer_support_assumed_byzantine": int(assumed_byzantine),
        "noise_score_peer_support_extra_honest_supporters": int(
            extra_honest_supporters
        ),
        "noise_score_peer_ranked_cosine_min": float(ranked_cosine.min().item()),
        "noise_score_peer_ranked_cosine_mean": float(ranked_cosine.mean().item()),
        "noise_score_peer_ranked_cosine_max": float(ranked_cosine.max().item()),
        "noise_score_peer_support_min": float(peer_support.min().item()),
        "noise_score_peer_support_mean": float(peer_support.mean().item()),
        "noise_score_peer_support_max": float(peer_support.max().item()),
        "noise_score_joint_support_min": float(support.min().item()),
        "noise_score_joint_support_mean": float(support.mean().item()),
        "noise_score_joint_support_max": float(support.max().item()),
        "noise_score_peer_support_is_private_postprocessing": True,
        "noise_score_peer_support_is_identification_certificate": False,
    }
    return support.to(dtype=vectors.dtype), diagnostics


def multi_krum_admissibility_trust_scores(
    vectors: torch.Tensor,
    *,
    assumed_byzantine: int,
    distance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    r"""Rank uploads by the classical Multi-Krum neighbourhood criterion.

    For every upload ``i``, let ``K_i`` contain its ``n-f-2`` nearest other
    uploads.  Its Krum score is

    .. math::

        k_i = \sum_{j\in K_i}\lVert x_i-x_j\rVert_2^2.

    Smaller values indicate stronger support from the cohort.  The returned
    trust is the reversed min--max normalisation of ``k_i``; only its ordering
    is used by the delayed admissibility filter.  Equal scores remain equal so
    no data-dependent hidden tie-break is introduced.

    This is deterministic post-processing of locally private uploads.  The
    usual Krum condition ``n >= 2f+3`` is checked, but this function alone does
    not claim that real honest updates satisfy Krum's distributional or
    angle assumptions.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("vectors must have shape (n,d) with n >= 2")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")
    n = int(vectors.shape[0])
    f = int(assumed_byzantine)
    if f < 0 or n < 2 * f + 3:
        raise ValueError("Multi-Krum requires n >= 2f+3 and f >= 0")
    if distance_ridge <= 0.0:
        raise ValueError("distance_ridge must be positive")

    neighbours = n - f - 2
    squared_distances = torch.cdist(vectors, vectors, p=2).square()
    diagonal = torch.eye(n, dtype=torch.bool, device=vectors.device)
    squared_distances = squared_distances.masked_fill(diagonal, float("inf"))
    krum_scores = squared_distances.topk(neighbours, largest=False, dim=1).values.sum(
        dim=1
    )
    score_min = krum_scores.min()
    score_max = krum_scores.max()
    span = score_max - score_min
    if float(span.item()) <= float(distance_ridge):
        trust = torch.ones_like(krum_scores)
    else:
        trust = (score_max - krum_scores) / span
    diagnostics: dict[str, float | int | bool] = {
        "noise_score_multi_krum_assumed_byzantine": f,
        "noise_score_multi_krum_neighbour_count": int(neighbours),
        "noise_score_multi_krum_score_min": float(score_min.item()),
        "noise_score_multi_krum_score_mean": float(krum_scores.mean().item()),
        "noise_score_multi_krum_score_max": float(score_max.item()),
        "noise_score_multi_krum_trust_min": float(trust.min().item()),
        "noise_score_multi_krum_trust_mean": float(trust.mean().item()),
        "noise_score_multi_krum_trust_max": float(trust.max().item()),
        "noise_score_multi_krum_is_private_postprocessing": True,
        "noise_score_multi_krum_is_universal_identification_certificate": False,
    }
    return trust.to(dtype=vectors.dtype), diagnostics


def pairwise_independence_trust_scores(
    vectors: torch.Tensor,
    upload_variances: torch.Tensor,
    *,
    low_separation: float = 0.20,
    full_separation: float = 0.70,
    variance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Detect implausibly identical uploads under independent local-DP noise.

    For two honest uploads whose independent isotropic noise variances are
    ``v_i`` and ``v_j``, the typical noise-only pairwise distance is of order

    .. math::

        \sqrt{d(v_i+v_j)}.

    We divide every pairwise distance by that public scale and retain the
    nearest-neighbour separation of each client.  Exact or nearly exact
    colluding copies then receive low trust, while independently noised
    messages should not collapse to zero separation.

    This is an anti-collusion diagnostic, not a universal Byzantine detector:
    adversaries that independently jitter their uploads may evade it.  It is
    therefore combined with, and audited separately from, directional trust.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("vectors must have shape (n,d) with n >= 2")
    if upload_variances.shape != (vectors.shape[0],):
        raise ValueError("upload_variances must align with vectors")
    if bool((upload_variances <= 0).any()) or not bool(
        torch.isfinite(upload_variances).all()
    ):
        raise ValueError("upload variances must be finite and positive")
    if not 0.0 <= low_separation < full_separation:
        raise ValueError("separation thresholds must satisfy 0 <= low < full")
    if variance_ridge <= 0.0:
        raise ValueError("variance_ridge must be positive")

    nearest = nearest_standardized_separations(
        vectors,
        upload_variances,
        variance_ridge=variance_ridge,
    )
    trust = (
        (nearest - float(low_separation))
        / (float(full_separation) - float(low_separation))
    ).clamp(min=0.0, max=1.0)
    diagnostics = {
        "noise_score_nearest_standardized_separation_min": float(nearest.min().item()),
        "noise_score_nearest_standardized_separation_mean": float(
            nearest.mean().item()
        ),
        "noise_score_nearest_standardized_separation_max": float(nearest.max().item()),
        "noise_score_independence_trust_min": float(trust.min().item()),
        "noise_score_independence_trust_mean": float(trust.mean().item()),
        "noise_score_independence_trust_max": float(trust.max().item()),
    }
    return trust, diagnostics


def nearest_standardized_separations(
    vectors: torch.Tensor,
    upload_variances: torch.Tensor,
    *,
    variance_ridge: float = 1e-12,
) -> torch.Tensor:
    r"""Nearest pairwise distance in public DP standard-deviation units."""

    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("vectors must have shape (n,d) with n >= 2")
    if upload_variances.shape != (vectors.shape[0],):
        raise ValueError("upload_variances must align with vectors")
    if bool((upload_variances <= 0).any()) or not bool(
        torch.isfinite(upload_variances).all()
    ):
        raise ValueError("upload variances must be finite and positive")
    if variance_ridge <= 0.0:
        raise ValueError("variance_ridge must be positive")

    pairwise = torch.cdist(vectors, vectors, p=2)
    dimension = float(vectors.shape[1])
    scale = torch.sqrt(
        dimension
        * (
            upload_variances[:, None]
            + upload_variances[None, :]
            + float(variance_ridge)
        )
    )
    standardized = pairwise / scale
    diagonal = torch.eye(vectors.shape[0], device=vectors.device, dtype=torch.bool)
    return standardized.masked_fill(diagonal, float("inf")).min(dim=1).values


def calibrated_independence_trust_scores(
    vectors: torch.Tensor,
    upload_variances: torch.Tensor,
    *,
    calibration_draws: int,
    calibration_seed: int,
    low_null_quantile: float = 0.05,
    full_trust_null_quantile: float = 0.50,
    trust_floor: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    r"""Calibrate anti-collusion trust under independent Gaussian uploads.

    For each client, artificial public-null cohorts are generated with the
    declared post-contraction covariance proxy.  The observed nearest-neighbour
    separation is converted to its empirical null CDF.  A separation below the
    lower null quantile receives ``trust_floor``; trust reaches one at the null
    median.  No private record or unnoised update is used by this simulation.
    """

    if calibration_draws < 20:
        raise ValueError("calibration_draws must be at least 20")
    if not 0.0 <= low_null_quantile < full_trust_null_quantile <= 1.0:
        raise ValueError("invalid null-quantile trust thresholds")
    if not 0.0 <= trust_floor <= 1.0:
        raise ValueError("trust_floor must lie in [0,1]")
    observed = nearest_standardized_separations(vectors, upload_variances)
    generator = torch.Generator(device="cpu").manual_seed(int(calibration_seed))
    null_rows = []
    standard_deviations = upload_variances.detach().cpu().double().sqrt()
    for _ in range(calibration_draws):
        noise = torch.randn(
            vectors.shape,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        null_vectors = noise * standard_deviations[:, None]
        null_rows.append(
            nearest_standardized_separations(
                null_vectors,
                upload_variances.detach().cpu().double(),
            )
        )
    null_samples = torch.stack(null_rows).to(observed)
    null_cdf = (null_samples <= observed[None, :]).double().mean(dim=0)
    calibrated = (
        (null_cdf - float(low_null_quantile))
        / (float(full_trust_null_quantile) - float(low_null_quantile))
    ).clamp(0.0, 1.0)
    trust = float(trust_floor) + (1.0 - float(trust_floor)) * calibrated
    diagnostics: dict[str, float | int | bool] = {
        "noise_score_independence_calibration_draws": int(calibration_draws),
        "noise_score_independence_calibration_seed": int(calibration_seed),
        "noise_score_independence_low_null_quantile": float(low_null_quantile),
        "noise_score_independence_full_trust_null_quantile": float(
            full_trust_null_quantile
        ),
        "noise_score_independence_null_cdf_min": float(null_cdf.min().item()),
        "noise_score_independence_null_cdf_mean": float(null_cdf.mean().item()),
        "noise_score_independence_null_cdf_max": float(null_cdf.max().item()),
        "noise_score_independence_trust_min": float(trust.min().item()),
        "noise_score_independence_trust_mean": float(trust.mean().item()),
        "noise_score_independence_trust_max": float(trust.max().item()),
        "noise_score_independence_is_null_calibrated": True,
    }
    return trust.to(dtype=vectors.dtype), diagnostics


def null_mc_moment_scores(
    distances: torch.Tensor,
    upload_variances: torch.Tensor,
    *,
    score_dimension: int,
    reference_builder: Callable[[torch.Tensor], torch.Tensor],
    null_center: torch.Tensor,
    calibration_draws: int,
    calibration_seed: int,
    z_clip: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    r"""Score residual energy against a deterministic Monte-Carlo null.

    Null uploads are independent isotropic Gaussian perturbations around the
    already-private robust reference.  The same reference rule is recomputed
    for every artificial cohort, so the moments include its finite-cohort and
    nonlinear effects.  The observed energy is centred by the simulated mean,
    divided by the simulated standard deviation, rectified and bounded.
    """

    if distances.ndim != 1 or upload_variances.shape != distances.shape:
        raise ValueError("distances and upload_variances must be aligned vectors")
    if null_center.shape != (score_dimension,):
        raise ValueError("null_center must match score_dimension")
    if calibration_draws < 20:
        raise ValueError("calibration_draws must be at least 20")
    if z_clip <= 0.0:
        raise ValueError("z_clip must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(calibration_seed))
    standard_deviations = upload_variances.detach().cpu().double().sqrt()
    centre_cpu = null_center.detach().cpu().double()
    energy_rows = []
    for _ in range(calibration_draws):
        noise = torch.randn(
            upload_variances.numel(),
            score_dimension,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )
        null_vectors_cpu = centre_cpu[None, :] + noise * standard_deviations[:, None]
        null_vectors = null_vectors_cpu.to(
            device=distances.device,
            dtype=null_center.dtype,
        )
        null_reference = reference_builder(null_vectors)
        energy_rows.append(
            torch.linalg.vector_norm(null_vectors - null_reference, dim=1).square()
        )
    energies = torch.stack(energy_rows)
    mean = energies.mean(dim=0)
    std = energies.std(dim=0, unbiased=True).clamp_min(1e-12)
    z = (distances.square() - mean) / std
    scores = (torch.relu(z) / float(z_clip)).clamp(max=1.0)
    diagnostics: dict[str, float | int | bool] = {
        "noise_score_null_mc_draws": int(calibration_draws),
        "noise_score_null_mc_seed": int(calibration_seed),
        "noise_score_null_mc_z_clip": float(z_clip),
        "noise_score_null_energy_mean_min": float(mean.min().item()),
        "noise_score_null_energy_mean_mean": float(mean.mean().item()),
        "noise_score_null_energy_mean_max": float(mean.max().item()),
        "noise_score_null_energy_std_min": float(std.min().item()),
        "noise_score_null_energy_std_mean": float(std.mean().item()),
        "noise_score_null_energy_std_max": float(std.max().item()),
        "noise_score_null_z_min": float(z.min().item()),
        "noise_score_null_z_mean": float(z.mean().item()),
        "noise_score_null_z_max": float(z.max().item()),
        "noise_score_null_mc_is_postprocessing": True,
        "noise_score_null_mc_covariance_is_proxy": True,
    }
    return scores, diagnostics


def robust_novelty_scores(
    novelty_scores: torch.Tensor,
    trust_scores: torch.Tensor,
    *,
    trust_floor: float = 0.0,
) -> torch.Tensor:
    r"""Combine a FAR novelty signal with a bounded robust trust gate.

    The product

    ``novelty * (trust_floor + (1-trust_floor) * trust)``

    remains in ``[0,1]``.  ``trust_floor`` is public and prevents the gate from
    making a client's novelty identically irrelevant when desired.
    """

    if novelty_scores.shape != trust_scores.shape or novelty_scores.ndim != 1:
        raise ValueError("novelty_scores and trust_scores must be aligned vectors")
    if not 0.0 <= trust_floor <= 1.0:
        raise ValueError("trust_floor must lie in [0,1]")
    if not bool(torch.isfinite(novelty_scores).all()) or not bool(
        torch.isfinite(trust_scores).all()
    ):
        raise ValueError("novelty and trust scores must be finite")
    if bool((novelty_scores < 0).any()) or bool((novelty_scores > 1).any()):
        raise ValueError("novelty scores must lie in [0,1]")
    if bool((trust_scores < 0).any()) or bool((trust_scores > 1).any()):
        raise ValueError("trust scores must lie in [0,1]")
    multiplier = float(trust_floor) + (1.0 - float(trust_floor)) * trust_scores
    return (novelty_scores * multiplier).clamp(min=0.0, max=1.0)


def separate_novelty_trust_scores(
    novelty_scores: torch.Tensor,
    trust_scores: torch.Tensor,
    *,
    trust_fraction: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Allocate a bounded logit budget to novelty and robust trust.

    Generation 4 multiplied novelty by trust.  A moderately uncertain trust
    estimate could therefore erase an honest tail score even when the novelty
    channel was correct.  Generation 5 keeps the two roles explicit:

    .. math::

        s_i=(1-\gamma)\nu_i+\gamma t_i,
        \qquad \gamma\in[0,1].

    where ``novelty`` :math:`\nu_i` and ``trust`` :math:`t_i` both lie in
    ``[0,1]``.  If the total public tilt is :math:`\alpha`, this is equivalent
    inside the softmax to

    .. math::

        \alpha_d\nu_i-\alpha_c(1-t_i),\qquad
        \alpha_d=(1-\gamma)\alpha,\quad
        \alpha_c=\gamma\alpha,

    because the omitted term :math:`-\alpha_c` is common to all logits and
    cancels in the softmax.  Thus novelty receives its own positive budget and
    suspicion receives a separate negative budget.  Since
    :math:`\alpha_d+\alpha_c=\alpha` and the composite score remains in
    ``[0,1]``, the existing public softmax-weight cap is unchanged.

    This construction is a utility/robustness heuristic, not a Byzantine
    identification theorem.  Its two components are reported separately and
    must pass a frozen end-to-end audit.
    """

    if novelty_scores.shape != trust_scores.shape or novelty_scores.ndim != 1:
        raise ValueError("novelty_scores and trust_scores must be aligned vectors")
    if not 0.0 <= trust_fraction <= 1.0:
        raise ValueError("trust_fraction must lie in [0,1]")
    if not bool(torch.isfinite(novelty_scores).all()) or not bool(
        torch.isfinite(trust_scores).all()
    ):
        raise ValueError("novelty and trust scores must be finite")
    if bool((novelty_scores < 0).any()) or bool((novelty_scores > 1).any()):
        raise ValueError("novelty scores must lie in [0,1]")
    if bool((trust_scores < 0).any()) or bool((trust_scores > 1).any()):
        raise ValueError("trust scores must lie in [0,1]")

    novelty_fraction = 1.0 - float(trust_fraction)
    combined = novelty_fraction * novelty_scores + float(trust_fraction) * trust_scores
    diagnostics: dict[str, float | bool] = {
        "noise_score_channels_separated": True,
        "noise_score_novelty_logit_fraction": novelty_fraction,
        "noise_score_trust_logit_fraction": float(trust_fraction),
        "noise_score_novelty_min": float(novelty_scores.min().item()),
        "noise_score_novelty_mean": float(novelty_scores.mean().item()),
        "noise_score_novelty_max": float(novelty_scores.max().item()),
        "noise_score_combined_trust_min": float(trust_scores.min().item()),
        "noise_score_combined_trust_mean": float(trust_scores.mean().item()),
        "noise_score_combined_trust_max": float(trust_scores.max().item()),
        "noise_score_combined_zero_rate": float(
            (combined <= 1e-12).float().mean().item()
        ),
    }
    return combined, diagnostics


def tierwise_midrank_scores(
    scores: torch.Tensor,
    public_scales: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Return exact midranks inside each discrete public-noise tier.

    For a client :math:`i` in public tier :math:`G_g`, define

    .. math::

        r_i=\frac{\#\{j\in G_g:s_j<s_i\}
        +\tfrac12\#\{j\in G_g:s_j=s_i\}}{|G_g|}.

    The output lies strictly between zero and one, preserves the score order
    inside each tier (with equal scores receiving equal midranks), and has
    exact tier mean one half.  Hence its empirical covariance with the tier
    scale is zero on the complete cohort, regardless of tier sizes.  Replacing
    ``b_g`` scores in one tier can move an unchanged honest normalised rank by
    at most ``b_g / |G_g|``.

    This is a deterministic post-processing rule for a small discrete set of
    public noise levels.  It guarantees tier-mean orthogonality, not
    statistical independence or Byzantine identification.
    """

    if scores.ndim != 1 or public_scales.shape != scores.shape:
        raise ValueError("scores and public_scales must be aligned vectors")
    if scores.numel() < 1:
        raise ValueError("at least one score is required")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(public_scales).all()
    ):
        raise ValueError("scores and public scales must be finite")
    if bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ValueError("scores must lie in [0,1]")
    if bool((public_scales <= 0).any()):
        raise ValueError("public scales must be strictly positive")

    ranks = torch.empty_like(scores)
    tier_mean_errors = []
    unique_scales = torch.unique(public_scales, sorted=True)
    for scale in unique_scales:
        mask = public_scales == scale
        values = scores[mask]
        less = (values[:, None] > values[None, :]).sum(dim=1).to(scores.dtype)
        equal = (values[:, None] == values[None, :]).sum(dim=1).to(scores.dtype)
        tier_ranks = (less + 0.5 * equal) / float(values.numel())
        ranks[mask] = tier_ranks
        tier_mean_errors.append(abs(float(tier_ranks.mean().item()) - 0.5))

    centered_ranks = ranks - ranks.mean()
    centered_scales = public_scales - public_scales.mean()
    covariance = float((centered_ranks @ centered_scales).item()) / float(
        scores.numel()
    )
    diagnostics: dict[str, float | bool] = {
        "noise_score_tier_midrank_applied": True,
        "noise_score_tier_count": float(unique_scales.numel()),
        "noise_score_tier_mean_max_error": float(max(tier_mean_errors)),
        "noise_score_tier_midrank_scale_covariance": covariance,
        "noise_score_tier_midrank_min": float(ranks.min().item()),
        "noise_score_tier_midrank_max": float(ranks.max().item()),
    }
    return ranks, diagnostics


def theil_sen_orthogonalize_scores_against_public_scale(
    scores: torch.Tensor,
    public_scales: torch.Tensor,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Remove a robust linear public-noise trend from bounded scores.

    The ordinary projection used in generation 4 estimates its slope with a
    mean covariance and can therefore be moved by Byzantine scores.  This
    function instead uses the median of all pairwise slopes with distinct
    public scales (the Theil--Sen slope):

    .. math::

        \widehat\beta_{\rm TS}=\operatorname{median}_{a_i\ne a_j}
        \frac{s_i-s_j}{a_i-a_j}.

    It then subtracts the fitted public-scale component and rescales between
    public winsorisation quantiles.  The transform preserves the ordering of
    clients that share the same public noise scale.  It is more resistant to
    contaminated scores than least squares, but no universal Byzantine
    guarantee is claimed for an arbitrary allocation of attackers.
    """

    if scores.ndim != 1 or public_scales.shape != scores.shape:
        raise ValueError("scores and public_scales must be aligned vectors")
    if scores.numel() < 2:
        raise ValueError("at least two scores are required")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(public_scales).all()
    ):
        raise ValueError("scores and public scales must be finite")
    if bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ValueError("scores must lie in [0,1]")
    if bool((public_scales <= 0).any()):
        raise ValueError("public scales must be strictly positive")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    scale_delta = public_scales[:, None] - public_scales[None, :]
    score_delta = scores[:, None] - scores[None, :]
    upper_triangle = torch.triu(
        torch.ones_like(scale_delta, dtype=torch.bool), diagonal=1
    )
    valid = upper_triangle & (scale_delta.abs() > float(ridge))
    slopes = score_delta[valid] / scale_delta[valid]
    if slopes.numel() == 0:
        return scores.clone(), {
            "noise_score_robust_orthogonalization_applied": False,
            "noise_score_robust_projection_slope": 0.0,
            "noise_score_robust_projection_num_slopes": 0.0,
        }

    slope = slopes.median()
    scale_centre = public_scales.median()
    residual = scores - slope * (public_scales - scale_centre)
    lower = torch.quantile(residual, float(lower_quantile))
    upper = torch.quantile(residual, float(upper_quantile))
    span = upper - lower
    if float(span) <= float(ridge):
        adjusted = torch.zeros_like(scores)
    else:
        adjusted = ((residual - lower) / span).clamp(min=0.0, max=1.0)
    diagnostics: dict[str, float | bool] = {
        "noise_score_robust_orthogonalization_applied": True,
        "noise_score_robust_projection_slope": float(slope.item()),
        "noise_score_robust_projection_num_slopes": float(slopes.numel()),
        "noise_score_robust_rescale_lower_quantile": float(lower_quantile),
        "noise_score_robust_rescale_upper_quantile": float(upper_quantile),
        "noise_score_robust_residual_span": float(span.item()),
    }
    return adjusted, diagnostics


def tierwise_robust_standardize_scores(
    scores: torch.Tensor,
    public_scales: torch.Tensor,
    *,
    z_clip: float = 3.0,
    ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Standardise scores robustly inside each discrete public-noise tier.

    Within every exactly matching public scale, the median and median absolute
    deviation (MAD) define a robust location and scale.  The resulting robust
    z-score is clipped at the public value ``z_clip`` and mapped to ``[0,1]``.
    This preserves within-tier order while making the centre of every tier
    equal to one half.  It is an ablation for experiments with a small public
    set of noise tiers; it is not intended for continuously varying scales.
    """

    if scores.ndim != 1 or public_scales.shape != scores.shape:
        raise ValueError("scores and public_scales must be aligned vectors")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(public_scales).all()
    ):
        raise ValueError("scores and public scales must be finite")
    if bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ValueError("scores must lie in [0,1]")
    if bool((public_scales <= 0).any()):
        raise ValueError("public scales must be strictly positive")
    if z_clip <= 0.0:
        raise ValueError("z_clip must be positive")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    adjusted = torch.empty_like(scores)
    flat_tiers = 0
    tier_scales = []
    unique_scales = torch.unique(public_scales, sorted=True)
    for scale in unique_scales:
        mask = public_scales == scale
        values = scores[mask]
        centre = values.median()
        mad = (values - centre).abs().median()
        robust_scale = 1.4826 * mad
        if float(robust_scale) <= float(ridge):
            lower = torch.quantile(values, 0.25)
            upper = torch.quantile(values, 0.75)
            robust_scale = (upper - lower) / 1.349
        if float(robust_scale) <= float(ridge):
            adjusted[mask] = 0.5
            flat_tiers += 1
            tier_scales.append(0.0)
            continue
        z = ((values - centre) / robust_scale).clamp(
            min=-float(z_clip), max=float(z_clip)
        )
        adjusted[mask] = 0.5 + z / (2.0 * float(z_clip))
        tier_scales.append(float(robust_scale.item()))

    diagnostics: dict[str, float | bool] = {
        "noise_score_tier_robust_standardization_applied": True,
        "noise_score_tier_count": float(unique_scales.numel()),
        "noise_score_tier_flat_count": float(flat_tiers),
        "noise_score_tier_z_clip": float(z_clip),
        "noise_score_tier_robust_scale_min": float(min(tier_scales)),
        "noise_score_tier_robust_scale_max": float(max(tier_scales)),
    }
    return adjusted.clamp(0.0, 1.0), diagnostics


def orthogonalize_scores_against_public_scale(
    scores: torch.Tensor,
    public_scales: torch.Tensor,
    *,
    ridge: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    r"""Remove the linear component of a score explained by public DP scale.

    Let ``a_i`` denote the public relative noise multiplier.  We form

    .. math::

        \widetilde s_i=s_i-
        \frac{\langle s-\bar s,a-\bar a\rangle}
             {\lVert a-\bar a\rVert_2^2}(a_i-\bar a),

    then apply an affine min--max map to ``[0,1]``.  In a non-degenerate
    cohort this makes the empirical Pearson covariance with ``a`` exactly
    zero before floating-point error.  Because ``a`` is public and the input
    scores are computed from already locally-private uploads, this operation
    is DP post-processing.

    This removes only *linear* dependence and can be affected by Byzantine
    scores.  It is therefore paired with a separate tier-mean diagnostic and
    robust trust gates rather than treated as a privacy or robustness proof.
    """

    if scores.ndim != 1 or public_scales.shape != scores.shape:
        raise ValueError("scores and public_scales must be aligned vectors")
    if not bool(torch.isfinite(scores).all()) or not bool(
        torch.isfinite(public_scales).all()
    ):
        raise ValueError("scores and public scales must be finite")
    if bool((scores < 0).any()) or bool((scores > 1).any()):
        raise ValueError("scores must lie in [0,1]")
    if bool((public_scales <= 0).any()):
        raise ValueError("public scales must be strictly positive")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    centered_scale = public_scales - public_scales.mean()
    denominator = centered_scale.square().sum()
    if float(denominator) <= float(ridge):
        return scores.clone(), {
            "noise_score_scale_orthogonalization_applied": False,
            "noise_score_scale_projection_coefficient": 0.0,
        }
    centered_score = scores - scores.mean()
    coefficient = (centered_score @ centered_scale) / denominator
    residual = scores - coefficient * centered_scale
    lower = residual.min()
    span = residual.max() - lower
    if float(span) <= float(ridge):
        result = torch.zeros_like(scores)
    else:
        result = (residual - lower) / span
    diagnostics: dict[str, float | bool] = {
        "noise_score_scale_orthogonalization_applied": True,
        "noise_score_scale_projection_coefficient": float(coefficient.item()),
        "noise_score_orthogonalized_span_before_rescaling": float(span.item()),
    }
    return result.clamp(0.0, 1.0), diagnostics


def standardize_distances(
    distances: torch.Tensor,
    public_scales: torch.Tensor | None,
    *,
    mode: str,
    reference_variance_factor: float | None = None,
    variance_ridge: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str | bool | None]]:
    """Standardise FAR residual norms using public DP-noise information.

    ``reference_variance_factor`` multiplies the cohort mean of ``a_i^2``.
    Its default is ``1/n``, corresponding to a mean-like reference.  This is
    a declared modelling proxy for centered clipping, not an exact covariance
    identity.  Mean-normalising the resulting divisors preserves the distance
    scale and makes the homogeneous-noise case an exact negative control.
    """

    resolved_mode = str(mode).lower()
    if resolved_mode not in NOISE_SCORE_MODES:
        raise ValueError(
            "noise_score_standardization must be one of "
            f"{sorted(NOISE_SCORE_MODES)}, got {resolved_mode!r}"
        )
    if distances.ndim != 1 or distances.numel() < 1:
        raise ValueError("distances must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
        raise ValueError("distances must be finite and non-negative")

    if resolved_mode in DIRECT_SCORE_MODES | ROBUST_DIRECT_SCORE_MODES:
        raise ValueError(
            f"{resolved_mode} produces bounded scores directly; call its "
            "dedicated scoring function instead of standardize_distances"
        )
    if resolved_mode == "none":
        divisors = torch.ones_like(distances)
        scales = None
        factor = None
    else:
        if public_scales is None or public_scales.shape != distances.shape:
            raise ValueError(
                "Noise-aware standardisation requires one aligned public scale "
                "per client"
            )
        scales = public_scales.to(device=distances.device, dtype=distances.dtype)
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError("Public DP noise scales must be finite and positive")
        if variance_ridge < 0:
            raise ValueError("noise_score_variance_ridge must be non-negative")
        if resolved_mode == "client_noise_scale":
            raw_divisors = scales
            factor = 0.0
        else:
            n = distances.numel()
            factor = (
                1.0 / n
                if reference_variance_factor is None
                else float(reference_variance_factor)
            )
            if factor < 0:
                raise ValueError(
                    "noise_score_reference_variance_factor must be non-negative"
                )
            reference_variance = factor * scales.square().mean()
            raw_divisors = torch.sqrt(
                scales.square() + reference_variance + float(variance_ridge)
            )
        divisors = raw_divisors / raw_divisors.mean().clamp_min(1e-15)

    standardized = distances / divisors.clamp_min(1e-15)
    homogeneous = (
        True
        if scales is None
        else bool(torch.allclose(scales, scales[:1], rtol=0.0, atol=1e-12))
    )
    diagnostics: dict[str, float | str | bool | None] = {
        "noise_score_standardization": resolved_mode,
        "noise_score_uses_public_parameters_only": True,
        "noise_score_covariance_is_proxy": bool(
            resolved_mode == "isotropic_dp_covariance_proxy"
        ),
        "noise_score_reference_variance_factor": factor,
        "noise_score_public_scale_homogeneous": homogeneous,
        "noise_score_divisor_min": float(divisors.min().item()),
        "noise_score_divisor_mean": float(divisors.mean().item()),
        "noise_score_divisor_max": float(divisors.max().item()),
        "noise_score_raw_distance_mean": float(distances.mean().item()),
        "noise_score_standardized_distance_mean": float(standardized.mean().item()),
    }
    if scales is not None:
        diagnostics.update(
            {
                "noise_score_public_scale_min": float(scales.min().item()),
                "noise_score_public_scale_mean": float(scales.mean().item()),
                "noise_score_public_scale_max": float(scales.max().item()),
            }
        )
    return standardized, divisors, diagnostics
