#!/usr/bin/env python3
"""Development-only mechanistic screen for the K6 TP-EIV predictor.

The screen compares three predictors built from the *same* newer rolling view:

``radial_y``
    ``G * Y / max(||Y||, r_min)``.  This is the positive K6 radial prescreen
    control and deliberately contains no noise correction.

``k6_tp_eiv``
    The same radial direction multiplied by a temporal confidence obtained
    from two disjoint, strictly-past rolling views.  Known DP covariance is
    propagated through both public clipping maps with independent ``d x d``
    blocks; no covariance of shape ``(L*n*d, L*n*d)`` is materialised.

``k6_uncorrected``
    The same temporal-confidence rule applied to raw alignment and energies.
    This paired ablation isolates the effect of the EIV moment correction.

The gate is computed once from rounds 1--15 and frozen before either view is
formed.  The older view uses rounds 16--19, the newer view uses rounds 20--23,
and the deployment/evaluation round is 24.  Thus both predictors are
measurable with respect to the transcript available before round 24.

This is a synthetic mechanism check, not an end-to-end FL result and not a
publication confirmation.  ``--run`` is fail-closed MPS-only and requires
``PYTORCH_ENABLE_MPS_FALLBACK=0``.  ``--dry-run`` and ``--validate`` perform
no scientific tensor computation and are suitable for CI on non-MPS hosts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k6_tp_eiv import (  # noqa: E402
    radial_confidence_predictor,
    temporal_eiv_confidence,
    temporal_eiv_corrected_moments,
)
from robustness.aggregators import clip_l2  # noqa: E402

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k6_tp_eiv_microbench_mps_v1"
DEFAULT_OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
ALGORITHM_PATH = ROOT / "algorithms/gaussian_aware_reference_k6_tp_eiv.py"
TEST_PATH = ROOT / "tests/test_run_gaussian_aware_reference_g0g_k6_tp_eiv_microbench.py"

CALIBRATION_SEEDS = tuple(202709110101 + 16 * index for index in range(12))
NULL_VALIDATION_SEEDS = tuple(202709115107 + 16 * index for index in range(12))
EVALUATION_SEEDS = tuple(202709120103 + 16 * index for index in range(12))
THREATS = ("none", "bitflip_x10", "model_replacement")
NOISE_REGIMES: dict[str, tuple[float, ...]] = {
    "homogeneous": (0.030,),
    "heteroscedastic": (0.012, 0.022, 0.040, 0.070),
}
STUDENT_T_975_DF11 = 2.200985160091638

PROTOCOL: dict[str, Any] = {
    "campaign_id": CAMPAIGN_ID,
    "scope": "M0_development_only_synthetic_mechanistic_screen",
    "distinct_from": (
        "P0_n25_full_development_protocol; M0 uses n=12,d=8 and cannot promote P0"
    ),
    "claims_excluded": [
        "end_to_end_federated_learning_utility",
        "classification_accuracy_or_fairness",
        "universal_byzantine_robustness",
        "publication_confirmation",
        "exact_post_clipping_gaussian_moments_outside_local_delta_method",
    ],
    "cohort": {
        "num_clients": 12,
        "num_byzantine": 2,
        "dimension": 8,
    },
    "timeline": {
        "gate_source_rounds": list(range(1, 16)),
        "gate_snapshot_round": 16,
        "older_view_rounds": [16, 17, 18, 19],
        "newer_view_rounds": [20, 21, 22, 23],
        "deployment_round": 24,
        "window_length": 4,
        "attack_onset_round": 20,
    },
    "geometry": {
        "server_clip_norm": 0.75,
        "residual_influence_cap": 0.13,
        "minimum_accepted_mass": 6.0,
        "minimum_direction_norm": 1.0e-6,
        "clip_boundary_margin_min": 1.0e-6,
    },
    "clean_process": {
        "mean_norm": 0.085,
        "client_heterogeneity_std": 0.010,
        "round_sampling_std": 0.006,
        "angular_drift_per_round": 0.012,
    },
    "privacy_noise": {
        "regimes": {key: list(value) for key, value in NOISE_REGIMES.items()},
        "known_isotropic_covariance_per_client": True,
        "attacker_covariance_model": "zero_after_arbitrary_replacement",
    },
    "temporal_gate": {
        "common_across_both_views": True,
        "computed_from_strict_past_only": True,
        "inner_mad_multiplier": 2.5,
        "outer_mad_multiplier": 4.5,
        "minimum_mad": 1.0e-6,
    },
    "eiv": {
        "cross_covariance": "exact_zero_from_disjoint_noise_streams",
        "covariance_propagation": "analytic_block_jacobians_after_both_clips",
        "dense_joint_covariance_materialized": False,
        "b_min": 1.0e-5,
        "b_min_common_across_all_cells": True,
        "b_min_status": "M0_public_simplification_not_a_P0_calibration",
        "tau_a_null_quantile": 0.975,
        "single_public_tau_a_across_noise_regimes": True,
        "separate_equal_quantile_tau_for_corrected_and_uncorrected": True,
    },
    "randomness": {
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "null_validation_seeds": list(NULL_VALIDATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "paired_candidates": True,
        "paired_threat_counterfactuals": True,
    },
    "screen_gates": {
        "status": "M0_exploratory_not_confirmatory",
        "independent_null_false_activation_rate_max": 0.10,
        "benign_relative_mse_loss_max": 0.10,
        "attacked_relative_mse_gain_mean_min": 0.05,
        "require_each_noise_regime_attacked_gain_positive": True,
    },
    "execution": {
        "required_device": "mps",
        "dtype": "float32",
        "pytorch_mps_fallback_required": "0",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _generator(device: torch.device, *parts: object) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(_stable_seed(*parts))


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value).clamp_min(1.0e-12)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    ordered = sorted(float(value) for value in values)
    location = probability * float(len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - float(lower)
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_protocol(protocol: Mapping[str, Any] = PROTOCOL) -> dict[str, Any]:
    timeline = protocol["timeline"]
    gate_rounds = tuple(int(value) for value in timeline["gate_source_rounds"])
    older = tuple(int(value) for value in timeline["older_view_rounds"])
    newer = tuple(int(value) for value in timeline["newer_view_rounds"])
    deployment = int(timeline["deployment_round"])
    snapshot = int(timeline["gate_snapshot_round"])
    length = int(timeline["window_length"])
    cohort = protocol["cohort"]
    geometry = protocol["geometry"]

    checks = {
        "calibration_null_validation_evaluation_seeds_pairwise_disjoint": not (
            (set(CALIBRATION_SEEDS) & set(NULL_VALIDATION_SEEDS))
            or (set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS))
            or (set(NULL_VALIDATION_SEEDS) & set(EVALUATION_SEEDS))
        ),
        "all_seeds_unique": len(
            set(CALIBRATION_SEEDS + NULL_VALIDATION_SEEDS + EVALUATION_SEEDS)
        )
        == len(CALIBRATION_SEEDS) + len(NULL_VALIDATION_SEEDS) + len(EVALUATION_SEEDS),
        "gate_sources_strictly_before_snapshot": max(gate_rounds) < snapshot,
        "gate_snapshot_before_both_windows": snapshot <= min(older + newer),
        "windows_have_registered_length": len(older) == len(newer) == length,
        "windows_disjoint": not (set(older) & set(newer)),
        "views_strictly_past": max(older + newer) < deployment,
        "attack_starts_after_older_view": int(timeline["attack_onset_round"])
        > max(older),
        "attack_starts_at_newer_view": int(timeline["attack_onset_round"])
        == min(newer),
        "valid_byzantine_fraction": 0
        < int(cohort["num_byzantine"])
        < int(cohort["num_clients"]) / 2,
        "positive_dimension": int(cohort["dimension"]) >= 2,
        "valid_mass_floor": 0.0
        < float(geometry["minimum_accepted_mass"])
        <= int(cohort["num_clients"]),
        "positive_caps": min(
            float(geometry["server_clip_norm"]),
            float(geometry["residual_influence_cap"]),
            float(geometry["minimum_direction_norm"]),
        )
        > 0.0,
        "required_threats_exact": THREATS
        == ("none", "bitflip_x10", "model_replacement"),
        "required_noise_regimes_exact": set(NOISE_REGIMES)
        == {"homogeneous", "heteroscedastic"},
        "one_public_threshold_per_candidate_across_noise_regimes": bool(
            protocol["eiv"]["single_public_tau_a_across_noise_regimes"]
        ),
        "candidate_thresholds_calibrated_separately_at_equal_quantile": bool(
            protocol["eiv"]["separate_equal_quantile_tau_for_corrected_and_uncorrected"]
        ),
        "b_min_is_common_public_M0_simplification": bool(
            protocol["eiv"]["b_min_common_across_all_cells"]
        )
        and float(protocol["eiv"]["b_min"]) > 0.0,
        "M0_is_distinct_from_P0": protocol["scope"]
        == "M0_development_only_synthetic_mechanistic_screen"
        and int(cohort["num_clients"]) == 12
        and int(cohort["dimension"]) == 8
        and "P0_n25" in str(protocol["distinct_from"]),
        "mps_only": protocol["execution"]
        == {
            "required_device": "mps",
            "dtype": "float32",
            "pytorch_mps_fallback_required": "0",
        },
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Invalid K6 microbench protocol: {failed}")
    return {
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": _canonical_hash(PROTOCOL),
        "checks": checks,
        "expected_calibration_contexts": len(CALIBRATION_SEEDS) * len(NOISE_REGIMES),
        "expected_null_validation_contexts": len(NULL_VALIDATION_SEEDS)
        * len(NOISE_REGIMES),
        "expected_evaluation_rows": len(EVALUATION_SEEDS)
        * len(NOISE_REGIMES)
        * len(THREATS),
    }


def _require_mps() -> tuple[torch.device, torch.dtype]:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError(
            "Set PYTORCH_ENABLE_MPS_FALLBACK=0; K6 refuses silent CPU fallback"
        )
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError("MPS is unavailable; K6 refuses CPU execution")
    device = torch.device("mps")
    probe = torch.arange(8, device=device, dtype=torch.float32)
    if probe.device.type != "mps" or probe.dtype != torch.float32:
        raise RuntimeError("MPS float32 runtime attestation failed")
    torch.mps.synchronize()
    return device, torch.float32


def _noise_std_by_client(regime: str, *, device: torch.device) -> torch.Tensor:
    if regime not in NOISE_REGIMES:
        raise ValueError(f"Unknown noise regime: {regime}")
    n = int(PROTOCOL["cohort"]["num_clients"])
    tiers = NOISE_REGIMES[regime]
    return torch.tensor(
        [tiers[index % len(tiers)] for index in range(n)],
        device=device,
        dtype=torch.float32,
    )


def _clip_with_jacobian(
    value: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project rows onto a ball and return exact piecewise Jacobian blocks."""

    if value.ndim != 2:
        raise ValueError("value must have shape (batch,d)")
    cap = float(radius)
    norms = torch.linalg.vector_norm(value, dim=1)
    inside = norms < cap
    safe_norms = norms.clamp_min(1.0e-12)
    unit = value / safe_norms[:, None]
    factors = torch.minimum(torch.ones_like(norms), cap / safe_norms)
    projected = value * factors[:, None]
    dimension = int(value.shape[1])
    identity = torch.eye(dimension, device=value.device, dtype=value.dtype)
    outside_jacobian = (cap / safe_norms)[:, None, None] * (
        identity[None, :, :] - unit[:, :, None] * unit[:, None, :]
    )
    jacobian = torch.where(
        inside[:, None, None], identity[None, :, :], outside_jacobian
    )
    boundary_margin = torch.abs(norms - cap)
    return projected, jacobian, norms, boundary_margin


def _clip_chain_with_jacobian(
    uploads: torch.Tensor, anchor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply server and residual clipping with exact local Jacobian blocks."""

    geometry = PROTOCOL["geometry"]
    server, jacobian_server, upload_norms, server_margins = _clip_with_jacobian(
        uploads, float(geometry["server_clip_norm"])
    )
    residual, jacobian_residual, residual_norms, residual_margins = _clip_with_jacobian(
        server - anchor[None, :],
        float(geometry["residual_influence_cap"]),
    )
    jacobian = jacobian_residual @ jacobian_server
    return (
        residual,
        jacobian,
        {
            "upload_norms": upload_norms,
            "server_clip_margins": server_margins,
            "server_clip_active": upload_norms > float(geometry["server_clip_norm"]),
            "pre_residual_norms": residual_norms,
            "residual_clip_margins": residual_margins,
            "residual_clip_active": residual_norms
            > float(geometry["residual_influence_cap"]),
        },
    )


def _clean_process(
    seed: int,
    *,
    device: torch.device,
    null_signal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean rounds 1--24 and the fixed honest-identity mask."""

    n = int(PROTOCOL["cohort"]["num_clients"])
    b = int(PROTOCOL["cohort"]["num_byzantine"])
    dimension = int(PROTOCOL["cohort"]["dimension"])
    clean_config = PROTOCOL["clean_process"]
    honest = torch.ones(n, device=device, dtype=torch.bool)
    honest[n - b :] = False
    if null_signal:
        return torch.zeros(24, n, dimension, device=device), honest

    direction = _unit(
        torch.randn(
            dimension,
            generator=_generator(device, CAMPAIGN_ID, "direction", seed),
            device=device,
            dtype=torch.float32,
        )
    )
    second = torch.randn(
        dimension,
        generator=_generator(device, CAMPAIGN_ID, "second", seed),
        device=device,
        dtype=torch.float32,
    )
    second = _unit(second - torch.dot(second, direction) * direction)
    offsets = torch.randn(
        n,
        dimension,
        generator=_generator(device, CAMPAIGN_ID, "offset", seed),
        device=device,
        dtype=torch.float32,
    ) * float(clean_config["client_heterogeneity_std"])
    offsets = offsets - offsets[honest].mean(dim=0, keepdim=True)
    rounds: list[torch.Tensor] = []
    for round_index in range(1, 25):
        angle = float(clean_config["angular_drift_per_round"]) * (round_index - 1)
        centre = float(clean_config["mean_norm"]) * (
            math.cos(angle) * direction + math.sin(angle) * second
        )
        jitter = torch.randn(
            n,
            dimension,
            generator=_generator(
                device, CAMPAIGN_ID, "clean-jitter", seed, round_index
            ),
            device=device,
            dtype=torch.float32,
        ) * float(clean_config["round_sampling_std"])
        rounds.append(centre[None, :] + offsets + jitter)
    return torch.stack(rounds), honest


def _private_uploads(
    clean: torch.Tensor,
    noise_std: torch.Tensor,
    *,
    seed: int,
    regime: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(
        clean.shape,
        generator=_generator(device, CAMPAIGN_ID, "dp-noise", seed, regime),
        device=device,
        dtype=torch.float32,
    )
    uploads = clean + noise * noise_std[None, :, None]
    variances = noise_std.square()[None, :, None].expand_as(uploads)
    return uploads, variances


def _apply_threat(
    uploads: torch.Tensor,
    clean: torch.Tensor,
    variances: torch.Tensor,
    honest: torch.Tensor,
    *,
    seed: int,
    threat: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace Byzantine messages only after the registered attack onset."""

    if threat not in THREATS:
        raise ValueError(f"Unknown threat: {threat}")
    result = uploads.clone()
    covariance = variances.clone()
    active = torch.zeros_like(honest)
    if threat == "none":
        return result, covariance, active
    active = ~honest
    start = int(PROTOCOL["timeline"]["attack_onset_round"]) - 1
    if threat == "bitflip_x10":
        result[start:, active] = -10.0 * clean[start:, active]
    else:
        dimension = int(PROTOCOL["cohort"]["dimension"])
        attack_direction = _unit(
            torch.randn(
                dimension,
                generator=_generator(
                    device, CAMPAIGN_ID, "replacement-direction", seed
                ),
                device=device,
                dtype=torch.float32,
            )
        )
        result[start:, active] = (
            10.0 * float(PROTOCOL["geometry"]["server_clip_norm"]) * attack_direction
        )
    covariance[start:, active] = 0.0
    return result, covariance, active


def _predictable_gate(
    clipped_prehistory: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build one client gate from rounds 1--15 and freeze it thereafter."""

    if clipped_prehistory.ndim != 3:
        raise ValueError("clipped_prehistory must have shape (round,n,d)")
    robust_centres = clipped_prehistory.median(dim=1).values
    deviations = torch.linalg.vector_norm(
        clipped_prehistory - robust_centres[:, None, :], dim=2
    )
    client_statistic = deviations.median(dim=0).values
    median = client_statistic.median()
    mad = (
        torch.abs(client_statistic - median)
        .median()
        .clamp_min(float(PROTOCOL["temporal_gate"]["minimum_mad"]))
    )
    inner = median + float(PROTOCOL["temporal_gate"]["inner_mad_multiplier"]) * mad
    outer = median + float(PROTOCOL["temporal_gate"]["outer_mad_multiplier"]) * mad
    gate = ((outer - client_statistic) / (outer - inner)).clamp(0.0, 1.0)
    return gate, {
        "source_round_min": 1,
        "source_round_max": 15,
        "snapshot_round": 16,
        "client_statistic": client_statistic.detach().cpu().tolist(),
        "median": float(median.item()),
        "mad": float(mad.item()),
        "inner": float(inner.item()),
        "outer": float(outer.item()),
        "gate": gate.detach().cpu().tolist(),
        "gate_hash": _tensor_hash(gate),
        "past_only": True,
        "common_across_views": True,
    }


def _view_and_covariance(
    uploads: torch.Tensor,
    coordinate_variances: torch.Tensor,
    gate: torch.Tensor,
    anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute one fixed-gate view and blockwise post-chain covariance."""

    if uploads.ndim != 3 or not uploads.is_floating_point():
        raise ValueError("uploads must be a floating tensor with shape (L,n,d)")
    if coordinate_variances.shape != uploads.shape or not bool(
        torch.isfinite(coordinate_variances).all()
    ):
        raise ValueError("coordinate_variances must be finite with shape (L,n,d)")
    if bool((coordinate_variances < 0.0).any()):
        raise ValueError("coordinate_variances must be non-negative")
    length, n, dimension = (int(value) for value in uploads.shape)
    if gate.shape != (n,) or anchor.shape != (dimension,):
        raise ValueError("gate and anchor shapes do not match uploads")
    if (
        not bool(torch.isfinite(gate).all())
        or bool((gate < 0.0).any())
        or bool((gate > 1.0).any())
    ):
        raise ValueError("gate must be finite and lie in [0,1]")
    flat_uploads = uploads.reshape(length * n, dimension)
    residuals, jacobians, clipping = _clip_chain_with_jacobian(flat_uploads, anchor)
    residuals = residuals.reshape(length, n, dimension)
    jacobians = jacobians.reshape(length, n, dimension, dimension)
    total_gate_mass = gate.sum()
    denominator = length * torch.maximum(
        total_gate_mass,
        torch.as_tensor(
            float(PROTOCOL["geometry"]["minimum_accepted_mass"]),
            device=uploads.device,
            dtype=uploads.dtype,
        ),
    )
    coefficients = gate / denominator
    raw_view = torch.sum(coefficients[None, :, None] * residuals, dim=(0, 1))
    cap = float(PROTOCOL["geometry"]["residual_influence_cap"])
    view = clip_l2(raw_view, cap)
    view_margin = cap - torch.linalg.vector_norm(raw_view)
    if float(view_margin.item()) <= float(
        PROTOCOL["geometry"]["clip_boundary_margin_min"]
    ):
        raise RuntimeError(
            "Pooled-view projection is active or too close to its boundary; "
            "the registered identity-Jacobian certificate is invalid"
        )

    covariance = torch.zeros(
        dimension, dimension, device=uploads.device, dtype=uploads.dtype
    )
    block_count = 0
    variances = coordinate_variances.reshape(length, n, dimension)
    for round_offset in range(length):
        for client in range(n):
            jacobian = jacobians[round_offset, client]
            weight = coefficients[client]
            # Each summand is explicitly B B^T, hence PSD by construction.
            # Keeping the dxd factor local avoids ever forming the dense
            # covariance of all L*n*d primitive Gaussian coordinates.
            factor = (
                weight
                * jacobian
                * torch.sqrt(variances[round_offset, client]).clamp_min(0.0)[None, :]
            )
            covariance = covariance + factor @ factor.T
            block_count += 1
    covariance = 0.5 * (covariance + covariance.T)
    tolerance = (
        256.0
        * torch.finfo(uploads.dtype).eps
        * dimension
        * max(1.0, float(covariance.abs().max().item()))
    )
    identity = torch.eye(dimension, device=uploads.device, dtype=uploads.dtype)
    _, cholesky_info = torch.linalg.cholesky_ex(
        covariance + tolerance * identity, check_errors=False
    )
    if int(cholesky_info.item()) != 0:
        raise RuntimeError("Structured post-chain covariance failed its PSD audit")

    minimum_margin = min(
        float(clipping["server_clip_margins"].min().item()),
        float(clipping["residual_clip_margins"].min().item()),
        float(view_margin.item()),
    )
    if minimum_margin <= float(PROTOCOL["geometry"]["clip_boundary_margin_min"]):
        raise RuntimeError(
            "A clipping point is too close to a non-differentiable boundary"
        )
    diagnostics = {
        "history_length": length,
        "num_clients": n,
        "dimension": dimension,
        "gate_mass": float(total_gate_mass.item()),
        "deployed_denominator": float(denominator.item()),
        "denominator_floor_active": bool(
            float(total_gate_mass.item())
            < float(PROTOCOL["geometry"]["minimum_accepted_mass"])
        ),
        "view_norm": float(torch.linalg.vector_norm(view).item()),
        "view_projection_margin": float(view_margin.item()),
        "server_clip_active_fraction": float(
            clipping["server_clip_active"].float().mean().item()
        ),
        "residual_clip_active_fraction": float(
            clipping["residual_clip_active"].float().mean().item()
        ),
        "minimum_piecewise_differentiability_margin": minimum_margin,
        "covariance_trace": float(torch.trace(covariance).item()),
        "covariance_min_diagonal": float(torch.diagonal(covariance).min().item()),
        "covariance_max_diagonal": float(torch.diagonal(covariance).max().item()),
        "covariance_psd": True,
        "covariance_psd_certificate": "sum_of_B_B_transpose_blocks",
        "covariance_psd_cholesky_with_public_jitter": True,
        "covariance_psd_cholesky_info": int(cholesky_info.item()),
        "covariance_psd_cholesky_jitter": tolerance,
        "covariance_block_count": block_count,
        "dense_joint_covariance_materialized": False,
        "final_view_projection_inactive": True,
        "delta_method_caveat": (
            "locally exact linear propagation within the observed clipping "
            "region; not an exact global moment identity across clip boundaries"
        ),
    }
    return view, covariance, diagnostics


def _context(
    seed: int,
    regime: str,
    threat: str,
    *,
    device: torch.device,
    null_signal: bool,
    tau_a_corrected: float,
    tau_a_uncorrected: float,
) -> dict[str, Any]:
    clean, honest = _clean_process(seed, device=device, null_signal=null_signal)
    noise_std = _noise_std_by_client(regime, device=device)
    uploads, variances = _private_uploads(
        clean, noise_std, seed=seed, regime=regime, device=device
    )
    uploads, variances, active_byzantine = _apply_threat(
        uploads,
        clean,
        variances,
        honest,
        seed=seed,
        threat=threat,
        device=device,
    )
    dimension = int(PROTOCOL["cohort"]["dimension"])
    anchor = torch.zeros(dimension, device=device, dtype=torch.float32)
    prehistory_rounds = PROTOCOL["timeline"]["gate_source_rounds"]
    prehistory_uploads = uploads[min(prehistory_rounds) - 1 : max(prehistory_rounds)]
    clipped_prehistory, _, _ = _clip_chain_with_jacobian(
        prehistory_uploads.reshape(-1, dimension), anchor
    )
    clipped_prehistory = clipped_prehistory.reshape(
        len(prehistory_rounds), int(PROTOCOL["cohort"]["num_clients"]), dimension
    )
    gate, gate_diagnostics = _predictable_gate(clipped_prehistory)

    older_rounds = PROTOCOL["timeline"]["older_view_rounds"]
    newer_rounds = PROTOCOL["timeline"]["newer_view_rounds"]
    older, covariance_older, older_diagnostics = _view_and_covariance(
        uploads[min(older_rounds) - 1 : max(older_rounds)],
        variances[min(older_rounds) - 1 : max(older_rounds)],
        gate,
        anchor,
    )
    newer, covariance_newer, newer_diagnostics = _view_and_covariance(
        uploads[min(newer_rounds) - 1 : max(newer_rounds)],
        variances[min(newer_rounds) - 1 : max(newer_rounds)],
        gate,
        anchor,
    )
    cross_covariance = torch.zeros_like(covariance_older)
    alignment, energy_x, energy_y, moment_diagnostics = temporal_eiv_corrected_moments(
        older,
        newer,
        covariance_older=covariance_older,
        covariance_newer=covariance_newer,
        cross_covariance_older_newer=cross_covariance,
        return_diagnostics=True,
    )
    confidence, confidence_diagnostics = temporal_eiv_confidence(
        alignment,
        energy_x,
        energy_y,
        b_min=float(PROTOCOL["eiv"]["b_min"]),
        tau_a=float(tau_a_corrected),
        return_diagnostics=True,
    )
    raw_alignment = torch.dot(older, newer)
    raw_energy_x = torch.dot(older, older)
    raw_energy_y = torch.dot(newer, newer)
    uncorrected_confidence, uncorrected_confidence_diagnostics = (
        temporal_eiv_confidence(
            raw_alignment,
            raw_energy_x,
            raw_energy_y,
            b_min=float(PROTOCOL["eiv"]["b_min"]),
            tau_a=float(tau_a_uncorrected),
            return_diagnostics=True,
        )
    )
    radial, radial_diagnostics = radial_confidence_predictor(
        newer,
        1.0,
        influence_cap=float(PROTOCOL["geometry"]["residual_influence_cap"]),
        minimum_direction_norm=float(PROTOCOL["geometry"]["minimum_direction_norm"]),
        return_diagnostics=True,
    )
    k6, k6_diagnostics = radial_confidence_predictor(
        newer,
        confidence,
        influence_cap=float(PROTOCOL["geometry"]["residual_influence_cap"]),
        minimum_direction_norm=float(PROTOCOL["geometry"]["minimum_direction_norm"]),
        return_diagnostics=True,
    )
    k6_uncorrected, k6_uncorrected_diagnostics = radial_confidence_predictor(
        newer,
        uncorrected_confidence,
        influence_cap=float(PROTOCOL["geometry"]["residual_influence_cap"]),
        minimum_direction_norm=float(PROTOCOL["geometry"]["minimum_direction_norm"]),
        return_diagnostics=True,
    )
    target_round = int(PROTOCOL["timeline"]["deployment_round"]) - 1
    target = clip_l2(
        clean[target_round, honest].mean(dim=0),
        float(PROTOCOL["geometry"]["residual_influence_cap"]),
    )
    radial_mse = float(torch.mean((radial - target).square()).item())
    k6_mse = float(torch.mean((k6 - target).square()).item())
    k6_uncorrected_mse = float(torch.mean((k6_uncorrected - target).square()).item())
    relative_gain = (radial_mse - k6_mse) / max(radial_mse, 1.0e-15)
    return {
        "seed": int(seed),
        "noise_regime": regime,
        "threat": threat,
        "null_signal": bool(null_signal),
        # The pairing identifier deliberately excludes the threat: all three
        # threat counterfactuals reuse the same clean path and Gaussian draws.
        "pairing_id": f"{seed}|{regime}",
        "cell_id": f"{seed}|{regime}|{threat}",
        "device": str(device),
        "dtype": str(older.dtype),
        "gate_hash": gate_diagnostics["gate_hash"],
        "gate_source_round_max": gate_diagnostics["source_round_max"],
        "gate_snapshot_round": gate_diagnostics["snapshot_round"],
        "gate_common_across_views": gate_diagnostics["common_across_views"],
        "older_rounds": json.dumps(older_rounds),
        "newer_rounds": json.dumps(newer_rounds),
        "views_disjoint": not bool(set(older_rounds) & set(newer_rounds)),
        "deployment_round": int(PROTOCOL["timeline"]["deployment_round"]),
        "attack_onset_round": int(PROTOCOL["timeline"]["attack_onset_round"]),
        "active_byzantine_count": int(active_byzantine.sum().item()),
        "gate_mean": float(gate.mean().item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "older_norm": float(torch.linalg.vector_norm(older).item()),
        "newer_norm": float(torch.linalg.vector_norm(newer).item()),
        "target_norm": float(torch.linalg.vector_norm(target).item()),
        "corrected_alignment": float(alignment.item()),
        "corrected_energy_x": float(energy_x.item()),
        "corrected_energy_y": float(energy_y.item()),
        "raw_alignment": float(raw_alignment.item()),
        "raw_energy_x": float(raw_energy_x.item()),
        "raw_energy_y": float(raw_energy_y.item()),
        "tau_a": float(tau_a_corrected),
        "tau_a_corrected": float(tau_a_corrected),
        "tau_a_uncorrected": float(tau_a_uncorrected),
        "confidence": float(confidence.item()),
        "uncorrected_confidence": float(uncorrected_confidence.item()),
        "radial_predictor_norm": radial_diagnostics["predictor_norm"],
        "k6_predictor_norm": k6_diagnostics["predictor_norm"],
        "k6_uncorrected_predictor_norm": k6_uncorrected_diagnostics["predictor_norm"],
        "radial_mse": radial_mse,
        "k6_mse": k6_mse,
        "k6_uncorrected_mse": k6_uncorrected_mse,
        "k6_relative_mse_gain_vs_radial": relative_gain,
        "k6_corrected_relative_mse_gain_vs_uncorrected": (
            (k6_uncorrected_mse - k6_mse) / max(k6_uncorrected_mse, 1.0e-15)
        ),
        "older_covariance_trace": older_diagnostics["covariance_trace"],
        "newer_covariance_trace": newer_diagnostics["covariance_trace"],
        "older_covariance_min_diagonal": older_diagnostics["covariance_min_diagonal"],
        "newer_covariance_min_diagonal": newer_diagnostics["covariance_min_diagonal"],
        "covariance_psd": bool(
            older_diagnostics["covariance_psd"] and newer_diagnostics["covariance_psd"]
        ),
        "covariance_block_count": int(
            older_diagnostics["covariance_block_count"]
            + newer_diagnostics["covariance_block_count"]
        ),
        "dense_joint_covariance_materialized": False,
        "cross_covariance_trace": 0.0,
        "cross_covariance_zero_by_construction": True,
        "minimum_clip_margin": min(
            older_diagnostics["minimum_piecewise_differentiability_margin"],
            newer_diagnostics["minimum_piecewise_differentiability_margin"],
        ),
        "server_clip_active_fraction": statistics.fmean(
            [
                older_diagnostics["server_clip_active_fraction"],
                newer_diagnostics["server_clip_active_fraction"],
            ]
        ),
        "residual_clip_active_fraction": statistics.fmean(
            [
                older_diagnostics["residual_clip_active_fraction"],
                newer_diagnostics["residual_clip_active_fraction"],
            ]
        ),
        "moment_formula": moment_diagnostics["formula_alignment"],
        "confidence_formula": confidence_diagnostics["formula"],
        "uncorrected_confidence_formula": uncorrected_confidence_diagnostics["formula"],
        "predictor_formula": k6_diagnostics["formula"],
        "current_round_input_used": False,
        "target_used_for_predictor": False,
    }


def _calibrate_tau(
    device: torch.device,
) -> tuple[float, float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    alignments: list[float] = []
    for seed in CALIBRATION_SEEDS:
        for regime in NOISE_REGIMES:
            row = _context(
                seed,
                regime,
                "none",
                device=device,
                null_signal=True,
                tau_a_corrected=0.0,
                tau_a_uncorrected=0.0,
            )
            rows.append(row)
            alignments.append(float(row["corrected_alignment"]))
    tau_corrected = max(
        0.0,
        _quantile(alignments, float(PROTOCOL["eiv"]["tau_a_null_quantile"])),
    )
    tau_uncorrected = max(
        0.0,
        _quantile(
            [float(row["raw_alignment"]) for row in rows],
            float(PROTOCOL["eiv"]["tau_a_null_quantile"]),
        ),
    )
    return tau_corrected, tau_uncorrected, rows


def _validate_tau_under_independent_null(
    device: torch.device, tau_a_corrected: float, tau_a_uncorrected: float
) -> list[dict[str, Any]]:
    """Evaluate the frozen threshold on null seeds excluded from calibration."""

    return [
        _context(
            seed,
            regime,
            "none",
            device=device,
            null_signal=True,
            tau_a_corrected=tau_a_corrected,
            tau_a_uncorrected=tau_a_uncorrected,
        )
        for seed in NULL_VALIDATION_SEEDS
        for regime in NOISE_REGIMES
    ]


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["noise_regime"]), str(row["threat"]))].append(row)
    result: list[dict[str, Any]] = []
    for (regime, threat), group in sorted(grouped.items()):
        result.append(
            {
                "noise_regime": regime,
                "threat": threat,
                "seeds": len(group),
                "radial_mse_mean": statistics.fmean(
                    float(row["radial_mse"]) for row in group
                ),
                "k6_mse_mean": statistics.fmean(float(row["k6_mse"]) for row in group),
                "k6_uncorrected_mse_mean": statistics.fmean(
                    float(row["k6_uncorrected_mse"]) for row in group
                ),
                "k6_relative_mse_gain_mean": statistics.fmean(
                    float(row["k6_relative_mse_gain_vs_radial"]) for row in group
                ),
                "confidence_mean": statistics.fmean(
                    float(row["confidence"]) for row in group
                ),
                "uncorrected_confidence_mean": statistics.fmean(
                    float(row["uncorrected_confidence"]) for row in group
                ),
                "k6_corrected_relative_mse_gain_vs_uncorrected_mean": (
                    statistics.fmean(
                        float(row["k6_corrected_relative_mse_gain_vs_uncorrected"])
                        for row in group
                    )
                ),
                "minimum_clip_margin": min(
                    float(row["minimum_clip_margin"]) for row in group
                ),
                "all_covariances_psd": all(
                    bool(row["covariance_psd"]) for row in group
                ),
            }
        )
    return result


def _ratio_gain(control: float, candidate: float) -> float:
    return (float(control) - float(candidate)) / max(float(control), 1.0e-15)


def _seed_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse child cells before inference so seeds are the only replicates."""

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["noise_regime"]))].append(row)
    result: list[dict[str, Any]] = []
    for (seed, regime), group in sorted(grouped.items()):
        attacked = [row for row in group if row["threat"] != "none"]
        benign = [row for row in group if row["threat"] == "none"]
        if len(attacked) != 2 or len(benign) != 1:
            raise RuntimeError("Each seed/noise unit must contain three threat cells")
        radial_attacked = sum(float(row["radial_mse"]) for row in attacked)
        corrected_attacked = sum(float(row["k6_mse"]) for row in attacked)
        uncorrected_attacked = sum(float(row["k6_uncorrected_mse"]) for row in attacked)
        corrected_all = sum(float(row["k6_mse"]) for row in group)
        uncorrected_all = sum(float(row["k6_uncorrected_mse"]) for row in group)
        result.append(
            {
                "seed": seed,
                "noise_regime": regime,
                "attacked_cell_count": len(attacked),
                "benign_cell_count": len(benign),
                "radial_attacked_mse_mean": statistics.fmean(
                    float(row["radial_mse"]) for row in attacked
                ),
                "k6_corrected_attacked_mse_mean": statistics.fmean(
                    float(row["k6_mse"]) for row in attacked
                ),
                "k6_uncorrected_attacked_mse_mean": statistics.fmean(
                    float(row["k6_uncorrected_mse"]) for row in attacked
                ),
                "radial_benign_mse": float(benign[0]["radial_mse"]),
                "k6_corrected_benign_mse": float(benign[0]["k6_mse"]),
                "k6_uncorrected_benign_mse": float(benign[0]["k6_uncorrected_mse"]),
                "k6_corrected_confidence_all_cells_mean": statistics.fmean(
                    float(row["confidence"]) for row in group
                ),
                "k6_uncorrected_confidence_all_cells_mean": statistics.fmean(
                    float(row["uncorrected_confidence"]) for row in group
                ),
                "k6_corrected_gain_vs_radial_attacked": _ratio_gain(
                    radial_attacked, corrected_attacked
                ),
                "k6_corrected_gain_vs_uncorrected_all_cells": _ratio_gain(
                    uncorrected_all, corrected_all
                ),
                "k6_corrected_gain_vs_uncorrected_attacked": _ratio_gain(
                    uncorrected_attacked, corrected_attacked
                ),
                "k6_corrected_benign_loss_vs_radial": -float(
                    benign[0]["k6_relative_mse_gain_vs_radial"]
                ),
            }
        )
    return result


def _student_ci(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if len(numbers) != len(EVALUATION_SEEDS):
        raise ValueError("M0 Student intervals require exactly 12 outer seeds")
    mean = statistics.fmean(numbers)
    standard_deviation = statistics.stdev(numbers)
    half_width = STUDENT_T_975_DF11 * standard_deviation / math.sqrt(len(numbers))
    return {
        "n_independent_seeds": len(numbers),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "low": mean - half_width,
        "high": mean + half_width,
        "student_t_critical_975_df11": STUDENT_T_975_DF11,
    }


def _contrast_intervals(
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "k6_corrected_gain_vs_radial_attacked",
        "k6_corrected_gain_vs_uncorrected_all_cells",
        "k6_corrected_gain_vs_uncorrected_attacked",
        "k6_corrected_benign_loss_vs_radial",
    )
    by_regime = {
        regime: {
            metric: _student_ci(
                [
                    float(row[metric])
                    for row in seed_rows
                    if row["noise_regime"] == regime
                ]
            )
            for metric in metrics
        }
        for regime in NOISE_REGIMES
    }
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_seed[int(row["seed"])].append(row)
    pooled = {
        metric: _student_ci(
            [
                statistics.fmean(float(row[metric]) for row in by_seed[seed])
                for seed in sorted(by_seed)
            ]
        )
        for metric in metrics
    }
    return {
        "independent_unit": "outer_seed",
        "child_cells_are_not_pseudoreplicates": True,
        "interval": "two_sided_95_percent_student_t_df11",
        "by_noise_regime": by_regime,
        "pooled_equal_weight_noise_regimes_within_seed": pooled,
    }


def _decision(
    calibration_rows: Sequence[Mapping[str, Any]],
    null_validation_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gates = PROTOCOL["screen_gates"]
    tau_corrected = float(evaluation_rows[0]["tau_a_corrected"])
    tau_uncorrected = float(evaluation_rows[0]["tau_a_uncorrected"])
    calibration_false_activation = statistics.fmean(
        float(float(row["corrected_alignment"]) > tau_corrected)
        for row in calibration_rows
    )
    calibration_uncorrected_false_activation = statistics.fmean(
        float(float(row["raw_alignment"]) > tau_uncorrected) for row in calibration_rows
    )
    independent_corrected_false_activation = statistics.fmean(
        float(float(row["confidence"]) > 0.0) for row in null_validation_rows
    )
    independent_uncorrected_false_activation = statistics.fmean(
        float(float(row["uncorrected_confidence"]) > 0.0)
        for row in null_validation_rows
    )
    seed_rows = _seed_summary_rows(evaluation_rows)
    intervals = _contrast_intervals(seed_rows)
    benign_loss = statistics.fmean(
        float(row["k6_corrected_benign_loss_vs_radial"]) for row in seed_rows
    )
    attacked_gain = statistics.fmean(
        float(row["k6_corrected_gain_vs_radial_attacked"]) for row in seed_rows
    )
    per_regime_attacked = {
        regime: statistics.fmean(
            float(row["k6_corrected_gain_vs_radial_attacked"])
            for row in seed_rows
            if row["noise_regime"] == regime
        )
        for regime in NOISE_REGIMES
    }
    corrected_vs_uncorrected = {
        regime: statistics.fmean(
            float(row["k6_corrected_gain_vs_uncorrected_all_cells"])
            for row in seed_rows
            if row["noise_regime"] == regime
        )
        for regime in NOISE_REGIMES
    }
    validity = {
        "calibration_row_count_exact": len(calibration_rows)
        == len(CALIBRATION_SEEDS) * len(NOISE_REGIMES),
        "null_validation_row_count_exact": len(null_validation_rows)
        == len(NULL_VALIDATION_SEEDS) * len(NOISE_REGIMES),
        "row_count_exact": len(evaluation_rows)
        == len(EVALUATION_SEEDS) * len(NOISE_REGIMES) * len(THREATS),
        "calibration_all_mps_float32": all(
            row["device"] == "mps" and row["dtype"] == "torch.float32"
            for row in calibration_rows
        ),
        "null_validation_all_mps_float32": all(
            row["device"] == "mps" and row["dtype"] == "torch.float32"
            for row in null_validation_rows
        ),
        "all_mps_float32": all(
            row["device"] == "mps" and row["dtype"] == "torch.float32"
            for row in evaluation_rows
        ),
        "registered_seed_sets_exact": (
            {int(row["seed"]) for row in calibration_rows} == set(CALIBRATION_SEEDS)
            and {int(row["seed"]) for row in null_validation_rows}
            == set(NULL_VALIDATION_SEEDS)
            and {int(row["seed"]) for row in evaluation_rows} == set(EVALUATION_SEEDS)
        ),
        "candidate_thresholds_frozen_across_evaluation": (
            {float(row["tau_a_corrected"]) for row in evaluation_rows}
            == {tau_corrected}
            and {float(row["tau_a_uncorrected"]) for row in evaluation_rows}
            == {tau_uncorrected}
        ),
        "all_past_only": all(
            int(row["gate_source_round_max"])
            < min(json.loads(str(row["older_rounds"])))
            and not bool(row["current_round_input_used"])
            and not bool(row["target_used_for_predictor"])
            for row in evaluation_rows
        ),
        "all_windows_disjoint": all(
            bool(row["views_disjoint"]) for row in evaluation_rows
        ),
        "all_covariances_psd": all(
            bool(row["covariance_psd"]) for row in evaluation_rows
        ),
        "no_dense_covariance": all(
            not bool(row["dense_joint_covariance_materialized"])
            for row in evaluation_rows
        ),
        "all_clip_margins_positive": all(
            float(row["minimum_clip_margin"])
            > float(PROTOCOL["geometry"]["clip_boundary_margin_min"])
            for row in evaluation_rows
        ),
        "same_gate_hash_across_threat_counterfactuals": all(
            len(
                {
                    str(row["gate_hash"])
                    for row in evaluation_rows
                    if int(row["seed"]) == seed and row["noise_regime"] == regime
                }
            )
            == 1
            for seed in EVALUATION_SEEDS
            for regime in NOISE_REGIMES
        ),
    }
    scientific = {
        "independent_corrected_null_false_activation_controlled": (
            independent_corrected_false_activation
            <= float(gates["independent_null_false_activation_rate_max"])
        ),
        "independent_uncorrected_null_false_activation_controlled": (
            independent_uncorrected_false_activation
            <= float(gates["independent_null_false_activation_rate_max"])
        ),
        "benign_loss_controlled": benign_loss
        <= float(gates["benign_relative_mse_loss_max"]),
        "attacked_gain_mean": attacked_gain
        >= float(gates["attacked_relative_mse_gain_mean_min"]),
        "each_noise_regime_attacked_gain_positive": all(
            value > 0.0 for value in per_regime_attacked.values()
        ),
    }
    passed = all(validity.values()) and all(scientific.values())
    return {
        "validity_checks": validity,
        "scientific_checks": scientific,
        "all_checks_pass": passed,
        "decision": (
            "advance_to_locked_k6_development"
            if passed
            else "do_not_promote_k6_from_microbench"
        ),
        "calibration_in_sample_false_activation_rate_descriptive": (
            calibration_false_activation
        ),
        "calibration_in_sample_uncorrected_false_activation_rate_descriptive": (
            calibration_uncorrected_false_activation
        ),
        "independent_corrected_null_false_activation_rate": (
            independent_corrected_false_activation
        ),
        "independent_uncorrected_null_false_activation_rate": (
            independent_uncorrected_false_activation
        ),
        "benign_relative_mse_loss": benign_loss,
        "attacked_relative_mse_gain": attacked_gain,
        "attacked_relative_mse_gain_by_noise_regime": per_regime_attacked,
        "corrected_relative_mse_gain_vs_uncorrected_by_noise_regime": (
            corrected_vs_uncorrected
        ),
        "paired_seed_level_student_intervals": intervals,
        "screen_gates_status": "M0_exploratory_not_confirmatory",
        "summary_rows": list(summaries),
        "scope_warning": (
            "Passing authorizes only a locked synthetic K6 development study; "
            "it is not evidence of end-to-end FL utility or Byzantine robustness."
        ),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    static = _validate_protocol()
    device, _ = _require_mps()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    tau_corrected, tau_uncorrected, calibration_rows = _calibrate_tau(device)
    null_validation_rows = _validate_tau_under_independent_null(
        device, tau_corrected, tau_uncorrected
    )
    evaluation_rows = [
        _context(
            seed,
            regime,
            threat,
            device=device,
            null_signal=False,
            tau_a_corrected=tau_corrected,
            tau_a_uncorrected=tau_uncorrected,
        )
        for seed in EVALUATION_SEEDS
        for regime in NOISE_REGIMES
        for threat in THREATS
    ]
    torch.mps.synchronize()
    summaries = _summary_rows(evaluation_rows)
    seed_summaries = _seed_summary_rows(evaluation_rows)
    contrast_intervals = _contrast_intervals(seed_summaries)
    decision = _decision(
        calibration_rows, null_validation_rows, evaluation_rows, summaries
    )
    _write_csv(output / "null_calibration_rows.csv", calibration_rows)
    _write_csv(output / "independent_null_validation_rows.csv", null_validation_rows)
    _write_csv(output / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "seed_summary.csv", seed_summaries)
    _write_json(output / "contrast_intervals.json", contrast_intervals)
    _write_json(
        output / "tau_a_calibration.json",
        {
            "tau_a_corrected": tau_corrected,
            "tau_a_uncorrected": tau_uncorrected,
            "null_quantile": PROTOCOL["eiv"]["tau_a_null_quantile"],
            "one_public_threshold_per_candidate_across_noise_regimes": True,
            "corrected_and_uncorrected_thresholds_calibrated_separately": True,
            "calibration_seed_count": len(CALIBRATION_SEEDS),
            "calibration_context_count": len(calibration_rows),
            "independent_null_validation_seed_count": len(NULL_VALIDATION_SEEDS),
            "independent_null_validation_context_count": len(null_validation_rows),
            "evaluation_seeds_used": False,
            "accuracy_or_target_used": False,
            "calibration_activation_rate_is_in_sample_and_descriptive_only": True,
            "scientific_false_activation_gate_uses_independent_null_seeds": True,
        },
    )
    _write_json(output / "decision.json", decision)
    _write_json(output / "resolved_protocol.json", PROTOCOL)
    artifact_names = (
        "null_calibration_rows.csv",
        "independent_null_validation_rows.csv",
        "evaluation_rows.csv",
        "summary.csv",
        "seed_summary.csv",
        "contrast_intervals.json",
        "tau_a_calibration.json",
        "decision.json",
        "resolved_protocol.json",
    )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "scope": PROTOCOL["scope"],
        "device": "mps",
        "dtype": "float32",
        "mps_fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "protocol_sha256": static["protocol_sha256"],
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha256(
                Path(__file__).resolve()
            ),
            str(ALGORITHM_PATH.relative_to(ROOT)): _sha256(ALGORITHM_PATH),
            **(
                {str(TEST_PATH.relative_to(ROOT)): _sha256(TEST_PATH)}
                if TEST_PATH.is_file()
                else {}
            ),
        },
        "artifact_sha256": {name: _sha256(output / name) for name in artifact_names},
        "calibration_seed_registry_sha256": _canonical_hash(CALIBRATION_SEEDS),
        "null_validation_seed_registry_sha256": _canonical_hash(NULL_VALIDATION_SEEDS),
        "evaluation_seed_registry_sha256": _canonical_hash(EVALUATION_SEEDS),
        "holdout_opened": False,
        "current_round_input_used": False,
        "dense_joint_covariance_materialized": False,
        "tau_a_corrected": tau_corrected,
        "tau_a_uncorrected": tau_uncorrected,
        "all_checks_pass": bool(decision["all_checks_pass"]),
        "decision": decision["decision"],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _plan() -> dict[str, Any]:
    validation = _validate_protocol()
    return {
        **validation,
        "mode": "static_only_no_scientific_tensor_computation",
        "required_command_environment": "PYTORCH_ENABLE_MPS_FALLBACK=0",
        "output": str(DEFAULT_OUTPUT),
        "candidate_pair": ["radial_y", "k6_uncorrected", "k6_tp_eiv"],
        "noise_regimes": list(NOISE_REGIMES),
        "threats": list(THREATS),
        "timeline": PROTOCOL["timeline"],
        "structured_covariance_blocks_per_context": (
            2
            * int(PROTOCOL["timeline"]["window_length"])
            * int(PROTOCOL["cohort"]["num_clients"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.run:
        result = run(args.output)
    else:
        result = _plan()
        result["requested_mode"] = "validate" if args.validate else "dry-run"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
