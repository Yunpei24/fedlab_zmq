#!/usr/bin/env python3
"""Run the preregistered G0g-K4 temporal causal gate screen on MPS."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_fixed_anchor_dual_gated_reference,
    gaussian_aware_fixed_anchor_scalar_gated_reference,
    gaussian_aware_fixed_anchor_temporal_gated_reference,
    gaussian_aware_temporal_standardized_messages,
)
from robustness.aggregators import centered_clipping, clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k1 as base  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k3 as k3  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

PRIMARY = "g0g_k4_temporal_causal_gate"
COUNTERFACTUAL = "g0g_k4_no_compromise_counterfactual_control"
AWARE = "g0g_k2"
BLIND = "g0g_k2_sigma_blind"
K3 = "g0g_k3_dual_gate"
CANDIDATES = ("fcc", BLIND, AWARE, K3, PRIMARY, COUNTERFACTUAL)
TIERS = (1.0, 1.5, 2.0)
FROZEN_K2_SHA256 = (
    "98a30cad8a2f92e6843346e9a9512bd321e8302b9b6e17b447b2641fffb544a5"
)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id",
        "scope",
        "scientific_contract",
        "frozen_calibration",
        "cohort",
        "privacy_noise",
        "references",
        "temporal",
        "honest_dynamics",
        "temporal_calibration",
        "aggregation",
        "threats",
        "no_compromise_control",
        "randomness",
        "candidates",
        "gates",
        "execution",
    }
    if set(config) != expected:
        raise ValueError("The frozen G0g-K4 top-level schema changed")
    if config["campaign_id"] != "gaussian_aware_reference_g0g_k4_tcg_mps_v1":
        raise ValueError("Unexpected G0g-K4 campaign id")
    contract = config["scientific_contract"]
    required_contract = {
        "estimand": "equal_client_mean_of_clean_honest_updates",
        "reference_only": True,
        "far_weights_used": False,
        "delay_weights_used": False,
        "accuracy_used": False,
        "persistent_authenticated_identities_required": True,
        "sybil_resistance_assumed": True,
        "compromise_before_enrollment_supported": False,
        "base_anchor": "exogenous_oracle_common_centre_with_fixed_public_error",
        "anchor_is_past_measurable": True,
        "covariance_role": "authenticated_dp_covariance_only",
        "heterogeneity_in_temporal_whitener": False,
        "history_gate_uses_current_upload": False,
        "history_gate_is_past_measurable": True,
        "final_gate_uses_current_upload": True,
        "final_gate_is_past_measurable": False,
        "temporal_gate_rule": "causal_window_against_fixed_clean_enrollment",
        "final_gate_rule": "elementwise_minimum_k2_aware_and_temporal",
        "counterfactual_control_role": (
            "oracle_common_random_number_no_compromise_history"
        ),
        "counterfactual_control_deployable": False,
        "influence_cap_depends_on_covariance": False,
        "inverse_variance_weighting": False,
        "gate_sum_normalization": False,
        "global_l2_cap": True,
        "current_cohort_replace_one_only": True,
        "user_trajectory_sensitivity_claimed": False,
        "development_only_screen": True,
    }
    if contract != required_contract:
        raise ValueError("The frozen G0g-K4 scientific contract changed")
    if config["frozen_calibration"] != {
        "path": (
            "results/ldp_gradient_far/"
            "gaussian_aware_reference_g0g_k2_mps_v1/calibration.json"
        ),
        "sha256": FROZEN_K2_SHA256,
        "source_campaign_id": "gaussian_aware_reference_g0g_k2_mps_v1",
        "reuse_thresholds_exactly": True,
        "recalibrate": False,
    }:
        raise ValueError("The frozen K2 calibration registry changed")
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    if n < 3 or not 0 <= b < n / 2:
        raise ValueError("G0g-K4 needs n>=3 and 0<=b<n/2")
    if sum(blocks) != int(cohort["dimension"]) or any(width <= 0 for width in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity standard deviation is required per block")
    if len(config["privacy_noise"]["block_std_multipliers"]) != len(blocks):
        raise ValueError("One DP-noise multiplier is required per block")
    noise = config["privacy_noise"]
    base_std = float(noise["base_std"])
    block_multipliers = [
        float(value) for value in noise["block_std_multipliers"]
    ]
    if not math.isfinite(base_std) or base_std <= 0.0 or any(
        not math.isfinite(value) or value <= 0.0 for value in block_multipliers
    ):
        raise ValueError("Authenticated DP block covariances must be positive")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for regime in noise["regimes"]
        for value in regime["client_std_multipliers"]
    ):
        raise ValueError("Every public client DP-noise tier must be positive")
    variance_floor = float(config["references"]["variance_floor"])
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("The numerical covariance ridge must be positive")
    temporal = config["temporal"]
    exact_temporal = {
        "enrollment_rounds": 8,
        "monitoring_rounds": 4,
        "history_window": 4,
        "first_temporal_gate_round": 13,
        "total_rounds": 36,
        "attack_start_round": 13,
        "attack_end_round": 24,
        "recovery_start_round": 25,
        "standardized_clip_formula": "2_sqrt_dimension",
        "standardized_clip_norm": 16.0,
        "whitening_covariance": "authenticated_dp_block_diagonal_only",
        "variance_floor_role": "numerical_ridge_only",
        "nominal_mean_difference_scale": "sqrt_1_over_L_plus_1_over_W0",
        "temporal_gate_shape": "linear_ramp",
        "detection_gate_threshold": 0.5,
        "detection_consecutive_rounds": 2,
        "first_eligible_detection_round": 15,
        "detection_deadline_round": 17,
        "recovery_gate_threshold": 0.9,
        "recovery_consecutive_rounds": 2,
        "first_fully_clean_window_round": 29,
        "recovery_deadline_round": 30,
    }
    if temporal != exact_temporal:
        raise ValueError("The frozen G0g-K4 temporal contract changed")
    expected_clip = 2.0 * math.sqrt(float(cohort["dimension"]))
    if not math.isclose(
        float(temporal["standardized_clip_norm"]), expected_clip, abs_tol=1.0e-12
    ):
        raise ValueError("H_z must equal 2*sqrt(d)")
    calibration = config["temporal_calibration"]
    expected_trajectories = (
        len(calibration["roots"])
        * int(calibration["trajectories_per_root_per_context"])
    )
    if expected_trajectories != int(calibration["expected_trajectories_per_context"]):
        raise ValueError("Temporal calibration trajectory count changed")
    if not 1 <= int(calibration["c0_order_statistic_one_indexed"]) < int(
        calibration["c1_order_statistic_one_indexed"]
    ) <= expected_trajectories:
        raise ValueError("Temporal calibration ranks must satisfy 1<=c0<c1<=m")
    expected_contexts = (
        len(base._noise_cells(config))
        * len(cohort["honest_outliers"]["geometries"])
        * len(config["honest_dynamics"]["names"])
    )
    if expected_contexts != int(calibration["expected_contexts"]):
        raise ValueError("Temporal calibration context count changed")
    if tuple(str(value) for value in config["candidates"]["names"]) != CANDIDATES:
        raise ValueError("The frozen G0g-K4 candidate set changed")
    if config["candidates"]["primary"] != PRIMARY:
        raise ValueError("Unexpected G0g-K4 primary candidate")
    if config["candidates"]["counterfactual_history_control"] != COUNTERFACTUAL:
        raise ValueError("Unexpected G0g-K4 counterfactual-history candidate")
    if config["execution"] != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
    }:
        raise ValueError("Production G0g-K4 must use MPS without CPU fallback")
    randomness = config["randomness"]
    calibration_roots = {int(value) for value in calibration["roots"]}
    development = {int(value) for value in randomness["development_seeds"]}
    holdout = {int(value) for value in randomness["holdout_seeds"]}
    if set(randomness) != {
        "development_seeds",
        "holdout_seeds",
        "holdout_rule",
        "replace_one_trials_per_seed_noise_cell",
        "pair_standard_noise_across_regimes_candidates_threats_and_schedules",
        "counterfactual_history_rule",
        "counterfactual_uses_common_random_numbers",
        "counterfactual_is_oracle_only",
    }:
        raise ValueError("The frozen G0g-K4 randomness schema changed")
    if randomness["counterfactual_history_rule"] != (
        "same_identity_offset_drift_tier_and_dp_noise_without_attack"
    ):
        raise ValueError("The counterfactual history construction changed")
    if not bool(randomness["counterfactual_uses_common_random_numbers"]):
        raise ValueError("The counterfactual must use common random numbers")
    if not bool(randomness["counterfactual_is_oracle_only"]):
        raise ValueError("The counterfactual control must remain oracle-only")
    if len(calibration_roots) != len(calibration["roots"]):
        raise ValueError("Temporal calibration roots must be unique")
    if calibration_roots & development or calibration_roots & holdout:
        raise ValueError("Calibration roots overlap development or holdout seeds")
    if development & holdout:
        raise ValueError("Development and holdout seeds overlap")
    if (
        len(development) != 5
        or len(development) != len(randomness["development_seeds"])
        or len(holdout) != 7
        or len(holdout) != len(randomness["holdout_seeds"])
    ):
        raise ValueError("G0g-K4 requires five development and seven holdout seeds")
    if randomness["holdout_rule"] != "open_only_if_all_development_gates_pass":
        raise ValueError("The frozen holdout rule changed")
    if config["threats"]["schedules"] != [
        "persistent",
        "intermittent_2_on_1_off",
    ]:
        raise ValueError("The frozen G0g-K4 threat schedules changed")
    if config["threats"]["intermittent_pattern"] != [True, True, False]:
        raise ValueError("The frozen intermittent pattern changed")
    if config["no_compromise_control"] != {
        "threat_name": "none",
        "schedule_name": "no_compromise",
        "one_trajectory_per_seed_noise_geometry_dynamics": True,
        "active_history_rounds": [13, 36],
    }:
        raise ValueError("The frozen no-compromise control changed")
    if not math.isclose(
        float(config["references"]["total_client_influence_cap"]),
        float(config["references"]["fcc_radius"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("K4 and FCC must use the same global L2 cap")


def _load_frozen_k2_calibration(
    config: Mapping[str, Any], *, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration, provenance = k3._load_frozen_calibration(config, root=root)
    source = {int(value) for value in calibration.get("calibration_seeds", [])}
    reserved = {
        int(value) for value in config["temporal_calibration"]["roots"]
    } | {int(value) for value in config["randomness"]["development_seeds"]} | {
        int(value) for value in config["randomness"]["holdout_seeds"]
    }
    if source & reserved:
        raise ValueError(
            "K2 calibration seeds overlap K4 calibration, development or holdout"
        )
    provenance = dict(provenance)
    provenance["disjoint_from_all_k4_seed_sets"] = True
    return calibration, provenance


def _phase(config: Mapping[str, Any], round_index: int) -> str:
    temporal = config["temporal"]
    if round_index <= int(temporal["enrollment_rounds"]):
        return "enrollment"
    if round_index < int(temporal["attack_start_round"]):
        return "monitoring"
    if round_index <= int(temporal["attack_end_round"]):
        return "attack"
    return "recovery"


def _attack_active(
    config: Mapping[str, Any], *, round_index: int, schedule: str
) -> bool:
    temporal = config["temporal"]
    start = int(temporal["attack_start_round"])
    end = int(temporal["attack_end_round"])
    if schedule == "no_compromise":
        return False
    if not start <= round_index <= end:
        return False
    if schedule == "persistent":
        return True
    if schedule == "intermittent_2_on_1_off":
        pattern = [bool(value) for value in config["threats"]["intermittent_pattern"]]
        return pattern[(round_index - start) % len(pattern)]
    raise ValueError(f"Unknown attack schedule {schedule!r}")


def _expand_block_values(values: torch.Tensor, block_sizes: Sequence[int]) -> torch.Tensor:
    return torch.repeat_interleave(
        values,
        torch.tensor(block_sizes, dtype=torch.long, device=values.device),
        dim=-1,
    )


def _drift_increments(
    config: Mapping[str, Any],
    *,
    variances: torch.Tensor,
    seed: int,
    geometry: str,
    dynamics: str,
) -> torch.Tensor:
    n = int(config["cohort"]["num_clients"])
    dimension = int(config["cohort"]["dimension"])
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    if dynamics == "stationary":
        return torch.zeros(
            (n, dimension),
            dtype=oracle._RUNTIME_DTYPE,
            device=oracle._RUNTIME_DEVICE,
        )
    if dynamics != "bounded_drift":
        raise ValueError(f"Unknown honest dynamics {dynamics!r}")
    raw = torch.randn(
        n,
        dimension,
        generator=oracle._generator("g0g-k4-drift", seed, geometry),
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    )
    ridge = float(config["references"]["variance_floor"])
    scales = _expand_block_values((variances + ridge).sqrt(), blocks)
    raw_mahalanobis = torch.linalg.vector_norm(raw / scales, dim=1).clamp_min(
        torch.finfo(raw.dtype).tiny
    )
    step = float(config["honest_dynamics"]["drift_step_mahalanobis"])
    increments = raw * (step / raw_mahalanobis)[:, None]
    increments = increments - increments.mean(dim=0, keepdim=True)
    centred_norms = torch.linalg.vector_norm(increments / scales, dim=1)
    maximum = float(centred_norms.max().item())
    if maximum > step:
        increments = increments * (step / maximum)
    return increments


def _trajectory_components(
    config: dict[str, Any],
    *,
    seed: int,
    regime: dict[str, Any],
    permutation: str,
    geometry: str,
    dynamics: str,
) -> dict[str, torch.Tensor]:
    base_clean, outliers, centre, anchor = oracle._honest_clean_vectors(
        config,
        seed=seed,
        draw=0,
        geometry=geometry,
        include_outliers=True,
    )
    variances, tiers = oracle._noise_variances(config, regime, permutation)
    increments = _drift_increments(
        config,
        variances=variances,
        seed=seed,
        geometry=geometry,
        dynamics=dynamics,
    )
    return {
        "base_clean": base_clean,
        "outliers": outliers,
        "centre": centre,
        "anchor": anchor,
        "variances": variances,
        "tiers": tiers,
        "drift_increments": increments,
    }


def _clean_at_round(components: Mapping[str, torch.Tensor], round_index: int) -> torch.Tensor:
    return components["base_clean"] + float(round_index - 1) * components[
        "drift_increments"
    ]


def _target_at_round(
    clean: torch.Tensor,
    latent_byzantine: torch.Tensor,
    *,
    phase: str,
    compromised_during_attack_phase: bool = True,
) -> torch.Tensor:
    """Return the frozen oracle estimand for one longitudinal phase."""

    if phase == "attack" and compromised_during_attack_phase:
        return clean[~latent_byzantine].mean(dim=0)
    if phase in {"enrollment", "monitoring", "attack", "recovery"}:
        return clean.mean(dim=0)
    raise ValueError(f"Unknown longitudinal phase {phase!r}")


def _private_vectors(
    config: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    *,
    seed: int,
    round_index: int,
    geometry: str,
) -> torch.Tensor:
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    clean = _clean_at_round(components, round_index)
    return base._paired_private_noise(
        clean,
        components["variances"],
        blocks,
        seed=seed,
        draw=round_index,
        geometry=geometry,
    )


def _standardized_messages(
    config: Mapping[str, Any],
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    variances: torch.Tensor,
) -> torch.Tensor:
    return gaussian_aware_temporal_standardized_messages(
        vectors,
        anchor=anchor,
        noise_variances=variances,
        block_sizes=tuple(int(value) for value in config["cohort"]["block_sizes"]),
        variance_floor=float(config["references"]["variance_floor"]),
        standardized_clip_norm=float(config["temporal"]["standardized_clip_norm"]),
        return_diagnostics=False,
    )


def _temporal_statistics(
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
    *,
    window: int,
    enrollment_size: int,
) -> torch.Tensor:
    if len(history) < window:
        raise ValueError("Temporal history is shorter than the frozen window")
    stacked = torch.stack(list(history[-window:]), dim=1)
    nominal_scale = math.sqrt(1.0 / float(window) + 1.0 / float(enrollment_size))
    return torch.linalg.vector_norm(stacked.mean(dim=1) - enrollment_mean, dim=1) / (
        nominal_scale
    )


def _order_statistic(values: Sequence[float], rank_one_indexed: int) -> float:
    if not values or not 1 <= rank_one_indexed <= len(values):
        raise ValueError("Invalid one-indexed order statistic")
    return float(sorted(float(value) for value in values)[rank_one_indexed - 1])


def _temporal_contexts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for regime, permutation in base._noise_cells(config):
        for geometry in config["cohort"]["honest_outliers"]["geometries"]:
            for dynamics in config["honest_dynamics"]["names"]:
                result.append(
                    {
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": str(permutation),
                        "outlier_geometry": str(geometry),
                        "honest_dynamics": str(dynamics),
                        "regime": regime,
                    }
                )
    return result


def _trajectory_id(
    *,
    seed: int,
    regime_name: str,
    permutation: str,
    geometry: str,
    dynamics: str,
    threat: str,
    schedule: str,
) -> str:
    return "|".join(
        (
            str(seed),
            regime_name,
            permutation,
            geometry,
            dynamics,
            threat,
            schedule,
        )
    )


def _development_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate the exact preregistered attack and clean trajectories."""

    cells: list[dict[str, Any]] = []
    clean_threat = str(config["no_compromise_control"]["threat_name"])
    clean_schedule = str(config["no_compromise_control"]["schedule_name"])
    for seed_value in config["randomness"]["development_seeds"]:
        seed = int(seed_value)
        for regime, permutation_value in base._noise_cells(config):
            permutation = str(permutation_value)
            for geometry_value in config["cohort"]["honest_outliers"]["geometries"]:
                geometry = str(geometry_value)
                for dynamics_value in config["honest_dynamics"]["names"]:
                    dynamics = str(dynamics_value)
                    for threat_value in config["threats"]["names"]:
                        threat = str(threat_value)
                        for schedule_value in config["threats"]["schedules"]:
                            schedule = str(schedule_value)
                            cells.append(
                                {
                                    "seed": seed,
                                    "regime": regime,
                                    "permutation": permutation,
                                    "geometry": geometry,
                                    "dynamics": dynamics,
                                    "threat": threat,
                                    "schedule": schedule,
                                    "trajectory_id": _trajectory_id(
                                        seed=seed,
                                        regime_name=str(regime["name"]),
                                        permutation=permutation,
                                        geometry=geometry,
                                        dynamics=dynamics,
                                        threat=threat,
                                        schedule=schedule,
                                    ),
                                }
                            )
                    cells.append(
                        {
                            "seed": seed,
                            "regime": regime,
                            "permutation": permutation,
                            "geometry": geometry,
                            "dynamics": dynamics,
                            "threat": clean_threat,
                            "schedule": clean_schedule,
                            "trajectory_id": _trajectory_id(
                                seed=seed,
                                regime_name=str(regime["name"]),
                                permutation=permutation,
                                geometry=geometry,
                                dynamics=dynamics,
                                threat=clean_threat,
                                schedule=clean_schedule,
                            ),
                        }
                    )
    identifiers = [str(cell["trajectory_id"]) for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Development trajectory identifiers are not unique")
    return cells


def _calibrate_temporal(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = config["temporal_calibration"]
    temporal = config["temporal"]
    window = int(temporal["history_window"])
    enrollment_size = int(temporal["enrollment_rounds"])
    c0_rank = int(settings["c0_order_statistic_one_indexed"])
    c1_rank = int(settings["c1_order_statistic_one_indexed"])
    context_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for context in _temporal_contexts(config):
        maxima: list[float] = []
        context_name = "|".join(
            [
                context["noise_regime"],
                context["noise_permutation"],
                context["outlier_geometry"],
                context["honest_dynamics"],
            ]
        )
        for root_seed in settings["roots"]:
            for draw in range(int(settings["trajectories_per_root_per_context"])):
                seed = oracle._seed("g0g-k4-calibration", root_seed, draw, context_name)
                components = _trajectory_components(
                    config,
                    seed=seed,
                    regime=context["regime"],
                    permutation=context["noise_permutation"],
                    geometry=context["outlier_geometry"],
                    dynamics=context["honest_dynamics"],
                )
                history: list[torch.Tensor] = []
                enrollment_mean: torch.Tensor | None = None
                maximum = 0.0
                for round_index in range(1, int(temporal["total_rounds"]) + 1):
                    observed = _private_vectors(
                        config,
                        components,
                        seed=seed,
                        round_index=round_index,
                        geometry=context["outlier_geometry"],
                    )
                    vectors = clip_l2(
                        observed, float(config["aggregation"]["server_clip_norm"])
                    )
                    standardized = _standardized_messages(
                        config,
                        vectors,
                        anchor=components["anchor"],
                        variances=components["variances"],
                    )
                    if round_index == enrollment_size:
                        enrollment_mean = torch.stack(history + [standardized]).mean(
                            dim=0
                        )
                    if round_index >= int(temporal["first_temporal_gate_round"]):
                        if enrollment_mean is None:
                            raise RuntimeError("Enrollment mean was not initialized")
                        statistics = _temporal_statistics(
                            history,
                            enrollment_mean,
                            window=window,
                            enrollment_size=enrollment_size,
                        )
                        maximum = max(maximum, float(statistics.max().item()))
                    history.append(standardized)
                maxima.append(maximum)
                trajectory_rows.append(
                    {
                        "context": context_name,
                        "root_seed": int(root_seed),
                        "draw": draw,
                        "trajectory_seed": int(seed),
                        "maximum_temporal_statistic": maximum,
                    }
                )
        c0 = _order_statistic(maxima, c0_rank)
        c1 = _order_statistic(maxima, c1_rank)
        if not c1 > c0:
            raise RuntimeError(f"Temporal calibration produced c1<=c0 in {context_name}")
        context_records.append(
            {
                "context": context_name,
                "trajectory_count": len(maxima),
                "c0_order_statistic": c0,
                "c1_order_statistic": c1,
                "c0_rank_one_indexed": c0_rank,
                "c1_rank_one_indexed": c1_rank,
            }
        )
    deployed_c0 = max(float(row["c0_order_statistic"]) for row in context_records)
    deployed_c1 = max(float(row["c1_order_statistic"]) for row in context_records)
    if not deployed_c1 > deployed_c0:
        raise RuntimeError("Deployed temporal thresholds must satisfy c1>c0")
    return (
        {
            "protocol": "independent_clean_trajectory_maximum_order_statistics",
            "contexts": context_records,
            "context_count": len(context_records),
            "trajectories_per_context": int(
                settings["expected_trajectories_per_context"]
            ),
            "maximum_domain": "all_clients_and_rounds_13_to_36",
            "c0_rank_one_indexed": c0_rank,
            "c1_rank_one_indexed": c1_rank,
            "deployed_threshold_rule": "maximum_context_threshold",
            "deployed_c0": deployed_c0,
            "deployed_c1": deployed_c1,
            "coverage_scope": settings["coverage_scope"],
            "simultaneous_future_coverage_claimed": False,
            "clipping_and_bounded_drift_included": True,
            "holdout_opened": False,
        },
        trajectory_rows,
    )


def _radii(
    config: Mapping[str, Any],
    variances: torch.Tensor,
    calibration: Mapping[str, Any],
    *,
    regime_name: str,
    blind: bool,
) -> torch.Tensor:
    return k3._radii(
        config,
        variances,
        calibration,
        regime_name=regime_name,
        blind=blind,
    )


def _current_reference(
    method: str,
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    blind_radii: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate one memoryless comparator with K4's frozen public constants."""

    if method == "fcc":
        return (
            centered_clipping(
                vectors,
                anchor=anchor,
                tau=float(config["references"]["fcc_radius"]),
            ),
            {},
        )
    common = {
        "anchor": anchor,
        "block_sizes": tuple(int(v) for v in config["cohort"]["block_sizes"]),
        "influence_cap": float(config["references"]["total_client_influence_cap"]),
        "gate_transition_width": float(
            config["references"]["current_gate_transition_width"]
        ),
        "return_diagnostics": True,
    }
    if method == BLIND:
        return gaussian_aware_fixed_anchor_scalar_gated_reference(
            vectors, statistical_radii=blind_radii, **common
        )
    if method == AWARE:
        return gaussian_aware_fixed_anchor_scalar_gated_reference(
            vectors, statistical_radii=aware_radii, **common
        )
    if method == K3:
        return gaussian_aware_fixed_anchor_dual_gated_reference(
            vectors,
            statistical_radii=aware_radii,
            common_statistical_radii=blind_radii,
            **common,
        )
    raise ValueError(f"Unknown memoryless G0g-K4 comparator {method!r}")


def _k4_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
    thresholds: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    window = int(config["temporal"]["history_window"])
    return gaussian_aware_fixed_anchor_temporal_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=aware_radii,
        temporal_standardized_history=torch.stack(list(history[-window:]), dim=1),
        enrollment_standardized_mean=enrollment_mean,
        enrollment_size=int(config["temporal"]["enrollment_rounds"]),
        temporal_gate_inner_threshold=float(thresholds["deployed_c0"]),
        temporal_gate_outer_threshold=float(thresholds["deployed_c1"]),
        block_sizes=tuple(int(v) for v in config["cohort"]["block_sizes"]),
        influence_cap=float(config["references"]["total_client_influence_cap"]),
        current_gate_transition_width=float(
            config["references"]["current_gate_transition_width"]
        ),
        return_diagnostics=True,
    )


def _temporal_identity_diagnostics(
    diagnostics: Mapping[str, Any], n: int
) -> dict[str, Any]:
    """Augment K2-aware diagnostics for rounds where h is fixed to one."""

    result = dict(diagnostics)
    current = [float(value) for value in diagnostics["gates_by_client"]]
    result.update(
        {
            "current_gates_by_client": current,
            "temporal_statistics_by_client": [0.0] * n,
            "temporal_gates_by_client": [1.0] * n,
            "gates_by_client": current,
            "temporal_gate_mean": 1.0,
            "temporal_trigger_fraction": 0.0,
            "temporal_detection_fraction": 0.0,
            "temporal_history_is_strictly_prior_to_current_upload": True,
            "normalization_by_gate_sum": False,
        }
    )
    return result


def _contribution_mass(
    diagnostics: Mapping[str, Any], latent_byzantine: torch.Tensor
) -> float:
    values = diagnostics.get("client_contribution_norms")
    if values is None:
        return float("nan")
    norms = torch.tensor(
        values, dtype=oracle._RUNTIME_DTYPE, device=oracle._RUNTIME_DEVICE
    )
    total = float(norms.sum().item())
    return float(norms[latent_byzantine].sum().item()) / total if total else 0.0


def _finite_tree(value: Any) -> bool:
    """Check every numeric diagnostic recursively while ignoring text labels."""

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _evaluate_trajectory(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    *,
    seed: int,
    regime: dict[str, Any],
    permutation: str,
    geometry: str,
    dynamics: str,
    threat: str,
    schedule: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = int(config["cohort"]["num_clients"])
    b = int(config["cohort"]["num_byzantine"])
    temporal = config["temporal"]
    components = _trajectory_components(
        config,
        seed=seed,
        regime=regime,
        permutation=permutation,
        geometry=geometry,
        dynamics=dynamics,
    )
    latent_byzantine = torch.zeros(
        n, dtype=torch.bool, device=oracle._RUNTIME_DEVICE
    )
    latent_byzantine[n - b :] = True
    honest = ~latent_byzantine
    regular = honest & ~components["outliers"]
    aware_radii = _radii(
        config,
        components["variances"],
        k2_calibration,
        regime_name=str(regime["name"]),
        blind=False,
    )
    blind_radii = _radii(
        config,
        components["variances"],
        k2_calibration,
        regime_name=str(regime["name"]),
        blind=True,
    )
    history: list[torch.Tensor] = []
    counterfactual_history: list[torch.Tensor] = []
    enrollment_mean: torch.Tensor | None = None
    counterfactual_enrollment_mean: torch.Tensor | None = None
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    temporal_gates_by_round: dict[int, torch.Tensor] = {}
    trajectory_id = _trajectory_id(
        seed=seed,
        regime_name=str(regime["name"]),
        permutation=permutation,
        geometry=geometry,
        dynamics=dynamics,
        threat=threat,
        schedule=schedule,
    )
    for round_index in range(1, int(temporal["total_rounds"]) + 1):
        clean = _clean_at_round(components, round_index)
        no_compromise_observed = _private_vectors(
            config,
            components,
            seed=seed,
            round_index=round_index,
            geometry=geometry,
        )
        observed = no_compromise_observed
        attack_active = _attack_active(
            config, round_index=round_index, schedule=schedule
        )
        if attack_active:
            observed, active_byzantine = oracle._replace_with_attack(
                observed,
                config,
                threat=threat,
                severity=float(config["threats"]["severity"]),
                seed=oracle._seed(
                    "g0g-k4-attack",
                    seed,
                    regime["name"],
                    permutation,
                    geometry,
                    dynamics,
                    threat,
                    round_index,
                ),
            )
        else:
            active_byzantine = torch.zeros_like(latent_byzantine)
        vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
        counterfactual_vectors = clip_l2(
            no_compromise_observed,
            float(config["aggregation"]["server_clip_norm"]),
        )
        standardized = _standardized_messages(
            config,
            vectors,
            anchor=components["anchor"],
            variances=components["variances"],
        )
        counterfactual_standardized = _standardized_messages(
            config,
            counterfactual_vectors,
            anchor=components["anchor"],
            variances=components["variances"],
        )
        if round_index == int(temporal["enrollment_rounds"]):
            enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)
            counterfactual_enrollment_mean = torch.stack(
                counterfactual_history + [counterfactual_standardized]
            ).mean(dim=0)

        outputs: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for method in ("fcc", BLIND, AWARE, K3):
            outputs[method], diagnostics[method] = _current_reference(
                method,
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                blind_radii=blind_radii,
                config=config,
            )
        if round_index < int(temporal["first_temporal_gate_round"]):
            outputs[PRIMARY] = outputs[AWARE].clone()
            outputs[COUNTERFACTUAL] = outputs[AWARE].clone()
            diagnostics[PRIMARY] = _temporal_identity_diagnostics(
                diagnostics[AWARE], n
            )
            diagnostics[COUNTERFACTUAL] = _temporal_identity_diagnostics(
                diagnostics[AWARE], n
            )
        else:
            if enrollment_mean is None or counterfactual_enrollment_mean is None:
                raise RuntimeError("The clean enrollment baseline is missing")
            outputs[PRIMARY], diagnostics[PRIMARY] = _k4_reference(
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                history=history,
                enrollment_mean=enrollment_mean,
                thresholds=temporal_calibration,
                config=config,
            )
            outputs[COUNTERFACTUAL], diagnostics[COUNTERFACTUAL] = _k4_reference(
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                history=counterfactual_history,
                enrollment_mean=counterfactual_enrollment_mean,
                thresholds=temporal_calibration,
                config=config,
            )
            diagnostics[COUNTERFACTUAL]["counterfactual_oracle_only"] = True
            diagnostics[COUNTERFACTUAL]["counterfactual_history_attacked"] = False
            diagnostics[COUNTERFACTUAL][
                "counterfactual_common_random_numbers"
            ] = True
            temporal_gates_by_round[round_index] = torch.tensor(
                diagnostics[PRIMARY]["temporal_gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )

        phase = _phase(config, round_index)
        target = _target_at_round(
            clean,
            latent_byzantine,
            phase=phase,
            compromised_during_attack_phase=threat != "none",
        )
        pairing_id = f"{trajectory_id}|{round_index}"
        for method in CANDIDATES:
            diagnostic = diagnostics[method]
            contribution_norms = diagnostic.get("client_contribution_norms")
            max_contribution = (
                max(float(value) for value in contribution_norms)
                if contribution_norms is not None
                else float("nan")
            )
            temporal_gates = diagnostic.get("temporal_gates_by_client")
            all_temporal_one = bool(
                temporal_gates is not None
                and all(abs(float(value) - 1.0) <= 1.0e-7 for value in temporal_gates)
            )
            round_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "pairing_id": pairing_id,
                    "seed": int(seed),
                    "noise_regime": str(regime["name"]),
                    "noise_permutation": permutation,
                    "outlier_geometry": geometry,
                    "honest_dynamics": dynamics,
                    "threat": threat,
                    "schedule": schedule,
                    "round": round_index,
                    "phase": phase,
                    "attack_active": attack_active,
                    "candidate": method,
                    "reference_error": float(
                        torch.linalg.vector_norm(outputs[method] - target).item()
                    ),
                    "difference_to_k2_aware": float(
                        torch.linalg.vector_norm(outputs[method] - outputs[AWARE]).item()
                    ),
                    "temporal_gate_all_one": all_temporal_one,
                    "temporal_gate_mean": float(
                        diagnostic.get("temporal_gate_mean", float("nan"))
                    ),
                    "current_gate_mean": float(
                        diagnostic.get(
                            "current_gate_mean",
                            diagnostic.get("gate_mean", float("nan")),
                        )
                    ),
                    "final_gate_mean": float(
                        diagnostic.get("gate_mean", float("nan"))
                    ),
                    "byzantine_contribution_mass": _contribution_mass(
                        diagnostic, latent_byzantine
                    ),
                    "max_client_contribution_norm": max_contribution,
                    "contribution_cap_respected": bool(
                        diagnostic.get("client_contribution_cap_respected", True)
                    ),
                    "normalization_by_gate_sum": bool(
                        diagnostic.get("normalization_by_gate_sum", False)
                    ),
                    "all_finite": bool(torch.isfinite(outputs[method]).all())
                    and _finite_tree(diagnostic),
                }
            )
        if round_index >= int(temporal["first_temporal_gate_round"]):
            diagnostic = diagnostics[PRIMARY]
            statistics = diagnostic["temporal_statistics_by_client"]
            current_gates = diagnostic["current_gates_by_client"]
            temporal_gates = diagnostic["temporal_gates_by_client"]
            final_gates = diagnostic["gates_by_client"]
            contribution_norms = diagnostic["client_contribution_norms"]
            counterfactual_diagnostic = diagnostics[COUNTERFACTUAL]
            for client in range(n):
                client_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "seed": int(seed),
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": permutation,
                        "outlier_geometry": geometry,
                        "honest_dynamics": dynamics,
                        "threat": threat,
                        "schedule": schedule,
                        "round": round_index,
                        "phase": phase,
                        "attack_active": attack_active,
                        "client": client,
                        "public_noise_tier": float(
                            components["tiers"][client].item()
                        ),
                        "latent_byzantine": bool(latent_byzantine[client]),
                        "active_byzantine": bool(active_byzantine[client]),
                        "honest_outlier": bool(components["outliers"][client]),
                        "honest_regular": bool(regular[client]),
                        "temporal_statistic": float(statistics[client]),
                        "current_gate": float(current_gates[client]),
                        "temporal_gate": float(temporal_gates[client]),
                        "counterfactual_temporal_statistic": float(
                            counterfactual_diagnostic[
                                "temporal_statistics_by_client"
                            ][client]
                        ),
                        "counterfactual_temporal_gate": float(
                            counterfactual_diagnostic["temporal_gates_by_client"][
                                client
                            ]
                        ),
                        "final_gate": float(final_gates[client]),
                        "gate_drop_vs_k2_aware": max(
                            0.0,
                            float(current_gates[client]) - float(final_gates[client]),
                        ),
                        "contribution_norm": float(contribution_norms[client]),
                    }
                )
        history.append(standardized)
        counterfactual_history.append(counterfactual_standardized)

    trajectory_rows = _trajectory_summaries(
        round_rows,
        temporal_gates_by_round,
        latent_byzantine,
        config=config,
    )
    return round_rows, client_rows, trajectory_rows


def _first_two_round_event(
    gates: Mapping[int, torch.Tensor],
    *,
    client: int,
    first_end_round: int,
    last_end_round: int,
    threshold: float,
    direction: str,
) -> int | None:
    for end_round in range(first_end_round, last_end_round + 1):
        if end_round not in gates or end_round - 1 not in gates:
            continue
        pair = (float(gates[end_round - 1][client]), float(gates[end_round][client]))
        if direction == "below" and all(value <= threshold for value in pair):
            return end_round
        if direction == "above" and all(value >= threshold for value in pair):
            return end_round
    return None


def _trajectory_summaries(
    round_rows: list[dict[str, Any]],
    temporal_gates_by_round: Mapping[int, torch.Tensor],
    latent_byzantine: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not round_rows:
        raise ValueError("Cannot summarize an empty trajectory")
    temporal = config["temporal"]
    base_row = round_rows[0]
    byzantine_ids = torch.nonzero(latent_byzantine, as_tuple=False).reshape(-1).tolist()
    detection_ends = [
        _first_two_round_event(
            temporal_gates_by_round,
            client=int(client),
            first_end_round=int(temporal["first_eligible_detection_round"]),
            last_end_round=int(temporal["detection_deadline_round"]),
            threshold=float(temporal["detection_gate_threshold"]),
            direction="below",
        )
        for client in byzantine_ids
    ]
    recovery_ends = [
        _first_two_round_event(
            temporal_gates_by_round,
            client=int(client),
            first_end_round=int(temporal["first_fully_clean_window_round"]) + 1,
            last_end_round=int(temporal["recovery_deadline_round"]),
            threshold=float(temporal["recovery_gate_threshold"]),
            direction="above",
        )
        for client in byzantine_ids
    ]
    result: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selected = [row for row in round_rows if row["candidate"] == candidate]
        by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_phase[str(row["phase"])].append(row)
        result.append(
            {
                "trajectory_id": base_row["trajectory_id"],
                "seed": int(base_row["seed"]),
                "noise_regime": base_row["noise_regime"],
                "noise_permutation": base_row["noise_permutation"],
                "outlier_geometry": base_row["outlier_geometry"],
                "honest_dynamics": base_row["honest_dynamics"],
                "threat": base_row["threat"],
                "schedule": base_row["schedule"],
                "candidate": candidate,
                "enrollment_auc": base._finite_mean(
                    row["reference_error"] for row in by_phase["enrollment"]
                ),
                "monitoring_auc": base._finite_mean(
                    row["reference_error"] for row in by_phase["monitoring"]
                ),
                "attack_auc": base._finite_mean(
                    row["reference_error"] for row in by_phase["attack"]
                ),
                "active_attack_auc": base._finite_mean(
                    row["reference_error"]
                    for row in by_phase["attack"]
                    if bool(row["attack_active"])
                ),
                "recovery_auc": base._finite_mean(
                    row["reference_error"] for row in by_phase["recovery"]
                ),
                "post_enrollment_auc": base._finite_mean(
                    row["reference_error"]
                    for phase in ("attack", "recovery")
                    for row in by_phase[phase]
                ),
                "attack_byzantine_contribution_mass": base._finite_mean(
                    row["byzantine_contribution_mass"] for row in by_phase["attack"]
                ),
                "detection_rate_within_deadline": (
                    sum(value is not None for value in detection_ends)
                    / float(len(detection_ends))
                    if candidate == PRIMARY and detection_ends
                    else float("nan")
                ),
                "mean_detection_end_round": (
                    base._finite_mean(
                        float(value)
                        for value in detection_ends
                        if value is not None
                    )
                    if candidate == PRIMARY
                    else float("nan")
                ),
                "recovery_rate_within_deadline": (
                    sum(value is not None for value in recovery_ends)
                    / float(len(recovery_ends))
                    if candidate == PRIMARY and recovery_ends
                    else float("nan")
                ),
                "mean_recovery_end_round": (
                    base._finite_mean(
                        float(value) for value in recovery_ends if value is not None
                    )
                    if candidate == PRIMARY
                    else float("nan")
                ),
            }
        )
    return result


def _replace_one_audit(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit only the conditional current-cohort replace-one certificate."""

    rows: list[dict[str, Any]] = []
    round_index = int(config["temporal"]["first_temporal_gate_round"])
    enrollment_size = int(config["temporal"]["enrollment_rounds"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    for seed_value in config["randomness"]["development_seeds"]:
        seed = int(seed_value)
        for regime, permutation in base._noise_cells(config):
            components = _trajectory_components(
                config,
                seed=seed,
                regime=regime,
                permutation=permutation,
                geometry="orthogonal",
                dynamics="stationary",
            )
            history: list[torch.Tensor] = []
            enrollment_mean: torch.Tensor | None = None
            current_vectors: torch.Tensor | None = None
            for past_round in range(1, round_index + 1):
                observed = _private_vectors(
                    config,
                    components,
                    seed=seed,
                    round_index=past_round,
                    geometry="orthogonal",
                )
                vectors = clip_l2(
                    observed, float(config["aggregation"]["server_clip_norm"])
                )
                standardized = _standardized_messages(
                    config,
                    vectors,
                    anchor=components["anchor"],
                    variances=components["variances"],
                )
                if past_round == enrollment_size:
                    enrollment_mean = torch.stack(history + [standardized]).mean(
                        dim=0
                    )
                if past_round < round_index:
                    history.append(standardized)
                else:
                    current_vectors = vectors
            if enrollment_mean is None or current_vectors is None:
                raise RuntimeError("Replace-one audit failed to build fixed history")
            aware_radii = _radii(
                config,
                components["variances"],
                k2_calibration,
                regime_name=str(regime["name"]),
                blind=False,
            )
            for trial in range(trials):
                replaced = (
                    oracle._seed(
                        "g0g-k4-replaced-client",
                        seed,
                        regime["name"],
                        permutation,
                        trial,
                    )
                    % current_vectors.shape[0]
                )
                neighbour = current_vectors.clone()
                proposal = torch.randn(
                    current_vectors.shape[1],
                    generator=oracle._generator(
                        "g0g-k4-replacement",
                        seed,
                        regime["name"],
                        permutation,
                        trial,
                    ),
                    dtype=current_vectors.dtype,
                    device=current_vectors.device,
                )
                neighbour[replaced] = clip_l2(
                    100.0 * proposal[None, :],
                    float(config["aggregation"]["server_clip_norm"]),
                )[0]
                left, diagnostics = _k4_reference(
                    current_vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    history=history,
                    enrollment_mean=enrollment_mean,
                    thresholds=temporal_calibration,
                    config=config,
                )
                right, _ = _k4_reference(
                    neighbour,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    history=history,
                    enrollment_mean=enrollment_mean,
                    thresholds=temporal_calibration,
                    config=config,
                )
                difference = float(torch.linalg.vector_norm(left - right).item())
                bound = float(diagnostics["replace_one_bound"])
                rows.append(
                    {
                        "seed": seed,
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": permutation,
                        "trial": trial,
                        "round": round_index,
                        "replaced_client": int(replaced),
                        "same_past": True,
                        "same_covariances": True,
                        "same_anchor": True,
                        "observed_replace_one_difference": difference,
                        "theoretical_replace_one_bound": bound,
                        "ratio_observed_to_bound": difference / bound,
                        "violation": difference > bound + 1.0e-6,
                    }
                )
    return rows


def _screen_development(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for cell in _development_cells(config):
        current_rounds, current_clients, current_trajectories = _evaluate_trajectory(
            config,
            k2_calibration,
            temporal_calibration,
            seed=int(cell["seed"]),
            regime=cell["regime"],
            permutation=str(cell["permutation"]),
            geometry=str(cell["geometry"]),
            dynamics=str(cell["dynamics"]),
            threat=str(cell["threat"]),
            schedule=str(cell["schedule"]),
        )
        if any(
            str(row["trajectory_id"]) != str(cell["trajectory_id"])
            for row in current_rounds + current_clients + current_trajectories
        ):
            raise RuntimeError("A development cell emitted a wrong trajectory id")
        round_rows.extend(current_rounds)
        client_rows.extend(current_clients)
        trajectory_rows.extend(current_trajectories)
    return round_rows, client_rows, trajectory_rows


def _paired_seed_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    competitor: str,
    predicate,
    metric: str = "attack_auc",
) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["candidate"] in {PRIMARY, competitor} and predicate(row):
            grouped[(int(row["seed"]), str(row["candidate"]))].append(
                float(row[metric])
            )
    seeds = sorted(
        seed
        for seed, candidate in grouped
        if candidate == PRIMARY and (seed, competitor) in grouped
    )
    return [
        base._finite_mean(grouped[(seed, PRIMARY)])
        - base._finite_mean(grouped[(seed, competitor)])
        for seed in seeds
    ]


def _paired_ratio(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate,
    metric: str,
) -> float:
    selected_candidate = [
        float(row[metric])
        for row in rows
        if row["candidate"] == candidate and predicate(row)
    ]
    selected_baseline = [
        float(row[metric])
        for row in rows
        if row["candidate"] == baseline and predicate(row)
    ]
    return base._finite_mean(selected_candidate) / max(
        base._finite_mean(selected_baseline), 1.0e-12
    )


def _relative_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    competitor: str,
    predicate,
    metric: str = "attack_auc",
) -> float:
    ratio = _paired_ratio(
        rows,
        candidate=PRIMARY,
        baseline=competitor,
        predicate=predicate,
        metric=metric,
    )
    return 1.0 - ratio


def _representative_client_rows(
    client_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    threat = str(config["threats"]["names"][0])
    return [
        row
        for row in client_rows
        if row["threat"] == threat and row["schedule"] == "persistent"
    ]


def _false_trigger_diagnostics(
    client_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    selected = [
        row
        for row in _representative_client_rows(client_rows, config)
        if bool(row["honest_regular"])
    ]
    client_round_rate = base._finite_mean(
        1.0 if float(row["temporal_gate"]) < 1.0 - 1.0e-7 else 0.0
        for row in selected
    )
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        by_trajectory[str(row["trajectory_id"])].append(
            float(row["temporal_statistic"])
        )
    c0 = float(config.get("_deployed_temporal_c0", float("nan")))
    trajectory_rate = base._finite_mean(
        1.0 if max(values) > c0 else 0.0 for values in by_trajectory.values()
    )
    tier_rates: dict[str, float] = {}
    tier_gate_means: dict[str, float] = {}
    heteroscedastic = [
        row for row in selected if row["noise_regime"] == "heteroscedastic"
    ]
    for tier in TIERS:
        tier_rows = [
            row
            for row in heteroscedastic
            if math.isclose(float(row["public_noise_tier"]), tier, abs_tol=1.0e-7)
        ]
        if tier_rows:
            tier_rates[str(tier)] = base._finite_mean(
                1.0
                if float(row["temporal_gate"]) < 1.0 - 1.0e-7
                else 0.0
                for row in tier_rows
            )
            tier_gate_means[str(tier)] = base._finite_mean(
                float(row["temporal_gate"]) for row in tier_rows
            )
    return {
        "client_round_rate": client_round_rate,
        "trajectory_rate": trajectory_rate,
        "tier_rates": tier_rates,
        "tier_gate_means": tier_gate_means,
        "tier_rate_gap": (
            max(tier_rates.values()) - min(tier_rates.values())
            if len(tier_rates) > 1
            else 0.0
        ),
        "tier_gate_mean_gap": (
            max(tier_gate_means.values()) - min(tier_gate_means.values())
            if len(tier_gate_means) > 1
            else 0.0
        ),
        "representative_client_rounds": len(selected),
        "representative_trajectories": len(by_trajectory),
    }


def _screen_completeness(
    round_rows: Sequence[Mapping[str, Any]],
    client_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    replace_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact preregistered identifiers, not only aggregate row counts."""

    expected_trajectory_ids = {
        str(cell["trajectory_id"]) for cell in _development_cells(config)
    }
    total_rounds = int(config["temporal"]["total_rounds"])
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    n = int(config["cohort"]["num_clients"])

    round_keys: set[tuple[str, str, int]] = set()
    trajectory_keys: set[tuple[str, str]] = set()
    client_keys: set[tuple[str, int, int]] = set()
    duplicate_round = duplicate_trajectory = duplicate_client = 0
    unexpected_round = unexpected_trajectory = unexpected_client = 0

    for row in round_rows:
        key = (
            str(row["trajectory_id"]),
            str(row["candidate"]),
            int(row["round"]),
        )
        valid = (
            key[0] in expected_trajectory_ids
            and key[1] in CANDIDATES
            and 1 <= key[2] <= total_rounds
        )
        if not valid:
            unexpected_round += 1
        elif key in round_keys:
            duplicate_round += 1
        else:
            round_keys.add(key)
    for row in trajectory_rows:
        key = (str(row["trajectory_id"]), str(row["candidate"]))
        valid = key[0] in expected_trajectory_ids and key[1] in CANDIDATES
        if not valid:
            unexpected_trajectory += 1
        elif key in trajectory_keys:
            duplicate_trajectory += 1
        else:
            trajectory_keys.add(key)
    for row in client_rows:
        key = (
            str(row["trajectory_id"]),
            int(row["round"]),
            int(row["client"]),
        )
        valid = (
            key[0] in expected_trajectory_ids
            and first_gate <= key[1] <= total_rounds
            and 0 <= key[2] < n
        )
        if not valid:
            unexpected_client += 1
        elif key in client_keys:
            duplicate_client += 1
        else:
            client_keys.add(key)

    expected_round = len(expected_trajectory_ids) * len(CANDIDATES) * total_rounds
    expected_trajectory = len(expected_trajectory_ids) * len(CANDIDATES)
    expected_client = len(expected_trajectory_ids) * (total_rounds - first_gate + 1) * n

    expected_replace_keys = {
        (int(seed), str(regime["name"]), str(permutation), trial)
        for seed in config["randomness"]["development_seeds"]
        for regime, permutation in base._noise_cells(config)
        for trial in range(
            int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
        )
    }
    replace_keys: set[tuple[int, str, str, int]] = set()
    duplicate_replace = unexpected_replace = 0
    for row in replace_rows:
        key = (
            int(row["seed"]),
            str(row["noise_regime"]),
            str(row["noise_permutation"]),
            int(row["trial"]),
        )
        if key not in expected_replace_keys:
            unexpected_replace += 1
        elif key in replace_keys:
            duplicate_replace += 1
        else:
            replace_keys.add(key)

    unique_fraction = min(
        len(round_keys) / float(expected_round),
        len(trajectory_keys) / float(expected_trajectory),
        len(client_keys) / float(expected_client),
        len(replace_keys) / float(len(expected_replace_keys)),
    )
    exact = bool(
        unique_fraction == 1.0
        and not any(
            (
                duplicate_round,
                duplicate_trajectory,
                duplicate_client,
                duplicate_replace,
                unexpected_round,
                unexpected_trajectory,
                unexpected_client,
                unexpected_replace,
            )
        )
    )
    return {
        "exact_identifier_completeness": exact,
        "complete_fraction": unique_fraction,
        "expected_trajectory_ids": len(expected_trajectory_ids),
        "missing_trajectory_ids": len(
            expected_trajectory_ids
            - {str(row["trajectory_id"]) for row in trajectory_rows}
        ),
        "round_rows": {
            "observed": len(round_rows),
            "expected": expected_round,
            "missing_unique_keys": expected_round - len(round_keys),
            "duplicate_keys": duplicate_round,
            "unexpected_rows": unexpected_round,
        },
        "trajectory_rows": {
            "observed": len(trajectory_rows),
            "expected": expected_trajectory,
            "missing_unique_keys": expected_trajectory - len(trajectory_keys),
            "duplicate_keys": duplicate_trajectory,
            "unexpected_rows": unexpected_trajectory,
        },
        "client_rows": {
            "observed": len(client_rows),
            "expected": expected_client,
            "missing_unique_keys": expected_client - len(client_keys),
            "duplicate_keys": duplicate_client,
            "unexpected_rows": unexpected_client,
        },
        "replace_one_rows": {
            "observed": len(replace_rows),
            "expected": len(expected_replace_keys),
            "missing_unique_keys": len(expected_replace_keys) - len(replace_keys),
            "duplicate_keys": duplicate_replace,
            "unexpected_rows": unexpected_replace,
        },
    }


def _evaluate_gates(
    round_rows: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    replace_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    device_name: str,
    temporal_c0: float,
) -> dict[str, Any]:
    gates = config["gates"]
    separated = {
        str(value) for value in config["threats"]["separated_for_primary_gate"]
    }
    completeness = _screen_completeness(
        round_rows, client_rows, trajectory_rows, replace_rows, config
    )
    expected_trajectories = int(completeness["expected_trajectory_ids"])
    finite_fraction = base._finite_mean(
        1.0 if bool(row["all_finite"]) else 0.0 for row in round_rows
    )
    certified = [row for row in round_rows if row["candidate"] != "fcc"]
    cap_violations = sum(
        not bool(row["contribution_cap_respected"]) for row in certified
    )
    gate_sum_normalization_violations = sum(
        bool(row["normalization_by_gate_sum"]) for row in certified
    )
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    identity_rows = [
        row
        for row in round_rows
        if row["candidate"] == PRIMARY and bool(row["temporal_gate_all_one"])
    ]
    identity_max = max(float(row["difference_to_k2_aware"]) for row in identity_rows)

    mutable_config = dict(config)
    mutable_config["_deployed_temporal_c0"] = temporal_c0
    false_triggers = _false_trigger_diagnostics(client_rows, mutable_config)
    representative = _representative_client_rows(client_rows, config)
    outlier_gate_drop = base._finite_mean(
        float(row["gate_drop_vs_k2_aware"])
        for row in representative
        if bool(row["honest_outlier"]) and not bool(row["latent_byzantine"])
    )

    def no_compromise(row: Mapping[str, Any]) -> bool:
        return bool(
            row["threat"] == config["no_compromise_control"]["threat_name"]
            and row["schedule"]
            == config["no_compromise_control"]["schedule_name"]
        )

    no_compromise_ratio = _paired_ratio(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=AWARE,
        predicate=no_compromise,
        metric="post_enrollment_auc",
    )

    def persistent_separated(row: Mapping[str, Any]) -> bool:
        return bool(row["schedule"] == "persistent" and row["threat"] in separated)

    def byzantine_high(row: Mapping[str, Any]) -> bool:
        return bool(
            persistent_separated(row)
            and row["noise_regime"] == "heteroscedastic"
            and row["noise_permutation"] == "byzantine_high"
        )

    def intermittent(row: Mapping[str, Any]) -> bool:
        return bool(row["schedule"] == "intermittent_2_on_1_off")

    def persistent_alie(row: Mapping[str, Any]) -> bool:
        return bool(row["schedule"] == "persistent" and row["threat"] == "alie")

    main_gain = _relative_gain(
        trajectory_rows, competitor=AWARE, predicate=persistent_separated
    )
    main_ci = base._ci95(
        _paired_seed_contrasts(
            trajectory_rows, competitor=AWARE, predicate=persistent_separated
        )
    )
    high_gain = _relative_gain(
        trajectory_rows, competitor=AWARE, predicate=byzantine_high
    )
    high_ci = base._ci95(
        _paired_seed_contrasts(
            trajectory_rows, competitor=AWARE, predicate=byzantine_high
        )
    )
    counterfactual_gain = _relative_gain(
        trajectory_rows, competitor=COUNTERFACTUAL, predicate=byzantine_high
    )
    counterfactual_ci = base._ci95(
        _paired_seed_contrasts(
            trajectory_rows, competitor=COUNTERFACTUAL, predicate=byzantine_high
        )
    )
    k3_ratio = _paired_ratio(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K3,
        predicate=persistent_separated,
        metric="attack_auc",
    )
    intermittent_ratio = _paired_ratio(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=AWARE,
        predicate=intermittent,
        metric="attack_auc",
    )
    intermittent_active_ratio = _paired_ratio(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=AWARE,
        predicate=intermittent,
        metric="active_attack_auc",
    )
    alie_ratio = _paired_ratio(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=AWARE,
        predicate=persistent_alie,
        metric="attack_auc",
    )
    control_ratio = max(intermittent_ratio, alie_ratio)
    byzantine_mass_reduction = _relative_gain(
        trajectory_rows,
        competitor=AWARE,
        predicate=persistent_separated,
        metric="attack_byzantine_contribution_mass",
    )
    primary_persistent = [
        row
        for row in trajectory_rows
        if row["candidate"] == PRIMARY and persistent_separated(row)
    ]
    detection_rate = base._finite_mean(
        row["detection_rate_within_deadline"] for row in primary_persistent
    )
    recovery_rate = base._finite_mean(
        row["recovery_rate_within_deadline"] for row in primary_persistent
    )
    complete_fraction = float(completeness["complete_fraction"])
    observed = {
        "device": device_name,
        "development_trajectories": expected_trajectories,
        "development_round_rows": len(round_rows),
        "expected_round_rows": completeness["round_rows"]["expected"],
        "development_trajectory_rows": len(trajectory_rows),
        "expected_trajectory_rows": completeness["trajectory_rows"]["expected"],
        "development_client_rows": len(client_rows),
        "expected_client_rows": completeness["client_rows"]["expected"],
        "complete_fraction": complete_fraction,
        "identifier_completeness": completeness,
        "finite_metric_fraction": finite_fraction,
        "contribution_cap_violations": cap_violations,
        "gate_sum_normalization_violations": gate_sum_normalization_violations,
        "replace_one_trials": len(replace_rows),
        "replace_one_violations": replace_violations,
        "replace_one_max_ratio_to_bound": max(
            float(row["ratio_observed_to_bound"]) for row in replace_rows
        ),
        "k4_k2_identity_max_abs_error": identity_max,
        "false_triggers": false_triggers,
        "honest_outlier_gate_drop_vs_k2_aware": outlier_gate_drop,
        "no_compromise_post_enrollment_error_ratio_to_k2_aware": (
            no_compromise_ratio
        ),
        "persistent_separated_attack_auc_gain_vs_k2_aware": main_gain,
        "persistent_separated_attack_auc_difference_seed_ci95": main_ci,
        "byzantine_high_attack_auc_gain_vs_k2_aware": high_gain,
        "byzantine_high_attack_auc_difference_seed_ci95": high_ci,
        "persistent_separated_byzantine_mass_reduction_vs_k2_aware": (
            byzantine_mass_reduction
        ),
        "persistent_separated_detection_rate_within_deadline": detection_rate,
        "persistent_separated_recovery_rate_within_deadline": recovery_rate,
        "persistent_separated_attack_auc_ratio_to_k3": k3_ratio,
        "byzantine_high_attack_auc_gain_vs_counterfactual_history": (
            counterfactual_gain
        ),
        "byzantine_high_attack_auc_difference_vs_counterfactual_seed_ci95": (
            counterfactual_ci
        ),
        "intermittent_or_alie_attack_auc_ratio_to_k2_aware": control_ratio,
        "intermittent_attack_auc_ratio_to_k2_aware": intermittent_ratio,
        "intermittent_active_attack_auc_ratio_to_k2_aware": (
            intermittent_active_ratio
        ),
        "persistent_alie_attack_auc_ratio_to_k2_aware": alie_ratio,
    }
    checks = {
        "complete": bool(completeness["exact_identifier_completeness"])
        and complete_fraction >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "production_device": device_name == str(gates["production_device_required"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "no_gate_sum_normalization": gate_sum_normalization_violations == 0,
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "k4_k2_identity_when_temporal_gate_one": identity_max
        <= float(gates["k4_k2_identity_when_temporal_gate_one_abs_error_max"]),
        "regular_client_round_false_trigger": false_triggers["client_round_rate"]
        <= float(gates["regular_client_round_false_trigger_rate_max"]),
        "regular_trajectory_false_trigger": false_triggers[
            "trajectory_rate"
        ]
        <= float(gates["regular_trajectory_false_trigger_rate_max"]),
        "pooled_regular_false_trigger_tier_gap": false_triggers["tier_rate_gap"]
        <= float(gates["pooled_regular_false_trigger_tier_gap_max"]),
        "pooled_regular_gate_mean_tier_gap": false_triggers[
            "tier_gate_mean_gap"
        ]
        <= float(gates["pooled_regular_gate_mean_tier_gap_max"]),
        "honest_outlier_gate_drop": outlier_gate_drop
        <= float(gates["honest_outlier_gate_drop_vs_k2_aware_max"]),
        "no_compromise_post_enrollment_noninferiority": no_compromise_ratio
        <= float(
            gates["no_compromise_post_enrollment_error_ratio_to_k2_aware_max"]
        ),
        "persistent_separated_gain": main_gain
        >= float(gates["persistent_separated_attack_auc_gain_vs_k2_aware_min"]),
        "persistent_separated_ci": float(main_ci["high"])
        <= float(
            gates["persistent_separated_attack_auc_difference_ci95_high_max"]
        ),
        "byzantine_high_gain": high_gain
        >= float(gates["byzantine_high_attack_auc_gain_vs_k2_aware_min"]),
        "byzantine_high_ci": float(high_ci["high"])
        <= float(gates["byzantine_high_attack_auc_difference_ci95_high_max"]),
        "byzantine_mass_reduction": byzantine_mass_reduction
        >= float(
            gates["persistent_separated_byzantine_mass_reduction_vs_k2_aware_min"]
        ),
        "detection": detection_rate
        >= float(gates["persistent_separated_detection_rate_within_deadline_min"]),
        "recovery": recovery_rate
        >= float(gates["persistent_separated_recovery_rate_within_deadline_min"]),
        "noninferiority_vs_k3": k3_ratio
        <= float(gates["persistent_separated_attack_auc_ratio_to_k3_max"]),
        "gain_vs_counterfactual_history": counterfactual_gain
        >= float(
            gates["byzantine_high_attack_auc_gain_vs_counterfactual_history_min"]
        ),
        "ci_vs_counterfactual_history": float(counterfactual_ci["high"])
        <= float(
            gates[
                "byzantine_high_attack_auc_difference_vs_counterfactual_ci95_high_max"
            ]
        ),
        "intermittent_or_alie_noninferiority": control_ratio
        <= float(gates["intermittent_or_alie_attack_auc_ratio_to_k2_aware_max"]),
    }
    return {
        "decision": (
            "promote_to_holdout" if all(checks.values()) else "stop_after_development"
        ),
        "all_gates_pass": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "holdout_opened": False,
    }


def _summaries(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in trajectory_rows:
        grouped[
            (
                str(row["candidate"]),
                str(row["noise_regime"]),
                str(row["threat"]),
                str(row["schedule"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (candidate, regime, threat, schedule), rows in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "noise_regime": regime,
                "threat": threat,
                "schedule": schedule,
                "n_trajectories": len(rows),
                "monitoring_auc_mean": base._finite_mean(
                    row["monitoring_auc"] for row in rows
                ),
                "attack_auc_mean": base._finite_mean(
                    row["attack_auc"] for row in rows
                ),
                "attack_auc_std": base._finite_std(
                    row["attack_auc"] for row in rows
                ),
                "active_attack_auc_mean": base._finite_mean(
                    row["active_attack_auc"] for row in rows
                ),
                "recovery_auc_mean": base._finite_mean(
                    row["recovery_auc"] for row in rows
                ),
                "post_enrollment_auc_mean": base._finite_mean(
                    row["post_enrollment_auc"] for row in rows
                ),
                "attack_byzantine_contribution_mass_mean": base._finite_mean(
                    row["attack_byzantine_contribution_mass"] for row in rows
                ),
            }
        )
    return result


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    decision: Mapping[str, Any],
) -> None:
    observed = decision["observed"]
    failed = [name for name, passed in decision["checks"].items() if not passed]
    lines = [
        "# G0g-K4-TCG — résultat développement",
        "",
        "> Ce rapport est généré après l'écran. Les fichiers CSV et JSON bruts "
        "restent les artefacts de preuve. Aucun holdout n'est ouvert par ce runner.",
        "",
        "## Décision",
        "",
        f"- Décision : **{decision['decision']}**.",
        f"- Tous les gates passent : **{decision['all_gates_pass']}**.",
        "- Holdout ouvert : **non**.",
        f"- Gates échoués : {', '.join(failed) if failed else 'aucun'}.",
        "",
        "## Contrastes principaux",
        "",
        "| Mesure | Valeur |",
        "|---|---:|",
        f"| Gain d'AUC attaque K4 vs K2-aware | {observed['persistent_separated_attack_auc_gain_vs_k2_aware']:.4f} |",
        f"| Borne haute IC95 de la différence | {observed['persistent_separated_attack_auc_difference_seed_ci95']['high']:.6f} |",
        f"| Gain byzantine-high | {observed['byzantine_high_attack_auc_gain_vs_k2_aware']:.4f} |",
        f"| Réduction de la masse Byzantine | {observed['persistent_separated_byzantine_mass_reduction_vs_k2_aware']:.4f} |",
        f"| Détection avant deadline | {observed['persistent_separated_detection_rate_within_deadline']:.4f} |",
        f"| Récupération avant deadline | {observed['persistent_separated_recovery_rate_within_deadline']:.4f} |",
        f"| Ratio AUC K4/K3 | {observed['persistent_separated_attack_auc_ratio_to_k3']:.4f} |",
        f"| Ratio propre post-enrôlement K4/K2-aware | {observed['no_compromise_post_enrollment_error_ratio_to_k2_aware']:.4f} |",
        f"| Gain K4 vs historique contrefactuel no-compromise | {observed['byzantine_high_attack_auc_gain_vs_counterfactual_history']:.4f} |",
        f"| Ratio intermittent, tours d'attaque actifs uniquement | {observed['intermittent_active_attack_auc_ratio_to_k2_aware']:.4f} |",
        "",
        "## Certificats et intégrité",
        "",
        f"- device observé : `{observed['device']}`; requis : `mps`;",
        f"- violations du cap global : {observed['contribution_cap_violations']};",
        f"- violations replace-one courant : {observed['replace_one_violations']} sur {observed['replace_one_trials']};",
        f"- maximum observé / borne $2G/n$ : {observed['replace_one_max_ratio_to_bound']:.6f};",
        f"- complétude exacte par identifiants : {observed['identifier_completeness']['exact_identifier_completeness']};",
        "- le certificat reste conditionnel au même passé, à la même ancre et aux mêmes covariances;",
        "- seul le facteur historique est mesurable par le passé; le gate final dépend encore de l'upload courant via K2-aware.",
        "",
        "## Portée",
        "",
        "K4 est ici un écran synthétique de référence, sous identités persistantes "
        "et compromis post-enrôlement. Ce résultat ne porte pas encore sur "
        "l'accuracy, la fairness ou une sensibilité de trajectoire utilisateur.",
        "",
        f"Résultats bruts : `{output_dir.relative_to(ROOT)}`.",
        f"Holdout réservé : `{config['randomness']['holdout_seeds']}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    k2_calibration, k2_provenance = _load_frozen_k2_calibration(config)
    device, dtype = oracle._configure_runtime("mps")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": base._sha256(config_path),
        "source_sha256": {
            "runner": base._sha256(Path(__file__).resolve()),
            "algorithm": base._sha256(ROOT / "algorithms/gaussian_aware_reference.py"),
            "shared_k1_runner": base._sha256(
                ROOT / "scripts/run_gaussian_aware_reference_g0g_k1.py"
            ),
            "shared_k2_runner": base._sha256(
                ROOT / "scripts/run_gaussian_aware_reference_g0g_k2.py"
            ),
            "shared_k3_runner": base._sha256(
                ROOT / "scripts/run_gaussian_aware_reference_g0g_k3.py"
            ),
            "oracle_runner": base._sha256(
                ROOT / "scripts/run_gaussian_aware_reference_oracle.py"
            ),
            "robust_aggregators": base._sha256(
                ROOT / "robustness/aggregators.py"
            ),
        },
        "device": str(device),
        "dtype": str(dtype),
        "mps_required_without_fallback": True,
        "k2_calibration_sha256_verified": k2_provenance["sha256_verified"],
        "holdout_rule": config["randomness"]["holdout_rule"],
        "reserved_holdout_seeds": config["randomness"]["holdout_seeds"],
        "holdout_opened": False,
        "status": "running_temporal_calibration",
    }
    base._atomic_json(output_dir / "manifest.json", manifest)
    base._atomic_json(output_dir / "k2_calibration_provenance.json", k2_provenance)
    try:
        temporal_calibration, temporal_calibration_rows = _calibrate_temporal(config)
        base._atomic_json(
            output_dir / "temporal_calibration.json", temporal_calibration
        )
        base._write_csv(
            output_dir / "temporal_calibration_trajectories.csv",
            temporal_calibration_rows,
        )
        manifest["status"] = "running_development_screen"
        manifest["temporal_c0"] = temporal_calibration["deployed_c0"]
        manifest["temporal_c1"] = temporal_calibration["deployed_c1"]
        base._atomic_json(output_dir / "manifest.json", manifest)
        round_rows, client_rows, trajectory_rows = _screen_development(
            config, k2_calibration, temporal_calibration
        )
        replace_rows = _replace_one_audit(
            config, k2_calibration, temporal_calibration
        )
        decision = _evaluate_gates(
            round_rows,
            client_rows,
            trajectory_rows,
            replace_rows,
            config,
            device_name=str(device),
            temporal_c0=float(temporal_calibration["deployed_c0"]),
        )
        summaries = _summaries(trajectory_rows)
        base._write_csv(output_dir / "development_round_rows.csv", round_rows)
        base._write_csv(output_dir / "development_client_rows.csv", client_rows)
        base._write_csv(
            output_dir / "development_trajectory_rows.csv", trajectory_rows
        )
        base._write_csv(output_dir / "replace_one_audit.csv", replace_rows)
        base._write_csv(output_dir / "summary.csv", summaries)
        base._atomic_json(output_dir / "decision.json", decision)
        manifest["status"] = "completed_development"
        manifest["development_decision"] = decision["decision"]
        manifest["all_development_gates_pass"] = decision["all_gates_pass"]
        manifest["holdout_opened"] = False
        base._atomic_json(output_dir / "manifest.json", manifest)
        _write_report(
            report_path,
            config=config,
            output_dir=output_dir,
            decision=decision,
        )
        return decision
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure_type"] = type(error).__name__
        manifest["failure_message"] = str(error)
        manifest["holdout_opened"] = False
        base._atomic_json(output_dir / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_G0g_K4_TCG_Report.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
