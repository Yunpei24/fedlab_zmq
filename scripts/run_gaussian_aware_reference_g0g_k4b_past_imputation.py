#!/usr/bin/env python3
"""Run the preregistered G0g-K4b temporal-mixture screen on MPS.

This is a development-only runner.  It has no code path that evaluates the
reserved holdout seeds and never recalibrates either K2 or K4 thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k4b import (  # noqa: E402
    FULL_TEMPORAL_MISSING_SLOT,
    INCREMENTAL_TEMPORAL_SUPPRESSION,
    fixed_denominator_past_predictor,
    gaussian_aware_fixed_anchor_past_imputed_reference,
    pointwise_optimal_full_imputation_predictor,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k1 as base  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

FCC = "fcc"
K2 = "g0g_k2"
K3 = "g0g_k3_dual_gate"
K4 = "g0g_k4_temporal_causal_gate"
PRIMARY = "g0g_k4b_full_temporal_missing_slot_imputation"
DELTA = "g0g_k4b_incremental_temporal_suppression_imputation"
STATIC = "g0g_k4b_static_clean_window_full_imputation"
ORACLE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (FCC, K2, K3, K4, PRIMARY, DELTA, STATIC, ORACLE)
CERTIFIED_CANDIDATES = (K2, K3, K4, PRIMARY, DELTA, STATIC, ORACLE)
FROZEN_K2_SHA256 = "98a30cad8a2f92e6843346e9a9512bd321e8302b9b6e17b447b2641fffb544a5"
FROZEN_K4_TEMPORAL_SHA256 = (
    "6ca81b5b8897ec19509a569b54fdf9edeac8c6dc65c7b0c4a890142882603075"
)
FROZEN_C0 = 10.823704719543457
FROZEN_C1 = 11.615572929382324
DEFAULT_CONFIG_PATH = (
    ROOT
    / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml"
)
DEFAULT_LOCK_PATH = (
    ROOT
    / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.lock.json"
)
LOCKED_K4B_PATHS = {
    "algorithms/gaussian_aware_reference_k4b.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml",
    "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "tests/test_gaussian_aware_reference_g0g_k4b.py",
    "tests/test_run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "output/analysis/Gaussian_Aware_G0g_K4b_Past_Imputation_Protocol_PreRun.md",
}
LOCKED_DEPENDENCY_PATHS = {
    "algorithms/gaussian_aware_reference.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k1.yaml",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k2.yaml",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k3.yaml",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg.yaml",
    "scripts/run_gaussian_aware_reference_g0g_k1.py",
    "scripts/run_gaussian_aware_reference_g0g_k2.py",
    "scripts/run_gaussian_aware_reference_g0g_k3.py",
    "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
    "scripts/run_gaussian_aware_reference_oracle.py",
    "robustness/aggregators.py",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1/calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1/temporal_calibration.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify_preregistration_lock(
    lock_path: Path,
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Verify the immutable K4b registry before any output path is created."""

    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing K4b preregistration lock: {lock_path}")
    try:
        registry = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Invalid K4b preregistration lock: {lock_path}") from error
    expected_top = {
        "schema_version",
        "campaign_id",
        "locked_files",
        "dependencies",
        "lock_file_self_hash_embedded",
        "publication_requirement",
    }
    if set(registry) != expected_top:
        raise RuntimeError("K4b preregistration-lock schema mismatch")
    if registry["schema_version"] != 1 or registry["campaign_id"] != (
        "gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
    ):
        raise RuntimeError("K4b preregistration-lock identity mismatch")
    if registry["lock_file_self_hash_embedded"] is not False:
        raise RuntimeError("The preregistration lock must not claim a self-hash")
    if registry["publication_requirement"] != (
        "publish_this_lock_file_sha256_in_research_log_or_chat_before_run"
    ):
        raise RuntimeError("K4b lock publication requirement changed")
    locked = registry["locked_files"]
    dependencies = registry["dependencies"]
    if not isinstance(locked, Mapping) or set(locked) != LOCKED_K4B_PATHS:
        raise RuntimeError("K4b locked-file registry mismatch")
    if not isinstance(dependencies, Mapping) or set(dependencies) != (
        LOCKED_DEPENDENCY_PATHS
    ):
        raise RuntimeError("K4b dependency registry mismatch")
    configured_path = (
        ROOT
        / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml"
    ).resolve()
    if config_path.resolve() != configured_path:
        raise RuntimeError("K4b runner accepts only the preregistered config path")
    for relative, expected_hash in {**locked, **dependencies}.items():
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing preregistered file: {relative}")
        observed_hash = _sha256(path)
        if observed_hash != str(expected_hash):
            raise RuntimeError(
                f"Preregistration hash mismatch for {relative}: "
                f"{observed_hash} != {expected_hash}"
            )
    return {
        "path": str(lock_path.resolve()),
        "sha256": _sha256(lock_path),
        "verified": True,
        "locked_files": len(locked),
        "dependencies": len(dependencies),
    }


def _seed_values(value: Any, *, parent_key: str = "") -> set[int]:
    """Collect registered RNG roots/seeds from one prior campaign config."""

    result: set[int] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "seed" in key_text or key_text == "roots":
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    for element in item:
                        if isinstance(element, int) and not isinstance(element, bool):
                            result.add(int(element))
                        else:
                            result |= _seed_values(element, parent_key=key_text)
                elif isinstance(item, int) and not isinstance(item, bool):
                    result.add(int(item))
            result |= _seed_values(item, parent_key=key_text)
    elif (
        isinstance(value, int)
        and not isinstance(value, bool)
        and ("seed" in parent_key or parent_key == "roots")
    ):
        result.add(int(value))
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id",
        "scope",
        "preregistration_lock",
        "scientific_contract",
        "frozen_calibrations",
        "frozen_source_hashes",
        "prior_campaign_seed_registry",
        "cohort",
        "privacy_noise",
        "references",
        "temporal",
        "past_imputation",
        "honest_dynamics",
        "aggregation",
        "threats",
        "no_compromise_control",
        "randomness",
        "candidates",
        "gates",
        "execution",
    }
    if set(config) != expected:
        raise ValueError("The frozen G0g-K4b top-level schema changed")
    if (
        config["campaign_id"]
        != "gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1"
    ):
        raise ValueError("Unexpected G0g-K4b campaign id")
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
        "screen_conditioned_on_semi_oracle_anchor": True,
        "end_to_end_deployability_claimed": False,
        "predictor_deployability": ("conditional_on_supplied_past_or_public_anchor"),
        "anchor_is_past_measurable": True,
        "covariance_role": "authenticated_dp_covariance_only",
        "history_gate_uses_current_upload": False,
        "history_gate_is_past_measurable": True,
        "final_gate_uses_current_upload": True,
        "final_gate_is_past_measurable": False,
        "final_gate_rule": "elementwise_minimum_k2_aware_and_frozen_k4_temporal",
        "aggregate_denominator": "public_cohort_size_n",
        "gate_sum_normalization": False,
        "primary_mechanism": (
            "historical_confidence_mixture_full_temporal_missing_slot_imputation"
        ),
        "primary_imputation_rule": "one_minus_history_gate",
        "incremental_ablation_rule": "current_gate_minus_joint_gate",
        "predictor_feedback_from_k4b_output": False,
        "predictor_randomness": "none",
        "influence_cap_depends_on_covariance": False,
        "inverse_variance_weighting": False,
        "global_l2_cap": True,
        "current_cohort_replace_one_only": True,
        "user_trajectory_sensitivity_claimed": False,
        "development_only_screen": True,
    }
    if contract != required_contract:
        raise ValueError("The frozen G0g-K4b scientific contract changed")
    if config["preregistration_lock"] != {
        "path": (
            "configs/ldp_gradient_far/"
            "gaussian_aware_reference_g0g_k4b_past_imputation.lock.json"
        ),
        "verify_before_output_creation": True,
        "publish_lock_sha256_before_scientific_run": True,
    }:
        raise ValueError("The K4b preregistration-lock contract changed")

    frozen = config["frozen_calibrations"]
    if set(frozen) != {"k2", "k4_temporal", "reuse_thresholds_exactly", "recalibrate"}:
        raise ValueError("The frozen calibration registry changed")
    if not bool(frozen["reuse_thresholds_exactly"]) or bool(frozen["recalibrate"]):
        raise ValueError("K4b must reuse K2/K4 thresholds without recalibration")
    if frozen["k2"]["sha256"] != FROZEN_K2_SHA256:
        raise ValueError("Unexpected frozen K2 hash")
    temporal_frozen = frozen["k4_temporal"]
    if temporal_frozen["sha256"] != FROZEN_K4_TEMPORAL_SHA256:
        raise ValueError("Unexpected frozen K4 temporal hash")
    if not math.isclose(
        float(temporal_frozen["deployed_c0"]), FROZEN_C0, abs_tol=1e-12
    ):
        raise ValueError("Frozen temporal c0 changed")
    if not math.isclose(
        float(temporal_frozen["deployed_c1"]), FROZEN_C1, abs_tol=1e-12
    ):
        raise ValueError("Frozen temporal c1 changed")
    expected_source_hashes = {
        "algorithms/gaussian_aware_reference.py": (
            "35b352d8a550fcbf75a71a7d64e74029d613f3548609469ee3727d7f11efb15f"
        ),
        "scripts/run_gaussian_aware_reference_g0g_k1.py": (
            "ed0b2865d964ed77c2495e31f9a614f4e98d7312cce6a322287d392326e4133d"
        ),
        "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py": (
            "4da1338e86ae9f6c587c181f4ca4738c1f08c9c1f7a27f4b3dbd20a64c3f2bd6"
        ),
        "scripts/run_gaussian_aware_reference_g0g_k2.py": (
            "1b4b2becc927b7e207ee63561cb56b2dde05682f369c38992a08c707500de818"
        ),
        "scripts/run_gaussian_aware_reference_g0g_k3.py": (
            "0725ce5392cc5633a6e440fba9b4b29039f7ed23bde87439851614a56960009b"
        ),
        "scripts/run_gaussian_aware_reference_oracle.py": (
            "913aa40897c272c5c1995c9a1aac7674904c0abe7b7a297d0944d18c10228974"
        ),
        "robustness/aggregators.py": (
            "1918b6ad9301c46d6ce8d5813320f322ff24b7db4adef31c2bc97de27794257f"
        ),
    }
    if config["frozen_source_hashes"] != expected_source_hashes:
        raise ValueError("The frozen K2/K3/K4 comparator source registry changed")
    for relative, expected_hash in expected_source_hashes.items():
        observed_hash = _sha256(ROOT / relative)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Frozen comparator source mismatch for {relative}: "
                f"{observed_hash} != {expected_hash}"
            )

    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    blocks = tuple(int(value) for value in cohort["block_sizes"])
    if n != 25 or b != 5 or not 0 <= b < n / 2:
        raise ValueError("The preregistered K4b cohort is n=25, b=5")
    if sum(blocks) != int(cohort["dimension"]) or any(value <= 0 for value in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity value is required per block")
    noise = config["privacy_noise"]
    if len(noise["block_std_multipliers"]) != len(blocks):
        raise ValueError("One DP-noise multiplier is required per block")
    if float(noise["base_std"]) <= 0.0:
        raise ValueError("base_std must be positive")
    for regime in noise["regimes"]:
        if any(float(value) <= 0.0 for value in regime["client_std_multipliers"]):
            raise ValueError("Authenticated DP-noise tiers must be positive")

    temporal = config["temporal"]
    expected_temporal = {
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
    if temporal != expected_temporal:
        raise ValueError("K4b must reproduce the frozen K4 temporal contract")
    expected_clip = 2.0 * math.sqrt(float(cohort["dimension"]))
    if not math.isclose(float(temporal["standardized_clip_norm"]), expected_clip):
        raise ValueError("The standardized temporal cap must equal 2*sqrt(d)")

    imputation = config["past_imputation"]
    required_imputation = {
        "history_length": 4,
        "maximum_byzantine_clients": 5,
        "minimum_accepted_mass": 20,
        "per_round_predictor": "clip_G_of_sum_g_c_over_max_m_min_sum_g",
        "window_predictor": "clip_G_of_mean_per_round_predictors",
        "primary_mechanism": (
            "historical_confidence_mixture_full_temporal_missing_slot_imputation"
        ),
        "primary_formula": "g_times_c_plus_one_minus_h_times_p",
        "exact_incremental_ablation": ("incremental_temporal_suppression_imputation"),
        "exact_incremental_formula": "g_times_c_plus_gamma_minus_g_times_p",
        "rolling_predictor": "rolling_strictly_prior_rounds",
        "static_ablation_predictor": (
            "fixed_last_clean_window_rounds_9_to_12_full_imputation"
        ),
        "static_clean_window": [9, 10, 11, 12],
        "oracle_predictor": "pointwise_optimal_full_imputation_projected_to_B_G",
        "oracle_formula": (
            "Proj_BG_of_(n_times_(target_minus_anchor)_minus_sum_g_c)"
            "_over_sum_(one_minus_h)"
        ),
        "oracle_deployable": False,
        "oracle_privacy_claimed": False,
        "direct_current_mass_definition": "norm_of_g_i_c_i",
        "imputed_mass_definition": "norm_of_imputation_coefficient_times_p",
        "total_slot_mass_definition": ("norm_of_direct_plus_imputed_descriptive_only"),
        "predictor_error_target": (
            "current_honest_mean_of_clipped_residuals_evaluation_only"
        ),
        "historical_contamination_counterfactual_available": False,
    }
    if imputation != required_imputation:
        raise ValueError("The preregistered K4b imputation rule changed")
    if int(imputation["minimum_accepted_mass"]) != n - int(
        imputation["maximum_byzantine_clients"]
    ):
        raise ValueError("m_min must equal n-b_max")
    if int(imputation["history_length"]) != int(temporal["history_window"]):
        raise ValueError("Predictor and temporal gate must use the same L=4")

    if tuple(str(value) for value in config["candidates"]["names"]) != CANDIDATES:
        raise ValueError("The frozen K4b candidate matrix changed")
    if config["candidates"]["primary"] != PRIMARY:
        raise ValueError("Unexpected K4b primary candidate")
    if config["candidates"]["exact_incremental_ablation"] != DELTA:
        raise ValueError("Unexpected exact incremental K4b ablation")
    if config["candidates"]["static_clean_window_ablation"] != STATIC:
        raise ValueError("Unexpected K4b static ablation")
    if config["candidates"]["oracle_diagnostic"] != ORACLE:
        raise ValueError("Unexpected K4b oracle diagnostic")
    if config["candidates"]["oracle_metadata"] != {
        "deployable": False,
        "privacy_claimed": False,
        "uses_target": True,
    }:
        raise ValueError("Oracle metadata must remain explicit")
    if config["execution"] != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
        "development_only": True,
        "holdout_code_path_present": False,
    }:
        raise ValueError("K4b production is MPS-only and development-only")

    randomness = config["randomness"]
    if set(randomness) != {
        "development_seeds",
        "holdout_seeds",
        "holdout_rule",
        "replace_one_trials_per_seed_noise_cell",
        "common_random_numbers",
        "no_new_predictor_rng",
    }:
        raise ValueError("The frozen K4b randomness schema changed")
    development = [int(value) for value in randomness["development_seeds"]]
    holdout = [int(value) for value in randomness["holdout_seeds"]]
    if len(development) != 5 or len(set(development)) != 5:
        raise ValueError("K4b requires five unique development seeds")
    if len(holdout) != 7 or len(set(holdout)) != 7:
        raise ValueError("K4b requires seven unique holdout seeds")
    if set(development) & set(holdout):
        raise ValueError("Development and holdout seeds overlap")
    if randomness["holdout_rule"] != (
        "inaccessible_to_development_runner_even_if_all_gates_pass"
    ):
        raise ValueError("The K4b development runner must never open holdout")
    if randomness["common_random_numbers"] != (
        "strict_across_candidates_within_trajectory"
    ) or not bool(randomness["no_new_predictor_rng"]):
        raise ValueError("K4b requires strict CRN and no predictor RNG")
    prior_seeds: set[int] = set()
    for registered in config["prior_campaign_seed_registry"]:
        path = _as_path(str(registered))
        if not path.is_file():
            raise ValueError(f"Missing prior seed registry: {path}")
        prior_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        prior_seeds |= _seed_values(prior_config)
    overlap = (set(development) | set(holdout)) & prior_seeds
    if overlap:
        raise ValueError(f"K4b seeds overlap K1--K4 registries: {sorted(overlap)}")

    if config["threats"]["schedules"] != [
        "persistent",
        "intermittent_2_on_1_off",
    ]:
        raise ValueError("The frozen K4 threat schedules changed")
    if config["threats"]["intermittent_pattern"] != [True, True, False]:
        raise ValueError("The frozen intermittent pattern changed")
    if config["no_compromise_control"] != {
        "threat_name": "none",
        "schedule_name": "no_compromise",
        "one_trajectory_per_seed_noise_geometry_dynamics": True,
        "active_history_rounds": [13, 36],
    }:
        raise ValueError("The frozen no-compromise control changed")
    cap = float(config["references"]["total_client_influence_cap"])
    if not math.isclose(cap, 0.13, abs_tol=1e-12):
        raise ValueError("The frozen influence cap changed")
    if not math.isclose(2.0 * cap / n, 0.0104, abs_tol=1e-12):
        raise ValueError("The preregistered current replace-one bound changed")


def _load_json_with_hash(
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _as_path(str(entry["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Frozen calibration is missing: {path}")
    observed = _sha256(path)
    expected = str(entry["sha256"])
    if observed != expected:
        raise RuntimeError(
            f"Frozen calibration hash mismatch for {path}: {observed} != {expected}"
        )
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    return value, {
        "path": str(path.relative_to(ROOT)),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "sha256_verified": True,
    }


def _load_frozen_calibrations(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_calibrations"]
    k2_calibration, k2_provenance = _load_json_with_hash(frozen["k2"])
    temporal, temporal_provenance = _load_json_with_hash(frozen["k4_temporal"])
    if not math.isclose(float(temporal["deployed_c0"]), FROZEN_C0, abs_tol=1e-12):
        raise RuntimeError("Observed K4 temporal c0 differs from preregistration")
    if not math.isclose(float(temporal["deployed_c1"]), FROZEN_C1, abs_tol=1e-12):
        raise RuntimeError("Observed K4 temporal c1 differs from preregistration")
    return (
        k2_calibration,
        temporal,
        {
            "k2": k2_provenance,
            "k4_temporal": temporal_provenance,
            "recalibrated": False,
        },
    )


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
    cells: list[dict[str, Any]] = []
    threats = [str(value) for value in config["threats"]["names"]]
    schedules = [str(value) for value in config["threats"]["schedules"]]
    clean_threat = str(config["no_compromise_control"]["threat_name"])
    clean_schedule = str(config["no_compromise_control"]["schedule_name"])
    for seed_value in config["randomness"]["development_seeds"]:
        seed = int(seed_value)
        for regime, permutation in base._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for dynamics in config["honest_dynamics"]["names"]:
                    base_cell = {
                        "seed": seed,
                        "regime": regime,
                        "permutation": str(permutation),
                        "geometry": str(geometry),
                        "dynamics": str(dynamics),
                    }
                    for threat in threats:
                        for schedule in schedules:
                            cell = dict(base_cell, threat=threat, schedule=schedule)
                            cell["trajectory_id"] = _trajectory_id(
                                seed=seed,
                                regime_name=str(regime["name"]),
                                permutation=str(permutation),
                                geometry=str(geometry),
                                dynamics=str(dynamics),
                                threat=threat,
                                schedule=schedule,
                            )
                            cells.append(cell)
                    clean = dict(
                        base_cell,
                        threat=clean_threat,
                        schedule=clean_schedule,
                    )
                    clean["trajectory_id"] = _trajectory_id(
                        seed=seed,
                        regime_name=str(regime["name"]),
                        permutation=str(permutation),
                        geometry=str(geometry),
                        dynamics=str(dynamics),
                        threat=clean_threat,
                        schedule=clean_schedule,
                    )
                    cells.append(clean)
    identifiers = [str(cell["trajectory_id"]) for cell in cells]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("The K4b development matrix contains duplicate IDs")
    if len(cells) != 900:
        raise RuntimeError(
            f"The K4b matrix must contain 900 trajectories, got {len(cells)}"
        )
    return cells


def _clipped_residuals(
    vectors: torch.Tensor, anchor: torch.Tensor, cap: float
) -> torch.Tensor:
    residuals = vectors - anchor[None, :]
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("K4b residual subtraction overflowed")
    return clip_l2(residuals, cap)


def _predictor_from_rounds(
    residuals_by_round: Mapping[int, torch.Tensor],
    gates_by_round: Mapping[int, torch.Tensor],
    rounds: Sequence[int],
    *,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected = tuple(int(value) for value in rounds)
    if len(selected) != int(config["past_imputation"]["history_length"]):
        raise ValueError("K4b predictor requires exactly L=4 prior rounds")
    if any(
        value not in residuals_by_round or value not in gates_by_round
        for value in selected
    ):
        raise RuntimeError("K4b predictor requested an unavailable prior round")
    predictor, diagnostics = fixed_denominator_past_predictor(
        torch.stack([residuals_by_round[value] for value in selected]),
        torch.stack([gates_by_round[value] for value in selected]),
        minimum_accepted_mass=float(config["past_imputation"]["minimum_accepted_mass"]),
        influence_cap=float(config["references"]["total_client_influence_cap"]),
        return_diagnostics=True,
    )
    diagnostics["source_rounds"] = list(selected)
    diagnostics["maximum_source_round"] = max(selected)
    diagnostics["source_gate_rule"] = "frozen_k4_minimum_current_and_history_gate"
    diagnostics["feedback_from_k4b_output"] = False
    return predictor, diagnostics


def _k4b_reference(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
    predictor: torch.Tensor,
    predictor_role: str,
    imputation_mode: str,
    deployable: bool,
    privacy_claimed: bool,
    temporal_calibration: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    window = int(config["temporal"]["history_window"])
    return gaussian_aware_fixed_anchor_past_imputed_reference(
        vectors,
        anchor=anchor,
        statistical_radii=aware_radii,
        temporal_standardized_history=torch.stack(list(history[-window:]), dim=1),
        enrollment_standardized_mean=enrollment_mean,
        enrollment_size=int(config["temporal"]["enrollment_rounds"]),
        temporal_gate_inner_threshold=float(temporal_calibration["deployed_c0"]),
        temporal_gate_outer_threshold=float(temporal_calibration["deployed_c1"]),
        predictor=predictor,
        block_sizes=tuple(int(value) for value in config["cohort"]["block_sizes"]),
        influence_cap=float(config["references"]["total_client_influence_cap"]),
        current_gate_transition_width=float(
            config["references"]["current_gate_transition_width"]
        ),
        predictor_role=predictor_role,
        imputation_mode=imputation_mode,
        deployable=deployable,
        privacy_claimed=privacy_claimed,
        return_diagnostics=True,
    )


def _frozen_k4_comparator(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
    temporal_calibration: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Delegate exactly to the frozen K4 implementation and thresholds."""

    return k4._k4_reference(
        vectors,
        anchor=anchor,
        aware_radii=aware_radii,
        history=history,
        enrollment_mean=enrollment_mean,
        thresholds=temporal_calibration,
        config=config,
    )


def _candidate_metadata(candidate: str) -> tuple[bool, bool]:
    """Return end-to-end deployability and privacy-claim flags.

    Every candidate is conditioned on the supplied semi-oracle anchor in this
    isolation screen, hence none is claimed end-to-end deployable.  Only the
    pointwise target oracle also forfeits the post-processing privacy claim.
    """

    if candidate == ORACLE:
        return False, False
    return False, True


def _component_mass(
    diagnostics: Mapping[str, Any],
    latent_byzantine: torch.Tensor,
    norm_key: str,
) -> dict[str, float]:
    """Return component mass and Byzantine share without conflating components."""

    values = diagnostics.get(norm_key)
    if values is None and norm_key in {
        "direct_current_contribution_norms",
        "total_slot_contribution_norms",
    }:
        values = diagnostics.get("client_contribution_norms")
    if values is None and norm_key == "imputed_contribution_norms":
        values = [0.0] * int(latent_byzantine.numel())
    if values is None:
        return {"total": float("nan"), "byzantine": float("nan"), "share": float("nan")}
    norms = torch.tensor(
        values,
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    )
    if norms.shape != latent_byzantine.shape or not bool(torch.isfinite(norms).all()):
        raise ValueError(f"Invalid component norms for {norm_key}")
    total = float(norms.sum().item())
    byzantine = float(norms[latent_byzantine].sum().item())
    return {
        "total": total,
        "byzantine": byzantine,
        "share": byzantine / total if total > 0.0 else 0.0,
    }


def _identity_diagnostics(
    source: Mapping[str, Any], n: int, *, candidate: str
) -> dict[str, Any]:
    result = k4._temporal_identity_diagnostics(source, n)
    result.update(
        {
            "predictor_role": "inactive_before_first_causal_gate",
            "predictor_deployable": False,
            "predictor_deployability": (
                "conditional_on_supplied_past_or_public_anchor"
                if candidate != ORACLE
                else "nondeployable_pointwise_target_oracle"
            ),
            "screen_conditioned_on_semi_oracle_anchor": True,
            "end_to_end_deployability_claimed": False,
            "privacy_claimed": candidate != ORACLE,
            "predictor_norm": 0.0,
            "imputation_coefficient_mean": 0.0,
            "coefficient_sum_max": max(
                float(value) for value in result["gates_by_client"]
            ),
            "client_contribution_cap_respected": bool(
                source.get("client_contribution_cap_respected", True)
            ),
            "direct_current_contribution_norms": list(
                source.get("client_contribution_norms", [0.0] * n)
            ),
            "imputed_contribution_norms": [0.0] * n,
            "total_slot_contribution_norms": list(
                source.get("client_contribution_norms", [0.0] * n)
            ),
        }
    )
    return result


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
    components = k4._trajectory_components(
        config,
        seed=seed,
        regime=regime,
        permutation=permutation,
        geometry=geometry,
        dynamics=dynamics,
    )
    latent_byzantine = torch.zeros(n, dtype=torch.bool, device=oracle._RUNTIME_DEVICE)
    latent_byzantine[n - b :] = True
    honest = ~latent_byzantine
    regular = honest & ~components["outliers"]
    aware_radii = k4._radii(
        config,
        components["variances"],
        k2_calibration,
        regime_name=str(regime["name"]),
        blind=False,
    )
    blind_radii = k4._radii(
        config,
        components["variances"],
        k2_calibration,
        regime_name=str(regime["name"]),
        blind=True,
    )
    history: list[torch.Tensor] = []
    enrollment_mean: torch.Tensor | None = None
    residuals_by_round: dict[int, torch.Tensor] = {}
    gates_by_round: dict[int, torch.Tensor] = {}
    static_predictor: torch.Tensor | None = None
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
        clean = k4._clean_at_round(components, round_index)
        phase = k4._phase(config, round_index)
        target = k4._target_at_round(
            clean,
            latent_byzantine,
            phase=phase,
            compromised_during_attack_phase=threat != "none",
        )
        observed = k4._private_vectors(
            config,
            components,
            seed=seed,
            round_index=round_index,
            geometry=geometry,
        )
        attack_active = k4._attack_active(
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
        standardized = k4._standardized_messages(
            config,
            vectors,
            anchor=components["anchor"],
            variances=components["variances"],
        )
        if round_index == int(temporal["enrollment_rounds"]):
            enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)

        outputs: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        outputs[FCC], diagnostics[FCC] = k4._current_reference(
            FCC,
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            blind_radii=blind_radii,
            config=config,
        )
        outputs[K2], diagnostics[K2] = k4._current_reference(
            K2,
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            blind_radii=blind_radii,
            config=config,
        )
        outputs[K3], diagnostics[K3] = k4._current_reference(
            K3,
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            blind_radii=blind_radii,
            config=config,
        )

        first_gate = int(temporal["first_temporal_gate_round"])
        predictor_diagnostics: dict[str, dict[str, Any]] = {}
        k4_reproduction_error = 0.0
        if round_index < first_gate:
            outputs[K4] = outputs[K2].clone()
            diagnostics[K4] = k4._temporal_identity_diagnostics(diagnostics[K2], n)
            for candidate in (PRIMARY, DELTA, STATIC, ORACLE):
                outputs[candidate] = outputs[K2].clone()
                diagnostics[candidate] = _identity_diagnostics(
                    diagnostics[K2], n, candidate=candidate
                )
                predictor_diagnostics[candidate] = {
                    "source_rounds": [],
                    "maximum_source_round": None,
                }
        else:
            if enrollment_mean is None:
                raise RuntimeError("The clean enrollment baseline is missing")
            if len(history) < int(temporal["history_window"]):
                raise RuntimeError("The strictly prior temporal history is incomplete")
            outputs[K4], diagnostics[K4] = _frozen_k4_comparator(
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                history=history,
                enrollment_mean=enrollment_mean,
                temporal_calibration=temporal_calibration,
                config=config,
            )
            reproduced_k4, _ = k4._k4_reference(
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                history=history,
                enrollment_mean=enrollment_mean,
                thresholds=temporal_calibration,
                config=config,
            )
            k4_reproduction_error = float(
                torch.linalg.vector_norm(outputs[K4] - reproduced_k4).item()
            )
            prior_rounds = tuple(
                range(
                    round_index - int(config["past_imputation"]["history_length"]),
                    round_index,
                )
            )
            rolling_predictor, rolling_diagnostic = _predictor_from_rounds(
                residuals_by_round,
                gates_by_round,
                prior_rounds,
                config=config,
            )
            if static_predictor is None:
                static_predictor, static_diagnostic = _predictor_from_rounds(
                    residuals_by_round,
                    gates_by_round,
                    config["past_imputation"]["static_clean_window"],
                    config=config,
                )
            else:
                static_diagnostic = {
                    "source_rounds": list(
                        config["past_imputation"]["static_clean_window"]
                    ),
                    "maximum_source_round": max(
                        config["past_imputation"]["static_clean_window"]
                    ),
                }
            current_clipped = _clipped_residuals(
                vectors,
                components["anchor"],
                float(config["references"]["total_client_influence_cap"]),
            )
            k4_final_gates = torch.tensor(
                diagnostics[K4]["gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )
            k4_history_gates = torch.tensor(
                diagnostics[K4]["temporal_gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )
            oracle_predictor, oracle_diagnostic = (
                pointwise_optimal_full_imputation_predictor(
                    current_clipped,
                    k4_final_gates,
                    k4_history_gates,
                    target_direction=target - components["anchor"],
                    influence_cap=float(
                        config["references"]["total_client_influence_cap"]
                    ),
                    return_diagnostics=True,
                )
            )
            oracle_diagnostic.update(
                {
                    "source_rounds": [],
                    "maximum_source_round": None,
                    "uses_current_oracle_target": True,
                    "deployable": False,
                    "privacy_claimed": False,
                }
            )
            definitions = {
                PRIMARY: (
                    rolling_predictor,
                    "rolling_strictly_past_historical_confidence_mixture",
                    True,
                    True,
                    FULL_TEMPORAL_MISSING_SLOT,
                    rolling_diagnostic,
                ),
                DELTA: (
                    rolling_predictor,
                    "rolling_strictly_past_incremental_suppression_ablation",
                    True,
                    True,
                    INCREMENTAL_TEMPORAL_SUPPRESSION,
                    rolling_diagnostic,
                ),
                STATIC: (
                    static_predictor,
                    "constant_last_clean_window_full_missing_slot_ablation",
                    True,
                    True,
                    FULL_TEMPORAL_MISSING_SLOT,
                    static_diagnostic,
                ),
                ORACLE: (
                    oracle_predictor,
                    "nondeployable_pointwise_optimal_full_imputation_oracle",
                    False,
                    False,
                    FULL_TEMPORAL_MISSING_SLOT,
                    oracle_diagnostic,
                ),
            }
            for candidate, definition in definitions.items():
                (
                    predictor,
                    role,
                    deployable,
                    privacy_claimed,
                    imputation_mode,
                    predictor_diagnostic,
                ) = definition
                outputs[candidate], diagnostics[candidate] = _k4b_reference(
                    vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    history=history,
                    enrollment_mean=enrollment_mean,
                    predictor=predictor,
                    predictor_role=role,
                    imputation_mode=imputation_mode,
                    deployable=deployable,
                    privacy_claimed=privacy_claimed,
                    temporal_calibration=temporal_calibration,
                    config=config,
                )
                predictor_diagnostics[candidate] = predictor_diagnostic
            temporal_gates_by_round[round_index] = torch.tensor(
                diagnostics[K4]["temporal_gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )

        pairing_id = f"{trajectory_id}|{round_index}"
        current_clipped_for_evaluation = _clipped_residuals(
            vectors,
            components["anchor"],
            float(config["references"]["total_client_influence_cap"]),
        )
        evaluation_honest = (
            honest
            if phase == "attack" and threat != "none"
            else torch.ones_like(honest)
        )
        honest_clipped_direction = current_clipped_for_evaluation[
            evaluation_honest
        ].mean(dim=0)
        for candidate in CANDIDATES:
            diagnostic = diagnostics[candidate]
            contribution_norms = diagnostic.get("client_contribution_norms")
            max_contribution = (
                max(float(value) for value in contribution_norms)
                if contribution_norms is not None
                else float("nan")
            )
            temporal_gates = diagnostic.get("temporal_gates_by_client")
            all_history_one = bool(
                temporal_gates is not None
                and all(abs(float(value) - 1.0) <= 1.0e-7 for value in temporal_gates)
            )
            deployable, privacy_claimed = _candidate_metadata(candidate)
            predictor_diagnostic = predictor_diagnostics.get(candidate, {})
            predictor_values = diagnostic.get("predictor_by_dimension")
            if predictor_values is None:
                predictor_error = float("nan")
            else:
                predictor_tensor = torch.tensor(
                    predictor_values,
                    dtype=oracle._RUNTIME_DTYPE,
                    device=oracle._RUNTIME_DEVICE,
                )
                predictor_error = float(
                    torch.linalg.vector_norm(
                        predictor_tensor - honest_clipped_direction
                    ).item()
                )
            direct_mass = _component_mass(
                diagnostic,
                latent_byzantine,
                "direct_current_contribution_norms",
            )
            imputed_mass = _component_mass(
                diagnostic,
                latent_byzantine,
                "imputed_contribution_norms",
            )
            total_slot_mass = _component_mass(
                diagnostic,
                latent_byzantine,
                "total_slot_contribution_norms",
            )
            round_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "pairing_id": pairing_id,
                    "crn_key": pairing_id,
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
                    "candidate": candidate,
                    "deployable": deployable,
                    "end_to_end_deployability_claimed": False,
                    "screen_conditioned_on_semi_oracle_anchor": True,
                    "predictor_deployability": (
                        "conditional_on_supplied_past_or_public_anchor"
                        if candidate in {PRIMARY, DELTA, STATIC}
                        else (
                            "nondeployable_pointwise_target_oracle"
                            if candidate == ORACLE
                            else "not_applicable"
                        )
                    ),
                    "privacy_claimed": privacy_claimed,
                    "reference_error": float(
                        torch.linalg.vector_norm(outputs[candidate] - target).item()
                    ),
                    "difference_to_k2": float(
                        torch.linalg.vector_norm(
                            outputs[candidate] - outputs[K2]
                        ).item()
                    ),
                    "difference_to_k4": float(
                        torch.linalg.vector_norm(
                            outputs[candidate] - outputs[K4]
                        ).item()
                    ),
                    "history_gate_all_one": all_history_one,
                    "temporal_gate_mean": float(
                        diagnostic.get("temporal_gate_mean", float("nan"))
                    ),
                    "current_gate_mean": float(
                        diagnostic.get(
                            "current_gate_mean",
                            diagnostic.get("gate_mean", float("nan")),
                        )
                    ),
                    "final_gate_mean": float(diagnostic.get("gate_mean", float("nan"))),
                    "imputation_coefficient_mean": float(
                        diagnostic.get("imputation_coefficient_mean", 0.0)
                    ),
                    "predictor_norm": float(diagnostic.get("predictor_norm", 0.0)),
                    "predictor_error_vs_current_honest_clipped_direction": (
                        predictor_error
                    ),
                    "predictor_error_uses_oracle_honest_labels_for_evaluation": True,
                    "historical_contamination_counterfactual_available": False,
                    "historical_contamination_separately_identified": False,
                    "predictor_max_source_round": predictor_diagnostic.get(
                        "maximum_source_round"
                    ),
                    "predictor_feedback_from_k4b_output": bool(
                        predictor_diagnostic.get("feedback_from_k4b_output", False)
                    ),
                    "predictor_is_strictly_past": bool(
                        predictor_diagnostic.get("maximum_source_round") is None
                        or int(predictor_diagnostic["maximum_source_round"])
                        < round_index
                    )
                    if candidate != ORACLE
                    else False,
                    "uses_oracle_target": candidate == ORACLE
                    and round_index >= first_gate,
                    "k4_comparator_reproduction_error": k4_reproduction_error,
                    "byzantine_direct_current_mass_share": direct_mass["share"],
                    "byzantine_imputed_mass_share": imputed_mass["share"],
                    "byzantine_total_slot_mass_share": total_slot_mass["share"],
                    "direct_current_mass_total": direct_mass["total"],
                    "imputed_mass_total": imputed_mass["total"],
                    "total_slot_mass_total": total_slot_mass["total"],
                    "max_client_contribution_norm": max_contribution,
                    "contribution_cap_respected": bool(
                        diagnostic.get("client_contribution_cap_respected", True)
                    ),
                    "normalization_by_gate_sum": bool(
                        diagnostic.get("normalization_by_gate_sum", False)
                    ),
                    "all_finite": bool(torch.isfinite(outputs[candidate]).all())
                    and k4._finite_tree(diagnostic),
                }
            )

        current_gates = torch.tensor(
            diagnostics[K4]["current_gates_by_client"],
            dtype=oracle._RUNTIME_DTYPE,
            device=oracle._RUNTIME_DEVICE,
        )
        temporal_gates = torch.tensor(
            diagnostics[K4]["temporal_gates_by_client"],
            dtype=oracle._RUNTIME_DTYPE,
            device=oracle._RUNTIME_DEVICE,
        )
        final_gates = torch.minimum(current_gates, temporal_gates)
        residuals_by_round[round_index] = _clipped_residuals(
            vectors,
            components["anchor"],
            float(config["references"]["total_client_influence_cap"]),
        )
        gates_by_round[round_index] = final_gates

        if round_index >= first_gate:
            statistics = diagnostics[K4]["temporal_statistics_by_client"]
            primary_direct = diagnostics[PRIMARY]["direct_current_contribution_norms"]
            primary_imputed = diagnostics[PRIMARY]["imputed_contribution_norms"]
            primary_total = diagnostics[PRIMARY]["total_slot_contribution_norms"]
            delta_direct = diagnostics[DELTA]["direct_current_contribution_norms"]
            delta_imputed = diagnostics[DELTA]["imputed_contribution_norms"]
            delta_total = diagnostics[DELTA]["total_slot_contribution_norms"]
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
                        "public_noise_tier": float(components["tiers"][client].item()),
                        "latent_byzantine": bool(latent_byzantine[client]),
                        "active_byzantine": bool(active_byzantine[client]),
                        "honest_outlier": bool(components["outliers"][client]),
                        "honest_regular": bool(regular[client]),
                        "temporal_statistic": float(statistics[client]),
                        "current_gate": float(current_gates[client]),
                        "temporal_gate": float(temporal_gates[client]),
                        "final_gate": float(final_gates[client]),
                        "gate_drop_vs_k2_aware": max(
                            0.0,
                            float(current_gates[client]) - float(final_gates[client]),
                        ),
                        "primary_direct_current_contribution_norm": float(
                            primary_direct[client]
                        ),
                        "primary_imputed_contribution_norm": float(
                            primary_imputed[client]
                        ),
                        "primary_total_slot_contribution_norm": float(
                            primary_total[client]
                        ),
                        "delta_direct_current_contribution_norm": float(
                            delta_direct[client]
                        ),
                        "delta_imputed_contribution_norm": float(delta_imputed[client]),
                        "delta_total_slot_contribution_norm": float(
                            delta_total[client]
                        ),
                    }
                )
        history.append(standardized)

    trajectory_rows = _trajectory_summaries(
        round_rows,
        temporal_gates_by_round,
        latent_byzantine,
        config=config,
    )
    return round_rows, client_rows, trajectory_rows


def _trajectory_summaries(
    round_rows: Sequence[Mapping[str, Any]],
    temporal_gates_by_round: Mapping[int, torch.Tensor],
    latent_byzantine: torch.Tensor,
    *,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not round_rows:
        raise ValueError("Cannot summarize an empty K4b trajectory")
    temporal = config["temporal"]
    base_row = round_rows[0]
    byzantine_ids = torch.nonzero(latent_byzantine, as_tuple=False).reshape(-1).tolist()
    detection = [
        k4._first_two_round_event(
            temporal_gates_by_round,
            client=int(client),
            first_end_round=int(temporal["first_eligible_detection_round"]),
            last_end_round=int(temporal["detection_deadline_round"]),
            threshold=float(temporal["detection_gate_threshold"]),
            direction="below",
        )
        for client in byzantine_ids
    ]
    recovery = [
        k4._first_two_round_event(
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
        by_phase: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            by_phase[str(row["phase"])].append(row)
        deployable, privacy_claimed = _candidate_metadata(candidate)
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
                "deployable": deployable,
                "privacy_claimed": privacy_claimed,
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
                "attack_byzantine_direct_current_mass_share": base._finite_mean(
                    row["byzantine_direct_current_mass_share"]
                    for row in by_phase["attack"]
                ),
                "attack_byzantine_imputed_mass_share": base._finite_mean(
                    row["byzantine_imputed_mass_share"] for row in by_phase["attack"]
                ),
                "attack_byzantine_total_slot_mass_share_descriptive": base._finite_mean(
                    row["byzantine_total_slot_mass_share"] for row in by_phase["attack"]
                ),
                "attack_predictor_error_vs_current_honest_clipped_direction": (
                    base._finite_mean(
                        row["predictor_error_vs_current_honest_clipped_direction"]
                        for row in by_phase["attack"]
                    )
                    if candidate in {PRIMARY, DELTA, STATIC, ORACLE}
                    else float("nan")
                ),
                "detection_rate_within_deadline": (
                    sum(value is not None for value in detection)
                    / float(len(detection))
                    if candidate == PRIMARY and detection
                    else float("nan")
                ),
                "recovery_rate_within_deadline": (
                    sum(value is not None for value in recovery) / float(len(recovery))
                    if candidate == PRIMARY and recovery
                    else float("nan")
                ),
            }
        )
    return result


def _screen_development(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for cell in _development_cells(config):
        rounds, clients, trajectories = _evaluate_trajectory(
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
        expected_id = str(cell["trajectory_id"])
        if any(
            str(row["trajectory_id"]) != expected_id
            for row in rounds + clients + trajectories
        ):
            raise RuntimeError("A K4b development cell emitted a wrong trajectory ID")
        round_rows.extend(rounds)
        client_rows.extend(clients)
        trajectory_rows.extend(trajectories)
    return round_rows, client_rows, trajectory_rows


def _replace_one_audit(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit current-cohort sensitivity with past, predictor and h fixed."""

    rows: list[dict[str, Any]] = []
    round_index = int(config["temporal"]["first_temporal_gate_round"])
    enrollment_rounds = int(config["temporal"]["enrollment_rounds"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    cap = float(config["references"]["total_client_influence_cap"])
    for seed_value in config["randomness"]["development_seeds"]:
        seed = int(seed_value)
        for regime, permutation in base._noise_cells(config):
            components = k4._trajectory_components(
                config,
                seed=seed,
                regime=regime,
                permutation=permutation,
                geometry="orthogonal",
                dynamics="stationary",
            )
            aware_radii = k4._radii(
                config,
                components["variances"],
                k2_calibration,
                regime_name=str(regime["name"]),
                blind=False,
            )
            blind_radii = k4._radii(
                config,
                components["variances"],
                k2_calibration,
                regime_name=str(regime["name"]),
                blind=True,
            )
            history: list[torch.Tensor] = []
            enrollment_mean: torch.Tensor | None = None
            residuals_by_round: dict[int, torch.Tensor] = {}
            gates_by_round: dict[int, torch.Tensor] = {}
            current_vectors: torch.Tensor | None = None
            for past_round in range(1, round_index + 1):
                observed = k4._private_vectors(
                    config,
                    components,
                    seed=seed,
                    round_index=past_round,
                    geometry="orthogonal",
                )
                vectors = clip_l2(
                    observed, float(config["aggregation"]["server_clip_norm"])
                )
                standardized = k4._standardized_messages(
                    config,
                    vectors,
                    anchor=components["anchor"],
                    variances=components["variances"],
                )
                if past_round == enrollment_rounds:
                    enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)
                if past_round < round_index:
                    _, current_diag = k4._current_reference(
                        K2,
                        vectors,
                        anchor=components["anchor"],
                        aware_radii=aware_radii,
                        blind_radii=blind_radii,
                        config=config,
                    )
                    residuals_by_round[past_round] = _clipped_residuals(
                        vectors, components["anchor"], cap
                    )
                    gates_by_round[past_round] = torch.tensor(
                        current_diag["gates_by_client"],
                        dtype=oracle._RUNTIME_DTYPE,
                        device=oracle._RUNTIME_DEVICE,
                    )
                    history.append(standardized)
                else:
                    current_vectors = vectors
            if enrollment_mean is None or current_vectors is None:
                raise RuntimeError("K4b replace-one audit failed to construct past")
            predictor, predictor_diag = _predictor_from_rounds(
                residuals_by_round,
                gates_by_round,
                range(round_index - 4, round_index),
                config=config,
            )
            for trial in range(trials):
                replaced = (
                    oracle._seed(
                        "g0g-k4b-replaced-client",
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
                        "g0g-k4b-replacement",
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
                common = {
                    "anchor": components["anchor"],
                    "aware_radii": aware_radii,
                    "history": history,
                    "enrollment_mean": enrollment_mean,
                    "predictor": predictor,
                    "predictor_role": "fixed_past_for_replace_one_audit",
                    "deployable": False,
                    "privacy_claimed": True,
                    "temporal_calibration": temporal_calibration,
                    "config": config,
                }
                for candidate, mode in (
                    (PRIMARY, FULL_TEMPORAL_MISSING_SLOT),
                    (DELTA, INCREMENTAL_TEMPORAL_SUPPRESSION),
                ):
                    left, diagnostics = _k4b_reference(
                        current_vectors, imputation_mode=mode, **common
                    )
                    right, _ = _k4b_reference(neighbour, imputation_mode=mode, **common)
                    difference = float(torch.linalg.vector_norm(left - right).item())
                    bound = float(diagnostics["replace_one_bound"])
                    history_gate_min = min(
                        float(value)
                        for value in diagnostics["temporal_gates_by_client"]
                    )
                    rows.append(
                        {
                            "audit_scenario": "clean_first_causal_gate",
                            "candidate": candidate,
                            "seed": seed,
                            "noise_regime": str(regime["name"]),
                            "noise_permutation": str(permutation),
                            "trial": trial,
                            "round": round_index,
                            "replaced_client": int(replaced),
                            "same_past": True,
                            "same_predictor": True,
                            "same_history_gate": True,
                            "same_covariances": True,
                            "same_anchor": True,
                            "predictor_max_source_round": predictor_diag[
                                "maximum_source_round"
                            ],
                            "observed_replace_one_difference": difference,
                            "theoretical_replace_one_bound": bound,
                            "ratio_observed_to_bound": difference / bound,
                            "violation": difference > bound + 1.0e-6,
                            "history_gate_min": history_gate_min,
                            "history_gate_has_suppression": (
                                history_gate_min < 1.0 - 1.0e-7
                            ),
                        }
                    )
    if int(config["cohort"]["num_clients"]) == 25:
        rows.extend(
            _attacked_replace_one_audit(
                config,
                k2_calibration,
                temporal_calibration,
            )
        )
    return rows


def _attacked_replace_one_audit(
    config: dict[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit the production primitive in an attacked cell with ``h < 1``.

    The history and predictor are constructed once and then held fixed across
    the neighbouring current cohorts.  This is exactly the conditional
    current-cohort replace-one statement certified by ``2G/n``.
    """

    regimes = [
        (regime, permutation)
        for regime, permutation in base._noise_cells(config)
        if str(regime["name"]) == "heteroscedastic"
        and str(permutation) == "byzantine_high"
    ]
    if len(regimes) != 1:
        raise RuntimeError("Expected one heteroscedastic/byzantine_high audit cell")
    regime, permutation = regimes[0]
    audit_round = 17
    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    cap = float(config["references"]["total_client_influence_cap"])
    server_cap = float(config["aggregation"]["server_clip_norm"])
    enrollment_rounds = int(config["temporal"]["enrollment_rounds"])
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    rows: list[dict[str, Any]] = []
    for seed_value in config["randomness"]["development_seeds"]:
        seed = int(seed_value)
        components = k4._trajectory_components(
            config,
            seed=seed,
            regime=regime,
            permutation=str(permutation),
            geometry="orthogonal",
            dynamics="stationary",
        )
        aware_radii = k4._radii(
            config,
            components["variances"],
            k2_calibration,
            regime_name=str(regime["name"]),
            blind=False,
        )
        blind_radii = k4._radii(
            config,
            components["variances"],
            k2_calibration,
            regime_name=str(regime["name"]),
            blind=True,
        )
        history: list[torch.Tensor] = []
        enrollment_mean: torch.Tensor | None = None
        residuals_by_round: dict[int, torch.Tensor] = {}
        gates_by_round: dict[int, torch.Tensor] = {}
        current_vectors: torch.Tensor | None = None
        for round_index in range(1, audit_round + 1):
            observed = k4._private_vectors(
                config,
                components,
                seed=seed,
                round_index=round_index,
                geometry="orthogonal",
            )
            if k4._attack_active(
                config,
                round_index=round_index,
                schedule="persistent",
            ):
                observed, _ = oracle._replace_with_attack(
                    observed,
                    config,
                    threat="bitflip_x10",
                    severity=float(config["threats"]["severity"]),
                    seed=oracle._seed(
                        "g0g-k4-attack",
                        seed,
                        regime["name"],
                        permutation,
                        "orthogonal",
                        "stationary",
                        "bitflip_x10",
                        round_index,
                    ),
                )
            vectors = clip_l2(observed, server_cap)
            standardized = k4._standardized_messages(
                config,
                vectors,
                anchor=components["anchor"],
                variances=components["variances"],
            )
            if round_index == enrollment_rounds:
                enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)
            if round_index == audit_round:
                current_vectors = vectors
                break
            if round_index < first_gate:
                _, diag = k4._current_reference(
                    K2,
                    vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    blind_radii=blind_radii,
                    config=config,
                )
                gates = torch.tensor(
                    diag["gates_by_client"],
                    dtype=oracle._RUNTIME_DTYPE,
                    device=oracle._RUNTIME_DEVICE,
                )
            else:
                if enrollment_mean is None:
                    raise RuntimeError("Missing enrollment in attacked audit")
                _, diag = _frozen_k4_comparator(
                    vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    history=history,
                    enrollment_mean=enrollment_mean,
                    temporal_calibration=temporal_calibration,
                    config=config,
                )
                gates = torch.tensor(
                    diag["gates_by_client"],
                    dtype=oracle._RUNTIME_DTYPE,
                    device=oracle._RUNTIME_DEVICE,
                )
            residuals_by_round[round_index] = _clipped_residuals(
                vectors, components["anchor"], cap
            )
            gates_by_round[round_index] = gates
            history.append(standardized)
        if enrollment_mean is None or current_vectors is None:
            raise RuntimeError("Failed to construct attacked replace-one cell")
        predictor, predictor_diag = _predictor_from_rounds(
            residuals_by_round,
            gates_by_round,
            range(audit_round - 4, audit_round),
            config=config,
        )
        common = {
            "anchor": components["anchor"],
            "aware_radii": aware_radii,
            "history": history,
            "enrollment_mean": enrollment_mean,
            "predictor": predictor,
            "predictor_role": "fixed_attacked_past_for_replace_one_audit",
            "deployable": False,
            "privacy_claimed": True,
            "temporal_calibration": temporal_calibration,
            "config": config,
        }
        _, baseline_diag = _k4b_reference(
            current_vectors,
            imputation_mode=FULL_TEMPORAL_MISSING_SLOT,
            **common,
        )
        history_gate_min = min(
            float(value) for value in baseline_diag["temporal_gates_by_client"]
        )
        if not history_gate_min < 1.0 - 1.0e-7:
            raise RuntimeError(
                "Attacked replace-one production cell did not produce h<1"
            )
        for trial in range(trials):
            replaced = (
                oracle._seed(
                    "g0g-k4b-attacked-replaced-client",
                    seed,
                    trial,
                )
                % current_vectors.shape[0]
            )
            neighbour = current_vectors.clone()
            proposal = torch.randn(
                current_vectors.shape[1],
                generator=oracle._generator(
                    "g0g-k4b-attacked-replacement",
                    seed,
                    trial,
                ),
                dtype=current_vectors.dtype,
                device=current_vectors.device,
            )
            neighbour[replaced] = clip_l2(100.0 * proposal[None, :], server_cap)[0]
            for candidate, mode in (
                (PRIMARY, FULL_TEMPORAL_MISSING_SLOT),
                (DELTA, INCREMENTAL_TEMPORAL_SUPPRESSION),
            ):
                left, diag = _k4b_reference(
                    current_vectors, imputation_mode=mode, **common
                )
                right, _ = _k4b_reference(neighbour, imputation_mode=mode, **common)
                difference = float(torch.linalg.vector_norm(left - right).item())
                bound = float(diag["replace_one_bound"])
                rows.append(
                    {
                        "audit_scenario": "persistent_bitflip_round17_h_below_one",
                        "candidate": candidate,
                        "seed": seed,
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": str(permutation),
                        "trial": trial,
                        "round": audit_round,
                        "replaced_client": int(replaced),
                        "same_past": True,
                        "same_predictor": True,
                        "same_history_gate": True,
                        "same_covariances": True,
                        "same_anchor": True,
                        "predictor_max_source_round": predictor_diag[
                            "maximum_source_round"
                        ],
                        "observed_replace_one_difference": difference,
                        "theoretical_replace_one_bound": bound,
                        "ratio_observed_to_bound": difference / bound,
                        "violation": difference > bound + 1.0e-6,
                        "history_gate_min": history_gate_min,
                        "history_gate_has_suppression": True,
                    }
                )
    return rows


def _paired_seed_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate: Callable[[Mapping[str, Any]], bool],
    metric: str = "attack_auc",
) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["candidate"] in {candidate, baseline} and predicate(row):
            grouped[(int(row["seed"]), str(row["candidate"]))].append(
                float(row[metric])
            )
    seeds = sorted(
        seed
        for seed, method in grouped
        if method == candidate and (seed, baseline) in grouped
    )
    return [
        base._finite_mean(grouped[(seed, candidate)])
        - base._finite_mean(grouped[(seed, baseline)])
        for seed in seeds
    ]


def _paired_ratio(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate: Callable[[Mapping[str, Any]], bool],
    metric: str,
) -> float:
    candidate_values = [
        float(row[metric])
        for row in rows
        if row["candidate"] == candidate and predicate(row)
    ]
    baseline_values = [
        float(row[metric])
        for row in rows
        if row["candidate"] == baseline and predicate(row)
    ]
    if not candidate_values or not baseline_values:
        raise RuntimeError(f"Missing paired contrast {candidate} vs {baseline}")
    return base._finite_mean(candidate_values) / max(
        base._finite_mean(baseline_values), 1.0e-12
    )


def _paired_seed_log_ratios(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate: Callable[[Mapping[str, Any]], bool],
    metric: str,
) -> dict[str, Any]:
    """Compute one paired log-ratio per seed and a descriptive pooled ratio."""

    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["candidate"] in {candidate, baseline} and predicate(row):
            value = float(row[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"Non-positive/non-finite value in log-ratio {metric}: {value}"
                )
            grouped[(int(row["seed"]), str(row["candidate"]))].append(value)
    seeds = sorted(
        seed
        for seed, method in grouped
        if method == candidate and (seed, baseline) in grouped
    )
    log_ratios: list[float] = []
    ratios_by_seed: dict[str, float] = {}
    log_ratios_by_seed: dict[str, float] = {}
    for seed in seeds:
        candidate_mean = base._finite_mean(grouped[(seed, candidate)])
        baseline_mean = base._finite_mean(grouped[(seed, baseline)])
        ratio = candidate_mean / baseline_mean
        ratios_by_seed[str(seed)] = ratio
        log_ratio = math.log(ratio)
        log_ratios_by_seed[str(seed)] = log_ratio
        log_ratios.append(log_ratio)
    if not log_ratios:
        raise RuntimeError(f"Missing paired seed log-ratio {candidate} vs {baseline}")
    ci = base._ci95(log_ratios)
    pooled_ratio = _paired_ratio(
        rows,
        candidate=candidate,
        baseline=baseline,
        predicate=predicate,
        metric=metric,
    )
    return {
        "n_seed_pairs": len(log_ratios),
        "seed_ids": seeds,
        "ratios_by_seed": ratios_by_seed,
        "log_ratios_by_seed": log_ratios_by_seed,
        "log_ratio_ci95": ci,
        "exp_log_ratio_ci95_high": math.exp(float(ci["high"])),
        "geometric_mean_ratio": math.exp(base._finite_mean(log_ratios)),
        "pooled_ratio_descriptive": pooled_ratio,
    }


def _relative_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate: Callable[[Mapping[str, Any]], bool],
    metric: str = "attack_auc",
) -> float:
    return 1.0 - _paired_ratio(
        rows,
        candidate=candidate,
        baseline=baseline,
        predicate=predicate,
        metric=metric,
    )


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
    rate = base._finite_mean(
        1.0 if float(row["temporal_gate"]) < 1.0 - 1.0e-7 else 0.0 for row in selected
    )
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        by_trajectory[str(row["trajectory_id"])].append(
            float(row["temporal_statistic"])
        )
    c0 = float(config["frozen_calibrations"]["k4_temporal"]["deployed_c0"])
    trajectory_rate = base._finite_mean(
        1.0 if max(values) > c0 else 0.0 for values in by_trajectory.values()
    )
    tier_rates: dict[str, float] = {}
    tier_gate_means: dict[str, float] = {}
    heteroscedastic = [
        row for row in selected if row["noise_regime"] == "heteroscedastic"
    ]
    for tier in k4.TIERS:
        tier_rows = [
            row
            for row in heteroscedastic
            if math.isclose(float(row["public_noise_tier"]), tier, abs_tol=1.0e-7)
        ]
        if tier_rows:
            tier_rates[str(tier)] = base._finite_mean(
                1.0 if float(row["temporal_gate"]) < 1.0 - 1.0e-7 else 0.0
                for row in tier_rows
            )
            tier_gate_means[str(tier)] = base._finite_mean(
                float(row["temporal_gate"]) for row in tier_rows
            )
    return {
        "client_round_rate": rate,
        "trajectory_rate": trajectory_rate,
        "tier_rates": tier_rates,
        "tier_gate_means": tier_gate_means,
        "tier_rate_gap": max(tier_rates.values()) - min(tier_rates.values()),
        "tier_gate_mean_gap": max(tier_gate_means.values())
        - min(tier_gate_means.values()),
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
    expected_ids = {str(cell["trajectory_id"]) for cell in _development_cells(config)}
    total_rounds = int(config["temporal"]["total_rounds"])
    first_gate = int(config["temporal"]["first_temporal_gate_round"])
    n = int(config["cohort"]["num_clients"])
    expected_round_keys = {
        (identifier, candidate, round_index)
        for identifier in expected_ids
        for candidate in CANDIDATES
        for round_index in range(1, total_rounds + 1)
    }
    expected_trajectory_keys = {
        (identifier, candidate)
        for identifier in expected_ids
        for candidate in CANDIDATES
    }
    expected_client_keys = {
        (identifier, round_index, client)
        for identifier in expected_ids
        for round_index in range(first_gate, total_rounds + 1)
        for client in range(n)
    }
    expected_replace_keys = {
        (
            "clean_first_causal_gate",
            candidate,
            int(seed),
            str(regime["name"]),
            str(permutation),
            trial,
        )
        for seed in config["randomness"]["development_seeds"]
        for regime, permutation in base._noise_cells(config)
        for candidate in (PRIMARY, DELTA)
        for trial in range(
            int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
        )
    }
    expected_replace_keys |= {
        (
            "persistent_bitflip_round17_h_below_one",
            candidate,
            int(seed),
            "heteroscedastic",
            "byzantine_high",
            trial,
        )
        for seed in config["randomness"]["development_seeds"]
        for candidate in (PRIMARY, DELTA)
        for trial in range(
            int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
        )
    }

    def observed_keys(
        rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
    ) -> tuple[set[tuple], int]:
        values = [tuple(row[field] for field in fields) for row in rows]
        return set(values), len(values) - len(set(values))

    round_keys, round_duplicates = observed_keys(
        round_rows, ("trajectory_id", "candidate", "round")
    )
    trajectory_keys, trajectory_duplicates = observed_keys(
        trajectory_rows, ("trajectory_id", "candidate")
    )
    client_keys, client_duplicates = observed_keys(
        client_rows, ("trajectory_id", "round", "client")
    )
    replace_keys, replace_duplicates = observed_keys(
        replace_rows,
        (
            "audit_scenario",
            "candidate",
            "seed",
            "noise_regime",
            "noise_permutation",
            "trial",
        ),
    )
    sets = (
        (round_keys, expected_round_keys, round_duplicates),
        (trajectory_keys, expected_trajectory_keys, trajectory_duplicates),
        (client_keys, expected_client_keys, client_duplicates),
        (replace_keys, expected_replace_keys, replace_duplicates),
    )
    exact = all(
        observed == expected and duplicates == 0
        for observed, expected, duplicates in sets
    )
    fractions = [
        len(observed & expected) / float(len(expected))
        for observed, expected, _ in sets
    ]
    return {
        "exact_identifier_completeness": exact,
        "complete_fraction": min(fractions),
        "expected_trajectory_ids": len(expected_ids),
        "round_rows_expected": len(expected_round_keys),
        "trajectory_rows_expected": len(expected_trajectory_keys),
        "client_rows_expected": len(expected_client_keys),
        "replace_rows_expected": len(expected_replace_keys),
        "duplicate_counts": {
            "round": round_duplicates,
            "trajectory": trajectory_duplicates,
            "client": client_duplicates,
            "replace": replace_duplicates,
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
) -> dict[str, Any]:
    gates = config["gates"]
    separated = {
        str(value) for value in config["threats"]["separated_for_primary_gate"]
    }
    headroom_threats = {
        str(value) for value in config["threats"]["oracle_headroom_threats"]
    }
    completeness = _screen_completeness(
        round_rows, client_rows, trajectory_rows, replace_rows, config
    )
    finite_fraction = base._finite_mean(
        1.0 if bool(row["all_finite"]) else 0.0 for row in round_rows
    )
    certified = [row for row in round_rows if row["candidate"] in CERTIFIED_CANDIDATES]
    cap_violations = sum(
        not bool(row["contribution_cap_respected"]) for row in certified
    )
    normalization_violations = sum(
        bool(row["normalization_by_gate_sum"]) for row in certified
    )
    causal_violations = sum(
        not bool(row["predictor_is_strictly_past"])
        for row in round_rows
        if row["candidate"] in {PRIMARY, DELTA, STATIC}
    )
    predictor_feedback_violations = sum(
        bool(row["predictor_feedback_from_k4b_output"])
        for row in round_rows
        if row["candidate"] in {PRIMARY, DELTA, STATIC}
    )
    oracle_label_violations = sum(
        not (
            row["candidate"] == ORACLE
            and not bool(row["deployable"])
            and not bool(row["privacy_claimed"])
            and bool(row["uses_oracle_target"])
            == (
                int(row["round"])
                >= int(config["temporal"]["first_temporal_gate_round"])
            )
        )
        for row in round_rows
        if row["candidate"] == ORACLE
    )
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    k4_identity = max(
        float(row["difference_to_k2"])
        for row in round_rows
        if row["candidate"] == K4 and bool(row["history_gate_all_one"])
    )
    k4_reproduction = max(
        float(row["k4_comparator_reproduction_error"])
        for row in round_rows
        if row["candidate"] == K4
    )
    k4b_identity = max(
        float(row["difference_to_k2"])
        for row in round_rows
        if row["candidate"] in {PRIMARY, DELTA} and bool(row["history_gate_all_one"])
    )
    false_triggers = _false_trigger_diagnostics(client_rows, config)
    representative = _representative_client_rows(client_rows, config)
    outlier_drop = base._finite_mean(
        float(row["gate_drop_vs_k2_aware"])
        for row in representative
        if bool(row["honest_outlier"]) and not bool(row["latent_byzantine"])
    )

    def persistent_for(threats: set[str]) -> Callable[[Mapping[str, Any]], bool]:
        return lambda row: bool(
            row["schedule"] == "persistent" and row["threat"] in threats
        )

    def persistent_threat(name: str) -> Callable[[Mapping[str, Any]], bool]:
        return lambda row: bool(
            row["schedule"] == "persistent" and row["threat"] == name
        )

    def intermittent(row: Mapping[str, Any]) -> bool:
        return bool(row["schedule"] == "intermittent_2_on_1_off")

    def clean(row: Mapping[str, Any]) -> bool:
        return bool(row["schedule"] == "no_compromise" and row["threat"] == "none")

    headroom_predicate = persistent_for(headroom_threats)
    separated_predicate = persistent_for(separated)
    oracle_gain = _relative_gain(
        trajectory_rows,
        candidate=ORACLE,
        baseline=K4,
        predicate=headroom_predicate,
    )
    oracle_seed_differences = _paired_seed_differences(
        trajectory_rows,
        candidate=ORACLE,
        baseline=K4,
        predicate=headroom_predicate,
    )
    oracle_ci = base._ci95(oracle_seed_differences)
    primary_headroom_gain = _relative_gain(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K4,
        predicate=headroom_predicate,
    )
    primary_headroom_seed_differences = _paired_seed_differences(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K4,
        predicate=headroom_predicate,
    )
    primary_headroom_ci = base._ci95(primary_headroom_seed_differences)
    primary_k2_gain = _relative_gain(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K2,
        predicate=separated_predicate,
    )
    primary_k2_seed_differences = _paired_seed_differences(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K2,
        predicate=separated_predicate,
    )
    primary_k2_ci = base._ci95(primary_k2_seed_differences)
    mean_k4 = base._finite_mean(
        row["attack_auc"]
        for row in trajectory_rows
        if row["candidate"] == K4 and headroom_predicate(row)
    )
    mean_primary = base._finite_mean(
        row["attack_auc"]
        for row in trajectory_rows
        if row["candidate"] == PRIMARY and headroom_predicate(row)
    )
    mean_delta = base._finite_mean(
        row["attack_auc"]
        for row in trajectory_rows
        if row["candidate"] == DELTA and headroom_predicate(row)
    )
    mean_oracle = base._finite_mean(
        row["attack_auc"]
        for row in trajectory_rows
        if row["candidate"] == ORACLE and headroom_predicate(row)
    )
    oracle_headroom = mean_k4 - mean_oracle
    captured_headroom = mean_k4 - mean_primary
    delta_captured_headroom = mean_k4 - mean_delta
    capture_fraction = (
        captured_headroom / oracle_headroom if oracle_headroom > 0.0 else float("-inf")
    )
    delta_capture_fraction = (
        delta_captured_headroom / oracle_headroom
        if oracle_headroom > 0.0
        else float("-inf")
    )
    ipm_noninferiority = _paired_seed_log_ratios(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K4,
        predicate=persistent_threat("ipm"),
        metric="attack_auc",
    )
    alie_noninferiority = _paired_seed_log_ratios(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K4,
        predicate=persistent_threat("alie"),
        metric="attack_auc",
    )
    intermittent_noninferiority = _paired_seed_log_ratios(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K4,
        predicate=intermittent,
        metric="attack_auc",
    )
    clean_noninferiority = _paired_seed_log_ratios(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K2,
        predicate=clean,
        metric="post_enrollment_auc",
    )
    byzantine_mass_reduction = _relative_gain(
        trajectory_rows,
        candidate=PRIMARY,
        baseline=K2,
        predicate=separated_predicate,
        metric="attack_byzantine_direct_current_mass_share",
    )
    delta_headroom_gain = _relative_gain(
        trajectory_rows,
        candidate=DELTA,
        baseline=K4,
        predicate=headroom_predicate,
    )
    delta_k2_gain = _relative_gain(
        trajectory_rows,
        candidate=DELTA,
        baseline=K2,
        predicate=separated_predicate,
    )
    delta_byzantine_direct_mass_reduction = _relative_gain(
        trajectory_rows,
        candidate=DELTA,
        baseline=K2,
        predicate=separated_predicate,
        metric="attack_byzantine_direct_current_mass_share",
    )
    primary_predictor_error = base._finite_mean(
        row["attack_predictor_error_vs_current_honest_clipped_direction"]
        for row in trajectory_rows
        if row["candidate"] == PRIMARY
    )
    delta_predictor_error = base._finite_mean(
        row["attack_predictor_error_vs_current_honest_clipped_direction"]
        for row in trajectory_rows
        if row["candidate"] == DELTA
    )
    primary_persistent = [
        row
        for row in trajectory_rows
        if row["candidate"] == PRIMARY and separated_predicate(row)
    ]
    detection_rate = base._finite_mean(
        row["detection_rate_within_deadline"] for row in primary_persistent
    )
    recovery_rate = base._finite_mean(
        row["recovery_rate_within_deadline"] for row in primary_persistent
    )

    observed = {
        "device": device_name,
        "development_trajectories": completeness["expected_trajectory_ids"],
        "development_round_rows": len(round_rows),
        "development_client_rows": len(client_rows),
        "development_trajectory_rows": len(trajectory_rows),
        "completeness": completeness,
        "finite_metric_fraction": finite_fraction,
        "contribution_cap_violations": cap_violations,
        "gate_sum_normalization_violations": normalization_violations,
        "causal_predictor_violations": causal_violations,
        "predictor_feedback_violations": predictor_feedback_violations,
        "oracle_metadata_violations": oracle_label_violations,
        "replace_one_trials": len(replace_rows),
        "replace_one_violations": replace_violations,
        "replace_one_max_ratio_to_bound": max(
            float(row["ratio_observed_to_bound"]) for row in replace_rows
        ),
        "k4_before_or_all_one_history_identity_max_abs_error": k4_identity,
        "k4_comparator_reproduction_max_abs_error": k4_reproduction,
        "k4b_k2_identity_max_abs_error": k4b_identity,
        "false_triggers": false_triggers,
        "honest_outlier_gate_drop_vs_k2_aware": outlier_drop,
        "oracle_bf_mr_persistent_gain_vs_k4": oracle_gain,
        "oracle_bf_mr_persistent_seed_difference_raw_count": len(
            oracle_seed_differences
        ),
        "oracle_bf_mr_persistent_seed_difference_count": int(oracle_ci["n"]),
        "oracle_bf_mr_persistent_difference_seed_ci95": oracle_ci,
        "primary_bf_mr_persistent_gain_vs_k4": primary_headroom_gain,
        "primary_bf_mr_persistent_seed_difference_raw_count": len(
            primary_headroom_seed_differences
        ),
        "primary_bf_mr_persistent_seed_difference_count": int(primary_headroom_ci["n"]),
        "primary_bf_mr_persistent_difference_seed_ci95": primary_headroom_ci,
        "primary_persistent_separated_gain_vs_k2": primary_k2_gain,
        "primary_persistent_separated_seed_difference_raw_count": len(
            primary_k2_seed_differences
        ),
        "primary_persistent_separated_seed_difference_count": int(primary_k2_ci["n"]),
        "primary_persistent_separated_difference_seed_ci95": primary_k2_ci,
        "oracle_headroom_absolute": oracle_headroom,
        "primary_captured_headroom_absolute": captured_headroom,
        "primary_oracle_headroom_capture_fraction": capture_fraction,
        "delta_oracle_headroom_capture_fraction_descriptive": delta_capture_fraction,
        "primary_persistent_ipm_attack_auc_log_ratio": ipm_noninferiority,
        "primary_persistent_alie_attack_auc_log_ratio": alie_noninferiority,
        "primary_intermittent_attack_auc_log_ratio": intermittent_noninferiority,
        "clean_primary_post_enrollment_auc_log_ratio": clean_noninferiority,
        "persistent_separated_byzantine_mass_reduction_vs_k2": byzantine_mass_reduction,
        "delta_bf_mr_persistent_gain_vs_k4_descriptive": delta_headroom_gain,
        "delta_persistent_separated_gain_vs_k2_descriptive": delta_k2_gain,
        "delta_persistent_separated_direct_mass_reduction_vs_k2_descriptive": (
            delta_byzantine_direct_mass_reduction
        ),
        "primary_predictor_error_vs_honest_clipped_direction": (
            primary_predictor_error
        ),
        "delta_predictor_error_vs_honest_clipped_direction": delta_predictor_error,
        "historical_contamination_separately_identified": False,
        "persistent_separated_detection_rate_within_deadline": detection_rate,
        "persistent_separated_recovery_rate_within_deadline": recovery_rate,
    }
    checks = {
        "complete": bool(completeness["exact_identifier_completeness"])
        and float(completeness["complete_fraction"])
        >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "production_device": device_name == str(gates["production_device_required"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "no_gate_sum_normalization": normalization_violations == 0,
        "causal_predictor": causal_violations == 0,
        "no_predictor_feedback": predictor_feedback_violations == 0,
        "oracle_metadata": oracle_label_violations == 0,
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "k4_comparator_reproduction": k4_reproduction
        <= float(gates["k4_comparator_reproduction_abs_error_max"]),
        "k4b_k2_identity_when_history_one": k4b_identity
        <= float(gates["k4b_k2_identity_when_history_gate_one_abs_error_max"]),
        "regular_client_round_false_trigger": false_triggers["client_round_rate"]
        <= float(gates["regular_client_round_false_trigger_rate_max"]),
        "regular_trajectory_false_trigger": false_triggers["trajectory_rate"]
        <= float(gates["regular_trajectory_false_trigger_rate_max"]),
        "pooled_regular_false_trigger_tier_gap": false_triggers["tier_rate_gap"]
        <= float(gates["pooled_regular_false_trigger_tier_gap_max"]),
        "pooled_regular_gate_mean_tier_gap": false_triggers["tier_gate_mean_gap"]
        <= float(gates["pooled_regular_gate_mean_tier_gap_max"]),
        "honest_outlier_gate_drop": outlier_drop
        <= float(gates["honest_outlier_gate_drop_vs_k2_aware_max"]),
        "byzantine_mass_reduction": byzantine_mass_reduction
        >= float(
            gates["persistent_separated_byzantine_mass_reduction_vs_k2_aware_min"]
        ),
        "detection": detection_rate
        >= float(gates["persistent_separated_detection_rate_within_deadline_min"]),
        "recovery": recovery_rate
        >= float(gates["persistent_separated_recovery_rate_within_deadline_min"]),
        "oracle_headroom_gain": oracle_gain
        >= float(gates["oracle_bf_mr_persistent_gain_vs_k4_min"]),
        "oracle_effectiveness_seed_count": int(oracle_ci["n"]) == 5,
        "oracle_headroom_ci": float(oracle_ci["high"])
        <= float(gates["oracle_bf_mr_persistent_difference_ci95_high_max"]),
        "primary_gain_vs_k4": primary_headroom_gain
        >= float(gates["primary_bf_mr_persistent_gain_vs_k4_min"]),
        "primary_vs_k4_effectiveness_seed_count": int(primary_headroom_ci["n"]) == 5,
        "primary_ci_vs_k4": float(primary_headroom_ci["high"])
        <= float(gates["primary_bf_mr_persistent_difference_ci95_high_max"]),
        "primary_gain_vs_k2": primary_k2_gain
        >= float(gates["primary_persistent_separated_gain_vs_k2_min"]),
        "primary_vs_k2_effectiveness_seed_count": int(primary_k2_ci["n"]) == 5,
        "primary_ci_vs_k2": float(primary_k2_ci["high"])
        <= float(gates["primary_persistent_separated_difference_ci95_high_max"]),
        "headroom_capture": capture_fraction
        >= float(gates["primary_oracle_headroom_capture_fraction_min"]),
        "ipm_noninferiority_vs_k4": ipm_noninferiority["n_seed_pairs"] == 5
        and ipm_noninferiority["exp_log_ratio_ci95_high"]
        <= float(gates["primary_persistent_ipm_exp_log_ratio_ci95_high_to_k4_max"]),
        "alie_noninferiority_vs_k4": alie_noninferiority["n_seed_pairs"] == 5
        and alie_noninferiority["exp_log_ratio_ci95_high"]
        <= float(gates["primary_persistent_alie_exp_log_ratio_ci95_high_to_k4_max"]),
        "intermittent_noninferiority_vs_k4": (
            intermittent_noninferiority["n_seed_pairs"] == 5
            and intermittent_noninferiority["exp_log_ratio_ci95_high"]
            <= float(gates["primary_intermittent_exp_log_ratio_ci95_high_to_k4_max"])
        ),
        "clean_noninferiority_vs_k2": clean_noninferiority["n_seed_pairs"] == 5
        and clean_noninferiority["exp_log_ratio_ci95_high"]
        <= float(
            gates["clean_primary_post_enrollment_exp_log_ratio_ci95_high_to_k2_max"]
        ),
    }
    oracle_headroom_pass = bool(
        checks["oracle_headroom_gain"]
        and checks["oracle_effectiveness_seed_count"]
        and checks["oracle_headroom_ci"]
    )
    return {
        "decision": "promote_to_separate_holdout_runner"
        if all(checks.values())
        else "stop_after_development",
        "all_gates_pass": all(checks.values()),
        "oracle_headroom_mechanism_viable": oracle_headroom_pass,
        "stop_mechanism_if_oracle_headroom_fails": not oracle_headroom_pass,
        "checks": checks,
        "observed": observed,
        "holdout_opened": False,
    }


def _summaries(trajectory_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                "deployable": False,
                "end_to_end_deployability_claimed": False,
                "screen_conditioned_on_semi_oracle_anchor": True,
                "privacy_claimed": candidate != ORACLE,
                "attack_auc_mean": base._finite_mean(row["attack_auc"] for row in rows),
                "attack_auc_std": base._finite_std(row["attack_auc"] for row in rows),
                "active_attack_auc_mean": base._finite_mean(
                    row["active_attack_auc"] for row in rows
                ),
                "recovery_auc_mean": base._finite_mean(
                    row["recovery_auc"] for row in rows
                ),
                "post_enrollment_auc_mean": base._finite_mean(
                    row["post_enrollment_auc"] for row in rows
                ),
                "attack_byzantine_direct_current_mass_share_mean": base._finite_mean(
                    row["attack_byzantine_direct_current_mass_share"] for row in rows
                ),
                "attack_byzantine_imputed_mass_share_mean": base._finite_mean(
                    row["attack_byzantine_imputed_mass_share"] for row in rows
                ),
                "attack_byzantine_total_slot_mass_share_mean_descriptive": (
                    base._finite_mean(
                        row["attack_byzantine_total_slot_mass_share_descriptive"]
                        for row in rows
                    )
                ),
                "attack_predictor_error_vs_honest_clipped_direction_mean": (
                    base._finite_mean(
                        row[
                            "attack_predictor_error_vs_current_honest_clipped_direction"
                        ]
                        for row in rows
                    )
                    if candidate in {PRIMARY, DELTA, STATIC, ORACLE}
                    else float("nan")
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
        "# G0g-K4b — mélange de confiance historique conditionné à une ancre semi-oracle",
        "",
        "> Rapport généré après l'écran de développement. Les seuils K2 et K4 "
        "sont gelés; aucun holdout n'est accessible à ce runner.",
        "",
        "## Décision",
        "",
        f"- Décision : **{decision['decision']}**.",
        f"- Tous les gates passent : **{decision['all_gates_pass']}**.",
        f"- Headroom oracle viable : **{decision['oracle_headroom_mechanism_viable']}**.",
        "- Holdout ouvert : **non**.",
        "- Revendication end-to-end déployable : **non** (ancre semi-oracle).",
        f"- Gates échoués : {', '.join(failed) if failed else 'aucun'}.",
        "",
        "## Contrastes préenregistrés",
        "",
        "| Mesure | Valeur |",
        "|---|---:|",
        f"| Gain oracle vs K4, BF/MR persistants | {observed['oracle_bf_mr_persistent_gain_vs_k4']:.4f} |",
        f"| IC95 haut oracle−K4 | {observed['oracle_bf_mr_persistent_difference_seed_ci95']['high']:.6f} |",
        f"| Gain K4b primaire vs K4, BF/MR persistants | {observed['primary_bf_mr_persistent_gain_vs_k4']:.4f} |",
        f"| IC95 haut K4b−K4 | {observed['primary_bf_mr_persistent_difference_seed_ci95']['high']:.6f} |",
        f"| Gain K4b primaire vs K2, attaques séparées | {observed['primary_persistent_separated_gain_vs_k2']:.4f} |",
        f"| Fraction du headroom oracle capturée | {observed['primary_oracle_headroom_capture_fraction']:.4f} |",
        f"| IC95 haut exp(log-ratio) K4b/K4, IPM persistant | {observed['primary_persistent_ipm_attack_auc_log_ratio']['exp_log_ratio_ci95_high']:.4f} |",
        f"| IC95 haut exp(log-ratio) K4b/K4, ALIE persistant | {observed['primary_persistent_alie_attack_auc_log_ratio']['exp_log_ratio_ci95_high']:.4f} |",
        f"| IC95 haut exp(log-ratio) K4b/K4, intermittent | {observed['primary_intermittent_attack_auc_log_ratio']['exp_log_ratio_ci95_high']:.4f} |",
        f"| IC95 haut exp(log-ratio) propre K4b/K2 | {observed['clean_primary_post_enrollment_auc_log_ratio']['exp_log_ratio_ci95_high']:.4f} |",
        f"| Ablation A : gain vs K4, BF/MR persistants (descriptif) | {observed['delta_bf_mr_persistent_gain_vs_k4_descriptive']:.4f} |",
        "",
        "## Certificats et portée",
        "",
        f"- Device observé : `{observed['device']}`; requis : `mps`.",
        f"- Violations du cap : {observed['contribution_cap_violations']}.",
        f"- Violations replace-one courant : {observed['replace_one_violations']} / {observed['replace_one_trials']}.",
        f"- Maximum observé / borne $2G/n=0,0104$ : {observed['replace_one_max_ratio_to_bound']:.6f}.",
        f"- Violations de causalité du prédicteur déployable : {observed['causal_predictor_violations']}.",
        f"- Violations de l'absence de feedback K4b : {observed['predictor_feedback_violations']}.",
        f"- Violations du marquage oracle : {observed['oracle_metadata_violations']}.",
        "- L'oracle est l'optimum pointwise projeté du mécanisme B; il utilise la cible, n'est pas déployable et ne porte aucune revendication de confidentialité.",
        "- La masse adversariale gatée directe, la masse imputée et la masse totale de slot sont enregistrées séparément; seul le premier contraste est gaté contre K2.",
        "- Sans trajectoire contrefactuelle, une contamination du prédicteur par l'historique n'est pas identifiable séparément du drift et du bruit.",
        "- Le certificat replace-one est conditionnel à un passé, un prédicteur, une ancre et des covariances fixés; ce n'est pas une sensibilité de trajectoire utilisateur.",
        "",
        f"Résultats bruts : `{output_dir.relative_to(ROOT)}`.",
        f"Holdout réservé mais inaccessible : `{config['randomness']['holdout_seeds']}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    lock_provenance = _verify_preregistration_lock(
        lock_path.resolve(),
        config_path=config_path.resolve(),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    k2_calibration, temporal_calibration, provenance = _load_frozen_calibrations(config)
    device, dtype = oracle._configure_runtime("mps")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": base._sha256(config_path),
        "source_sha256": {
            "runner": base._sha256(Path(__file__).resolve()),
            "k4b_primitive": base._sha256(
                ROOT / "algorithms/gaussian_aware_reference_k4b.py"
            ),
            **{
                relative: base._sha256(ROOT / relative)
                for relative in config["frozen_source_hashes"]
            },
        },
        "device": str(device),
        "dtype": str(dtype),
        "mps_required_without_fallback": True,
        "development_only": True,
        "screen_conditioned_on_semi_oracle_anchor": True,
        "end_to_end_deployability_claimed": False,
        "predictor_deployability": ("conditional_on_supplied_past_or_public_anchor"),
        "historical_contamination_counterfactual_available": False,
        "historical_contamination_separately_identified": False,
        "preregistration_lock": lock_provenance,
        "calibration_provenance": provenance,
        "frozen_source_hashes_verified": True,
        "reserved_holdout_seeds": config["randomness"]["holdout_seeds"],
        "holdout_opened": False,
        "status": "running_development_screen",
    }
    base._atomic_json(output_dir / "manifest.json", manifest)
    base._atomic_json(output_dir / "frozen_calibration_provenance.json", provenance)
    try:
        round_rows, client_rows, trajectory_rows = _screen_development(
            config, k2_calibration, temporal_calibration
        )
        replace_rows = _replace_one_audit(config, k2_calibration, temporal_calibration)
        decision = _evaluate_gates(
            round_rows,
            client_rows,
            trajectory_rows,
            replace_rows,
            config,
            device_name=str(device),
        )
        base._write_csv(output_dir / "development_round_rows.csv", round_rows)
        base._write_csv(output_dir / "development_client_rows.csv", client_rows)
        base._write_csv(output_dir / "development_trajectory_rows.csv", trajectory_rows)
        base._write_csv(output_dir / "replace_one_audit.csv", replace_rows)
        base._write_csv(output_dir / "summary.csv", _summaries(trajectory_rows))
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
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/Gaussian_Aware_G0g_K4b_Past_Imputation_Report.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        args.lock.resolve(),
    )


if __name__ == "__main__":
    main()
