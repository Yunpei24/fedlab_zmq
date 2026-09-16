#!/usr/bin/env python3
"""Run the preregistered nested-MC G0g-K4c-CH screen on MPS.

K4c-CH is a development-only current-randomness conditional compensatory
headroom experiment. It estimates a clean-current-state-privileged
conditional-MSE semi-oracle with construction children and evaluates it on an
independent child stream. It is not measurable with respect to the observable
past alone and has no holdout execution path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k4b import (  # noqa: E402
    FULL_TEMPORAL_MISSING_SLOT,
    pointwise_optimal_full_imputation_predictor,
)
from algorithms.gaussian_aware_reference_k4c_ch import (  # noqa: E402
    current_randomness_conditional_mse_semi_oracle,
    fixed_denominator_imputed_reference_from_sum,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k1 as base  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k4b_past_imputation as k4b,
)
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
SEMI_ORACLE = "g0g_k4c_ch_current_randomness_conditional_mse_semi_oracle"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, SEMI_ORACLE, POINTWISE)

FROZEN_K2_SHA256 = "98a30cad8a2f92e6843346e9a9512bd321e8302b9b6e17b447b2641fffb544a5"
FROZEN_K4_SHA256 = "6ca81b5b8897ec19509a569b54fdf9edeac8c6dc65c7b0c4a890142882603075"
FROZEN_C0 = 10.823704719543457
FROZEN_C1 = 11.615572929382324
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k4c_causal_headroom.lock.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
)
DEFAULT_REPORT = ROOT / (
    "output/analysis/Gaussian_Aware_G0g_K4c_Causal_Headroom_Report.md"
)
LOCKED_PATHS = {
    "algorithms/gaussian_aware_reference_k4c_ch.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4c_causal_headroom.yaml",
    "scripts/audit_gaussian_aware_reference_g0g_k4c_ch.py",
    "scripts/run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "tests/test_audit_gaussian_aware_reference_g0g_k4c_ch.py",
    "tests/test_gaussian_aware_reference_g0g_k4c_ch.py",
    "tests/test_run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "output/analysis/Gaussian_Aware_G0g_K4c_Causal_Headroom_Protocol_PreRun.md",
}
DEPENDENCY_PATHS = {
    "algorithms/gaussian_aware_reference.py",
    "algorithms/gaussian_aware_reference_k4b.py",
    "scripts/run_gaussian_aware_reference_g0g_k1.py",
    "scripts/run_gaussian_aware_reference_g0g_k2.py",
    "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
    "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "scripts/run_gaussian_aware_reference_oracle.py",
    "robustness/aggregators.py",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1/calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1/temporal_calibration.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_metric(value: Any) -> str:
    number = float(value) if value is not None else float("nan")
    return f"{number:.6f}" if math.isfinite(number) else "NA"


def _manual_k4_formula_error(
    reference: torch.Tensor,
    anchor: torch.Tensor,
    direct_sum: torch.Tensor,
    num_clients: int,
) -> float:
    """Check K4 against its fixed-denominator identity independently."""

    manual = anchor + direct_sum / float(num_clients)
    return float(torch.linalg.vector_norm(reference - manual).item())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _verify_lock(lock_path: Path, config_path: Path) -> dict[str, Any]:
    if lock_path.resolve() != DEFAULT_LOCK.resolve():
        raise RuntimeError("Production accepts only the preregistered lock path")
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing preregistration lock: {lock_path}")
    registry = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "campaign_id",
        "locked_files",
        "dependencies",
        "lock_file_self_hash_embedded",
        "publication_requirement",
    }
    if set(registry) != expected:
        raise RuntimeError("K4c-CH lock schema mismatch")
    if registry["schema_version"] != 1 or registry["campaign_id"] != (
        "gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
    ):
        raise RuntimeError("K4c-CH lock identity mismatch")
    if registry["lock_file_self_hash_embedded"] is not False:
        raise RuntimeError("The lock must not claim a self-hash")
    if registry["publication_requirement"] != (
        "publish_this_lock_file_sha256_in_research_log_or_chat_before_run"
    ):
        raise RuntimeError("K4c-CH lock publication rule changed")
    if set(registry["locked_files"]) != LOCKED_PATHS:
        raise RuntimeError("K4c-CH locked-file registry mismatch")
    if set(registry["dependencies"]) != DEPENDENCY_PATHS:
        raise RuntimeError("K4c-CH dependency registry mismatch")
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RuntimeError("Production accepts only the preregistered config")
    for relative, expected_hash in {
        **registry["locked_files"],
        **registry["dependencies"],
    }.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise RuntimeError(f"Preregistration hash mismatch: {relative}")
    return {
        "path": str(lock_path.resolve()),
        "sha256": _sha256(lock_path),
        "verified": True,
        "locked_files": len(LOCKED_PATHS),
        "dependencies": len(DEPENDENCY_PATHS),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id",
        "scope",
        "preregistration_lock",
        "scientific_contract",
        "frozen_calibrations",
        "cohort",
        "privacy_noise",
        "references",
        "temporal",
        "honest_dynamics",
        "aggregation",
        "threats",
        "nested_monte_carlo",
        "randomness",
        "candidates",
        "statistical_analysis",
        "gates",
        "execution",
    }
    if set(config) != expected:
        raise ValueError("K4c-CH top-level schema changed")
    if config["campaign_id"] != (
        "gaussian_aware_reference_g0g_k4c_causal_headroom_mps_v1"
    ):
        raise ValueError("Unexpected K4c-CH campaign id")
    contract = config["scientific_contract"]
    declared_lock = Path(str(config["preregistration_lock"]["path"]))
    if not declared_lock.is_absolute():
        declared_lock = ROOT / declared_lock
    if declared_lock.resolve() != DEFAULT_LOCK.resolve():
        raise ValueError("YAML preregistration lock path changed")
    if config["preregistration_lock"] != {
        "path": (
            "configs/ldp_gradient_far/"
            "gaussian_aware_reference_g0g_k4c_causal_headroom.lock.json"
        ),
        "verify_before_output_creation": True,
        "publish_lock_sha256_before_scientific_run": True,
        "external_publication_is_procedural_precondition": True,
    }:
        raise ValueError("K4c-CH lock preconditions changed")
    if contract["conditional_loss"] != "squared_l2_reference_error":
        raise ValueError("K4c-CH must optimize conditional MSE")
    if contract["semi_oracle_is_exact_for_expected_l2_norm"] is not False:
        raise ValueError("K4c-CH must not claim optimality for E||error||")
    if contract["ordinary_l2_error_role"] != "descriptive_only":
        raise ValueError("Ordinary norm error must remain descriptive")
    if contract["compensatory_headroom_components_separately_identified"] is not False:
        raise ValueError("K4c-CH cannot attribute compensatory headroom components")
    if contract["no_noise_no_attack_counterfactual_included"] is not False:
        raise ValueError("K4c-CH does not include a causal no-noise/no-attack contrast")
    if contract["pointwise_oracle_used_for_construction"] is not False:
        raise ValueError("Pointwise oracles are forbidden in construction")
    if contract["pointwise_oracle_role"] != (
        "nondeployable_decision_benchmark_headroom_denominator_only"
    ):
        raise ValueError("Pointwise oracle decision role changed")
    if contract["conditions_on_latent_clean_current_state"] is not True:
        raise ValueError("K4c-CH must disclose its privileged clean-state conditioning")
    if contract["observable_past_only_predictor_constructed"] is not False:
        raise ValueError("K4c-CH must not claim a transcript-only predictor")
    if contract["pass_authorizes_holdout_or_promotion"] is not False:
        raise ValueError("A K4c-CH pass cannot authorize promotion or holdout")
    if contract["valid_scientific_fail_stops_branch"] is not True:
        raise ValueError(
            "Only a valid screen with a scientific failure may stop this branch"
        )
    if contract["conditioning_sigma_field"] != (
        "G_t_minus_observable_past_latent_clean_current_state_"
        "simulator_labels_and_configured_current_dgp"
    ):
        raise ValueError("K4c-CH conditioning sigma-field changed")
    if contract["semi_oracle_is_observable_past_measurable"] is not False:
        raise ValueError("The semi-oracle cannot be labelled past-only")
    if contract["semi_oracle_is_guaranteed_exact_optimum_at_64_children"] is not False:
        raise ValueError("Finite-MC construction cannot claim the exact optimum")
    if not contract["uses_simulator_honest_byzantine_labels"]:
        raise ValueError("K4c-CH must disclose simulator-only identity labels")
    if not contract["knows_configured_current_noise_and_attack_generator"]:
        raise ValueError("K4c-CH must disclose knowledge of the configured DGP")
    if contract["arbitrary_adaptive_byzantine_behavior_supported"] is not False:
        raise ValueError("K4c-CH does not cover arbitrary adaptive attacks")
    if contract["aggregate_denominator"] != "public_cohort_size_n":
        raise ValueError("K4c-CH requires the fixed public denominator")
    if contract["imputation_rule"] != "one_minus_history_gate":
        raise ValueError("K4c-CH must retain the K4b primary mechanism")
    if contract["missing_slot_zero_tolerance_formula"] != (
        "64_times_machine_epsilon_of_runtime_dtype"
    ):
        raise ValueError("Missing-slot zero tolerance changed")
    if contract["development_only_screen"] is not True:
        raise ValueError("K4c-CH is development-only")
    if tuple(config["candidates"]["names"]) != CANDIDATES:
        raise ValueError("K4c-CH candidate set changed")
    if config["candidates"]["semi_oracle_primary"] != SEMI_ORACLE:
        raise ValueError("Unexpected K4c-CH semi-oracle candidate")
    if config["candidates"]["pointwise_decision_benchmark"] != POINTWISE:
        raise ValueError("Unexpected pointwise decision benchmark")
    frozen = config["frozen_calibrations"]
    if frozen["k2"]["sha256"] != FROZEN_K2_SHA256:
        raise ValueError("Unexpected K2 calibration hash")
    if frozen["k4_temporal"]["sha256"] != FROZEN_K4_SHA256:
        raise ValueError("Unexpected K4 calibration hash")
    if not frozen["reuse_thresholds_exactly"] or frozen["recalibrate"]:
        raise ValueError("K2/K4 thresholds must remain frozen")
    nested = config["nested_monte_carlo"]
    construction = int(nested["construction_children"])
    if construction != int(nested["construction_split_a_children"]) + int(
        nested["construction_split_b_children"]
    ):
        raise ValueError("Construction split must partition all children")
    if construction < 2 or int(nested["evaluation_children"]) < 2:
        raise ValueError("Nested streams require at least two children")
    if nested["construction_stream_tag"] == nested["evaluation_stream_tag"]:
        raise ValueError("Construction/evaluation tags must differ")
    if nested["projection_rule"] != "once_after_both_conditional_expectations":
        raise ValueError("Projection must follow the two expectations")
    if nested["average_pointwise_projected_oracles"] is not False:
        raise ValueError("Averaging pointwise projected oracles is forbidden")
    if nested["conditional_expectations"]["conditioning"] != (
        "G_t_minus_frozen_observable_past_plus_fixed_latent_clean_"
        "current_state_simulator_labels_and_configured_current_dgp"
    ):
        raise ValueError("Nested-MC conditioning was weakened or mislabelled")
    frozen_children = set(nested["frozen_across_children"])
    if not {"clean_client_offsets", "drift"}.issubset(frozen_children):
        raise ValueError("The privileged latent clean state must be frozen explicitly")
    analysis = config["statistical_analysis"]
    seeds = tuple(
        int(value) for value in config["randomness"]["development_outer_seeds"]
    )
    holdout = tuple(
        int(value) for value in config["randomness"]["reserved_holdout_seeds"]
    )
    if len(seeds) != int(analysis["num_independent_units"]) or len(set(seeds)) != len(
        seeds
    ):
        raise ValueError("Outer-seed registry does not match the analysis")
    if set(seeds) & set(holdout):
        raise ValueError("Development and reserved holdout seeds overlap")
    if len(holdout) != 7 or len(set(holdout)) != 7:
        raise ValueError("K4c-CH requires seven untouched reserved holdout seeds")
    if config["randomness"]["holdout_rule"] != (
        "no_holdout_trajectory_or_data_execution_path_registry_metadata_"
        "validation_allowed"
    ):
        raise ValueError("Holdout metadata/read boundary changed")
    prior_seeds: set[int] = set()
    for pattern in config["randomness"]["prior_seed_registry_globs"]:
        for path in ROOT.glob(str(pattern)):
            prior_config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if (
                not isinstance(prior_config, Mapping)
                or prior_config.get("campaign_id") == config["campaign_id"]
            ):
                continue
            prior_seeds |= k4b._seed_values(prior_config)
    collision = (set(seeds) | set(holdout)) & prior_seeds
    if collision:
        raise ValueError(f"K4c-CH seeds overlap prior registries: {sorted(collision)}")
    if (
        analysis["unit"] != "outer_seed"
        or not analysis["no_child_or_history_pseudoreplication"]
    ):
        raise ValueError("The statistical unit must be the outer seed")
    if analysis["eligible_history_rule"] != (
        "all_preregistered_histories_with_R_positive_no_outcome_filter"
    ):
        raise ValueError("Outcome-dependent history selection is forbidden")
    status_logic = analysis["decision_status_logic"]
    if status_logic != {
        "invalid_or_inconclusive_if_any_validity_check_fails": True,
        "evaluate_scientific_branch_stop_only_after_all_validity_checks_pass": True,
        "all_checks_pass_status": "authorize_transcript_only_predictor_study",
        "valid_but_scientific_check_fails_status": (
            "stop_missing_slot_imputation_branch"
        ),
        "validity_check_fails_status": "invalid_or_inconclusive_screen",
    }:
        raise ValueError("K4c-CH decision-status logic changed")
    if int(analysis["frozen_histories_total"]) != 576:
        raise ValueError("The frozen production matrix must contain 576 histories")
    if int(analysis["child_draws_total"]) != 73728:
        raise ValueError("The nested production matrix must contain 73728 child draws")
    if (
        int(analysis["construction_child_seeds_total"]) != 36864
        or int(analysis["evaluation_child_seeds_total"]) != 36864
    ):
        raise ValueError("The nested RNG registries must each contain 36864 seeds")
    if int(config["randomness"]["replace_one_trials_per_seed_noise_cell"]) != int(
        config["cohort"]["num_clients"]
    ):
        raise ValueError(
            "Replace-one implementation audit must cover every client slot"
        )
    expected_replace_trials = (
        len(seeds)
        * sum(
            len(tuple(regime["permutations"]))
            for regime in config["privacy_noise"]["regimes"]
        )
        * int(config["cohort"]["num_clients"])
    )
    if int(config["gates"]["replace_one_exact_trials"]) != expected_replace_trials:
        raise ValueError(
            "Replace-one exact-trial gate does not match balanced coverage"
        )
    expected_eligible_per_cell = (
        len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["honest_dynamics"]["names"])
        * len(config["threats"]["names"])
        * len(config["temporal"]["assessment_rounds"])
    )
    if (
        int(config["gates"]["eligible_histories_per_seed_noise_cell_exact"])
        != expected_eligible_per_cell
    ):
        raise ValueError("Eligible-history cell composition gate changed")
    if float(config["gates"]["eligible_R_positive_history_fraction_min"]) != 1.0:
        raise ValueError("Every preregistered history must be R-eligible")
    execution = config["execution"]
    if execution != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
        "development_only": True,
        "holdout_code_path_present": False,
        "expected_runtime_minutes_mps": [35, 60],
    }:
        raise ValueError("Production K4c-CH must be MPS-only without holdout")


def _load_calibrations(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    k2, temporal, provenance = k4b._load_frozen_calibrations(config)
    if not math.isclose(float(temporal["deployed_c0"]), FROZEN_C0, abs_tol=1e-12):
        raise RuntimeError("Frozen K4 c0 changed")
    if not math.isclose(float(temporal["deployed_c1"]), FROZEN_C1, abs_tol=1e-12):
        raise RuntimeError("Frozen K4 c1 changed")
    return k2, temporal, provenance


def _attest_external_lock_publication(
    verified_lock: Mapping[str, Any], published_lock_sha256: str
) -> dict[str, Any]:
    """Require the exact externally published lock hash before any output."""

    provided = str(published_lock_sha256).strip().lower()
    expected = str(verified_lock["sha256"]).lower()
    if provided != expected:
        raise RuntimeError(
            "The --published-lock-sha256 attestation must equal the verified "
            "lock hash. Publish that hash externally before running."
        )
    return {
        "procedural_external_publication_attested": True,
        "published_lock_sha256": provided,
        "machine_verifies_external_log_itself": False,
    }


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], str]]:
    return base._noise_cells(config)


def _noise_cell_names(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(regime["name"]), str(permutation))
        for regime, permutation in _noise_cells(config)
    ]


def _outer_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in config["randomness"]["development_outer_seeds"]:
        for regime, permutation in _noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for dynamics in config["honest_dynamics"]["names"]:
                    for threat in config["threats"]["names"]:
                        result.append(
                            {
                                "seed": int(seed),
                                "regime": regime,
                                "permutation": str(permutation),
                                "geometry": str(geometry),
                                "dynamics": str(dynamics),
                                "threat": str(threat),
                            }
                        )
    expected = int(config["statistical_analysis"]["frozen_histories_total"]) // len(
        config["temporal"]["assessment_rounds"]
    )
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} outer trajectories, got {len(result)}")
    return result


def _history_id(cell: Mapping[str, Any], round_index: int) -> str:
    return "|".join(
        (
            str(cell["seed"]),
            str(cell["regime"]["name"]),
            str(cell["permutation"]),
            str(cell["geometry"]),
            str(cell["dynamics"]),
            str(cell["threat"]),
            str(round_index),
        )
    )


def _child_seed(
    config: Mapping[str, Any],
    cell: Mapping[str, Any],
    round_index: int,
    stream: str,
    child: int,
) -> int:
    tag = str(config["nested_monte_carlo"][f"{stream}_stream_tag"])
    return oracle._seed(
        tag,
        cell["seed"],
        cell["regime"]["name"],
        cell["permutation"],
        cell["geometry"],
        cell["dynamics"],
        cell["threat"],
        round_index,
        child,
    )


def _current_child(
    config: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    cell: Mapping[str, Any],
    *,
    round_index: int,
    stream: str,
    child: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seed = _child_seed(config, cell, round_index, stream, child)
    clean = k4._clean_at_round(components, round_index)
    observed = base._paired_private_noise(
        clean,
        components["variances"],
        tuple(int(value) for value in config["cohort"]["block_sizes"]),
        seed=seed,
        draw=round_index,
        geometry=str(cell["geometry"]),
    )
    attacked, _ = oracle._replace_with_attack(
        observed,
        config,
        threat=str(cell["threat"]),
        severity=float(config["threats"]["severity"]),
        seed=oracle._seed("k4c-ch-child-attack", stream, seed),
    )
    return attacked, clean, seed


def _current_components(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    latent_byzantine: torch.Tensor,
    *,
    regime_name: str,
    vectors: torch.Tensor,
    clean: torch.Tensor,
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
) -> dict[str, Any]:
    aware_radii = k4._radii(
        config,
        components["variances"],
        k2_calibration,
        regime_name=regime_name,
        blind=False,
    )
    k4_reference, diagnostics = k4._k4_reference(
        vectors,
        anchor=components["anchor"],
        aware_radii=aware_radii,
        history=history,
        enrollment_mean=enrollment_mean,
        thresholds=temporal_calibration,
        config=config,
    )
    gates = torch.tensor(
        diagnostics["gates_by_client"],
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    )
    history_gates = torch.tensor(
        diagnostics["temporal_gates_by_client"],
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    )
    clipped = k4b._clipped_residuals(
        vectors,
        components["anchor"],
        float(config["references"]["total_client_influence_cap"]),
    )
    direct_sum = torch.sum(gates[:, None] * clipped, dim=0)
    target = clean[~latent_byzantine].mean(dim=0)
    return {
        "k4_reference": k4_reference,
        "diagnostics": diagnostics,
        "gates": gates,
        "history_gates": history_gates,
        "clipped": clipped,
        "direct_sum": direct_sum,
        "target": target,
        "target_direction": target - components["anchor"],
        "missing_slot_mass": float(torch.sum(1.0 - history_gates).item()),
    }


def _evaluate_snapshot(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    cell: Mapping[str, Any],
    *,
    round_index: int,
    history: Sequence[torch.Tensor],
    enrollment_mean: torch.Tensor,
    residuals_by_round: Mapping[int, torch.Tensor],
    gates_by_round: Mapping[int, torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    n = int(config["cohort"]["num_clients"])
    b = int(config["cohort"]["num_byzantine"])
    latent_byzantine = torch.zeros(n, dtype=torch.bool, device=oracle._RUNTIME_DEVICE)
    latent_byzantine[n - b :] = True
    construction_count = int(config["nested_monte_carlo"]["construction_children"])
    evaluation_count = int(config["nested_monte_carlo"]["evaluation_children"])
    cap = float(config["references"]["total_client_influence_cap"])
    history_id = _history_id(cell, round_index)
    source_rounds = tuple(range(round_index - 4, round_index))
    rolling, rolling_diagnostics = k4b._predictor_from_rounds(
        residuals_by_round,
        gates_by_round,
        source_rounds,
        config={
            **config,
            "past_imputation": {
                "history_length": 4,
                "minimum_accepted_mass": n - b,
            },
        },
    )

    construction_targets: list[torch.Tensor] = []
    construction_sums: list[torch.Tensor] = []
    construction_seeds: list[int] = []
    fixed_history_gate: torch.Tensor | None = None
    for child in range(construction_count):
        observed, clean, child_seed = _current_child(
            config,
            components,
            cell,
            round_index=round_index,
            stream="construction",
            child=child,
        )
        vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
        values = _current_components(
            config,
            k2_calibration,
            temporal_calibration,
            components,
            latent_byzantine,
            regime_name=str(cell["regime"]["name"]),
            vectors=vectors,
            clean=clean,
            history=history,
            enrollment_mean=enrollment_mean,
        )
        if fixed_history_gate is None:
            fixed_history_gate = values["history_gates"].clone()
        elif not torch.equal(fixed_history_gate, values["history_gates"]):
            raise RuntimeError("History gate changed across current-noise children")
        construction_targets.append(values["target_direction"])
        construction_sums.append(values["direct_sum"])
        construction_seeds.append(child_seed)
    if fixed_history_gate is None:
        raise RuntimeError("No construction children were generated")
    missing_mass = float(torch.sum(1.0 - fixed_history_gate).item())
    target_matrix = torch.stack(construction_targets)
    direct_matrix = torch.stack(construction_sums)
    semi_oracle, semi_oracle_diagnostics = (
        current_randomness_conditional_mse_semi_oracle(
            target_matrix,
            direct_matrix,
            missing_slot_mass=missing_mass,
            num_clients=n,
            influence_cap=cap,
            return_diagnostics=True,
        )
    )
    split_a_count = int(config["nested_monte_carlo"]["construction_split_a_children"])
    split_a = current_randomness_conditional_mse_semi_oracle(
        target_matrix[:split_a_count],
        direct_matrix[:split_a_count],
        missing_slot_mass=missing_mass,
        num_clients=n,
        influence_cap=cap,
    )
    split_b = current_randomness_conditional_mse_semi_oracle(
        target_matrix[split_a_count:],
        direct_matrix[split_a_count:],
        missing_slot_mass=missing_mass,
        num_clients=n,
        influence_cap=cap,
    )

    child_rows: list[dict[str, Any]] = []
    evaluation_seeds: list[int] = []
    first_eval: dict[str, Any] | None = None
    for child in range(evaluation_count):
        observed, clean, child_seed = _current_child(
            config,
            components,
            cell,
            round_index=round_index,
            stream="evaluation",
            child=child,
        )
        vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
        values = _current_components(
            config,
            k2_calibration,
            temporal_calibration,
            components,
            latent_byzantine,
            regime_name=str(cell["regime"]["name"]),
            vectors=vectors,
            clean=clean,
            history=history,
            enrollment_mean=enrollment_mean,
        )
        if not torch.equal(fixed_history_gate, values["history_gates"]):
            raise RuntimeError("Frozen history gate changed on evaluation stream")
        if not math.isclose(values["missing_slot_mass"], missing_mass, abs_tol=1e-7):
            raise RuntimeError("Missing-slot mass changed across children")
        pointwise = pointwise_optimal_full_imputation_predictor(
            values["clipped"],
            values["gates"],
            values["history_gates"],
            target_direction=values["target_direction"],
            influence_cap=cap,
        )
        aware_radii = k4._radii(
            config,
            components["variances"],
            k2_calibration,
            regime_name=str(cell["regime"]["name"]),
            blind=False,
        )
        references = {
            K4: values["k4_reference"],
            K4B: fixed_denominator_imputed_reference_from_sum(
                anchor=components["anchor"],
                direct_sum=values["direct_sum"],
                predictor=rolling,
                missing_slot_mass=missing_mass,
                num_clients=n,
                influence_cap=cap,
            ),
            SEMI_ORACLE: fixed_denominator_imputed_reference_from_sum(
                anchor=components["anchor"],
                direct_sum=values["direct_sum"],
                predictor=semi_oracle,
                missing_slot_mass=missing_mass,
                num_clients=n,
                influence_cap=cap,
            ),
            POINTWISE: fixed_denominator_imputed_reference_from_sum(
                anchor=components["anchor"],
                direct_sum=values["direct_sum"],
                predictor=pointwise,
                missing_slot_mass=missing_mass,
                num_clients=n,
                influence_cap=cap,
            ),
        }
        references[K2], _ = k4._current_reference(
            k4.AWARE,
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            blind_radii=aware_radii,
            config=config,
        )
        reproduced_k4, _ = k4b._frozen_k4_comparator(
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            history=history,
            enrollment_mean=enrollment_mean,
            temporal_calibration=temporal_calibration,
            config=config,
        )
        reproduced_k4b, _ = k4b._k4b_reference(
            vectors,
            anchor=components["anchor"],
            aware_radii=aware_radii,
            history=history,
            enrollment_mean=enrollment_mean,
            predictor=rolling,
            predictor_role="frozen_k4b_rolling_past_imputation",
            imputation_mode=FULL_TEMPORAL_MISSING_SLOT,
            deployable=False,
            privacy_claimed=False,
            temporal_calibration=temporal_calibration,
            config=config,
        )
        frozen_k4_comparator_error = float(
            torch.linalg.vector_norm(references[K4] - reproduced_k4).item()
        )
        k4_manual_formula_error = _manual_k4_formula_error(
            references[K4],
            components["anchor"],
            values["direct_sum"],
            n,
        )
        k4b_reproduction_error = float(
            torch.linalg.vector_norm(references[K4B] - reproduced_k4b).item()
        )
        manual_semi_oracle = components["anchor"] + (
            values["direct_sum"] + missing_mass * semi_oracle
        ) / float(n)
        fixed_denominator_formula_error = float(
            torch.linalg.vector_norm(
                references[SEMI_ORACLE] - manual_semi_oracle
            ).item()
        )
        contributions = (
            values["gates"][:, None] * values["clipped"]
            + (1.0 - values["history_gates"])[:, None] * semi_oracle[None, :]
        )
        cap_ok = bool(
            (
                torch.linalg.vector_norm(contributions, dim=1)
                <= cap + 64.0 * torch.finfo(contributions.dtype).eps
            ).all()
        )
        if first_eval is None:
            first_eval = {
                "vectors": vectors.clone(),
                "history_gate": values["history_gates"].clone(),
                "predictor": semi_oracle.clone(),
            }
        squared_errors = {
            candidate: float(torch.sum((reference - values["target"]).square()).item())
            for candidate, reference in references.items()
        }
        l2_errors = {
            candidate: float(
                torch.linalg.vector_norm(reference - values["target"]).item()
            )
            for candidate, reference in references.items()
        }
        for candidate, reference in references.items():
            child_rows.append(
                {
                    "history_id": history_id,
                    "seed": int(cell["seed"]),
                    "noise_regime": str(cell["regime"]["name"]),
                    "noise_permutation": str(cell["permutation"]),
                    "outlier_geometry": str(cell["geometry"]),
                    "honest_dynamics": str(cell["dynamics"]),
                    "threat": str(cell["threat"]),
                    "assessment_round": round_index,
                    "evaluation_child": child,
                    "evaluation_child_seed": child_seed,
                    "crn_key": f"{history_id}|evaluation|{child}",
                    "candidate": candidate,
                    "squared_reference_error": squared_errors[candidate],
                    "reference_error_l2_descriptive": l2_errors[candidate],
                    "pointwise_excess_mse_over_candidate": (
                        squared_errors[POINTWISE] - squared_errors[candidate]
                        if candidate in {K4, K4B, SEMI_ORACLE}
                        else 0.0
                    ),
                    "missing_slot_mass": missing_mass,
                    "predictor_norm": float(
                        torch.linalg.vector_norm(
                            semi_oracle
                            if candidate == SEMI_ORACLE
                            else rolling
                            if candidate == K4B
                            else pointwise
                            if candidate == POINTWISE
                            else torch.zeros_like(semi_oracle)
                        ).item()
                    ),
                    "semi_oracle_predictor_hash": hashlib.sha256(
                        semi_oracle.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "semi_oracle_predictor_fixed_across_evaluation_children": True,
                    "construction_stream_read": candidate == SEMI_ORACLE,
                    "pointwise_oracle": candidate == POINTWISE,
                    "pointwise_oracle_used_in_construction": False,
                    "conditions_on_latent_clean_current_state": (
                        candidate == SEMI_ORACLE
                    ),
                    "predictor_observable_past_only": (
                        True
                        if candidate == K4B
                        else False
                        if candidate in {SEMI_ORACLE, POINTWISE}
                        else None
                    ),
                    "fixed_denominator_n": n,
                    "normalization_by_gate_sum": False,
                    "k4_manual_formula_error": k4_manual_formula_error,
                    "frozen_k4_comparator_error_descriptive": (
                        frozen_k4_comparator_error
                    ),
                    "k4b_comparator_reproduction_error": k4b_reproduction_error,
                    "fixed_denominator_formula_error": (
                        fixed_denominator_formula_error
                    ),
                    "contribution_cap_respected": cap_ok,
                    "all_finite": (
                        bool(torch.isfinite(reference).all())
                        and math.isfinite(squared_errors[candidate])
                        and math.isfinite(l2_errors[candidate])
                        and math.isfinite(
                            squared_errors[POINTWISE] - squared_errors[candidate]
                        )
                        and math.isfinite(missing_mass)
                        and math.isfinite(k4_manual_formula_error)
                        and math.isfinite(frozen_k4_comparator_error)
                        and math.isfinite(k4b_reproduction_error)
                        and math.isfinite(fixed_denominator_formula_error)
                    ),
                }
            )
        evaluation_seeds.append(child_seed)
    if first_eval is None:
        raise RuntimeError("No evaluation children were generated")
    overlap = len(set(construction_seeds) & set(evaluation_seeds))
    construction_seed_registry = ",".join(str(value) for value in construction_seeds)
    evaluation_seed_registry = ",".join(str(value) for value in evaluation_seeds)
    split_disagreement = float(torch.linalg.vector_norm(split_a - split_b).item())
    history_row = {
        "history_id": history_id,
        "seed": int(cell["seed"]),
        "noise_regime": str(cell["regime"]["name"]),
        "noise_permutation": str(cell["permutation"]),
        "outlier_geometry": str(cell["geometry"]),
        "honest_dynamics": str(cell["dynamics"]),
        "threat": str(cell["threat"]),
        "assessment_round": round_index,
        "construction_children": construction_count,
        "evaluation_children": evaluation_count,
        "construction_evaluation_seed_overlap": overlap,
        "construction_child_seed_registry": construction_seed_registry,
        "evaluation_child_seed_registry": evaluation_seed_registry,
        "construction_child_seed_registry_sha256": hashlib.sha256(
            construction_seed_registry.encode("utf-8")
        ).hexdigest(),
        "evaluation_child_seed_registry_sha256": hashlib.sha256(
            evaluation_seed_registry.encode("utf-8")
        ).hexdigest(),
        "construction_child_seed_unique_count": len(set(construction_seeds)),
        "evaluation_child_seed_unique_count": len(set(evaluation_seeds)),
        "missing_slot_mass": missing_mass,
        "eligible_R_positive": missing_mass > 64.0 * torch.finfo(semi_oracle.dtype).eps,
        "rolling_predictor_norm": float(torch.linalg.vector_norm(rolling).item()),
        "semi_oracle_predictor_norm": float(
            torch.linalg.vector_norm(semi_oracle).item()
        ),
        "semi_oracle_raw_predictor_norm": semi_oracle_diagnostics["raw_predictor_norm"],
        "semi_oracle_projection_active": semi_oracle_diagnostics["raw_predictor_norm"]
        > cap,
        "split_a_b_predictor_distance": split_disagreement,
        "split_a_b_aggregate_scale_distance": missing_mass
        / float(n)
        * split_disagreement,
        "predictor_source_rounds": ",".join(
            str(value) for value in rolling_diagnostics["source_rounds"]
        ),
        "predictor_max_source_round": rolling_diagnostics["maximum_source_round"],
        "semi_oracle_uses_evaluation_target": False,
        "semi_oracle_uses_evaluation_noise": False,
        "semi_oracle_uses_k4b_feedback": False,
        "semi_oracle_conditions_on_latent_clean_current_state": True,
        "semi_oracle_uses_simulator_honest_byzantine_labels": True,
        "semi_oracle_knows_configured_current_noise_attack_dgp": True,
        "semi_oracle_conditioning_sigma_field": (
            "G_t_minus_observable_past_latent_clean_current_state_"
            "simulator_labels_and_configured_current_dgp"
        ),
        "arbitrary_adaptive_byzantine_behavior_supported": False,
        "semi_oracle_is_observable_past_measurable": False,
        "projection_applied_once_after_expectations": True,
        "pointwise_oracles_averaged": False,
    }
    audit_context = {
        "history_id": history_id,
        "vectors": first_eval["vectors"],
        "history": list(history),
        "enrollment_mean": enrollment_mean,
        "aware_radii": k4._radii(
            config,
            components["variances"],
            k2_calibration,
            regime_name=str(cell["regime"]["name"]),
            blind=False,
        ),
        "anchor": components["anchor"],
        "predictor": first_eval["predictor"],
        "missing_slot_mass": missing_mass,
    }
    return history_row, child_rows, audit_context


def _outer_observed(
    config: Mapping[str, Any],
    components: Mapping[str, torch.Tensor],
    cell: Mapping[str, Any],
    round_index: int,
) -> torch.Tensor:
    observed = k4._private_vectors(
        config,
        components,
        seed=int(cell["seed"]),
        round_index=round_index,
        geometry=str(cell["geometry"]),
    )
    if (
        int(config["temporal"]["attack_start_round"])
        <= round_index
        <= int(config["temporal"]["attack_end_round"])
    ):
        observed, _ = oracle._replace_with_attack(
            observed,
            config,
            threat=str(cell["threat"]),
            severity=float(config["threats"]["severity"]),
            seed=oracle._seed(
                "g0g-k4-attack",
                cell["seed"],
                cell["regime"]["name"],
                cell["permutation"],
                cell["geometry"],
                cell["dynamics"],
                cell["threat"],
                round_index,
            ),
        )
    return observed


def _evaluate_outer_trajectory(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    components = k4._trajectory_components(
        config,
        seed=int(cell["seed"]),
        regime=dict(cell["regime"]),
        permutation=str(cell["permutation"]),
        geometry=str(cell["geometry"]),
        dynamics=str(cell["dynamics"]),
    )
    history: list[torch.Tensor] = []
    enrollment_mean: torch.Tensor | None = None
    residuals: dict[int, torch.Tensor] = {}
    gates: dict[int, torch.Tensor] = {}
    history_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    replace_contexts: list[dict[str, Any]] = []
    assessment = {int(value) for value in config["temporal"]["assessment_rounds"]}
    total = int(config["temporal"]["total_rounds_needed_for_frozen_pasts"])
    for round_index in range(1, total + 1):
        if round_index in assessment:
            if enrollment_mean is None:
                raise RuntimeError("Enrollment baseline missing before snapshot")
            history_row, children, context = _evaluate_snapshot(
                config,
                k2_calibration,
                temporal_calibration,
                components,
                cell,
                round_index=round_index,
                history=history,
                enrollment_mean=enrollment_mean,
                residuals_by_round=residuals,
                gates_by_round=gates,
            )
            history_rows.append(history_row)
            child_rows.extend(children)
            replace_contexts.append(context)
        observed = _outer_observed(config, components, cell, round_index)
        vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
        standardized = k4._standardized_messages(
            config,
            vectors,
            anchor=components["anchor"],
            variances=components["variances"],
        )
        if round_index == int(config["temporal"]["enrollment_rounds"]):
            enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)
        aware_radii = k4._radii(
            config,
            components["variances"],
            k2_calibration,
            regime_name=str(cell["regime"]["name"]),
            blind=False,
        )
        if round_index < int(config["temporal"]["first_temporal_gate_round"]):
            _, diagnostics = k4._current_reference(
                k4.AWARE,
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                blind_radii=aware_radii,
                config=config,
            )
        else:
            if enrollment_mean is None:
                raise RuntimeError("Enrollment baseline missing")
            _, diagnostics = k4._k4_reference(
                vectors,
                anchor=components["anchor"],
                aware_radii=aware_radii,
                history=history,
                enrollment_mean=enrollment_mean,
                thresholds=temporal_calibration,
                config=config,
            )
        residuals[round_index] = k4b._clipped_residuals(
            vectors,
            components["anchor"],
            float(config["references"]["total_client_influence_cap"]),
        )
        gates[round_index] = torch.tensor(
            diagnostics["gates_by_client"],
            dtype=oracle._RUNTIME_DTYPE,
            device=oracle._RUNTIME_DEVICE,
        )
        history.append(standardized)
    return history_rows, child_rows, replace_contexts


def _replace_one_rows(
    config: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cap = float(config["references"]["total_client_influence_cap"])
    n = int(config["cohort"]["num_clients"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_noise_cell"])
    selected: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for context in contexts:
        parts = str(context["history_id"]).split("|")
        key = (int(parts[0]), parts[1], parts[2])
        if parts[3:] == ["aligned", "stationary", "bitflip_x10", "17"]:
            selected[key] = context
    rows: list[dict[str, Any]] = []
    for key, context in sorted(selected.items()):
        original = context["vectors"]
        for trial in range(trials):
            client = trial % n
            neighbour = original.clone()
            direction = torch.zeros(
                original.shape[1],
                dtype=original.dtype,
                device=original.device,
            )
            direction[trial % original.shape[1]] = cap
            neighbour[client] = context["anchor"] - direction
            common = {
                "anchor": context["anchor"],
                "aware_radii": context["aware_radii"],
                "history": context["history"],
                "enrollment_mean": context["enrollment_mean"],
                "predictor": context["predictor"],
                "predictor_role": (
                    "fixed_current_randomness_conditional_clean_state_mse_semi_oracle"
                ),
                "imputation_mode": FULL_TEMPORAL_MISSING_SLOT,
                "deployable": False,
                "privacy_claimed": False,
                "temporal_calibration": temporal_calibration,
                "config": config,
            }
            left, left_diagnostics = k4b._k4b_reference(original, **common)
            right, _ = k4b._k4b_reference(neighbour, **common)
            difference = float(torch.linalg.vector_norm(left - right).item())
            bound = 2.0 * cap / float(n)
            rows.append(
                {
                    "seed": key[0],
                    "noise_regime": key[1],
                    "noise_permutation": key[2],
                    "trial": trial,
                    "replaced_client": client,
                    "history_id": context["history_id"],
                    "same_past": True,
                    "same_predictor": True,
                    "same_history_gate": True,
                    "certificate_scope": (
                        "evaluation_map_only_with_fixed_anchor_past_history_"
                        "radii_and_semi_oracle_predictor"
                    ),
                    "end_to_end_semi_oracle_sensitivity_claimed": False,
                    "history_gate_has_suppression": min(
                        left_diagnostics["temporal_gates_by_client"]
                    )
                    < 1.0,
                    "observed_difference": difference,
                    "theoretical_bound": bound,
                    "ratio_to_bound": difference / bound,
                    "violation": difference > bound + 1.0e-6,
                }
            )
    return rows


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def _parse_child_seed_registry(value: Any) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(token) for token in text.split(",")]


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return {
            "n": len(finite),
            "mean": _mean(finite),
            "sd": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
        }
    mean = statistics.fmean(finite)
    sd = statistics.stdev(finite)
    half = float(t_critical) * sd / math.sqrt(len(finite))
    return {
        "n": len(finite),
        "mean": mean,
        "sd": sd,
        "low": mean - half,
        "high": mean + half,
    }


def _decision_status(
    validity_checks: Mapping[str, bool], scientific_checks: Mapping[str, bool]
) -> str:
    """Apply the preregistered asymmetric validity/science decision rule."""

    if not all(validity_checks.values()):
        return "invalid_or_inconclusive_screen"
    if not all(scientific_checks.values()):
        return "stop_missing_slot_imputation_branch"
    return "authorize_transcript_only_predictor_study"


def _matrix_is_exact(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Verify the exact preregistered history and child-key composition."""

    expected_histories = {
        "|".join(
            (
                str(seed),
                str(regime["name"]),
                str(permutation),
                str(geometry),
                str(dynamics),
                str(threat),
                str(round_index),
            )
        )
        for seed in config["randomness"]["development_outer_seeds"]
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
        for geometry in config["cohort"]["honest_outliers"]["geometries"]
        for dynamics in config["honest_dynamics"]["names"]
        for threat in config["threats"]["names"]
        for round_index in config["temporal"]["assessment_rounds"]
    }
    observed_history_ids = [str(row.get("history_id", "")) for row in history_rows]
    if (
        len(observed_history_ids) != len(expected_histories)
        or set(observed_history_ids) != expected_histories
    ):
        return False
    evaluation_children = range(
        int(config["nested_monte_carlo"]["evaluation_children"])
    )
    expected_child_keys = {
        (history_id, child, candidate)
        for history_id in expected_histories
        for child in evaluation_children
        for candidate in CANDIDATES
    }
    observed_child_keys = [
        (
            str(row.get("history_id", "")),
            int(row.get("evaluation_child", -1)),
            str(row.get("candidate", "")),
        )
        for row in child_rows
    ]
    return (
        len(observed_child_keys) == len(expected_child_keys)
        and set(observed_child_keys) == expected_child_keys
    )


def _replace_one_coverage_is_exact(
    config: Mapping[str, Any], replace_rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Check one balanced replacement of every identity in every audit context."""

    n = int(config["cohort"]["num_clients"])
    expected = {
        (
            int(seed),
            str(regime["name"]),
            str(permutation),
            client,
            client,
            "|".join(
                (
                    str(seed),
                    str(regime["name"]),
                    str(permutation),
                    "aligned",
                    "stationary",
                    "bitflip_x10",
                    "17",
                )
            ),
        )
        for seed in config["randomness"]["development_outer_seeds"]
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
        for client in range(n)
    }
    observed = [
        (
            int(row.get("seed", -1)),
            str(row.get("noise_regime", "")),
            str(row.get("noise_permutation", "")),
            int(row.get("trial", -1)),
            int(row.get("replaced_client", -1)),
            str(row.get("history_id", "")),
        )
        for row in replace_rows
    ]
    return len(observed) == len(expected) and set(observed) == expected


def _summarize(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
    replace_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_history: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    history_meta = {str(row["history_id"]): row for row in history_rows}
    for row in child_rows:
        by_history[str(row["history_id"])][str(row["candidate"])].append(
            float(row["squared_reference_error"])
        )
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in config["randomness"]["development_outer_seeds"]:
        ids = [
            key for key, row in history_meta.items() if int(row["seed"]) == int(seed)
        ]
        eligible_noise_cell_counts = {
            f"{regime_name}|{permutation}": sum(
                bool(history_meta[history_id]["eligible_R_positive"])
                and str(history_meta[history_id]["noise_regime"]) == regime_name
                and str(history_meta[history_id]["noise_permutation"]) == permutation
                for history_id in ids
            )
            for regime_name, permutation in _noise_cell_names(config)
        }
        eligible_noise_cell_composition_exact = all(
            count
            == int(config["gates"]["eligible_histories_per_seed_noise_cell_exact"])
            for count in eligible_noise_cell_counts.values()
        )
        point_headrooms: list[float] = []
        semi_oracle_headrooms: list[float] = []
        semi_oracle_minus_k4b: list[float] = []
        candidate_mse_sums: dict[str, float] = defaultdict(float)
        regime_candidate_mse_sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        split_disagreement_squared_sum = 0.0
        for history_id in ids:
            means = {
                candidate: _mean(by_history[history_id][candidate])
                for candidate in CANDIDATES
            }
            if not bool(history_meta[history_id]["eligible_R_positive"]):
                continue
            regime_name = str(history_meta[history_id]["noise_regime"])
            for candidate in CANDIDATES:
                candidate_mse_sums[candidate] += means[candidate]
                regime_candidate_mse_sums[regime_name][candidate] += means[candidate]
            point = means[K4] - means[POINTWISE]
            semi_oracle_headroom = means[K4] - means[SEMI_ORACLE]
            point_headrooms.append(point)
            semi_oracle_headrooms.append(semi_oracle_headroom)
            semi_oracle_minus_k4b.append(means[SEMI_ORACLE] - means[K4B])
            split_distance = float(
                history_meta[history_id]["split_a_b_aggregate_scale_distance"]
            )
            split_disagreement_squared_sum += split_distance * split_distance
        k4_mse_sum = candidate_mse_sums[K4]
        k4b_mse_sum = candidate_mse_sums[K4B]
        semi_oracle_mse_sum = candidate_mse_sums[SEMI_ORACLE]
        pointwise_mse_sum = candidate_mse_sums[POINTWISE]
        point_sum = k4_mse_sum - pointwise_mse_sum
        semi_oracle_sum = k4_mse_sum - semi_oracle_mse_sum
        regime_statistics: dict[str, float] = {}
        positive_finite_mse_denominators = [k4_mse_sum, k4b_mse_sum]
        positive_finite_pointwise_headroom_denominators = [point_sum]
        for regime_name in ("homogeneous", "heteroscedastic"):
            sums = regime_candidate_mse_sums[regime_name]
            regime_k4 = sums[K4]
            regime_pointwise_headroom = regime_k4 - sums[POINTWISE]
            regime_semi_headroom = regime_k4 - sums[SEMI_ORACLE]
            positive_finite_mse_denominators.append(regime_k4)
            positive_finite_pointwise_headroom_denominators.append(
                regime_pointwise_headroom
            )
            regime_statistics.update(
                {
                    f"{regime_name}_pointwise_relative_mse_headroom_vs_k4": (
                        regime_pointwise_headroom / regime_k4
                        if regime_k4 > 0.0
                        else float("nan")
                    ),
                    f"{regime_name}_semi_oracle_relative_mse_gain_vs_k4": (
                        regime_semi_headroom / regime_k4
                        if regime_k4 > 0.0
                        else float("nan")
                    ),
                    f"{regime_name}_capture_fraction_descriptive": (
                        regime_semi_headroom / regime_pointwise_headroom
                        if regime_pointwise_headroom > 0.0
                        else float("nan")
                    ),
                }
            )
        per_seed[int(seed)] = {
            "seed": int(seed),
            "eligible_histories": len(point_headrooms),
            "total_histories": len(ids),
            "eligible_histories_by_noise_cell": json.dumps(
                eligible_noise_cell_counts, sort_keys=True, separators=(",", ":")
            ),
            "eligible_noise_cell_composition_exact": (
                eligible_noise_cell_composition_exact
            ),
            "capture_fraction": (
                semi_oracle_sum / point_sum if point_sum > 0.0 else float("nan")
            ),
            "pointwise_relative_mse_headroom_vs_k4": (
                point_sum / k4_mse_sum if k4_mse_sum > 0.0 else float("nan")
            ),
            "semi_oracle_relative_mse_gain_vs_k4": (
                semi_oracle_sum / k4_mse_sum if k4_mse_sum > 0.0 else float("nan")
            ),
            "semi_oracle_headroom_mse": _mean(semi_oracle_headrooms),
            "pointwise_headroom_mse": _mean(point_headrooms),
            "semi_oracle_minus_k4b_mse": _mean(semi_oracle_minus_k4b),
            "k4_two_snapshot_integrated_mse": k4_mse_sum,
            "k4b_two_snapshot_integrated_mse": k4b_mse_sum,
            "semi_oracle_two_snapshot_integrated_mse": semi_oracle_mse_sum,
            "pointwise_two_snapshot_integrated_mse": pointwise_mse_sum,
            "semi_oracle_relative_mse_gain_vs_k4b": (
                (k4b_mse_sum - semi_oracle_mse_sum) / k4b_mse_sum
                if k4b_mse_sum > 0.0
                else float("nan")
            ),
            "positive_finite_mse_denominators": all(
                math.isfinite(value) and value > 0.0
                for value in positive_finite_mse_denominators
            ),
            "positive_finite_pointwise_headroom_denominators": all(
                math.isfinite(value) and value > 0.0
                for value in positive_finite_pointwise_headroom_denominators
            ),
            "construction_split_disagreement_mse_ratio": (
                split_disagreement_squared_sum / k4_mse_sum
                if k4_mse_sum > 0.0
                else float("nan")
            ),
            **regime_statistics,
        }
    seed_rows = [
        per_seed[int(seed)] for seed in config["randomness"]["development_outer_seeds"]
    ]
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    capture_ci = _ci([float(row["capture_fraction"]) for row in seed_rows], tcrit)
    pointwise_relative_k4_ci = _ci(
        [float(row["pointwise_relative_mse_headroom_vs_k4"]) for row in seed_rows],
        tcrit,
    )
    semi_oracle_relative_k4_ci = _ci(
        [float(row["semi_oracle_relative_mse_gain_vs_k4"]) for row in seed_rows],
        tcrit,
    )
    headroom_ci = _ci(
        [float(row["semi_oracle_headroom_mse"]) for row in seed_rows], tcrit
    )
    difference_ci = _ci(
        [float(row["semi_oracle_minus_k4b_mse"]) for row in seed_rows], tcrit
    )
    relative_gain_ci = _ci(
        [float(row["semi_oracle_relative_mse_gain_vs_k4b"]) for row in seed_rows],
        tcrit,
    )
    homogeneous_relative_k4_ci = _ci(
        [
            float(row["homogeneous_semi_oracle_relative_mse_gain_vs_k4"])
            for row in seed_rows
        ],
        tcrit,
    )
    heteroscedastic_relative_k4_ci = _ci(
        [
            float(row["heteroscedastic_semi_oracle_relative_mse_gain_vs_k4"])
            for row in seed_rows
        ],
        tcrit,
    )
    homogeneous_capture_ci_descriptive = _ci(
        [float(row["homogeneous_capture_fraction_descriptive"]) for row in seed_rows],
        tcrit,
    )
    heteroscedastic_capture_ci_descriptive = _ci(
        [
            float(row["heteroscedastic_capture_fraction_descriptive"])
            for row in seed_rows
        ],
        tcrit,
    )
    expected_histories = int(config["statistical_analysis"]["frozen_histories_total"])
    expected_child_rows = (
        expected_histories
        * int(config["nested_monte_carlo"]["evaluation_children"])
        * len(CANDIDATES)
    )
    declared_per_history_overlap = sum(
        int(row["construction_evaluation_seed_overlap"]) for row in history_rows
    )
    construction_seed_registry = [
        seed
        for row in history_rows
        for seed in _parse_child_seed_registry(row["construction_child_seed_registry"])
    ]
    evaluation_seed_registry = [
        seed
        for row in history_rows
        for seed in _parse_child_seed_registry(row["evaluation_child_seed_registry"])
    ]
    construction_seed_duplicates = len(construction_seed_registry) - len(
        set(construction_seed_registry)
    )
    evaluation_seed_duplicates = len(evaluation_seed_registry) - len(
        set(evaluation_seed_registry)
    )
    global_seed_overlap = len(
        set(construction_seed_registry) & set(evaluation_seed_registry)
    )
    cap_violations = sum(
        not bool(row["contribution_cap_respected"])
        for row in child_rows
        if row["candidate"] == SEMI_ORACLE
    )
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    eligible_fraction = _mean(
        1.0 if bool(row["eligible_R_positive"]) else 0.0 for row in history_rows
    )
    predictor_variation = max(
        (
            len(
                {
                    str(row["semi_oracle_predictor_hash"])
                    for row in child_rows
                    if str(row["history_id"]) == history_id
                    and str(row["candidate"]) == SEMI_ORACLE
                }
            )
            - 1
            for history_id in history_meta
        ),
        default=0,
    )
    gate_sum_violations = sum(
        bool(row["normalization_by_gate_sum"]) for row in child_rows
    )
    denominator_violations = sum(
        int(row["fixed_denominator_n"]) != int(config["cohort"]["num_clients"])
        or float(row["fixed_denominator_formula_error"])
        > float(config["gates"]["fixed_denominator_formula_abs_error_max"])
        for row in child_rows
    )
    k4_manual_formula_max = max(
        (float(row["k4_manual_formula_error"]) for row in child_rows),
        default=float("nan"),
    )
    frozen_k4_comparator_max = max(
        (float(row["frozen_k4_comparator_error_descriptive"]) for row in child_rows),
        default=float("nan"),
    )
    k4b_reproduction_max = max(
        (float(row["k4b_comparator_reproduction_error"]) for row in child_rows),
        default=float("nan"),
    )
    split_ratios = [
        float(row["construction_split_disagreement_mse_ratio"]) for row in seed_rows
    ]
    cap_saturation_rate = _mean(
        1.0 if bool(row["semi_oracle_projection_active"]) else 0.0
        for row in history_rows
    )
    pointwise_dominance_excess_mse_max = max(
        (
            float(row["pointwise_excess_mse_over_candidate"])
            for row in child_rows
            if str(row["candidate"]) in {K4, K4B, SEMI_ORACLE}
        ),
        default=float("nan"),
    )
    pooled_restricted_k4b = sum(
        float(row["k4b_two_snapshot_integrated_mse"]) for row in seed_rows
    )
    pooled_restricted_semi_oracle = sum(
        float(row["semi_oracle_two_snapshot_integrated_mse"]) for row in seed_rows
    )
    positive_finite_mse_denominator_count = sum(
        bool(row["positive_finite_mse_denominators"]) for row in seed_rows
    )
    positive_finite_pointwise_headroom_denominator_count = sum(
        bool(row["positive_finite_pointwise_headroom_denominators"])
        for row in seed_rows
    )
    eligible_noise_cell_composition_count = sum(
        bool(row["eligible_noise_cell_composition_exact"]) for row in seed_rows
    )
    observed = {
        "frozen_histories": len(history_rows),
        "evaluation_child_rows": len(child_rows),
        "expected_frozen_histories": expected_histories,
        "expected_evaluation_child_rows": expected_child_rows,
        "complete_fraction": min(
            len(history_rows) / expected_histories,
            len(child_rows) / expected_child_rows,
        ),
        "matrix_exact": _matrix_is_exact(config, history_rows, child_rows),
        "finite_metric_fraction": _mean(
            1.0 if bool(row["all_finite"]) else 0.0 for row in child_rows
        ),
        "construction_evaluation_seed_overlap": global_seed_overlap,
        "declared_per_history_construction_evaluation_seed_overlap": (
            declared_per_history_overlap
        ),
        "construction_child_seed_count": len(construction_seed_registry),
        "evaluation_child_seed_count": len(evaluation_seed_registry),
        "construction_child_seed_duplicates": construction_seed_duplicates,
        "evaluation_child_seed_duplicates": evaluation_seed_duplicates,
        "eligible_R_positive_history_fraction": eligible_fraction,
        "eligible_noise_cell_composition_seed_count": (
            eligible_noise_cell_composition_count
        ),
        "semi_oracle_capture_fraction_seed_ci95": capture_ci,
        "pointwise_relative_mse_headroom_vs_k4_seed_ci95": (pointwise_relative_k4_ci),
        "semi_oracle_relative_mse_gain_vs_k4_seed_ci95": (semi_oracle_relative_k4_ci),
        "semi_oracle_headroom_mse_seed_ci95": headroom_ci,
        "semi_oracle_minus_k4b_mse_seed_ci95": difference_ci,
        "semi_oracle_relative_mse_gain_vs_k4b_seed_ci95": relative_gain_ci,
        "semi_oracle_relative_mse_gain_vs_k4b_pooled_descriptive": (
            (pooled_restricted_k4b - pooled_restricted_semi_oracle)
            / pooled_restricted_k4b
            if pooled_restricted_k4b > 0.0
            else float("nan")
        ),
        "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": (
            homogeneous_relative_k4_ci
        ),
        "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95": (
            heteroscedastic_relative_k4_ci
        ),
        "homogeneous_capture_fraction_seed_ci95_descriptive": (
            homogeneous_capture_ci_descriptive
        ),
        "heteroscedastic_capture_fraction_seed_ci95_descriptive": (
            heteroscedastic_capture_ci_descriptive
        ),
        "positive_finite_mse_denominator_count": (
            positive_finite_mse_denominator_count
        ),
        "positive_finite_pointwise_headroom_denominator_count": (
            positive_finite_pointwise_headroom_denominator_count
        ),
        "construction_split_disagreement_mse_ratio_mean": _mean(split_ratios),
        "construction_split_disagreement_mse_ratio_max": max(
            split_ratios, default=float("nan")
        ),
        "semi_oracle_projection_cap_saturation_rate": cap_saturation_rate,
        "predictor_variation_across_evaluation_children": predictor_variation,
        "k4_manual_formula_abs_error_max": k4_manual_formula_max,
        "frozen_k4_comparator_abs_error_max_descriptive": (frozen_k4_comparator_max),
        "k4b_comparator_reproduction_abs_error_max": k4b_reproduction_max,
        "gate_sum_normalization_violations": gate_sum_violations,
        "fixed_denominator_violations": denominator_violations,
        "pointwise_oracle_dominance_excess_mse_max": (
            pointwise_dominance_excess_mse_max
        ),
        "contribution_cap_violations": cap_violations,
        "replace_one_trials": len(replace_rows),
        "replace_one_exact_coverage": _replace_one_coverage_is_exact(
            config, replace_rows
        ),
        "replace_one_violations": replace_violations,
        "replace_one_max_ratio_to_bound": max(
            (float(row["ratio_to_bound"]) for row in replace_rows), default=float("nan")
        ),
        "pointwise_oracle_used_in_construction": sum(
            bool(row["pointwise_oracle_used_in_construction"]) for row in child_rows
        ),
        "ordinary_l2_error_used_for_gate": False,
        "device": str(oracle._RUNTIME_DEVICE),
    }
    gates = config["gates"]
    exact_seed_count = int(gates["exact_seed_capture_count"])
    validity_checks = {
        "complete": observed["complete_fraction"]
        >= float(gates["complete_fraction_min"]),
        "matrix_exact": bool(observed["matrix_exact"]),
        "finite": observed["finite_metric_fraction"]
        >= float(gates["finite_metric_fraction_min"]),
        "production_device": observed["device"]
        == str(gates["production_device_required"]),
        "construction_evaluation_independent": (
            global_seed_overlap
            <= int(gates["construction_evaluation_seed_overlap_max"])
            and declared_per_history_overlap
            <= int(gates["construction_evaluation_seed_overlap_max"])
        ),
        "rng_global_unique": (
            len(construction_seed_registry)
            == int(gates["construction_child_seed_exact_count"])
            and len(evaluation_seed_registry)
            == int(gates["evaluation_child_seed_exact_count"])
            and construction_seed_duplicates
            <= int(gates["global_child_seed_duplicate_max"])
            and evaluation_seed_duplicates
            <= int(gates["global_child_seed_duplicate_max"])
        ),
        "predictor_fixed_across_evaluation_children": predictor_variation
        <= float(gates["predictor_variation_across_evaluation_children_max"]),
        "pointwise_not_used_in_construction": observed[
            "pointwise_oracle_used_in_construction"
        ]
        <= int(gates["pointwise_oracle_used_in_construction_max"]),
        "exact_outer_seed_count": len(seed_rows) == exact_seed_count,
        "positive_finite_mse_denominators": (
            not bool(gates["positive_finite_mse_denominators_required"])
            or positive_finite_mse_denominator_count == exact_seed_count
        ),
        "construction_mc_stability": observed[
            "construction_split_disagreement_mse_ratio_max"
        ]
        <= float(gates["construction_split_disagreement_mse_ratio_max"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "replace_one_complete": (
            len(replace_rows) == int(gates["replace_one_exact_trials"])
            and bool(observed["replace_one_exact_coverage"])
        ),
        "k4_manual_formula": k4_manual_formula_max
        <= float(gates["k4_manual_formula_abs_error_max"]),
        "k4b_reproduction": k4b_reproduction_max
        <= float(gates["k4b_comparator_reproduction_abs_error_max"]),
        "no_gate_sum_normalization": gate_sum_violations
        <= int(gates["gate_sum_normalization_violation_max"]),
        "fixed_denominator": denominator_violations
        <= int(gates["fixed_denominator_violation_max"]),
        "pointwise_oracle_dominance": pointwise_dominance_excess_mse_max
        <= float(gates["pointwise_oracle_dominance_excess_mse_max"]),
    }
    scientific_checks = {
        "eligible_R": eligible_fraction
        >= float(gates["eligible_R_positive_history_fraction_min"]),
        "eligible_noise_cell_composition": (
            eligible_noise_cell_composition_count == exact_seed_count
        ),
        "positive_finite_pointwise_headroom_denominators": (
            not bool(gates["positive_finite_pointwise_headroom_denominators_required"])
            or positive_finite_pointwise_headroom_denominator_count == exact_seed_count
        ),
        "exact_scientific_seed_count": all(
            int(interval["n"]) == exact_seed_count
            for interval in (
                capture_ci,
                pointwise_relative_k4_ci,
                semi_oracle_relative_k4_ci,
                difference_ci,
                relative_gain_ci,
                homogeneous_relative_k4_ci,
                heteroscedastic_relative_k4_ci,
            )
        ),
        "material_pointwise_headroom_mean": float(pointwise_relative_k4_ci["mean"])
        >= float(gates["pointwise_relative_mse_headroom_vs_k4_mean_min"]),
        "material_pointwise_headroom_ci": float(pointwise_relative_k4_ci["low"])
        > float(
            gates[
                "pointwise_relative_mse_headroom_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "material_semi_oracle_gain_vs_k4_mean": float(
            semi_oracle_relative_k4_ci["mean"]
        )
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4_mean_min"]),
        "material_semi_oracle_gain_vs_k4_ci": float(semi_oracle_relative_k4_ci["low"])
        > float(
            gates[
                "semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "capture_mean": float(capture_ci["mean"])
        >= float(gates["semi_oracle_capture_fraction_mean_min"]),
        "capture_ci_low": float(capture_ci["low"])
        > float(gates["semi_oracle_capture_fraction_ci95_low_strictly_greater_than"]),
        "gain_vs_k4b": float(relative_gain_ci["mean"])
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4b_min"]),
        "gain_vs_k4b_ci": float(relative_gain_ci["low"])
        >= float(gates["semi_oracle_relative_mse_gain_vs_k4b_seed_ci95_low_min"]),
        "ci_vs_k4b": float(difference_ci["high"])
        <= float(gates["semi_oracle_minus_k4b_mse_seed_ci95_high_max"]),
        "homogeneous_gain_vs_k4_ci": float(homogeneous_relative_k4_ci["low"])
        > float(
            gates[
                "homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
        "heteroscedastic_gain_vs_k4_ci": float(heteroscedastic_relative_k4_ci["low"])
        > float(
            gates[
                "heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95_low_strictly_greater_than"
            ]
        ),
    }
    classification = config["statistical_analysis"]["gate_classification"]
    if tuple(classification["validity"]) != tuple(validity_checks):
        raise RuntimeError("Configured validity-gate classification changed")
    if tuple(classification["scientific"]) != tuple(scientific_checks):
        raise RuntimeError("Configured scientific-gate classification changed")
    validity_pass = all(validity_checks.values())
    scientific_pass = all(scientific_checks.values())
    status = _decision_status(validity_checks, scientific_checks)
    checks = {**validity_checks, **scientific_checks}
    decision = {
        "decision": status,
        "all_gates_pass": validity_pass and scientific_pass,
        "checks": checks,
        "validity_checks": validity_checks,
        "scientific_checks": scientific_checks,
        "validity_pass": validity_pass,
        "scientific_checks_pass": scientific_pass if validity_pass else None,
        "observed": observed,
        "pointwise_oracle_role": (
            "nondeployable_decision_benchmark_headroom_denominator_only"
        ),
        "semi_oracle_role": (
            "dgp_specific_finite_monte_carlo_mechanism_conditioned_on_observable_"
            "past_latent_clean_current_state_simulator_labels_and_configured_"
            "current_noise_attack_law"
        ),
        "pass_does_not_authorize_promotion_or_holdout": True,
        "failure_interpretation": (
            "no_scientific_inference_when_invalid_or_inconclusive_otherwise_"
            "the_preregistered_finite_mc_privileged_mechanism_fails_required_"
            "gates_for_this_dgp_and_stops_this_fixed_denominator_branch"
        ),
        "success_interpretation": (
            "headroom_exists_but_a_separate_observable_past_only_predictor_"
            "must_still_be_constructed_and_tested"
        ),
        "universal_impossibility_claimed": False,
        "exact_conditional_optimum_failure_claimed": False,
        "arbitrary_adaptive_attack_claimed": False,
        "loss_used_for_all_scientific_gates": "squared_l2_reference_error",
        "holdout_opened": False,
    }
    return seed_rows, decision


def _report(decision: Mapping[str, Any], output: Path) -> None:
    observed = decision["observed"]
    failed = [name for name, value in decision["checks"].items() if not value]
    if decision["decision"] == "invalid_or_inconclusive_screen":
        interpretation = (
            "A validity check failed: this screen is invalid/inconclusive and "
            "supports no scientific branch inference."
        )
    elif decision["decision"] == "stop_missing_slot_imputation_branch":
        interpretation = (
            "All validity checks passed, but the preregistered finite-MC "
            "privileged mechanism missed at least one scientific gate for this "
            "DGP; stop this fixed-denominator branch."
        )
    else:
        interpretation = (
            "All checks passed; this only authorizes a later observable-past-only "
            "predictor study."
        )
    lines = [
        "# G0g-K4c-CH — current-randomness conditional compensatory headroom",
        "",
        f"- Decision: **{decision['decision']}**.",
        f"- Failed gates: {', '.join(failed) if failed else 'none'}.",
        f"- Interpretation: {interpretation}",
        "- Pointwise oracle: nondeployable decision benchmark and headroom denominator; never used in construction or promotion.",
        "- K4c-CH semi-oracle conditions on the latent clean current state; it is not transcript-only.",
        "- It also uses simulator labels and the configured current attack/noise generator.",
        "- Headroom is compensatory total current-reference error; DP, attack, gate/clipping and anchor components are not separately identified.",
        "- Two-snapshot integrated ordinary L2 error: descriptive only; every scientific gate uses squared error.",
        "- PASS only authorizes a later transcript-only predictor study; never promotion or holdout.",
        "- This finite-MC screen never establishes failure of the exact conditional optimum or a universal impossibility result.",
        "- Holdout opened: no.",
        "",
        "| Quantity | Observed |",
        "|---|---:|",
        f"| Pointwise relative MSE headroom vs K4 (seed mean) | {_format_metric(observed['pointwise_relative_mse_headroom_vs_k4_seed_ci95']['mean'])} |",
        f"| Pointwise relative MSE headroom vs K4 (CI95 low) | {_format_metric(observed['pointwise_relative_mse_headroom_vs_k4_seed_ci95']['low'])} |",
        f"| Semi-oracle relative MSE gain vs K4 (seed mean) | {_format_metric(observed['semi_oracle_relative_mse_gain_vs_k4_seed_ci95']['mean'])} |",
        f"| Semi-oracle relative MSE gain vs K4 (CI95 low) | {_format_metric(observed['semi_oracle_relative_mse_gain_vs_k4_seed_ci95']['low'])} |",
        f"| Semi-oracle capture mean | {_format_metric(observed['semi_oracle_capture_fraction_seed_ci95']['mean'])} |",
        f"| Semi-oracle capture seed SD | {_format_metric(observed['semi_oracle_capture_fraction_seed_ci95']['sd'])} |",
        f"| Semi-oracle capture CI95 low | {_format_metric(observed['semi_oracle_capture_fraction_seed_ci95']['low'])} |",
        f"| Homogeneous relative MSE gain vs K4 (CI95 low) | {_format_metric(observed['homogeneous_semi_oracle_relative_mse_gain_vs_k4_seed_ci95']['low'])} |",
        f"| Heteroscedastic relative MSE gain vs K4 (CI95 low) | {_format_metric(observed['heteroscedastic_semi_oracle_relative_mse_gain_vs_k4_seed_ci95']['low'])} |",
        f"| Homogeneous capture, descriptive (seed mean) | {_format_metric(observed['homogeneous_capture_fraction_seed_ci95_descriptive']['mean'])} |",
        f"| Heteroscedastic capture, descriptive (seed mean) | {_format_metric(observed['heteroscedastic_capture_fraction_seed_ci95_descriptive']['mean'])} |",
        f"| Relative MSE gain vs K4b (seed mean) | {_format_metric(observed['semi_oracle_relative_mse_gain_vs_k4b_seed_ci95']['mean'])} |",
        f"| Relative MSE gain vs K4b (CI95 low) | {_format_metric(observed['semi_oracle_relative_mse_gain_vs_k4b_seed_ci95']['low'])} |",
        f"| Max split-half MC MSE ratio | {_format_metric(observed['construction_split_disagreement_mse_ratio_max'])} |",
        f"| Predictor cap saturation rate | {_format_metric(observed['semi_oracle_projection_cap_saturation_rate'])} |",
        f"| Eligible histories | {_format_metric(observed['eligible_R_positive_history_fraction'])} |",
        f"| Replace-one violations | {observed['replace_one_violations']} |",
        f"| Maximum replace-one ratio | {_format_metric(observed['replace_one_max_ratio_to_bound'])} |",
        "",
        f"Raw results: `{output.relative_to(ROOT)}`.",
    ]
    _atomic_text(DEFAULT_REPORT, "\n".join(lines) + "\n")


def run(
    config_path: Path,
    lock_path: Path,
    output: Path,
    *,
    published_lock_sha256: str,
) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("Production accepts only the preregistered output path")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    lock = _verify_lock(lock_path, config_path)
    publication = _attest_external_lock_publication(lock, published_lock_sha256)
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError("K4c-CH refuses CPU or non-float32 production")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing K4c-CH result directory: {output}"
        )
    output.mkdir(parents=True)
    k2_calibration, temporal_calibration, provenance = _load_calibrations(config)
    manifest = {
        "campaign_id": config["campaign_id"],
        "status": "running_development",
        "device": "mps",
        "dtype": str(oracle._RUNTIME_DTYPE),
        "development_only": True,
        "holdout_opened": False,
        "conditional_loss": "squared_l2_reference_error",
        "ordinary_l2_error_role": "descriptive_only",
        "pointwise_oracle_role": (
            "nondeployable_decision_benchmark_headroom_denominator_only"
        ),
        "semi_oracle_conditioning": (
            "G_t_minus_observable_past_plus_latent_clean_current_state_"
            "simulator_labels_and_configured_current_dgp"
        ),
        "semi_oracle_is_finite_monte_carlo_approximation": True,
        "compensatory_headroom_components_separately_identified": False,
        "observable_past_only_predictor_constructed": False,
        "pass_authorizes_holdout_or_promotion": False,
        "preregistration_lock": lock,
        "lock_publication_attestation": publication,
        "config_sha256": _sha256(config_path),
        "calibration_provenance": provenance,
    }
    _write_json(output / "manifest.json", manifest)
    history_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for cell in _outer_cells(config):
        histories, children, audit_contexts = _evaluate_outer_trajectory(
            config, k2_calibration, temporal_calibration, cell
        )
        history_rows.extend(histories)
        child_rows.extend(children)
        contexts.extend(audit_contexts)
    replace_rows = _replace_one_rows(config, temporal_calibration, contexts)
    seed_rows, decision = _summarize(config, history_rows, child_rows, replace_rows)
    _write_csv(output / "frozen_history_rows.csv", history_rows)
    _write_csv(output / "evaluation_child_rows.csv", child_rows)
    _write_csv(output / "seed_summary.csv", seed_rows)
    _write_csv(output / "replace_one_audit.csv", replace_rows)
    _write_json(output / "decision.json", decision)
    _write_json(output / "frozen_calibration_provenance.json", provenance)
    manifest.update(
        {
            "status": "completed_development",
            "all_gates_pass": decision["all_gates_pass"],
            "development_decision": decision["decision"],
            "frozen_histories": len(history_rows),
            "evaluation_child_rows": len(child_rows),
            "holdout_opened": False,
        }
    )
    _write_json(output / "manifest.json", manifest)
    _report(decision, output)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the locked screen")
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--published-lock-sha256",
        help=(
            "exact lock SHA-256 already published in the research log/chat; "
            "required with --run"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not args.run:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _validate_config(config)
        print(
            json.dumps(
                {
                    "campaign_id": config["campaign_id"],
                    "device_required": "mps",
                    "frozen_histories": config["statistical_analysis"][
                        "frozen_histories_total"
                    ],
                    "child_draws": config["statistical_analysis"]["child_draws_total"],
                    "scientific_run_started": False,
                },
                indent=2,
            )
        )
        return 0
    if args.published_lock_sha256 is None:
        parser.error(
            "--run requires --published-lock-sha256 after external publication"
        )
    decision = run(
        config_path,
        lock_path,
        output,
        published_lock_sha256=args.published_lock_sha256,
    )
    print(json.dumps(_json_safe(decision), indent=2, sort_keys=True, allow_nan=False))
    return 0 if decision["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
