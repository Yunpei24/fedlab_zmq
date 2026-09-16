#!/usr/bin/env python3
"""G0f: decisive covariance-budgeted robust-reference experiment.

G0f keeps the G0e FCC pilot, offline cross-fitted calibration, finite public
Huber solver, bounded blend and replace-one certificate.  Its sole scientific
change is how the total client influence budget ``G`` is allocated across
blocks.  The main candidate uses authenticated public radii ``a[i,b]`` and

``G[i,b] = G * a[i,b] / max_j ||a[j]||_2``.

The campaign includes a per-client-normalized (scale-blind) ablation, an equal
cap ablation, the original G0e correction, FCC, RFA, trimmed mean and the
uniform mean.  Development is a hard gate: holdout artifacts are never read or
generated unless every pre-registered development criterion passes.

Production execution is MPS-only and float32.  CPU is used only by unit tests
for pure deterministic helpers.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
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

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_budget_allocated_correction,
    gaussian_aware_crossfit_bounded_correction,
)
from robustness.aggregators import clip_l2, coordinate_median  # noqa: E402
from scripts import run_gaussian_aware_reference_g0e as g0e  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

BASELINES = ("uniform_mean", "fcc", "rfa", "trimmed_mean", "coordinate_median")
CORRECTIONS = (
    "g0e",
    "g0f_equal_cap",
    "g0f_global_covariance",
    "g0f_per_client_scale_blind",
)
CANDIDATES = (*BASELINES, *CORRECTIONS)
PRIMARY_CANDIDATE = "g0f_global_covariance"
POLICY_BY_CANDIDATE = {
    "g0f_equal_cap": "equal_cap",
    "g0f_global_covariance": "global_covariance",
    "g0f_per_client_scale_blind": "per_client_scale_blind",
}


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _critical_t95(n: int) -> float:
    return {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(n, 1.96)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Rows for {path} do not share one schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_sha256(payload: Any) -> str:
    """Hash a JSON artifact independently of whitespace and key order."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _software_sha256() -> dict[str, str]:
    """Fingerprint every source that can change a resumed G0f checkpoint."""

    paths = (
        Path(__file__).resolve(),
        ROOT / "algorithms/gaussian_aware_reference.py",
        ROOT / "scripts/run_gaussian_aware_reference_g0e.py",
        ROOT / "scripts/run_gaussian_aware_reference_oracle.py",
        ROOT / "robustness/aggregators.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing G0f source dependency: {path}")
        result[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor value, dtype and shape after a canonical CPU transfer."""

    canonical = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + canonical.numpy().tobytes()).hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Return dependency/runtime versions relevant to deterministic resume."""

    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "pyyaml": str(yaml.__version__),
        "platform": platform.platform(),
    }


def _assert_manifest_matches(
    manifest_path: Path, expected: Mapping[str, Any], *, stage: str
) -> None:
    """Reject any source, dependency, config or calibration drift."""

    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed != dict(expected):
        changed = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise RuntimeError(f"G0f manifest mismatch {stage}; changed fields: {changed}")


def _holdout_artifacts(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.rglob("*")
        if any("holdout" in part.lower() for part in path.relative_to(output_dir).parts)
    )


def _g0e_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the exact G0e calibration view without mutating G0f config."""

    view = copy.deepcopy(dict(config))
    references = view["references"]
    references["g0e_public_budgets"] = copy.deepcopy(references["g0f_public_budgets"])
    return view


def _validate_config(config: dict[str, Any]) -> None:
    expected_top_level = {
        "campaign_id",
        "scope",
        "excluded_prior_seeds",
        "scientific_contract",
        "cohort",
        "privacy_noise",
        "references",
        "aggregation",
        "threats",
        "randomness",
        "holdout_registry",
        "gates",
        "selection",
        "execution",
    }
    if set(config) != expected_top_level:
        raise ValueError("The frozen G0f top-level protocol schema changed")
    if config["campaign_id"] != "gaussian_aware_reference_g0f_mps_v1":
        raise ValueError("Unexpected G0f campaign id")
    if config["scope"] != (
        "preregistered_reference_only_global_covariance_budget_allocation"
    ):
        raise ValueError("Unexpected G0f campaign scope")
    excluded = [int(value) for value in config["excluded_prior_seeds"]]
    if not excluded or len(excluded) != len(set(excluded)):
        raise ValueError("excluded_prior_seeds must be a non-empty unique registry")

    contract = config["scientific_contract"]
    expected_contract = {
        "estimand": "equal_client_mean_of_clean_honest_updates",
        "reference_only": True,
        "far_weights_used": False,
        "accuracy_used": False,
        "inverse_variance_estimand_forbidden": True,
        "base_reference": "fcc",
        "correction_is_bounded_around_fcc": True,
        "calibration_chain": "server_clip_then_leave_one_out_fcc",
        "calibration_is_offline_and_cross_fitted": True,
        "crossfit_references_are_not_online_solver_inputs": True,
        "allocation_radii": "deployed_public_radii_before_cap",
        "allocation_radius_semantics": "effective_null_radius",
        "allocation_radius_components": (
            "dp_variance_plus_heterogeneity_plus_reference_variance_plus_floor"
        ),
        "allocation_radii_are_public_authenticated": True,
        "client_declared_covariance_forbidden": True,
        "parameters_are_derived_from_public_budgets": True,
        "holdout_used_for_selection": False,
        "holdout_blocked_until_development_passes": True,
    }
    if contract != expected_contract:
        raise ValueError("The frozen G0f scientific contract changed")
    required_true = (
        "reference_only",
        "inverse_variance_estimand_forbidden",
        "correction_is_bounded_around_fcc",
        "calibration_is_offline_and_cross_fitted",
        "crossfit_references_are_not_online_solver_inputs",
        "allocation_radii_are_public_authenticated",
        "client_declared_covariance_forbidden",
        "holdout_blocked_until_development_passes",
    )
    if any(not bool(contract.get(key, False)) for key in required_true):
        raise ValueError("Every G0f scientific-contract safeguard must be true")
    if contract["estimand"] != "equal_client_mean_of_clean_honest_updates":
        raise ValueError("G0f must retain the equal-client estimand")
    if contract["base_reference"] != "fcc":
        raise ValueError("G0f must retain the certified FCC pilot")
    if contract["allocation_radii"] != "deployed_public_radii_before_cap":
        raise ValueError("G0f allocation must use deployed public pre-cap radii")
    if contract.get("far_weights_used") or contract.get("accuracy_used"):
        raise ValueError("G0f is reference-only")
    if contract.get("holdout_used_for_selection"):
        raise ValueError("Holdout selection is forbidden")

    execution = config["execution"]
    if set(execution) != {"required_device", "tensor_dtype", "allow_cpu_fallback"}:
        raise ValueError("The frozen execution schema changed")
    if execution["required_device"] != "mps" or bool(execution["allow_cpu_fallback"]):
        raise ValueError("G0f production execution is MPS-only")
    if execution["tensor_dtype"] != "float32":
        raise ValueError("G0f production execution requires float32")
    if config["selection"]["candidate_grid"] != "forbidden":
        raise ValueError("G0f is one frozen candidate, not a tuning grid")
    if config["selection"]["primary_candidate"] != PRIMARY_CANDIDATE:
        raise ValueError("The frozen primary candidate must be global covariance")
    if set(config["selection"]["reported_candidates"]) != set(CANDIDATES):
        raise ValueError("Exactly nine pre-registered candidates are required")

    randomness = config["randomness"]
    if not bool(randomness["pair_noise_across_regimes"]):
        raise ValueError("G0f requires paired standard noise across regimes")
    if not bool(randomness["pair_noise_across_tier_permutations"]):
        raise ValueError("G0f requires paired noise across tier permutations")
    folds = [[int(value) for value in fold] for fold in randomness["calibration_folds"]]
    if len(folds) != 2 or any(len(fold) < 2 for fold in folds):
        raise ValueError("G0f requires two calibration folds of at least two seeds")
    groups = (
        [value for fold in folds for value in fold],
        [int(value) for value in randomness["development_seeds"]],
    )
    all_seeds = [value for group in groups for value in group]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Calibration, development, and holdout seeds must be disjoint")
    if set(all_seeds) & set(int(value) for value in config["excluded_prior_seeds"]):
        raise ValueError("G0f reuses a previously inspected seed")
    if len(groups[1]) != 5:
        raise ValueError("G0f requires exactly five development seeds")
    if "holdout_seeds" in randomness:
        raise ValueError("Holdout seeds must not appear in the primary G0f config")
    registry = config["holdout_registry"]
    if set(registry) != {"path", "sha256", "expected_seed_count"}:
        raise ValueError("The execution-gated holdout registry contract is malformed")
    if str(registry["path"]) != (
        "configs/ldp_gradient_far/gaussian_aware_reference_g0f_holdout.yaml"
    ):
        raise ValueError("Unexpected G0f holdout registry path")
    digest = str(registry["sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Holdout registry SHA-256 must be a lowercase hex digest")
    if digest != "5c6f65adb249f095519dbb06c0a3e5ed834e77f2fb46cb6d33bbe8e34d9fe6a0":
        raise ValueError("The frozen holdout registry commitment changed")
    if int(registry["expected_seed_count"]) != 7:
        raise ValueError("G0f requires seven execution-gated holdout seeds")

    cohort = config["cohort"]
    if set(cohort) != {
        "num_clients",
        "num_byzantine",
        "dimension",
        "block_sizes",
        "honest_mean_norm",
        "heterogeneity_std_by_block",
        "honest_outliers",
    }:
        raise ValueError("The frozen cohort schema changed")
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    if (n, b) != (25, 5):
        raise ValueError("The decisive G0f protocol is frozen at n=25, b=5")
    if sum(int(value) for value in cohort["block_sizes"]) != int(cohort["dimension"]):
        raise ValueError("block_sizes must sum to dimension")
    if int(config["references"]["trimmed_mean"]["trim_count"]) != b:
        raise ValueError("trimmed_mean trim_count must equal the Byzantine count")
    if list(cohort["block_sizes"]) != [16, 16, 16, 16]:
        raise ValueError("The frozen G0f block partition changed")
    if float(cohort["honest_mean_norm"]) != 0.08 or list(
        cohort["heterogeneity_std_by_block"]
    ) != [0.012, 0.018, 0.025, 0.035]:
        raise ValueError("The frozen honest cohort geometry changed")
    if cohort["honest_outliers"] != {
        "count": 5,
        "shift_norm": 0.10,
        "geometries": ["aligned", "orthogonal"],
    }:
        raise ValueError("The frozen honest-outlier protocol changed")
    if config["aggregation"] != {
        "server_clip_norm": 0.42,
        "clip_before_reference": True,
    }:
        raise ValueError("The frozen server-clipping protocol changed")
    if set(config["threats"]["names"]) != {
        "none",
        "alie",
        "ipm",
        "bitflip_x10",
        "model_replacement",
    }:
        raise ValueError("All five frozen threat cells are required")
    threats = config["threats"]
    if set(threats) != {
        "names",
        "separated_for_gates",
        "evasive_controls",
        "alie_z",
        "ipm_scale",
        "bitflip_scale",
        "model_replacement_scale",
        "development_severities",
        "holdout_severities",
    }:
        raise ValueError("The frozen G0f threat schema changed")
    if threats["separated_for_gates"] != [
        "ipm",
        "bitflip_x10",
        "model_replacement",
    ] or threats["evasive_controls"] != ["alie"]:
        raise ValueError("The frozen separated/evasive threat roles changed")
    if (
        float(threats["alie_z"]) != 1.5
        or float(threats["ipm_scale"]) != 3.0
        or float(threats["bitflip_scale"]) != 10.0
        or float(threats["model_replacement_scale"]) != 5.0
        or list(threats["development_severities"]) != [0.75, 1.0, 1.25]
        or list(threats["holdout_severities"]) != [0.60, 0.90, 1.10, 1.40]
    ):
        raise ValueError("The frozen attack severity protocol changed")
    noise_cells = {
        (str(regime["name"]), str(permutation))
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
    }
    if noise_cells != {
        ("homogeneous", "identity"),
        ("heteroscedastic", "identity"),
        ("heteroscedastic", "reverse"),
        ("heteroscedastic", "byzantine_high"),
        ("heteroscedastic", "byzantine_low"),
    }:
        raise ValueError("Exactly the five frozen public noise cells are required")
    privacy_noise = config["privacy_noise"]
    if (
        set(privacy_noise)
        != {
            "base_std",
            "block_std_multipliers",
            "regimes",
        }
        or float(privacy_noise["base_std"]) != 0.012
    ):
        raise ValueError("The frozen privacy-noise schema changed")
    if list(privacy_noise["block_std_multipliers"]) != [0.70, 1.00, 1.35, 1.70]:
        raise ValueError("The frozen block noise geometry changed")
    if privacy_noise["regimes"] != [
        {
            "name": "homogeneous",
            "client_std_multipliers": [1.0],
            "permutations": ["identity"],
        },
        {
            "name": "heteroscedastic",
            "client_std_multipliers": [1.0, 1.5, 2.0],
            "permutations": [
                "identity",
                "reverse",
                "byzantine_high",
                "byzantine_low",
            ],
        },
    ]:
        raise ValueError("The frozen client noise-tier protocol changed")

    references = config["references"]
    if set(references) != {
        "public_anchor",
        "public_anchor_error_norm",
        "fcc",
        "rfa",
        "trimmed_mean",
        "g0f_public_budgets",
    }:
        raise ValueError("The frozen reference schema changed")
    if (
        references["public_anchor"] != "lagged_public_proxy"
        or float(references["public_anchor_error_norm"]) != 0.02
    ):
        raise ValueError("The frozen public anchor changed")
    if references["fcc"] != {"radius": 0.13}:
        raise ValueError("The frozen FCC pilot changed")
    if references["rfa"] != {
        "max_iter": 80,
        "tolerance": 1.0e-8,
        "smoothing": 1.0e-8,
    }:
        raise ValueError("The frozen RFA comparator changed")
    if references["trimmed_mean"] != {"trim_count": 5}:
        raise ValueError("The frozen trimmed-mean comparator changed")

    budgets = config["references"]["g0f_public_budgets"]
    for key in (
        "regular_honest_false_tail_rate",
        "correction_contamination_budget_fraction_of_fcc",
        "correction_radius_budget_fraction_of_fcc_radius",
        "target_solver_contraction",
        "solver_error_tolerance",
        "replace_one_bound_max",
        "variance_floor",
    ):
        if not math.isfinite(float(budgets[key])) or float(budgets[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0.0 < float(budgets["regular_honest_false_tail_rate"]) < 0.5:
        raise ValueError("false-tail rate must lie in (0,0.5)")
    if not 0.0 < float(budgets["target_solver_contraction"]) < 1.0:
        raise ValueError("target solver contraction must lie in (0,1)")
    if budgets != {
        "regular_honest_false_tail_rate": 0.10,
        "correction_contamination_budget_fraction_of_fcc": 1.0,
        "correction_radius_budget_fraction_of_fcc_radius": 0.20,
        "target_solver_contraction": 0.3333333333333333,
        "solver_error_tolerance": 1.0e-6,
        "replace_one_bound_max": 0.030,
        "variance_floor": 1.0e-12,
    }:
        raise ValueError("The frozen G0f public budget equations changed")

    expected_randomness_keys = {
        "calibration_folds",
        "calibration_draws_per_seed",
        "calibration_contexts",
        "pair_noise_across_regimes",
        "pair_noise_across_tier_permutations",
        "development_seeds",
        "development_draws_per_seed",
        "holdout_draws_per_seed",
        "replace_one_trials_per_seed_cell",
    }
    if set(randomness) != expected_randomness_keys:
        raise ValueError("The frozen G0f randomness schema changed")
    if folds != [[2026092701, 2026092711], [2026092803, 2026092817]]:
        raise ValueError("The frozen calibration identities changed")
    if groups[1] != [
        2026101001,
        2026101013,
        2026101027,
        2026101039,
        2026101051,
    ]:
        raise ValueError("The frozen development identities changed")
    if (
        int(randomness["calibration_draws_per_seed"]) != 32
        or int(randomness["development_draws_per_seed"]) != 1
        or int(randomness["holdout_draws_per_seed"]) != 2
        or int(randomness["replace_one_trials_per_seed_cell"]) != 4
    ):
        raise ValueError("The frozen G0f draw counts changed")
    expected_contexts = [
        {
            "name": "regular_only",
            "include_outliers": False,
            "geometry": "orthogonal",
        },
        {
            "name": "bounded_outliers_aligned",
            "include_outliers": True,
            "geometry": "aligned",
        },
        {
            "name": "bounded_outliers_orthogonal",
            "include_outliers": True,
            "geometry": "orthogonal",
        },
    ]
    if randomness["calibration_contexts"] != expected_contexts:
        raise ValueError("The frozen cross-fitted calibration contexts changed")

    gates = config["gates"]
    required_gates = (
        "clean_reference_error_ratio_to_fcc_max",
        "heteroscedastic_attacked_relative_gain_min",
        "separated_byzantine_endpoint_clipped_gradient_share_per_block_max",
        "diagnostic_group_observations_min",
        "diagnostic_group_seed_clusters_min",
        "regular_tier_clients_per_observation_min",
    )
    expected_gate_keys = {
        "complete_fraction_min",
        "stability_complete_fraction_min",
        "finite_primary_metric_fraction_min",
        "clean_reference_error_ratio_to_fcc_max",
        "attacked_reference_error_ratio_to_fcc_max",
        "evasive_reference_error_ratio_to_fcc_max",
        "heteroscedastic_attacked_difference_ci95_high_max",
        "heteroscedastic_attacked_relative_gain_min",
        "regular_honest_false_tail_upper_excess_max",
        "separated_byzantine_endpoint_clipped_gradient_share_max",
        "separated_byzantine_endpoint_clipped_gradient_share_per_block_max",
        "allocation_shape_deviation_min",
        "global_allocation_ratio_range_max",
        "complete_client_budget_max",
        "g0e_equal_cap_equivalence_tolerance",
        "replace_one_bound_max",
        "empirical_replace_one_violation_max",
        "solver_gradient_residual_max",
        "correction_radius_violation_max",
        "diagnostic_group_observations_min",
        "diagnostic_group_seed_clusters_min",
        "regular_tier_clients_per_observation_min",
    }
    if set(gates) != expected_gate_keys:
        raise ValueError("The frozen G0f gate schema changed")
    missing_gates = [key for key in required_gates if key not in gates]
    if missing_gates:
        raise ValueError(f"Missing frozen G0f gates: {missing_gates}")
    if float(gates["clean_reference_error_ratio_to_fcc_max"]) != 1.02:
        raise ValueError("The clean G0f/FCC gate must be frozen at 1.02")
    if float(gates["heteroscedastic_attacked_relative_gain_min"]) != 0.05:
        raise ValueError("The relative-gain gate must be frozen at 5 percent")
    if (
        float(
            gates["separated_byzantine_endpoint_clipped_gradient_share_per_block_max"]
        )
        != 0.25
    ):
        raise ValueError("The per-block Byzantine influence gate must be 0.25")
    if int(gates["diagnostic_group_observations_min"]) < 5:
        raise ValueError("Each diagnostic group requires at least five observations")
    if int(gates["diagnostic_group_seed_clusters_min"]) < 5:
        raise ValueError("Each diagnostic group requires at least five seed clusters")
    if int(gates["regular_tier_clients_per_observation_min"]) < 2:
        raise ValueError("Each regular public tier needs at least two honest clients")
    expected_gate_values = {
        "complete_fraction_min": 1.0,
        "stability_complete_fraction_min": 1.0,
        "finite_primary_metric_fraction_min": 1.0,
        "clean_reference_error_ratio_to_fcc_max": 1.02,
        "attacked_reference_error_ratio_to_fcc_max": 1.05,
        "evasive_reference_error_ratio_to_fcc_max": 1.05,
        "heteroscedastic_attacked_difference_ci95_high_max": 0.0,
        "heteroscedastic_attacked_relative_gain_min": 0.05,
        "regular_honest_false_tail_upper_excess_max": 0.05,
        "separated_byzantine_endpoint_clipped_gradient_share_max": 0.25,
        "separated_byzantine_endpoint_clipped_gradient_share_per_block_max": 0.25,
        "allocation_shape_deviation_min": 0.05,
        "global_allocation_ratio_range_max": 1.0e-6,
        "complete_client_budget_max": 0.13,
        "g0e_equal_cap_equivalence_tolerance": 1.0e-6,
        "replace_one_bound_max": 0.030,
        "empirical_replace_one_violation_max": 0,
        "solver_gradient_residual_max": 1.0e-6,
        "correction_radius_violation_max": 0,
        "diagnostic_group_observations_min": 5,
        "diagnostic_group_seed_clusters_min": 5,
        "regular_tier_clients_per_observation_min": 2,
    }
    if gates != expected_gate_values:
        raise ValueError("The frozen G0f gate values changed")
    expected_selection = {
        "phase": "development_gate_only",
        "primary_candidate": PRIMARY_CANDIDATE,
        "reported_candidates": list(CANDIDATES),
        "candidate_grid": "forbidden",
        "holdout_used_for_selection": False,
        "if_development_fails": "stop_without_reading_or_generating_holdout",
        "promotion_rule": "all_preregistered_development_and_holdout_gates_pass",
    }
    if config["selection"] != expected_selection:
        raise ValueError("The frozen G0f selection contract changed")
    if (
        _expected_pairings(config, "development") != 650
        or _expected_pairings(config, "development") * len(CANDIDATES) != 5850
    ):
        raise ValueError("The frozen development detail cardinality changed")
    expected_stability = (
        len(randomness["development_seeds"])
        * _noise_cell_count(config)
        * int(randomness["replace_one_trials_per_seed_cell"])
    )
    if expected_stability != 100 or expected_stability * len(CANDIDATES) != 900:
        raise ValueError("The frozen development stability cardinality changed")


def _derive_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    derived = g0e._derive_parameters(_g0e_view(config))
    result = dict(derived)
    result["derivation_version"] = "g0f_global_covariance_budget_v1"
    result.pop("influence_cap_per_block", None)
    result["allocation_radii"] = "deployed_public_radii_before_cap"
    result["allocation_global_normalizer"] = "max_i_l2_norm_a_i"
    result["allocation_policies"] = dict(POLICY_BY_CANDIDATE)
    result["covariance_provenance"] = "public_authenticated"
    return result


def _load_execution_gated_holdout_seeds(config: Mapping[str, Any]) -> list[int]:
    """Open and verify the holdout registry only after the development gate."""

    registry = config["holdout_registry"]
    relative = Path(str(registry["path"]))
    if relative.is_absolute():
        raise ValueError("Holdout registry path must be repository-relative")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Holdout registry escapes the repository") from error
    content = path.read_bytes()
    observed_digest = hashlib.sha256(content).hexdigest()
    if observed_digest != str(registry["sha256"]):
        raise RuntimeError("Execution-gated holdout registry commitment mismatch")
    payload = yaml.safe_load(content.decode("utf-8"))
    if set(payload) != {"campaign_id", "purpose", "holdout_seeds"}:
        raise ValueError("Malformed execution-gated holdout registry")
    if payload["campaign_id"] != config["campaign_id"]:
        raise ValueError("Holdout registry belongs to another campaign")
    if payload["purpose"] != "preregistered_execution_gated_holdout_identities":
        raise ValueError("Unexpected holdout registry purpose")
    seeds = [int(value) for value in payload["holdout_seeds"]]
    if len(seeds) != int(registry["expected_seed_count"]) or len(seeds) != len(
        set(seeds)
    ):
        raise ValueError("Holdout registry seed cardinality is invalid")
    known = {
        int(value)
        for fold in config["randomness"]["calibration_folds"]
        for value in fold
    } | {int(value) for value in config["randomness"]["development_seeds"]}
    if set(seeds) & known:
        raise ValueError("Holdout identities overlap calibration or development")
    if set(seeds) & {int(value) for value in config["excluded_prior_seeds"]}:
        raise ValueError("Holdout registry reuses a previously inspected seed")
    return seeds


def _calibrate(
    config: dict[str, Any], derived: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # G0e's exact calibration chain is reused.  The oracle natively supports
    # every frozen public tier assignment, so no mutable monkey-patch is used.
    artifact, rows = g0e._calibrate(_g0e_view(config), derived)
    artifact = dict(artifact)
    artifact["consumer"] = "g0f_deployed_public_pre_cap_allocation_radii"
    artifact["covariance_provenance"] = "public_authenticated"
    artifact["client_declared_covariance_forbidden"] = True
    return artifact, rows


def _radii_for_cell(
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    regime: str,
    permutation: str,
    noise_variances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return g0e._radii_for_cell(
        _g0e_view(config),
        derived,
        calibration,
        regime=regime,
        permutation=permutation,
        noise_variances=noise_variances,
    )


def _noise_variances(
    config: Mapping[str, Any], regime: Mapping[str, Any], permutation: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed public variances, including Byzantine-tier assignments.

    ``byzantine_high`` and ``byzantine_low`` preserve the public tier multiset
    and move respectively the largest or smallest ``b`` tiers onto the fixed
    Byzantine identities (the final ``b`` clients).  No observed upload or
    candidate output enters this assignment.
    """

    return oracle._noise_variances(dict(config), dict(regime), permutation)


def _paired_standard_normal(
    shape: Sequence[int],
    *,
    seed: int,
    draw: int,
    geometry: str,
) -> torch.Tensor:
    """Expose the exact standard-normal tensor shared by all noise cells."""

    return torch.randn(
        tuple(int(value) for value in shape),
        generator=oracle._generator("g0f-paired-private-noise", seed, draw, geometry),
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    )


def _paired_private_noise(
    clean: torch.Tensor,
    noise_variances: torch.Tensor,
    block_sizes: Sequence[int],
    *,
    seed: int,
    draw: int,
    geometry: str,
) -> torch.Tensor:
    """Use one standard-normal draw for every noise regime and assignment."""

    coordinate_std = oracle._expand_block_values(noise_variances.sqrt(), block_sizes)
    standard = _paired_standard_normal(
        clean.shape,
        seed=seed,
        draw=draw,
        geometry=geometry,
    )
    return clean + coordinate_std * standard


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for width in block_sizes:
        result.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(result)


def _candidate_diagnostics(
    *,
    vectors: torch.Tensor,
    reference: torch.Tensor,
    crossfit_references: torch.Tensor,
    statistical_radii: torch.Tensor,
    deployed_radii: torch.Tensor,
    diagnostics: Mapping[str, Any],
    regular_honest: torch.Tensor,
    honest_outliers: torch.Tensor,
    byzantine: torch.Tensor,
    noise_tiers: torch.Tensor,
    block_sizes: Sequence[int],
) -> dict[str, Any]:
    slices = _block_slices(block_sizes)
    caps = torch.tensor(
        diagnostics["allocated_block_budgets"],
        dtype=vectors.dtype,
        device=vectors.device,
    )
    effective = torch.minimum(deployed_radii, caps)
    returned_residual_norms = torch.empty_like(statistical_radii)
    endpoint_residual_norms = torch.empty_like(statistical_radii)
    crossfit_norms = torch.empty_like(statistical_radii)
    endpoint_clipped_gradient_norms = torch.empty_like(statistical_radii)
    pre_blend_reference = torch.tensor(
        diagnostics["pre_blend_reference"],
        dtype=vectors.dtype,
        device=vectors.device,
    )
    for block, block_slice in enumerate(slices):
        returned_residual_norms[:, block] = torch.linalg.vector_norm(
            vectors[:, block_slice] - reference[None, block_slice], dim=1
        )
        endpoint_residual_norms[:, block] = torch.linalg.vector_norm(
            vectors[:, block_slice] - pre_blend_reference[None, block_slice], dim=1
        )
        crossfit_norms[:, block] = torch.linalg.vector_norm(
            vectors[:, block_slice] - crossfit_references[:, block_slice], dim=1
        )
        endpoint_clipped_gradient_norms[:, block] = torch.minimum(
            endpoint_residual_norms[:, block], effective[:, block]
        )
    statistical_tail = crossfit_norms > statistical_radii
    cap_active = (deployed_radii > caps) & (endpoint_residual_norms > caps)
    endpoint_client_gradient_norm = torch.linalg.vector_norm(
        endpoint_clipped_gradient_norms, dim=1
    )
    endpoint_gradient_norm_total = float(endpoint_client_gradient_norm.sum().item())
    byzantine_endpoint_share = (
        float(endpoint_client_gradient_norm[byzantine].sum().item())
        / endpoint_gradient_norm_total
        if bool(byzantine.any().item()) and endpoint_gradient_norm_total > 0.0
        else float("nan")
    )
    byzantine_endpoint_share_by_block: list[float] = []
    for block in range(len(slices)):
        block_total = float(endpoint_clipped_gradient_norms[:, block].sum().item())
        byzantine_endpoint_share_by_block.append(
            float(endpoint_clipped_gradient_norms[byzantine, block].sum().item())
            / block_total
            if bool(byzantine.any().item()) and block_total > 0.0
            else float("nan")
        )

    honest = regular_honest | honest_outliers
    honest_count = int(honest.sum().item())
    outlier_count = int(honest_outliers.sum().item())
    if honest_count and outlier_count:
        standardized_energy = (
            (returned_residual_norms / deployed_radii).square().sum(dim=1)
        )
        honest_indices = torch.where(honest)[0]
        selected_local = torch.topk(
            standardized_energy[honest_indices], k=outlier_count
        ).indices
        selected = honest_indices[selected_local]
        outlier_recall = float(honest_outliers[selected].float().mean().item())
    else:
        outlier_recall = float("nan")

    regular_tail = (
        float(statistical_tail[regular_honest].float().mean().item())
        if bool(regular_honest.any().item())
        else float("nan")
    )
    regular_cap = (
        float(cap_active[regular_honest].float().mean().item())
        if bool(regular_honest.any().item())
        else float("nan")
    )
    equal = float(diagnostics["complete_client_influence_cap"]) / math.sqrt(
        float(len(slices))
    )
    shape_deviation = float(
        torch.linalg.vector_norm(caps - equal, dim=1).mean().item()
        / max(float(diagnostics["complete_client_influence_cap"]), 1.0e-30)
    )
    allocation = diagnostics["allocation_diagnostics"]
    tier_counts = [
        int((regular_honest & torch.isclose(noise_tiers, tier)).sum().item())
        for tier in torch.unique(noise_tiers)
        if bool((regular_honest & torch.isclose(noise_tiers, tier)).any().item())
    ]
    return {
        "regular_honest_statistical_tail_rate": regular_tail,
        "regular_honest_cap_activation_rate": regular_cap,
        "honest_outlier_topk_recall": outlier_recall,
        "byzantine_endpoint_clipped_gradient_share": byzantine_endpoint_share,
        "byzantine_endpoint_clipped_gradient_share_by_block": (
            byzantine_endpoint_share_by_block
        ),
        "byzantine_endpoint_share_estimand": (
            "norm_share_of_client_huber_gradients_at_unblended_final_iterate_p_K"
        ),
        "byzantine_endpoint_share_is_causal_decomposition": False,
        "regular_clients_per_public_tier_min": min(tier_counts),
        "allocation_shape_deviation_from_equal": shape_deviation,
        "allocation_ratio_range": float(allocation["budget_to_radius_ratio_max"])
        - float(allocation["budget_to_radius_ratio_min"]),
        "allocated_client_norm_max": float(allocation["allocated_client_norm_max"]),
    }


def _correction(
    name: str,
    vectors: torch.Tensor,
    *,
    pilot: torch.Tensor,
    crossfit: torch.Tensor,
    statistical: torch.Tensor,
    deployed: torch.Tensor,
    derived: Mapping[str, Any],
    block_sizes: Sequence[int],
    num_replacements_for_diagnostics: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    common = dict(
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=statistical,
        deployed_radii=deployed,
        pilot_replace_one_bound=float(derived["pilot_replace_one_bound"]),
        block_sizes=block_sizes,
        regularization=float(derived["regularization"]),
        correction_budget=float(derived["correction_budget"]),
        num_steps=int(derived["num_steps"]),
        return_diagnostics=True,
    )
    if name == "g0e":
        num_blocks = len(block_sizes)
        cap = float(derived["influence_cap_total"]) / math.sqrt(float(num_blocks))
        return gaussian_aware_crossfit_bounded_correction(
            vectors, influence_cap=[cap] * num_blocks, **common
        )
    return gaussian_aware_budget_allocated_correction(
        vectors,
        total_influence_budget=float(derived["influence_cap_total"]),
        allocation_radii=deployed,
        allocation_policy=POLICY_BY_CANDIDATE[name],
        allocation_radius_provenance="effective_null_radius",
        covariance_provenance="public_authenticated",
        num_replacements_for_diagnostics=num_replacements_for_diagnostics,
        **common,
    )


def _g0e_diagnostics_as_g0f(
    diagnostics: Mapping[str, Any],
    *,
    n: int,
    num_blocks: int,
) -> dict[str, Any]:
    """Expose G0e equal caps in the common G0f diagnostic schema."""

    cap = float(diagnostics["complete_client_influence_cap"]) / math.sqrt(
        float(num_blocks)
    )
    caps = [[cap] * num_blocks for _ in range(n)]
    return {
        **diagnostics,
        "allocated_block_budgets": caps,
        "allocation_diagnostics": {
            "budget_to_radius_ratio_min": float("nan"),
            "budget_to_radius_ratio_max": float("nan"),
            "allocated_client_norm_max": float(
                diagnostics["complete_client_influence_cap"]
            ),
        },
    }


def _evaluate_pairing(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    pairing_id: str,
    vectors: torch.Tensor,
    clean: torch.Tensor,
    observed_private: torch.Tensor,
    standard_noise: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
    noise_tiers: torch.Tensor,
    centre: torch.Tensor,
    regime: str,
    permutation: str,
    geometry: str,
    threat: str,
    severity: float,
    seed: int,
    draw: int,
) -> list[dict[str, Any]]:
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    bounded = clip_l2(vectors, float(config["aggregation"]["server_clip_norm"]))
    pilot = g0e._comparator_reference("fcc", bounded, config, anchor=anchor)
    crossfit = g0e._fcc_leave_one_out(
        bounded, anchor=anchor, radius=float(config["references"]["fcc"]["radius"])
    )
    statistical, deployed = _radii_for_cell(
        config,
        derived,
        calibration,
        regime=regime,
        permutation=permutation,
        noise_variances=noise_variances,
    )
    references: dict[str, torch.Tensor] = {}
    diagnostics_by_name: dict[str, dict[str, Any]] = {}
    for name in BASELINES:
        if name == "fcc":
            references[name] = pilot
        elif name == "coordinate_median":
            references[name] = coordinate_median(bounded)
        else:
            references[name] = g0e._comparator_reference(
                name, bounded, config, anchor=anchor
            )
    for name in CORRECTIONS:
        reference, diagnostics = _correction(
            name,
            bounded,
            pilot=pilot,
            crossfit=crossfit,
            statistical=statistical,
            deployed=deployed,
            derived=derived,
            block_sizes=block_sizes,
            num_replacements_for_diagnostics=int(config["cohort"]["num_byzantine"]),
        )
        references[name] = reference
        diagnostics_by_name[name] = (
            _g0e_diagnostics_as_g0f(
                diagnostics,
                n=int(bounded.shape[0]),
                num_blocks=len(block_sizes),
            )
            if name == "g0e"
            else diagnostics
        )
    equal_equivalence = float(
        torch.linalg.vector_norm(references["g0e"] - references["g0f_equal_cap"]).item()
    )
    honest = ~byzantine_mask
    regular_honest = honest & ~outlier_mask
    target = clean[honest].mean(dim=0)
    coordinate_std = oracle._expand_block_values(noise_variances.sqrt(), block_sizes)
    pairing_hashes = {
        "clean_cohort_sha256": _tensor_sha256(clean),
        "standard_noise_z_sha256": _tensor_sha256(standard_noise),
        "observed_private_cohort_sha256": _tensor_sha256(observed_private),
        "noise_rescaled_source_z_sha256": _tensor_sha256(standard_noise),
        "coordinate_noise_std_sha256": _tensor_sha256(coordinate_std),
        "attacked_cohort_sha256": _tensor_sha256(vectors),
        "tier_assignment_sha256": _tensor_sha256(noise_tiers),
        "outlier_mask_sha256": _tensor_sha256(outlier_mask),
        "byzantine_mask_sha256": _tensor_sha256(byzantine_mask),
    }
    rows: list[dict[str, Any]] = []
    for name in CANDIDATES:
        reference = references[name]
        is_correction = name in CORRECTIONS
        metrics: dict[str, Any] = {}
        diagnostics = diagnostics_by_name.get(name)
        if is_correction and diagnostics is not None:
            metrics = _candidate_diagnostics(
                vectors=bounded,
                reference=reference,
                crossfit_references=crossfit,
                statistical_radii=statistical,
                deployed_radii=deployed,
                diagnostics=diagnostics,
                regular_honest=regular_honest,
                honest_outliers=honest & outlier_mask,
                byzantine=byzantine_mask,
                noise_tiers=noise_tiers,
                block_sizes=block_sizes,
            )
        rows.append(
            {
                "campaign_id": config["campaign_id"],
                "phase": phase,
                "pairing_id": pairing_id,
                "candidate": name,
                "is_primary_candidate": name == PRIMARY_CANDIDATE,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "severity": severity,
                "seed": seed,
                "draw": draw,
                **pairing_hashes,
                "reference_error": float(
                    torch.linalg.vector_norm(reference - target).item()
                ),
                "reference_error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "reference_error_ratio_to_uniform": float("nan"),
                "reference_error_ratio_to_fcc": float("nan"),
                "regular_honest_statistical_tail_rate": metrics.get(
                    "regular_honest_statistical_tail_rate", float("nan")
                ),
                "regular_honest_cap_activation_rate": metrics.get(
                    "regular_honest_cap_activation_rate", float("nan")
                ),
                "honest_outlier_topk_recall": metrics.get(
                    "honest_outlier_topk_recall", float("nan")
                ),
                "byzantine_endpoint_clipped_gradient_share": metrics.get(
                    "byzantine_endpoint_clipped_gradient_share", float("nan")
                ),
                "byzantine_endpoint_clipped_gradient_share_by_block": metrics.get(
                    "byzantine_endpoint_clipped_gradient_share_by_block", None
                ),
                "regular_clients_per_public_tier_min": metrics.get(
                    "regular_clients_per_public_tier_min", float("nan")
                ),
                "allocation_shape_deviation_from_equal": metrics.get(
                    "allocation_shape_deviation_from_equal", float("nan")
                ),
                "allocation_ratio_range": metrics.get(
                    "allocation_ratio_range", float("nan")
                ),
                "allocated_client_norm_max": metrics.get(
                    "allocated_client_norm_max", float("nan")
                ),
                "g0e_equal_cap_reference_difference": (
                    equal_equivalence
                    if name in {"g0e", "g0f_equal_cap"}
                    else float("nan")
                ),
                "solver_gradient_residual": (
                    float(diagnostics["gradient_residual_norm"])
                    if diagnostics is not None
                    else float("nan")
                ),
                "correction_budget_violation": (
                    not bool(diagnostics["correction_budget_respected"])
                    if diagnostics is not None
                    else None
                ),
                "finite_solver_replace_one_bound": (
                    float(diagnostics["finite_solver_replace_one_bound"])
                    if diagnostics is not None
                    else float("nan")
                ),
                "resolved_device": str(bounded.device),
                "tensor_dtype": str(bounded.dtype).replace("torch.", ""),
            }
        )
    uniform_error = next(
        float(row["reference_error"])
        for row in rows
        if row["candidate"] == "uniform_mean"
    )
    fcc_error = next(
        float(row["reference_error"]) for row in rows if row["candidate"] == "fcc"
    )
    if uniform_error <= 0.0 or fcc_error <= 0.0:
        raise RuntimeError("A paired baseline has zero reference error")
    for row in rows:
        row["reference_error_ratio_to_uniform"] = (
            float(row["reference_error"]) / uniform_error
        )
        row["reference_error_ratio_to_fcc"] = float(row["reference_error"]) / fcc_error
    return rows


def _phase_rows(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, tiers = _noise_variances(config, regime, permutation)
            for seed_index, seed_value in enumerate(seeds, start=1):
                seed = int(seed_value)
                print(
                    f"[G0f] {phase} seed {seed_index}/{len(seeds)}: {seed}",
                    flush=True,
                )
                for draw in range(draws_per_seed):
                    for geometry_value in config["cohort"]["honest_outliers"][
                        "geometries"
                    ]:
                        geometry = str(geometry_value)
                        clean, outliers, centre, anchor = oracle._honest_clean_vectors(
                            config,
                            seed=seed,
                            draw=draw,
                            geometry=geometry,
                            include_outliers=True,
                        )
                        observed = _paired_private_noise(
                            clean,
                            variances,
                            blocks,
                            seed=seed,
                            draw=draw,
                            geometry=geometry,
                        )
                        standard_noise = _paired_standard_normal(
                            clean.shape,
                            seed=seed,
                            draw=draw,
                            geometry=geometry,
                        )
                        for threat_value in config["threats"]["names"]:
                            threat = str(threat_value)
                            levels = [1.0] if threat == "none" else severities
                            for severity in levels:
                                attacked, byzantine = oracle._replace_with_attack(
                                    observed,
                                    config,
                                    threat=threat,
                                    severity=float(severity),
                                    seed=oracle._seed(
                                        "g0f-attack", seed, draw, geometry, threat
                                    ),
                                )
                                pairing_id = (
                                    f"{phase}:{regime_name}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{seed}:{draw}"
                                )
                                rows.extend(
                                    _evaluate_pairing(
                                        config=config,
                                        derived=derived,
                                        calibration=calibration,
                                        phase=phase,
                                        pairing_id=pairing_id,
                                        vectors=attacked,
                                        clean=clean,
                                        observed_private=observed,
                                        standard_noise=standard_noise,
                                        outlier_mask=outliers,
                                        byzantine_mask=byzantine,
                                        anchor=anchor,
                                        noise_variances=variances,
                                        noise_tiers=tiers,
                                        centre=centre,
                                        regime=regime_name,
                                        permutation=permutation,
                                        geometry=geometry,
                                        threat=threat,
                                        severity=float(severity),
                                        seed=seed,
                                        draw=draw,
                                    )
                                )
    return rows


def _noise_cell_count(config: Mapping[str, Any]) -> int:
    return sum(
        len(regime["permutations"]) for regime in config["privacy_noise"]["regimes"]
    )


def _expected_pairings_per_seed(config: Mapping[str, Any], phase: str) -> int:
    draws = int(config["randomness"][f"{phase}_draws_per_seed"])
    levels = config["threats"][f"{phase}_severities"]
    geometries = len(config["cohort"]["honest_outliers"]["geometries"])
    threats = 1 + (len(config["threats"]["names"]) - 1) * len(levels)
    return draws * _noise_cell_count(config) * geometries * threats


def _expected_pairings(config: Mapping[str, Any], phase: str) -> int:
    return len(config["randomness"][f"{phase}_seeds"]) * _expected_pairings_per_seed(
        config, phase
    )


def _validate_checkpoint(
    rows: Any,
    *,
    config: Mapping[str, Any],
    phase: str,
    seed: int,
    kind: str,
    source: Path,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Empty checkpoint {source}")
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    expected = (
        _expected_pairings_per_seed(config, phase) * len(CANDIDATES)
        if kind == "detail"
        else _noise_cell_count(config) * trials * len(CANDIDATES)
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"Incomplete {kind} checkpoint {source}: {len(rows)}/{expected}"
        )
    for row in rows:
        if row.get("phase") != phase or int(row.get("seed", -1)) != seed:
            raise RuntimeError(f"Wrong phase or seed in checkpoint {source}")
        if not str(row.get("resolved_device", "")).startswith("mps"):
            raise RuntimeError(f"Non-MPS checkpoint rejected: {source}")
        if row.get("tensor_dtype") != "float32":
            raise RuntimeError(f"Non-float32 checkpoint rejected: {source}")
        if row.get("candidate") not in CANDIDATES:
            raise RuntimeError(f"Unknown candidate in checkpoint {source}")
    if kind == "detail":
        paired: dict[str, set[str]] = defaultdict(set)
        pairing_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        identities: list[tuple[str, str]] = []
        for row in rows:
            pairing_id = str(row["pairing_id"])
            candidate = str(row["candidate"])
            paired[pairing_id].add(candidate)
            pairing_rows[pairing_id].append(row)
            identities.append((pairing_id, candidate))
        if len(identities) != len(set(identities)):
            raise RuntimeError(f"Duplicate detail rows in {source}")
        if len(paired) != _expected_pairings_per_seed(config, phase) or any(
            values != set(CANDIDATES) for values in paired.values()
        ):
            raise RuntimeError(f"Incomplete candidate pairing in {source}")
        expected_detail_grid: set[tuple[str, str]] = set()
        for regime in config["privacy_noise"]["regimes"]:
            for permutation in regime["permutations"]:
                for draw in range(int(config["randomness"][f"{phase}_draws_per_seed"])):
                    for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                        for threat in config["threats"]["names"]:
                            severity_values = (
                                [1.0]
                                if threat == "none"
                                else config["threats"][f"{phase}_severities"]
                            )
                            for severity in severity_values:
                                expected_pairing_id = (
                                    f"{phase}:{regime['name']}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{seed}:{draw}"
                                )
                                expected_detail_grid.update(
                                    (expected_pairing_id, candidate)
                                    for candidate in CANDIDATES
                                )
        if set(identities) != expected_detail_grid:
            raise RuntimeError(f"Detail checkpoint grid mismatch in {source}")
        for row in rows:
            expected_pairing_id = (
                f"{phase}:{row['noise_regime']}:{row['noise_permutation']}:"
                f"{row['outlier_geometry']}:{row['threat']}:"
                f"{float(row['severity']):.3f}:{seed}:{int(row['draw'])}"
            )
            if row["pairing_id"] != expected_pairing_id:
                raise RuntimeError(f"Incoherent pairing metadata in {source}")
        hash_fields = (
            "clean_cohort_sha256",
            "standard_noise_z_sha256",
            "observed_private_cohort_sha256",
            "noise_rescaled_source_z_sha256",
            "coordinate_noise_std_sha256",
            "attacked_cohort_sha256",
            "tier_assignment_sha256",
            "outlier_mask_sha256",
            "byzantine_mask_sha256",
        )
        for pairing_id, grouped in pairing_rows.items():
            for field in hash_fields:
                values = {str(row.get(field, "")) for row in grouped}
                if len(values) != 1:
                    raise RuntimeError(
                        f"Unpaired {field} within {pairing_id} in {source}"
                    )
                digest = next(iter(values))
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise RuntimeError(f"Invalid {field} in {source}")
            if any(
                not all(
                    math.isfinite(float(row[field]))
                    for field in (
                        "reference_error",
                        "reference_error_ratio_to_uniform",
                        "reference_error_ratio_to_fcc",
                    )
                )
                for row in grouped
            ):
                raise RuntimeError(f"Non-finite primary metric in {source}")
        representative_rows = [grouped[0] for grouped in pairing_rows.values()]
        across_noise: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        within_noise_cell: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for row in representative_rows:
            across_noise[(int(row["draw"]), str(row["outlier_geometry"]))].append(row)
            within_noise_cell[
                (
                    str(row["noise_regime"]),
                    str(row["noise_permutation"]),
                    int(row["draw"]),
                    str(row["outlier_geometry"]),
                )
            ].append(row)
        for grouped in across_noise.values():
            for field in (
                "clean_cohort_sha256",
                "standard_noise_z_sha256",
                "noise_rescaled_source_z_sha256",
                "outlier_mask_sha256",
            ):
                if len({str(row[field]) for row in grouped}) != 1:
                    raise RuntimeError(
                        f"Cross-regime pairing violation for {field} in {source}"
                    )
        for grouped in within_noise_cell.values():
            for field in (
                "observed_private_cohort_sha256",
                "coordinate_noise_std_sha256",
                "tier_assignment_sha256",
            ):
                if len({str(row[field]) for row in grouped}) != 1:
                    raise RuntimeError(
                        f"Within-cell pairing violation for {field} in {source}"
                    )
    else:
        identities = [
            (
                str(row["noise_regime"]),
                str(row["noise_permutation"]),
                int(row["trial"]),
                str(row["candidate"]),
            )
            for row in rows
        ]
        if len(identities) != len(set(identities)):
            raise RuntimeError(f"Duplicate stability rows in {source}")
        expected_grid = {
            (
                str(regime["name"]),
                str(permutation),
                trial,
                candidate,
            )
            for regime in config["privacy_noise"]["regimes"]
            for permutation in regime["permutations"]
            for trial in range(trials)
            for candidate in CANDIDATES
        }
        if set(identities) != expected_grid:
            raise RuntimeError(f"Incomplete stability grid in {source}")


def _cached_phase_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in seeds:
        seed = int(value)
        checkpoint = checkpoint_dir / f"{phase}_detail_seed_{seed}.json"
        if resume and checkpoint.exists():
            produced = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            produced = _phase_rows(
                config=config,
                derived=derived,
                calibration=calibration,
                phase=phase,
                seeds=[seed],
                draws_per_seed=draws_per_seed,
                severities=severities,
            )
            _atomic_json(checkpoint, produced)
        _validate_checkpoint(
            produced,
            config=config,
            phase=phase,
            seed=seed,
            kind="detail",
            source=checkpoint,
        )
        rows.extend(produced)
    return rows


def _replace_one_rows(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    fcc_radius = float(config["references"]["fcc"]["radius"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = _noise_variances(config, regime, permutation)
            statistical, deployed = _radii_for_cell(
                config,
                derived,
                calibration,
                regime=regime_name,
                permutation=permutation,
                noise_variances=variances,
            )
            for seed_value in seeds:
                seed = int(seed_value)
                for trial in range(trials):
                    clean, _, _, anchor = oracle._honest_clean_vectors(
                        config,
                        seed=seed,
                        draw=trial,
                        geometry="orthogonal",
                        include_outliers=False,
                    )
                    observed = _paired_private_noise(
                        clean,
                        variances,
                        blocks,
                        seed=seed,
                        draw=trial,
                        geometry="g0f-replace",
                    )
                    left_vectors = clip_l2(observed, server_clip)
                    replaced = trial % int(left_vectors.shape[0])
                    replacement = torch.randn(
                        left_vectors.shape[1],
                        generator=oracle._generator(
                            "g0f-replacement", phase, seed, trial
                        ),
                        dtype=left_vectors.dtype,
                        device=left_vectors.device,
                    )
                    replacement = clip_l2(
                        (2.0 * server_clip * replacement)[None, :], server_clip
                    )[0]
                    right_vectors = left_vectors.clone()
                    right_vectors[replaced] = replacement
                    left_pilot = g0e._comparator_reference(
                        "fcc", left_vectors, config, anchor=anchor
                    )
                    right_pilot = g0e._comparator_reference(
                        "fcc", right_vectors, config, anchor=anchor
                    )
                    left_crossfit = g0e._fcc_leave_one_out(
                        left_vectors, anchor=anchor, radius=fcc_radius
                    )
                    right_crossfit = g0e._fcc_leave_one_out(
                        right_vectors, anchor=anchor, radius=fcc_radius
                    )
                    for name in CANDIDATES:
                        if name in CORRECTIONS:
                            left, diagnostics = _correction(
                                name,
                                left_vectors,
                                pilot=left_pilot,
                                crossfit=left_crossfit,
                                statistical=statistical,
                                deployed=deployed,
                                derived=derived,
                                block_sizes=blocks,
                                num_replacements_for_diagnostics=int(
                                    config["cohort"]["num_byzantine"]
                                ),
                            )
                            right, _ = _correction(
                                name,
                                right_vectors,
                                pilot=right_pilot,
                                crossfit=right_crossfit,
                                statistical=statistical,
                                deployed=deployed,
                                derived=derived,
                                block_sizes=blocks,
                                num_replacements_for_diagnostics=int(
                                    config["cohort"]["num_byzantine"]
                                ),
                            )
                            bound = float(
                                diagnostics["finite_solver_replace_one_bound"]
                            )
                            certificate = "fcc_pilot_plus_fixed_public_radii_finite_k"
                        else:
                            if name == "fcc":
                                left, right = left_pilot, right_pilot
                            elif name == "coordinate_median":
                                left = coordinate_median(left_vectors)
                                right = coordinate_median(right_vectors)
                            else:
                                left = g0e._comparator_reference(
                                    name, left_vectors, config, anchor=anchor
                                )
                                right = g0e._comparator_reference(
                                    name, right_vectors, config, anchor=anchor
                                )
                            if name == "uniform_mean":
                                bound = 2.0 * server_clip / len(left_vectors)
                                certificate = "replace_one_after_server_clip"
                            elif name == "fcc":
                                bound = 2.0 * fcc_radius / len(left_vectors)
                                certificate = "replace_one_fixed_anchor"
                            elif name == "trimmed_mean":
                                trim = float(
                                    config["references"]["trimmed_mean"]["trim_count"]
                                )
                                bound = (
                                    2.0
                                    * server_clip
                                    * math.sqrt(float(left_vectors.shape[1]))
                                    / (float(len(left_vectors)) - 2.0 * trim)
                                )
                                certificate = "dimension_dependent_only"
                            elif name in {"rfa", "coordinate_median"}:
                                bound = float("nan")
                                certificate = "no_dimension_free_certificate"
                            else:
                                raise RuntimeError(f"Unhandled comparator {name}")
                        observed_delta = float(
                            torch.linalg.vector_norm(left - right).item()
                        )
                        rows.append(
                            {
                                "phase": phase,
                                "candidate": name,
                                "noise_regime": regime_name,
                                "noise_permutation": permutation,
                                "seed": seed,
                                "trial": trial,
                                "replaced_client": replaced,
                                "observed_delta": observed_delta,
                                "theoretical_bound": bound,
                                "certificate": certificate,
                                "violation": (
                                    bool(observed_delta > bound + 1.0e-6)
                                    if math.isfinite(bound)
                                    else None
                                ),
                                "resolved_device": str(left_vectors.device),
                                "tensor_dtype": str(left_vectors.dtype).replace(
                                    "torch.", ""
                                ),
                            }
                        )
    return rows


def _cached_stability_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in seeds:
        seed = int(value)
        checkpoint = checkpoint_dir / f"{phase}_stability_seed_{seed}.json"
        if resume and checkpoint.exists():
            produced = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            produced = _replace_one_rows(
                config=config,
                derived=derived,
                calibration=calibration,
                phase=phase,
                seeds=[seed],
            )
            _atomic_json(checkpoint, produced)
        _validate_checkpoint(
            produced,
            config=config,
            phase=phase,
            seed=seed,
            kind="stability",
            source=checkpoint,
        )
        rows.extend(produced)
    return rows


def _paired_ci95(
    rows: Sequence[dict[str, Any]],
    *,
    left: str,
    right: str,
    predicate: Any,
) -> tuple[float, float, float, int]:
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    seeds: dict[str, int] = {}
    for row in rows:
        if predicate(row) and row["candidate"] in {left, right}:
            pair = str(row["pairing_id"])
            paired[pair][str(row["candidate"])] = float(row["reference_error"])
            seeds[pair] = int(row["seed"])
    per_seed: dict[int, list[float]] = defaultdict(list)
    for pair, values in paired.items():
        if set(values) != {left, right}:
            raise RuntimeError(f"Incomplete comparison {pair}")
        per_seed[seeds[pair]].append(values[left] - values[right])
    seed_means = [_finite_mean(values) for _, values in sorted(per_seed.items())]
    mean = _finite_mean(seed_means)
    if len(seed_means) < 2:
        return mean, float("-inf"), float("inf"), len(seed_means)
    half = (
        _critical_t95(len(seed_means))
        * statistics.stdev(seed_means)
        / math.sqrt(len(seed_means))
    )
    return mean, mean - half, mean + half, len(seed_means)


def _paired_relative_gain_by_seed(
    rows: Sequence[dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    predicate: Any,
) -> tuple[float, list[dict[str, float]]]:
    """Return mean seed-level ``(baseline-candidate)/baseline`` gains."""

    paired: dict[str, dict[str, float]] = defaultdict(dict)
    seeds: dict[str, int] = {}
    for row in rows:
        if predicate(row) and row["candidate"] in {candidate, baseline}:
            pair = str(row["pairing_id"])
            paired[pair][str(row["candidate"])] = float(row["reference_error"])
            seeds[pair] = int(row["seed"])
    per_seed: dict[int, list[float]] = defaultdict(list)
    for pair, values in paired.items():
        if set(values) != {candidate, baseline}:
            raise RuntimeError(f"Incomplete relative-gain pairing {pair}")
        denominator = values[baseline]
        if denominator <= 0.0:
            raise RuntimeError("Relative-gain baseline error must be positive")
        per_seed[seeds[pair]].append((denominator - values[candidate]) / denominator)
    evidence = [
        {"seed": float(seed), "relative_gain": _finite_mean(values)}
        for seed, values in sorted(per_seed.items())
    ]
    return _finite_mean(row["relative_gain"] for row in evidence), evidence


def _worst_group(rows: Sequence[dict[str, Any]], field: str, *, maximum: bool) -> float:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["noise_regime"],
            row["noise_permutation"],
            row["outlier_geometry"],
            row["threat"],
            row["severity"],
        )
        groups[key].append(float(row[field]))
    values = [_finite_mean(group) for group in groups.values()]
    return max(values) if maximum else min(values)


def _worst_seed_clustered_block_mean(
    rows: Sequence[dict[str, Any]], field: str
) -> float:
    """Worst cell/block mean, averaging draws then seed clusters equally."""

    within_seed: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        block_values = row[field]
        if not isinstance(block_values, list) or not block_values:
            raise RuntimeError(f"Missing block diagnostic {field}")
        for block, value in enumerate(block_values):
            key = (
                row["noise_regime"],
                row["noise_permutation"],
                row["outlier_geometry"],
                row["threat"],
                row["severity"],
                block,
                int(row["seed"]),
            )
            within_seed[key].append(float(value))
    across_seeds: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, draw_values in within_seed.items():
        across_seeds[key[:-1]].append(_finite_mean(draw_values))
    if not across_seeds:
        raise RuntimeError(f"No observations for block diagnostic {field}")
    return max(_finite_mean(seed_values) for seed_values in across_seeds.values())


def _summarize(
    rows: Sequence[dict[str, Any]],
    stability: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == PRIMARY_CANDIDATE]
    expected = _expected_pairings(config, phase)
    clean = [row for row in selected if row["threat"] == "none"]
    separated = set(config["threats"]["separated_for_gates"])
    evasive = set(config["threats"]["evasive_controls"])
    attacked = [row for row in selected if row["threat"] in separated]
    attacked_byzantine_high = [
        row for row in attacked if row["noise_permutation"] == "byzantine_high"
    ]
    evasive_rows = [row for row in selected if row["threat"] in evasive]
    stable = [row for row in stability if row["candidate"] == PRIMARY_CANDIDATE]
    expected_stability = (
        len(config["randomness"][f"{phase}_seeds"])
        * _noise_cell_count(config)
        * int(config["randomness"]["replace_one_trials_per_seed_cell"])
    )

    def predicate(row: Mapping[str, Any]) -> bool:
        return bool(
            row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
        )

    vs_fcc = _paired_ci95(
        rows, left=PRIMARY_CANDIDATE, right="fcc", predicate=predicate
    )
    vs_g0e = _paired_ci95(
        rows, left=PRIMARY_CANDIDATE, right="g0e", predicate=predicate
    )
    vs_equal = _paired_ci95(
        rows,
        left=PRIMARY_CANDIDATE,
        right="g0f_equal_cap",
        predicate=predicate,
    )
    vs_scale_blind = _paired_ci95(
        rows,
        left=PRIMARY_CANDIDATE,
        right="g0f_per_client_scale_blind",
        predicate=predicate,
    )
    gain_vs_fcc, gain_vs_fcc_evidence = _paired_relative_gain_by_seed(
        rows,
        candidate=PRIMARY_CANDIDATE,
        baseline="fcc",
        predicate=predicate,
    )
    gain_vs_equal, gain_vs_equal_evidence = _paired_relative_gain_by_seed(
        rows,
        candidate=PRIMARY_CANDIDATE,
        baseline="g0f_equal_cap",
        predicate=predicate,
    )
    target_tail = float(
        config["references"]["g0f_public_budgets"]["regular_honest_false_tail_rate"]
    )
    summary: dict[str, Any] = {
        "phase": phase,
        "candidate": PRIMARY_CANDIDATE,
        "observations": len(selected),
        "expected_observations": expected,
        "complete_fraction": len(selected) / expected,
        "finite_primary_fraction": _finite_mean(
            float(
                math.isfinite(float(row["reference_error"]))
                and math.isfinite(float(row["reference_error_ratio_to_fcc"]))
            )
            for row in selected
        ),
        "clean_error_ratio_to_fcc_worst_group": _worst_group(
            clean, "reference_error_ratio_to_fcc", maximum=True
        ),
        "attacked_error_ratio_to_fcc_worst_group": _worst_group(
            attacked, "reference_error_ratio_to_fcc", maximum=True
        ),
        "evasive_error_ratio_to_fcc_worst_group": _worst_group(
            evasive_rows, "reference_error_ratio_to_fcc", maximum=True
        ),
        "heteroscedastic_attacked_difference_to_fcc_mean": vs_fcc[0],
        "heteroscedastic_attacked_difference_to_fcc_ci95_low": vs_fcc[1],
        "heteroscedastic_attacked_difference_to_fcc_ci95_high": vs_fcc[2],
        "heteroscedastic_attacked_difference_to_g0e_ci95_high": vs_g0e[2],
        "heteroscedastic_attacked_difference_to_equal_cap_ci95_high": vs_equal[2],
        "heteroscedastic_attacked_difference_to_scale_blind_ci95_high": (
            vs_scale_blind[2]
        ),
        "heteroscedastic_attacked_relative_gain_vs_fcc_seed_mean": gain_vs_fcc,
        "heteroscedastic_attacked_relative_gain_vs_equal_cap_seed_mean": (
            gain_vs_equal
        ),
        "relative_gain_vs_fcc_by_seed": gain_vs_fcc_evidence,
        "relative_gain_vs_equal_cap_by_seed": gain_vs_equal_evidence,
        "regular_false_tail_upper_excess_worst_group": max(
            0.0,
            _worst_group(clean, "regular_honest_statistical_tail_rate", maximum=True)
            - target_tail,
        ),
        "honest_outlier_topk_recall_worst_group": _worst_group(
            clean, "honest_outlier_topk_recall", maximum=False
        ),
        "separated_byzantine_endpoint_clipped_gradient_share_worst_group": _worst_group(
            attacked_byzantine_high,
            "byzantine_endpoint_clipped_gradient_share",
            maximum=True,
        ),
        "separated_byzantine_endpoint_clipped_gradient_share_worst_group_block": (
            _worst_seed_clustered_block_mean(
                attacked_byzantine_high,
                "byzantine_endpoint_clipped_gradient_share_by_block",
            )
        ),
        "allocation_shape_deviation_min_heteroscedastic": min(
            float(row["allocation_shape_deviation_from_equal"])
            for row in selected
            if row["noise_regime"] == "heteroscedastic"
        ),
        "allocation_ratio_range_max": max(
            float(row["allocation_ratio_range"]) for row in selected
        ),
        "allocated_client_norm_max": max(
            float(row["allocated_client_norm_max"]) for row in selected
        ),
        "g0e_equal_cap_reference_difference_max": max(
            float(row["g0e_equal_cap_reference_difference"])
            for row in rows
            if row["candidate"] == "g0f_equal_cap"
        ),
        "stability_observations": len(stable),
        "expected_stability_observations": expected_stability,
        "stability_complete_fraction": len(stable) / expected_stability,
        "replace_one_bound_max": max(float(row["theoretical_bound"]) for row in stable),
        "replace_one_observed_max": max(float(row["observed_delta"]) for row in stable),
        "replace_one_violation_count": sum(bool(row["violation"]) for row in stable),
        "solver_gradient_residual_max": max(
            float(row["solver_gradient_residual"]) for row in selected
        ),
        "correction_radius_violation_count": sum(
            bool(row["correction_budget_violation"]) for row in selected
        ),
        "regular_clients_per_public_tier_min": min(
            int(row["regular_clients_per_public_tier_min"]) for row in selected
        ),
    }
    group_observations: dict[tuple[Any, ...], int] = defaultdict(int)
    group_seeds: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in selected:
        key = (
            row["noise_regime"],
            row["noise_permutation"],
            row["outlier_geometry"],
            row["threat"],
            row["severity"],
        )
        group_observations[key] += 1
        group_seeds[key].add(int(row["seed"]))
    summary["diagnostic_group_observations_min"] = min(group_observations.values())
    summary["diagnostic_group_seed_clusters_min"] = min(
        len(value) for value in group_seeds.values()
    )
    gates = config["gates"]
    checks = {
        "complete": summary["complete_fraction"]
        >= float(gates["complete_fraction_min"]),
        "finite": summary["finite_primary_fraction"]
        >= float(gates["finite_primary_metric_fraction_min"]),
        "clean_vs_fcc": summary["clean_error_ratio_to_fcc_worst_group"]
        <= float(gates["clean_reference_error_ratio_to_fcc_max"]),
        "attacked_vs_fcc": summary["attacked_error_ratio_to_fcc_worst_group"]
        <= float(gates["attacked_reference_error_ratio_to_fcc_max"]),
        "evasive_vs_fcc": summary["evasive_error_ratio_to_fcc_worst_group"]
        <= float(gates["evasive_reference_error_ratio_to_fcc_max"]),
        "heteroscedastic_vs_fcc": summary[
            "heteroscedastic_attacked_difference_to_fcc_ci95_high"
        ]
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "heteroscedastic_vs_g0e": summary[
            "heteroscedastic_attacked_difference_to_g0e_ci95_high"
        ]
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "heteroscedastic_vs_equal_cap": summary[
            "heteroscedastic_attacked_difference_to_equal_cap_ci95_high"
        ]
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "heteroscedastic_vs_scale_blind": summary[
            "heteroscedastic_attacked_difference_to_scale_blind_ci95_high"
        ]
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "tail_calibration": summary["regular_false_tail_upper_excess_worst_group"]
        <= float(gates["regular_honest_false_tail_upper_excess_max"]),
        "relative_gain_vs_fcc": summary[
            "heteroscedastic_attacked_relative_gain_vs_fcc_seed_mean"
        ]
        >= float(gates["heteroscedastic_attacked_relative_gain_min"]),
        "relative_gain_vs_equal_cap": summary[
            "heteroscedastic_attacked_relative_gain_vs_equal_cap_seed_mean"
        ]
        >= float(gates["heteroscedastic_attacked_relative_gain_min"]),
        "byzantine_endpoint_gradient_share": summary[
            "separated_byzantine_endpoint_clipped_gradient_share_worst_group"
        ]
        <= float(gates["separated_byzantine_endpoint_clipped_gradient_share_max"]),
        "byzantine_share_by_block": summary[
            "separated_byzantine_endpoint_clipped_gradient_share_worst_group_block"
        ]
        <= float(
            gates["separated_byzantine_endpoint_clipped_gradient_share_per_block_max"]
        ),
        "covariance_active": summary["allocation_shape_deviation_min_heteroscedastic"]
        >= float(gates["allocation_shape_deviation_min"]),
        "global_ratio_constant": summary["allocation_ratio_range_max"]
        <= float(gates["global_allocation_ratio_range_max"]),
        "client_budget": summary["allocated_client_norm_max"]
        <= float(config["gates"]["complete_client_budget_max"]),
        "equal_cap_equivalence": summary["g0e_equal_cap_reference_difference_max"]
        <= float(gates["g0e_equal_cap_equivalence_tolerance"]),
        "diagnostic_group_observations": summary["diagnostic_group_observations_min"]
        >= int(gates["diagnostic_group_observations_min"]),
        "diagnostic_group_seed_clusters": summary["diagnostic_group_seed_clusters_min"]
        >= int(gates["diagnostic_group_seed_clusters_min"]),
        "regular_tier_minimum_clients": summary["regular_clients_per_public_tier_min"]
        >= int(gates["regular_tier_clients_per_observation_min"]),
        "stability_complete": summary["stability_complete_fraction"]
        >= float(gates["stability_complete_fraction_min"]),
        "replace_one_bound": summary["replace_one_bound_max"]
        <= float(gates["replace_one_bound_max"]),
        "replace_one_empirical": summary["replace_one_violation_count"]
        <= int(gates["empirical_replace_one_violation_max"]),
        "solver_residual": summary["solver_gradient_residual_max"]
        <= float(gates["solver_gradient_residual_max"]),
        "correction_radius": summary["correction_radius_violation_count"]
        <= int(gates["correction_radius_violation_max"]),
    }
    for name, value in checks.items():
        summary[f"gate_{name}"] = bool(value)
    summary["gate_fail_count"] = sum(not value for value in checks.values())
    summary["failed_gates"] = [name for name, value in checks.items() if not value]
    summary["passes_all_gates"] = all(checks.values())
    return summary


def _candidate_summary(
    rows: Sequence[dict[str, Any]], stability: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == name]
        stable = [row for row in stability if row["candidate"] == name]
        result.append(
            {
                "candidate": name,
                "observations": len(selected),
                "reference_error_mean": _finite_mean(
                    row["reference_error"] for row in selected
                ),
                "clean_reference_error_mean": _finite_mean(
                    row["reference_error"]
                    for row in selected
                    if row["threat"] == "none"
                ),
                "attacked_reference_error_mean": _finite_mean(
                    row["reference_error"]
                    for row in selected
                    if row["threat"] != "none"
                ),
                "replace_one_observed_max": max(
                    float(row["observed_delta"]) for row in stable
                ),
                "replace_one_theoretical_bound": max(
                    (
                        float(row["theoretical_bound"])
                        for row in stable
                        if math.isfinite(float(row["theoretical_bound"]))
                    ),
                    default=float("nan"),
                ),
            }
        )
    return result


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    development: Mapping[str, Any],
    decision: Mapping[str, Any],
    output_dir: Path,
    holdout: Mapping[str, Any] | None = None,
) -> None:
    verdict = (
        "**Arrêt au développement : aucun artefact holdout n'a été généré ou lu.**"
        if decision["holdout_status"] == "blocked_by_development_gate"
        else (
            "**G0f est promu après développement et holdout.**"
            if decision["promote"]
            else "**G0f échoue au holdout et n'est pas promu.**"
        )
    )
    lines = [
        "# G0f — allocation covariance-aware d'un budget d'influence certifié",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "G0f conserve l'estimand égal-client. La covariance publique ne pondère "
        "pas le centre quadratique : elle répartit seulement un budget total G.",
        "",
        "## Construction figée",
        "",
        "`a[i,b]` est le rayon public déployé avant cap : rayon statistique plus "
        "marge FCC figée. Avec `A=max_i ||a_i||_2`, le candidat utilise "
        "`G[i,b]=G*a[i,b]/A`; ainsi `||G_i||_2<=G`.",
        "",
        f"- G : `{derived['influence_cap_total']}` ;",
        f"- gamma : `{derived['regularization']}` ;",
        f"- K : `{derived['num_steps']}` ;",
        f"- beta : `{derived['beta_finite_solver']}` ;",
        f"- borne replace-one finite-K : `{derived['finite_solver_replace_one_bound']}` ;",
        f"- seuils calibrés : `{calibration['standardized_thresholds']}`.",
        "",
        "La borne totale est `Delta_pilot + 2*beta*G*(1-q^K)/(gamma*n)`.",
        "",
        "## Gates",
        "",
        f"- développement : `{development['passes_all_gates']}` ;",
        f"- échecs : `{development['failed_gates']}` ;",
        f"- différence hétéroscédastique vs FCC, IC95 haut : "
        f"`{development['heteroscedastic_attacked_difference_to_fcc_ci95_high']}` ;",
        f"- différence hétéroscédastique vs G0e, IC95 haut : "
        f"`{development['heteroscedastic_attacked_difference_to_g0e_ci95_high']}` ;",
        f"- rappel top-k honest outliers, pire cellule : "
        f"`{development['honest_outlier_topk_recall_worst_group']}` ;",
        f"- masse Byzantine, pire cellule séparée : "
        f"`{development['separated_byzantine_endpoint_clipped_gradient_share_worst_group']}` ;",
        "  Cette quantité est la part en norme des gradients Huber au dernier "
        "itéré non mélangé p_K ; ce n'est pas une décomposition causale de la trajectoire.",
        f"- max replace-one observé / borne : "
        f"`{development['replace_one_observed_max']}` / "
        f"`{development['replace_one_bound_max']}`.",
        "",
        "## Reproductibilité",
        "",
        f"- sorties : `{output_dir.resolve()}` ;",
        f"- seeds développement : `{config['randomness']['development_seeds']}` ;",
        f"- statut holdout : `{decision['holdout_status']}` ;",
        "- MPS float32 obligatoire, aucun fallback CPU.",
    ]
    if holdout is not None:
        lines += [
            "",
            "## Holdout préenregistré, à exécution conditionnée par le gate",
            "",
            f"- passe tous les gates : `{holdout['passes_all_gates']}` ;",
            f"- échecs : `{holdout['failed_gates']}`.",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    _validate_config(config)
    derived = _derive_parameters(config)
    if float(derived["finite_solver_replace_one_bound"]) > float(
        config["gates"]["replace_one_bound_max"]
    ):
        raise RuntimeError("Public G0f budgets violate the replace-one gate")
    if float(derived["influence_cap_total"]) > float(
        config["gates"]["complete_client_budget_max"]
    ):
        raise RuntimeError("The derived G budget exceeds its pre-registered gate")

    device, dtype = oracle._configure_runtime("mps")
    if device.type != "mps" or dtype != torch.float32:
        raise RuntimeError("G0f requires real MPS float32 execution")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"{output_dir} is not empty; pass --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    software_hashes = _software_sha256()
    runtime_versions = _runtime_versions()
    manifest_path = output_dir / "run_manifest.json"
    previous_manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preliminary_expected = {
            "campaign_id": config["campaign_id"],
            "config_sha256": fingerprint,
            "software_sha256": software_hashes,
            "runtime_versions": runtime_versions,
            "resolved_device": "mps",
            "tensor_dtype": "float32",
            "candidates": list(CANDIDATES),
        }
        for key, expected_value in preliminary_expected.items():
            if previous_manifest.get(key) != expected_value:
                raise RuntimeError(f"Refusing to resume: manifest mismatch for {key}")
    elif resume and any(output_dir.iterdir()):
        raise RuntimeError("Refusing to resume a non-empty directory without manifest")

    # Calibration is deterministic and recomputed from fresh calibration
    # identities.  Its canonical hash is checked before any checkpoint can be
    # reused, preventing a mixed-calibration resume.
    calibration, calibration_rows = _calibrate(config, derived)
    derived_hash = _canonical_sha256(derived)
    calibration_hash = _canonical_sha256(calibration)
    calibration_rows_hash = _canonical_sha256(calibration_rows)
    complete_manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": fingerprint,
        "software_sha256": software_hashes,
        "runtime_versions": runtime_versions,
        "derived_parameters_sha256": derived_hash,
        "calibration_artifact_sha256": calibration_hash,
        "calibration_probes_sha256": calibration_rows_hash,
        "resolved_device": "mps",
        "tensor_dtype": "float32",
        "candidates": list(CANDIDATES),
    }
    if previous_manifest is not None:
        for key in (
            "derived_parameters_sha256",
            "calibration_artifact_sha256",
            "calibration_probes_sha256",
        ):
            if previous_manifest.get(key) != complete_manifest[key]:
                raise RuntimeError(f"Refusing to resume: manifest mismatch for {key}")
        _assert_manifest_matches(
            manifest_path, complete_manifest, stage="before checkpoint reuse"
        )
    else:
        _atomic_json(manifest_path, complete_manifest)

    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _atomic_json(output_dir / "derived_parameters.json", derived)
    _atomic_json(output_dir / "calibration_artifact.json", calibration)
    _write_csv(output_dir / "calibration_probes.csv", calibration_rows)
    checkpoint_dir = output_dir / "_checkpoints"
    development_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
        draws_per_seed=int(config["randomness"]["development_draws_per_seed"]),
        severities=config["threats"]["development_severities"],
    )
    development_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
    )
    development = _summarize(
        development_rows, development_stability, config, "development"
    )
    _write_csv(output_dir / "development_detail.csv", development_rows)
    _write_csv(output_dir / "development_stability.csv", development_stability)
    _write_csv(output_dir / "development_summary.csv", [development])
    _write_csv(
        output_dir / "development_candidates.csv",
        _candidate_summary(development_rows, development_stability),
    )

    if not bool(development["passes_all_gates"]):
        stale = _holdout_artifacts(output_dir)
        if stale:
            raise RuntimeError(
                "Development failed but holdout artifacts exist: "
                + ", ".join(str(path.relative_to(output_dir)) for path in stale)
            )
        decision = {
            "campaign_id": config["campaign_id"],
            "development_passes": False,
            "development_gate_fail_count": int(development["gate_fail_count"]),
            "failed_development_gates": development["failed_gates"],
            "holdout_status": "blocked_by_development_gate",
            "holdout_passes": None,
            "promote": False,
            "resolved_device": "mps",
            "tensor_dtype": "float32",
        }
        _atomic_json(output_dir / "decision.json", decision)
        _write_report(
            report_path,
            config=config,
            derived=derived,
            calibration=calibration,
            development=development,
            decision=decision,
            output_dir=output_dir,
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return decision

    # Re-fingerprint sources immediately before opening the separately
    # committed holdout registry.  This catches edits made during a long
    # development run and prevents mixed-code holdout execution.
    pre_holdout_manifest = dict(complete_manifest)
    pre_holdout_manifest["software_sha256"] = _software_sha256()
    pre_holdout_manifest["runtime_versions"] = _runtime_versions()
    _assert_manifest_matches(
        manifest_path, pre_holdout_manifest, stage="before holdout registry access"
    )
    holdout_seeds = _load_execution_gated_holdout_seeds(config)
    holdout_config = copy.deepcopy(config)
    holdout_config["randomness"]["holdout_seeds"] = holdout_seeds
    holdout_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=holdout_config,
        derived=derived,
        calibration=calibration,
        phase="holdout",
        seeds=holdout_seeds,
        draws_per_seed=int(config["randomness"]["holdout_draws_per_seed"]),
        severities=config["threats"]["holdout_severities"],
    )
    holdout_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=holdout_config,
        derived=derived,
        calibration=calibration,
        phase="holdout",
        seeds=holdout_seeds,
    )
    holdout = _summarize(holdout_rows, holdout_stability, holdout_config, "holdout")
    _write_csv(output_dir / "holdout_detail.csv", holdout_rows)
    _write_csv(output_dir / "holdout_stability.csv", holdout_stability)
    _write_csv(output_dir / "holdout_summary.csv", [holdout])
    _write_csv(
        output_dir / "holdout_candidates.csv",
        _candidate_summary(holdout_rows, holdout_stability),
    )
    decision = {
        "campaign_id": config["campaign_id"],
        "development_passes": True,
        "development_gate_fail_count": 0,
        "holdout_status": "executed_after_development_pass",
        "holdout_passes": bool(holdout["passes_all_gates"]),
        "holdout_gate_fail_count": int(holdout["gate_fail_count"]),
        "failed_holdout_gates": holdout["failed_gates"],
        "promote": bool(holdout["passes_all_gates"]),
        "resolved_device": "mps",
        "tensor_dtype": "float32",
    }
    _atomic_json(output_dir / "decision.json", decision)
    _write_report(
        report_path,
        config=config,
        derived=derived,
        calibration=calibration,
        development=development,
        holdout=holdout,
        decision=decision,
        output_dir=output_dir,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0f.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0f_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_Robust_Reference_G0f_MPS.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
