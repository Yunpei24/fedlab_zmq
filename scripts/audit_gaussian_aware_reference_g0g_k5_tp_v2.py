#!/usr/bin/env python3
"""Independent fail-closed audit for the technical G0g-K5-TP-v2 rerun.

The audit intentionally does not import the K5-v2 runner or its algorithm
module.  It rebuilds the effective configuration, child-seed registries,
ridge solutions, lambda decisions and (when present) evaluation estimands
from persisted artifacts.  It also verifies that the first v2 attempt stopped
before data generation.  Production numerical reconstruction is MPS float32
only; unit tests may exercise the pure helpers on CPU.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/ldp_gradient_far/k5_v2/gaussian_aware_reference_g0g_k5_tp_v2.yaml"
)
DEFAULT_LOCK = (
    ROOT / "configs/ldp_gradient_far/k5_v2/"
    "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2.lock.json"
)
DEFAULT_RESULTS = (
    ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
V1_RESULTS = (
    ROOT
    / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)
TECHNICAL_V2_MPS_V1_RESULTS = (
    ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v1"
)

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K5_1D = "g0g_k5_tp_v2_one_dimensional_control"
K5 = "g0g_k5_tp_v2_five_feature_ridge"
K4C = "g0g_k4c_ch_privileged_benchmark"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, K5_1D, K5, K4C, POINTWISE)
FEATURE_NAMES = (
    "rolling_accepted_mean",
    "accepted_mean_t_minus_1",
    "accepted_mean_delta_t_minus_1_t_minus_2",
    "fixed_denominator_delta_t_minus_1_t_minus_2",
    "rolling_fixed_denominator_direction",
)
FIT_STREAMS = ("train_target", "calibration_target", "calibration_evaluation")
FORBIDDEN_FIT_STREAMS = ("evaluation_target", "evaluation", "holdout")
RNG_STREAM_ORDER = FIT_STREAMS + FORBIDDEN_FIT_STREAMS
FIT_AUDITED_ARTIFACT_NAMES = (
    "frozen_predictor.json",
    "fit_decision.json",
    "fit_sufficient_statistics.json",
    "fit_rng_registry.json",
    "manifest.json",
)
PUBLICATION_REQUIREMENT = (
    "publish_lock_sha256_before_fit_and_frozen_predictor_sha256_and_"
    "independent_fit_audit_sha256_before_evaluation"
)
LAMBDA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
REMOVED_V1_FEATURE = "fixed_denominator_direction_t_minus_1"
TRAIN_SEEDS = (
    2027043003,
    2027043017,
    2027043029,
    2027043041,
    2027043053,
    2027043067,
    2027043079,
    2027043091,
    2027043107,
    2027043121,
    2027043133,
    2027043149,
)
CALIBRATION_SEEDS = (
    2027051001,
    2027051017,
    2027051033,
    2027051049,
    2027051065,
    2027051081,
    2027051097,
    2027051113,
)
EVALUATION_SEEDS = (
    2027050201,
    2027050217,
    2027050233,
    2027050249,
    2027050265,
    2027050281,
    2027050297,
    2027050313,
    2027050329,
    2027050345,
    2027050361,
    2027050377,
)
HOLDOUT_SEEDS = (
    2027044009,
    2027044021,
    2027044033,
    2027044047,
    2027044059,
    2027044071,
    2027044087,
)

PREDICTOR_SCHEMA = {
    "schema_version",
    "campaign_id",
    "candidate",
    "feature_names",
    "removed_v1_feature",
    "feature_normalization_formula",
    "feature_scales",
    "coefficients",
    "selected_lambda",
    "one_dimensional_control",
    "influence_cap",
    "public_cohort_size",
    "fit_split",
    "lambda_selection_coefficients_fit_on",
    "observable_past_only_at_inference",
    "privileged_training_supervision",
    "amendment_config_sha256",
    "base_config_sha256",
    "resolved_config_sha256",
    "lock_sha256",
    "fit_rng_registry_structured_sha256",
    "v1_provenance_sha256",
    "evaluation_generated_before_freeze",
    "holdout_opened",
}

LOCKED_PATHS = {
    "algorithms/gaussian_aware_reference_k5_tp_v2.py",
    "configs/ldp_gradient_far/k5_v2/gaussian_aware_reference_g0g_k5_tp_v2.yaml",
    "output/analysis/Gaussian_Aware_G0g_K5_TP_V2_Technical_Rerun_Addendum.md",
    "scripts/audit_gaussian_aware_reference_g0g_k5_tp_v2.py",
    "scripts/run_gaussian_aware_reference_g0g_k5_tp_v2.py",
    "tests/test_audit_gaussian_aware_reference_g0g_k5_tp_v2.py",
    "tests/test_gaussian_aware_reference_g0g_k5_tp_v2.py",
    "tests/test_run_gaussian_aware_reference_g0g_k5_tp_v2.py",
}
DEPENDENCY_PATHS = {
    "algorithms/gaussian_aware_reference.py",
    "algorithms/gaussian_aware_reference_k4b.py",
    "algorithms/gaussian_aware_reference_k4c_ch.py",
    "algorithms/gaussian_aware_reference_k5_tp.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor.yaml",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor.lock.json",
    "configs/ldp_gradient_far/k5_v2/gaussian_aware_reference_g0g_k5_tp_v2.lock.json",
    "output/analysis/Gaussian_Aware_G0g_K5_TP_V2_Protocol_PreRun.md",
    "robustness/aggregators.py",
    "scripts/run_gaussian_aware_reference_g0g_k1.py",
    "scripts/run_gaussian_aware_reference_g0g_k2.py",
    "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
    "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "scripts/run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "scripts/run_gaussian_aware_reference_g0g_k5_transcript_predictor.py",
    "scripts/run_gaussian_aware_reference_oracle.py",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1/calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1/temporal_calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/manifest.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/fit_decision.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/feature_design_diagnostics.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/fit_rng_registry.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/frozen_predictor.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/lambda_calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/fit_history_rows.csv",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1/independent_fit_audit.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v1/manifest.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v1/resolved_config.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_artifact_hash_inventory(results: Path) -> dict[str, str | None]:
    """Return the complete five-name inventory, using null for missing files."""

    return {
        name: _sha256(results / name) if (results / name).is_file() else None
        for name in FIT_AUDITED_ARTIFACT_NAMES
    }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be canonically hashed")
    return value


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialized_json_hash(value: Any) -> str:
    """Hash the exact canonical artifact representation used by the runner."""

    encoded = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if (
                not fieldnames
                or any(not str(name).strip() for name in fieldnames)
                or len(fieldnames) != len(set(fieldnames))
            ):
                raise ValueError(f"CSV header is empty, duplicated or invalid: {path}")
            rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV artifact: {path}") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV artifact is not rectangular: {path}")
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: Any, right: Any, tolerance: float = 2e-6) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    )


def _device_type(value: Any) -> str | None:
    try:
        return torch.device(str(value)).type
    except (RuntimeError, TypeError):
        return None


def _flatten_mapping(
    value: Any, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        result: dict[tuple[str, ...], Any] = {}
        for key, item in value.items():
            result.update(_flatten_mapping(item, (*prefix, str(key))))
        return result
    return {prefix: value}


def _resolved_delta_paths(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> set[tuple[str, ...]]:
    old = _flatten_mapping(before)
    new = _flatten_mapping(after)
    sentinel = object()
    return {
        key
        for key in set(old) | set(new)
        if old.get(key, sentinel) != new.get(key, sentinel)
    }


def _load_config_bundle(config_path: Path) -> dict[str, Any]:
    """Independently rebuild the effective v2 config and its three hashes."""

    amendment = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(amendment, Mapping)
        or amendment.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("invalid K5-v2 amendment config")
    preregistration = amendment.get("preregistration_lock", {})
    if (
        preregistration.get("publication_requirement") != PUBLICATION_REQUIREMENT
        or preregistration.get("publish_lock_sha256_before_fit") is not True
        or preregistration.get("evaluation_requires_published_frozen_predictor_sha256")
        is not True
        or preregistration.get("evaluation_requires_published_fit_audit_sha256")
        is not True
    ):
        raise ValueError("K5-v2 publication requirement changed")
    base_path = ROOT / str(amendment["base_protocol"]["config_path"])
    base_hash = _sha256(base_path)
    if base_hash != str(amendment["base_protocol"]["config_sha256"]):
        raise RuntimeError("K5-v2 base config hash mismatch")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    resolved = copy.deepcopy(base)
    resolved["campaign_id"] = CAMPAIGN_ID
    resolved["scope"] = amendment["scope"]
    resolved["scientific_contract"]["inference_predictor_name"] = K5
    resolved["features"]["order"] = list(amendment["features"]["order"])
    resolved["gates"]["feature_count_exact"] = int(
        amendment["features"]["feature_count_exact"]
    )
    resolved["gates"]["flattened_feature_rank_exact"] = int(
        amendment["features"]["flattened_feature_rank_exact"]
    )
    for key in (
        "train_outer_seeds",
        "calibration_outer_seeds",
        "evaluation_outer_seeds",
        "reserved_holdout_seeds",
    ):
        resolved["randomness"][key] = list(amendment["randomness"][key])
    resolved["randomness"]["train_seed_role"] = "reuse_v1_training_outer_seeds"
    resolved["randomness"][
        "calibration_and_evaluation_seed_rule"
    ] = "new_calibration_seeds_and_exact_unopened_v1_evaluation_seeds"
    for key in (
        "train_target_stream_tag",
        "calibration_target_stream_tag",
        "calibration_evaluation_stream_tag",
        "evaluation_target_stream_tag",
        "evaluation_stream_tag",
    ):
        resolved["nested_monte_carlo"][key] = amendment["nested_monte_carlo_streams"][
            key
        ]
    resolved["preregistration_lock"] = {
        "path": amendment["preregistration_lock"]["path"],
        "verify_before_output_creation": True,
        "publish_lock_sha256_before_fit": True,
        "external_publication_is_procedural_precondition": True,
    }
    resolved["candidates"]["names"] = list(CANDIDATES)
    resolved["candidates"]["primary"] = K5
    resolved["candidates"]["learned_one_dimensional_control"] = K5_1D
    resolved["execution"]["device_identity_check"] = "torch_device_type_equals_mps"
    resolved["execution"]["output_directory"] = amendment["execution"][
        "output_directory"
    ]
    if tuple(resolved["features"]["order"]) != FEATURE_NAMES:
        raise ValueError("K5-v2 feature schema mismatch")
    if tuple(float(value) for value in resolved["ridge"]["lambda_grid"]) != LAMBDA_GRID:
        raise ValueError("K5-v2 lambda grid mismatch")
    observed_seed_sets = (
        tuple(int(value) for value in resolved["randomness"]["train_outer_seeds"]),
        tuple(
            int(value) for value in resolved["randomness"]["calibration_outer_seeds"]
        ),
        tuple(int(value) for value in resolved["randomness"]["evaluation_outer_seeds"]),
        tuple(int(value) for value in resolved["randomness"]["reserved_holdout_seeds"]),
    )
    if observed_seed_sets != (
        TRAIN_SEEDS,
        CALIBRATION_SEEDS,
        EVALUATION_SEEDS,
        HOLDOUT_SEEDS,
    ) or any(
        set(observed_seed_sets[left]) & set(observed_seed_sets[right])
        for left in range(4)
        for right in range(left + 1, 4)
    ):
        raise ValueError("K5-v2 outer-seed registries changed or overlap")
    if any(
        resolved["gates"].get(key) != value or base["gates"].get(key) != value
        for key, value in amendment["scientific_gates_must_equal_v1"].items()
    ):
        raise ValueError("K5-v2 scientific gates differ from v1")
    if resolved["execution"].get("output_directory") != str(
        DEFAULT_RESULTS.relative_to(ROOT)
    ):
        raise ValueError("K5-v2 resolved output directory changed")
    return {
        "amendment": amendment,
        "base": base,
        "resolved": resolved,
        "base_path": base_path,
        "hashes": {
            "amendment_config_sha256": _sha256(config_path),
            "base_config_sha256": base_hash,
            "resolved_config_sha256": _serialized_json_hash(resolved),
        },
    }


def _verify_lock(lock_path: Path, config_path: Path) -> dict[str, Any]:
    if (
        lock_path.resolve() != DEFAULT_LOCK.resolve()
        or config_path.resolve() != DEFAULT_CONFIG.resolve()
    ):
        return {"pass": False, "error": "non_preregistered_path"}
    try:
        lock = _read_json(lock_path)
    except (OSError, ValueError) as exc:
        return {"pass": False, "error": f"lock_unreadable:{exc}"}
    checks = {
        "schema": set(lock)
        == {
            "schema_version",
            "campaign_id",
            "locked_files",
            "dependencies",
            "lock_file_self_hash_embedded",
            "publication_requirement",
        }
        and lock.get("schema_version") == 1
        and lock.get("campaign_id") == CAMPAIGN_ID,
        "inventory": set(lock.get("locked_files", {})) == LOCKED_PATHS
        and set(lock.get("dependencies", {})) == DEPENDENCY_PATHS,
        "no_self_hash": lock.get("lock_file_self_hash_embedded") is False,
        "publication_contract": lock.get("publication_requirement")
        == PUBLICATION_REQUIREMENT,
    }
    mismatches: list[str] = []
    for section in ("locked_files", "dependencies"):
        for relative, expected in lock.get(section, {}).items():
            path = ROOT / str(relative)
            if not path.is_file() or _sha256(path) != str(expected):
                mismatches.append(str(relative))
    checks["file_hashes"] = not mismatches
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "mismatches": mismatches,
        "lock_sha256": _sha256(lock_path),
    }


def _audit_v1_provenance(amendment: Mapping[str, Any]) -> dict[str, Any]:
    declared = amendment["v1_pre_evaluation_provenance"]
    hashes_ok = _sha256(ROOT / declared["lock_path"]) == declared["lock_sha256"]
    mismatches: list[str] = []
    for name, expected in declared["required_artifact_sha256"].items():
        path = V1_RESULTS / str(name)
        if not path.is_file() or _sha256(path) != str(expected):
            mismatches.append(str(name))
    manifest = _read_json(V1_RESULTS / "manifest.json")
    decision = _read_json(V1_RESULTS / "fit_decision.json")
    design = _read_json(V1_RESULTS / "feature_design_diagnostics.json")
    registry = _read_json(V1_RESULTS / "fit_rng_registry.json")
    predictor = _read_json(V1_RESULTS / "frozen_predictor.json")
    false_checks = {
        key for key, value in decision["checks"].items() if value is not True
    }
    fit_rng = [int(seed) for stream in FIT_STREAMS for seed in registry[stream]]
    checks = {
        "declared_hashes": hashes_ok and not mismatches,
        "invalid_fit_exact": manifest.get("status")
        == "fit_invalid_evaluation_forbidden"
        and decision.get("decision") == "invalid_fit_evaluation_forbidden"
        and false_checks == set(declared["expected_v1_invalid_checks"]),
        "rank_five": design["train"]["flattened_design_rank"] == 5
        and design["train_plus_calibration"]["flattened_design_rank"] == 5,
        "mps_semantic": _device_type(design["final_fit"]["fit_device"]) == "mps",
        "evaluation_absent": manifest.get("evaluation_trajectory_count_generated") == 0
        and registry.get("evaluation_target") == []
        and registry.get("evaluation") == []
        and not (V1_RESULTS / "evaluation").exists(),
        "holdout_closed": manifest.get("holdout_opened") is False
        and registry.get("holdout") == [],
        "rng_unique_and_hashed": len(fit_rng) == len(set(fit_rng))
        and _canonical_hash(fit_rng) == predictor.get("fit_rng_registry_sha256"),
    }
    return {"pass": all(checks.values()), "checks": checks, "mismatches": mismatches}


def _audit_technical_v2_mps_v1_failure_provenance(
    amendment: Mapping[str, Any], resolved_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit the data-free boundary; keep the reported cause non-machine-verified."""

    declared = amendment["technical_v2_mps_v1_failure_provenance"]
    expected = {
        "campaign_id": "gaussian_aware_reference_g0g_k5_tp_v2_mps_v1",
        "lock_path": (
            "configs/ldp_gradient_far/k5_v2/"
            "gaussian_aware_reference_g0g_k5_tp_v2.lock.json"
        ),
        "lock_sha256": (
            "e96669612ff7aec599f6199bd841ca16a5a3aa8cb1526551ea6e2b810cc330e2"
        ),
        "locked_runner_sha256": (
            "ee3e4fe2b17fb0fc7c9fca24f2c32f5414f534477607801338235da6de6c62ab"
        ),
        "results_directory": (
            "results/ldp_gradient_far/" "gaussian_aware_reference_g0g_k5_tp_v2_mps_v1"
        ),
        "manifest_sha256": (
            "26a310935e01bf4ded252c7a700b3f47780276b8c6f03d8d04ac6af1cfa98415"
        ),
        "resolved_config_sha256": (
            "3bd9d41522551f2a4cb6ef9c6ad6683ad9adbc6fea79839c536a897f2e42c215"
        ),
        "artifact_inventory_exact": ["manifest.json", "resolved_config.json"],
        "manifest_status": "fit_running_evaluation_forbidden",
        "evaluation_trajectory_count_exact": 0,
        "fit_history_row_count_exact": 0,
        "holdout_opened": False,
        "failure_boundary": "after_manifest_before_any_fit_or_evaluation_data",
        "failure_boundary_machine_verified": True,
        "failure_exception_attested": "AttributeError",
        "missing_symbol_attested": "v1._read_csv",
        "persistent_traceback_available": False,
        "failure_cause_attested_not_machine_verified": True,
        "failure_cause_is_scientific_gate": False,
        "cause_attestation_source": (
            "contemporaneous_runtime_observation_and_code_review"
        ),
        "scientific_design_changed_before_rerun": False,
        "train_seeds_changed_before_rerun": False,
        "calibration_seeds_changed_before_rerun": False,
        "evaluation_seeds_changed_before_rerun": False,
        "scientific_gates_changed_before_rerun": False,
    }
    cause_keys = {
        "failure_exception_attested",
        "missing_symbol_attested",
        "persistent_traceback_available",
        "failure_cause_attested_not_machine_verified",
        "failure_cause_is_scientific_gate",
        "cause_attestation_source",
    }
    documentation_only_keys = {
        "scientific_design_changed_before_rerun",
        "train_seeds_changed_before_rerun",
        "calibration_seeds_changed_before_rerun",
        "evaluation_seeds_changed_before_rerun",
        "scientific_gates_changed_before_rerun",
    }
    boundary_keys = set(expected) - cause_keys - documentation_only_keys
    boundary_declaration_exact = all(
        declared.get(key) == expected[key] for key in boundary_keys
    )
    cause_attestation_schema_exact = all(
        declared.get(key) == expected[key] for key in cause_keys
    )
    lock_path = ROOT / str(declared["lock_path"])
    manifest_path = TECHNICAL_V2_MPS_V1_RESULTS / "manifest.json"
    resolved_path = TECHNICAL_V2_MPS_V1_RESULTS / "resolved_config.json"
    old_lock = _read_json(lock_path)
    manifest = _read_json(manifest_path)
    resolved = _read_json(resolved_path)
    expected_resolved_delta = {
        ("campaign_id",),
        ("execution", "output_directory"),
        ("preregistration_lock", "path"),
    }
    resolved_delta = _resolved_delta_paths(resolved, resolved_config)
    inventory = sorted(
        str(path.relative_to(TECHNICAL_V2_MPS_V1_RESULTS))
        for path in TECHNICAL_V2_MPS_V1_RESULTS.rglob("*")
        if path.is_file()
    )
    checks = {
        "boundary_declaration_exact": boundary_declaration_exact,
        "old_lock_sha256_exact": _sha256(lock_path) == declared["lock_sha256"],
        "old_lock_campaign_exact": old_lock.get("campaign_id")
        == declared["campaign_id"],
        "old_locked_runner_identity_exact": old_lock.get("locked_files", {}).get(
            "scripts/run_gaussian_aware_reference_g0g_k5_tp_v2.py"
        )
        == declared["locked_runner_sha256"],
        "artifact_hashes_exact": _sha256(manifest_path) == declared["manifest_sha256"]
        and _sha256(resolved_path) == declared["resolved_config_sha256"],
        "artifact_inventory_exact": inventory == declared["artifact_inventory_exact"],
        "manifest_boundary_exact": manifest.get("campaign_id")
        == declared["campaign_id"]
        and manifest.get("status") == declared["manifest_status"]
        and int(manifest.get("evaluation_trajectory_count_generated", -1))
        == int(declared["evaluation_trajectory_count_exact"])
        and manifest.get("holdout_opened") is declared["holdout_opened"],
        "resolved_identity_exact": resolved.get("campaign_id")
        == declared["campaign_id"]
        and manifest.get("resolved_config_sha256")
        == declared["resolved_config_sha256"],
        "zero_fit_rows_and_no_evaluation": len(
            list(TECHNICAL_V2_MPS_V1_RESULTS.rglob("*.csv"))
        )
        == int(declared["fit_history_row_count_exact"])
        == 0
        and not (TECHNICAL_V2_MPS_V1_RESULTS / "evaluation").exists(),
        "resolved_config_delta_exact": resolved_delta == expected_resolved_delta,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "resolved_config_delta": [".".join(path) for path in sorted(resolved_delta)],
        "failure_boundary_machine_verified": all(checks.values()),
        "failure_cause_attested_not_machine_verified": bool(
            declared.get("failure_cause_attested_not_machine_verified")
        ),
        "failure_cause_is_scientific_gate": bool(
            declared.get("failure_cause_is_scientific_gate")
        ),
        "failure_cause_attestation": {
            "exception": declared.get("failure_exception_attested"),
            "missing_symbol": declared.get("missing_symbol_attested"),
            "persistent_traceback_available": bool(
                declared.get("persistent_traceback_available")
            ),
            "source": declared.get("cause_attestation_source"),
            "metadata_schema_exact": cause_attestation_schema_exact,
            "contributes_to_machine_provenance_pass": False,
        },
        "fit_history_row_count": 0,
        "evaluation_trajectory_count": 0,
        "holdout_opened": False,
    }


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(regime["name"]), str(permutation))
        for regime in config["privacy_noise"]["regimes"]
        for permutation in regime["permutations"]
    ]


def _history_records(
    config: Mapping[str, Any], split: str
) -> list[tuple[str, int, str, str, str, str, str, int]]:
    seed_key = {
        "train": "train_outer_seeds",
        "calibration": "calibration_outer_seeds",
        "evaluation": "evaluation_outer_seeds",
    }[split]
    records: list[tuple[str, int, str, str, str, str, str, int]] = []
    for seed in config["randomness"][seed_key]:
        for regime, permutation in _noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for dynamics in config["honest_dynamics"]["names"]:
                    for threat in config["threats"]["names"]:
                        for round_index in config["temporal"]["assessment_rounds"]:
                            parts = (
                                split,
                                int(seed),
                                regime,
                                permutation,
                                str(geometry),
                                str(dynamics),
                                str(threat),
                                int(round_index),
                            )
                            history_id = "|".join(str(value) for value in parts)
                            records.append((history_id, *parts[1:]))
    return records


def _stable_seed(*parts: object) -> int:
    """Independent copy of the preregistered SHA-256 seed derivation."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _expected_rng_stream_records(
    config: Mapping[str, Any], *, split: str, stream: str, children: int
) -> list[dict[str, Any]]:
    tag = str(config["nested_monte_carlo"][f"{stream}_stream_tag"])
    return [
        {
            "history_id": history_id,
            "stream": stream,
            "stream_tag": tag,
            "outer_seed": seed,
            "noise_regime": regime,
            "noise_permutation": permutation,
            "outlier_geometry": geometry,
            "honest_dynamics": dynamics,
            "threat": threat,
            "assessment_round": round_index,
            "child_seeds": [
                _stable_seed(
                    tag,
                    seed,
                    regime,
                    permutation,
                    geometry,
                    dynamics,
                    threat,
                    round_index,
                    child,
                )
                for child in range(children)
            ],
        }
        for (
            history_id,
            seed,
            regime,
            permutation,
            geometry,
            dynamics,
            threat,
            round_index,
        ) in _history_records(config, split)
    ]


def _flatten_rng_records(records: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(seed) for record in records for seed in record.get("child_seeds", ())]


def _rng_registry(
    streams: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stream_order": list(RNG_STREAM_ORDER),
        "streams": {
            key: [dict(record) for record in streams.get(key, ())]
            for key in RNG_STREAM_ORDER
        },
    }
    return {**payload, "payload_sha256": _canonical_hash(payload)}


def _expected_fit_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config["nested_monte_carlo"]
    streams = {
        "train_target": _expected_rng_stream_records(
            config,
            split="train",
            stream="train_target",
            children=int(nested["train_target_construction_children"]),
        ),
        "calibration_target": _expected_rng_stream_records(
            config,
            split="calibration",
            stream="calibration_target",
            children=int(nested["calibration_target_construction_children"]),
        ),
        "calibration_evaluation": _expected_rng_stream_records(
            config,
            split="calibration",
            stream="calibration_evaluation",
            children=int(nested["calibration_evaluation_children"]),
        ),
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    return _rng_registry(streams)


def _select_lambda(
    rows: Sequence[Mapping[str, Any]], tolerance: float
) -> tuple[float, dict[str, Any]]:
    if len(rows) != len(LAMBDA_GRID):
        raise ValueError("lambda table does not contain eight rows")
    parsed: list[tuple[float, float]] = []
    for row in rows:
        values = [float(value) for value in row["calibration_seed_mse"]]
        if len(values) != 8 or not all(math.isfinite(value) for value in values):
            raise ValueError("each lambda needs eight finite seed-level MSEs")
        score = statistics.fmean(values)
        if not _close(score, row["equal_seed_mean_projected_aggregate_mse"], 1e-10):
            raise ValueError("calibration score is not the equal-seed mean")
        parsed.append((float(row["ridge_lambda"]), score))
    if {value for value, _ in parsed} != set(LAMBDA_GRID):
        raise ValueError("lambda table differs from the preregistered grid")
    minimum = min(score for _, score in parsed)
    selected = max(value for value, score in parsed if score <= minimum + tolerance)
    return selected, {"scores": dict(parsed), "minimum": minimum}


def _solve_sufficient_statistics(
    diagnostic: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    gram = torch.tensor(
        diagnostic["normalized_gram"], dtype=torch.float32, device=device
    )
    rhs = torch.tensor(diagnostic["normalized_rhs"], dtype=torch.float32, device=device)
    if (
        gram.ndim != 2
        or gram.shape[0] != gram.shape[1]
        or rhs.shape != (gram.shape[0],)
    ):
        raise ValueError("invalid ridge sufficient-statistic shapes")
    ridge_lambda = float(diagnostic["ridge_lambda"])
    system = gram + ridge_lambda * torch.eye(
        int(gram.shape[0]), dtype=torch.float32, device=device
    )
    coefficients = torch.linalg.solve(system, rhs)
    inverse = torch.linalg.solve(
        system,
        torch.eye(system.shape[0], dtype=torch.float32, device=device),
    )
    condition = float(
        (
            torch.max(torch.sum(torch.abs(system), dim=1))
            * torch.max(torch.sum(torch.abs(inverse), dim=1))
        ).item()
    )
    relative_residual = float(
        (
            torch.linalg.vector_norm(system @ coefficients - rhs)
            / torch.clamp(
                torch.linalg.vector_norm(rhs), min=torch.finfo(torch.float32).eps
            )
        ).item()
    )
    return {
        "coefficients": coefficients,
        "condition_number": condition,
        "normal_equation_relative_residual": relative_residual,
    }


def _audit_ridge_solution(
    diagnostic: Mapping[str, Any],
    *,
    expected_count: int,
    expected_lambda: float,
    stored_coefficients: Sequence[float],
    expected_feature_names: Sequence[str],
    device: torch.device,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        stored = [float(value) for value in stored_coefficients]
        solved = _solve_sufficient_statistics(diagnostic, device)
        actual = solved["coefficients"]
        if len(stored) != expected_count or not all(math.isfinite(v) for v in stored):
            errors.append("coefficient_schema")
        elif not torch.allclose(
            actual,
            torch.tensor(stored, dtype=torch.float32, device=device),
            atol=2e-6,
            rtol=2e-6,
        ):
            errors.append("coefficient_reproduction")
        if not math.isclose(
            float(diagnostic["ridge_lambda"]),
            expected_lambda,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append("lambda")
        if tuple(diagnostic.get("feature_names", ())) != tuple(expected_feature_names):
            errors.append("feature_names")
        if _device_type(diagnostic.get("fit_device")) != "mps":
            errors.append("device")
        if diagnostic.get("fit_dtype") != "torch.float32":
            errors.append("dtype")
        condition = solved["condition_number"]
        residual = solved["normal_equation_relative_residual"]
        if not math.isfinite(condition) or condition > float(
            gates["regularized_system_condition_number_max"]
        ):
            errors.append("condition_bound")
        if not math.isfinite(residual) or residual > float(
            gates["normal_equation_relative_residual_max"]
        ):
            errors.append("residual_bound")
        if not _close(
            condition, diagnostic["condition_number_regularized_system"], 2e-4
        ):
            errors.append("condition_reproduction")
        if not _close(residual, diagnostic["normal_equation_relative_residual"], 2e-4):
            errors.append("residual_reproduction")
        for key in ("objective", "weighted_mse_before_projection", "ridge_penalty"):
            if key in diagnostic and not _finite_number(diagnostic[key]):
                errors.append(f"nonfinite_{key}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errors.append(f"exception:{exc}")
    return {"pass": not errors, "errors": errors}


def _grid_mapping(value: Any) -> dict[float, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("ridge grid diagnostics must be an object")
    result: dict[float, Mapping[str, Any]] = {}
    for raw, diagnostic in value.items():
        numeric = float(raw)
        if numeric in result:
            raise ValueError("duplicate numeric lambda key")
        if not isinstance(diagnostic, Mapping):
            raise ValueError("ridge diagnostic must be an object")
        result[numeric] = diagnostic
    if set(result) != set(LAMBDA_GRID):
        raise ValueError("ridge diagnostics grid mismatch")
    return result


def _phase_boundary_ok(
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    predictor: Mapping[str, Any],
    *,
    evaluation_directory_exists: bool,
) -> bool:
    return (
        manifest.get("status") == "fit_completed_evaluation_locked"
        and manifest.get("fit_validity_pass") is True
        and decision.get("all_validity_checks_pass") is True
        and isinstance(decision.get("checks"), Mapping)
        and bool(decision["checks"])
        and all(value is True for value in decision["checks"].values())
        and manifest.get("evaluation_trajectory_count_generated") == 0
        and decision.get("evaluation_trajectory_count_generated") == 0
        and manifest.get("holdout_opened") is False
        and decision.get("holdout_opened") is False
        and predictor.get("evaluation_generated_before_freeze") is False
        and predictor.get("holdout_opened") is False
        and not evaluation_directory_exists
    )


def _post_evaluation_boundary_ok(
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    predictor: Mapping[str, Any],
) -> bool:
    """Verify that evaluation only advanced the root phase metadata."""

    return bool(
        manifest.get("status") == "completed_development"
        and manifest.get("fit_validity_pass") is True
        and manifest.get("evaluation_trajectory_count_generated") == 576
        and decision.get("all_validity_checks_pass") is True
        and isinstance(decision.get("checks"), Mapping)
        and bool(decision["checks"])
        and all(value is True for value in decision["checks"].values())
        and decision.get("evaluation_trajectory_count_generated") == 0
        and manifest.get("holdout_opened") is False
        and decision.get("holdout_opened") is False
        and predictor.get("evaluation_generated_before_freeze") is False
        and predictor.get("holdout_opened") is False
    )


def _artifact_hash_check(
    artifact: Mapping[str, Any], expected: Mapping[str, str]
) -> bool:
    nested = artifact.get("config_hashes")
    return all(
        artifact.get(key) == value
        or (isinstance(nested, Mapping) and nested.get(key) == value)
        for key, value in expected.items()
    )


def _sufficient_statistics_consistent(
    statistics_artifact: Mapping[str, Any],
    calibration: Mapping[str, Any],
    design: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    """Cross-link the separately persisted matrices to all 18 ridge solves."""

    try:
        if set(statistics_artifact) != {
            "schema_version",
            "normalization",
            "train",
            "train_plus_calibration",
            "one_dimensional",
            "lambda_grid",
            "selected_lambda",
            "selected_lambda_one_dimensional",
            "calibration_primary_seed_mse_by_lambda",
            "calibration_one_dimensional_seed_mse_by_lambda",
        }:
            return False
        if (
            statistics_artifact["schema_version"] != 1
            or statistics_artifact["normalization"]
            != "dimension_times_sum_seed_balanced_weights"
            or tuple(float(value) for value in statistics_artifact["lambda_grid"])
            != LAMBDA_GRID
            or statistics_artifact["calibration_primary_seed_mse_by_lambda"]
            != calibration["primary"]
            or statistics_artifact["calibration_one_dimensional_seed_mse_by_lambda"]
            != calibration["one_dimensional"]
        ):
            return False
        train = statistics_artifact["train"]
        refit = statistics_artifact["train_plus_calibration"]
        one = statistics_artifact["one_dimensional"]
        if not (
            train["histories"] == 576
            and train["dimension"] == int(config["cohort"]["dimension"])
            and train["feature_count"] == 5
            and _close(train["aggregate_weight_sum"], 12.0, 1e-6)
            and refit["histories"] == 960
            and refit["dimension"] == int(config["cohort"]["dimension"])
            and refit["feature_count"] == 5
            and _close(refit["aggregate_weight_sum"], 20.0, 1e-6)
            and train["feature_scales"] == design["train"]["deployed_scales"]
            and refit["feature_scales"]
            == design["train_plus_calibration"]["deployed_scales"]
        ):
            return False
        primary = _grid_mapping(calibration["train_grid_fit_diagnostics"])
        control = _grid_mapping(calibration["train_grid_1d_fit_diagnostics"])
        for ridge_lambda in LAMBDA_GRID:
            if not (
                primary[ridge_lambda]["normalized_gram"] == train["normalized_gram"]
                and primary[ridge_lambda]["normalized_rhs"] == train["normalized_rhs"]
                and primary[ridge_lambda]["feature_scales"] == train["feature_scales"]
                and control[ridge_lambda]["normalized_gram"]
                == one["train_normalized_gram"]
                and control[ridge_lambda]["normalized_rhs"]
                == one["train_normalized_rhs"]
            ):
                return False
        return bool(
            design["final_fit"]["normalized_gram"] == refit["normalized_gram"]
            and design["final_fit"]["normalized_rhs"] == refit["normalized_rhs"]
            and design["final_fit"]["feature_scales"] == refit["feature_scales"]
            and design["final_1d_fit"]["normalized_gram"]
            == one["train_plus_calibration_normalized_gram"]
            and design["final_1d_fit"]["normalized_rhs"]
            == one["train_plus_calibration_normalized_rhs"]
            and statistics_artifact["selected_lambda"]
            == calibration.get(
                "selected_lambda", statistics_artifact["selected_lambda"]
            )
            and statistics_artifact["selected_lambda_one_dimensional"]
            == calibration.get(
                "selected_lambda_one_dimensional",
                statistics_artifact["selected_lambda_one_dimensional"],
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _audit_fit(
    bundle: Mapping[str, Any], results: Path, device: torch.device
) -> dict[str, Any]:
    required = {
        "resolved": results / "resolved_config.json",
        "manifest": results / "manifest.json",
        "decision": results / "fit_decision.json",
        "predictor": results / "frozen_predictor.json",
        "calibration": results / "lambda_calibration.json",
        "design": results / "feature_design_diagnostics.json",
        "registry": results / "fit_rng_registry.json",
        "statistics": results / "fit_sufficient_statistics.json",
        "rows": results / "fit_history_rows.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "pass": False,
            "checks": {"artifacts_present": False},
            "missing": missing,
        }
    resolved = _read_json(required["resolved"])
    manifest = _read_json(required["manifest"])
    decision = _read_json(required["decision"])
    predictor = _read_json(required["predictor"])
    calibration = _read_json(required["calibration"])
    design = _read_json(required["design"])
    registry = _read_json(required["registry"])
    statistics_artifact = _read_json(required["statistics"])
    rows = _read_csv(required["rows"])
    config = bundle["resolved"]
    hashes = bundle["hashes"]
    predictor_sha = _sha256(required["predictor"])

    expected_registry = _expected_fit_registry(config)
    registry_exact = registry == expected_registry
    streams = registry.get("streams", {})
    fit_rng = [
        int(value)
        for key in FIT_STREAMS
        for value in _flatten_rng_records(streams.get(key, ()))
    ]
    flat_rng_hash = _canonical_hash(fit_rng)
    structured_rng_hash = expected_registry["payload_sha256"]
    rng_hash_objects = (manifest, decision, predictor)
    structured_hash_ok = all(
        obj.get("fit_rng_registry_structured_sha256") == structured_rng_hash
        for obj in rng_hash_objects
    )

    expected_train = {record[0] for record in _history_records(config, "train")}
    expected_cal = {record[0] for record in _history_records(config, "calibration")}
    train_rows = [row for row in rows if row.get("split") == "train"]
    cal_rows = [row for row in rows if row.get("split") == "calibration"]
    train_ids = [row.get("history_id") for row in train_rows]
    cal_ids = [row.get("history_id") for row in cal_rows]

    tolerance = float(calibration.get("tie_absolute_tolerance", float("nan")))
    ridge_checks: dict[str, Any] = {}
    try:
        selected, _ = _select_lambda(calibration["primary"], tolerance)
        selected_1d, _ = _select_lambda(calibration["one_dimensional"], tolerance)
        primary_grid = _grid_mapping(calibration["train_grid_fit_diagnostics"])
        one_grid = _grid_mapping(calibration["train_grid_1d_fit_diagnostics"])
        for ridge_lambda in LAMBDA_GRID:
            diag = primary_grid[ridge_lambda]
            ridge_checks[f"primary_{ridge_lambda:g}"] = _audit_ridge_solution(
                diag,
                expected_count=5,
                expected_lambda=ridge_lambda,
                stored_coefficients=diag.get("coefficients", []),
                expected_feature_names=FEATURE_NAMES,
                device=device,
                gates=config["gates"],
            )
            diag_1d = one_grid[ridge_lambda]
            ridge_checks[f"one_dimensional_{ridge_lambda:g}"] = _audit_ridge_solution(
                diag_1d,
                expected_count=1,
                expected_lambda=ridge_lambda,
                stored_coefficients=diag_1d.get("coefficients", []),
                expected_feature_names=FEATURE_NAMES[:1],
                device=device,
                gates=config["gates"],
            )
        ridge_checks["final"] = _audit_ridge_solution(
            design["final_fit"],
            expected_count=5,
            expected_lambda=selected,
            stored_coefficients=predictor.get("coefficients", []),
            expected_feature_names=FEATURE_NAMES,
            device=device,
            gates=config["gates"],
        )
        one = predictor.get("one_dimensional_control", {})
        ridge_checks["final_one_dimensional"] = _audit_ridge_solution(
            design["final_1d_fit"],
            expected_count=1,
            expected_lambda=selected_1d,
            stored_coefficients=[one.get("coefficient")],
            expected_feature_names=FEATURE_NAMES[:1],
            device=device,
            gates=config["gates"],
        )
        lambda_ok = (
            selected
            == predictor.get("selected_lambda")
            == decision.get("selected_lambda")
            == statistics_artifact.get("selected_lambda")
            and selected_1d
            == one.get("selected_lambda")
            == decision.get("selected_lambda_one_dimensional")
            == statistics_artifact.get("selected_lambda_one_dimensional")
            and tolerance == 1e-12
            and calibration.get("tie_break") == "largest_lambda"
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        ridge_checks["exception"] = {"pass": False, "errors": [str(exc)]}
        lambda_ok = False

    rank_train = design.get("train", {})
    rank_refit = design.get("train_plus_calibration", {})
    expected_rank_tolerances = (
        576 * 64 * torch.finfo(torch.float32).eps,
        960 * 64 * torch.finfo(torch.float32).eps,
    )
    rank_ok = all(
        (
            item.get("feature_count") == 5
            and item.get("flattened_design_rank") == 5
            and item.get("floor_active_count") == 0
            and _finite_number(item.get("minimum_centered_rms"))
            and float(item["minimum_centered_rms"]) > 0.0
            and _device_type(item.get("rank_diagnostics", {}).get("compute_device"))
            == "mps"
            and item.get("rank_diagnostics", {}).get("compute_dtype") == "torch.float32"
            and _close(
                item.get("rank_diagnostics", {}).get("relative_tolerance"),
                expected_tolerance,
                1e-7,
            )
        )
        for item, expected_tolerance in zip(
            (rank_train, rank_refit), expected_rank_tolerances, strict=True
        )
    )
    weights = design.get("weight_audit", {})
    train_seed_keys = {
        str(value) for value in config["randomness"]["train_outer_seeds"]
    }
    refit_seed_keys = train_seed_keys | {
        str(value) for value in config["randomness"]["calibration_outer_seeds"]
    }
    train_mass = weights.get("balanced_train_sum_by_seed", {})
    refit_mass = weights.get("balanced_train_plus_calibration_sum_by_seed", {})
    weight_ok = (
        set(train_mass) == train_seed_keys
        and set(refit_mass) == refit_seed_keys
        and all(
            _close(value, 1.0, 1e-6)
            for value in (*train_mass.values(), *refit_mass.values())
        )
    )
    evaluation_exists = (results / "evaluation").exists()
    phase_ok = (
        _post_evaluation_boundary_ok(manifest, decision, predictor)
        if evaluation_exists
        else _phase_boundary_ok(
            manifest,
            decision,
            predictor,
            evaluation_directory_exists=False,
        )
    )
    checks = {
        "artifacts_present": True,
        "resolved_config_exact": resolved == config
        and _sha256(required["resolved"]) == hashes["resolved_config_sha256"],
        "campaign_and_predictor_schema": set(predictor) == PREDICTOR_SCHEMA
        and predictor.get("schema_version") == 1
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and predictor.get("campaign_id") == CAMPAIGN_ID
        and predictor.get("candidate") == K5
        and tuple(predictor.get("feature_names", ())) == FEATURE_NAMES
        and predictor.get("removed_v1_feature") == REMOVED_V1_FEATURE
        and predictor.get("observable_past_only_at_inference") is True
        and predictor.get("privileged_training_supervision")
        == "k4c_ch_synthetic_targets"
        and predictor.get("public_cohort_size") == int(config["cohort"]["num_clients"])
        and _close(
            predictor.get("influence_cap"),
            config["references"]["total_client_influence_cap"],
            1e-12,
        )
        and len(predictor.get("coefficients", ())) == 5
        and len(predictor.get("feature_scales", ())) == 5
        and all(_finite_number(value) for value in predictor.get("coefficients", ()))
        and all(
            _finite_number(value) and float(value) > 0.0
            for value in predictor.get("feature_scales", ())
        ),
        "config_hashes": _artifact_hash_check(manifest, hashes)
        and _artifact_hash_check(predictor, hashes),
        "predictor_hash": predictor_sha
        == manifest.get("frozen_predictor_sha256")
        == decision.get("frozen_predictor_sha256"),
        "fit_matrix": len(train_ids) == len(expected_train)
        and len(set(train_ids)) == len(train_ids)
        and set(train_ids) == expected_train
        and len(cal_ids) == len(expected_cal)
        and len(set(cal_ids)) == len(cal_ids)
        and set(cal_ids) == expected_cal,
        "strict_past": all(
            int(row["feature_max_source_round"]) == int(row["assessment_round"]) - 1
            and row["feature_source_rounds"]
            == ",".join(
                str(value)
                for value in range(
                    int(row["assessment_round"]) - 4,
                    int(row["assessment_round"]),
                )
            )
            and int(row["inference_payload_forbidden_current_fields"]) == 0
            for row in rows
        ),
        "v1_train_reproduced": len(train_rows) == 576
        and all(row.get("v1_train_row_reproduced") == "True" for row in train_rows),
        "feature_rank_and_scales": rank_ok,
        "seed_balancing": weight_ok,
        "sufficient_statistics_cross_linked": _sufficient_statistics_consistent(
            statistics_artifact, calibration, design, config
        ),
        "lambda_selection": lambda_ok,
        "all_18_ridge_solutions": len(ridge_checks) == 18
        and all(item.get("pass") is True for item in ridge_checks.values()),
        "rng_registry_exact": registry_exact,
        "rng_unique": len(fit_rng) == 86016 and len(fit_rng) == len(set(fit_rng)),
        "rng_structured_hash": structured_hash_ok,
        "phase_boundary": phase_ok,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "ridge_solutions": ridge_checks,
        "hashes": hashes,
        "rng": {
            "flat_sha256": flat_rng_hash,
            "structured_sha256": structured_rng_hash,
            "count": len(fit_rng),
        },
        "selected_lambda": predictor.get("selected_lambda"),
        "selected_lambda_one_dimensional": predictor.get(
            "one_dimensional_control", {}
        ).get("selected_lambda"),
    }


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if len(numbers) < 2 or not all(math.isfinite(value) for value in numbers):
        raise ValueError("CI requires at least two finite outer-seed values")
    mean = statistics.fmean(numbers)
    half = float(t_critical) * statistics.stdev(numbers) / math.sqrt(len(numbers))
    return {"n": len(numbers), "mean": mean, "low": mean - half, "high": mean + half}


def _ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0.0
    ):
        return float("nan")
    return numerator / denominator


def _csv_bool(value: Any) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def _headroom_diagnostics(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for row in seed_rows:
        relative = float(
            row.get("relative_k4_k4c_headroom", row.get("relative_headroom"))
        )
        near_zero = bool(
            row.get(
                "headroom_numerically_near_zero",
                float(row["k4_minus_k4c"])
                <= 1e-12 * max(1.0, abs(float(row["k4_integrated_mse"]))),
            )
        )
        small = bool(row.get("headroom_small_relative_warning", relative < 0.05))
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "relative_headroom": relative,
                "split_mc_over_headroom": float(row["split_mc_over_headroom"]),
                "numerically_near_zero": near_zero,
                "small_relative_warning": small,
            }
        )
    relative = [row["relative_headroom"] for row in per_seed]
    split = [row["split_mc_over_headroom"] for row in per_seed]
    finite_relative = [value for value in relative if math.isfinite(value)]
    finite_split = [value for value in split if math.isfinite(value)]
    near_zero_count = sum(row["numerically_near_zero"] for row in per_seed)
    return {
        "per_seed": per_seed,
        "relative_headroom_minimum": min(finite_relative, default=float("nan")),
        "relative_headroom_median": (
            statistics.median(finite_relative) if finite_relative else float("nan")
        ),
        "split_mc_over_headroom_minimum": min(finite_split, default=float("nan")),
        "split_mc_over_headroom_median": (
            statistics.median(finite_split) if finite_split else float("nan")
        ),
        "split_mc_over_headroom_maximum": max(finite_split, default=float("nan")),
        "numerically_near_zero_seed_count": near_zero_count,
        "small_relative_headroom_warning_seed_count": sum(
            row["small_relative_warning"] for row in per_seed
        ),
        "warning": (
            "K4-K4c is numerically near zero for at least one seed; capture ratios are unstable."
            if near_zero_count
            else "none"
        ),
        "changes_scientific_decision": False,
    }


def _recompute_evaluation(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, str]],
    child_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_history: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    meta = {row["history_id"]: row for row in history_rows}
    for row in child_rows:
        value = float(row["squared_reference_error"])
        if not math.isfinite(value):
            raise ValueError("non-finite evaluation metric")
        by_history[row["history_id"]][row["candidate"]].append(value)
    seed_rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        ids = [key for key, row in meta.items() if int(row["seed"]) == int(seed)]
        sums: dict[str, float] = defaultdict(float)
        regimes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        split_sq = 0.0
        for history_id in ids:
            for candidate in CANDIDATES:
                values = by_history[history_id][candidate]
                if len(values) != int(
                    config["nested_monte_carlo"]["evaluation_children"]
                ):
                    raise ValueError("evaluation child multiplicity mismatch")
                value = statistics.fmean(values)
                sums[candidate] += value
                regimes[meta[history_id]["noise_regime"]][candidate] += value
            split_sq += (
                float(
                    meta[history_id]["privileged_target_split_aggregate_scale_distance"]
                )
                ** 2
            )
        headroom = sums[K4] - sums[K4C]
        near_zero = headroom <= 1e-12 * max(1.0, abs(sums[K4]))
        relative_headroom = _ratio(headroom, sums[K4])
        seed_rows.append(
            {
                "seed": int(seed),
                "histories": len(ids),
                "k4_integrated_mse": sums[K4],
                "k4b_integrated_mse": sums[K4B],
                "one_dimensional_integrated_mse": sums[K5_1D],
                "k5_integrated_mse": sums[K5],
                "k4c_integrated_mse": sums[K4C],
                "pointwise_integrated_mse": sums[POINTWISE],
                "gain_vs_k4b": _ratio(sums[K4B] - sums[K5], sums[K4B]),
                "gain_vs_k4": _ratio(sums[K4] - sums[K5], sums[K4]),
                "gain_vs_one_dimensional": _ratio(sums[K5_1D] - sums[K5], sums[K5_1D]),
                "ch_capture_fraction": _ratio(sums[K4] - sums[K5], headroom),
                "k4_minus_k4c": headroom,
                "k4_minus_k4c_positive": headroom > 0.0,
                "relative_k4_k4c_headroom": relative_headroom,
                "headroom_numerically_near_zero": near_zero,
                "headroom_small_relative_warning": bool(
                    math.isfinite(relative_headroom) and relative_headroom < 0.05
                ),
                "split_target_disagreement_mse": split_sq,
                "split_mc_over_headroom": _ratio(split_sq, headroom),
                "homogeneous_gain_vs_k4b": _ratio(
                    regimes["homogeneous"][K4B] - regimes["homogeneous"][K5],
                    regimes["homogeneous"][K4B],
                ),
                "heteroscedastic_gain_vs_k4b": _ratio(
                    regimes["heteroscedastic"][K4B] - regimes["heteroscedastic"][K5],
                    regimes["heteroscedastic"][K4B],
                ),
                "privileged_target_split_disagreement_mse_ratio": _ratio(
                    split_sq, sums[K4]
                ),
            }
        )
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    contrasts = (
        "gain_vs_k4b",
        "gain_vs_k4",
        "gain_vs_one_dimensional",
        "ch_capture_fraction",
        "homogeneous_gain_vs_k4b",
        "heteroscedastic_gain_vs_k4b",
    )
    cis = {key: _ci([float(row[key]) for row in seed_rows], tcrit) for key in contrasts}
    return seed_rows, cis, _headroom_diagnostics(seed_rows)


def _seed_summary_matches(
    recomputed: Sequence[Mapping[str, Any]], stored: Sequence[Mapping[str, str]]
) -> bool:
    bool_keys = {
        "k4_minus_k4c_positive",
        "headroom_numerically_near_zero",
        "headroom_small_relative_warning",
    }
    by_seed = {int(row["seed"]): row for row in stored}
    if len(by_seed) != len(stored) or set(by_seed) != {
        int(row["seed"]) for row in recomputed
    }:
        return False
    try:
        for row in recomputed:
            observed = by_seed[int(row["seed"])]
            if set(observed) != set(row):
                return False
            for key, value in row.items():
                if key == "seed":
                    if int(observed[key]) != int(value):
                        return False
                elif key == "histories":
                    if int(observed[key]) != int(value):
                        return False
                elif key in bool_keys:
                    if _csv_bool(observed[key]) is not bool(value):
                        return False
                elif not _close(observed[key], value, 2e-6):
                    return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _confidence_intervals_match(
    stored: Any, recomputed: Mapping[str, Mapping[str, Any]]
) -> bool:
    if not isinstance(stored, Mapping) or set(stored) != set(recomputed):
        return False
    try:
        return all(
            set(stored[key]) == {"n", "mean", "low", "high"}
            and int(stored[key]["n"]) == int(value["n"])
            and all(
                _close(stored[key][field], value[field], 2e-6)
                for field in ("mean", "low", "high")
            )
            for key, value in recomputed.items()
        )
    except (KeyError, TypeError, ValueError):
        return False


def _publication_attestation_ok(value: Any, *, name: str, expected_sha256: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("procedural_external_publication_attested") is True
        and value.get(f"published_{name}_sha256") == expected_sha256
        and value.get("machine_verifies_external_log_itself") is False
    )


def _independent_fit_audit_attestation_ok(value: Any, audit_path: Path) -> bool:
    """Verify the externally anchored pre-evaluation audit without a hash cycle."""

    if not isinstance(value, Mapping) or not audit_path.is_file():
        return False
    try:
        fit_audit = _read_json(audit_path)
        actual_sha256 = _sha256(audit_path)
    except (OSError, ValueError):
        return False
    audited = fit_audit.get("audited_artifact_sha256")
    return bool(
        value.get("verified_before_evaluation") is True
        and value.get("pre_evaluation_external_anchor_required") is True
        and value.get("sha256") == actual_sha256
        and value.get("audited_artifact_sha256") == audited
        and isinstance(audited, Mapping)
        and set(audited) == set(FIT_AUDITED_ARTIFACT_NAMES)
        and all(_is_sha256(item) for item in audited.values())
        and fit_audit.get("schema_version") == 1
        and fit_audit.get("campaign_id") == CAMPAIGN_ID
        and fit_audit.get("audit_scope") == "fit_and_freeze_pre_evaluation"
        and fit_audit.get("all_checks_pass") is True
        and all(bool(item) for item in fit_audit.get("checks", {}).values())
        and _publication_attestation_ok(
            value.get("publication_attestation"),
            name="fit_audit",
            expected_sha256=actual_sha256,
        )
    )


def _audit_evaluation(
    bundle: Mapping[str, Any], results: Path, fit_audit: Mapping[str, Any]
) -> dict[str, Any]:
    directory = results / "evaluation"
    required = {
        "history": directory / "history_rows.csv",
        "children": directory / "evaluation_child_rows.csv",
        "seeds": directory / "seed_summary.csv",
        "replace": directory / "replace_one_audit.csv",
        "registry": directory / "evaluation_rng_registry.json",
        "decision": directory / "decision.json",
        "manifest": directory / "manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "pass": False,
            "checks": {"artifacts_present": False},
            "missing": missing,
        }
    config = bundle["resolved"]
    history_rows = _read_csv(required["history"])
    child_rows = _read_csv(required["children"])
    stored_seed_rows = _read_csv(required["seeds"])
    replace_rows = _read_csv(required["replace"])
    registry = _read_json(required["registry"])
    stored_decision = _read_json(required["decision"])
    manifest = _read_json(required["manifest"])
    root_manifest = _read_json(results / "manifest.json")
    predictor = _read_json(results / "frozen_predictor.json")

    expected_records = _history_records(config, "evaluation")
    expected_ids = {record[0] for record in expected_records}
    observed_ids = [row["history_id"] for row in history_rows]
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    expected_keys = {
        (history_id, candidate, child)
        for history_id in expected_ids
        for candidate in CANDIDATES
        for child in range(children)
    }
    observed_keys = [
        (row["history_id"], row["candidate"], int(row["evaluation_child"]))
        for row in child_rows
    ]
    target_records = _expected_rng_stream_records(
        config,
        split="evaluation",
        stream="evaluation_target",
        children=int(
            config["nested_monte_carlo"]["evaluation_target_construction_children"]
        ),
    )
    evaluation_records = _expected_rng_stream_records(
        config, split="evaluation", stream="evaluation", children=children
    )
    expected_registry = _rng_registry(
        {
            "train_target": [],
            "calibration_target": [],
            "calibration_evaluation": [],
            "evaluation_target": target_records,
            "evaluation": evaluation_records,
            "holdout": [],
        }
    )
    expected_target = _flatten_rng_records(target_records)
    expected_eval = _flatten_rng_records(evaluation_records)
    expected_seed_by_key = {
        (str(record["history_id"]), child): int(seed)
        for record in evaluation_records
        for child, seed in enumerate(record["child_seeds"])
    }
    try:
        child_seed_exact = all(
            int(row["evaluation_child_seed"])
            == expected_seed_by_key[(row["history_id"], int(row["evaluation_child"]))]
            for row in child_rows
        )
        seed_rows, cis, headroom = _recompute_evaluation(
            config, history_rows, child_rows
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "pass": False,
            "checks": {"evaluation_recomputation": False},
            "error": str(exc),
        }

    gates = config["gates"]
    scientific = {
        "gain_vs_k4b_mean": cis["gain_vs_k4b"]["mean"]
        >= float(gates["primary_gain_vs_k4b_mean_min"]),
        "gain_vs_k4b_ci": cis["gain_vs_k4b"]["low"]
        > float(gates["primary_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "gain_vs_k4_mean": cis["gain_vs_k4"]["mean"]
        >= float(gates["primary_gain_vs_k4_mean_min"]),
        "gain_vs_k4_ci": cis["gain_vs_k4"]["low"]
        > float(gates["primary_gain_vs_k4_ci95_low_strictly_greater_than"]),
        "gain_vs_one_dimensional_mean": cis["gain_vs_one_dimensional"]["mean"]
        >= float(gates["primary_gain_vs_one_dimensional_mean_min"]),
        "gain_vs_one_dimensional_ci": cis["gain_vs_one_dimensional"]["low"]
        > float(
            gates["primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"]
        ),
        "capture_mean": cis["ch_capture_fraction"]["mean"]
        >= float(gates["ch_capture_fraction_mean_min"]),
        "capture_ci": cis["ch_capture_fraction"]["low"]
        > float(gates["ch_capture_fraction_ci95_low_strictly_greater_than"]),
        "homogeneous_gain_ci": cis["homogeneous_gain_vs_k4b"]["low"]
        > float(gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "heteroscedastic_gain_ci": cis["heteroscedastic_gain_vs_k4b"]["low"]
        > float(gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"]),
    }
    fit_registry = _read_json(results / "fit_rng_registry.json")
    fit_rng = [
        value
        for key in FIT_STREAMS
        for value in _flatten_rng_records(fit_registry.get("streams", {}).get(key, ()))
    ]
    all_rng = fit_rng + expected_target + expected_eval
    predictor_hash = _sha256(results / "frozen_predictor.json")
    try:
        replace_ok = (
            len(replace_rows) == int(gates["replace_one_exact_trials"])
            and sum(_csv_bool(row["violation"]) for row in replace_rows)
            <= int(gates["replace_one_violation_max"])
            and all(
                _close(row["theoretical_bound"], gates["replace_one_bound"], 1e-8)
                and float(row["observed_difference"])
                <= float(row["theoretical_bound"]) + 1e-6
                for row in replace_rows
            )
        )
    except (KeyError, TypeError, ValueError):
        replace_ok = False
    registry_hash = expected_registry["payload_sha256"]
    validity = {
        "fit_audit_pass": fit_audit.get("pass") is True,
        "evaluation_device_mps_float32": _device_type(manifest.get("device")) == "mps"
        and manifest.get("dtype") == "torch.float32",
        "history_matrix": len(observed_ids) == 576
        and len(set(observed_ids)) == 576
        and set(observed_ids) == expected_ids,
        "child_matrix": len(observed_keys) == len(expected_keys)
        and len(set(observed_keys)) == len(observed_keys)
        and set(observed_keys) == expected_keys,
        "rng_registry_and_each_seed": registry == expected_registry
        and child_seed_exact,
        "rng_hash_cross_linked": manifest.get(
            "evaluation_rng_registry_structured_sha256"
        )
        == registry_hash
        == root_manifest.get("evaluation_rng_registry_structured_sha256"),
        "rng_counts_and_global_uniqueness": len(expected_target)
        == int(gates["evaluation_target_child_seeds_exact"])
        and len(expected_eval) == int(gates["evaluation_child_seeds_exact"])
        and len(all_rng) == len(set(all_rng)),
        "seed_summary_reproduced": _seed_summary_matches(seed_rows, stored_seed_rows),
        "confidence_intervals_reproduced": _confidence_intervals_match(
            stored_decision.get("confidence_intervals"), cis
        ),
        "replace_one": replace_ok,
        "formulae": max(
            (float(row["k4_manual_formula_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["k4_manual_formula_abs_error_max"])
        and max(
            (float(row["k4b_reproduction_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["k4b_reproduction_abs_error_max"])
        and max(
            (float(row["fixed_denominator_formula_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["fixed_denominator_formula_abs_error_max"]),
        "fixed_denominator_and_cap": all(
            int(row["fixed_denominator_n"]) == int(config["cohort"]["num_clients"])
            and not _csv_bool(row["normalization_by_gate_sum"])
            and (
                row["candidate"] != K5
                or (
                    _csv_bool(row["contribution_cap_respected"])
                    and float(row["max_slot_contribution_norm"])
                    <= float(config["references"]["total_client_influence_cap"])
                    + 64.0 * torch.finfo(torch.float32).eps
                )
            )
            for row in child_rows
        ),
        "strict_past": all(
            int(row["feature_max_source_round"]) == int(row["assessment_round"]) - 1
            and row["feature_source_rounds"]
            == ",".join(
                str(value)
                for value in range(
                    int(row["assessment_round"]) - 4,
                    int(row["assessment_round"]),
                )
            )
            and int(row["forbidden_current_inference_field_count"]) == 0
            for row in history_rows
        ),
        "predictor_hash_stable": manifest.get("frozen_predictor_sha256")
        == predictor_hash
        == root_manifest.get("frozen_predictor_sha256_after_evaluation")
        and all(
            row.get("frozen_predictor_sha256") == predictor_hash for row in history_rows
        )
        and all(
            row.get("frozen_predictor_sha256") == predictor_hash
            and _csv_bool(row["frozen_predictor_fixed"])
            for row in child_rows
        ),
        "publication_attestations": _publication_attestation_ok(
            manifest.get("lock_publication_attestation"),
            name="lock",
            expected_sha256=str(predictor["lock_sha256"]),
        )
        and _publication_attestation_ok(
            manifest.get("predictor_publication_attestation"),
            name="predictor",
            expected_sha256=predictor_hash,
        )
        and _independent_fit_audit_attestation_ok(
            manifest.get("independent_fit_audit_attestation"),
            results / "independent_fit_audit.json",
        )
        and root_manifest.get("independent_fit_audit_attestation")
        == manifest.get("independent_fit_audit_attestation"),
        "mc_stability": max(
            (
                float(row["privileged_target_split_disagreement_mse_ratio"])
                for row in seed_rows
            ),
            default=float("inf"),
        )
        <= float(gates["privileged_target_split_disagreement_mse_ratio_max"]),
        "headroom_positive": len(seed_rows) == 12
        and sum(bool(row["k4_minus_k4c_positive"]) for row in seed_rows)
        == int(gates["positive_k4_minus_k4c_denominator_seed_count_exact"]),
        "headroom_non_gating": headroom["changes_scientific_decision"] is False,
        "holdout_closed": manifest.get("holdout_opened") is False
        and root_manifest.get("holdout_opened") is False
        and stored_decision.get("holdout_opened") is False
        and registry.get("streams", {}).get("holdout") == [],
        "completed_manifest": manifest.get("status") == "completed_development"
        and root_manifest.get("status") == "completed_development"
        and root_manifest.get("evaluation_trajectory_count_generated") == 576,
    }
    validity_pass = all(validity.values())
    scientific_pass = validity_pass and all(scientific.values())
    expected_decision = (
        "authorize_end_to_end_development_screen"
        if scientific_pass
        else (
            "stop_five_feature_linear_predictor_instance"
            if validity_pass
            else "invalid_or_inconclusive_screen"
        )
    )
    stored_match = bool(
        stored_decision.get("validity_pass") is validity_pass
        and stored_decision.get("scientific_checks_pass") is scientific_pass
        and stored_decision.get("all_gates_pass") is (validity_pass and scientific_pass)
        and stored_decision.get("decision") == expected_decision
        and stored_decision.get("scientific_checks") == scientific
        and stored_decision.get("headroom_diagnostics_non_gating") == headroom
        and stored_decision.get("holdout_opened") is False
        and manifest.get("decision") == expected_decision
        and manifest.get("all_gates_pass") is (validity_pass and scientific_pass)
    )
    validity["stored_decision_reproduced"] = stored_match
    passed = all(validity.values())
    return {
        "pass": passed,
        "all_checks_pass": passed,
        "validity_checks": validity,
        "scientific_checks": scientific,
        "confidence_intervals": cis,
        "headroom_diagnostics_non_gating": headroom,
        "decision_recomputed": expected_decision,
        "seed_rows_recomputed": seed_rows,
    }


def run_audit(
    *, config_path: Path, lock_path: Path, results: Path, device: torch.device
) -> dict[str, Any]:
    bundle = _load_config_bundle(config_path)
    lock = _verify_lock(lock_path, config_path)
    v1 = _audit_v1_provenance(bundle["amendment"])
    technical_rerun = _audit_technical_v2_mps_v1_failure_provenance(
        bundle["amendment"], bundle["resolved"]
    )
    fit = _audit_fit(bundle, results, device)
    evaluation_exists = (results / "evaluation").exists()
    evaluation = _audit_evaluation(bundle, results, fit) if evaluation_exists else None
    all_checks = (
        lock.get("pass") is True
        and v1.get("pass") is True
        and technical_rerun.get("pass") is True
        and fit.get("pass") is True
    )
    if evaluation_exists:
        all_checks = (
            all_checks and evaluation is not None and evaluation.get("pass") is True
        )
    audited_artifact_sha256 = _fit_artifact_hash_inventory(results)
    top_checks = {
        "preregistration_lock_integrity": lock.get("pass") is True,
        "v1_pre_evaluation_provenance": v1.get("pass") is True,
        "technical_v2_mps_v1_failure_provenance": technical_rerun.get("pass") is True,
        "fit_and_freeze_recomputed": fit.get("pass") is True,
        "required_artifact_hash_inventory": set(audited_artifact_sha256)
        == set(FIT_AUDITED_ARTIFACT_NAMES)
        and all(_is_sha256(value) for value in audited_artifact_sha256.values()),
        "evaluation_recomputed_if_present": not evaluation_exists
        or (evaluation is not None and evaluation.get("pass") is True),
        "holdout_closed": fit.get("checks", {}).get("phase_boundary") is True,
    }
    all_checks = all_checks and all(top_checks.values())
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "audit_scope": (
            "fit_and_evaluation"
            if evaluation_exists
            else "fit_and_freeze_pre_evaluation"
        ),
        "audit_device": str(device),
        "independent_of_v2_runner": True,
        "checks": top_checks,
        "audited_artifact_sha256": audited_artifact_sha256,
        "config_hashes": bundle["hashes"],
        "lock_audit": lock,
        "v1_provenance_audit": v1,
        "technical_v2_mps_v1_failure_provenance_audit": technical_rerun,
        "fit_audit": fit,
        "evaluation_present": evaluation_exists,
        "evaluation_audit": evaluation,
        "holdout_opened": False,
        "all_checks_pass": all_checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args(argv)
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("K5-v2 independent production audit requires real MPS")
    device = torch.device("mps")
    report = run_audit(
        config_path=args.config.resolve(),
        lock_path=args.lock.resolve(),
        results=args.results.resolve(),
        device=device,
    )
    output = (
        args.results.resolve() / "evaluation" / "independent_postrun_audit.json"
        if report["evaluation_present"]
        else args.results.resolve() / "independent_fit_audit.json"
    )
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
