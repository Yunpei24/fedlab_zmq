#!/usr/bin/env python3
"""Fit/freeze then evaluate the technical K5-TP-v2 rerun on MPS.

V2 is a transparent post-v1-fit, pre-evaluation structural revision.  It
removes one redundant past-only feature, uses new calibration outer seeds and
keeps the still-unopened v1 evaluation registry and every scientific gate.
This mps-v2 rerun only repairs the reported missing CSV-reader symbol from the
first v2 attempt, whose data-free failure boundary is machine-verifiable.  The
reported exception itself has no persisted traceback and is not a machine-
verified claim.  The command has an irreversible fit/hash-publication/
evaluation boundary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from algorithms.gaussian_aware_reference_k5_tp_v2 import (  # noqa: E402
    FEATURE_NAMES,
    REMOVED_V1_FEATURE,
    fit_shared_scalar_ridge,
    forbidden_current_field_count,
    training_feature_scales,
    transcript_past_feature_dictionary,
    transcript_past_predictor,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k4b_past_imputation as k4b,
)
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k4c_causal_headroom as k4c,
)
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k5_transcript_predictor as v1,
)
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
K2 = v1.K2
K4 = v1.K4
K4B = v1.K4B
K5_1D = "g0g_k5_tp_v2_one_dimensional_control"
K5 = "g0g_k5_tp_v2_five_feature_ridge"
K4C = v1.K4C
POINTWISE = v1.POINTWISE
CANDIDATES = (K2, K4, K4B, K5_1D, K5, K4C, POINTWISE)
RNG_STREAM_ORDER = (
    "train_target",
    "calibration_target",
    "calibration_evaluation",
    "evaluation_target",
    "evaluation",
    "holdout",
)
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

DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/k5_v2/gaussian_aware_reference_g0g_k5_tp_v2.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/k5_v2/"
    "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2.lock.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
V1_RESULTS = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)
TECHNICAL_V2_MPS_V1_RESULTS = ROOT / (
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k5_tp_v2_mps_v1"
)
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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a rectangular CSV artifact and reject malformed rows or headers."""

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


def _progress(phase: str, *, completed: int, total: int, outer_seed: int) -> None:
    """Emit one deterministic, line-buffered progress record per outer seed."""

    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "phase": phase,
                "outer_seed": int(outer_seed),
                "outer_seeds_completed": int(completed),
                "outer_seeds_total": int(total),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers)
    half = float(t_critical) * sd / math.sqrt(len(numbers))
    return {"n": len(numbers), "mean": mean, "low": mean - half, "high": mean + half}


def _stable_seed(*parts: object) -> int:
    """Reproduce the scientific child-seed derivation without global RNG state."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _expected_rng_stream_records(
    config: Mapping[str, Any],
    *,
    split: str,
    seeds: Sequence[int],
    stream: str,
    children: int,
) -> list[dict[str, Any]]:
    """Return one deterministic record per history, with child index by list offset."""

    tag = str(config["nested_monte_carlo"][f"{stream}_stream_tag"])
    rows: list[dict[str, Any]] = []
    for cell in v1._cells(config, seeds):
        for round_index in config["temporal"]["assessment_rounds"]:
            history_id = v1._history_id(split, cell, int(round_index))
            child_seeds = [
                _stable_seed(
                    tag,
                    int(cell["seed"]),
                    str(cell["regime"]["name"]),
                    str(cell["permutation"]),
                    str(cell["geometry"]),
                    str(cell["dynamics"]),
                    str(cell["threat"]),
                    int(round_index),
                    child,
                )
                for child in range(int(children))
            ]
            rows.append(
                {
                    "history_id": history_id,
                    "stream": stream,
                    "stream_tag": tag,
                    "outer_seed": int(cell["seed"]),
                    "noise_regime": str(cell["regime"]["name"]),
                    "noise_permutation": str(cell["permutation"]),
                    "outlier_geometry": str(cell["geometry"]),
                    "honest_dynamics": str(cell["dynamics"]),
                    "threat": str(cell["threat"]),
                    "assessment_round": int(round_index),
                    "child_seeds": child_seeds,
                }
            )
    return rows


def _flatten_rng_records(records: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(seed) for record in records for seed in record.get("child_seeds", [])]


def _rng_registry(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stream_order": list(RNG_STREAM_ORDER),
        "streams": {
            key: [dict(record) for record in streams.get(key, [])]
            for key in RNG_STREAM_ORDER
        },
    }
    return {**payload, "payload_sha256": _canonical_hash(payload)}


def _validate_rng_registry(
    registry: Mapping[str, Any],
    expected_streams: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    if set(registry) != {
        "schema_version",
        "stream_order",
        "streams",
        "payload_sha256",
    }:
        return False
    payload = {
        "schema_version": registry["schema_version"],
        "stream_order": registry["stream_order"],
        "streams": registry["streams"],
    }
    expected = _rng_registry(expected_streams)
    return (
        registry["schema_version"] == 1
        and tuple(registry["stream_order"]) == RNG_STREAM_ORDER
        and registry["streams"] == expected["streams"]
        and registry["payload_sha256"] == _canonical_hash(payload)
        and registry["payload_sha256"] == expected["payload_sha256"]
    )


def _load_amended_config(
    path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    amendment = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = ROOT / amendment["base_protocol"]["config_path"]
    if _sha256(base_path) != amendment["base_protocol"]["config_sha256"]:
        raise RuntimeError("K5-v2 base protocol hash changed")
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["campaign_id"] = CAMPAIGN_ID
    config["scope"] = amendment["scope"]
    config["scientific_contract"]["inference_predictor_name"] = K5
    config["features"]["order"] = list(amendment["features"]["order"])
    config["gates"]["feature_count_exact"] = int(
        amendment["features"]["feature_count_exact"]
    )
    config["gates"]["flattened_feature_rank_exact"] = int(
        amendment["features"]["flattened_feature_rank_exact"]
    )
    for key in (
        "train_outer_seeds",
        "calibration_outer_seeds",
        "evaluation_outer_seeds",
        "reserved_holdout_seeds",
    ):
        config["randomness"][key] = list(amendment["randomness"][key])
    config["randomness"]["train_seed_role"] = "reuse_v1_training_outer_seeds"
    config["randomness"][
        "calibration_and_evaluation_seed_rule"
    ] = "new_calibration_seeds_and_exact_unopened_v1_evaluation_seeds"
    streams = amendment["nested_monte_carlo_streams"]
    for key in (
        "train_target_stream_tag",
        "calibration_target_stream_tag",
        "calibration_evaluation_stream_tag",
        "evaluation_target_stream_tag",
        "evaluation_stream_tag",
    ):
        config["nested_monte_carlo"][key] = streams[key]
    config["preregistration_lock"] = {
        "path": amendment["preregistration_lock"]["path"],
        "verify_before_output_creation": True,
        "publish_lock_sha256_before_fit": True,
        "external_publication_is_procedural_precondition": True,
    }
    config["candidates"]["names"] = list(CANDIDATES)
    config["candidates"]["primary"] = K5
    config["candidates"]["learned_one_dimensional_control"] = K5_1D
    config["execution"]["device_identity_check"] = "torch_device_type_equals_mps"
    config["execution"]["output_directory"] = amendment["execution"]["output_directory"]
    return config, amendment


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
    base: Mapping[str, Any], resolved: Mapping[str, Any]
) -> set[tuple[str, ...]]:
    before = _flatten_mapping(base)
    after = _flatten_mapping(resolved)
    return {
        key
        for key in set(before) | set(after)
        if before.get(key, object()) != after.get(key, object())
    }


def _v1_provenance(amendment: Mapping[str, Any]) -> dict[str, Any]:
    declared = amendment["v1_pre_evaluation_provenance"]
    if _sha256(ROOT / declared["lock_path"]) != declared["lock_sha256"]:
        raise RuntimeError("K5-v1 lock provenance changed")
    mismatches: list[str] = []
    for name, expected in declared["required_artifact_sha256"].items():
        path = V1_RESULTS / name
        if not path.is_file() or _sha256(path) != str(expected):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"K5-v1 artifact provenance changed: {mismatches}")
    manifest = json.loads((V1_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (V1_RESULTS / "fit_decision.json").read_text(encoding="utf-8")
    )
    design = json.loads(
        (V1_RESULTS / "feature_design_diagnostics.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (V1_RESULTS / "fit_rng_registry.json").read_text(encoding="utf-8")
    )
    false_checks = {key for key, value in decision["checks"].items() if not value}
    expected_false = set(declared["expected_v1_invalid_checks"])
    concatenated = [
        int(value)
        for key in ("train_target", "calibration_target", "calibration_evaluation")
        for value in registry[key]
    ]
    checks = {
        "v1_status_invalid_fit": manifest["status"]
        == "fit_invalid_evaluation_forbidden"
        and decision["decision"] == "invalid_fit_evaluation_forbidden",
        "only_declared_v1_checks_failed": false_checks == expected_false,
        "v1_effective_rank_five": int(design["train"]["flattened_design_rank"])
        == int(declared["v1_effective_rank_train"])
        == 5
        and int(design["train_plus_calibration"]["flattened_design_rank"])
        == int(declared["v1_effective_rank_refit"])
        == 5,
        "v1_fit_was_mps_semantic": torch.device(design["final_fit"]["fit_device"]).type
        == "mps",
        "v1_evaluation_never_generated": int(
            manifest["evaluation_trajectory_count_generated"]
        )
        == int(declared["v1_evaluation_trajectory_count_exact"])
        == 0
        and registry["evaluation_target"] == []
        and registry["evaluation"] == [],
        "v1_holdout_closed": manifest["holdout_opened"] is False
        and declared["v1_holdout_opened"] is False
        and registry["holdout"] == []
        and not (V1_RESULTS / "evaluation").exists(),
        "v1_rng_unique": len(concatenated) == len(set(concatenated)),
        "v1_rng_hash_native": _canonical_hash(concatenated)
        == json.loads(
            (V1_RESULTS / "frozen_predictor.json").read_text(encoding="utf-8")
        )["fit_rng_registry_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"K5-v1 provenance audit failed: {checks}")
    return {
        "checks": checks,
        "all_checks_pass": True,
        "lock_sha256": declared["lock_sha256"],
        "frozen_predictor_sha256": declared["required_artifact_sha256"][
            "frozen_predictor.json"
        ],
    }


def _technical_v2_mps_v1_failure_provenance(
    amendment: Mapping[str, Any], resolved_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the first v2 data-free boundary; preserve cause as attestation."""

    declared = amendment["technical_v2_mps_v1_failure_provenance"]
    expected_declaration = {
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
    boundary_keys = set(expected_declaration) - cause_keys - documentation_only_keys
    boundary_declaration_exact = all(
        declared.get(key) == expected_declaration[key] for key in boundary_keys
    )
    cause_attestation_schema_exact = all(
        declared.get(key) == expected_declaration[key] for key in cause_keys
    )
    lock_path = ROOT / str(declared["lock_path"])
    manifest_path = TECHNICAL_V2_MPS_V1_RESULTS / "manifest.json"
    resolved_path = TECHNICAL_V2_MPS_V1_RESULTS / "resolved_config.json"
    old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    expected_resolved_delta = {
        ("campaign_id",),
        ("execution", "output_directory"),
        ("preregistration_lock", "path"),
    }
    resolved_delta = _resolved_delta_paths(resolved, resolved_config)
    observed_inventory = sorted(
        str(path.relative_to(TECHNICAL_V2_MPS_V1_RESULTS))
        for path in TECHNICAL_V2_MPS_V1_RESULTS.rglob("*")
        if path.is_file()
    )
    fit_csvs = list(TECHNICAL_V2_MPS_V1_RESULTS.rglob("*.csv"))
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
        "artifact_inventory_exact": observed_inventory
        == declared["artifact_inventory_exact"],
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
        "zero_fit_rows_and_no_evaluation": len(fit_csvs)
        == int(declared["fit_history_row_count_exact"])
        == 0
        and not (TECHNICAL_V2_MPS_V1_RESULTS / "evaluation").exists(),
        "resolved_config_delta_exact": resolved_delta == expected_resolved_delta,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K5-v2 technical-rerun provenance audit failed: {checks}")
    return {
        "checks": checks,
        "all_checks_pass": True,
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


def _validate_config(config: Mapping[str, Any], amendment: Mapping[str, Any]) -> None:
    if (
        config["campaign_id"] != CAMPAIGN_ID
        or tuple(config["features"]["order"]) != FEATURE_NAMES
    ):
        raise ValueError("Unexpected K5-v2 identity or feature schema")
    if amendment["structural_amendment"]["removed_feature"] != REMOVED_V1_FEATURE:
        raise ValueError("K5-v2 removed-feature declaration changed")
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
    if (
        int(config["gates"]["feature_count_exact"]) != 5
        or int(config["gates"]["flattened_feature_rank_exact"]) != 5
    ):
        raise ValueError("K5-v2 requires exactly five full-rank features")
    base = yaml.safe_load(
        (ROOT / amendment["base_protocol"]["config_path"]).read_text(encoding="utf-8")
    )
    allowed_delta = {
        ("campaign_id",),
        ("scope",),
        ("scientific_contract", "inference_predictor_name"),
        ("features", "order"),
        ("gates", "feature_count_exact"),
        ("gates", "flattened_feature_rank_exact"),
        ("randomness", "calibration_outer_seeds"),
        ("randomness", "train_seed_role"),
        ("randomness", "calibration_and_evaluation_seed_rule"),
        ("nested_monte_carlo", "calibration_target_stream_tag"),
        ("nested_monte_carlo", "calibration_evaluation_stream_tag"),
        ("preregistration_lock", "path"),
        ("candidates", "names"),
        ("candidates", "primary"),
        ("candidates", "learned_one_dimensional_control"),
        ("execution", "device_identity_check"),
        ("execution", "output_directory"),
    }
    actual_delta = _resolved_delta_paths(base, config)
    if actual_delta != allowed_delta:
        raise ValueError(
            f"K5-v2 resolved-config delta differs from its allowlist: "
            f"missing={sorted(allowed_delta - actual_delta)}, "
            f"extra={sorted(actual_delta - allowed_delta)}"
        )
    scientific = amendment["scientific_gates_must_equal_v1"]
    if any(
        float(config["gates"][key]) != float(value) for key, value in scientific.items()
    ):
        raise ValueError("K5-v2 scientific gates differ from v1")
    if any(
        float(base["gates"][key]) != float(value) for key, value in scientific.items()
    ):
        raise ValueError("Declared K5-v1 scientific gates are inaccurate")
    train = tuple(int(value) for value in config["randomness"]["train_outer_seeds"])
    calibration = tuple(
        int(value) for value in config["randomness"]["calibration_outer_seeds"]
    )
    evaluation = tuple(
        int(value) for value in config["randomness"]["evaluation_outer_seeds"]
    )
    holdout = tuple(
        int(value) for value in config["randomness"]["reserved_holdout_seeds"]
    )
    if train != tuple(int(value) for value in base["randomness"]["train_outer_seeds"]):
        raise ValueError("K5-v2 must reuse the exact v1 training outer seeds")
    if evaluation != tuple(
        int(value) for value in base["randomness"]["evaluation_outer_seeds"]
    ):
        raise ValueError("K5-v2 must preserve the unopened v1 evaluation seeds")
    if holdout != tuple(
        int(value) for value in base["randomness"]["reserved_holdout_seeds"]
    ):
        raise ValueError("K5-v2 must preserve the closed v1 holdout registry")
    if (len(train), len(calibration), len(evaluation), len(holdout)) != (12, 8, 12, 7):
        raise ValueError("K5-v2 split sizes changed")
    if calibration != (
        2027051001,
        2027051017,
        2027051033,
        2027051049,
        2027051065,
        2027051081,
        2027051097,
        2027051113,
    ):
        raise ValueError("K5-v2 calibration seed registry changed")
    groups = tuple(map(set, (train, calibration, evaluation, holdout)))
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("K5-v2 outer-seed registries overlap")
    snapshot = amendment["prior_registry_snapshot"]
    if snapshot != {
        "timing": "before_k5_v2_preregistration",
        "scanner": (
            "k4b_seed_values_over_ldp_gradient_far_and_dt_ldp_far_yaml_"
            "excluding_this_file"
        ),
        "distinct_integer_count": 271,
        "canonical_sorted_integer_sha256": (
            "2abf8caaa27f55b23e73527126d207db6a3b3e4da255584bcc8007e85af0a4b0"
        ),
        "new_calibration_seed_overlap": [],
    }:
        raise ValueError("K5-v2 frozen prior-seed-registry attestation changed")
    nested = config["nested_monte_carlo"]
    if (
        len(
            {
                nested[key]
                for key in (
                    "train_target_stream_tag",
                    "calibration_target_stream_tag",
                    "calibration_evaluation_stream_tag",
                    "evaluation_target_stream_tag",
                    "evaluation_stream_tag",
                )
            }
        )
        != 5
    ):
        raise ValueError("K5-v2 child stream tags must be distinct")
    if (
        config["execution"].get("device_identity_check")
        != "torch_device_type_equals_mps"
    ):
        raise ValueError("K5-v2 device gate must compare torch device types")
    if config["execution"].get("output_directory") != str(
        DEFAULT_OUTPUT.relative_to(ROOT)
    ):
        raise ValueError("K5-v2 resolved output directory changed")
    if (
        config["execution"]["required_device"] != "mps"
        or config["execution"]["allow_cpu_fallback"] is not False
    ):
        raise ValueError("K5-v2 is MPS-only without fallback")
    if (
        tuple(config["candidates"]["names"]) != CANDIDATES
        or config["candidates"]["primary"] != K5
        or config["candidates"]["learned_one_dimensional_control"] != K5_1D
    ):
        raise ValueError("K5-v2 candidate matrix changed")
    if config["scientific_contract"]["inference_predictor_name"] != K5:
        raise ValueError("K5-v2 inference predictor identity changed")
    if (
        config["features"]["normalization_formula"]
        != ("sqrt_sum_h_norm_V_j_h_squared_over_H_times_dimension")
        or int(config["features"]["history_length"]) != 4
    ):
        raise ValueError("K5-v2 feature normalization/history contract changed")
    if tuple(float(value) for value in config["ridge"]["lambda_grid"]) != (
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    ):
        raise ValueError("K5-v2 ridge grid changed")
    nested = config["nested_monte_carlo"]
    expected_counts = {
        "train_target_child_seeds_exact": 576
        * int(nested["train_target_construction_children"]),
        "calibration_target_child_seeds_exact": 384
        * int(nested["calibration_target_construction_children"]),
        "calibration_evaluation_child_seeds_exact": 384
        * int(nested["calibration_evaluation_children"]),
        "evaluation_target_child_seeds_exact": 576
        * int(nested["evaluation_target_construction_children"]),
        "evaluation_child_seeds_exact": 576 * int(nested["evaluation_children"]),
    }
    if any(
        int(config["gates"][key]) != value for key, value in expected_counts.items()
    ):
        raise ValueError("K5-v2 child-seed count contract changed")
    _v1_provenance(amendment)
    _technical_v2_mps_v1_failure_provenance(amendment, config)


def _verify_lock(lock_path: Path, config_path: Path) -> dict[str, Any]:
    if (
        lock_path.resolve() != DEFAULT_LOCK.resolve()
        or config_path.resolve() != DEFAULT_CONFIG.resolve()
    ):
        raise RuntimeError("K5-v2 production accepts only preregistered paths")
    registry = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "campaign_id",
        "locked_files",
        "dependencies",
        "lock_file_self_hash_embedded",
        "publication_requirement",
    }
    if set(registry) != expected_keys or registry["schema_version"] != 1:
        raise RuntimeError("K5-v2 lock schema mismatch")
    if registry["campaign_id"] != CAMPAIGN_ID:
        raise RuntimeError("K5-v2 lock identity mismatch")
    if (
        set(registry["locked_files"]) != LOCKED_PATHS
        or set(registry["dependencies"]) != DEPENDENCY_PATHS
    ):
        raise RuntimeError("K5-v2 lock inventory mismatch")
    if (
        registry["lock_file_self_hash_embedded"] is not False
        or registry["publication_requirement"] != PUBLICATION_REQUIREMENT
    ):
        raise RuntimeError("K5-v2 publication contract changed")
    for relative, expected in {
        **registry["locked_files"],
        **registry["dependencies"],
    }.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != str(expected):
            raise RuntimeError(f"K5-v2 preregistration hash mismatch: {relative}")
    return {
        "path": str(lock_path.resolve()),
        "sha256": _sha256(lock_path),
        "verified": True,
    }


def _attest(actual: str, published: str, *, name: str) -> dict[str, Any]:
    if str(actual).strip().lower() != str(published).strip().lower():
        raise RuntimeError(f"Published {name} SHA-256 does not match the verified file")
    return {
        "procedural_external_publication_attested": True,
        f"published_{name}_sha256": str(published).strip().lower(),
        "machine_verifies_external_log_itself": False,
    }


def _snapshot_contexts(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    *,
    split: str,
    seeds: Sequence[int],
) -> Iterator[dict[str, Any]]:
    cap = float(config["references"]["total_client_influence_cap"])
    minimum = float(config["features"]["minimum_accepted_mass"])
    for cell in v1._cells(config, seeds):
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
        assessment = {int(value) for value in config["temporal"]["assessment_rounds"]}
        total = int(config["temporal"]["total_rounds_needed_for_frozen_pasts"])
        for round_index in range(1, total + 1):
            if round_index in assessment:
                if enrollment_mean is None:
                    raise RuntimeError(
                        "Enrollment baseline missing before K5-v2 snapshot"
                    )
                source_rounds = tuple(range(round_index - 4, round_index))
                feature, feature_diagnostics = transcript_past_feature_dictionary(
                    torch.stack([residuals[value] for value in source_rounds]),
                    torch.stack([gates[value] for value in source_rounds]),
                    minimum_accepted_mass=minimum,
                    influence_cap=cap,
                    return_diagnostics=True,
                )
                full_v1_feature = v1.transcript_past_feature_dictionary(
                    torch.stack([residuals[value] for value in source_rounds]),
                    torch.stack([gates[value] for value in source_rounds]),
                    minimum_accepted_mass=minimum,
                    influence_cap=cap,
                )
                yield {
                    "split": split,
                    "history_id": v1._history_id(split, cell, round_index),
                    "cell": cell,
                    "round_index": round_index,
                    "components": components,
                    "history": tuple(value.clone() for value in history),
                    "enrollment_mean": enrollment_mean.clone(),
                    "feature": feature,
                    "feature_diagnostics": feature_diagnostics,
                    "full_v1_feature_hash": _tensor_hash(full_v1_feature),
                    "source_rounds": source_rounds,
                    "feature_max_source_round": max(source_rounds),
                }
            observed = k4c._outer_observed(config, components, cell, round_index)
            vectors = clip_l2(
                observed, float(config["aggregation"]["server_clip_norm"])
            )
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
                vectors, components["anchor"], cap
            )
            gates[round_index] = torch.tensor(
                diagnostics["gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )
            history.append(standardized)


def _fit_grid(
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    lambdas: Sequence[float],
) -> tuple[dict[float, torch.Tensor], dict[float, dict[str, Any]]]:
    coefficients: dict[float, torch.Tensor] = {}
    diagnostics: dict[float, dict[str, Any]] = {}
    for value in lambdas:
        theta, diag = fit_shared_scalar_ridge(
            features,
            targets,
            weights,
            ridge_lambda=float(value),
            feature_scales=scales,
            return_diagnostics=True,
        )
        coefficients[float(value)] = theta
        diag["coefficients"] = theta.detach().cpu().tolist()
        diagnostics[float(value)] = diag
    return coefficients, diagnostics


def _finite_flat(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_flat(item) for item in value)
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _single_solution_safe(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    ridge_lambda: float,
    feature_count: int,
    histories: int,
    aggregate_weight_sum: float,
) -> bool:
    try:
        gram = row.get("normalized_gram")
        rhs = row.get("normalized_rhs")
        coefficients = row.get("coefficients")
        scales = row.get("feature_scales")
        return bool(
            torch.device(str(row.get("fit_device"))).type == "mps"
            and row.get("fit_dtype") == "torch.float32"
            and int(row.get("histories", -1)) == histories
            and int(row.get("dimension", -1)) == int(config["cohort"]["dimension"])
            and math.isclose(float(row.get("ridge_lambda")), ridge_lambda, abs_tol=0.0)
            and row.get("condition_number_norm") == "infinity_exact_via_solve"
            and math.isfinite(float(row.get("condition_number_regularized_system")))
            and float(row.get("condition_number_regularized_system"))
            <= float(config["gates"]["regularized_system_condition_number_max"])
            and math.isfinite(float(row.get("normal_equation_relative_residual")))
            and float(row.get("normal_equation_relative_residual"))
            <= float(config["gates"]["normal_equation_relative_residual_max"])
            and math.isclose(
                float(row.get("aggregate_weight_sum")),
                float(aggregate_weight_sum),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and isinstance(gram, list)
            and len(gram) == feature_count
            and all(
                isinstance(line, list) and len(line) == feature_count for line in gram
            )
            and isinstance(rhs, list)
            and len(rhs) == feature_count
            and isinstance(coefficients, list)
            and len(coefficients) == feature_count
            and isinstance(scales, list)
            and len(scales) == feature_count
            and _finite_flat(gram)
            and _finite_flat(rhs)
            and _finite_flat(coefficients)
            and _finite_flat(scales)
            and all(float(value) > 0.0 for value in scales)
            and all(
                math.isfinite(float(row.get(key)))
                for key in (
                    "weighted_mse_before_projection",
                    "ridge_penalty",
                    "objective",
                    "coefficient_l2_norm",
                )
            )
        )
    except (TypeError, ValueError, RuntimeError):
        return False


def _all_grid_solutions_safe(
    diagnostics: Mapping[float, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    lambdas: Sequence[float],
    feature_count: int,
    histories: int,
    aggregate_weight_sum: float,
) -> bool:
    expected = {float(value) for value in lambdas}
    return set(diagnostics) == expected and all(
        _single_solution_safe(
            diagnostics[value],
            config,
            ridge_lambda=value,
            feature_count=feature_count,
            histories=histories,
            aggregate_weight_sum=aggregate_weight_sum,
        )
        for value in expected
    )


def _candidate_references(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    context: Mapping[str, Any],
    values: Mapping[str, Any],
    vectors: torch.Tensor,
    *,
    k5_predictor: torch.Tensor,
    one_dimensional_predictor: torch.Tensor,
    privileged_predictor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rolling = context["feature"][0]
    pointwise = pointwise_optimal_full_imputation_predictor(
        values["clipped"],
        values["gates"],
        values["history_gates"],
        target_direction=values["target_direction"],
        influence_cap=float(config["references"]["total_client_influence_cap"]),
    )
    aware_radii = k4._radii(
        config,
        context["components"]["variances"],
        k2_calibration,
        regime_name=str(context["cell"]["regime"]["name"]),
        blind=False,
    )
    k2_reference, _ = k4._current_reference(
        k4.AWARE,
        vectors,
        anchor=context["components"]["anchor"],
        aware_radii=aware_radii,
        blind_radii=aware_radii,
        config=config,
    )
    fixed = v1._fixed_reference
    return {
        K2: k2_reference,
        K4: values["k4_reference"],
        K4B: fixed(context, values, rolling, config),
        K5_1D: fixed(context, values, one_dimensional_predictor, config),
        K5: fixed(context, values, k5_predictor, config),
        K4C: fixed(context, values, privileged_predictor, config),
        POINTWISE: fixed(context, values, pointwise, config),
    }


def _evaluation_matrix_is_exact(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
) -> bool:
    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    expected_histories = {
        v1._history_id("evaluation", cell, int(round_index))
        for cell in v1._cells(config, seeds)
        for round_index in config["temporal"]["assessment_rounds"]
    }
    observed_histories = [str(row["history_id"]) for row in history_rows]
    if (
        len(observed_histories) != len(expected_histories)
        or set(observed_histories) != expected_histories
    ):
        return False
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    expected = {
        (history_id, candidate, child)
        for history_id in expected_histories
        for candidate in CANDIDATES
        for child in range(children)
    }
    observed = [
        (str(row["history_id"]), str(row["candidate"]), int(row["evaluation_child"]))
        for row in child_rows
    ]
    return len(observed) == len(expected) and set(observed) == expected


def fit_freeze(
    config_path: Path,
    lock_path: Path,
    output: Path,
    *,
    published_lock_sha256: str,
) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("K5-v2 fit accepts only the preregistered output directory")
    config, amendment = _load_amended_config(config_path)
    _validate_config(config, amendment)
    provenance = _v1_provenance(amendment)
    technical_rerun_provenance = _technical_v2_mps_v1_failure_provenance(
        amendment, config
    )
    lock = _verify_lock(lock_path, config_path)
    publication = _attest(lock["sha256"], published_lock_sha256, name="lock")
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError("K5-v2 fit refuses CPU, fallback, or non-float32 runtime")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite K5-v2 directory: {output}")
    output.mkdir(parents=True)
    base_config_path = ROOT / amendment["base_protocol"]["config_path"]
    _write_json(output / "resolved_config.json", config)
    amendment_config_sha256 = _sha256(config_path)
    base_config_sha256 = _sha256(base_config_path)
    resolved_config_sha256 = _sha256(output / "resolved_config.json")
    k2_calibration, temporal_calibration, calibration_provenance = (
        v1._load_calibrations(config)
    )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "fit_running_evaluation_forbidden",
        "device": str(oracle._RUNTIME_DEVICE),
        "dtype": str(oracle._RUNTIME_DTYPE),
        "holdout_opened": False,
        "evaluation_trajectory_count_generated": 0,
        "preregistration_lock": lock,
        "lock_publication_attestation": publication,
        "amendment_config_sha256": amendment_config_sha256,
        "base_config_sha256": base_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "v1_pre_evaluation_provenance": provenance,
        "technical_v2_mps_v1_failure_provenance": technical_rerun_provenance,
        "calibration_provenance": calibration_provenance,
    }
    _write_json(output / "manifest.json", manifest)

    train_seeds = [int(value) for value in config["randomness"]["train_outer_seeds"]]
    calibration_seeds = [
        int(value) for value in config["randomness"]["calibration_outer_seeds"]
    ]
    lambdas = [float(value) for value in config["ridge"]["lambda_grid"]]
    v1_rows = {
        row["history_id"]: row
        for row in _read_csv(V1_RESULTS / "fit_history_rows.csv")
        if row["split"] == "train"
    }
    v1_registry = json.loads(
        (V1_RESULTS / "fit_rng_registry.json").read_text(encoding="utf-8")
    )
    fit_rows: list[dict[str, Any]] = []
    rng_registry: dict[str, list[int]] = {
        "train_target": [],
        "calibration_target": [],
        "calibration_evaluation": [],
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    train_features: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    train_weights: list[float] = []
    train_seed_ids: list[int] = []
    train_reuse_checks: list[bool] = []
    train_progress_seed: int | None = None
    train_progress_completed = 0
    for context in _snapshot_contexts(
        config,
        k2_calibration,
        temporal_calibration,
        split="train",
        seeds=train_seeds,
    ):
        current_seed = int(context["cell"]["seed"])
        if train_progress_seed is not None and current_seed != train_progress_seed:
            train_progress_completed += 1
            _progress(
                "fit_train",
                completed=train_progress_completed,
                total=len(train_seeds),
                outer_seed=train_progress_seed,
            )
        train_progress_seed = current_seed
        target = v1._privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="train_target",
            children=int(
                config["nested_monte_carlo"]["train_target_construction_children"]
            ),
            split_diagnostic=False,
        )
        missing = float(target["missing_slot_mass"])
        if missing <= 0.0:
            raise RuntimeError("K5-v2 training requires positive missing-slot mass")
        expected = v1_rows.get(context["history_id"])
        reused_exactly = bool(
            expected
            and context["full_v1_feature_hash"] == expected["feature_hash"]
            and _tensor_hash(target["predictor"]) == expected["privileged_target_hash"]
        )
        train_reuse_checks.append(reused_exactly)
        train_features.append(context["feature"])
        train_targets.append(target["predictor"])
        train_weights.append((missing / float(config["cohort"]["num_clients"])) ** 2)
        train_seed_ids.append(int(context["cell"]["seed"]))
        rng_registry["train_target"].extend(target["child_seeds"])
        fit_rows.append(
            {
                "split": "train",
                "history_id": context["history_id"],
                "seed": int(context["cell"]["seed"]),
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "feature_source_rounds": ",".join(
                    str(value) for value in context["source_rounds"]
                ),
                "missing_slot_mass": missing,
                "feature_hash": _tensor_hash(context["feature"]),
                "full_v1_feature_hash": context["full_v1_feature_hash"],
                "privileged_target_hash": _tensor_hash(target["predictor"]),
                "privileged_target_norm": float(target["predictor_norm"]),
                "v1_train_row_reproduced": reused_exactly,
                "inference_payload_forbidden_current_fields": (
                    forbidden_current_field_count(
                        {
                            "past_features": True,
                            "feature_scales": True,
                            "coefficients": True,
                        }
                    )
                ),
            }
        )
    if train_progress_seed is not None:
        train_progress_completed += 1
        _progress(
            "fit_train",
            completed=train_progress_completed,
            total=len(train_seeds),
            outer_seed=train_progress_seed,
        )
    train_x = torch.stack(train_features)
    train_y = torch.stack(train_targets)
    train_balanced_weights = v1._seed_balanced_weights(train_weights, train_seed_ids)
    train_w = torch.tensor(
        train_balanced_weights, device=oracle._RUNTIME_DEVICE, dtype=torch.float32
    )
    rms_floor = float(config["features"]["rms_floor"])
    train_scales, train_design = training_feature_scales(
        train_x, rms_floor=rms_floor, return_diagnostics=True
    )
    grid_theta, grid_fit = _fit_grid(train_x, train_y, train_w, train_scales, lambdas)
    grid_1d_theta, grid_1d_fit = _fit_grid(
        train_x[:, :1], train_y, train_w, train_scales[:1], lambdas
    )

    calibration_features: list[torch.Tensor] = []
    calibration_targets: list[torch.Tensor] = []
    calibration_weights: list[float] = []
    calibration_seed_ids: list[int] = []
    calibration_sum: dict[tuple[float, int], float] = defaultdict(float)
    calibration_count: dict[tuple[float, int], int] = defaultdict(int)
    calibration_1d_sum: dict[tuple[float, int], float] = defaultdict(float)
    calibration_1d_count: dict[tuple[float, int], int] = defaultdict(int)
    calibration_progress_seed: int | None = None
    calibration_progress_completed = 0
    for context in _snapshot_contexts(
        config,
        k2_calibration,
        temporal_calibration,
        split="calibration",
        seeds=calibration_seeds,
    ):
        current_seed = int(context["cell"]["seed"])
        if (
            calibration_progress_seed is not None
            and current_seed != calibration_progress_seed
        ):
            calibration_progress_completed += 1
            _progress(
                "fit_calibration",
                completed=calibration_progress_completed,
                total=len(calibration_seeds),
                outer_seed=calibration_progress_seed,
            )
        calibration_progress_seed = current_seed
        target = v1._privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="calibration_target",
            children=int(
                config["nested_monte_carlo"]["calibration_target_construction_children"]
            ),
            split_diagnostic=False,
        )
        missing = float(target["missing_slot_mass"])
        if missing <= 0.0:
            raise RuntimeError("K5-v2 calibration requires positive missing-slot mass")
        calibration_features.append(context["feature"])
        calibration_targets.append(target["predictor"])
        calibration_weights.append(
            (missing / float(config["cohort"]["num_clients"])) ** 2
        )
        seed = int(context["cell"]["seed"])
        calibration_seed_ids.append(seed)
        rng_registry["calibration_target"].extend(target["child_seeds"])
        predictors = {
            value: transcript_past_predictor(
                context["feature"],
                grid_theta[value],
                influence_cap=float(config["references"]["total_client_influence_cap"]),
                feature_scales=train_scales,
            )
            for value in lambdas
        }
        predictors_1d = {
            value: transcript_past_predictor(
                context["feature"][:1],
                grid_1d_theta[value],
                influence_cap=float(config["references"]["total_client_influence_cap"]),
                feature_scales=train_scales[:1],
            )
            for value in lambdas
        }
        for child in range(
            int(config["nested_monte_carlo"]["calibration_evaluation_children"])
        ):
            values, _, child_seed = v1._child_values(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                stream="calibration_evaluation",
                child=child,
            )
            rng_registry["calibration_evaluation"].append(child_seed)
            for value in lambdas:
                reference = v1._fixed_reference(
                    context, values, predictors[value], config
                )
                reference_1d = v1._fixed_reference(
                    context, values, predictors_1d[value], config
                )
                calibration_sum[(value, seed)] += float(
                    torch.sum((reference - values["target"]).square()).item()
                )
                calibration_count[(value, seed)] += 1
                calibration_1d_sum[(value, seed)] += float(
                    torch.sum((reference_1d - values["target"]).square()).item()
                )
                calibration_1d_count[(value, seed)] += 1
        fit_rows.append(
            {
                "split": "calibration",
                "history_id": context["history_id"],
                "seed": seed,
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "feature_source_rounds": ",".join(
                    str(value) for value in context["source_rounds"]
                ),
                "missing_slot_mass": missing,
                "feature_hash": _tensor_hash(context["feature"]),
                "privileged_target_hash": _tensor_hash(target["predictor"]),
                "privileged_target_norm": float(target["predictor_norm"]),
                "v1_train_row_reproduced": False,
                "inference_payload_forbidden_current_fields": 0,
            }
        )
    if calibration_progress_seed is not None:
        calibration_progress_completed += 1
        _progress(
            "fit_calibration",
            completed=calibration_progress_completed,
            total=len(calibration_seeds),
            outer_seed=calibration_progress_seed,
        )

    tolerance = float(config["ridge"]["selection_tie_absolute_tolerance"])
    selected, calibration_rows = v1._select_lambda(
        lambdas,
        calibration_sum,
        calibration_count,
        calibration_seeds,
        tolerance=tolerance,
    )
    selected_1d, calibration_1d_rows = v1._select_lambda(
        lambdas,
        calibration_1d_sum,
        calibration_1d_count,
        calibration_seeds,
        tolerance=tolerance,
    )
    calibration_x = torch.stack(calibration_features)
    calibration_y = torch.stack(calibration_targets)
    combined_x = torch.cat((train_x, calibration_x), dim=0)
    combined_y = torch.cat((train_y, calibration_y), dim=0)
    combined_balanced_weights = v1._seed_balanced_weights(
        train_weights + calibration_weights,
        train_seed_ids + calibration_seed_ids,
    )
    combined_w = torch.tensor(
        combined_balanced_weights,
        device=oracle._RUNTIME_DEVICE,
        dtype=torch.float32,
    )
    combined_scales, combined_design = training_feature_scales(
        combined_x, rms_floor=rms_floor, return_diagnostics=True
    )
    final_theta, final_fit = fit_shared_scalar_ridge(
        combined_x,
        combined_y,
        combined_w,
        ridge_lambda=selected,
        feature_scales=combined_scales,
        return_diagnostics=True,
    )
    final_1d_theta, final_1d_fit = fit_shared_scalar_ridge(
        combined_x[:, :1],
        combined_y,
        combined_w,
        ridge_lambda=selected_1d,
        feature_scales=combined_scales[:1],
        return_diagnostics=True,
    )
    final_fit["coefficients"] = final_theta.detach().cpu().tolist()
    final_1d_fit["coefficients"] = final_1d_theta.detach().cpu().tolist()
    all_fit_rng = [
        int(value)
        for key in ("train_target", "calibration_target", "calibration_evaluation")
        for value in rng_registry[key]
    ]
    expected_streams = {
        "train_target": _expected_rng_stream_records(
            config,
            split="train",
            seeds=train_seeds,
            stream="train_target",
            children=int(
                config["nested_monte_carlo"]["train_target_construction_children"]
            ),
        ),
        "calibration_target": _expected_rng_stream_records(
            config,
            split="calibration",
            seeds=calibration_seeds,
            stream="calibration_target",
            children=int(
                config["nested_monte_carlo"]["calibration_target_construction_children"]
            ),
        ),
        "calibration_evaluation": _expected_rng_stream_records(
            config,
            split="calibration",
            seeds=calibration_seeds,
            stream="calibration_evaluation",
            children=int(
                config["nested_monte_carlo"]["calibration_evaluation_children"]
            ),
        ),
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    fit_rng_registry = _rng_registry(expected_streams)
    actual_rng_matches_derivation = all(
        rng_registry[key] == _flatten_rng_records(expected_streams[key])
        for key in ("train_target", "calibration_target", "calibration_evaluation")
    )
    _write_json(output / "fit_rng_registry.json", fit_rng_registry)
    persisted_fit_rng_registry = json.loads(
        (output / "fit_rng_registry.json").read_text(encoding="utf-8")
    )
    fit_rng_hash = str(fit_rng_registry["payload_sha256"])
    gates = config["gates"]
    fit_checks = {
        "device_mps": oracle._RUNTIME_DEVICE.type == "mps",
        "fit_dtype_float32": final_theta.dtype == torch.float32,
        "fit_device_type_mps": final_theta.device.type == "mps",
        "train_histories_exact": len(train_features)
        == int(gates["train_histories_exact"]),
        "calibration_histories_exact": len(calibration_features)
        == int(gates["calibration_histories_exact"]),
        "train_matrix_exact": v1._fit_history_matrix_is_exact(
            config, fit_rows, split="train", seeds=train_seeds
        ),
        "calibration_matrix_exact": v1._fit_history_matrix_is_exact(
            config, fit_rows, split="calibration", seeds=calibration_seeds
        ),
        "v1_train_data_reproduced_exactly": len(train_reuse_checks) == 576
        and all(train_reuse_checks)
        and rng_registry["train_target"] == v1_registry["train_target"],
        "feature_rank_train_exact_five": int(train_design["flattened_design_rank"])
        == 5,
        "feature_rank_refit_exact_five": int(combined_design["flattened_design_rank"])
        == 5,
        "feature_rank_computed_on_mps": torch.device(
            train_design["rank_diagnostics"]["compute_device"]
        ).type
        == "mps"
        and torch.device(combined_design["rank_diagnostics"]["compute_device"]).type
        == "mps",
        "feature_scale_floor_inactive": int(train_design["floor_active_count"]) == 0
        and int(combined_design["floor_active_count"]) == 0,
        "minimum_centered_rms_positive": float(train_design["minimum_centered_rms"])
        > 0.0
        and float(combined_design["minimum_centered_rms"]) > 0.0,
        "features_strictly_past": all(
            int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1
            for row in fit_rows
        ),
        "feature_source_rounds_exact": all(
            row["feature_source_rounds"]
            == ",".join(
                str(value)
                for value in range(
                    int(row["assessment_round"]) - 4,
                    int(row["assessment_round"]),
                )
            )
            for row in fit_rows
        ),
        "forbidden_current_fields_zero": all(
            int(row["inference_payload_forbidden_current_fields"]) == 0
            for row in fit_rows
        ),
        "fit_rng_counts_exact": len(rng_registry["train_target"])
        == int(gates["train_target_child_seeds_exact"])
        and len(rng_registry["calibration_target"])
        == int(gates["calibration_target_child_seeds_exact"])
        and len(rng_registry["calibration_evaluation"])
        == int(gates["calibration_evaluation_child_seeds_exact"]),
        "fit_rng_globally_unique": len(all_fit_rng) == len(set(all_fit_rng)),
        "fit_rng_each_seed_matches_native_derivation": actual_rng_matches_derivation,
        "fit_rng_structured_registry_round_trip": _validate_rng_registry(
            persisted_fit_rng_registry, expected_streams
        ),
        "evaluation_rng_absent": rng_registry["evaluation_target"] == []
        and rng_registry["evaluation"] == [],
        "holdout_rng_absent": rng_registry["holdout"] == [],
        "selected_lambdas_in_grid": selected in lambdas and selected_1d in lambdas,
        "calibration_grid_counts_exact": all(
            calibration_count[(value, seed)] == 48 * 64
            and calibration_1d_count[(value, seed)] == 48 * 64
            for value in lambdas
            for seed in calibration_seeds
        ),
        "calibration_grid_metrics_finite": all(
            math.isfinite(float(row["equal_seed_mean_projected_aggregate_mse"]))
            and len(row["calibration_seed_mse"]) == 8
            and _finite_flat(row["calibration_seed_mse"])
            for row in calibration_rows + calibration_1d_rows
        ),
        "all_primary_grid_solutions_safe": _all_grid_solutions_safe(
            grid_fit,
            config,
            lambdas=lambdas,
            feature_count=5,
            histories=576,
            aggregate_weight_sum=12.0,
        ),
        "all_one_dimensional_grid_solutions_safe": _all_grid_solutions_safe(
            grid_1d_fit,
            config,
            lambdas=lambdas,
            feature_count=1,
            histories=576,
            aggregate_weight_sum=12.0,
        ),
        "final_primary_solution_safe": _single_solution_safe(
            final_fit,
            config,
            ridge_lambda=selected,
            feature_count=5,
            histories=960,
            aggregate_weight_sum=20.0,
        ),
        "final_one_dimensional_solution_safe": _single_solution_safe(
            final_1d_fit,
            config,
            ridge_lambda=selected_1d,
            feature_count=1,
            histories=960,
            aggregate_weight_sum=20.0,
        ),
    }
    predictor = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "candidate": K5,
        "feature_names": list(FEATURE_NAMES),
        "removed_v1_feature": REMOVED_V1_FEATURE,
        "feature_normalization_formula": "sqrt(sum_h ||V_j,h||_2^2 / (H*d))",
        "feature_scales": combined_scales.detach().cpu().tolist(),
        "coefficients": final_theta.detach().cpu().tolist(),
        "selected_lambda": selected,
        "one_dimensional_control": {
            "feature_name": FEATURE_NAMES[0],
            "feature_scale": float(combined_scales[0].item()),
            "coefficient": float(final_1d_theta[0].item()),
            "selected_lambda": selected_1d,
        },
        "influence_cap": float(config["references"]["total_client_influence_cap"]),
        "public_cohort_size": int(config["cohort"]["num_clients"]),
        "fit_split": "v1_train_seeds_plus_new_v2_calibration_after_selection",
        "lambda_selection_coefficients_fit_on": "v1_train_outer_seeds_only",
        "observable_past_only_at_inference": True,
        "privileged_training_supervision": "k4c_ch_synthetic_targets",
        "amendment_config_sha256": amendment_config_sha256,
        "base_config_sha256": base_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "lock_sha256": lock["sha256"],
        "fit_rng_registry_structured_sha256": fit_rng_hash,
        "v1_provenance_sha256": _canonical_hash(provenance),
        "evaluation_generated_before_freeze": False,
        "holdout_opened": False,
    }
    predictor_path = output / "frozen_predictor.json"
    _write_json(predictor_path, predictor)
    predictor_sha = _sha256(predictor_path)
    fit_decision = {
        "all_validity_checks_pass": all(fit_checks.values()),
        "checks": fit_checks,
        "decision": (
            "predictor_frozen_evaluation_locked_pending_external_hash_publication"
            if all(fit_checks.values())
            else "invalid_fit_evaluation_forbidden"
        ),
        "selected_lambda": selected,
        "selected_lambda_one_dimensional": selected_1d,
        "fit_rng_registry_structured_sha256": fit_rng_hash,
        "frozen_predictor_sha256": predictor_sha,
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    _write_csv(output / "fit_history_rows.csv", fit_rows)
    _write_json(
        output / "lambda_calibration.json",
        {
            "primary": calibration_rows,
            "one_dimensional": calibration_1d_rows,
            "tie_absolute_tolerance": tolerance,
            "tie_break": "largest_lambda",
            "calibration_seed_ids": calibration_seeds,
            "train_grid_fit_diagnostics": {
                str(key): value for key, value in grid_fit.items()
            },
            "train_grid_1d_fit_diagnostics": {
                str(key): value for key, value in grid_1d_fit.items()
            },
        },
    )
    _write_json(
        output / "feature_design_diagnostics.json",
        {
            "train": train_design,
            "train_plus_calibration": combined_design,
            "final_fit": final_fit,
            "final_1d_fit": final_1d_fit,
            "weight_audit": {
                "raw_train_sum_by_seed": v1._weights_by_seed(
                    train_weights, train_seed_ids
                ),
                "raw_train_plus_calibration_sum_by_seed": v1._weights_by_seed(
                    train_weights + calibration_weights,
                    train_seed_ids + calibration_seed_ids,
                ),
                "balanced_train_sum_by_seed": v1._weights_by_seed(
                    train_balanced_weights, train_seed_ids
                ),
                "balanced_train_plus_calibration_sum_by_seed": v1._weights_by_seed(
                    combined_balanced_weights,
                    train_seed_ids + calibration_seed_ids,
                ),
                "normalization_rule": "each_outer_seed_sums_to_one",
            },
        },
    )
    _write_json(
        output / "fit_sufficient_statistics.json",
        {
            "schema_version": 1,
            "normalization": "dimension_times_sum_seed_balanced_weights",
            "train": {
                "histories": 576,
                "dimension": int(config["cohort"]["dimension"]),
                "feature_count": 5,
                "feature_scales": train_scales.detach().cpu().tolist(),
                "normalized_gram": grid_fit[lambdas[0]]["normalized_gram"],
                "normalized_rhs": grid_fit[lambdas[0]]["normalized_rhs"],
                "aggregate_weight_sum": float(train_w.sum().item()),
                "raw_weight_sum_by_seed": v1._weights_by_seed(
                    train_weights, train_seed_ids
                ),
                "balanced_weight_sum_by_seed": v1._weights_by_seed(
                    train_balanced_weights, train_seed_ids
                ),
            },
            "train_plus_calibration": {
                "histories": 960,
                "dimension": int(config["cohort"]["dimension"]),
                "feature_count": 5,
                "feature_scales": combined_scales.detach().cpu().tolist(),
                "normalized_gram": final_fit["normalized_gram"],
                "normalized_rhs": final_fit["normalized_rhs"],
                "aggregate_weight_sum": float(combined_w.sum().item()),
                "raw_weight_sum_by_seed": v1._weights_by_seed(
                    train_weights + calibration_weights,
                    train_seed_ids + calibration_seed_ids,
                ),
                "balanced_weight_sum_by_seed": v1._weights_by_seed(
                    combined_balanced_weights,
                    train_seed_ids + calibration_seed_ids,
                ),
            },
            "one_dimensional": {
                "train_normalized_gram": grid_1d_fit[lambdas[0]]["normalized_gram"],
                "train_normalized_rhs": grid_1d_fit[lambdas[0]]["normalized_rhs"],
                "train_plus_calibration_normalized_gram": final_1d_fit[
                    "normalized_gram"
                ],
                "train_plus_calibration_normalized_rhs": final_1d_fit["normalized_rhs"],
            },
            "lambda_grid": lambdas,
            "selected_lambda": selected,
            "selected_lambda_one_dimensional": selected_1d,
            "calibration_primary_seed_mse_by_lambda": calibration_rows,
            "calibration_one_dimensional_seed_mse_by_lambda": (calibration_1d_rows),
        },
    )
    _write_json(output / "fit_decision.json", fit_decision)
    manifest.update(
        {
            "status": (
                "fit_completed_evaluation_locked"
                if all(fit_checks.values())
                else "fit_invalid_evaluation_forbidden"
            ),
            "fit_validity_pass": all(fit_checks.values()),
            "frozen_predictor_sha256": predictor_sha,
            "fit_rng_registry_structured_sha256": fit_rng_hash,
            "evaluation_trajectory_count_generated": 0,
        }
    )
    _write_json(output / "manifest.json", manifest)
    return fit_decision


_PREDICTOR_KEYS = {
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


def _validate_frozen_predictor(
    predictor: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
    amendment_config_sha256: str,
    resolved_config_sha256: str,
    lock_sha256: str,
) -> None:
    """Fail closed on every deployable field of the frozen predictor."""

    if set(predictor) != _PREDICTOR_KEYS:
        raise RuntimeError("Frozen K5-v2 predictor schema changed")
    if (
        predictor["schema_version"] != 1
        or predictor["campaign_id"] != CAMPAIGN_ID
        or predictor["candidate"] != K5
        or tuple(predictor["feature_names"]) != FEATURE_NAMES
        or predictor["removed_v1_feature"] != REMOVED_V1_FEATURE
    ):
        raise RuntimeError("Frozen K5-v2 predictor identity changed")
    if predictor["feature_normalization_formula"] != (
        "sqrt(sum_h ||V_j,h||_2^2 / (H*d))"
    ):
        raise RuntimeError("Frozen K5-v2 feature normalization changed")
    coefficients = predictor["coefficients"]
    scales = predictor["feature_scales"]
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != 5
        or not _finite_flat(coefficients)
        or not isinstance(scales, list)
        or len(scales) != 5
        or not _finite_flat(scales)
        or not all(float(value) > 0.0 for value in scales)
    ):
        raise RuntimeError("Frozen K5-v2 coefficients/scales are invalid")
    grid = {float(value) for value in config["ridge"]["lambda_grid"]}
    if (
        not math.isfinite(float(predictor["selected_lambda"]))
        or float(predictor["selected_lambda"]) not in grid
    ):
        raise RuntimeError("Frozen K5-v2 lambda is outside the locked grid")
    one = predictor["one_dimensional_control"]
    if set(one) != {
        "feature_name",
        "feature_scale",
        "coefficient",
        "selected_lambda",
    }:
        raise RuntimeError("Frozen K5-v2 one-dimensional schema changed")
    if (
        one["feature_name"] != FEATURE_NAMES[0]
        or not math.isfinite(float(one["feature_scale"]))
        or float(one["feature_scale"]) <= 0.0
        or not math.isfinite(float(one["coefficient"]))
        or float(one["selected_lambda"]) not in grid
    ):
        raise RuntimeError("Frozen K5-v2 one-dimensional control is invalid")
    if not math.isclose(
        float(predictor["influence_cap"]),
        float(config["references"]["total_client_influence_cap"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ) or int(predictor["public_cohort_size"]) != int(config["cohort"]["num_clients"]):
        raise RuntimeError("Frozen K5-v2 public cap/denominator changed")
    expected_literals = {
        "fit_split": "v1_train_seeds_plus_new_v2_calibration_after_selection",
        "lambda_selection_coefficients_fit_on": "v1_train_outer_seeds_only",
        "privileged_training_supervision": "k4c_ch_synthetic_targets",
    }
    if any(predictor[key] != value for key, value in expected_literals.items()):
        raise RuntimeError("Frozen K5-v2 fitting provenance changed")
    if (
        predictor["observable_past_only_at_inference"] is not True
        or predictor["evaluation_generated_before_freeze"] is not False
        or predictor["holdout_opened"] is not False
    ):
        raise RuntimeError("Frozen K5-v2 phase/inference declaration is invalid")
    base_sha = amendment["base_protocol"]["config_sha256"]
    provenance_sha = _canonical_hash(_v1_provenance(amendment))
    expected_hashes = {
        "amendment_config_sha256": amendment_config_sha256,
        "base_config_sha256": base_sha,
        "resolved_config_sha256": resolved_config_sha256,
        "lock_sha256": lock_sha256,
        "v1_provenance_sha256": provenance_sha,
    }
    if any(predictor[key] != value for key, value in expected_hashes.items()):
        raise RuntimeError("Frozen K5-v2 hash provenance mismatch")
    if (
        not isinstance(predictor["fit_rng_registry_structured_sha256"], str)
        or len(predictor["fit_rng_registry_structured_sha256"]) != 64
    ):
        raise RuntimeError("Frozen K5-v2 RNG registry hash is malformed")


def _verify_independent_fit_audit(
    output: Path, published_fit_audit_sha256: str
) -> dict[str, Any]:
    """Bind evaluation to the published audit and all five audited artifacts."""

    audit_path = output / "independent_fit_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Independent K5-v2 fit audit artifact is missing")
    audit_sha256 = _sha256(audit_path)
    publication = _attest(audit_sha256, published_fit_audit_sha256, name="fit_audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audited = audit.get("audited_artifact_sha256")
    if not isinstance(audited, Mapping) or set(audited) != set(
        FIT_AUDITED_ARTIFACT_NAMES
    ):
        raise RuntimeError("Independent K5-v2 fit audit hash inventory is incomplete")
    current = {
        name: _sha256(output / name)
        for name in FIT_AUDITED_ARTIFACT_NAMES
        if (output / name).is_file()
    }
    checks = {
        "schema": audit.get("schema_version") == 1,
        "campaign": audit.get("campaign_id") == CAMPAIGN_ID,
        "scope": audit.get("audit_scope") == "fit_and_freeze_pre_evaluation",
        "independent": audit.get("independent_of_v2_runner") is True,
        "audit_passed": audit.get("all_checks_pass") is True
        and all(bool(value) for value in audit.get("checks", {}).values()),
        "fit_artifact_inventory_exact": set(current) == set(FIT_AUDITED_ARTIFACT_NAMES),
        "all_fit_artifact_hashes_unchanged": current == dict(audited),
        "holdout_closed": audit.get("holdout_opened") is False,
        "evaluation_absent_when_fit_audited": audit.get("evaluation_present") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Independent K5-v2 fit audit or audited artifacts changed: {checks}"
        )
    return {
        "schema_version": 1,
        "path": str(audit_path.resolve()),
        "sha256": audit_sha256,
        "verified_before_evaluation": True,
        "pre_evaluation_external_anchor_required": True,
        "publication_attestation": publication,
        "self_hash_embedded_in_fit_artifacts": False,
        "cycle_avoidance": "audit_hash_attested_in_evaluation_manifest_only",
        "anti_replay_limit": (
            "runtime_verifies_the_caller_supplied_published_sha256_but_does_not_"
            "query_or_timestamp_the_external_publication_record"
        ),
        "audited_artifact_sha256": current,
        "checks": checks,
    }


def _load_frozen_predictor(
    output: Path,
    published_predictor_sha256: str,
    published_fit_audit_sha256: str,
    *,
    config: Mapping[str, Any],
    amendment: Mapping[str, Any],
    config_path: Path,
    lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "fit_decision.json").read_text(encoding="utf-8"))
    if (
        manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("status") != "fit_completed_evaluation_locked"
        or manifest.get("fit_validity_pass") is not True
        or decision.get("all_validity_checks_pass") is not True
        or not all(bool(value) for value in decision.get("checks", {}).values())
    ):
        raise RuntimeError("K5-v2 evaluation requires a valid completed fit")
    if (
        int(manifest.get("evaluation_trajectory_count_generated", -1)) != 0
        or decision.get("evaluation_trajectory_count_generated") != 0
        or manifest.get("holdout_opened") is not False
        or decision.get("holdout_opened") is not False
        or (output / "evaluation").exists()
    ):
        raise RuntimeError("K5-v2 evaluation/holdout was opened before publication")
    resolved_path = output / "resolved_config.json"
    resolved_sha = _sha256(resolved_path)
    if (
        resolved_sha != manifest.get("resolved_config_sha256")
        or json.loads(resolved_path.read_text(encoding="utf-8")) != config
    ):
        raise RuntimeError("K5-v2 resolved config changed after fit")
    predictor_path = output / "frozen_predictor.json"
    actual = _sha256(predictor_path)
    if actual != manifest.get("frozen_predictor_sha256") or actual != decision.get(
        "frozen_predictor_sha256"
    ):
        raise RuntimeError("Frozen K5-v2 predictor changed after fit")
    publication = _attest(actual, published_predictor_sha256, name="predictor")
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    _validate_frozen_predictor(
        predictor,
        config=config,
        amendment=amendment,
        amendment_config_sha256=_sha256(config_path),
        resolved_config_sha256=resolved_sha,
        lock_sha256=lock_sha256,
    )
    fit_streams = {
        "train_target": _expected_rng_stream_records(
            config,
            split="train",
            seeds=config["randomness"]["train_outer_seeds"],
            stream="train_target",
            children=int(
                config["nested_monte_carlo"]["train_target_construction_children"]
            ),
        ),
        "calibration_target": _expected_rng_stream_records(
            config,
            split="calibration",
            seeds=config["randomness"]["calibration_outer_seeds"],
            stream="calibration_target",
            children=int(
                config["nested_monte_carlo"]["calibration_target_construction_children"]
            ),
        ),
        "calibration_evaluation": _expected_rng_stream_records(
            config,
            split="calibration",
            seeds=config["randomness"]["calibration_outer_seeds"],
            stream="calibration_evaluation",
            children=int(
                config["nested_monte_carlo"]["calibration_evaluation_children"]
            ),
        ),
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    registry_path = output / "fit_rng_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        not _validate_rng_registry(registry, fit_streams)
        or registry["payload_sha256"] != predictor["fit_rng_registry_structured_sha256"]
        or registry["payload_sha256"]
        != decision.get("fit_rng_registry_structured_sha256")
        or registry["payload_sha256"]
        != manifest.get("fit_rng_registry_structured_sha256")
    ):
        raise RuntimeError("K5-v2 structured fit RNG registry is invalid")
    fit_audit_attestation = _verify_independent_fit_audit(
        output, published_fit_audit_sha256
    )
    return predictor, publication, fit_audit_attestation


def _evaluation_noise_composition_is_exact(
    config: Mapping[str, Any], history_rows: Sequence[Mapping[str, Any]]
) -> bool:
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        for regime, permutation in v1._noise_cells(config):
            count = sum(
                int(row["seed"]) == int(seed)
                and str(row["noise_regime"]) == str(regime["name"])
                and str(row["noise_permutation"]) == str(permutation)
                for row in history_rows
            )
            if count != 16:
                return False
    return True


def _ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator == 0.0
    ):
        return float("nan")
    return numerator / denominator


def _summarize_evaluation(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
    replace_rows: Sequence[Mapping[str, Any]],
    validity_extra: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_history: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    meta = {str(row["history_id"]): row for row in history_rows}
    for row in child_rows:
        by_history[str(row["history_id"])][str(row["candidate"])].append(
            float(row["squared_reference_error"])
        )
    seed_rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        ids = [key for key, row in meta.items() if int(row["seed"]) == int(seed)]
        sums: dict[str, float] = defaultdict(float)
        regime_sums: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        split_sq = 0.0
        for history_id in ids:
            means = {
                candidate: _mean(by_history[history_id][candidate])
                for candidate in CANDIDATES
            }
            regime = str(meta[history_id]["noise_regime"])
            for candidate, value in means.items():
                sums[candidate] += value
                regime_sums[regime][candidate] += value
            split_sq += (
                float(
                    meta[history_id]["privileged_target_split_aggregate_scale_distance"]
                )
                ** 2
            )
        headroom = sums[K4] - sums[K4C]
        near_zero = headroom <= 1e-12 * max(1.0, abs(sums[K4]))
        relative_headroom = _ratio(headroom, sums[K4])
        split_over_headroom = _ratio(split_sq, headroom)
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
                "split_mc_over_headroom": split_over_headroom,
                "homogeneous_gain_vs_k4b": _ratio(
                    regime_sums["homogeneous"][K4B] - regime_sums["homogeneous"][K5],
                    regime_sums["homogeneous"][K4B],
                ),
                "heteroscedastic_gain_vs_k4b": _ratio(
                    regime_sums["heteroscedastic"][K4B]
                    - regime_sums["heteroscedastic"][K5],
                    regime_sums["heteroscedastic"][K4B],
                ),
                "privileged_target_split_disagreement_mse_ratio": _ratio(
                    split_sq, sums[K4]
                ),
            }
        )
    ci_keys = (
        "gain_vs_k4b",
        "gain_vs_k4",
        "gain_vs_one_dimensional",
        "ch_capture_fraction",
        "homogeneous_gain_vs_k4b",
        "heteroscedastic_gain_vs_k4b",
    )
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    cis: dict[str, dict[str, Any]] = {}
    for key in ci_keys:
        values = [float(row[key]) for row in seed_rows]
        cis[key] = (
            _ci(values, tcrit)
            if len(values) == 12 and all(math.isfinite(value) for value in values)
            else {
                "n": len(values),
                "mean": float("nan"),
                "low": float("nan"),
                "high": float("nan"),
            }
        )
    relative = [float(row["relative_k4_k4c_headroom"]) for row in seed_rows]
    split_headroom = [float(row["split_mc_over_headroom"]) for row in seed_rows]
    finite_relative = [value for value in relative if math.isfinite(value)]
    finite_split_headroom = [value for value in split_headroom if math.isfinite(value)]
    headroom_diagnostics = {
        "per_seed": [
            {
                "seed": row["seed"],
                "relative_headroom": row["relative_k4_k4c_headroom"],
                "split_mc_over_headroom": row["split_mc_over_headroom"],
                "numerically_near_zero": row["headroom_numerically_near_zero"],
                "small_relative_warning": row["headroom_small_relative_warning"],
            }
            for row in seed_rows
        ],
        "relative_headroom_minimum": min(finite_relative, default=float("nan")),
        "relative_headroom_median": (
            statistics.median(finite_relative) if finite_relative else float("nan")
        ),
        "split_mc_over_headroom_minimum": min(
            finite_split_headroom, default=float("nan")
        ),
        "split_mc_over_headroom_median": (
            statistics.median(finite_split_headroom)
            if finite_split_headroom
            else float("nan")
        ),
        "split_mc_over_headroom_maximum": max(
            finite_split_headroom, default=float("nan")
        ),
        "numerically_near_zero_seed_count": sum(
            bool(row["headroom_numerically_near_zero"]) for row in seed_rows
        ),
        "small_relative_headroom_warning_seed_count": sum(
            bool(row["headroom_small_relative_warning"]) for row in seed_rows
        ),
        "warning": (
            "K4-K4c is numerically near zero for at least one seed; capture ratios are unstable."
            if any(bool(row["headroom_numerically_near_zero"]) for row in seed_rows)
            else "none"
        ),
        "changes_scientific_decision": False,
    }
    gates = config["gates"]
    finite_child_metrics = all(
        math.isfinite(float(row["squared_reference_error"])) for row in child_rows
    )
    validity = {
        **validity_extra,
        "evaluation_matrix_exact": _evaluation_matrix_is_exact(
            config, history_rows, child_rows
        ),
        "evaluation_noise_cell_composition_exact": (
            _evaluation_noise_composition_is_exact(config, history_rows)
        ),
        "evaluation_outer_seed_count_exact": len(seed_rows) == 12
        and {int(row["seed"]) for row in seed_rows}
        == {int(value) for value in config["randomness"]["evaluation_outer_seeds"]},
        "evaluation_histories_exact": len(history_rows)
        == int(gates["evaluation_histories_exact"]),
        "evaluation_child_rows_exact": len(child_rows)
        == int(gates["evaluation_histories_exact"])
        * int(config["nested_monte_carlo"]["evaluation_children"])
        * len(CANDIDATES),
        "replace_one_trials_exact": len(replace_rows)
        == int(gates["replace_one_exact_trials"]),
        "replace_one_no_violation": sum(bool(row["violation"]) for row in replace_rows)
        <= int(gates["replace_one_violation_max"]),
        "finite_metrics": finite_child_metrics
        and all(math.isfinite(float(row[key])) for row in seed_rows for key in ci_keys),
        "strict_past_features": all(
            int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1
            for row in history_rows
        ),
        "feature_source_rounds_exact": all(
            row["feature_source_rounds"]
            == ",".join(
                str(value)
                for value in range(
                    int(row["assessment_round"]) - 4,
                    int(row["assessment_round"]),
                )
            )
            for row in history_rows
        ),
        "forbidden_current_inference_fields_zero": max(
            (
                int(row["forbidden_current_inference_field_count"])
                for row in history_rows
            ),
            default=1,
        )
        <= int(gates["forbidden_current_inference_field_count_max"]),
        "fixed_predictor_across_children": all(
            bool(row["frozen_predictor_fixed"]) for row in child_rows
        ),
        "fixed_denominator_n": all(
            int(row["fixed_denominator_n"]) == int(config["cohort"]["num_clients"])
            for row in child_rows
        ),
        "no_gate_sum_normalization": all(
            not bool(row["normalization_by_gate_sum"]) for row in child_rows
        ),
        "contribution_cap": all(
            bool(row["contribution_cap_respected"])
            for row in child_rows
            if row["candidate"] == K5
        ),
        "k4_manual_formula": max(
            (float(row["k4_manual_formula_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["k4_manual_formula_abs_error_max"]),
        "k4b_reproduction": max(
            (float(row["k4b_reproduction_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["k4b_reproduction_abs_error_max"]),
        "fixed_denominator_formula": max(
            (float(row["fixed_denominator_formula_error"]) for row in child_rows),
            default=float("inf"),
        )
        <= float(gates["fixed_denominator_formula_abs_error_max"]),
        "positive_headroom_denominators": sum(
            bool(row["k4_minus_k4c_positive"]) for row in seed_rows
        )
        == int(gates["positive_k4_minus_k4c_denominator_seed_count_exact"]),
        "privileged_target_mc_stability": max(
            (
                float(row["privileged_target_split_disagreement_mse_ratio"])
                for row in seed_rows
            ),
            default=float("inf"),
        )
        <= float(gates["privileged_target_split_disagreement_mse_ratio_max"]),
    }
    scientific = {
        "gain_vs_k4b_mean": float(cis["gain_vs_k4b"]["mean"])
        >= float(gates["primary_gain_vs_k4b_mean_min"]),
        "gain_vs_k4b_ci": float(cis["gain_vs_k4b"]["low"])
        > float(gates["primary_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "gain_vs_k4_mean": float(cis["gain_vs_k4"]["mean"])
        >= float(gates["primary_gain_vs_k4_mean_min"]),
        "gain_vs_k4_ci": float(cis["gain_vs_k4"]["low"])
        > float(gates["primary_gain_vs_k4_ci95_low_strictly_greater_than"]),
        "gain_vs_one_dimensional_mean": float(cis["gain_vs_one_dimensional"]["mean"])
        >= float(gates["primary_gain_vs_one_dimensional_mean_min"]),
        "gain_vs_one_dimensional_ci": float(cis["gain_vs_one_dimensional"]["low"])
        > float(
            gates["primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"]
        ),
        "capture_mean": float(cis["ch_capture_fraction"]["mean"])
        >= float(gates["ch_capture_fraction_mean_min"]),
        "capture_ci": float(cis["ch_capture_fraction"]["low"])
        > float(gates["ch_capture_fraction_ci95_low_strictly_greater_than"]),
        "homogeneous_gain_ci": float(cis["homogeneous_gain_vs_k4b"]["low"])
        > float(gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "heteroscedastic_gain_ci": float(cis["heteroscedastic_gain_vs_k4b"]["low"])
        > float(gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"]),
    }
    validity_pass = all(validity.values())
    scientific_pass = all(scientific.values()) if validity_pass else False
    decision = (
        "authorize_end_to_end_development_screen"
        if validity_pass and scientific_pass
        else (
            "stop_five_feature_linear_predictor_instance"
            if validity_pass
            else "invalid_or_inconclusive_screen"
        )
    )
    return seed_rows, {
        "validity_pass": validity_pass,
        "scientific_checks_pass": scientific_pass,
        "all_gates_pass": validity_pass and scientific_pass,
        "decision": decision,
        "validity_checks": validity,
        "scientific_checks": scientific,
        "confidence_intervals": cis,
        "headroom_diagnostics_non_gating": headroom_diagnostics,
        "holdout_opened": False,
        "pass_authorizes_holdout_or_promotion": False,
        "pass_authorizes_end_to_end_development_screen": True,
    }


def evaluate_frozen(
    config_path: Path,
    lock_path: Path,
    output: Path,
    *,
    published_lock_sha256: str,
    published_predictor_sha256: str,
    published_fit_audit_sha256: str,
) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError(
            "K5-v2 evaluation accepts only the preregistered output directory"
        )
    config, amendment = _load_amended_config(config_path)
    _validate_config(config, amendment)
    lock = _verify_lock(lock_path, config_path)
    lock_publication = _attest(lock["sha256"], published_lock_sha256, name="lock")
    (
        predictor_artifact,
        predictor_publication,
        independent_fit_audit_attestation,
    ) = _load_frozen_predictor(
        output,
        published_predictor_sha256,
        published_fit_audit_sha256,
        config=config,
        amendment=amendment,
        config_path=config_path,
        lock_sha256=lock["sha256"],
    )
    evaluation_dir = output / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing K5-v2 evaluation")
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError(
            "K5-v2 evaluation refuses CPU, fallback, or non-float32 runtime"
        )
    k2_calibration, temporal_calibration, _ = v1._load_calibrations(config)
    coefficients = torch.tensor(
        predictor_artifact["coefficients"],
        device=oracle._RUNTIME_DEVICE,
        dtype=torch.float32,
    )
    scales = torch.tensor(
        predictor_artifact["feature_scales"],
        device=oracle._RUNTIME_DEVICE,
        dtype=torch.float32,
    )
    one = predictor_artifact["one_dimensional_control"]
    coefficients_1d = torch.tensor(
        [one["coefficient"]], device=oracle._RUNTIME_DEVICE, dtype=torch.float32
    )
    scales_1d = torch.tensor(
        [one["feature_scale"]], device=oracle._RUNTIME_DEVICE, dtype=torch.float32
    )
    predictor_path = output / "frozen_predictor.json"
    predictor_hash = _sha256(predictor_path)
    evaluation_dir.mkdir(parents=True)
    _write_json(
        evaluation_dir / "manifest.json",
        {
            "status": "evaluation_running",
            "device": str(oracle._RUNTIME_DEVICE),
            "dtype": str(oracle._RUNTIME_DTYPE),
            "frozen_predictor_sha256": predictor_hash,
            "lock_publication_attestation": lock_publication,
            "predictor_publication_attestation": predictor_publication,
            "independent_fit_audit_attestation": (independent_fit_audit_attestation),
            "holdout_opened": False,
        },
    )
    history_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    audit_contexts: list[dict[str, Any]] = []
    target_seeds: list[int] = []
    evaluation_seeds: list[int] = []
    outer_seeds = [
        int(value) for value in config["randomness"]["evaluation_outer_seeds"]
    ]
    evaluation_progress_seed: int | None = None
    evaluation_progress_completed = 0
    for context in _snapshot_contexts(
        config,
        k2_calibration,
        temporal_calibration,
        split="evaluation",
        seeds=outer_seeds,
    ):
        current_seed = int(context["cell"]["seed"])
        if (
            evaluation_progress_seed is not None
            and current_seed != evaluation_progress_seed
        ):
            evaluation_progress_completed += 1
            _progress(
                "evaluate_frozen",
                completed=evaluation_progress_completed,
                total=len(outer_seeds),
                outer_seed=evaluation_progress_seed,
            )
        evaluation_progress_seed = current_seed
        target = v1._privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="evaluation_target",
            children=int(
                config["nested_monte_carlo"]["evaluation_target_construction_children"]
            ),
            split_diagnostic=True,
        )
        target_seeds.extend(int(value) for value in target["child_seeds"])
        predictor, predictor_diag = transcript_past_predictor(
            context["feature"],
            coefficients,
            influence_cap=float(config["references"]["total_client_influence_cap"]),
            feature_scales=scales,
            return_diagnostics=True,
        )
        predictor_1d = transcript_past_predictor(
            context["feature"][:1],
            coefficients_1d,
            influence_cap=float(config["references"]["total_client_influence_cap"]),
            feature_scales=scales_1d,
        )
        inference_payload = {
            "past_feature_dictionary_sha256": _tensor_hash(context["feature"]),
            "frozen_coefficients_sha256": _tensor_hash(coefficients),
            "frozen_feature_scales_sha256": _tensor_hash(scales),
        }
        history_rows.append(
            {
                "history_id": context["history_id"],
                "seed": int(context["cell"]["seed"]),
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "feature_source_rounds": ",".join(
                    str(value) for value in context["source_rounds"]
                ),
                "feature_hash": _tensor_hash(context["feature"]),
                "k5_predictor_hash": _tensor_hash(predictor),
                "k5_predictor_norm": float(predictor_diag["predictor_norm"]),
                "k5_projection_active": bool(predictor_diag["projection_active"]),
                "one_dimensional_predictor_norm": float(
                    torch.linalg.vector_norm(predictor_1d).item()
                ),
                "missing_slot_mass": float(target["missing_slot_mass"]),
                "privileged_predictor_norm": float(target["predictor_norm"]),
                "privileged_target_split_aggregate_scale_distance": float(
                    target["split_aggregate_scale_distance"]
                ),
                "frozen_predictor_sha256": predictor_hash,
                "forbidden_current_inference_field_count": (
                    forbidden_current_field_count(inference_payload)
                ),
            }
        )
        first_audit_added = False
        for child in range(int(config["nested_monte_carlo"]["evaluation_children"])):
            values, vectors, child_seed = v1._child_values(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                stream="evaluation",
                child=child,
            )
            if not math.isclose(
                float(values["missing_slot_mass"]),
                float(target["missing_slot_mass"]),
                abs_tol=1e-7,
            ):
                raise RuntimeError("Evaluation past missing-slot mass changed")
            evaluation_seeds.append(int(child_seed))
            references = _candidate_references(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                values,
                vectors,
                k5_predictor=predictor,
                one_dimensional_predictor=predictor_1d,
                privileged_predictor=target["predictor"],
            )
            n = int(config["cohort"]["num_clients"])
            manual_k4 = context["components"]["anchor"] + values["direct_sum"] / float(
                n
            )
            k4_error = float(
                torch.linalg.vector_norm(references[K4] - manual_k4).item()
            )
            aware_radii = k4._radii(
                config,
                context["components"]["variances"],
                k2_calibration,
                regime_name=str(context["cell"]["regime"]["name"]),
                blind=False,
            )
            reproduced_k4b, _ = k4b._k4b_reference(
                vectors,
                anchor=context["components"]["anchor"],
                aware_radii=aware_radii,
                history=context["history"],
                enrollment_mean=context["enrollment_mean"],
                predictor=context["feature"][0],
                predictor_role="independent_frozen_k4b_rolling_past_imputation",
                imputation_mode=FULL_TEMPORAL_MISSING_SLOT,
                deployable=False,
                privacy_claimed=True,
                temporal_calibration=temporal_calibration,
                config=config,
            )
            k4b_error = float(
                torch.linalg.vector_norm(references[K4B] - reproduced_k4b).item()
            )
            manual_k5 = context["components"]["anchor"] + (
                values["direct_sum"] + float(values["missing_slot_mass"]) * predictor
            ) / float(n)
            fixed_error = float(
                torch.linalg.vector_norm(references[K5] - manual_k5).item()
            )
            contributions = (
                values["gates"][:, None] * values["clipped"]
                + (1.0 - values["history_gates"])[:, None] * predictor[None, :]
            )
            cap = float(config["references"]["total_client_influence_cap"])
            tolerance = 64.0 * torch.finfo(contributions.dtype).eps
            norms = torch.linalg.vector_norm(contributions, dim=1)
            cap_ok = bool((norms <= cap + tolerance).all())
            max_norm = float(torch.max(norms).item())
            if not first_audit_added and (
                str(context["cell"]["geometry"]) == "aligned"
                and str(context["cell"]["dynamics"]) == "stationary"
                and str(context["cell"]["threat"]) == "bitflip_x10"
                and int(context["round_index"]) == 17
            ):
                audit_contexts.append(
                    {
                        "history_id": context["history_id"],
                        "seed": int(context["cell"]["seed"]),
                        "noise_regime": str(context["cell"]["regime"]["name"]),
                        "noise_permutation": str(context["cell"]["permutation"]),
                        "vectors": vectors.clone(),
                        "anchor": context["components"]["anchor"],
                        "aware_radii": aware_radii,
                        "history": context["history"],
                        "enrollment_mean": context["enrollment_mean"],
                        "predictor": predictor.clone(),
                    }
                )
                first_audit_added = True
            for candidate, reference in references.items():
                child_rows.append(
                    {
                        "history_id": context["history_id"],
                        "seed": int(context["cell"]["seed"]),
                        "noise_regime": str(context["cell"]["regime"]["name"]),
                        "noise_permutation": str(context["cell"]["permutation"]),
                        "outlier_geometry": str(context["cell"]["geometry"]),
                        "honest_dynamics": str(context["cell"]["dynamics"]),
                        "threat": str(context["cell"]["threat"]),
                        "assessment_round": int(context["round_index"]),
                        "evaluation_child": child,
                        "evaluation_child_seed": int(child_seed),
                        "candidate": candidate,
                        "squared_reference_error": float(
                            torch.sum((reference - values["target"]).square()).item()
                        ),
                        "reference_error_l2_descriptive": float(
                            torch.linalg.vector_norm(
                                reference - values["target"]
                            ).item()
                        ),
                        "frozen_predictor_sha256": predictor_hash,
                        "frozen_predictor_fixed": _tensor_hash(predictor)
                        == history_rows[-1]["k5_predictor_hash"],
                        "predictor_observable_past_only": candidate in {K4B, K5_1D, K5},
                        "fixed_denominator_n": n,
                        "normalization_by_gate_sum": False,
                        "contribution_cap_respected": (
                            cap_ok if candidate == K5 else True
                        ),
                        "max_slot_contribution_norm": (
                            max_norm if candidate == K5 else 0.0
                        ),
                        "k4_manual_formula_error": k4_error,
                        "k4b_reproduction_error": k4b_error,
                        "fixed_denominator_formula_error": fixed_error,
                    }
                )
    if evaluation_progress_seed is not None:
        evaluation_progress_completed += 1
        _progress(
            "evaluate_frozen",
            completed=evaluation_progress_completed,
            total=len(outer_seeds),
            outer_seed=evaluation_progress_seed,
        )
    replace_rows = v1._replace_one_audit(config, temporal_calibration, audit_contexts)
    expected_evaluation_streams = {
        "train_target": [],
        "calibration_target": [],
        "calibration_evaluation": [],
        "evaluation_target": _expected_rng_stream_records(
            config,
            split="evaluation",
            seeds=outer_seeds,
            stream="evaluation_target",
            children=int(
                config["nested_monte_carlo"]["evaluation_target_construction_children"]
            ),
        ),
        "evaluation": _expected_rng_stream_records(
            config,
            split="evaluation",
            seeds=outer_seeds,
            stream="evaluation",
            children=int(config["nested_monte_carlo"]["evaluation_children"]),
        ),
        "holdout": [],
    }
    evaluation_registry = _rng_registry(expected_evaluation_streams)
    actual_evaluation_derivation = target_seeds == _flatten_rng_records(
        expected_evaluation_streams["evaluation_target"]
    ) and evaluation_seeds == _flatten_rng_records(
        expected_evaluation_streams["evaluation"]
    )
    fit_registry = json.loads(
        (output / "fit_rng_registry.json").read_text(encoding="utf-8")
    )
    fit_seeds = [
        value
        for stream in ("train_target", "calibration_target", "calibration_evaluation")
        for value in _flatten_rng_records(fit_registry["streams"][stream])
    ]
    all_seeds = fit_seeds + target_seeds + evaluation_seeds
    _write_json(evaluation_dir / "evaluation_rng_registry.json", evaluation_registry)
    persisted_evaluation_registry = json.loads(
        (evaluation_dir / "evaluation_rng_registry.json").read_text(encoding="utf-8")
    )
    lock_after = _verify_lock(lock_path, config_path)
    validity_extra = {
        "device_type_mps": oracle._RUNTIME_DEVICE.type == "mps",
        "dtype_float32": oracle._RUNTIME_DTYPE == torch.float32,
        "evaluation_target_seed_count_exact": len(target_seeds)
        == int(config["gates"]["evaluation_target_child_seeds_exact"]),
        "evaluation_seed_count_exact": len(evaluation_seeds)
        == int(config["gates"]["evaluation_child_seeds_exact"]),
        "evaluation_rng_each_seed_matches_native_derivation": actual_evaluation_derivation,
        "evaluation_rng_structured_registry_round_trip": _validate_rng_registry(
            persisted_evaluation_registry, expected_evaluation_streams
        ),
        "global_child_seed_unique": len(all_seeds) == len(set(all_seeds)),
        "evaluation_target_and_evaluation_disjoint": not (
            set(target_seeds) & set(evaluation_seeds)
        ),
        "evaluation_disjoint_from_fit": not (
            set(fit_seeds) & (set(target_seeds) | set(evaluation_seeds))
        ),
        "frozen_predictor_hash_unchanged": _sha256(predictor_path) == predictor_hash,
        "lock_hash_unchanged": lock_after["sha256"] == lock["sha256"],
        "resolved_config_hash_unchanged": _sha256(output / "resolved_config.json")
        == predictor_artifact["resolved_config_sha256"],
        "holdout_not_opened": True,
    }
    seed_rows, decision = _summarize_evaluation(
        config, history_rows, child_rows, replace_rows, validity_extra
    )
    _write_csv(evaluation_dir / "history_rows.csv", history_rows)
    _write_csv(evaluation_dir / "evaluation_child_rows.csv", child_rows)
    _write_csv(evaluation_dir / "seed_summary.csv", seed_rows)
    _write_csv(evaluation_dir / "replace_one_audit.csv", replace_rows)
    _write_json(evaluation_dir / "decision.json", decision)
    if _sha256(predictor_path) != predictor_hash:
        raise RuntimeError("Frozen K5-v2 predictor changed during evaluation")
    _write_json(
        evaluation_dir / "manifest.json",
        {
            "status": "completed_development",
            "device": str(oracle._RUNTIME_DEVICE),
            "dtype": str(oracle._RUNTIME_DTYPE),
            "frozen_predictor_sha256": predictor_hash,
            "lock_publication_attestation": lock_publication,
            "predictor_publication_attestation": predictor_publication,
            "independent_fit_audit_attestation": (independent_fit_audit_attestation),
            "evaluation_rng_registry_structured_sha256": evaluation_registry[
                "payload_sha256"
            ],
            "histories": len(history_rows),
            "child_rows": len(child_rows),
            "decision": decision["decision"],
            "all_gates_pass": decision["all_gates_pass"],
            "headroom_warning": decision["headroom_diagnostics_non_gating"]["warning"],
            "holdout_opened": False,
        },
    )
    root_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    root_manifest.update(
        {
            "status": "completed_development",
            "evaluation_trajectory_count_generated": len(history_rows),
            "evaluation_decision": decision["decision"],
            "all_gates_pass": decision["all_gates_pass"],
            "frozen_predictor_sha256_after_evaluation": _sha256(predictor_path),
            "evaluation_rng_registry_structured_sha256": evaluation_registry[
                "payload_sha256"
            ],
            "independent_fit_audit_attestation": (independent_fit_audit_attestation),
            "holdout_opened": False,
        }
    )
    _write_json(output / "manifest.json", root_manifest)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("validate", "fit-freeze", "evaluate-frozen"),
        default="validate",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--published-lock-sha256")
    parser.add_argument("--published-predictor-sha256")
    parser.add_argument("--published-fit-audit-sha256")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config, amendment = _load_amended_config(config_path)
    _validate_config(config, amendment)
    if args.phase == "validate":
        print(
            json.dumps(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "phase": "validate_only_no_data_generation",
                    "amendment_config_sha256": _sha256(config_path),
                    "base_config_sha256": amendment["base_protocol"]["config_sha256"],
                    "device_required": "mps",
                    "feature_names": list(FEATURE_NAMES),
                    "rank_expected_exact": 5,
                    "train_outer_seeds": config["randomness"]["train_outer_seeds"],
                    "calibration_outer_seeds": config["randomness"][
                        "calibration_outer_seeds"
                    ],
                    "evaluation_outer_seeds_unopened": config["randomness"][
                        "evaluation_outer_seeds"
                    ],
                    "lock_exists": lock_path.is_file(),
                    "evaluation_generated": False,
                    "holdout_opened": False,
                },
                indent=2,
            )
        )
        return 0
    if not args.published_lock_sha256:
        parser.error("fit/evaluation requires --published-lock-sha256")
    if args.phase == "fit-freeze":
        decision = fit_freeze(
            config_path,
            lock_path,
            output,
            published_lock_sha256=args.published_lock_sha256,
        )
        print(json.dumps(_json_safe(decision), indent=2, sort_keys=True))
        return 0 if decision["all_validity_checks_pass"] else 2
    if not args.published_predictor_sha256:
        parser.error("evaluate-frozen requires --published-predictor-sha256")
    if not args.published_fit_audit_sha256:
        parser.error("evaluate-frozen requires --published-fit-audit-sha256")
    decision = evaluate_frozen(
        config_path,
        lock_path,
        output,
        published_lock_sha256=args.published_lock_sha256,
        published_predictor_sha256=args.published_predictor_sha256,
        published_fit_audit_sha256=args.published_fit_audit_sha256,
    )
    print(json.dumps(_json_safe(decision), indent=2, sort_keys=True))
    return 0 if decision["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
