#!/usr/bin/env python3
"""Fail-closed supplemental audit before any G0g-K5 evaluation.

This audit only inspects artifacts persisted by the K5 fit/freeze phase.  It
does not import the scientific runner, generate random seeds, fit a model, or
evaluate a scientific metric.  Its purpose is to independently enforce the
phase boundary immediately before the frozen-predictor hash is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.lock.json"
)
DEFAULT_RESULTS = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)
OUTPUT_NAME = "pre_evaluation_supplemental_audit.json"

FIT_STREAMS = (
    "train_target",
    "calibration_target",
    "calibration_evaluation",
)
FORBIDDEN_STREAMS = ("evaluation_target", "evaluation", "holdout")
GRID_DIAGNOSTIC_FIELDS = {
    "condition_number_regularized_system",
    "normal_equation_relative_residual",
    "fit_dtype",
    "fit_device",
    "coefficients",
    "ridge_lambda",
}
FINAL_DIAGNOSTIC_FIELDS = GRID_DIAGNOSTIC_FIELDS - {"coefficients"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON list")
    return value


def _finite_coefficients(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    try:
        coefficients = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) for item in coefficients)


def _is_float32(value: Any) -> bool:
    return str(value) in {"float32", "torch.float32"}


def _is_mps(value: Any) -> bool:
    text = str(value)
    return text == "mps" or text.startswith("mps:")


def _diagnostic_check(
    diagnostic: Any,
    *,
    expected_lambda: float | None,
    expected_coefficient_count: int,
    coefficients: Any | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    if not isinstance(diagnostic, Mapping):
        return False, ["diagnostic_not_an_object"], {}
    required = GRID_DIAGNOSTIC_FIELDS if coefficients is None else FINAL_DIAGNOSTIC_FIELDS
    missing = sorted(required - set(diagnostic))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))

    def finite_number(field: str) -> float | None:
        try:
            number = float(diagnostic[field])
        except (KeyError, TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    condition = finite_number("condition_number_regularized_system")
    residual = finite_number("normal_equation_relative_residual")
    ridge_lambda = finite_number("ridge_lambda")
    if condition is None or condition > 1.0e7:
        errors.append("condition_number_not_finite_or_above_1e7")
    if residual is None or residual > 1.0e-4:
        errors.append("normal_residual_not_finite_or_above_1e-4")
    if not _is_float32(diagnostic.get("fit_dtype")):
        errors.append("fit_dtype_not_float32")
    if not _is_mps(diagnostic.get("fit_device")):
        errors.append("fit_device_not_mps")
    if expected_lambda is not None and (
        ridge_lambda is None
        or not math.isclose(ridge_lambda, expected_lambda, rel_tol=0.0, abs_tol=1.0e-15)
    ):
        errors.append("ridge_lambda_does_not_match_grid_key")
    coefficient_payload = diagnostic.get("coefficients") if coefficients is None else coefficients
    if not _finite_coefficients(coefficient_payload):
        errors.append("coefficients_missing_empty_or_nonfinite")
    elif len(coefficient_payload) != expected_coefficient_count:
        errors.append("coefficient_count_mismatch")
    details = {
        "condition_number": condition,
        "normal_equation_relative_residual": residual,
        "fit_dtype": diagnostic.get("fit_dtype"),
        "fit_device": diagnostic.get("fit_device"),
        "ridge_lambda": ridge_lambda,
        "coefficient_count": (
            len(coefficient_payload) if isinstance(coefficient_payload, list) else None
        ),
        "errors": errors,
    }
    return not errors, errors, details


def _grid_audit(
    calibration: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        expected = {float(item) for item in config["ridge"]["lambda_grid"]}
    except (KeyError, TypeError, ValueError) as exc:
        return False, {}, [f"invalid_config_lambda_grid:{exc}"]
    for artifact_key in (
        "train_grid_fit_diagnostics",
        "train_grid_1d_fit_diagnostics",
    ):
        expected_coefficient_count = (
            1 if artifact_key == "train_grid_1d_fit_diagnostics" else 6
        )
        payload = calibration.get(artifact_key)
        if not isinstance(payload, Mapping):
            errors.append(f"{artifact_key}:missing_or_not_an_object")
            details[artifact_key] = {}
            continue
        parsed: dict[float, tuple[str, Any]] = {}
        duplicate_numeric_keys = False
        for raw_key, diagnostic in payload.items():
            try:
                numeric_key = float(raw_key)
            except (TypeError, ValueError):
                errors.append(f"{artifact_key}:invalid_lambda_key:{raw_key}")
                continue
            if numeric_key in parsed:
                duplicate_numeric_keys = True
            parsed[numeric_key] = (str(raw_key), diagnostic)
        if duplicate_numeric_keys:
            errors.append(f"{artifact_key}:duplicate_numeric_lambda_key")
        if set(parsed) != expected:
            errors.append(f"{artifact_key}:lambda_grid_mismatch")
        artifact_details: dict[str, Any] = {}
        for numeric_key, (raw_key, diagnostic) in sorted(parsed.items()):
            passed, diag_errors, diag_details = _diagnostic_check(
                diagnostic,
                expected_lambda=numeric_key,
                expected_coefficient_count=expected_coefficient_count,
            )
            artifact_details[raw_key] = {"pass": passed, **diag_details}
            errors.extend(f"{artifact_key}[{raw_key}]:{item}" for item in diag_errors)
        details[artifact_key] = artifact_details
    return not errors, details, errors


def _balanced_mass_audit(
    design: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    try:
        audit = _require_mapping(design["weight_audit"], "weight_audit")
        expected_train = {str(int(seed)) for seed in config["randomness"]["train_outer_seeds"]}
        expected_refit = expected_train | {
            str(int(seed)) for seed in config["randomness"]["calibration_outer_seeds"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        return False, {}, [f"invalid_weight_audit_schema:{exc}"]
    for key, expected_seeds in (
        ("balanced_train_sum_by_seed", expected_train),
        ("balanced_train_plus_calibration_sum_by_seed", expected_refit),
    ):
        payload = audit.get(key)
        current: dict[str, Any] = {"expected_seeds": sorted(expected_seeds)}
        if not isinstance(payload, Mapping):
            errors.append(f"{key}:missing_or_not_an_object")
            current["observed"] = None
            details[key] = current
            continue
        observed = {str(seed): value for seed, value in payload.items()}
        current["observed"] = observed
        if set(observed) != expected_seeds:
            errors.append(f"{key}:seed_set_mismatch")
        invalid: dict[str, Any] = {}
        for seed, value in observed.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                invalid[seed] = value
                continue
            if not math.isfinite(number) or not math.isclose(
                number, 1.0, rel_tol=0.0, abs_tol=1.0e-6
            ):
                invalid[seed] = value
        if invalid:
            errors.append(f"{key}:mass_not_one")
        current["invalid_masses"] = invalid
        details[key] = current
    return not errors, details, errors


def _lock_inventory_audit(lock: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    mismatches: list[str] = []
    checked = 0
    for section in ("locked_files", "dependencies"):
        inventory = lock.get(section)
        if not isinstance(inventory, Mapping):
            mismatches.append(f"{section}:missing_or_not_an_object")
            continue
        for relative, expected in inventory.items():
            path = ROOT / str(relative)
            checked += 1
            if not path.is_file() or _sha256(path) != str(expected):
                mismatches.append(f"{section}:{relative}")
    return not mismatches, {"checked_files": checked, "mismatches": mismatches}


def audit_pre_evaluation(
    *, config_path: Path, lock_path: Path, results: Path
) -> dict[str, Any]:
    """Inspect frozen fit artifacts and return a JSON-safe fail-closed audit."""

    artifact_paths = {
        "config": config_path,
        "lock": lock_path,
        "manifest": results / "manifest.json",
        "fit_decision": results / "fit_decision.json",
        "frozen_predictor": results / "frozen_predictor.json",
        "lambda_calibration": results / "lambda_calibration.json",
        "feature_design_diagnostics": results / "feature_design_diagnostics.json",
        "fit_rng_registry": results / "fit_rng_registry.json",
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit": "g0g_k5_pre_evaluation_supplement",
        "scope": "persisted_fit_artifacts_only_no_seed_generation_no_scientific_computation",
        "results_directory": str(results.resolve()),
        "checks": {},
        "violations": [],
        "artifact_hashes": {},
    }
    if missing:
        report["checks"]["all_required_artifacts_present"] = False
        report["violations"].append("missing_artifacts:" + ",".join(sorted(missing)))
        report["pass"] = False
        return report
    report["checks"]["all_required_artifacts_present"] = True

    try:
        config = _require_mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config"
        )
        lock = _require_mapping(_read_json(lock_path), "lock")
        manifest = _require_mapping(_read_json(artifact_paths["manifest"]), "manifest")
        decision = _require_mapping(
            _read_json(artifact_paths["fit_decision"]), "fit_decision"
        )
        predictor = _require_mapping(
            _read_json(artifact_paths["frozen_predictor"]), "frozen_predictor"
        )
        calibration = _require_mapping(
            _read_json(artifact_paths["lambda_calibration"]), "lambda_calibration"
        )
        design = _require_mapping(
            _read_json(artifact_paths["feature_design_diagnostics"]),
            "feature_design_diagnostics",
        )
        registry = _require_mapping(
            _read_json(artifact_paths["fit_rng_registry"]), "fit_rng_registry"
        )
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        report["checks"]["artifacts_parse_and_schema"] = False
        report["violations"].append(f"artifact_parse_or_schema_failure:{exc}")
        report["pass"] = False
        return report
    report["checks"]["artifacts_parse_and_schema"] = True

    config_hash = _sha256(config_path)
    lock_hash = _sha256(lock_path)
    predictor_hash = _sha256(artifact_paths["frozen_predictor"])
    report["artifact_hashes"] = {
        "config_sha256": config_hash,
        "lock_sha256": lock_hash,
        "frozen_predictor_sha256": predictor_hash,
    }

    inventory_pass, inventory_details = _lock_inventory_audit(lock)
    report["checks"]["lock_inventory_integrity"] = inventory_pass
    report["lock_inventory"] = inventory_details
    if not inventory_pass:
        report["violations"].append("lock_inventory_hash_mismatch")

    campaign_id = config.get("campaign_id")
    hashes_pass = (
        lock.get("campaign_id") == campaign_id
        and predictor.get("campaign_id") == campaign_id
        and manifest.get("campaign_id") == campaign_id
        and manifest.get("config_sha256") == config_hash
        and predictor.get("config_sha256") == config_hash
        and predictor.get("lock_sha256") == lock_hash
        and isinstance(manifest.get("preregistration_lock"), Mapping)
        and manifest["preregistration_lock"].get("sha256") == lock_hash
        and isinstance(manifest.get("lock_publication_attestation"), Mapping)
        and manifest["lock_publication_attestation"].get("published_lock_sha256")
        == lock_hash
        and manifest["lock_publication_attestation"].get(
            "procedural_external_publication_attested"
        )
        is True
        and manifest.get("frozen_predictor_sha256") == predictor_hash
        and decision.get("frozen_predictor_sha256") == predictor_hash
    )
    report["checks"]["config_lock_predictor_hashes_coherent"] = hashes_pass
    if not hashes_pass:
        report["violations"].append("config_lock_predictor_hash_incoherence")

    rng_errors: list[str] = []
    fit_rng: list[int] = []
    expected_registry_keys = set(FIT_STREAMS) | set(FORBIDDEN_STREAMS)
    if set(registry) != expected_registry_keys:
        rng_errors.append("rng_registry_key_set_mismatch")
    for key in FIT_STREAMS:
        try:
            values = _require_list(registry[key], f"fit_rng_registry.{key}")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                rng_errors.append(f"{key}:contains_non_integer_seed")
            else:
                fit_rng.extend(values)
        except (KeyError, TypeError) as exc:
            rng_errors.append(f"{key}:{exc}")
    for key in FORBIDDEN_STREAMS:
        if registry.get(key) != []:
            rng_errors.append(f"{key}:must_be_empty_before_evaluation")
    computed_rng_hash = _canonical_hash(fit_rng)
    if predictor.get("fit_rng_registry_sha256") != computed_rng_hash:
        rng_errors.append("fit_rng_registry_sha256_mismatch")
    report["fit_rng_registry"] = {
        "canonical_payload": "concatenated_train_target_then_calibration_target_then_calibration_evaluation",
        "canonical_fit_stream_hash": computed_rng_hash,
        "canonical_full_registry_hash_for_traceability": _canonical_hash(registry),
        "predictor_declared_hash": predictor.get("fit_rng_registry_sha256"),
        "fit_seed_count": len(fit_rng),
        "errors": rng_errors,
    }
    if len(fit_rng) != len(set(fit_rng)):
        rng_errors.append("fit_streams_contain_duplicate_seed")
    report["checks"]["fit_rng_registry_hash_and_phase_boundary"] = not rng_errors
    report["violations"].extend(f"rng:{item}" for item in rng_errors)

    grid_pass, grid_details, grid_errors = _grid_audit(calibration, config)
    report["checks"]["all_primary_and_one_dimensional_grid_solutions_safe"] = grid_pass
    report["ridge_grid"] = grid_details
    report["violations"].extend(f"ridge_grid:{item}" for item in grid_errors)

    final_errors: list[str] = []
    final_details: dict[str, Any] = {}
    final_specs = (
        ("final_fit", predictor.get("coefficients"), 6),
        (
            "final_1d_fit",
            [predictor.get("one_dimensional_control", {}).get("coefficient")]
            if isinstance(predictor.get("one_dimensional_control"), Mapping)
            else None,
            1,
        ),
    )
    for name, coefficients, expected_count in final_specs:
        passed, errors, details = _diagnostic_check(
            design.get(name),
            expected_lambda=None,
            expected_coefficient_count=expected_count,
            coefficients=coefficients,
        )
        final_details[name] = {"pass": passed, **details}
        final_errors.extend(f"{name}:{item}" for item in errors)
    report["checks"]["final_refit_solutions_safe"] = not final_errors
    report["final_refits"] = final_details
    report["violations"].extend(f"final_refit:{item}" for item in final_errors)

    masses_pass, masses_details, mass_errors = _balanced_mass_audit(design, config)
    report["checks"]["balanced_weight_mass_is_one_per_seed"] = masses_pass
    report["balanced_weights"] = masses_details
    report["violations"].extend(f"balanced_weights:{item}" for item in mass_errors)

    manifest_evaluation_count = manifest.get("evaluation_trajectory_count_generated")
    decision_evaluation_count = decision.get("evaluation_trajectory_count_generated")
    decision_checks = decision.get("checks")
    phase_pass = (
        manifest.get("status") == "fit_completed_evaluation_locked"
        and manifest.get("fit_validity_pass") is True
        and decision.get("all_validity_checks_pass") is True
        and isinstance(decision_checks, Mapping)
        and bool(decision_checks)
        and all(value is True for value in decision_checks.values())
        and type(manifest_evaluation_count) is int
        and manifest_evaluation_count == 0
        and type(decision_evaluation_count) is int
        and decision_evaluation_count == 0
        and manifest.get("holdout_opened") is False
        and decision.get("holdout_opened") is False
        and predictor.get("evaluation_generated_before_freeze") is False
        and predictor.get("holdout_opened") is False
    )
    report["checks"]["evaluation_absent_and_holdout_closed"] = phase_pass
    if not phase_pass:
        report["violations"].append("evaluation_or_holdout_phase_boundary_violation")

    report["pass"] = all(bool(value) for value in report["checks"].values())
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    report = audit_pre_evaluation(
        config_path=args.config.resolve(),
        lock_path=args.lock.resolve(),
        results=args.results.resolve(),
    )
    output = args.results.resolve() / OUTPUT_NAME
    _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
