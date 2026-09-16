from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch

from scripts import audit_gaussian_aware_reference_g0g_k5_tp_v2 as audit


def _lambda_row(value: float, score: float) -> dict:
    return {
        "ridge_lambda": value,
        "calibration_seed_mse": [score] * 8,
        "equal_seed_mean_projected_aggregate_mse": score,
    }


def test_lambda_selection_is_seed_balanced_and_uses_largest_tie() -> None:
    rows = [
        _lambda_row(value, 3.0 + index) for index, value in enumerate(audit.LAMBDA_GRID)
    ]
    rows[0] = {
        "ridge_lambda": audit.LAMBDA_GRID[0],
        "calibration_seed_mse": [0.0] * 7 + [8.0],
        "equal_seed_mean_projected_aggregate_mse": 1.0,
    }
    rows[1] = _lambda_row(audit.LAMBDA_GRID[1], 1.0 + 5e-13)
    selected, diagnostics = audit._select_lambda(rows, 1e-12)
    assert selected == audit.LAMBDA_GRID[1]
    assert diagnostics["minimum"] == pytest.approx(1.0)


def test_lambda_selection_rejects_non_equal_seed_mean() -> None:
    rows = [
        _lambda_row(value, float(index + 1))
        for index, value in enumerate(audit.LAMBDA_GRID)
    ]
    rows[0]["equal_seed_mean_projected_aggregate_mse"] = 99.0
    with pytest.raises(ValueError, match="equal-seed"):
        audit._select_lambda(rows, 1e-12)


def test_seed_derivation_is_sha256_stable_and_registry_is_structured() -> None:
    parts = (
        "k5_tp_v2_calibration_target_construction",
        2027051001,
        "homogeneous",
        "identity",
        "aligned",
        "stationary",
        "bitflip_x10",
        17,
        0,
    )
    payload = "|".join(str(value) for value in parts).encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )
    assert audit._stable_seed(*parts) == expected

    bundle = audit._load_config_bundle(audit.DEFAULT_CONFIG)
    registry = audit._expected_fit_registry(bundle["resolved"])
    assert set(registry) == {
        "schema_version",
        "stream_order",
        "streams",
        "payload_sha256",
    }
    assert tuple(registry["stream_order"]) == audit.RNG_STREAM_ORDER
    streams = registry["streams"]
    assert len(audit._flatten_rng_records(streams["train_target"])) == 36864
    assert len(audit._flatten_rng_records(streams["calibration_target"])) == 24576
    assert len(audit._flatten_rng_records(streams["calibration_evaluation"])) == 24576
    assert (
        streams["evaluation_target"]
        == streams["evaluation"]
        == streams["holdout"]
        == []
    )
    generated = [
        seed
        for stream in audit.FIT_STREAMS
        for seed in audit._flatten_rng_records(streams[stream])
    ]
    assert len(generated) == len(set(generated)) == 86016
    payload = {
        key: registry[key] for key in ("schema_version", "stream_order", "streams")
    }
    assert registry["payload_sha256"] == audit._canonical_hash(payload)


def test_audit_rebuilds_new_output_and_verifies_technical_failure_boundary() -> None:
    bundle = audit._load_config_bundle(audit.DEFAULT_CONFIG)
    assert bundle["resolved"]["campaign_id"] == (
        "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
    )
    assert bundle["resolved"]["execution"]["output_directory"] == str(
        audit.DEFAULT_RESULTS.relative_to(audit.ROOT)
    )
    report = audit._audit_technical_v2_mps_v1_failure_provenance(
        bundle["amendment"], bundle["resolved"]
    )
    assert report["pass"] is True
    assert report["fit_history_row_count"] == 0
    assert report["evaluation_trajectory_count"] == 0
    assert report["failure_boundary_machine_verified"] is True
    assert report["failure_cause_attested_not_machine_verified"] is True
    assert report["failure_cause_is_scientific_gate"] is False
    tampered = copy.deepcopy(bundle["resolved"])
    tampered["aggregation"]["server_clip_norm"] += 0.01
    failed = audit._audit_technical_v2_mps_v1_failure_provenance(
        bundle["amendment"], tampered
    )
    assert failed["pass"] is False
    assert failed["checks"]["resolved_config_delta_exact"] is False


def test_sufficient_statistics_solver_reproduces_solution_on_cpu() -> None:
    diagnostic = {
        "normalized_gram": [[2.0, 0.0], [0.0, 3.0]],
        "normalized_rhs": [4.0, 9.0],
        "ridge_lambda": 1.0,
    }
    result = audit._solve_sufficient_statistics(diagnostic, torch.device("cpu"))
    assert torch.allclose(result["coefficients"], torch.tensor([4.0 / 3.0, 9.0 / 4.0]))
    assert result["condition_number"] == pytest.approx(4.0 / 3.0)
    assert result["normal_equation_relative_residual"] < 1e-6


def test_headroom_diagnostics_are_explicitly_non_gating() -> None:
    rows = [
        {
            "seed": 1,
            "k4_integrated_mse": 10.0,
            "k4_minus_k4c": 2.0,
            "relative_headroom": 0.2,
            "split_mc_over_headroom": 0.1,
        },
        {
            "seed": 2,
            "k4_integrated_mse": 20.0,
            "k4_minus_k4c": 8.0,
            "relative_headroom": 0.4,
            "split_mc_over_headroom": 0.3,
        },
    ]
    result = audit._headroom_diagnostics(rows)
    assert result["relative_headroom_minimum"] == pytest.approx(0.2)
    assert result["relative_headroom_median"] == pytest.approx(0.3)
    assert result["split_mc_over_headroom_maximum"] == pytest.approx(0.3)
    assert result["numerically_near_zero_seed_count"] == 0
    assert result["warning"] == "none"
    assert result["changes_scientific_decision"] is False


def test_fit_artifact_hash_inventory_always_has_five_named_entries(
    tmp_path: Path,
) -> None:
    empty = audit._fit_artifact_hash_inventory(tmp_path)
    assert tuple(empty) == audit.FIT_AUDITED_ARTIFACT_NAMES
    assert all(value is None for value in empty.values())
    for name in audit.FIT_AUDITED_ARTIFACT_NAMES:
        (tmp_path / name).write_text(f"artifact:{name}\n", encoding="utf-8")
    complete = audit._fit_artifact_hash_inventory(tmp_path)
    assert tuple(complete) == audit.FIT_AUDITED_ARTIFACT_NAMES
    assert all(audit._is_sha256(value) for value in complete.values())


def test_independent_fit_audit_attestation_rejects_audit_mutation(
    tmp_path: Path,
) -> None:
    artifact_hashes = {name: "a" * 64 for name in audit.FIT_AUDITED_ARTIFACT_NAMES}
    audit_path = tmp_path / "independent_fit_audit.json"
    audit._write_json(
        audit_path,
        {
            "schema_version": 1,
            "campaign_id": audit.CAMPAIGN_ID,
            "audit_scope": "fit_and_freeze_pre_evaluation",
            "checks": {"all_fit_checks": True},
            "audited_artifact_sha256": artifact_hashes,
            "all_checks_pass": True,
        },
    )
    audit_sha256 = audit._sha256(audit_path)
    attestation = {
        "verified_before_evaluation": True,
        "pre_evaluation_external_anchor_required": True,
        "sha256": audit_sha256,
        "audited_artifact_sha256": artifact_hashes,
        "publication_attestation": {
            "procedural_external_publication_attested": True,
            "published_fit_audit_sha256": audit_sha256,
            "machine_verifies_external_log_itself": False,
        },
    }
    assert audit._independent_fit_audit_attestation_ok(attestation, audit_path)
    audit._write_json(
        audit_path,
        {
            "schema_version": 1,
            "campaign_id": audit.CAMPAIGN_ID,
            "audit_scope": "fit_and_freeze_pre_evaluation",
            "checks": {"all_fit_checks": True},
            "audited_artifact_sha256": {
                **artifact_hashes,
                "manifest.json": "b" * 64,
            },
            "all_checks_pass": True,
        },
    )
    assert not audit._independent_fit_audit_attestation_ok(attestation, audit_path)


def test_fit_evaluation_guard_is_fail_closed() -> None:
    manifest = {
        "status": "fit_completed_evaluation_locked",
        "fit_validity_pass": True,
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    decision = {
        "all_validity_checks_pass": True,
        "checks": {"rank": True, "rng": True},
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    predictor = {"evaluation_generated_before_freeze": False, "holdout_opened": False}
    assert audit._phase_boundary_ok(
        manifest, decision, predictor, evaluation_directory_exists=False
    )
    assert not audit._phase_boundary_ok(
        manifest, decision, predictor, evaluation_directory_exists=True
    )
    decision["checks"]["rng"] = False
    assert not audit._phase_boundary_ok(
        manifest, decision, predictor, evaluation_directory_exists=False
    )


def test_post_evaluation_boundary_accepts_only_valid_phase_advance() -> None:
    manifest = {
        "status": "completed_development",
        "fit_validity_pass": True,
        "evaluation_trajectory_count_generated": 576,
        "holdout_opened": False,
    }
    decision = {
        "all_validity_checks_pass": True,
        "checks": {"fit": True},
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    predictor = {"evaluation_generated_before_freeze": False, "holdout_opened": False}
    assert audit._post_evaluation_boundary_ok(manifest, decision, predictor)
    manifest["evaluation_trajectory_count_generated"] = 575
    assert not audit._post_evaluation_boundary_ok(manifest, decision, predictor)


def test_audit_source_does_not_import_v2_runner() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    forbidden = "from scripts import run_gaussian_aware_reference_g0g_k5_tp_v2"
    assert forbidden not in source
