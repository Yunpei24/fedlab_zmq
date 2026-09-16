from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import audit_gaussian_aware_reference_g0g_k5_pre_eval_supplement as audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _diagnostic(ridge_lambda: float, coefficient_count: int) -> dict:
    return {
        "ridge_lambda": ridge_lambda,
        "condition_number_regularized_system": 12.0,
        "normal_equation_relative_residual": 2.0e-7,
        "fit_dtype": "torch.float32",
        "fit_device": "mps:0",
        "coefficients": [0.25] * coefficient_count,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    lock_path = tmp_path / "lock.json"
    results = tmp_path / "results"
    results.mkdir()
    config = {
        "campaign_id": "test-k5",
        "ridge": {"lambda_grid": [0.1, 1.0]},
        "randomness": {
            "train_outer_seeds": [11, 12],
            "calibration_outer_seeds": [21],
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_hash = audit._sha256(config_path)
    lock = {
        "schema_version": 1,
        "campaign_id": "test-k5",
        "locked_files": {"config.yaml": config_hash},
        "dependencies": {},
    }
    _write_json(lock_path, lock)
    lock_hash = audit._sha256(lock_path)
    registry = {
        "train_target": [101, 102],
        "calibration_target": [201],
        "calibration_evaluation": [301, 302],
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    _write_json(results / "fit_rng_registry.json", registry)
    rng_hash = audit._canonical_hash([101, 102, 201, 301, 302])
    predictor = {
        "campaign_id": "test-k5",
        "config_sha256": config_hash,
        "lock_sha256": lock_hash,
        "fit_rng_registry_sha256": rng_hash,
        "coefficients": [0.1] * 6,
        "one_dimensional_control": {"coefficient": 0.2},
        "evaluation_generated_before_freeze": False,
        "holdout_opened": False,
    }
    _write_json(results / "frozen_predictor.json", predictor)
    predictor_hash = audit._sha256(results / "frozen_predictor.json")
    manifest = {
        "campaign_id": "test-k5",
        "status": "fit_completed_evaluation_locked",
        "fit_validity_pass": True,
        "config_sha256": config_hash,
        "preregistration_lock": {"sha256": lock_hash},
        "lock_publication_attestation": {
            "published_lock_sha256": lock_hash,
            "procedural_external_publication_attested": True,
        },
        "frozen_predictor_sha256": predictor_hash,
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    decision = {
        "all_validity_checks_pass": True,
        "checks": {"synthetic_fixture_fit_valid": True},
        "frozen_predictor_sha256": predictor_hash,
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    calibration = {
        "train_grid_fit_diagnostics": {
            "0.1": _diagnostic(0.1, 6),
            "1.0": _diagnostic(1.0, 6),
        },
        "train_grid_1d_fit_diagnostics": {
            "0.1": _diagnostic(0.1, 1),
            "1.0": _diagnostic(1.0, 1),
        },
    }
    final = _diagnostic(0.1, 6)
    final_1d = _diagnostic(1.0, 1)
    final.pop("coefficients")
    final_1d.pop("coefficients")
    design = {
        "final_fit": final,
        "final_1d_fit": final_1d,
        "weight_audit": {
            "balanced_train_sum_by_seed": {"11": 1.0, "12": 1.0},
            "balanced_train_plus_calibration_sum_by_seed": {
                "11": 1.0,
                "12": 1.0,
                "21": 1.0,
            },
        },
    }
    _write_json(results / "manifest.json", manifest)
    _write_json(results / "fit_decision.json", decision)
    _write_json(results / "lambda_calibration.json", calibration)
    _write_json(results / "feature_design_diagnostics.json", design)
    return {"config": config_path, "lock": lock_path, "results": results}


def _run(paths: dict[str, Path]) -> dict:
    return audit.audit_pre_evaluation(
        config_path=paths["config"],
        lock_path=paths["lock"],
        results=paths["results"],
    )


def test_valid_frozen_fit_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    report = _run(paths)
    assert report["pass"] is True
    assert all(report["checks"].values())
    assert report["violations"] == []


def test_rng_registry_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    path = paths["results"] / "fit_rng_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["train_target"].append(999)
    _write_json(path, registry)
    report = _run(paths)
    assert report["pass"] is False
    assert report["checks"]["fit_rng_registry_hash_and_phase_boundary"] is False
    assert "rng:fit_rng_registry_sha256_mismatch" in report["violations"]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("condition_number_regularized_system", 1.0e7 + 1.0, "condition_number"),
        ("normal_equation_relative_residual", 1.01e-4, "normal_residual"),
        ("fit_dtype", "torch.float64", "fit_dtype"),
        ("fit_device", "cpu", "fit_device"),
        ("coefficients", [float("nan")], "coefficients"),
    ],
)
def test_any_unsafe_grid_solution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    fragment: str,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    path = paths["results"] / "lambda_calibration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_grid_1d_fit_diagnostics"]["1.0"][field] = value
    _write_json(path, payload)
    report = _run(paths)
    assert report["pass"] is False
    assert report["checks"]["all_primary_and_one_dimensional_grid_solutions_safe"] is False
    assert any(fragment in item for item in report["violations"])


def test_unbalanced_seed_mass_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    path = paths["results"] / "feature_design_diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weight_audit"]["balanced_train_sum_by_seed"]["12"] = 0.999
    _write_json(path, payload)
    report = _run(paths)
    assert report["pass"] is False
    assert report["checks"]["balanced_weight_mass_is_one_per_seed"] is False


@pytest.mark.parametrize(
    ("artifact", "field", "value"),
    [
        ("manifest.json", "evaluation_trajectory_count_generated", 1),
        ("manifest.json", "holdout_opened", True),
        ("fit_decision.json", "evaluation_trajectory_count_generated", 2),
        ("fit_decision.json", "holdout_opened", True),
    ],
)
def test_open_evaluation_or_holdout_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    value: object,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    path = paths["results"] / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)
    report = _run(paths)
    assert report["pass"] is False
    assert report["checks"]["evaluation_absent_and_holdout_closed"] is False


def test_hash_incoherence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    path = paths["results"] / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    _write_json(path, payload)
    report = _run(paths)
    assert report["pass"] is False
    assert report["checks"]["config_lock_predictor_hashes_coherent"] is False


def test_main_writes_fail_closed_evidence_for_missing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["results"] / "fit_decision.json").unlink()
    return_code = audit.main(
        [
            "--config",
            str(paths["config"]),
            "--lock",
            str(paths["lock"]),
            "--results",
            str(paths["results"]),
        ]
    )
    output = paths["results"] / audit.OUTPUT_NAME
    assert return_code == 1
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["checks"]["all_required_artifacts_present"] is False
