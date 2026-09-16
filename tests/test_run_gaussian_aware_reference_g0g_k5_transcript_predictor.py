from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts import run_gaussian_aware_reference_g0g_k5_transcript_predictor as runner


def _config() -> dict:
    return yaml.safe_load(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _predictor(config: dict, lock_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "candidate": runner.K5,
        "feature_names": list(runner.FEATURE_NAMES),
        "feature_normalization_formula": "sqrt(sum_h ||V_j,h||_2^2 / (H*d))",
        "feature_scales": [1.0] * 6,
        "coefficients": [0.0] * 6,
        "selected_lambda": 0.1,
        "one_dimensional_control": {
            "feature_name": runner.FEATURE_NAMES[0],
            "feature_scale": 1.0,
            "coefficient": 0.0,
            "selected_lambda": 1.0,
        },
        "influence_cap": config["references"]["total_client_influence_cap"],
        "public_cohort_size": config["cohort"]["num_clients"],
        "fit_split": "train_plus_calibration_after_lambda_selection_on_calibration",
        "lambda_selection_coefficients_fit_on": "train_only",
        "observable_past_only_at_inference": True,
        "privileged_training_supervision": "k4c_ch_synthetic_targets",
        "config_sha256": runner._sha256(runner.DEFAULT_CONFIG),
        "lock_sha256": lock_sha256,
        "fit_rng_registry_sha256": "f" * 64,
        "evaluation_generated_before_freeze": False,
        "holdout_opened": False,
    }


def test_locked_config_validates_without_generating_data() -> None:
    runner._validate_config(_config())


def test_split_registries_are_pairwise_disjoint() -> None:
    config = _config()
    groups = [
        set(config["randomness"][key])
        for key in (
            "train_outer_seeds",
            "calibration_outer_seeds",
            "evaluation_outer_seeds",
            "reserved_holdout_seeds",
        )
    ]
    assert all(not (groups[i] & groups[j]) for i in range(4) for j in range(i + 1, 4))


def test_lambda_selection_is_seed_balanced() -> None:
    lambdas = [0.1, 1.0]
    # Lambda 0.1 has the smaller pooled sum, but lambda 1.0 is better after
    # each calibration seed receives equal weight.
    sums = {(0.1, 1): 0.0, (0.1, 2): 100.0, (1.0, 1): 10.0, (1.0, 2): 10.0}
    counts = {(0.1, 1): 1000, (0.1, 2): 1, (1.0, 1): 1, (1.0, 2): 1}
    selected, _ = runner._select_lambda(
        lambdas, sums, counts, [1, 2], tolerance=1e-12
    )
    assert selected == 1.0


def test_lambda_tie_break_selects_largest_within_tolerance() -> None:
    lambdas = [0.1, 1.0, 10.0]
    sums = {
        (0.1, 1): 1.0,
        (1.0, 1): 1.0 + 5e-13,
        (10.0, 1): 1.0 + 2e-12,
    }
    counts = {(value, 1): 1 for value in lambdas}
    selected, _ = runner._select_lambda(
        lambdas, sums, counts, [1], tolerance=1e-12
    )
    assert selected == 1.0


def test_seed_balanced_weights_give_each_seed_equal_total_mass() -> None:
    weights = runner._seed_balanced_weights([1.0, 3.0, 10.0], [7, 7, 9])
    assert weights == pytest.approx([0.25, 0.75, 1.0])


def test_predictor_hash_must_be_published_before_evaluation(tmp_path: Path) -> None:
    config = _config()
    lock_sha = "e" * 64
    manifest = {
        "status": "fit_completed_evaluation_locked",
        "evaluation_trajectory_count_generated": 0,
    }
    fit_decision = {"all_validity_checks_pass": True}
    predictor = _predictor(config, lock_sha)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "fit_decision.json").write_text(json.dumps(fit_decision), encoding="utf-8")
    (tmp_path / "frozen_predictor.json").write_text(json.dumps(predictor), encoding="utf-8")
    digest = runner._sha256(tmp_path / "frozen_predictor.json")
    manifest["frozen_predictor_sha256"] = digest
    fit_decision["frozen_predictor_sha256"] = digest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "fit_decision.json").write_text(json.dumps(fit_decision), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Published predictor"):
        runner._load_frozen_predictor(
            tmp_path,
            "0" * 64,
            config=config,
            config_sha256=runner._sha256(runner.DEFAULT_CONFIG),
            lock_sha256=lock_sha,
        )
    loaded, publication = runner._load_frozen_predictor(
        tmp_path,
        digest,
        config=config,
        config_sha256=runner._sha256(runner.DEFAULT_CONFIG),
        lock_sha256=lock_sha,
    )
    assert loaded == predictor
    assert publication["procedural_external_publication_attested"] is True


def test_phase_boundary_rejects_prior_evaluation_generation(tmp_path: Path) -> None:
    manifest = {
        "status": "fit_completed_evaluation_locked",
        "evaluation_trajectory_count_generated": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "fit_decision.json").write_text(
        json.dumps({"all_validity_checks_pass": True}), encoding="utf-8"
    )
    (tmp_path / "frozen_predictor.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="before the phase boundary"):
        runner._load_frozen_predictor(
            tmp_path,
            runner._sha256(tmp_path / "frozen_predictor.json"),
            config=_config(),
            config_sha256=runner._sha256(runner.DEFAULT_CONFIG),
            lock_sha256="e" * 64,
        )


def test_lock_publication_attestation_is_exact() -> None:
    with pytest.raises(RuntimeError, match="Published lock"):
        runner._attest("a" * 64, "b" * 64, name="lock")
    result = runner._attest("A" * 64, "a" * 64, name="lock")
    assert result["published_lock_sha256"] == "a" * 64


def test_frozen_predictor_schema_rejects_unknown_or_nonfinite_values() -> None:
    config = _config()
    predictor = _predictor(config, "e" * 64)
    runner._validate_frozen_predictor(
        predictor,
        config=config,
        config_sha256=runner._sha256(runner.DEFAULT_CONFIG),
        lock_sha256="e" * 64,
    )
    predictor["unknown"] = True
    with pytest.raises(RuntimeError, match="schema"):
        runner._validate_frozen_predictor(
            predictor,
            config=config,
            config_sha256=runner._sha256(runner.DEFAULT_CONFIG),
            lock_sha256="e" * 64,
        )
