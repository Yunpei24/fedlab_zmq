from __future__ import annotations

import ast
import copy
import importlib
import json
from pathlib import Path

import pytest
import torch

from scripts import run_gaussian_aware_reference_g0g_k5_tp_v2 as runner


def _bundle() -> tuple[dict, dict]:
    return runner._load_amended_config(runner.DEFAULT_CONFIG)


def test_amended_config_is_valid_and_split_registries_are_exact() -> None:
    config, amendment = _bundle()
    runner._validate_config(config, amendment)
    assert tuple(config["features"]["order"]) == runner.FEATURE_NAMES
    assert config["gates"]["flattened_feature_rank_exact"] == 5
    groups = [
        set(config["randomness"][name])
        for name in (
            "train_outer_seeds",
            "calibration_outer_seeds",
            "evaluation_outer_seeds",
            "reserved_holdout_seeds",
        )
    ]
    assert [len(group) for group in groups] == [12, 8, 12, 7]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))
    assert config["campaign_id"] == "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
    assert config["execution"]["output_directory"] == str(
        runner.DEFAULT_OUTPUT.relative_to(runner.ROOT)
    )
    provenance = runner._technical_v2_mps_v1_failure_provenance(amendment, config)
    assert provenance["all_checks_pass"] is True
    assert provenance["fit_history_row_count"] == 0
    assert provenance["evaluation_trajectory_count"] == 0
    assert provenance["failure_boundary_machine_verified"] is True
    assert provenance["failure_cause_attested_not_machine_verified"] is True
    assert provenance["failure_cause_is_scientific_gate"] is False
    prior_resolved = json.loads(
        (runner.TECHNICAL_V2_MPS_V1_RESULTS / "resolved_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert runner._resolved_delta_paths(prior_resolved, config) == {
        ("campaign_id",),
        ("execution", "output_directory"),
        ("preregistration_lock", "path"),
    }
    tampered = copy.deepcopy(config)
    tampered["aggregation"]["server_clip_norm"] += 0.01
    with pytest.raises(RuntimeError, match="provenance audit failed"):
        runner._technical_v2_mps_v1_failure_provenance(amendment, tampered)


def test_structured_rng_registry_checks_every_derived_seed() -> None:
    config, _ = _bundle()
    streams = {
        "train_target": runner._expected_rng_stream_records(
            config,
            split="train",
            seeds=config["randomness"]["train_outer_seeds"][:1],
            stream="train_target",
            children=2,
        ),
        "calibration_target": [],
        "calibration_evaluation": [],
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    registry = runner._rng_registry(streams)
    assert runner._validate_rng_registry(registry, streams)
    tampered = copy.deepcopy(registry)
    tampered["streams"]["train_target"][0]["child_seeds"][0] += 1
    assert not runner._validate_rng_registry(tampered, streams)


def test_solution_gate_accepts_mps_and_mps_zero_semantically() -> None:
    config, _ = _bundle()
    base = {
        "fit_device": "mps",
        "fit_dtype": "torch.float32",
        "histories": 2,
        "dimension": config["cohort"]["dimension"],
        "ridge_lambda": 0.1,
        "condition_number_norm": "infinity_exact_via_solve",
        "condition_number_regularized_system": 2.0,
        "normal_equation_relative_residual": 1.0e-7,
        "aggregate_weight_sum": 1.0,
        "normalized_gram": [[1.0]],
        "normalized_rhs": [1.0],
        "coefficients": [0.5],
        "feature_scales": [1.0],
        "weighted_mse_before_projection": 0.1,
        "ridge_penalty": 0.01,
        "objective": 0.11,
        "coefficient_l2_norm": 0.5,
    }
    for device in ("mps", "mps:0"):
        row = {**base, "fit_device": device}
        assert runner._single_solution_safe(
            row,
            config,
            ridge_lambda=0.1,
            feature_count=1,
            histories=2,
            aggregate_weight_sum=1.0,
        )
    assert not runner._single_solution_safe(
        {**base, "fit_device": "cpu"},
        config,
        ridge_lambda=0.1,
        feature_count=1,
        histories=2,
        aggregate_weight_sum=1.0,
    )


def test_headroom_diagnostics_are_published_but_not_a_gate() -> None:
    config, _ = _bundle()
    histories = []
    children = []
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        history_id = f"synthetic-{seed}"
        histories.append(
            {
                "history_id": history_id,
                "seed": seed,
                "noise_regime": "homogeneous",
                "noise_permutation": "identity",
                "privileged_target_split_aggregate_scale_distance": 0.01,
                "feature_max_source_round": 16,
                "assessment_round": 17,
                "feature_source_rounds": "13,14,15,16",
                "forbidden_current_inference_field_count": 0,
            }
        )
        values = {
            runner.K2: 8.0,
            runner.K4: 10.0,
            runner.K4B: 9.0,
            runner.K5_1D: 8.5,
            runner.K5: 7.0,
            runner.K4C: 5.0,
            runner.POINTWISE: 4.0,
        }
        for candidate, value in values.items():
            children.append(
                {
                    "history_id": history_id,
                    "candidate": candidate,
                    "evaluation_child": 0,
                    "squared_reference_error": value,
                    "frozen_predictor_fixed": True,
                    "fixed_denominator_n": config["cohort"]["num_clients"],
                    "normalization_by_gate_sum": False,
                    "contribution_cap_respected": True,
                    "k4_manual_formula_error": 0.0,
                    "k4b_reproduction_error": 0.0,
                    "fixed_denominator_formula_error": 0.0,
                }
            )
    _, decision = runner._summarize_evaluation(
        config, histories, children, [], {"synthetic_validity": False}
    )
    diagnostics = decision["headroom_diagnostics_non_gating"]
    assert diagnostics["relative_headroom_minimum"] == 0.5
    assert diagnostics["changes_scientific_decision"] is False
    assert decision["validity_pass"] is False


def test_runner_source_has_no_holdout_generation_path() -> None:
    source = runner.Path(runner.__file__).read_text(encoding="utf-8")
    assert 'stream="holdout"' not in source
    assert "fit-freeze" in source and "evaluate-frozen" in source
    assert torch.device("mps:0").type == "mps"


def test_local_csv_reader_replaces_missing_v1_helper(tmp_path: Path) -> None:
    assert not hasattr(runner.v1, "_read_csv")
    artifact = tmp_path / "rows.csv"
    artifact.write_text("history_id,split\nh-1,train\n", encoding="utf-8")
    assert runner._read_csv(artifact) == [{"history_id": "h-1", "split": "train"}]
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("history_id,history_id\nh-1,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        runner._read_csv(malformed)


def test_every_project_symbol_imported_or_reused_by_runner_exists() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_aliases = {
        "k4": runner.k4,
        "k4b": runner.k4b,
        "k4c": runner.k4c,
        "oracle": runner.oracle,
        "v1": runner.v1,
    }
    missing_reused = sorted(
        {
            f"{node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
            and not hasattr(module_aliases[node.value.id], node.attr)
        }
    )
    assert missing_reused == []

    missing_imported: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(("algorithms.", "robustness.")):
            continue
        module = importlib.import_module(node.module)
        for symbol in node.names:
            if symbol.name != "*" and not hasattr(module, symbol.name):
                missing_imported.append(f"{node.module}.{symbol.name}")
    assert sorted(missing_imported) == []


def test_independent_fit_audit_binds_all_five_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    for name in runner.FIT_AUDITED_ARTIFACT_NAMES:
        (tmp_path / name).write_text(f"frozen:{name}\n", encoding="utf-8")
    hashes = {
        name: runner._sha256(tmp_path / name)
        for name in runner.FIT_AUDITED_ARTIFACT_NAMES
    }
    runner._write_json(
        tmp_path / "independent_fit_audit.json",
        {
            "schema_version": 1,
            "campaign_id": runner.CAMPAIGN_ID,
            "audit_scope": "fit_and_freeze_pre_evaluation",
            "independent_of_v2_runner": True,
            "checks": {"all_fit_checks": True},
            "audited_artifact_sha256": hashes,
            "evaluation_present": False,
            "holdout_opened": False,
            "all_checks_pass": True,
        },
    )
    published_audit_sha256 = runner._sha256(tmp_path / "independent_fit_audit.json")
    attestation = runner._verify_independent_fit_audit(tmp_path, published_audit_sha256)
    assert attestation["verified_before_evaluation"] is True
    assert attestation["pre_evaluation_external_anchor_required"] is True
    assert (
        attestation["publication_attestation"]["published_fit_audit_sha256"]
        == published_audit_sha256
    )
    assert set(attestation["audited_artifact_sha256"]) == set(
        runner.FIT_AUDITED_ARTIFACT_NAMES
    )
    assert len(attestation["sha256"]) == 64

    (tmp_path / "fit_decision.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="audited artifacts changed"):
        runner._verify_independent_fit_audit(tmp_path, published_audit_sha256)

    (tmp_path / "fit_decision.json").write_text(
        "frozen:fit_decision.json\n", encoding="utf-8"
    )
    audit_path = tmp_path / "independent_fit_audit.json"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["audited_artifact_sha256"]["manifest.json"] = "b" * 64
    runner._write_json(audit_path, audit_payload)
    with pytest.raises(RuntimeError, match="Published fit_audit SHA-256"):
        runner._verify_independent_fit_audit(tmp_path, published_audit_sha256)


def test_independent_fit_audit_rejects_incomplete_hash_inventory(
    tmp_path: Path,
) -> None:
    runner._write_json(
        tmp_path / "independent_fit_audit.json",
        {
            "schema_version": 1,
            "campaign_id": runner.CAMPAIGN_ID,
            "audit_scope": "fit_and_freeze_pre_evaluation",
            "independent_of_v2_runner": True,
            "checks": {"all_fit_checks": True},
            "audited_artifact_sha256": {"frozen_predictor.json": "a" * 64},
            "evaluation_present": False,
            "holdout_opened": False,
            "all_checks_pass": True,
        },
    )
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        runner._verify_independent_fit_audit(
            tmp_path, runner._sha256(tmp_path / "independent_fit_audit.json")
        )


def test_evaluation_cli_requires_published_fit_audit_sha256() -> None:
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--phase",
                "evaluate-frozen",
                "--published-lock-sha256",
                "a" * 64,
                "--published-predictor-sha256",
                "b" * 64,
            ]
        )
    assert error.value.code == 2
