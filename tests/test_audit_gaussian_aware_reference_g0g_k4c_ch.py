"""Focused tests for the independent K4c-CH post-run auditor."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
import yaml

from scripts import audit_gaussian_aware_reference_g0g_k4c_ch as audit


def test_auditor_stops_at_incomplete_manifest_before_any_other_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": audit.CAMPAIGN_ID,
                "status": "running_development",
            }
        ),
        encoding="utf-8",
    )

    def forbidden_csv(*args: object, **kwargs: object) -> None:
        raise AssertionError("raw CSV accessed before completed manifest")

    monkeypatch.setattr(audit, "_read_csv", forbidden_csv)
    with pytest.raises(RuntimeError, match="raw artifacts were not opened"):
        audit.audit_results(tmp_path / "missing.yaml", tmp_path)


def test_auditor_does_not_import_the_scientific_runner() -> None:
    source = inspect.getsource(audit)
    forbidden = "import run_gaussian_aware_reference_g0g_k4c_causal_headroom"
    assert forbidden not in source
    assert "from scripts.run_gaussian_aware_reference_g0g_k4c" not in source


def _minimal_config() -> dict:
    return {
        "randomness": {
            "development_outer_seeds": [1, 2],
        },
        "privacy_noise": {
            "regimes": [
                {"name": "homogeneous", "permutations": ["identity"]},
                {"name": "heteroscedastic", "permutations": ["identity"]},
            ]
        },
        "cohort": {
            "honest_outliers": {"geometries": ["aligned"]},
            "num_clients": 2,
        },
        "honest_dynamics": {"names": ["stationary"]},
        "threats": {"names": ["bitflip_x10"]},
        "temporal": {"assessment_rounds": [17]},
        "nested_monte_carlo": {
            "construction_children": 2,
            "evaluation_children": 1,
            "construction_stream_tag": "construct",
            "evaluation_stream_tag": "evaluate",
        },
        "statistical_analysis": {
            "construction_child_seeds_total": 8,
            "evaluation_child_seeds_total": 4,
        },
        "gates": {
            "eligible_histories_per_seed_noise_cell_exact": 1,
        },
        "references": {"total_client_influence_cap": 0.13},
    }


def _history(seed: int, regime: str) -> dict:
    permutation = "identity"
    identifier = f"{seed}|{regime}|{permutation}|aligned|stationary|bitflip_x10|17"
    return {
        "history_id": identifier,
        "seed": seed,
        "noise_regime": regime,
        "noise_permutation": permutation,
        "outlier_geometry": "aligned",
        "honest_dynamics": "stationary",
        "threat": "bitflip_x10",
        "assessment_round": 17,
        "eligible_R_positive": True,
        "split_a_b_aggregate_scale_distance": 0.01,
    }


def _child(history: dict, candidate: str, error: float) -> dict:
    return {
        "history_id": history["history_id"],
        "seed": history["seed"],
        "candidate": candidate,
        "squared_reference_error": error,
    }


def test_seed_ratios_use_all_R_positive_histories_and_ratio_of_sums() -> None:
    config = _minimal_config()
    histories: list[dict] = []
    children: list[dict] = []
    for seed in (1, 2):
        for index, regime in enumerate(("homogeneous", "heteroscedastic")):
            history = _history(seed, regime)
            histories.append(history)
            errors = {
                audit.K2: 1.1,
                audit.K4: 1.0,
                audit.K4B: 0.9,
                audit.SEMI_ORACLE: 0.8 if index == 0 else 1.1,
                audit.POINTWISE: 0.5 if index == 0 else 1.0,
            }
            children.extend(
                _child(history, candidate, error) for candidate, error in errors.items()
            )

    rows = audit._recompute_seed_rows(config, histories, children)
    assert len(rows) == 2
    assert all(row["eligible_histories"] == 2 for row in rows)
    assert all(row["capture_fraction"] == pytest.approx(0.2) for row in rows)
    assert all(
        row["pointwise_relative_mse_headroom_vs_k4"] == pytest.approx(0.25)
        for row in rows
    )
    assert all(
        row["semi_oracle_relative_mse_gain_vs_k4"] == pytest.approx(0.05)
        for row in rows
    )
    assert all(
        row["positive_finite_pointwise_headroom_denominators"] is False for row in rows
    )


def test_exact_matrix_rejects_duplicate_even_when_row_count_is_unchanged() -> None:
    config = _minimal_config()
    histories = [
        _history(seed, regime)
        for seed in (1, 2)
        for regime in (
            "homogeneous",
            "heteroscedastic",
        )
    ]
    children: list[dict] = []
    tag = config["nested_monte_carlo"]["evaluation_stream_tag"]
    for history in histories:
        for candidate in audit.CANDIDATES:
            child = _child(history, candidate, 1.0)
            child.update(
                {
                    "noise_regime": history["noise_regime"],
                    "noise_permutation": history["noise_permutation"],
                    "outlier_geometry": history["outlier_geometry"],
                    "honest_dynamics": history["honest_dynamics"],
                    "threat": history["threat"],
                    "assessment_round": 17,
                    "evaluation_child": 0,
                    "evaluation_child_seed": audit._stable_seed(
                        tag,
                        history["seed"],
                        history["noise_regime"],
                        history["noise_permutation"],
                        history["outlier_geometry"],
                        history["honest_dynamics"],
                        history["threat"],
                        17,
                        0,
                    ),
                    "crn_key": f"{history['history_id']}|evaluation|0",
                }
            )
            children.append(child)
        history.update(
            {
                "construction_children": 2,
                "evaluation_children": 1,
                "construction_evaluation_seed_overlap": 0,
            }
        )
        specification = {
            "seed": history["seed"],
            "noise_regime": history["noise_regime"],
            "noise_permutation": history["noise_permutation"],
            "outlier_geometry": history["outlier_geometry"],
            "honest_dynamics": history["honest_dynamics"],
            "threat": history["threat"],
            "assessment_round": history["assessment_round"],
        }
        construction = [
            audit._stable_seed(
                config["nested_monte_carlo"]["construction_stream_tag"],
                *specification.values(),
                child,
            )
            for child in range(2)
        ]
        evaluation = [
            audit._stable_seed(
                config["nested_monte_carlo"]["evaluation_stream_tag"],
                *specification.values(),
                child,
            )
            for child in range(1)
        ]
        construction_payload = ",".join(str(value) for value in construction)
        evaluation_payload = ",".join(str(value) for value in evaluation)
        history.update(
            {
                "construction_child_seed_registry": construction_payload,
                "evaluation_child_seed_registry": evaluation_payload,
                "construction_child_seed_registry_sha256": hashlib.sha256(
                    construction_payload.encode("utf-8")
                ).hexdigest(),
                "evaluation_child_seed_registry_sha256": hashlib.sha256(
                    evaluation_payload.encode("utf-8")
                ).hexdigest(),
                "construction_child_seed_unique_count": len(set(construction)),
                "evaluation_child_seed_unique_count": len(set(evaluation)),
            }
        )
    passed, _ = audit._matrix_audit(config, histories, children)
    assert passed

    duplicated = list(children)
    duplicated[-1] = dict(duplicated[0])
    passed, details = audit._matrix_audit(config, histories, duplicated)
    assert not passed
    assert details["checks"]["child_keys_exact"] is False


def test_three_state_decision_never_turns_invalidity_into_scientific_stop() -> None:
    assert (
        audit._decision_status({"valid": False}, {"gain": False})
        == "invalid_or_inconclusive_screen"
    )
    assert (
        audit._decision_status({"valid": True}, {"gain": False})
        == "stop_missing_slot_imputation_branch"
    )
    assert (
        audit._decision_status({"valid": True}, {"gain": True})
        == "authorize_transcript_only_predictor_study"
    )


def test_student_interval_uses_configured_outer_seed_units() -> None:
    result = audit._ci([1.0, 2.0, 3.0], 4.302652729911275)
    expected_half = 4.302652729911275 / math.sqrt(3.0)
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["sd"] == pytest.approx(1.0)
    assert result["low"] == pytest.approx(2.0 - expected_half)
    assert result["high"] == pytest.approx(2.0 + expected_half)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _full_audit_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    zero_headroom: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / ("zero" if zero_headroom else "positive")
    root.mkdir()
    results = root / "results"
    results.mkdir()
    config_path = root / "config.yaml"
    lock_path = root / "lock.json"
    locked_path = root / "locked.txt"
    dependency_path = root / "dependency.txt"
    locked_path.write_text("locked\n", encoding="utf-8")
    dependency_path.write_text("dependency\n", encoding="utf-8")

    config = yaml.safe_load(audit.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["preregistration_lock"]["path"] = str(lock_path)
    config["randomness"]["development_outer_seeds"] = [1, 2]
    config["privacy_noise"]["regimes"] = [
        {"name": "homogeneous", "permutations": ["identity"]},
        {"name": "heteroscedastic", "permutations": ["identity"]},
    ]
    config["cohort"]["num_clients"] = 2
    config["cohort"]["honest_outliers"]["geometries"] = ["aligned"]
    config["honest_dynamics"]["names"] = ["stationary"]
    config["threats"]["names"] = ["bitflip_x10"]
    config["temporal"]["assessment_rounds"] = [17]
    config["nested_monte_carlo"].update(
        {
            "construction_children": 2,
            "construction_split_a_children": 1,
            "construction_split_b_children": 1,
            "evaluation_children": 1,
        }
    )
    config["statistical_analysis"].update(
        {
            "num_independent_units": 2,
            "histories_per_outer_seed": 2,
            "frozen_histories_total": 4,
            "construction_child_seeds_total": 8,
            "evaluation_child_seeds_total": 4,
            "t_critical_df11": 12.706204736432095,
        }
    )
    config["gates"].update(
        {
            "construction_child_seed_exact_count": 8,
            "evaluation_child_seed_exact_count": 4,
            "eligible_histories_per_seed_noise_cell_exact": 1,
            "exact_seed_capture_count": 2,
            "replace_one_exact_trials": 8,
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(audit, "DEFAULT_LOCK", lock_path)
    monkeypatch.setattr(audit, "LOCKED_PATHS", {"locked.txt"})
    monkeypatch.setattr(audit, "DEPENDENCY_PATHS", {"dependency.txt"})

    histories: list[dict] = []
    children: list[dict] = []
    construction_tag = config["nested_monte_carlo"]["construction_stream_tag"]
    evaluation_tag = config["nested_monte_carlo"]["evaluation_stream_tag"]
    for seed in (1, 2):
        for regime in ("homogeneous", "heteroscedastic"):
            specification = (
                seed,
                regime,
                "identity",
                "aligned",
                "stationary",
                "bitflip_x10",
                17,
            )
            history_id = "|".join(str(value) for value in specification)
            construction = [
                audit._stable_seed(construction_tag, *specification, child)
                for child in range(2)
            ]
            evaluation = [audit._stable_seed(evaluation_tag, *specification, 0)]
            construction_text = ",".join(str(value) for value in construction)
            evaluation_text = ",".join(str(value) for value in evaluation)
            histories.append(
                {
                    "history_id": history_id,
                    "seed": seed,
                    "noise_regime": regime,
                    "noise_permutation": "identity",
                    "outlier_geometry": "aligned",
                    "honest_dynamics": "stationary",
                    "threat": "bitflip_x10",
                    "assessment_round": 17,
                    "construction_children": 2,
                    "evaluation_children": 1,
                    "construction_evaluation_seed_overlap": 0,
                    "construction_child_seed_registry": construction_text,
                    "evaluation_child_seed_registry": evaluation_text,
                    "construction_child_seed_registry_sha256": hashlib.sha256(
                        construction_text.encode("utf-8")
                    ).hexdigest(),
                    "evaluation_child_seed_registry_sha256": hashlib.sha256(
                        evaluation_text.encode("utf-8")
                    ).hexdigest(),
                    "construction_child_seed_unique_count": 2,
                    "evaluation_child_seed_unique_count": 1,
                    "missing_slot_mass": 1.0,
                    "eligible_R_positive": True,
                    "rolling_predictor_norm": 0.01,
                    "semi_oracle_predictor_norm": 0.02,
                    "semi_oracle_raw_predictor_norm": 0.02,
                    "semi_oracle_projection_active": False,
                    "split_a_b_predictor_distance": 0.0,
                    "split_a_b_aggregate_scale_distance": 0.0,
                    "semi_oracle_uses_evaluation_target": False,
                    "semi_oracle_uses_evaluation_noise": False,
                    "semi_oracle_uses_k4b_feedback": False,
                    "semi_oracle_conditions_on_latent_clean_current_state": True,
                    "semi_oracle_uses_simulator_honest_byzantine_labels": True,
                    "semi_oracle_knows_configured_current_noise_attack_dgp": True,
                    "arbitrary_adaptive_byzantine_behavior_supported": False,
                    "semi_oracle_is_observable_past_measurable": False,
                    "projection_applied_once_after_expectations": True,
                    "pointwise_oracles_averaged": False,
                }
            )
            errors = (
                {
                    audit.K2: 1.1,
                    audit.K4: 1.0,
                    audit.K4B: 1.0,
                    audit.SEMI_ORACLE: 1.0,
                    audit.POINTWISE: 1.0,
                }
                if zero_headroom
                else {
                    audit.K2: 1.1,
                    audit.K4: 1.0,
                    audit.K4B: 0.8,
                    audit.SEMI_ORACLE: 0.5,
                    audit.POINTWISE: 0.4,
                }
            )
            for candidate, squared_error in errors.items():
                pointwise_excess = (
                    errors[audit.POINTWISE] - squared_error
                    if candidate in {audit.K4, audit.K4B, audit.SEMI_ORACLE}
                    else 0.0
                )
                children.append(
                    {
                        "history_id": history_id,
                        "seed": seed,
                        "noise_regime": regime,
                        "noise_permutation": "identity",
                        "outlier_geometry": "aligned",
                        "honest_dynamics": "stationary",
                        "threat": "bitflip_x10",
                        "assessment_round": 17,
                        "evaluation_child": 0,
                        "evaluation_child_seed": evaluation[0],
                        "crn_key": f"{history_id}|evaluation|0",
                        "candidate": candidate,
                        "squared_reference_error": squared_error,
                        "reference_error_l2_descriptive": math.sqrt(squared_error),
                        "pointwise_excess_mse_over_candidate": pointwise_excess,
                        "missing_slot_mass": 1.0,
                        "predictor_norm": 0.02,
                        "k4_manual_formula_error": 0.0,
                        "frozen_k4_comparator_error_descriptive": 0.0,
                        "k4b_comparator_reproduction_error": 0.0,
                        "fixed_denominator_formula_error": 0.0,
                        "all_finite": True,
                        "contribution_cap_respected": True,
                        "semi_oracle_predictor_hash": f"p-{history_id}",
                        "semi_oracle_predictor_fixed_across_evaluation_children": True,
                        "construction_stream_read": candidate == audit.SEMI_ORACLE,
                        "pointwise_oracle": candidate == audit.POINTWISE,
                        "pointwise_oracle_used_in_construction": False,
                        "conditions_on_latent_clean_current_state": candidate
                        == audit.SEMI_ORACLE,
                        "normalization_by_gate_sum": False,
                        "fixed_denominator_n": 2,
                    }
                )

    replacements = []
    bound = 2.0 * 0.13 / 2.0
    for seed in (1, 2):
        for regime in ("homogeneous", "heteroscedastic"):
            history_id = f"{seed}|{regime}|identity|aligned|stationary|bitflip_x10|17"
            for client in range(2):
                replacements.append(
                    {
                        "seed": seed,
                        "noise_regime": regime,
                        "noise_permutation": "identity",
                        "trial": client,
                        "replaced_client": client,
                        "history_id": history_id,
                        "same_past": True,
                        "same_predictor": True,
                        "same_history_gate": True,
                        "certificate_scope": (
                            "evaluation_map_only_with_fixed_anchor_past_history_"
                            "radii_and_semi_oracle_predictor"
                        ),
                        "end_to_end_semi_oracle_sensitivity_claimed": False,
                        "observed_difference": 0.0,
                        "theoretical_bound": bound,
                        "ratio_to_bound": 0.0,
                        "violation": False,
                    }
                )

    provisional_manifest = {"device": "mps"}
    matrix_exact, _ = audit._matrix_audit(config, histories, children)
    replacement_exact, _ = audit._replace_coverage(config, replacements)
    assert matrix_exact and replacement_exact
    seed_rows = audit._recompute_seed_rows(config, histories, children)
    decision = audit._recompute_decision(
        config,
        provisional_manifest,
        histories,
        children,
        replacements,
        seed_rows,
        matrix_exact=True,
        replacement_exact=True,
    )
    _write_csv(results / "frozen_history_rows.csv", histories)
    _write_csv(results / "evaluation_child_rows.csv", children)
    _write_csv(results / "seed_summary.csv", seed_rows)
    _write_csv(results / "replace_one_audit.csv", replacements)
    (results / "decision.json").write_text(
        json.dumps(audit._json_safe(decision), allow_nan=False),
        encoding="utf-8",
    )

    lock = {
        "schema_version": 1,
        "campaign_id": audit.CAMPAIGN_ID,
        "locked_files": {"locked.txt": audit._sha256(locked_path)},
        "dependencies": {"dependency.txt": audit._sha256(dependency_path)},
        "lock_file_self_hash_embedded": False,
        "publication_requirement": (
            "publish_this_lock_file_sha256_in_research_log_or_chat_before_run"
        ),
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha = audit._sha256(lock_path)
    manifest = {
        "campaign_id": audit.CAMPAIGN_ID,
        "status": "completed_development",
        "device": "mps",
        "dtype": "torch.float32",
        "development_only": True,
        "holdout_opened": False,
        "observable_past_only_predictor_constructed": False,
        "pointwise_oracle_role": (
            "nondeployable_decision_benchmark_headroom_denominator_only"
        ),
        "semi_oracle_is_finite_monte_carlo_approximation": True,
        "development_decision": decision["decision"],
        "all_gates_pass": decision["all_gates_pass"],
        "frozen_histories": len(histories),
        "evaluation_child_rows": len(children),
        "config_sha256": audit._sha256(config_path),
        "preregistration_lock": {
            "verified": True,
            "path": str(lock_path.resolve()),
            "sha256": lock_sha,
            "locked_files": 1,
            "dependencies": 1,
        },
        "lock_publication_attestation": {
            "procedural_external_publication_attested": True,
            "published_lock_sha256": lock_sha,
            "machine_verifies_external_log_itself": False,
        },
    }
    (results / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return config_path, results


def test_compact_full_artifact_audit_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, results = _full_audit_fixture(tmp_path, monkeypatch)
    outcome = audit.audit_results(config, results)
    assert outcome["all_checks_pass"] is True


@pytest.mark.parametrize(
    "tamper", ["lock", "matrix", "replace", "manual_k4", "nonfinite"]
)
def test_full_artifact_audit_fails_closed_on_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    config, results = _full_audit_fixture(tmp_path, monkeypatch)
    if tamper == "lock":
        (audit.ROOT / "locked.txt").write_text("tampered\n", encoding="utf-8")
    elif tamper == "matrix":
        rows = audit._read_csv(results / "evaluation_child_rows.csv")
        rows[-1]["evaluation_child_seed"] = "7"
        _write_csv(results / "evaluation_child_rows.csv", rows)
    elif tamper == "replace":
        rows = audit._read_csv(results / "replace_one_audit.csv")
        rows[-1]["replaced_client"] = "0"
        _write_csv(results / "replace_one_audit.csv", rows)
    elif tamper == "manual_k4":
        rows = audit._read_csv(results / "evaluation_child_rows.csv")
        rows[-1]["k4_manual_formula_error"] = "0.1"
        _write_csv(results / "evaluation_child_rows.csv", rows)
    else:
        rows = audit._read_csv(results / "evaluation_child_rows.csv")
        rows[-1]["squared_reference_error"] = "nan"
        _write_csv(results / "evaluation_child_rows.csv", rows)
    assert audit.audit_results(config, results)["all_checks_pass"] is False


def test_zero_headroom_is_a_valid_scientific_stop_with_standard_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, results = _full_audit_fixture(tmp_path, monkeypatch, zero_headroom=True)
    raw_decision = (results / "decision.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_decision
    assert (
        json.loads(raw_decision)["observed"]["semi_oracle_capture_fraction_seed_ci95"][
            "mean"
        ]
        is None
    )
    outcome = audit.audit_results(config, results)
    assert outcome["all_checks_pass"] is True
    assert outcome["recomputed_decision"]["validity_pass"] is True
    assert outcome["recomputed_decision"]["decision"] == (
        "stop_missing_slot_imputation_branch"
    )
