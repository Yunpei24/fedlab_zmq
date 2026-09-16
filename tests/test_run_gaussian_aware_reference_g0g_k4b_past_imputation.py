"""Contract tests for the preregistered G0g-K4b development runner."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4
from scripts import run_gaussian_aware_reference_g0g_k4b_past_imputation as k4b
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.yaml"
)
LOCK = (
    ROOT
    / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k4b_past_imputation.lock.json"
)


@pytest.fixture
def production_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def reduced_config(production_config: dict) -> dict:
    config = copy.deepcopy(production_config)
    config["cohort"].update(
        {
            "num_clients": 6,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    config["cohort"]["honest_outliers"].update({"count": 1, "geometries": ["aligned"]})
    config["privacy_noise"].update(
        {
            "block_std_multipliers": [0.8, 1.4],
            "regimes": [
                {
                    "name": "heteroscedastic",
                    "client_std_multipliers": [1.0, 1.5, 2.0],
                    "permutations": ["byzantine_high"],
                }
            ],
        }
    )
    config["temporal"]["standardized_clip_norm"] = 2.0 * math.sqrt(8.0)
    config["honest_dynamics"]["names"] = ["stationary"]
    config["past_imputation"].update(
        {
            "maximum_byzantine_clients": 2,
            "minimum_accepted_mass": 4,
        }
    )
    config["threats"].update(
        {
            "names": ["ipm"],
            "separated_for_primary_gate": ["ipm"],
            "oracle_headroom_threats": ["ipm"],
        }
    )
    config["randomness"].update(
        {
            "development_seeds": [7201, 7213],
            "replace_one_trials_per_seed_noise_cell": 1,
        }
    )
    oracle._configure_runtime("cpu")
    return config


def test_production_config_hashes_seeds_and_exact_matrix(
    production_config: dict,
) -> None:
    k4b._validate_config(production_config)
    _, temporal, provenance = k4b._load_frozen_calibrations(production_config)
    assert provenance["k2"]["sha256_verified"] is True
    assert provenance["k4_temporal"]["observed_sha256"] == (
        "6ca81b5b8897ec19509a569b54fdf9edeac8c6dc65c7b0c4a890142882603075"
    )
    assert temporal["deployed_c0"] == pytest.approx(k4b.FROZEN_C0)
    assert temporal["deployed_c1"] == pytest.approx(k4b.FROZEN_C1)
    cells = k4b._development_cells(production_config)
    assert len(cells) == 900
    assert len({cell["trajectory_id"] for cell in cells}) == 900
    assert sum(cell["schedule"] == "no_compromise" for cell in cells) == 100
    new_seeds = set(production_config["randomness"]["development_seeds"]) | set(
        production_config["randomness"]["holdout_seeds"]
    )
    prior: set[int] = set()
    for path in production_config["prior_campaign_seed_registry"]:
        prior |= k4b._seed_values(
            yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        )
    assert not (new_seeds & prior)
    assert (
        production_config["scientific_contract"][
            "screen_conditioned_on_semi_oracle_anchor"
        ]
        is True
    )
    assert (
        production_config["scientific_contract"]["end_to_end_deployability_claimed"]
        is False
    )
    assert k4b.DELTA in production_config["candidates"]["names"]


def test_predictor_rounds_are_strictly_past_and_static_window_is_fixed(
    reduced_config: dict,
) -> None:
    cap = float(reduced_config["references"]["total_client_influence_cap"])
    residuals = {
        round_index: torch.full((6, 8), round_index / 1000.0)
        for round_index in range(1, 14)
    }
    # Ensure every residual is a valid already-clipped c_i.
    residuals = {
        key: value / max(1.0, float(torch.linalg.vector_norm(value[0])) / cap)
        for key, value in residuals.items()
    }
    gates = {key: torch.ones(6) for key in residuals}
    _, rolling = k4b._predictor_from_rounds(
        residuals, gates, [9, 10, 11, 12], config=reduced_config
    )
    _, static = k4b._predictor_from_rounds(
        residuals,
        gates,
        reduced_config["past_imputation"]["static_clean_window"],
        config=reduced_config,
    )
    assert rolling["maximum_source_round"] == 12 < 13
    assert static["source_rounds"] == [9, 10, 11, 12]


def test_frozen_k4_comparator_reproduces_original_exactly(
    reduced_config: dict,
) -> None:
    k2_calibration, temporal_calibration, _ = k4b._load_frozen_calibrations(
        reduced_config
    )
    regime = reduced_config["privacy_noise"]["regimes"][0]
    components = k4._trajectory_components(
        reduced_config,
        seed=7201,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
    )
    radii = k4._radii(
        reduced_config,
        components["variances"],
        k2_calibration,
        regime_name="heteroscedastic",
        blind=False,
    )
    history = []
    enrollment = None
    vectors = None
    for round_index in range(1, 14):
        observed = k4._private_vectors(
            reduced_config,
            components,
            seed=7201,
            round_index=round_index,
            geometry="aligned",
        )
        vectors = k4.clip_l2(
            observed, float(reduced_config["aggregation"]["server_clip_norm"])
        )
        standardized = k4._standardized_messages(
            reduced_config,
            vectors,
            anchor=components["anchor"],
            variances=components["variances"],
        )
        if round_index == 8:
            enrollment = torch.stack(history + [standardized]).mean(dim=0)
        if round_index < 13:
            history.append(standardized)
    assert vectors is not None and enrollment is not None
    left, left_diag = k4b._frozen_k4_comparator(
        vectors,
        anchor=components["anchor"],
        aware_radii=radii,
        history=history,
        enrollment_mean=enrollment,
        temporal_calibration=temporal_calibration,
        config=reduced_config,
    )
    right, right_diag = k4._k4_reference(
        vectors,
        anchor=components["anchor"],
        aware_radii=radii,
        history=history,
        enrollment_mean=enrollment,
        thresholds=temporal_calibration,
        config=reduced_config,
    )
    assert torch.equal(left, right)
    assert left_diag["gates_by_client"] == right_diag["gates_by_client"]


def test_one_reduced_trajectory_has_strict_crn_and_explicit_oracle_labels(
    reduced_config: dict,
) -> None:
    k2_calibration, temporal_calibration, _ = k4b._load_frozen_calibrations(
        reduced_config
    )
    regime = reduced_config["privacy_noise"]["regimes"][0]
    round_rows, client_rows, trajectory_rows = k4b._evaluate_trajectory(
        reduced_config,
        k2_calibration,
        temporal_calibration,
        seed=7201,
        regime=regime,
        permutation="byzantine_high",
        geometry="aligned",
        dynamics="stationary",
        threat="ipm",
        schedule="persistent",
    )
    assert len(round_rows) == 36 * len(k4b.CANDIDATES)
    assert len(client_rows) == 24 * 6
    assert len(trajectory_rows) == len(k4b.CANDIDATES)
    by_round: dict[int, set[str]] = {}
    for row in round_rows:
        by_round.setdefault(int(row["round"]), set()).add(str(row["crn_key"]))
    assert all(len(keys) == 1 for keys in by_round.values())
    assert all(not bool(row["end_to_end_deployability_claimed"]) for row in round_rows)
    assert all(
        bool(row["screen_conditioned_on_semi_oracle_anchor"]) for row in round_rows
    )
    primary = [row for row in round_rows if row["candidate"] == k4b.PRIMARY]
    assert all(bool(row["predictor_is_strictly_past"]) for row in primary)
    assert not any(bool(row["predictor_feedback_from_k4b_output"]) for row in primary)
    assert all(
        row["predictor_max_source_round"] is None
        or int(row["predictor_max_source_round"]) < int(row["round"])
        for row in primary
    )
    oracle_rows = [row for row in round_rows if row["candidate"] == k4b.ORACLE]
    assert all(not bool(row["deployable"]) for row in oracle_rows)
    assert all(not bool(row["privacy_claimed"]) for row in oracle_rows)
    assert all(
        bool(row["uses_oracle_target"]) == (int(row["round"]) >= 13)
        for row in oracle_rows
    )
    before = [row for row in primary if int(row["round"]) < 13]
    assert max(float(row["difference_to_k2"]) for row in before) == 0.0
    after_oracle = [row for row in oracle_rows if int(row["round"]) >= 13]
    after_k4 = {
        int(row["round"]): row
        for row in round_rows
        if row["candidate"] == k4b.K4 and int(row["round"]) >= 13
    }
    assert all(
        float(row["reference_error"])
        <= float(after_k4[int(row["round"])]["reference_error"]) + 1.0e-6
        for row in after_oracle
    )
    assert all(
        "byzantine_direct_current_mass_share" in row
        and "byzantine_imputed_mass_share" in row
        and "byzantine_total_slot_mass_share" in row
        for row in round_rows
    )


def test_current_replace_one_audit_uses_fixed_past_predictor_and_bound(
    reduced_config: dict,
) -> None:
    k2_calibration, temporal_calibration, _ = k4b._load_frozen_calibrations(
        reduced_config
    )
    rows = k4b._replace_one_audit(reduced_config, k2_calibration, temporal_calibration)
    assert len(rows) == 4
    assert all(row["same_past"] for row in rows)
    assert all(row["same_predictor"] for row in rows)
    assert all(row["same_history_gate"] for row in rows)
    assert all(
        float(row["theoretical_replace_one_bound"]) == pytest.approx(2 * 0.13 / 6)
        for row in rows
    )
    assert not any(row["violation"] for row in rows)
    assert not any(row["history_gate_has_suppression"] for row in rows)
    assert {row["candidate"] for row in rows} == {k4b.PRIMARY, k4b.DELTA}


def test_production_attacked_replace_one_cell_has_h_below_one_and_no_violation(
    production_config: dict,
) -> None:
    config = copy.deepcopy(production_config)
    config["randomness"]["development_seeds"] = [
        production_config["randomness"]["development_seeds"][0]
    ]
    config["randomness"]["replace_one_trials_per_seed_noise_cell"] = 1
    oracle._configure_runtime("cpu")
    k2_calibration, temporal_calibration, _ = k4b._load_frozen_calibrations(config)
    rows = k4b._attacked_replace_one_audit(config, k2_calibration, temporal_calibration)
    assert len(rows) == 2
    assert {row["candidate"] for row in rows} == {k4b.PRIMARY, k4b.DELTA}
    assert all(float(row["history_gate_min"]) < 1.0 for row in rows)
    assert all(row["same_past"] and row["same_predictor"] for row in rows)
    assert not any(row["violation"] for row in rows)


def test_completeness_checks_exact_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        k4b, "_development_cells", lambda _config: [{"trajectory_id": "t"}]
    )
    config = {
        "temporal": {"total_rounds": 3, "first_temporal_gate_round": 2},
        "cohort": {"num_clients": 2},
        "randomness": {
            "development_seeds": [1],
            "replace_one_trials_per_seed_noise_cell": 1,
        },
    }
    monkeypatch.setattr(
        k4b.base,
        "_noise_cells",
        lambda _config: [({"name": "homogeneous"}, "identity")],
    )
    rounds = [
        {"trajectory_id": "t", "candidate": candidate, "round": round_index}
        for candidate in k4b.CANDIDATES
        for round_index in range(1, 4)
    ]
    trajectories = [
        {"trajectory_id": "t", "candidate": candidate} for candidate in k4b.CANDIDATES
    ]
    clients = [
        {"trajectory_id": "t", "round": round_index, "client": client}
        for round_index in range(2, 4)
        for client in range(2)
    ]
    replacements = [
        {
            "audit_scenario": scenario,
            "candidate": candidate,
            "seed": 1,
            "noise_regime": regime,
            "noise_permutation": permutation,
            "trial": 0,
        }
        for scenario, regime, permutation in (
            ("clean_first_causal_gate", "homogeneous", "identity"),
            (
                "persistent_bitflip_round17_h_below_one",
                "heteroscedastic",
                "byzantine_high",
            ),
        )
        for candidate in (k4b.PRIMARY, k4b.DELTA)
    ]
    complete = k4b._screen_completeness(
        rounds, clients, trajectories, replacements, config
    )
    assert complete["exact_identifier_completeness"] is True
    incomplete = k4b._screen_completeness(
        rounds[:-1], clients, trajectories, replacements, config
    )
    assert incomplete["exact_identifier_completeness"] is False


def test_runner_is_mps_only_and_has_no_holdout_evaluation_path(
    production_config: dict,
) -> None:
    source = inspect.getsource(k4b.run)
    main_source = inspect.getsource(k4b.main)
    assert 'oracle._configure_runtime("mps")' in source
    assert 'choices=("mps",)' in main_source
    assert "holdout_seeds" not in inspect.getsource(k4b._screen_development)
    assert production_config["execution"]["allow_cpu_fallback"] is False
    assert production_config["execution"]["holdout_code_path_present"] is False


def test_paired_seed_log_ratios_preserve_sign_and_use_exactly_five_pairs() -> None:
    rows = []
    for seed, ratio in zip(range(5), [0.95, 0.97, 1.0, 1.01, 0.99], strict=True):
        rows.extend(
            [
                {"seed": seed, "candidate": k4b.PRIMARY, "metric": 10.0 * ratio},
                {"seed": seed, "candidate": k4b.K4, "metric": 10.0},
            ]
        )
    result = k4b._paired_seed_log_ratios(
        rows,
        candidate=k4b.PRIMARY,
        baseline=k4b.K4,
        predicate=lambda _row: True,
        metric="metric",
    )
    assert result["n_seed_pairs"] == 5
    assert result["ratios_by_seed"]["0"] == pytest.approx(0.95)
    assert result["ratios_by_seed"]["3"] == pytest.approx(1.01)
    assert result["log_ratios_by_seed"]["0"] == pytest.approx(math.log(0.95))
    assert result["log_ratios_by_seed"]["3"] == pytest.approx(math.log(1.01))
    assert result["geometric_mean_ratio"] == pytest.approx(
        math.exp(sum(math.log(value) for value in [0.95, 0.97, 1.0, 1.01, 0.99]) / 5)
    )


def test_evaluate_gates_uses_five_seed_cis_and_log_ratio_upper_bounds(
    production_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        k4b,
        "_screen_completeness",
        lambda *_args: {
            "exact_identifier_completeness": True,
            "complete_fraction": 1.0,
            "expected_trajectory_ids": 1,
        },
    )
    monkeypatch.setattr(
        k4b,
        "_false_trigger_diagnostics",
        lambda *_args: {
            "client_round_rate": 0.0,
            "trajectory_rate": 0.0,
            "tier_rate_gap": 0.0,
            "tier_gate_mean_gap": 0.0,
        },
    )
    monkeypatch.setattr(
        k4b,
        "_representative_client_rows",
        lambda *_args: [
            {
                "gate_drop_vs_k2_aware": 0.0,
                "honest_outlier": True,
                "latent_byzantine": False,
            }
        ],
    )
    round_rows = [
        {
            "candidate": candidate,
            "all_finite": True,
            "contribution_cap_respected": True,
            "normalization_by_gate_sum": False,
            "predictor_is_strictly_past": True,
            "predictor_feedback_from_k4b_output": False,
            "deployable": False,
            "privacy_claimed": candidate != k4b.ORACLE,
            "uses_oracle_target": candidate == k4b.ORACLE,
            "round": 13,
            "history_gate_all_one": True,
            "difference_to_k2": 0.0,
            "k4_comparator_reproduction_error": 0.0,
        }
        for candidate in k4b.CANDIDATES
    ]
    trajectory_rows = []
    for seed in range(5):
        cells = [
            (
                "bitflip_x10",
                "persistent",
                {
                    k4b.K4: 10.0,
                    k4b.ORACLE: 8.0,
                    k4b.PRIMARY: 9.0,
                    k4b.DELTA: 9.5,
                    k4b.K2: 11.0,
                },
            ),
            (
                "model_replacement",
                "persistent",
                {
                    k4b.K4: 10.0,
                    k4b.ORACLE: 8.0,
                    k4b.PRIMARY: 9.0,
                    k4b.DELTA: 9.5,
                    k4b.K2: 11.0,
                },
            ),
            (
                "ipm",
                "persistent",
                {k4b.K4: 10.0, k4b.PRIMARY: 10.0, k4b.DELTA: 10.0, k4b.K2: 11.0},
            ),
            (
                "alie",
                "persistent",
                {k4b.K4: 10.0, k4b.PRIMARY: 10.0, k4b.DELTA: 10.0, k4b.K2: 11.0},
            ),
            ("ipm", "intermittent_2_on_1_off", {k4b.K4: 10.0, k4b.PRIMARY: 10.0}),
            ("none", "no_compromise", {k4b.K2: 10.0, k4b.PRIMARY: 10.0}),
        ]
        for threat, schedule, values in cells:
            for candidate, attack_auc in values.items():
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "candidate": candidate,
                        "threat": threat,
                        "schedule": schedule,
                        "attack_auc": attack_auc,
                        "post_enrollment_auc": attack_auc,
                        "attack_byzantine_direct_current_mass_share": (
                            0.20 if candidate in {k4b.PRIMARY, k4b.DELTA} else 0.25
                        ),
                        "detection_rate_within_deadline": 1.0,
                        "recovery_rate_within_deadline": 1.0,
                        "attack_predictor_error_vs_current_honest_clipped_direction": 0.1,
                    }
                )
    decision = k4b._evaluate_gates(
        round_rows,
        [],
        trajectory_rows,
        [{"violation": False, "ratio_observed_to_bound": 0.5}],
        production_config,
        device_name="mps",
    )
    observed = decision["observed"]
    assert observed["oracle_bf_mr_persistent_seed_difference_count"] == 5
    assert observed["primary_bf_mr_persistent_seed_difference_count"] == 5
    assert observed["primary_persistent_separated_seed_difference_count"] == 5
    assert observed["primary_persistent_ipm_attack_auc_log_ratio"]["n_seed_pairs"] == 5
    assert decision["checks"]["ipm_noninferiority_vs_k4"] is True
    assert decision["checks"]["clean_noninferiority_vs_k2"] is True
    four_seed_rows = [row for row in trajectory_rows if int(row["seed"]) != 4]
    four_seed_decision = k4b._evaluate_gates(
        round_rows,
        [],
        four_seed_rows,
        [{"violation": False, "ratio_observed_to_bound": 0.5}],
        production_config,
        device_name="mps",
    )
    assert four_seed_decision["checks"]["oracle_effectiveness_seed_count"] is False
    assert (
        four_seed_decision["checks"]["primary_vs_k4_effectiveness_seed_count"] is False
    )
    assert four_seed_decision["checks"]["ipm_noninferiority_vs_k4"] is False

    original_paired_seed_differences = k4b._paired_seed_differences

    def with_one_nonfinite_seed_difference(*args, **kwargs):
        differences = original_paired_seed_differences(*args, **kwargs)
        assert len(differences) == 5
        differences[-1] = float("nan")
        return differences

    monkeypatch.setattr(
        k4b, "_paired_seed_differences", with_one_nonfinite_seed_difference
    )
    nonfinite_seed_decision = k4b._evaluate_gates(
        round_rows,
        [],
        trajectory_rows,
        [{"violation": False, "ratio_observed_to_bound": 0.5}],
        production_config,
        device_name="mps",
    )
    nonfinite_observed = nonfinite_seed_decision["observed"]
    assert nonfinite_observed["oracle_bf_mr_persistent_seed_difference_raw_count"] == 5
    assert nonfinite_observed["oracle_bf_mr_persistent_seed_difference_count"] == 4
    assert nonfinite_observed["primary_bf_mr_persistent_seed_difference_count"] == 4
    assert nonfinite_observed["primary_persistent_separated_seed_difference_count"] == 4
    assert nonfinite_seed_decision["checks"]["oracle_effectiveness_seed_count"] is False
    assert (
        nonfinite_seed_decision["checks"]["primary_vs_k4_effectiveness_seed_count"]
        is False
    )
    assert (
        nonfinite_seed_decision["checks"]["primary_vs_k2_effectiveness_seed_count"]
        is False
    )


def test_lock_verification_precedes_output_creation(tmp_path: Path) -> None:
    registry = json.loads(LOCK.read_text(encoding="utf-8"))
    first = next(iter(registry["locked_files"]))
    registry["locked_files"][first] = "0" * 64
    bad_lock = tmp_path / "bad-lock.json"
    bad_lock.write_text(json.dumps(registry), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        k4b.run(CONFIG, output, tmp_path / "report.md", bad_lock)
    assert not output.exists()


def test_successful_simulated_run_still_keeps_holdout_closed(
    production_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, list[int]] = {}

    def fake_screen(config: dict, *_args):
        seen["seeds"] = list(config["randomness"]["development_seeds"])
        return [], [], []

    monkeypatch.setattr(
        k4b.oracle,
        "_configure_runtime",
        lambda _name: (torch.device("mps"), torch.float32),
    )
    monkeypatch.setattr(k4b, "_screen_development", fake_screen)
    monkeypatch.setattr(k4b, "_replace_one_audit", lambda *_args: [])
    monkeypatch.setattr(
        k4b,
        "_evaluate_gates",
        lambda *_args, **_kwargs: {
            "decision": "promote_to_separate_holdout_runner",
            "all_gates_pass": True,
            "oracle_headroom_mechanism_viable": True,
            "checks": {},
            "observed": {},
            "holdout_opened": False,
        },
    )
    monkeypatch.setattr(k4b, "_summaries", lambda _rows: [])
    monkeypatch.setattr(k4b, "_write_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(k4b.base, "_write_csv", lambda *_args, **_kwargs: None)
    output = tmp_path / "result"
    decision = k4b.run(CONFIG, output, tmp_path / "report.md", LOCK)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert decision["holdout_opened"] is False
    assert manifest["holdout_opened"] is False
    assert seen["seeds"] == production_config["randomness"]["development_seeds"]
    assert not (
        set(seen["seeds"]) & set(production_config["randomness"]["holdout_seeds"])
    )
