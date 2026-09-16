from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0e as g0e
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0e.yaml"


@pytest.fixture()
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _small(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["cohort"].update(
        {
            "num_clients": 8,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    result["cohort"]["honest_outliers"]["count"] = 2
    result["privacy_noise"]["block_std_multipliers"] = [0.8, 1.2]
    result["privacy_noise"]["regimes"] = [
        {
            "name": "homogeneous",
            "client_std_multipliers": [1.0],
            "permutations": ["identity"],
        }
    ]
    result["randomness"]["calibration_draws_per_seed"] = 2
    result["randomness"]["calibration_contexts"] = [
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
    ]
    result["references"]["trimmed_mean"]["trim_count"] = 2
    return result


def test_config_and_public_budget_derivation(config: dict) -> None:
    g0e._validate_config(config)
    derived = g0e._derive_parameters(config)
    assert derived["regularization"] == pytest.approx(1.0)
    # Replacement contamination is tied to the FCC bound, hence G=rho=0.13.
    assert derived["influence_cap_total"] == pytest.approx(0.13)
    assert 2 * config["cohort"]["num_byzantine"] * derived["influence_cap_total"] / (
        derived["regularization"] * config["cohort"]["num_clients"]
    ) == pytest.approx(0.052)
    assert derived["num_steps"] == 12
    assert derived["num_steps"] % 2 == 0
    assert derived["solver_gradient_residual_bound"] <= 1.0e-6
    assert derived["finite_correction_norm_bound"] == pytest.approx(0.026)
    assert derived["beta_finite_solver"] > derived["beta_asymptotic_control"]
    assert derived["finite_solver_replace_one_bound"] == pytest.approx(0.01248)
    assert derived["total_reference_replacement_contamination_bound"] == pytest.approx(
        0.0624
    )


def test_validation_rejects_cpu_seed_reuse_and_grid(config: dict) -> None:
    bad = copy.deepcopy(config)
    bad["execution"]["allow_cpu_fallback"] = True
    with pytest.raises(ValueError, match="forbids CPU fallback"):
        g0e._validate_config(bad)

    bad = copy.deepcopy(config)
    bad["randomness"]["holdout_seeds"][0] = 5003
    with pytest.raises(ValueError, match="reuses a seed"):
        g0e._validate_config(bad)

    bad = copy.deepcopy(config)
    bad["selection"]["candidate_grid"] = "allowed"
    with pytest.raises(ValueError, match="cannot tune a candidate grid"):
        g0e._validate_config(bad)


def test_split_conformal_rank_is_finite_sample_correct() -> None:
    values = torch.arange(1.0, 11.0)
    threshold, rank = g0e._conformal_quantile(values, 0.20)
    # ceil((10+1)*0.8) = 9.
    assert rank == 9
    assert threshold == pytest.approx(9.0)


def test_calibration_uses_one_public_regular_probe_per_stratum(
    config: dict,
) -> None:
    small = _small(config)
    oracle._configure_runtime("cpu")
    artifact, rows = g0e._calibrate(small, g0e._derive_parameters(small))
    keys = [
        (
            row["score_fold"],
            row["seed"],
            row["draw"],
            row["calibration_context"],
            row["noise_regime"],
            row["noise_permutation"],
            row["noise_tier"],
            row["block"],
        )
        for row in rows
    ]
    assert len(keys) == len(set(keys))
    assert len(rows) == 2 * 2 * 2 * 2 * 1 * 1 * 1 * 2
    assert artifact["common_threshold_is_maximum_over_folds_and_preregistered_strata"]
    assert artifact["deployment_reference_variance_is_maximum_over_folds_and_contexts"]
    assert artifact[
        "calibration_and_evaluation_share_preregistered_population_centre_law_and_public_anchor_rule"
    ]
    for block, common in enumerate(artifact["standardized_thresholds"]):
        relevant = [
            row["threshold"]
            for row in artifact["stratum_quantiles"]
            if row["block"] == block
        ]
        assert common == pytest.approx(max(relevant))
    assert set(artifact["calibration_tail_rate_by_context"]) == {
        "regular_only",
        "bounded_outliers_aligned",
    }
    assert artifact["calibration_tail_rate_by_stratum"]


def test_calibration_draws_have_independent_centres_with_evaluation_law(
    config: dict,
) -> None:
    small = _small(config)
    oracle._configure_runtime("cpu")
    regime = small["privacy_noise"]["regimes"][0]
    context = small["randomness"]["calibration_contexts"][0]
    first = g0e._null_sample(
        small,
        seed=2026092301,
        draw=0,
        regime=regime,
        permutation="identity",
        context=context,
    )
    second = g0e._null_sample(
        small,
        seed=2026092301,
        draw=1,
        regime=regime,
        permutation="identity",
        context=context,
    )
    first_anchor = first[3]
    second_anchor = second[3]
    expected_anchor_norm = (
        small["cohort"]["honest_mean_norm"] ** 2
        + small["references"]["public_anchor_error_norm"] ** 2
    ) ** 0.5
    assert torch.linalg.vector_norm(first_anchor).item() == pytest.approx(
        expected_anchor_norm
    )
    assert torch.linalg.vector_norm(second_anchor).item() == pytest.approx(
        expected_anchor_norm
    )
    assert not torch.allclose(first_anchor, second_anchor)


def test_statistical_and_deployed_radii_are_distinct(config: dict) -> None:
    small = _small(config)
    oracle._configure_runtime("cpu")
    derived = g0e._derive_parameters(small)
    calibration, _ = g0e._calibrate(small, derived)
    regime = small["privacy_noise"]["regimes"][0]
    variances, _ = oracle._noise_variances(small, regime, "identity")
    statistical, deployed = g0e._radii_for_cell(
        small,
        derived,
        calibration,
        regime="homogeneous",
        permutation="identity",
        noise_variances=variances,
    )
    assert torch.all(deployed > statistical)
    assert torch.allclose(
        deployed - statistical,
        torch.full_like(statistical, derived["pilot_replace_one_bound"]),
    )


def test_diagnostic_gates_use_worst_cell_block_not_global_average(
    config: dict,
) -> None:
    small = _small(config)

    def candidate_row(threat: str, pairing: str) -> dict:
        return {
            "phase": "development",
            "pairing_id": pairing,
            "candidate": "g0e",
            "noise_regime": "heteroscedastic",
            "noise_permutation": "identity",
            "outlier_geometry": "aligned",
            "threat": threat,
            "severity": 1.0,
            "seed": 6007,
            "reference_error": 1.0,
            "reference_error_ratio_to_uniform": 1.0,
            "reference_error_ratio_to_fcc": 1.0,
            # Scalars look benign; block 1 deliberately violates every gate.
            "regular_honest_statistical_tail_rate": 0.10,
            "regular_honest_influence_cap_activation_rate": 0.10,
            "honest_outlier_not_cap_limited_rate": 0.40,
            "byzantine_influence_share": 0.20,
            "regular_honest_statistical_tail_rate_by_block": [0.10, 0.30],
            "regular_honest_influence_cap_activation_rate_by_block": [0.10, 0.30],
            "honest_outlier_statistical_tail_rate_by_block": [0.10, 0.10],
            "honest_outlier_influence_cap_activation_rate_by_block": [0.10, 0.60],
            "honest_outlier_not_cap_limited_rate_by_block": [0.90, 0.40],
            "byzantine_influence_cap_activation_rate_by_block": [0.10, 0.10],
            "byzantine_influence_share_by_block": [0.20, 0.30],
            "regular_honest_diagnostics_by_noise_tier_block": [
                {
                    "noise_tier": 1.0,
                    "block": 0,
                    "regular_client_count": 4,
                    "regular_honest_statistical_tail_rate": 0.10,
                    "regular_honest_influence_cap_activation_rate": 0.10,
                },
                {
                    "noise_tier": 1.0,
                    "block": 1,
                    "regular_client_count": 4,
                    "regular_honest_statistical_tail_rate": 0.30,
                    "regular_honest_influence_cap_activation_rate": 0.30,
                },
            ],
            "solver_gradient_residual": 0.0,
            "correction_budget_violation": False,
        }

    rows = [
        candidate_row("none", "clean"),
        candidate_row("ipm", "attack"),
        candidate_row("alie", "evasive"),
        {
            **candidate_row("ipm", "attack"),
            "candidate": "fcc",
            "reference_error": 1.1,
        },
    ]
    stability = [
        {
            "candidate": "g0e",
            "theoretical_bound": 0.012,
            "observed_delta": 0.006,
            "violation": False,
        }
    ]
    summary = g0e._summarize(rows, stability, small, "development")
    assert summary[
        "regular_honest_statistical_tail_rate_upper_excess_worst_group_block"
    ] == pytest.approx(0.20)
    assert summary[
        "regular_honest_influence_cap_activation_rate_worst_group_block"
    ] == pytest.approx(0.30)
    assert summary[
        "honest_outlier_all_blocks_not_cap_limited_rate_worst_group"
    ] == pytest.approx(0.40)
    assert summary["gate_regular_tail_calibration"] is False
    assert summary["gate_regular_not_cap_limited"] is False
    assert summary["gate_honest_outlier_retention"] is False


def test_crossfit_diagnostic_does_not_change_online_reference(config: dict) -> None:
    small = _small(config)
    oracle._configure_runtime("cpu")
    n = small["cohort"]["num_clients"]
    d = small["cohort"]["dimension"]
    vectors = torch.randn(n, d, dtype=torch.float64) * 0.02
    anchor = torch.zeros(d, dtype=torch.float64)
    pilot = g0e._comparator_reference("fcc", vectors, small, anchor=anchor)
    crossfit = g0e._fcc_leave_one_out(
        vectors, anchor=anchor, radius=small["references"]["fcc"]["radius"]
    )
    derived = g0e._derive_parameters(small)
    statistical = torch.full((n, 2), 0.04, dtype=torch.float64)
    deployed = statistical + derived["pilot_replace_one_bound"]
    kwargs = dict(
        pilot=pilot,
        statistical_radii=statistical,
        deployed_radii=deployed,
        pilot_replace_one_bound=derived["pilot_replace_one_bound"],
        block_sizes=[4, 4],
        influence_cap=derived["influence_cap_per_block"],
        regularization=derived["regularization"],
        correction_budget=derived["correction_budget"],
        num_steps=derived["num_steps"],
    )
    left = g0e.gaussian_aware_crossfit_bounded_correction(
        vectors, crossfit_references=crossfit, **kwargs
    )
    right = g0e.gaussian_aware_crossfit_bounded_correction(
        vectors, crossfit_references=crossfit + 10.0, **kwargs
    )
    assert torch.equal(left, right)


def test_expected_pairing_count_is_derived_from_config(config: dict) -> None:
    # 5 seeds * 1 draw * 3 noise cells * 2 geometries * (1+4*3 levels)
    assert g0e._expected_pairings(config, "development") == 390
    # 7 seeds * 2 draws * 3 cells * 2 geometries * (1+4*4 levels)
    assert g0e._expected_pairings(config, "holdout") == 1428
    assert g0e._expected_stability_observations(config, "development") == 60
    assert g0e._expected_stability_observations(config, "holdout") == 84


def test_resume_checkpoint_rejects_cpu_partial_and_wrong_seed(
    config: dict, tmp_path: Path
) -> None:
    small = _small(config)
    candidates = (*g0e.COMPARATORS, g0e.CANDIDATE)
    pairings = g0e._expected_pairings_per_seed(small, "development")
    rows = [
        {
            "phase": "development",
            "seed": 6007,
            "resolved_device": "mps",
            "tensor_dtype": "float32",
            "pairing_id": f"pair-{pairing}",
            "candidate": candidate,
        }
        for pairing in range(pairings)
        for candidate in candidates
    ]
    source = tmp_path / "development_detail_seed_6007.json"
    g0e._validate_checkpoint_rows(
        rows,
        config=small,
        phase="development",
        seed=6007,
        kind="detail",
        source=source,
    )
    indexed_mps_rows = copy.deepcopy(rows)
    for row in indexed_mps_rows:
        row["resolved_device"] = "mps:0"
    g0e._validate_checkpoint_rows(
        indexed_mps_rows,
        config=small,
        phase="development",
        seed=6007,
        kind="detail",
        source=source,
    )
    with pytest.raises(RuntimeError, match="Incomplete detail checkpoint"):
        g0e._validate_checkpoint_rows(
            rows[:-1],
            config=small,
            phase="development",
            seed=6007,
            kind="detail",
            source=source,
        )
    cpu_rows = copy.deepcopy(rows)
    for row in cpu_rows:
        row["resolved_device"] = "cpu"
    with pytest.raises(RuntimeError, match="Non-MPS checkpoint"):
        g0e._validate_checkpoint_rows(
            cpu_rows,
            config=small,
            phase="development",
            seed=6007,
            kind="detail",
            source=source,
        )
    wrong_seed = copy.deepcopy(rows)
    for row in wrong_seed:
        row["seed"] = 6101
    with pytest.raises(RuntimeError, match="Wrong phase or seed"):
        g0e._validate_checkpoint_rows(
            wrong_seed,
            config=small,
            phase="development",
            seed=6007,
            kind="detail",
            source=source,
        )


def test_stability_checkpoint_has_its_own_exact_cardinality(
    config: dict, tmp_path: Path
) -> None:
    small = _small(config)
    candidates = (*g0e.COMPARATORS, g0e.CANDIDATE)
    trials = int(small["randomness"]["replace_one_trials_per_seed_cell"])
    rows = [
        {
            "phase": "development",
            "seed": 6007,
            "resolved_device": "mps",
            "tensor_dtype": "float32",
            "noise_regime": "homogeneous",
            "noise_permutation": "identity",
            "trial": trial,
            "candidate": candidate,
        }
        for trial in range(trials)
        for candidate in candidates
    ]
    assert len(rows) == g0e._expected_stability_rows_per_seed(small)
    source = tmp_path / "development_stability_seed_6007.json"
    g0e._validate_checkpoint_rows(
        rows,
        config=small,
        phase="development",
        seed=6007,
        kind="stability",
        source=source,
    )
    with pytest.raises(RuntimeError, match="Incomplete stability checkpoint"):
        g0e._validate_checkpoint_rows(
            rows[:-1],
            config=small,
            phase="development",
            seed=6007,
            kind="stability",
            source=source,
        )


def test_run_never_opens_holdout_when_development_fails(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "g0e.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        g0e.oracle,
        "_configure_runtime",
        lambda _: (torch.device("mps"), torch.float32),
    )
    monkeypatch.setattr(
        g0e,
        "_calibrate",
        lambda *_: (
            {
                "standardized_thresholds": [1.0] * 4,
                "calibration_tail_rate_by_block": [0.1] * 4,
                "calibration_tail_rate_by_context": {"regular_only": 0.1},
            },
            [{"row": 1}],
        ),
    )

    def fake_phase(**kwargs: object) -> list[dict]:
        calls.append(str(kwargs["phase"]))
        return [{"phase": kwargs["phase"]}]

    monkeypatch.setattr(g0e, "_cached_phase_rows", fake_phase)
    monkeypatch.setattr(
        g0e,
        "_cached_stability_rows",
        lambda **kwargs: [{"phase": kwargs["phase"]}],
    )
    monkeypatch.setattr(
        g0e,
        "_summarize",
        lambda *args: {
            "passes_all_gates": False,
            "gate_fail_count": 2,
        },
    )
    monkeypatch.setattr(g0e, "_comparator_summaries", lambda *_: [{"x": 1}])
    monkeypatch.setattr(g0e, "_diagnostic_group_summaries", lambda *_, **__: [{"x": 1}])
    monkeypatch.setattr(
        g0e, "_regular_tier_block_group_summaries", lambda *_, **__: [{"x": 1}]
    )
    monkeypatch.setattr(g0e, "_write_report", lambda *_, **__: None)
    decision = g0e.run(
        config_path,
        tmp_path / "results",
        tmp_path / "report.md",
    )
    assert calls == ["development"]
    assert decision["holdout_status"] == "blocked_by_development_gate"
    assert not any((tmp_path / "results").glob("holdout*"))
    saved = json.loads((tmp_path / "results/decision.json").read_text())
    assert saved["promote"] is False


def test_failed_development_rejects_nested_holdout_checkpoint(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "g0e.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "results"
    nested = output / "_checkpoints/holdout_detail_seed_7001.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        g0e.oracle,
        "_configure_runtime",
        lambda _: (torch.device("mps"), torch.float32),
    )
    monkeypatch.setattr(
        g0e,
        "_calibrate",
        lambda *_: (
            {
                "standardized_thresholds": [1.0] * 4,
                "calibration_tail_rate_by_block": [0.1] * 4,
                "calibration_tail_rate_by_context": {"regular_only": 0.1},
            },
            [{"row": 1}],
        ),
    )
    monkeypatch.setattr(
        g0e, "_cached_phase_rows", lambda **kwargs: [{"phase": kwargs["phase"]}]
    )
    monkeypatch.setattr(
        g0e,
        "_cached_stability_rows",
        lambda **kwargs: [{"phase": kwargs["phase"]}],
    )
    monkeypatch.setattr(
        g0e,
        "_summarize",
        lambda *args: {"passes_all_gates": False, "gate_fail_count": 1},
    )
    monkeypatch.setattr(g0e, "_comparator_summaries", lambda *_: [{"x": 1}])
    monkeypatch.setattr(g0e, "_diagnostic_group_summaries", lambda *_, **__: [{"x": 1}])
    monkeypatch.setattr(
        g0e, "_regular_tier_block_group_summaries", lambda *_, **__: [{"x": 1}]
    )
    monkeypatch.setattr(g0e, "_write_report", lambda *_, **__: None)
    with pytest.raises(RuntimeError, match="holdout artifacts exist"):
        g0e.run(config_path, output, tmp_path / "report.md", resume=True)
    artifacts = g0e._holdout_artifacts(output)
    assert nested in artifacts


def test_algorithm_rejects_deployed_radius_below_statistical(config: dict) -> None:
    small = _small(config)
    n = small["cohort"]["num_clients"]
    d = small["cohort"]["dimension"]
    vectors = torch.zeros(n, d)
    pilot = torch.zeros(d)
    crossfit = torch.zeros(n, d)
    with pytest.raises(ValueError, match="at least statistical_radii"):
        g0e.gaussian_aware_crossfit_bounded_correction(
            vectors,
            pilot=pilot,
            crossfit_references=crossfit,
            statistical_radii=torch.full((n, 2), 0.2),
            deployed_radii=torch.full((n, 2), 0.1),
            pilot_replace_one_bound=0.01,
            block_sizes=[4, 4],
            influence_cap=[0.05, 0.05],
            regularization=1.0,
            correction_budget=0.02,
            num_steps=4,
        )
