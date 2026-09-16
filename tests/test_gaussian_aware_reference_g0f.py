"""Unit, ablation, and certificate tests for the G0f reference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from algorithms.gaussian_aware_reference import (
    allocate_gaussian_aware_block_budgets,
    gaussian_aware_budget_allocated_correction,
)
from robustness.aggregators import (
    centered_clipping,
    centered_clipping_leave_one_out,
)
from scripts import run_gaussian_aware_reference_g0f as g0f
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0f.yaml"


@pytest.fixture()
def campaign_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _inputs(dtype: torch.dtype = torch.float64):
    vectors = torch.tensor(
        [
            [0.20, -0.10, 0.05, 0.03],
            [-0.10, 0.15, 0.02, -0.04],
            [0.05, 0.05, -0.01, 0.06],
            [0.10, -0.05, 0.08, -0.02],
        ],
        dtype=dtype,
    )
    anchor = torch.zeros(4, dtype=dtype)
    pilot = centered_clipping(vectors, anchor=anchor, tau=0.3)
    crossfit = centered_clipping_leave_one_out(vectors, anchor=anchor, tau=0.3)
    deployed = torch.tensor(
        [[0.10, 0.20], [0.20, 0.40], [0.15, 0.30], [0.25, 0.50]],
        dtype=dtype,
    )
    return vectors, pilot, crossfit, deployed


def test_global_covariance_allocation_has_constant_ratio_and_l2_certificate():
    radii = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]], dtype=torch.float64)
    budgets, diagnostics = allocate_gaussian_aware_block_budgets(
        radii,
        total_influence_budget=0.7,
        policy="global_covariance",
    )
    expected_a = torch.linalg.vector_norm(radii, dim=1).max()
    assert torch.allclose(budgets, 0.7 * radii / expected_a)
    assert torch.allclose(budgets / radii, torch.full_like(radii, 0.7 / expected_a))
    assert torch.linalg.vector_norm(budgets, dim=1).max() <= 0.7 + 1e-12
    assert diagnostics["public_radius_row_norm_max_A"] == pytest.approx(
        expected_a.item()
    )
    assert diagnostics["budget_to_radius_ratio_min"] == pytest.approx(
        diagnostics["budget_to_radius_ratio_max"]
    )
    assert diagnostics["preserves_between_client_covariance_scale"] is True
    assert diagnostics["complete_client_budget_respected"] is True


def test_scale_blind_ablation_removes_between_client_scale():
    radii = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float64)
    budgets, diagnostics = allocate_gaussian_aware_block_budgets(
        radii,
        total_influence_budget=0.5,
        policy="per_client_scale_blind",
    )
    assert torch.allclose(budgets[0], budgets[1])
    assert torch.allclose(
        torch.linalg.vector_norm(budgets, dim=1),
        torch.full((2,), 0.5, dtype=radii.dtype),
    )
    assert diagnostics["preserves_between_client_covariance_scale"] is False
    assert diagnostics["preserves_within_client_block_shape"] is True


def test_equal_cap_ablation_ignores_all_covariance_geometry():
    radii = torch.tensor([[1.0, 7.0], [4.0, 2.0]], dtype=torch.float32)
    budgets, diagnostics = allocate_gaussian_aware_block_budgets(
        radii,
        total_influence_budget=0.8,
        policy="equal_cap",
    )
    assert torch.allclose(budgets, torch.full_like(radii, 0.8 / 2**0.5))
    assert torch.allclose(
        torch.linalg.vector_norm(budgets, dim=1), torch.full((2,), 0.8)
    )
    assert diagnostics["preserves_between_client_covariance_scale"] is False
    assert diagnostics["preserves_within_client_block_shape"] is False


@pytest.mark.parametrize(
    "policy",
    ["global_covariance", "per_client_scale_blind", "equal_cap"],
)
def test_all_g0f_policies_are_deterministic_and_budgeted(policy: str):
    vectors, pilot, crossfit, deployed = _inputs()
    kwargs = dict(
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=deployed - 0.01,
        deployed_radii=deployed,
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        allocation_policy=policy,
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
        return_diagnostics=True,
    )
    first, first_diagnostics = gaussian_aware_budget_allocated_correction(
        vectors, **kwargs
    )
    second, second_diagnostics = gaussian_aware_budget_allocated_correction(
        vectors, **kwargs
    )
    assert torch.equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["allocation_policy"] == policy
    assert first_diagnostics["complete_client_budget_respected"] is True
    assert first_diagnostics["correction_budget_respected"] is True
    assert torch.linalg.vector_norm(first - pilot) <= 0.04 + 1e-12
    json.dumps(first_diagnostics, allow_nan=False)


def test_g0f_global_policy_uses_deployed_pre_cap_radii_by_default():
    vectors, pilot, crossfit, deployed = _inputs()
    _, diagnostics = gaussian_aware_budget_allocated_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=deployed - 0.01,
        deployed_radii=deployed,
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        allocation_policy="global_covariance",
        allocation_radius_provenance="effective_null_radius",
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
        return_diagnostics=True,
    )
    allocated = torch.tensor(
        diagnostics["allocated_block_budgets"], dtype=deployed.dtype
    )
    ratio = allocated / deployed
    assert ratio.max().item() == pytest.approx(ratio.min().item())
    assert diagnostics["allocation_radius_min"] == pytest.approx(deployed.min().item())
    assert diagnostics["allocation_radius_max"] == pytest.approx(deployed.max().item())
    assert diagnostics["allocation_source"] == "effective_null_radius"
    assert diagnostics["allocation_numerically_equals_deployed_radii"] is True
    assert diagnostics["global_effective_fraction_theorem_applicable"] is True
    assert diagnostics["effective_radius_to_deployed_ratio_spread"] == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert diagnostics["effective_solver_radius_definition"].startswith("min(")
    assert (
        diagnostics["crossfit_diagnostics_require_separate_release_accounting"] is False
    )
    assert diagnostics["crossfit_privacy_status"].startswith("server_postprocessing")


def test_crossfit_references_remain_diagnostic_only():
    vectors, pilot, crossfit, deployed = _inputs()
    common = dict(
        pilot=pilot,
        statistical_radii=deployed - 0.01,
        deployed_radii=deployed,
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
        return_diagnostics=True,
    )
    first, first_diagnostics = gaussian_aware_budget_allocated_correction(
        vectors, crossfit_references=crossfit, **common
    )
    second, second_diagnostics = gaussian_aware_budget_allocated_correction(
        vectors, crossfit_references=crossfit + 50.0, **common
    )
    assert torch.equal(first, second)
    assert first_diagnostics["crossfit_references_affect_returned_reference"] is False
    assert (
        first_diagnostics["statistical_tail_fraction_crossfit"]
        != second_diagnostics["statistical_tail_fraction_crossfit"]
    )


def test_finite_k_replace_one_certificate_includes_pilot_bound():
    generator = torch.Generator().manual_seed(2026)
    n = 10
    vectors = 0.12 * torch.randn(n, 4, generator=generator, dtype=torch.float64)
    neighbour = vectors.clone()
    neighbour[3] = torch.tensor([40.0, -30.0, 20.0, -10.0])
    anchor = torch.zeros(4, dtype=torch.float64)
    pilot_radius = 0.35
    pilot_bound = 2.0 * pilot_radius / n
    deployed = torch.tensor(
        [[0.15 + 0.01 * index, 0.24 + 0.02 * index] for index in range(n)],
        dtype=torch.float64,
    )

    def evaluate(cohort: torch.Tensor, with_diagnostics: bool):
        pilot = centered_clipping(cohort, anchor=anchor, tau=pilot_radius)
        crossfit = centered_clipping_leave_one_out(
            cohort, anchor=anchor, tau=pilot_radius
        )
        return gaussian_aware_budget_allocated_correction(
            cohort,
            pilot=pilot,
            crossfit_references=crossfit,
            statistical_radii=deployed - 0.02,
            deployed_radii=deployed,
            pilot_replace_one_bound=pilot_bound,
            total_influence_budget=0.18,
            block_sizes=[2, 2],
            regularization=0.7,
            correction_budget=0.05,
            num_steps=10,
            return_diagnostics=with_diagnostics,
        )

    left, diagnostics = evaluate(vectors, True)
    right = evaluate(neighbour, False)
    observed = float(torch.linalg.vector_norm(left - right).item())
    expected = pilot_bound + (
        2.0
        * diagnostics["beta"]
        * 0.18
        * (1.0 - diagnostics["solver_contraction"] ** 10)
        / (0.7 * n)
    )
    assert diagnostics["finite_solver_replace_one_bound"] == pytest.approx(expected)
    assert diagnostics["finite_solver_error_bound_after_blend"] == pytest.approx(
        diagnostics["beta"] * diagnostics["finite_solver_error_bound"]
    )
    assert diagnostics[
        "finite_b_replacement_bound_by_replacement_path"
    ] == pytest.approx(diagnostics["num_replacements_for_diagnostics"] * expected)
    assert diagnostics["finite_solver_replace_one_bound"] >= pilot_bound
    assert observed <= diagnostics["finite_solver_replace_one_bound"] + 1e-12


@pytest.mark.parametrize("num_replacements", [2, 3])
def test_b_replacement_path_certificate_matches_actual_b_neighbour(
    num_replacements: int,
):
    generator = torch.Generator().manual_seed(9271 + num_replacements)
    n = 12
    vectors = 0.08 * torch.randn(n, 6, generator=generator, dtype=torch.float64)
    neighbour = vectors.clone()
    neighbour[:num_replacements] = 25.0 * torch.randn(
        num_replacements,
        6,
        generator=torch.Generator().manual_seed(103 + num_replacements),
        dtype=torch.float64,
    )
    anchor = torch.zeros(6, dtype=torch.float64)
    pilot_radius = 0.24
    pilot_bound = 2.0 * pilot_radius / n
    deployed = torch.tensor(
        [[0.12 + 0.003 * i, 0.18 + 0.004 * i, 0.24 + 0.005 * i] for i in range(n)],
        dtype=torch.float64,
    )

    def evaluate(cohort: torch.Tensor, diagnostics: bool):
        pilot = centered_clipping(cohort, anchor=anchor, tau=pilot_radius)
        crossfit = centered_clipping_leave_one_out(
            cohort, anchor=anchor, tau=pilot_radius
        )
        return gaussian_aware_budget_allocated_correction(
            cohort,
            pilot=pilot,
            crossfit_references=crossfit,
            statistical_radii=deployed - 0.01,
            deployed_radii=deployed,
            pilot_replace_one_bound=pilot_bound,
            total_influence_budget=0.15,
            block_sizes=[2, 2, 2],
            allocation_policy="global_covariance",
            regularization=0.6,
            correction_budget=0.04,
            num_steps=9,
            num_replacements_for_diagnostics=num_replacements,
            return_diagnostics=diagnostics,
        )

    left, diagnostics = evaluate(vectors, True)
    right = evaluate(neighbour, False)
    observed = float(torch.linalg.vector_norm(left - right).item())
    path_bound = diagnostics["finite_b_replacement_bound_by_replacement_path"]
    assert path_bound == pytest.approx(
        num_replacements * diagnostics["finite_solver_replace_one_bound"]
    )
    assert diagnostics["b_replacement_path_bound_formula"].startswith("b*(")
    assert observed <= path_bound + 1.0e-12


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"covariance_provenance": "client_declared"}, "public_authenticated"),
        ({"allocation_policy": "unknown"}, "Unknown allocation policy"),
        ({"total_influence_budget": 0.0}, "finite and positive"),
        ({"allocation_radii": -0.1}, "strictly positive"),
        ({"regularization": 0.0}, "strictly positive"),
        ({"regularization": 1.0e-20}, "not numerically resolvable"),
        ({"correction_budget": -0.1}, "non-negative"),
        ({"num_steps": 0}, "integer >= 1"),
    ],
)
def test_invalid_g0f_configuration_is_rejected(override, message):
    vectors, pilot, crossfit, deployed = _inputs()
    kwargs = dict(
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=deployed - 0.01,
        deployed_radii=deployed,
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        gaussian_aware_budget_allocated_correction(vectors, **kwargs)


def test_half_precision_input_keeps_float32_certificate_output():
    vectors, pilot, crossfit, deployed = _inputs(dtype=torch.float16)
    result = gaussian_aware_budget_allocated_correction(
        vectors,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=(deployed - 0.01).to(torch.float32),
        deployed_radii=deployed.to(torch.float32),
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
    )
    assert result.dtype == torch.float32
    assert bool(torch.isfinite(result).all())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_g0f_runs_in_float32_on_mps():
    vectors, pilot, crossfit, deployed = _inputs(dtype=torch.float32)
    result = gaussian_aware_budget_allocated_correction(
        vectors.to("mps"),
        pilot=pilot.to("mps"),
        crossfit_references=crossfit.to("mps"),
        statistical_radii=(deployed - 0.01).to("mps"),
        deployed_radii=deployed.to("mps"),
        pilot_replace_one_bound=0.15,
        total_influence_budget=0.2,
        block_sizes=[2, 2],
        regularization=0.5,
        correction_budget=0.04,
        num_steps=8,
    )
    assert result.device.type == "mps"
    assert result.dtype == torch.float32
    assert bool(torch.isfinite(result).all().cpu())


def test_decisive_config_and_exact_pairing_cardinality(campaign_config: dict):
    g0f._validate_config(campaign_config)
    assert len(g0f.CANDIDATES) == 9
    assert "coordinate_median" in g0f.CANDIDATES
    assert g0f._noise_cell_count(campaign_config) == 5
    assert g0f._expected_pairings_per_seed(campaign_config, "development") == 130
    assert g0f._expected_pairings(campaign_config, "development") == 650
    assert (
        g0f._expected_pairings(campaign_config, "development") * len(g0f.CANDIDATES)
        == 5850
    )


def test_public_byzantine_tier_assignments_preserve_multiset(
    campaign_config: dict,
):
    oracle._configure_runtime("cpu")
    regime = next(
        item
        for item in campaign_config["privacy_noise"]["regimes"]
        if item["name"] == "heteroscedastic"
    )
    identity_variance, identity_tiers = g0f._noise_variances(
        campaign_config, regime, "identity"
    )
    high_variance, high_tiers = g0f._noise_variances(
        campaign_config, regime, "byzantine_high"
    )
    low_variance, low_tiers = g0f._noise_variances(
        campaign_config, regime, "byzantine_low"
    )
    b = campaign_config["cohort"]["num_byzantine"]
    assert torch.equal(identity_tiers.sort().values, high_tiers.sort().values)
    assert torch.equal(identity_tiers.sort().values, low_tiers.sort().values)
    assert high_tiers[-b:].min() >= high_tiers[:-b].max()
    assert low_tiers[-b:].max() <= low_tiers[:-b].min()
    assert torch.all(high_variance > 0)
    assert torch.all(low_variance > 0)


def test_standard_noise_is_paired_across_regimes_and_assignments(
    campaign_config: dict,
):
    oracle._configure_runtime("cpu")
    clean = torch.zeros(25, 64, dtype=torch.float64)
    blocks = campaign_config["cohort"]["block_sizes"]
    homogeneous = campaign_config["privacy_noise"]["regimes"][0]
    heterogeneous = campaign_config["privacy_noise"]["regimes"][1]
    variance_hom, _ = g0f._noise_variances(campaign_config, homogeneous, "identity")
    variance_high, _ = g0f._noise_variances(
        campaign_config, heterogeneous, "byzantine_high"
    )
    first = g0f._paired_private_noise(
        clean,
        variance_hom,
        blocks,
        seed=8009,
        draw=0,
        geometry="aligned",
    )
    second = g0f._paired_private_noise(
        clean,
        variance_high,
        blocks,
        seed=8009,
        draw=0,
        geometry="aligned",
    )
    first_std = oracle._expand_block_values(variance_hom.sqrt(), blocks)
    second_std = oracle._expand_block_values(variance_high.sqrt(), blocks)
    # The runner exposes the underlying Z directly: exact equality here is
    # stronger and avoids roundoff from multiply-then-divide reconstruction.
    first_z = g0f._paired_standard_normal(
        clean.shape, seed=8009, draw=0, geometry="aligned"
    )
    second_z = g0f._paired_standard_normal(
        clean.shape, seed=8009, draw=0, geometry="aligned"
    )
    assert torch.equal(first_z, second_z)
    assert torch.allclose(first, clean + first_std * first_z)
    assert torch.allclose(second, clean + second_std * second_z)


def test_derived_finite_k_bound_has_required_total_formula(campaign_config: dict):
    derived = g0f._derive_parameters(campaign_config)
    q = float(derived["target_solver_contraction"])
    beta = float(derived["beta_finite_solver"])
    total = float(derived["influence_cap_total"])
    gamma = float(derived["regularization"])
    n = int(campaign_config["cohort"]["num_clients"])
    k = int(derived["num_steps"])
    expected = float(derived["pilot_replace_one_bound"]) + (
        2.0 * beta * total * (1.0 - q**k) / (gamma * n)
    )
    assert derived["finite_solver_replace_one_bound"] == pytest.approx(expected)
    assert derived["allocation_radii"] == "deployed_public_radii_before_cap"


def test_calibration_supports_all_five_public_noise_cells(campaign_config: dict):
    small = copy.deepcopy(campaign_config)
    small["cohort"].update(
        {
            "num_clients": 8,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    small["cohort"]["honest_outliers"]["count"] = 2
    small["privacy_noise"]["block_std_multipliers"] = [0.8, 1.2]
    small["references"]["trimmed_mean"]["trim_count"] = 2
    small["randomness"]["calibration_draws_per_seed"] = 2
    small["randomness"]["calibration_contexts"] = [
        {
            "name": "regular_only",
            "include_outliers": False,
            "geometry": "orthogonal",
        }
    ]
    oracle._configure_runtime("cpu")
    artifact, rows = g0f._calibrate(small, g0f._derive_parameters(small))
    cells = {(row["noise_regime"], row["noise_permutation"]) for row in rows}
    assert cells == {
        ("homogeneous", "identity"),
        ("heteroscedastic", "identity"),
        ("heteroscedastic", "reverse"),
        ("heteroscedastic", "byzantine_high"),
        ("heteroscedastic", "byzantine_low"),
    }
    assert artifact["consumer"].startswith("g0f_")
    assert artifact["covariance_provenance"] == "public_authenticated"


def test_failed_development_never_materializes_holdout(
    campaign_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "g0f.yaml"
    config_path.write_text(
        yaml.safe_dump(campaign_config, sort_keys=False), encoding="utf-8"
    )
    phases: list[str] = []
    monkeypatch.setattr(
        g0f.oracle,
        "_configure_runtime",
        lambda _: (torch.device("mps"), torch.float32),
    )
    monkeypatch.setattr(
        g0f,
        "_calibrate",
        lambda *_: ({"standardized_thresholds": [1.0] * 4}, [{"row": 1}]),
    )

    def fake_phase(**kwargs):
        phases.append(str(kwargs["phase"]))
        return [{"phase": kwargs["phase"]}]

    monkeypatch.setattr(g0f, "_cached_phase_rows", fake_phase)
    monkeypatch.setattr(
        g0f,
        "_cached_stability_rows",
        lambda **kwargs: [{"phase": kwargs["phase"]}],
    )
    monkeypatch.setattr(
        g0f,
        "_summarize",
        lambda *_: {
            "passes_all_gates": False,
            "gate_fail_count": 1,
            "failed_gates": ["decisive_failure"],
        },
    )
    monkeypatch.setattr(g0f, "_candidate_summary", lambda *_: [{"x": 1}])
    monkeypatch.setattr(g0f, "_write_report", lambda *_, **__: None)
    monkeypatch.setattr(
        g0f,
        "_load_execution_gated_holdout_seeds",
        lambda *_: pytest.fail("holdout registry was opened after a failed dev gate"),
    )
    decision = g0f.run(
        config_path,
        tmp_path / "results",
        tmp_path / "report.md",
    )
    assert phases == ["development"]
    assert decision["holdout_status"] == "blocked_by_development_gate"
    assert not g0f._holdout_artifacts(tmp_path / "results")
    resolved = yaml.safe_load(
        (tmp_path / "results/resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert "holdout_seeds" not in resolved["randomness"]
    manifest_path = tmp_path / "results/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["software_sha256"]) == {
        "scripts/run_gaussian_aware_reference_g0f.py",
        "algorithms/gaussian_aware_reference.py",
        "scripts/run_gaussian_aware_reference_g0e.py",
        "scripts/run_gaussian_aware_reference_oracle.py",
        "robustness/aggregators.py",
    }
    assert len(manifest["calibration_artifact_sha256"]) == 64
    manifest["software_sha256"]["robustness/aggregators.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest mismatch for software_sha256"):
        g0f.run(
            config_path,
            tmp_path / "results",
            tmp_path / "report.md",
            resume=True,
        )


def test_byzantine_block_gate_is_clustered_by_seed_before_worst_cell():
    rows = []
    for seed, draws in {11: [0.10, 0.30], 29: [0.20]}.items():
        for draw, block_zero in enumerate(draws):
            rows.append(
                {
                    "noise_regime": "heteroscedastic",
                    "noise_permutation": "byzantine_high",
                    "outlier_geometry": "aligned",
                    "threat": "ipm",
                    "severity": 1.0,
                    "seed": seed,
                    "draw": draw,
                    "byzantine_endpoint_clipped_gradient_share_by_block": [
                        block_zero,
                        0.05,
                    ],
                }
            )
    # Seed 11 mean is .20 and seed 29 mean is .20; unequal draw counts must
    # not turn the result into the raw-draw mean or raw maximum (.30).
    assert g0f._worst_seed_clustered_block_mean(
        rows, "byzantine_endpoint_clipped_gradient_share_by_block"
    ) == pytest.approx(0.20)


def test_byzantine_endpoint_share_uses_unblended_p_k_not_returned_reference():
    vectors = torch.tensor([[0.0], [10.0]], dtype=torch.float64)
    metrics = g0f._candidate_diagnostics(
        vectors=vectors,
        # Deliberately swap which client is close: using this returned/blended
        # reference would assign the Byzantine client zero gradient mass.
        reference=torch.tensor([10.0], dtype=torch.float64),
        crossfit_references=torch.zeros_like(vectors),
        statistical_radii=torch.full((2, 1), 100.0, dtype=torch.float64),
        deployed_radii=torch.ones((2, 1), dtype=torch.float64),
        diagnostics={
            "allocated_block_budgets": [[1.0], [1.0]],
            "pre_blend_reference": [0.0],
            "complete_client_influence_cap": 1.0,
            "allocation_diagnostics": {
                "budget_to_radius_ratio_min": 1.0,
                "budget_to_radius_ratio_max": 1.0,
                "allocated_client_norm_max": 1.0,
            },
        },
        regular_honest=torch.tensor([True, False]),
        honest_outliers=torch.tensor([False, False]),
        byzantine=torch.tensor([False, True]),
        noise_tiers=torch.ones(2, dtype=torch.float64),
        block_sizes=[1],
    )
    assert metrics["byzantine_endpoint_clipped_gradient_share"] == pytest.approx(1.0)
    assert metrics["byzantine_endpoint_clipped_gradient_share_by_block"] == [1.0]
    assert metrics["byzantine_endpoint_share_is_causal_decomposition"] is False


def test_execution_gated_holdout_registry_matches_frozen_commitment(
    campaign_config: dict,
):
    assert g0f._load_execution_gated_holdout_seeds(campaign_config) == [
        2026102003,
        2026102019,
        2026102031,
        2026102047,
        2026102059,
        2026102071,
        2026102083,
    ]


def _minimal_paired_checkpoint_rows(config: dict) -> list[dict]:
    rows = []
    seed = 17
    digest = "a" * 64
    for regime in config["privacy_noise"]["regimes"]:
        for permutation in regime["permutations"]:
            pairing = (
                f"development:{regime['name']}:{permutation}:aligned:"
                f"none:1.000:{seed}:0"
            )
            for candidate in g0f.CANDIDATES:
                rows.append(
                    {
                        "phase": "development",
                        "seed": seed,
                        "draw": 0,
                        "candidate": candidate,
                        "pairing_id": pairing,
                        "noise_regime": regime["name"],
                        "noise_permutation": permutation,
                        "outlier_geometry": "aligned",
                        "threat": "none",
                        "severity": 1.0,
                        "resolved_device": "mps:0",
                        "tensor_dtype": "float32",
                        "reference_error": 1.0,
                        "reference_error_ratio_to_uniform": 1.0,
                        "reference_error_ratio_to_fcc": 1.0,
                        "clean_cohort_sha256": digest,
                        "standard_noise_z_sha256": digest,
                        "observed_private_cohort_sha256": digest,
                        "noise_rescaled_source_z_sha256": digest,
                        "coordinate_noise_std_sha256": digest,
                        "attacked_cohort_sha256": digest,
                        "tier_assignment_sha256": digest,
                        "outlier_mask_sha256": digest,
                        "byzantine_mask_sha256": digest,
                    }
                )
    return rows


def test_checkpoint_rejects_cross_regime_standard_noise_mismatch(
    campaign_config: dict, tmp_path: Path
):
    small = copy.deepcopy(campaign_config)
    small["privacy_noise"]["regimes"] = [
        {
            "name": "homogeneous",
            "client_std_multipliers": [1.0],
            "permutations": ["identity"],
        },
        {
            "name": "heteroscedastic",
            "client_std_multipliers": [1.0, 2.0],
            "permutations": ["identity"],
        },
    ]
    small["cohort"]["honest_outliers"]["geometries"] = ["aligned"]
    small["threats"]["names"] = ["none"]
    small["randomness"]["development_draws_per_seed"] = 1
    rows = _minimal_paired_checkpoint_rows(small)
    g0f._validate_checkpoint(
        rows,
        config=small,
        phase="development",
        seed=17,
        kind="detail",
        source=tmp_path / "valid.json",
    )
    for row in rows:
        if row["noise_regime"] == "heteroscedastic":
            row["standard_noise_z_sha256"] = "b" * 64
            row["noise_rescaled_source_z_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="Cross-regime pairing violation"):
        g0f._validate_checkpoint(
            rows,
            config=small,
            phase="development",
            seed=17,
            kind="detail",
            source=tmp_path / "corrupted.json",
        )
