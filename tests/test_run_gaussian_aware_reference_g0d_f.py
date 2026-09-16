from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from scripts import run_gaussian_aware_reference_g0d_f as g0d
from scripts import run_gaussian_aware_reference_oracle as oracle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0d_f.yaml"


@pytest.fixture()
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_registered_matrix_has_twelve_unique_candidates(config: dict) -> None:
    g0d._validate_config(config)
    specs = g0d._candidate_specs(config)
    assert len(specs) == 12
    assert len({spec["id"] for spec in specs}) == 12
    assert {spec["quantile"] for spec in specs} == {0.80, 0.90, 0.94}
    assert {spec["influence_cap_total"] for spec in specs} == {0.22, 0.30}
    assert {spec["regularization"] for spec in specs} == {0.10, 0.25}
    assert set(config["threats"]["separated_for_gates"]) == {
        "ipm",
        "bitflip_x10",
        "model_replacement",
    }
    assert set(config["threats"]["evasive_controls"]) == {"alie"}


def test_validation_rejects_seed_reuse_and_cpu_fallback(config: dict) -> None:
    reused = copy.deepcopy(config)
    reused["randomness"]["holdout_seeds"][0] = 28
    with pytest.raises(ValueError, match="reuses a seed"):
        g0d._validate_config(reused)

    fallback = copy.deepcopy(config)
    fallback["execution"]["allow_cpu_fallback"] = True
    with pytest.raises(ValueError, match="forbids CPU fallback"):
        g0d._validate_config(fallback)


def test_mps_compatible_quantile_and_coordinate_median() -> None:
    values = torch.tensor([4.0, 1.0, 3.0, 2.0])
    assert g0d._empirical_quantile(values, 0.50).item() == pytest.approx(2.5)
    assert g0d._empirical_quantile(values, 0.75).item() == pytest.approx(3.25)
    vectors = torch.tensor([[0.0, 9.0], [2.0, 1.0], [4.0, 5.0], [6.0, 3.0]])
    assert torch.equal(g0d._coordinate_median(vectors), torch.tensor([3.0, 4.0]))


def test_null_thresholds_are_pooled_not_identity_specific(config: dict) -> None:
    small = copy.deepcopy(config)
    small["randomness"]["null_calibration_draws_per_seed"] = 2
    oracle._configure_runtime("cpu")
    thresholds, rows = g0d._null_thresholds(small)
    for quantile in (0.80, 0.90, 0.94):
        assert (
            thresholds[("heteroscedastic", "identity", quantile)]
            == thresholds[("heteroscedastic", "reverse", quantile)]
        )
    assert all(
        row["calibration_pool"] == "all_regimes_and_permutations" for row in rows
    )
    assert all(0.0 <= row["empirical_tail_rate"] <= 1.0 for row in rows)
    for block in range(4):
        ordered = [
            thresholds[("homogeneous", "identity", quantile)][block]
            for quantile in (0.80, 0.90, 0.94)
        ]
        assert ordered == sorted(ordered)


def test_all_comparator_references_are_finite(config: dict) -> None:
    oracle._configure_runtime("cpu")
    n = int(config["cohort"]["num_clients"])
    d = int(config["cohort"]["dimension"])
    vectors = torch.randn(n, d, dtype=torch.float64) * 0.01
    anchor = torch.zeros(d, dtype=torch.float64)
    regime = config["privacy_noise"]["regimes"][1]
    variances, _ = oracle._noise_variances(config, regime, "identity")
    for name in g0d.COMPARATORS:
        reference, diagnostics = g0d._reference(
            name,
            vectors,
            config,
            anchor=anchor,
            noise_variances=variances,
        )
        assert reference.shape == (d,)
        assert torch.isfinite(reference).all()
        if name == "g0c_locked":
            assert diagnostics["finite_solver_replace_one_bound"] <= 0.12


def test_replace_one_rows_distinguish_empirical_and_certified(config: dict) -> None:
    small = copy.deepcopy(config)
    small["privacy_noise"]["regimes"] = [small["privacy_noise"]["regimes"][0]]
    small["randomness"]["replace_one_trials_per_seed_cell"] = 1
    oracle._configure_runtime("cpu")
    spec = g0d._candidate_specs(small)[-1]
    thresholds = {("homogeneous", "identity", spec["quantile"]): [4.0] * 4}
    rows = g0d._replace_one_audit(
        config=small,
        phase="unit",
        seeds=[4001],
        specs=[spec],
        calibrated=thresholds,
    )
    by_name = {row["candidate"]: row for row in rows}
    assert set(g0d.COMPARATORS).issubset(by_name)
    assert spec["id"] in by_name
    assert by_name[spec["id"]]["theoretical_bound"] <= 0.12
    assert by_name[spec["id"]]["violation"] is False
    assert math.isnan(by_name["coordinate_median"]["theoretical_bound"])
    assert by_name["coordinate_median"]["violation"] is None
    assert "dimension_dependent" in by_name["trimmed_mean"]["certificate"]


def test_resume_reads_seed_checkpoint_without_recomputation(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "development_detail_seed_4001.json"
    g0d._atomic_json(checkpoint, [{"sentinel": 7}])

    def fail_if_called(**_: object) -> list[dict]:
        raise AssertionError("resume should not recompute a valid seed checkpoint")

    monkeypatch.setattr(g0d, "_phase_rows", fail_if_called)
    rows = g0d._cached_phase_rows(
        checkpoint_dir=tmp_path,
        resume=True,
        config=config,
        phase="development",
        seeds=[4001],
        draws_per_seed=1,
        severities=[1.0],
        specs=g0d._candidate_specs(config),
        calibrated={},
    )
    assert rows == [{"sentinel": 7}]
