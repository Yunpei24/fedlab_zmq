from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from robustness.aggregators import centered_clipping_leave_one_out, clip_l2
from scripts import run_gaussian_aware_reference_oracle as oracle
from scripts.run_gaussian_aware_reference_g0b import run

ROOT = Path(__file__).resolve().parents[1]


def test_g0b_locks_development_before_disjoint_holdout(tmp_path):
    config = yaml.safe_load(
        (
            ROOT
            / "configs/ldp_gradient_far/gaussian_aware_reference_g0b.yaml"
        ).read_text(encoding="utf-8")
    )
    config["cohort"].update(
        {
            "num_clients": 10,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    config["cohort"]["honest_outliers"].update(
        {"count": 2, "geometries": ["orthogonal"]}
    )
    config["privacy_noise"]["block_std_multipliers"] = [0.8, 1.2]
    config["privacy_noise"]["regimes"] = [
        {
            "name": "heteroscedastic",
            "client_std_multipliers": [1.0, 2.0],
            "permutations": ["identity"],
        }
    ]
    config["references"]["candidates"] = [
        {"id": "uniform_mean", "method": "uniform_mean"},
        {"id": "fcc", "method": "fcc"},
        {
            "id": "sigma_huber_g020",
            "method": "f_sigma_huber",
            "influence_cap_total": 0.20,
            "regularization": 0.20,
        },
    ]
    config["references"]["f_sigma_huber"]["num_steps"] = 3
    config["selection"]["candidate_references"] = ["sigma_huber_g020"]
    config["threats"].update(
        {
            "names": ["none", "ipm"],
            "separated_for_gates": ["ipm"],
            "evasive_controls": [],
            "development_severities": [1.0],
            "holdout_severities": [1.0],
        }
    )
    config["randomness"].update(
        {
            "null_calibration_seed": 9001,
            "null_calibration_draws": 3,
            "development_seeds": [9011, 9013, 9017],
            "development_draws_per_seed": 1,
            "holdout_seeds": [9029, 9041, 9043, 9049, 9059],
            "holdout_draws_per_seed": 1,
        }
    )
    config_path = tmp_path / "g0b.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = tmp_path / "results"
    report = tmp_path / "report.md"
    decision = run(
        config_path,
        output,
        report,
        device="cpu",
        test_only_allow_cpu=True,
    )

    assert decision["holdout_used_for_selection"] is False
    assert decision["accuracy_used_for_selection"] is False
    assert decision["development_seeds"] == [9011, 9013, 9017]
    assert decision["holdout_seeds"] == [9029, 9041, 9043, 9049, 9059]
    assert set(decision["development_seeds"]).isdisjoint(decision["holdout_seeds"])
    assert decision["resolved_device"] == "cpu"

    lock = json.loads((output / "development_lock.json").read_text())
    assert lock["candidate"] == "sigma_huber_g020"
    assert lock["holdout_used_for_selection"] is False
    with (output / "development_detail.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        development = list(csv.DictReader(handle))
    paired = Counter(row["pairing_id"] for row in development)
    # Uniform contributes one mode, FCC and Huber each contribute three.
    assert set(paired.values()) == {7}
    assert {
        row["weight_mode"]
        for row in development
        if row["candidate"] == "sigma_huber_g020"
    } == {"reference_only", "novelty_only", "novelty_confidence"}
    assert all(row["phase"] == "development" for row in development)
    assert (output / "holdout_summary.csv").exists()
    assert report.exists()


def test_noise_draw_is_strictly_paired_across_tier_permutations():
    oracle._configure_runtime("cpu")
    clean = torch.zeros(6, 4, dtype=torch.float64)
    first_variance = torch.tensor(
        [[1.0], [4.0], [9.0], [1.0], [4.0], [9.0]], dtype=torch.float64
    )
    second_variance = first_variance.flip(0)
    first = oracle._add_private_noise(
        clean,
        first_variance,
        [4],
        seed=77,
        draw=2,
        regime="heteroscedastic",
        permutation="identity",
        geometry="orthogonal",
        pair_permutations=True,
    )
    second = oracle._add_private_noise(
        clean,
        second_variance,
        [4],
        seed=77,
        draw=2,
        regime="heteroscedastic",
        permutation="reverse",
        geometry="orthogonal",
        pair_permutations=True,
    )
    first_z = first / first_variance.sqrt()
    second_z = second / second_variance.sqrt()
    assert torch.allclose(first_z, second_z, atol=1e-15, rtol=1e-15)


def test_post_server_clip_pipeline_is_invariant_to_larger_collinear_upload():
    oracle._configure_runtime("cpu")
    low = torch.tensor(
        [[2.0, 0.0], [0.1, 0.2], [-0.1, 0.1], [0.0, -0.2]],
        dtype=torch.float64,
    )
    high = low.clone()
    high[0] = torch.tensor([20.0, 0.0], dtype=torch.float64)
    bounded_low = clip_l2(low, 1.0)
    bounded_high = clip_l2(high, 1.0)
    assert torch.equal(bounded_low, bounded_high)

    anchor = torch.zeros(2, dtype=torch.float64)
    loo_low = centered_clipping_leave_one_out(
        bounded_low, anchor=anchor, tau=0.5
    )
    loo_high = centered_clipping_leave_one_out(
        bounded_high, anchor=anchor, tau=0.5
    )
    assert torch.equal(loo_low, loo_high)
    scores_low = oracle._standardized_scores(
        bounded_low,
        loo_low,
        {
            "cohort": {"block_sizes": [2], "heterogeneity_std_by_block": [0.1]},
            "score": {"variance_floor": 1e-12},
        },
        noise_variances=torch.ones(4, 1, dtype=torch.float64),
        reference_variances=torch.ones(4, 1, dtype=torch.float64),
    )[0]
    scores_high = oracle._standardized_scores(
        bounded_high,
        loo_high,
        {
            "cohort": {"block_sizes": [2], "heterogeneity_std_by_block": [0.1]},
            "score": {"variance_floor": 1e-12},
        },
        noise_variances=torch.ones(4, 1, dtype=torch.float64),
        reference_variances=torch.ones(4, 1, dtype=torch.float64),
    )[0]
    assert torch.equal(scores_low, scores_high)
    score_config = {
        "score": {
            "novelty_start_z": 1.0,
            "novelty_full_z": 2.5,
            "rejection_start_z": 3.0,
            "rejection_full_z": 5.0,
            "trust_floor": 0.02,
            "alpha": 0.8,
        }
    }
    weights_low, _ = oracle._oracle_weights(
        scores_low, score_config, mode="novelty_confidence"
    )
    weights_high, _ = oracle._oracle_weights(
        scores_high, score_config, mode="novelty_confidence"
    )
    assert torch.equal(weights_low, weights_high)
    assert torch.equal(
        (weights_low[:, None] * bounded_low).sum(dim=0),
        (weights_high[:, None] * bounded_high).sum(dim=0),
    )
