from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.analyze_ldp_gradient_far_positioning import (
    _max_drawdown,
    _normalized_auc,
    analyze_campaign,
    collect_runs,
    summarize_phases,
)
from scripts.run_ldp_gradient_far_positioning import (
    DEFAULT_MATRIX,
    load_campaign,
    resolved_config,
    task_output_dir,
)


def _payload(campaign, task, *, wrong_seed: bool = False, incomplete: bool = False):
    config = resolved_config(campaign, task)
    algo = copy.deepcopy(config["training"]["algo_config"])
    # run_experiment propagates the CLI device into the algorithm metrics.
    algo["device"] = "mps"
    rounds_count = int(config["training"]["num_rounds"])
    if incomplete:
        rounds_count -= 1
    rounds = []
    for round_num in range(1, rounds_count + 1):
        accuracy = 0.10 + 0.01 * min(round_num, 10)
        rounds.append(
            {
                "round_num": round_num,
                "test_accuracy": accuracy,
                "client_accuracy_mean": accuracy - 0.01,
                "client_accuracy_variance_pct2": 4.0,
                "worst20_accuracy_pct": 8.0,
                "best20_worst20_gap_pct": 12.0,
                "test_loss": 2.0 - accuracy,
                "train_loss": 1.9 - accuracy,
                "survival_ratio": 1.0,
                "privacy_sampling_scheme": "fixed_without_replacement",
                "privacy_adjacency": "replace_one",
                "privacy_epsilon_max": 4.0 if algo["enable_dp"] else None,
                "privacy_delta": 1e-5 if algo["enable_dp"] else None,
                "privacy_model_noise_multiplier_min": 1.0,
                "privacy_model_noise_multiplier_mean": 1.2,
                "privacy_model_noise_multiplier_max": 1.5,
                "privacy_clip_rate_mean": 0.25,
                "far_server_clip_rate": 0.10,
                "far_score_span": 0.50,
                "far_logit_range": 1.0,
                "max_client_weight": 0.15,
                "weight_entropy": 2.1,
                "effective_num_clients": 8.5,
                "far_noise_amplification_vs_uniform": 1.1,
                "byzantine_weight_mass_oracle": 0.18,
                "far_reference_honest_center_error_oracle": 0.3,
                "far_weight_effective_noise_corr_oracle": 0.2,
                "far_honest_noisy_clean_score_corr_oracle": 0.7,
                "far_honest_clean_tilting_bias_norm_oracle": 0.04,
                "far_honest_fixed_weight_dp_noise_norm_oracle": 0.05,
                "far_byzantine_displacement_norm_oracle": 0.06,
                "far_aggregate_error_to_clean_honest_center_norm_oracle": 0.09,
            }
        )
    return {
        "algorithm": "ldp_gradient_far",
        "dataset": config["data"]["dataset"],
        "config": algo,
        "summary": {
            "num_rounds": int(config["training"]["num_rounds"]),
            "seed": task.seed + int(wrong_seed),
            "partition_seed": task.seed,
            "dataset": config["data"]["dataset"],
            "model": config["model"]["architecture"],
            "num_clients": config["clients"]["num_clients"],
        },
        "rounds": rounds,
    }


def _write_metrics(campaign, task, payload) -> Path:
    path = task_output_dir(campaign, task) / "experiment" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_auc_and_drawdown_have_percentage_point_interpretation():
    trajectory = [(1, 10.0), (2, 30.0), (3, 20.0)]
    assert _normalized_auc(trajectory) == pytest.approx(22.5)
    assert _max_drawdown(trajectory) == pytest.approx(10.0)
    assert _normalized_auc([(7, 42.0)]) == pytest.approx(42.0)


def test_partial_campaign_separates_complete_invalid_and_missing(tmp_path):
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    first, second = campaign.tasks[:2]
    _write_metrics(campaign, first, _payload(campaign, first))
    _write_metrics(campaign, second, _payload(campaign, second, incomplete=True))

    records = collect_runs(campaign)
    assert len(records) == len(campaign.tasks)
    assert records[0]["status"] == "complete"
    assert records[1]["status"] == "invalid"
    assert sum(row["status"] == "missing" for row in records) == len(records) - 2
    assert records[0]["test_accuracy_final_pct"] == pytest.approx(20.0)
    phase = summarize_phases(campaign, records)[0]
    assert phase == {
        "phase": "a_activation_screen",
        "role": "development_screen",
        "description": "LeNet-5 Tanh versus ReLU on MNIST/Fashion-MNIST and n=10/25.",
        "expected": 8,
        "complete": 1,
        "invalid": 1,
        "missing": 6,
    }


def test_protocol_mismatch_is_invalid_not_silently_accepted(tmp_path):
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    task = campaign.tasks[0]
    _write_metrics(campaign, task, _payload(campaign, task, wrong_seed=True))
    row = collect_runs(campaign)[0]
    assert row["status"] == "invalid"
    assert "seed" in row["reason"]


def test_analysis_exports_descriptive_artifacts_without_writing_gate(tmp_path):
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    task = campaign.tasks[0]
    _write_metrics(campaign, task, _payload(campaign, task))
    output_dir = tmp_path / "analysis"
    report = tmp_path / "report.md"

    result = analyze_campaign(campaign, output_dir=output_dir, report_path=report)

    assert result["analysis_kind"] == "descriptive_only"
    assert result["automated_gate_decision"] is False
    assert result["counts"] == {
        "expected": len(campaign.tasks),
        "complete": 1,
        "invalid": 0,
        "missing": len(campaign.tasks) - 1,
    }
    assert (output_dir / "run_level.csv").exists()
    assert (output_dir / "variant_summary.csv").exists()
    assert (output_dir / "phase_status.csv").exists()
    assert (output_dir / "analysis.json").exists()
    assert "descriptif uniquement" in report.read_text(encoding="utf-8")
    assert not (campaign.output_root / "_gates").exists()


def test_dp_oracle_metrics_are_reported_but_labeled_oracle(tmp_path):
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    task = next(task for task in campaign.tasks if task.phase_id == "d_privacy_screen")
    _write_metrics(campaign, task, _payload(campaign, task))
    row = next(
        item
        for item in collect_runs(campaign)
        if item["phase"] == task.phase_id and item["run_id"] == task.run_id
    )
    assert row["status"] == "complete"
    assert row["privacy_epsilon_final"] == pytest.approx(4.0)
    assert row["byzantine_weight_mass_median_oracle"] == pytest.approx(0.18)
    assert row["honest_noisy_clean_score_corr_median_oracle"] == pytest.approx(0.7)
    assert all(
        key.endswith("_oracle")
        for key in row
        if "honest_center" in key or "byzantine_displacement" in key
    )


def test_analyzer_reconstructs_nonnegative_logit_range_for_negative_alpha(tmp_path):
    campaign = replace(load_campaign(DEFAULT_MATRIX), output_root=tmp_path / "results")
    task = next(
        task
        for task in campaign.tasks
        if task.phase_id == "c_alpha_reference_screen"
        and resolved_config(campaign, task)["training"]["algo_config"]["far_alpha"]
        == -2.0
    )
    payload = _payload(campaign, task)
    for round_row in payload["rounds"]:
        round_row["far_score_span"] = 0.5
        # Emulate the legacy producer bug. The analyzer must not preserve it.
        round_row["far_logit_range"] = -1.0
    _write_metrics(campaign, task, payload)
    row = next(
        item
        for item in collect_runs(campaign)
        if item["phase"] == task.phase_id and item["run_id"] == task.run_id
    )
    assert row["far_logit_range_final"] == pytest.approx(1.0)
    assert row["far_logit_range_median"] == pytest.approx(1.0)
