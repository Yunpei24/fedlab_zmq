from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.multiseed_summary import (
    SUMMARY_COLUMNS,
    best_value_indices,
    load_run_summary,
    method_label,
    run_selector_label,
    summarize_runs,
)


def _write_run(
    root: Path,
    *,
    method_folder: str,
    seed: int,
    algorithm: str,
    config: dict,
    client_accuracy: float,
    test_accuracy: float,
    variance: float,
    worst20: float,
    gap: float,
) -> Path:
    run_dir = (
        root
        / "algorithm_fidelity_v4"
        / "exp1_fairness_no_attack"
        / "non_private"
        / "none"
        / method_folder
        / f"seed{seed}"
        / f"{algorithm}_mnist_s{seed}"
    )
    run_dir.mkdir(parents=True)
    payload = {
        "algorithm": algorithm,
        "config": config,
        "rounds": [
            {
                "round_num": 1,
                "client_accuracy_mean": 0.1,
                "test_accuracy": 0.1,
                "client_accuracy_variance": 0.01,
                "worst20_accuracy": 0.01,
                "best20_worst20_gap": 0.02,
            },
            {
                "round_num": 40,
                "client_accuracy_mean": client_accuracy,
                "test_accuracy": test_accuracy,
                "client_accuracy_variance": variance,
                "worst20_accuracy": worst20,
                "best20_worst20_gap": gap,
            },
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir


def test_far_alpha_is_part_of_method_and_selector_identity(tmp_path: Path) -> None:
    run_a = _write_run(
        tmp_path,
        method_folder="far_alpha_0p1",
        seed=28,
        algorithm="far",
        config={"far_alpha": 0.1},
        client_accuracy=0.91,
        test_accuracy=0.90,
        variance=0.001,
        worst20=0.82,
        gap=0.10,
    )
    run_b = _write_run(
        tmp_path,
        method_folder="far_alpha_1p0",
        seed=28,
        algorithm="far",
        config={"far_alpha": 1.0},
        client_accuracy=0.92,
        test_accuracy=0.91,
        variance=0.002,
        worst20=0.83,
        gap=0.09,
    )
    data_a = json.loads((run_a / "metrics.json").read_text(encoding="utf-8"))
    data_b = json.loads((run_b / "metrics.json").read_text(encoding="utf-8"))

    assert method_label(data_a, run_a) == "FAR (α=0.1)"
    assert method_label(data_b, run_b) == "FAR (α=1)"
    assert run_selector_label(data_a, run_a) != run_selector_label(data_b, run_b)


def test_summary_aggregates_final_round_across_seeds(tmp_path: Path) -> None:
    run_28 = _write_run(
        tmp_path,
        method_folder="far_alpha_0p4",
        seed=28,
        algorithm="far",
        config={"far_alpha": 0.4},
        client_accuracy=0.90,
        test_accuracy=0.92,
        variance=0.001,
        worst20=0.80,
        gap=0.12,
    )
    run_36 = _write_run(
        tmp_path,
        method_folder="far_alpha_0p4",
        seed=36,
        algorithm="far",
        config={"far_alpha": 0.4},
        client_accuracy=0.94,
        test_accuracy=0.96,
        variance=0.003,
        worst20=0.84,
        gap=0.08,
    )

    table, seeds = summarize_runs(
        [load_run_summary(run_28), load_run_summary(run_36)], include_std=True
    )

    assert list(table.columns) == SUMMARY_COLUMNS
    assert table.shape == (1, 6)
    assert table.iloc[0]["Méthode"] == "FAR (α=0.4)"
    assert table.iloc[0]["Client Acc. (%)"] == "92.000 ± 2.828"
    assert table.iloc[0]["Test Acc. (%)"] == "94.000 ± 2.828"
    # Raw variance values 0.001 and 0.003 become 10 and 30 pp².
    assert table.iloc[0]["Var (pp²)"] == "20.000 ± 14.142"
    assert table.iloc[0]["Worst-20 (%)"] == "82.000 ± 2.828"
    assert table.iloc[0]["Gap Δk (pp)"] == "10.000 ± 2.828"
    assert seeds == {"FAR (α=0.4)": [28, 36]}


def test_percent_encoded_metrics_are_not_scaled_twice(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        method_folder="fedavg",
        seed=28,
        algorithm="fedavg",
        config={},
        client_accuracy=0.90,
        test_accuracy=0.91,
        variance=0.001,
        worst20=0.80,
        gap=0.10,
    )
    metrics_path = run_dir / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    data["rounds"][-1].update(
        {
            "client_accuracy_variance_pct2": 12.5,
            "worst20_accuracy_pct": 81.5,
            "best20_worst20_gap_pct": 9.25,
        }
    )
    metrics_path.write_text(json.dumps(data), encoding="utf-8")

    summary = load_run_summary(run_dir)
    assert summary.variance_pct2 == pytest.approx(12.5)
    assert summary.worst20_pct == pytest.approx(81.5)
    assert summary.gap_pct == pytest.approx(9.25)


def test_best_values_use_the_correct_direction_and_keep_ties() -> None:
    table = pd.DataFrame(
        [
            {
                "Méthode": "A",
                "Client Acc. (%)": "92.000 ± 1.000",
                "Test Acc. (%)": "91.000 ± 1.000",
                "Var (pp²)": "12.000 ± 2.000",
                "Worst-20 (%)": "82.000 ± 1.000",
                "Gap Δk (pp)": "9.000 ± 1.000",
            },
            {
                "Méthode": "B",
                "Client Acc. (%)": "94.000 ± 8.000",
                "Test Acc. (%)": "93.000 ± 2.000",
                "Var (pp²)": "10.000 ± 4.000",
                "Worst-20 (%)": "85.000 ± 2.000",
                "Gap Δk (pp)": "7.000 ± 3.000",
            },
            {
                "Méthode": "C",
                "Client Acc. (%)": "94.000 ± 0.500",
                "Test Acc. (%)": "92.000 ± 0.500",
                "Var (pp²)": "11.000 ± 1.000",
                "Worst-20 (%)": "84.000 ± 1.000",
                "Gap Δk (pp)": "8.000 ± 1.000",
            },
        ]
    )

    best = best_value_indices(table)

    assert best["Client Acc. (%)"] == {1, 2}
    assert best["Test Acc. (%)"] == {1}
    assert best["Var (pp²)"] == {1}
    assert best["Worst-20 (%)"] == {1}
    assert best["Gap Δk (pp)"] == {1}
