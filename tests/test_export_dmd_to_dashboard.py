from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "export_dmd_to_dashboard.py"
SPEC = importlib.util.spec_from_file_location("export_dmd_to_dashboard", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_export_preserves_dmd_metrics_and_adds_unit_explicit_aliases(tmp_path: Path) -> None:
    run_dir = tmp_path / "dmd_cb_emnist" / "alpha_0p1" / "benchmark_seed91"
    run_dir.mkdir(parents=True)
    method = "margin_mean_cb_fixed_zero_stale_usv_r025"
    round_csv = run_dir / f"round_metrics_{method}_seed91.csv"
    client_csv = run_dir / f"client_metrics_{method}_seed91.csv"
    pd.DataFrame(
        {
            "seed": [91, 91],
            "method": [method, method],
            "round": [1, 2],
            "test_accuracy": [0.25, 0.30],
            "test_loss": [2.0, 1.8],
            "worst20_accuracy": [0.10, 0.12],
            "client_accuracy_variance": [0.02, 0.01],
            "mean_client_balanced_accuracy": [0.20, 0.24],
            "worst20_client_balanced_accuracy": [0.08, 0.11],
            "client_balanced_accuracy_variance": [0.012, 0.010],
            "best_worst_client_balanced_accuracy_gap": [0.40, 0.35],
            "weighted_client_loss_variance": [0.2, 0.15],
            "canonical_cb_deficit_mean": [0.5, 0.4],
            "canonical_cb_deficit_cvar20": [0.8, 0.7],
            "stale_usv_value": [float("nan"), 0.12],
        }
    ).to_csv(round_csv, index=False)
    pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "client_id": [0, 1, 0, 1],
            "accuracy": [0.1, 0.3, 0.2, 0.4],
            "balanced_accuracy": [0.08, 0.32, 0.18, 0.42],
            "loss": [2.2, 1.8, 2.0, 1.6],
            "canonical_cb_deficit": [0.7, 0.3, 0.6, 0.2],
        }
    ).to_csv(client_csv, index=False)

    artifacts = MODULE.discover_artifacts(tmp_path)
    assert len(artifacts) == 1
    output_root = tmp_path / "dashboard"
    metrics_path = MODULE.export_artifact(
        artifacts[0], input_root=tmp_path, output_root=output_root
    )
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert data["algorithm"] == method
    assert data["dataset"] == "emnist"
    assert data["summary"]["num_rounds"] == 2
    assert data["summary"]["best_accuracy"] == 0.30
    first, second = data["rounds"]
    assert first["round_num"] == 1
    assert first["worst20_balanced_accuracy_pct"] == 8.0
    assert first["client_balanced_accuracy_variance_pct2"] == 120.0
    assert first["best_worst_balanced_accuracy_gap_pct"] == 40.0
    assert first["balanced_performance_fairness"] == 0.2
    assert first["client_balanced_accuracy_values_oracle"] == [0.08, 0.32]
    assert first["stale_usv_value"] is None
    assert second["client_dmd_cb_values_oracle"] == [0.6, 0.2]
    assert "best20_worst20_gap_pct" not in first
    assert data["export"]["source_manifest"] is None
    assert data["export"]["source_summary"] is None


def test_export_rejects_duplicate_rounds(tmp_path: Path) -> None:
    run_dir = tmp_path / "alpha_0p3" / "benchmark_seed92"
    run_dir.mkdir(parents=True)
    path = run_dir / "round_metrics_fedavg_seed92.csv"
    pd.DataFrame({"round": [1, 1], "test_accuracy": [0.1, 0.2]}).to_csv(
        path, index=False
    )
    artifact = MODULE.discover_artifacts(tmp_path)[0]
    try:
        MODULE.export_artifact(
            artifact, input_root=tmp_path, output_root=tmp_path / "dashboard"
        )
    except ValueError as error:
        assert "duplicate round" in str(error)
    else:
        raise AssertionError("duplicate rounds must be rejected")
