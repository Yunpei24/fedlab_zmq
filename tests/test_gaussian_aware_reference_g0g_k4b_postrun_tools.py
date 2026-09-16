"""Light contract tests for the independent K4b post-run tools."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import audit_gaussian_aware_reference_g0g_k4b_results as audit
from scripts import plot_gaussian_aware_g0g_k4b as figures


def _running_manifest(directory: Path) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": audit.CAMPAIGN_ID,
                "status": "running_development_screen",
            }
        ),
        encoding="utf-8",
    )


def test_auditor_does_not_open_raw_artifacts_before_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _running_manifest(tmp_path)

    def forbidden_read_csv(*args: object, **kwargs: object) -> None:
        raise AssertionError("A raw CSV was opened before completion")

    monkeypatch.setattr(audit.pd, "read_csv", forbidden_read_csv)
    with pytest.raises(RuntimeError, match="raw artifacts were not opened"):
        audit.audit_results(
            tmp_path / "missing.yaml", tmp_path, tmp_path / "audit.json"
        )
    assert not (tmp_path / "audit.json").exists()


def test_figure_tool_does_not_open_raw_artifacts_before_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _running_manifest(tmp_path)

    def forbidden_read_csv(*args: object, **kwargs: object) -> None:
        raise AssertionError("A raw CSV was opened before completion")

    monkeypatch.setattr(figures.pd, "read_csv", forbidden_read_csv)
    with pytest.raises(RuntimeError, match="figures did not open raw results"):
        figures._validate_sources(tmp_path)


def test_student_ci_uses_five_seed_units() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = audit._ci95(values)
    expected_half_width = 2.776 * math.sqrt(2.5) / math.sqrt(5.0)
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["low"] == pytest.approx(3.0 - expected_half_width)
    assert result["high"] == pytest.approx(3.0 + expected_half_width)


def test_log_ratio_is_computed_once_per_seed_before_ci() -> None:
    rows: list[dict[str, object]] = []
    for seed in range(5):
        for context, baseline in enumerate((1.0 + seed, 3.0 + seed)):
            trajectory = f"{seed}|{context}"
            rows.extend(
                [
                    {
                        "trajectory_id": trajectory,
                        "seed": seed,
                        "candidate": audit.K4,
                        "attack_auc": baseline,
                    },
                    {
                        "trajectory_id": trajectory,
                        "seed": seed,
                        "candidate": audit.PRIMARY,
                        "attack_auc": 0.9 * baseline,
                    },
                ]
            )
    frame = pd.DataFrame(rows)
    result = audit._paired_seed_log_ratios(
        frame,
        pd.Series(True, index=frame.index),
        candidate=audit.PRIMARY,
        baseline=audit.K4,
        metric="attack_auc",
    )
    assert result["n_seed_pairs"] == 5
    assert result["seed_ids"] == [0, 1, 2, 3, 4]
    assert result["geometric_mean_ratio"] == pytest.approx(0.9)
    assert result["exp_log_ratio_ci95_high"] == pytest.approx(0.9)
    assert result["pooled_ratio_descriptive"] == pytest.approx(0.9)


def test_component_masses_are_rebuilt_from_client_rows() -> None:
    clients = pd.DataFrame(
        {
            "trajectory_id": ["cell", "cell"],
            "round": [13, 13],
            "latent_byzantine_bool": [False, True],
            "primary_direct_current_contribution_norm": [0.01, 0.03],
            "primary_imputed_contribution_norm": [0.02, 0.01],
            "primary_total_slot_contribution_norm": [0.025, 0.035],
            "delta_direct_current_contribution_norm": [0.01, 0.03],
            "delta_imputed_contribution_norm": [0.005, 0.005],
            "delta_total_slot_contribution_norm": [0.012, 0.032],
        }
    )
    round_rows: list[dict[str, object]] = []
    for candidate, direct, imputed, total in (
        (audit.PRIMARY, (0.04, 0.75), (0.03, 1.0 / 3.0), (0.06, 7.0 / 12.0)),
        (audit.DELTA, (0.04, 0.75), (0.01, 0.50), (0.044, 8.0 / 11.0)),
    ):
        round_rows.append(
            {
                "trajectory_id": "cell",
                "round": 13,
                "candidate": candidate,
                "direct_current_mass_total": direct[0],
                "byzantine_direct_current_mass_share": direct[1],
                "imputed_mass_total": imputed[0],
                "byzantine_imputed_mass_share": imputed[1],
                "total_slot_mass_total": total[0],
                "byzantine_total_slot_mass_share": total[1],
                "max_client_contribution_norm": 0.035
                if candidate == audit.PRIMARY
                else 0.032,
            }
        )
    config = {
        "temporal": {"first_temporal_gate_round": 13},
        "references": {"total_client_influence_cap": 0.13},
    }
    passed, details = audit._component_mass_audit(
        pd.DataFrame(round_rows), clients, config
    )
    assert passed
    assert all(all(checks.values()) for checks in details.values())


def test_float32_mass_reduction_tolerance_is_local_and_detects_real_mismatch() -> None:
    values = np.asarray([0.13] * 25, dtype=np.float32)
    reconstructed = pd.Series([float(values.astype(np.float64).sum())])
    recorded_float32 = pd.Series([float(values.sum(dtype=np.float32))])

    assert audit._float32_reduction_close(reconstructed, recorded_float32, max_terms=25)
    assert not audit._float32_reduction_close(
        reconstructed,
        pd.Series([float(recorded_float32.iloc[0]) + 1.0e-4]),
        max_terms=25,
    )


def test_figure_aggregation_averages_contexts_within_seed_first() -> None:
    frame = pd.DataFrame(
        [
            {"seed": seed, "candidate": "method", "context": context, "metric": value}
            for seed, values in enumerate(
                ([1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0], [5.0, 7.0])
            )
            for context, value in enumerate(values)
        ]
    )
    seed_values = figures._seed_values(frame, value="metric", axes=["candidate"])
    assert seed_values.sort_values("seed")["value"].tolist() == pytest.approx(
        [2.0, 3.0, 4.0, 5.0, 6.0]
    )
    summary = figures._mean_sd(seed_values, axes=["candidate"])
    assert int(summary.loc[0, "n_seeds"]) == 5
    assert float(summary.loc[0, "mean"]) == pytest.approx(4.0)
    assert float(summary.loc[0, "sd"]) == pytest.approx(math.sqrt(2.5))
