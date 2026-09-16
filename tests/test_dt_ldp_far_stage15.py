from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.run_dt_ldp_far_stage15_crossfit_peer_support_audit import run

ROOT = Path(__file__).resolve().parents[1]


def test_stage15_smoke_writes_two_profiles_and_separate_reference_gates(tmp_path):
    source = yaml.safe_load(
        (
            ROOT / "configs/dt_ldp_far/stage15_crossfit_peer_support_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    source["cohort"]["ambient_dimension"] = 16
    source["novelty"]["dimension"] = 16
    source["cohort"]["noise_permutations"] = ["identity"]
    source["cohort"]["outlier_geometries"] = ["aligned"]
    source["cohort"]["threats"] = ["none", "ipm"]
    source["robust_reference"]["candidates"] = ["fcc", "huber"]
    source["randomness"]["signal_seeds"] = [28]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    output = tmp_path / "results"
    report = tmp_path / "report.md"
    run(
        config,
        output,
        report,
        calibration_draws_override=20,
        null_holdout_draws_override=20,
        signal_draws_override=1,
    )
    assert report.exists()
    assert (output / "summary.json").exists()
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 4
    assert set(summary["profile"]) == {
        "crossfit_covariance",
        "crossfit_covariance_plus_peer_support",
    }
    assert "passes_score_gates" in summary
    assert "passes_full_gates" in summary
