from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml

from robustness.aggregators import guarded_aggregate
from scripts.run_dt_ldp_far_stage14b_reference_trust_audit import run

ROOT = Path(__file__).resolve().parents[1]


def test_guarded_aggregate_stays_in_reference_ball():
    reference = torch.tensor([0.5, -0.5], dtype=torch.float64)
    raw = torch.tensor([4.0, 2.0], dtype=torch.float64)
    guarded = guarded_aggregate(raw, reference, radius=0.3)
    assert torch.linalg.vector_norm(guarded - reference) <= 0.3 + 1e-12


def test_stage14b_smoke_writes_all_artifacts(tmp_path):
    source = yaml.safe_load(
        (ROOT / "configs/dt_ldp_far/stage14b_reference_trust_guard.yaml").read_text()
    )
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
    candidates = pd.read_csv(output / "candidate_summary.csv")
    guards = pd.read_csv(output / "guard_summary.csv")
    assert len(candidates) == 8
    assert len(guards) == 40
    assert set(candidates["trust_profile"]) == {
        "none",
        "directional_independence",
    }
