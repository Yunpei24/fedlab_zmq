from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.run_dt_ldp_far_stage16_oracle_margin_audit import (
    _required_support_fraction,
    run,
)

ROOT = Path(__file__).resolve().parents[1]


def test_required_support_fraction_matches_two_logit_inequality():
    import torch

    novelty = torch.tensor([0.2, 0.3, 0.8, 0.9], dtype=torch.float64)
    support = torch.tensor([0.8, 0.7, 0.3, 0.2], dtype=torch.float64)
    byzantine = torch.tensor([False, False, True, True])
    required, margin, novelty_gap = _required_support_fraction(
        novelty, support, byzantine
    )
    assert margin > 0.0
    assert novelty_gap > 0.0
    assert 0.0 < required < 1.0
    assert required == novelty_gap / (novelty_gap + margin)


def test_stage16_smoke_writes_oracle_and_profile_summaries(tmp_path):
    source = yaml.safe_load(
        (
            ROOT / "configs/dt_ldp_far/stage16_oracle_margin_two_logit_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    source["cohort"]["ambient_dimension"] = 16
    source["novelty"]["dimension"] = 16
    source["cohort"]["noise_permutations"] = ["identity"]
    source["cohort"]["outlier_geometries"] = ["aligned"]
    source["cohort"]["threats"] = ["ipm"]
    source["cohort"]["attack_severities"] = [1.0]
    source["robust_reference"]["candidates"] = ["fcc", "huber"]
    source["randomness"]["signal_seeds"] = [71]
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
    profiles = pd.read_csv(output / "profile_summary.csv")
    references = pd.read_csv(output / "reference_summary.csv")
    assert len(profiles) == 6
    assert len(references) == 2
    assert set(profiles["support_logit_fraction"]) == {0.0, 0.5, 0.75}
    assert (output / "oracle_detail.csv").exists()
