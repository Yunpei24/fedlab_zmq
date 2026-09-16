from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_dt_ldp_far_stage18_conditional_byzantine_holdout import run

ROOT = Path(__file__).resolve().parents[1]


def test_stage18_smoke_writes_conditional_decision(tmp_path, monkeypatch):
    source = yaml.safe_load(
        (
            ROOT / "configs/dt_ldp_far/stage18_conditional_byzantine_holdout.yaml"
        ).read_text(encoding="utf-8")
    )
    source["cohort"]["num_clients"] = 10
    source["cohort"]["num_byzantine"] = 2
    source["cohort"]["effective_score_dimension"] = 8
    source["cohort"]["noise_permutations"] = ["identity"]
    source["cohort"]["outlier_geometries"] = ["aligned"]
    source["cohort"]["minority_structures"] = [
        source["cohort"]["minority_structures"][2]
    ]
    source["robust_reference"]["candidates"] = ["fcc"]
    source["threats"]["separated"] = ["ipm"]
    source["threats"]["evasive_controls"] = ["alie"]
    source["threats"]["attack_severities"] = [1.0]
    source["randomness"]["signal_seeds"] = [307]
    source["decision"]["primary_structure"] = "coherent_f_plus_2_primary"
    source["stage17_selection"]["required_total_clients"] = 10
    source["stage17_selection"]["required_effective_score_dimension"] = 8
    selection_dir = tmp_path / "stage17"
    selection_dir.mkdir()
    (selection_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_cell": {
                    "total_clients": 10,
                    "effective_score_dimension": 8,
                    "upload_noise_std": 0.012,
                }
            }
        ),
        encoding="utf-8",
    )
    source["stage17_selection"]["path"] = "stage17/selection.json"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_dt_ldp_far_stage18_conditional_byzantine_holdout.ROOT",
        tmp_path,
    )
    output = tmp_path / "results"
    report = tmp_path / "report.md"
    primary = run(
        config,
        output,
        report,
        calibration_draws_override=20,
        null_holdout_draws_override=20,
        signal_draws_override=1,
    )
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 3
    assert primary["minority_structure"] == "coherent_f_plus_2_primary"
    assert (output / "decision.json").exists()
    assert report.exists()
