from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.run_dt_ldp_far_stage17_feasibility_map import run

ROOT = Path(__file__).resolve().parents[1]


def test_stage17_smoke_writes_phase_map_and_machine_decision(tmp_path):
    source = yaml.safe_load(
        (
            ROOT / "configs/dt_ldp_far/stage17_reference_snr_feasibility_map.yaml"
        ).read_text(encoding="utf-8")
    )
    source["grid"]["total_clients"] = [10]
    source["grid"]["effective_score_dimensions"] = [8]
    source["grid"]["upload_noise_stds"] = [0.006]
    source["grid"]["noise_permutations"] = ["identity"]
    source["grid"]["outlier_geometries"] = ["aligned"]
    source["randomness"]["signal_seeds"] = [211]
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
    summary = pd.read_csv(output / "cell_summary.csv")
    assert len(summary) == 1
    assert (output / "selection.json").exists()
    assert (output / "signal_detail.csv").exists()
    assert report.exists()
