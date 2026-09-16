from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from scripts.run_gaussian_aware_reference_oracle import run

ROOT = Path(__file__).resolve().parents[1]


def test_gaussian_aware_reference_oracle_is_strictly_paired(tmp_path):
    source = yaml.safe_load(
        (
            ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_oracle.yaml"
        ).read_text(encoding="utf-8")
    )
    source["cohort"].update(
        {
            "num_clients": 10,
            "num_byzantine": 2,
            "dimension": 8,
            "block_sizes": [4, 4],
            "heterogeneity_std_by_block": [0.012, 0.025],
        }
    )
    source["cohort"]["honest_outliers"]["count"] = 2
    source["privacy_noise"]["block_std_multipliers"] = [0.8, 1.2]
    source["privacy_noise"]["regimes"][1]["permutations"] = ["identity"]
    source["references"]["f_sigma_huber"]["influence_cap"] = [0.15, 0.15]
    source["references"]["f_sigma_huber"]["num_steps"] = 3
    source["references"]["rfa"]["max_iter"] = 8
    source["threats"]["names"] = ["none", "ipm"]
    source["threats"]["severities"] = [1.0]
    source["threats"]["separated_for_gates"] = ["ipm"]
    source["threats"]["evasive_controls"] = []
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    output = tmp_path / "results"
    report = tmp_path / "report.md"
    decision = run(
        config,
        output,
        report,
        calibration_draws_override=3,
        draws_per_seed_override=1,
        seeds_override=[28],
    )

    with (output / "paired_detail.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = source["references"]["candidates"]
    counts = Counter(row["pairing_id"] for row in rows)
    assert counts
    assert set(counts.values()) == {len(methods)}
    assert decision["paired_rows"] == len(rows)
    assert {row["reference"] for row in rows} == set(methods)

    uniform_ipm = next(
        row
        for row in rows
        if row["reference"] == "uniform_mean" and row["threat"] == "ipm"
    )
    assert abs(float(uniform_ipm["byzantine_weight_mass"]) - 0.2) < 1e-12
    assert (output / "null_calibration.csv").exists()
    assert (output / "summary.csv").exists()
    decision_payload = json.loads((output / "decision.json").read_text())
    assert decision_payload["accuracy_used_for_selection"] is False
    assert decision_payload["requested_device"] == "cpu"
    assert decision_payload["resolved_device"] == "cpu"
    assert decision_payload["tensor_dtype"] == "float64"
    assert decision_payload["silent_cpu_fallback_allowed"] is False
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert resolved["execution"] == {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "tensor_dtype": "float64",
        "silent_cpu_fallback_allowed": False,
    }
    assert report.exists()
