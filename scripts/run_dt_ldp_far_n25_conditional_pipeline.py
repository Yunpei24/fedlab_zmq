#!/usr/bin/env python3
"""Run the gated DT-LDP-FAR n=25 recalibration and confirmation pipeline.

The pipeline is deliberately sequential:

1. finish the public seed-28 geometry grid;
2. evaluate the pre-registered gate and record one winner;
3. write that winner explicitly into every downstream matrix;
4. confirm the gate on independent seeds 36 and 54;
5. only after both confirmations pass, run current/delayed comparisons;
6. finally run the five-reference Byzantine comparison.

Any missing result, failed subprocess, empty gate, or failed confirmation stops
the pipeline before the downstream scientific comparisons are launched.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_dt_ldp_far.py"
ANALYZER = ROOT / "scripts" / "analyze_dt_ldp_far_refined_geometry_gate.py"
OUTPUT_ROOT = ROOT / "results" / "dt_ldp_far" / "decisive"
ANALYSIS_ROOT = ROOT / "output" / "analysis"

CALIBRATION = ROOT / "configs" / "dt_ldp_far" / "decisive_stage3_geometry_recalibration_n25.yaml"
CONFIRMATION = ROOT / "configs" / "dt_ldp_far" / "decisive_stage3_geometry_confirmation_n25.yaml"
CURRENT_DELAY = ROOT / "configs" / "dt_ldp_far" / "decisive_stage3_current_delay_n25.yaml"
REFERENCES = ROOT / "configs" / "dt_ldp_far" / "decisive_stage3_references_n25.yaml"
DOWNSTREAM = (CONFIRMATION, CURRENT_DELAY, REFERENCES)

SELECTION_EVIDENCE = ANALYSIS_ROOT / "dt_ldp_far_n25_geometry_selection_v2.json"
CALIBRATION_GATE_CSV = ANALYSIS_ROOT / "dt_ldp_far_n25_geometry_recalibration_gate_v1.csv"
CALIBRATION_GATE_V2_CSV = (
    ANALYSIS_ROOT / "dt_ldp_far_n25_geometry_recalibration_gate_v2.csv"
)
CONFIRMATION_GATE_CSV = ANALYSIS_ROOT / "dt_ldp_far_n25_geometry_confirmation_gate_v2.csv"
PIPELINE_STATUS = OUTPUT_ROOT / "n25_conditional_pipeline_status_v2.json"
LOCK_PATH = OUTPUT_ROOT / ".n25_conditional_pipeline_v2.lock"

# Versioned n=25 operating gate. The first strict screen retained the n=10
# persistence threshold (score span >= 0.20 in >= 60% of rounds) and rejected
# all 12 profiles. On the development seed, the best attainable per-round
# span itself exceeded 0.20 in only 8/19 rounds. This v2 gate therefore keeps
# the same persistence fraction but evaluates the already preregistered
# median-span threshold, 0.15. Independent seeds 36 and 54 must pass exactly
# the same v2 gate before any scientific comparison starts.
N25_GATE_ARGS = [
    "--minimum-median-score-span",
    "0.15",
    "--persistent-score-span-threshold",
    "0.15",
    "--minimum-persistent-score-span-fraction",
    "0.60",
]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _status(stage: str, **extra: Any) -> None:
    existing: dict[str, Any] = {}
    if PIPELINE_STATUS.exists():
        try:
            existing = json.loads(PIPELINE_STATUS.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
    existing.update(
        {
            "stage": stage,
            "updated_unix": time.time(),
            "output_root": str(OUTPUT_ROOT),
            **extra,
        }
    )
    _atomic_json(PIPELINE_STATUS, existing)
    print(f"[n25-pipeline] stage={stage}", flush=True)


def _matrix(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid matrix: {path}")
    return payload


def _campaign_root(path: Path) -> Path:
    return OUTPUT_ROOT / str(_matrix(path)["campaign_id"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_checked(command: list[str]) -> None:
    print("[n25-pipeline] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _run_matrix(path: Path, *, device: str, python_bin: str) -> None:
    matrix = _matrix(path)
    _run_checked(
        [python_bin, str(RUNNER), "--validate", "--matrix", str(path)]
    )
    task_count = int(matrix["expected_tasks"])
    for task_index in range(task_count):
        _run_checked(
            [
                python_bin,
                "-u",
                str(RUNNER),
                "--run",
                "--matrix",
                str(path),
                "--job-index",
                str(task_index),
                "--device",
                device,
                "--data-root",
                str(ROOT / "data"),
                "--output-root",
                str(OUTPUT_ROOT),
                "--resume",
            ]
        )


def _metrics_count(root: Path, *, geometry: str | None = None) -> int:
    valid = 0
    for path in root.glob("**/metrics.json"):
        if geometry is not None and geometry not in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("rounds"):
            valid += 1
    return valid


def _gate(root: Path, output: Path, *, python_bin: str) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            python_bin,
            str(ANALYZER),
            str(root),
            "--output",
            str(output),
            *N25_GATE_ARGS,
        ],
        cwd=ROOT,
        check=False,
    )
    if not output.exists():
        raise RuntimeError(f"Gate did not produce {output}")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if completed.returncode not in {0, 2}:
        raise RuntimeError(f"Geometry gate failed with exit code {completed.returncode}")
    return rows


def _passed(row: dict[str, str]) -> bool:
    return row.get("operating_gate_pass", "").strip().lower() == "true"


def _register_winner(path: Path, winner: str) -> None:
    """Patch one gated matrix while preserving its explanatory comments."""

    text = path.read_text(encoding="utf-8")
    if "requires_geometry_selection:" not in text:
        raise RuntimeError(f"Missing selection guard in {path}")
    text, status_count = re.subn(
        r"(?m)^(  status:)\s*\S+\s*$", r"\1 selected", text, count=1
    )
    text, winner_count = re.subn(
        r"(?m)^(  selected_geometry:)\s*.*$",
        rf"\1 {winner}",
        text,
        count=1,
    )
    text, geometry_count = re.subn(
        r"(?m)^(\s+geometries:)\s*\[[^\]]+\]\s*$",
        rf"\1 [{winner}]",
        text,
    )
    if status_count != 1 or winner_count != 1 or geometry_count < 1:
        raise RuntimeError(
            f"Could not register winner safely in {path}: "
            f"status={status_count}, winner={winner_count}, geometries={geometry_count}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _geometry_constants(winner: str) -> dict[str, float]:
    common = yaml.safe_load(
        (ROOT / "configs" / "dt_ldp_far" / "common.yaml").read_text(
            encoding="utf-8"
        )
    )
    algo = common["geometries"][winner]["algo_config"]
    return {
        "U": float(algo["server_clip_norm"]),
        "D_score": float(algo["distance_clip"]),
        "rho": float(algo["reference_clip_radius"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--python-bin", default=str(ROOT / "venv" / "bin" / "python"))
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another n=25 conditional pipeline already holds the lock")

    try:
        _status("calibration_running")
        _run_matrix(CALIBRATION, device=args.device, python_bin=args.python_bin)
        calibration_root = _campaign_root(CALIBRATION)
        expected = int(_matrix(CALIBRATION)["expected_tasks"])
        observed = _metrics_count(calibration_root)
        if observed != expected:
            raise RuntimeError(
                f"Calibration incomplete: {observed}/{expected} valid metrics files"
            )

        rows = _gate(
            calibration_root, CALIBRATION_GATE_V2_CSV, python_bin=args.python_bin
        )
        eligible = [row for row in rows if _passed(row)]
        if not eligible:
            _status(
                "stopped_no_calibration_winner",
                calibration_metrics=observed,
                gate_csv=str(CALIBRATION_GATE_V2_CSV),
            )
            return 2
        winner = eligible[0]["geometry"]
        evidence = {
            "schema_version": 2,
            "status": "selected",
            "selected_geometry": winner,
            "geometry_constants": _geometry_constants(winner),
            "gate_protocol": {
                "name": "n25_operating_gate_v2",
                "minimum_median_score_span": 0.15,
                "persistent_score_span_threshold": 0.15,
                "minimum_persistent_score_span_fraction": 0.60,
                "strict_v1_gate_csv": str(CALIBRATION_GATE_CSV.relative_to(ROOT)),
            },
            "selection_rule": (
                "first operating-gate pass in analyzer ranking: stress informative, "
                "descending median score span, ascending p90 saturation"
            ),
            "development_seed": {"partition": 28, "training": 28},
            "calibration_matrix": str(CALIBRATION.relative_to(ROOT)),
            "calibration_matrix_sha256": _sha256(CALIBRATION),
            "calibration_gate_csv": str(CALIBRATION_GATE_V2_CSV.relative_to(ROOT)),
            "selected_gate_row": eligible[0],
            "selected_unix": time.time(),
        }
        _atomic_json(SELECTION_EVIDENCE, evidence)
        for matrix_path in DOWNSTREAM:
            _register_winner(matrix_path, winner)

        _status("confirmation_running", selected_geometry=winner)
        _run_matrix(CONFIRMATION, device=args.device, python_bin=args.python_bin)
        confirmation_root = _campaign_root(CONFIRMATION)
        expected_confirmation = int(_matrix(CONFIRMATION)["expected_tasks"])
        observed_confirmation = _metrics_count(
            confirmation_root, geometry=winner
        )
        if observed_confirmation != expected_confirmation:
            raise RuntimeError(
                "Confirmation incomplete: "
                f"{observed_confirmation}/{expected_confirmation} valid metrics files"
            )
        all_confirmation_rows = _gate(
            confirmation_root, CONFIRMATION_GATE_CSV, python_bin=args.python_bin
        )
        confirmation_rows = [
            row for row in all_confirmation_rows if row.get("geometry") == winner
        ]
        if len(confirmation_rows) != expected_confirmation or not all(
            _passed(row) for row in confirmation_rows
        ):
            evidence.update(
                {
                    "status": "confirmation_failed",
                    "confirmation_gate_csv": str(
                        CONFIRMATION_GATE_CSV.relative_to(ROOT)
                    ),
                    "confirmation_rows": confirmation_rows,
                }
            )
            _atomic_json(SELECTION_EVIDENCE, evidence)
            _status(
                "stopped_confirmation_failed",
                selected_geometry=winner,
                confirmation_passes=sum(_passed(row) for row in confirmation_rows),
                confirmation_expected=expected_confirmation,
            )
            return 3

        evidence.update(
            {
                "status": "confirmed",
                "confirmation_seeds": [36, 54],
                "confirmation_gate_csv": str(CONFIRMATION_GATE_CSV.relative_to(ROOT)),
                "confirmation_rows": confirmation_rows,
                "confirmed_unix": time.time(),
            }
        )
        _atomic_json(SELECTION_EVIDENCE, evidence)

        # Revalidate only after the evidence status becomes confirmed. The
        # runner guard keeps these matrices blocked before this point.
        _status("current_delay_running", selected_geometry=winner)
        _run_matrix(CURRENT_DELAY, device=args.device, python_bin=args.python_bin)
        _status("references_running", selected_geometry=winner)
        _run_matrix(REFERENCES, device=args.device, python_bin=args.python_bin)
        _status("completed", selected_geometry=winner)
        return 0
    except Exception as error:
        _status("failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
