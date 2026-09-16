#!/usr/bin/env python3
"""Analyze the matched masked DP-FedAvg screen (SC-FAR Step 1H)."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT / "configs/scpfar/paper1/s0_step1h_fmnist_masked_dpfedavg_eps10_screen.yaml"
)
DEFAULT_RESULTS = ROOT / "results/scfar_paper1_fmnist_step1h_masked_dpfedavg_v1"
DEFAULT_FULL_RESULTS = ROOT / "results/scfar_paper1_fmnist_step1f_dp_feasibility_v1"
DEFAULT_REPORT = ROOT / "output/analysis/SC_FAR_Step1H_Masked_DPFedAvg_Analysis.md"
DEFAULT_SELECTION = DEFAULT_RESULTS / "selection.json"


@dataclass(frozen=True)
class PrivateDimensionResult:
    mode: str
    privacy_id: str
    active_dimension: int
    final_test_accuracy: float
    final_client_accuracy: float
    final_variance_pct2: float
    final_worst20_accuracy: float
    final_gap_pct: float
    realized_epsilon: float | None
    noise_multiplier: float
    sensitivity: float
    noise_std: float
    median_noise_norm: float
    median_clean_aggregate_norm: float
    median_noise_to_clean_ratio: float
    median_user_clip_rate: float


def _finite_median(rounds: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rounds
        if row.get(key) is not None and float(row[key]) == float(row[key])
    ]
    return float(statistics.median(values)) if values else 0.0


def _float_or_zero(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _read_task(
    manifest_path: Path, *, default_dimension: int | None = None
) -> PrivateDimensionResult:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "complete_reused"}:
        raise RuntimeError(
            f"Task {manifest.get('task_id')} has status {manifest.get('status')}"
        )
    if manifest.get("compliance_issues"):
        raise RuntimeError(f"Task {manifest.get('task_id')} violates its protocol")
    metrics_paths = sorted(manifest_path.parent.glob("**/metrics.json"))
    if len(metrics_paths) != 1:
        raise RuntimeError(
            f"Expected one metrics.json below {manifest_path.parent}, found "
            f"{len(metrics_paths)}"
        )
    payload = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    expected = int(manifest["config"]["training"]["num_rounds"])
    if len(rounds) != expected:
        raise RuntimeError(
            f"Task {manifest.get('task_id')} has {len(rounds)}/{expected} rounds"
        )
    final = rounds[-1]
    axes = manifest["config"]["reproduction"]["matrix_axes"]
    mode = str(final.get("active_parameter_mode", "full"))
    active_dimension = final.get("scfar_active_parameter_dimension")
    if active_dimension is None:
        active_dimension = final.get("active_parameter_count", default_dimension)
    if active_dimension is None:
        raise RuntimeError(
            f"No active dimension recorded for {manifest.get('task_id')}"
        )
    realized_epsilon = final.get("privacy_epsilon")
    return PrivateDimensionResult(
        mode=mode,
        privacy_id=str(axes["privacy"]),
        active_dimension=int(active_dimension),
        final_test_accuracy=float(final["test_accuracy"]),
        final_client_accuracy=float(final["client_accuracy_mean"]),
        final_variance_pct2=float(final["client_accuracy_variance_pct2"]),
        final_worst20_accuracy=float(final["worst20_accuracy"]),
        final_gap_pct=float(final["best20_worst20_gap_pct"]),
        realized_epsilon=(
            float(realized_epsilon) if realized_epsilon is not None else None
        ),
        noise_multiplier=_float_or_zero(final.get("central_noise_multiplier")),
        sensitivity=_float_or_zero(final.get("scfar_sensitivity")),
        noise_std=_float_or_zero(final.get("central_noise_std")),
        median_noise_norm=_finite_median(rounds, "central_noise_norm"),
        median_clean_aggregate_norm=_finite_median(
            rounds, "scfar_clean_aggregate_norm"
        ),
        median_noise_to_clean_ratio=_finite_median(
            rounds, "central_noise_to_clean_aggregate_ratio"
        ),
        median_user_clip_rate=_finite_median(rounds, "scfar_user_clip_rate"),
    )


def _load_step1h(results_root: Path) -> list[PrivateDimensionResult]:
    manifests = sorted(results_root.glob("**/scfar_paper1_task_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No Step 1H manifests below {results_root}")
    return [_read_task(path) for path in manifests]


def _load_full_controls(results_root: Path) -> list[PrivateDimensionResult]:
    controls: list[PrivateDimensionResult] = []
    for path in sorted(results_root.glob("**/scfar_paper1_task_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        if int(config.get("training", {}).get("num_rounds", -1)) != 40:
            continue
        axes = config.get("reproduction", {}).get("matrix_axes", {})
        if axes.get("privacy") not in {"no_dp", "eps10"}:
            continue
        controls.append(_read_task(path, default_dimension=61706))
    if {row.privacy_id for row in controls} != {"no_dp", "eps10"}:
        raise RuntimeError("Could not resolve both full-update Step 1F controls")
    return controls


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _render(
    rows: list[PrivateDimensionResult],
    *,
    accuracy_gate: float,
    worst20_gate: float,
    ratio_gate: float,
    promoted_modes: list[str],
) -> str:
    lines = [
        "# SC-FAR-DP — Step 1H: masked DP-FedAvg feasibility",
        "",
        "## Decision",
        "",
    ]
    if promoted_modes:
        lines.append(
            "The following masks pass every preregistered epsilon=10 gate and "
            f"may proceed to lower budgets and multi-seed confirmation: "
            f"**{', '.join(promoted_modes)}**."
        )
    else:
        lines.append(
            "No public mask passes all epsilon=10 feasibility gates. The "
            "predeclared sequential rule therefore stops the lower-epsilon "
            "campaign. This does not refute central DP in general; it rejects "
            "this fixed C=1.4, 40-round, randomly initialized masked mechanism."
        )
    lines.extend(
        [
            "",
            "## Protocol and gates",
            "",
            "- Fashion-MNIST / LeNet-5; 25 clients; full participation; no client death.",
            "- 40 rounds; 2 local epochs; paired seeds (101, 28).",
            "- User-level central DP, replace-one adjacency, q=1 without amplification.",
            "- Public clipping threshold C=1.4 for every dimension; delta=1e-5.",
            f"- Epsilon=10 gates: Test Accuracy >= {_pct(accuracy_gate)}%, "
            f"Worst-20 >= {_pct(worst20_gate)}%, and median noise/clean <= "
            f"{ratio_gate:.1f}.",
            "- Full-update rows are reused from the already completed Step 1F, "
            "which has the same horizon, seeds, clipping and privacy accountant.",
            "",
            "## Results",
            "",
            "| Active mask | Privacy | d_active | Test Acc. (%) | Client Acc. (%) | Var (pp^2) | Worst-20 (%) | Gap (pp) | eps realized | sigma | Delta2 | noise std/coord | median ||Z|| | median ||A_clean|| | median noise/clean | median clip | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (-item.active_dimension, item.privacy_id != "no_dp"),
    ):
        passed = (
            row.privacy_id == "eps10"
            and row.final_test_accuracy >= accuracy_gate
            and row.final_worst20_accuracy >= worst20_gate
            and row.median_noise_to_clean_ratio <= ratio_gate
        )
        epsilon = (
            "infinity"
            if row.realized_epsilon is None
            else f"{row.realized_epsilon:.4f}"
        )
        lines.append(
            f"| {row.mode} | {row.privacy_id} | {row.active_dimension:,} | "
            f"{_pct(row.final_test_accuracy)} | {_pct(row.final_client_accuracy)} | "
            f"{row.final_variance_pct2:.2f} | {_pct(row.final_worst20_accuracy)} | "
            f"{row.final_gap_pct:.2f} | {epsilon} | {_fmt(row.noise_multiplier)} | "
            f"{_fmt(row.sensitivity)} | {_fmt(row.noise_std)} | "
            f"{_fmt(row.median_noise_norm)} | "
            f"{_fmt(row.median_clean_aggregate_norm)} | "
            f"{_fmt(row.median_noise_to_clean_ratio)} | "
            f"{100.0 * row.median_user_clip_rate:.1f}% | "
            f"{'pass' if passed else 'fail' if row.privacy_id == 'eps10' else 'control'} |"
        )
    lines.extend(
        [
            "",
            "## What the dimension screen identifies",
            "",
            "At fixed C and epsilon, the Gaussian standard deviation per active "
            "coordinate is unchanged; only the number of released coordinates "
            "changes. Therefore the typical Euclidean noise norm scales as "
            "sqrt(d_active). Improvement over the full control is evidence for "
            "a dimension effect. Failure despite a smaller dimension means that "
            "dimension reduction alone is insufficient under the frozen clipping "
            "and horizon—not that partial training can never work.",
            "",
            "## Interpretation boundary",
            "",
            "This one-seed development screen cannot establish a publication-level "
            "utility claim. Any passing cell must be frozen and repeated on the "
            "three independent seed pairs before SC-FAR reweighting is introduced.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    matrix_path: Path,
    results_root: Path,
    full_results_root: Path,
    report_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    gates = matrix["preregistration"]["feasibility_gate"]
    accuracy_gate = float(gates["final_test_accuracy_min"])
    worst20_gate = float(gates["final_worst20_accuracy_min"])
    ratio_gate = float(gates["median_noise_to_clean_aggregate_ratio_max"])
    step1h_rows = _load_step1h(results_root)
    expected = sum(
        int(experiment["expected_tasks"]) for experiment in matrix["experiments"]
    )
    if len(step1h_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} Step 1H tasks, found {len(step1h_rows)}"
        )
    rows = _load_full_controls(full_results_root) + step1h_rows
    promoted_modes = [
        row.mode
        for row in step1h_rows
        if row.privacy_id == "eps10"
        and row.final_test_accuracy >= accuracy_gate
        and row.final_worst20_accuracy >= worst20_gate
        and row.median_noise_to_clean_ratio <= ratio_gate
    ]
    payload = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "decision_rule": {
            "epsilon": 10.0,
            "final_test_accuracy_min": accuracy_gate,
            "final_worst20_accuracy_min": worst20_gate,
            "median_noise_to_clean_aggregate_ratio_max": ratio_gate,
        },
        "promoted_to_lower_epsilon_and_multiseed": promoted_modes,
        "rows": [row.__dict__ for row in rows],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render(
            rows,
            accuracy_gate=accuracy_gate,
            worst20_gate=worst20_gate,
            ratio_gate=ratio_gate,
            promoted_modes=promoted_modes,
        ),
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--full-results-root", type=Path, default=DEFAULT_FULL_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(
        matrix_path=args.matrix.resolve(),
        results_root=args.results_root.resolve(),
        full_results_root=args.full_results_root.resolve(),
        report_path=args.report.resolve(),
        selection_path=args.selection.resolve(),
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
