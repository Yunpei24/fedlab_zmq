#!/usr/bin/env python3
"""Analyze Step 1I and freeze one clipping threshold per active mask."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT / "configs/scpfar/paper1/s0_step1i_fmnist_mask_clip_calibration.yaml"
)
DEFAULT_RESULTS = ROOT / "results/scfar_paper1_fmnist_step1i_mask_clip_calibration_v1"
DEFAULT_REPORT = ROOT / "output/analysis/SC_FAR_Step1I_Mask_Clip_Calibration.md"
DEFAULT_SELECTION = DEFAULT_RESULTS / "selection.json"


@dataclass(frozen=True)
class CalibrationResult:
    mode: str
    user_clip_norm: float
    active_dimension: int
    final_test_accuracy: float
    final_client_accuracy: float
    final_variance_pct2: float
    final_worst20_accuracy: float
    final_gap_pct: float
    median_user_clip_rate: float
    median_clean_aggregate_norm: float
    median_preclip_norm_p50: float
    passed: bool


def _median(rounds: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rounds
        if row.get(key) is not None and float(row[key]) == float(row[key])
    ]
    if not values:
        raise RuntimeError(f"Missing required round metric {key!r}")
    return float(statistics.median(values))


def _load_rows(
    results_root: Path,
    *,
    test_min: float,
    worst20_min: float,
    clip_min: float,
    clip_max: float,
) -> list[CalibrationResult]:
    manifests = sorted(results_root.glob("**/scfar_paper1_task_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No task manifests below {results_root}")
    rows: list[CalibrationResult] = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"complete", "complete_reused"}:
            raise RuntimeError(
                f"Task {manifest.get('task_id')} has status {manifest.get('status')}"
            )
        if manifest.get("compliance_issues"):
            raise RuntimeError(f"Task {manifest.get('task_id')} violates its protocol")
        metrics_paths = sorted(path.parent.glob("**/metrics.json"))
        if len(metrics_paths) != 1:
            raise RuntimeError(
                f"Expected one metrics.json below {path.parent}, found {len(metrics_paths)}"
            )
        payload = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
        rounds = payload.get("rounds", [])
        expected = int(manifest["config"]["training"]["num_rounds"])
        if len(rounds) != expected:
            raise RuntimeError(
                f"Task {manifest.get('task_id')} has {len(rounds)}/{expected} rounds"
            )
        final = rounds[-1]
        clip_rate = _median(rounds, "scfar_user_clip_rate")
        row = CalibrationResult(
            mode=str(final["active_parameter_mode"]),
            user_clip_norm=float(
                manifest["config"]["training"]["algo_config"]["user_clip_norm"]
            ),
            active_dimension=int(final["scfar_active_parameter_dimension"]),
            final_test_accuracy=float(final["test_accuracy"]),
            final_client_accuracy=float(final["client_accuracy_mean"]),
            final_variance_pct2=float(final["client_accuracy_variance_pct2"]),
            final_worst20_accuracy=float(final["worst20_accuracy"]),
            final_gap_pct=float(final["best20_worst20_gap_pct"]),
            median_user_clip_rate=clip_rate,
            median_clean_aggregate_norm=_median(
                rounds, "scfar_clean_aggregate_norm"
            ),
            median_preclip_norm_p50=_median(rounds, "scfar_preclip_norm_p50"),
            passed=(
                float(final["test_accuracy"]) >= test_min
                and float(final["worst20_accuracy"]) >= worst20_min
                and clip_min < clip_rate < clip_max
            ),
        )
        rows.append(row)
    return rows


def _render(
    rows: list[CalibrationResult],
    *,
    selected: dict[str, float],
    refinement_candidates: dict[str, float],
    test_min: float,
    worst20_min: float,
    clip_min: float,
    clip_max: float,
) -> str:
    lines = [
        "# SC-FAR-DP — Step 1I: joint mask/clipping calibration",
        "",
        "## Frozen decision rule",
        "",
        "The clipping grid was declared before execution. For each public mask, "
        "the selected value is the **smallest** C satisfying all three gates:",
        "",
        f"- final Test Accuracy >= {100 * test_min:.1f}%;",
        f"- final Worst-20 Accuracy >= {100 * worst20_min:.1f}%;",
        f"- {100 * clip_min:.1f}% < median client-update clipping rate < "
        f"{100 * clip_max:.1f}%.",
        "",
        "This is a development calibration, not a claim that hyperparameter "
        "selection itself is differentially private.",
        "",
        "## Results",
        "",
        "| Active mask | d_active | C | Test Acc. (%) | Client Acc. (%) | Var (pp^2) | Worst-20 (%) | Gap (pp) | Median clip (%) | Median preclip p50 norm | Median clean aggregate norm | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(rows, key=lambda item: (item.mode, item.user_clip_norm)):
        lines.append(
            f"| {row.mode} | {row.active_dimension:,} | {row.user_clip_norm:.2f} | "
            f"{100 * row.final_test_accuracy:.2f} | "
            f"{100 * row.final_client_accuracy:.2f} | "
            f"{row.final_variance_pct2:.2f} | "
            f"{100 * row.final_worst20_accuracy:.2f} | "
            f"{row.final_gap_pct:.2f} | "
            f"{100 * row.median_user_clip_rate:.1f} | "
            f"{row.median_preclip_norm_p50:.3f} | "
            f"{row.median_clean_aggregate_norm:.3f} | "
            f"{'pass' if row.passed else 'fail'} |"
        )
    lines.extend(["", "## Selection", ""])
    if selected:
        for mode, value in sorted(selected.items()):
            lines.append(f"- `{mode}`: C={value:g}.")
    else:
        lines.append("No mask has a valid clipping threshold in the frozen grid.")
    if refinement_candidates:
        lines.extend(
            [
                "",
                "## Deterministic refinement candidates",
                "",
                "For every unresolved mask, the largest essentially-unclipped "
                "probe arm is used only to observe the pre-clipping update-norm "
                "distribution. The refinement candidate is fixed as",
                "",
                "`C_refine = median over rounds of the per-round median "
                "pre-clipping client-update norm`.",
                "",
                "This targets a non-degenerate clipping rate without selecting "
                "C from private-model accuracy.",
                "",
            ]
        )
        for mode, value in sorted(refinement_candidates.items()):
            lines.append(f"- `{mode}`: C_refine={value:g}.")
    lines.extend(
        [
            "",
            "A selected pair `(mask, C)` may enter the matched horizon screen "
            "at T in {5, 10, 20}. A rejected mask is not rescued by inspecting "
            "private accuracy at another epsilon.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *, matrix_path: Path, results_root: Path, report_path: Path, selection_path: Path
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    gates = matrix["preregistration"]["feasibility_gate"]
    test_min = float(gates["final_test_accuracy_min"])
    worst20_min = float(gates["final_worst20_accuracy_min"])
    clip_min = float(gates["median_user_clip_rate_min_exclusive"])
    clip_max = float(gates["median_user_clip_rate_max_exclusive"])
    rows = _load_rows(
        results_root,
        test_min=test_min,
        worst20_min=worst20_min,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    expected = sum(int(item["expected_tasks"]) for item in matrix["experiments"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} results, found {len(rows)}")
    modes = sorted({row.mode for row in rows})
    selected: dict[str, float] = {}
    refinement_candidates: dict[str, float] = {}
    for mode in modes:
        passing = sorted(
            row.user_clip_norm for row in rows if row.mode == mode and row.passed
        )
        if passing:
            selected[mode] = passing[0]
            continue
        probes = sorted(
            (
                row
                for row in rows
                if row.mode == mode and row.median_user_clip_rate <= clip_min
            ),
            key=lambda row: row.user_clip_norm,
        )
        if probes:
            refinement_candidates[mode] = round(
                probes[-1].median_preclip_norm_p50, 6
            )
    payload = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "decision_rule": {
            "final_test_accuracy_min": test_min,
            "final_worst20_accuracy_min": worst20_min,
            "median_user_clip_rate_min_exclusive": clip_min,
            "median_user_clip_rate_max_exclusive": clip_max,
            "selection": "smallest_C_passing_all_gates_per_mask",
        },
        "selected_clip_norm_by_mode": selected,
        "refinement_candidate_by_mode": refinement_candidates,
        "excluded_modes": [
            mode
            for mode in modes
            if mode not in selected and mode not in refinement_candidates
        ],
        "rows": [asdict(row) for row in rows],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render(
            rows,
            selected=selected,
            refinement_candidates=refinement_candidates,
            test_min=test_min,
            worst20_min=worst20_min,
            clip_min=clip_min,
            clip_max=clip_max,
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(
        matrix_path=args.matrix.resolve(),
        results_root=args.results_root.resolve(),
        report_path=args.report.resolve(),
        selection_path=args.selection.resolve(),
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
