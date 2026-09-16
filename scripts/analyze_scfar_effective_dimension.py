#!/usr/bin/env python3
"""Analyze the public-mask utility screen for SC-FAR-DP development.

The script refuses incomplete or protocol-invalid tasks, applies the frozen
utility gates declared in the matrix, and writes both a machine-readable
selection and a concise Markdown audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT / "configs/scpfar/paper1/s0_step1g_fmnist_effective_dimension_nodp_screen.yaml"
)
DEFAULT_RESULTS = ROOT / "results/scfar_paper1_fmnist_step1g_effective_dimension_v1"
DEFAULT_REPORT = ROOT / "output/analysis/SC_FAR_Step1G_Effective_Dimension_Screen.md"
DEFAULT_SELECTION = DEFAULT_RESULTS / "selection.json"


@dataclass(frozen=True)
class ScreenResult:
    method: str
    mode: str
    active_count: int
    full_count: int
    final_test_accuracy: float
    final_client_accuracy: float
    final_variance_pct2: float
    final_worst20_accuracy: float
    final_gap_pct: float
    per_round_uplink_bytes: int
    rounds: int

    @property
    def active_fraction(self) -> float:
        return self.active_count / self.full_count


def _find_metrics(task_dir: Path) -> Path:
    paths = sorted(task_dir.glob("**/metrics.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one metrics.json below {task_dir}, found {len(paths)}"
        )
    return paths[0]


def _load_results(results_root: Path) -> list[ScreenResult]:
    manifests = sorted(results_root.glob("**/scfar_paper1_task_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No task manifest found below {results_root}")
    rows: list[ScreenResult] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"complete", "complete_reused"}:
            raise RuntimeError(
                f"Task {manifest.get('task_id')} has status {manifest.get('status')}"
            )
        if manifest.get("compliance_issues"):
            raise RuntimeError(
                f"Task {manifest.get('task_id')} has protocol compliance issues"
            )
        metrics_path = _find_metrics(manifest_path.parent)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rounds = metrics.get("rounds", [])
        expected_rounds = int(manifest["config"]["training"]["num_rounds"])
        if len(rounds) != expected_rounds:
            raise RuntimeError(
                f"Task {manifest.get('task_id')} has {len(rounds)}/{expected_rounds} rounds"
            )
        final = rounds[-1]
        axes = manifest["config"]["reproduction"]["matrix_axes"]
        rows.append(
            ScreenResult(
                method=str(axes["method"]),
                mode=str(final["active_parameter_mode"]),
                active_count=int(final["active_parameter_count"]),
                full_count=int(final["full_parameter_count"]),
                final_test_accuracy=float(final["test_accuracy"]),
                final_client_accuracy=float(final["client_accuracy_mean"]),
                final_variance_pct2=float(final["client_accuracy_variance_pct2"]),
                final_worst20_accuracy=float(final["worst20_accuracy"]),
                final_gap_pct=float(final["best20_worst20_gap_pct"]),
                per_round_uplink_bytes=int(final["total_bytes_sent"]),
                rounds=len(rounds),
            )
        )
    return sorted(rows, key=lambda row: row.active_count, reverse=True)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _render_report(
    rows: list[ScreenResult],
    *,
    accuracy_gate: float,
    worst20_gate: float,
    promoted_modes: list[str],
) -> str:
    lines = [
        "# SC-FAR-DP — Step 1G: effective private-dimension screen",
        "",
        "## Decision",
        "",
    ]
    if promoted_modes:
        lines.append(
            "The following non-full public masks pass both preregistered "
            f"no-DP utility gates and are promoted to DP-FedAvg: "
            f"**{', '.join(promoted_modes)}**."
        )
    else:
        lines.append(
            "No non-full public mask passes both preregistered no-DP utility "
            "gates. A private masked comparison is therefore not launched: "
            "the bottleneck is the non-private optimization ceiling, not yet "
            "the central-DP mechanism."
        )
    lines.extend(
        [
            "",
            "This is a development result on one paired seed, not a final paper result.",
            "",
            "## Frozen protocol",
            "",
            "- Fashion-MNIST / LeNet-5; client-Dirichlet balanced, beta = 0.1.",
            "- 25 clients, full participation, no dropout or dead client.",
            "- 40 rounds, 2 local epochs, no DP, paired seeds (101, 28).",
            "- The active mask is public and architecture-only. Frozen coordinates "
            "never depend on private data and are absent from the uploaded vector.",
            f"- Gates: Test Accuracy >= {_percent(accuracy_gate)}% and "
            f"Worst-20 >= {_percent(worst20_gate)}%.",
            "",
            "## Final-round results",
            "",
            "| Mask | Active dimension | Fraction of full | Test Acc. (%) | Client Acc. (%) | Var (pp^2) | Worst-20 (%) | Gap (pp) | Uplink/round | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        passed = (
            row.final_test_accuracy >= accuracy_gate
            and row.final_worst20_accuracy >= worst20_gate
        )
        lines.append(
            f"| {row.mode} | {row.active_count:,} | "
            f"{100.0 * row.active_fraction:.2f}% | "
            f"{_percent(row.final_test_accuracy)} | "
            f"{_percent(row.final_client_accuracy)} | "
            f"{row.final_variance_pct2:.2f} | "
            f"{_percent(row.final_worst20_accuracy)} | "
            f"{row.final_gap_pct:.2f} | {row.per_round_uplink_bytes:,} B | "
            f"{'pass' if passed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Passing the no-DP screen only shows that the mask has a usable "
            "optimization ceiling from the declared public initialization. It "
            "does not establish differential privacy, robustness, or a benefit "
            "of SC-FAR. Those questions belong to the subsequent matched "
            "DP-FedAvg and SC-FAR comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    matrix_path: Path,
    results_root: Path,
    report_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    gates = matrix["preregistration"]["utility_gate"]
    accuracy_gate = float(gates["final_test_accuracy_min"])
    worst20_gate = float(gates["final_worst20_accuracy_min"])
    rows = _load_results(results_root)
    if len(rows) != int(matrix["experiments"][0]["expected_tasks"]):
        raise RuntimeError(
            f"Expected {matrix['experiments'][0]['expected_tasks']} completed tasks, "
            f"found {len(rows)}"
        )
    promoted_modes = [
        row.mode
        for row in rows
        if row.mode != "full"
        and row.final_test_accuracy >= accuracy_gate
        and row.final_worst20_accuracy >= worst20_gate
    ]
    payload = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "decision_rule": {
            "final_test_accuracy_min": accuracy_gate,
            "final_worst20_accuracy_min": worst20_gate,
            "full_mask_is_control_only": True,
        },
        "promoted_modes": promoted_modes,
        "results": [row.__dict__ for row in rows],
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    report_path.write_text(
        _render_report(
            rows,
            accuracy_gate=accuracy_gate,
            worst20_gate=worst20_gate,
            promoted_modes=promoted_modes,
        ),
        encoding="utf-8",
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
