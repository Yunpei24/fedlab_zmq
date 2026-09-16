#!/usr/bin/env python3
"""Analyze the matched Step 1J mask/clipping/horizon screen."""

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
    ROOT / "configs/scpfar/paper1/s0_step1j_fmnist_joint_mask_clip_horizon.yaml"
)
DEFAULT_RESULTS = ROOT / "results/scfar_paper1_fmnist_step1j_joint_horizon_v1"
DEFAULT_REPORT = ROOT / "output/analysis/SC_FAR_Step1J_Joint_Horizon_Screen.md"
DEFAULT_SELECTION = DEFAULT_RESULTS / "selection.json"


@dataclass(frozen=True)
class HorizonResult:
    mode: str
    user_clip_norm: float
    active_dimension: int
    rounds: int
    privacy_id: str
    final_test_accuracy: float
    final_client_accuracy: float
    final_variance_pct2: float
    final_worst20_accuracy: float
    final_gap_pct: float
    median_user_clip_rate: float
    median_clean_aggregate_norm: float
    median_noise_norm: float
    median_noise_to_clean_ratio: float
    sensitivity: float
    noise_multiplier: float
    noise_std: float
    realized_epsilon: float | None


def _median(rounds: list[dict[str, Any]], key: str, *, default: float = 0.0) -> float:
    values = [
        float(row[key])
        for row in rounds
        if row.get(key) is not None and float(row[key]) == float(row[key])
    ]
    return float(statistics.median(values)) if values else default


def _load_rows(results_root: Path) -> list[HorizonResult]:
    manifests = sorted(results_root.glob("**/scfar_paper1_task_manifest.json"))
    if not manifests:
        raise RuntimeError(f"No task manifests below {results_root}")
    rows: list[HorizonResult] = []
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
        axes = manifest["config"]["reproduction"]["matrix_axes"]
        realized_epsilon = final.get("privacy_epsilon")
        rows.append(
            HorizonResult(
                mode=str(final["active_parameter_mode"]),
                user_clip_norm=float(final["scfar_user_clip_norm"]),
                active_dimension=int(final["scfar_active_parameter_dimension"]),
                rounds=expected,
                privacy_id=str(axes["privacy"]),
                final_test_accuracy=float(final["test_accuracy"]),
                final_client_accuracy=float(final["client_accuracy_mean"]),
                final_variance_pct2=float(final["client_accuracy_variance_pct2"]),
                final_worst20_accuracy=float(final["worst20_accuracy"]),
                final_gap_pct=float(final["best20_worst20_gap_pct"]),
                median_user_clip_rate=_median(rounds, "scfar_user_clip_rate"),
                median_clean_aggregate_norm=_median(
                    rounds, "scfar_clean_aggregate_norm"
                ),
                median_noise_norm=_median(rounds, "central_noise_norm"),
                median_noise_to_clean_ratio=_median(
                    rounds, "central_noise_to_clean_aggregate_ratio"
                ),
                sensitivity=float(final.get("scfar_sensitivity") or 0.0),
                noise_multiplier=float(final.get("central_noise_multiplier") or 0.0),
                noise_std=float(final.get("central_noise_std") or 0.0),
                realized_epsilon=(
                    float(realized_epsilon) if realized_epsilon is not None else None
                ),
            )
        )
    return rows


def _render(
    rows: list[HorizonResult],
    *,
    selected: dict[str, int],
    private_gates: dict[str, float],
    control_gates: dict[str, float],
) -> str:
    lines = [
        "# SC-FAR-DP — Step 1J: joint mask, clipping and horizon screen",
        "",
        "## Decision",
        "",
    ]
    if selected:
        for mode, rounds in sorted(selected.items()):
            lines.append(f"- `{mode}` passes first at T={rounds} releases.")
    else:
        lines.append(
            "No calibrated reduced-dimensional mask passes the matched no-DP "
            "and epsilon=10 gates at T in {5, 10, 20}."
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Mask | d_active | C | T | Privacy | Test Acc. (%) | Client Acc. (%) | Var (pp^2) | Worst-20 (%) | Gap (pp) | Median clip (%) | Median clean norm | Median noise norm | Median noise/clean | Delta2 | sigma | noise std/coord | eps realized |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item.mode, item.rounds, item.privacy_id)):
        epsilon = "infinity" if row.realized_epsilon is None else f"{row.realized_epsilon:.4f}"
        lines.append(
            f"| {row.mode} | {row.active_dimension:,} | {row.user_clip_norm:.6g} | "
            f"{row.rounds} | {row.privacy_id} | {100 * row.final_test_accuracy:.2f} | "
            f"{100 * row.final_client_accuracy:.2f} | {row.final_variance_pct2:.2f} | "
            f"{100 * row.final_worst20_accuracy:.2f} | {row.final_gap_pct:.2f} | "
            f"{100 * row.median_user_clip_rate:.1f} | "
            f"{row.median_clean_aggregate_norm:.3f} | {row.median_noise_norm:.3f} | "
            f"{row.median_noise_to_clean_ratio:.3f} | {row.sensitivity:.4f} | "
            f"{row.noise_multiplier:.4f} | {row.noise_std:.4f} | {epsilon} |"
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
            f"Matched no-DP: Test >= {100 * control_gates['final_test_accuracy_min']:.1f}%, "
            f"Worst-20 >= {100 * control_gates['final_worst20_accuracy_min']:.1f}%, "
            f"and {100 * control_gates['median_user_clip_rate_min_exclusive']:.1f}% "
            f"< median clip < {100 * control_gates['median_user_clip_rate_max_exclusive']:.1f}%.",
            "",
            f"Epsilon=10: Test >= {100 * private_gates['final_test_accuracy_min']:.1f}%, "
            f"Worst-20 >= {100 * private_gates['final_worst20_accuracy_min']:.1f}%, "
            f"and median noise/clean <= "
            f"{private_gates['median_noise_to_clean_aggregate_ratio_max']:.1f}.",
            "",
            "The smallest passing T is selected because it minimizes the number "
            "of composed releases while retaining the preregistered utility floor.",
            "",
            "This remains a development screen. A passing configuration must be "
            "frozen before three-seed confirmation and before introducing SC-FAR tilting.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *, matrix_path: Path, results_root: Path, report_path: Path, selection_path: Path
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    private_gates = {
        key: float(value)
        for key, value in matrix["preregistration"]["private_feasibility_gate"].items()
    }
    control_gates = {
        key: float(value)
        for key, value in matrix["preregistration"]["matched_control_gate"].items()
    }
    rows = _load_rows(results_root)
    expected = (
        sum(int(item["expected_tasks"]) for item in matrix["experiments"])
        * len(matrix["preregistration"]["horizon_rounds"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} results, found {len(rows)}")

    selected: dict[str, int] = {}
    for mode in sorted({row.mode for row in rows}):
        for rounds in sorted({row.rounds for row in rows if row.mode == mode}):
            pair = {
                row.privacy_id: row
                for row in rows
                if row.mode == mode and row.rounds == rounds
            }
            if set(pair) != {"no_dp", "eps10"}:
                raise RuntimeError(f"Incomplete matched pair for {mode}, T={rounds}")
            control = pair["no_dp"]
            private = pair["eps10"]
            control_pass = (
                control.final_test_accuracy
                >= control_gates["final_test_accuracy_min"]
                and control.final_worst20_accuracy
                >= control_gates["final_worst20_accuracy_min"]
                and control_gates["median_user_clip_rate_min_exclusive"]
                < control.median_user_clip_rate
                < control_gates["median_user_clip_rate_max_exclusive"]
            )
            private_pass = (
                private.final_test_accuracy
                >= private_gates["final_test_accuracy_min"]
                and private.final_worst20_accuracy
                >= private_gates["final_worst20_accuracy_min"]
                and private.median_noise_to_clean_ratio
                <= private_gates["median_noise_to_clean_aggregate_ratio_max"]
                and private.realized_epsilon is not None
                and private.realized_epsilon <= 10.0001
            )
            if control_pass and private_pass:
                selected[mode] = rounds
                break

    payload = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "selected_horizon_by_mode": selected,
        "promoted_modes": sorted(selected),
        "excluded_modes": sorted({row.mode for row in rows}.difference(selected)),
        "private_feasibility_gate": private_gates,
        "matched_control_gate": control_gates,
        "rows": [asdict(row) for row in rows],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render(
            rows,
            selected=selected,
            private_gates=private_gates,
            control_gates=control_gates,
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
