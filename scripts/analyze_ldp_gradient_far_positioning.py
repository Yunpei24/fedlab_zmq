#!/usr/bin/env python3
"""Descriptive, reproducible analysis of the LDP-gradient-FAR positioning campaign.

This script deliberately does *not* evaluate or write campaign gates.  It
inventories every registered task, separates missing, invalid and complete
runs, and writes run-level, variant-level and phase-level summaries.  Gate
promotion remains an explicit scientific decision made outside this tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ldp_gradient_far_positioning import (  # noqa: E402
    DEFAULT_MATRIX,
    Campaign,
    PositioningTask,
    load_campaign,
    resolved_config,
    task_output_dir,
)


DEFAULT_OUTPUT_DIR = ROOT / "output" / "analysis" / "ldp_gradient_far_positioning_v1"
DEFAULT_REPORT = ROOT / "output" / "analysis" / "LDP_Gradient_FAR_Positioning_Descriptive_Analysis.md"


# Values taken from the final available evaluated round.
FINAL_METRICS: dict[str, tuple[str, float]] = {
    "test_accuracy_final_pct": ("test_accuracy", 100.0),
    "client_accuracy_final_pct": ("client_accuracy_mean", 100.0),
    "variance_final_pp2": ("client_accuracy_variance_pct2", 1.0),
    "worst20_final_pct": ("worst20_accuracy_pct", 1.0),
    "gap_final_pp": ("best20_worst20_gap_pct", 1.0),
    "test_loss_final": ("test_loss", 1.0),
    "train_loss_final": ("train_loss", 1.0),
    "privacy_epsilon_final": ("privacy_epsilon_max", 1.0),
    "privacy_delta_final": ("privacy_delta", 1.0),
    "privacy_noise_multiplier_min_final": (
        "privacy_model_noise_multiplier_min",
        1.0,
    ),
    "privacy_noise_multiplier_mean_final": (
        "privacy_model_noise_multiplier_mean",
        1.0,
    ),
    "privacy_noise_multiplier_max_final": (
        "privacy_model_noise_multiplier_max",
        1.0,
    ),
    "privacy_local_clip_rate_final": ("privacy_clip_rate_mean", 1.0),
    "server_clip_rate_final": ("far_server_clip_rate", 1.0),
    "far_score_span_final": ("far_score_span", 1.0),
    "far_logit_range_final": ("far_logit_range", 1.0),
    "far_max_weight_final": ("max_client_weight", 1.0),
    "far_weight_entropy_final": ("weight_entropy", 1.0),
    "far_effective_clients_final": ("effective_num_clients", 1.0),
    "far_weight_concentration_final": (
        "far_noise_amplification_vs_uniform",
        1.0,
    ),
}


# Dynamic mechanism diagnostics are summarized by their median over rounds.
MEDIAN_METRICS: dict[str, str] = {
    "privacy_local_clip_rate_median": "privacy_clip_rate_mean",
    "server_clip_rate_median": "far_server_clip_rate",
    "far_score_span_median": "far_score_span",
    "far_logit_range_median": "far_logit_range",
    "far_max_weight_median": "max_client_weight",
    "far_weight_entropy_median": "weight_entropy",
    "far_effective_clients_median": "effective_num_clients",
    "far_weight_concentration_median": "far_noise_amplification_vs_uniform",
    "byzantine_weight_mass_median_oracle": "byzantine_weight_mass_oracle",
    "reference_honest_center_error_median_oracle": (
        "far_reference_honest_center_error_oracle"
    ),
    "weight_dp_noise_corr_median_oracle": "far_weight_dp_noise_corr_oracle",
    "distance_dp_noise_corr_median_oracle": "far_distance_dp_noise_corr_oracle",
    "weight_effective_noise_corr_median_oracle": (
        "far_weight_effective_noise_corr_oracle"
    ),
    "noisy_clean_score_corr_median_oracle": (
        "far_noisy_clean_score_corr_oracle"
    ),
    "honest_noisy_clean_score_corr_median_oracle": (
        "far_honest_noisy_clean_score_corr_oracle"
    ),
    "honest_clean_top_tail_recall_median_oracle": (
        "far_honest_clean_top_tail_recall_oracle"
    ),
    "fixed_weight_fresh_noise_sq_error_median_oracle": (
        "far_fixed_weight_fresh_noise_sq_error_oracle"
    ),
    "total_fresh_noise_sq_error_median_oracle": (
        "far_total_fresh_noise_sq_error_oracle"
    ),
    "reweighting_component_sq_norm_median_oracle": (
        "far_reweighting_component_sq_norm_oracle"
    ),
    "honest_weight_mass_median_oracle": "far_honest_weight_mass_oracle",
    "honest_weight_l1_from_uniform_median_oracle": (
        "far_honest_conditioned_weight_l1_from_uniform_oracle"
    ),
    "honest_clean_tilting_bias_norm_median_oracle": (
        "far_honest_clean_tilting_bias_norm_oracle"
    ),
    "honest_fixed_weight_dp_noise_norm_median_oracle": (
        "far_honest_fixed_weight_dp_noise_norm_oracle"
    ),
    "byzantine_displacement_norm_median_oracle": (
        "far_byzantine_displacement_norm_oracle"
    ),
    "aggregate_error_to_clean_honest_center_median_oracle": (
        "far_aggregate_error_to_clean_honest_center_norm_oracle"
    ),
    "error_decomposition_residual_norm_median_oracle": (
        "far_error_decomposition_residual_norm_oracle"
    ),
}


RUN_NUMERIC_METRICS = (
    "test_accuracy_auc_pct",
    "client_accuracy_auc_pct",
    "test_accuracy_best_pct",
    "test_accuracy_max_drawdown_pp",
    "round_to_best_test_accuracy",
    *FINAL_METRICS.keys(),
    *MEDIAN_METRICS.keys(),
)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_equals(value: Any, expected: int) -> bool:
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError, OverflowError):
        return False


def _float_equals(value: Any, expected: float) -> bool:
    observed = _finite_float(value)
    return observed is not None and math.isclose(
        observed, float(expected), abs_tol=1e-12
    )


def _series(
    rounds: Sequence[Mapping[str, Any]], key: str, *, scale: float = 1.0
) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rounds, start=1):
        value = _finite_float(row.get(key))
        round_num = _finite_float(row.get("round_num", row.get("round", index)))
        if value is not None and round_num is not None:
            values.append((int(round_num), value * scale))
    values.sort(key=lambda pair: pair[0])
    return values


def _last_value(
    rounds: Sequence[Mapping[str, Any]], key: str, *, scale: float = 1.0
) -> float | None:
    values = _series(rounds, key, scale=scale)
    return values[-1][1] if values else None


def _median_value(
    rounds: Sequence[Mapping[str, Any]], key: str, *, scale: float = 1.0
) -> float | None:
    values = [value for _, value in _series(rounds, key, scale=scale)]
    return statistics.median(values) if values else None


def _normalized_auc(values: Sequence[tuple[int, float]]) -> float | None:
    """Return trapezoidal AUC divided by the observed round interval.

    The result has the same unit as the original trajectory (percentage points
    for accuracy).  A one-point trajectory returns that point rather than an
    undefined integral.
    """

    if not values:
        return None
    if len(values) == 1:
        return values[0][1]
    span = values[-1][0] - values[0][0]
    if span <= 0:
        return statistics.fmean(value for _, value in values)
    area = sum(
        (right_round - left_round) * (left_value + right_value) / 2.0
        for (left_round, left_value), (right_round, right_value) in zip(
            values, values[1:]
        )
    )
    return area / span


def _max_drawdown(values: Sequence[tuple[int, float]]) -> float | None:
    if not values:
        return None
    peak = values[0][1]
    drawdown = 0.0
    for _, value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return drawdown


def _noise_profile(config: Mapping[str, Any]) -> tuple[str, float, float]:
    scales = config.get("privacy_noise_multiplier_scale_by_client")
    if not scales:
        return "homogeneous", 1.0, 1.0
    finite = [_finite_float(value) for value in scales]
    values = [value for value in finite if value is not None]
    if len(values) != len(scales):
        return "invalid", math.nan, math.nan
    minimum, maximum = min(values), max(values)
    label = "homogeneous" if math.isclose(minimum, maximum) else "heteroscedastic"
    return label, minimum, maximum


def _critical_protocol_errors(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    task: PositioningTask,
) -> list[str]:
    summary = payload.get("summary")
    rounds = payload.get("rounds")
    config = payload.get("config")
    if not isinstance(summary, Mapping):
        return ["summary absent or not a mapping"]
    if not isinstance(rounds, list):
        return ["rounds absent or not a list"]
    if not isinstance(config, Mapping):
        return ["config absent or not a mapping"]

    errors: list[str] = []
    expected_rounds = int(expected["training"]["num_rounds"])
    expected_algo = expected["training"]["algo_config"]
    expected_attack = expected_algo.get("attack", {})
    checks = {
        "algorithm": payload.get("algorithm") == "ldp_gradient_far",
        "summary.num_rounds": _int_equals(
            summary.get("num_rounds"), expected_rounds
        ),
        "round count": len(rounds) == expected_rounds,
        "seed": _int_equals(summary.get("seed"), task.seed),
        "partition seed": _int_equals(summary.get("partition_seed"), task.seed),
        "dataset": summary.get("dataset") == expected["data"]["dataset"],
        "model": summary.get("model") == expected["model"]["architecture"],
        "num_clients": _int_equals(
            summary.get("num_clients"), int(expected["clients"]["num_clients"])
        ),
        "device": str(config.get("device")) == "mps",
        "far_alpha": _float_equals(
            config.get("far_alpha"), float(expected_algo["far_alpha"])
        ),
        "reference": config.get("robust_reference")
        == expected_algo["robust_reference"],
        "local clip": _float_equals(
            config.get("clip_norm"), float(expected_algo["clip_norm"])
        ),
        "DP state": bool(config.get("enable_dp"))
        == bool(expected_algo["enable_dp"]),
        "attack": str((config.get("attack") or {}).get("name", "none"))
        == str(expected_attack.get("name", "none")),
    }
    if bool(expected_algo["enable_dp"]):
        checks["target epsilon"] = _float_equals(
            config.get("target_epsilon"), float(expected_algo["target_epsilon"])
        )
        last = rounds[-1] if rounds else {}
        checks["fixed without replacement"] = (
            last.get("privacy_sampling_scheme") == "fixed_without_replacement"
        )
        checks["replace-one"] = last.get("privacy_adjacency") == "replace_one"
    else:
        checks["no-DP target cleared"] = config.get("target_epsilon") is None
        checks["no-DP noise zero"] = _float_equals(
            config.get("noise_multiplier"), 0.0
        )
    for label, passed in checks.items():
        if not passed:
            errors.append(label)
    if rounds:
        observed_rounds = [
            int(value)
            for row in rounds
            if (value := _finite_float(row.get("round_num"))) is not None
        ]
        if (
            len(observed_rounds) != len(rounds)
            or observed_rounds != list(range(1, expected_rounds + 1))
        ):
            errors.append("round indices are not exactly 1..T")
    return errors


def _discover_task_metrics(
    campaign: Campaign, task: PositioningTask
) -> tuple[str, Path | None, Mapping[str, Any] | None, str | None]:
    directory = task_output_dir(campaign, task)
    paths = sorted(directory.glob("**/metrics.json")) if directory.exists() else []
    if not paths:
        return "missing", None, None, "no metrics.json"
    valid: list[tuple[Path, Mapping[str, Any]]] = []
    reasons: list[str] = []
    expected = resolved_config(campaign, task)
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"{path}: unreadable JSON ({exc})")
            continue
        if not isinstance(payload, Mapping):
            reasons.append(f"{path}: top-level JSON is not an object")
            continue
        errors = _critical_protocol_errors(payload, expected, task)
        if errors:
            reasons.append(f"{path}: {', '.join(errors)}")
        else:
            valid.append((path, payload))
    if len(valid) == 1:
        return "complete", valid[0][0], valid[0][1], None
    if len(valid) > 1:
        return "invalid", None, None, "multiple complete metrics.json files"
    return "invalid", None, None, "; ".join(reasons)


def _extract_complete_run(
    campaign: Campaign,
    task: PositioningTask,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected = resolved_config(campaign, task)
    algo = expected["training"]["algo_config"]
    attack = algo.get("attack", {})
    rounds: list[Mapping[str, Any]] = payload["rounds"]  # type: ignore[assignment]
    test_series = _series(rounds, "test_accuracy", scale=100.0)
    client_series = _series(rounds, "client_accuracy_mean", scale=100.0)
    noise_profile, noise_scale_min, noise_scale_max = _noise_profile(algo)

    record: dict[str, Any] = {
        "global_index": task.global_index,
        "phase_index": task.phase_index,
        "phase": task.phase_id,
        "phase_role": task.phase_role,
        "variant_id": task.variant_id,
        "run_id": task.run_id,
        "status": "complete",
        "reason": None,
        "metrics_path": str(path.resolve()),
        "seed": task.seed,
        "dataset": expected["data"]["dataset"],
        "model": expected["model"]["architecture"],
        "num_clients": int(expected["clients"]["num_clients"]),
        "num_rounds": int(expected["training"]["num_rounds"]),
        "local_clip_C": float(algo["clip_norm"]),
        "server_clip_U": float(algo["far_server_clip_norm"]),
        "far_alpha": float(algo["far_alpha"]),
        "robust_reference": str(algo["robust_reference"]),
        "dp_enabled": bool(algo["enable_dp"]),
        "target_epsilon": (
            float(algo["target_epsilon"])
            if algo.get("target_epsilon") is not None
            else None
        ),
        "privacy_delta_target": float(algo.get("delta", math.nan)),
        "noise_profile": noise_profile,
        "noise_scale_min": noise_scale_min,
        "noise_scale_max": noise_scale_max,
        "attack_enabled": bool(attack.get("enabled", False)),
        "attack": str(attack.get("name", "none")),
        "num_byzantine": int(attack.get("num_byzantine", 0)),
        "byzantine_fraction": (
            int(attack.get("num_byzantine", 0))
            / int(expected["clients"]["num_clients"])
        ),
        "survival_ratio_final": _last_value(rounds, "survival_ratio"),
        "test_accuracy_auc_pct": _normalized_auc(test_series),
        "client_accuracy_auc_pct": _normalized_auc(client_series),
        "test_accuracy_final_pct": test_series[-1][1] if test_series else None,
        "test_accuracy_best_pct": (
            max(value for _, value in test_series) if test_series else None
        ),
        "test_accuracy_max_drawdown_pp": _max_drawdown(test_series),
        "round_to_best_test_accuracy": (
            max(test_series, key=lambda pair: pair[1])[0] if test_series else None
        ),
    }
    for output_name, (input_name, scale) in FINAL_METRICS.items():
        # Some quantities above are already computed directly; keep the direct
        # trajectory computation as the canonical value.
        if output_name not in record:
            record[output_name] = _last_value(rounds, input_name, scale=scale)
    for output_name, input_name in MEDIAN_METRICS.items():
        record[output_name] = _median_value(rounds, input_name)
    # Reconstruct the mathematical range rather than trusting legacy signed
    # producer output: range(alpha*s) = |alpha|*(s_max-s_min) >= 0.
    canonical_logit_ranges = [
        (round_num, abs(float(algo["far_alpha"])) * score_span)
        for round_num, score_span in _series(rounds, "far_score_span")
    ]
    if canonical_logit_ranges:
        record["far_logit_range_final"] = canonical_logit_ranges[-1][1]
        record["far_logit_range_median"] = statistics.median(
            value for _, value in canonical_logit_ranges
        )
    return record


def _empty_run_record(
    campaign: Campaign,
    task: PositioningTask,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    expected = resolved_config(campaign, task)
    algo = expected["training"]["algo_config"]
    attack = algo.get("attack", {})
    profile, minimum, maximum = _noise_profile(algo)
    record: dict[str, Any] = {
        "global_index": task.global_index,
        "phase_index": task.phase_index,
        "phase": task.phase_id,
        "phase_role": task.phase_role,
        "variant_id": task.variant_id,
        "run_id": task.run_id,
        "status": status,
        "reason": reason,
        "metrics_path": None,
        "seed": task.seed,
        "dataset": expected["data"]["dataset"],
        "model": expected["model"]["architecture"],
        "num_clients": int(expected["clients"]["num_clients"]),
        "num_rounds": int(expected["training"]["num_rounds"]),
        "local_clip_C": float(algo["clip_norm"]),
        "server_clip_U": float(algo["far_server_clip_norm"]),
        "far_alpha": float(algo["far_alpha"]),
        "robust_reference": str(algo["robust_reference"]),
        "dp_enabled": bool(algo["enable_dp"]),
        "target_epsilon": algo.get("target_epsilon"),
        "privacy_delta_target": float(algo.get("delta", math.nan)),
        "noise_profile": profile,
        "noise_scale_min": minimum,
        "noise_scale_max": maximum,
        "attack_enabled": bool(attack.get("enabled", False)),
        "attack": str(attack.get("name", "none")),
        "num_byzantine": int(attack.get("num_byzantine", 0)),
        "byzantine_fraction": int(attack.get("num_byzantine", 0))
        / int(expected["clients"]["num_clients"]),
        "survival_ratio_final": None,
    }
    for metric in RUN_NUMERIC_METRICS:
        record.setdefault(metric, None)
    return record


def collect_runs(campaign: Campaign) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in campaign.tasks:
        status, path, payload, reason = _discover_task_metrics(campaign, task)
        if status == "complete" and path is not None and payload is not None:
            records.append(_extract_complete_run(campaign, task, path, payload))
        else:
            records.append(_empty_run_record(campaign, task, status, reason))
    return records


def _sample_stats(values: Iterable[Any]) -> dict[str, Any]:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    if not finite:
        return {"count": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "sd": statistics.stdev(finite) if len(finite) > 1 else None,
        "min": min(finite),
        "max": max(finite),
    }


def summarize_variants(
    campaign: Campaign, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    expected_counts: dict[tuple[str, str], int] = {}
    for task in campaign.tasks:
        key = (task.phase_id, task.variant_id)
        expected_counts[key] = expected_counts.get(key, 0) + 1
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["phase"]), str(row["variant_id"]))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1])):
        rows = groups[key]
        complete = [row for row in rows if row["status"] == "complete"]
        first = rows[0]
        summary: dict[str, Any] = {
            "phase": key[0],
            "variant_id": key[1],
            "expected_runs": expected_counts[key],
            "complete_runs": len(complete),
            "invalid_runs": sum(row["status"] == "invalid" for row in rows),
            "missing_runs": sum(row["status"] == "missing" for row in rows),
            "seeds_complete": sorted(int(row["seed"]) for row in complete),
        }
        for field in (
            "dataset",
            "model",
            "num_clients",
            "num_rounds",
            "local_clip_C",
            "server_clip_U",
            "far_alpha",
            "robust_reference",
            "dp_enabled",
            "target_epsilon",
            "noise_profile",
            "noise_scale_min",
            "noise_scale_max",
            "attack",
            "num_byzantine",
        ):
            summary[field] = first.get(field)
        for metric in RUN_NUMERIC_METRICS:
            summary[metric] = _sample_stats(row.get(metric) for row in complete)
        summaries.append(summary)
    return summaries


def summarize_phases(
    campaign: Campaign, records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        rows = [row for row in records if row["phase"] == phase_id]
        summaries.append(
            {
                "phase": phase_id,
                "role": str(phase.get("role", "unspecified")),
                "description": str(phase.get("description", "")),
                "expected": len(rows),
                "complete": sum(row["status"] == "complete" for row in rows),
                "invalid": sum(row["status"] == "invalid" for row in rows),
                "missing": sum(row["status"] == "missing" for row in rows),
            }
        )
    return summaries


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def _flatten_variant_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Mapping) and {"count", "mean", "sd", "min", "max"} <= set(
            value
        ):
            for stat in ("count", "mean", "sd", "min", "max"):
                flat[f"{key}_{stat}"] = value.get(stat)
        else:
            flat[key] = value
    return flat


def _fmt_number(value: Any, digits: int = 2) -> str:
    number = _finite_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _fmt_stat(value: Any, digits: int = 2) -> str:
    if not isinstance(value, Mapping):
        return "NA"
    mean = _finite_float(value.get("mean"))
    sd = _finite_float(value.get("sd"))
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _status_cell(row: Mapping[str, Any]) -> str:
    return f"{row['complete_runs']}/{row['expected_runs']}"


def _phase_markdown_rows(
    phase: str, summaries: Sequence[Mapping[str, Any]]
) -> list[str]:
    rows = [row for row in summaries if row["phase"] == phase]
    if phase == "a_activation_screen":
        lines = [
            "| Variante | Exécutions | n | Dataset | Modèle | AUC acc. (%) | Finale (%) | Meilleure (%) | Drawdown (pp) | Client acc. (%) | Var. (pp²) | Worst-20 (%) | Gap (pp) | Loss finale |",
            "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['variant_id']}` | {_status_cell(row)} | {row['num_clients']} | "
                f"{row['dataset']} | {row['model']} | "
                f"{_fmt_stat(row['test_accuracy_auc_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_best_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_max_drawdown_pp'])} | "
                f"{_fmt_stat(row['client_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['variance_final_pp2'])} | "
                f"{_fmt_stat(row['worst20_final_pct'])} | "
                f"{_fmt_stat(row['gap_final_pp'])} | "
                f"{_fmt_stat(row['test_loss_final'], 4)} |"
            )
        return lines

    if phase == "b_local_clip_screen":
        lines = [
            "| Variante | Exécutions | n | C | AUC acc. (%) | Finale (%) | Meilleure (%) | Drawdown (pp) | Client acc. (%) | Var. (pp²) | Worst-20 (%) | Gap (pp) | Clip local médian | Clip serveur médian |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['variant_id']}` | {_status_cell(row)} | {row['num_clients']} | "
                f"{_fmt_number(row['local_clip_C'])} | "
                f"{_fmt_stat(row['test_accuracy_auc_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_best_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_max_drawdown_pp'])} | "
                f"{_fmt_stat(row['client_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['variance_final_pp2'])} | "
                f"{_fmt_stat(row['worst20_final_pct'])} | "
                f"{_fmt_stat(row['gap_final_pp'])} | "
                f"{_fmt_stat(row['privacy_local_clip_rate_median'], 3)} | "
                f"{_fmt_stat(row['server_clip_rate_median'], 3)} |"
            )
        return lines

    if phase == "c_alpha_reference_screen":
        lines = [
            "| Variante | Exécutions | n | α | Référence | AUC acc. (%) | Finale (%) | Meilleure (%) | Drawdown (pp) | Client acc. (%) | Var. (pp²) | Worst-20 (%) | Gap (pp) | q max médian | nΣq² médian | Entropie médiane | Plage des logits médiane |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['variant_id']}` | {_status_cell(row)} | {row['num_clients']} | "
                f"{_fmt_number(row['far_alpha'])} | {row['robust_reference']} | "
                f"{_fmt_stat(row['test_accuracy_auc_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_best_pct'])} | "
                f"{_fmt_stat(row['test_accuracy_max_drawdown_pp'])} | "
                f"{_fmt_stat(row['client_accuracy_final_pct'])} | "
                f"{_fmt_stat(row['variance_final_pp2'])} | "
                f"{_fmt_stat(row['worst20_final_pct'])} | "
                f"{_fmt_stat(row['gap_final_pp'])} | "
                f"{_fmt_stat(row['far_max_weight_median'], 4)} | "
                f"{_fmt_stat(row['far_weight_concentration_median'], 3)} | "
                f"{_fmt_stat(row['far_weight_entropy_median'], 3)} | "
                f"{_fmt_stat(row['far_logit_range_median'], 3)} |"
            )
        return lines

    # DP, heterogeneous-noise, Byzantine and confirmation phases share a
    # performance table, then distinct privacy/clipping and oracle tables.
    lines = [
        "| Variante | Exécutions | n | Attaque | Bruit | α | Référence | AUC acc. (%) | Finale (%) | Meilleure (%) | Drawdown (pp) | Client acc. (%) | Var. (pp²) | Worst-20 (%) | Gap (pp) |",
        "|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | {_status_cell(row)} | {row['num_clients']} | "
            f"{row['attack']} | {row['noise_profile']} "
            f"[{_fmt_number(row['noise_scale_min'], 1)}, {_fmt_number(row['noise_scale_max'], 1)}] | "
            f"{_fmt_number(row['far_alpha'])} | {row['robust_reference']} | "
            f"{_fmt_stat(row['test_accuracy_auc_pct'])} | "
            f"{_fmt_stat(row['test_accuracy_final_pct'])} | "
            f"{_fmt_stat(row['test_accuracy_best_pct'])} | "
            f"{_fmt_stat(row['test_accuracy_max_drawdown_pp'])} | "
            f"{_fmt_stat(row['client_accuracy_final_pct'])} | "
            f"{_fmt_stat(row['variance_final_pp2'])} | "
            f"{_fmt_stat(row['worst20_final_pct'])} | "
            f"{_fmt_stat(row['gap_final_pp'])} |"
        )
    lines.extend(
        [
            "",
            "Confidentialité et clipping :",
            "",
            "| Variante | ε cible | ε réalisé | δ réalisé | σ min. | σ moyen | σ max. | Clip local médian | Clip serveur médian |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | "
            f"{_fmt_number(row['target_epsilon'])} | "
            f"{_fmt_stat(row['privacy_epsilon_final'], 4)} | "
            f"{_fmt_stat(row['privacy_delta_final'], 7)} | "
            f"{_fmt_stat(row['privacy_noise_multiplier_min_final'], 4)} | "
            f"{_fmt_stat(row['privacy_noise_multiplier_mean_final'], 4)} | "
            f"{_fmt_stat(row['privacy_noise_multiplier_max_final'], 4)} | "
            f"{_fmt_stat(row['privacy_local_clip_rate_median'], 3)} | "
            f"{_fmt_stat(row['server_clip_rate_median'], 3)} |"
        )
    lines.extend(
        [
            "",
            "Diagnostics mécanistiques (oracle de simulation, jamais entrée de l'algorithme) :",
            "",
            "| Variante | Masse byzantine | Erreur référence–centre honnête | Corr. poids–bruit | Corr. score bruité–propre | Biais tilting honnête | Bruit DP à poids fixes | Déplacement byzantin | Erreur agrégat–centre honnête |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['variant_id']}` | "
            f"{_fmt_stat(row['byzantine_weight_mass_median_oracle'], 4)} | "
            f"{_fmt_stat(row['reference_honest_center_error_median_oracle'], 4)} | "
            f"{_fmt_stat(row['weight_effective_noise_corr_median_oracle'], 3)} | "
            f"{_fmt_stat(row['honest_noisy_clean_score_corr_median_oracle'], 3)} | "
            f"{_fmt_stat(row['honest_clean_tilting_bias_norm_median_oracle'], 4)} | "
            f"{_fmt_stat(row['honest_fixed_weight_dp_noise_norm_median_oracle'], 4)} | "
            f"{_fmt_stat(row['byzantine_displacement_norm_median_oracle'], 4)} | "
            f"{_fmt_stat(row['aggregate_error_to_clean_honest_center_median_oracle'], 4)} |"
        )
    return lines


def render_report(
    campaign: Campaign,
    records: Sequence[Mapping[str, Any]],
    phase_summaries: Sequence[Mapping[str, Any]],
    variant_summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> str:
    complete = sum(row["status"] == "complete" for row in records)
    invalid = sum(row["status"] == "invalid" for row in records)
    missing = sum(row["status"] == "missing" for row in records)
    lines = [
        "# Analyse descriptive — campagne de positionnement LDP-Gradient-FAR",
        "",
        "**Statut méthodologique : descriptif uniquement.** Ce document ne prend "
        "aucune décision de gate et ne promeut aucune configuration. Les choix "
        "scientifiques doivent être enregistrés séparément après inspection des "
        "résultats et des confondants.",
        "",
        f"- Matrice : `{campaign.matrix_path.resolve()}`",
        f"- Résultats attendus : **{len(records)}**",
        f"- Complets et valides : **{complete}**",
        f"- Invalides/partiels : **{invalid}**",
        f"- Absents : **{missing}**",
        f"- Export reproductible : `{output_dir.resolve()}`",
        "",
        "## 1. Complétude par phase",
        "",
        "| Phase | Rôle | Complets | Invalides | Absents | Total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in phase_summaries:
        lines.append(
            f"| `{row['phase']}` | {row['role']} | {row['complete']} | "
            f"{row['invalid']} | {row['missing']} | {row['expected']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Définitions des indicateurs",
            "",
            "- **AUC accuracy** : intégrale trapézoïdale de la Test Accuracy, "
            "divisée par l'intervalle de rounds observé. Elle mesure la qualité "
            "de toute la trajectoire, et pas seulement le dernier tour.",
            "- **Drawdown maximal** : plus grande baisse, en points, entre le "
            "meilleur niveau déjà atteint et un tour ultérieur. Il quantifie une "
            "instabilité ou un effondrement après apprentissage.",
            "- **Variance, Worst-20 et gap** : métriques finales de fairness "
            "inter-clients. Une variance et un gap plus faibles ne sont favorables "
            "que si l'accuracy ne s'effondre pas.",
            "- **Concentration `nΣq²`** : vaut 1 pour des poids uniformes et "
            "augmente quand quelques clients dominent. L'entropie et le nombre "
            "effectif de clients donnent les vues complémentaires.",
            "- **Span logit signé** : quantité enregistrée "
            "`α(s_max-s_min)`. Sa valeur est négative pour un α négatif ; sa "
            "valeur absolue mesure l'étendue des logits de la softmax.",
            "- Les quantités suffixées **oracle** utilisent des informations "
            "accessibles uniquement au simulateur (clients byzantins ou gradient "
            "avant bruit). Elles servent à expliquer, jamais à entraîner.",
            "",
            "Les valeurs dynamiques de poids, score, clipping et oracle sont des "
            "médianes sur les tours. Les métriques d'accuracy/fairness dites "
            "finales proviennent du dernier tour évalué. Les cellules à plusieurs "
            "seeds sont affichées en moyenne ± écart-type échantillonnal ; avec une "
            "seule seed, aucun écart-type n'est inventé.",
        ]
    )
    for index, phase in enumerate(campaign.phases, start=3):
        phase_id = str(phase["id"])
        status = next(row for row in phase_summaries if row["phase"] == phase_id)
        lines.extend(
            [
                "",
                f"## {index}. {phase_id}",
                "",
                str(phase.get("description", "")),
                "",
                f"Complétude : **{status['complete']}/{status['expected']}** "
                f"(invalides : {status['invalid']}, absents : {status['missing']}).",
                "",
                *_phase_markdown_rows(phase_id, variant_summaries),
            ]
        )

    unresolved = [row for row in records if row["status"] != "complete"]
    lines.extend(["", "## Annexe — tâches non résolues", ""])
    if not unresolved:
        lines.append("Toutes les tâches enregistrées disposent d'un fichier valide.")
    else:
        lines.extend(
            [
                "| Index | Phase | Run | Statut | Motif |",
                "|---:|---|---|---|---|",
            ]
        )
        for row in unresolved:
            reason = str(row.get("reason") or "").replace("|", "\\|")
            lines.append(
                f"| {row['global_index']} | `{row['phase']}` | "
                f"`{row['run_id']}` | {row['status']} | {reason} |"
            )
    lines.extend(
        [
            "",
            "---",
            "",
            "Ce rapport est volontairement muet sur `promote`/`stop`. Il fournit "
            "les observations nécessaires à une décision humaine préenregistrée, "
            "sans transformer un résumé descriptif en règle de sélection post hoc.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_campaign(
    campaign: Campaign,
    *,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    records = collect_runs(campaign)
    phase_summaries = summarize_phases(campaign, records)
    variant_summaries = summarize_variants(campaign, records)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_csv = output_dir / "run_level.csv"
    variant_csv = output_dir / "variant_summary.csv"
    phase_csv = output_dir / "phase_status.csv"
    json_path = output_dir / "analysis.json"
    _write_csv(run_csv, records)
    _write_csv(variant_csv, [_flatten_variant_summary(row) for row in variant_summaries])
    _write_csv(phase_csv, phase_summaries)

    payload = {
        "schema_version": 1,
        "analysis_kind": "descriptive_only",
        "automated_gate_decision": False,
        "campaign_id": campaign.matrix["campaign_id"],
        "matrix": str(campaign.matrix_path.resolve()),
        "output_root": str(campaign.output_root.resolve()),
        "counts": {
            "expected": len(records),
            "complete": sum(row["status"] == "complete" for row in records),
            "invalid": sum(row["status"] == "invalid" for row in records),
            "missing": sum(row["status"] == "missing" for row in records),
        },
        "phase_status": phase_summaries,
        "runs": records,
        "variant_summaries": variant_summaries,
        "artifacts": {
            "run_level_csv": str(run_csv.resolve()),
            "variant_summary_csv": str(variant_csv.resolve()),
            "phase_status_csv": str(phase_csv.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            campaign,
            records,
            phase_summaries,
            variant_summaries,
            output_dir,
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    campaign = load_campaign(args.matrix)
    if args.results is not None:
        campaign = replace(campaign, output_root=args.results.resolve())
    payload = analyze_campaign(
        campaign,
        output_dir=args.output_dir.resolve(),
        report_path=args.report.resolve(),
    )
    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("automated_gate_decision=false")


if __name__ == "__main__":
    main()
