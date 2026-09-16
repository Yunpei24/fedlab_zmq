#!/usr/bin/env python3
"""Audit, gate and summarize the RCIG/LDP-gradient-FAR campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rcig_ldp_gradient_far import (  # noqa: E402
    DEFAULT_MATRIX,
    RCIGCampaign,
    RCIGTask,
    campaign_scientific_hash,
    load_campaign,
    resolved_config,
    task_output_dir,
)

DEFAULT_OUTPUT_DIR = ROOT / "output" / "analysis" / "rcig_ldp_gradient_far_v1"
DEFAULT_REPORT = (
    ROOT / "output" / "analysis" / "RCIG_LDP_Gradient_FAR_End_to_End_Report.md"
)
MODES = ("full", "isotropic", "euclidean")
ERROR_SUFFIX = "error_to_clean_honest_center_oracle"


@dataclass
class RunRecord:
    task: RCIGTask
    status: str
    reason: str
    metrics_path: Path | None
    payload: dict[str, Any] | None
    expected: dict[str, Any]

    @property
    def rounds(self) -> list[dict[str, Any]]:
        if self.payload is None:
            return []
        rows = self.payload.get("rounds")
        return rows if isinstance(rows, list) else []


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool01(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    number = _finite(value)
    if number is None or number not in {0.0, 1.0}:
        return None
    return number


def _find_metrics(output_dir: Path) -> Path | None:
    paths = sorted(output_dir.glob("**/metrics.json"))
    return paths[0] if len(paths) == 1 else None


def _ready_rows(record: RunRecord) -> list[dict[str, Any]]:
    return [row for row in record.rounds if bool(row.get("rcig_history_ready", False))]


def _attack_active(config: Mapping[str, Any], round_num: int) -> bool:
    attack = config["training"]["algo_config"].get("attack", {})
    if not bool(attack.get("enabled", False)):
        return False
    start = attack.get("active_round_start")
    end = attack.get("active_round_end")
    if start is None and end is None:
        return True
    return int(start) <= round_num <= int(end)


def _strict_past(row: Mapping[str, Any]) -> bool:
    declared = _bool01(row.get("rcig_reference_strictly_past"))
    round_num = int(row.get("round_num", row.get("round", -1)))
    gate_min = _finite(row.get("rcig_gate_source_round_min"))
    gate_max = _finite(row.get("rcig_gate_source_round_max"))
    # The implementation calls the two views ``older`` and ``newer``.  Keep
    # the shorter historical aliases readable for old synthetic fixtures, but
    # validate the names emitted by real runs first.
    old_min = _finite(row.get("rcig_older_round_min"))
    old_max = _finite(row.get("rcig_older_round_max"))
    new_min = _finite(row.get("rcig_newer_round_min"))
    new_max = _finite(row.get("rcig_newer_round_max"))
    if old_min is None:
        old_min = _finite(row.get("rcig_old_round_min"))
    if old_max is None:
        old_max = _finite(row.get("rcig_old_round_max"))
    if new_min is None:
        new_min = _finite(row.get("rcig_new_round_min"))
    if new_max is None:
        new_max = _finite(row.get("rcig_new_round_max"))
    values = (gate_min, gate_max, old_min, old_max, new_min, new_max)
    if declared != 1.0 or any(value is None for value in values):
        return False
    assert all(value is not None for value in values)
    return bool(
        gate_min <= gate_max < old_min <= old_max < new_min <= new_max < round_num
    )


def _mode_value(row: Mapping[str, Any], mode: str, suffix: str) -> float | None:
    converter = _bool01 if suffix == "gate_active" else _finite
    explicit = converter(row.get(f"rcig_{mode}_{suffix}"))
    if explicit is not None:
        return explicit
    selected_mode = str(row.get("rcig_reference_mode", ""))
    if selected_mode == mode:
        aliases = {
            "innovation_stat": "rcig_innovation_stat",
            "gate_active": "rcig_gate_active",
            ERROR_SUFFIX: f"rcig_reference_{ERROR_SUFFIX}",
        }
        if suffix in aliases:
            return converter(row.get(aliases[suffix]))
    return None


def _critical_protocol_errors(
    payload: Mapping[str, Any], expected: Mapping[str, Any], task: RCIGTask
) -> list[str]:
    errors: list[str] = []
    rounds = payload.get("rounds")
    summary = payload.get("summary")
    config = payload.get("config")
    if not isinstance(rounds, list):
        return ["rounds missing"]
    if not isinstance(summary, Mapping):
        return ["summary missing"]
    if not isinstance(config, Mapping):
        return ["algorithm config missing"]
    expected_rounds = int(expected["training"]["num_rounds"])
    expected_algo = expected["training"]["algo_config"]
    checks = {
        "algorithm": payload.get("algorithm") == "ldp_gradient_far",
        "round_count": len(rounds) == expected_rounds,
        "summary_rounds": int(summary.get("num_rounds", -1)) == expected_rounds,
        "seed": int(summary.get("seed", -1)) == task.seed,
        "partition_seed": int(summary.get("partition_seed", -1)) == task.seed,
        "dataset": summary.get("dataset") == "fashionmnist",
        "model": summary.get("model") == "lenet5_tanh",
        "num_clients": int(summary.get("num_clients", -1)) == 25,
        "device": config.get("device") == "mps",
        "reference": config.get("robust_reference")
        == expected_algo.get("robust_reference"),
        "alpha": math.isclose(
            float(config.get("far_alpha", math.nan)),
            float(expected_algo.get("far_alpha", math.nan)),
            abs_tol=1.0e-12,
        ),
        "local_clip": math.isclose(
            float(config.get("clip_norm", math.nan)), 4.0, abs_tol=1.0e-12
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    if rounds:
        for index, row in enumerate(rounds, start=1):
            if row.get("privacy_sampling_scheme") != "fixed_without_replacement":
                errors.append(f"round_{index}_sampling")
                break
            if row.get("privacy_adjacency") != "replace_one":
                errors.append(f"round_{index}_adjacency")
                break
    if expected_algo.get("robust_reference") == "rcig_temporal":
        ready = [row for row in rounds if bool(row.get("rcig_history_ready", False))]
        if not ready:
            errors.append("no_rcig_ready_round")
        else:
            if not all(_strict_past(row) for row in ready):
                errors.append("rcig_not_strictly_past")
            if not all(
                _bool01(row.get("rcig_covariance_psd_certified")) == 1.0
                for row in ready
            ):
                errors.append("rcig_covariance_not_psd")
            if not all(
                int(row.get("rcig_public_subspace_dimension", -1)) == 64
                for row in ready
            ):
                errors.append("rcig_wrong_public_subspace")
            for mode in MODES:
                required = (
                    _mode_value(ready[-1], mode, "innovation_stat"),
                    _mode_value(ready[-1], mode, "gate_active"),
                    _mode_value(ready[-1], mode, ERROR_SUFFIX),
                )
                if any(value is None for value in required):
                    errors.append(f"missing_rcig_{mode}_counterfactual")
    return errors


def collect_runs(campaign: RCIGCampaign) -> list[RunRecord]:
    records: list[RunRecord] = []
    calibration_gate = campaign.output_root / "_gates" / "r0_calibration_frozen.json"
    calibration_promoted = False
    if calibration_gate.exists():
        try:
            calibration_promoted = (
                json.loads(calibration_gate.read_text(encoding="utf-8")).get("decision")
                == "promote"
            )
        except (OSError, json.JSONDecodeError):
            calibration_promoted = False
    for task in campaign.tasks:
        inject = calibration_promoted or task.phase_id == "r0_calibration_frozen"
        expected = resolved_config(campaign, task, inject_threshold=inject)
        output_dir = task_output_dir(campaign, task)
        metrics_path = _find_metrics(output_dir)
        status_path = output_dir / "orchestration_status.json"
        if metrics_path is None or not status_path.exists():
            records.append(
                RunRecord(
                    task, "missing", "artifact absent", metrics_path, None, expected
                )
            )
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append(
                RunRecord(task, "invalid", str(exc), metrics_path, None, expected)
            )
            continue
        if status.get("status") != "completed":
            records.append(
                RunRecord(
                    task,
                    "invalid",
                    "orchestration not completed",
                    metrics_path,
                    payload,
                    expected,
                )
            )
            continue
        errors = _critical_protocol_errors(payload, expected, task)
        records.append(
            RunRecord(
                task,
                "complete" if not errors else "invalid",
                "; ".join(errors),
                metrics_path,
                payload,
                expected,
            )
        )
    return records


def _phase_records(records: Sequence[RunRecord], phase_id: str) -> list[RunRecord]:
    return [record for record in records if record.task.phase_id == phase_id]


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _cp_upper(successes: int, trials: int, alpha: float = 0.025) -> float:
    """Exact one-sided Clopper-Pearson upper endpoint by bisection."""

    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    if successes == trials:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(trials, index)
            * probability**index
            * (1.0 - probability) ** (trials - index)
            for index in range(successes + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if cdf(midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return high


_T_ONE_SIDED_95 = {
    1: 6.313752,
    2: 2.919986,
    3: 2.353363,
    4: 2.131847,
    5: 2.015048,
    6: 1.943180,
    7: 1.894579,
    8: 1.859548,
    9: 1.833113,
    10: 1.812461,
    11: 1.795885,
    12: 1.782288,
    15: 1.753050,
    17: 1.739607,
    20: 1.724718,
    24: 1.710882,
    30: 1.697261,
    35: 1.689572,
    40: 1.683851,
    60: 1.670649,
    120: 1.657651,
}


def _conservative_t95(df: int) -> float:
    eligible = [key for key in _T_ONE_SIDED_95 if key <= df]
    return _T_ONE_SIDED_95[max(eligible)] if eligible else math.inf


def _one_sided_interval(values: Sequence[float]) -> tuple[float, float, float]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    if not numbers:
        return math.nan, -math.inf, math.inf
    mean = statistics.fmean(numbers)
    if len(numbers) == 1:
        return mean, -math.inf, math.inf
    margin = (
        _conservative_t95(len(numbers) - 1)
        * statistics.stdev(numbers)
        / math.sqrt(len(numbers))
    )
    return mean, mean - margin, mean + margin


def _common_evidence(records: Sequence[RunRecord]) -> dict[str, float]:
    total = len(records)
    complete = [record for record in records if record.status == "complete"]
    return {
        "complete_fraction": len(complete) / total if total else 0.0,
        "invalid_runs": float(sum(record.status == "invalid" for record in records)),
        "mps_fraction": (
            sum(
                record.payload is not None
                and record.payload.get("config", {}).get("device") == "mps"
                and all(
                    _finite(row.get("rcig_private_gradient_mps_fraction")) == 1.0
                    for row in record.rounds
                )
                for record in complete
            )
            / len(complete)
            if complete
            else 0.0
        ),
    }


def _causality_evidence(records: Sequence[RunRecord]) -> dict[str, float]:
    ready_rows = [
        row
        for record in records
        if record.status == "complete"
        for row in _ready_rows(record)
    ]
    return {
        "strict_past_fraction": (
            sum(_strict_past(row) for row in ready_rows) / len(ready_rows)
            if ready_rows
            else 0.0
        ),
        "covariance_psd_fraction": (
            sum(
                _bool01(row.get("rcig_covariance_psd_certified")) == 1.0
                for row in ready_rows
            )
            / len(ready_rows)
            if ready_rows
            else 0.0
        ),
    }


def _calibration_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence: dict[str, Any] = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    complete = [record for record in records if record.status == "complete"]
    private_ok = []
    maxima: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in complete:
        private_ok.append(
            all(
                row.get("privacy_sampling_scheme") == "fixed_without_replacement"
                and row.get("privacy_adjacency") == "replace_one"
                for row in record.rounds
            )
        )
        regime = record.task.axis_values["noise_regime"]
        ready = _ready_rows(record)
        for mode in MODES:
            values = [_mode_value(row, mode, "innovation_stat") for row in ready]
            finite = [value for value in values if value is not None]
            if finite:
                maxima[(regime, mode)].append(max(finite))
    evidence["private_gradient_protocol_fraction"] = (
        sum(private_ok) / len(private_ok) if private_ok else 0.0
    )
    thresholds: dict[str, dict[str, float]] = {}
    expected_cells = 0
    finite_cells = 0
    for regime in ("homogeneous", "heteroscedastic"):
        thresholds[regime] = {}
        for mode in MODES:
            expected_cells += 1
            values = maxima.get((regime, mode), [])
            if values:
                threshold = _nearest_rank(values, 0.99)
                thresholds[regime][mode] = threshold
                finite_cells += int(math.isfinite(threshold) and threshold > 0.0)
    evidence["finite_threshold_fraction"] = finite_cells / expected_cells
    evidence["thresholds_by_regime_and_mode"] = thresholds
    evidence["threshold_statistical_unit_count"] = {
        f"{regime}/{mode}": len(maxima.get((regime, mode), []))
        for regime in ("homogeneous", "heteroscedastic")
        for mode in MODES
    }
    return evidence


def _null_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence: dict[str, Any] = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    complete = [record for record in records if record.status == "complete"]
    cp_cells: dict[str, Any] = {}
    upper_bounds: list[float] = []
    losses: list[float] = []
    for regime in ("homogeneous", "heteroscedastic"):
        cell = [
            record
            for record in complete
            if record.task.axis_values["noise_regime"] == regime
        ]
        for mode in MODES:
            activations = 0
            valid = 0
            for record in cell:
                values = [
                    _mode_value(row, mode, "gate_active") for row in _ready_rows(record)
                ]
                finite = [value for value in values if value is not None]
                if finite:
                    valid += 1
                    activations += int(any(value >= 0.5 for value in finite))
            upper = _cp_upper(activations, valid) if valid else 1.0
            upper_bounds.append(upper)
            cp_cells[f"{regime}/{mode}"] = {
                "activations": activations,
                "trials": valid,
                "cp97_5_upper": upper,
            }
        for record in cell:
            full = [
                _mode_value(row, "full", ERROR_SUFFIX) for row in _ready_rows(record)
            ]
            identity = [
                _finite(row.get(f"rcig_identity_new_{ERROR_SUFFIX}"))
                for row in _ready_rows(record)
            ]
            paired = [
                (candidate - baseline) / max(baseline, 1.0e-15)
                for candidate, baseline in zip(full, identity)
                if candidate is not None and baseline is not None
            ]
            if paired:
                losses.append(statistics.fmean(paired))
    mean, _low, upper = _one_sided_interval(losses)
    evidence.update(
        {
            "max_cp97_5_false_activation_upper": max(upper_bounds, default=1.0),
            "max_no_attack_reference_loss_one_sided_upper": upper,
            "no_attack_reference_loss_mean": mean,
            "false_activation_cells": cp_cells,
        }
    )
    return evidence


def _mechanistic_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence: dict[str, Any] = _common_evidence(records)
    evidence.update(_causality_evidence(records))
    rcig_gains: list[float] = []
    iso_gains: list[float] = []
    euclidean_gains: list[float] = []
    active_flags: list[float] = []
    for record in records:
        if record.status != "complete":
            continue
        active_rows = [
            row
            for row in _ready_rows(record)
            if _attack_active(
                record.expected,
                int(row.get("round_num", row.get("round", -1))),
            )
        ]
        per_run: dict[str, list[float]] = defaultdict(list)
        for row in active_rows:
            full = _mode_value(row, "full", ERROR_SUFFIX)
            isotropic = _mode_value(row, "isotropic", ERROR_SUFFIX)
            euclidean = _mode_value(row, "euclidean", ERROR_SUFFIX)
            identity = _finite(row.get(f"rcig_identity_new_{ERROR_SUFFIX}"))
            if full is not None and identity is not None:
                per_run["identity"].append((identity - full) / max(identity, 1.0e-15))
            if full is not None and isotropic is not None:
                per_run["isotropic"].append(
                    (isotropic - full) / max(isotropic, 1.0e-15)
                )
            if full is not None and euclidean is not None:
                per_run["euclidean"].append(
                    (euclidean - full) / max(euclidean, 1.0e-15)
                )
            active = _mode_value(row, "full", "gate_active")
            if active is not None:
                active_flags.append(float(active >= 0.5))
        if per_run["identity"]:
            rcig_gains.append(statistics.fmean(per_run["identity"]))
        if per_run["isotropic"]:
            iso_gains.append(statistics.fmean(per_run["isotropic"]))
        if per_run["euclidean"]:
            euclidean_gains.append(statistics.fmean(per_run["euclidean"]))
    gain_mean, gain_low, _ = _one_sided_interval(rcig_gains)
    iso_mean, iso_low, _ = _one_sided_interval(iso_gains)
    euc_mean, euc_low, _ = _one_sided_interval(euclidean_gains)
    evidence.update(
        {
            "attacked_mse_gain_vs_identity_new_one_sided_low": gain_low,
            "attacked_mse_gain_vs_identity_new_mean": gain_mean,
            "attack_gate_activation_rate": (
                statistics.fmean(active_flags) if active_flags else 0.0
            ),
            "full_gain_vs_isotropic_one_sided_low": iso_low,
            "full_gain_vs_isotropic_mean": iso_mean,
            "full_gain_vs_euclidean_one_sided_low": euc_low,
            "full_gain_vs_euclidean_mean": euc_mean,
            "paired_run_count": len(rcig_gains),
        }
    )
    return evidence


def _last(
    rowset: Sequence[Mapping[str, Any]], key: str, scale: float = 1.0
) -> float | None:
    values = [_finite(row.get(key)) for row in rowset]
    finite = [value for value in values if value is not None]
    return finite[-1] * scale if finite else None


def _end_to_end_row(record: RunRecord) -> dict[str, Any]:
    rounds = record.rounds
    return {
        "phase": record.task.phase_id,
        "run_id": record.task.run_id,
        "seed": record.task.seed,
        **record.task.axis_values,
        "status": record.status,
        "reason": record.reason,
        "test_accuracy_pct": _last(rounds, "test_accuracy", 100.0),
        "client_accuracy_pct": _last(rounds, "client_accuracy_mean", 100.0),
        "test_loss": _last(rounds, "test_loss"),
        "variance_pp2": _last(rounds, "client_accuracy_variance_pct2"),
        "worst20_pct": _last(rounds, "worst20_accuracy_pct"),
        "gap_pp": _last(rounds, "best20_worst20_gap_pct"),
        "epsilon": _last(rounds, "privacy_epsilon_max"),
        "byzantine_weight_mass_oracle": _last(rounds, "byzantine_weight_mass_oracle"),
        "rcig_gate_active": _last(rounds, "rcig_gate_active"),
        "rcig_newer_view_trust": _last(rounds, "rcig_newer_view_trust"),
    }


def _paired_metric(
    rows: Sequence[Mapping[str, Any]],
    reference: str,
    comparator: str,
    key: str,
    *,
    reverse: bool = False,
) -> list[float]:
    index = {
        (
            row.get("noise_regime"),
            row.get("attack_schedule"),
            row.get("seed"),
            row.get("reference"),
        ): row
        for row in rows
        if row.get("status") == "complete"
    }
    values: list[float] = []
    cells = {
        (row.get("noise_regime"), row.get("attack_schedule"), row.get("seed"))
        for row in rows
        if row.get("reference") == reference and row.get("status") == "complete"
    }
    for regime, attack, seed in sorted(cells):
        left = index.get((regime, attack, seed, reference))
        right = index.get((regime, attack, seed, comparator))
        if left is None or right is None:
            continue
        a, b = _finite(left.get(key)), _finite(right.get(key))
        if a is not None and b is not None:
            values.append((b - a) if reverse else (a - b))
    return values


def _end_to_end_development_evidence(records: Sequence[RunRecord]) -> dict[str, Any]:
    evidence: dict[str, Any] = _common_evidence(records)
    rows = [_end_to_end_row(record) for record in records]
    complete = [row for row in rows if row["status"] == "complete"]
    epsilon_errors = [
        abs(float(row["epsilon"]) - 4.0)
        for row in complete
        if row["epsilon"] is not None
    ]
    none = [row for row in complete if row.get("attack_schedule") == "none"]
    rcig_none = next((row for row in none if row.get("reference") == "rcig_full"), None)
    baselines = [
        row
        for row in none
        if row.get("reference") in {"uniform", "fcc", "rfa", "trmean"}
    ]
    if rcig_none and baselines:
        deficit = max(float(row["test_accuracy_pct"]) for row in baselines) - float(
            rcig_none["test_accuracy_pct"]
        )
    else:
        deficit = 1.0e30
    attacked = [row for row in complete if row.get("attack_schedule") != "none"]
    worst_deltas = _paired_metric(attacked, "rcig_full", "fcc", "worst20_pct")
    gap_reductions = _paired_metric(
        attacked, "rcig_full", "fcc", "gap_pp", reverse=True
    )
    persistent = [
        row for row in attacked if "persistent" in str(row.get("attack_schedule"))
    ]
    accuracy_deltas = _paired_metric(
        persistent, "rcig_full", "fcc", "test_accuracy_pct"
    )
    evidence.update(
        {
            "max_abs_epsilon_error": max(epsilon_errors, default=1.0e30),
            "rcig_full_no_attack_accuracy_deficit_vs_best_baseline_pp": deficit,
            "rcig_full_attacked_worst20_delta_vs_fcc_pp": (
                statistics.fmean(worst_deltas) if worst_deltas else -1.0e30
            ),
            "rcig_full_attacked_gap_reduction_vs_fcc_pp": (
                statistics.fmean(gap_reductions) if gap_reductions else -1.0e30
            ),
            "rcig_full_persistent_accuracy_delta_vs_fcc_pp": (
                statistics.fmean(accuracy_deltas) if accuracy_deltas else -1.0e30
            ),
        }
    )
    return evidence


def evaluate_gate(campaign: RCIGCampaign, phase_id: str) -> dict[str, Any]:
    records = _phase_records(collect_runs(campaign), phase_id)
    if phase_id == "r0_calibration_frozen":
        return _calibration_evidence(records)
    if phase_id == "r1_null_validation_frozen":
        return _null_evidence(records)
    if phase_id == "r2_attack_mechanism_frozen":
        return _mechanistic_evidence(records)
    if phase_id == "r3_end_to_end_development":
        return _end_to_end_development_evidence(records)
    raise ValueError(f"phase {phase_id!r} has no preregistered gate evaluator")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: Iterable[Any]) -> str:
    finite = [value for item in values if (value := _finite(item)) is not None]
    if not finite:
        return "—"
    if len(finite) == 1:
        return f"{finite[0]:.3f}"
    return f"{statistics.fmean(finite):.3f} ± {statistics.stdev(finite):.3f}"


def analyze_campaign(
    campaign: RCIGCampaign,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    records = collect_runs(campaign)
    run_rows = [_end_to_end_row(record) for record in records]
    _write_csv(output_dir / "run_level.csv", run_rows)
    phase_rows = []
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        cell = _phase_records(records, phase_id)
        phase_rows.append(
            {
                "phase": phase_id,
                "expected": len(cell),
                "complete": sum(row.status == "complete" for row in cell),
                "invalid": sum(row.status == "invalid" for row in cell),
                "missing": sum(row.status == "missing" for row in cell),
            }
        )
    _write_csv(output_dir / "phase_status.csv", phase_rows)

    complete_e2e = [
        row
        for row in run_rows
        if row["status"] == "complete"
        and row["phase"].startswith("r")
        and "end_to_end" in row["phase"]
    ]
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in complete_e2e:
        groups[
            (
                row["phase"],
                str(row.get("noise_regime")),
                str(row.get("reference")),
                str(row.get("attack_schedule")),
            )
        ].append(row)
    summary_rows = []
    for (phase, regime, reference, attack), cell in sorted(groups.items()):
        summary_rows.append(
            {
                "phase": phase,
                "noise_regime": regime,
                "reference": reference,
                "attack": attack,
                "n": len(cell),
                "test_accuracy_pct": _mean_sd(row["test_accuracy_pct"] for row in cell),
                "client_accuracy_pct": _mean_sd(
                    row["client_accuracy_pct"] for row in cell
                ),
                "variance_pp2": _mean_sd(row["variance_pp2"] for row in cell),
                "worst20_pct": _mean_sd(row["worst20_pct"] for row in cell),
                "gap_pp": _mean_sd(row["gap_pp"] for row in cell),
                "test_loss": _mean_sd(row["test_loss"] for row in cell),
            }
        )
    _write_csv(output_dir / "end_to_end_summary.csv", summary_rows)

    state = {
        "campaign_id": campaign.matrix["campaign_id"],
        "campaign_scientific_hash": campaign_scientific_hash(campaign),
        "counts": {
            "expected": len(records),
            "complete": sum(record.status == "complete" for record in records),
            "invalid": sum(record.status == "invalid" for record in records),
            "missing": sum(record.status == "missing" for record in records),
        },
        "phase_status": phase_rows,
        "model_replacement_status": campaign.matrix["scientific_scope"][
            "model_replacement_status"
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# RCIG dans LDP-Gradient-FAR — état de la campagne end-to-end",
        "",
        f"Hash scientifique verrouillé : `{campaign_scientific_hash(campaign)}`.",
        "",
        "## État des phases",
        "",
        "| Phase | Attendus | Complets | Invalides | Manquants |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            f"| {row['phase']} | {row['expected']} | {row['complete']} | {row['invalid']} | {row['missing']} |"
        )
    lines.extend(
        [
            "",
            "## Résultats end-to-end disponibles",
            "",
            "Les cellules sont des moyennes ± écarts-types entre seeds. Les métriques clientes sont calculées sur les clients honnêtes configurés.",
            "",
            "| Phase | Bruit | Référence | Attaque | n | Test Acc. (%) | Worst-20 (%) | Gap (pp) | Var. (pp²) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['phase']} | {row['noise_regime']} | {row['reference']} | {row['attack']} | {row['n']} | {row['test_accuracy_pct']} | {row['worst20_pct']} | {row['gap_pp']} | {row['variance_pp2']} |"
        )
    lines.extend(
        [
            "",
            "## Limite verrouillée",
            "",
            "L'API Byzantine end-to-end actuelle ne fournit pas une attaque `model_replacement` distincte. BF×10 est conservée comme attaque de grande norme, sans être renommée ni interprétée comme un model replacement. Aucun claim sur model replacement ne sera formulé avant son implémentation dédiée.",
            "",
            "Les diagnostics suffixés `oracle` servent uniquement à évaluer le mécanisme. Ils ne sont jamais utilisés pour construire la référence, les poids ou la sortie du protocole.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evaluate-gate")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.matrix)
    if args.evaluate_gate:
        evidence = evaluate_gate(campaign, args.evaluate_gate)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"{args.evaluate_gate}_gate_evidence.json"
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(path)
        return
    state = analyze_campaign(
        campaign, output_dir=args.output_dir, report_path=args.report
    )
    print(json.dumps(state["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
