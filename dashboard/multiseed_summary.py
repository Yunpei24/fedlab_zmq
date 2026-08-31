"""Utilities for scientifically comparable multi-seed dashboard summaries.

The Streamlit application is intentionally kept out of this module so that the
method identity and aggregation rules can be unit-tested without starting a UI.
All fairness metrics are read at the final completed round of each selected run.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "Méthode",
    "Client Acc. (%)",
    "Test Acc. (%)",
    "Var (pp²)",
    "Worst-20 (%)",
    "Gap Δk (pp)",
]

BEST_DIRECTION = {
    "Client Acc. (%)": "max",
    "Test Acc. (%)": "max",
    "Var (pp²)": "min",
    "Worst-20 (%)": "max",
    "Gap Δk (pp)": "min",
}


@dataclass(frozen=True)
class RunSummary:
    """Final-round values required by the internship-style comparison table."""

    method: str
    seed: int | None
    client_accuracy_pct: float
    test_accuracy_pct: float
    variance_pct2: float
    worst20_pct: float
    gap_pct: float
    condition: str
    run_dir: pathlib.Path


def _format_number(value: object) -> str:
    """Format a scalar without meaningless trailing zeros."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(numeric):
        return "?"
    return f"{numeric:g}"


def _algorithm(data: Mapping[str, object]) -> str:
    direct = data.get("algorithm")
    if direct:
        return str(direct).lower()

    config = data.get("config")
    if isinstance(config, Mapping) and config.get("algorithm"):
        return str(config["algorithm"]).lower()

    experiment = data.get("experiment")
    if isinstance(experiment, Mapping):
        training = experiment.get("training")
        if isinstance(training, Mapping) and training.get("algorithm"):
            return str(training["algorithm"]).lower()
        if experiment.get("algorithm"):
            return str(experiment["algorithm"]).lower()
    return "unknown"


def _algo_config(data: Mapping[str, object]) -> Mapping[str, object]:
    config = data.get("config")
    if not isinstance(config, Mapping):
        return {}
    nested = config.get("algo_config")
    return nested if isinstance(nested, Mapping) else config


def _path_parameter(run_dir: pathlib.Path, prefix: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(prefix)}_(m?\d+(?:p\d+)?)$")
    for part in reversed(run_dir.parts):
        match = pattern.match(part)
        if not match:
            continue
        encoded = match.group(1).replace("p", ".")
        if encoded.startswith("m"):
            encoded = "-" + encoded[1:]
        try:
            return float(encoded)
        except ValueError:
            return None
    return None


def method_label(data: Mapping[str, object], run_dir: pathlib.Path) -> str:
    """Return a parameter-aware method identity shared by all its seeds.

    FAR's tilt is deliberately part of the identity: FAR(alpha=-0.1) and
    FAR(alpha=0.4) are different experimental methods and must never collapse
    into one dashboard series.
    """

    algo = _algorithm(data)
    config = _algo_config(data)

    if algo == "far":
        alpha = config.get("far_alpha")
        if alpha is None:
            alpha = _path_parameter(run_dir, "far_alpha")
        return f"FAR (α={_format_number(alpha)})"

    if algo in {"dpfar", "dp_far"}:
        alpha = config.get("far_alpha")
        if alpha is None:
            alpha = _path_parameter(run_dir, "far_alpha")
        return f"DP-FAR (α={_format_number(alpha)})"

    if algo in {"scpartialfar", "sc_partial_far", "fairpartfar_dp"}:
        tau = config.get("tilt_tau", config.get("far_alpha"))
        return f"SC-Partial-FAR-DP (τ={_format_number(tau)})"

    if algo == "fedfair":
        fairness_lambda = config.get("fairness_lambda")
        return f"FedFair (λ={_format_number(fairness_lambda)})"

    if algo in {"qffl", "q-ffl"}:
        q_value = config.get("q")
        return f"q-FFL (q={_format_number(q_value)})"

    if algo == "fedfdp":
        fairness_lambda = config.get("fairness_lambda", config.get("lambda"))
        if fairness_lambda is None:
            return "FedFDP"
        return f"FedFDP (λ={_format_number(fairness_lambda)})"

    display_names = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
    }
    return display_names.get(algo, algo.upper() if algo != "unknown" else "Unknown")


def run_seed(data: Mapping[str, object], run_dir: pathlib.Path) -> int | None:
    config = data.get("config")
    if isinstance(config, Mapping) and config.get("seed") is not None:
        try:
            return int(config["seed"])
        except (TypeError, ValueError):
            pass

    for part in reversed(run_dir.parts):
        match = re.fullmatch(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
        match = re.search(r"_s(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def experimental_condition(run_dir: pathlib.Path) -> str:
    """Infer the comparable scenario represented by a reproduction path."""

    parts = list(run_dir.parts)
    for index, part in enumerate(parts):
        if not re.match(r"^exp\d+_", part):
            continue
        condition = []
        if index > 0 and re.search(r"(?:fidelity|revision|protocol).*v\d+", parts[index - 1]):
            condition.append(parts[index - 1])
        condition.extend(parts[index : min(index + 3, len(parts))])
        return " / ".join(condition)
    return "Sélection courante"


def run_selector_label(data: Mapping[str, object], run_dir: pathlib.Path) -> str:
    """A unique, readable label for one concrete method/seed run."""

    seed = run_seed(data, run_dir)
    condition = experimental_condition(run_dir)
    seed_text = f"seed {seed}" if seed is not None else run_dir.name
    return f"{method_label(data, run_dir)} | {condition} | {seed_text}"


def _last_round(rounds: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not rounds:
        raise ValueError("metrics.json contains no completed round")

    def round_index(item: Mapping[str, object]) -> float:
        for key in ("round_num", "round", "t"):
            if item.get(key) is not None:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    continue
        return -1.0

    return max(rounds, key=round_index)


def _metric_percent(
    row: Mapping[str, object], percent_key: str, fraction_key: str, factor: float
) -> float:
    value = row.get(percent_key)
    if value is not None:
        return float(value)
    value = row.get(fraction_key)
    if value is None:
        return float("nan")
    return float(value) * factor


def load_run_summary(run_dir: str | pathlib.Path) -> RunSummary:
    run_path = pathlib.Path(run_dir)
    metrics_path = run_path if run_path.name == "metrics.json" else run_path / "metrics.json"
    with metrics_path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    row = _last_round(data.get("rounds", []))
    actual_run_dir = metrics_path.parent
    return RunSummary(
        method=method_label(data, actual_run_dir),
        seed=run_seed(data, actual_run_dir),
        client_accuracy_pct=_metric_percent(
            row, "client_accuracy_mean_pct", "client_accuracy_mean", 100.0
        ),
        test_accuracy_pct=_metric_percent(row, "test_accuracy_pct", "test_accuracy", 100.0),
        variance_pct2=_metric_percent(
            row, "client_accuracy_variance_pct2", "client_accuracy_variance", 10_000.0
        ),
        worst20_pct=_metric_percent(row, "worst20_accuracy_pct", "worst20_accuracy", 100.0),
        gap_pct=_metric_percent(
            row, "best20_worst20_gap_pct", "best20_worst20_gap", 100.0
        ),
        condition=experimental_condition(actual_run_dir),
        run_dir=actual_run_dir,
    )


def _mean_std(values: Iterable[float]) -> tuple[float, float | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), None
    if array.size == 1:
        return float(array[0]), None
    return float(array.mean()), float(array.std(ddof=1))


def _display(mean: float, std: float | None, decimals: int, include_std: bool) -> str:
    if not math.isfinite(mean):
        return "—"
    if include_std and std is not None and math.isfinite(std):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def summarize_runs(
    summaries: Sequence[RunSummary], *, include_std: bool = True, decimals: int = 3
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Aggregate final-round metrics by method across the selected seeds."""

    grouped: dict[str, list[RunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.method].append(summary)

    rows = []
    seed_map: dict[str, list[int]] = {}
    metric_specs = (
        ("Client Acc. (%)", "client_accuracy_pct"),
        ("Test Acc. (%)", "test_accuracy_pct"),
        ("Var (pp²)", "variance_pct2"),
        ("Worst-20 (%)", "worst20_pct"),
        ("Gap Δk (pp)", "gap_pct"),
    )
    for method in sorted(grouped, key=_method_sort_key):
        method_runs = grouped[method]
        row = {"Méthode": method}
        for column, attribute in metric_specs:
            mean, std = _mean_std(getattr(run, attribute) for run in method_runs)
            row[column] = _display(mean, std, decimals, include_std)
        rows.append(row)
        seed_map[method] = sorted(
            {run.seed for run in method_runs if run.seed is not None}
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), seed_map


def best_value_indices(table: pd.DataFrame) -> dict[str, set[object]]:
    """Identify the scientifically best mean value in every metric column.

    Summary cells can contain ``mean ± sample_std``. Ranking deliberately uses
    the mean only; the uncertainty remains visible but does not change which
    point estimate is highlighted. All ties are retained.
    """

    best: dict[str, set[object]] = {}
    for column, direction in BEST_DIRECTION.items():
        if column not in table.columns:
            continue
        means = pd.to_numeric(
            table[column].astype(str).str.split("±", n=1).str[0].str.strip(),
            errors="coerce",
        )
        valid = means.dropna()
        if valid.empty:
            best[column] = set()
            continue
        optimum = valid.max() if direction == "max" else valid.min()
        best[column] = {
            index
            for index, value in valid.items()
            if np.isclose(float(value), float(optimum), rtol=1e-12, atol=1e-12)
        }
    return best


def style_best_values(table: pd.DataFrame):
    """Return a Styler that bolds every best metric value, including ties."""

    best = best_value_indices(table)

    def styles(data: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame("", index=data.index, columns=data.columns)
        for column, indices in best.items():
            for index in indices:
                result.at[index, column] = "font-weight: 800"
        return result

    return table.style.apply(styles, axis=None)


def _method_sort_key(method: str) -> tuple[int, float, str]:
    base_order = {"FedAvg": 0, "FedFair": 1, "q-FFL": 2, "FAR": 3, "DP-FAR": 4}
    base = method.split(" (", 1)[0]
    match = re.search(r"[ατ]=(-?\d+(?:\.\d+)?)", method)
    parameter = float(match.group(1)) if match else 0.0
    return base_order.get(base, 99), parameter, method
