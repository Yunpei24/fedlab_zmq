#!/usr/bin/env python3
"""Audit the deterministic mass attenuation inherited by the K4b predictor.

This development-only diagnostic regenerates past transcript features from
the already-consumed K5-v2 development seeds. It never opens the reserved
holdout and never generates a current-round privileged target. Its purpose is
to check whether the large K5 one-dimensional coefficient is numerically
compatible with the public denominator max(m_min, accepted_mass).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gaussian_aware_reference_g0g_k5_tp_v2 as k5  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

DEFAULT_CONFIG = (
    ROOT / "configs/ldp_gradient_far/k5_v2/"
    "gaussian_aware_reference_g0g_k5_tp_v2.yaml"
)
DEFAULT_K5_RESULTS = (
    ROOT / "results/ldp_gradient_far/" "gaussian_aware_reference_g0g_k5_tp_v2_mps_v2"
)
DEFAULT_OUTPUT = (
    ROOT / "output/analysis/" "Gaussian_Aware_G0g_K6_Mass_Attenuation_Development.json"
)


def _quantile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated empirical quantile."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(value) for value in values)
    position = probability * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - float(lower)
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float]:
    """Summarize one finite scalar sample."""

    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("summary values must be non-empty and finite")
    return {
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
    }


@torch.no_grad()
def run(config_path: Path, k5_results: Path, output: Path) -> dict[str, Any]:
    """Regenerate K5 past contexts on MPS and audit their accepted masses."""

    config, amendment = k5._load_amended_config(config_path)
    k5._validate_config(config, amendment)
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps":
        raise RuntimeError("This scientific diagnostic refuses a non-MPS runtime")
    predictor = json.loads(
        (k5_results / "frozen_predictor.json").read_text(encoding="utf-8")
    )
    one = predictor["one_dimensional_control"]
    learned_effective_multiplier = float(one["coefficient"]) / float(
        one["feature_scale"]
    )
    k2_calibration, temporal_calibration, _ = k5.v1._load_calibrations(config)
    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    minimum = float(config["features"]["minimum_accepted_mass"])
    length = int(config["features"]["history_length"])

    per_round_mass: list[float] = []
    per_round_contraction: list[float] = []
    total_window_mass: list[float] = []
    rolling_contraction: list[float] = []
    pooled_contraction: list[float] = []
    constant_direction_pool_gain: list[float] = []
    rows: list[dict[str, Any]] = []
    for context in k5._snapshot_contexts(
        config,
        k2_calibration,
        temporal_calibration,
        split="evaluation_development_mass_audit",
        seeds=seeds,
    ):
        diagnostics = context["feature_diagnostics"]
        masses = [float(value) for value in diagnostics["accepted_mass_chronological"]]
        denominators = [
            float(value) for value in diagnostics["denominator_chronological"]
        ]
        if len(masses) != length or len(denominators) != length:
            raise RuntimeError("Unexpected K5 history length in mass diagnostic")
        contractions = [
            mass / denominator if denominator > 0.0 else 0.0
            for mass, denominator in zip(masses, denominators, strict=True)
        ]
        window_mass = sum(masses)
        pool_denominator = max(float(length), window_mass)
        pool_factor = window_mass / pool_denominator
        rolling_factor = statistics.fmean(contractions)
        gain = pool_factor / rolling_factor if rolling_factor > 0.0 else math.inf
        if not math.isfinite(gain):
            raise RuntimeError("Zero rolling contraction encountered")
        per_round_mass.extend(masses)
        per_round_contraction.extend(contractions)
        total_window_mass.append(window_mass)
        rolling_contraction.append(rolling_factor)
        pooled_contraction.append(pool_factor)
        constant_direction_pool_gain.append(gain)
        rows.append(
            {
                "history_id": str(context["history_id"]),
                "outer_seed": int(context["cell"]["seed"]),
                "assessment_round": int(context["round_index"]),
                "accepted_masses": masses,
                "denominators": denominators,
                "window_mass": window_mass,
                "rolling_contraction": rolling_factor,
                "pooled_contraction": pool_factor,
                "constant_direction_pool_gain": gain,
            }
        )

    result = {
        "scope": "development_only_k5_consumed_evaluation_seeds",
        "device": str(oracle._RUNTIME_DEVICE),
        "dtype": str(oracle._RUNTIME_DTYPE),
        "reserved_holdout_opened": False,
        "current_round_privileged_target_generated": False,
        "outer_seed_count": len(seeds),
        "history_count": len(rows),
        "history_length": length,
        "k4b_minimum_accepted_mass_per_round": minimum,
        "k6_provisional_pooled_mass_floor_per_window": length,
        "learned_k5_1d_effective_multiplier": learned_effective_multiplier,
        "per_round_accepted_mass": _summary(per_round_mass),
        "per_round_contraction_mass_over_denominator": _summary(per_round_contraction),
        "window_total_accepted_mass": _summary(total_window_mass),
        "rolling_constant_direction_contraction": _summary(rolling_contraction),
        "pooled_constant_direction_contraction": _summary(pooled_contraction),
        "constant_direction_pooled_over_k4b_gain_proxy": _summary(
            constant_direction_pool_gain
        ),
        "fraction_k4b_floor_active_per_round": statistics.fmean(
            float(mass < minimum) for mass in per_round_mass
        ),
        "fraction_k6_pool_floor_active_per_window": statistics.fmean(
            float(mass < float(length)) for mass in total_window_mass
        ),
        "interpretation": (
            "The pooled/K4b ratio is exact only under a constant direction over "
            "the four rounds. It is a denominator-attenuation diagnostic, not "
            "an accuracy or target-MSE result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k5-results", type=Path, default=DEFAULT_K5_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.k5_results, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
