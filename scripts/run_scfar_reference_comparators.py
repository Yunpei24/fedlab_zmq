#!/usr/bin/env python3
"""S0.5 empirical replace-one audit of SC-FAR reference comparators.

This experiment compares the arithmetic mean, coordinate-wise median (CM),
trimmed mean and RFA/geometric median on fixed-size neighbouring cohorts.  It
measures both the shift of the reference ``F`` and the shift of the complete
deterministic SC-FAR aggregate.

Two kinds of neighbouring cohorts are used:

* random L2-clipped Gaussian cohorts, which describe typical behaviour under
  the chosen synthetic distribution;
* deterministic stress cohorts, which prevent benign random draws from
  hiding the global replace-one instability of robust comparators.

Only the arithmetic mean receives its elementary global reference certificate
``2C/n``.  CM, trimmed mean and RFA are empirical comparators: observed
``1/n``-like scaling is not promoted to a DP certificate.  Independently of
the reference, the final aggregate remains a convex combination of user-level
clipped updates and therefore retains the conservative global bound ``2C``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robustness.aggregators import clip_l2  # noqa: E402
from scripts.run_scfar_sensitivity import (  # noqa: E402
    build_public_anchor,
    scfar_chain,
)

DEFAULT_METHODS = ("mean", "cm", "trmean", "rfa")
COMPARATOR_ALIASES = {
    "mean": "mean",
    "cm": "cm",
    "median": "cm",
    "coordinate_median": "cm",
    "trmean": "trmean",
    "trimmedmean": "trmean",
    "trimmed_mean": "trmean",
    "rfa": "rfa",
    "geometric_median": "rfa",
}


def _parse_values(raw: str, cast) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one comma-separated value is required")
    return values


def _canonical_methods(raw: str) -> list[str]:
    methods: list[str] = []
    for value in raw.split(","):
        key = value.strip().lower()
        if not key:
            continue
        canonical = COMPARATOR_ALIASES.get(key)
        if canonical not in DEFAULT_METHODS:
            raise ValueError(
                f"S0.5 accepts only {', '.join(DEFAULT_METHODS)}; got {value!r}"
            )
        if canonical not in methods:
            methods.append(canonical)
    if not methods:
        raise ValueError("At least one S0.5 comparator is required")
    return methods


def _random_pair(
    *, n: int, dimension: int, clip_norm: float, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, int]:
    first = clip_l2(
        torch.randn(n, dimension, generator=generator, dtype=torch.float64),
        clip_norm,
    )
    replaced = int(torch.randint(n, (1,), generator=generator).item())
    second = first.clone()
    second[replaced] = clip_l2(
        torch.randn(dimension, generator=generator, dtype=torch.float64),
        clip_norm,
    )
    return first, second, replaced


def _signed_axis(count: int, value: float, dimension: int) -> torch.Tensor:
    rows = torch.zeros(count, dimension, dtype=torch.float64)
    if count:
        rows[:, 0] = value
    return rows


def _majority_stress_pair(
    *, n: int, dimension: int, clip_norm: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Neighbouring cohorts that stress CM and the geometric median.

    For odd ``n``, the common ``n-1`` rows contain the same number of ``-C``
    and ``+C`` points.  Replacing the last row changes the strict majority and
    moves the one-dimensional median from ``-C`` to ``+C``.  With even ``n``
    and midpoint CM, the same construction still produces a macroscopic shift
    even though the exact extremal value differs.
    """

    negative_common = (n - 1) // 2
    positive_common = (n - 1) - negative_common
    common = torch.cat(
        (
            _signed_axis(negative_common, -clip_norm, dimension),
            _signed_axis(positive_common, clip_norm, dimension),
        )
    )
    before = torch.cat((common, _signed_axis(1, -clip_norm, dimension)))
    after = torch.cat((common, _signed_axis(1, clip_norm, dimension)))
    return before, after, n - 1


def _trim_boundary_stress_pair(
    *, n: int, dimension: int, clip_norm: float, f: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Neighbouring cohorts crossing both retained trimmed-mean boundaries."""

    kept = n - 2 * f
    if kept < 1:
        raise ValueError("trimmed-mean stress case requires n-2f >= 1")
    common = torch.cat(
        (
            _signed_axis(f, -clip_norm, dimension),
            _signed_axis(kept - 1, 0.0, dimension),
            _signed_axis(f, clip_norm, dimension),
        )
    )
    before = torch.cat((common, _signed_axis(1, -clip_norm, dimension)))
    after = torch.cat((common, _signed_axis(1, clip_norm, dimension)))
    return before, after, n - 1


def _evaluate_pair(
    *,
    first: torch.Tensor,
    second: torch.Tensor,
    replaced: int,
    scenario: str,
    trial: int,
    method: str,
    anchor: torch.Tensor,
    num_byzantine: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    common = {
        "method": method,
        "anchor": anchor,
        "num_byzantine": num_byzantine,
        "clip_norm": args.clip,
        "distance_clip": args.distance_clip,
        "alpha": args.alpha,
        "kappa_w": args.kappa_w,
        "tau": args.tau,
        "huber_gamma": 1.0,
        "huber_num_steps": 1,
    }
    before = scfar_chain(first, **common)
    after = scfar_chain(second, **common)
    reference_shift = float(
        torch.linalg.vector_norm(before["reference"] - after["reference"])
    )
    aggregate_shift = float(
        torch.linalg.vector_norm(before["aggregate"] - after["aggregate"])
    )
    mean_reference_bound = 2.0 * args.clip / len(first) if method == "mean" else None
    return {
        "audit": "S0.5_reference_comparators",
        "scenario": scenario,
        "trial": trial,
        "n": len(first),
        "dimension": args.dimension,
        "method": method,
        "num_byzantine_f": num_byzantine,
        "replaced_index": replaced,
        "clip_norm_C": args.clip,
        "distance_clip_Dmax": args.distance_clip,
        "alpha_requested": args.alpha,
        "alpha_effective": before["alpha_effective"],
        "kappa_w": args.kappa_w,
        "reference_certificate": (
            "2C_over_n_arithmetic_mean" if method == "mean" else "none_empirical_only"
        ),
        "reference_bound": mean_reference_bound,
        "reference_shift": reference_shift,
        "reference_shift_over_C": reference_shift / args.clip,
        "reference_certificate_holds": (
            reference_shift <= mean_reference_bound + args.tolerance
            if mean_reference_bound is not None
            else None
        ),
        "aggregate_global_bound": 2.0 * args.clip,
        "aggregate_shift": aggregate_shift,
        "aggregate_shift_over_2C": aggregate_shift / (2.0 * args.clip),
        "aggregate_global_bound_holds": aggregate_shift
        <= 2.0 * args.clip + args.tolerance,
        "max_weight_before": float(before["weights"].max()),
        "max_weight_after": float(after["weights"].max()),
    }


def run_audit(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_values = _parse_values(args.n, int)
    methods = _canonical_methods(args.methods)
    if args.dimension < 1 or args.trials < 1:
        raise ValueError("dimension and trials must be positive")
    if args.clip <= 0 or args.distance_clip <= 0:
        raise ValueError("clip and distance_clip must be positive")
    if not 0.0 <= args.byzantine_fraction < 0.5:
        raise ValueError("byzantine_fraction must lie in [0, 0.5)")

    generator = torch.Generator().manual_seed(args.seed)
    anchor = build_public_anchor(
        dimension=args.dimension,
        mode=args.anchor_mode,
        norm=args.anchor_norm,
        seed=args.anchor_seed,
    )
    records: list[dict[str, Any]] = []
    for n in n_values:
        if n < 3:
            raise ValueError("S0.5 requires n >= 3")
        if args.kappa_w >= n:
            raise ValueError(f"kappa_w={args.kappa_w} must be smaller than n={n}")
        f = min(int(math.floor(args.byzantine_fraction * n)), (n - 1) // 2)
        pairs: list[tuple[str, int, torch.Tensor, torch.Tensor, int]] = []
        for trial in range(args.trials):
            first, second, replaced = _random_pair(
                n=n,
                dimension=args.dimension,
                clip_norm=args.clip,
                generator=generator,
            )
            pairs.append(("random_l2", trial, first, second, replaced))
        for scenario, builder in (
            ("majority_axis_stress", _majority_stress_pair),
            ("trim_boundary_stress", _trim_boundary_stress_pair),
        ):
            kwargs = {
                "n": n,
                "dimension": args.dimension,
                "clip_norm": args.clip,
            }
            if scenario == "trim_boundary_stress":
                kwargs["f"] = f
            first, second, replaced = builder(**kwargs)
            pairs.append((scenario, -1, first, second, replaced))

        for method in methods:
            for scenario, trial, first, second, replaced in pairs:
                records.append(
                    _evaluate_pair(
                        first=first,
                        second=second,
                        replaced=replaced,
                        scenario=scenario,
                        trial=trial,
                        method=method,
                        anchor=anchor,
                        num_byzantine=f,
                        args=args,
                    )
                )

    groups: dict[str, Any] = {}
    for n in n_values:
        for method in methods:
            for scenario in (
                "random_l2",
                "majority_axis_stress",
                "trim_boundary_stress",
            ):
                subset = [
                    row
                    for row in records
                    if row["n"] == n
                    and row["method"] == method
                    and row["scenario"] == scenario
                ]
                groups[f"n={n}/{method}/{scenario}"] = {
                    "n": n,
                    "method": method,
                    "scenario": scenario,
                    "samples": len(subset),
                    "reference_certificate": subset[0]["reference_certificate"],
                    "reference_bound": subset[0]["reference_bound"],
                    "reference_shift": {
                        "mean": statistics.fmean(
                            float(row["reference_shift"]) for row in subset
                        ),
                        "max": max(float(row["reference_shift"]) for row in subset),
                    },
                    "aggregate_shift": {
                        "mean": statistics.fmean(
                            float(row["aggregate_shift"]) for row in subset
                        ),
                        "max": max(float(row["aggregate_shift"]) for row in subset),
                    },
                    "aggregate_global_bound": 2.0 * args.clip,
                    "aggregate_bound_violations": sum(
                        not bool(row["aggregate_global_bound_holds"]) for row in subset
                    ),
                }

    summary = {
        "schema_version": 1,
        "audit": "S0.5_reference_comparators",
        "status": "empirical_falsification_audit_not_dp_certificate",
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "methods": methods,
        "interpretation": {
            "mean": "global reference certificate 2C/n; not Byzantine robust",
            "cm": "empirical robust comparator; no O(C/n) certificate claimed",
            "trmean": "empirical robust comparator; no O(C/n) certificate claimed",
            "rfa": "empirical robust comparator; no O(C/n) certificate claimed",
            "final_aggregate": "conservative global replace-one bound 2C",
        },
        "groups": groups,
        "all_mean_certificates_hold": all(
            row["reference_certificate_holds"] is not False
            for row in records
            if row["method"] == "mean"
        ),
        "all_aggregate_2C_bounds_hold": all(
            bool(row["aggregate_global_bound_holds"]) for row in records
        ),
    }
    return records, summary


def write_outputs(
    records: list[dict[str, Any]], summary: dict[str, Any], output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "replace_one_comparators.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", default="5,10,20,40")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--distance-clip", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=2.0)
    parser.add_argument("--byzantine-fraction", type=float, default=0.2)
    parser.add_argument(
        "--anchor-mode", choices=("zero", "fixed_random"), default="zero"
    )
    parser.add_argument("--anchor-norm", type=float, default=0.5)
    parser.add_argument("--anchor-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--output", default="results/scpfar/sensitivity_s0_5")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, summary = run_audit(args)
    output = Path(args.output)
    write_outputs(records, summary, output)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
