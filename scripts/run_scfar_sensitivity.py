#!/usr/bin/env python3
"""Empirical replace-one audit for Sensitivity-Controlled FAR.

This simulator is intentionally independent of neural-network training.  It
measures the three terms appearing in the sensitivity proof:

* displacement of the robust reference ``F``;
* L1 drift of the softmax weights;
* displacement of the final deterministic aggregate.

The output is a falsification aid, not a proof of differential privacy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.sc_partial_far_dp import (
    alpha_max_for_weight_factor,
    bounded_distance_scores,
    clip_rows,
)
from robustness.aggregators import aggregate_vectors


def scfar(vectors, *, method, f, clip_norm, distance_clip, alpha, kappa_w):
    clipped, _ = clip_rows(vectors, clip_norm)
    reference = aggregate_vectors(clipped, method, num_byzantine=f)
    distances = torch.linalg.vector_norm(clipped - reference, dim=1)
    scores = bounded_distance_scores(distances, distance_clip)
    alpha_effective = min(alpha, alpha_max_for_weight_factor(len(vectors), kappa_w))
    weights = torch.softmax(alpha_effective * scores, dim=0)
    aggregate = (weights[:, None] * clipped).sum(0)
    return aggregate, reference, weights, alpha_effective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default="5,10,20,40")
    parser.add_argument("--methods", default="mean,cm,rfa,cm_nnm")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--distance-clip", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=2.0)
    parser.add_argument("--byzantine-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/scpfar/sensitivity")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    records = []
    for n in [int(value) for value in args.n.split(",")]:
        if args.kappa_w >= n:
            raise ValueError(f"kappa_w={args.kappa_w} must be smaller than n={n}")
        f = min(int(args.byzantine_fraction * n), max(0, (n - 1) // 2))
        for method in args.methods.split(","):
            for trial in range(args.trials):
                first = torch.randn(n, args.dimension, generator=generator)
                first, _ = clip_rows(first, args.clip)
                second = first.clone()
                replacement = torch.randn(args.dimension, generator=generator)
                replacement = replacement * min(
                    1.0, args.clip / float(torch.linalg.vector_norm(replacement))
                )
                second[0] = replacement
                a, r, q, alpha_eff = scfar(
                    first,
                    method=method,
                    f=f,
                    clip_norm=args.clip,
                    distance_clip=args.distance_clip,
                    alpha=args.alpha,
                    kappa_w=args.kappa_w,
                )
                ap, rp, qp, _ = scfar(
                    second,
                    method=method,
                    f=f,
                    clip_norm=args.clip,
                    distance_clip=args.distance_clip,
                    alpha=args.alpha,
                    kappa_w=args.kappa_w,
                )
                records.append(
                    {
                        "n": n,
                        "method": method,
                        "trial": trial,
                        "f": f,
                        "alpha_effective": alpha_eff,
                        "aggregate_shift_over_c": float(torch.linalg.vector_norm(a - ap))
                        / args.clip,
                        "reference_shift": float(torch.linalg.vector_norm(r - rp)),
                        "weight_l1_shift": float(torch.linalg.vector_norm(q - qp, ord=1)),
                        "max_weight": float(q.max()),
                    }
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "replace_one_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for n in sorted({record["n"] for record in records}):
        for method in sorted({record["method"] for record in records}):
            subset = [
                record
                for record in records
                if record["n"] == n and record["method"] == method
            ]
            key = f"n={n}/{method}"
            summary[key] = {
                field: max(float(record[field]) for record in subset)
                for field in (
                    "aggregate_shift_over_c",
                    "reference_shift",
                    "weight_l1_shift",
                    "max_weight",
                )
            }
    with (output / "summary.json").open("w") as handle:
        json.dump({"config": vars(args), "maxima": summary}, handle, indent=2)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
