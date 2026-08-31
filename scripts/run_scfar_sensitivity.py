#!/usr/bin/env python3
"""End-to-end replace-one audit for Sensitivity-Controlled FAR (S0.3).

The script isolates the deterministic SC-FAR chain from neural-network
training and measures, for neighbouring fixed-size cohorts,

``cohort -> F -> bounded scores -> softmax weights -> aggregate``.

``centered_clipping`` (``F_CC``) is the primary reference and
``regularized_huber`` is the finite-step ablation. Both use the same public
anchor, generated independently of every audited cohort. Mean, coordinate
median, RFA and CM(NNM) are retained as empirical comparators. The elementary
``2C/n`` stability certificate is reported for the arithmetic mean, while no
unproved ``O(C/n)`` certificate is assigned to the robust comparators.

The CSV contains every replace-one pair. The JSON contains the public
configuration, method roles, theoretical certificates when available and
violation counts. These outputs are falsification diagnostics, not a proof
of differential privacy and not DP-safe telemetry for a production run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.sc_partial_far_dp import (  # noqa: E402
    alpha_max_for_weight_factor,
    bounded_distance_scores,
    certified_scfar_sensitivity,
    clip_rows,
    softmax_weight_factor_bound,
)
from robustness.aggregators import (  # noqa: E402
    aggregate_vectors,
    centered_clipping,
    clip_l2,
    regularized_huber_reference,
)


METHOD_ALIASES = {
    "cc": "centered_clipping",
    "f_cc": "centered_clipping",
    "centered_clipping": "centered_clipping",
    "huber": "regularized_huber",
    "huber_regularized": "regularized_huber",
    "regularized_huber": "regularized_huber",
    "median": "cm",
    "coordinate_median": "cm",
    "cm": "cm",
    "geometric_median": "rfa",
    "rfa": "rfa",
    "cm(nnm)": "cm_nnm",
    "cm_nnm": "cm_nnm",
    "mean": "mean",
}

METHOD_ROLES = {
    "centered_clipping": "primary_certified_reference",
    "regularized_huber": "certified_finite_step_ablation",
    "mean": "empirical_comparator",
    "cm": "empirical_robust_comparator",
    "rfa": "empirical_robust_comparator",
    "cm_nnm": "empirical_robust_comparator",
}


def _canonical_method(name: str) -> str:
    key = name.strip().lower()
    if key not in METHOD_ALIASES:
        available = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown reference {name!r}. Available aliases: {available}")
    return METHOD_ALIASES[key]


def _parse_csv_values(raw: str, cast) -> list[Any]:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one comma-separated value is required")
    return values


def build_public_anchor(
    *,
    dimension: int,
    mode: str,
    norm: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return an anchor generated without reading the audited cohort.

    ``zero`` is the cleanest first-round construction. ``fixed_random`` is a
    reproducible stress test whose seed and radius are public parameters.
    Neither mode computes an anchor statistic from the current clients.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if norm < 0:
        raise ValueError("anchor_norm must be non-negative")
    if mode == "zero":
        return torch.zeros(dimension, dtype=dtype)
    if mode != "fixed_random":
        raise ValueError("anchor_mode must be 'zero' or 'fixed_random'")
    generator = torch.Generator().manual_seed(int(seed))
    direction = torch.randn(dimension, generator=generator, dtype=dtype)
    direction_norm = torch.linalg.vector_norm(direction)
    if float(direction_norm) == 0.0 or norm == 0.0:
        return torch.zeros(dimension, dtype=dtype)
    return direction * (float(norm) / direction_norm)


def _reference_and_certificate(
    vectors: torch.Tensor,
    *,
    method: str,
    anchor: torch.Tensor,
    clip_norm: float,
    tau: float,
    huber_gamma: float,
    huber_num_steps: int,
    num_byzantine: int,
) -> tuple[torch.Tensor, float | None, str, dict[str, float | int]]:
    """Evaluate ``F`` and return its absolute replace-one certificate."""

    n = int(vectors.shape[0])
    effective_radius = min(float(clip_norm), float(tau))
    if method == "centered_clipping":
        reference = centered_clipping(vectors, anchor=anchor, tau=tau)
        return (
            reference,
            2.0 * effective_radius / n,
            "centered_clipping_global",
            {"reference_effective_radius": effective_radius},
        )
    if method == "regularized_huber":
        reference, diagnostics = regularized_huber_reference(
            vectors,
            anchor=anchor,
            tau=tau,
            gamma=huber_gamma,
            num_steps=huber_num_steps,
            return_diagnostics=True,
        )
        contraction = float(diagnostics["huber_contraction"])
        stability = (
            2.0
            * effective_radius
            / (float(huber_gamma) * n)
            * (1.0 - contraction ** int(huber_num_steps))
        )
        return (
            reference,
            stability,
            "regularized_huber_fixed_steps",
            {
                "reference_effective_radius": effective_radius,
                **{key: value for key, value in diagnostics.items()},
            },
        )

    if method == "mean":
        # This certificate is useful as a stability sanity check only: the
        # arithmetic mean is not Byzantine robust. For replace-one cohorts,
        # ||mean(U)-mean(U')|| = ||u_k-u'_k||/n <= 2C/n.
        return (
            vectors.mean(dim=0),
            2.0 * float(clip_norm) / n,
            "arithmetic_mean_global_not_byzantine_robust",
            {},
        )

    reference = aggregate_vectors(
        vectors,
        method,
        num_byzantine=num_byzantine,
    )
    return reference, None, "none_comparator_only", {}


def scfar_chain(
    vectors: torch.Tensor,
    *,
    method: str,
    anchor: torch.Tensor,
    num_byzantine: int,
    clip_norm: float,
    distance_clip: float,
    alpha: float,
    kappa_w: float,
    tau: float,
    huber_gamma: float,
    huber_num_steps: int,
) -> dict[str, Any]:
    """Evaluate all deterministic objects in one SC-FAR execution."""

    clipped, clip_factors = clip_rows(vectors, clip_norm)
    reference, stability, certificate, reference_diagnostics = (
        _reference_and_certificate(
            clipped,
            method=method,
            anchor=anchor,
            clip_norm=clip_norm,
            tau=tau,
            huber_gamma=huber_gamma,
            huber_num_steps=huber_num_steps,
            num_byzantine=num_byzantine,
        )
    )
    distances = torch.linalg.vector_norm(clipped - reference, dim=1)
    scores = bounded_distance_scores(distances, distance_clip)
    alpha_max = alpha_max_for_weight_factor(len(vectors), kappa_w)
    alpha_effective = min(float(alpha), alpha_max)
    weights = torch.softmax(alpha_effective * scores, dim=0)
    aggregate = (weights[:, None] * clipped).sum(dim=0)
    analytical_kappa = softmax_weight_factor_bound(len(vectors), alpha_effective)
    return {
        "clipped": clipped,
        "clip_factors": clip_factors,
        "reference": reference,
        "reference_stability": stability,
        "reference_certificate": certificate,
        "reference_diagnostics": reference_diagnostics,
        "distances": distances,
        "scores": scores,
        "weights": weights,
        "aggregate": aggregate,
        "alpha_effective": alpha_effective,
        "alpha_max": alpha_max,
        "analytical_kappa": analytical_kappa,
    }


def _replace_one_pair(
    *,
    n: int,
    dimension: int,
    clip_norm: float,
    generator: torch.Generator,
    replacement_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = clip_l2(
        torch.randn(n, dimension, generator=generator, dtype=torch.float64),
        clip_norm,
    )
    second = first.clone()
    second[replacement_index] = clip_l2(
        torch.randn(dimension, generator=generator, dtype=torch.float64),
        clip_norm,
    )
    return first, second


def _certificate_bounds(
    *,
    n: int,
    clip_norm: float,
    distance_clip: float,
    alpha_effective: float,
    analytical_kappa: float,
    reference_stability: float | None,
) -> dict[str, float | None]:
    if reference_stability is None:
        return {
            "reference_bound": None,
            "unchanged_score_linf_bound": None,
            "replaced_score_bound": None,
            "score_l1_bound": None,
            "weight_l1_bound": None,
            "aggregate_bound": 2.0 * clip_norm,
            "aggregate_refined_bound": None,
        }

    eta_reference = min(1.0, reference_stability / distance_clip)
    eta_direct = min(1.0, (2.0 * clip_norm + reference_stability) / distance_clip)
    score_l1_bound = eta_direct + (n - 1) * eta_reference
    weight_l1_bound = (
        2.0 * alpha_effective * ((analytical_kappa / n) * eta_direct + eta_reference)
    )
    aggregate_refined = min(
        2.0 * clip_norm,
        2.0 * clip_norm * analytical_kappa / n + clip_norm * weight_l1_bound,
    )
    aggregate_certified = certified_scfar_sensitivity(
        n=n,
        clip_norm=clip_norm,
        distance_clip=distance_clip,
        alpha=alpha_effective,
        kappa_bound=analytical_kappa,
        reference_stability=reference_stability,
    )
    return {
        "reference_bound": reference_stability,
        "unchanged_score_linf_bound": eta_reference,
        "replaced_score_bound": eta_direct,
        "score_l1_bound": score_l1_bound,
        "weight_l1_bound": weight_l1_bound,
        "aggregate_bound": aggregate_certified,
        "aggregate_refined_bound": aggregate_refined,
    }


def audit_replace_one(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Run S0.3 and return trial records plus a serialisable summary."""

    n_values = _parse_csv_values(args.n, int)
    methods = [_canonical_method(value) for value in args.methods.split(",")]
    methods = list(dict.fromkeys(methods))
    if args.trials < 1:
        raise ValueError("trials must be positive")
    if args.clip <= 0 or args.distance_clip <= 0 or args.tau <= 0:
        raise ValueError("clip, distance_clip and tau must be positive")
    if args.huber_gamma <= 0 or args.huber_num_steps < 1:
        raise ValueError("Huber requires gamma>0 and a positive fixed step count")
    if not 0.0 <= args.byzantine_fraction < 0.5:
        raise ValueError("byzantine_fraction must lie in [0, 0.5)")

    anchor = build_public_anchor(
        dimension=args.dimension,
        mode=args.anchor_mode,
        norm=args.anchor_norm,
        seed=args.anchor_seed,
    )
    anchor_digest = hashlib.sha256(anchor.numpy().tobytes()).hexdigest()
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict] = []

    for n in n_values:
        if n < 2:
            raise ValueError("Every cohort size n must be at least two")
        if args.kappa_w >= n:
            raise ValueError(f"kappa_w={args.kappa_w} must be smaller than n={n}")
        f = min(int(args.byzantine_fraction * n), max(0, (n - 1) // 2))
        for method in methods:
            for trial in range(args.trials):
                replaced = trial % n
                first, second = _replace_one_pair(
                    n=n,
                    dimension=args.dimension,
                    clip_norm=args.clip,
                    generator=generator,
                    replacement_index=replaced,
                )
                common = {
                    "method": method,
                    "anchor": anchor,
                    "num_byzantine": f,
                    "clip_norm": args.clip,
                    "distance_clip": args.distance_clip,
                    "alpha": args.alpha,
                    "kappa_w": args.kappa_w,
                    "tau": args.tau,
                    "huber_gamma": args.huber_gamma,
                    "huber_num_steps": args.huber_num_steps,
                }
                before = scfar_chain(first, **common)
                after = scfar_chain(second, **common)
                stability = before["reference_stability"]
                if stability != after["reference_stability"]:
                    raise RuntimeError(
                        "A public certificate cannot depend on the cohort"
                    )
                bounds = _certificate_bounds(
                    n=n,
                    clip_norm=args.clip,
                    distance_clip=args.distance_clip,
                    alpha_effective=before["alpha_effective"],
                    analytical_kappa=before["analytical_kappa"],
                    reference_stability=stability,
                )

                score_delta = (before["scores"] - after["scores"]).abs()
                unchanged_mask = torch.ones(n, dtype=torch.bool)
                unchanged_mask[replaced] = False
                unchanged_score_linf = float(score_delta[unchanged_mask].max())
                reference_shift = float(
                    torch.linalg.vector_norm(before["reference"] - after["reference"])
                )
                weight_l1_shift = float(
                    torch.linalg.vector_norm(
                        before["weights"] - after["weights"], ord=1
                    )
                )
                aggregate_shift = float(
                    torch.linalg.vector_norm(before["aggregate"] - after["aggregate"])
                )
                tolerance = float(args.certificate_tolerance)

                record = {
                    "n": n,
                    "method": method,
                    "method_role": METHOD_ROLES[method],
                    "trial": trial,
                    "replaced_index": replaced,
                    "dimension": args.dimension,
                    "f": f,
                    "public_anchor_mode": args.anchor_mode,
                    "public_anchor_norm": float(torch.linalg.vector_norm(anchor)),
                    "public_anchor_sha256": anchor_digest,
                    "clip_norm_C": args.clip,
                    "distance_clip_Dmax": args.distance_clip,
                    "reference_tau": args.tau,
                    "huber_gamma": args.huber_gamma,
                    "huber_num_steps": args.huber_num_steps,
                    "alpha_requested": args.alpha,
                    "alpha_effective": before["alpha_effective"],
                    "alpha_max": before["alpha_max"],
                    "analytical_kappa": before["analytical_kappa"],
                    "configured_kappa_w": args.kappa_w,
                    "reference_certificate": before["reference_certificate"],
                    "huber_step_size": before["reference_diagnostics"].get(
                        "huber_step_size"
                    ),
                    "huber_contraction": before["reference_diagnostics"].get(
                        "huber_contraction"
                    ),
                    "huber_gradient_residual_before": before[
                        "reference_diagnostics"
                    ].get("huber_gradient_residual"),
                    "huber_gradient_residual_after": after["reference_diagnostics"].get(
                        "huber_gradient_residual"
                    ),
                    "reference_shift": reference_shift,
                    "score_linf_shift": float(score_delta.max()),
                    "score_l1_shift": float(score_delta.sum()),
                    "unchanged_score_linf_shift": unchanged_score_linf,
                    "replaced_score_shift": float(score_delta[replaced]),
                    "weight_l1_shift": weight_l1_shift,
                    "aggregate_shift": aggregate_shift,
                    "aggregate_shift_over_C": aggregate_shift / args.clip,
                    "max_weight_before": float(before["weights"].max()),
                    "max_weight_after": float(after["weights"].max()),
                    "reference_bound": bounds["reference_bound"],
                    "unchanged_score_linf_bound": bounds["unchanged_score_linf_bound"],
                    "replaced_score_bound": bounds["replaced_score_bound"],
                    "score_l1_bound": bounds["score_l1_bound"],
                    "weight_l1_bound": bounds["weight_l1_bound"],
                    "aggregate_bound": bounds["aggregate_bound"],
                    "aggregate_refined_bound": bounds["aggregate_refined_bound"],
                }
                for measured, bound in (
                    ("reference", "reference_bound"),
                    ("unchanged_score_linf", "unchanged_score_linf_bound"),
                    ("replaced_score", "replaced_score_bound"),
                    ("score_l1", "score_l1_bound"),
                    ("weight_l1", "weight_l1_bound"),
                    ("aggregate", "aggregate_bound"),
                    ("aggregate_refined", "aggregate_refined_bound"),
                ):
                    measured_key = {
                        "reference": "reference_shift",
                        "unchanged_score_linf": "unchanged_score_linf_shift",
                        "replaced_score": "replaced_score_shift",
                        "score_l1": "score_l1_shift",
                        "weight_l1": "weight_l1_shift",
                        "aggregate": "aggregate_shift",
                        "aggregate_refined": "aggregate_shift",
                    }[measured]
                    bound_value = record[bound]
                    record[f"{measured}_certificate_holds"] = (
                        None
                        if bound_value is None
                        else bool(record[measured_key] <= bound_value + tolerance)
                    )
                records.append(record)

    summary = _summarise(records, args, anchor, anchor_digest, methods)
    return records, summary


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarise(
    records: list[dict],
    args: argparse.Namespace,
    anchor: torch.Tensor,
    anchor_digest: str,
    methods: list[str],
) -> dict:
    groups: dict[str, dict] = {}
    fields = (
        "reference_shift",
        "score_linf_shift",
        "score_l1_shift",
        "unchanged_score_linf_shift",
        "replaced_score_shift",
        "weight_l1_shift",
        "aggregate_shift",
        "aggregate_shift_over_C",
        "max_weight_before",
        "max_weight_after",
    )
    certificate_fields = tuple(
        key for key in records[0] if key.endswith("_certificate_holds")
    )
    for n in sorted({int(record["n"]) for record in records}):
        for method in methods:
            subset = [
                record
                for record in records
                if int(record["n"]) == n and record["method"] == method
            ]
            key = f"n={n}/{method}"
            groups[key] = {
                "n": n,
                "method": method,
                "role": METHOD_ROLES[method],
                "trials": len(subset),
                "reference_certificate": subset[0]["reference_certificate"],
                "bounds": {
                    field: subset[0][field]
                    for field in (
                        "reference_bound",
                        "unchanged_score_linf_bound",
                        "replaced_score_bound",
                        "score_l1_bound",
                        "weight_l1_bound",
                        "aggregate_bound",
                        "aggregate_refined_bound",
                    )
                },
                "statistics": {
                    field: {
                        "mean": statistics.fmean(
                            float(record[field]) for record in subset
                        ),
                        "p95": _quantile(
                            (float(record[field]) for record in subset), 0.95
                        ),
                        "max": max(float(record[field]) for record in subset),
                    }
                    for field in fields
                },
                "certificate_violations": {
                    field: sum(record[field] is False for record in subset)
                    for field in certificate_fields
                },
                "certificate_checks_evaluated": {
                    field: sum(record[field] is not None for record in subset)
                    for field in certificate_fields
                },
            }

    config = {key: value for key, value in vars(args).items() if key != "output"}
    config["methods_canonical"] = methods
    return {
        "schema_version": 2,
        "audit": "S0.3_replace_one_F_scores_weights_aggregate",
        "warning": (
            "Synthetic oracle diagnostics only; outputs are not a DP proof and "
            "must not be published as auxiliary per-client telemetry."
        ),
        "config": config,
        "public_anchor": {
            "construction": args.anchor_mode,
            "independent_of_current_cohort": True,
            "seed": args.anchor_seed if args.anchor_mode == "fixed_random" else None,
            "requested_norm": args.anchor_norm,
            "realised_norm": float(torch.linalg.vector_norm(anchor)),
            "sha256": anchor_digest,
            "values": [float(value) for value in anchor],
        },
        "method_roles": {method: METHOD_ROLES[method] for method in methods},
        "groups": groups,
    }


def write_outputs(records: list[dict], summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "replace_one_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", default="5,10,20,40")
    parser.add_argument(
        "--methods",
        default="centered_clipping,regularized_huber,mean,cm,rfa,cm_nnm",
        help=(
            "Comma-separated references. F_CC is primary, regularized Huber "
            "is the ablation; mean/CM/RFA/CM-NNM are uncertified comparators."
        ),
    )
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--clip", type=float, default=1.0, help="User bound C")
    parser.add_argument(
        "--distance-clip",
        type=float,
        default=2.0,
        help="Public score scale D_max",
    )
    parser.add_argument("--tau", "--reference-tau", dest="tau", type=float, default=1.0)
    parser.add_argument("--huber-gamma", type=float, default=1.0)
    parser.add_argument("--huber-num-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=2.0)
    parser.add_argument("--byzantine-fraction", type=float, default=0.2)
    parser.add_argument(
        "--anchor-mode", choices=("zero", "fixed_random"), default="zero"
    )
    parser.add_argument("--anchor-norm", type=float, default=0.5)
    parser.add_argument("--anchor-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--certificate-tolerance", type=float, default=1e-10)
    parser.add_argument("--output", default="results/scpfar/sensitivity_s0_3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, summary = audit_replace_one(args)
    output = Path(args.output)
    write_outputs(records, summary, output)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
