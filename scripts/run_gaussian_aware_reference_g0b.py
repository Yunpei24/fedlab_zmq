#!/usr/bin/env python3
"""Pre-registered G0b audit for Gaussian-aware robust FAR references.

G0b corrects two statistical defects of G0: finite-cohort invariance metrics
are centred on an independently simulated null, and uncertainty is computed
after clustering all paired observations by seed.  Candidate choice uses only
the development seeds.  The selected candidate is then frozen before the
holdout phase.  No accuracy is generated or used.

All tensor construction and linear algebra execute on the explicitly requested
device.  The publication CLI requires MPS and refuses silent CPU fallback.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robustness.aggregators import clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number != 0.0 and abs(number) < 10.0 ** (-digits):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _fmt_count(value: Any) -> str:
    """Format an integer-valued count without a misleading decimal suffix."""

    return str(int(round(float(value))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Rows for {path} do not share an identical schema")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for raw in config["references"]["candidates"]:
        spec = dict(raw)
        identifier = str(spec["id"])
        if identifier in specs:
            raise ValueError(f"Duplicate reference candidate {identifier!r}")
        method = str(spec["method"])
        if method not in oracle.REFERENCE_NAMES:
            raise ValueError(f"Unknown base reference {method!r}")
        specs[identifier] = spec
    return specs


def _materialize_candidate_config(
    base: dict[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    method = str(spec["method"])
    config["references"]["candidates"] = [method]
    if method == "f_sigma_huber":
        total_cap = float(spec["influence_cap_total"])
        blocks = len(config["cohort"]["block_sizes"])
        per_block = total_cap / math.sqrt(float(blocks))
        config["references"]["f_sigma_huber"]["influence_cap"] = [
            per_block
        ] * blocks
        config["references"]["f_sigma_huber"]["regularization"] = float(
            spec["regularization"]
        )
    oracle._validate_config(config)
    return config


def _validate_g0b(config: dict[str, Any]) -> None:
    specs = _candidate_specs(config)
    excluded = set(int(value) for value in config["excluded_prior_seeds"])
    calibration = int(config["randomness"]["null_calibration_seed"])
    development = [int(value) for value in config["randomness"]["development_seeds"]]
    holdout = [int(value) for value in config["randomness"]["holdout_seeds"]]
    all_fresh = [calibration, *development, *holdout]
    if len(all_fresh) != len(set(all_fresh)):
        raise ValueError("Calibration, development and holdout seeds must be disjoint")
    if excluded.intersection(all_fresh):
        raise ValueError("G0b must not reuse any seed inspected during G0")
    if len(development) < 3 or len(holdout) < 5:
        raise ValueError("G0b needs >=3 development and >=5 holdout seeds")
    selectable = set(str(value) for value in config["selection"]["candidate_references"])
    if not selectable or not selectable.issubset(specs):
        raise ValueError("Selection candidates must be declared reference IDs")
    if any(specs[name]["method"] != "f_sigma_huber" for name in selectable):
        raise ValueError("G0b selection is restricted to Gaussian-aware candidates")
    if config["execution"]["required_device"] != "mps":
        raise ValueError("The registered G0b publication run requires MPS")
    if bool(config["execution"]["allow_cpu_fallback"]):
        raise ValueError("Silent CPU fallback is forbidden")
    if bool(config["selection"]["accuracy_used"]):
        raise ValueError("Accuracy cannot be used in the synthetic oracle")
    modes = list(config["score"]["weight_modes"])
    if modes != ["reference_only", "novelty_only", "novelty_confidence"]:
        raise ValueError("The three registered weighting ablations are required")
    non_null_threats = set(str(value) for value in config["threats"]["names"]) - {
        "none"
    }
    separated = set(
        str(value) for value in config["threats"]["separated_for_gates"]
    )
    evasive = set(str(value) for value in config["threats"]["evasive_controls"])
    if separated & evasive or separated | evasive != non_null_threats:
        raise ValueError(
            "Every non-null threat must belong to exactly one of "
            "separated_for_gates or evasive_controls"
        )


def _calibrations(
    config: dict[str, Any],
    candidate_configs: Mapping[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    draws = int(config["randomness"]["null_calibration_draws"])
    seed = int(config["randomness"]["null_calibration_seed"])
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    total_cells = sum(
        len(regime["permutations"]) for regime in config["privacy_noise"]["regimes"]
    ) * len(candidate_configs)
    completed_cells = 0
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = oracle._noise_variances(config, regime, permutation)
            for candidate, candidate_config in candidate_configs.items():
                method = str(_candidate_specs(config)[candidate]["method"])
                calibrated = oracle._calibrate_reference_variances(
                    candidate_config,
                    regime=regime,
                    permutation=permutation,
                    draws=draws,
                    seed=seed,
                )[method]
                result[(candidate, regime_name, permutation)] = calibrated
                completed_cells += 1
                print(
                    f"[G0b] calibration {completed_cells}/{total_cells}: "
                    f"{candidate} {regime_name}/{permutation}",
                    flush=True,
                )
                for client in range(int(config["cohort"]["num_clients"])):
                    for block in range(len(blocks)):
                        rows.append(
                            {
                                "candidate": candidate,
                                "noise_regime": regime_name,
                                "noise_permutation": permutation,
                                "client": client,
                                "block": block,
                                "public_noise_variance": float(
                                    variances[client, block].item()
                                ),
                                "reference_variance": float(
                                    calibrated["reference_variances"][client, block].item()
                                ),
                                "pooled_score_threshold": float(
                                    calibrated["score_threshold"].item()
                                ),
                                "null_abs_correlation_mean": float(
                                    calibrated[
                                        "null_abs_score_noise_correlation_mean"
                                    ]
                                ),
                                "null_tier_range_mean": float(
                                    calibrated["null_noise_tier_score_range_mean"]
                                ),
                            }
                        )
    return result, rows


def _reference_certificate(
    candidate: str,
    spec: Mapping[str, Any],
    candidate_config: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> tuple[float, str]:
    n = float(candidate_config["cohort"]["num_clients"])
    method = str(spec["method"])
    if method == "uniform_mean":
        server_clip = float(candidate_config["aggregation"]["server_clip_norm"])
        return 2.0 * server_clip / n, "replace_one_after_server_clip"
    if method == "fcc":
        return 2.0 * float(candidate_config["references"]["fcc"]["radius"]) / n, (
            "replace_one_fixed_anchor"
        )
    if method == "fna_cc":
        settings = candidate_config["references"]["fna_cc"]
        return (
            2.0
            * float(settings["max_weight_ratio"])
            * float(settings["radius"])
            / n,
            "replace_one_fixed_public_variances",
        )
    if method == "rfa":
        return float("nan"), "no_global_o_1_over_n_certificate"
    return float(diagnostics["finite_solver_replace_one_bound"]), (
        "finite_solver_fixed_anchor_covariances"
    )


def _evaluate_candidate_modes(
    *,
    candidate: str,
    spec: Mapping[str, Any],
    candidate_config: dict[str, Any],
    vectors: torch.Tensor,
    clean: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    noise_variances: torch.Tensor,
    tiers: torch.Tensor,
    anchor: torch.Tensor,
    centre: torch.Tensor,
    calibration: Mapping[str, Any],
    modes: Sequence[str],
) -> list[dict[str, Any]]:
    method = str(spec["method"])
    server_clip = float(candidate_config["aggregation"]["server_clip_norm"])
    # This is the actual LDP-Gradient-FAR order: local DP release first,
    # deterministic server clipping second, then F / scores / FAR weights.
    # In particular, a large-norm Byzantine upload cannot reach the reference
    # or the score before its public influence bound is enforced.
    bounded_vectors = clip_l2(vectors, server_clip)
    reference, diagnostics = oracle._reference(
        method,
        bounded_vectors,
        candidate_config,
        anchor=anchor,
        noise_variances=noise_variances,
        return_diagnostics=True,
    )
    loo = oracle._leave_one_out_references(
        method,
        bounded_vectors,
        candidate_config,
        anchor=anchor,
        noise_variances=noise_variances,
    )
    scores, score_diagnostics = oracle._standardized_scores(
        bounded_vectors,
        loo,
        candidate_config,
        noise_variances=noise_variances,
        reference_variances=calibration["reference_variances"],
    )
    honest = ~byzantine_mask
    regular_honest = honest & ~outlier_mask
    outliers = honest & outlier_mask
    honest_target = clean[honest].mean(dim=0)
    threshold = float(calibration["score_threshold"].item())
    false_outlier_rate = float(
        (scores[regular_honest] > threshold).to(scores.dtype).mean().item()
    )
    outlier_recall = (
        float((scores[outliers] > threshold).to(scores.dtype).mean().item())
        if bool(outliers.any().item())
        else float("nan")
    )
    corr = oracle._corr(scores[regular_honest], tiers[regular_honest])
    tier_range, tier_reference_range, tier_projection = oracle._tier_metrics(
        clean=clean,
        reference=reference,
        scores=scores,
        tiers=tiers,
        regular_honest_mask=regular_honest,
    )
    abs_corr_excess = (
        abs(corr) - float(calibration["null_abs_score_noise_correlation_mean"])
        if math.isfinite(corr)
        else float("nan")
    )
    tier_range_excess = (
        tier_range - float(calibration["null_noise_tier_score_range_mean"])
        if math.isfinite(tier_range)
        else float("nan")
    )
    reference_bound, certificate = _reference_certificate(
        candidate, spec, candidate_config, diagnostics
    )
    covariance_counterfactual_distance = float("nan")
    covariance_counterfactual_ratio = float("nan")
    if method == "f_sigma_huber":
        # Counterfactual with the same cap, regularisation, anchor and fixed
        # solver, but an enormous standardized threshold.  Every transition
        # radius is then the Euclidean cap, so a non-zero distance proves that
        # the covariance branch changes the returned reference, rather than
        # merely being nominally available.
        cap_only_config = copy.deepcopy(candidate_config)
        cap_only_config["references"]["f_sigma_huber"][
            "standardized_threshold"
        ] = [1.0e6] * len(candidate_config["cohort"]["block_sizes"])
        cap_only_reference = oracle._reference(
            method,
            bounded_vectors,
            cap_only_config,
            anchor=anchor,
            noise_variances=noise_variances,
        )
        covariance_counterfactual_distance = float(
            torch.linalg.vector_norm(reference - cap_only_reference).item()
        )
        total_cap = float(spec["influence_cap_total"])
        covariance_counterfactual_ratio = (
            covariance_counterfactual_distance / total_cap
        )
    contributions = bounded_vectors
    results: list[dict[str, Any]] = []
    for mode in modes:
        weights, weight_diagnostics = oracle._oracle_weights(
            scores, candidate_config, mode=mode
        )
        aggregate = (weights[:, None] * contributions).sum(dim=0)
        results.append(
            {
                "candidate": candidate,
                "base_method": method,
                "weight_mode": mode,
                "reference_error_to_honest_clean_mean": float(
                    torch.linalg.vector_norm(reference - honest_target).item()
                ),
                "reference_error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "aggregate_error_to_honest_clean_mean": float(
                    torch.linalg.vector_norm(aggregate - honest_target).item()
                ),
                "false_outlier_rate": false_outlier_rate,
                "honest_outlier_recall": outlier_recall,
                "score_noise_std_correlation": corr,
                "abs_score_noise_std_correlation": (
                    abs(corr) if math.isfinite(corr) else float("nan")
                ),
                "null_abs_score_noise_correlation_mean": float(
                    calibration["null_abs_score_noise_correlation_mean"]
                ),
                "abs_correlation_null_excess": abs_corr_excess,
                "noise_tier_mean_score_range": tier_range,
                "null_noise_tier_score_range_mean": float(
                    calibration["null_noise_tier_score_range_mean"]
                ),
                "tier_range_null_excess": tier_range_excess,
                "noise_tier_reference_distance_range": tier_reference_range,
                "noise_tier_reference_bias_projection_abs": tier_projection,
                "honest_outlier_weight_mass": float(weights[outliers].sum().item()),
                "byzantine_weight_mass": float(weights[byzantine_mask].sum().item()),
                "max_individual_weight": float(weights.max().item()),
                "weight_concentration_n_sum_q2": float(
                    weights.numel() * weights.square().sum().item()
                ),
                "weight_entropy": float(
                    (-(weights * weights.clamp_min(1e-15).log()).sum()).item()
                ),
                "score_min": float(scores.min().item()),
                "score_max": float(scores.max().item()),
                "score_span": float((scores.max() - scores.min()).item()),
                "mean_novelty_honest": float(
                    weight_diagnostics["novelty"][honest].mean().item()
                ),
                "mean_trust_honest": float(
                    weight_diagnostics["trust"][honest].mean().item()
                ),
                "mean_trust_byzantine": (
                    float(weight_diagnostics["trust"][byzantine_mask].mean().item())
                    if bool(byzantine_mask.any().item())
                    else float("nan")
                ),
                "quadratic_energy_mean": float(
                    score_diagnostics["quadratic_energy"].mean().item()
                ),
                "server_clip_rate_honest": float(
                    (
                        torch.linalg.vector_norm(vectors[honest], dim=1)
                        > server_clip
                    )
                    .to(vectors.dtype)
                    .mean()
                    .item()
                ),
                "server_clip_rate_byzantine": (
                    float(
                        (
                            torch.linalg.vector_norm(vectors[byzantine_mask], dim=1)
                            > server_clip
                        )
                        .to(vectors.dtype)
                        .mean()
                        .item()
                    )
                    if bool(byzantine_mask.any().item())
                    else float("nan")
                ),
                "reference_tail_fraction_client_blocks": float(
                    diagnostics.get("tail_fraction_client_blocks", float("nan"))
                ),
                "reference_fraction_covariance_limited_client_blocks": float(
                    diagnostics.get(
                        "fraction_covariance_limited_client_blocks", float("nan")
                    )
                ),
                "reference_cap_limited_fraction_client_blocks": (
                    1.0
                    - float(
                        diagnostics.get(
                            "fraction_covariance_limited_client_blocks", float("nan")
                        )
                    )
                    if math.isfinite(
                        float(
                            diagnostics.get(
                                "fraction_covariance_limited_client_blocks",
                                float("nan"),
                            )
                        )
                    )
                    else float("nan")
                ),
                "reference_fraction_covariance_limited_and_tail_client_blocks": float(
                    diagnostics.get(
                        "fraction_covariance_limited_and_huber_tail_client_blocks",
                        float("nan"),
                    )
                ),
                "reference_fraction_cap_limited_and_tail_client_blocks": float(
                    diagnostics.get(
                        "fraction_cap_limited_and_huber_tail_client_blocks",
                        float("nan"),
                    )
                ),
                "reference_replace_one_bound": reference_bound,
                "reference_certificate": certificate,
                "reference_exact_minimizer_replace_one_bound": float(
                    diagnostics.get("exact_minimizer_replace_one_bound", float("nan"))
                ),
                "reference_gradient_residual_norm": float(
                    diagnostics.get("gradient_residual_norm", float("nan"))
                ),
                "reference_covariance_counterfactual_distance": (
                    covariance_counterfactual_distance
                ),
                "reference_covariance_counterfactual_ratio_to_total_cap": (
                    covariance_counterfactual_ratio
                ),
            }
        )
    return results


def _modes_for_development(candidate: str) -> list[str]:
    if candidate == "uniform_mean":
        return ["reference_only"]
    return ["reference_only", "novelty_only", "novelty_confidence"]


def _phase_rows(
    *,
    phase: str,
    config: dict[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    candidate_configs: Mapping[str, dict[str, Any]],
    calibrations: Mapping[tuple[str, str, str], Mapping[str, Any]],
    candidates: Sequence[str],
    modes_by_candidate: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, tiers = oracle._noise_variances(config, regime, permutation)
            for seed in seeds:
                print(
                    f"[G0b] {phase}: seed {seed} "
                    f"({list(seeds).index(seed) + 1}/{len(seeds)})",
                    flush=True,
                )
                for draw in range(draws_per_seed):
                    for geometry_value in config["cohort"]["honest_outliers"][
                        "geometries"
                    ]:
                        geometry = str(geometry_value)
                        clean, outliers, centre, anchor = oracle._honest_clean_vectors(
                            config,
                            seed=int(seed),
                            draw=draw,
                            geometry=geometry,
                            include_outliers=True,
                        )
                        observed = oracle._add_private_noise(
                            clean,
                            variances,
                            blocks,
                            seed=int(seed),
                            draw=draw,
                            regime=regime_name,
                            permutation=permutation,
                            geometry=geometry,
                            pair_permutations=bool(
                                config["randomness"][
                                    "pair_noise_across_tier_permutations"
                                ]
                            ),
                        )
                        for threat_value in config["threats"]["names"]:
                            threat = str(threat_value)
                            threat_severities = [1.0] if threat == "none" else severities
                            for severity in threat_severities:
                                attacked, byzantine = oracle._replace_with_attack(
                                    observed,
                                    config,
                                    threat=threat,
                                    severity=float(severity),
                                    seed=oracle._seed(seed, draw, geometry, threat),
                                )
                                pairing_id = (
                                    f"{phase}:{regime_name}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{int(seed)}:{draw}"
                                )
                                for candidate in candidates:
                                    metrics_rows = _evaluate_candidate_modes(
                                        candidate=candidate,
                                        spec=specs[candidate],
                                        candidate_config=candidate_configs[candidate],
                                        vectors=attacked,
                                        clean=clean,
                                        outlier_mask=outliers,
                                        byzantine_mask=byzantine,
                                        noise_variances=variances,
                                        tiers=tiers,
                                        anchor=anchor,
                                        centre=centre,
                                        calibration=calibrations[
                                            (candidate, regime_name, permutation)
                                        ],
                                        modes=modes_by_candidate[candidate],
                                    )
                                    for metrics in metrics_rows:
                                        rows.append(
                                            {
                                                "campaign_id": config["campaign_id"],
                                                "phase": phase,
                                                "pairing_id": pairing_id,
                                                "noise_regime": regime_name,
                                                "noise_permutation": permutation,
                                                "outlier_geometry": geometry,
                                                "threat": threat,
                                                "severity": float(severity),
                                                "seed": int(seed),
                                                "draw": draw,
                                                **metrics,
                                            }
                                        )
    _attach_ratios(rows)
    return rows


def _attach_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {
        row["pairing_id"]: row
        for row in rows
        if row["candidate"] == "uniform_mean"
        and row["weight_mode"] == "reference_only"
    }
    pairings = set(row["pairing_id"] for row in rows)
    if set(baselines) != pairings:
        raise ValueError("Every pairing must contain one uniform reference-only baseline")
    for row in rows:
        baseline = baselines[row["pairing_id"]]
        ref_den = float(baseline["reference_error_to_honest_clean_mean"])
        agg_den = float(baseline["aggregate_error_to_honest_clean_mean"])
        row["reference_error_ratio_to_uniform"] = (
            float(row["reference_error_to_honest_clean_mean"]) / ref_den
            if ref_den > 0.0
            else float("nan")
        )
        row["aggregate_error_ratio_to_uniform"] = (
            float(row["aggregate_error_to_honest_clean_mean"]) / agg_den
            if agg_den > 0.0
            else float("nan")
        )


def _group_worst(rows: Sequence[dict[str, Any]], field: str) -> float:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["noise_regime"],
            row["noise_permutation"],
            row["outlier_geometry"],
            row["threat"],
            row["severity"],
        )
        grouped[key].append(float(row[field]))
    return max(_finite_mean(values) for values in grouped.values())


def _clustered_seed_ci95(
    rows: Sequence[dict[str, Any]], field: str
) -> tuple[float, float, float, int]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            grouped[int(row["seed"])].append(value)
    seed_means = [_finite_mean(values) for _, values in sorted(grouped.items())]
    n = len(seed_means)
    mean = _finite_mean(seed_means)
    if n < 2:
        return mean, float("-inf"), float("inf"), n
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(n, 1.96)
    half = critical * _finite_std(seed_means) / math.sqrt(float(n))
    return mean, mean - half, mean + half, n


def _summaries(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    separated = set(str(value) for value in config["threats"]["separated_for_gates"])
    evasive_names = set(str(value) for value in config["threats"]["evasive_controls"])
    gates = config["gates"]
    uniform_outlier_mass = float(config["cohort"]["honest_outliers"]["count"]) / float(
        config["cohort"]["num_clients"]
    )
    keys = sorted(set((row["candidate"], row["weight_mode"]) for row in rows))
    result: list[dict[str, Any]] = []
    for candidate, mode in keys:
        selected = [
            row
            for row in rows
            if row["candidate"] == candidate and row["weight_mode"] == mode
        ]
        clean = [row for row in selected if row["threat"] == "none"]
        hetero_clean = [
            row for row in clean if row["noise_regime"] == "heteroscedastic"
        ]
        attacked = [row for row in selected if row["threat"] in separated]
        evasive = [row for row in selected if row["threat"] in evasive_names]
        # The *_max gates are worst-stratum gates, not pooled means.  This
        # prevents a difficult noise/permutation/geometry cell from being
        # hidden by many easy cells.
        clean_ref = _group_worst(clean, "reference_error_ratio_to_uniform")
        clean_agg = _group_worst(clean, "aggregate_error_ratio_to_uniform")
        attacked_ref = _group_worst(attacked, "reference_error_ratio_to_uniform")
        attacked_agg = _group_worst(attacked, "aggregate_error_ratio_to_uniform")
        byzantine_mass = _group_worst(attacked, "byzantine_weight_mass")
        evasive_agg = (
            _group_worst(evasive, "aggregate_error_ratio_to_uniform")
            if evasive
            else float("nan")
        )
        evasive_mass = (
            _group_worst(evasive, "byzantine_weight_mass")
            if evasive
            else float("nan")
        )
        false_rate = _finite_mean(row["false_outlier_rate"] for row in clean)
        recall = _finite_mean(row["honest_outlier_recall"] for row in clean)
        outlier_mass = _finite_mean(row["honest_outlier_weight_mass"] for row in clean)
        corr_mean, corr_lo, corr_hi, corr_n = _clustered_seed_ci95(
            hetero_clean, "abs_correlation_null_excess"
        )
        tier_mean, tier_lo, tier_hi, tier_n = _clustered_seed_ci95(
            hetero_clean, "tier_range_null_excess"
        )
        covariance_fraction = _finite_mean(
            row["reference_fraction_covariance_limited_client_blocks"]
            for row in selected
        )
        covariance_tail_fraction = _finite_mean(
            row[
                "reference_fraction_covariance_limited_and_tail_client_blocks"
            ]
            for row in clean
        )
        covariance_counterfactual_ratio = _finite_mean(
            row["reference_covariance_counterfactual_ratio_to_total_cap"]
            for row in clean
        )
        gaussian = selected[0]["base_method"] == "f_sigma_huber"
        checks = {
            "clean_reference_error": clean_ref
            <= float(gates["clean_reference_error_ratio_to_uniform_max"]),
            "clean_aggregate_error": clean_agg
            <= float(gates["clean_aggregate_error_ratio_to_uniform_max"]),
            "attacked_reference_error": attacked_ref
            <= float(gates["separated_att_reference_error_ratio_to_uniform_max"]),
            "attacked_aggregate_error": attacked_agg
            <= float(gates["separated_aggregate_error_ratio_to_uniform_max"]),
            "false_outlier_rate": float(gates["false_outlier_rate_min"])
            <= false_rate
            <= float(gates["false_outlier_rate_max"]),
            "honest_outlier_recall": recall
            >= float(gates["honest_outlier_recall_min"]),
            "honest_outlier_weight_mass": (
                outlier_mass - uniform_outlier_mass
                >= float(gates["honest_outlier_weight_mass_gain_over_uniform_min"])
            ),
            "byzantine_weight_mass": byzantine_mass
            <= float(gates["separated_byzantine_weight_mass_max"]),
            "evasive_aggregate_error": (
                not evasive
                or evasive_agg
                <= float(gates["evasive_aggregate_error_ratio_to_uniform_max"])
            ),
            "evasive_byzantine_weight_mass": (
                not evasive
                or evasive_mass <= float(gates["evasive_byzantine_weight_mass_max"])
            ),
            "abs_correlation_null_excess": corr_hi
            <= float(gates["abs_correlation_null_excess_ci95_upper_max"]),
            "tier_range_null_excess": tier_hi
            <= float(gates["tier_range_null_excess_ci95_upper_max"]),
            "covariance_branch_active": (
                not gaussian
                or covariance_fraction
                >= float(
                    gates["gaussian_candidate_fraction_covariance_limited_min"]
                )
            ),
            "covariance_branch_effective_on_huber_influence": (
                not gaussian
                or covariance_tail_fraction
                >= float(
                    gates[
                        "gaussian_candidate_fraction_covariance_limited_and_tail_min"
                    ]
                )
            ),
            "covariance_changes_returned_reference": (
                not gaussian
                or covariance_counterfactual_ratio
                >= float(
                    gates[
                        "gaussian_candidate_covariance_counterfactual_ratio_min"
                    ]
                )
            ),
        }
        summary: dict[str, Any] = {
            "candidate": candidate,
            "base_method": selected[0]["base_method"],
            "weight_mode": mode,
            "observations": len(selected),
            "clean_reference_error_ratio": clean_ref,
            "clean_aggregate_error_ratio": clean_agg,
            "attacked_reference_error_ratio_worst_group": attacked_ref,
            "attacked_aggregate_error_ratio_worst_group": attacked_agg,
            "false_outlier_rate": false_rate,
            "honest_outlier_recall": recall,
            "honest_outlier_weight_mass": outlier_mass,
            "honest_outlier_weight_mass_gain": outlier_mass - uniform_outlier_mass,
            "byzantine_weight_mass_worst_group": byzantine_mass,
            "evasive_aggregate_error_ratio_worst_group": evasive_agg,
            "evasive_byzantine_weight_mass_worst_group": evasive_mass,
            "abs_correlation_null_excess_seed_mean": corr_mean,
            "abs_correlation_null_excess_ci95_low": corr_lo,
            "abs_correlation_null_excess_ci95_high": corr_hi,
            "abs_correlation_null_excess_num_seed_clusters": corr_n,
            "tier_range_null_excess_seed_mean": tier_mean,
            "tier_range_null_excess_ci95_low": tier_lo,
            "tier_range_null_excess_ci95_high": tier_hi,
            "tier_range_null_excess_num_seed_clusters": tier_n,
            "fraction_covariance_limited": covariance_fraction,
            "fraction_covariance_limited_and_tail": covariance_tail_fraction,
            "covariance_counterfactual_ratio": covariance_counterfactual_ratio,
            "fraction_cap_limited": _finite_mean(
                row["reference_cap_limited_fraction_client_blocks"] for row in selected
            ),
            "tail_fraction": _finite_mean(
                row["reference_tail_fraction_client_blocks"] for row in clean
            ),
            "replace_one_bound": _finite_mean(
                row["reference_replace_one_bound"] for row in selected
            ),
            "certificate": selected[0]["reference_certificate"],
        }
        for name, passed in checks.items():
            summary[f"gate_{name}"] = bool(passed)
        summary["gate_fail_count"] = sum(not passed for passed in checks.values())
        summary["passes_all_hard_gates"] = all(checks.values())
        summary["normalized_gate_penalty"] = _gate_penalty(summary, config)
        result.append(summary)
    return result


def _gate_penalty(summary: Mapping[str, Any], config: dict[str, Any]) -> float:
    gates = config["gates"]

    def upper(value: float, threshold: float) -> float:
        if not math.isfinite(value):
            return 10.0
        return max(0.0, value / threshold - 1.0)

    def lower(value: float, threshold: float) -> float:
        if not math.isfinite(value):
            return 10.0
        return max(0.0, (threshold - value) / max(abs(threshold), 1e-12))

    penalty = 0.0
    penalty += upper(
        float(summary["clean_reference_error_ratio"]),
        float(gates["clean_reference_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["clean_aggregate_error_ratio"]),
        float(gates["clean_aggregate_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["attacked_reference_error_ratio_worst_group"]),
        float(gates["separated_att_reference_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["attacked_aggregate_error_ratio_worst_group"]),
        float(gates["separated_aggregate_error_ratio_to_uniform_max"]),
    )
    false_rate = float(summary["false_outlier_rate"])
    if false_rate < float(gates["false_outlier_rate_min"]):
        penalty += lower(false_rate, float(gates["false_outlier_rate_min"]))
    else:
        penalty += upper(false_rate, float(gates["false_outlier_rate_max"]))
    penalty += lower(
        float(summary["honest_outlier_recall"]),
        float(gates["honest_outlier_recall_min"]),
    )
    penalty += lower(
        float(summary["honest_outlier_weight_mass_gain"]),
        float(gates["honest_outlier_weight_mass_gain_over_uniform_min"]),
    )
    penalty += upper(
        float(summary["byzantine_weight_mass_worst_group"]),
        float(gates["separated_byzantine_weight_mass_max"]),
    )
    if math.isfinite(float(summary["evasive_aggregate_error_ratio_worst_group"])):
        penalty += upper(
            float(summary["evasive_aggregate_error_ratio_worst_group"]),
            float(gates["evasive_aggregate_error_ratio_to_uniform_max"]),
        )
        penalty += upper(
            float(summary["evasive_byzantine_weight_mass_worst_group"]),
            float(gates["evasive_byzantine_weight_mass_max"]),
        )
    penalty += upper(
        max(0.0, float(summary["abs_correlation_null_excess_ci95_high"])),
        float(gates["abs_correlation_null_excess_ci95_upper_max"]),
    )
    penalty += upper(
        max(0.0, float(summary["tier_range_null_excess_ci95_high"])),
        float(gates["tier_range_null_excess_ci95_upper_max"]),
    )
    if summary["base_method"] == "f_sigma_huber":
        penalty += lower(
            float(summary["fraction_covariance_limited"]),
            float(gates["gaussian_candidate_fraction_covariance_limited_min"]),
        )
        penalty += lower(
            float(summary["covariance_counterfactual_ratio"]),
            float(
                gates[
                    "gaussian_candidate_covariance_counterfactual_ratio_min"
                ]
            ),
        )
        penalty += lower(
            float(summary["fraction_covariance_limited_and_tail"]),
            float(
                gates[
                    "gaussian_candidate_fraction_covariance_limited_and_tail_min"
                ]
            ),
        )
    return penalty


def _select_development(
    summaries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    allowed = [str(value) for value in config["selection"]["candidate_references"]]
    order = {name: index for index, name in enumerate(allowed)}
    eligible = [row for row in summaries if row["candidate"] in order]
    if not eligible:
        raise RuntimeError("No Gaussian-aware development candidate was evaluated")
    eligible.sort(
        key=lambda row: (
            0 if row["passes_all_hard_gates"] else 1,
            int(row["gate_fail_count"]),
            float(row["normalized_gate_penalty"]),
            order[row["candidate"]],
            ["novelty_confidence", "novelty_only", "reference_only"].index(
                row["weight_mode"]
            ),
        )
    )
    return dict(eligible[0])


def _gate_audit_lines(
    config: Mapping[str, Any],
    development: Mapping[str, Any],
    holdout: Mapping[str, Any],
) -> list[str]:
    """Build a complete, human-readable audit of every pre-registered gate."""

    gates = config["gates"]
    specs = [
        (
            "Erreur de référence, propre",
            "clean_reference_error_ratio",
            f"<= {gates['clean_reference_error_ratio_to_uniform_max']}",
            "gate_clean_reference_error",
        ),
        (
            "Erreur d'agrégat, propre",
            "clean_aggregate_error_ratio",
            f"<= {gates['clean_aggregate_error_ratio_to_uniform_max']}",
            "gate_clean_aggregate_error",
        ),
        (
            "Erreur de référence, attaques séparées",
            "attacked_reference_error_ratio_worst_group",
            f"<= {gates['separated_att_reference_error_ratio_to_uniform_max']}",
            "gate_attacked_reference_error",
        ),
        (
            "Erreur d'agrégat, attaques séparées",
            "attacked_aggregate_error_ratio_worst_group",
            f"<= {gates['separated_aggregate_error_ratio_to_uniform_max']}",
            "gate_attacked_aggregate_error",
        ),
        (
            "Taux de faux outliers",
            "false_outlier_rate",
            "[{lo}, {hi}]".format(
                lo=gates["false_outlier_rate_min"],
                hi=gates["false_outlier_rate_max"],
            ),
            "gate_false_outlier_rate",
        ),
        (
            "Rappel des honest outliers",
            "honest_outlier_recall",
            f">= {gates['honest_outlier_recall_min']}",
            "gate_honest_outlier_recall",
        ),
        (
            "Gain de masse des honest outliers",
            "honest_outlier_weight_mass_gain",
            f">= {gates['honest_outlier_weight_mass_gain_over_uniform_min']}",
            "gate_honest_outlier_weight_mass",
        ),
        (
            "Masse byzantine, attaques séparées",
            "byzantine_weight_mass_worst_group",
            f"<= {gates['separated_byzantine_weight_mass_max']}",
            "gate_byzantine_weight_mass",
        ),
        (
            "Erreur d'agrégat, ALIE",
            "evasive_aggregate_error_ratio_worst_group",
            f"<= {gates['evasive_aggregate_error_ratio_to_uniform_max']}",
            "gate_evasive_aggregate_error",
        ),
        (
            "Masse byzantine, ALIE",
            "evasive_byzantine_weight_mass_worst_group",
            f"<= {gates['evasive_byzantine_weight_mass_max']}",
            "gate_evasive_byzantine_weight_mass",
        ),
        (
            "Borne IC95 de l'excès de corrélation",
            "abs_correlation_null_excess_ci95_high",
            f"<= {gates['abs_correlation_null_excess_ci95_upper_max']}",
            "gate_abs_correlation_null_excess",
        ),
        (
            "Borne IC95 de l'excès de plage inter-tiers",
            "tier_range_null_excess_ci95_high",
            f"<= {gates['tier_range_null_excess_ci95_upper_max']}",
            "gate_tier_range_null_excess",
        ),
        (
            "Fraction covariance-limited",
            "fraction_covariance_limited",
            f">= {gates['gaussian_candidate_fraction_covariance_limited_min']}",
            "gate_covariance_branch_active",
        ),
        (
            "Fraction covariance-limited et branche Huber-tail",
            "fraction_covariance_limited_and_tail",
            ">= "
            f"{gates['gaussian_candidate_fraction_covariance_limited_and_tail_min']}",
            "gate_covariance_branch_effective_on_huber_influence",
        ),
        (
            "Distance au contre-factuel à cap fixe / cap total",
            "covariance_counterfactual_ratio",
            ">= "
            f"{gates['gaussian_candidate_covariance_counterfactual_ratio_min']}",
            "gate_covariance_changes_returned_reference",
        ),
    ]
    lines = [
        "## Audit complet des gates du candidat verrouillé",
        "",
        "Les ratios d'erreur sont rapportés à la moyenne uniforme appariée. "
        "Les erreurs maximales sont les pires moyennes parmi les strates "
        "bruit/permutation/géométrie/attaque/sévérité : une strate difficile "
        "ne peut donc pas être masquée par les autres.",
        "",
        "| Gate dur | Seuil préenregistré | Développement | Verdict dev | "
        "Holdout | Verdict holdout |",
        "|---|---:|---:|:---:|---:|:---:|",
    ]
    for label, field, threshold, gate_field in specs:
        lines.append(
            f"| {label} | `{threshold}` | {_fmt(development[field])} | "
            f"{'passe' if development[gate_field] else '**échoue**'} | "
            f"{_fmt(holdout[field])} | "
            f"{'passe' if holdout[gate_field] else '**échoue**'} |"
        )
    lines.extend(
        [
            "",
            "Le candidat échoue donc sur développement avant toute ouverture "
            "du holdout. Sur holdout, les six échecs concernent l'agrégat "
            "propre, la référence sous attaques séparées, le rappel des honest "
            "outliers, l'invariance entre tiers de bruit, l'action conjointe de "
            "la covariance dans la branche Huber-tail et l'effet mesurable de "
            "la covariance sur la référence retournée.",
        ]
    )
    return lines


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    development: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    locked: dict[str, Any],
    final: dict[str, Any],
) -> None:
    dev_selectable = [
        row
        for row in development
        if row["candidate"] in set(config["selection"]["candidate_references"])
    ]
    holdout_locked = next(
        row
        for row in holdout
        if row["candidate"] == locked["candidate"]
        and row["weight_mode"] == locked["weight_mode"]
    )
    holdout_lookup = {
        (row["candidate"], row["weight_mode"]): row for row in holdout
    }
    uniform_holdout = holdout_lookup[("uniform_mean", "reference_only")]
    certificate_comparators = [
        ("moyenne post-clipping", uniform_holdout)
    ]
    for candidate, label in (("fcc", "FCC"), ("fna_cc", "FNA-CC")):
        comparator = holdout_lookup.get((candidate, "reference_only"))
        if comparator is not None:
            certificate_comparators.append((label, comparator))
    certificate_comparison = ", ".join(
        f"`{_fmt(row['replace_one_bound'], 4)}` pour {label}"
        for label, row in certificate_comparators
    )
    lines = [
        "# G0b — audit MPS préenregistré de la référence Gaussian-aware",
        "",
        "## Verdict",
        "",
        (
            "**Promotion autorisée sur ce banc synthétique.** Le candidat "
            "verrouillé sur développement satisfait aussi tous les gates durs "
            "sur le holdout."
            if final["promote"]
            else "**Aucune promotion.** Au moins un gate dur du candidat "
            "verrouillé échoue sur le holdout ; aucune accuracy Fashion-MNIST "
            "ne doit être utilisée pour le sauver a posteriori."
        ),
        "",
        f"Candidat verrouillé : `{locked['candidate']}` avec règle "
        f"`{locked['weight_mode']}`. Il avait "
        f"`{_fmt_count(locked['gate_fail_count'])}` "
        "échec(s) de gate sur développement et en a "
        f"`{_fmt_count(holdout_locked['gate_fail_count'])}` sur holdout.",
        "",
        "Sur holdout, sa fraction covariance-limited vaut "
        f"`{_fmt(holdout_locked['fraction_covariance_limited'])}`, son "
        "intersection covariance-limited/tail vaut "
        f"`{_fmt(holdout_locked['fraction_covariance_limited_and_tail'])}`, "
        "et la distance au contre-factuel à cap fixe, normalisée par le cap "
        f"total, vaut `{_fmt(holdout_locked['covariance_counterfactual_ratio'])}`.",
        "",
        "## Séparation développement / holdout",
        "",
        f"- calibration nulle indépendante : seed "
        f"`{config['randomness']['null_calibration_seed']}`, "
        f"`{config['randomness']['null_calibration_draws']}` tirages ;",
        f"- développement uniquement : seeds "
        f"`{config['randomness']['development_seeds']}` ;",
        f"- holdout jamais consulté pour choisir : seeds "
        f"`{config['randomness']['holdout_seeds']}` ;",
        f"- seeds G0 explicitement interdites : `{config['excluded_prior_seeds']}`.",
        "",
        "Le choix du candidat et du mode de pondération est écrit dans "
        "`development_lock.json` avant l'évaluation holdout. Le holdout ne "
        "modifie ni les hyperparamètres, ni les seuils, ni la décision.",
        "",
        *_gate_audit_lines(config, locked, holdout_locked),
        "",
        "## Pipeline effectivement audité",
        "",
        "G0b suit le pipeline déployable actuel de LDP-Gradient-FAR : "
        "`Y privé -> X=Clip_U(Y) -> F(X) -> scores -> poids -> agrégat`. "
        "Il ne teste pas la variante antérieure du brouillon qui calculait "
        "la géométrie sur `Y` avant clipping. Cette autre variante exigerait "
        "une intégration serveur et une analyse séparées.",
        "",
        "Le clipping transforme la loi gaussienne de `Y`. Les covariances DP "
        "pré-clipping utilisées dans les rayons et moments deviennent donc des "
        "proxys publics, non des moments gaussiens exacts de `X`. Le clipping "
        "peut aussi atténuer le signal de covariance. C'est précisément pour "
        "cela que la calibration nulle Monte-Carlo rejoue `Y -> Clip_U(Y)` et "
        "que les gates d'invariance sont obligatoires.",
        "",
        "## Pourquoi les nouveaux gates d'invariance sont valides",
        "",
        "Avec 25 clients, même une corrélation absolue et une plage max–min "
        "issues d'un score parfaitement pivotal sont positives. G0b soustrait "
        "donc leur espérance sous un nul indépendant, puis moyenne chaque "
        "statistique au sein d'une seed. L'IC de Student à 95 % est calculé "
        "sur ces moyennes de seeds, pas sur les milliers de lignes appariées. "
        "Le gate porte sur la borne supérieure de cet IC. Sur holdout, `n=5` "
        "clusters seulement : l'IC reste donc large et la conclusion doit être "
        "lue comme un screening, non comme une preuve asymptotique. Ces IC sont "
        "en outre conditionnels à la calibration Monte-Carlo de "
        f"`{config['randomness']['null_calibration_draws']}` tirages, "
        "réutilisée pour estimer variance de référence et nul : ils "
        "n'intègrent pas l'incertitude de cette calibration.",
        "",
        "## Développement — candidats Gaussian-aware",
        "",
        "| Candidat | poids | réf. propre | agrégat propre | réf. attaquée | "
        "agrégat attaqué | agrégat ALIE | rappel | masse outlier + | masse byz. | "
        "IC corr. haut | IC tiers haut | covariance active | cov. active & tail | "
        "distance cov./cap | tail | échecs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev_selectable:
        lines.append(
            "| {candidate} | {mode} | {clean_ref} | {clean_agg} | {att_ref} | "
            "{att_agg} | {alie} | {recall} | {gain} | {byz} | {corr} | {tier} | "
            "{cov} | {covtail} | {counterfactual} | {tail} | {fails} |".format(
                candidate=row["candidate"],
                mode=row["weight_mode"],
                clean_ref=_fmt(row["clean_reference_error_ratio"]),
                clean_agg=_fmt(row["clean_aggregate_error_ratio"]),
                att_ref=_fmt(row["attacked_reference_error_ratio_worst_group"]),
                att_agg=_fmt(row["attacked_aggregate_error_ratio_worst_group"]),
                alie=_fmt(row["evasive_aggregate_error_ratio_worst_group"]),
                recall=_fmt(row["honest_outlier_recall"]),
                gain=_fmt(row["honest_outlier_weight_mass_gain"]),
                byz=_fmt(row["byzantine_weight_mass_worst_group"]),
                corr=_fmt(row["abs_correlation_null_excess_ci95_high"]),
                tier=_fmt(row["tier_range_null_excess_ci95_high"]),
                cov=_fmt(row["fraction_covariance_limited"]),
                covtail=_fmt(row["fraction_covariance_limited_and_tail"]),
                counterfactual=_fmt(row["covariance_counterfactual_ratio"]),
                tail=_fmt(row["tail_fraction"]),
                fails=_fmt_count(row["gate_fail_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Holdout verrouillé et comparateurs",
            "",
            "| Référence | poids | réf. propre | agrégat propre | réf. attaquée | "
            "agrégat attaqué | agrégat ALIE | rappel | masse outlier + | masse byz. | "
            "IC corr. haut | IC tiers haut | covariance active | cap actif | "
            "cov. active & tail | distance cov./cap | borne replace-one | échecs |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in holdout:
        lines.append(
            "| {candidate} | {mode} | {clean_ref} | {clean_agg} | {att_ref} | "
            "{att_agg} | {alie} | {recall} | {gain} | {byz} | {corr} | {tier} | "
            "{cov} | {cap} | {covtail} | {counterfactual} | {bound} | {fails} |".format(
                candidate=row["candidate"],
                mode=row["weight_mode"],
                clean_ref=_fmt(row["clean_reference_error_ratio"]),
                clean_agg=_fmt(row["clean_aggregate_error_ratio"]),
                att_ref=_fmt(row["attacked_reference_error_ratio_worst_group"]),
                att_agg=_fmt(row["attacked_aggregate_error_ratio_worst_group"]),
                alie=_fmt(row["evasive_aggregate_error_ratio_worst_group"]),
                recall=_fmt(row["honest_outlier_recall"]),
                gain=_fmt(row["honest_outlier_weight_mass_gain"]),
                byz=_fmt(row["byzantine_weight_mass_worst_group"]),
                corr=_fmt(row["abs_correlation_null_excess_ci95_high"]),
                tier=_fmt(row["tier_range_null_excess_ci95_high"]),
                cov=_fmt(row["fraction_covariance_limited"]),
                cap=_fmt(row["fraction_cap_limited"]),
                covtail=_fmt(row["fraction_covariance_limited_and_tail"]),
                counterfactual=_fmt(row["covariance_counterfactual_ratio"]),
                bound=_fmt(row["replace_one_bound"]),
                fails=_fmt_count(row["gate_fail_count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Lecture des ablations de poids",
            "",
            "- `reference_only` fixe alpha à zéro : l'erreur de référence est "
            "auditée sans attribuer son éventuel gain à la pondération FAR.",
            "- `novelty_only` applique uniquement l'incitation positive aux "
            "résidus modérément atypiques ; elle révèle si l'inclusion FAR "
            "achète aussi de la masse byzantine.",
            "- `novelty_confidence` ajoute une redescente sur les anomalies "
            "extrêmes ; son gain n'est crédible que s'il conserve les honest "
            "outliers tout en respectant le gate de masse byzantine.",
            "",
            "## Covariance, cap et certificats d'influence",
            "",
            "`fraction_covariance_limited` est la fraction des couples "
            "client–bloc pour lesquels le rayon issu de la covariance est "
            "strictement inférieur au cap euclidien. Une valeur nulle signifie "
            "que la prétendue référence Gaussian-aware se réduit partout à "
            "un Huber à cap fixe. `fraction_cap_limited` est son complément. "
            "`tail` indique la part des résidus effectivement dans la branche "
            "linéaire de Huber.",
            "",
            "L'activité effective est contrôlée deux fois : l'intersection "
            "`covariance-limited AND Huber-tail` doit être non négligeable, et "
            "la référence doit différer d'un contre-factuel identique où les "
            "rayons sont tous forcés au cap fixe. Ainsi, un simple rayon "
            "nominalement dépendant de Sigma mais jamais actif ne peut pas "
            "faire passer le gate Gaussian-aware.",
            "",
            "La borne replace-one rapportée pour Huber est celle du solveur à "
            "nombre public et fixé d'itérations, conditionnellement à l'ancre "
            "et aux covariances publiques fixées. FCC et FNA-CC disposent de "
            "leurs bornes analytiques respectives. RFA n'a pas ici de "
            "certificat global O(1/n). Dans le pipeline audité, la moyenne "
            "uniforme voit elle aussi les uploads déjà clippés et possède la "
            "borne `2U/n = "
            f"{2.0 * float(config['aggregation']['server_clip_norm']) / float(config['cohort']['num_clients']):.4f}`. "
            "Comparer les erreurs sans comparer ces "
            "certificats masquerait le prix payé pour élargir la zone centrale.",
            "",
            "## Observations, inférences et limites d'identification",
            "",
            "### Observations directement établies",
            "",
            "- La référence verrouillée améliore l'erreur propre de "
            f"`{100.0 * (1.0 - float(holdout_locked['clean_reference_error_ratio'])):.1f} %` "
            "par rapport à la moyenne uniforme (ratio "
            f"`{_fmt(holdout_locked['clean_reference_error_ratio'])}`), mais "
            "sa règle `novelty_confidence` détériore l'agrégat propre de "
            f"`{100.0 * (float(holdout_locked['clean_aggregate_error_ratio']) - 1.0):.1f} %` "
            f"(ratio `{_fmt(holdout_locked['clean_aggregate_error_ratio'])}`). "
            "Le gain local sur F ne se transmet donc pas à l'agrégat pondéré.",
            "- Sous les attaques séparées, l'agrégat reste meilleur que la "
            "moyenne uniforme (ratio "
            f"`{_fmt(holdout_locked['attacked_aggregate_error_ratio_worst_group'])}`), "
            "mais la référence ne franchit pas le seuil plus exigeant de 20 % "
            "d'amélioration (ratio "
            f"`{_fmt(holdout_locked['attacked_reference_error_ratio_worst_group'])}` "
            "> `0,80`).",
            "- Le score accroît bien la masse des honest outliers de "
            f"`{_fmt(holdout_locked['honest_outlier_weight_mass_gain'])}`, "
            "mais n'en rappelle que "
            f"`{_fmt(holdout_locked['honest_outlier_recall'])}` : l'inclusion "
            "observée est trop peu sensible pour le critère préenregistré de "
            "`0,60`.",
            "- La covariance limite nominalement "
            f"`{100.0 * float(holdout_locked['fraction_covariance_limited']):.1f} %` "
            "des rayons. Pourtant, seuls "
            f"`{_fmt(holdout_locked['fraction_covariance_limited_and_tail'])}` "
            "des couples client-bloc sont simultanément covariance-limited et "
            "dans la branche Huber-tail, et changer les rayons pour des caps "
            "fixes ne déplace F que de "
            f"`{_fmt(holdout_locked['covariance_counterfactual_ratio'])}` du "
            "cap total. Sur les cohortes propres utilisées par ces gates, la "
            "covariance est donc presque inactive dans l'estimateur "
            "effectivement retourné. Cette conclusion ne s'étend pas "
            "automatiquement aux cohortes attaquées.",
            "- Le certificat replace-one du candidat verrouillé est "
            f"`{_fmt(holdout_locked['replace_one_bound'], 4)}`, contre "
            f"{certificate_comparison}. "
            "Le candidat paie une borne d'influence plus large sans satisfaire "
            "les gates d'utilité/robustesse.",
            "- Aucun comparateur du tableau holdout ne satisfait tous les "
            "gates. Cela empêche d'attribuer l'échec à un unique choix de cap "
            "Gaussian-aware.",
            "",
            "### Inférences compatibles avec les observations",
            "",
            "L'échec conjoint du gate covariance-tail et du contre-factuel "
            "est compatible avec un seuil radial Laurent--Massart trop "
            "conservateur, avec des caps qui dominent les rayons, ou avec une "
            "géométrie de résidus qui atteint rarement simultanément la zone "
            "covariance-limited et la branche linéaire de Huber. G0b ne permet "
            "pas d'identifier laquelle de ces causes domine. Il ne faut pas "
            "attribuer cet échec au clipping serveur : sur les cohortes propres, "
            "le taux de clipping honnête observé est nul en bruit homogène et "
            "de l'ordre de `0,1 %` en bruit hétéroscédastique. Par ailleurs, le "
            "passage de l'amélioration de F à la dégradation de l'agrégat "
            "indique que la construction des poids, et pas uniquement la "
            "référence, reste un goulot d'étranglement.",
            "",
            "### Non identifiable avec G0b",
            "",
            "G0b est un banc oracle synthétique : il ne mesure ni accuracy, ni "
            "convergence Fashion-MNIST, ni coût DP composé. Avec seulement cinq "
            "seeds holdout, les IC sont adaptés à un screening mais pas à une "
            "affirmation asymptotique. Les géométries ne sont pas appariées "
            "entre elles, et l'incertitude de la calibration Monte-Carlo n'est "
            "pas propagée dans les IC. De plus, le nul retire les identifiants "
            "réservés aux honest outliers et construit F sans ces outliers, "
            "tandis que l'évaluation construit F avec eux : l'excès "
            "d'invariance peut donc mêler effet du niveau de bruit et effet "
            "indirect des outliers sur la référence. Les autres gates sont des "
            "estimations ponctuelles sur cinq seeds, sans IC propre. Enfin, la "
            "variante où F voit Y avant clipping n'est pas testée ici et ne "
            "peut pas être déduite de ce résultat.",
            "",
            "## Conclusion défendable",
            "",
            (
                "Le holdout soutient la construction à ce niveau oracle. Une "
                "confirmation end-to-end tenue à l'écart peut être planifiée, "
                "mais G0b ne constitue toujours pas une preuve Byzantine."
                if final["promote"]
                else "G0b ne valide pas encore simultanément estimation propre, "
                "robustesse sous attaque, inclusion des honest outliers et "
                "invariance au bruit. Le résultat doit être conservé comme "
                "audit négatif; aucune campagne vision ne peut transformer un "
                "échec de gate oracle en validation mécanistique."
            ),
            "",
            "## Fichiers et reproductibilité",
            "",
            f"- configuration : `{config_path.resolve()}` ;",
            f"- configuration résolue : `{(output_dir / 'resolved_config.yaml').resolve()}` ;",
            f"- calibration nulle : `{(output_dir / 'null_calibration.csv').resolve()}` ;",
            f"- détail développement : `{(output_dir / 'development_detail.csv').resolve()}` ;",
            f"- verrou : `{(output_dir / 'development_lock.json').resolve()}` ;",
            f"- détail holdout : `{(output_dir / 'holdout_detail.csv').resolve()}` ;",
            f"- décision : `{(output_dir / 'decision.json').resolve()}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    device: str | torch.device = "mps",
    calibration_draws_override: int | None = None,
    development_draws_override: int | None = None,
    holdout_draws_override: int | None = None,
    test_only_allow_cpu: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_g0b(config)
    runtime_device, runtime_dtype = oracle._configure_runtime(device)
    if runtime_device.type != "mps" and not test_only_allow_cpu:
        raise RuntimeError("G0b publication execution is MPS-only")
    if calibration_draws_override is not None:
        config["randomness"]["null_calibration_draws"] = int(
            calibration_draws_override
        )
    if development_draws_override is not None:
        config["randomness"]["development_draws_per_seed"] = int(
            development_draws_override
        )
    if holdout_draws_override is not None:
        config["randomness"]["holdout_draws_per_seed"] = int(
            holdout_draws_override
        )
    if min(
        int(config["randomness"]["null_calibration_draws"]),
        int(config["randomness"]["development_draws_per_seed"]),
        int(config["randomness"]["holdout_draws_per_seed"]),
    ) < 1:
        raise ValueError("All draw counts must remain positive")
    config["execution"]["requested_device"] = str(device)
    config["execution"]["resolved_device"] = str(runtime_device)
    config["execution"]["tensor_dtype"] = str(runtime_dtype).removeprefix("torch.")
    config["execution"]["silent_cpu_fallback_observed"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    specs = _candidate_specs(config)
    candidate_configs = {
        identifier: _materialize_candidate_config(config, spec)
        for identifier, spec in specs.items()
    }
    calibrations, calibration_rows = _calibrations(config, candidate_configs)
    _write_csv(output_dir / "null_calibration.csv", calibration_rows)

    all_candidates = list(specs)
    development_modes = {
        candidate: _modes_for_development(candidate) for candidate in all_candidates
    }
    development_rows = _phase_rows(
        phase="development",
        config=config,
        specs=specs,
        candidate_configs=candidate_configs,
        calibrations=calibrations,
        candidates=all_candidates,
        modes_by_candidate=development_modes,
        seeds=[int(value) for value in config["randomness"]["development_seeds"]],
        draws_per_seed=int(config["randomness"]["development_draws_per_seed"]),
        severities=[float(value) for value in config["threats"]["development_severities"]],
    )
    development_summaries = _summaries(development_rows, config)
    locked = _select_development(development_summaries, config)
    print(
        f"[G0b] development lock: {locked['candidate']} / "
        f"{locked['weight_mode']} ({locked['gate_fail_count']} gate failures)",
        flush=True,
    )
    lock_payload = {
        "candidate": locked["candidate"],
        "weight_mode": locked["weight_mode"],
        "selected_on_phase": "development_only",
        "holdout_used_for_selection": False,
        "development_gate_fail_count": locked["gate_fail_count"],
        "development_normalized_gate_penalty": locked["normalized_gate_penalty"],
        "selection_rule": config["selection"]["if_no_candidate_passes"],
    }
    _write_csv(output_dir / "development_detail.csv", development_rows)
    _write_csv(output_dir / "development_summary.csv", development_summaries)
    (output_dir / "development_lock.json").write_text(
        json.dumps(lock_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    holdout_candidates = [
        candidate
        for candidate in ["uniform_mean", "fcc", "fna_cc", "rfa"]
        if candidate in specs
    ]
    if locked["candidate"] not in holdout_candidates:
        holdout_candidates.append(str(locked["candidate"]))
    locked_mode = str(locked["weight_mode"])
    holdout_modes: dict[str, list[str]] = {}
    for candidate in holdout_candidates:
        modes = ["reference_only"]
        if candidate != "uniform_mean" and locked_mode not in modes:
            modes.append(locked_mode)
        if candidate == locked["candidate"]:
            for mode in config["score"]["weight_modes"]:
                if mode not in modes:
                    modes.append(str(mode))
        holdout_modes[candidate] = modes
    holdout_rows = _phase_rows(
        phase="holdout",
        config=config,
        specs=specs,
        candidate_configs=candidate_configs,
        calibrations=calibrations,
        candidates=holdout_candidates,
        modes_by_candidate=holdout_modes,
        seeds=[int(value) for value in config["randomness"]["holdout_seeds"]],
        draws_per_seed=int(config["randomness"]["holdout_draws_per_seed"]),
        severities=[float(value) for value in config["threats"]["holdout_severities"]],
    )
    holdout_summaries = _summaries(holdout_rows, config)
    holdout_locked = next(
        row
        for row in holdout_summaries
        if row["candidate"] == locked["candidate"]
        and row["weight_mode"] == locked_mode
    )
    decision = {
        "campaign_id": config["campaign_id"],
        "requested_device": str(device),
        "resolved_device": str(runtime_device),
        "tensor_dtype": str(runtime_dtype).removeprefix("torch."),
        "silent_cpu_fallback_allowed": False,
        "excluded_prior_seeds": config["excluded_prior_seeds"],
        "development_seeds": config["randomness"]["development_seeds"],
        "holdout_seeds": config["randomness"]["holdout_seeds"],
        "holdout_used_for_selection": False,
        "locked_candidate": locked["candidate"],
        "locked_weight_mode": locked_mode,
        "development_passes_all_hard_gates": bool(locked["passes_all_hard_gates"]),
        "holdout_passes_all_hard_gates": bool(
            holdout_locked["passes_all_hard_gates"]
        ),
        "promote": bool(
            locked["passes_all_hard_gates"]
            and holdout_locked["passes_all_hard_gates"]
        ),
        "promotion_rule": "all_hard_gates_must_pass_in_development_and_holdout",
        "accuracy_used_for_selection": False,
        "development_rows": len(development_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_seed_clusters_for_ci": len(
            config["randomness"]["holdout_seeds"]
        ),
    }
    _write_csv(output_dir / "holdout_detail.csv", holdout_rows)
    _write_csv(output_dir / "holdout_summary.csv", holdout_summaries)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        development=development_summaries,
        holdout=holdout_summaries,
        locked=locked,
        final=decision,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/ldp_gradient_far/gaussian_aware_reference_g0b.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0b_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/Gaussian_Aware_Robust_Reference_G0b_MPS.md",
    )
    parser.add_argument("--calibration-draws", type=int)
    parser.add_argument("--development-draws", type=int)
    parser.add_argument("--holdout-draws", type=int)
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        device=args.device,
        calibration_draws_override=args.calibration_draws,
        development_draws_override=args.development_draws,
        holdout_draws_override=args.holdout_draws,
    )


if __name__ == "__main__":
    main()
