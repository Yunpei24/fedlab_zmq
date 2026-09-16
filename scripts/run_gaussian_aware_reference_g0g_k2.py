#!/usr/bin/env python3
"""Run the preregistered G0g-K2 scalar-gate/global-cap screen on MPS."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_fixed_anchor_scalar_gated_reference,
)
from robustness.aggregators import (  # noqa: E402
    centered_clipping,
    clip_l2,
    coordinate_median,
    geometric_median,
    trimmed_mean,
)
from scripts import run_gaussian_aware_reference_g0g_k1 as base  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

PRIMARY = "g0g_k2"
BLIND = "g0g_k2_sigma_blind"
NO_GATE = "fixed_global_cap_no_gate"
CANDIDATES = (
    "uniform_mean",
    "fcc",
    "rfa",
    "trimmed_mean",
    "coordinate_median",
    NO_GATE,
    BLIND,
    PRIMARY,
)
TIERS = (1.0, 1.5, 2.0)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id",
        "scope",
        "scientific_contract",
        "excluded_prior_seeds",
        "cohort",
        "privacy_noise",
        "references",
        "aggregation",
        "threats",
        "randomness",
        "candidates",
        "gates",
        "execution",
    }
    if set(config) != expected:
        raise ValueError("The frozen G0g-K2 top-level schema changed")
    if config["campaign_id"] != "gaussian_aware_reference_g0g_k2_mps_v1":
        raise ValueError("Unexpected G0g-K2 campaign id")
    contract = config["scientific_contract"]
    required = {
        "estimand": "equal_client_mean_of_clean_honest_updates",
        "reference_only": True,
        "far_weights_used": False,
        "accuracy_used": False,
        "base_anchor": "fixed_public_or_prior_transcript",
        "covariance_role": "statistical_tolerance_only",
        "influence_cap_depends_on_covariance": False,
        "inverse_variance_weighting": False,
        "gate_sum_normalization": False,
        "global_l2_cap": True,
        "no_gate_is_exactly_fcc": True,
        "covariance_provenance": "public_authenticated",
        "client_declared_covariance_forbidden": True,
        "calibration_is_offline": True,
        "development_only_screen": True,
    }
    if contract != required:
        raise ValueError("The frozen G0g-K2 scientific contract changed")
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    if n < 3 or not 0 <= b < n / 2:
        raise ValueError("G0g-K2 needs n>=3 and 0<=b<n/2")
    if sum(blocks) != int(cohort["dimension"]) or any(width <= 0 for width in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity standard deviation is required per block")
    if tuple(str(value) for value in config["candidates"]["names"]) != CANDIDATES:
        raise ValueError("The frozen G0g-K2 candidate set changed")
    if config["candidates"]["primary"] != PRIMARY:
        raise ValueError("Unexpected primary candidate")
    execution = config["execution"]
    if execution != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
    }:
        raise ValueError("Production G0g-K2 must use MPS float32 without fallback")
    calibration = {int(value) for value in config["randomness"]["calibration_seeds"]}
    development = {int(value) for value in config["randomness"]["development_seeds"]}
    excluded = {int(value) for value in config["excluded_prior_seeds"]}
    if calibration & development or calibration & excluded or development & excluded:
        raise ValueError("Calibration, development and prior seed registries overlap")
    if config["randomness"]["calibration_probe_rule"] != (
        "dedicated_public_rng_one_regular_probe_per_independent_stratum_cohort"
    ):
        raise ValueError("The frozen K2 calibration probe rule changed")
    if config["randomness"]["calibration_deployed_threshold_rule"] != (
        "max_stratum_quantile_within_mode_and_noise_regime"
    ):
        raise ValueError("The frozen K2 deployed-threshold rule changed")
    if not 0.0 < float(config["references"]["statistical_false_tail_rate"]) < 1.0:
        raise ValueError("statistical_false_tail_rate must lie in (0,1)")
    if not math.isclose(
        float(config["references"]["total_client_influence_cap"]),
        float(config["references"]["fcc_radius"]),
        abs_tol=1e-12,
    ):
        raise ValueError("K2 no-gate identity requires influence cap == FCC radius")


def _client_statistics(
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    scales: torch.Tensor,
    block_sizes: Sequence[int],
) -> torch.Tensor:
    residuals = vectors - anchor[None, :]
    block_values = []
    for block_index, block_slice in enumerate(base._block_slices(block_sizes)):
        block_values.append(
            torch.linalg.vector_norm(residuals[:, block_slice], dim=1)
            / scales[:, block_index]
        )
    return torch.stack(block_values, dim=1).square().mean(dim=1).sqrt()


def _calibrate(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate K2 from one public probe per independent synthetic cohort.

    A calibration unit is an independently generated cohort for one public
    ``(context, noise regime, tier permutation)`` stratum.  Exactly one
    regular client is selected by a dedicated public RNG that never inspects
    the vectors or their scores.  A split-conformal quantile is computed in
    every stratum, separately for the aware and sigma-blind statistics.  The
    deployed threshold is the maximum stratum quantile within one
    ``(mode, noise regime)`` pair, hence it is conservative for every
    preregistered context and permutation.

    The construction removes the client/block pseudo-replication of K1.  Its
    coverage claim remains marginal for a new regular probe from the same
    synthetic stratum; it is not a simultaneous guarantee for all clients,
    a conditional guarantee by noise tier, or a DP certificate.
    """

    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    draws = int(config["randomness"]["calibration_draws_per_seed"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    pools: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    provisional: list[dict[str, Any]] = []
    for seed in config["randomness"]["calibration_seeds"]:
        for draw in range(draws):
            for context in config["randomness"]["calibration_contexts"]:
                geometry = str(context["geometry"])
                for regime, permutation in base._noise_cells(config):
                    regime_name = str(regime["name"])
                    context_name = str(context["name"])
                    cohort_seed = oracle._seed(
                        "g0g-k2-calibration-cohort",
                        seed,
                        draw,
                        context_name,
                        regime_name,
                        permutation,
                    )
                    clean, outliers, _, anchor = oracle._honest_clean_vectors(
                        config,
                        seed=cohort_seed,
                        draw=0,
                        geometry=geometry,
                        include_outliers=bool(context["include_outliers"]),
                    )
                    variances, tiers = oracle._noise_variances(
                        config, regime, permutation
                    )
                    observed = oracle._add_private_noise(
                        clean,
                        variances,
                        blocks,
                        seed=cohort_seed,
                        draw=0,
                        regime=regime_name,
                        permutation=permutation,
                        geometry=geometry,
                        pair_permutations=False,
                    )
                    bounded = clip_l2(observed, server_clip)
                    aware_scales = base._public_scale(config, variances)
                    blind_scales = aware_scales.mean(dim=0, keepdim=True).expand_as(
                        aware_scales
                    )
                    statistics_by_mode = {
                        "aware": _client_statistics(
                            bounded,
                            anchor=anchor,
                            scales=aware_scales,
                            block_sizes=blocks,
                        ),
                        "blind": _client_statistics(
                            bounded,
                            anchor=anchor,
                            scales=blind_scales,
                            block_sizes=blocks,
                        ),
                    }
                    eligible = torch.where(~outliers)[0]
                    if eligible.numel() < 1:
                        raise RuntimeError(
                            "Calibration stratum has no regular probe candidate"
                        )
                    probe_seed = oracle._seed(
                        "g0g-k2-public-probe",
                        seed,
                        draw,
                        context_name,
                        regime_name,
                        permutation,
                    )
                    probe_position = int(
                        torch.randint(
                            int(eligible.numel()),
                            (1,),
                            generator=oracle._generator(
                                "g0g-k2-public-probe-selection", probe_seed
                            ),
                            device=eligible.device,
                        ).item()
                    )
                    client = int(eligible[probe_position].item())
                    for mode, statistics in statistics_by_mode.items():
                        value = float(statistics[client].item())
                        stratum_key = (
                            mode,
                            regime_name,
                            permutation,
                            context_name,
                        )
                        pools[stratum_key].append(value)
                        provisional.append(
                            {
                                "seed": int(seed),
                                "draw": draw,
                                "context": context_name,
                                "noise_regime": regime_name,
                                "noise_permutation": permutation,
                                "mode": mode,
                                "cohort_seed": int(cohort_seed),
                                "probe_selection_seed": int(probe_seed),
                                "probe_client": client,
                                "probe_regular": bool(not outliers[client].item()),
                                "probe_outlier": bool(outliers[client].item()),
                                "noise_tier": float(tiers[client].item()),
                                "public_variance_min": float(
                                    variances[client].min().item()
                                ),
                                "public_variance_max": float(
                                    variances[client].max().item()
                                ),
                                "public_scale_min": float(
                                    (
                                        blind_scales
                                        if mode == "blind"
                                        else aware_scales
                                    )[client]
                                    .min()
                                    .item()
                                ),
                                "public_scale_max": float(
                                    (
                                        blind_scales
                                        if mode == "blind"
                                        else aware_scales
                                    )[client]
                                    .max()
                                    .item()
                                ),
                                "raw_client_statistic": value,
                            }
                        )
    miscoverage = float(config["references"]["statistical_false_tail_rate"])
    thresholds: dict[str, dict[str, float]] = {"aware": {}, "blind": {}}
    stratum_records: list[dict[str, Any]] = []
    stratum_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for (mode, regime, permutation, context), values in sorted(pools.items()):
        tensor = torch.tensor(
            values, dtype=oracle._RUNTIME_DTYPE, device=oracle._RUNTIME_DEVICE
        )
        threshold, rank = base._conformal_quantile(tensor, miscoverage)
        record = {
            "mode": mode,
            "noise_regime": regime,
            "noise_permutation": permutation,
            "context": context,
            "pool_size": len(values),
            "conformal_rank": rank,
            "finite_sample_marginal_coverage_lower_bound": rank
            / float(len(values) + 1),
            "stratum_threshold": threshold,
            "stratum_tail_count": sum(value > threshold for value in values),
            "stratum_tail_rate": sum(value > threshold for value in values)
            / float(len(values)),
        }
        stratum_records.append(record)
        stratum_lookup[(mode, regime, permutation, context)] = record
    deployed_sources: dict[str, dict[str, dict[str, Any]]] = {
        "aware": {},
        "blind": {},
    }
    for mode in thresholds:
        regimes = sorted(
            {regime for candidate_mode, regime, _, _ in pools if candidate_mode == mode}
        )
        for regime in regimes:
            eligible_records = [
                record
                for record in stratum_records
                if record["mode"] == mode and record["noise_regime"] == regime
            ]
            source = max(
                eligible_records,
                key=lambda record: float(record["stratum_threshold"]),
            )
            thresholds[mode][regime] = float(source["stratum_threshold"])
            deployed_sources[mode][regime] = {
                "noise_permutation": source["noise_permutation"],
                "context": source["context"],
                "stratum_threshold": source["stratum_threshold"],
            }
    rows: list[dict[str, Any]] = []
    for row in provisional:
        key = (
            str(row["mode"]),
            str(row["noise_regime"]),
            str(row["noise_permutation"]),
            str(row["context"]),
        )
        stratum = stratum_lookup[key]
        threshold = thresholds[key[0]][key[1]]
        rows.append(
            {
                **row,
                "stratum_pool_size": int(stratum["pool_size"]),
                "stratum_conformal_rank": int(stratum["conformal_rank"]),
                "stratum_threshold": float(stratum["stratum_threshold"]),
                "deployed_threshold": threshold,
                "normalized_client_statistic": float(row["raw_client_statistic"])
                / threshold,
                "stratum_tail": float(row["raw_client_statistic"])
                > float(stratum["stratum_threshold"]),
                "deployed_tail": float(row["raw_client_statistic"]) > threshold,
                "statistical_tail": float(row["raw_client_statistic"]) > threshold,
            }
        )
    cohort_keys = {
        (
            int(row["seed"]),
            int(row["draw"]),
            str(row["context"]),
            str(row["noise_regime"]),
            str(row["noise_permutation"]),
            int(row["cohort_seed"]),
        )
        for row in provisional
    }
    expected_cohorts = (
        len(config["randomness"]["calibration_seeds"])
        * draws
        * len(config["randomness"]["calibration_contexts"])
        * len(base._noise_cells(config))
    )
    if len(cohort_keys) != expected_cohorts:
        raise RuntimeError("Calibration cohorts are not uniquely keyed")
    expected_pool_size = len(config["randomness"]["calibration_seeds"]) * draws
    if any(len(values) != expected_pool_size for values in pools.values()):
        raise RuntimeError("A calibration stratum has an unexpected pool size")
    return (
        {
            "protocol": "offline_split_conformal_one_public_probe_per_independent_stratum_cohort",
            "coverage_scope": "marginal_regular_probe_same_synthetic_stratum",
            "simultaneous_all_clients_coverage_claimed": False,
            "conditional_noise_tier_coverage_claimed": False,
            "honest_outlier_coverage_claimed": False,
            "fashion_mnist_transfer_claimed": False,
            "dp_certificate_source": "global_cap_2G_over_n_not_calibration",
            "probe_selection": "dedicated_public_rng_independent_of_vectors_and_scores",
            "deployed_threshold_rule": "maximum_stratum_quantile_within_mode_and_noise_regime",
            "calibration_seeds": [
                int(value) for value in config["randomness"]["calibration_seeds"]
            ],
            "draws_per_seed": draws,
            "miscoverage": miscoverage,
            "thresholds": thresholds,
            "deployed_threshold_sources": deployed_sources,
            "strata": stratum_records,
            "expected_independent_cohorts": expected_cohorts,
            "observed_unique_cohorts": len(cohort_keys),
            "expected_pool_size_per_stratum": expected_pool_size,
            "calibration_and_development_disjoint": True,
            "holdout_opened": False,
        },
        rows,
    )


def _radii(
    config: Mapping[str, Any],
    variances: torch.Tensor,
    calibration: Mapping[str, Any],
    *,
    regime_name: str,
    blind: bool,
) -> torch.Tensor:
    scales = base._public_scale(config, variances)
    mode = "blind" if blind else "aware"
    if blind:
        scales = scales.mean(dim=0, keepdim=True).expand_as(scales)
    return scales * float(calibration["thresholds"][mode][regime_name])


def _reference(
    method: str,
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    blind_radii: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "uniform_mean":
        return vectors.mean(dim=0), {}
    if method == "fcc":
        return (
            centered_clipping(
                vectors,
                anchor=anchor,
                tau=float(config["references"]["fcc_radius"]),
            ),
            {},
        )
    if method == "rfa":
        settings = config["references"]["rfa"]
        return (
            geometric_median(
                vectors,
                max_iter=int(settings["max_iter"]),
                tol=float(settings["tolerance"]),
                smoothing=float(settings["smoothing"]),
            ),
            {},
        )
    if method == "trimmed_mean":
        return (
            trimmed_mean(
                vectors,
                int(config["references"]["trimmed_mean"]["trim_count"]),
            ),
            {},
        )
    if method == "coordinate_median":
        return coordinate_median(vectors), {}
    if method == NO_GATE:
        radii = torch.full_like(aware_radii, 1.0e12)
    elif method == BLIND:
        radii = blind_radii
    elif method == PRIMARY:
        radii = aware_radii
    else:
        raise ValueError(f"Unknown G0g-K2 candidate: {method}")
    return gaussian_aware_fixed_anchor_scalar_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=radii,
        block_sizes=tuple(int(value) for value in config["cohort"]["block_sizes"]),
        influence_cap=float(config["references"]["total_client_influence_cap"]),
        gate_transition_width=float(config["references"]["gate_transition_width"]),
        return_diagnostics=True,
    )


def _tier_fields(
    *,
    normalized: torch.Tensor,
    gates: torch.Tensor,
    tiers: torch.Tensor,
    regular: torch.Tensor,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for tier in TIERS:
        selected = regular & torch.isclose(
            tiers, torch.tensor(tier, dtype=tiers.dtype, device=tiers.device)
        )
        count = int(selected.sum().item())
        key = str(tier).replace(".", "p")
        result[f"regular_count_tier_{key}"] = count
        result[f"tail_count_tier_{key}"] = (
            int((normalized[selected] > 1.0).sum().item()) if count else 0
        )
        result[f"gate_sum_tier_{key}"] = (
            float(gates[selected].sum().item()) if count else 0.0
        )
    return result


def _evaluate_cell(
    config: dict[str, Any],
    calibration: Mapping[str, Any],
    *,
    seed: int,
    regime: dict[str, Any],
    permutation: str,
    geometry: str,
    threat: str,
) -> list[dict[str, Any]]:
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    clean, outliers, centre, anchor = oracle._honest_clean_vectors(
        config,
        seed=seed,
        draw=0,
        geometry=geometry,
        include_outliers=True,
    )
    variances, tiers = oracle._noise_variances(config, regime, permutation)
    observed = base._paired_private_noise(
        clean, variances, blocks, seed=seed, draw=0, geometry=geometry
    )
    attacked, byzantine = oracle._replace_with_attack(
        observed,
        config,
        threat=threat,
        severity=float(config["threats"]["severity"]),
        seed=oracle._seed(
            "g0g-k2-attack", seed, regime["name"], permutation, geometry, threat
        ),
    )
    server_clip = float(config["aggregation"]["server_clip_norm"])
    vectors = clip_l2(attacked, server_clip)
    aware_radii = _radii(
        config,
        variances,
        calibration,
        regime_name=str(regime["name"]),
        blind=False,
    )
    blind_radii = _radii(
        config,
        variances,
        calibration,
        regime_name=str(regime["name"]),
        blind=True,
    )
    honest = ~byzantine
    regular = honest & ~outliers
    target = clean[honest].mean(dim=0)
    pair_id = f"{seed}|{regime['name']}|{permutation}|{geometry}|{threat}"
    result: list[dict[str, Any]] = []
    for method in CANDIDATES:
        reference, diagnostics = _reference(
            method,
            vectors,
            anchor=anchor,
            aware_radii=aware_radii,
            blind_radii=blind_radii,
            config=config,
        )
        normalized_values = diagnostics.get("normalized_residuals_by_client")
        gate_values = diagnostics.get("gates_by_client")
        contribution_norms = diagnostics.get("client_contribution_norms")
        tier_fields = {
            f"{prefix}_tier_{str(tier).replace('.', 'p')}": 0
            for tier in TIERS
            for prefix in ("regular_count", "tail_count", "gate_sum")
        }
        tail_rate = float("nan")
        tier_gap = float("nan")
        gate_mean = float("nan")
        byzantine_share = float("nan")
        if normalized_values is not None and gate_values is not None:
            normalized = torch.tensor(
                normalized_values, dtype=vectors.dtype, device=vectors.device
            )
            gates = torch.tensor(
                gate_values, dtype=vectors.dtype, device=vectors.device
            )
            tail_rate = float((normalized[regular] > 1.0).float().mean().item())
            gate_mean = float(gates.mean().item())
            tier_fields = _tier_fields(
                normalized=normalized, gates=gates, tiers=tiers, regular=regular
            )
            rates = []
            for tier in TIERS:
                key = str(tier).replace(".", "p")
                count = int(tier_fields[f"regular_count_tier_{key}"])
                if count:
                    rates.append(float(tier_fields[f"tail_count_tier_{key}"]) / count)
            tier_gap = max(rates) - min(rates) if len(rates) > 1 else 0.0
        if contribution_norms is not None and bool(byzantine.any()):
            norms = torch.tensor(
                contribution_norms, dtype=vectors.dtype, device=vectors.device
            )
            total = float(norms.sum().item())
            byzantine_share = (
                float(norms[byzantine].sum().item()) / total if total else 0.0
            )
        result.append(
            {
                "pairing_id": pair_id,
                "seed": seed,
                "noise_regime": str(regime["name"]),
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "candidate": method,
                "reference_error": float(
                    torch.linalg.vector_norm(reference - target).item()
                ),
                "error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "regular_false_tail_rate": tail_rate,
                "regular_false_tail_tier_gap_cell": tier_gap,
                "gate_mean": gate_mean,
                "hard_reject_fraction": float(
                    diagnostics.get("hard_reject_fraction", float("nan"))
                ),
                "global_cap_active_fraction": float(
                    diagnostics.get("global_cap_active_fraction", float("nan"))
                ),
                "byzantine_realized_contribution_share": byzantine_share,
                "max_client_contribution_norm": (
                    max(float(value) for value in contribution_norms)
                    if contribution_norms is not None
                    else float("nan")
                ),
                "complete_client_influence_cap": float(
                    diagnostics.get("complete_client_influence_cap", float("nan"))
                ),
                "contribution_cap_respected": bool(
                    diagnostics.get("client_contribution_cap_respected", True)
                ),
                "replace_one_bound": float(
                    diagnostics.get("replace_one_bound", float("nan"))
                ),
                "server_clip_rate_honest": float(
                    (torch.linalg.vector_norm(attacked[honest], dim=1) > server_clip)
                    .float()
                    .mean()
                    .item()
                ),
                "server_clip_rate_byzantine": (
                    float(
                        (
                            torch.linalg.vector_norm(attacked[byzantine], dim=1)
                            > server_clip
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    if bool(byzantine.any())
                    else float("nan")
                ),
                "statistical_radius_min": float(
                    (blind_radii if method == BLIND else aware_radii).min().item()
                ),
                "statistical_radius_max": float(
                    (blind_radii if method == BLIND else aware_radii).max().item()
                ),
                "all_finite": bool(torch.isfinite(reference).all()),
                "gate_sum_normalized": bool(
                    diagnostics.get("normalization_by_gate_sum", False)
                ),
                **tier_fields,
            }
        )
    return result


def _attach_baselines(rows: list[dict[str, Any]]) -> None:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    for row in rows:
        fcc = lookup[(row["pairing_id"], "fcc")]
        blind = lookup[(row["pairing_id"], BLIND)]
        row["difference_vs_fcc"] = float(row["reference_error"]) - float(
            fcc["reference_error"]
        )
        row["ratio_to_fcc"] = float(row["reference_error"]) / max(
            float(fcc["reference_error"]), 1.0e-12
        )
        row["difference_vs_sigma_blind"] = float(row["reference_error"]) - float(
            blind["reference_error"]
        )
        row["ratio_to_sigma_blind"] = float(row["reference_error"]) / max(
            float(blind["reference_error"]), 1.0e-12
        )


def _replace_one_audit(
    config: dict[str, Any], calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in base._noise_cells(config):
            clean, _, _, anchor = oracle._honest_clean_vectors(
                config,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
                include_outliers=True,
            )
            variances, _ = oracle._noise_variances(config, regime, permutation)
            observed = base._paired_private_noise(
                clean,
                variances,
                blocks,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
            )
            vectors = clip_l2(
                observed, float(config["aggregation"]["server_clip_norm"])
            )
            aware = _radii(
                config,
                variances,
                calibration,
                regime_name=str(regime["name"]),
                blind=False,
            )
            blind = _radii(
                config,
                variances,
                calibration,
                regime_name=str(regime["name"]),
                blind=True,
            )
            for trial in range(trials):
                client = (
                    oracle._seed(
                        "g0g-k2-replace-client",
                        seed,
                        regime["name"],
                        permutation,
                        trial,
                    )
                    % vectors.shape[0]
                )
                neighbour = vectors.clone()
                replacement = torch.randn(
                    vectors.shape[1],
                    generator=oracle._generator(
                        "g0g-k2-replacement", seed, regime["name"], permutation, trial
                    ),
                    dtype=vectors.dtype,
                    device=vectors.device,
                )
                neighbour[client] = 100.0 * replacement
                left, diagnostics = _reference(
                    PRIMARY,
                    vectors,
                    anchor=anchor,
                    aware_radii=aware,
                    blind_radii=blind,
                    config=config,
                )
                right, _ = _reference(
                    PRIMARY,
                    neighbour,
                    anchor=anchor,
                    aware_radii=aware,
                    blind_radii=blind,
                    config=config,
                )
                observed_difference = float(
                    torch.linalg.vector_norm(left - right).item()
                )
                bound = float(diagnostics["replace_one_bound"])
                rows.append(
                    {
                        "seed": int(seed),
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": permutation,
                        "trial": trial,
                        "replaced_client": int(client),
                        "observed_replace_one_difference": observed_difference,
                        "theoretical_replace_one_bound": bound,
                        "ratio_observed_to_bound": observed_difference / bound,
                        "violation": observed_difference > bound + 1.0e-6,
                    }
                )
    return rows


def _selected(
    rows: list[dict[str, Any]], candidate: str, predicate
) -> list[dict[str, Any]]:
    return [row for row in rows if row["candidate"] == candidate and predicate(row)]


def _ratio(
    rows: list[dict[str, Any]], candidate: str, baseline: str, predicate
) -> float:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    selected = _selected(rows, candidate, predicate)
    numerator = base._finite_mean(row["reference_error"] for row in selected)
    denominator = base._finite_mean(
        lookup[(row["pairing_id"], baseline)]["reference_error"] for row in selected
    )
    return numerator / max(denominator, 1.0e-12)


def _seed_contrasts(
    rows: list[dict[str, Any]], competitor: str, threats: set[str]
) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["noise_regime"] == "heteroscedastic"
            and row["threat"] in threats
            and row["candidate"] in {PRIMARY, competitor}
        ):
            grouped[(int(row["seed"]), str(row["candidate"]))].append(
                float(row["reference_error"])
            )
    return [
        base._finite_mean(grouped[(seed, PRIMARY)])
        - base._finite_mean(grouped[(seed, competitor)])
        for seed in sorted({key[0] for key in grouped})
    ]


def _worst_group_ratio(
    rows: list[dict[str, Any]], *, baseline: str, threats: set[str]
) -> float:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["candidate"] == PRIMARY
            and row["noise_regime"] == "heteroscedastic"
            and row["threat"] in threats
        ):
            field = "ratio_to_fcc" if baseline == "fcc" else "ratio_to_sigma_blind"
            grouped[
                (
                    str(row["noise_permutation"]),
                    str(row["outlier_geometry"]),
                    str(row["threat"]),
                )
            ].append(float(row[field]))
    return max(base._finite_mean(values) for values in grouped.values())


def _pooled_tier_diagnostics(
    rows: list[dict[str, Any]], candidate: str
) -> dict[str, Any]:
    selected = _selected(
        rows,
        candidate,
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] == "none"
        ),
    )
    tail_rates: dict[str, float] = {}
    gate_means: dict[str, float] = {}
    for tier in TIERS:
        key = str(tier).replace(".", "p")
        count = sum(int(row[f"regular_count_tier_{key}"]) for row in selected)
        tails = sum(int(row[f"tail_count_tier_{key}"]) for row in selected)
        gates = sum(float(row[f"gate_sum_tier_{key}"]) for row in selected)
        if count:
            tail_rates[str(tier)] = tails / float(count)
            gate_means[str(tier)] = gates / float(count)
    return {
        "tail_rates": tail_rates,
        "gate_means": gate_means,
        "tail_rate_gap": max(tail_rates.values()) - min(tail_rates.values()),
        "gate_mean_gap": max(gate_means.values()) - min(gate_means.values()),
    }


def _evaluate_gates(
    rows: list[dict[str, Any]],
    replace_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    separated = set(
        str(value) for value in config["threats"]["separated_for_primary_gate"]
    )
    primary = [row for row in rows if row["candidate"] == PRIMARY]
    clean = [row for row in primary if row["threat"] == "none"]
    fixed_identity = max(
        abs(float(row["difference_vs_fcc"]))
        for row in rows
        if row["candidate"] == NO_GATE
    )
    tail_rate = base._finite_mean(row["regular_false_tail_rate"] for row in clean)
    pooled = _pooled_tier_diagnostics(rows, PRIMARY)
    homogeneous_ratio = _ratio(
        rows,
        PRIMARY,
        "fcc",
        lambda row: row["noise_regime"] == "homogeneous" and row["threat"] == "none",
    )
    heteroscedastic_ratio = _ratio(
        rows,
        PRIMARY,
        "fcc",
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] == "none"
        ),
    )
    fcc_contrasts = _seed_contrasts(rows, "fcc", separated)
    blind_contrasts = _seed_contrasts(rows, BLIND, separated)
    fcc_ci = base._ci95(fcc_contrasts)
    blind_ci = base._ci95(blind_contrasts)
    attacked = _selected(
        rows,
        PRIMARY,
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
        ),
    )
    mean_primary = base._finite_mean(row["reference_error"] for row in attacked)
    mean_fcc = mean_primary - float(fcc_ci["mean"])
    mean_blind = mean_primary - float(blind_ci["mean"])
    gain_fcc = (mean_fcc - mean_primary) / max(mean_fcc, 1.0e-12)
    gain_blind = (mean_blind - mean_primary) / max(mean_blind, 1.0e-12)
    byzantine_high_ratio = _ratio(
        rows,
        PRIMARY,
        BLIND,
        lambda row: (
            row["noise_regime"] == "heteroscedastic"
            and row["noise_permutation"] == "byzantine_high"
            and row["threat"] in separated
        ),
    )
    worst_fcc = _worst_group_ratio(rows, baseline="fcc", threats=separated)
    worst_alie = _worst_group_ratio(rows, baseline="fcc", threats={"alie"})
    byzantine_share = max(
        base._finite_mean(
            row["byzantine_realized_contribution_share"]
            for row in attacked
            if (row["noise_permutation"], row["outlier_geometry"], row["threat"])
            == group
        )
        for group in {
            (row["noise_permutation"], row["outlier_geometry"], row["threat"])
            for row in attacked
        }
    )
    expected = (
        len(config["randomness"]["development_seeds"])
        * len(base._noise_cells(config))
        * len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["threats"]["names"])
        * len(CANDIDATES)
    )
    complete_fraction = len(rows) / float(expected)
    finite_fraction = base._finite_mean(
        1.0 if row["all_finite"] else 0.0 for row in rows
    )
    cap_violations = sum(not bool(row["contribution_cap_respected"]) for row in primary)
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    checks = {
        "complete": complete_fraction >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "fixed_global_cap_identity": fixed_identity
        <= float(gates["fixed_global_cap_identity_abs_error_max"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "false_tail_rate": tail_rate <= float(gates["regular_false_tail_rate_max"]),
        "pooled_false_tail_tier_gap": float(pooled["tail_rate_gap"])
        <= float(gates["pooled_false_tail_tier_gap_max"]),
        "pooled_gate_mean_tier_gap": float(pooled["gate_mean_gap"])
        <= float(gates["pooled_gate_mean_tier_gap_max"]),
        "homogeneous_clean_noninferiority": homogeneous_ratio
        <= float(gates["homogeneous_clean_error_ratio_to_fcc_max"]),
        "heteroscedastic_clean_noninferiority": heteroscedastic_ratio
        <= float(gates["heteroscedastic_clean_error_ratio_to_fcc_max"]),
        "attacked_ci_vs_fcc": float(fcc_ci["high"])
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "attacked_gain_vs_fcc": gain_fcc
        >= float(gates["heteroscedastic_attacked_relative_gain_vs_fcc_min"]),
        "attacked_ci_vs_sigma_blind": float(blind_ci["high"])
        <= float(
            gates["heteroscedastic_attacked_sigma_blind_difference_ci95_high_max"]
        ),
        "attacked_gain_vs_sigma_blind": gain_blind
        >= float(gates["heteroscedastic_attacked_relative_gain_vs_sigma_blind_min"]),
        "byzantine_high_noninferiority_vs_sigma_blind": byzantine_high_ratio
        <= float(gates["byzantine_high_error_ratio_to_sigma_blind_max"]),
        "attacked_worst_group_ratio": worst_fcc
        <= float(gates["attacked_error_ratio_to_fcc_max"]),
        "alie_ratio": worst_alie <= float(gates["alie_error_ratio_to_fcc_max"]),
        "byzantine_contribution_share": byzantine_share
        <= float(gates["byzantine_realized_contribution_share_max"]),
    }
    return {
        "decision": "promote_to_holdout"
        if all(checks.values())
        else "stop_after_development",
        "all_gates_pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "development_rows": len(rows),
            "expected_development_rows": expected,
            "complete_fraction": complete_fraction,
            "finite_fraction": finite_fraction,
            "fixed_global_cap_max_abs_difference_vs_fcc": fixed_identity,
            "contribution_cap_violations": cap_violations,
            "replace_one_trials": len(replace_rows),
            "replace_one_violations": replace_violations,
            "replace_one_max_ratio_observed_to_bound": max(
                float(row["ratio_observed_to_bound"]) for row in replace_rows
            ),
            "regular_false_tail_rate": tail_rate,
            "pooled_tier_diagnostics": pooled,
            "homogeneous_clean_error_ratio_to_fcc": homogeneous_ratio,
            "heteroscedastic_clean_error_ratio_to_fcc": heteroscedastic_ratio,
            "heteroscedastic_attacked_difference_vs_fcc_seed_ci95": fcc_ci,
            "heteroscedastic_attacked_relative_gain_vs_fcc": gain_fcc,
            "heteroscedastic_attacked_difference_vs_sigma_blind_seed_ci95": blind_ci,
            "heteroscedastic_attacked_relative_gain_vs_sigma_blind": gain_blind,
            "byzantine_high_error_ratio_to_sigma_blind": byzantine_high_ratio,
            "heteroscedastic_attacked_worst_group_ratio_to_fcc": worst_fcc,
            "heteroscedastic_alie_worst_group_ratio_to_fcc": worst_alie,
            "heteroscedastic_attacked_byzantine_contribution_share_worst_group": byzantine_share,
        },
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["noise_regime"], row["threat"])].append(row)
    result: list[dict[str, Any]] = []
    for (candidate, regime, threat), selected in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "noise_regime": regime,
                "threat": threat,
                "n_rows": len(selected),
                "reference_error_mean": base._finite_mean(
                    row["reference_error"] for row in selected
                ),
                "reference_error_std": base._finite_std(
                    row["reference_error"] for row in selected
                ),
                "ratio_to_fcc_mean": base._finite_mean(
                    row["ratio_to_fcc"] for row in selected
                ),
                "ratio_to_sigma_blind_mean": base._finite_mean(
                    row["ratio_to_sigma_blind"] for row in selected
                ),
                "regular_false_tail_rate_mean": base._finite_mean(
                    row["regular_false_tail_rate"] for row in selected
                ),
                "gate_mean": base._finite_mean(row["gate_mean"] for row in selected),
                "byzantine_contribution_share_mean": base._finite_mean(
                    row["byzantine_realized_contribution_share"] for row in selected
                ),
            }
        )
    return result


def _make_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    key_methods = ["fcc", BLIND, PRIMARY]
    paths: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, regime in zip(axes, ("homogeneous", "heteroscedastic"), strict=True):
        threats = list(dict.fromkeys(str(row["threat"]) for row in rows))
        x = range(len(threats))
        width = 0.25
        for index, method in enumerate(key_methods):
            means = [
                base._finite_mean(
                    row["reference_error"]
                    for row in rows
                    if row["candidate"] == method
                    and row["noise_regime"] == regime
                    and row["threat"] == threat
                )
                for threat in threats
            ]
            axis.bar(
                [value + (index - 1) * width for value in x], means, width, label=method
            )
        axis.set_xticks(list(x), threats, rotation=25, ha="right")
        axis.set_title(
            "Bruit homogène" if regime == "homogeneous" else "Bruit hétéroscédastique"
        )
        axis.set_ylabel("Erreur L2 de référence")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    path = figure_dir / "k2_reference_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    separated = {"ipm", "bitflip_x10", "model_replacement"}
    selected = _selected(
        rows,
        PRIMARY,
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
        ),
    )
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_seed[int(row["seed"])].append(float(row["difference_vs_fcc"]))
    seeds = sorted(by_seed)
    differences = [base._finite_mean(by_seed[seed]) for seed in seeds]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.bar(
        [str(seed) for seed in seeds],
        differences,
        color=["#188977" if value <= 0.0 else "#d95f02" for value in differences],
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Erreur G0g-K2 − erreur FCC")
    axis.set_title("Contraste apparié : bruit hétéroscédastique, attaques séparées")
    axis.grid(axis="y", alpha=0.25)
    path = figure_dir / "k2_paired_difference_by_seed.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    calibration: Mapping[str, Any],
    decision: Mapping[str, Any],
    figures: Sequence[str],
) -> None:
    checks = decision["checks"]
    observed = decision["observed"]
    lines = [
        "# G0g-K2 — décision expérimentale",
        "",
        "## Verdict",
        "",
        "**"
        + (
            "PROMOTION AU HOLDOUT"
            if decision["all_gates_pass"]
            else "ARRÊT AU DÉVELOPPEMENT"
        )
        + "**",
        "",
        "K2 remplace le cap par bloc de K1 par un cap L2 global. Sans gate, le "
        "mécanisme est exactement FCC; la covariance ne règle que la tolérance "
        "statistique de la gate.",
        "",
        "| Test préenregistré | Observation | Verdict |",
        "|---|---:|:---:|",
        f"| Identité contrôle sans gate − FCC | {_fmt(observed['fixed_global_cap_max_abs_difference_vs_fcc'])} | {'✓' if checks['fixed_global_cap_identity'] else '✗'} |",
        f"| Ratio propre homogène / FCC | {_fmt(observed['homogeneous_clean_error_ratio_to_fcc'])} | {'✓' if checks['homogeneous_clean_noninferiority'] else '✗'} |",
        f"| Ratio propre hétéroscédastique / FCC | {_fmt(observed['heteroscedastic_clean_error_ratio_to_fcc'])} | {'✓' if checks['heteroscedastic_clean_noninferiority'] else '✗'} |",
        f"| Gain attaques séparées vs FCC | {_fmt(100 * observed['heteroscedastic_attacked_relative_gain_vs_fcc'], 2)} % | {'✓' if checks['attacked_gain_vs_fcc'] else '✗'} |",
        f"| IC95 G0g-K2−FCC, borne haute | {_fmt(observed['heteroscedastic_attacked_difference_vs_fcc_seed_ci95']['high'])} | {'✓' if checks['attacked_ci_vs_fcc'] else '✗'} |",
        f"| Gain attaques séparées vs sigma-blind | {_fmt(100 * observed['heteroscedastic_attacked_relative_gain_vs_sigma_blind'], 2)} % | {'✓' if checks['attacked_gain_vs_sigma_blind'] else '✗'} |",
        f"| Ratio vs sigma-blind, Byzantins niveau de bruit élevé | {_fmt(observed['byzantine_high_error_ratio_to_sigma_blind'])} | {'✓' if checks['byzantine_high_noninferiority_vs_sigma_blind'] else '✗'} |",
        f"| Faux-tail honnêtes réguliers | {_fmt(100 * observed['regular_false_tail_rate'], 2)} % | {'✓' if checks['false_tail_rate'] else '✗'} |",
        f"| Écart pooled faux-tail entre tiers | {_fmt(100 * observed['pooled_tier_diagnostics']['tail_rate_gap'], 2)} pp | {'✓' if checks['pooled_false_tail_tier_gap'] else '✗'} |",
        f"| Écart pooled de gate moyenne entre tiers | {_fmt(100 * observed['pooled_tier_diagnostics']['gate_mean_gap'], 2)} pp | {'✓' if checks['pooled_gate_mean_tier_gap'] else '✗'} |",
        f"| Part Byzantine maximale | {_fmt(100 * observed['heteroscedastic_attacked_byzantine_contribution_share_worst_group'], 2)} % | {'✓' if checks['byzantine_contribution_share'] else '✗'} |",
        "",
        "## Certificat",
        "",
        f"- cap : {observed['contribution_cap_violations']} violation;",
        f"- replace-one : {observed['replace_one_violations']} violation sur {observed['replace_one_trials']} essais;",
        f"- maximum observé / borne 2G/n : {_fmt(observed['replace_one_max_ratio_observed_to_bound'])};",
        "- borne analytique : 2 × 0,13 / 25 = 0,0104;",
        "- coût de confidentialité additionnel : zéro, par post-traitement.",
        "",
        "## Table synthétique",
        "",
        "| Référence | Bruit | Attaque | Erreur L2 moyenne ± écart-type | Ratio/FCC | Ratio/sigma-blind |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        if row["candidate"] not in {"fcc", BLIND, PRIMARY}:
            continue
        lines.append(
            f"| {row['candidate']} | {row['noise_regime']} | {row['threat']} | "
            f"{_fmt(row['reference_error_mean'])} ± {_fmt(row['reference_error_std'])} | "
            f"{_fmt(row['ratio_to_fcc_mean'])} | {_fmt(row['ratio_to_sigma_blind_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interprétation autorisée",
            "",
            "Ce screen mesure une erreur de référence synthétique. Il ne mesure ni "
            "accuracy, ni fairness, ni convergence end-to-end. La comparaison "
            "covariance-aware contre sigma-blind isole la valeur propre de l'information "
            "de covariance; la comparaison contre FCC isole la valeur de la gate entière.",
            "",
            f"Seuils conformal : `{calibration['thresholds']}`.",
            "",
        ]
    )
    if figures:
        lines.extend(["## Figures", ""])
        for figure in figures:
            lines.extend([f"![{Path(figure).stem}]({ROOT / figure})", ""])
    failed = [name for name, passed in checks.items() if not passed]
    lines.extend(
        [
            "## Étape suivante",
            "",
            (
                "Tous les gates passent : ouvrir le holdout à sept seeds, sans changer "
                "les paramètres."
                if not failed
                else "Gates échoués : **" + ", ".join(failed) + "**. K2 n'est pas "
                "promu; aucun résultat Fashion-MNIST ne doit être produit pour K2."
            ),
            "",
            f"Configuration : `{config_path.relative_to(ROOT)}`. Résultats : `{output_dir.relative_to(ROOT)}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    oracle._configure_runtime("mps")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        output_dir / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": base._sha256(config_path),
            "source_sha256": {
                "runner": base._sha256(Path(__file__).resolve()),
                "algorithm": base._sha256(
                    ROOT / "algorithms/gaussian_aware_reference.py"
                ),
                "shared_runner": base._sha256(
                    ROOT / "scripts/run_gaussian_aware_reference_g0g_k1.py"
                ),
            },
            "device": str(oracle._RUNTIME_DEVICE),
            "dtype": str(oracle._RUNTIME_DTYPE),
            "holdout_opened": False,
        },
    )
    calibration, calibration_rows = _calibrate(config)
    base._atomic_json(output_dir / "calibration.json", calibration)
    base._write_csv(output_dir / "calibration_rows.csv", calibration_rows)
    rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in base._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for threat in config["threats"]["names"]:
                    rows.extend(
                        _evaluate_cell(
                            config,
                            calibration,
                            seed=int(seed),
                            regime=regime,
                            permutation=permutation,
                            geometry=str(geometry),
                            threat=str(threat),
                        )
                    )
    _attach_baselines(rows)
    base._write_csv(output_dir / "development_rows.csv", rows)
    replace_rows = _replace_one_audit(config, calibration)
    base._write_csv(output_dir / "replace_one_rows.csv", replace_rows)
    summaries = _summaries(rows)
    base._write_csv(output_dir / "summary.csv", summaries)
    decision = _evaluate_gates(rows, replace_rows, config)
    decision["holdout_opened"] = False
    decision["privacy_claim"] = "zero additional local-DP cost by post-processing"
    base._atomic_json(output_dir / "decision.json", decision)
    figures = _make_figures(rows, output_dir)
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        summaries=summaries,
        calibration=calibration,
        decision=decision,
        figures=figures,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k2.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_G0g_K2_Monday_Report.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
