#!/usr/bin/env python3
"""Run the preregistered G0g-K1 Gaussian-aware reference screen.

The experiment separates two roles that were coupled in G0f:

* authenticated public DP covariance sets a statistical acceptance radius;
* one client-independent public cap sets the maximum authorised influence.

The script is reference-only.  It never reads an accuracy, never uses FAR
weights and never opens a holdout.  Production execution is MPS-only; the
small deterministic helpers remain importable on CPU for unit tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_fixed_anchor_gated_reference,
)
from robustness.aggregators import (  # noqa: E402
    centered_clipping,
    clip_l2,
    coordinate_median,
    geometric_median,
    trimmed_mean,
)
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

PRIMARY = "g0g_k1"
CANDIDATES = (
    "uniform_mean",
    "fcc",
    "rfa",
    "trimmed_mean",
    "coordinate_median",
    "fixed_cap_no_gate",
    "g0g_sigma_blind",
    PRIMARY,
)


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _t95(n: int) -> float:
    return {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(int(n), 1.96)


def _ci95(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    mean = float(statistics.fmean(finite))
    if len(finite) == 1:
        return {"n": 1, "mean": mean, "low": float("nan"), "high": float("nan")}
    half = _t95(len(finite)) * statistics.stdev(finite) / math.sqrt(len(finite))
    return {"n": len(finite), "mean": mean, "low": mean - half, "high": mean + half}


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for width in block_sizes:
        result.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(result)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Rows for {path} do not share one schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError("The frozen G0g top-level schema changed")
    if config["campaign_id"] != "gaussian_aware_reference_g0g_k1_mps_v1":
        raise ValueError("Unexpected G0g campaign id")
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
        "covariance_provenance": "public_authenticated",
        "client_declared_covariance_forbidden": True,
        "calibration_is_offline": True,
        "development_only_screen": True,
    }
    if contract != required:
        raise ValueError("The frozen G0g scientific contract changed")
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    if n < 3 or not 0 <= b < n / 2:
        raise ValueError("G0g needs n>=3 and 0<=b<n/2")
    if sum(blocks) != int(cohort["dimension"]) or any(width <= 0 for width in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity standard deviation is required per block")
    if len(config["privacy_noise"]["block_std_multipliers"]) != len(blocks):
        raise ValueError("One DP-noise multiplier is required per block")
    names = tuple(str(value) for value in config["candidates"]["names"])
    if names != CANDIDATES or config["candidates"]["primary"] != PRIMARY:
        raise ValueError("The frozen candidate set or order changed")
    if not 0.0 < float(config["references"]["statistical_false_tail_rate"]) < 1.0:
        raise ValueError("statistical_false_tail_rate must lie in (0,1)")
    if float(config["references"]["total_client_influence_cap"]) <= 0.0:
        raise ValueError("total_client_influence_cap must be positive")
    if float(config["references"]["gate_transition_width"]) <= 0.0:
        raise ValueError("gate_transition_width must be positive")
    execution = config["execution"]
    if execution != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
    }:
        raise ValueError(
            "Production G0g execution must be MPS float32 without fallback"
        )
    used = {
        int(value)
        for fold in config["randomness"]["calibration_folds"]
        for value in fold
    }
    development = {int(value) for value in config["randomness"]["development_seeds"]}
    excluded = {int(value) for value in config["excluded_prior_seeds"]}
    if used & development or used & excluded or development & excluded:
        raise ValueError(
            "Calibration, development and excluded seed registries must be disjoint"
        )


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], str]]:
    cells: list[tuple[dict[str, Any], str]] = []
    for raw in config["privacy_noise"]["regimes"]:
        regime = dict(raw)
        for permutation in regime["permutations"]:
            cells.append((regime, str(permutation)))
    return cells


def _paired_private_noise(
    clean: torch.Tensor,
    variances: torch.Tensor,
    block_sizes: Sequence[int],
    *,
    seed: int,
    draw: int,
    geometry: str,
) -> torch.Tensor:
    """Use one standard Gaussian tensor for all public noise assignments."""

    coordinate_std = oracle._expand_block_values(variances.sqrt(), block_sizes)
    standard = torch.randn(
        clean.shape,
        generator=oracle._generator("g0g-paired-private-noise", seed, draw, geometry),
        dtype=clean.dtype,
        device=clean.device,
    )
    return clean + coordinate_std * standard


def _public_scale(
    config: Mapping[str, Any],
    variances: torch.Tensor,
) -> torch.Tensor:
    """Return public root-second-moment scales for block residual norms."""

    blocks = torch.tensor(
        config["cohort"]["block_sizes"],
        dtype=variances.dtype,
        device=variances.device,
    )
    heterogeneity = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=variances.dtype,
        device=variances.device,
    ).square()
    dimension = float(config["cohort"]["dimension"])
    anchor_variance = (
        float(config["references"]["public_anchor_error_norm"]) ** 2 / dimension
    )
    floor = float(config["references"]["variance_floor"])
    return (
        blocks[None, :] * (variances + heterogeneity[None, :] + anchor_variance + floor)
    ).sqrt()


def _conformal_quantile(values: torch.Tensor, miscoverage: float) -> tuple[float, int]:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("Conformal pool must be non-empty")
    ordered = torch.sort(values).values
    rank = min(
        int(values.numel()),
        int(math.ceil((int(values.numel()) + 1) * (1.0 - float(miscoverage)))),
    )
    return float(ordered[rank - 1].item()), rank


def _calibrate(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate one common standardized threshold per public block."""

    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    slices = _block_slices(block_sizes)
    pools: dict[int, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    seeds = [
        int(value)
        for fold in config["randomness"]["calibration_folds"]
        for value in fold
    ]
    draws = int(config["randomness"]["calibration_draws_per_seed"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    for seed in seeds:
        for draw in range(draws):
            for context in config["randomness"]["calibration_contexts"]:
                geometry = str(context["geometry"])
                clean, outliers, _, anchor = oracle._honest_clean_vectors(
                    config,
                    seed=oracle._seed(
                        "g0g-calibration", seed, draw, str(context["name"])
                    ),
                    draw=0,
                    geometry=geometry,
                    include_outliers=bool(context["include_outliers"]),
                )
                for regime, permutation in _noise_cells(config):
                    variances, tiers = oracle._noise_variances(
                        config, regime, permutation
                    )
                    observed = _paired_private_noise(
                        clean,
                        variances,
                        block_sizes,
                        seed=seed,
                        draw=draw,
                        geometry=geometry,
                    )
                    bounded = clip_l2(observed, server_clip)
                    scales = _public_scale(config, variances)
                    residuals = bounded - anchor[None, :]
                    for client in torch.where(~outliers)[0].tolist():
                        for block, block_slice in enumerate(slices):
                            standardized = float(
                                (
                                    torch.linalg.vector_norm(
                                        residuals[client, block_slice]
                                    )
                                    / scales[client, block]
                                ).item()
                            )
                            pools[block].append(standardized)
                            rows.append(
                                {
                                    "seed": seed,
                                    "draw": draw,
                                    "context": str(context["name"]),
                                    "noise_regime": str(regime["name"]),
                                    "noise_permutation": permutation,
                                    "noise_tier": float(tiers[client].item()),
                                    "client": int(client),
                                    "block": block,
                                    "standardized_residual": standardized,
                                }
                            )
    miscoverage = float(config["references"]["statistical_false_tail_rate"])
    thresholds: list[float] = []
    ranks: list[int] = []
    for block in range(len(block_sizes)):
        values = torch.tensor(
            pools[block], dtype=oracle._RUNTIME_DTYPE, device=oracle._RUNTIME_DEVICE
        )
        threshold, rank = _conformal_quantile(values, miscoverage)
        thresholds.append(threshold)
        ranks.append(rank)
    for row in rows:
        row["deployed_standardized_threshold"] = thresholds[int(row["block"])]
        row["statistical_tail"] = bool(
            float(row["standardized_residual"]) > thresholds[int(row["block"])]
        )
    artifact = {
        "protocol": "offline_split_conformal_fixed_public_anchor_regular_honest_blocks",
        "seeds": seeds,
        "draws_per_seed": draws,
        "miscoverage": miscoverage,
        "standardized_thresholds_by_block": thresholds,
        "conformal_rank_by_block": ranks,
        "pool_size_by_block": [len(pools[index]) for index in range(len(block_sizes))],
        "calibration_and_development_disjoint": True,
        "covariance_role": "statistical_tolerance_only",
    }
    return artifact, rows


def _radii(
    config: Mapping[str, Any],
    variances: torch.Tensor,
    calibration: Mapping[str, Any],
) -> torch.Tensor:
    thresholds = torch.tensor(
        calibration["standardized_thresholds_by_block"],
        dtype=variances.dtype,
        device=variances.device,
    )
    return _public_scale(config, variances) * thresholds[None, :]


def _block_caps(config: Mapping[str, Any], vectors: torch.Tensor) -> torch.Tensor:
    count = len(config["cohort"]["block_sizes"])
    total = float(config["references"]["total_client_influence_cap"])
    return torch.full(
        (count,),
        total / math.sqrt(float(count)),
        dtype=vectors.dtype,
        device=vectors.device,
    )


def _reference(
    method: str,
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    radii: torch.Tensor,
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
        return trimmed_mean(
            vectors, int(config["references"]["trimmed_mean"]["trim_count"])
        ), {}
    if method == "coordinate_median":
        return coordinate_median(vectors), {}

    candidate_radii = radii
    if method == "fixed_cap_no_gate":
        candidate_radii = torch.full_like(radii, 1.0e12)
    elif method == "g0g_sigma_blind":
        candidate_radii = radii.mean(dim=0, keepdim=True).expand_as(radii)
    elif method != PRIMARY:
        raise ValueError(f"Unknown G0g candidate: {method}")
    result, diagnostics = gaussian_aware_fixed_anchor_gated_reference(
        vectors,
        anchor=anchor,
        statistical_radii=candidate_radii,
        block_sizes=tuple(int(value) for value in config["cohort"]["block_sizes"]),
        influence_cap=_block_caps(config, vectors),
        gate_transition_width=float(config["references"]["gate_transition_width"]),
        return_diagnostics=True,
    )
    return result, diagnostics


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
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    clean, outliers, centre, anchor = oracle._honest_clean_vectors(
        config,
        seed=seed,
        draw=0,
        geometry=geometry,
        include_outliers=True,
    )
    variances, tiers = oracle._noise_variances(config, regime, permutation)
    observed = _paired_private_noise(
        clean, variances, block_sizes, seed=seed, draw=0, geometry=geometry
    )
    attacked, byzantine = oracle._replace_with_attack(
        observed,
        config,
        threat=threat,
        severity=float(config["threats"]["severity"]),
        seed=oracle._seed(
            "g0g-attack", seed, str(regime["name"]), permutation, geometry, threat
        ),
    )
    server_clip = float(config["aggregation"]["server_clip_norm"])
    vectors = clip_l2(attacked, server_clip)
    public_radii = _radii(config, variances, calibration)
    honest = ~byzantine
    regular = honest & ~outliers
    target = clean[honest].mean(dim=0)
    pair_id = f"{seed}|{regime['name']}|{permutation}|{geometry}|{threat}"
    result: list[dict[str, Any]] = []
    for method in CANDIDATES:
        reference, diagnostics = _reference(
            method, vectors, anchor=anchor, radii=public_radii, config=config
        )
        error = float(torch.linalg.vector_norm(reference - target).item())
        normalized = diagnostics.get("normalized_residuals_by_client_block")
        contribution_norms = diagnostics.get("client_contribution_norms")
        tail_rate = float("nan")
        tier_gap = float("nan")
        byzantine_share = float("nan")
        if normalized is not None:
            normalized_tensor = torch.tensor(
                normalized, dtype=vectors.dtype, device=vectors.device
            )
            regular_tails = normalized_tensor[regular] > 1.0
            tail_rate = float(regular_tails.float().mean().item())
            unique_tiers = torch.unique(tiers[regular])
            if unique_tiers.numel() <= 1:
                tier_gap = 0.0
            else:
                tier_rates = [
                    float(
                        (normalized_tensor[regular & tiers.eq(tier)] > 1.0)
                        .float()
                        .mean()
                        .item()
                    )
                    for tier in unique_tiers
                ]
                tier_gap = max(tier_rates) - min(tier_rates)
        if contribution_norms is not None and bool(byzantine.any()):
            norms = torch.tensor(
                contribution_norms, dtype=vectors.dtype, device=vectors.device
            )
            total = float(norms.sum().item())
            byzantine_share = (
                float(norms[byzantine].sum().item()) / total if total > 0.0 else 0.0
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
                "reference_error": error,
                "error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "regular_false_tail_rate": tail_rate,
                "regular_false_tail_tier_gap": tier_gap,
                "gate_mean": float(diagnostics.get("gate_mean", float("nan"))),
                "hard_reject_fraction": float(
                    diagnostics.get("hard_reject_fraction", float("nan"))
                ),
                "cap_active_fraction": float(
                    diagnostics.get("cap_active_fraction", float("nan"))
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
                "noise_tier_min": float(tiers.min().item()),
                "noise_tier_max": float(tiers.max().item()),
                "statistical_radius_min": float(public_radii.min().item()),
                "statistical_radius_max": float(public_radii.max().item()),
                "all_finite": bool(torch.isfinite(reference).all()),
                "gate_sum_normalized": bool(
                    diagnostics.get("normalization_by_gate_sum", False)
                ),
            }
        )
    return result


def _attach_paired_baselines(rows: list[dict[str, Any]]) -> None:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    for row in rows:
        fcc = lookup[(row["pairing_id"], "fcc")]
        blind = lookup[(row["pairing_id"], "g0g_sigma_blind")]
        row["difference_vs_fcc"] = float(row["reference_error"]) - float(
            fcc["reference_error"]
        )
        row["ratio_to_fcc"] = float(row["reference_error"]) / max(
            float(fcc["reference_error"]), 1.0e-12
        )
        row["difference_vs_sigma_blind"] = float(row["reference_error"]) - float(
            blind["reference_error"]
        )


def _replace_one_audit(
    config: dict[str, Any], calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in _noise_cells(config):
            clean, _, _, anchor = oracle._honest_clean_vectors(
                config,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
                include_outliers=True,
            )
            variances, _ = oracle._noise_variances(config, regime, permutation)
            observed = _paired_private_noise(
                clean,
                variances,
                block_sizes,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
            )
            vectors = clip_l2(
                observed, float(config["aggregation"]["server_clip_norm"])
            )
            public_radii = _radii(config, variances, calibration)
            for trial in range(trials):
                client = (
                    oracle._seed(
                        "g0g-replace-client", seed, regime["name"], permutation, trial
                    )
                    % vectors.shape[0]
                )
                neighbour = vectors.clone()
                replacement = torch.randn(
                    vectors.shape[1],
                    generator=oracle._generator(
                        "g0g-replacement", seed, regime["name"], permutation, trial
                    ),
                    dtype=vectors.dtype,
                    device=vectors.device,
                )
                neighbour[client] = 100.0 * replacement
                left, diagnostics = _reference(
                    PRIMARY,
                    vectors,
                    anchor=anchor,
                    radii=public_radii,
                    config=config,
                )
                right, _ = _reference(
                    PRIMARY,
                    neighbour,
                    anchor=anchor,
                    radii=public_radii,
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


def _seed_contrasts(
    rows: list[dict[str, Any]],
    *,
    competitor: str,
    threats: set[str],
) -> list[float]:
    selected = [
        row
        for row in rows
        if row["noise_regime"] == "heteroscedastic"
        and row["threat"] in threats
        and row["candidate"] in {PRIMARY, competitor}
    ]
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(int(row["seed"]), str(row["candidate"]))].append(
            float(row["reference_error"])
        )
    values: list[float] = []
    for seed in sorted({key[0] for key in grouped}):
        values.append(
            _finite_mean(grouped[(seed, PRIMARY)])
            - _finite_mean(grouped[(seed, competitor)])
        )
    return values


def _ratio(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    predicate,
    baseline: str = "fcc",
) -> float:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    selected = [row for row in rows if row["candidate"] == candidate and predicate(row)]
    numerators = [float(row["reference_error"]) for row in selected]
    denominators = [
        float(lookup[(row["pairing_id"], baseline)]["reference_error"])
        for row in selected
    ]
    return _finite_mean(numerators) / max(_finite_mean(denominators), 1.0e-12)


def _worst_group_ratio(rows: list[dict[str, Any]], *, threats: set[str]) -> float:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["candidate"] == PRIMARY
            and row["noise_regime"] == "heteroscedastic"
            and row["threat"] in threats
        ):
            grouped[
                (
                    str(row["noise_permutation"]),
                    str(row["outlier_geometry"]),
                    str(row["threat"]),
                )
            ].append(float(row["ratio_to_fcc"]))
    return max(_finite_mean(values) for values in grouped.values())


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
    regular_false_tail = _finite_mean(row["regular_false_tail_rate"] for row in clean)
    tier_gap = max(float(row["regular_false_tail_tier_gap"]) for row in clean)
    homogeneous_ratio = _ratio(
        rows,
        candidate=PRIMARY,
        predicate=lambda row: (
            row["noise_regime"] == "homogeneous" and row["threat"] == "none"
        ),
    )
    heteroscedastic_ratio = _ratio(
        rows,
        candidate=PRIMARY,
        predicate=lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] == "none"
        ),
    )
    fcc_contrasts = _seed_contrasts(rows, competitor="fcc", threats=separated)
    blind_contrasts = _seed_contrasts(
        rows, competitor="g0g_sigma_blind", threats=separated
    )
    fcc_ci = _ci95(fcc_contrasts)
    blind_ci = _ci95(blind_contrasts)
    attacked_primary = [
        row
        for row in primary
        if row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
    ]
    mean_primary = _finite_mean(row["reference_error"] for row in attacked_primary)
    mean_fcc = mean_primary - float(fcc_ci["mean"])
    mean_blind = mean_primary - float(blind_ci["mean"])
    gain_fcc = (mean_fcc - mean_primary) / max(mean_fcc, 1.0e-12)
    gain_blind = (mean_blind - mean_primary) / max(mean_blind, 1.0e-12)
    worst_attacked_ratio = _worst_group_ratio(rows, threats=separated)
    worst_alie_ratio = _worst_group_ratio(rows, threats={"alie"})
    byzantine_share = max(
        _finite_mean(
            row["byzantine_realized_contribution_share"]
            for row in attacked_primary
            if (row["noise_permutation"], row["outlier_geometry"], row["threat"]) == key
        )
        for key in {
            (row["noise_permutation"], row["outlier_geometry"], row["threat"])
            for row in attacked_primary
        }
    )
    expected_rows = (
        len(config["randomness"]["development_seeds"])
        * len(_noise_cells(config))
        * len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["threats"]["names"])
        * len(CANDIDATES)
    )
    observed_fraction = len(rows) / float(expected_rows)
    finite_fraction = _finite_mean(1.0 if row["all_finite"] else 0.0 for row in rows)
    cap_violations = sum(not bool(row["contribution_cap_respected"]) for row in primary)
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    checks = {
        "complete": observed_fraction >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "false_tail_rate": regular_false_tail
        <= float(gates["regular_false_tail_rate_max"]),
        "false_tail_tier_gap": tier_gap
        <= float(gates["regular_false_tail_tier_gap_max"]),
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
        "attacked_worst_group_ratio": worst_attacked_ratio
        <= float(gates["attacked_error_ratio_to_fcc_max"]),
        "alie_ratio": worst_alie_ratio <= float(gates["alie_error_ratio_to_fcc_max"]),
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
            "expected_development_rows": expected_rows,
            "complete_fraction": observed_fraction,
            "finite_fraction": finite_fraction,
            "contribution_cap_violations": cap_violations,
            "replace_one_trials": len(replace_rows),
            "replace_one_violations": replace_violations,
            "regular_false_tail_rate": regular_false_tail,
            "regular_false_tail_tier_gap_worst_cell": tier_gap,
            "homogeneous_clean_error_ratio_to_fcc": homogeneous_ratio,
            "heteroscedastic_clean_error_ratio_to_fcc": heteroscedastic_ratio,
            "heteroscedastic_attacked_difference_vs_fcc_seed_ci95": fcc_ci,
            "heteroscedastic_attacked_relative_gain_vs_fcc": gain_fcc,
            "heteroscedastic_attacked_difference_vs_sigma_blind_seed_ci95": blind_ci,
            "heteroscedastic_attacked_relative_gain_vs_sigma_blind": gain_blind,
            "heteroscedastic_attacked_worst_group_ratio_to_fcc": worst_attacked_ratio,
            "heteroscedastic_alie_worst_group_ratio_to_fcc": worst_alie_ratio,
            "heteroscedastic_attacked_byzantine_contribution_share_worst_group": byzantine_share,
            "replace_one_max_ratio_observed_to_bound": max(
                float(row["ratio_observed_to_bound"]) for row in replace_rows
            ),
        },
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["candidate"]), str(row["noise_regime"]), str(row["threat"]))
        ].append(row)
    result: list[dict[str, Any]] = []
    for (candidate, regime, threat), selected in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "noise_regime": regime,
                "threat": threat,
                "n_rows": len(selected),
                "reference_error_mean": _finite_mean(
                    row["reference_error"] for row in selected
                ),
                "reference_error_std": _finite_std(
                    row["reference_error"] for row in selected
                ),
                "ratio_to_fcc_mean": _finite_mean(
                    row["ratio_to_fcc"] for row in selected
                ),
                "regular_false_tail_rate_mean": _finite_mean(
                    row["regular_false_tail_rate"] for row in selected
                ),
                "gate_mean": _finite_mean(row["gate_mean"] for row in selected),
                "byzantine_contribution_share_mean": _finite_mean(
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
    key_methods = ["fcc", "fixed_cap_no_gate", "g0g_sigma_blind", PRIMARY]
    paths: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, regime in zip(axes, ("homogeneous", "heteroscedastic"), strict=True):
        threats = list(dict.fromkeys(str(row["threat"]) for row in rows))
        x = range(len(threats))
        width = 0.19
        for index, method in enumerate(key_methods):
            means = [
                _finite_mean(
                    row["reference_error"]
                    for row in rows
                    if row["candidate"] == method
                    and row["noise_regime"] == regime
                    and row["threat"] == threat
                )
                for threat in threats
            ]
            axis.bar(
                [value + (index - 1.5) * width for value in x],
                means,
                width,
                label=method,
            )
        axis.set_xticks(list(x), threats, rotation=25, ha="right")
        axis.set_title(
            "Bruit homogène" if regime == "homogeneous" else "Bruit hétéroscédastique"
        )
        axis.set_ylabel("Erreur de référence (L2, plus faible = meilleur)")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    path = figure_dir / "reference_error_by_noise_and_attack.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    separated = {"ipm", "bitflip_x10", "model_replacement"}
    figure_rows = [
        row
        for row in rows
        if row["candidate"] == PRIMARY
        and row["noise_regime"] == "heteroscedastic"
        and row["threat"] in separated
    ]
    seed_values: dict[int, list[float]] = defaultdict(list)
    for row in figure_rows:
        seed_values[int(row["seed"])].append(float(row["difference_vs_fcc"]))
    seeds = sorted(seed_values)
    differences = [_finite_mean(seed_values[seed]) for seed in seeds]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    colors = ["#188977" if value <= 0 else "#d95f02" for value in differences]
    axis.bar([str(seed) for seed in seeds], differences, color=colors)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Seed (unité statistique)")
    axis.set_ylabel("Erreur G0g − erreur FCC")
    axis.set_title(
        "Contraste apparié sous bruit hétéroscédastique et attaques séparées"
    )
    axis.grid(axis="y", alpha=0.25)
    path = figure_dir / "paired_g0g_minus_fcc_by_seed.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    clean_primary = [
        row for row in rows if row["candidate"] == PRIMARY and row["threat"] == "none"
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    labels = ["homogène", "hétéroscédastique"]
    tail = [
        _finite_mean(
            row["regular_false_tail_rate"]
            for row in clean_primary
            if row["noise_regime"] == regime
        )
        for regime in ("homogeneous", "heteroscedastic")
    ]
    gate = [
        _finite_mean(
            row["gate_mean"] for row in clean_primary if row["noise_regime"] == regime
        )
        for regime in ("homogeneous", "heteroscedastic")
    ]
    x = [0, 1]
    axis.bar([value - 0.18 for value in x], tail, 0.36, label="fraction z>1")
    axis.bar([value + 0.18 for value in x], gate, 0.36, label="gate moyenne")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Calibration statistique de la gate (sans attaque)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    path = figure_dir / "statistical_gate_calibration.png"
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
    calibration: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    decision: Mapping[str, Any],
    figures: Sequence[str],
) -> None:
    observed = decision["observed"]
    checks = decision["checks"]
    key_rows = [
        row
        for row in summaries
        if row["candidate"] in {"fcc", "fixed_cap_no_gate", "g0g_sigma_blind", PRIMARY}
        and row["threat"] in {"none", "alie", "ipm", "bitflip_x10", "model_replacement"}
    ]
    lines = [
        "# G0g-K1 — rapport de décision pour lundi",
        "",
        "## Verdict",
        "",
        (
            "**PROMOTION vers un holdout indépendant.**"
            if decision["all_gates_pass"]
            else "**ARRÊT après le développement : cette instanciation n'est pas promue.**"
        ),
        "",
        "G0g-K1 utilise la covariance publique uniquement pour définir ce qui est "
        "statistiquement plausible. Le déplacement autorisé par client reste borné "
        "par le même cap G, quel que soit son niveau de bruit.",
        "",
        "Ce screen est synthétique et reference-only : il ne contient ni FAR, ni "
        "accuracy, ni Fashion-MNIST. Une réussite autorise seulement l'étape suivante; "
        "elle ne constitue pas encore une validation end-to-end.",
        "",
        "## Résultats qui répondent à l'hypothèse",
        "",
        "| Quantité préenregistrée | Valeur observée | Seuil | Verdict |",
        "|---|---:|---:|:---:|",
        f"| Ratio erreur propre homogène / FCC | {_fmt(observed['homogeneous_clean_error_ratio_to_fcc'])} | ≤ 1.02 | {'✓' if checks['homogeneous_clean_noninferiority'] else '✗'} |",
        f"| Ratio erreur propre hétéroscédastique / FCC | {_fmt(observed['heteroscedastic_clean_error_ratio_to_fcc'])} | ≤ 1.02 | {'✓' if checks['heteroscedastic_clean_noninferiority'] else '✗'} |",
        f"| Gain sous attaques séparées vs FCC | {_fmt(100.0 * observed['heteroscedastic_attacked_relative_gain_vs_fcc'], 2)} % | ≥ 5 % | {'✓' if checks['attacked_gain_vs_fcc'] else '✗'} |",
        f"| Borne haute IC95 de G0g−FCC | {_fmt(observed['heteroscedastic_attacked_difference_vs_fcc_seed_ci95']['high'])} | ≤ 0 | {'✓' if checks['attacked_ci_vs_fcc'] else '✗'} |",
        f"| Gain vs gate aveugle au bruit | {_fmt(100.0 * observed['heteroscedastic_attacked_relative_gain_vs_sigma_blind'], 2)} % | ≥ 3 % | {'✓' if checks['attacked_gain_vs_sigma_blind'] else '✗'} |",
        f"| Faux-tail honnêtes réguliers | {_fmt(100.0 * observed['regular_false_tail_rate'], 2)} % | ≤ 15 % | {'✓' if checks['false_tail_rate'] else '✗'} |",
        f"| Écart maximal faux-tail entre tiers | {_fmt(100.0 * observed['regular_false_tail_tier_gap_worst_cell'], 2)} pp | ≤ 10 pp | {'✓' if checks['false_tail_tier_gap'] else '✗'} |",
        f"| Part maximale des contributions byzantines | {_fmt(100.0 * observed['heteroscedastic_attacked_byzantine_contribution_share_worst_group'], 2)} % | ≤ 25 % | {'✓' if checks['byzantine_contribution_share'] else '✗'} |",
        "",
        "## Certificats déterministes",
        "",
        f"- violations du cap de contribution : **{observed['contribution_cap_violations']}**;",
        f"- audits replace-one : **{observed['replace_one_trials']}**, violations : **{observed['replace_one_violations']}**;",
        f"- maximum observé / borne 2G/n : **{_fmt(observed['replace_one_max_ratio_observed_to_bound'])}**;",
        "- la covariance n'intervient ni dans G ni dans la borne 2G/n;",
        "- coût local-DP supplémentaire de la référence : **0** (post-traitement).",
        "",
        "## Moyennes par régime et attaque",
        "",
        "| Référence | Bruit | Attaque | Erreur L2 moyenne ± écart-type | Ratio/FCC | Gate moyenne |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in key_rows:
        lines.append(
            "| {candidate} | {noise_regime} | {threat} | {mean} ± {std} | {ratio} | {gate} |".format(
                candidate=row["candidate"],
                noise_regime=row["noise_regime"],
                threat=row["threat"],
                mean=_fmt(row["reference_error_mean"]),
                std=_fmt(row["reference_error_std"]),
                ratio=_fmt(row["ratio_to_fcc_mean"]),
                gate=_fmt(row["gate_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Lecture scientifique",
            "",
            "**Observation.** Les nombres ci-dessus décrivent exactement la matrice "
            "préenregistrée et les mêmes réalisations pour chaque référence.",
            "",
            "**Inférence permise.** Si tous les gates passent, séparer tolérance "
            "statistique et influence fixe est supérieur, dans ce banc synthétique, "
            "à FCC et à une gate ignorant les niveaux de bruit. Si un gate échoue, "
            "l'instanciation K1 ne justifie pas une campagne vision coûteuse.",
            "",
            "**Non identifiable ici.** Accuracy, fairness, convergence et bénéfice FAR "
            "end-to-end ne peuvent pas être déduits de ce screen reference-only.",
            "",
            "## Calibration et traçabilité",
            "",
            f"- seuils standardisés par bloc : `{calibration['standardized_thresholds_by_block']}`;",
            f"- lignes de développement : `{observed['development_rows']}/{observed['expected_development_rows']}`;",
            f"- configuration figée : `{config_path.relative_to(ROOT)}`;",
            f"- résultats : `{output_dir.relative_to(ROOT)}`.",
        ]
    )
    if figures:
        lines.extend(["", "## Figures", ""])
        for figure in figures:
            lines.append(f"![{Path(figure).stem}]({ROOT / figure})")
            lines.append("")
    failed = [name for name, passed in checks.items() if not passed]
    lines.extend(
        [
            "## Décision suivante",
            "",
            (
                "Tous les gates passent : ouvrir un holdout à sept seeds, puis seulement "
                "en cas de confirmation intégrer G0g-K1 à LDP-Gradient-FAR sur Fashion-MNIST."
                if not failed
                else "Gates échoués : **" + ", ".join(failed) + "**. Ne pas ouvrir de "
                "holdout et ne pas lancer Fashion-MNIST pour K1; analyser ces échecs puis "
                "modifier le mécanisme sous un nouvel identifiant de protocole."
            ),
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
        raise RuntimeError(
            f"Refusing to overwrite an existing G0g result directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "algorithm": _sha256(ROOT / "algorithms/gaussian_aware_reference.py"),
            "oracle_helpers": _sha256(
                ROOT / "scripts/run_gaussian_aware_reference_oracle.py"
            ),
        },
        "device": str(oracle._RUNTIME_DEVICE),
        "dtype": str(oracle._RUNTIME_DTYPE),
        "torch_version": str(torch.__version__),
        "holdout_opened": False,
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    calibration, calibration_rows = _calibrate(config)
    _atomic_json(output_dir / "calibration.json", calibration)
    _write_csv(output_dir / "calibration_rows.csv", calibration_rows)

    rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in _noise_cells(config):
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
    _attach_paired_baselines(rows)
    _write_csv(output_dir / "development_rows.csv", rows)
    replace_rows = _replace_one_audit(config, calibration)
    _write_csv(output_dir / "replace_one_rows.csv", replace_rows)
    summaries = _summaries(rows)
    _write_csv(output_dir / "summary.csv", summaries)
    decision = _evaluate_gates(rows, replace_rows, config)
    decision["holdout_opened"] = False
    decision["privacy_claim"] = (
        "No extra local-DP cost: deterministic post-processing of locally private uploads"
    )
    _atomic_json(output_dir / "decision.json", decision)
    figures = _make_figures(rows, output_dir)
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        calibration=calibration,
        summaries=summaries,
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
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k1.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k1_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_G0g_K1_Monday_Report.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
