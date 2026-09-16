#!/usr/bin/env python3
"""Stage 14A: frozen synthetic audit of effective-null FAR scores.

The audit never trains a model and never uses final accuracy.  It compares a
moment-standardised score with an empirical-null-quantile control.  Both use
the same public coordinate subspace, the same exact one-step centered-clipping
leave-one-out reference and the same full-chain public null simulations.
"""

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
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.noise_aware_scores import (  # noqa: E402
    calibrated_null_energy_scores,
    shrink_null_energy_moments,
)
from robustness.aggregators import (  # noqa: E402
    centered_clipping_leave_one_out,
    clip_l2,
)


@dataclass(frozen=True)
class Candidate:
    family: str
    score_dimension: int
    individual_calibration_weight: float

    @property
    def name(self) -> str:
        weight = f"{self.individual_calibration_weight:.2f}".replace(".", "p")
        return f"{self.family}_d{self.score_dimension}_individual_w{weight}"


def _finite_mean(values: list[float]) -> float:
    selected = [value for value in values if math.isfinite(value)]
    return float(statistics.fmean(selected)) if selected else float("nan")


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    if x.numel() != y.numel() or x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-15:
        return float("nan")
    return float((x @ y / denominator).item())


def _ks_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().reshape(-1).sort().values
    right = right.double().reshape(-1).sort().values
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    support = torch.unique(torch.cat((left, right))).sort().values
    left_cdf = torch.searchsorted(left, support, right=True).double() / left.numel()
    right_cdf = torch.searchsorted(right, support, right=True).double() / right.numel()
    return float((left_cdf - right_cdf).abs().max().item())


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-15)


def _orthogonal(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return _unit(vector - (vector @ direction) * direction)


def _clean_honest_vectors(
    seed: int,
    *,
    n_honest: int,
    dimension: int,
    num_outliers: int,
    outlier_geometry: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    descent = _unit(torch.randn(dimension, generator=generator, dtype=torch.float64))
    centre = 0.08 * descent
    deviations = torch.randn(
        n_honest, dimension, generator=generator, dtype=torch.float64
    )
    deviations = torch.stack([_orthogonal(row, descent) for row in deviations])
    clean = centre + 0.018 * deviations

    outlier_mask = torch.zeros(n_honest, dtype=torch.bool)
    outlier_ids = (
        torch.linspace(1, n_honest - 2, steps=num_outliers).round().long().unique()
    )
    if len(outlier_ids) != num_outliers:
        raise RuntimeError("The fixed honest-outlier construction lost an ID")
    directions = torch.randn(
        num_outliers, dimension, generator=generator, dtype=torch.float64
    )
    directions = torch.stack([_orthogonal(row, descent) for row in directions])
    if outlier_geometry == "aligned":
        shifts = 0.09 * (0.60 * descent[None, :] + 0.80 * directions)
    elif outlier_geometry == "orthogonal":
        shifts = 0.09 * directions
    else:
        raise ValueError(f"Unknown outlier geometry {outlier_geometry!r}")
    clean[outlier_ids] += shifts
    outlier_mask[outlier_ids] = True
    return clean, outlier_mask


def _noise_scales(levels: list[float], permutation: str, n: int) -> torch.Tensor:
    repeated = (levels * ((n + len(levels) - 1) // len(levels)))[:n]
    base = torch.tensor(repeated, dtype=torch.float64)
    if permutation == "identity":
        return base
    if permutation == "rotate7":
        return torch.roll(base, shifts=7)
    if permutation == "reverse":
        return torch.flip(base, dims=(0,))
    raise ValueError(f"Unknown noise permutation {permutation!r}")


def _attack_vectors(
    honest_noisy: torch.Tensor,
    *,
    threat: str,
    num_byzantine: int,
) -> torch.Tensor:
    centre = honest_noisy.mean(dim=0)
    if threat == "ipm":
        vector = -3.0 * centre
    elif threat == "bitflip_x10":
        vector = -10.0 * centre
    elif threat == "alie":
        vector = centre - 1.5 * honest_noisy.std(dim=0, unbiased=False)
    else:
        raise ValueError(f"Unknown threat {threat!r}")
    return vector[None, :].repeat(num_byzantine, 1)


def _observed_full_cohort(
    clean_honest: torch.Tensor,
    scales: torch.Tensor,
    *,
    threat: str,
    num_byzantine: int,
    noise_std: float,
    server_clip_norm: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        clean_honest.shape, generator=generator, dtype=clean_honest.dtype
    )
    honest = clean_honest + noise_std * scales[: len(clean_honest), None] * noise
    if threat == "none":
        preclip = honest
        byzantine = torch.zeros(len(honest), dtype=torch.bool)
    else:
        attacks = _attack_vectors(
            honest, threat=threat, num_byzantine=num_byzantine
        )
        preclip = torch.cat((honest, attacks), dim=0)
        byzantine = torch.zeros(len(preclip), dtype=torch.bool)
        byzantine[len(honest) :] = True
    return clip_l2(preclip, server_clip_norm), byzantine


def _projection_order(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dimension, generator=generator)


def _fcc_loo_energies(
    vectors: torch.Tensor,
    *,
    coordinate_order: torch.Tensor,
    score_dimension: int,
    full_dimension: int,
    full_radius: float,
) -> torch.Tensor:
    indices = coordinate_order[:score_dimension].sort().values
    projected = vectors.index_select(1, indices)
    radius = full_radius * math.sqrt(score_dimension / full_dimension)
    references = centered_clipping_leave_one_out(
        projected,
        anchor=torch.zeros(score_dimension, dtype=projected.dtype),
        tau=radius,
    )
    return torch.linalg.vector_norm(projected - references, dim=1).square()


def _null_energy_samples(
    *,
    scales: torch.Tensor,
    coordinate_orders: dict[int, torch.Tensor],
    score_dimensions: list[int],
    full_dimension: int,
    full_radius: float,
    noise_std: float,
    server_clip_norm: float,
    draws: int,
    seed: int,
) -> dict[tuple[int, int], torch.Tensor]:
    rows: dict[tuple[int, int], list[torch.Tensor]] = defaultdict(list)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for _ in range(draws):
        noise = torch.randn(
            len(scales), full_dimension, generator=generator, dtype=torch.float64
        )
        vectors = clip_l2(noise_std * scales[:, None] * noise, server_clip_norm)
        for projection_seed, order in coordinate_orders.items():
            for dimension in score_dimensions:
                rows[(projection_seed, dimension)].append(
                    _fcc_loo_energies(
                        vectors,
                        coordinate_order=order,
                        score_dimension=dimension,
                        full_dimension=full_dimension,
                        full_radius=full_radius,
                    )
                )
    return {key: torch.stack(values) for key, values in rows.items()}


def _score_matrix(
    observed: torch.Tensor,
    calibration: torch.Tensor,
    *,
    family: str,
    individual_weight: float,
    z_clip: float,
    tail_probability: float,
    variance_ridge: float,
) -> torch.Tensor:
    if observed.ndim != 2 or calibration.ndim != 2:
        raise ValueError("observed and calibration energies must be matrices")
    if observed.shape[1] != calibration.shape[1]:
        raise ValueError("observed and calibration client counts must match")
    if family == "effective_moment":
        mean, variance = shrink_null_energy_moments(
            calibration,
            individual_moment_weight=individual_weight,
            variance_ridge=variance_ridge,
        )
        z = (observed - mean[None, :]) / torch.sqrt(variance)[None, :]
        return (torch.relu(z) / z_clip).clamp(max=1.0)
    if family == "empirical_quantile":
        individual = torch.empty_like(observed)
        for client in range(observed.shape[1]):
            ordered = calibration[:, client].sort().values
            individual[:, client] = torch.searchsorted(
                ordered, observed[:, client].contiguous(), right=True
            ).double() / len(ordered)
        pooled = calibration.reshape(-1).sort().values
        pooled_cdf = (
            torch.searchsorted(pooled, observed.reshape(-1), right=True).double()
            / len(pooled)
        ).reshape_as(observed)
        cdf = (1.0 - individual_weight) * pooled_cdf + individual_weight * individual
        return ((cdf - tail_probability) / (1.0 - tail_probability)).clamp(
            min=0.0, max=1.0
        )
    raise ValueError(f"Unknown score family {family!r}")


def _null_metrics(scores: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    repeated_scales = scales[None, :].expand_as(scores)
    tier_values = {
        float(tier): scores[:, scales == tier].reshape(-1)
        for tier in torch.unique(scales)
    }
    tier_means = [float(values.mean()) for values in tier_values.values()]
    ks_values = [
        _ks_distance(tier_values[left], tier_values[right])
        for left in tier_values
        for right in tier_values
        if left < right
    ]
    return {
        "null_score_noise_correlation": _corr(scores, repeated_scales),
        "null_noise_tier_score_range": max(tier_means) - min(tier_means),
        "null_pairwise_ks_max": max(ks_values, default=float("nan")),
        "null_score_mean": float(scores.mean()),
        "null_score_zero_rate": float((scores <= 1e-12).double().mean()),
    }


def _signal_metrics(
    *,
    scores: torch.Tensor,
    clean_distances: torch.Tensor,
    honest_outliers: torch.Tensor,
    byzantine_mask: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    honest = ~byzantine_mask
    honest_scores = scores[honest]
    top_count = min(5, honest_scores.numel())
    predicted_honest = torch.topk(honest_scores, top_count).indices
    honest_recall = float(honest_outliers[predicted_honest].double().mean())

    full_outliers = torch.zeros_like(byzantine_mask)
    full_outliers[: len(honest_outliers)] = honest_outliers
    predicted_global = torch.topk(scores, min(5, scores.numel())).indices
    global_recall = float(full_outliers[predicted_global].double().mean())
    weights = torch.softmax(alpha * scores, dim=0)
    return {
        "clean_geometry_correlation": _corr(honest_scores, clean_distances),
        "honest_outlier_recall_at_5": honest_recall,
        "global_honest_outlier_recall_at_5": global_recall,
        "honest_outlier_weight_mass": float(weights[full_outliers].sum()),
        "byzantine_weight_mass": float(weights[byzantine_mask].sum()),
        "max_individual_weight": float(weights.max()),
        "weight_concentration": float(scores.numel() * weights.square().sum()),
        "score_span": float(scores.max() - scores.min()),
        "score_degenerate": float(float(scores.max() - scores.min()) <= 1e-12),
    }


def _candidate_scores(
    candidate: Candidate,
    observed_energy: torch.Tensor,
    calibration_energy: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    score_config = config["scores"]
    mode = "moment" if candidate.family == "effective_moment" else "quantile"
    scores, _ = calibrated_null_energy_scores(
        observed_energy,
        calibration_energy,
        mode=mode,
        z_clip=float(score_config["moment_z_clip"]),
        tail_probability=float(score_config["quantile_tail_probability"]),
        individual_calibration_weight=candidate.individual_calibration_weight,
        variance_ridge=float(score_config["variance_ridge"]),
    )
    return scores


def _aggregate(
    candidates: list[Candidate],
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    gates: dict[str, float],
) -> list[dict[str, Any]]:
    summaries = []
    for candidate in candidates:
        null = [row for row in null_rows if row["candidate"] == candidate.name]
        signal = [row for row in signal_rows if row["candidate"] == candidate.name]
        clean = [row for row in signal if row["threat"] == "none"]
        attacked = [row for row in signal if row["threat"] != "none"]
        byzantine_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in attacked:
            key = (
                row["threat"],
                row["signal_seed"],
                row["projection_seed"],
                row["noise_permutation"],
                row["outlier_geometry"],
            )
            byzantine_groups[key].append(row["byzantine_weight_mass"])
        worst_byzantine_group = max(
            (_finite_mean(values) for values in byzantine_groups.values()),
            default=float("nan"),
        )
        summary: dict[str, Any] = {
            "candidate": candidate.name,
            "family": candidate.family,
            "score_dimension": candidate.score_dimension,
            "individual_calibration_weight": candidate.individual_calibration_weight,
            "null_abs_score_noise_correlation_max": max(
                (
                    abs(row["null_score_noise_correlation"])
                    if math.isfinite(row["null_score_noise_correlation"])
                    else float("inf")
                    for row in null
                ),
                default=float("inf"),
            ),
            "null_noise_tier_score_range_max": max(
                (row["null_noise_tier_score_range"] for row in null),
                default=float("inf"),
            ),
            "null_pairwise_ks_max": max(
                (row["null_pairwise_ks_max"] for row in null),
                default=float("inf"),
            ),
            "clean_geometry_correlation_mean": _finite_mean(
                [row["clean_geometry_correlation"] for row in clean]
            ),
            "honest_outlier_recall_at_5_mean": _finite_mean(
                [row["honest_outlier_recall_at_5"] for row in clean]
            ),
            "honest_outlier_weight_mass_mean": _finite_mean(
                [row["honest_outlier_weight_mass"] for row in clean]
            ),
            "score_degenerate_rate": _finite_mean(
                [row["score_degenerate"] for row in clean]
            ),
            "max_individual_weight_observed": max(
                (row["max_individual_weight"] for row in signal), default=float("inf")
            ),
            "attacked_byzantine_weight_mass_max_group_mean": worst_byzantine_group,
            "attacked_global_honest_outlier_recall_at_5_mean": _finite_mean(
                [row["global_honest_outlier_recall_at_5"] for row in attacked]
            ),
        }
        statistical_checks = {
            "null_noise_correlation": summary[
                "null_abs_score_noise_correlation_max"
            ]
            <= gates["null_abs_score_noise_correlation_max"],
            "null_tier_means": summary["null_noise_tier_score_range_max"]
            <= gates["null_noise_tier_score_range_max"],
            "null_distributions": summary["null_pairwise_ks_max"]
            <= gates["null_pairwise_ks_max"],
            "clean_geometry": summary["clean_geometry_correlation_mean"]
            >= gates["clean_geometry_correlation_min"],
            "honest_outlier_recall": summary["honest_outlier_recall_at_5_mean"]
            >= gates["honest_outlier_recall_at_5_min"],
            "honest_outlier_mass": summary["honest_outlier_weight_mass_mean"]
            > gates["honest_outlier_weight_mass_min_exclusive"],
            "non_degenerate": summary["score_degenerate_rate"]
            <= gates["score_degenerate_rate_max"],
            "weight_cap": summary["max_individual_weight_observed"]
            <= gates["max_individual_weight"] + 1e-10,
        }
        robustness_checks = {
            "byzantine_mass": summary[
                "attacked_byzantine_weight_mass_max_group_mean"
            ]
            <= gates["attacked_byzantine_weight_mass_max"],
            "attacked_outlier_recall": summary[
                "attacked_global_honest_outlier_recall_at_5_mean"
            ]
            >= gates["attacked_global_honest_outlier_recall_at_5_min"],
        }
        summary["statistical_gate_checks"] = statistical_checks
        summary["robustness_gate_checks"] = robustness_checks
        summary["passes_statistical_gates"] = all(statistical_checks.values())
        summary["passes_full_gates"] = summary["passes_statistical_gates"] and all(
            robustness_checks.values()
        )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _write_report(
    path: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    statistical = [row for row in summaries if row["passes_statistical_gates"]]
    full = [row for row in summaries if row["passes_full_gates"]]
    lines = [
        "# Stage 14A — Audit moments effectifs, shrinkage et LOO",
        "",
        "## Verdict",
        "",
        f"- Configurations évaluées : **{len(summaries)}**.",
        f"- Survivants statistiques : **{len(statistical)}**.",
        f"- Survivants de tous les gates, robustesse comprise : **{len(full)}**.",
        "",
        (
            "Les candidats statistiques peuvent être transférés au screen "
            "end-to-end Stage 14B sans modifier leurs constantes."
            if statistical
            else "Aucun candidat ne justifie un transfert end-to-end. Le protocole "
            "impose alors l'arrêt de cette famille de corrections radiales."
        ),
        "",
        "## Construction auditée",
        "",
        "Pour chaque sous-espace public et chaque client, la chaîne simulée est :",
        "",
        "```text",
        "bruit DP public -> clipping serveur -> coordonnées publiques",
        "-> F_CC leave-one-out -> énergie résiduelle",
        "```",
        "",
        "Le paramètre `individual_calibration_weight` est non ambigu : 0 utilise "
        "des moments/CDF entièrement communs et 1 une calibration entièrement "
        "spécifique au client. Le score principal corrige moyenne et variance "
        "nulles ; le quantile empirique est son contrôle apparié.",
        "",
        "## Résultats",
        "",
        "| Famille | d_s | Poids individuel | Corr. bruit absolue max | Écart tiers max | KS max | Corr. propre | Recall@5 | Masse outliers | Masse byz. max | Gate stat. | Gate complet |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in sorted(
        summaries,
        key=lambda item: (
            item["family"],
            item["score_dimension"],
            item["individual_calibration_weight"],
        ),
    ):
        lines.append(
            "| {family} | {dimension} | {weight:.2f} | {corr} | {tier} | {ks} | "
            "{clean} | {recall} | {outliers} | {byz} | {stat} | {full} |".format(
                family=row["family"],
                dimension=row["score_dimension"],
                weight=row["individual_calibration_weight"],
                corr=_fmt(row["null_abs_score_noise_correlation_max"]),
                tier=_fmt(row["null_noise_tier_score_range_max"]),
                ks=_fmt(row["null_pairwise_ks_max"]),
                clean=_fmt(row["clean_geometry_correlation_mean"]),
                recall=_fmt(row["honest_outlier_recall_at_5_mean"]),
                outliers=_fmt(row["honest_outlier_weight_mass_mean"]),
                byz=_fmt(row["attacked_byzantine_weight_mass_max_group_mean"]),
                stat="oui" if row["passes_statistical_gates"] else "non",
                full="oui" if row["passes_full_gates"] else "non",
            )
        )
    lines.extend(
        [
            "",
            "## Règle de lecture",
            "",
            "La corrélation nulle ne suffit pas : l'écart des moyennes par niveau "
            "de bruit et la distance de Kolmogorov–Smirnov contrôlent aussi les "
            "dépendances non linéaires. Corrélation propre, Recall@5 et masse des "
            "outliers contrôlent la conservation de l'hétérogénéité honnête.",
            "",
            "La masse byzantine est comparée à 5/25 = 0,20. Le cap individuel "
            "reste 2/25 = 0,08. Les oracles propres ne sont utilisés que pour "
            "l'audit et ne font pas partie d'un mécanisme déployable.",
            "",
            "## Traçabilité",
            "",
            f"- Configuration figée : `{config_path}`",
            f"- Résultats détaillés nuls : `{output_dir / 'null_holdout_detail.csv'}`",
            f"- Résultats détaillés signal/attaques : `{output_dir / 'signal_detail.csv'}`",
            f"- Synthèse machine : `{output_dir / 'summary.json'}`",
            "",
            "Aucune accuracy n'a été calculée ou utilisée pour sélectionner ces "
            "configurations.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    calibration_draws_override: int | None = None,
    null_holdout_draws_override: int | None = None,
    signal_draws_override: int | None = None,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cohort = config["cohort"]
    geometry = config["geometry"]
    randomness = config["randomness"]
    score_config = config["scores"]
    gates = config["gates"]

    n = int(cohort["num_clients"])
    num_byzantine = int(cohort["num_byzantine"])
    full_dimension = int(cohort["ambient_dimension"])
    score_dimensions = [int(value) for value in geometry["score_subspace_dimensions"]]
    projection_seeds = [int(value) for value in geometry["score_subspace_seeds"]]
    weights = [float(value) for value in score_config["individual_calibration_weights"]]
    families = [str(value) for value in score_config["families"]]
    candidates = [
        Candidate(family, dimension, weight)
        for family in families
        for dimension in score_dimensions
        for weight in weights
    ]
    if max(score_dimensions) > full_dimension:
        raise ValueError("A score subspace cannot exceed the ambient dimension")
    if geometry["score_subspace_mode"] != "public_coordinates":
        raise ValueError("Stage 14A requires public coordinate subspaces")

    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    null_holdout_draws = int(
        null_holdout_draws_override or randomness["null_holdout_draws"]
    )
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or null_holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Need >=20 null draws and >=1 signal draw")

    levels = [float(value) for value in cohort["public_noise_scales"]]
    permutations = [str(value) for value in cohort["noise_permutations"]]
    coordinate_orders = {
        seed: _projection_order(full_dimension, seed) for seed in projection_seeds
    }
    calibration_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    holdout_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    for permutation_index, permutation in enumerate(permutations):
        scales = _noise_scales(levels, permutation, n)
        calibration = _null_energy_samples(
            scales=scales,
            coordinate_orders=coordinate_orders,
            score_dimensions=score_dimensions,
            full_dimension=full_dimension,
            full_radius=float(geometry["fcc_radius_at_full_dimension"]),
            noise_std=float(cohort["noise_std"]),
            server_clip_norm=float(geometry["server_clip_norm"]),
            draws=calibration_draws,
            seed=int(randomness["calibration_seed"]) + 10_000 * permutation_index,
        )
        holdout = _null_energy_samples(
            scales=scales,
            coordinate_orders=coordinate_orders,
            score_dimensions=score_dimensions,
            full_dimension=full_dimension,
            full_radius=float(geometry["fcc_radius_at_full_dimension"]),
            noise_std=float(cohort["noise_std"]),
            server_clip_norm=float(geometry["server_clip_norm"]),
            draws=null_holdout_draws,
            seed=int(randomness["null_holdout_seed"]) + 10_000 * permutation_index,
        )
        for key, values in calibration.items():
            calibration_cache[(permutation, *key)] = values
        for key, values in holdout.items():
            holdout_cache[(permutation, *key)] = values

    null_rows: list[dict[str, Any]] = []
    for permutation in permutations:
        scales = _noise_scales(levels, permutation, n)
        for projection_seed in projection_seeds:
            for dimension in score_dimensions:
                calibration = calibration_cache[(permutation, projection_seed, dimension)]
                observed = holdout_cache[(permutation, projection_seed, dimension)]
                for candidate in candidates:
                    if candidate.score_dimension != dimension:
                        continue
                    scores = _score_matrix(
                        observed,
                        calibration,
                        family=candidate.family,
                        individual_weight=candidate.individual_calibration_weight,
                        z_clip=float(score_config["moment_z_clip"]),
                        tail_probability=float(score_config["quantile_tail_probability"]),
                        variance_ridge=float(score_config["variance_ridge"]),
                    )
                    null_rows.append(
                        {
                            "candidate": candidate.name,
                            "family": candidate.family,
                            "score_dimension": dimension,
                            "individual_calibration_weight": candidate.individual_calibration_weight,
                            "projection_seed": projection_seed,
                            "noise_permutation": permutation,
                            **_null_metrics(scores, scales),
                        }
                    )

    kappa = float(config["weights"]["kappa_w"])
    alpha = math.log(kappa * (n - 1) / (n - kappa))
    signal_rows: list[dict[str, Any]] = []
    signal_seeds = [int(value) for value in randomness["signal_seeds"]]
    geometries = [str(value) for value in cohort["outlier_geometries"]]
    threats = [str(value) for value in cohort["threats"]]
    for signal_seed in signal_seeds:
        for geometry_index, outlier_geometry in enumerate(geometries):
            for permutation_index, permutation in enumerate(permutations):
                scales = _noise_scales(levels, permutation, n)
                for threat_index, threat in enumerate(threats):
                    n_honest = n if threat == "none" else n - num_byzantine
                    clean, honest_outliers = _clean_honest_vectors(
                        signal_seed,
                        n_honest=n_honest,
                        dimension=full_dimension,
                        num_outliers=int(cohort["honest_outliers"]),
                        outlier_geometry=outlier_geometry,
                    )
                    clean_clipped = clip_l2(
                        clean, float(geometry["server_clip_norm"])
                    )
                    for draw in range(signal_draws):
                        observed, byzantine_mask = _observed_full_cohort(
                            clean,
                            scales,
                            threat=threat,
                            num_byzantine=num_byzantine,
                            noise_std=float(cohort["noise_std"]),
                            server_clip_norm=float(geometry["server_clip_norm"]),
                            seed=(
                                signal_seed * 1_000_000
                                + geometry_index * 100_000
                                + permutation_index * 10_000
                                + threat_index * 1_000
                                + draw
                            ),
                        )
                        for projection_seed, order in coordinate_orders.items():
                            for dimension in score_dimensions:
                                observed_energy = _fcc_loo_energies(
                                    observed,
                                    coordinate_order=order,
                                    score_dimension=dimension,
                                    full_dimension=full_dimension,
                                    full_radius=float(
                                        geometry["fcc_radius_at_full_dimension"]
                                    ),
                                )
                                clean_energy = _fcc_loo_energies(
                                    clean_clipped,
                                    coordinate_order=order,
                                    score_dimension=dimension,
                                    full_dimension=full_dimension,
                                    full_radius=float(
                                        geometry["fcc_radius_at_full_dimension"]
                                    ),
                                )
                                clean_distances = clean_energy.sqrt()
                                calibration = calibration_cache[
                                    (permutation, projection_seed, dimension)
                                ]
                                for candidate in candidates:
                                    if candidate.score_dimension != dimension:
                                        continue
                                    scores = _candidate_scores(
                                        candidate,
                                        observed_energy,
                                        calibration,
                                        config,
                                    )
                                    signal_rows.append(
                                        {
                                            "candidate": candidate.name,
                                            "family": candidate.family,
                                            "score_dimension": dimension,
                                            "individual_calibration_weight": candidate.individual_calibration_weight,
                                            "signal_seed": signal_seed,
                                            "draw": draw,
                                            "projection_seed": projection_seed,
                                            "noise_permutation": permutation,
                                            "outlier_geometry": outlier_geometry,
                                            "threat": threat,
                                            **_signal_metrics(
                                                scores=scores,
                                                clean_distances=clean_distances,
                                                honest_outliers=honest_outliers,
                                                byzantine_mask=byzantine_mask,
                                                alpha=alpha,
                                            ),
                                        }
                                    )

    summaries = _aggregate(candidates, null_rows, signal_rows, gates)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "null_holdout_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    summary_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"statistical_gate_checks", "robustness_gate_checks"}
        }
        for row in summaries
    ]
    _write_csv(output_dir / "summary.csv", summary_rows)
    payload = {
        "status": "completed",
        "campaign_id": config["campaign_id"],
        "config": config,
        "realized": {
            "calibration_draws": calibration_draws,
            "null_holdout_draws": null_holdout_draws,
            "signal_draws": signal_draws,
            "alpha": alpha,
            "weight_cap": kappa / n,
            "num_candidates": len(candidates),
            "num_null_cells": len(null_rows),
            "num_signal_cells": len(signal_rows),
        },
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        config_path=config_path,
        config=config,
        summaries=summaries,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "candidates": len(candidates),
                "statistical_survivors": sum(
                    bool(row["passes_statistical_gates"]) for row in summaries
                ),
                "full_survivors": sum(
                    bool(row["passes_full_gates"]) for row in summaries
                ),
                "report": str(report_path),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dt_ldp_far/stage14a_effective_null_moments.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/dt_ldp_far/score_quality_stage14a_effective_null_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage14A_Effective_Null_Moments_Audit.md",
    )
    parser.add_argument("--calibration-draws", type=int)
    parser.add_argument("--null-holdout-draws", type=int)
    parser.add_argument("--signal-draws", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.config,
        args.output_dir,
        args.report,
        calibration_draws_override=args.calibration_draws,
        null_holdout_draws_override=args.null_holdout_draws,
        signal_draws_override=args.signal_draws,
    )
