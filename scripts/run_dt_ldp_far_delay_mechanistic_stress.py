#!/usr/bin/env python3
"""Strictly paired mechanistic stress test for delayed FAR weights.

This script deliberately does *not* train a neural network and makes no new
privacy claim. It isolates the server-side mechanism that motivates
DT-LDP-FAR. Within every Monte-Carlo draw, the current-round and one-round
delayed rules receive exactly the same clean client vectors, the same fresh
perturbations, the same server clipping, and the same public geometry. The
only changed object is the score timestamp used to construct the weights.

All oracle quantities in this file are simulation diagnostics. They must not
be added to a claimed local-DP transcript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch


def alpha_max(n: int, kappa_w: float) -> float:
    """Largest alpha guaranteeing max_i omega_i <= kappa_w / n."""

    if n < 2 or not 1.0 <= float(kappa_w) < n:
        raise ValueError("Need n >= 2 and 1 <= kappa_w < n")
    return math.log(float(kappa_w) * (n - 1) / (n - float(kappa_w)))


def clip_rows(vectors: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        raise ValueError("Clipping radius must be positive")
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors * (float(radius) / norms.clamp_min(1e-15)).clamp(max=1.0)


def centered_clipping_reference(
    vectors: torch.Tensor, *, anchor: torch.Tensor, rho: float
) -> torch.Tensor:
    """One anchored centered-clipping step, batched over leading dimensions."""

    residuals = vectors - anchor.unsqueeze(-2)
    return anchor + clip_rows(residuals, rho).mean(dim=-2)


def bounded_scores(
    vectors: torch.Tensor, *, anchor: torch.Tensor, rho: float, d_score: float
) -> torch.Tensor:
    if d_score <= 0.0:
        raise ValueError("D_score must be positive")
    reference = centered_clipping_reference(vectors, anchor=anchor, rho=rho)
    distances = torch.linalg.vector_norm(vectors - reference.unsqueeze(-2), dim=-1)
    return (distances / float(d_score)).clamp(0.0, 1.0)


def aggregate(weights: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return (weights.unsqueeze(-1) * vectors).sum(dim=-2)


def _rowwise_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation for each row; undefined rows are returned as NaN."""

    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Expected equally-shaped [draws, clients] tensors")
    x0 = x.double() - x.double().mean(dim=1, keepdim=True)
    y0 = y.double() - y.double().mean(dim=1, keepdim=True)
    denominator = torch.linalg.vector_norm(x0, dim=1) * torch.linalg.vector_norm(
        y0, dim=1
    )
    numerator = (x0 * y0).sum(dim=1)
    result = torch.full_like(numerator, float("nan"))
    valid = denominator > 1e-15
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _finite_mean(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.mean().item()) if finite.numel() else None


def _finite_median(values: torch.Tensor) -> float | None:
    finite = values[torch.isfinite(values)]
    return float(finite.median().item()) if finite.numel() else None


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.double().flatten(), q).item())


@dataclass(frozen=True)
class PairedInputs:
    clean: torch.Tensor
    current_noisy: torch.Tensor
    past_noisy: torch.Tensor
    effective_fresh_perturbation: torch.Tensor
    randomness_pair_id: str


def make_paired_inputs(
    *,
    n: int,
    dimension: int,
    draws: int,
    seed: int,
    server_clip: float,
    noise_l2_scale: float,
    drift_l2_scale: float,
    heterogeneity_l2_scale: float,
    heteroscedastic_noise: bool,
) -> PairedInputs:
    """Generate one immutable source of randomness shared by every alpha arm."""

    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    consensus = torch.randn(dimension, generator=generator, dtype=dtype)
    consensus = consensus / consensus.norm().clamp_min(1e-15) * (0.20 * server_clip)

    offsets = torch.randn(n, dimension, generator=generator, dtype=dtype)
    offsets = offsets / math.sqrt(dimension) * float(heterogeneity_l2_scale)
    clean = clip_rows(consensus + offsets, server_clip)

    fresh_standard = torch.randn(
        draws, n, dimension, generator=generator, dtype=dtype
    ) / math.sqrt(dimension)
    past_standard = torch.randn(
        draws, n, dimension, generator=generator, dtype=dtype
    ) / math.sqrt(dimension)
    if heteroscedastic_noise:
        # A public, fixed scale pattern creates an explicit heteroscedastic
        # stress. The same pattern is used in every draw and in both arms.
        scales = torch.linspace(0.50, 1.50, n, dtype=dtype)
    else:
        scales = torch.ones(n, dtype=dtype)
    fresh_noise = fresh_standard * scales[None, :, None] * float(noise_l2_scale)
    past_noise = past_standard * scales[None, :, None] * float(noise_l2_scale)

    drift = torch.randn(
        draws, n, dimension, generator=generator, dtype=dtype
    ) / math.sqrt(dimension) * float(drift_l2_scale)
    clean_batch = clean.unsqueeze(0).expand(draws, -1, -1)
    current_noisy = clip_rows(clean_batch + fresh_noise, server_clip)
    past_noisy = clip_rows(clean_batch + drift + past_noise, server_clip)
    effective = current_noisy - clean_batch

    pair_description = (
        f"n={n};d={dimension};draws={draws};seed={seed};U={server_clip};"
        f"noise={noise_l2_scale};drift={drift_l2_scale};"
        f"heterogeneity={heterogeneity_l2_scale};heteroscedastic={heteroscedastic_noise}"
    )
    return PairedInputs(
        clean=clean,
        current_noisy=current_noisy,
        past_noisy=past_noisy,
        effective_fresh_perturbation=effective,
        randomness_pair_id=hashlib.sha256(pair_description.encode("utf-8")).hexdigest()[:16],
    )


def evaluate_cell(
    *,
    paired: PairedInputs,
    n: int,
    dimension: int,
    draws: int,
    seed: int,
    server_clip: float,
    d_score: float,
    rho: float,
    noise_l2_scale: float,
    drift_l2_scale: float,
    heterogeneity_l2_scale: float,
    heteroscedastic_noise: bool,
    alpha_profile: str,
    alpha: float,
    certificate_kappa_w: float | None,
) -> dict[str, float | int | str | bool | None]:
    anchor = torch.zeros(draws, dimension, dtype=paired.current_noisy.dtype)
    current_scores = bounded_scores(
        paired.current_noisy, anchor=anchor, rho=rho, d_score=d_score
    )
    past_scores = bounded_scores(
        paired.past_noisy, anchor=anchor, rho=rho, d_score=d_score
    )
    clean_anchor = torch.zeros(dimension, dtype=paired.clean.dtype)
    clean_scores = bounded_scores(
        paired.clean, anchor=clean_anchor, rho=rho, d_score=d_score
    )

    current_weights = torch.softmax(float(alpha) * current_scores, dim=1)
    delayed_weights = torch.softmax(float(alpha) * past_scores, dim=1)
    uniform_weights = torch.full_like(current_weights, 1.0 / n)
    clean_weights = torch.softmax(float(alpha) * clean_scores, dim=0)

    clean_batch = paired.clean.unsqueeze(0).expand(draws, -1, -1)
    current_noisy_aggregate = aggregate(current_weights, paired.current_noisy)
    delayed_noisy_aggregate = aggregate(delayed_weights, paired.current_noisy)
    uniform_noisy_aggregate = aggregate(uniform_weights, paired.current_noisy)

    # Holding each arm's weights fixed isolates the perturbation multiplied by
    # those weights. Comparing with the clean FAR target adds selection and
    # staleness effects and is therefore reported separately.
    current_clean_fixed_weights = aggregate(current_weights, clean_batch)
    delayed_clean_fixed_weights = aggregate(delayed_weights, clean_batch)
    uniform_clean = paired.clean.mean(dim=0).unsqueeze(0)
    clean_far_target = aggregate(clean_weights, paired.clean).unsqueeze(0)

    current_noise_error = (
        current_noisy_aggregate - current_clean_fixed_weights
    ).square().sum(dim=1)
    delayed_noise_error = (
        delayed_noisy_aggregate - delayed_clean_fixed_weights
    ).square().sum(dim=1)
    uniform_noise_error = (
        uniform_noisy_aggregate - uniform_clean
    ).square().sum(dim=1)
    current_net_error = (
        current_noisy_aggregate - clean_far_target
    ).square().sum(dim=1)
    delayed_net_error = (
        delayed_noisy_aggregate - clean_far_target
    ).square().sum(dim=1)
    staleness_error = (
        delayed_clean_fixed_weights - clean_far_target
    ).square().sum(dim=1)

    perturbation_norm = torch.linalg.vector_norm(
        paired.effective_fresh_perturbation, dim=2
    )
    current_corr = _rowwise_pearson(current_weights, perturbation_norm)
    delayed_corr = _rowwise_pearson(delayed_weights, perturbation_norm)
    current_spans = current_scores.max(dim=1).values - current_scores.min(dim=1).values
    current_logit_spans = float(alpha) * current_spans
    current_saturation = (current_scores >= 1.0 - 1e-12).double().mean(dim=1)
    current_max_weight = current_weights.max(dim=1).values
    delayed_max_weight = delayed_weights.max(dim=1).values
    current_concentration = n * current_weights.square().sum(dim=1)
    delayed_concentration = n * delayed_weights.square().sum(dim=1)

    current_noise_mse = float(current_noise_error.mean().item())
    delayed_noise_mse = float(delayed_noise_error.mean().item())
    current_net_mse = float(current_net_error.mean().item())
    delayed_net_mse = float(delayed_net_error.mean().item())
    reduction = (
        (current_noise_mse - delayed_noise_mse) / current_noise_mse
        if current_noise_mse > 0.0
        else 0.0
    )
    certified = certificate_kappa_w is not None and alpha <= alpha_max(
        n, certificate_kappa_w
    ) + 1e-12
    analytic_weight_cap = float(certificate_kappa_w) / n if certified else None
    return {
        "n": n,
        "dimension": dimension,
        "draws": draws,
        "seed": seed,
        "randomness_pair_id": paired.randomness_pair_id,
        "server_clip_U": server_clip,
        "D_score": d_score,
        "reference_radius_rho": rho,
        "noise_l2_scale": noise_l2_scale,
        "drift_l2_scale": drift_l2_scale,
        "heterogeneity_l2_scale": heterogeneity_l2_scale,
        "heteroscedastic_noise": heteroscedastic_noise,
        "alpha_profile": alpha_profile,
        "alpha": alpha,
        "certificate_kappa_w": certificate_kappa_w,
        "influence_certificate_claimed": certified,
        "analytic_weight_cap": analytic_weight_cap,
        "score_span_median": _quantile(current_spans, 0.50),
        "score_span_p90": _quantile(current_spans, 0.90),
        "logit_span_median": _quantile(current_logit_spans, 0.50),
        "score_saturation_rate_median": _quantile(current_saturation, 0.50),
        "score_saturation_rate_p90": _quantile(current_saturation, 0.90),
        "current_max_weight_median": _quantile(current_max_weight, 0.50),
        "delayed_max_weight_median": _quantile(delayed_max_weight, 0.50),
        "current_concentration_median": _quantile(current_concentration, 0.50),
        "delayed_concentration_median": _quantile(delayed_concentration, 0.50),
        "current_weight_fresh_perturbation_corr_mean": _finite_mean(current_corr),
        "current_weight_fresh_perturbation_corr_median": _finite_median(current_corr),
        "delayed_weight_fresh_perturbation_corr_mean": _finite_mean(delayed_corr),
        "delayed_weight_fresh_perturbation_corr_median": _finite_median(delayed_corr),
        "uniform_noise_mse": float(uniform_noise_error.mean().item()),
        "current_fixed_weight_noise_mse": current_noise_mse,
        "delayed_fixed_weight_noise_mse": delayed_noise_mse,
        "relative_delay_noise_mse_reduction": reduction,
        "current_net_mse": current_net_mse,
        "delayed_net_mse": delayed_net_mse,
        "staleness_mse": float(staleness_error.mean().item()),
        "delayed_net_win_rate": float(
            (delayed_net_error < current_net_error).double().mean().item()
        ),
    }


def _promotion_decision(row: dict[str, object]) -> tuple[bool, list[str]]:
    checks = {
        "score_span": float(row["score_span_median"]) >= 0.30,
        "logit_span": float(row["logit_span_median"]) >= 1.00,
        "limited_saturation": float(row["score_saturation_rate_p90"]) <= 0.25,
        "weight_concentration": float(row["current_concentration_median"]) >= 1.10,
        "maximum_weight": float(row["current_max_weight_median"])
        >= 1.50 / int(row["n"]),
        "current_self_selection": float(
            row["current_weight_fresh_perturbation_corr_median"] or 0.0
        )
        >= 0.20,
        "delayed_decoupling": abs(
            float(row["delayed_weight_fresh_perturbation_corr_median"] or 0.0)
        )
        <= 0.10,
        "noise_mse_reduction": float(row["relative_delay_noise_mse_reduction"])
        >= 0.05,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def _write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    promoted: list[dict[str, object]],
    robust_promotions: list[dict[str, object]],
    *,
    draws: int,
) -> None:
    top = sorted(
        [row for row in rows if float(row["alpha"]) > 0.0],
        key=lambda row: (
            bool(row["promoted_for_end_to_end"]),
            float(row["relative_delay_noise_mse_reduction"]),
            float(row["logit_span_median"]),
        ),
        reverse=True,
    )[:12]
    lines = [
        "# DT-LDP-FAR — stress mécanistique n=25 strictement apparié",
        "",
        "## Objet et portée",
        "",
        "Ce test isole le mécanisme serveur. Il ne constitue ni un entraînement de modèle "
        "ni une nouvelle dépense de confidentialité. Pour chaque cellule et chacun des "
        f"{draws} tirages, les variantes courante et retardée reçoivent exactement les "
        "mêmes vecteurs propres et la même perturbation fraîche. Seule la date des scores "
        "utilisés dans les poids change.",
        "",
        "Le poids courant dépend du score calculé sur la perturbation fraîche du même tour. "
        "Le poids retardé est calculé avec un tirage passé indépendant. La comparaison "
        "appariée retire donc la variance due à des bruits différents entre méthodes.",
        "",
        "## Gate de promotion",
        "",
        "Une cellule n'est promue vers un entraînement long que si toutes les conditions "
        "suivantes sont satisfaites : score-span médian ≥ 0,30 ; logit-span médian ≥ 1 ; "
        "saturation p90 ≤ 25 % ; concentration médiane n·Σωᵢ² ≥ 1,10 ; poids maximal "
        "médian ≥ 1,5/n ; corrélation courante poids–perturbation ≥ 0,20 ; corrélation "
        "retardée en valeur absolue ≤ 0,10 ; réduction de MSE du bruit ≥ 5 %.",
        "",
        f"Cellules seed-spécifiques promues : **{len(promoted)}** sur {len(rows)}. "
        f"Configurations passant sur les trois seeds : **{len(robust_promotions)}**.",
        "",
        "## Cellules les plus informatives",
        "",
        "| D_score | d | bruit hétéroscédastique | alpha | certifiée | S médian | R médian | ω_max | n·Σω² | corr. courant | corr. retardé | réduction MSE | promue |",
        "|---:|---:|:---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in top:
        lines.append(
            "| {D_score:.3f} | {dimension} | {hetero} | {alpha:.3f} | {cert} | "
            "{score:.3f} | {logit:.3f} | {wmax:.4f} | {conc:.3f} | {ccorr:.3f} | "
            "{dcorr:.3f} | {reduction:.1%} | {promoted} |".format(
                D_score=float(row["D_score"]),
                dimension=int(row["dimension"]),
                hetero="oui" if row["heteroscedastic_noise"] else "non",
                alpha=float(row["alpha"]),
                cert="oui" if row["influence_certificate_claimed"] else "non",
                score=float(row["score_span_median"]),
                logit=float(row["logit_span_median"]),
                wmax=float(row["current_max_weight_median"]),
                conc=float(row["current_concentration_median"]),
                ccorr=float(row["current_weight_fresh_perturbation_corr_median"] or 0.0),
                dcorr=float(row["delayed_weight_fresh_perturbation_corr_median"] or 0.0),
                reduction=float(row["relative_delay_noise_mse_reduction"]),
                promoted="oui" if row["promoted_for_end_to_end"] else "non",
            )
        )
    lines.extend(
        [
            "",
            "## Configurations robustes aux trois seeds",
            "",
            "| D_score | d | alpha | certifiée | S moyen | R moyen | n·Σω² moyen | corr. courant | corr. retardé | réduction MSE |",
            "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in robust_promotions:
        lines.append(
            "| {D_score:.3f} | {dimension} | {alpha:.3f} | {cert} | {score:.3f} | "
            "{logit:.3f} | {conc:.3f} | {ccorr:.3f} | {dcorr:.3f} | {reduction:.1%} |".format(
                D_score=float(row["D_score"]),
                dimension=int(row["dimension"]),
                alpha=float(row["alpha"]),
                cert="oui" if row["influence_certificate_claimed"] else "non",
                score=float(row["score_span_median_mean_over_seeds"]),
                logit=float(row["logit_span_median_mean_over_seeds"]),
                conc=float(row["current_concentration_median_mean_over_seeds"]),
                ccorr=float(
                    row[
                        "current_weight_fresh_perturbation_corr_median_mean_over_seeds"
                    ]
                ),
                dcorr=float(
                    row[
                        "delayed_weight_fresh_perturbation_corr_median_mean_over_seeds"
                    ]
                ),
                reduction=float(
                    row["relative_delay_noise_mse_reduction_mean_over_seeds"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Lecture de S et R",
            "",
            "Par définition, sᵢ = min(‖Xᵢ−F‖/D_score, 1), donc 0 ≤ sᵢ ≤ 1 et "
            "0 ≤ S = maxᵢ sᵢ − minᵢ sᵢ ≤ 1. Le contraste qui entre dans la softmax "
            "est toutefois R = alpha·S. Ainsi R ≥ 1 si et seulement si alpha ≥ 1/S "
            "(pour S > 0). Borner les scores ne borne donc pas R par 1 lorsque alpha > 1.",
            "",
            "Le fichier CSV conserve chaque cellule, son identifiant de randomness pairing "
            "et la liste exacte des critères échoués.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _robust_promotions(
    promoted_rows: list[dict[str, object]], expected_seeds: set[int]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in promoted_rows:
        key = (
            row["dimension"],
            row["heteroscedastic_noise"],
            row["D_score"],
            row["noise_l2_scale"],
            row["alpha_profile"],
            row["alpha"],
            row["certificate_kappa_w"],
        )
        groups.setdefault(key, []).append(row)
    robust: list[dict[str, object]] = []
    for key, values in groups.items():
        observed_seeds = {int(row["seed"]) for row in values}
        if observed_seeds != expected_seeds:
            continue
        numeric_fields = (
            "score_span_median",
            "logit_span_median",
            "current_max_weight_median",
            "current_concentration_median",
            "current_weight_fresh_perturbation_corr_median",
            "delayed_weight_fresh_perturbation_corr_median",
            "relative_delay_noise_mse_reduction",
            "delayed_net_win_rate",
        )
        summary = {
            "dimension": key[0],
            "heteroscedastic_noise": key[1],
            "D_score": key[2],
            "noise_l2_scale": key[3],
            "alpha_profile": key[4],
            "alpha": key[5],
            "certificate_kappa_w": key[6],
            "influence_certificate_claimed": all(
                bool(row["influence_certificate_claimed"]) for row in values
            ),
            "seeds": sorted(observed_seeds),
        }
        summary.update(
            {
                f"{field}_mean_over_seeds": mean(float(row[field]) for row in values)
                for field in numeric_fields
            }
        )
        robust.append(summary)
    robust.sort(
        key=lambda row: (
            bool(row["influence_certificate_claimed"]),
            float(row["relative_delay_noise_mse_reduction_mean_over_seeds"]),
        ),
        reverse=True,
    )
    return robust


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dt_ldp_far/mechanistic_n25_paired_v2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/analysis/DT_LDP_FAR_N25_Stress_Mecanistique_Apparie.md"),
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seeds", default="28,36,54")
    args = parser.parse_args()
    if args.draws < 10:
        raise SystemExit("--draws must be at least 10")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds selects no seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    n = 25
    server_clip = 0.42
    rho = 0.21
    alpha_profiles = [
        ("uniform", 0.0, 2.0),
        ("kappa2_boundary", alpha_max(n, 2.0), 2.0),
        ("kappa4_boundary", alpha_max(n, 4.0), 4.0),
        ("kappa10_boundary", alpha_max(n, 10.0), 10.0),
        ("diagnostic_alpha4", 4.0, None),
        ("diagnostic_alpha8", 8.0, None),
    ]
    base_cells = []
    for dimension in (4, 64):
        for heteroscedastic in (False, True):
            for d_score in (0.105, 0.210, 0.441):
                for noise_scale in (0.08, 0.16, 0.28):
                    base_cells.append(
                        {
                            "dimension": dimension,
                            "heteroscedastic_noise": heteroscedastic,
                            "D_score": d_score,
                            "noise_l2_scale": noise_scale,
                        }
                    )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for base_index, base in enumerate(base_cells):
            paired = make_paired_inputs(
                n=n,
                dimension=int(base["dimension"]),
                draws=args.draws,
                seed=seed * 1000 + base_index,
                server_clip=server_clip,
                noise_l2_scale=float(base["noise_l2_scale"]),
                drift_l2_scale=0.0,
                heterogeneity_l2_scale=0.0,
                heteroscedastic_noise=bool(base["heteroscedastic_noise"]),
            )
            for profile, alpha, kappa in alpha_profiles:
                row = evaluate_cell(
                    paired=paired,
                    n=n,
                    dimension=int(base["dimension"]),
                    draws=args.draws,
                    seed=seed,
                    server_clip=server_clip,
                    d_score=float(base["D_score"]),
                    rho=rho,
                    noise_l2_scale=float(base["noise_l2_scale"]),
                    drift_l2_scale=0.0,
                    heterogeneity_l2_scale=0.0,
                    heteroscedastic_noise=bool(base["heteroscedastic_noise"]),
                    alpha_profile=profile,
                    alpha=alpha,
                    certificate_kappa_w=kappa,
                )
                promoted, failed = _promotion_decision(row)
                row["promoted_for_end_to_end"] = promoted
                row["failed_promotion_checks"] = ";".join(failed)
                rows.append(row)

    csv_path = args.output_dir / "paired_mechanistic_cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    promoted = [row for row in rows if bool(row["promoted_for_end_to_end"])]
    robust_promotions = _robust_promotions(promoted, set(seeds))
    promoted_path = args.output_dir / "promoted_cells.json"
    promoted_path.write_text(
        json.dumps(promoted, indent=2, allow_nan=False), encoding="utf-8"
    )
    robust_path = args.output_dir / "promoted_configurations_all_seeds.json"
    robust_path.write_text(
        json.dumps(robust_promotions, indent=2, allow_nan=False), encoding="utf-8"
    )
    summary = {
        "schema_version": 2,
        "experiment": "n25 strictly paired mechanistic current-versus-delayed stress",
        "privacy_claim": "none; simulation-only server mechanism diagnostic",
        "pairing_invariant": (
            "within each cell current and delayed use identical clean vectors, fresh "
            "perturbations, clipping, reference geometry and alpha; only score time differs"
        ),
        "n": n,
        "seeds": seeds,
        "draws_per_cell": args.draws,
        "base_randomness_cells": len(base_cells) * len(seeds),
        "evaluated_cells": len(rows),
        "promoted_cells": len(promoted),
        "promoted_certified_cells": sum(
            bool(row["influence_certificate_claimed"]) for row in promoted
        ),
        "promoted_configurations_all_seeds": len(robust_promotions),
        "promoted_certified_configurations_all_seeds": sum(
            bool(row["influence_certificate_claimed"])
            for row in robust_promotions
        ),
        "mean_relative_reduction_promoted": (
            mean(float(row["relative_delay_noise_mse_reduction"]) for row in promoted)
            if promoted
            else None
        ),
        "outputs": {
            "all_cells_csv": str(csv_path),
            "promoted_cells_json": str(promoted_path),
            "promoted_configurations_all_seeds_json": str(robust_path),
            "report": str(args.report),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_markdown(
        args.report,
        rows,
        promoted,
        robust_promotions,
        draws=args.draws,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
