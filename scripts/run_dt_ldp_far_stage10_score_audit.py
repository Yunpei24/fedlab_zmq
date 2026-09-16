#!/usr/bin/env python3
"""Stage 10: controlled audit of noise-aware and Byzantine-aware FAR scores.

This is deliberately not an end-to-end Fashion-MNIST benchmark.  It creates
vector cohorts for which the clean honest geometry, public DP noise tiers,
honest outliers and Byzantine identities are known.  That oracle information
lets us reject scores that merely look promising in final accuracy while in
fact ranking privacy noise or adversarial messages.

The gates in ``SELECTION_GATES`` are declared in this file before the results
are generated.  Passing the synthetic audit is necessary, not sufficient:
survivors must subsequently be tested end-to-end on held-out Fashion-MNIST
seeds without changing these gates after seeing final accuracy.
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

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.noise_aware_scores import (
    debiased_distance_scores,
    directional_trust_scores,
    excess_energy_scores,
    mean_reference_residual_variances,
    null_quantile_scores,
    orthogonalize_scores_against_public_scale,
    pairwise_independence_trust_scores,
    robust_novelty_scores,
    shrink_residual_variances,
)
from robustness.aggregators import aggregate_vectors, centered_clipping, clip_l2


SELECTION_GATES = {
    "no_attack_clean_geometry_correlation_min": 0.25,
    "no_attack_honest_outlier_top5_recall_min": 0.55,
    "no_attack_abs_score_noise_correlation_max": 0.20,
    "no_attack_noise_tier_score_range_max": 0.10,
    "attacked_byzantine_weight_mass_max": 0.20,
    "attacked_global_honest_outlier_top5_recall_min": 0.35,
    "no_attack_degenerate_draw_rate_max": 0.10,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    shrinkage: float | None = None
    leave_one_out: bool = False
    alignment_gate: bool = False
    independence_gate: bool = False
    calibrated_independence_gate: bool = False
    noise_orthogonalization: bool = False
    trust_floor: float = 0.05


CANDIDATES = (
    Candidate("raw_bounded", "raw"),
    Candidate("debiased_beta0", "debiased", shrinkage=0.0),
    Candidate("debiased_beta05", "debiased", shrinkage=0.5),
    Candidate("excess_energy_beta0", "excess", shrinkage=0.0),
    Candidate("excess_energy_beta05", "excess", shrinkage=0.5),
    Candidate("null_quantile_beta0", "null_quantile", shrinkage=0.0),
    Candidate("null_quantile_beta05", "null_quantile", shrinkage=0.5),
    Candidate("public_null_mc_moment", "mc_moment"),
    Candidate("public_null_mc_quantile", "mc_quantile"),
    Candidate(
        "loo_public_null_mc_quantile",
        "mc_quantile",
        leave_one_out=True,
    ),
    Candidate(
        "loo_null_mc_quantile_alignment",
        "mc_quantile",
        leave_one_out=True,
        alignment_gate=True,
    ),
    Candidate(
        "loo_debiased_beta05_alignment",
        "debiased",
        shrinkage=0.5,
        leave_one_out=True,
        alignment_gate=True,
    ),
    Candidate(
        "loo_excess_beta0_dual_trust",
        "excess",
        shrinkage=0.0,
        leave_one_out=True,
        alignment_gate=True,
        independence_gate=True,
    ),
    Candidate(
        "loo_null_mc_quantile_dual_trust",
        "mc_quantile",
        leave_one_out=True,
        alignment_gate=True,
        independence_gate=True,
    ),
    Candidate(
        "public_null_mc_moment_independence",
        "mc_moment",
        independence_gate=True,
    ),
    Candidate(
        "loo_excess_beta0_calibrated_dual_trust",
        "excess",
        shrinkage=0.0,
        leave_one_out=True,
        alignment_gate=True,
        calibrated_independence_gate=True,
    ),
    Candidate(
        "loo_null_mc_quantile_calibrated_dual_trust",
        "mc_quantile",
        leave_one_out=True,
        alignment_gate=True,
        calibrated_independence_gate=True,
    ),
    Candidate(
        "public_null_mc_moment_calibrated_independence",
        "mc_moment",
        calibrated_independence_gate=True,
    ),
    Candidate(
        "loo_excess_calibrated_dual_trust_orthogonalized",
        "excess",
        shrinkage=0.0,
        leave_one_out=True,
        alignment_gate=True,
        calibrated_independence_gate=True,
        noise_orthogonalization=True,
    ),
    Candidate(
        "null_mc_moment_calibrated_independence_orthogonalized",
        "mc_moment",
        calibrated_independence_gate=True,
        noise_orthogonalization=True,
    ),
)


def _finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(statistics.fmean(values)) if values else float("nan")


def _finite_std(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-15:
        return float("nan")
    return float((x @ y / denominator).item())


def _reference(vectors: torch.Tensor, name: str, rho: float) -> torch.Tensor:
    if name == "fcc":
        return centered_clipping(
            vectors,
            anchor=torch.zeros(vectors.shape[1], dtype=vectors.dtype),
            tau=rho,
        )
    if name == "rfa":
        return aggregate_vectors(
            vectors,
            "rfa",
            num_byzantine=0,
            max_iter=60,
            tol=1e-7,
            smoothing=1e-8,
        )
    raise ValueError(f"Unknown reference {name!r}")


def _fcc_leave_one_out_references(
    vectors: torch.Tensor, rho: float
) -> torch.Tensor:
    """Exact leave-one-out references for one-step FCC with public zero anchor."""

    clipped = clip_l2(vectors, rho)
    return (clipped.sum(dim=0, keepdim=True) - clipped) / (vectors.shape[0] - 1)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-15)


def _orthogonal(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return _unit(vector - (vector @ direction) * direction)


def _clean_honest_vectors(
    seed: int,
    *,
    n_honest: int,
    dimension: int,
    outlier_geometry: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    descent = _unit(torch.randn(dimension, generator=generator, dtype=torch.float64))
    centre = 0.08 * descent
    deviations = torch.randn(
        n_honest, dimension, generator=generator, dtype=torch.float64
    )
    deviations = torch.stack([_orthogonal(row, descent) for row in deviations])
    clean = centre + 0.018 * deviations

    outlier_mask = torch.zeros(n_honest, dtype=torch.bool)
    outlier_ids = torch.linspace(1, n_honest - 2, steps=5).round().long().unique()
    if len(outlier_ids) != 5:
        raise RuntimeError("The fixed honest-outlier construction must contain five IDs")
    outlier_directions = torch.randn(
        5, dimension, generator=generator, dtype=torch.float64
    )
    outlier_directions = torch.stack(
        [_orthogonal(row, descent) for row in outlier_directions]
    )
    if outlier_geometry == "aligned":
        shifts = 0.09 * (
            0.60 * descent[None, :] + 0.80 * outlier_directions
        )
    elif outlier_geometry == "orthogonal":
        shifts = 0.09 * outlier_directions
    else:
        raise ValueError(outlier_geometry)
    clean[outlier_ids] += shifts
    outlier_mask[outlier_ids] = True
    return clean, outlier_mask


def _noise_scales(permutation: str, n: int) -> torch.Tensor:
    base = torch.tensor(([1.0, 1.5, 2.0] * ((n + 2) // 3))[:n], dtype=torch.float64)
    if permutation == "identity":
        return base
    if permutation == "rotate7":
        return torch.roll(base, shifts=7)
    if permutation == "reverse":
        return torch.flip(base, dims=(0,))
    raise ValueError(permutation)


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
        coordinate_std = honest_noisy.std(dim=0, unbiased=False)
        vector = centre - 1.5 * coordinate_std
    else:
        raise ValueError(threat)
    return vector[None, :].repeat(num_byzantine, 1)


def _make_observed_cohort(
    clean_honest: torch.Tensor,
    scales: torch.Tensor,
    *,
    threat: str,
    num_byzantine: int,
    noise_std: float,
    server_clip: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_honest = clean_honest.shape[0]
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        clean_honest.shape, generator=generator, dtype=clean_honest.dtype
    )
    honest_preclip = clean_honest + noise_std * scales[:n_honest, None] * noise
    if threat == "none":
        preclip = honest_preclip
        byzantine_mask = torch.zeros(n_honest, dtype=torch.bool)
    else:
        byzantine = _attack_vectors(
            honest_preclip,
            threat=threat,
            num_byzantine=num_byzantine,
        )
        preclip = torch.cat((honest_preclip, byzantine), dim=0)
        byzantine_mask = torch.zeros(preclip.shape[0], dtype=torch.bool)
        byzantine_mask[n_honest:] = True
    norms = torch.linalg.vector_norm(preclip, dim=1)
    contraction = (server_clip / norms.clamp_min(1e-15)).clamp(max=1.0)
    return clip_l2(preclip, server_clip), contraction, byzantine_mask


def _analytic_variances(
    scales: torch.Tensor,
    contraction: torch.Tensor,
    *,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    upload = (noise_std * scales * contraction).square().clamp_min(1e-12)
    current = mean_reference_residual_variances(upload)
    n = upload.numel()
    loo = upload + (upload.sum() - upload) / ((n - 1) ** 2)
    return upload, current, loo


def _nearest_standardized_separations(
    vectors: torch.Tensor,
    upload_variances: torch.Tensor,
) -> torch.Tensor:
    pairwise = torch.cdist(vectors, vectors, p=2)
    scale = torch.sqrt(
        vectors.shape[1]
        * (
            upload_variances[:, None]
            + upload_variances[None, :]
            + 1e-12
        )
    )
    standardized = pairwise / scale
    diagonal = torch.eye(vectors.shape[0], dtype=torch.bool)
    return standardized.masked_fill(diagonal, float("inf")).min(dim=1).values


def _public_null_calibration(
    *,
    scales: torch.Tensor,
    noise_std: float,
    server_clip: float,
    reference: str,
    rho: float,
    dimension: int,
    draws: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    current_energies = []
    loo_energies = []
    nearest_separations = []
    for _ in range(draws):
        noise = torch.randn(
            scales.shape[0], dimension, generator=generator, dtype=torch.float64
        )
        preclip = noise_std * scales[:, None] * noise
        norms = torch.linalg.vector_norm(preclip, dim=1)
        contraction = (server_clip / norms.clamp_min(1e-15)).clamp(max=1.0)
        vectors = clip_l2(preclip, server_clip)
        upload_variances = (noise_std * scales * contraction).square().clamp_min(
            1e-12
        )
        nearest_separations.append(
            _nearest_standardized_separations(vectors, upload_variances)
        )
        reference_vector = _reference(vectors, reference, rho)
        current_energies.append(
            torch.linalg.vector_norm(vectors - reference_vector, dim=1).square()
        )
        if reference == "fcc":
            loo_refs = _fcc_leave_one_out_references(vectors, rho)
            loo_energies.append(
                torch.linalg.vector_norm(vectors - loo_refs, dim=1).square()
            )
    current = torch.stack(current_energies)
    result = {
        "current_samples": current,
        "current_mean": current.mean(dim=0),
        "current_std": current.std(dim=0, unbiased=True).clamp_min(1e-12),
        "independence_samples": torch.stack(nearest_separations),
    }
    if loo_energies:
        loo = torch.stack(loo_energies)
        result.update(
            {
                "loo_samples": loo,
                "loo_mean": loo.mean(dim=0),
                "loo_std": loo.std(dim=0, unbiased=True).clamp_min(1e-12),
            }
        )
    return result


def _mc_quantile_scores(
    observed_energy: torch.Tensor,
    calibration_samples: torch.Tensor,
    *,
    lower_quantile: float = 0.90,
) -> torch.Tensor:
    quantile = (
        calibration_samples <= observed_energy[None, :]
    ).double().mean(dim=0)
    return ((quantile - lower_quantile) / (1.0 - lower_quantile)).clamp(0.0, 1.0)


def _candidate_scores(
    candidate: Candidate,
    *,
    vectors: torch.Tensor,
    reference: torch.Tensor,
    loo_references: torch.Tensor | None,
    current_variances: torch.Tensor,
    loo_variances: torch.Tensor,
    upload_variances: torch.Tensor,
    calibration: dict[str, torch.Tensor],
    score_dimension: int,
    distance_clip: float,
    public_scales: torch.Tensor,
) -> torch.Tensor:
    if candidate.leave_one_out:
        if loo_references is None:
            raise ValueError("leave-one-out candidate requires FCC references")
        references = loo_references
        variances = loo_variances
        calibration_prefix = "loo"
    else:
        references = reference
        variances = current_variances
        calibration_prefix = "current"
    distances = torch.linalg.vector_norm(vectors - references, dim=1)

    if candidate.family == "raw":
        scores = (distances / distance_clip).clamp(max=1.0)
    elif candidate.family in {"debiased", "excess", "null_quantile"}:
        variances = shrink_residual_variances(
            variances,
            shrinkage=float(candidate.shrinkage),
        )
        if candidate.family == "debiased":
            scores, _ = debiased_distance_scores(
                distances,
                variances,
                score_dimension=score_dimension,
                distance_clip=distance_clip,
            )
        elif candidate.family == "excess":
            scores, _ = excess_energy_scores(
                distances,
                variances,
                score_dimension=score_dimension,
                z_clip=5.0,
            )
        else:
            scores, _ = null_quantile_scores(
                distances,
                variances,
                score_dimension=score_dimension,
                lower_z=1.2815515655446004,
                upper_z=3.5,
            )
    elif candidate.family == "mc_moment":
        mean = calibration[f"{calibration_prefix}_mean"]
        std = calibration[f"{calibration_prefix}_std"]
        z = (distances.square() - mean) / std
        scores = (torch.relu(z) / 4.0).clamp(max=1.0)
    elif candidate.family == "mc_quantile":
        scores = _mc_quantile_scores(
            distances.square(),
            calibration[f"{calibration_prefix}_samples"],
        )
    else:
        raise ValueError(candidate.family)

    if candidate.alignment_gate:
        trust, _ = directional_trust_scores(
            vectors,
            references,
            reject_cosine=-0.10,
            full_trust_cosine=0.25,
        )
        scores = robust_novelty_scores(
            scores,
            trust,
            trust_floor=candidate.trust_floor,
        )
    if candidate.independence_gate:
        independence, _ = pairwise_independence_trust_scores(
            vectors,
            upload_variances,
            low_separation=0.20,
            full_separation=0.70,
        )
        scores = robust_novelty_scores(
            scores,
            independence,
            trust_floor=candidate.trust_floor,
        )
    if candidate.calibrated_independence_gate:
        nearest = _nearest_standardized_separations(vectors, upload_variances)
        null_cdf = (
            calibration["independence_samples"] <= nearest[None, :]
        ).double().mean(dim=0)
        # A separation in the lowest 5% of the public null receives no trust;
        # the gate reaches full trust at the null median.  These two constants
        # are fixed before the Generation-3 results are computed.
        independence = ((null_cdf - 0.05) / (0.50 - 0.05)).clamp(0.0, 1.0)
        scores = robust_novelty_scores(
            scores,
            independence,
            trust_floor=candidate.trust_floor,
        )
    if candidate.noise_orthogonalization:
        scores, _ = orthogonalize_scores_against_public_scale(
            scores,
            public_scales,
        )
    return scores


def _row_metrics(
    *,
    scores: torch.Tensor,
    vectors: torch.Tensor,
    clean_distances: torch.Tensor,
    honest_outliers: torch.Tensor,
    scales: torch.Tensor,
    byzantine_mask: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    honest_mask = ~byzantine_mask
    honest_scores = scores[honest_mask]
    n_honest = int(honest_mask.sum())
    topk_honest = min(5, n_honest)
    honest_rank = torch.topk(honest_scores, k=topk_honest).indices
    honest_recall = float(honest_outliers[honest_rank].float().mean())
    global_rank = torch.topk(scores, k=min(5, scores.numel())).indices
    full_outlier_mask = torch.zeros_like(byzantine_mask)
    full_outlier_mask[:n_honest] = honest_outliers
    global_recall = float(full_outlier_mask[global_rank].float().mean())
    weights = torch.softmax(alpha * scores, dim=0)
    honest_target = vectors[honest_mask].mean(dim=0)
    aggregate = (weights[:, None] * vectors).sum(dim=0)
    tier_means = []
    for tier in torch.unique(scales[:n_honest]):
        tier_means.append(float(honest_scores[scales[:n_honest] == tier].mean()))
    return {
        "clean_geometry_correlation": _corr(honest_scores, clean_distances),
        "score_noise_scale_correlation": _corr(
            honest_scores, scales[:n_honest]
        ),
        "noise_tier_score_range": max(tier_means) - min(tier_means),
        "honest_outlier_top5_recall": honest_recall,
        "global_honest_outlier_top5_recall": global_recall,
        "honest_outlier_weight_mass": float(weights[full_outlier_mask].sum()),
        "byzantine_weight_mass": float(weights[byzantine_mask].sum()),
        "byzantine_top5_rate": float(byzantine_mask[global_rank].float().mean()),
        "aggregate_error_to_observed_honest_mean": float(
            torch.linalg.vector_norm(aggregate - honest_target)
        ),
        "score_span": float(scores.max() - scores.min()),
        "weight_concentration": float(scores.numel() * weights.square().sum()),
        "score_zero_rate": float((scores <= 1e-12).float().mean()),
        "score_saturation_rate": float((scores >= 1.0 - 1e-7).float().mean()),
        "score_degenerate": float(float(scores.max() - scores.min()) <= 1e-8),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    candidates = sorted({row["candidate"] for row in rows})
    summaries = []
    for name in candidates:
        selected = [row for row in rows if row["candidate"] == name]
        clean = [row for row in selected if row["threat"] == "none"]
        attacked = [row for row in selected if row["threat"] != "none"]
        attack_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in attacked:
            attack_groups[
                (row["threat"], row["reference"], row["outlier_geometry"])
            ].append(row["byzantine_weight_mass"])
        max_group_byzantine_mass = max(
            (_finite_mean(values) for values in attack_groups.values()),
            default=float("nan"),
        )
        summary = {
            "candidate": name,
            "observations": len(selected),
            "clean_geometry_correlation_mean": _finite_mean(
                [row["clean_geometry_correlation"] for row in clean]
            ),
            "clean_geometry_correlation_std": _finite_std(
                [row["clean_geometry_correlation"] for row in clean]
            ),
            "honest_outlier_top5_recall_mean": _finite_mean(
                [row["honest_outlier_top5_recall"] for row in clean]
            ),
            "abs_score_noise_correlation_mean": _finite_mean(
                [abs(row["score_noise_scale_correlation"]) for row in clean]
            ),
            "noise_tier_score_range_mean": _finite_mean(
                [row["noise_tier_score_range"] for row in clean]
            ),
            "honest_outlier_weight_mass_mean": _finite_mean(
                [row["honest_outlier_weight_mass"] for row in clean]
            ),
            "attacked_byzantine_weight_mass_mean": _finite_mean(
                [row["byzantine_weight_mass"] for row in attacked]
            ),
            "attacked_byzantine_weight_mass_max_group_mean": max_group_byzantine_mass,
            "attacked_global_honest_outlier_top5_recall_mean": _finite_mean(
                [row["global_honest_outlier_top5_recall"] for row in attacked]
            ),
            "attacked_byzantine_top5_rate_mean": _finite_mean(
                [row["byzantine_top5_rate"] for row in attacked]
            ),
            "attacked_aggregate_error_mean": _finite_mean(
                [row["aggregate_error_to_observed_honest_mean"] for row in attacked]
            ),
            "no_attack_degenerate_draw_rate": _finite_mean(
                [row["score_degenerate"] for row in clean]
            ),
        }
        checks = {
            "clean_geometry": summary["clean_geometry_correlation_mean"]
            >= SELECTION_GATES["no_attack_clean_geometry_correlation_min"],
            "honest_outlier_recall": summary["honest_outlier_top5_recall_mean"]
            >= SELECTION_GATES["no_attack_honest_outlier_top5_recall_min"],
            "noise_invariance": summary["abs_score_noise_correlation_mean"]
            <= SELECTION_GATES["no_attack_abs_score_noise_correlation_max"],
            "noise_tier_invariance": summary["noise_tier_score_range_mean"]
            <= SELECTION_GATES["no_attack_noise_tier_score_range_max"],
            "byzantine_mass": summary[
                "attacked_byzantine_weight_mass_max_group_mean"
            ]
            <= SELECTION_GATES["attacked_byzantine_weight_mass_max"],
            "attacked_outlier_recall": summary[
                "attacked_global_honest_outlier_top5_recall_mean"
            ]
            >= SELECTION_GATES[
                "attacked_global_honest_outlier_top5_recall_min"
            ],
            "non_degenerate": summary["no_attack_degenerate_draw_rate"]
            <= SELECTION_GATES["no_attack_degenerate_draw_rate_max"],
        }
        summary["gate_checks"] = checks
        summary["passes_all_gates"] = all(checks.values())
        summaries.append(summary)
    return summaries


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _write_report(
    path: Path,
    *,
    detail_path: Path,
    rows: list[dict],
    summaries: list[dict],
    draws: int,
    calibration_draws: int,
    seeds: tuple[int, ...],
) -> None:
    passed = [row["candidate"] for row in summaries if row["passes_all_gates"]]
    lines = [
        "# DT-LDP-FAR — Stage 10: audit contrôlé de la qualité du score",
        "",
        "## Verdict",
        "",
        (
            "Les candidats qui satisfont **tous** les critères pré-enregistrés "
            f"sont : **{', '.join(passed)}**."
            if passed
            else "Aucun candidat ne satisfait simultanément tous les critères pré-enregistrés."
        ),
        "",
        "Ce verdict est un filtre de mécanisme, pas une validation d'accuracy. "
        "Un candidat ne peut être promu qu'après une expérience end-to-end sur "
        "Fashion-MNIST avec des seeds tenues à l'écart de cet audit.",
        "",
        "## Ce que signifie « données synthétiques »",
        "",
        "Il ne s'agit pas d'images synthétiques ni de Fashion-MNIST. Chaque "
        "observation est une cohorte artificielle de 25 vecteurs de dimension "
        "128. Le générateur connaît le centre honnête latent, cinq honest "
        "outliers, le niveau de bruit local-DP de chaque client et, sous attaque, "
        "les cinq identités byzantines. Ces oracles permettent de mesurer "
        "directement si un score suit le signal propre, le bruit ou l'attaque.",
        "",
        "Deux géométries d'honest outliers sont testées : une déviation restant "
        "partiellement alignée avec la direction honnête et une déviation "
        "orthogonale. Les niveaux de bruit publics suivent trois permutations des "
        "échelles 1, 1.5 et 2. Les attaques sont IPM, ALIE et Bit-Flip ×10.",
        "",
        "## Candidats construits",
        "",
        "| Candidat | Construction | Rôle |",
        "|---|---|---|",
        "| `raw_bounded` | distance bornée historique | témoin FAR |",
        "| `debiased_beta0` | retrait complet du plancher d'énergie DP | correction absolue |",
        "| `debiased_beta05` | même correction, variance shrinkée à 50 % | limiter la sur-correction |",
        "| `excess_energy_beta0` | énergie excédentaire standardisée | signal/bruit |",
        "| `excess_energy_beta05` | énergie excédentaire avec shrinkage | compromis signal/bruit |",
        "| `null_quantile_beta0` | quantile nul Wilson–Hilferty | seuil comparable entre dimensions |",
        "| `null_quantile_beta05` | quantile nul avec shrinkage | quantile moins dépendant de σ |",
        "| `public_null_mc_moment` | moments nuls simulés publiquement | intégrer clipping et référence non linéaire |",
        "| `public_null_mc_quantile` | CDF nulle simulée publiquement | calibration sans hypothèse χ² exacte |",
        "| `loo_public_null_mc_quantile` | quantile public, référence sans le client i | supprimer l'auto-masquage |",
        "| `loo_null_mc_quantile_alignment` | précédent × filtre directionnel | rejeter IPM/Bit-Flip anti-alignés |",
        "| `loo_debiased_beta05_alignment` | distance débiaisée LOO × filtre | alternative géométrique robuste |",
        "| `loo_excess_beta0_dual_trust` | énergie LOO × alignement × indépendance | signal fort et anti-collusion |",
        "| `loo_null_mc_quantile_dual_trust` | quantile LOO × deux filtres | calibration publique et anti-collusion |",
        "| `public_null_mc_moment_independence` | moment public × séparation DP | ablation anti-collusion sans direction |",
        "| `loo_excess_beta0_calibrated_dual_trust` | énergie LOO × alignement × CDF publique de séparation | neutraliser le biais σ du filtre |",
        "| `loo_null_mc_quantile_calibrated_dual_trust` | quantile LOO × deux filtres calibrés | candidat entièrement calibré |",
        "| `public_null_mc_moment_calibrated_independence` | moment public × CDF de séparation | ablation calibrée sans direction |",
        "| `loo_excess_calibrated_dual_trust_orthogonalized` | candidat précédent résidualisé sur σ public | retirer la dépendance linéaire restante |",
        "| `null_mc_moment_calibrated_independence_orthogonalized` | moment public robuste résidualisé | ablation sans filtre directionnel |",
        "",
        r"Le shrinkage suit $v_i^{(β)}=(1-β)v_i+β\bar v$. Le leave-one-out "
        "est évalué seulement avec $F_{CC}$, où il est calculable exactement et "
        "efficacement pour l'étape ancrée publique.",
        "",
        "## Protocole fixé avant lecture des résultats",
        "",
        f"- {draws} tirages évalués par combinaison ; {calibration_draws} tirages publics séparés pour les scores Monte-Carlo.",
        f"- Seeds : {', '.join(str(seed) for seed in seeds)} ; références : $F_{{CC}}$ et RFA.",
        "- Le même cap analytique est utilisé pour tous : $n=25$, $κ_w=2$, "
        r"$α=\log(2(n-1)/(n-2))$, donc $ω_i≤2/n$.",
        "- Le clipping serveur reste fixé ; cette étape ne change que le score.",
        "",
        "## Critères de sélection",
        "",
        "| Critère | Seuil | Information fournie |",
        "|---|---:|---|",
        f"| Corrélation score–géométrie propre, sans attaque | ≥ {SELECTION_GATES['no_attack_clean_geometry_correlation_min']:.2f} | le score retrouve le signal latent |",
        f"| Rappel top-5 des honest outliers, sans attaque | ≥ {SELECTION_GATES['no_attack_honest_outlier_top5_recall_min']:.2f} | les clients atypiques utiles restent visibles |",
        f"| Corrélation absolue score–niveau de bruit | ≤ {SELECTION_GATES['no_attack_abs_score_noise_correlation_max']:.2f} | le score ne classe pas principalement σ_i |",
        f"| Écart moyen des scores entre tiers de bruit | ≤ {SELECTION_GATES['no_attack_noise_tier_score_range_max']:.2f} | contrôle non redondant avec Pearson |",
        f"| Pire masse byzantine moyenne par scénario | ≤ {SELECTION_GATES['attacked_byzantine_weight_mass_max']:.2f} | les 20 % de Byzantins ne sont pas surpondérés |",
        f"| Rappel global des honest outliers sous attaque | ≥ {SELECTION_GATES['attacked_global_honest_outlier_top5_recall_min']:.2f} | les attaques ne chassent pas tout le signal utile du top-5 |",
        f"| Taux de scores dégénérés, sans attaque | ≤ {SELECTION_GATES['no_attack_degenerate_draw_rate_max']:.2f} | le score reste informatif |",
        "",
        "## Résultats agrégés",
        "",
        "| Candidat | Corr. propre | Rappel honest top-5 | Corr. bruit absolue | Écart tiers | Masse byz max | Rappel honest sous attaque | Dégénéré | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            "| {candidate} | {clean} | {recall} | {noise} | {tier} | {byz} | {attack_recall} | {degenerate} | {verdict} |".format(
                candidate=row["candidate"],
                clean=_fmt(row["clean_geometry_correlation_mean"]),
                recall=_fmt(row["honest_outlier_top5_recall_mean"]),
                noise=_fmt(row["abs_score_noise_correlation_mean"]),
                tier=_fmt(row["noise_tier_score_range_mean"]),
                byz=_fmt(row["attacked_byzantine_weight_mass_max_group_mean"]),
                attack_recall=_fmt(
                    row["attacked_global_honest_outlier_top5_recall_mean"]
                ),
                degenerate=_fmt(row["no_attack_degenerate_draw_rate"]),
                verdict="**PASS**" if row["passes_all_gates"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## Règle de décision pour la suite",
            "",
            "1. Les candidats `PASS` sont transférés tels quels vers "
            "Fashion-MNIST ; aucune constante n'est retouchée avec l'accuracy finale.",
            "2. S'il n'existe aucun `PASS`, les échecs par critère déterminent la "
            "prochaine construction ; on ne choisit pas simplement le meilleur "
            "score composite après coup.",
            "3. Sur Fashion-MNIST, les comparaisons doivent inclure le score FAR "
            "brut, le meilleur score passé, FedAvg-LDP et les références $F_{CC}$/RFA, "
            "avec randomness appariée et plusieurs seeds.",
            "",
            "## Limites",
            "",
            "Les vecteurs synthétiques ne reproduisent ni l'optimisation non convexe "
            "de LeNet-5 ni le drift temporel des clients. Les attaques sont des "
            "instanciations contrôlées, pas l'ensemble de toutes les stratégies "
            "adaptatives. Le modèle de covariance reste isotrope pour les candidats "
            "analytiques ; les deux candidats Monte-Carlo absorbent partiellement "
            "les non-linéarités du clipping et de la référence, mais restent calibrés "
            "sous une hypothèse nulle publique.",
            "",
            f"Détails : `{detail_path}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    output_dir: Path,
    report_path: Path,
    *,
    draws: int,
    calibration_draws: int,
    seeds: tuple[int, ...],
) -> None:
    n = 25
    num_byzantine = 5
    dimension = 128
    noise_std = 0.012
    server_clip = 0.45
    rho = 0.24
    distance_clip = 0.18
    alpha = math.log(2.0 * (n - 1) / (n - 2.0))
    permutations = ("identity", "rotate7", "reverse")
    references = ("fcc", "rfa")
    threats = ("none", "ipm", "alie", "bitflip_x10")
    outlier_geometries = ("aligned", "orthogonal")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for permutation in permutations:
        scales = _noise_scales(permutation, n)
        for reference_name in references:
            calibration_cache[(permutation, reference_name)] = (
                _public_null_calibration(
                    scales=scales,
                    noise_std=noise_std,
                    server_clip=server_clip,
                    reference=reference_name,
                    rho=rho,
                    dimension=dimension,
                    draws=calibration_draws,
                    seed=900_000 + 101 * len(calibration_cache),
                )
            )

    rows: list[dict] = []
    for seed in seeds:
        for outlier_geometry in outlier_geometries:
            for permutation in permutations:
                scales = _noise_scales(permutation, n)
                for threat in threats:
                    n_honest = n if threat == "none" else n - num_byzantine
                    clean, outlier_mask = _clean_honest_vectors(
                        seed,
                        n_honest=n_honest,
                        dimension=dimension,
                        outlier_geometry=outlier_geometry,
                    )
                    for draw in range(draws):
                        vectors, contraction, byzantine_mask = _make_observed_cohort(
                            clean,
                            scales,
                            threat=threat,
                            num_byzantine=num_byzantine,
                            noise_std=noise_std,
                            server_clip=server_clip,
                            seed=seed * 1_000_000
                            + 10_000 * outlier_geometries.index(outlier_geometry)
                            + 1_000 * permutations.index(permutation)
                            + 100 * threats.index(threat)
                            + draw,
                        )
                        upload_var, current_var, loo_var = _analytic_variances(
                            scales,
                            contraction,
                            noise_std=noise_std,
                        )
                        for reference_name in references:
                            reference = _reference(vectors, reference_name, rho)
                            loo_refs = (
                                _fcc_leave_one_out_references(vectors, rho)
                                if reference_name == "fcc"
                                else None
                            )
                            clean_reference = _reference(
                                clean, reference_name, rho
                            )
                            clean_distances = torch.linalg.vector_norm(
                                clean - clean_reference, dim=1
                            )
                            calibration = calibration_cache[
                                (permutation, reference_name)
                            ]
                            for candidate in CANDIDATES:
                                if candidate.leave_one_out and reference_name != "fcc":
                                    continue
                                scores = _candidate_scores(
                                    candidate,
                                    vectors=vectors,
                                    reference=reference,
                                    loo_references=loo_refs,
                                    current_variances=current_var,
                                    loo_variances=loo_var,
                                    upload_variances=upload_var,
                                    calibration=calibration,
                                    score_dimension=dimension,
                                    distance_clip=distance_clip,
                                    public_scales=scales,
                                )
                                metrics = _row_metrics(
                                    scores=scores,
                                    vectors=vectors,
                                    clean_distances=clean_distances,
                                    honest_outliers=outlier_mask,
                                    scales=scales,
                                    byzantine_mask=byzantine_mask,
                                    alpha=alpha,
                                )
                                rows.append(
                                    {
                                        "seed": seed,
                                        "draw": draw,
                                        "outlier_geometry": outlier_geometry,
                                        "noise_permutation": permutation,
                                        "reference": reference_name,
                                        "threat": threat,
                                        "candidate": candidate.name,
                                        **metrics,
                                    }
                                )

    fieldnames = list(rows[0])
    with (output_dir / "detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summaries = _aggregate(rows)
    payload = {
        "status": "completed",
        "protocol": {
            "n": n,
            "num_byzantine": num_byzantine,
            "dimension": dimension,
            "noise_std": noise_std,
            "server_clip": server_clip,
            "fcc_rho": rho,
            "distance_clip": distance_clip,
            "alpha": alpha,
            "kappa_w": 2.0,
            "draws": draws,
            "calibration_draws": calibration_draws,
            "seeds": list(seeds),
            "noise_permutations": list(permutations),
            "references": list(references),
            "threats": list(threats),
            "outlier_geometries": list(outlier_geometries),
        },
        "selection_gates": SELECTION_GATES,
        "candidate_definitions": [candidate.__dict__ for candidate in CANDIDATES],
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_path,
        detail_path=output_dir / "detail.csv",
        rows=rows,
        summaries=summaries,
        draws=draws,
        calibration_draws=calibration_draws,
        seeds=seeds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/dt_ldp_far/score_quality_stage10_synthetic_v4",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage10_Synthetic_Score_Audit_Generation4.md",
    )
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--calibration-draws", type=int, default=300)
    parser.add_argument(
        "--seeds",
        default="28,36,54",
        help="Comma-separated synthetic cohort seeds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.draws < 1 or args.calibration_draws < 20:
        raise SystemExit("draws must be >=1 and calibration-draws must be >=20")
    parsed_seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not parsed_seeds:
        raise SystemExit("at least one seed is required")
    run(
        args.output_dir,
        args.report,
        draws=args.draws,
        calibration_draws=args.calibration_draws,
        seeds=parsed_seeds,
    )
