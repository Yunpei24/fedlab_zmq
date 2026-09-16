#!/usr/bin/env python3
"""Stage 14B: frozen audit of robust references, trust and guarded output.

This is a vector-level preflight.  It deliberately separates the validated
Stage-14A novelty score from the robust reference used by the trust channel
and final guarded output.  No model accuracy is observed during selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.noise_aware_scores import (  # noqa: E402
    calibrated_null_energy_scores,
    directional_trust_scores,
    nearest_standardized_separations,
    separate_novelty_trust_scores,
)
from robustness.aggregators import (  # noqa: E402
    aggregate_vectors,
    centered_clipping,
    guarded_aggregate,
    regularized_huber_reference,
)
from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (  # noqa: E402
    _clean_honest_vectors,
    _corr,
    _fcc_loo_energies,
    _finite_mean,
    _ks_distance,
    _noise_scales,
    _unit,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference(vectors: torch.Tensor, name: str, config: dict[str, Any]) -> torch.Tensor:
    settings = config["robust_reference"]
    anchor = torch.zeros(vectors.shape[1], dtype=vectors.dtype, device=vectors.device)
    if name == "fcc":
        return centered_clipping(
            vectors, anchor=anchor, tau=float(settings["fcc_radius"])
        )
    if name == "huber":
        return regularized_huber_reference(
            vectors,
            anchor=anchor,
            tau=float(settings["huber_radius"]),
            gamma=float(settings["huber_gamma"]),
            num_steps=int(settings["huber_steps"]),
        )
    if name == "rfa":
        return aggregate_vectors(
            vectors,
            "rfa",
            max_iter=int(settings["rfa_max_iter"]),
            tol=float(settings["rfa_tolerance"]),
        )
    if name == "cm_nnm":
        return aggregate_vectors(
            vectors,
            "cm_nnm",
            num_byzantine=int(config["cohort"]["num_byzantine"]),
        )
    raise ValueError(f"Unknown robust reference {name!r}")


def _null_vectors(
    scales: torch.Tensor,
    *,
    draws: int,
    dimension: int,
    noise_std: float,
    server_clip_norm: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        draws,
        len(scales),
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    vectors = noise_std * scales[None, :, None] * noise
    norms = torch.linalg.vector_norm(vectors, dim=2, keepdim=True)
    factors = (float(server_clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)
    return vectors * factors


def _energies(vectors: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    dimension = int(config["score"]["dimension"])
    order = torch.arange(dimension)
    return _fcc_loo_energies(
        vectors,
        coordinate_order=order,
        score_dimension=dimension,
        full_dimension=dimension,
        full_radius=float(config["score"]["fcc_radius"]),
    )


def _energy_matrix(cohorts: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return torch.stack([_energies(vectors, config) for vectors in cohorts])


def _separation_matrix(
    cohorts: torch.Tensor, variances: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [nearest_standardized_separations(vectors, variances) for vectors in cohorts]
    )


def _independence_trust(
    vectors: torch.Tensor,
    variances: torch.Tensor,
    calibration: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    trust_config = config["trust"]
    observed = nearest_standardized_separations(vectors, variances)
    cdf = (calibration <= observed[None, :]).double().mean(dim=0)
    low = float(trust_config["independence_low_null_quantile"])
    full = float(trust_config["independence_full_trust_null_quantile"])
    floor = float(trust_config["trust_floor"])
    calibrated = ((cdf - low) / (full - low)).clamp(min=0.0, max=1.0)
    return floor + (1.0 - floor) * calibrated


def _dual_trust(
    vectors: torch.Tensor,
    reference: torch.Tensor,
    variances: torch.Tensor,
    separation_calibration: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    settings = config["trust"]
    directional, _ = directional_trust_scores(
        vectors,
        reference,
        reject_cosine=float(settings["direction_reject_cosine"]),
        full_trust_cosine=float(settings["direction_full_trust_cosine"]),
    )
    independence = _independence_trust(
        vectors, variances, separation_calibration, config
    )
    return torch.minimum(directional.double(), independence.double())


def _compose_score(
    novelty: torch.Tensor,
    trust: torch.Tensor,
    profile: str,
    config: dict[str, Any],
) -> torch.Tensor:
    if profile == "none":
        return novelty
    if profile == "directional_independence":
        combined, _ = separate_novelty_trust_scores(
            novelty,
            trust,
            trust_fraction=float(config["trust"]["trust_logit_fraction"]),
        )
        return combined
    raise ValueError(f"Unknown trust profile {profile!r}")


def _score(
    energies: torch.Tensor,
    calibration: torch.Tensor,
    family: str,
    config: dict[str, Any],
) -> torch.Tensor:
    settings = config["score"]
    mode = "moment" if family == "effective_moment" else "quantile"
    scores, _ = calibrated_null_energy_scores(
        energies,
        calibration,
        mode=mode,
        z_clip=float(settings["moment_z_clip"]),
        tail_probability=float(settings["quantile_tail_probability"]),
        individual_calibration_weight=float(
            settings["individual_calibration_weight"]
        ),
        variance_ridge=float(settings["variance_ridge"]),
    )
    return scores


def _null_metrics(scores: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    repeated = scales[None, :].expand_as(scores)
    tiers = {
        float(tier): scores[:, scales == tier].reshape(-1)
        for tier in torch.unique(scales)
    }
    means = [float(values.mean()) for values in tiers.values()]
    ks = [
        _ks_distance(tiers[left], tiers[right])
        for left in tiers
        for right in tiers
        if left < right
    ]
    return {
        "null_score_noise_correlation": _corr(scores, repeated),
        "null_noise_tier_score_range": max(means) - min(means),
        "null_pairwise_ks_max": max(ks, default=float("nan")),
    }


def _attack_vectors(
    honest_noisy: torch.Tensor,
    *,
    threat: str,
    count: int,
    scales: torch.Tensor,
    config: dict[str, Any],
    seed: int,
) -> torch.Tensor:
    centre = honest_noisy.mean(dim=0)
    if threat == "ipm":
        return (-3.0 * centre)[None, :].repeat(count, 1)
    if threat == "bitflip_x10":
        return (-10.0 * centre)[None, :].repeat(count, 1)
    if threat == "alie":
        return (
            centre - 1.5 * honest_noisy.std(dim=0, unbiased=False)
        )[None, :].repeat(count, 1)
    if threat == "adaptive_aligned_jitter":
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 99173)
        direction = _unit(centre)
        base = 0.90 * float(config["aggregation"]["server_clip_norm"]) * direction
        jitter = torch.randn(
            count,
            honest_noisy.shape[1],
            generator=generator,
            dtype=honest_noisy.dtype,
        )
        byzantine_scales = scales[-count:]
        return base[None, :] + (
            0.75
            * float(config["cohort"]["noise_std"])
            * byzantine_scales[:, None]
            * jitter
        )
    raise ValueError(f"Unknown threat {threat!r}")


def _observed(
    clean: torch.Tensor,
    scales: torch.Tensor,
    *,
    threat: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    honest = clean + (
        float(config["cohort"]["noise_std"])
        * scales[: len(clean), None]
        * noise
    )
    if threat == "none":
        vectors = honest
        byzantine = torch.zeros(len(vectors), dtype=torch.bool)
    else:
        count = int(config["cohort"]["num_byzantine"])
        attacks = _attack_vectors(
            honest,
            threat=threat,
            count=count,
            scales=scales,
            config=config,
            seed=seed,
        )
        vectors = torch.cat((honest, attacks), dim=0)
        byzantine = torch.zeros(len(vectors), dtype=torch.bool)
        byzantine[len(honest) :] = True
    radius = float(config["aggregation"]["server_clip_norm"])
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    vectors = vectors * (radius / norms.clamp_min(1e-12)).clamp(max=1.0)
    return vectors, byzantine


def _signal_metrics(
    scores: torch.Tensor,
    clean_distances: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    *,
    alpha: float,
) -> tuple[dict[str, float], torch.Tensor]:
    honest = ~byzantine_mask
    honest_scores = scores[honest]
    predicted_honest = torch.topk(honest_scores, min(5, len(honest_scores))).indices
    honest_recall = float(outlier_mask[predicted_honest].double().mean())
    full_outliers = torch.zeros_like(byzantine_mask)
    full_outliers[: len(outlier_mask)] = outlier_mask
    predicted_global = torch.topk(scores, min(5, len(scores))).indices
    weights = torch.softmax(float(alpha) * scores.double(), dim=0)
    return {
        "clean_geometry_correlation": _corr(honest_scores, clean_distances),
        "honest_outlier_recall_at_5": honest_recall,
        "global_honest_outlier_recall_at_5": float(
            full_outliers[predicted_global].double().mean()
        ),
        "honest_outlier_weight_mass": float(weights[full_outliers].sum()),
        "byzantine_weight_mass": float(weights[byzantine_mask].sum()),
        "max_individual_weight": float(weights.max()),
    }, weights


def _group_extreme(
    rows: list[dict[str, Any]], field: str, *, minimum: bool
) -> float:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["threat"],
            row["signal_seed"],
            row["noise_permutation"],
            row["outlier_geometry"],
        )
        grouped[key].append(float(row[field]))
    values = [_finite_mean(group) for group in grouped.values()]
    if not values:
        return float("nan")
    return min(values) if minimum else max(values)


def _summarize_candidates(
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config["gates"]
    keys = sorted(
        {
            (row["family"], row["robust_reference"], row["trust_profile"])
            for row in signal_rows
        }
    )
    summaries: list[dict[str, Any]] = []
    for family, reference, trust in keys:
        null = [
            row
            for row in null_rows
            if row["family"] == family
            and row["robust_reference"] == reference
            and row["trust_profile"] == trust
        ]
        rows = [
            row
            for row in signal_rows
            if row["family"] == family
            and row["robust_reference"] == reference
            and row["trust_profile"] == trust
        ]
        clean = [row for row in rows if row["threat"] == "none"]
        attacked = [row for row in rows if row["threat"] != "none"]
        no_trust = [
            row
            for row in signal_rows
            if row["family"] == family
            and row["robust_reference"] == reference
            and row["trust_profile"] == "none"
            and row["threat"] == "none"
        ]
        clean_recall = _finite_mean([row["honest_outlier_recall_at_5"] for row in clean])
        clean_mass = _finite_mean([row["honest_outlier_weight_mass"] for row in clean])
        base_recall = _finite_mean(
            [row["honest_outlier_recall_at_5"] for row in no_trust]
        )
        base_mass = _finite_mean(
            [row["honest_outlier_weight_mass"] for row in no_trust]
        )
        summary = {
            "family": family,
            "robust_reference": reference,
            "trust_profile": trust,
            "null_abs_score_noise_correlation_max": max(
                abs(float(row["null_score_noise_correlation"])) for row in null
            ),
            "null_noise_tier_score_range_max": max(
                float(row["null_noise_tier_score_range"]) for row in null
            ),
            "null_pairwise_ks_max": max(
                float(row["null_pairwise_ks_max"]) for row in null
            ),
            "clean_geometry_correlation_mean": _finite_mean(
                [row["clean_geometry_correlation"] for row in clean]
            ),
            "honest_outlier_recall_at_5_mean": clean_recall,
            "honest_outlier_weight_mass_mean": clean_mass,
            "max_individual_weight_observed": max(
                float(row["max_individual_weight"]) for row in rows
            ),
            "attacked_byzantine_weight_mass_worst_group": _group_extreme(
                attacked, "byzantine_weight_mass", minimum=False
            ),
            "attacked_global_recall_worst_group": _group_extreme(
                attacked, "global_honest_outlier_recall_at_5", minimum=True
            ),
            "robust_reference_error_ratio_worst_group": _group_extreme(
                attacked, "reference_error_over_honest_dispersion", minimum=False
            ),
            "trust_recall_loss": max(0.0, base_recall - clean_recall),
            "trust_honest_outlier_mass_loss": max(0.0, base_mass - clean_mass),
        }
        checks = {
            "null_corr": summary["null_abs_score_noise_correlation_max"]
            <= float(gates["null_abs_score_noise_correlation_max"]),
            "null_tier": summary["null_noise_tier_score_range_max"]
            <= float(gates["null_noise_tier_score_range_max"]),
            "null_ks": summary["null_pairwise_ks_max"]
            <= float(gates["null_pairwise_ks_max"]),
            "clean_corr": summary["clean_geometry_correlation_mean"]
            >= float(gates["clean_geometry_correlation_min"]),
            "clean_recall": clean_recall
            >= float(gates["honest_outlier_recall_at_5_min"]),
            "clean_mass": clean_mass
            > float(gates["honest_outlier_weight_mass_min_exclusive"]),
            "weight_cap": summary["max_individual_weight_observed"]
            <= float(gates["max_individual_weight"]) + 1e-10,
            "byzantine_mass": summary["attacked_byzantine_weight_mass_worst_group"]
            <= float(gates["attacked_byzantine_weight_mass_max"]),
            "attacked_recall": summary["attacked_global_recall_worst_group"]
            >= float(gates["attacked_global_honest_outlier_recall_at_5_min"]),
            "trust_recall_loss": summary["trust_recall_loss"]
            <= float(gates["trust_recall_loss_max"]),
            "trust_mass_loss": summary["trust_honest_outlier_mass_loss"]
            <= float(gates["trust_honest_outlier_mass_loss_max"]),
            "reference_quality": summary["robust_reference_error_ratio_worst_group"]
            <= float(gates["robust_reference_error_over_honest_dispersion_max"]),
        }
        summary["gate_checks"] = checks
        summary["passes_candidate_gates"] = all(checks.values())
        summaries.append(summary)
    return summaries


def _summarize_guards(
    output_rows: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config["gates"]
    candidate_pass = {
        (row["family"], row["robust_reference"], row["trust_profile"]): bool(
            row["passes_candidate_gates"]
        )
        for row in candidate_summaries
    }
    keys = sorted(
        {
            (
                row["family"],
                row["robust_reference"],
                row["trust_profile"],
                row["guard_fraction"],
            )
            for row in output_rows
        },
        key=lambda key: (key[0], key[1], key[2], str(key[3])),
    )
    summaries = []
    for family, reference, trust, guard in keys:
        rows = [
            row
            for row in output_rows
            if row["family"] == family
            and row["robust_reference"] == reference
            and row["trust_profile"] == trust
            and row["guard_fraction"] == guard
        ]
        clean = [row for row in rows if row["threat"] == "none"]
        attacked = [row for row in rows if row["threat"] != "none"]
        clean_ratio = _finite_mean(
            [row["aggregate_error"] / max(row["raw_error"], 1e-15) for row in clean]
        )
        reductions: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in attacked:
            key = (
                row["threat"],
                row["signal_seed"],
                row["noise_permutation"],
                row["outlier_geometry"],
            )
            reductions[key].append(
                1.0 - row["aggregate_error"] / max(row["raw_error"], 1e-15)
            )
        worst_reduction = min(
            (_finite_mean(values) for values in reductions.values()),
            default=float("nan"),
        )
        guard_is_raw = guard == "raw"
        guard_is_positive = not guard_is_raw and float(guard) > 0.0
        output_checks = {
            "positive_guard": guard_is_positive,
            "clean_cost": clean_ratio - 1.0
            <= float(gates["guarded_clean_error_increase_max"]),
            "attack_reduction": worst_reduction
            >= float(gates["guarded_attack_error_reduction_min"]),
        }
        summary = {
            "family": family,
            "robust_reference": reference,
            "trust_profile": trust,
            "guard_fraction": guard,
            "clean_error_ratio_to_raw": clean_ratio,
            "worst_attacked_error_reduction_vs_raw": worst_reduction,
            "candidate_gates_pass": candidate_pass[(family, reference, trust)],
            "output_gate_checks": output_checks,
            "passes_full_gates": candidate_pass[(family, reference, trust)]
            and all(output_checks.values()),
        }
        summaries.append(summary)
    return summaries


def _fmt(value: float) -> str:
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):.3f}"


def _write_report(
    path: Path,
    candidate_summaries: list[dict[str, Any]],
    guard_summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    candidate_survivors = [
        row for row in candidate_summaries if row["passes_candidate_gates"]
    ]
    full_survivors = [row for row in guard_summaries if row["passes_full_gates"]]
    lines = [
        "# Stage 14B — Référence, confiance et sortie gardée",
        "",
        "## Verdict",
        "",
        f"- Profils score/référence/confiance : **{len(candidate_summaries)}**.",
        f"- Profils franchissant leurs gates : **{len(candidate_survivors)}**.",
        f"- Sorties gardées franchissant tous les gates : **{len(full_survivors)}**.",
        "",
        "## Profils score, référence et confiance",
        "",
        "| Score | F robuste | Confiance | Corr. bruit | Corr. propre | Recall | Masse outliers | Masse byz. pire | Rappel attaqué pire | Erreur F / dispersion | Passe |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in candidate_summaries:
        lines.append(
            "| {family} | {reference} | {trust} | {noise} | {clean} | {recall} | {mass} | {byz} | {attacked} | {ferror} | {passed} |".format(
                family=row["family"],
                reference=row["robust_reference"],
                trust=row["trust_profile"],
                noise=_fmt(row["null_abs_score_noise_correlation_max"]),
                clean=_fmt(row["clean_geometry_correlation_mean"]),
                recall=_fmt(row["honest_outlier_recall_at_5_mean"]),
                mass=_fmt(row["honest_outlier_weight_mass_mean"]),
                byz=_fmt(row["attacked_byzantine_weight_mass_worst_group"]),
                attacked=_fmt(row["attacked_global_recall_worst_group"]),
                ferror=_fmt(row["robust_reference_error_ratio_worst_group"]),
                passed="oui" if row["passes_candidate_gates"] else "non",
            )
        )
    lines.extend(
        [
            "",
            "## Sorties gardées admissibles",
            "",
        ]
    )
    if full_survivors:
        lines.extend(
            [
                "| Score | F robuste | Confiance | Rayon / U | Coût propre | Réduction attaquée pire |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in full_survivors:
            lines.append(
                "| {family} | {reference} | {trust} | {guard} | {cost} | {reduction} |".format(
                    family=row["family"],
                    reference=row["robust_reference"],
                    trust=row["trust_profile"],
                    guard=row["guard_fraction"],
                    cost=_fmt(row["clean_error_ratio_to_raw"] - 1.0),
                    reduction=_fmt(row["worst_attacked_error_reduction_vs_raw"]),
                )
            )
    else:
        lines.append(
            "Aucune sortie ne satisfait simultanément les critères de score, de confiance, de référence et de garde."
        )
    lines.extend(
        [
            "",
            "## Interprétation",
            "",
            "Le score corrige la confusion entre géométrie honnête et niveau de bruit DP. La référence robuste estime un centre. La confiance cherche des incohérences directionnelles ou de collusion. Enfin, la garde transmet une borne conditionnelle sur la référence à la sortie. Ces quatre propriétés sont auditées séparément afin qu'une bonne moyenne ne masque pas l'échec d'un composant.",
            "",
            "Même avec une référence parfaite, cinq Byzantins de score un face à vingt honnêtes de score nul reçoivent une masse `5 exp(alpha) / (20 + 5 exp(alpha))`, soit environ `0,343` à la borne alpha de ce protocole. Une meilleure référence ne suffit donc pas si elle sert seulement à produire un score radial croissant ; elle doit aussi intervenir dans la confiance ou contraindre la sortie.",
            "",
            "Une attaque alignée et jitterée est incluse pour rappeler la limite d'identifiabilité : un message adversarial qui ressemble à un honest outlier ne peut pas être détecté universellement par une règle purement géométrique.",
            "",
            "Si aucun profil ne passe, le protocole interdit le transfert Fashion-MNIST et oriente la suite vers une référence temporelle construite à partir d'uploads privés passés, plutôt que vers un réglage de score choisi après lecture de l'accuracy.",
            "",
            "## Traçabilité",
            "",
            f"- Détails nuls : `{output_dir / 'null_detail.csv'}`",
            f"- Détails signal/attaques : `{output_dir / 'signal_detail.csv'}`",
            f"- Détails des sorties : `{output_dir / 'output_detail.csv'}`",
            f"- Synthèse candidats : `{output_dir / 'candidate_summary.csv'}`",
            f"- Synthèse gardes : `{output_dir / 'guard_summary.csv'}`",
            "",
            "Aucune accuracy n'a été calculée pour sélectionner les profils.",
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
    randomness = config["randomness"]
    n = int(cohort["num_clients"])
    dimension = int(cohort["ambient_dimension"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    levels = [float(value) for value in cohort["public_noise_scales"]]
    permutations = [str(value) for value in cohort["noise_permutations"]]
    references = [str(value) for value in config["robust_reference"]["candidates"]]
    families = [str(value) for value in config["score"]["families"]]
    trust_profiles = [str(value) for value in config["trust"]["profiles"]]
    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    holdout_draws = int(
        null_holdout_draws_override or randomness["null_holdout_draws"]
    )
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Stage 14B requires >=20 null draws and >=1 signal draw")

    alpha = math.log(
        float(config["aggregation"]["kappa_w"]) * (n - 1)
        / (n - float(config["aggregation"]["kappa_w"]))
    )
    calibration_energy: dict[str, torch.Tensor] = {}
    calibration_separation: dict[str, torch.Tensor] = {}
    null_rows: list[dict[str, Any]] = []
    for permutation_index, permutation in enumerate(permutations):
        scales = _noise_scales(levels, permutation, n)
        variances = (float(cohort["noise_std"]) * scales).square()
        calibration = _null_vectors(
            scales,
            draws=calibration_draws,
            dimension=dimension,
            noise_std=float(cohort["noise_std"]),
            server_clip_norm=server_clip,
            seed=int(randomness["calibration_seed"]) + 10_000 * permutation_index,
        )
        holdout = _null_vectors(
            scales,
            draws=holdout_draws,
            dimension=dimension,
            noise_std=float(cohort["noise_std"]),
            server_clip_norm=server_clip,
            seed=int(randomness["null_holdout_seed"]) + 10_000 * permutation_index,
        )
        energies = _energy_matrix(calibration, config)
        separations = _separation_matrix(calibration, variances)
        calibration_energy[permutation] = energies
        calibration_separation[permutation] = separations
        holdout_energy = _energy_matrix(holdout, config)
        novelty_by_family = {
            family: torch.stack(
                [
                    _score(row, energies, family, config)
                    for row in holdout_energy
                ]
            )
            for family in families
        }
        for reference_name in references:
            trust_rows = []
            for vectors in holdout:
                robust_reference = _reference(vectors, reference_name, config)
                trust_rows.append(
                    _dual_trust(
                        vectors,
                        robust_reference,
                        variances,
                        separations,
                        config,
                    )
                )
            dual_trust = torch.stack(trust_rows)
            for family in families:
                for profile in trust_profiles:
                    combined = torch.stack(
                        [
                            _compose_score(
                                novelty_by_family[family][index],
                                dual_trust[index],
                                profile,
                                config,
                            )
                            for index in range(holdout_draws)
                        ]
                    )
                    null_rows.append(
                        {
                            "family": family,
                            "robust_reference": reference_name,
                            "trust_profile": profile,
                            "noise_permutation": permutation,
                            **_null_metrics(combined, scales),
                        }
                    )

    signal_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    threats = [str(value) for value in cohort["threats"]]
    geometries = [str(value) for value in cohort["outlier_geometries"]]
    guard_values = config["aggregation"]["guard_radius_fractions"]
    for signal_seed in [int(value) for value in randomness["signal_seeds"]]:
        for geometry_index, outlier_geometry in enumerate(geometries):
            for permutation_index, permutation in enumerate(permutations):
                scales = _noise_scales(levels, permutation, n)
                variances = (float(cohort["noise_std"]) * scales).square()
                for threat_index, threat in enumerate(threats):
                    n_honest = n if threat == "none" else n - int(
                        cohort["num_byzantine"]
                    )
                    clean, outliers = _clean_honest_vectors(
                        signal_seed,
                        n_honest=n_honest,
                        dimension=dimension,
                        num_outliers=int(cohort["honest_outliers"]),
                        outlier_geometry=outlier_geometry,
                    )
                    clean_norms = torch.linalg.vector_norm(clean, dim=1, keepdim=True)
                    clean = clean * (server_clip / clean_norms.clamp_min(1e-12)).clamp(
                        max=1.0
                    )
                    clean_distances = _energies(clean, config).sqrt()
                    target = clean.mean(dim=0)
                    dispersion = torch.sqrt(
                        torch.linalg.vector_norm(clean - target, dim=1).square().mean()
                    ).clamp_min(1e-12)
                    for draw in range(signal_draws):
                        run_seed = (
                            signal_seed * 1_000_000
                            + geometry_index * 100_000
                            + permutation_index * 10_000
                            + threat_index * 1_000
                            + draw
                        )
                        observed, byzantine = _observed(
                            clean,
                            scales,
                            threat=threat,
                            config=config,
                            seed=run_seed,
                        )
                        observed_energy = _energies(observed, config)
                        novelty_by_family = {
                            family: _score(
                                observed_energy,
                                calibration_energy[permutation],
                                family,
                                config,
                            )
                            for family in families
                        }
                        for reference_name in references:
                            robust_reference = _reference(
                                observed, reference_name, config
                            )
                            reference_error_ratio = float(
                                torch.linalg.vector_norm(robust_reference - target)
                                / dispersion
                            )
                            dual_trust = _dual_trust(
                                observed,
                                robust_reference,
                                variances,
                                calibration_separation[permutation],
                                config,
                            )
                            for family in families:
                                for profile in trust_profiles:
                                    combined = _compose_score(
                                        novelty_by_family[family],
                                        dual_trust,
                                        profile,
                                        config,
                                    )
                                    metrics, weights = _signal_metrics(
                                        combined,
                                        clean_distances,
                                        outliers,
                                        byzantine,
                                        alpha=alpha,
                                    )
                                    common = {
                                        "family": family,
                                        "robust_reference": reference_name,
                                        "trust_profile": profile,
                                        "signal_seed": signal_seed,
                                        "draw": draw,
                                        "noise_permutation": permutation,
                                        "outlier_geometry": outlier_geometry,
                                        "threat": threat,
                                        "reference_error_over_honest_dispersion": reference_error_ratio,
                                    }
                                    signal_rows.append({**common, **metrics})
                                    raw = (weights[:, None] * observed).sum(dim=0)
                                    raw_error = float(
                                        torch.linalg.vector_norm(raw - target)
                                    )
                                    for guard in guard_values:
                                        if guard == "raw":
                                            aggregate = raw
                                            label: str | float = "raw"
                                        else:
                                            fraction = float(guard)
                                            aggregate = guarded_aggregate(
                                                raw,
                                                robust_reference,
                                                radius=fraction * server_clip,
                                            )
                                            label = fraction
                                        output_rows.append(
                                            {
                                                **common,
                                                "guard_fraction": label,
                                                "raw_error": raw_error,
                                                "aggregate_error": float(
                                                    torch.linalg.vector_norm(
                                                        aggregate - target
                                                    )
                                                ),
                                            }
                                        )

    candidate_summaries = _summarize_candidates(null_rows, signal_rows, config)
    guard_summaries = _summarize_guards(
        output_rows, candidate_summaries, config
    )
    _write_csv(output_dir / "null_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    _write_csv(output_dir / "output_detail.csv", output_rows)
    flat_candidates = [
        {key: value for key, value in row.items() if key != "gate_checks"}
        for row in candidate_summaries
    ]
    flat_guards = [
        {key: value for key, value in row.items() if key != "output_gate_checks"}
        for row in guard_summaries
    ]
    _write_csv(output_dir / "candidate_summary.csv", flat_candidates)
    _write_csv(output_dir / "guard_summary.csv", flat_guards)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "candidate_summaries": candidate_summaries,
                "guard_summaries": guard_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_path,
        candidate_summaries,
        guard_summaries,
        output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "candidate_profiles": len(candidate_summaries),
                "candidate_survivors": sum(
                    bool(row["passes_candidate_gates"])
                    for row in candidate_summaries
                ),
                "guard_survivors": sum(
                    bool(row["passes_full_gates"]) for row in guard_summaries
                ),
                "report": str(report_path),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dt_ldp_far/stage14b_reference_trust_guard.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/dt_ldp_far/score_quality_stage14b_reference_trust_guard_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage14B_Reference_Trust_Guard_Audit.md",
    )
    parser.add_argument("--calibration-draws", type=int)
    parser.add_argument("--null-holdout-draws", type=int)
    parser.add_argument("--signal-draws", type=int)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        calibration_draws_override=args.calibration_draws,
        null_holdout_draws_override=args.null_holdout_draws,
        signal_draws_override=args.signal_draws,
    )


if __name__ == "__main__":
    main()
