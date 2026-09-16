#!/usr/bin/env python3
"""Stage 15: frozen cross-fitted novelty and peer-support audit.

This vector-level audit answers two distinct questions without training a
model or inspecting final accuracy:

1. Does the Stage-14A cross-fitted covariance-corrected novelty retain honest
   geometric signal while remaining neutral to public DP-noise tiers?
2. Can a directional peer-support condition reduce Byzantine mass without
   erasing honest outliers, and with which robust reference?

The score and robust-reference gates are reported separately.  This prevents
a good score from hiding a poor reference, or vice versa.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
    ranked_directional_peer_support_scores,
    separate_novelty_trust_scores,
)
from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (  # noqa: E402
    _clean_honest_vectors,
    _corr,
    _finite_mean,
    _ks_distance,
    _noise_scales,
)
from scripts.run_dt_ldp_far_stage14b_reference_trust_audit import (  # noqa: E402
    _energy_matrix,
    _independence_trust,
    _observed,
    _reference,
    _separation_matrix,
)


BASE_PROFILE = "crossfit_covariance"
SUPPORTED_PROFILE = "crossfit_covariance_plus_peer_support"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _novelty_scores(
    observed_energy: torch.Tensor,
    calibration_energy: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    settings = config["novelty"]
    scores, _ = calibrated_null_energy_scores(
        observed_energy,
        calibration_energy,
        mode="moment",
        z_clip=float(settings["moment_z_clip"]),
        individual_calibration_weight=float(settings["individual_calibration_weight"]),
        variance_ridge=float(settings["variance_ridge"]),
    )
    return scores.double()


def _joint_support(
    vectors: torch.Tensor,
    reference: torch.Tensor,
    variances: torch.Tensor,
    separation_calibration: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    settings = config["support"]
    directional, diagnostics = ranked_directional_peer_support_scores(
        vectors,
        reference,
        assumed_byzantine=int(config["cohort"]["num_byzantine"]),
        reference_reject_cosine=float(settings["reference_reject_cosine"]),
        reference_full_support_cosine=float(settings["reference_full_support_cosine"]),
        peer_reject_cosine=float(settings["peer_reject_cosine"]),
        peer_full_support_cosine=float(settings["peer_full_support_cosine"]),
        extra_honest_supporters=int(settings["extra_honest_supporters"]),
    )
    independence = _independence_trust(
        vectors,
        variances,
        separation_calibration,
        config={
            "trust": {
                "independence_low_null_quantile": float(
                    settings["independence_low_null_quantile"]
                ),
                "independence_full_trust_null_quantile": float(
                    settings["independence_full_trust_null_quantile"]
                ),
                "trust_floor": float(settings["independence_trust_floor"]),
            }
        },
    ).double()
    joint = torch.minimum(directional.double(), independence)
    diagnostics.update(
        {
            "noise_score_independence_support_min": float(independence.min()),
            "noise_score_independence_support_mean": float(independence.mean()),
            "noise_score_independence_support_max": float(independence.max()),
            "noise_score_final_support_min": float(joint.min()),
            "noise_score_final_support_mean": float(joint.mean()),
            "noise_score_final_support_max": float(joint.max()),
        }
    )
    return joint, diagnostics


def _compose(
    novelty: torch.Tensor,
    support: torch.Tensor,
    profile: str,
    config: dict[str, Any],
) -> torch.Tensor:
    if profile == BASE_PROFILE:
        return novelty
    if profile == SUPPORTED_PROFILE:
        combined, _ = separate_novelty_trust_scores(
            novelty,
            support,
            trust_fraction=float(config["support"]["trust_logit_fraction"]),
        )
        return combined.double()
    raise ValueError(f"Unknown Stage-15 profile {profile!r}")


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
        "null_score_mean": float(scores.mean()),
    }


def _signal_metrics(
    *,
    scores: torch.Tensor,
    support: torch.Tensor,
    clean_distances: torch.Tensor,
    honest_outliers: torch.Tensor,
    byzantine: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
) -> dict[str, float]:
    honest = ~byzantine
    honest_scores = scores[honest]
    predicted_honest = torch.topk(honest_scores, min(5, honest_scores.numel())).indices
    full_outliers = torch.zeros_like(byzantine)
    full_outliers[: len(honest_outliers)] = honest_outliers
    predicted_global = torch.topk(scores, min(5, scores.numel())).indices
    weights = torch.softmax(float(alpha) * scores, dim=0)
    aggregate = (weights[:, None] * observed).sum(dim=0)
    byzantine_support = support[byzantine]
    return {
        "clean_geometry_correlation": _corr(honest_scores, clean_distances),
        "honest_outlier_recall_at_5": float(
            honest_outliers[predicted_honest].double().mean()
        ),
        "global_honest_outlier_recall_at_5": float(
            full_outliers[predicted_global].double().mean()
        ),
        "honest_outlier_weight_mass": float(weights[full_outliers].sum()),
        "byzantine_weight_mass": float(weights[byzantine].sum()),
        "max_individual_weight": float(weights.max()),
        "weight_concentration": float(scores.numel() * weights.square().sum()),
        "support_honest_mean": float(support[honest].mean()),
        "support_byzantine_mean": (
            float(byzantine_support.mean())
            if byzantine_support.numel()
            else float("nan")
        ),
        "support_honest_minus_byzantine": (
            float(support[honest].mean() - byzantine_support.mean())
            if byzantine_support.numel()
            else float("nan")
        ),
        "aggregate_error": float(torch.linalg.vector_norm(aggregate - target)),
    }


def _group_extreme(rows: list[dict[str, Any]], field: str, *, minimum: bool) -> float:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            continue
        key = (
            row["threat"],
            row["signal_seed"],
            row["noise_permutation"],
            row["outlier_geometry"],
        )
        grouped[key].append(value)
    means = [_finite_mean(values) for values in grouped.values()]
    if not means:
        return float("nan")
    return min(means) if minimum else max(means)


def _summaries(
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config["gates"]
    keys = sorted({(row["profile"], row["robust_reference"]) for row in signal_rows})
    output: list[dict[str, Any]] = []
    for profile, reference in keys:
        null = [
            row
            for row in null_rows
            if row["profile"] == profile and row["robust_reference"] == reference
        ]
        rows = [
            row
            for row in signal_rows
            if row["profile"] == profile and row["robust_reference"] == reference
        ]
        clean = [row for row in rows if row["threat"] == "none"]
        attacked = [row for row in rows if row["threat"] != "none"]
        baseline = [
            row
            for row in signal_rows
            if row["profile"] == BASE_PROFILE
            and row["robust_reference"] == reference
            and row["threat"] == "none"
        ]
        clean_recall = _finite_mean(
            [float(row["honest_outlier_recall_at_5"]) for row in clean]
        )
        clean_mass = _finite_mean(
            [float(row["honest_outlier_weight_mass"]) for row in clean]
        )
        baseline_recall = _finite_mean(
            [float(row["honest_outlier_recall_at_5"]) for row in baseline]
        )
        baseline_mass = _finite_mean(
            [float(row["honest_outlier_weight_mass"]) for row in baseline]
        )
        summary: dict[str, Any] = {
            "profile": profile,
            "robust_reference": reference,
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
                [float(row["clean_geometry_correlation"]) for row in clean]
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
            "support_honest_minus_byzantine_worst_group": _group_extreme(
                attacked, "support_honest_minus_byzantine", minimum=True
            ),
            "support_clean_recall_loss": max(0.0, baseline_recall - clean_recall),
            "support_clean_outlier_mass_loss": max(0.0, baseline_mass - clean_mass),
            "robust_reference_error_over_honest_dispersion_worst_group": _group_extreme(
                attacked,
                "reference_error_over_honest_dispersion",
                minimum=False,
            ),
            "attacked_aggregate_error_worst_group": _group_extreme(
                attacked, "aggregate_error", minimum=False
            ),
        }
        score_checks = {
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
        }
        if profile == SUPPORTED_PROFILE:
            score_checks.update(
                {
                    "support_separation": summary[
                        "support_honest_minus_byzantine_worst_group"
                    ]
                    >= float(gates["support_honest_minus_byzantine_min"]),
                    "support_recall_cost": summary["support_clean_recall_loss"]
                    <= float(gates["support_clean_recall_loss_max"]),
                    "support_mass_cost": summary["support_clean_outlier_mass_loss"]
                    <= float(gates["support_clean_outlier_mass_loss_max"]),
                }
            )
        reference_check = summary[
            "robust_reference_error_over_honest_dispersion_worst_group"
        ] <= float(gates["robust_reference_error_over_honest_dispersion_max"])
        summary["score_gate_checks"] = score_checks
        summary["reference_gate_check"] = reference_check
        summary["passes_score_gates"] = all(score_checks.values())
        summary["passes_full_gates"] = bool(
            summary["passes_score_gates"] and reference_check
        )
        output.append(summary)
    return output


def _attack_summaries(signal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attacks without hiding the worst paired group."""

    keys = sorted(
        {
            (row["profile"], row["robust_reference"], row["threat"])
            for row in signal_rows
            if row["threat"] != "none"
        }
    )
    output = []
    for profile, reference, threat in keys:
        rows = [
            row
            for row in signal_rows
            if row["profile"] == profile
            and row["robust_reference"] == reference
            and row["threat"] == threat
        ]
        output.append(
            {
                "profile": profile,
                "robust_reference": reference,
                "threat": threat,
                "byzantine_weight_mass_mean": _finite_mean(
                    [float(row["byzantine_weight_mass"]) for row in rows]
                ),
                "byzantine_weight_mass_worst_group": _group_extreme(
                    rows, "byzantine_weight_mass", minimum=False
                ),
                "global_honest_outlier_recall_at_5_mean": _finite_mean(
                    [float(row["global_honest_outlier_recall_at_5"]) for row in rows]
                ),
                "global_honest_outlier_recall_at_5_worst_group": _group_extreme(
                    rows, "global_honest_outlier_recall_at_5", minimum=True
                ),
                "support_honest_minus_byzantine_worst_group": _group_extreme(
                    rows, "support_honest_minus_byzantine", minimum=True
                ),
                "aggregate_error_mean": _finite_mean(
                    [float(row["aggregate_error"]) for row in rows]
                ),
            }
        )
    return output


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    attack_summaries: list[dict[str, Any]],
) -> None:
    score_survivors = [row for row in summaries if row["passes_score_gates"]]
    full_survivors = [row for row in summaries if row["passes_full_gates"]]
    lines = [
        "# Stage 15 — Score cross-fitted et soutien directionnel inter-clients",
        "",
        "## Verdict",
        "",
        f"- Profils score/référence audités : **{len(summaries)}**.",
        f"- Profils franchissant les gates du score : **{len(score_survivors)}**.",
        f"- Profils franchissant aussi le gate de référence : **{len(full_survivors)}**.",
        "",
        "Aucune accuracy n'a été calculée. La décision repose uniquement sur les "
        "propriétés du score, des poids, des attaques et de la référence.",
        "",
        "## Deux candidats construits",
        "",
        "1. **Cross-fitted covariance** : énergie de l'upload autour de "
        "`F_CC,-i`, calibrée client par client sous la chaîne nulle complète "
        "bruit DP → clipping serveur → référence leave-one-out. C'est la "
        "formalisation propre du survivant Stage 14A, pas un renommage présenté "
        "comme une nouveauté.",
        "2. **Cross-fitted covariance + peer support** : le même signal de "
        "nouveauté reçoit un budget de logit séparé pour le soutien directionnel. "
        "Un message doit être aligné avec la référence robuste et soutenu par au "
        "moins `f+1` pairs; la confiance d'indépendance pénalise en plus les copies "
        "collusives trop proches.",
        "",
        "Le second candidat n'est pas un certificat d'identification byzantine. "
        "Une attaque alignée avec assez d'honnêtes peut être indiscernable d'un "
        "sous-groupe honnête utile; l'attaque adaptative du protocole teste "
        "explicitement cette limite.",
        "",
        "## Résultats agrégés",
        "",
        "| Score | F | Corr. bruit | Corr. propre | Recall propre | Masse outliers | Masse byz. pire | Recall attaqué pire | Soutien H−B pire | Erreur F/disp. pire | Gate score | Gate complet |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            "| {profile} | {reference} | {noise} | {clean} | {recall} | {mass} | {byz} | {attacked} | {support} | {ferror} | {score_pass} | {full_pass} |".format(
                profile=row["profile"],
                reference=row["robust_reference"].upper(),
                noise=_fmt(row["null_abs_score_noise_correlation_max"]),
                clean=_fmt(row["clean_geometry_correlation_mean"]),
                recall=_fmt(row["honest_outlier_recall_at_5_mean"]),
                mass=_fmt(row["honest_outlier_weight_mass_mean"]),
                byz=_fmt(row["attacked_byzantine_weight_mass_worst_group"]),
                attacked=_fmt(row["attacked_global_recall_worst_group"]),
                support=_fmt(row["support_honest_minus_byzantine_worst_group"]),
                ferror=_fmt(
                    row["robust_reference_error_over_honest_dispersion_worst_group"]
                ),
                score_pass="oui" if row["passes_score_gates"] else "non",
                full_pass="oui" if row["passes_full_gates"] else "non",
            )
        )
    supported = [row for row in summaries if row["profile"] == SUPPORTED_PROFILE]
    lines.extend(
        [
            "",
            "## Critères qui bloquent la promotion",
            "",
            "| F | Gates du score non satisfaits | Gate de référence |",
            "|---|---|---|",
        ]
    )
    for row in supported:
        failed = [
            name for name, passed in row["score_gate_checks"].items() if not passed
        ]
        lines.append(
            "| {reference} | {failed} | {reference_gate} |".format(
                reference=row["robust_reference"].upper(),
                failed=", ".join(failed) if failed else "aucun",
                reference_gate="oui" if row["reference_gate_check"] else "non",
            )
        )

    by_reference = {row["robust_reference"]: row for row in supported}
    if "fcc" in by_reference:
        supported_fcc = by_reference["fcc"]
        base_fcc = next(
            row
            for row in summaries
            if row["profile"] == BASE_PROFILE and row["robust_reference"] == "fcc"
        )
        mass_drop = (
            base_fcc["attacked_byzantine_weight_mass_worst_group"]
            - supported_fcc["attacked_byzantine_weight_mass_worst_group"]
        )
        error_drop = 1.0 - (
            supported_fcc["attacked_aggregate_error_worst_group"]
            / base_fcc["attacked_aggregate_error_worst_group"]
        )
        lines.extend(
            [
                "",
                "## Effet causal du canal de soutien, à géométrie identique",
                "",
                (
                    "Avec FCC, ajouter le soutien fait passer la pire masse "
                    f"byzantine de `{base_fcc['attacked_byzantine_weight_mass_worst_group']:.3f}` "
                    f"à `{supported_fcc['attacked_byzantine_weight_mass_worst_group']:.3f}` "
                    f"(baisse absolue `{mass_drop:.3f}`). L'erreur agrégée du "
                    f"pire groupe baisse de `{100.0 * error_drop:.1f} %`."
                ),
                "",
                (
                    "Ce gain coûte "
                    f"`{supported_fcc['support_clean_recall_loss']:.3f}` de rappel "
                    "propre et "
                    f"`{supported_fcc['support_clean_outlier_mass_loss']:.3f}` "
                    "de masse attribuée aux honest outliers. Le compromis est "
                    "réel, mais il ne suffit pas à franchir tous les gates."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Ventilation du candidat soutenu par attaque",
            "",
            "Les colonnes « pire » prennent le groupe apparié le moins favorable "
            "sur seed, permutation de bruit et géométrie, et non une moyenne qui "
            "pourrait masquer un échec.",
            "",
            "| F | Attaque | Masse byz. moyenne | Masse byz. pire | Recall moyen | Recall pire | Soutien H−B pire |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in attack_summaries:
        if row["profile"] != SUPPORTED_PROFILE:
            continue
        lines.append(
            "| {reference} | {threat} | {mass_mean} | {mass_worst} | {recall_mean} | {recall_worst} | {support} |".format(
                reference=row["robust_reference"].upper(),
                threat=row["threat"],
                mass_mean=_fmt(row["byzantine_weight_mass_mean"]),
                mass_worst=_fmt(row["byzantine_weight_mass_worst_group"]),
                recall_mean=_fmt(row["global_honest_outlier_recall_at_5_mean"]),
                recall_worst=_fmt(row["global_honest_outlier_recall_at_5_worst_group"]),
                support=_fmt(row["support_honest_minus_byzantine_worst_group"]),
            )
        )
    reference_rank = sorted(
        supported,
        key=lambda row: row[
            "robust_reference_error_over_honest_dispersion_worst_group"
        ],
    )
    if reference_rank:
        reference_values = ", ".join(
            "{name} `{value:.3f}`".format(
                name=row["robust_reference"].upper(),
                value=row["robust_reference_error_over_honest_dispersion_worst_group"],
            )
            for row in reference_rank
        )
        reference_verdict = (
            "Au moins une respecte la borne préenregistrée."
            if any(row["reference_gate_check"] for row in reference_rank)
            else "Aucune ne respecte toutefois la borne préenregistrée `1,00`."
        )
        lines.extend(
            [
                "",
                "## Ce que S15 établit sur le choix de F",
                "",
                (
                    "Classement par pire erreur de référence divisée par la "
                    f"dispersion honnête : {reference_values}. {reference_verdict}"
                ),
                "",
                "Les métriques de poids sont identiques à la précision affichée "
                "pour les trois références : dans ce régime, le minimum entre "
                "soutien directionnel, soutien des pairs et indépendance est "
                "piloté par les deux derniers canaux. Le screen ne permet donc "
                "pas d'affirmer que FCC est universellement meilleure; il montre "
                "seulement qu'elle a la plus faible erreur de référence parmi "
                "les trois candidats testés.",
            ]
        )
    lines.extend(
        [
            "",
            "## Règle de décision",
            "",
            "Le candidat soutenu n'est promu que s'il réduit la masse byzantine "
            "jusqu'à sa masse uniforme `5/25 = 0,20`, conserve le rappel des "
            "honest outliers, sépare leur soutien de celui des Byzantins et "
            "respecte le cap individuel `2/25 = 0,08` pour chaque attaque et "
            "chaque géométrie. Le gate de référence exige en plus que l'erreur "
            "maximale de `F` ne dépasse pas la dispersion honnête.",
            "",
            (
                "Au moins un profil complet est admissible pour un screen "
                "Fashion-MNIST préenregistré."
                if full_survivors
                else "Aucun profil complet n'est admissible : le protocole "
                "interdit de lancer Fashion-MNIST pour choisir après coup le "
                "meilleur réglage."
            ),
            "",
            "## Traçabilité",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Détails nuls : `{(output_dir / 'null_detail.csv').resolve()}`",
            f"- Détails signal/attaques : `{(output_dir / 'signal_detail.csv').resolve()}`",
            f"- Synthèse : `{(output_dir / 'summary.csv').resolve()}`",
            f"- Synthèse par attaque : `{(output_dir / 'attack_summary.csv').resolve()}`",
            f"- JSON complet : `{(output_dir / 'summary.json').resolve()}`",
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
    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    holdout_draws = int(null_holdout_draws_override or randomness["null_holdout_draws"])
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Stage 15 requires >=20 null draws and >=1 signal draw")
    if int(cohort["honest_outliers"]) > n - int(cohort["num_byzantine"]):
        raise ValueError("honest_outliers exceed the attacked honest cohort")

    server_clip = 0.42
    # Stage 15 intentionally reuses the frozen Stage-14 geometry.
    if "aggregation" in config:
        server_clip = float(config["aggregation"]["server_clip_norm"])
    else:
        server_clip = 0.42
    references = [str(value) for value in config["robust_reference"]["candidates"]]
    profiles = [str(value) for value in config["support"]["profiles"]]
    permutations = [str(value) for value in cohort["noise_permutations"]]
    levels = [float(value) for value in cohort["public_noise_scales"]]
    alpha = math.log(
        float(config["weights"]["kappa_w"])
        * (n - 1)
        / (n - float(config["weights"]["kappa_w"]))
    )

    calibration_energy: dict[str, torch.Tensor] = {}
    calibration_separation: dict[str, torch.Tensor] = {}
    null_rows: list[dict[str, Any]] = []
    for permutation_index, permutation in enumerate(permutations):
        scales = _noise_scales(levels, permutation, n)
        variances = (float(cohort["noise_std"]) * scales).square()
        # The Stage-14 helpers use aggregation.server_clip_norm and score fields.
        helper_config = {
            **config,
            "aggregation": {"server_clip_norm": server_clip},
            "score": {
                "dimension": dimension,
                "fcc_radius": float(config["novelty"]["fcc_radius"]),
            },
        }
        generator = torch.Generator(device="cpu").manual_seed(
            int(randomness["calibration_seed"]) + 10_000 * permutation_index
        )
        std = float(cohort["noise_std"]) * scales
        calibration = (
            torch.randn(
                calibration_draws,
                n,
                dimension,
                generator=generator,
                dtype=torch.float64,
            )
            * std[None, :, None]
        )
        norms = torch.linalg.vector_norm(calibration, dim=2, keepdim=True)
        calibration = calibration * (server_clip / norms.clamp_min(1e-12)).clamp(
            max=1.0
        )
        holdout_generator = torch.Generator(device="cpu").manual_seed(
            int(randomness["null_holdout_seed"]) + 10_000 * permutation_index
        )
        holdout = (
            torch.randn(
                holdout_draws,
                n,
                dimension,
                generator=holdout_generator,
                dtype=torch.float64,
            )
            * std[None, :, None]
        )
        norms = torch.linalg.vector_norm(holdout, dim=2, keepdim=True)
        holdout = holdout * (server_clip / norms.clamp_min(1e-12)).clamp(max=1.0)
        calibration_energy[permutation] = _energy_matrix(calibration, helper_config)
        calibration_separation[permutation] = _separation_matrix(calibration, variances)
        holdout_novelty = torch.stack(
            [
                _novelty_scores(
                    energy,
                    calibration_energy[permutation],
                    config,
                )
                for energy in _energy_matrix(holdout, helper_config)
            ]
        )
        for reference_name in references:
            supports = []
            for vectors in holdout:
                reference = _reference(vectors, reference_name, helper_config)
                support, _ = _joint_support(
                    vectors,
                    reference,
                    variances,
                    calibration_separation[permutation],
                    config,
                )
                supports.append(support)
            support_matrix = torch.stack(supports)
            for profile in profiles:
                combined = torch.stack(
                    [
                        _compose(
                            holdout_novelty[index],
                            support_matrix[index],
                            profile,
                            config,
                        )
                        for index in range(holdout_draws)
                    ]
                )
                null_rows.append(
                    {
                        "profile": profile,
                        "robust_reference": reference_name,
                        "noise_permutation": permutation,
                        **_null_metrics(combined, scales),
                    }
                )

    signal_rows: list[dict[str, Any]] = []
    geometries = [str(value) for value in cohort["outlier_geometries"]]
    threats = [str(value) for value in cohort["threats"]]
    helper_config = {
        **config,
        "aggregation": {"server_clip_norm": server_clip},
        "score": {
            "dimension": dimension,
            "fcc_radius": float(config["novelty"]["fcc_radius"]),
        },
    }
    for signal_seed in [int(value) for value in randomness["signal_seeds"]]:
        for geometry_index, geometry in enumerate(geometries):
            for permutation_index, permutation in enumerate(permutations):
                scales = _noise_scales(levels, permutation, n)
                variances = (float(cohort["noise_std"]) * scales).square()
                for threat_index, threat in enumerate(threats):
                    n_honest = (
                        n if threat == "none" else n - int(cohort["num_byzantine"])
                    )
                    clean, outliers = _clean_honest_vectors(
                        signal_seed,
                        n_honest=n_honest,
                        dimension=dimension,
                        num_outliers=int(cohort["honest_outliers"]),
                        outlier_geometry=geometry,
                    )
                    clean_norms = torch.linalg.vector_norm(clean, dim=1, keepdim=True)
                    clean = clean * (server_clip / clean_norms.clamp_min(1e-12)).clamp(
                        max=1.0
                    )
                    clean_distances = _energy_matrix(clean[None, :, :], helper_config)[
                        0
                    ].sqrt()
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
                            config=helper_config,
                            seed=run_seed,
                        )
                        observed_energy = _energy_matrix(
                            observed[None, :, :], helper_config
                        )[0]
                        novelty = _novelty_scores(
                            observed_energy,
                            calibration_energy[permutation],
                            config,
                        )
                        for reference_name in references:
                            reference = _reference(
                                observed, reference_name, helper_config
                            )
                            support, support_diagnostics = _joint_support(
                                observed,
                                reference,
                                variances,
                                calibration_separation[permutation],
                                config,
                            )
                            reference_error_ratio = float(
                                torch.linalg.vector_norm(reference - target)
                                / dispersion
                            )
                            for profile in profiles:
                                scores = _compose(novelty, support, profile, config)
                                common = {
                                    "profile": profile,
                                    "robust_reference": reference_name,
                                    "signal_seed": signal_seed,
                                    "draw": draw,
                                    "noise_permutation": permutation,
                                    "outlier_geometry": geometry,
                                    "threat": threat,
                                    "reference_error_over_honest_dispersion": reference_error_ratio,
                                    "peer_ranked_cosine_mean": float(
                                        support_diagnostics[
                                            "noise_score_peer_ranked_cosine_mean"
                                        ]
                                    ),
                                }
                                signal_rows.append(
                                    {
                                        **common,
                                        **_signal_metrics(
                                            scores=scores,
                                            support=support,
                                            clean_distances=clean_distances,
                                            honest_outliers=outliers,
                                            byzantine=byzantine,
                                            observed=observed,
                                            target=target,
                                            alpha=alpha,
                                        ),
                                    }
                                )

    summaries = _summaries(null_rows, signal_rows, config)
    attack_summaries = _attack_summaries(signal_rows)
    _write_csv(output_dir / "null_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    _write_csv(
        output_dir / "summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"score_gate_checks", "reference_gate_check"}
            }
            for row in summaries
        ],
    )
    _write_csv(output_dir / "attack_summary.csv", attack_summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "summaries": summaries,
                "attack_summaries": attack_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        summaries=summaries,
        attack_summaries=attack_summaries,
    )
    print(
        json.dumps(
            {
                "profiles": len(summaries),
                "score_survivors": sum(
                    bool(row["passes_score_gates"]) for row in summaries
                ),
                "full_survivors": sum(
                    bool(row["passes_full_gates"]) for row in summaries
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
        default=ROOT / "configs/dt_ldp_far/stage15_crossfit_peer_support_audit.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/dt_ldp_far/score_quality_stage15_crossfit_peer_support_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage15_Crossfit_Peer_Support_Audit.md",
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
