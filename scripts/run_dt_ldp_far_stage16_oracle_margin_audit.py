#!/usr/bin/env python3
"""Stage 16: oracle noise, support margins and two-logit audit.

The experiment uses fresh seeds relative to Stage 15 and strictly paired
honest noise across attack severities and score profiles.  It never trains a
model and never inspects accuracy.  Membership oracles are diagnostics only:
they indicate whether the current LDP-noise regime already makes reference
estimation difficult, but they are not minimax lower bounds.
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

from algorithms.noise_aware_scores import separate_novelty_trust_scores  # noqa: E402
from robustness.aggregators import centered_clipping  # noqa: E402
from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (  # noqa: E402
    _clean_honest_vectors,
    _corr,
    _finite_mean,
    _noise_scales,
)
from scripts.run_dt_ldp_far_stage14b_reference_trust_audit import (  # noqa: E402
    _energy_matrix,
    _reference,
    _separation_matrix,
)
from scripts.run_dt_ldp_far_stage15_crossfit_peer_support_audit import (  # noqa: E402
    _joint_support,
    _novelty_scores,
    _null_metrics,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clip_rows(vectors: torch.Tensor, radius: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    return vectors * (float(radius) / norms.clamp_min(1e-12)).clamp(max=1.0)


def _attack_vectors(
    honest: torch.Tensor,
    *,
    threat: str,
    count: int,
    scales: torch.Tensor,
    severity: float,
    config: dict[str, Any],
    seed: int,
) -> torch.Tensor:
    """Build paired attacks whose public severity changes no honest randomness."""

    centre = honest.mean(dim=0)
    if threat == "ipm":
        return (-3.0 * float(severity) * centre)[None, :].repeat(count, 1)
    if threat == "bitflip_x10":
        return (-10.0 * float(severity) * centre)[None, :].repeat(count, 1)
    if threat == "alie":
        return (centre - 1.5 * float(severity) * honest.std(dim=0, unbiased=False))[
            None, :
        ].repeat(count, 1)
    if threat == "adaptive_aligned_jitter":
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 99173)
        direction = centre / torch.linalg.vector_norm(centre).clamp_min(1e-12)
        server_clip = float(config["aggregation"]["server_clip_norm"])
        base_norm = min(0.98 * server_clip, 0.90 * severity * server_clip)
        base = base_norm * direction
        jitter = torch.randn(
            count,
            honest.shape[1],
            generator=generator,
            dtype=honest.dtype,
        )
        byzantine_scales = scales[-count:]
        return base[None, :] + (
            0.75
            * float(severity)
            * float(config["cohort"]["noise_std"])
            * byzantine_scales[:, None]
            * jitter
        )
    raise ValueError(f"Unknown threat {threat!r}")


def _paired_observed(
    clean: torch.Tensor,
    scales: torch.Tensor,
    *,
    threat: str,
    severity: float,
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    honest_noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    honest_preclip = clean + (
        float(config["cohort"]["noise_std"]) * scales[: len(clean), None] * honest_noise
    )
    server_clip = float(config["aggregation"]["server_clip_norm"])
    honest = _clip_rows(honest_preclip, server_clip)
    count = int(config["cohort"]["num_byzantine"])
    attack = _attack_vectors(
        honest,
        threat=threat,
        count=count,
        scales=scales,
        severity=severity,
        config=config,
        seed=seed,
    )
    observed = torch.cat((honest, _clip_rows(attack, server_clip)), dim=0)
    byzantine = torch.zeros(len(observed), dtype=torch.bool)
    byzantine[len(honest) :] = True
    return observed, byzantine, honest


def _profile_score(
    novelty: torch.Tensor,
    support: torch.Tensor,
    support_fraction: float,
) -> torch.Tensor:
    if support_fraction == 0.0:
        return novelty
    score, _ = separate_novelty_trust_scores(
        novelty,
        support,
        trust_fraction=float(support_fraction),
    )
    return score.double()


def _quantile_margin(
    honest_values: torch.Tensor,
    byzantine_values: torch.Tensor,
) -> tuple[float, float, float]:
    honest_low = float(torch.quantile(honest_values.double(), 0.10))
    byzantine_high = float(torch.quantile(byzantine_values.double(), 0.90))
    return honest_low - byzantine_high, honest_low, byzantine_high


def _required_support_fraction(
    novelty: torch.Tensor,
    support: torch.Tensor,
    byzantine: torch.Tensor,
) -> tuple[float, float, float]:
    """Return the empirical quantile condition for the two-logit rule.

    For ``l=(1-gamma)*nu+gamma*t``, a sufficient quantile condition is

    ``gamma * Delta_t >= (1-gamma) * Delta_nu``.

    This is not a universal per-client certificate because it uses 10/90 %
    quantiles.  It diagnoses whether a useful bulk separation exists.
    """

    honest = ~byzantine
    support_margin, _, _ = _quantile_margin(support[honest], support[byzantine])
    novelty_gap = float(
        torch.quantile(novelty[byzantine].double(), 0.90)
        - torch.quantile(novelty[honest].double(), 0.10)
    )
    if novelty_gap <= 0.0:
        required = 0.0
    elif support_margin <= 0.0:
        required = float("inf")
    else:
        required = novelty_gap / (novelty_gap + support_margin)
    return required, support_margin, novelty_gap


def _group_values(
    rows: list[dict[str, Any]], field: str
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            continue
        key = (
            row["threat"],
            row["severity"],
            row["signal_seed"],
            row["noise_permutation"],
            row["outlier_geometry"],
        )
        grouped[key].append(value)
    return {key: _finite_mean(values) for key, values in grouped.items()}


def _group_extreme(rows: list[dict[str, Any]], field: str, *, minimum: bool) -> float:
    values = list(_group_values(rows, field).values())
    if not values:
        return float("nan")
    return min(values) if minimum else max(values)


def _group_rate(rows: list[dict[str, Any]], field: str, predicate) -> float:
    values = list(_group_values(rows, field).values())
    if not values:
        return float("nan")
    return sum(bool(predicate(value)) for value in values) / len(values)


def _summarize_profiles(
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    gates = config["gates"]
    keys = sorted({(row["profile"], row["robust_reference"]) for row in signal_rows})
    summaries = []
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
        summary: dict[str, Any] = {
            "profile": profile,
            "robust_reference": reference,
            "support_logit_fraction": float(rows[0]["support_logit_fraction"]),
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
            "honest_outlier_recall_at_5_mean": _finite_mean(
                [float(row["honest_outlier_recall_at_5"]) for row in clean]
            ),
            "honest_outlier_weight_mass_mean": _finite_mean(
                [float(row["honest_outlier_weight_mass"]) for row in clean]
            ),
            "max_individual_weight_observed": max(
                float(row["max_individual_weight"]) for row in rows
            ),
            "attacked_byzantine_weight_mass_worst_group": _group_extreme(
                attacked, "byzantine_weight_mass", minimum=False
            ),
            "attacked_global_recall_worst_group": _group_extreme(
                attacked, "global_honest_outlier_recall_at_5", minimum=True
            ),
            "positive_quantile_support_margin_group_rate": _group_rate(
                attacked, "support_quantile_margin", lambda value: value > 0.0
            ),
            "empirical_two_logit_condition_group_rate": _group_rate(
                attacked,
                "two_logit_condition_slack",
                lambda value: value >= 0.0,
            ),
            "reference_error_over_honest_dispersion_worst_group": _group_extreme(
                attacked,
                "reference_error_over_honest_dispersion",
                minimum=False,
            ),
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
            "clean_recall": summary["honest_outlier_recall_at_5_mean"]
            >= float(gates["honest_outlier_recall_at_5_min"]),
            "clean_mass": summary["honest_outlier_weight_mass_mean"]
            > float(gates["honest_outlier_weight_mass_min_exclusive"]),
            "weight_cap": summary["max_individual_weight_observed"]
            <= float(gates["max_individual_weight"]) + 1e-10,
            "byzantine_mass": summary["attacked_byzantine_weight_mass_worst_group"]
            <= float(gates["attacked_byzantine_weight_mass_max"]),
            "attacked_recall": summary["attacked_global_recall_worst_group"]
            >= float(gates["attacked_global_honest_outlier_recall_at_5_min"]),
        }
        if summary["support_logit_fraction"] > 0.0:
            checks.update(
                {
                    "positive_support_margin": summary[
                        "positive_quantile_support_margin_group_rate"
                    ]
                    >= float(gates["positive_quantile_support_margin_group_rate_min"]),
                    "two_logit_condition": summary[
                        "empirical_two_logit_condition_group_rate"
                    ]
                    >= float(gates["empirical_two_logit_condition_group_rate_min"]),
                }
            )
        reference_check = summary[
            "reference_error_over_honest_dispersion_worst_group"
        ] <= float(gates["robust_reference_error_over_honest_dispersion_max"])
        summary["score_gate_checks"] = checks
        summary["reference_gate_check"] = reference_check
        summary["passes_score_gates"] = all(checks.values())
        summary["passes_full_gates"] = bool(
            summary["passes_score_gates"] and reference_check
        )
        summaries.append(summary)
    return summaries


def _summarize_references(
    oracle_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    threshold = float(
        config["gates"]["robust_reference_error_over_honest_dispersion_max"]
    )
    output = []
    for reference in sorted({row["robust_reference"] for row in oracle_rows}):
        rows = [row for row in oracle_rows if row["robust_reference"] == reference]
        output.append(
            {
                "robust_reference": reference,
                "deployed_error_ratio_mean": _finite_mean(
                    [float(row["deployed_reference_error_ratio"]) for row in rows]
                ),
                "deployed_error_ratio_worst_group": _group_extreme(
                    rows, "deployed_reference_error_ratio", minimum=False
                ),
                "membership_oracle_mean_error_ratio_mean": _finite_mean(
                    [float(row["membership_oracle_mean_error_ratio"]) for row in rows]
                ),
                "membership_oracle_mean_error_ratio_worst_group": _group_extreme(
                    rows, "membership_oracle_mean_error_ratio", minimum=False
                ),
                "membership_oracle_fcc_error_ratio_mean": _finite_mean(
                    [float(row["membership_oracle_fcc_error_ratio"]) for row in rows]
                ),
                "public_preclip_noise_floor_ratio_mean": _finite_mean(
                    [float(row["public_preclip_noise_floor_ratio"]) for row in rows]
                ),
                "deployed_excess_over_membership_oracle_mean": _finite_mean(
                    [
                        float(row["deployed_excess_over_membership_oracle"])
                        for row in rows
                    ]
                ),
                "deployed_reference_gate_passes": _group_extreme(
                    rows, "deployed_reference_error_ratio", minimum=False
                )
                <= threshold,
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
    config: dict[str, Any],
    profile_summaries: list[dict[str, Any]],
    reference_summaries: list[dict[str, Any]],
) -> None:
    primary_name = str(config["decision"]["primary_profile"])
    primary = [row for row in profile_summaries if row["profile"] == primary_name]
    primary_passes = any(row["passes_full_gates"] for row in primary)
    lines = [
        "# Stage 16 — Diagnostic oracle, marge de soutien et deux logits",
        "",
        "## Décision",
        "",
        f"- Profil principal préenregistré : `{primary_name}`.",
        f"- Références testées : **{len(reference_summaries)}**.",
        f"- Profil principal complet admissible : **{'oui' if primary_passes else 'non'}**.",
        "",
        "Aucune accuracy n'a été calculée. Les seeds 71, 83 et 97, ainsi que "
        "les amplitudes 0,75, 1,00 et 1,25, sont indépendantes de S15.",
        "",
        "## Diagnostic oracle de la référence",
        "",
        "L'oracle d'appartenance retire artificiellement les Byzantins avant "
        "d'estimer la moyenne. Il quantifie la difficulté induite par le bruit "
        "LDP même lorsque l'identité des attaquants est connue. Ce n'est ni un "
        "estimateur déployable ni une borne minimax d'impossibilité.",
        "",
        "| F déployée | Erreur F moyenne | Erreur F pire | Oracle moyenne | Oracle pire | Oracle FCC | Plancher bruit public | Excès F−oracle | Gate F |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in reference_summaries:
        lines.append(
            "| {reference} | {mean} | {worst} | {oracle} | {oracle_worst} | {oracle_fcc} | {floor} | {excess} | {passed} |".format(
                reference=row["robust_reference"].upper(),
                mean=_fmt(row["deployed_error_ratio_mean"]),
                worst=_fmt(row["deployed_error_ratio_worst_group"]),
                oracle=_fmt(row["membership_oracle_mean_error_ratio_mean"]),
                oracle_worst=_fmt(
                    row["membership_oracle_mean_error_ratio_worst_group"]
                ),
                oracle_fcc=_fmt(row["membership_oracle_fcc_error_ratio_mean"]),
                floor=_fmt(row["public_preclip_noise_floor_ratio_mean"]),
                excess=_fmt(row["deployed_excess_over_membership_oracle_mean"]),
                passed="oui" if row["deployed_reference_gate_passes"] else "non",
            )
        )
    lines.extend(
        [
            "",
            "## Marge et pondération à deux logits",
            "",
            "Le logit testé est",
            "",
            "```text",
            "ell_i = alpha_max [(1-gamma) novelty_i + gamma support_i]",
            "```",
            "",
            "équivalent, à une constante commune près, à une récompense de "
            "nouveauté et une pénalité de manque de soutien. Le contrôle S15 "
            "utilise `gamma=0,50`; le profil principal utilise `gamma=0,75`.",
            "",
            "La condition diagnostique sur les quantiles est",
            "",
            "```text",
            "gamma * [Q10(support_H) - Q90(support_B)]",
            "    >= (1-gamma) * [Q90(novelty_B) - Q10(novelty_H)].",
            "```",
            "",
            "Elle ne constitue pas un certificat universel client par client; "
            "elle vérifie si une séparation du gros de la population existe.",
            "",
            "| Profil | F | gamma | Corr. bruit | Corr. propre | Recall propre | Masse byz. pire | Recall attaqué pire | Marges + | Condition 2-logits | Poids max | Gate score | Gate complet |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in profile_summaries:
        lines.append(
            "| {profile} | {reference} | {gamma} | {noise} | {clean} | {recall} | {byz} | {attacked} | {margin} | {condition} | {weight} | {score_pass} | {full_pass} |".format(
                profile=row["profile"],
                reference=row["robust_reference"].upper(),
                gamma=_fmt(row["support_logit_fraction"], 2),
                noise=_fmt(row["null_abs_score_noise_correlation_max"]),
                clean=_fmt(row["clean_geometry_correlation_mean"]),
                recall=_fmt(row["honest_outlier_recall_at_5_mean"]),
                byz=_fmt(row["attacked_byzantine_weight_mass_worst_group"]),
                attacked=_fmt(row["attacked_global_recall_worst_group"]),
                margin=_fmt(row["positive_quantile_support_margin_group_rate"]),
                condition=_fmt(row["empirical_two_logit_condition_group_rate"]),
                weight=_fmt(row["max_individual_weight_observed"]),
                score_pass="oui" if row["passes_score_gates"] else "non",
                full_pass="oui" if row["passes_full_gates"] else "non",
            )
        )
    fcc_profiles = {
        row["profile"]: row
        for row in profile_summaries
        if row["robust_reference"] == "fcc"
    }
    fcc_reference = next(
        (row for row in reference_summaries if row["robust_reference"] == "fcc"),
        None,
    )
    if fcc_reference is not None and {
        "crossfit_covariance",
        "two_logit_support_50_control",
        primary_name,
    }.issubset(fcc_profiles):
        base = fcc_profiles["crossfit_covariance"]
        control = fcc_profiles["two_logit_support_50_control"]
        primary_fcc = fcc_profiles[primary_name]
        lines.extend(
            [
                "",
                "## Interprétation scientifique",
                "",
                "### 1. La difficulté de F vient d'abord du régime LDP",
                "",
                (
                    "L'erreur moyenne de l'oracle d'appartenance vaut "
                    f"`{fcc_reference['membership_oracle_mean_error_ratio_mean']:.3f}` "
                    "fois la dispersion honnête, presque exactement le plancher "
                    "gaussien public pré-clipping "
                    f"`{fcc_reference['public_preclip_noise_floor_ratio_mean']:.3f}`. "
                    "FCC déployé vaut "
                    f"`{fcc_reference['deployed_error_ratio_mean']:.3f}`, soit "
                    "seulement "
                    f"`{fcc_reference['deployed_excess_over_membership_oracle_mean']:.3f}` "
                    "au-dessus de l'oracle moyenne."
                ),
                "",
                "Ce résultat ne prouve pas une impossibilité minimax, mais il "
                "montre que le seuil `1,00` n'est pas réaliste pour une référence "
                "moyenne instantanée dans ce régime `(n=25,d=256,bruit LDP)`.",
                "",
                "### 2. Une marge de soutien positive ne suffit pas",
                "",
                (
                    "Une marge quantile de soutien est positive dans "
                    f"`{100.0 * primary_fcc['positive_quantile_support_margin_group_rate']:.1f} %` "
                    "des groupes, mais elle est assez grande pour compenser la "
                    "nouveauté byzantine dans seulement "
                    f"`{100.0 * primary_fcc['empirical_two_logit_condition_group_rate']:.1f} %` "
                    "des groupes. Les deux propriétés ne doivent donc pas être "
                    "confondues."
                ),
                "",
                "### 3. Augmenter la pénalité produit un compromis défavorable",
                "",
                (
                    "De `gamma=0,50` à `gamma=0,75`, la pire masse byzantine "
                    f"baisse de `{control['attacked_byzantine_weight_mass_worst_group']:.3f}` "
                    f"à `{primary_fcc['attacked_byzantine_weight_mass_worst_group']:.3f}`. "
                    "En contrepartie, la corrélation à la géométrie honnête tombe "
                    f"de `{control['clean_geometry_correlation_mean']:.3f}` à "
                    f"`{primary_fcc['clean_geometry_correlation_mean']:.3f}`, le "
                    f"rappel propre de `{control['honest_outlier_recall_at_5_mean']:.3f}` "
                    f"à `{primary_fcc['honest_outlier_recall_at_5_mean']:.3f}` et "
                    "le pire rappel sous attaque de "
                    f"`{control['attacked_global_recall_worst_group']:.3f}` à "
                    f"`{primary_fcc['attacked_global_recall_worst_group']:.3f}`."
                ),
                "",
                "Cette évolution réfute, pour cette famille de scores et cette "
                "matrice, l'idée qu'une pénalité directionnelle plus forte puisse "
                "simultanément préserver les honest outliers et fermer la masse "
                "byzantine par simple réglage de `gamma`.",
            ]
        )
    lines.extend(["", "## Critères bloquants du profil principal", ""])
    for row in primary:
        failed = [key for key, passed in row["score_gate_checks"].items() if not passed]
        if not row["reference_gate_check"]:
            failed.append("reference_quality")
        lines.append(
            f"- **{row['robust_reference'].upper()}** : "
            + (", ".join(failed) if failed else "aucun")
            + "."
        )
    lines.extend(
        [
            "",
            "## Conclusion opérationnelle",
            "",
            (
                "Le profil principal peut être transféré vers un screen "
                "Fashion-MNIST préenregistré."
                if primary_passes
                else "Le profil principal ne franchit pas tous les gates. "
                "Aucun screen Fashion-MNIST ne doit être lancé pour ajuster "
                "après coup gamma, les seuils de soutien ou la référence."
            ),
            "",
            "## Traçabilité",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Détails oracle : `{(output_dir / 'oracle_detail.csv').resolve()}`",
            f"- Détails marges/poids : `{(output_dir / 'signal_detail.csv').resolve()}`",
            f"- Détails nuls : `{(output_dir / 'null_detail.csv').resolve()}`",
            f"- Synthèse profils : `{(output_dir / 'profile_summary.csv').resolve()}`",
            f"- Synthèse références : `{(output_dir / 'reference_summary.csv').resolve()}`",
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
    num_byzantine = int(cohort["num_byzantine"])
    dimension = int(cohort["ambient_dimension"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    holdout_draws = int(null_holdout_draws_override or randomness["null_holdout_draws"])
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Stage 16 requires >=20 null draws and >=1 signal draw")

    helper_config = {
        **config,
        "score": {
            "dimension": dimension,
            "fcc_radius": float(config["novelty"]["fcc_radius"]),
        },
    }
    profiles = list(config["profiles"])
    references = [str(value) for value in config["robust_reference"]["candidates"]]
    levels = [float(value) for value in cohort["public_noise_scales"]]
    permutations = [str(value) for value in cohort["noise_permutations"]]
    alpha_max = math.log(
        float(config["aggregation"]["kappa_w"])
        * (n - 1)
        / (n - float(config["aggregation"]["kappa_w"]))
    )

    calibration_energy: dict[str, torch.Tensor] = {}
    calibration_separation: dict[str, torch.Tensor] = {}
    null_rows: list[dict[str, Any]] = []
    for permutation_index, permutation in enumerate(permutations):
        scales = _noise_scales(levels, permutation, n)
        variances = (float(cohort["noise_std"]) * scales).square()
        std = variances.sqrt()
        calibration_generator = torch.Generator(device="cpu").manual_seed(
            int(randomness["calibration_seed"]) + 10_000 * permutation_index
        )
        calibration = (
            torch.randn(
                calibration_draws,
                n,
                dimension,
                generator=calibration_generator,
                dtype=torch.float64,
            )
            * std[None, :, None]
        )
        calibration = _clip_rows(calibration, server_clip)
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
        holdout = _clip_rows(holdout, server_clip)
        calibration_energy[permutation] = _energy_matrix(calibration, helper_config)
        calibration_separation[permutation] = _separation_matrix(calibration, variances)
        holdout_novelty = torch.stack(
            [
                _novelty_scores(energy, calibration_energy[permutation], config)
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
                fraction = float(profile["support_logit_fraction"])
                score_matrix = torch.stack(
                    [
                        _profile_score(
                            holdout_novelty[index],
                            support_matrix[index],
                            fraction,
                        )
                        for index in range(holdout_draws)
                    ]
                )
                null_rows.append(
                    {
                        "profile": str(profile["name"]),
                        "robust_reference": reference_name,
                        "noise_permutation": permutation,
                        **_null_metrics(score_matrix, scales),
                    }
                )

    signal_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    geometries = [str(value) for value in cohort["outlier_geometries"]]
    threats = [str(value) for value in cohort["threats"]]
    severities = [float(value) for value in cohort["attack_severities"]]
    for signal_seed in [int(value) for value in randomness["signal_seeds"]]:
        for geometry_index, geometry in enumerate(geometries):
            for permutation_index, permutation in enumerate(permutations):
                scales = _noise_scales(levels, permutation, n)
                variances = (float(cohort["noise_std"]) * scales).square()
                n_honest = n - num_byzantine
                clean, outliers = _clean_honest_vectors(
                    signal_seed,
                    n_honest=n_honest,
                    dimension=dimension,
                    num_outliers=int(cohort["honest_outliers"]),
                    outlier_geometry=geometry,
                )
                clean = _clip_rows(clean, server_clip)
                clean_distances = _energy_matrix(clean[None, :, :], helper_config)[
                    0
                ].sqrt()
                target = clean.mean(dim=0)
                dispersion = torch.sqrt(
                    torch.linalg.vector_norm(clean - target, dim=1).square().mean()
                ).clamp_min(1e-12)
                public_noise_floor = math.sqrt(
                    dimension * float(variances[:n_honest].sum()) / n_honest**2
                )
                for threat_index, threat in enumerate(threats):
                    for severity in severities:
                        for draw in range(signal_draws):
                            # Severity is deliberately absent from this seed: the
                            # honest noise and adaptive-attack jitter are paired.
                            run_seed = (
                                signal_seed * 1_000_000
                                + geometry_index * 100_000
                                + permutation_index * 10_000
                                + threat_index * 1_000
                                + draw
                            )
                            observed, byzantine, honest_observed = _paired_observed(
                                clean,
                                scales,
                                threat=threat,
                                severity=severity,
                                config=config,
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
                            oracle_mean = honest_observed.mean(dim=0)
                            oracle_fcc = centered_clipping(
                                honest_observed,
                                anchor=torch.zeros(dimension, dtype=torch.float64),
                                tau=float(config["robust_reference"]["fcc_radius"]),
                            )
                            oracle_mean_ratio = float(
                                torch.linalg.vector_norm(oracle_mean - target)
                                / dispersion
                            )
                            oracle_fcc_ratio = float(
                                torch.linalg.vector_norm(oracle_fcc - target)
                                / dispersion
                            )
                            for reference_name in references:
                                reference = _reference(
                                    observed, reference_name, helper_config
                                )
                                reference_ratio = float(
                                    torch.linalg.vector_norm(reference - target)
                                    / dispersion
                                )
                                support, _ = _joint_support(
                                    observed,
                                    reference,
                                    variances,
                                    calibration_separation[permutation],
                                    config,
                                )
                                required_fraction, margin, novelty_gap = (
                                    _required_support_fraction(
                                        novelty, support, byzantine
                                    )
                                )
                                common = {
                                    "robust_reference": reference_name,
                                    "signal_seed": signal_seed,
                                    "draw": draw,
                                    "noise_permutation": permutation,
                                    "outlier_geometry": geometry,
                                    "threat": threat,
                                    "severity": severity,
                                }
                                oracle_rows.append(
                                    {
                                        **common,
                                        "deployed_reference_error_ratio": reference_ratio,
                                        "membership_oracle_mean_error_ratio": oracle_mean_ratio,
                                        "membership_oracle_fcc_error_ratio": oracle_fcc_ratio,
                                        "public_preclip_noise_floor_ratio": public_noise_floor
                                        / float(dispersion),
                                        "deployed_excess_over_membership_oracle": reference_ratio
                                        - oracle_mean_ratio,
                                    }
                                )
                                honest = ~byzantine
                                for profile in profiles:
                                    fraction = float(profile["support_logit_fraction"])
                                    scores = _profile_score(novelty, support, fraction)
                                    weights = torch.softmax(alpha_max * scores, dim=0)
                                    full_outliers = torch.zeros_like(byzantine)
                                    full_outliers[: len(outliers)] = outliers
                                    honest_scores = scores[honest]
                                    predicted_honest = torch.topk(
                                        honest_scores,
                                        min(5, honest_scores.numel()),
                                    ).indices
                                    predicted_global = torch.topk(
                                        scores, min(5, scores.numel())
                                    ).indices
                                    aggregate = (weights[:, None] * observed).sum(dim=0)
                                    condition_slack = (
                                        fraction * margin
                                        - (1.0 - fraction) * novelty_gap
                                    )
                                    signal_rows.append(
                                        {
                                            **common,
                                            "profile": str(profile["name"]),
                                            "profile_role": str(profile["role"]),
                                            "support_logit_fraction": fraction,
                                            "clean_geometry_correlation": _corr(
                                                honest_scores, clean_distances
                                            ),
                                            "honest_outlier_recall_at_5": float(
                                                outliers[predicted_honest]
                                                .double()
                                                .mean()
                                            ),
                                            "global_honest_outlier_recall_at_5": float(
                                                full_outliers[predicted_global]
                                                .double()
                                                .mean()
                                            ),
                                            "honest_outlier_weight_mass": float(
                                                weights[full_outliers].sum()
                                            ),
                                            "byzantine_weight_mass": float(
                                                weights[byzantine].sum()
                                            ),
                                            "max_individual_weight": float(
                                                weights.max()
                                            ),
                                            "aggregate_error": float(
                                                torch.linalg.vector_norm(
                                                    aggregate - target
                                                )
                                            ),
                                            "reference_error_over_honest_dispersion": reference_ratio,
                                            "support_quantile_margin": margin,
                                            "novelty_quantile_gap": novelty_gap,
                                            "required_support_logit_fraction": required_fraction,
                                            "two_logit_condition_slack": condition_slack,
                                            "support_honest_mean": float(
                                                support[honest].mean()
                                            ),
                                            "support_byzantine_mean": float(
                                                support[byzantine].mean()
                                            ),
                                        }
                                    )

    # Clean profiles use the same paired draws with no Byzantine messages.
    for signal_seed in [int(value) for value in randomness["signal_seeds"]]:
        for geometry_index, geometry in enumerate(geometries):
            for permutation_index, permutation in enumerate(permutations):
                scales = _noise_scales(levels, permutation, n)
                variances = (float(cohort["noise_std"]) * scales).square()
                clean, outliers = _clean_honest_vectors(
                    signal_seed,
                    n_honest=n,
                    dimension=dimension,
                    num_outliers=int(cohort["honest_outliers"]),
                    outlier_geometry=geometry,
                )
                clean = _clip_rows(clean, server_clip)
                clean_distances = _energy_matrix(clean[None, :, :], helper_config)[
                    0
                ].sqrt()
                target = clean.mean(dim=0)
                for draw in range(signal_draws):
                    run_seed = (
                        signal_seed * 1_000_000
                        + geometry_index * 100_000
                        + permutation_index * 10_000
                        + 99_000
                        + draw
                    )
                    generator = torch.Generator(device="cpu").manual_seed(run_seed)
                    observed = clean + (
                        float(cohort["noise_std"])
                        * scales[:, None]
                        * torch.randn(
                            clean.shape, generator=generator, dtype=clean.dtype
                        )
                    )
                    observed = _clip_rows(observed, server_clip)
                    observed_energy = _energy_matrix(
                        observed[None, :, :], helper_config
                    )[0]
                    novelty = _novelty_scores(
                        observed_energy,
                        calibration_energy[permutation],
                        config,
                    )
                    for reference_name in references:
                        reference = _reference(observed, reference_name, helper_config)
                        support, _ = _joint_support(
                            observed,
                            reference,
                            variances,
                            calibration_separation[permutation],
                            config,
                        )
                        reference_ratio = float(
                            torch.linalg.vector_norm(reference - target)
                            / torch.sqrt(
                                torch.linalg.vector_norm(clean - target, dim=1)
                                .square()
                                .mean()
                            ).clamp_min(1e-12)
                        )
                        for profile in profiles:
                            fraction = float(profile["support_logit_fraction"])
                            scores = _profile_score(novelty, support, fraction)
                            weights = torch.softmax(alpha_max * scores, dim=0)
                            predicted = torch.topk(
                                scores, min(5, scores.numel())
                            ).indices
                            signal_rows.append(
                                {
                                    "robust_reference": reference_name,
                                    "signal_seed": signal_seed,
                                    "draw": draw,
                                    "noise_permutation": permutation,
                                    "outlier_geometry": geometry,
                                    "threat": "none",
                                    "severity": 0.0,
                                    "profile": str(profile["name"]),
                                    "profile_role": str(profile["role"]),
                                    "support_logit_fraction": fraction,
                                    "clean_geometry_correlation": _corr(
                                        scores, clean_distances
                                    ),
                                    "honest_outlier_recall_at_5": float(
                                        outliers[predicted].double().mean()
                                    ),
                                    "global_honest_outlier_recall_at_5": float(
                                        outliers[predicted].double().mean()
                                    ),
                                    "honest_outlier_weight_mass": float(
                                        weights[outliers].sum()
                                    ),
                                    "byzantine_weight_mass": 0.0,
                                    "max_individual_weight": float(weights.max()),
                                    "aggregate_error": float(
                                        torch.linalg.vector_norm(
                                            (weights[:, None] * observed).sum(dim=0)
                                            - target
                                        )
                                    ),
                                    "reference_error_over_honest_dispersion": reference_ratio,
                                    "support_quantile_margin": float("nan"),
                                    "novelty_quantile_gap": float("nan"),
                                    "required_support_logit_fraction": float("nan"),
                                    "two_logit_condition_slack": float("nan"),
                                    "support_honest_mean": float(support.mean()),
                                    "support_byzantine_mean": float("nan"),
                                }
                            )

    profile_summaries = _summarize_profiles(null_rows, signal_rows, config)
    reference_summaries = _summarize_references(oracle_rows, config)
    _write_csv(output_dir / "null_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    _write_csv(output_dir / "oracle_detail.csv", oracle_rows)
    _write_csv(
        output_dir / "profile_summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"score_gate_checks", "reference_gate_check"}
            }
            for row in profile_summaries
        ],
    )
    _write_csv(output_dir / "reference_summary.csv", reference_summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "profile_summaries": profile_summaries,
                "reference_summaries": reference_summaries,
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
        config=config,
        profile_summaries=profile_summaries,
        reference_summaries=reference_summaries,
    )
    primary = str(config["decision"]["primary_profile"])
    print(
        json.dumps(
            {
                "profiles": len(profile_summaries),
                "primary_full_survivors": sum(
                    row["profile"] == primary and bool(row["passes_full_gates"])
                    for row in profile_summaries
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
        default=ROOT / "configs/dt_ldp_far/stage16_oracle_margin_two_logit_audit.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/dt_ldp_far/score_quality_stage16_oracle_margin_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage16_Oracle_Margin_Two_Logit_Audit.md",
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
