#!/usr/bin/env python3
"""Stage 18: frozen conditional Byzantine holdout for DT-LDP-FAR scoring.

The primary claim is deliberately conditional: a coherent honest minority has
at least ``f+2`` members, so every member can have ``f+1`` aligned peers while
a Byzantine coalition of size at most ``f`` cannot self-certify using only its
own messages.  Dispersed minorities, minorities of size ``f`` and attacks
aligned with honest traffic are retained as negative/evasive controls.
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

from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (  # noqa: E402
    _clean_honest_vectors,
    _corr,
    _finite_mean,
    _noise_scales,
    _orthogonal,
    _unit,
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
from scripts.run_dt_ldp_far_stage16_oracle_margin_audit import (  # noqa: E402
    _paired_observed,
    _profile_score,
    _quantile_margin,
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


def _outlier_count(rule: str, f: int) -> int:
    if rule == "f":
        return f
    if rule == "f_plus_2":
        return f + 2
    raise ValueError(f"Unknown outlier-count rule {rule!r}")


def _coherent_honest_vectors(
    seed: int,
    *,
    n_honest: int,
    dimension: int,
    num_outliers: int,
    outlier_geometry: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a public-directionally coherent honest minority."""

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
        raise RuntimeError("The coherent minority construction lost an ID")
    shared = _orthogonal(
        torch.randn(dimension, generator=generator, dtype=torch.float64), descent
    )
    jitters = torch.randn(
        num_outliers, dimension, generator=generator, dtype=torch.float64
    )
    jitters = torch.stack([_orthogonal(row, descent) for row in jitters])
    if outlier_geometry == "aligned":
        direction = _unit(0.60 * descent + 0.80 * shared)
    elif outlier_geometry == "orthogonal":
        direction = shared
    else:
        raise ValueError(f"Unknown outlier geometry {outlier_geometry!r}")
    shifts = 0.09 * torch.stack(
        [_unit(direction + 0.08 * jitter) for jitter in jitters]
    )
    clean[outlier_ids] += shifts
    outlier_mask[outlier_ids] = True
    return clean, outlier_mask


def _clean_vectors(
    seed: int,
    *,
    n_honest: int,
    dimension: int,
    num_outliers: int,
    outlier_geometry: str,
    coherence: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coherence == "coherent":
        return _coherent_honest_vectors(
            seed,
            n_honest=n_honest,
            dimension=dimension,
            num_outliers=num_outliers,
            outlier_geometry=outlier_geometry,
        )
    if coherence == "dispersed":
        return _clean_honest_vectors(
            seed,
            n_honest=n_honest,
            dimension=dimension,
            num_outliers=num_outliers,
            outlier_geometry=outlier_geometry,
        )
    raise ValueError(f"Unknown coherence {coherence!r}")


def _validate_stage17_selection(config: dict[str, Any], root: Path) -> None:
    settings = config["stage17_selection"]
    path = root / str(settings["path"])
    selection = json.loads(path.read_text(encoding="utf-8"))["selected_cell"]
    if selection is None:
        raise ValueError("Stage 17 did not authorize Stage 18")
    expected = {
        "total_clients": int(settings["required_total_clients"]),
        "effective_score_dimension": int(
            settings["required_effective_score_dimension"]
        ),
        "upload_noise_std": float(settings["required_upload_noise_std"]),
    }
    for key, value in expected.items():
        observed = selection[key]
        if isinstance(value, float):
            valid = math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = int(observed) == value
        if not valid:
            raise ValueError(
                f"Stage 18 config disagrees with Stage 17 selection for {key}"
            )


def _mean_by_group(
    rows: list[dict[str, Any]], field: str, *, include_severity: bool = True
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key: tuple[Any, ...] = (
            row["threat"],
            row["signal_seed"],
            row["noise_permutation"],
            row["outlier_geometry"],
        )
        if include_severity:
            key += (row["severity"],)
        grouped[key].append(float(row[field]))
    return {key: _finite_mean(values) for key, values in grouped.items()}


def _summaries(
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    separated = set(str(value) for value in config["threats"]["separated"])
    gates = config["gates"]
    output = []
    keys = sorted(
        {
            (row["minority_structure"], row["profile"], row["robust_reference"])
            for row in signal_rows
        }
    )
    for structure, profile, reference in keys:
        rows = [
            row
            for row in signal_rows
            if row["minority_structure"] == structure
            and row["profile"] == profile
            and row["robust_reference"] == reference
        ]
        clean = [row for row in rows if row["threat"] == "none"]
        attacked = [row for row in rows if row["threat"] in separated]
        evasive = [row for row in rows if row["threat"] not in separated | {"none"}]
        outlier_count = int(rows[0]["num_honest_outliers"])
        n = int(config["cohort"]["num_clients"])
        uniform_mass = outlier_count / n
        error_ratios = _mean_by_group(attacked, "aggregate_error_ratio_to_uniform")
        byz_mass = _mean_by_group(attacked, "byzantine_weight_mass")
        attacked_recall = _mean_by_group(attacked, "global_honest_outlier_recall")
        support_margin = _mean_by_group(attacked, "support_quantile_margin")
        summary: dict[str, Any] = {
            "minority_structure": structure,
            "profile": profile,
            "robust_reference": reference,
            "support_logit_fraction": float(rows[0]["support_logit_fraction"]),
            "num_honest_outliers": outlier_count,
            "uniform_honest_outlier_mass": uniform_mass,
            "null_abs_score_noise_correlation_max": max(
                abs(float(row["null_score_noise_correlation"]))
                for row in null_rows
                if row["profile"] == profile and row["robust_reference"] == reference
            ),
            "null_noise_tier_score_range_max": max(
                float(row["null_noise_tier_score_range"])
                for row in null_rows
                if row["profile"] == profile and row["robust_reference"] == reference
            ),
            "clean_geometry_correlation_mean": _finite_mean(
                [float(row["clean_geometry_correlation"]) for row in clean]
            ),
            "honest_outlier_recall_mean": _finite_mean(
                [float(row["honest_outlier_recall"]) for row in clean]
            ),
            "honest_outlier_weight_mass_mean": _finite_mean(
                [float(row["honest_outlier_weight_mass"]) for row in clean]
            ),
            "honest_outlier_weight_mass_gain_over_uniform": _finite_mean(
                [float(row["honest_outlier_weight_mass"]) for row in clean]
            )
            - uniform_mass,
            "max_individual_weight_observed": max(
                float(row["max_individual_weight"]) for row in rows
            ),
            "separated_byzantine_weight_mass_worst_group": max(byz_mass.values()),
            "separated_global_honest_outlier_recall_worst_group": min(
                attacked_recall.values()
            ),
            "separated_aggregate_error_mean_ratio_to_uniform": _finite_mean(
                list(error_ratios.values())
            ),
            "separated_aggregate_error_worst_group_ratio_to_uniform": max(
                error_ratios.values()
            ),
            "positive_support_margin_group_rate": sum(
                value > 0.0 for value in support_margin.values()
            )
            / len(support_margin),
            "evasive_byzantine_weight_mass_mean": _finite_mean(
                [float(row["byzantine_weight_mass"]) for row in evasive]
            ),
            "evasive_aggregate_error_ratio_mean": _finite_mean(
                [float(row["aggregate_error_ratio_to_uniform"]) for row in evasive]
            ),
            "deployed_reference_error_ratio_mean": _finite_mean(
                [float(row["reference_error_ratio"]) for row in attacked]
            ),
        }
        checks = {
            "null_corr": summary["null_abs_score_noise_correlation_max"]
            <= float(gates["null_abs_score_noise_correlation_max"]),
            "null_tier": summary["null_noise_tier_score_range_max"]
            <= float(gates["null_noise_tier_score_range_max"]),
            "clean_corr": summary["clean_geometry_correlation_mean"]
            >= float(gates["clean_geometry_correlation_mean_min"]),
            "clean_recall": summary["honest_outlier_recall_mean"]
            >= float(gates["honest_outlier_recall_mean_min"]),
            "clean_mass_gain": summary["honest_outlier_weight_mass_gain_over_uniform"]
            >= float(gates["honest_outlier_weight_mass_gain_over_uniform_min"]),
            "weight_cap": summary["max_individual_weight_observed"]
            <= float(gates["max_individual_weight"]) + 1e-10,
            "separated_byz_mass": summary["separated_byzantine_weight_mass_worst_group"]
            <= float(gates["separated_byzantine_weight_mass_worst_group_max"]),
            "separated_recall": summary[
                "separated_global_honest_outlier_recall_worst_group"
            ]
            >= float(gates["separated_global_honest_outlier_recall_worst_group_min"]),
            "aggregate_mean": summary["separated_aggregate_error_mean_ratio_to_uniform"]
            <= float(gates["separated_aggregate_error_mean_ratio_to_uniform_max"]),
            "aggregate_worst": summary[
                "separated_aggregate_error_worst_group_ratio_to_uniform"
            ]
            <= float(
                gates["separated_aggregate_error_worst_group_ratio_to_uniform_max"]
            ),
            "support_margin": summary["positive_support_margin_group_rate"]
            >= float(gates["positive_support_margin_group_rate_min"]),
        }
        summary["gate_checks"] = checks
        summary["passes_conditional_score_gates"] = all(checks.values())
        output.append(summary)

    # Reference competition is evaluated within each structure/profile.
    for row in output:
        peers = [
            item
            for item in output
            if item["minority_structure"] == row["minority_structure"]
            and item["profile"] == row["profile"]
        ]
        best = min(float(item["deployed_reference_error_ratio_mean"]) for item in peers)
        row["reference_excess_over_best"] = (
            float(row["deployed_reference_error_ratio_mean"]) - best
        )
        row["reference_is_competitive"] = row["reference_excess_over_best"] <= float(
            gates["fcc_reference_competitive_tolerance"]
        )
        row["passes_full_conditional_gates"] = bool(
            row["passes_conditional_score_gates"] and row["reference_is_competitive"]
        )
    return output


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    primary: dict[str, Any],
) -> None:
    lines = [
        "# Stage 18 — Validation Byzantine conditionnelle sur holdout",
        "",
        "## Claim préenregistré",
        "",
        "Le claim testé n'est pas une identification Byzantine universelle. "
        "Il porte sur un régime identifiable : une minorité honnête cohérente "
        "contient `f+2` clients, donc chaque membre dispose de `f+1` pairs "
        "possibles, tandis qu'une coalition de taille `f` ne peut pas "
        "s'auto-certifier seule. IPM et Bit-Flip sont les attaques séparées "
        "du claim. ALIE et l'attaque adaptative alignée sont des contrôles "
        "évasifs explicitement hors garantie.",
        "",
        "Aucune accuracy n'a servi à choisir les profils, seuils ou seeds.",
        "",
        "## Synthèse",
        "",
        "| Structure | Profil | F | corr. propre | rappel propre | gain masse outliers | masse byz. pire | rappel attaqué pire | erreur/uniforme moy. | erreur/uniforme pire | marge + | masse byz. évasive | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            "| {structure} | {profile} | {reference} | {corr:.3f} | {recall:.3f} | {gain:+.3f} | {byz:.3f} | {attacked:.3f} | {mean:.3f} | {worst:.3f} | {margin:.1%} | {evasive:.3f} | {passed} |".format(
                structure=row["minority_structure"],
                profile=row["profile"],
                reference=row["robust_reference"].upper(),
                corr=row["clean_geometry_correlation_mean"],
                recall=row["honest_outlier_recall_mean"],
                gain=row["honest_outlier_weight_mass_gain_over_uniform"],
                byz=row["separated_byzantine_weight_mass_worst_group"],
                attacked=row["separated_global_honest_outlier_recall_worst_group"],
                mean=row["separated_aggregate_error_mean_ratio_to_uniform"],
                worst=row["separated_aggregate_error_worst_group_ratio_to_uniform"],
                margin=row["positive_support_margin_group_rate"],
                evasive=row["evasive_byzantine_weight_mass_mean"],
                passed="oui" if row["passes_full_conditional_gates"] else "non",
            )
        )
    lines.extend(["", "## Décision primaire", ""])
    if primary["passes_full_conditional_gates"]:
        lines.extend(
            [
                "Le profil principal franchit tous les gates conditionnels "
                "sur les cinq seeds holdout. Cela constitue une **validation "
                "synthétique conditionnelle** de la construction : neutralité "
                "au niveau de bruit, conservation d'une minorité honnête "
                "cohérente, contrôle de masse Byzantine sous attaques séparées, "
                "cap individuel et erreur d'agrégation non supérieure au contrôle uniforme.",
                "",
                "Cette validation autorise une confirmation Fashion-MNIST. "
                "Elle ne transforme pas les attaques évasives en cas résolus.",
            ]
        )
    else:
        failed = [key for key, passed in primary["gate_checks"].items() if not passed]
        if not primary["reference_is_competitive"]:
            failed.append("reference_competitive")
        lines.extend(
            [
                "Le profil principal échoue au holdout indépendant.",
                "",
                "Critères bloquants : " + ", ".join(failed) + ".",
                "",
                "Le résultat constitue une validation négative pour le claim "
                "conditionnel tel qu'il a été figé. Les seuils ne doivent pas "
                "être réajustés sur ces seeds.",
            ]
        )
    lines.extend(
        [
            "",
            "## Traçabilité",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Détails nuls : `{(output_dir / 'null_detail.csv').resolve()}`",
            f"- Détails signal/attaques : `{(output_dir / 'signal_detail.csv').resolve()}`",
            f"- Synthèse : `{(output_dir / 'summary.csv').resolve()}`",
            f"- Décision machine : `{(output_dir / 'decision.json').resolve()}`",
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
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_stage17_selection(config, ROOT)
    cohort = config["cohort"]
    randomness = config["randomness"]
    n = int(cohort["num_clients"])
    f = int(cohort["num_byzantine"])
    dimension = int(cohort["effective_score_dimension"])
    noise_std = float(cohort["noise_std"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    holdout_draws = int(null_holdout_draws_override or randomness["null_holdout_draws"])
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Stage 18 requires >=20 null draws and >=1 signal draw")
    alpha_max = math.log(
        float(config["aggregation"]["kappa_w"])
        * (n - 1)
        / (n - float(config["aggregation"]["kappa_w"]))
    )
    helper_config = {
        **config,
        "score": {
            "dimension": dimension,
            "fcc_radius": config["novelty"]["fcc_radius"],
        },
    }
    levels = [float(value) for value in cohort["public_noise_scales"]]
    permutations = [str(value) for value in cohort["noise_permutations"]]
    profiles = list(config["profiles"])
    references = [str(value) for value in config["robust_reference"]["candidates"]]
    calibration_energy: dict[str, torch.Tensor] = {}
    calibration_separation: dict[str, torch.Tensor] = {}
    null_rows: list[dict[str, Any]] = []

    for permutation_index, permutation in enumerate(permutations):
        scales = _noise_scales(levels, permutation, n)
        variances = (noise_std * scales).square()
        generator = torch.Generator(device="cpu").manual_seed(
            int(randomness["calibration_seed"]) + permutation_index * 10_000
        )
        calibration = (
            noise_std
            * scales[None, :, None]
            * torch.randn(
                calibration_draws,
                n,
                dimension,
                generator=generator,
                dtype=torch.float64,
            )
        )
        calibration = _clip_rows(calibration, server_clip)
        calibration_energy[permutation] = _energy_matrix(calibration, helper_config)
        calibration_separation[permutation] = _separation_matrix(calibration, variances)
        holdout_generator = torch.Generator(device="cpu").manual_seed(
            int(randomness["null_holdout_seed"]) + permutation_index * 10_000
        )
        holdout = (
            noise_std
            * scales[None, :, None]
            * torch.randn(
                holdout_draws,
                n,
                dimension,
                generator=holdout_generator,
                dtype=torch.float64,
            )
        )
        holdout = _clip_rows(holdout, server_clip)
        novelty_matrix = torch.stack(
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
                            novelty_matrix[index], support_matrix[index], fraction
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

    threats = (
        ["none"]
        + [str(value) for value in config["threats"]["separated"]]
        + [str(value) for value in config["threats"]["evasive_controls"]]
    )
    severities = [float(value) for value in config["threats"]["attack_severities"]]
    signal_rows: list[dict[str, Any]] = []
    for structure in config["cohort"]["minority_structures"]:
        structure_name = str(structure["name"])
        outlier_count = _outlier_count(str(structure["outlier_count_rule"]), f)
        coherence = str(structure["coherence"])
        for signal_seed in [int(value) for value in randomness["signal_seeds"]]:
            for geometry_index, outlier_geometry in enumerate(
                [str(value) for value in cohort["outlier_geometries"]]
            ):
                for permutation_index, permutation in enumerate(permutations):
                    scales = _noise_scales(levels, permutation, n)
                    variances = (noise_std * scales).square()
                    for threat_index, threat in enumerate(threats):
                        n_honest = n if threat == "none" else n - f
                        clean, outliers = _clean_vectors(
                            signal_seed,
                            n_honest=n_honest,
                            dimension=dimension,
                            num_outliers=outlier_count,
                            outlier_geometry=outlier_geometry,
                            coherence=coherence,
                        )
                        clean = _clip_rows(clean, server_clip)
                        clean_distances = _energy_matrix(
                            clean[None, :, :], helper_config
                        )[0].sqrt()
                        target = clean.mean(dim=0)
                        dispersion = torch.sqrt(
                            torch.linalg.vector_norm(clean - target, dim=1)
                            .square()
                            .mean()
                        ).clamp_min(1e-12)
                        threat_severities = [1.0] if threat == "none" else severities
                        for severity in threat_severities:
                            for draw in range(signal_draws):
                                run_seed = (
                                    signal_seed * 10_000_000
                                    + geometry_index * 1_000_000
                                    + permutation_index * 100_000
                                    + threat_index * 10_000
                                    + draw
                                )
                                if threat == "none":
                                    generator = torch.Generator(
                                        device="cpu"
                                    ).manual_seed(run_seed)
                                    observed = clean + noise_std * scales[
                                        :, None
                                    ] * torch.randn(
                                        clean.shape,
                                        generator=generator,
                                        dtype=clean.dtype,
                                    )
                                    observed = _clip_rows(observed, server_clip)
                                    byzantine = torch.zeros(n, dtype=torch.bool)
                                else:
                                    observed, byzantine, _ = _paired_observed(
                                        clean,
                                        scales,
                                        threat=threat,
                                        severity=severity,
                                        config=config,
                                        seed=run_seed,
                                    )
                                novelty = _novelty_scores(
                                    _energy_matrix(observed[None, :, :], helper_config)[
                                        0
                                    ],
                                    calibration_energy[permutation],
                                    config,
                                )
                                full_outliers = torch.zeros(n, dtype=torch.bool)
                                full_outliers[:n_honest] = outliers
                                uniform = observed.mean(dim=0)
                                uniform_error = torch.linalg.vector_norm(
                                    uniform - target
                                ).clamp_min(1e-12)
                                for reference_name in references:
                                    reference = _reference(
                                        observed, reference_name, helper_config
                                    )
                                    support, _ = _joint_support(
                                        observed,
                                        reference,
                                        variances,
                                        calibration_separation[permutation],
                                        config,
                                    )
                                    honest = ~byzantine
                                    margin = (
                                        _quantile_margin(
                                            support[honest], support[byzantine]
                                        )[0]
                                        if bool(byzantine.any())
                                        else float("nan")
                                    )
                                    for profile in profiles:
                                        fraction = float(
                                            profile["support_logit_fraction"]
                                        )
                                        scores = _profile_score(
                                            novelty, support, fraction
                                        )
                                        weights = torch.softmax(
                                            alpha_max * scores, dim=0
                                        )
                                        aggregate = (weights[:, None] * observed).sum(
                                            dim=0
                                        )
                                        honest_scores = scores[honest]
                                        predicted_honest = torch.topk(
                                            honest_scores, outlier_count
                                        ).indices
                                        predicted_global = torch.topk(
                                            scores, outlier_count
                                        ).indices
                                        signal_rows.append(
                                            {
                                                "minority_structure": structure_name,
                                                "minority_role": str(structure["role"]),
                                                "coherence": coherence,
                                                "num_honest_outliers": outlier_count,
                                                "profile": str(profile["name"]),
                                                "support_logit_fraction": fraction,
                                                "robust_reference": reference_name,
                                                "signal_seed": signal_seed,
                                                "draw": draw,
                                                "noise_permutation": permutation,
                                                "outlier_geometry": outlier_geometry,
                                                "threat": threat,
                                                "severity": severity,
                                                "clean_geometry_correlation": _corr(
                                                    honest_scores, clean_distances
                                                ),
                                                "honest_outlier_recall": float(
                                                    outliers[predicted_honest]
                                                    .double()
                                                    .mean()
                                                ),
                                                "global_honest_outlier_recall": float(
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
                                                "uniform_aggregate_error": float(
                                                    uniform_error
                                                ),
                                                "aggregate_error_ratio_to_uniform": float(
                                                    torch.linalg.vector_norm(
                                                        aggregate - target
                                                    )
                                                    / uniform_error
                                                ),
                                                "support_quantile_margin": margin,
                                                "reference_error_ratio": float(
                                                    torch.linalg.vector_norm(
                                                        reference - target
                                                    )
                                                    / dispersion
                                                ),
                                            }
                                        )

    summaries = _summaries(null_rows, signal_rows, config)
    decision = config["decision"]
    primary = next(
        row
        for row in summaries
        if row["minority_structure"] == decision["primary_structure"]
        and row["profile"] == decision["primary_profile"]
        and row["robust_reference"] == decision["primary_reference"]
    )
    _write_csv(output_dir / "null_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    _write_csv(
        output_dir / "summary.csv",
        [
            {key: value for key, value in row.items() if key != "gate_checks"}
            for row in summaries
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    machine = {
        "config": str(config_path.resolve()),
        "primary": primary,
        "fashion_mnist_authorized": bool(primary["passes_full_conditional_gates"]),
    }
    (output_dir / "decision.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        summaries=summaries,
        primary=primary,
    )
    print(json.dumps(machine, indent=2, sort_keys=True))
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dt_ldp_far/stage18_conditional_byzantine_holdout.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/dt_ldp_far/score_quality_stage18_conditional_byzantine_holdout_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage18_Conditional_Byzantine_Holdout.md",
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
