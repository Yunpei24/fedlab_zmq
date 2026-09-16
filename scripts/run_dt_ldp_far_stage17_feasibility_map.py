#!/usr/bin/env python3
"""Stage 17: preregistered reference-SNR and clean-signal feasibility map.

This synthetic audit does not use model accuracy or attacks.  It asks a
necessary question before any further Byzantine-score tuning: in which
``(n, d_score, upload-noise)`` regimes can a membership-oracle FCC reference
and the frozen covariance-calibrated novelty score recover honest geometry?

``d_score`` is an *effective informative representation dimension*.  A lower
value is therefore a structural assumption (for example, a public layer or
publicly specified subspace retaining the relevant signal).  The script does
not claim that an arbitrary random projection improves signal-to-noise ratio.
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

from algorithms.noise_aware_scores import calibrated_null_energy_scores  # noqa: E402
from robustness.aggregators import centered_clipping, clip_l2  # noqa: E402
from scripts.run_dt_ldp_far_stage14a_effective_moments_audit import (  # noqa: E402
    _clean_honest_vectors,
    _corr,
    _fcc_loo_energies,
    _finite_mean,
    _noise_scales,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), probability))


def _clip_cohorts(cohorts: torch.Tensor, radius: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(cohorts, dim=-1, keepdim=True)
    return cohorts * (float(radius) / norms.clamp_min(1e-12)).clamp(max=1.0)


def _energies(cohort: torch.Tensor, *, radius: float) -> torch.Tensor:
    dimension = cohort.shape[1]
    return _fcc_loo_energies(
        cohort,
        coordinate_order=torch.arange(dimension),
        score_dimension=dimension,
        full_dimension=dimension,
        full_radius=float(radius),
    )


def _score(
    energies: torch.Tensor,
    calibration: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    scores, _ = calibrated_null_energy_scores(
        energies,
        calibration,
        mode="moment",
        z_clip=float(config["geometry"]["moment_z_clip"]),
        individual_calibration_weight=float(
            config["geometry"]["individual_calibration_weight"]
        ),
        variance_ridge=float(config["geometry"]["variance_ridge"]),
    )
    return scores.double()


def _null_summary(scores: torch.Tensor, scales: torch.Tensor) -> dict[str, float]:
    repeated = scales[None, :].expand_as(scores)
    means = [float(scores[:, scales == tier].mean()) for tier in torch.unique(scales)]
    return {
        "null_score_noise_correlation": _corr(scores, repeated),
        "null_noise_tier_score_range": max(means) - min(means),
    }


def _cell_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        int(row["total_clients"]),
        int(row["effective_score_dimension"]),
        float(row["upload_noise_std"]),
    )


def _summarize_cells(
    null_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    null_by_cell: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(list)
    signal_by_cell: dict[tuple[int, int, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in null_rows:
        null_by_cell[_cell_key(row)].append(row)
    for row in signal_rows:
        signal_by_cell[_cell_key(row)].append(row)

    gates = config["gates"]
    output = []
    for key in sorted(signal_by_cell):
        rows = signal_by_cell[key]
        null = null_by_cell[key]
        group_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_rows[
                (
                    row["signal_seed"],
                    row["noise_permutation"],
                    row["outlier_geometry"],
                )
            ].append(row)
        group_passes = []
        for group in group_rows.values():
            group_passes.append(
                _finite_mean(
                    [float(row["membership_oracle_fcc_error_ratio"]) for row in group]
                )
                <= float(gates["membership_oracle_fcc_error_ratio_mean_max"])
                and _finite_mean(
                    [float(row["clean_geometry_correlation"]) for row in group]
                )
                >= float(gates["clean_geometry_correlation_mean_min"])
                and _finite_mean([float(row["honest_outlier_recall"]) for row in group])
                >= float(gates["honest_outlier_recall_mean_min"])
            )

        fcc_errors = [float(row["membership_oracle_fcc_error_ratio"]) for row in rows]
        summary: dict[str, Any] = {
            "total_clients": key[0],
            "num_byzantine_reserved": int(
                round(key[0] * config["grid"]["byzantine_fraction"])
            ),
            "effective_score_dimension": key[1],
            "upload_noise_std": key[2],
            "null_abs_score_noise_correlation_max": max(
                abs(float(row["null_score_noise_correlation"])) for row in null
            ),
            "null_noise_tier_score_range_max": max(
                float(row["null_noise_tier_score_range"]) for row in null
            ),
            "membership_oracle_mean_error_ratio_mean": _finite_mean(
                [float(row["membership_oracle_mean_error_ratio"]) for row in rows]
            ),
            "membership_oracle_fcc_error_ratio_mean": _finite_mean(fcc_errors),
            "membership_oracle_fcc_error_ratio_p90": _quantile(fcc_errors, 0.90),
            "public_preclip_noise_floor_ratio_mean": _finite_mean(
                [float(row["public_preclip_noise_floor_ratio"]) for row in rows]
            ),
            "clean_geometry_correlation_mean": _finite_mean(
                [float(row["clean_geometry_correlation"]) for row in rows]
            ),
            "honest_outlier_recall_mean": _finite_mean(
                [float(row["honest_outlier_recall"]) for row in rows]
            ),
            "group_pass_rate": sum(group_passes) / len(group_passes),
            "num_groups": len(group_passes),
        }
        checks = {
            "null_corr": summary["null_abs_score_noise_correlation_max"]
            <= float(gates["null_abs_score_noise_correlation_max"]),
            "null_tier": summary["null_noise_tier_score_range_max"]
            <= float(gates["null_noise_tier_score_range_max"]),
            "fcc_mean": summary["membership_oracle_fcc_error_ratio_mean"]
            <= float(gates["membership_oracle_fcc_error_ratio_mean_max"]),
            "fcc_p90": summary["membership_oracle_fcc_error_ratio_p90"]
            <= float(gates["membership_oracle_fcc_error_ratio_p90_max"]),
            "noise_floor": summary["public_preclip_noise_floor_ratio_mean"]
            <= float(gates["public_preclip_noise_floor_ratio_mean_max"]),
            "clean_corr": summary["clean_geometry_correlation_mean"]
            >= float(gates["clean_geometry_correlation_mean_min"]),
            "clean_recall": summary["honest_outlier_recall_mean"]
            >= float(gates["honest_outlier_recall_mean_min"]),
            "group_rate": summary["group_pass_rate"]
            >= float(gates["minimum_group_pass_rate"]),
        }
        summary["gate_checks"] = checks
        summary["passes_all_gates"] = all(checks.values())
        output.append(summary)
    return output


def _select_cell(
    summaries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any] | None:
    survivors = [row for row in summaries if row["passes_all_gates"]]
    if not survivors:
        return None
    canonical = float(config["decision"]["canonical_noise_std"])
    canonical_survivors = [
        row for row in survivors if math.isclose(row["upload_noise_std"], canonical)
    ]
    candidates = canonical_survivors or survivors
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["upload_noise_std"]),
            int(row["total_clients"]),
            -int(row["effective_score_dimension"]),
        ),
    )[0]


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> None:
    lines = [
        "# Stage 17 — Carte de faisabilité référence–bruit–signal",
        "",
        "## Question et règle de décision",
        "",
        "Cette étape cherche une **condition nécessaire** avant toute nouvelle "
        "optimisation adversariale : une référence FCC calculée uniquement sur "
        "les clients honnêtes doit être suffisamment précise et le score figé "
        "doit encore retrouver la géométrie honnête. Aucune accuracy et aucune "
        "attaque ne sont utilisées.",
        "",
        "La dimension indiquée est une dimension effective informative. Une "
        "réduction n'est exploitable en pratique que si un sous-espace public "
        "préserve réellement le signal. Une projection aléatoire aveugle réduit "
        "en général signal et bruit ensemble et n'est pas présentée comme un gain.",
        "",
        "La sélection est automatique : bruit le plus élevé, puis plus petit "
        "nombre de clients, puis dimension la plus grande parmi les cellules "
        "qui franchissent tous les seuils préenregistrés.",
        "",
        "## Résultats",
        "",
        "| n | d score | bruit upload | FCC oracle moy. | FCC oracle p90 | plancher bruit | corr. géométrie | rappel outliers | groupes passés | décision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            "| {n} | {d} | {noise:.3f} | {fcc:.3f} | {p90:.3f} | {floor:.3f} | {corr:.3f} | {recall:.3f} | {rate:.1%} | {passed} |".format(
                n=row["total_clients"],
                d=row["effective_score_dimension"],
                noise=row["upload_noise_std"],
                fcc=row["membership_oracle_fcc_error_ratio_mean"],
                p90=row["membership_oracle_fcc_error_ratio_p90"],
                floor=row["public_preclip_noise_floor_ratio_mean"],
                corr=row["clean_geometry_correlation_mean"],
                recall=row["honest_outlier_recall_mean"],
                rate=row["group_pass_rate"],
                passed="oui" if row["passes_all_gates"] else "non",
            )
        )
    lines.extend(["", "## Verdict", ""])
    if selected is None:
        lines.append(
            "Aucune cellule ne franchit les critères. La chaîne géométrique "
            "n'a pas de régime admissible dans cette grille et Stage 18 ne doit "
            "pas être lancé."
        )
    else:
        lines.extend(
            [
                "Une cellule est admissible selon la règle figée :",
                "",
                f"- `n={selected['total_clients']}` ;",
                f"- `d_score={selected['effective_score_dimension']}` ;",
                f"- écart-type effectif de bruit upload `{selected['upload_noise_std']:.3f}`.",
                "",
                "Ce résultat valide seulement la faisabilité pré-attaque. Il "
                "autorise Stage 18 ; il ne valide pas encore la séparation "
                "Byzantine ni un bénéfice end-to-end.",
            ]
        )
    lines.extend(
        [
            "",
            "## Traçabilité",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Détails nuls : `{(output_dir / 'null_detail.csv').resolve()}`",
            f"- Détails signal : `{(output_dir / 'signal_detail.csv').resolve()}`",
            f"- Synthèse : `{(output_dir / 'cell_summary.csv').resolve()}`",
            f"- Décision machine : `{(output_dir / 'selection.json').resolve()}`",
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
) -> dict[str, Any] | None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    grid = config["grid"]
    geometry = config["geometry"]
    randomness = config["randomness"]
    calibration_draws = int(
        calibration_draws_override or randomness["calibration_draws"]
    )
    holdout_draws = int(null_holdout_draws_override or randomness["null_holdout_draws"])
    signal_draws = int(signal_draws_override or randomness["signal_draws"])
    if calibration_draws < 20 or holdout_draws < 20 or signal_draws < 1:
        raise ValueError("Stage 17 requires >=20 null draws and >=1 signal draw")

    null_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    for n in [int(value) for value in grid["total_clients"]]:
        f = int(round(n * float(grid["byzantine_fraction"])))
        n_honest = n - f
        outlier_count = int(round(n * float(grid["honest_outlier_fraction_of_total"])))
        if not 0 < f < n_honest or not 0 < outlier_count < n_honest:
            raise ValueError("Invalid Byzantine/outlier count in Stage 17 grid")
        for dimension in [int(value) for value in grid["effective_score_dimensions"]]:
            for noise_std in [float(value) for value in grid["upload_noise_stds"]]:
                for permutation_index, permutation in enumerate(
                    [str(value) for value in grid["noise_permutations"]]
                ):
                    scales = _noise_scales(
                        [float(value) for value in grid["public_noise_scales"]],
                        permutation,
                        n_honest,
                    )
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(randomness["calibration_seed"])
                        + n * 1_000_000
                        + dimension * 1_000
                        + int(round(noise_std * 1_000_000))
                        + permutation_index
                    )
                    calibration = (
                        noise_std
                        * scales[None, :, None]
                        * torch.randn(
                            calibration_draws,
                            n_honest,
                            dimension,
                            generator=generator,
                            dtype=torch.float64,
                        )
                    )
                    calibration = _clip_cohorts(
                        calibration, float(geometry["server_clip_norm"])
                    )
                    calibration_energy = torch.stack(
                        [
                            _energies(row, radius=float(geometry["fcc_radius"]))
                            for row in calibration
                        ]
                    )
                    holdout_generator = torch.Generator(device="cpu").manual_seed(
                        int(randomness["null_holdout_seed"])
                        + n * 1_000_000
                        + dimension * 1_000
                        + int(round(noise_std * 1_000_000))
                        + permutation_index
                    )
                    holdout = (
                        noise_std
                        * scales[None, :, None]
                        * torch.randn(
                            holdout_draws,
                            n_honest,
                            dimension,
                            generator=holdout_generator,
                            dtype=torch.float64,
                        )
                    )
                    holdout = _clip_cohorts(
                        holdout, float(geometry["server_clip_norm"])
                    )
                    holdout_scores = torch.stack(
                        [
                            _score(
                                _energies(row, radius=float(geometry["fcc_radius"])),
                                calibration_energy,
                                config,
                            )
                            for row in holdout
                        ]
                    )
                    null_rows.append(
                        {
                            "total_clients": n,
                            "effective_score_dimension": dimension,
                            "upload_noise_std": noise_std,
                            "noise_permutation": permutation,
                            **_null_summary(holdout_scores, scales),
                        }
                    )

                    for signal_seed in [
                        int(value) for value in randomness["signal_seeds"]
                    ]:
                        for geometry_index, outlier_geometry in enumerate(
                            [str(value) for value in grid["outlier_geometries"]]
                        ):
                            clean, outliers = _clean_honest_vectors(
                                signal_seed,
                                n_honest=n_honest,
                                dimension=dimension,
                                num_outliers=outlier_count,
                                outlier_geometry=outlier_geometry,
                            )
                            clean = clip_l2(clean, float(geometry["server_clip_norm"]))
                            target = clean.mean(dim=0)
                            dispersion = torch.sqrt(
                                torch.linalg.vector_norm(clean - target, dim=1)
                                .square()
                                .mean()
                            ).clamp_min(1e-12)
                            clean_distances = _energies(
                                clean, radius=float(geometry["fcc_radius"])
                            ).sqrt()
                            noise_floor = math.sqrt(
                                dimension
                                * float((noise_std * scales).square().sum())
                                / n_honest**2
                            ) / float(dispersion)
                            for draw in range(signal_draws):
                                signal_generator = torch.Generator(
                                    device="cpu"
                                ).manual_seed(
                                    signal_seed * 10_000_000
                                    + n * 100_000
                                    + dimension * 100
                                    + permutation_index * 10
                                    + geometry_index * 1_000_000
                                    + draw
                                )
                                observed = clean + noise_std * scales[
                                    :, None
                                ] * torch.randn(
                                    clean.shape,
                                    generator=signal_generator,
                                    dtype=clean.dtype,
                                )
                                observed = clip_l2(
                                    observed, float(geometry["server_clip_norm"])
                                )
                                scores = _score(
                                    _energies(
                                        observed,
                                        radius=float(geometry["fcc_radius"]),
                                    ),
                                    calibration_energy,
                                    config,
                                )
                                oracle_mean = observed.mean(dim=0)
                                oracle_fcc = centered_clipping(
                                    observed,
                                    anchor=torch.zeros(dimension, dtype=torch.float64),
                                    tau=float(geometry["fcc_radius"]),
                                )
                                predicted = torch.topk(scores, outlier_count).indices
                                signal_rows.append(
                                    {
                                        "total_clients": n,
                                        "num_byzantine_reserved": f,
                                        "num_honest": n_honest,
                                        "num_honest_outliers": outlier_count,
                                        "effective_score_dimension": dimension,
                                        "upload_noise_std": noise_std,
                                        "noise_permutation": permutation,
                                        "signal_seed": signal_seed,
                                        "draw": draw,
                                        "outlier_geometry": outlier_geometry,
                                        "membership_oracle_mean_error_ratio": float(
                                            torch.linalg.vector_norm(
                                                oracle_mean - target
                                            )
                                            / dispersion
                                        ),
                                        "membership_oracle_fcc_error_ratio": float(
                                            torch.linalg.vector_norm(
                                                oracle_fcc - target
                                            )
                                            / dispersion
                                        ),
                                        "public_preclip_noise_floor_ratio": noise_floor,
                                        "clean_geometry_correlation": _corr(
                                            scores, clean_distances
                                        ),
                                        "honest_outlier_recall": float(
                                            outliers[predicted].double().mean()
                                        ),
                                    }
                                )

    summaries = _summarize_cells(null_rows, signal_rows, config)
    selected = _select_cell(summaries, config)
    _write_csv(output_dir / "null_detail.csv", null_rows)
    _write_csv(output_dir / "signal_detail.csv", signal_rows)
    _write_csv(
        output_dir / "cell_summary.csv",
        [
            {key: value for key, value in row.items() if key != "gate_checks"}
            for row in summaries
        ],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "config": str(config_path.resolve()),
        "num_cells": len(summaries),
        "num_survivors": sum(bool(row["passes_all_gates"]) for row in summaries),
        "selected_cell": selected,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"config": str(config_path.resolve()), "cell_summaries": summaries},
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
        selected=selected,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/dt_ldp_far/stage17_reference_snr_feasibility_map.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/dt_ldp_far/score_quality_stage17_reference_snr_feasibility_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/DT_LDP_FAR_Stage17_Reference_SNR_Feasibility_Map.md",
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
