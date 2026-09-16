#!/usr/bin/env python3
"""G0e: cross-fitted, FCC-anchored Gaussian-aware reference audit.

G0e is deliberately a reference-only experiment.  It uses an independent
offline calibration population to freeze client/block radii for the exact
``Clip_U -> FCC_{-i}`` residual geometry.  At evaluation time the deployed
estimator is a bounded correction around the ordinary FCC pilot; the
leave-one-out references are diagnostics only and never enter the optimizer.

There is exactly one candidate.  Its false-tail probability, regularisation,
influence cap, blend coefficient and public iteration count are derived from
pre-registered public budgets.  Development is a hard gate: holdout files are
neither generated nor read unless every development criterion passes.

Production execution is MPS-only.  Unit tests may call pure helpers on CPU.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_crossfit_bounded_correction,
)
from robustness.aggregators import (  # noqa: E402
    centered_clipping,
    clip_l2,
    geometric_median,
    trimmed_mean,
)
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

COMPARATORS = ("uniform_mean", "fcc", "rfa", "trimmed_mean")
CANDIDATE = "g0e"
PRIMARY_FIELDS = (
    "reference_error",
    "reference_error_ratio_to_uniform",
    "reference_error_ratio_to_fcc",
)


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
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


def _holdout_artifacts(output_dir: Path) -> list[Path]:
    """Find every holdout-named artifact, including nested checkpoints."""

    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.rglob("*")
        if any("holdout" in part.lower() for part in path.relative_to(output_dir).parts)
    )


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for width in block_sizes:
        result.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(result)


def _validate_config(config: dict[str, Any]) -> None:
    contract = config["scientific_contract"]
    required_true = (
        "reference_only",
        "inverse_variance_estimand_forbidden",
        "correction_is_bounded_around_fcc",
        "calibration_is_offline_and_cross_fitted",
        "crossfit_references_are_not_online_solver_inputs",
        "parameters_are_derived_from_public_budgets",
        "holdout_blocked_until_development_passes",
    )
    if any(not bool(contract[key]) for key in required_true):
        raise ValueError("Every G0e scientific-contract safeguard must be true")
    if bool(contract["far_weights_used"]) or bool(contract["accuracy_used"]):
        raise ValueError("G0e is reference-only and cannot use FAR weights or accuracy")
    if bool(contract["holdout_used_for_selection"]):
        raise ValueError("Holdout use for G0e selection is forbidden")
    if contract["estimand"] != "equal_client_mean_of_clean_honest_updates":
        raise ValueError("G0e must retain the equal-client estimand")
    if contract["base_reference"] != "fcc":
        raise ValueError("The G0e correction must be anchored at FCC")
    if contract["calibration_chain"] != "server_clip_then_leave_one_out_fcc":
        raise ValueError("G0e calibration must reproduce Clip_U -> FCC_-i")
    if (
        contract["calibration_centre_law"]
        != "same_preregistered_population_centre_law_and_public_anchor_rule"
    ):
        raise ValueError("Calibration and evaluation must use the same centre law")

    execution = config["execution"]
    if execution["required_device"] != "mps" or bool(execution["allow_cpu_fallback"]):
        raise ValueError("G0e is MPS-only and forbids CPU fallback")
    if str(config["selection"]["candidate_grid"]) != "forbidden":
        raise ValueError("G0e cannot tune a candidate grid")
    if bool(config["selection"]["holdout_used_for_selection"]):
        raise ValueError("G0e cannot use holdout for selection")

    random = config["randomness"]
    folds = [[int(value) for value in fold] for fold in random["calibration_folds"]]
    if len(folds) != 2 or any(len(fold) < 2 for fold in folds):
        raise ValueError("G0e requires exactly two calibration folds of >=2 seeds")
    groups = (
        [value for fold in folds for value in fold],
        [int(value) for value in random["development_seeds"]],
        [int(value) for value in random["holdout_seeds"]],
    )
    fresh = [value for group in groups for value in group]
    if len(fresh) != len(set(fresh)):
        raise ValueError("Calibration, development and holdout seeds must be disjoint")
    if set(fresh) & set(int(value) for value in config["excluded_prior_seeds"]):
        raise ValueError("G0e reuses a seed inspected by G0--G0d")
    if len(groups[1]) < 3 or len(groups[2]) < 5:
        raise ValueError("Need >=3 development and >=5 holdout seeds")
    if int(random["calibration_draws_per_seed"]) < 2:
        raise ValueError("Calibration needs at least two draws per seed")
    if int(random["replace_one_trials_per_seed_cell"]) < 1:
        raise ValueError("Replace-one audit needs at least one trial per seed/cell")

    n = int(config["cohort"]["num_clients"])
    b = int(config["cohort"]["num_byzantine"])
    blocks = [int(value) for value in config["cohort"]["block_sizes"]]
    if not 0 < b < n / 2:
        raise ValueError("G0e requires 0 < num_byzantine < n/2")
    if sum(blocks) != int(config["cohort"]["dimension"]):
        raise ValueError("block_sizes must sum to dimension")
    trim = int(config["references"]["trimmed_mean"]["trim_count"])
    if trim < 0 or 2 * trim >= n:
        raise ValueError("Invalid trimmed-mean trim_count")

    budgets = config["references"]["g0e_public_budgets"]
    if not 0.0 < float(budgets["regular_honest_false_tail_rate"]) < 0.5:
        raise ValueError("false-tail budget must lie in (0,0.5)")
    if not 0.0 < float(budgets["target_solver_contraction"]) < 1.0:
        raise ValueError("target_solver_contraction must lie in (0,1)")
    for key in (
        "correction_contamination_budget_fraction_of_fcc",
        "correction_radius_budget_fraction_of_fcc_radius",
        "solver_error_tolerance",
        "replace_one_bound_max",
        "variance_floor",
    ):
        if not math.isfinite(float(budgets[key])) or float(budgets[key]) <= 0.0:
            raise ValueError(f"{key} must be finite and positive")

    expected_threats = {"none", "alie", "ipm", "bitflip_x10", "model_replacement"}
    if set(str(value) for value in config["threats"]["names"]) != expected_threats:
        raise ValueError("All five pre-registered threat cells are required")
    separated = set(str(value) for value in config["threats"]["separated_for_gates"])
    evasive = set(str(value) for value in config["threats"]["evasive_controls"])
    if separated != {"ipm", "bitflip_x10", "model_replacement"} or evasive != {"alie"}:
        raise ValueError("The separated/evasive attack split is frozen")


def _derive_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive every G0e solver parameter from public scientific budgets."""

    n = int(config["cohort"]["num_clients"])
    b = int(config["cohort"]["num_byzantine"])
    num_blocks = len(config["cohort"]["block_sizes"])
    tau = float(config["references"]["fcc"]["radius"])
    budgets = config["references"]["g0e_public_budgets"]
    contraction = float(budgets["target_solver_contraction"])
    gamma = 0.5 * (1.0 / contraction - 1.0)
    fcc_contamination_bound = 2.0 * float(b) * tau / float(n)
    unmixed_bias = (
        float(budgets["correction_contamination_budget_fraction_of_fcc"])
        * fcc_contamination_bound
    )
    # Under replacement contamination, b client terms may each change by
    # 2G.  Thus 2*b*G/(gamma*n) <= B_byz defines the correction-only cap G.
    influence_cap_total = unmixed_bias * gamma * float(n) / (2.0 * float(b))
    correction_budget = (
        float(budgets["correction_radius_budget_fraction_of_fcc_radius"]) * tau
    )
    tolerance = float(budgets["solver_error_tolerance"])
    gradient_ratio = (1.0 + gamma) * influence_cap_total / (gamma * tolerance)
    if gradient_ratio <= 1.0:
        num_steps = 1
    else:
        num_steps = max(
            1,
            int(math.ceil(math.log(gradient_ratio) / math.log(1.0 / contraction))),
        )
    # The symmetric optimal step can alternate signs, so the final public
    # iteration count is rounded to the next even integer.
    if num_steps % 2:
        num_steps += 1
    finite_fraction = 1.0 - contraction**num_steps
    beta_asymptotic = min(1.0, gamma * correction_budget / influence_cap_total)
    beta = min(
        1.0,
        gamma * correction_budget / (influence_cap_total * finite_fraction),
    )
    if beta <= 0.0:
        raise ValueError("Derived beta must be positive")
    block_cap = influence_cap_total / math.sqrt(float(num_blocks))
    pilot_bound = 2.0 * tau / float(n)
    loo_bound = 2.0 * tau / float(n - 1)
    eta = 2.0 / (1.0 + 2.0 * gamma)
    geometric_sum = finite_fraction / (1.0 - contraction)
    direct_finite = beta * 2.0 * eta * influence_cap_total * geometric_sum / float(n)
    finite_replace_one = pilot_bound + direct_finite
    return {
        "derivation_version": "g0e_public_budget_equations_v1",
        "false_tail_probability": float(budgets["regular_honest_false_tail_rate"]),
        "conformal_quantile_probability": 1.0
        - float(budgets["regular_honest_false_tail_rate"]),
        "regularization": gamma,
        "target_solver_contraction": contraction,
        "influence_cap_total": influence_cap_total,
        "influence_cap_per_block": [block_cap] * num_blocks,
        "maximum_unmixed_correction_byzantine_bias": unmixed_bias,
        "correction_contamination_budget_fraction_of_fcc": float(
            budgets["correction_contamination_budget_fraction_of_fcc"]
        ),
        "correction_radius_budget_fraction_of_fcc_radius": float(
            budgets["correction_radius_budget_fraction_of_fcc_radius"]
        ),
        "fcc_replacement_contamination_bound": fcc_contamination_bound,
        "raw_correction_replacement_contamination_bound": unmixed_bias,
        "blended_correction_replacement_contamination_bound": (
            beta * finite_fraction * unmixed_bias
        ),
        "total_reference_replacement_contamination_bound": (
            fcc_contamination_bound + beta * finite_fraction * unmixed_bias
        ),
        "correction_budget": correction_budget,
        "beta": beta,
        "beta_finite_solver": beta,
        "beta_asymptotic_control": beta_asymptotic,
        "finite_solver_geometric_fraction": finite_fraction,
        "num_steps": num_steps,
        "solver_error_tolerance": tolerance,
        "solver_distance_to_exact_bound": (
            contraction**num_steps * influence_cap_total / gamma
        ),
        "solver_gradient_residual_bound": (
            (1.0 + gamma) * contraction**num_steps * influence_cap_total / gamma
        ),
        "finite_correction_norm_bound": (
            beta * influence_cap_total * finite_fraction / gamma
        ),
        "pilot_replace_one_bound": pilot_bound,
        "leave_one_out_fcc_replace_one_bound": loo_bound,
        "crossfit_to_full_fcc_margin_per_block": pilot_bound,
        "finite_solver_replace_one_bound": finite_replace_one,
        "replace_one_bound_budget": float(budgets["replace_one_bound_max"]),
    }


def _conformal_quantile(values: torch.Tensor, miscoverage: float) -> tuple[float, int]:
    """Finite-sample split-conformal upper quantile and one-based rank."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("values must be a non-empty vector")
    if not 0.0 < float(miscoverage) < 1.0:
        raise ValueError("miscoverage must lie strictly in (0,1)")
    ordered = torch.sort(values).values
    rank = min(
        int(ordered.numel()),
        int(math.ceil((int(ordered.numel()) + 1) * (1.0 - miscoverage))),
    )
    return float(ordered[rank - 1].item()), rank


def _fcc_leave_one_out(
    vectors: torch.Tensor, *, anchor: torch.Tensor, radius: float
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError("FCC leave-one-out requires a matrix with n>=2")
    indices = torch.arange(vectors.shape[0], device=vectors.device)
    return torch.stack(
        [
            centered_clipping(vectors[indices != index], anchor=anchor, tau=radius)
            for index in range(int(vectors.shape[0]))
        ]
    )


def _null_sample(
    config: dict[str, Any],
    *,
    seed: int,
    draw: int,
    regime: Mapping[str, Any],
    permutation: str,
    context: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    # One public base seed may supply many Monte-Carlo draws, but each draw
    # must have its own centre and anchor to be one independent conformal
    # cohort.  Hashing (seed, draw) preserves the evaluation law marginally
    # while avoiding the shared-centre dependence of _population_geometry.
    cohort_seed = oracle._seed("g0e-calibration-cohort", seed, draw)
    clean, outlier_mask, _, anchor = oracle._honest_clean_vectors(
        config,
        seed=cohort_seed,
        draw=0,
        geometry=str(context["geometry"]),
        include_outliers=bool(context["include_outliers"]),
    )
    # The mask is used only to exclude bounded honest outliers from the
    # regular false-tail calibration; their context still moves FCC_-i.
    variances, _ = oracle._noise_variances(config, dict(regime), permutation)
    observed = oracle._add_private_noise(
        clean,
        variances,
        blocks,
        seed=cohort_seed,
        draw=0,
        regime=str(regime["name"]),
        permutation=permutation,
        geometry="g0e-null",
        pair_permutations=bool(
            config["randomness"]["pair_noise_across_tier_permutations"]
        ),
    )
    bounded = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
    loo = _fcc_leave_one_out(
        bounded,
        anchor=anchor,
        radius=float(config["references"]["fcc"]["radius"]),
    )
    return bounded, loo, variances, anchor, outlier_mask


def _fit_crossfit_reference_variance(
    config: dict[str, Any], seeds: Sequence[int]
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Fit per-client/block FCC_-i variance using only the supplied fold."""

    slices = _block_slices(config["cohort"]["block_sizes"])
    draws = int(config["randomness"]["calibration_draws_per_seed"])
    result: dict[tuple[str, str, str], torch.Tensor] = {}
    for regime in config["privacy_noise"]["regimes"]:
        name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            for context in config["randomness"]["calibration_contexts"]:
                context_name = str(context["name"])
                samples: list[torch.Tensor] = []
                for seed in seeds:
                    for draw in range(draws):
                        _, loo, _, _, _ = _null_sample(
                            config,
                            seed=int(seed),
                            draw=draw,
                            regime=regime,
                            permutation=permutation,
                            context=context,
                        )
                        samples.append(loo)
                stacked = torch.stack(samples)  # draws x n x d
                centred = stacked - stacked.mean(dim=0, keepdim=True)
                cell = torch.empty(
                    (stacked.shape[1], len(slices)),
                    dtype=stacked.dtype,
                    device=stacked.device,
                )
                for block_index, block_slice in enumerate(slices):
                    cell[:, block_index] = (
                        centred[:, :, block_slice].square().mean(dim=(0, 2))
                    )
                result[(name, permutation, context_name)] = cell
    return result


def _calibrate(
    config: dict[str, Any], derived: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cross-fit reference variance and exact-chain conformal tail radii."""

    folds = [
        [int(value) for value in fold]
        for fold in config["randomness"]["calibration_folds"]
    ]
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    slices = _block_slices(block_sizes)
    heterogeneity = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=oracle._RUNTIME_DTYPE,
        device=oracle._RUNTIME_DEVICE,
    ).square()
    floor = float(config["references"]["g0e_public_budgets"]["variance_floor"])
    draws = int(config["randomness"]["calibration_draws_per_seed"])
    standardized_by_fold_stratum: dict[
        tuple[int, str, str, str, float, int], list[float]
    ] = defaultdict(list)
    provisional: list[dict[str, Any]] = []
    # Each scoring fold uses nuisance scales fitted on the disjoint fold.  We
    # retain both public fits for deployment below instead of refitting on the
    # union, which would otherwise feed each calibration observation back into
    # the scale used at deployment.
    fitted_reference_variances = [
        _fit_crossfit_reference_variance(config, fold) for fold in folds
    ]
    for score_fold in range(2):
        fit_fold = 1 - score_fold
        reference_variances = fitted_reference_variances[fit_fold]
        for regime in config["privacy_noise"]["regimes"]:
            regime_name = str(regime["name"])
            for permutation_value in regime["permutations"]:
                permutation = str(permutation_value)
                _, tiers = oracle._noise_variances(config, regime, permutation)
                tier_values = sorted(
                    set(float(value) for value in tiers.cpu().tolist())
                )
                for context in config["randomness"]["calibration_contexts"]:
                    context_name = str(context["name"])
                    ref_var = reference_variances[
                        (regime_name, permutation, context_name)
                    ]
                    for seed in folds[score_fold]:
                        for draw in range(draws):
                            bounded, loo, variances, _, outlier_mask = _null_sample(
                                config,
                                seed=seed,
                                draw=draw,
                                regime=regime,
                                permutation=permutation,
                                context=context,
                            )
                            scale = (
                                variances + heterogeneity[None, :] + ref_var + floor
                            ).sqrt()
                            regular = ~outlier_mask
                            for tier in tier_values:
                                tier_mask = torch.isclose(
                                    tiers,
                                    torch.tensor(
                                        tier, dtype=tiers.dtype, device=tiers.device
                                    ),
                                )
                                eligible = torch.where(regular & tier_mask)[0]
                                if eligible.numel() < 1:
                                    raise RuntimeError(
                                        "A calibration stratum has no regular probe"
                                    )
                                public_hash = oracle._seed(
                                    "g0e-public-probe",
                                    score_fold,
                                    seed,
                                    draw,
                                    regime_name,
                                    permutation,
                                    context_name,
                                    tier,
                                )
                                client = int(
                                    eligible[public_hash % int(eligible.numel())].item()
                                )
                                for block_index, block_slice in enumerate(slices):
                                    residual = torch.linalg.vector_norm(
                                        bounded[client, block_slice]
                                        - loo[client, block_slice]
                                    )
                                    standardized = float(
                                        (residual / scale[client, block_index]).item()
                                    )
                                    key = (
                                        score_fold,
                                        context_name,
                                        regime_name,
                                        permutation,
                                        tier,
                                        block_index,
                                    )
                                    standardized_by_fold_stratum[key].append(
                                        standardized
                                    )
                                    provisional.append(
                                        {
                                            "score_fold": score_fold,
                                            "variance_fit_fold": fit_fold,
                                            "seed": seed,
                                            "draw": draw,
                                            "calibration_context": context_name,
                                            "noise_regime": regime_name,
                                            "noise_permutation": permutation,
                                            "noise_tier": tier,
                                            "probe_client": client,
                                            "block": block_index,
                                            "standardized_crossfit_residual": standardized,
                                        }
                                    )

    miscoverage = float(derived["false_tail_probability"])
    stratum_quantiles: dict[
        tuple[int, str, str, str, float, int], tuple[float, int, int]
    ] = {}
    for key, values in standardized_by_fold_stratum.items():
        tensor = torch.tensor(
            values, dtype=oracle._RUNTIME_DTYPE, device=oracle._RUNTIME_DEVICE
        )
        threshold, rank = _conformal_quantile(tensor, miscoverage)
        stratum_quantiles[key] = (threshold, rank, len(values))
    thresholds = [
        max(
            threshold
            for key, (threshold, _, _) in stratum_quantiles.items()
            if int(key[-1]) == block
        )
        for block in range(len(block_sizes))
    ]
    rows: list[dict[str, Any]] = []
    for row in provisional:
        block = int(row["block"])
        key = (
            int(row["score_fold"]),
            str(row["calibration_context"]),
            str(row["noise_regime"]),
            str(row["noise_permutation"]),
            float(row["noise_tier"]),
            block,
        )
        stratum_threshold, rank, pool_size = stratum_quantiles[key]
        common_threshold = thresholds[block]
        rows.append(
            {
                **row,
                "conformal_miscoverage": miscoverage,
                "stratum_conformal_rank_one_based": rank,
                "stratum_conformal_pool_size": pool_size,
                "stratum_standardized_threshold": stratum_threshold,
                "deployed_common_standardized_threshold": common_threshold,
                "statistical_tail": bool(
                    float(row["standardized_crossfit_residual"]) > common_threshold
                ),
            }
        )

    full_ref_variances: dict[tuple[str, str], torch.Tensor] = {}
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            # Neither the scoring fold nor the evaluation context is observed
            # by the online mechanism.  Deploy the public elementwise maximum
            # over both disjoint nuisance fits and all pre-registered null
            # contexts.  This is no smaller than the scale used for any
            # calibration probe and avoids a post-crossfit nuisance refit.
            cells = [
                fitted_reference_variances[fit_fold][
                    (regime_name, permutation, str(context["name"]))
                ]
                for fit_fold in range(2)
                for context in config["randomness"]["calibration_contexts"]
            ]
            full_ref_variances[(regime_name, permutation)] = (
                torch.stack(cells).max(dim=0).values
            )
    artifact = {
        "calibration_protocol": (
            "two_fold_crossfit_exact_ClipU_FCC_minus_i_one_public_probe_"
            "per_independent_cohort_and_stratum"
        ),
        "calibration_and_evaluation_share_preregistered_population_centre_law_and_public_anchor_rule": True,
        "calibration_seeds_and_draws_are_disjoint_from_evaluation": True,
        "folds": folds,
        "standardized_thresholds": thresholds,
        "common_threshold_is_maximum_over_folds_and_preregistered_strata": True,
        "deployment_reference_variance_is_maximum_over_folds_and_contexts": True,
        "stratum_quantiles": [
            {
                "score_fold": key[0],
                "calibration_context": key[1],
                "noise_regime": key[2],
                "noise_permutation": key[3],
                "noise_tier": key[4],
                "block": key[5],
                "threshold": value[0],
                "rank_one_based": value[1],
                "pool_size": value[2],
            }
            for key, value in sorted(stratum_quantiles.items())
        ],
        "calibration_tail_rate_by_stratum": [
            {
                "score_fold": key[0],
                "calibration_context": key[1],
                "noise_regime": key[2],
                "noise_permutation": key[3],
                "noise_tier": key[4],
                "block": key[5],
                "probe_count": len(values),
                "tail_count_against_common_threshold": sum(
                    float(value) > thresholds[int(key[5])] for value in values
                ),
                "tail_rate_against_common_threshold": _finite_mean(
                    float(float(value) > thresholds[int(key[5])]) for value in values
                ),
            }
            for key, values in sorted(standardized_by_fold_stratum.items())
        ],
        "false_tail_probability": miscoverage,
        "crossfit_to_full_fcc_margin_per_block": float(
            derived["crossfit_to_full_fcc_margin_per_block"]
        ),
        "reference_variance_by_cell": {
            f"{regime}|{permutation}": tensor.detach().cpu().tolist()
            for (regime, permutation), tensor in full_ref_variances.items()
        },
        "calibration_tail_rate_by_block": [
            _finite_mean(
                float(row["statistical_tail"])
                for row in rows
                if int(row["block"]) == block
            )
            for block in range(len(block_sizes))
        ],
        "calibration_tail_rate_by_context": {
            str(context["name"]): _finite_mean(
                float(row["statistical_tail"])
                for row in rows
                if row["calibration_context"] == str(context["name"])
            )
            for context in config["randomness"]["calibration_contexts"]
        },
    }
    return artifact, rows


def _radii_for_cell(
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    regime: str,
    permutation: str,
    noise_variances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_var = torch.tensor(
        calibration["reference_variance_by_cell"][f"{regime}|{permutation}"],
        dtype=noise_variances.dtype,
        device=noise_variances.device,
    )
    heterogeneity = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=noise_variances.dtype,
        device=noise_variances.device,
    ).square()
    floor = float(config["references"]["g0e_public_budgets"]["variance_floor"])
    thresholds = torch.tensor(
        calibration["standardized_thresholds"],
        dtype=noise_variances.dtype,
        device=noise_variances.device,
    )
    statistical = (
        noise_variances + heterogeneity[None, :] + ref_var + floor
    ).sqrt() * thresholds[None, :]
    deployed = statistical + float(derived["crossfit_to_full_fcc_margin_per_block"])
    return statistical, deployed


def _comparator_reference(
    name: str,
    vectors: torch.Tensor,
    config: Mapping[str, Any],
    *,
    anchor: torch.Tensor,
) -> torch.Tensor:
    if name == "uniform_mean":
        return vectors.mean(dim=0)
    if name == "fcc":
        return centered_clipping(
            vectors,
            anchor=anchor,
            tau=float(config["references"]["fcc"]["radius"]),
        )
    if name == "rfa":
        settings = config["references"]["rfa"]
        return geometric_median(
            vectors,
            max_iter=int(settings["max_iter"]),
            tol=float(settings["tolerance"]),
            smoothing=float(settings["smoothing"]),
        )
    if name == "trimmed_mean":
        return trimmed_mean(
            vectors, f=int(config["references"]["trimmed_mean"]["trim_count"])
        )
    raise ValueError(f"Unknown comparator {name!r}")


def _masked_rate(flags: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    return float(flags[mask].to(torch.float32).mean().item())


def _masked_rate_by_block(flags: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Return one client-averaged rate per block without pooling blocks."""

    if flags.ndim != 2 or mask.ndim != 1 or flags.shape[0] != mask.shape[0]:
        raise ValueError("Block flags and client mask have incompatible shapes")
    if not bool(mask.any().item()):
        return [float("nan")] * int(flags.shape[1])
    selected = flags[mask].to(torch.float32)
    return [float(value) for value in selected.mean(dim=0).tolist()]


def _regular_tier_block_rates(
    statistical_tail: torch.Tensor,
    cap_active: torch.Tensor,
    regular_honest: torch.Tensor,
    noise_tiers: torch.Tensor,
) -> list[dict[str, Any]]:
    """Stratify regular-client diagnostics by public noise tier and block."""

    if noise_tiers.ndim != 1 or noise_tiers.shape[0] != regular_honest.shape[0]:
        raise ValueError("noise_tiers must contain one public tier per client")
    result: list[dict[str, Any]] = []
    tier_values = sorted(
        set(float(value) for value in noise_tiers.detach().cpu().tolist())
    )
    for tier in tier_values:
        tier_tensor = torch.tensor(
            tier, dtype=noise_tiers.dtype, device=noise_tiers.device
        )
        tier_mask = torch.isclose(noise_tiers, tier_tensor)
        eligible = regular_honest & tier_mask
        if not bool(eligible.any().item()):
            continue
        tail_rates = _masked_rate_by_block(statistical_tail, eligible)
        cap_rates = _masked_rate_by_block(cap_active, eligible)
        for block in range(int(statistical_tail.shape[1])):
            result.append(
                {
                    "noise_tier": tier,
                    "block": block,
                    "regular_client_count": int(eligible.sum().item()),
                    "regular_honest_statistical_tail_rate": tail_rates[block],
                    "regular_honest_influence_cap_activation_rate": cap_rates[block],
                }
            )
    if not result:
        raise RuntimeError("No regular-client noise-tier stratum is available")
    return result


def _candidate_diagnostics(
    *,
    vectors: torch.Tensor,
    crossfit_references: torch.Tensor,
    statistical_radii: torch.Tensor,
    deployed_radii: torch.Tensor,
    diagnostics: Mapping[str, Any],
    regular_honest: torch.Tensor,
    honest_outliers: torch.Tensor,
    byzantine: torch.Tensor,
    noise_tiers: torch.Tensor,
    influence_caps: Sequence[float],
    block_sizes: Sequence[int],
) -> dict[str, Any]:
    """Compute mask-specific statistics without conflating tail and cap."""

    slices = _block_slices(block_sizes)
    statistical_tail = torch.zeros_like(statistical_radii, dtype=torch.bool)
    for block_index, block_slice in enumerate(slices):
        residual_norm = torch.linalg.vector_norm(
            vectors[:, block_slice] - crossfit_references[:, block_slice], dim=1
        )
        statistical_tail[:, block_index] = (
            residual_norm > statistical_radii[:, block_index]
        )

    cap_active_blocks = torch.tensor(
        diagnostics["influence_cap_active_client_blocks"],
        dtype=torch.bool,
        device=vectors.device,
    )
    if cap_active_blocks.shape != statistical_tail.shape:
        raise RuntimeError("G0e diagnostic masks have an invalid shape")
    cap_active_clients = cap_active_blocks.any(dim=1)

    # Reconstruct the final Huber influence norms solely for the influence
    # share diagnostic.  This does not alter the returned estimator.
    point = torch.tensor(
        diagnostics["pre_blend_reference"],
        dtype=vectors.dtype,
        device=vectors.device,
    )
    caps = torch.tensor(influence_caps, dtype=vectors.dtype, device=vectors.device)
    influence_by_block = torch.zeros(
        (vectors.shape[0], len(slices)),
        dtype=vectors.dtype,
        device=vectors.device,
    )
    for block_index, block_slice in enumerate(slices):
        norm = torch.linalg.vector_norm(
            point[None, block_slice] - vectors[:, block_slice], dim=1
        )
        effective = torch.minimum(
            deployed_radii[:, block_index],
            torch.full_like(deployed_radii[:, block_index], caps[block_index]),
        )
        influence_by_block[:, block_index] = torch.minimum(norm, effective)
    influence_norm = influence_by_block.square().sum(dim=1).sqrt()
    total = float(influence_norm.sum().item())
    byzantine_share = (
        float(influence_norm[byzantine].sum().item()) / total
        if bool(byzantine.any().item()) and total > 0.0
        else float("nan")
    )
    byzantine_share_by_block = []
    for block_index in range(len(slices)):
        block_total = float(influence_by_block[:, block_index].sum().item())
        byzantine_share_by_block.append(
            float(influence_by_block[byzantine, block_index].sum().item()) / block_total
            if bool(byzantine.any().item()) and block_total > 0.0
            else float("nan")
        )
    regular_tail_by_block = _masked_rate_by_block(statistical_tail, regular_honest)
    regular_cap_by_block = _masked_rate_by_block(cap_active_blocks, regular_honest)
    outlier_tail_by_block = _masked_rate_by_block(statistical_tail, honest_outliers)
    outlier_cap_by_block = _masked_rate_by_block(cap_active_blocks, honest_outliers)
    byzantine_cap_by_block = _masked_rate_by_block(cap_active_blocks, byzantine)
    regular_tier_block = _regular_tier_block_rates(
        statistical_tail,
        cap_active_blocks,
        regular_honest,
        noise_tiers,
    )
    return {
        "regular_honest_statistical_tail_rate": _masked_rate(
            statistical_tail, regular_honest
        ),
        "regular_honest_influence_cap_activation_rate": _masked_rate(
            cap_active_blocks, regular_honest
        ),
        "honest_outlier_statistical_tail_rate": _masked_rate(
            statistical_tail, honest_outliers
        ),
        "honest_outlier_influence_cap_activation_rate": _masked_rate(
            cap_active_blocks, honest_outliers
        ),
        "honest_outlier_not_cap_limited_rate": (
            1.0 - _masked_rate(cap_active_clients, honest_outliers)
            if bool(honest_outliers.any().item())
            else float("nan")
        ),
        "byzantine_influence_cap_activation_rate": _masked_rate(
            cap_active_blocks, byzantine
        ),
        "byzantine_influence_share": byzantine_share,
        "regular_honest_statistical_tail_rate_by_block": regular_tail_by_block,
        "regular_honest_influence_cap_activation_rate_by_block": (regular_cap_by_block),
        "honest_outlier_statistical_tail_rate_by_block": outlier_tail_by_block,
        "honest_outlier_influence_cap_activation_rate_by_block": (outlier_cap_by_block),
        "honest_outlier_not_cap_limited_rate_by_block": [
            1.0 - value for value in outlier_cap_by_block
        ],
        "byzantine_influence_cap_activation_rate_by_block": (byzantine_cap_by_block),
        "byzantine_influence_share_by_block": byzantine_share_by_block,
        "regular_honest_diagnostics_by_noise_tier_block": regular_tier_block,
    }


def _evaluate_pairing(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    pairing_id: str,
    vectors: torch.Tensor,
    clean: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
    noise_tiers: torch.Tensor,
    centre: torch.Tensor,
    regime: str,
    permutation: str,
    geometry: str,
    threat: str,
    severity: float,
    seed: int,
    draw: int,
) -> list[dict[str, Any]]:
    bounded = clip_l2(vectors, float(config["aggregation"]["server_clip_norm"]))
    pilot = _comparator_reference("fcc", bounded, config, anchor=anchor)
    crossfit = _fcc_leave_one_out(
        bounded,
        anchor=anchor,
        radius=float(config["references"]["fcc"]["radius"]),
    )
    statistical_radii, deployed_radii = _radii_for_cell(
        config,
        derived,
        calibration,
        regime=regime,
        permutation=permutation,
        noise_variances=noise_variances,
    )
    candidate, diagnostics = gaussian_aware_crossfit_bounded_correction(
        bounded,
        pilot=pilot,
        crossfit_references=crossfit,
        statistical_radii=statistical_radii,
        deployed_radii=deployed_radii,
        pilot_replace_one_bound=float(derived["pilot_replace_one_bound"]),
        block_sizes=config["cohort"]["block_sizes"],
        influence_cap=derived["influence_cap_per_block"],
        regularization=float(derived["regularization"]),
        correction_budget=float(derived["correction_budget"]),
        num_steps=int(derived["num_steps"]),
        return_diagnostics=True,
    )
    if bool(diagnostics["crossfit_references_affect_returned_reference"]):
        raise RuntimeError("Crossfit diagnostics changed the deployed estimator")

    honest = ~byzantine_mask
    target = clean[honest].mean(dim=0)
    regular_honest = honest & ~outlier_mask
    mask_metrics = _candidate_diagnostics(
        vectors=bounded,
        crossfit_references=crossfit,
        statistical_radii=statistical_radii,
        deployed_radii=deployed_radii,
        diagnostics=diagnostics,
        regular_honest=regular_honest,
        honest_outliers=honest & outlier_mask,
        byzantine=byzantine_mask,
        noise_tiers=noise_tiers,
        influence_caps=derived["influence_cap_per_block"],
        block_sizes=config["cohort"]["block_sizes"],
    )

    entries: list[tuple[str, torch.Tensor, Mapping[str, Any] | None]] = []
    for name in COMPARATORS:
        reference = (
            pilot
            if name == "fcc"
            else _comparator_reference(name, bounded, config, anchor=anchor)
        )
        entries.append((name, reference, None))
    entries.append((CANDIDATE, candidate, diagnostics))
    rows: list[dict[str, Any]] = []
    for name, reference, candidate_diagnostics in entries:
        is_candidate = name == CANDIDATE
        rows.append(
            {
                "campaign_id": config["campaign_id"],
                "phase": phase,
                "pairing_id": pairing_id,
                "candidate": name,
                "is_g0e_candidate": is_candidate,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "severity": severity,
                "seed": seed,
                "draw": draw,
                "reference_error": float(
                    torch.linalg.vector_norm(reference - target).item()
                ),
                "reference_error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "reference_error_ratio_to_uniform": float("nan"),
                "reference_error_ratio_to_fcc": float("nan"),
                "regular_honest_statistical_tail_rate": (
                    mask_metrics["regular_honest_statistical_tail_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "regular_honest_influence_cap_activation_rate": (
                    mask_metrics["regular_honest_influence_cap_activation_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "honest_outlier_statistical_tail_rate": (
                    mask_metrics["honest_outlier_statistical_tail_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "honest_outlier_influence_cap_activation_rate": (
                    mask_metrics["honest_outlier_influence_cap_activation_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "honest_outlier_not_cap_limited_rate": (
                    mask_metrics["honest_outlier_not_cap_limited_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "byzantine_influence_cap_activation_rate": (
                    mask_metrics["byzantine_influence_cap_activation_rate"]
                    if is_candidate
                    else float("nan")
                ),
                "byzantine_influence_share": (
                    mask_metrics["byzantine_influence_share"]
                    if is_candidate
                    else float("nan")
                ),
                "regular_honest_statistical_tail_rate_by_block": (
                    mask_metrics["regular_honest_statistical_tail_rate_by_block"]
                    if is_candidate
                    else None
                ),
                "regular_honest_influence_cap_activation_rate_by_block": (
                    mask_metrics[
                        "regular_honest_influence_cap_activation_rate_by_block"
                    ]
                    if is_candidate
                    else None
                ),
                "honest_outlier_statistical_tail_rate_by_block": (
                    mask_metrics["honest_outlier_statistical_tail_rate_by_block"]
                    if is_candidate
                    else None
                ),
                "honest_outlier_influence_cap_activation_rate_by_block": (
                    mask_metrics[
                        "honest_outlier_influence_cap_activation_rate_by_block"
                    ]
                    if is_candidate
                    else None
                ),
                "honest_outlier_not_cap_limited_rate_by_block": (
                    mask_metrics["honest_outlier_not_cap_limited_rate_by_block"]
                    if is_candidate
                    else None
                ),
                "byzantine_influence_cap_activation_rate_by_block": (
                    mask_metrics["byzantine_influence_cap_activation_rate_by_block"]
                    if is_candidate
                    else None
                ),
                "byzantine_influence_share_by_block": (
                    mask_metrics["byzantine_influence_share_by_block"]
                    if is_candidate
                    else None
                ),
                "regular_honest_diagnostics_by_noise_tier_block": (
                    mask_metrics["regular_honest_diagnostics_by_noise_tier_block"]
                    if is_candidate
                    else None
                ),
                "solver_gradient_residual": (
                    float(candidate_diagnostics["gradient_residual_norm"])
                    if is_candidate
                    else float("nan")
                ),
                "observed_correction_norm": (
                    float(candidate_diagnostics["observed_correction_norm"])
                    if is_candidate
                    else float("nan")
                ),
                "correction_budget": (
                    float(candidate_diagnostics["correction_budget"])
                    if is_candidate
                    else float("nan")
                ),
                "correction_budget_violation": (
                    not bool(candidate_diagnostics["correction_budget_respected"])
                    if is_candidate
                    else None
                ),
                "finite_solver_replace_one_bound": (
                    float(candidate_diagnostics["finite_solver_replace_one_bound"])
                    if is_candidate
                    else float("nan")
                ),
                "resolved_device": str(bounded.device),
                "tensor_dtype": str(bounded.dtype).replace("torch.", ""),
            }
        )
    uniform = next(row for row in rows if row["candidate"] == "uniform_mean")
    fcc = next(row for row in rows if row["candidate"] == "fcc")
    if uniform["reference_error"] <= 0.0 or fcc["reference_error"] <= 0.0:
        raise RuntimeError("A paired baseline has zero error")
    for row in rows:
        row["reference_error_ratio_to_uniform"] = float(row["reference_error"]) / float(
            uniform["reference_error"]
        )
        row["reference_error_ratio_to_fcc"] = float(row["reference_error"]) / float(
            fcc["reference_error"]
        )
    return rows


def _phase_rows(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, tiers = oracle._noise_variances(config, regime, permutation)
            for seed_index, seed_value in enumerate(seeds, start=1):
                seed = int(seed_value)
                print(
                    f"[G0e] {phase} seed {seed_index}/{len(seeds)}: {seed}",
                    flush=True,
                )
                for draw in range(draws_per_seed):
                    for geometry_value in config["cohort"]["honest_outliers"][
                        "geometries"
                    ]:
                        geometry = str(geometry_value)
                        clean, outliers, centre, anchor = oracle._honest_clean_vectors(
                            config,
                            seed=seed,
                            draw=draw,
                            geometry=geometry,
                            include_outliers=True,
                        )
                        observed = oracle._add_private_noise(
                            clean,
                            variances,
                            blocks,
                            seed=seed,
                            draw=draw,
                            regime=regime_name,
                            permutation=permutation,
                            geometry=geometry,
                            pair_permutations=bool(
                                config["randomness"][
                                    "pair_noise_across_tier_permutations"
                                ]
                            ),
                        )
                        for threat_value in config["threats"]["names"]:
                            threat = str(threat_value)
                            levels = [1.0] if threat == "none" else severities
                            for severity in levels:
                                attacked, byzantine = oracle._replace_with_attack(
                                    observed,
                                    config,
                                    threat=threat,
                                    severity=float(severity),
                                    seed=oracle._seed(
                                        "g0e-attack", seed, draw, geometry, threat
                                    ),
                                )
                                pairing_id = (
                                    f"{phase}:{regime_name}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{seed}:{draw}"
                                )
                                rows.extend(
                                    _evaluate_pairing(
                                        config=config,
                                        derived=derived,
                                        calibration=calibration,
                                        phase=phase,
                                        pairing_id=pairing_id,
                                        vectors=attacked,
                                        clean=clean,
                                        outlier_mask=outliers,
                                        byzantine_mask=byzantine,
                                        anchor=anchor,
                                        noise_variances=variances,
                                        noise_tiers=tiers,
                                        centre=centre,
                                        regime=regime_name,
                                        permutation=permutation,
                                        geometry=geometry,
                                        threat=threat,
                                        severity=float(severity),
                                        seed=seed,
                                        draw=draw,
                                    )
                                )
    return rows


def _noise_cell_count(config: Mapping[str, Any]) -> int:
    return sum(
        len(regime["permutations"]) for regime in config["privacy_noise"]["regimes"]
    )


def _expected_pairings_per_seed(config: Mapping[str, Any], phase: str) -> int:
    draws = int(config["randomness"][f"{phase}_draws_per_seed"])
    severities = config["threats"][f"{phase}_severities"]
    geometries = len(config["cohort"]["honest_outliers"]["geometries"])
    threat_levels = 1 + (len(config["threats"]["names"]) - 1) * len(severities)
    return draws * _noise_cell_count(config) * geometries * threat_levels


def _expected_detail_rows_per_seed(config: Mapping[str, Any], phase: str) -> int:
    return _expected_pairings_per_seed(config, phase) * (len(COMPARATORS) + 1)


def _expected_stability_rows_per_seed(config: Mapping[str, Any]) -> int:
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    return _noise_cell_count(config) * trials * (len(COMPARATORS) + 1)


def _validate_checkpoint_rows(
    rows: Any,
    *,
    config: Mapping[str, Any],
    phase: str,
    seed: int,
    kind: str,
    source: Path,
) -> None:
    """Reject partial, CPU, wrong-phase or wrong-seed resume checkpoints."""

    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Invalid empty checkpoint {source}")
    if kind == "detail":
        expected = _expected_detail_rows_per_seed(config, phase)
    elif kind == "stability":
        expected = _expected_stability_rows_per_seed(config)
    else:
        raise ValueError(f"Unknown checkpoint kind {kind!r}")
    if len(rows) != expected:
        raise RuntimeError(
            f"Incomplete {kind} checkpoint {source}: {len(rows)}/{expected} rows"
        )
    required_candidates = set((*COMPARATORS, CANDIDATE))
    for row in rows:
        if str(row.get("phase")) != phase or int(row.get("seed", -1)) != int(seed):
            raise RuntimeError(f"Wrong phase or seed in checkpoint {source}")
        resolved_device = str(row.get("resolved_device"))
        if resolved_device != "mps" and not resolved_device.startswith("mps:"):
            raise RuntimeError(f"Non-MPS checkpoint rejected: {source}")
        if str(row.get("tensor_dtype")) != "float32":
            raise RuntimeError(f"Non-float32 checkpoint rejected: {source}")
        if str(row.get("candidate")) not in required_candidates:
            raise RuntimeError(f"Unknown candidate in checkpoint {source}")
    if kind == "detail":
        identities = [(str(row["pairing_id"]), str(row["candidate"])) for row in rows]
        if len(identities) != len(set(identities)):
            raise RuntimeError(f"Duplicate detail rows in checkpoint {source}")
        pairing_candidates: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            pairing_candidates[str(row["pairing_id"])].add(str(row["candidate"]))
        if len(pairing_candidates) != _expected_pairings_per_seed(config, phase) or any(
            candidates != required_candidates
            for candidates in pairing_candidates.values()
        ):
            raise RuntimeError(f"Incomplete candidate pairing in checkpoint {source}")
    else:
        identities = [
            (
                str(row["noise_regime"]),
                str(row["noise_permutation"]),
                int(row["trial"]),
                str(row["candidate"]),
            )
            for row in rows
        ]
        if len(identities) != len(set(identities)):
            raise RuntimeError(f"Duplicate stability rows in checkpoint {source}")


def _cached_phase_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        checkpoint = checkpoint_dir / f"{phase}_detail_seed_{seed}.json"
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            _validate_checkpoint_rows(
                cached,
                config=config,
                phase=phase,
                seed=seed,
                kind="detail",
                source=checkpoint,
            )
            print(f"[G0e] resume {checkpoint.name}", flush=True)
            rows.extend(cached)
            continue
        produced = _phase_rows(
            config=config,
            derived=derived,
            calibration=calibration,
            phase=phase,
            seeds=[seed],
            draws_per_seed=draws_per_seed,
            severities=severities,
        )
        _validate_checkpoint_rows(
            produced,
            config=config,
            phase=phase,
            seed=seed,
            kind="detail",
            source=checkpoint,
        )
        _atomic_json(checkpoint, produced)
        rows.extend(produced)
    return rows


def _replace_one_audit(
    *,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    fcc_radius = float(config["references"]["fcc"]["radius"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = oracle._noise_variances(config, regime, permutation)
            statistical_radii, deployed_radii = _radii_for_cell(
                config,
                derived,
                calibration,
                regime=regime_name,
                permutation=permutation,
                noise_variances=variances,
            )
            for seed_value in seeds:
                seed = int(seed_value)
                for trial in range(trials):
                    clean, _, _, anchor = oracle._honest_clean_vectors(
                        config,
                        seed=seed,
                        draw=trial,
                        geometry="orthogonal",
                        include_outliers=False,
                    )
                    observed = oracle._add_private_noise(
                        clean,
                        variances,
                        blocks,
                        seed=seed,
                        draw=trial,
                        regime=regime_name,
                        permutation=permutation,
                        geometry="g0e-replace",
                        pair_permutations=bool(
                            config["randomness"]["pair_noise_across_tier_permutations"]
                        ),
                    )
                    bounded = clip_l2(observed, server_clip)
                    index = trial % int(bounded.shape[0])
                    replacement = torch.randn(
                        bounded.shape[1],
                        generator=oracle._generator(
                            "g0e-replacement", phase, seed, trial
                        ),
                        dtype=bounded.dtype,
                        device=bounded.device,
                    )
                    replacement = clip_l2(
                        (2.0 * server_clip * replacement)[None, :], server_clip
                    )[0]
                    neighbour = bounded.clone()
                    neighbour[index] = replacement

                    left_pilot = _comparator_reference(
                        "fcc", bounded, config, anchor=anchor
                    )
                    right_pilot = _comparator_reference(
                        "fcc", neighbour, config, anchor=anchor
                    )
                    left_crossfit = _fcc_leave_one_out(
                        bounded, anchor=anchor, radius=fcc_radius
                    )
                    right_crossfit = _fcc_leave_one_out(
                        neighbour, anchor=anchor, radius=fcc_radius
                    )
                    left_g0e, diagnostics = gaussian_aware_crossfit_bounded_correction(
                        bounded,
                        pilot=left_pilot,
                        crossfit_references=left_crossfit,
                        statistical_radii=statistical_radii,
                        deployed_radii=deployed_radii,
                        pilot_replace_one_bound=float(
                            derived["pilot_replace_one_bound"]
                        ),
                        block_sizes=blocks,
                        influence_cap=derived["influence_cap_per_block"],
                        regularization=float(derived["regularization"]),
                        correction_budget=float(derived["correction_budget"]),
                        num_steps=int(derived["num_steps"]),
                        return_diagnostics=True,
                    )
                    right_g0e = gaussian_aware_crossfit_bounded_correction(
                        neighbour,
                        pilot=right_pilot,
                        crossfit_references=right_crossfit,
                        statistical_radii=statistical_radii,
                        deployed_radii=deployed_radii,
                        pilot_replace_one_bound=float(
                            derived["pilot_replace_one_bound"]
                        ),
                        block_sizes=blocks,
                        influence_cap=derived["influence_cap_per_block"],
                        regularization=float(derived["regularization"]),
                        correction_budget=float(derived["correction_budget"]),
                        num_steps=int(derived["num_steps"]),
                    )
                    pairs: list[tuple[str, torch.Tensor, torch.Tensor, float, str]] = [
                        (
                            CANDIDATE,
                            left_g0e,
                            right_g0e,
                            float(diagnostics["finite_solver_replace_one_bound"]),
                            "fcc_pilot_plus_fixed_radius_finite_solver",
                        )
                    ]
                    for name in COMPARATORS:
                        left = (
                            left_pilot
                            if name == "fcc"
                            else _comparator_reference(
                                name, bounded, config, anchor=anchor
                            )
                        )
                        right = (
                            right_pilot
                            if name == "fcc"
                            else _comparator_reference(
                                name, neighbour, config, anchor=anchor
                            )
                        )
                        if name == "uniform_mean":
                            bound = 2.0 * server_clip / float(bounded.shape[0])
                            certificate = "replace_one_after_server_clip"
                        elif name == "fcc":
                            bound = 2.0 * fcc_radius / float(bounded.shape[0])
                            certificate = "replace_one_fixed_anchor"
                        elif name == "trimmed_mean":
                            n = float(bounded.shape[0])
                            d = float(bounded.shape[1])
                            trim = float(
                                config["references"]["trimmed_mean"]["trim_count"]
                            )
                            bound = 2.0 * server_clip * math.sqrt(d) / (n - 2.0 * trim)
                            certificate = "dimension_dependent_only"
                        else:
                            bound = float("nan")
                            certificate = "no_dimension_free_certificate"
                        pairs.append((name, left, right, bound, certificate))
                    for name, left, right, bound, certificate in pairs:
                        observed_delta = float(
                            torch.linalg.vector_norm(left - right).item()
                        )
                        rows.append(
                            {
                                "phase": phase,
                                "candidate": name,
                                "noise_regime": regime_name,
                                "noise_permutation": permutation,
                                "seed": seed,
                                "trial": trial,
                                "replaced_client": index,
                                "observed_delta": observed_delta,
                                "theoretical_bound": bound,
                                "certificate": certificate,
                                "violation": (
                                    bool(observed_delta > bound + 1.0e-6)
                                    if math.isfinite(bound)
                                    else None
                                ),
                                "resolved_device": str(bounded.device),
                                "tensor_dtype": str(bounded.dtype).replace(
                                    "torch.", ""
                                ),
                            }
                        )
    return rows


def _cached_stability_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    phase: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        checkpoint = checkpoint_dir / f"{phase}_stability_seed_{seed}.json"
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            _validate_checkpoint_rows(
                cached,
                config=config,
                phase=phase,
                seed=seed,
                kind="stability",
                source=checkpoint,
            )
            print(f"[G0e] resume {checkpoint.name}", flush=True)
            rows.extend(cached)
            continue
        produced = _replace_one_audit(
            config=config,
            derived=derived,
            calibration=calibration,
            phase=phase,
            seeds=[seed],
        )
        _validate_checkpoint_rows(
            produced,
            config=config,
            phase=phase,
            seed=seed,
            kind="stability",
            source=checkpoint,
        )
        _atomic_json(checkpoint, produced)
        rows.extend(produced)
    return rows


def _critical_t95(n: int) -> float:
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
    }.get(n, 1.96)


def _paired_ci95(
    rows: Sequence[dict[str, Any]],
    *,
    left: str,
    right: str,
    predicate: Any,
) -> tuple[float, float, float, int]:
    paired: dict[str, dict[str, float]] = defaultdict(dict)
    seeds: dict[str, int] = {}
    for row in rows:
        if predicate(row) and row["candidate"] in {left, right}:
            pairing = str(row["pairing_id"])
            paired[pairing][str(row["candidate"])] = float(row["reference_error"])
            seeds[pairing] = int(row["seed"])
    per_seed: dict[int, list[float]] = defaultdict(list)
    for pairing, values in paired.items():
        if set(values) != {left, right}:
            raise RuntimeError(f"Incomplete paired comparison {pairing}")
        per_seed[seeds[pairing]].append(values[left] - values[right])
    seed_means = [_finite_mean(values) for _, values in sorted(per_seed.items())]
    mean = _finite_mean(seed_means)
    if len(seed_means) < 2:
        return mean, float("-inf"), float("inf"), len(seed_means)
    half = (
        _critical_t95(len(seed_means))
        * statistics.stdev(seed_means)
        / math.sqrt(len(seed_means))
    )
    return mean, mean - half, mean + half, len(seed_means)


def _group_worst(rows: Sequence[dict[str, Any]], field: str) -> float:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["noise_regime"],
            row["noise_permutation"],
            row["outlier_geometry"],
            row["threat"],
            row["severity"],
        )
        grouped[key].append(float(row[field]))
    return max(_finite_mean(values) for values in grouped.values())


_BLOCK_DIAGNOSTIC_FIELDS = (
    "regular_honest_statistical_tail_rate",
    "regular_honest_influence_cap_activation_rate",
    "honest_outlier_statistical_tail_rate",
    "honest_outlier_influence_cap_activation_rate",
    "honest_outlier_not_cap_limited_rate",
    "byzantine_influence_cap_activation_rate",
    "byzantine_influence_share",
)


def _diagnostic_group_summaries(
    rows: Sequence[dict[str, Any]], *, phase: str, num_blocks: int
) -> list[dict[str, Any]]:
    """Preserve every evaluation cell and block before applying gates."""

    grouped: dict[tuple[str, str, str, str, float, int], dict[str, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    seed_clusters: dict[tuple[str, str, str, str, float, int], set[int]] = defaultdict(
        set
    )
    observations: dict[tuple[str, str, str, str, float, int], int] = defaultdict(int)
    for row in rows:
        if row["candidate"] != CANDIDATE:
            continue
        for block in range(num_blocks):
            key = (
                str(row["noise_regime"]),
                str(row["noise_permutation"]),
                str(row["outlier_geometry"]),
                str(row["threat"]),
                float(row["severity"]),
                block,
            )
            observations[key] += 1
            seed_clusters[key].add(int(row["seed"]))
            for field in _BLOCK_DIAGNOSTIC_FIELDS:
                values = row[f"{field}_by_block"]
                if not isinstance(values, list) or len(values) != num_blocks:
                    raise RuntimeError(
                        f"Missing {num_blocks}-block diagnostic {field!r}"
                    )
                grouped[key][field].append(float(values[block]))
            grouped[key]["honest_outlier_all_blocks_not_cap_limited_rate"].append(
                float(row["honest_outlier_not_cap_limited_rate"])
            )

    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        regime, permutation, geometry, threat, severity, block = key
        result.append(
            {
                "phase": phase,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "severity": severity,
                "block": block,
                "observations": observations[key],
                "seed_clusters": len(seed_clusters[key]),
                **{
                    field: _finite_mean(grouped[key][field])
                    for field in _BLOCK_DIAGNOSTIC_FIELDS
                },
                "honest_outlier_all_blocks_not_cap_limited_rate": _finite_mean(
                    grouped[key]["honest_outlier_all_blocks_not_cap_limited_rate"]
                ),
            }
        )
    if not result:
        raise RuntimeError("No G0e diagnostic group could be constructed")
    return result


def _regular_tier_block_group_summaries(
    rows: Sequence[dict[str, Any]], *, phase: str
) -> list[dict[str, Any]]:
    """Aggregate regular diagnostics without pooling public noise tiers."""

    grouped: dict[
        tuple[str, str, str, str, float, float, int], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    seed_clusters: dict[tuple[str, str, str, str, float, float, int], set[int]] = (
        defaultdict(set)
    )
    for row in rows:
        if row["candidate"] != CANDIDATE:
            continue
        diagnostics = row["regular_honest_diagnostics_by_noise_tier_block"]
        if not isinstance(diagnostics, list) or not diagnostics:
            raise RuntimeError("Missing regular noise-tier diagnostics")
        for item in diagnostics:
            key = (
                str(row["noise_regime"]),
                str(row["noise_permutation"]),
                str(row["outlier_geometry"]),
                str(row["threat"]),
                float(row["severity"]),
                float(item["noise_tier"]),
                int(item["block"]),
            )
            grouped[key]["tail"].append(
                float(item["regular_honest_statistical_tail_rate"])
            )
            grouped[key]["cap"].append(
                float(item["regular_honest_influence_cap_activation_rate"])
            )
            grouped[key]["regular_clients"].append(float(item["regular_client_count"]))
            seed_clusters[key].add(int(row["seed"]))

    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        regime, permutation, geometry, threat, severity, tier, block = key
        result.append(
            {
                "phase": phase,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "severity": severity,
                "noise_tier": tier,
                "block": block,
                "observations": len(grouped[key]["tail"]),
                "seed_clusters": len(seed_clusters[key]),
                "regular_clients_per_observation_min": int(
                    min(grouped[key]["regular_clients"])
                ),
                "regular_honest_statistical_tail_rate": _finite_mean(
                    grouped[key]["tail"]
                ),
                "regular_honest_influence_cap_activation_rate": _finite_mean(
                    grouped[key]["cap"]
                ),
            }
        )
    if not result:
        raise RuntimeError("No regular noise-tier diagnostic group was constructed")
    return result


def _expected_pairings(config: Mapping[str, Any], phase: str) -> int:
    seeds = config["randomness"][f"{phase}_seeds"]
    return len(seeds) * _expected_pairings_per_seed(config, phase)


def _expected_stability_observations(config: Mapping[str, Any], phase: str) -> int:
    """Expected G0e-only stability rows for all seeds in one phase."""

    seeds = config["randomness"][f"{phase}_seeds"]
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    return len(seeds) * _noise_cell_count(config) * trials


def _summarize(
    rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == CANDIDATE]
    clean = [row for row in selected if row["threat"] == "none"]
    separated_names = set(config["threats"]["separated_for_gates"])
    evasive_names = set(config["threats"]["evasive_controls"])
    attacked = [row for row in selected if row["threat"] in separated_names]
    evasive = [row for row in selected if row["threat"] in evasive_names]
    stable = [row for row in stability_rows if row["candidate"] == CANDIDATE]
    diagnostic_groups = _diagnostic_group_summaries(
        selected,
        phase=phase,
        num_blocks=len(config["cohort"]["block_sizes"]),
    )
    regular_tier_groups = _regular_tier_block_group_summaries(selected, phase=phase)
    clean_diagnostic_groups = [
        row for row in diagnostic_groups if row["threat"] == "none"
    ]
    separated_diagnostic_groups = [
        row for row in diagnostic_groups if row["threat"] in separated_names
    ]
    clean_regular_tier_groups = [
        row for row in regular_tier_groups if row["threat"] == "none"
    ]
    if not clean_diagnostic_groups or not separated_diagnostic_groups:
        raise RuntimeError("Missing clean or separated diagnostic groups")
    if not clean_regular_tier_groups:
        raise RuntimeError("Missing clean regular-client noise-tier groups")
    expected = _expected_pairings(config, phase)
    complete_fraction = len(selected) / float(expected) if expected else 0.0
    expected_stability = _expected_stability_observations(config, phase)
    stability_complete_fraction = (
        len(stable) / float(expected_stability) if expected_stability else 0.0
    )
    finite_fraction = _finite_mean(
        float(all(math.isfinite(float(row[field])) for field in PRIMARY_FIELDS))
        for row in selected
    )
    difference = _paired_ci95(
        rows,
        left=CANDIDATE,
        right="fcc",
        predicate=lambda row: (
            row["noise_regime"] == "heteroscedastic"
            and row["threat"] in separated_names
        ),
    )
    target_tail = float(
        config["references"]["g0e_public_budgets"]["regular_honest_false_tail_rate"]
    )
    summary: dict[str, Any] = {
        "phase": phase,
        "candidate": CANDIDATE,
        "observations": len(selected),
        "expected_observations": expected,
        "complete_fraction": complete_fraction,
        "stability_observations": len(stable),
        "expected_stability_observations": expected_stability,
        "stability_complete_fraction": stability_complete_fraction,
        "finite_primary_metric_fraction": finite_fraction,
        "clean_error_ratio_to_uniform_worst_group": _group_worst(
            clean, "reference_error_ratio_to_uniform"
        ),
        "attacked_error_ratio_to_fcc_worst_group": _group_worst(
            attacked, "reference_error_ratio_to_fcc"
        ),
        "evasive_error_ratio_to_fcc_worst_group": _group_worst(
            evasive, "reference_error_ratio_to_fcc"
        ),
        "heteroscedastic_attacked_error_difference_to_fcc_seed_mean": difference[0],
        "heteroscedastic_attacked_error_difference_to_fcc_ci95_low": difference[1],
        "heteroscedastic_attacked_error_difference_to_fcc_ci95_high": difference[2],
        "heteroscedastic_attacked_error_difference_seed_clusters": difference[3],
        "regular_honest_statistical_tail_rate": _finite_mean(
            row["regular_honest_statistical_tail_rate"] for row in clean
        ),
        "regular_honest_statistical_tail_rate_abs_error": abs(
            _finite_mean(row["regular_honest_statistical_tail_rate"] for row in clean)
            - target_tail
        ),
        "regular_honest_statistical_tail_rate_abs_error_worst_group_block": max(
            abs(float(row["regular_honest_statistical_tail_rate"]) - target_tail)
            for row in clean_regular_tier_groups
        ),
        "regular_honest_statistical_tail_rate_upper_excess_worst_group_block": max(
            float(row["regular_honest_statistical_tail_rate"]) - target_tail
            for row in clean_regular_tier_groups
        ),
        "regular_honest_influence_cap_activation_rate": _finite_mean(
            row["regular_honest_influence_cap_activation_rate"] for row in clean
        ),
        "regular_honest_influence_cap_activation_rate_worst_group_block": max(
            float(row["regular_honest_influence_cap_activation_rate"])
            for row in clean_regular_tier_groups
        ),
        "honest_outlier_not_cap_limited_rate": _finite_mean(
            row["honest_outlier_not_cap_limited_rate"] for row in clean
        ),
        "honest_outlier_all_blocks_not_cap_limited_rate_worst_group": min(
            float(row["honest_outlier_all_blocks_not_cap_limited_rate"])
            for row in clean_diagnostic_groups
        ),
        "separated_byzantine_influence_share_worst_group": max(
            float(row["byzantine_influence_share"])
            for row in separated_diagnostic_groups
        ),
        "diagnostic_group_observations_min": min(
            int(row["observations"]) for row in diagnostic_groups
        ),
        "diagnostic_group_seed_clusters_min": min(
            int(row["seed_clusters"]) for row in diagnostic_groups
        ),
        "regular_tier_clients_per_observation_min": min(
            int(row["regular_clients_per_observation_min"])
            for row in regular_tier_groups
        ),
        "replace_one_bound_max": max(float(row["theoretical_bound"]) for row in stable),
        "replace_one_observed_max": max(float(row["observed_delta"]) for row in stable),
        "replace_one_violation_count": sum(bool(row["violation"]) for row in stable),
        "solver_gradient_residual_max": max(
            float(row["solver_gradient_residual"]) for row in selected
        ),
        "correction_radius_violation_count": sum(
            bool(row["correction_budget_violation"]) for row in selected
        ),
    }
    gates = config["gates"]
    checks = {
        "complete": complete_fraction >= float(gates["complete_fraction_min"]),
        "stability_complete": stability_complete_fraction
        >= float(gates["stability_complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_primary_metric_fraction_min"]),
        "clean_vs_uniform": summary["clean_error_ratio_to_uniform_worst_group"]
        <= float(gates["clean_reference_error_ratio_to_uniform_max"]),
        "attacked_vs_fcc": summary["attacked_error_ratio_to_fcc_worst_group"]
        <= float(gates["attacked_reference_error_ratio_to_fcc_max"]),
        "evasive_vs_fcc": summary["evasive_error_ratio_to_fcc_worst_group"]
        <= float(gates["evasive_reference_error_ratio_to_fcc_max"]),
        "heteroscedastic_attacked_improvement": summary[
            "heteroscedastic_attacked_error_difference_to_fcc_ci95_high"
        ]
        <= float(
            gates["heteroscedastic_attacked_error_difference_to_fcc_ci95_high_max"]
        ),
        "regular_tail_calibration": summary[
            "regular_honest_statistical_tail_rate_upper_excess_worst_group_block"
        ]
        <= float(gates["regular_honest_statistical_tail_rate_upper_excess_max"]),
        "regular_not_cap_limited": summary[
            "regular_honest_influence_cap_activation_rate_worst_group_block"
        ]
        <= float(gates["regular_honest_influence_cap_activation_rate_max"]),
        "honest_outlier_retention": summary[
            "honest_outlier_all_blocks_not_cap_limited_rate_worst_group"
        ]
        >= float(gates["honest_outlier_not_cap_limited_rate_min"]),
        "byzantine_influence_share": summary[
            "separated_byzantine_influence_share_worst_group"
        ]
        <= float(gates["separated_byzantine_influence_share_max"]),
        "diagnostic_group_observations": summary["diagnostic_group_observations_min"]
        >= int(gates["diagnostic_group_observations_min"]),
        "diagnostic_group_seed_clusters": summary["diagnostic_group_seed_clusters_min"]
        >= int(gates["diagnostic_group_seed_clusters_min"]),
        "regular_tier_minimum_clients": summary[
            "regular_tier_clients_per_observation_min"
        ]
        >= int(gates["regular_tier_clients_per_observation_min"]),
        "replace_one_bound": summary["replace_one_bound_max"]
        <= float(gates["replace_one_bound_max"]),
        "replace_one_empirical": summary["replace_one_violation_count"]
        <= int(gates["empirical_replace_one_violation_max"]),
        "solver_residual": summary["solver_gradient_residual_max"]
        <= float(gates["solver_gradient_residual_max"]),
        "correction_radius": summary["correction_radius_violation_count"]
        <= int(gates["correction_radius_violation_max"]),
    }
    for name, passed in checks.items():
        summary[f"gate_{name}"] = bool(passed)
    summary["gate_fail_count"] = sum(not value for value in checks.values())
    summary["passes_all_gates"] = all(checks.values())
    return summary


def _comparator_summaries(
    rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    separated = set(config["threats"]["separated_for_gates"])
    evasive = set(config["threats"]["evasive_controls"])
    for name in (*COMPARATORS, CANDIDATE):
        selected = [row for row in rows if row["candidate"] == name]
        stable = [row for row in stability_rows if row["candidate"] == name]
        clean = [row for row in selected if row["threat"] == "none"]
        attacked = [row for row in selected if row["threat"] in separated]
        evasive_rows = [row for row in selected if row["threat"] in evasive]
        finite_bounds = [
            float(row["theoretical_bound"])
            for row in stable
            if math.isfinite(float(row["theoretical_bound"]))
        ]
        result.append(
            {
                "candidate": name,
                "observations": len(selected),
                "clean_reference_error_mean": _finite_mean(
                    row["reference_error"] for row in clean
                ),
                "clean_error_ratio_to_uniform_worst_group": _group_worst(
                    clean, "reference_error_ratio_to_uniform"
                ),
                "attacked_reference_error_mean": _finite_mean(
                    row["reference_error"] for row in attacked
                ),
                "attacked_error_ratio_to_fcc_worst_group": _group_worst(
                    attacked, "reference_error_ratio_to_fcc"
                ),
                "evasive_reference_error_mean": _finite_mean(
                    row["reference_error"] for row in evasive_rows
                ),
                "evasive_error_ratio_to_fcc_worst_group": _group_worst(
                    evasive_rows, "reference_error_ratio_to_fcc"
                ),
                "replace_one_observed_max": max(
                    float(row["observed_delta"]) for row in stable
                ),
                "replace_one_theoretical_bound": (
                    max(finite_bounds) if finite_bounds else "N/A"
                ),
                "certificate": ";".join(
                    sorted(set(str(row["certificate"]) for row in stable))
                ),
            }
        )
    return result


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    calibration: Mapping[str, Any],
    development: Mapping[str, Any],
    decision: Mapping[str, Any],
    output_dir: Path,
    holdout: Mapping[str, Any] | None = None,
) -> None:
    if decision["holdout_status"] == "blocked_by_development_gate":
        verdict = "**Arrêt au développement. Le holdout n'a été ni généré ni lu.**"
    elif bool(decision["promote"]):
        verdict = "**Promotion synthétique autorisée.**"
    else:
        verdict = "**Aucune promotion après évaluation du holdout.**"
    lines = [
        "# G0e — correction Gaussian-aware bornée autour de FCC",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "G0e est un audit de référence synthétique : ni poids FAR, ni "
        "accuracy, ni entraînement de réseau ne sont utilisés.",
        "",
        "## Paramètres dérivés des budgets publics",
        "",
        f"- regularisation gamma : `{derived['regularization']}` ;",
        f"- cap client complet G : `{derived['influence_cap_total']}` ;",
        f"- itérations publiques K : `{derived['num_steps']}` ;",
        f"- beta finite-K : `{derived['beta_finite_solver']}` ;",
        f"- beta asymptotique de contrôle : `{derived['beta_asymptotic_control']}` ;",
        f"- budget de correction autour de FCC : `{derived['correction_budget']}` ;",
        f"- borne replace-one finite-K : `{derived['finite_solver_replace_one_bound']}`.",
        "",
        "Le budget Byzantine concerne uniquement la correction. Le pilote FCC "
        "a une borne de contamination distincte. Pour b remplacements :",
        "",
        f"- FCC : `{derived['fcc_replacement_contamination_bound']}` ;",
        f"- correction brute : `{derived['raw_correction_replacement_contamination_bound']}` ;",
        f"- correction mélangée : `{derived['blended_correction_replacement_contamination_bound']}` ;",
        f"- référence complète : `{derived['total_reference_replacement_contamination_bound']}`.",
        "",
        "## Calibration offline",
        "",
        "La calibration reproduit Clip_U puis FCC leave-one-out. Un seul client "
        "régulier probe par cohorte indépendante et par strate alimente chaque "
        "quantile. Les contextes réguliers et honest-outliers bornés sont "
        "séparés; les outliers ne sont jamais étiquetés réguliers. Le seuil "
        "commun est le maximum public sur les folds et strates.",
        "",
        f"- seuils standardisés : `{calibration['standardized_thresholds']}` ;",
        f"- taux de queue par bloc : `{calibration['calibration_tail_rate_by_block']}` ;",
        f"- taux par contexte : `{calibration['calibration_tail_rate_by_context']}`.",
        "",
        "Le statistical-tail utilise le rayon sans marge. Le cap-active est "
        "mesuré séparément avec le rayon déployé. Les références cross-fittées "
        "sont des diagnostics et ne sont pas des entrées du solveur online.",
        "Les gates réguliers de queue et de cap sont évalués sur le pire tier "
        "de bruit, la pire cellule préenregistrée et le pire bloc. La rétention "
        "des honest outliers exige qu'aucun de leurs blocs ne soit cap-actif, "
        "dans la pire cellule. Aucune moyenne globale ne peut masquer un échec.",
        "",
        "## Gates",
        "",
        "| Phase | Propre/uniforme | Attaques/FCC | ALIE/FCC | CI95 diff hétéro vs FCC | Excès queue pire tier-cellule-bloc | Cap régulier pire tier-cellule-bloc | Outliers sans aucun bloc capé, pire cellule | Masse Byzantine pire cellule-bloc | Borne replace | Max observé | Résidu | Échecs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def add_gate_row(label: str, row: Mapping[str, Any]) -> None:
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | [{:.4f}, {:.4f}] | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {:.4f} | {:.2e} | {} |".format(
                label,
                float(row["clean_error_ratio_to_uniform_worst_group"]),
                float(row["attacked_error_ratio_to_fcc_worst_group"]),
                float(row["evasive_error_ratio_to_fcc_worst_group"]),
                float(row["heteroscedastic_attacked_error_difference_to_fcc_ci95_low"]),
                float(
                    row["heteroscedastic_attacked_error_difference_to_fcc_ci95_high"]
                ),
                float(
                    row[
                        "regular_honest_statistical_tail_rate_upper_excess_worst_group_block"
                    ]
                ),
                float(
                    row[
                        "regular_honest_influence_cap_activation_rate_worst_group_block"
                    ]
                ),
                float(
                    row["honest_outlier_all_blocks_not_cap_limited_rate_worst_group"]
                ),
                float(row["separated_byzantine_influence_share_worst_group"]),
                float(row["replace_one_bound_max"]),
                float(row["replace_one_observed_max"]),
                float(row["solver_gradient_residual_max"]),
                int(row["gate_fail_count"]),
            )
        )

    add_gate_row("Développement", development)
    if holdout is not None:
        add_gate_row("Holdout", holdout)
    lines += [
        "",
        "## Reproductibilité",
        "",
        f"- sorties : `{output_dir.resolve()}` ;",
        f"- calibration : `{config['randomness']['calibration_folds']}` ;",
        f"- développement : `{config['randomness']['development_seeds']}` ;",
        f"- holdout prévu : `{config['randomness']['holdout_seeds']}` ;",
        f"- statut holdout : `{decision['holdout_status']}` ;",
        f"- stabilité développement : `{development['stability_observations']}/{development['expected_stability_observations']}` ;",
        "- runtime : MPS float32, aucun fallback CPU.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    _validate_config(config)
    derived = _derive_parameters(config)
    if (
        float(derived["finite_correction_norm_bound"])
        > float(derived["correction_budget"]) + 1.0e-12
    ):
        raise RuntimeError("Derived finite correction violates B_corr")
    if float(derived["solver_gradient_residual_bound"]) > float(
        derived["solver_error_tolerance"]
    ):
        raise RuntimeError("Derived K does not certify the gradient residual")
    if float(derived["finite_solver_replace_one_bound"]) > float(
        derived["replace_one_bound_budget"]
    ):
        raise RuntimeError("Public budgets are incompatible with stability gate")

    runtime_device, runtime_dtype = oracle._configure_runtime("mps")
    if runtime_device.type != "mps" or runtime_dtype != torch.float32:
        raise RuntimeError("G0e requires real MPS float32 execution")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"{output_dir} is not empty; pass --resume to reuse exact checkpoints"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != fingerprint:
            raise RuntimeError("Refusing to resume checkpoints with a changed config")
    else:
        _atomic_json(
            manifest_path,
            {
                "campaign_id": config["campaign_id"],
                "config_sha256": fingerprint,
                "resolved_device": "mps",
                "tensor_dtype": "float32",
            },
        )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _atomic_json(output_dir / "derived_parameters.json", derived)
    checkpoint_dir = output_dir / "_checkpoints"

    calibration, calibration_rows = _calibrate(config, derived)
    _atomic_json(output_dir / "calibration_artifact.json", calibration)
    _write_csv(output_dir / "calibration_probes.csv", calibration_rows)

    development_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
        draws_per_seed=int(config["randomness"]["development_draws_per_seed"]),
        severities=config["threats"]["development_severities"],
    )
    development_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
    )
    development = _summarize(
        development_rows, development_stability, config, "development"
    )
    development_comparators = _comparator_summaries(
        development_rows, development_stability, config
    )
    development_diagnostic_groups = _diagnostic_group_summaries(
        development_rows,
        phase="development",
        num_blocks=len(config["cohort"]["block_sizes"]),
    )
    development_regular_tier_groups = _regular_tier_block_group_summaries(
        development_rows, phase="development"
    )
    _write_csv(output_dir / "development_detail.csv", development_rows)
    _write_csv(output_dir / "development_stability.csv", development_stability)
    _write_csv(output_dir / "development_summary.csv", [development])
    _write_csv(output_dir / "development_comparators.csv", development_comparators)
    _write_csv(
        output_dir / "development_diagnostic_groups.csv",
        development_diagnostic_groups,
    )
    _write_csv(
        output_dir / "development_regular_tier_block_diagnostics.csv",
        development_regular_tier_groups,
    )

    development_pass = bool(development["passes_all_gates"])
    if not development_pass:
        stale_holdout = _holdout_artifacts(output_dir)
        if stale_holdout:
            raise RuntimeError(
                "Development failed but holdout artifacts exist; refusing leakage: "
                + ", ".join(str(path.relative_to(output_dir)) for path in stale_holdout)
            )
        decision = {
            "campaign_id": config["campaign_id"],
            "requested_device": "mps",
            "resolved_device": "mps",
            "tensor_dtype": "float32",
            "reference_only": True,
            "holdout_used_for_selection": False,
            "development_passes": False,
            "development_gate_fail_count": int(development["gate_fail_count"]),
            "holdout_status": "blocked_by_development_gate",
            "holdout_passes": None,
            "promote": False,
        }
        _atomic_json(output_dir / "decision.json", decision)
        _write_report(
            report_path,
            config=config,
            derived=derived,
            calibration=calibration,
            development=development,
            decision=decision,
            output_dir=output_dir,
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return decision

    # Materialize holdout only after the immutable candidate passes dev.
    holdout_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="holdout",
        seeds=config["randomness"]["holdout_seeds"],
        draws_per_seed=int(config["randomness"]["holdout_draws_per_seed"]),
        severities=config["threats"]["holdout_severities"],
    )
    holdout_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        derived=derived,
        calibration=calibration,
        phase="holdout",
        seeds=config["randomness"]["holdout_seeds"],
    )
    holdout = _summarize(holdout_rows, holdout_stability, config, "holdout")
    holdout_comparators = _comparator_summaries(holdout_rows, holdout_stability, config)
    holdout_diagnostic_groups = _diagnostic_group_summaries(
        holdout_rows,
        phase="holdout",
        num_blocks=len(config["cohort"]["block_sizes"]),
    )
    holdout_regular_tier_groups = _regular_tier_block_group_summaries(
        holdout_rows, phase="holdout"
    )
    _write_csv(output_dir / "holdout_detail.csv", holdout_rows)
    _write_csv(output_dir / "holdout_stability.csv", holdout_stability)
    _write_csv(output_dir / "holdout_summary.csv", [holdout])
    _write_csv(output_dir / "holdout_comparators.csv", holdout_comparators)
    _write_csv(output_dir / "holdout_diagnostic_groups.csv", holdout_diagnostic_groups)
    _write_csv(
        output_dir / "holdout_regular_tier_block_diagnostics.csv",
        holdout_regular_tier_groups,
    )
    holdout_pass = bool(holdout["passes_all_gates"])
    decision = {
        "campaign_id": config["campaign_id"],
        "requested_device": "mps",
        "resolved_device": "mps",
        "tensor_dtype": "float32",
        "reference_only": True,
        "holdout_used_for_selection": False,
        "development_passes": True,
        "development_gate_fail_count": 0,
        "holdout_status": "executed_after_development_pass",
        "holdout_passes": holdout_pass,
        "holdout_gate_fail_count": int(holdout["gate_fail_count"]),
        "promote": holdout_pass,
    }
    _atomic_json(output_dir / "decision.json", decision)
    _write_report(
        report_path,
        config=config,
        derived=derived,
        calibration=calibration,
        development=development,
        holdout=holdout,
        decision=decision,
        output_dir=output_dir,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0e.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0e_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_Robust_Reference_G0e_MPS.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
