#!/usr/bin/env python3
"""G0d-F: null-calibrated, reference-only Gaussian-aware audit.

This campaign deliberately excludes FAR weights, model training and accuracy.
It calibrates public blockwise Huber transition thresholds on an independent
synthetic null that includes honest heterogeneity, client-side Gaussian noise
and the deterministic server clip.  Candidate selection uses development
seeds only; one locked candidate is then evaluated once on fresh holdout seeds.

The executable is MPS-only and refuses silent CPU fallback.  CPU is used only
by unit tests that call pure helpers directly.
"""

from __future__ import annotations

import argparse
import copy
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

from algorithms.gaussian_aware_reference import (
    gaussian_aware_huber_reference,
)  # noqa: E402
from robustness.aggregators import (  # noqa: E402
    centered_clipping,
    clip_l2,
    geometric_median,
    trimmed_mean,
)
from scripts import run_gaussian_aware_reference_g0c as g0c  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

COMPARATORS = (
    "uniform_mean",
    "fcc",
    "rfa",
    "trimmed_mean",
    "coordinate_median",
    "g0c_locked",
)
PRIMARY_FINITE_FIELDS = (
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"Rows for {path} do not share one schema")
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


def _cached_phase_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
    specs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, float], Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        checkpoint = checkpoint_dir / f"{phase}_detail_seed_{seed}.json"
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(cached, list) or not cached:
                raise RuntimeError(f"Invalid checkpoint {checkpoint}")
            rows.extend(cached)
            print(f"[G0d-F] resume {checkpoint.name}", flush=True)
            continue
        produced = _phase_rows(
            config=config,
            phase=phase,
            seeds=[seed],
            draws_per_seed=draws_per_seed,
            severities=severities,
            specs=specs,
            calibrated=calibrated,
        )
        _atomic_json(checkpoint, produced)
        rows.extend(produced)
    return rows


def _cached_stability_rows(
    *,
    checkpoint_dir: Path,
    resume: bool,
    config: dict[str, Any],
    phase: str,
    seeds: Sequence[int],
    specs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, float], Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_value in seeds:
        seed = int(seed_value)
        checkpoint = checkpoint_dir / f"{phase}_stability_seed_{seed}.json"
        if resume and checkpoint.exists():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(cached, list) or not cached:
                raise RuntimeError(f"Invalid checkpoint {checkpoint}")
            rows.extend(cached)
            print(f"[G0d-F] resume {checkpoint.name}", flush=True)
            continue
        produced = _replace_one_audit(
            config=config,
            phase=phase,
            seeds=[seed],
            specs=specs,
            calibrated=calibrated,
        )
        _atomic_json(checkpoint, produced)
        rows.extend(produced)
    return rows


def _candidate_id(quantile: float, cap: float, regularization: float) -> str:
    return (
        f"g0d_q{round(100 * quantile):02d}"
        f"_g{round(100 * cap):03d}_r{round(100 * regularization):03d}"
    )


def _candidate_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["references"]["g0d_grid"]
    return [
        {
            "id": _candidate_id(float(quantile), float(cap), float(regularization)),
            "quantile": float(quantile),
            "influence_cap_total": float(cap),
            "regularization": float(regularization),
            "num_steps": int(grid["num_steps"]),
            "variance_floor": float(grid["variance_floor"]),
        }
        for quantile in grid["null_tail_quantiles"]
        for cap in grid["influence_cap_totals"]
        for regularization in grid["regularizations"]
    ]


def _validate_config(config: dict[str, Any]) -> None:
    contract = config["scientific_contract"]
    required_contract = {
        "reference_only": True,
        "far_weights_used": False,
        "accuracy_used": False,
        "holdout_used_for_selection": False,
        "null_calibration_is_independent": True,
        "inverse_variance_estimand_forbidden": True,
    }
    for key, expected in required_contract.items():
        if bool(contract[key]) is not expected:
            raise ValueError(f"scientific_contract.{key} must be {expected}")
    if contract["estimand"] != "equal_client_mean_of_clean_honest_updates":
        raise ValueError("G0d-F must retain the equal-client estimand")
    execution = config["execution"]
    if execution["required_device"] != "mps" or bool(execution["allow_cpu_fallback"]):
        raise ValueError("G0d-F is MPS-only and forbids CPU fallback")

    random = config["randomness"]
    groups = (
        [int(value) for value in random["null_calibration_seeds"]],
        [int(value) for value in random["development_seeds"]],
        [int(value) for value in random["holdout_seeds"]],
    )
    fresh = [value for group in groups for value in group]
    if len(fresh) != len(set(fresh)):
        raise ValueError("Calibration, development and holdout seeds must be disjoint")
    if set(fresh) & set(int(value) for value in config["excluded_prior_seeds"]):
        raise ValueError("G0d-F reuses a seed inspected by G0/G0b/G0c")
    if len(groups[0]) < 2 or len(groups[1]) < 3 or len(groups[2]) < 5:
        raise ValueError("Need >=2 calibration, >=3 development and >=5 holdout seeds")
    if int(random["null_calibration_draws_per_seed"]) < 2:
        raise ValueError("Null calibration needs at least two draws per seed")
    if int(random["replace_one_trials_per_seed_cell"]) < 1:
        raise ValueError("Replace-one audit needs at least one trial per seed/cell")

    oracle._validate_config(_oracle_validation_config(config))
    quantiles = [
        float(value)
        for value in config["references"]["g0d_grid"]["null_tail_quantiles"]
    ]
    if not quantiles or any(not 0.5 < value < 1.0 for value in quantiles):
        raise ValueError("Null tail quantiles must lie strictly in (0.5,1)")
    if quantiles != sorted(set(quantiles)):
        raise ValueError("Null tail quantiles must be unique and sorted")
    if tuple(str(value) for value in config["threats"]["names"]) != (
        "none",
        "alie",
        "ipm",
        "bitflip_x10",
        "model_replacement",
    ):
        raise ValueError("All five pre-registered threat cells are required")
    non_null = set(config["threats"]["names"]) - {"none"}
    separated = set(config["threats"]["separated_for_gates"])
    evasive = set(config["threats"]["evasive_controls"])
    if separated & evasive or separated | evasive != non_null:
        raise ValueError("Every non-null attack must be exactly separated or evasive")
    if evasive != {"alie"} or separated != {
        "ipm",
        "bitflip_x10",
        "model_replacement",
    }:
        raise ValueError("G0d-F must retain the pre-registered G0c attack split")
    trim = int(config["references"]["trimmed_mean"]["trim_count"])
    n = int(config["cohort"]["num_clients"])
    if trim < 0 or 2 * trim >= n:
        raise ValueError("Invalid trimmed-mean trim_count")
    if bool(config["selection"]["holdout_used_for_selection"]):
        raise ValueError("Holdout use in candidate selection is forbidden")


def _oracle_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize only fields required by the shared tensor simulator."""

    result = copy.deepcopy(dict(config))
    result["references"]["candidates"] = ["f_sigma_huber"]
    result["references"]["f_sigma_huber"] = {
        "standardized_threshold": [1.0] * len(config["cohort"]["block_sizes"]),
        "null_tail_probability": 0.01,
        "influence_cap": [0.05] * len(config["cohort"]["block_sizes"]),
        "regularization": 0.25,
        "num_steps": 2,
        "variance_floor": float(config["references"]["g0d_grid"]["variance_floor"]),
    }
    result["score"] = {
        "novelty_start_z": 0.5,
        "novelty_full_z": 2.0,
        "rejection_start_z": 3.0,
        "rejection_full_z": 5.0,
        "trust_floor": 0.01,
    }
    return result


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for width in block_sizes:
        result.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(result)


def _effective_variances(
    config: Mapping[str, Any], noise_variances: torch.Tensor
) -> torch.Tensor:
    heterogeneity = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=noise_variances.dtype,
        device=noise_variances.device,
    ).square()
    floor = float(config["references"]["g0d_grid"]["variance_floor"])
    return noise_variances + heterogeneity[None, :] + floor


def _coordinate_median(vectors: torch.Tensor) -> torch.Tensor:
    """MPS-compatible midpoint coordinate median."""

    ordered = torch.sort(vectors, dim=0).values
    n = int(vectors.shape[0])
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _empirical_quantile(values: torch.Tensor, probability: float) -> torch.Tensor:
    """Linear empirical quantile using only sort/index operations on MPS."""

    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("values must be a non-empty vector")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    ordered = torch.sort(values).values
    position = float(probability) * float(ordered.numel() - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - float(lower)
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _null_thresholds(
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str, float], list[float]], list[dict[str, Any]]]:
    """Calibrate block radii before any development or holdout identity.

    The simulated population centre is known only inside this independent
    public null generator.  It is never estimated from a development or
    holdout cohort.  The emitted quantiles are ordinary public constants at
    evaluation time.
    """

    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    slices = _block_slices(blocks)
    server_clip = float(config["aggregation"]["server_clip_norm"])
    quantiles = [
        float(value)
        for value in config["references"]["g0d_grid"]["null_tail_quantiles"]
    ]
    thresholds: dict[tuple[str, str, float], list[float]] = {}
    rows: list[dict[str, Any]] = []
    cell_samples: dict[tuple[str, str], list[torch.Tensor]] = {}
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = oracle._noise_variances(config, regime, permutation)
            effective = _effective_variances(config, variances)
            per_block: list[list[torch.Tensor]] = [[] for _ in blocks]
            for seed in config["randomness"]["null_calibration_seeds"]:
                for draw in range(
                    int(config["randomness"]["null_calibration_draws_per_seed"])
                ):
                    zero = torch.zeros(
                        int(config["cohort"]["dimension"]),
                        dtype=oracle._RUNTIME_DTYPE,
                        device=oracle._RUNTIME_DEVICE,
                    )
                    clean, _, _, _ = oracle._honest_clean_vectors(
                        config,
                        seed=int(seed),
                        draw=draw,
                        geometry="orthogonal",
                        include_outliers=False,
                        centre_override=zero,
                    )
                    observed = oracle._add_private_noise(
                        clean,
                        variances,
                        blocks,
                        seed=int(seed),
                        draw=draw,
                        regime=regime_name,
                        permutation=permutation,
                        geometry="g0d-null",
                        pair_permutations=bool(
                            config["randomness"]["pair_noise_across_tier_permutations"]
                        ),
                    )
                    bounded = clip_l2(observed, server_clip)
                    # The calibration follows the actual reference/scoring
                    # geometry rather than measuring distance to the latent
                    # simulator centre.  Uniform leave-one-out is used as a
                    # fixed, candidate-independent pilot, so no G0d candidate
                    # gains a favourable candidate-specific calibration and
                    # no client is used in the reference defining its own
                    # residual.  The entire pilot is discarded after emitting
                    # the public blockwise thresholds.
                    loo = (bounded.sum(dim=0, keepdim=True) - bounded) / float(
                        bounded.shape[0] - 1
                    )
                    for block_index, block_slice in enumerate(slices):
                        residual = bounded[:, block_slice] - loo[:, block_slice]
                        norm = torch.linalg.vector_norm(residual, dim=1)
                        # Public variance of the uniform leave-one-out pilot.
                        # Under independent client mechanisms this is
                        # sum_{j!=i} V_j/(n-1)^2.  It is included because the
                        # null residual is X_i-F_{-i}, not X_i minus a fixed
                        # oracle centre.  Server clipping can only make this
                        # analytic pre-clip variance conservative.
                        loo_variance = (
                            effective[:, block_index].sum() - effective[:, block_index]
                        ) / float((bounded.shape[0] - 1) ** 2)
                        standardized = (
                            norm / (effective[:, block_index] + loo_variance).sqrt()
                        )
                        per_block[block_index].append(standardized)
            cell_samples[(regime_name, permutation)] = [
                torch.cat(values) for values in per_block
            ]

    # One pooled threshold per block/quantile is shared by every noise cell.
    # Consequently identity/reverse invariance is tested rather than enforced
    # by an identity-specific calibration.  Standardisation by the public
    # effective variance makes pooling homogeneous and heteroscedastic cells
    # scientifically meaningful.
    pooled = [
        torch.cat([samples[block] for samples in cell_samples.values()])
        for block in range(len(blocks))
    ]
    for quantile in quantiles:
        values = [
            float(_empirical_quantile(block, quantile).item()) for block in pooled
        ]
        for regime_name, permutation in cell_samples:
            thresholds[(regime_name, permutation, quantile)] = values
        for (regime_name, permutation), samples in cell_samples.items():
            for block_index, (threshold, sample) in enumerate(
                zip(values, samples, strict=True)
            ):
                rows.append(
                    {
                        "noise_regime": regime_name,
                        "noise_permutation": permutation,
                        "calibration_pool": "all_regimes_and_permutations",
                        "quantile": quantile,
                        "block": block_index,
                        "threshold": threshold,
                        "empirical_tail_rate": float(
                            (sample > threshold).to(sample.dtype).mean().item()
                        ),
                        "num_null_residuals_in_cell": int(sample.numel()),
                        "num_null_residuals_in_pool": int(pooled[block_index].numel()),
                    }
                )
    return thresholds, rows


def _huber_kwargs(
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    thresholds: Sequence[float],
    *,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
) -> dict[str, Any]:
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    total_cap = float(spec["influence_cap_total"])
    heterogeneity = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=noise_variances.dtype,
        device=noise_variances.device,
    ).square()
    return {
        "anchor": anchor,
        "noise_variances": noise_variances,
        "block_sizes": blocks,
        "heterogeneity_variances": heterogeneity,
        "standardized_threshold": list(float(value) for value in thresholds),
        "influence_cap": [total_cap / math.sqrt(len(blocks))] * len(blocks),
        "regularization": float(spec["regularization"]),
        "num_steps": int(spec["num_steps"]),
        "variance_floor": float(spec["variance_floor"]),
    }


def _g0c_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["references"]["g0c_locked"]
    return {
        "id": "g0c_locked",
        "influence_cap_total": float(source["influence_cap_total"]),
        "regularization": float(source["regularization"]),
        "num_steps": int(source["num_steps"]),
        "variance_floor": float(config["references"]["g0d_grid"]["variance_floor"]),
    }


def _reference(
    name: str,
    vectors: torch.Tensor,
    config: Mapping[str, Any],
    *,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
    candidate_spec: Mapping[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if name == "uniform_mean":
        return vectors.mean(dim=0), {}
    if name == "fcc":
        return (
            centered_clipping(
                vectors,
                anchor=anchor,
                tau=float(config["references"]["fcc"]["radius"]),
            ),
            {},
        )
    if name == "rfa":
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
    if name == "trimmed_mean":
        return (
            trimmed_mean(
                vectors, f=int(config["references"]["trimmed_mean"]["trim_count"])
            ),
            {},
        )
    if name == "coordinate_median":
        return _coordinate_median(vectors), {}
    if name == "g0c_locked":
        source = config["references"]["g0c_locked"]
        frozen_thresholds = g0c._radial_thresholds(
            config["cohort"]["block_sizes"],
            tail_probability=float(source["radial_base_tail_probability"]),
            scale=float(source["radial_scale"]),
        )
        spec = _g0c_spec(config)
        return gaussian_aware_huber_reference(
            vectors,
            **_huber_kwargs(
                config,
                spec,
                frozen_thresholds,
                anchor=anchor,
                noise_variances=noise_variances,
            ),
            return_diagnostics=True,
        )
    if name.startswith("g0d_"):
        if candidate_spec is None or thresholds is None:
            raise ValueError("A G0d candidate requires its spec and calibrated radii")
        return gaussian_aware_huber_reference(
            vectors,
            **_huber_kwargs(
                config,
                candidate_spec,
                thresholds,
                anchor=anchor,
                noise_variances=noise_variances,
            ),
            return_diagnostics=True,
        )
    raise ValueError(f"Unknown reference {name!r}")


def _tail_rate(
    vectors: torch.Tensor,
    reference: torch.Tensor,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    thresholds: Sequence[float],
    noise_variances: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    slices = _block_slices(config["cohort"]["block_sizes"])
    effective = _effective_variances(config, noise_variances)
    total_cap = float(spec["influence_cap_total"])
    cap = total_cap / math.sqrt(len(slices))
    flags: list[torch.Tensor] = []
    for block_index, block_slice in enumerate(slices):
        radii = torch.minimum(
            effective[:, block_index].sqrt() * float(thresholds[block_index]),
            torch.full_like(effective[:, block_index], cap),
        )
        residual = torch.linalg.vector_norm(
            vectors[:, block_slice] - reference[block_slice], dim=1
        )
        flags.append(residual > radii)
    stacked = torch.stack(flags, dim=1)
    if not bool(mask.any().item()):
        return float("nan")
    return float(stacked[mask].to(vectors.dtype).mean().item())


def _evaluate_pairing(
    *,
    config: dict[str, Any],
    phase: str,
    pairing_id: str,
    vectors: torch.Tensor,
    clean: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
    centre: torch.Tensor,
    regime: str,
    permutation: str,
    geometry: str,
    threat: str,
    severity: float,
    seed: int,
    draw: int,
    specs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, float], Sequence[float]],
) -> list[dict[str, Any]]:
    bounded = clip_l2(vectors, float(config["aggregation"]["server_clip_norm"]))
    honest = ~byzantine_mask
    target = clean[honest].mean(dim=0)
    regular_honest = honest & ~outlier_mask
    entries: list[tuple[str, Mapping[str, Any] | None]] = [
        (name, None) for name in COMPARATORS
    ] + [(str(spec["id"]), spec) for spec in specs]
    rows: list[dict[str, Any]] = []
    for name, spec in entries:
        thresholds = (
            calibrated[(regime, permutation, float(spec["quantile"]))]
            if spec is not None
            else None
        )
        reference, diagnostics = _reference(
            name,
            bounded,
            config,
            anchor=anchor,
            noise_variances=noise_variances,
            candidate_spec=spec,
            thresholds=thresholds,
        )
        error = float(torch.linalg.vector_norm(reference - target).item())
        population_error = float(torch.linalg.vector_norm(reference - centre).item())
        is_candidate = spec is not None
        rows.append(
            {
                "campaign_id": config["campaign_id"],
                "phase": phase,
                "pairing_id": pairing_id,
                "candidate": name,
                "is_g0d_candidate": is_candidate,
                "noise_regime": regime,
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "severity": severity,
                "seed": seed,
                "draw": draw,
                "reference_error": error,
                "reference_error_to_population_centre": population_error,
                "reference_error_ratio_to_uniform": float("nan"),
                "reference_error_ratio_to_fcc": float("nan"),
                "regular_honest_tail_rate": (
                    _tail_rate(
                        bounded,
                        reference,
                        config,
                        spec,
                        thresholds,
                        noise_variances,
                        regular_honest,
                    )
                    if is_candidate
                    else float("nan")
                ),
                "replace_one_bound": float(
                    diagnostics.get("finite_solver_replace_one_bound", float("nan"))
                ),
                "solver_gradient_residual": float(
                    diagnostics.get("gradient_residual_norm", float("nan"))
                ),
                "covariance_limited_and_tail_fraction": float(
                    diagnostics.get(
                        "fraction_covariance_limited_and_huber_tail_client_blocks",
                        float("nan"),
                    )
                ),
                "resolved_device": str(bounded.device),
                "tensor_dtype": str(bounded.dtype).replace("torch.", ""),
            }
        )
    uniform = next(row for row in rows if row["candidate"] == "uniform_mean")
    fcc = next(row for row in rows if row["candidate"] == "fcc")
    if uniform["reference_error"] <= 0.0 or fcc["reference_error"] <= 0.0:
        raise RuntimeError("A paired baseline has zero error; ratios are undefined")
    for row in rows:
        row["reference_error_ratio_to_uniform"] = (
            row["reference_error"] / uniform["reference_error"]
        )
        row["reference_error_ratio_to_fcc"] = (
            row["reference_error"] / fcc["reference_error"]
        )
    return rows


def _phase_rows(
    *,
    config: dict[str, Any],
    phase: str,
    seeds: Sequence[int],
    draws_per_seed: int,
    severities: Sequence[float],
    specs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, float], Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = oracle._noise_variances(config, regime, permutation)
            for seed_index, seed_value in enumerate(seeds, start=1):
                seed = int(seed_value)
                print(
                    f"[G0d-F] {phase} seed {seed_index}/{len(seeds)}: {seed}",
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
                                        "g0d-attack", seed, draw, geometry, threat
                                    ),
                                )
                                pairing_id = (
                                    f"{phase}:{regime_name}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{seed}:{draw}"
                                )
                                rows.extend(
                                    _evaluate_pairing(
                                        config=config,
                                        phase=phase,
                                        pairing_id=pairing_id,
                                        vectors=attacked,
                                        clean=clean,
                                        outlier_mask=outliers,
                                        byzantine_mask=byzantine,
                                        anchor=anchor,
                                        noise_variances=variances,
                                        centre=centre,
                                        regime=regime_name,
                                        permutation=permutation,
                                        geometry=geometry,
                                        threat=threat,
                                        severity=float(severity),
                                        seed=seed,
                                        draw=draw,
                                        specs=specs,
                                        calibrated=calibrated,
                                    )
                                )
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


def _identity_reverse_ci95(
    rows: Sequence[dict[str, Any]],
) -> tuple[float, float, float, int]:
    clean = [
        row
        for row in rows
        if row["threat"] == "none" and row["noise_regime"] == "heteroscedastic"
    ]
    pairs: dict[tuple[int, int, str], dict[str, float]] = defaultdict(dict)
    for row in clean:
        key = (int(row["seed"]), int(row["draw"]), str(row["outlier_geometry"]))
        pairs[key][str(row["noise_permutation"])] = float(
            row["reference_error_ratio_to_uniform"]
        )
    per_seed: dict[int, list[float]] = defaultdict(list)
    for (seed, _, _), values in pairs.items():
        if set(values) != {"identity", "reverse"}:
            raise RuntimeError("Identity/reverse pairing is incomplete")
        per_seed[seed].append(values["identity"] - values["reverse"])
    seed_means = [_finite_mean(values) for _, values in sorted(per_seed.items())]
    mean = _finite_mean(seed_means)
    if len(seed_means) < 2:
        return mean, float("-inf"), float("inf"), len(seed_means)
    sd = statistics.stdev(seed_means)
    half = _critical_t95(len(seed_means)) * sd / math.sqrt(len(seed_means))
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


def _replace_one_audit(
    *,
    config: dict[str, Any],
    phase: str,
    seeds: Sequence[int],
    specs: Sequence[Mapping[str, Any]],
    calibrated: Mapping[tuple[str, str, float], Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    server_clip = float(config["aggregation"]["server_clip_norm"])
    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation_value in regime["permutations"]:
            permutation = str(permutation_value)
            variances, _ = oracle._noise_variances(config, regime, permutation)
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
                        geometry="g0d-replace",
                        pair_permutations=bool(
                            config["randomness"]["pair_noise_across_tier_permutations"]
                        ),
                    )
                    bounded = clip_l2(observed, server_clip)
                    index = trial % int(bounded.shape[0])
                    replacement = torch.randn(
                        bounded.shape[1],
                        generator=oracle._generator(
                            "g0d-replacement", phase, seed, trial
                        ),
                        dtype=bounded.dtype,
                        device=bounded.device,
                    )
                    replacement = clip_l2(
                        (2.0 * server_clip * replacement)[None, :], server_clip
                    )[0]
                    neighbour = bounded.clone()
                    neighbour[index] = replacement
                    entries: list[tuple[str, Mapping[str, Any] | None]] = [
                        (name, None) for name in COMPARATORS
                    ] + [(str(spec["id"]), spec) for spec in specs]
                    for name, spec in entries:
                        thresholds = (
                            calibrated[
                                (regime_name, permutation, float(spec["quantile"]))
                            ]
                            if spec is not None
                            else None
                        )
                        left, diagnostics = _reference(
                            name,
                            bounded,
                            config,
                            anchor=anchor,
                            noise_variances=variances,
                            candidate_spec=spec,
                            thresholds=thresholds,
                        )
                        right, _ = _reference(
                            name,
                            neighbour,
                            config,
                            anchor=anchor,
                            noise_variances=variances,
                            candidate_spec=spec,
                            thresholds=thresholds,
                        )
                        observed_delta = float(
                            torch.linalg.vector_norm(left - right).item()
                        )
                        if spec is not None or name == "g0c_locked":
                            bound = float(
                                diagnostics["finite_solver_replace_one_bound"]
                            )
                            certificate = "finite_solver_fixed_anchor_covariances"
                        elif name == "uniform_mean":
                            bound = 2.0 * server_clip / float(bounded.shape[0])
                            certificate = "replace_one_after_server_clip"
                        elif name == "fcc":
                            bound = (
                                2.0
                                * float(config["references"]["fcc"]["radius"])
                                / float(bounded.shape[0])
                            )
                            certificate = "replace_one_fixed_anchor"
                        elif name == "trimmed_mean":
                            n = float(bounded.shape[0])
                            d = float(bounded.shape[1])
                            trim = float(
                                config["references"]["trimmed_mean"]["trim_count"]
                            )
                            bound = 2.0 * server_clip * math.sqrt(d) / (n - 2.0 * trim)
                            certificate = (
                                "dimension_dependent_bound_only;"
                                "no_dimension_free_o_U_over_n_l2_certificate"
                            )
                        else:
                            bound = float("nan")
                            certificate = (
                                "no_dimension_free_o_U_over_n_l2_certificate_"
                                "under_g0d_contract"
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
                            }
                        )
    return rows


def _penalty(summary: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    gates = config["gates"]

    def upper(value: float, limit: float) -> float:
        return 10.0 if not math.isfinite(value) else max(0.0, value / limit - 1.0)

    def lower(value: float, limit: float) -> float:
        return 10.0 if not math.isfinite(value) else max(0.0, 1.0 - value / limit)

    penalty = upper(
        float(summary["clean_error_ratio_worst_group"]),
        float(gates["clean_reference_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["attacked_error_ratio_to_uniform_worst_group"]),
        float(gates["attacked_reference_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["attacked_error_ratio_to_fcc_worst_group"]),
        float(gates["attacked_reference_error_ratio_to_fcc_max"]),
    )
    penalty += upper(
        float(summary["evasive_error_ratio_to_uniform_worst_group"]),
        float(gates["evasive_reference_error_ratio_to_uniform_max"]),
    )
    penalty += upper(
        float(summary["evasive_error_ratio_to_fcc_worst_group"]),
        float(gates["evasive_reference_error_ratio_to_fcc_max"]),
    )
    rate = float(summary["regular_honest_tail_rate"])
    if rate < float(gates["regular_honest_tail_rate_min"]):
        penalty += lower(rate, float(gates["regular_honest_tail_rate_min"]))
    else:
        penalty += upper(rate, float(gates["regular_honest_tail_rate_max"]))
    penalty += upper(
        max(
            abs(float(summary["identity_reverse_ci95_low"])),
            abs(float(summary["identity_reverse_ci95_high"])),
        ),
        float(gates["identity_reverse_error_ratio_ci95_abs_max"]),
    )
    penalty += upper(
        float(summary["replace_one_bound_max"]),
        float(gates["replace_one_bound_max"]),
    )
    penalty += upper(
        float(summary["solver_gradient_residual_max"]),
        float(gates["solver_gradient_residual_max"]),
    )
    penalty += lower(
        float(summary["covariance_limited_and_tail_fraction"]),
        float(gates["covariance_limited_and_tail_fraction_min"]),
    )
    return penalty


def _summaries(
    rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gates = config["gates"]
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        identifier = str(spec["id"])
        selected = [row for row in rows if row["candidate"] == identifier]
        clean = [row for row in selected if row["threat"] == "none"]
        attacked = [
            row
            for row in selected
            if row["threat"] in set(config["threats"]["separated_for_gates"])
        ]
        evasive = [
            row
            for row in selected
            if row["threat"] in set(config["threats"]["evasive_controls"])
        ]
        stable = [row for row in stability_rows if row["candidate"] == identifier]
        expected = len({str(row["pairing_id"]) for row in rows})
        complete_fraction = len(selected) / float(expected) if expected else 0.0
        finite_fraction = _finite_mean(
            float(
                all(math.isfinite(float(row[field])) for field in PRIMARY_FINITE_FIELDS)
            )
            for row in selected
        )
        ci_mean, ci_low, ci_high, ci_n = _identity_reverse_ci95(selected)
        summary: dict[str, Any] = {
            "candidate": identifier,
            "observations": len(selected),
            "complete_fraction": complete_fraction,
            "finite_primary_metric_fraction": finite_fraction,
            "clean_error_ratio_worst_group": _group_worst(
                clean, "reference_error_ratio_to_uniform"
            ),
            "attacked_error_ratio_to_uniform_worst_group": _group_worst(
                attacked, "reference_error_ratio_to_uniform"
            ),
            "attacked_error_ratio_to_fcc_worst_group": _group_worst(
                attacked, "reference_error_ratio_to_fcc"
            ),
            "evasive_error_ratio_to_uniform_worst_group": _group_worst(
                evasive, "reference_error_ratio_to_uniform"
            ),
            "evasive_error_ratio_to_fcc_worst_group": _group_worst(
                evasive, "reference_error_ratio_to_fcc"
            ),
            "regular_honest_tail_rate": _finite_mean(
                row["regular_honest_tail_rate"] for row in clean
            ),
            "identity_reverse_seed_mean": ci_mean,
            "identity_reverse_ci95_low": ci_low,
            "identity_reverse_ci95_high": ci_high,
            "identity_reverse_seed_clusters": ci_n,
            "replace_one_bound_max": max(
                float(row["theoretical_bound"]) for row in stable
            ),
            "replace_one_observed_max": max(
                float(row["observed_delta"]) for row in stable
            ),
            "replace_one_violation_count": sum(
                bool(row["violation"]) for row in stable
            ),
            "solver_gradient_residual_max": max(
                float(row["solver_gradient_residual"]) for row in selected
            ),
            "covariance_limited_and_tail_fraction": _finite_mean(
                row["covariance_limited_and_tail_fraction"] for row in clean
            ),
        }
        checks = {
            "complete": complete_fraction >= float(gates["complete_fraction_min"]),
            "finite": finite_fraction
            >= float(gates["finite_primary_metric_fraction_min"]),
            "clean_error": summary["clean_error_ratio_worst_group"]
            <= float(gates["clean_reference_error_ratio_to_uniform_max"]),
            "attacked_vs_uniform": summary[
                "attacked_error_ratio_to_uniform_worst_group"
            ]
            <= float(gates["attacked_reference_error_ratio_to_uniform_max"]),
            "attacked_vs_fcc": summary["attacked_error_ratio_to_fcc_worst_group"]
            <= float(gates["attacked_reference_error_ratio_to_fcc_max"]),
            "evasive_vs_uniform": summary["evasive_error_ratio_to_uniform_worst_group"]
            <= float(gates["evasive_reference_error_ratio_to_uniform_max"]),
            "evasive_vs_fcc": summary["evasive_error_ratio_to_fcc_worst_group"]
            <= float(gates["evasive_reference_error_ratio_to_fcc_max"]),
            "honest_tail_rate": float(gates["regular_honest_tail_rate_min"])
            <= summary["regular_honest_tail_rate"]
            <= float(gates["regular_honest_tail_rate_max"]),
            "identity_reverse_invariance": max(abs(ci_low), abs(ci_high))
            <= float(gates["identity_reverse_error_ratio_ci95_abs_max"]),
            "replace_one_bound": summary["replace_one_bound_max"]
            <= float(gates["replace_one_bound_max"]),
            "replace_one_empirical": summary["replace_one_violation_count"]
            <= int(gates["empirical_replace_one_violation_max"]),
            "solver_residual": summary["solver_gradient_residual_max"]
            <= float(gates["solver_gradient_residual_max"]),
            "covariance_effective": summary["covariance_limited_and_tail_fraction"]
            >= float(gates["covariance_limited_and_tail_fraction_min"]),
        }
        for name, passed in checks.items():
            summary[f"gate_{name}"] = bool(passed)
        summary["gate_fail_count"] = sum(not value for value in checks.values())
        summary["passes_all_gates"] = all(checks.values())
        summary["normalized_gate_penalty"] = _penalty(summary, config)
        summaries.append(summary)
    return summaries


def _comparator_summaries(
    rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate]
        clean = [row for row in selected if row["threat"] == "none"]
        attacked = [
            row
            for row in selected
            if row["threat"] in set(config["threats"]["separated_for_gates"])
        ]
        evasive = [
            row
            for row in selected
            if row["threat"] in set(config["threats"]["evasive_controls"])
        ]
        stable = [row for row in stability_rows if row["candidate"] == candidate]
        ci_mean, ci_low, ci_high, ci_n = _identity_reverse_ci95(selected)
        certificates = sorted(set(str(row["certificate"]) for row in stable))
        summaries.append(
            {
                "candidate": candidate,
                "observations": len(selected),
                "clean_error_ratio_to_uniform_worst_group": _group_worst(
                    clean, "reference_error_ratio_to_uniform"
                ),
                "attacked_error_ratio_to_uniform_worst_group": _group_worst(
                    attacked, "reference_error_ratio_to_uniform"
                ),
                "attacked_error_ratio_to_fcc_worst_group": _group_worst(
                    attacked, "reference_error_ratio_to_fcc"
                ),
                "evasive_error_ratio_to_uniform_worst_group": _group_worst(
                    evasive, "reference_error_ratio_to_uniform"
                ),
                "evasive_error_ratio_to_fcc_worst_group": _group_worst(
                    evasive, "reference_error_ratio_to_fcc"
                ),
                "clean_population_error_mean": _finite_mean(
                    row["reference_error_to_population_centre"] for row in clean
                ),
                "identity_reverse_seed_mean": ci_mean,
                "identity_reverse_ci95_low": ci_low,
                "identity_reverse_ci95_high": ci_high,
                "identity_reverse_seed_clusters": ci_n,
                "empirical_replace_one_max": max(
                    float(row["observed_delta"]) for row in stable
                ),
                "global_o_1_over_n_certificate": ";".join(certificates),
                "theoretical_bound": (
                    max(
                        float(row["theoretical_bound"])
                        for row in stable
                        if math.isfinite(float(row["theoretical_bound"]))
                    )
                    if any(
                        math.isfinite(float(row["theoretical_bound"])) for row in stable
                    )
                    else "N/A"
                ),
                "huber_tail_covariance_solver_metrics": (
                    "reported_in_candidate_gate_table"
                    if candidate.startswith("g0d_")
                    else "N/A"
                ),
            }
        )
    return summaries


def _select(
    summaries: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    order = {str(spec["id"]): index for index, spec in enumerate(specs)}
    ranked = sorted(
        (dict(row) for row in summaries),
        key=lambda row: (
            int(row["gate_fail_count"]),
            float(row["normalized_gate_penalty"]),
            order[str(row["candidate"])],
        ),
    )
    return ranked[0]


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    lock: Mapping[str, Any],
    holdout: Mapping[str, Any],
    comparators: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    output_dir: Path,
) -> None:
    lines = [
        "# G0d-F — référence Gaussian-aware calibrée sous le nul",
        "",
        "## Verdict",
        "",
        (
            "**Promotion synthétique autorisée.** Le candidat verrouillé passe "
            "tous les gates en développement et sur le holdout."
            if decision["promote"]
            else "**Aucune promotion.** Le candidat verrouillé échoue au moins "
            "un gate de développement ou de holdout."
        ),
        "",
        "Cet audit n'utilise ni poids FAR, ni accuracy, ni entraînement de "
        "réseau. Les seuils ont été calibrés sur des seeds nulles indépendantes "
        "avant tout développement; le holdout n'a participé à aucun choix.",
        "",
        "## Verrou et gates",
        "",
        f"- candidat : `{lock['candidate']}`;",
        f"- échecs développement : `{lock['gate_fail_count']}`;",
        f"- échecs holdout : `{holdout['gate_fail_count']}`;",
        f"- décision : `promote={str(decision['promote']).lower()}`.",
        "",
        "Les attaques séparées sont exactement `ipm`, `bitflip_x10` et "
        "`model_replacement`. `alie` est le contrôle furtif, avec son gate "
        "distinct. Les ensembles sont disjoints et couvrent toutes les attaques.",
        "",
        "| Phase | Propre / uniforme | Séparées / uniforme | Séparées / FCC | ALIE / uniforme | ALIE / FCC | Queue honnête | IC95 identity-reverse | Borne replace-one | Max observé | Résidu solveur | Covariance & tail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| Développement | {c:.3f} | {a:.3f} | {f:.3f} | {eu:.3f} | {ef:.3f} | {t:.3f} | [{lo:.3f}, {hi:.3f}] | {b:.3f} | {o:.3f} | {r:.2e} | {v:.3f} |".format(
            c=float(lock["clean_error_ratio_worst_group"]),
            a=float(lock["attacked_error_ratio_to_uniform_worst_group"]),
            f=float(lock["attacked_error_ratio_to_fcc_worst_group"]),
            eu=float(lock["evasive_error_ratio_to_uniform_worst_group"]),
            ef=float(lock["evasive_error_ratio_to_fcc_worst_group"]),
            t=float(lock["regular_honest_tail_rate"]),
            lo=float(lock["identity_reverse_ci95_low"]),
            hi=float(lock["identity_reverse_ci95_high"]),
            b=float(lock["replace_one_bound_max"]),
            o=float(lock["replace_one_observed_max"]),
            r=float(lock["solver_gradient_residual_max"]),
            v=float(lock["covariance_limited_and_tail_fraction"]),
        ),
        "| Holdout | {c:.3f} | {a:.3f} | {f:.3f} | {eu:.3f} | {ef:.3f} | {t:.3f} | [{lo:.3f}, {hi:.3f}] | {b:.3f} | {o:.3f} | {r:.2e} | {v:.3f} |".format(
            c=float(holdout["clean_error_ratio_worst_group"]),
            a=float(holdout["attacked_error_ratio_to_uniform_worst_group"]),
            f=float(holdout["attacked_error_ratio_to_fcc_worst_group"]),
            eu=float(holdout["evasive_error_ratio_to_uniform_worst_group"]),
            ef=float(holdout["evasive_error_ratio_to_fcc_worst_group"]),
            t=float(holdout["regular_honest_tail_rate"]),
            lo=float(holdout["identity_reverse_ci95_low"]),
            hi=float(holdout["identity_reverse_ci95_high"]),
            b=float(holdout["replace_one_bound_max"]),
            o=float(holdout["replace_one_observed_max"]),
            r=float(holdout["solver_gradient_residual_max"]),
            v=float(holdout["covariance_limited_and_tail_fraction"]),
        ),
        "",
        "## Comparateurs appariés sur holdout",
        "",
        "| Référence | Propre / uniforme | Séparées / uniforme | Séparées / FCC | ALIE / uniforme | ALIE / FCC | Erreur population propre | IC95 identity-reverse | Delta replace observé | Certificat global |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparators:
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | [{:.3f}, {:.3f}] | {:.4f} | `{}` |".format(
                row["candidate"],
                float(row["clean_error_ratio_to_uniform_worst_group"]),
                float(row["attacked_error_ratio_to_uniform_worst_group"]),
                float(row["attacked_error_ratio_to_fcc_worst_group"]),
                float(row["evasive_error_ratio_to_uniform_worst_group"]),
                float(row["evasive_error_ratio_to_fcc_worst_group"]),
                float(row["clean_population_error_mean"]),
                float(row["identity_reverse_ci95_low"]),
                float(row["identity_reverse_ci95_high"]),
                float(row["empirical_replace_one_max"]),
                row["global_o_1_over_n_certificate"],
            )
        )
    lines += [
        "",
        "## Interprétation",
        "",
        "Le seuil empirique ne vient d'aucun résultat d'accuracy et n'est pas "
        "recalculé sur la cohorte observée. La simulation nulle publique inclut "
        "la dispersion honnête, la covariance DP authentifiée et le clipping "
        "serveur; ses quantiles deviennent ensuite des hyperparamètres publics.",
        "",
        "Les bornes replace-one concernent le candidat G0d-F à ancre et "
        "covariances publiques fixées. CM, trMean et RFA sont des comparateurs "
        "d'erreur robuste; aucune certification O(1/n) n'est inférée de leurs "
        "seules performances empiriques.",
        "",
        "## Reproductibilité",
        "",
        f"- sorties : `{output_dir.resolve()}`;",
        f"- seeds nulles : `{config['randomness']['null_calibration_seeds']}`;",
        f"- seeds développement : `{config['randomness']['development_seeds']}`;",
        f"- seeds holdout : `{config['randomness']['holdout_seeds']}`;",
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
    runtime_device, runtime_dtype = oracle._configure_runtime("mps")
    if runtime_device.type != "mps" or runtime_dtype != torch.float32:
        raise RuntimeError("G0d-F requires real MPS float32 execution")
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
    checkpoint_dir = output_dir / "_checkpoints"

    specs = _candidate_specs(config)
    calibrated, calibration_rows = _null_thresholds(config)
    _write_csv(output_dir / "null_calibration.csv", calibration_rows)

    development_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
        draws_per_seed=int(config["randomness"]["development_draws_per_seed"]),
        severities=config["threats"]["development_severities"],
        specs=specs,
        calibrated=calibrated,
    )
    development_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        phase="development",
        seeds=config["randomness"]["development_seeds"],
        specs=specs,
        calibrated=calibrated,
    )
    development_summaries = _summaries(
        development_rows, development_stability, specs, config
    )
    lock = _select(development_summaries, specs)
    lock_payload = {
        **lock,
        "selected_on_phase": "development_only",
        "holdout_used_for_selection": False,
    }
    _write_csv(output_dir / "development_detail.csv", development_rows)
    _write_csv(output_dir / "development_stability.csv", development_stability)
    _write_csv(output_dir / "development_summary.csv", development_summaries)
    _atomic_json(output_dir / "development_lock.json", lock_payload)

    locked_spec = next(spec for spec in specs if spec["id"] == lock["candidate"])
    holdout_rows = _cached_phase_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        phase="holdout",
        seeds=config["randomness"]["holdout_seeds"],
        draws_per_seed=int(config["randomness"]["holdout_draws_per_seed"]),
        severities=config["threats"]["holdout_severities"],
        specs=[locked_spec],
        calibrated=calibrated,
    )
    holdout_stability = _cached_stability_rows(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        config=config,
        phase="holdout",
        seeds=config["randomness"]["holdout_seeds"],
        specs=[locked_spec],
        calibrated=calibrated,
    )
    holdout = _summaries(holdout_rows, holdout_stability, [locked_spec], config)[0]
    comparator_summaries = _comparator_summaries(
        holdout_rows,
        holdout_stability,
        [*COMPARATORS, str(locked_spec["id"])],
        config,
    )
    development_pass = bool(lock["passes_all_gates"])
    holdout_pass = bool(holdout["passes_all_gates"])
    decision = {
        "campaign_id": config["campaign_id"],
        "requested_device": "mps",
        "resolved_device": "mps",
        "tensor_dtype": "float32",
        "silent_cpu_fallback_allowed": False,
        "reference_only": True,
        "far_weights_used": False,
        "accuracy_used": False,
        "null_calibration_independent": True,
        "holdout_used_for_selection": False,
        "locked_candidate": lock["candidate"],
        "development_passes": development_pass,
        "holdout_passes": holdout_pass,
        "development_gate_fail_count": int(lock["gate_fail_count"]),
        "holdout_gate_fail_count": int(holdout["gate_fail_count"]),
        "promote": bool(development_pass and holdout_pass),
        "promotion_rule": "all_preregistered_development_and_holdout_gates_pass",
    }
    _write_csv(output_dir / "holdout_detail.csv", holdout_rows)
    _write_csv(output_dir / "holdout_stability.csv", holdout_stability)
    _write_csv(output_dir / "holdout_summary.csv", [holdout])
    _write_csv(output_dir / "holdout_comparators.csv", comparator_summaries)
    _atomic_json(output_dir / "decision.json", decision)
    _write_report(
        report_path,
        config=config,
        lock=lock,
        holdout=holdout,
        comparators=comparator_summaries,
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
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0d_f.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0d_f_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_Robust_Reference_G0d_F_MPS.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only checkpoints whose config SHA-256 matches exactly.",
    )
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
