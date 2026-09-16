#!/usr/bin/env python3
"""Fit/freeze then evaluate the preregistered G0g-K5-TP predictor on MPS.

The two phases are intentionally separate.  ``fit-freeze`` is allowed to use
privileged K4c-CH labels on synthetic train/calibration seeds.  It freezes the
feature scales, lambda and coefficients.  ``evaluate-frozen`` refuses to
generate an evaluation trajectory until the exact frozen-predictor file hash
has been published externally and supplied on the command line.
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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k4b import (  # noqa: E402
    FULL_TEMPORAL_MISSING_SLOT,
    pointwise_optimal_full_imputation_predictor,
)
from algorithms.gaussian_aware_reference_k5_tp import (  # noqa: E402
    FEATURE_NAMES,
    fit_shared_scalar_ridge,
    forbidden_current_field_count,
    training_feature_scales,
    transcript_past_feature_dictionary,
    transcript_past_predictor,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k4_tcg as k4  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k4b_past_imputation as k4b,
)
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k4c_causal_headroom as k4c,
)
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

K2 = "g0g_k2"
K4 = "g0g_k4_temporal_causal_gate"
K4B = "g0g_k4b_rolling_past_imputation"
K5_1D = "g0g_k5_tp_one_dimensional_control"
K5 = "g0g_k5_tp_shared_scalar_ridge"
K4C = "g0g_k4c_ch_privileged_benchmark"
POINTWISE = "g0g_k4b_pointwise_optimal_oracle"
CANDIDATES = (K2, K4, K4B, K5_1D, K5, K4C, POINTWISE)

DEFAULT_CONFIG = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.yaml"
)
DEFAULT_LOCK = ROOT / (
    "configs/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor.lock.json"
)
DEFAULT_OUTPUT = ROOT / (
    "results/ldp_gradient_far/"
    "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
)
LOCKED_PATHS = {
    "algorithms/gaussian_aware_reference_k5_tp.py",
    "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k5_transcript_predictor.yaml",
    "scripts/audit_gaussian_aware_reference_g0g_k5_tp.py",
    "scripts/run_gaussian_aware_reference_g0g_k5_transcript_predictor.py",
    "tests/test_audit_gaussian_aware_reference_g0g_k5_tp.py",
    "tests/test_gaussian_aware_reference_g0g_k5_tp.py",
    "tests/test_run_gaussian_aware_reference_g0g_k5_transcript_predictor.py",
    "output/analysis/Gaussian_Aware_G0g_K5_TP_Protocol_PreRun.md",
}
DEPENDENCY_PATHS = {
    "algorithms/gaussian_aware_reference.py",
    "algorithms/gaussian_aware_reference_k4b.py",
    "algorithms/gaussian_aware_reference_k4c_ch.py",
    "scripts/run_gaussian_aware_reference_g0g_k1.py",
    "scripts/run_gaussian_aware_reference_g0g_k2.py",
    "scripts/run_gaussian_aware_reference_g0g_k4_tcg.py",
    "scripts/run_gaussian_aware_reference_g0g_k4b_past_imputation.py",
    "scripts/run_gaussian_aware_reference_g0g_k4c_causal_headroom.py",
    "scripts/run_gaussian_aware_reference_oracle.py",
    "robustness/aggregators.py",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k2_mps_v1/calibration.json",
    "results/ldp_gradient_far/gaussian_aware_reference_g0g_k4_tcg_mps_v1/temporal_calibration.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else float("nan")


def _ci(values: Sequence[float], t_critical: float) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        return {"n": len(numbers), "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    mean = statistics.fmean(numbers)
    if len(numbers) == 1:
        return {"n": 1, "mean": mean, "low": mean, "high": mean}
    sd = statistics.stdev(numbers)
    half = float(t_critical) * sd / math.sqrt(len(numbers))
    return {"n": len(numbers), "mean": mean, "low": mean - half, "high": mean + half}


def _verify_lock(lock_path: Path, config_path: Path) -> dict[str, Any]:
    if lock_path.resolve() != DEFAULT_LOCK.resolve() or config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise RuntimeError("Production accepts only the preregistered K5 paths")
    registry = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "campaign_id",
        "locked_files",
        "dependencies",
        "lock_file_self_hash_embedded",
        "publication_requirement",
    }
    if set(registry) != expected_keys:
        raise RuntimeError("K5 lock schema mismatch")
    if registry["schema_version"] != 1 or registry["campaign_id"] != (
        "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1"
    ):
        raise RuntimeError("K5 lock identity mismatch")
    if registry["lock_file_self_hash_embedded"] is not False:
        raise RuntimeError("The K5 lock must not claim a self-hash")
    if registry["publication_requirement"] != (
        "publish_lock_sha256_before_fit_and_frozen_predictor_sha256_before_evaluation"
    ):
        raise RuntimeError("K5 publication contract changed")
    if set(registry["locked_files"]) != LOCKED_PATHS:
        raise RuntimeError("K5 locked-file registry mismatch")
    if set(registry["dependencies"]) != DEPENDENCY_PATHS:
        raise RuntimeError("K5 dependency registry mismatch")
    for relative, expected_hash in {
        **registry["locked_files"],
        **registry["dependencies"],
    }.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise RuntimeError(f"K5 preregistration hash mismatch: {relative}")
    return {"path": str(lock_path.resolve()), "sha256": _sha256(lock_path), "verified": True}


def _attest(actual: str, published: str, *, name: str) -> dict[str, Any]:
    expected = str(actual).strip().lower()
    provided = str(published).strip().lower()
    if provided != expected:
        raise RuntimeError(f"Published {name} SHA-256 does not match the verified file")
    return {
        "procedural_external_publication_attested": True,
        f"published_{name}_sha256": provided,
        "machine_verifies_external_log_itself": False,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("campaign_id") != "gaussian_aware_reference_g0g_k5_transcript_predictor_mps_v1":
        raise ValueError("Unexpected K5 campaign id")
    if tuple(config["features"]["order"]) != FEATURE_NAMES:
        raise ValueError("K5 feature schema differs from the implementation")
    if config["features"]["normalization_formula"] != (
        "sqrt_sum_h_norm_V_j_h_squared_over_H_times_dimension"
    ):
        raise ValueError("K5 per-coordinate RMS formula changed")
    if int(config["features"]["history_length"]) != 4:
        raise ValueError("K5 requires L=4")
    if config["features"]["rank_diagnostic"] != (
        "two_pass_modified_gram_schmidt_on_unit_norm_columns"
    ) or config["features"]["rank_relative_tolerance"] != (
        "max_H_times_d_and_J_times_float32_epsilon"
    ):
        raise ValueError("K5 MPS rank diagnostic changed")
    if int(config["features"]["minimum_accepted_mass"]) != 20:
        raise ValueError("K5 requires the frozen public accepted-mass floor n-b=20")
    if tuple(float(value) for value in config["ridge"]["lambda_grid"]) != (
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
    ):
        raise ValueError("K5 lambda grid changed")
    ridge = config["ridge"]
    if ridge["refit_after_selection"] != "train_plus_calibration":
        raise ValueError("K5 final coefficients must be refitted on train+calibration")
    if ridge["condition_number_norm"] != "infinity_exact_via_solve":
        raise ValueError("K5 MPS condition-number diagnostic changed")
    if ridge["calibration_metric"] != "equal_seed_mean_of_integrated_projected_aggregate_mse":
        raise ValueError("K5 lambda selection metric changed")
    if float(ridge["selection_tie_absolute_tolerance"]) != 1e-12 or ridge["tie_break"] != "largest_lambda":
        raise ValueError("K5 lambda tie rule changed")
    streams = [
        str(value)
        for key, value in config["nested_monte_carlo"].items()
        if key.endswith("_stream_tag")
    ]
    if len(streams) != 5 or len(set(streams)) != 5:
        raise ValueError("All five K5 child streams must be distinct")
    randomness = config["randomness"]
    train = tuple(int(v) for v in randomness["train_outer_seeds"])
    calibration = tuple(int(v) for v in randomness["calibration_outer_seeds"])
    evaluation = tuple(int(v) for v in randomness["evaluation_outer_seeds"])
    holdout = tuple(int(v) for v in randomness["reserved_holdout_seeds"])
    if (len(train), len(calibration), len(evaluation), len(holdout)) != (12, 8, 12, 7):
        raise ValueError("K5 split sizes changed")
    groups = (set(train), set(calibration), set(evaluation), set(holdout))
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("K5 outer-seed splits overlap")
    expected_train = {
        2027043003, 2027043017, 2027043029, 2027043041,
        2027043053, 2027043067, 2027043079, 2027043091,
        2027043107, 2027043121, 2027043133, 2027043149,
    }
    if set(train) != expected_train:
        raise ValueError("Only the frozen K4c development seeds may supervise training")
    prior: set[int] = set()
    for pattern in ("configs/ldp_gradient_far/*.yaml", "configs/dt_ldp_far/*.yaml"):
        for path in ROOT.glob(pattern):
            if path.resolve() == DEFAULT_CONFIG.resolve():
                continue
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                prior |= k4b._seed_values(value)
    collision = (set(calibration) | set(evaluation)) & prior
    if collision:
        raise ValueError(f"New K5 calibration/evaluation seeds collide: {sorted(collision)}")
    analysis = config["statistical_analysis"]
    if (int(analysis["train_histories_total"]), int(analysis["calibration_histories_total"]), int(analysis["evaluation_histories_total"])) != (576, 384, 576):
        raise ValueError("K5 history matrix sizes changed")
    nested = config["nested_monte_carlo"]
    expected_seed_counts = {
        "train_target_child_seeds_exact": 576 * int(nested["train_target_construction_children"]),
        "calibration_target_child_seeds_exact": 384 * int(nested["calibration_target_construction_children"]),
        "calibration_evaluation_child_seeds_exact": 384 * int(nested["calibration_evaluation_children"]),
        "evaluation_target_child_seeds_exact": 576 * int(nested["evaluation_target_construction_children"]),
        "evaluation_child_seeds_exact": 576 * int(nested["evaluation_children"]),
    }
    for key, expected in expected_seed_counts.items():
        if int(config["gates"][key]) != expected:
            raise ValueError(f"K5 seed-count gate changed: {key}")
    if config["execution"] != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "ridge_fit_dtype": "float32",
        "ridge_fit_device_must_equal_runtime_device": True,
        "rank_diagnostic_device_must_equal_runtime_device": True,
        "allow_cpu_fallback": False,
        "development_only": True,
        "holdout_code_path_present": False,
        "fit_phase_name": "fit_freeze",
        "evaluation_phase_name": "evaluate_frozen",
        "expected_runtime_minutes_mps": [75, 130],
    }:
        raise ValueError("K5 production execution contract changed")


def _noise_cells(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], str]]:
    return k4c._noise_cells(config)


def _cells(config: Mapping[str, Any], seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for regime, permutation in _noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for dynamics in config["honest_dynamics"]["names"]:
                    for threat in config["threats"]["names"]:
                        rows.append(
                            {
                                "seed": int(seed),
                                "regime": regime,
                                "permutation": str(permutation),
                                "geometry": str(geometry),
                                "dynamics": str(dynamics),
                                "threat": str(threat),
                            }
                        )
    return rows


def _history_id(split: str, cell: Mapping[str, Any], round_index: int) -> str:
    return "|".join(
        (
            split,
            str(cell["seed"]),
            str(cell["regime"]["name"]),
            str(cell["permutation"]),
            str(cell["geometry"]),
            str(cell["dynamics"]),
            str(cell["threat"]),
            str(round_index),
        )
    )


def _fit_history_matrix_is_exact(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    seeds: Sequence[int],
) -> bool:
    expected = {
        _history_id(split, cell, int(round_index))
        for cell in _cells(config, seeds)
        for round_index in config["temporal"]["assessment_rounds"]
    }
    observed = [str(row["history_id"]) for row in rows if row["split"] == split]
    return len(observed) == len(expected) and set(observed) == expected


def _snapshot_contexts(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    *,
    split: str,
    seeds: Sequence[int],
) -> Iterator[dict[str, Any]]:
    cap = float(config["references"]["total_client_influence_cap"])
    minimum = float(config["features"]["minimum_accepted_mass"])
    for cell in _cells(config, seeds):
        components = k4._trajectory_components(
            config,
            seed=int(cell["seed"]),
            regime=dict(cell["regime"]),
            permutation=str(cell["permutation"]),
            geometry=str(cell["geometry"]),
            dynamics=str(cell["dynamics"]),
        )
        history: list[torch.Tensor] = []
        enrollment_mean: torch.Tensor | None = None
        residuals: dict[int, torch.Tensor] = {}
        gates: dict[int, torch.Tensor] = {}
        assessment = {int(value) for value in config["temporal"]["assessment_rounds"]}
        total = int(config["temporal"]["total_rounds_needed_for_frozen_pasts"])
        for round_index in range(1, total + 1):
            if round_index in assessment:
                if enrollment_mean is None:
                    raise RuntimeError("Enrollment baseline missing before K5 snapshot")
                source_rounds = tuple(range(round_index - 4, round_index))
                feature, feature_diagnostics = transcript_past_feature_dictionary(
                    torch.stack([residuals[value] for value in source_rounds]),
                    torch.stack([gates[value] for value in source_rounds]),
                    minimum_accepted_mass=minimum,
                    influence_cap=cap,
                    return_diagnostics=True,
                )
                yield {
                    "split": split,
                    "history_id": _history_id(split, cell, round_index),
                    "cell": cell,
                    "round_index": round_index,
                    "components": components,
                    "history": tuple(value.clone() for value in history),
                    "enrollment_mean": enrollment_mean.clone(),
                    "feature": feature,
                    "feature_diagnostics": feature_diagnostics,
                    "source_rounds": source_rounds,
                    "feature_max_source_round": max(source_rounds),
                }
            observed = k4c._outer_observed(config, components, cell, round_index)
            vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
            standardized = k4._standardized_messages(
                config,
                vectors,
                anchor=components["anchor"],
                variances=components["variances"],
            )
            if round_index == int(config["temporal"]["enrollment_rounds"]):
                enrollment_mean = torch.stack(history + [standardized]).mean(dim=0)
            aware_radii = k4._radii(
                config,
                components["variances"],
                k2_calibration,
                regime_name=str(cell["regime"]["name"]),
                blind=False,
            )
            if round_index < int(config["temporal"]["first_temporal_gate_round"]):
                _, diagnostics = k4._current_reference(
                    k4.AWARE,
                    vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    blind_radii=aware_radii,
                    config=config,
                )
            else:
                if enrollment_mean is None:
                    raise RuntimeError("Enrollment baseline missing")
                _, diagnostics = k4._k4_reference(
                    vectors,
                    anchor=components["anchor"],
                    aware_radii=aware_radii,
                    history=history,
                    enrollment_mean=enrollment_mean,
                    thresholds=temporal_calibration,
                    config=config,
                )
            residuals[round_index] = k4b._clipped_residuals(
                vectors, components["anchor"], cap
            )
            gates[round_index] = torch.tensor(
                diagnostics["gates_by_client"],
                dtype=oracle._RUNTIME_DTYPE,
                device=oracle._RUNTIME_DEVICE,
            )
            history.append(standardized)


def _child_values(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    stream: str,
    child: int,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    cell = context["cell"]
    observed, clean, child_seed = k4c._current_child(
        config,
        context["components"],
        cell,
        round_index=int(context["round_index"]),
        stream=stream,
        child=child,
    )
    vectors = clip_l2(observed, float(config["aggregation"]["server_clip_norm"]))
    n = int(config["cohort"]["num_clients"])
    b = int(config["cohort"]["num_byzantine"])
    latent_byzantine = torch.zeros(n, dtype=torch.bool, device=oracle._RUNTIME_DEVICE)
    latent_byzantine[n - b :] = True
    values = k4c._current_components(
        config,
        k2_calibration,
        temporal_calibration,
        context["components"],
        latent_byzantine,
        regime_name=str(cell["regime"]["name"]),
        vectors=vectors,
        clean=clean,
        history=context["history"],
        enrollment_mean=context["enrollment_mean"],
    )
    return values, vectors, child_seed


def _privileged_target(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    stream: str,
    children: int,
    split_diagnostic: bool,
) -> dict[str, Any]:
    targets: list[torch.Tensor] = []
    direct_sums: list[torch.Tensor] = []
    seeds: list[int] = []
    history_gate: torch.Tensor | None = None
    for child in range(children):
        values, _, child_seed = _child_values(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream=stream,
            child=child,
        )
        if history_gate is None:
            history_gate = values["history_gates"].clone()
        elif not torch.equal(history_gate, values["history_gates"]):
            raise RuntimeError("Past gate changed across current-randomness children")
        targets.append(values["target_direction"])
        direct_sums.append(values["direct_sum"])
        seeds.append(child_seed)
    if history_gate is None:
        raise RuntimeError("No target-construction child generated")
    missing_mass = float(torch.sum(1.0 - history_gate).item())
    target_matrix = torch.stack(targets)
    direct_matrix = torch.stack(direct_sums)
    cap = float(config["references"]["total_client_influence_cap"])
    n = int(config["cohort"]["num_clients"])
    predictor, diagnostics = k4c.current_randomness_conditional_mse_semi_oracle(
        target_matrix,
        direct_matrix,
        missing_slot_mass=missing_mass,
        num_clients=n,
        influence_cap=cap,
        return_diagnostics=True,
    )
    split_distance = 0.0
    if split_diagnostic:
        half = children // 2
        left = k4c.current_randomness_conditional_mse_semi_oracle(
            target_matrix[:half],
            direct_matrix[:half],
            missing_slot_mass=missing_mass,
            num_clients=n,
            influence_cap=cap,
        )
        right = k4c.current_randomness_conditional_mse_semi_oracle(
            target_matrix[half:],
            direct_matrix[half:],
            missing_slot_mass=missing_mass,
            num_clients=n,
            influence_cap=cap,
        )
        split_distance = (missing_mass / float(n)) * float(
            torch.linalg.vector_norm(left - right).item()
        )
    return {
        "predictor": predictor,
        "missing_slot_mass": missing_mass,
        "history_gate": history_gate,
        "child_seeds": seeds,
        "split_aggregate_scale_distance": split_distance,
        "raw_predictor_norm": float(diagnostics["raw_predictor_norm"]),
        "predictor_norm": float(diagnostics["predictor_norm"]),
    }


def _fit_grid(
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    lambdas: Sequence[float],
) -> tuple[dict[float, torch.Tensor], dict[float, dict[str, Any]]]:
    coefficients: dict[float, torch.Tensor] = {}
    diagnostics: dict[float, dict[str, Any]] = {}
    for value in lambdas:
        theta, diag = fit_shared_scalar_ridge(
            features,
            targets,
            weights,
            ridge_lambda=float(value),
            feature_scales=scales,
            return_diagnostics=True,
        )
        coefficients[float(value)] = theta
        diag["coefficients"] = theta.detach().cpu().tolist()
        diagnostics[float(value)] = diag
    return coefficients, diagnostics


def _select_lambda(
    lambdas: Sequence[float],
    sums: Mapping[tuple[float, int], float],
    counts: Mapping[tuple[float, int], int],
    calibration_seeds: Sequence[int],
    *,
    tolerance: float,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for ridge_lambda in lambdas:
        seed_mse = [
            sums[(float(ridge_lambda), int(seed))]
            / float(counts[(float(ridge_lambda), int(seed))])
            for seed in calibration_seeds
        ]
        rows.append(
            {
                "ridge_lambda": float(ridge_lambda),
                "equal_seed_mean_projected_aggregate_mse": statistics.fmean(seed_mse),
                "calibration_seed_mse": seed_mse,
            }
        )
    minimum = min(float(row["equal_seed_mean_projected_aggregate_mse"]) for row in rows)
    eligible = [
        float(row["ridge_lambda"])
        for row in rows
        if float(row["equal_seed_mean_projected_aggregate_mse"]) <= minimum + tolerance
    ]
    return max(eligible), rows


def _seed_balanced_weights(
    raw_weights: Sequence[float], seed_ids: Sequence[int]
) -> list[float]:
    """Normalize positive history weights to sum to one within each seed."""

    if len(raw_weights) != len(seed_ids) or not raw_weights:
        raise ValueError("raw_weights and seed_ids must be non-empty and aligned")
    totals: dict[int, float] = defaultdict(float)
    for raw, seed in zip(raw_weights, seed_ids, strict=True):
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("seed-balanced raw weights must be finite and positive")
        totals[int(seed)] += value
    return [
        float(raw) / totals[int(seed)]
        for raw, seed in zip(raw_weights, seed_ids, strict=True)
    ]


def _weights_by_seed(weights: Sequence[float], seed_ids: Sequence[int]) -> dict[str, float]:
    totals: dict[int, float] = defaultdict(float)
    for value, seed in zip(weights, seed_ids, strict=True):
        totals[int(seed)] += float(value)
    return {str(seed): totals[seed] for seed in sorted(totals)}


def _fixed_reference(
    context: Mapping[str, Any],
    values: Mapping[str, Any],
    predictor: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    return k4c.fixed_denominator_imputed_reference_from_sum(
        anchor=context["components"]["anchor"],
        direct_sum=values["direct_sum"],
        predictor=predictor,
        missing_slot_mass=float(values["missing_slot_mass"]),
        num_clients=int(config["cohort"]["num_clients"]),
        influence_cap=float(config["references"]["total_client_influence_cap"]),
    )


def _load_calibrations(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return k4c._load_calibrations(config)


def fit_freeze(
    config_path: Path,
    lock_path: Path,
    output: Path,
    *,
    published_lock_sha256: str,
) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("K5 fit accepts only the preregistered output directory")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    lock = _verify_lock(lock_path, config_path)
    publication = _attest(lock["sha256"], published_lock_sha256, name="lock")
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError("K5 fit refuses CPU, fallback, or non-float32 runtime")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing K5 directory: {output}")
    output.mkdir(parents=True)
    k2_calibration, temporal_calibration, provenance = _load_calibrations(config)
    manifest = {
        "campaign_id": config["campaign_id"],
        "status": "fit_running_evaluation_forbidden",
        "device": "mps",
        "dtype": str(oracle._RUNTIME_DTYPE),
        "holdout_opened": False,
        "evaluation_trajectory_count_generated": 0,
        "preregistration_lock": lock,
        "lock_publication_attestation": publication,
        "config_sha256": _sha256(config_path),
        "calibration_provenance": provenance,
    }
    _write_json(output / "manifest.json", manifest)
    train_seeds = [int(v) for v in config["randomness"]["train_outer_seeds"]]
    cal_seeds = [int(v) for v in config["randomness"]["calibration_outer_seeds"]]
    lambdas = [float(v) for v in config["ridge"]["lambda_grid"]]
    train_features: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    train_weights: list[float] = []
    train_seed_ids: list[int] = []
    fit_rows: list[dict[str, Any]] = []
    rng_registry: dict[str, list[int]] = {
        "train_target": [],
        "calibration_target": [],
        "calibration_evaluation": [],
        "evaluation_target": [],
        "evaluation": [],
        "holdout": [],
    }
    for context in _snapshot_contexts(
        config, k2_calibration, temporal_calibration, split="train", seeds=train_seeds
    ):
        target = _privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="train_target",
            children=int(config["nested_monte_carlo"]["train_target_construction_children"]),
            split_diagnostic=False,
        )
        missing = float(target["missing_slot_mass"])
        if missing <= 0.0:
            raise RuntimeError("K5 training requires positive missing-slot mass")
        train_features.append(context["feature"])
        train_targets.append(target["predictor"])
        train_weights.append((missing / float(config["cohort"]["num_clients"])) ** 2)
        train_seed_ids.append(int(context["cell"]["seed"]))
        rng_registry["train_target"].extend(target["child_seeds"])
        fit_rows.append(
            {
                "split": "train",
                "history_id": context["history_id"],
                "seed": int(context["cell"]["seed"]),
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "missing_slot_mass": missing,
                "feature_hash": _tensor_hash(context["feature"]),
                "privileged_target_hash": _tensor_hash(target["predictor"]),
                "privileged_target_norm": float(target["predictor_norm"]),
                "inference_payload_forbidden_current_fields": forbidden_current_field_count(
                    {"past_features": True, "feature_scales": True, "coefficients": True}
                ),
            }
        )
    train_x = torch.stack(train_features)
    train_y = torch.stack(train_targets)
    train_balanced_weights = _seed_balanced_weights(train_weights, train_seed_ids)
    train_w = torch.tensor(
        train_balanced_weights,
        device=oracle._RUNTIME_DEVICE,
        dtype=torch.float32,
    )
    rms_floor = float(config["features"]["rms_floor"])
    train_scales, train_design = training_feature_scales(
        train_x, rms_floor=rms_floor, return_diagnostics=True
    )
    grid_theta, grid_fit = _fit_grid(train_x, train_y, train_w, train_scales, lambdas)
    grid_1d_theta, grid_1d_fit = _fit_grid(
        train_x[:, :1], train_y, train_w, train_scales[:1], lambdas
    )
    cal_features: list[torch.Tensor] = []
    cal_targets: list[torch.Tensor] = []
    cal_weights: list[float] = []
    cal_seed_ids: list[int] = []
    cal_sum: dict[tuple[float, int], float] = defaultdict(float)
    cal_count: dict[tuple[float, int], int] = defaultdict(int)
    cal_1d_sum: dict[tuple[float, int], float] = defaultdict(float)
    cal_1d_count: dict[tuple[float, int], int] = defaultdict(int)
    for context in _snapshot_contexts(
        config, k2_calibration, temporal_calibration, split="calibration", seeds=cal_seeds
    ):
        target = _privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="calibration_target",
            children=int(config["nested_monte_carlo"]["calibration_target_construction_children"]),
            split_diagnostic=False,
        )
        missing = float(target["missing_slot_mass"])
        if missing <= 0.0:
            raise RuntimeError("K5 calibration requires positive missing-slot mass")
        cal_features.append(context["feature"])
        cal_targets.append(target["predictor"])
        cal_weights.append((missing / float(config["cohort"]["num_clients"])) ** 2)
        cal_seed_ids.append(int(context["cell"]["seed"]))
        rng_registry["calibration_target"].extend(target["child_seeds"])
        predictors = {
            value: transcript_past_predictor(
                context["feature"],
                grid_theta[value],
                influence_cap=float(config["references"]["total_client_influence_cap"]),
                feature_scales=train_scales,
            )
            for value in lambdas
        }
        predictors_1d = {
            value: transcript_past_predictor(
                context["feature"][:1],
                grid_1d_theta[value],
                influence_cap=float(config["references"]["total_client_influence_cap"]),
                feature_scales=train_scales[:1],
            )
            for value in lambdas
        }
        for child in range(int(config["nested_monte_carlo"]["calibration_evaluation_children"])):
            values, _, child_seed = _child_values(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                stream="calibration_evaluation",
                child=child,
            )
            if not math.isclose(
                float(values["missing_slot_mass"]), missing, abs_tol=1e-7
            ):
                raise RuntimeError("Calibration past missing-slot mass changed")
            rng_registry["calibration_evaluation"].append(child_seed)
            seed = int(context["cell"]["seed"])
            for value in lambdas:
                reference = _fixed_reference(context, values, predictors[value], config)
                error = float(torch.sum((reference - values["target"]).square()).item())
                cal_sum[(value, seed)] += error
                cal_count[(value, seed)] += 1
                reference_1d = _fixed_reference(context, values, predictors_1d[value], config)
                error_1d = float(torch.sum((reference_1d - values["target"]).square()).item())
                cal_1d_sum[(value, seed)] += error_1d
                cal_1d_count[(value, seed)] += 1
        fit_rows.append(
            {
                "split": "calibration",
                "history_id": context["history_id"],
                "seed": int(context["cell"]["seed"]),
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "missing_slot_mass": missing,
                "feature_hash": _tensor_hash(context["feature"]),
                "privileged_target_hash": _tensor_hash(target["predictor"]),
                "privileged_target_norm": float(target["predictor_norm"]),
                "inference_payload_forbidden_current_fields": 0,
            }
        )
    tolerance = float(config["ridge"]["selection_tie_absolute_tolerance"])
    selected, calibration_rows = _select_lambda(
        lambdas, cal_sum, cal_count, cal_seeds, tolerance=tolerance
    )
    selected_1d, calibration_1d_rows = _select_lambda(
        lambdas, cal_1d_sum, cal_1d_count, cal_seeds, tolerance=tolerance
    )
    combined_x = torch.cat((train_x, torch.stack(cal_features)), dim=0)
    combined_y = torch.cat((train_y, torch.stack(cal_targets)), dim=0)
    combined_balanced_weights = _seed_balanced_weights(
        train_weights + cal_weights, train_seed_ids + cal_seed_ids
    )
    combined_w = torch.tensor(
        combined_balanced_weights,
        device=oracle._RUNTIME_DEVICE,
        dtype=torch.float32,
    )
    combined_scales, combined_design = training_feature_scales(
        combined_x, rms_floor=rms_floor, return_diagnostics=True
    )
    final_theta, final_fit = fit_shared_scalar_ridge(
        combined_x,
        combined_y,
        combined_w,
        ridge_lambda=selected,
        feature_scales=combined_scales,
        return_diagnostics=True,
    )
    final_1d_theta, final_1d_fit = fit_shared_scalar_ridge(
        combined_x[:, :1],
        combined_y,
        combined_w,
        ridge_lambda=selected_1d,
        feature_scales=combined_scales[:1],
        return_diagnostics=True,
    )
    all_fit_rng = (
        rng_registry["train_target"]
        + rng_registry["calibration_target"]
        + rng_registry["calibration_evaluation"]
    )
    gates = config["gates"]
    fit_checks = {
        "device_mps": oracle._RUNTIME_DEVICE.type == "mps",
        "fit_dtype_float32": final_theta.dtype == torch.float32,
        "fit_device_equals_runtime": final_theta.device == oracle._RUNTIME_DEVICE,
        "train_histories_exact": len(train_features) == int(gates["train_histories_exact"]),
        "calibration_histories_exact": len(cal_features) == int(gates["calibration_histories_exact"]),
        "train_matrix_exact": _fit_history_matrix_is_exact(
            config, fit_rows, split="train", seeds=train_seeds
        ),
        "calibration_matrix_exact": _fit_history_matrix_is_exact(
            config, fit_rows, split="calibration", seeds=cal_seeds
        ),
        "feature_rank_train_exact": int(train_design["flattened_design_rank"]) == int(gates["flattened_feature_rank_exact"]),
        "feature_rank_refit_exact": int(combined_design["flattened_design_rank"]) == int(gates["flattened_feature_rank_exact"]),
        "feature_rank_computed_on_mps": torch.device(
            train_design["rank_diagnostics"]["compute_device"]
        ).type
        == "mps"
        and torch.device(
            combined_design["rank_diagnostics"]["compute_device"]
        ).type
        == "mps",
        "feature_scale_floor_inactive_train": int(train_design["floor_active_count"]) <= int(gates["feature_scale_floor_active_count_max"]),
        "feature_scale_floor_inactive_refit": int(combined_design["floor_active_count"]) <= int(gates["feature_scale_floor_active_count_max"]),
        "minimum_centered_rms_positive_train": float(train_design["minimum_centered_rms"]) > float(gates["minimum_centered_feature_rms_strictly_greater_than"]),
        "minimum_centered_rms_positive_refit": float(combined_design["minimum_centered_rms"]) > float(gates["minimum_centered_feature_rms_strictly_greater_than"]),
        "features_strictly_past": all(int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1 for row in fit_rows),
        "forbidden_current_inference_fields_zero": max(int(row["inference_payload_forbidden_current_fields"]) for row in fit_rows) <= int(gates["forbidden_current_inference_field_count_max"]),
        "train_target_seed_count_exact": len(rng_registry["train_target"]) == int(gates["train_target_child_seeds_exact"]),
        "calibration_target_seed_count_exact": len(rng_registry["calibration_target"]) == int(gates["calibration_target_child_seeds_exact"]),
        "calibration_evaluation_seed_count_exact": len(rng_registry["calibration_evaluation"]) == int(gates["calibration_evaluation_child_seeds_exact"]),
        "fit_rng_globally_unique": len(all_fit_rng) == len(set(all_fit_rng)),
        "evaluation_rng_not_generated": len(rng_registry["evaluation_target"]) == 0 and len(rng_registry["evaluation"]) == 0,
        "holdout_rng_not_generated": len(rng_registry["holdout"]) == 0,
        "selected_lambda_in_grid": selected in lambdas and selected_1d in lambdas,
        "regularized_condition_number": max(float(final_fit["condition_number_regularized_system"]), float(final_1d_fit["condition_number_regularized_system"])) <= float(gates["regularized_system_condition_number_max"]),
        "condition_number_norm_exact": final_fit["condition_number_norm"]
        == "infinity_exact_via_solve"
        and final_1d_fit["condition_number_norm"]
        == "infinity_exact_via_solve",
        "normal_equation_residual": max(float(final_fit["normal_equation_relative_residual"]), float(final_1d_fit["normal_equation_relative_residual"])) <= float(gates["normal_equation_relative_residual_max"]),
    }
    predictor = {
        "schema_version": 1,
        "campaign_id": config["campaign_id"],
        "candidate": K5,
        "feature_names": list(FEATURE_NAMES),
        "feature_normalization_formula": "sqrt(sum_h ||V_j,h||_2^2 / (H*d))",
        "feature_scales": combined_scales.detach().cpu().tolist(),
        "coefficients": final_theta.detach().cpu().tolist(),
        "selected_lambda": selected,
        "one_dimensional_control": {
            "feature_name": FEATURE_NAMES[0],
            "feature_scale": float(combined_scales[0].item()),
            "coefficient": float(final_1d_theta[0].item()),
            "selected_lambda": selected_1d,
        },
        "influence_cap": float(config["references"]["total_client_influence_cap"]),
        "public_cohort_size": int(config["cohort"]["num_clients"]),
        "fit_split": "train_plus_calibration_after_lambda_selection_on_calibration",
        "lambda_selection_coefficients_fit_on": "train_only",
        "observable_past_only_at_inference": True,
        "privileged_training_supervision": "k4c_ch_synthetic_targets",
        "config_sha256": _sha256(config_path),
        "lock_sha256": lock["sha256"],
        "fit_rng_registry_sha256": _canonical_hash(all_fit_rng),
        "evaluation_generated_before_freeze": False,
        "holdout_opened": False,
    }
    predictor_path = output / "frozen_predictor.json"
    _write_json(predictor_path, predictor)
    predictor_sha = _sha256(predictor_path)
    fit_decision = {
        "all_validity_checks_pass": all(fit_checks.values()),
        "checks": fit_checks,
        "decision": "predictor_frozen_evaluation_locked_pending_external_hash_publication" if all(fit_checks.values()) else "invalid_fit_evaluation_forbidden",
        "selected_lambda": selected,
        "selected_lambda_one_dimensional": selected_1d,
        "frozen_predictor_sha256": predictor_sha,
        "evaluation_trajectory_count_generated": 0,
        "holdout_opened": False,
    }
    _write_csv(output / "fit_history_rows.csv", fit_rows)
    _write_json(
        output / "lambda_calibration.json",
        {
            "primary": calibration_rows,
            "one_dimensional": calibration_1d_rows,
            "tie_absolute_tolerance": tolerance,
            "tie_break": "largest_lambda",
            "calibration_seed_ids": cal_seeds,
            "train_grid_fit_diagnostics": {str(key): value for key, value in grid_fit.items()},
            "train_grid_1d_fit_diagnostics": {str(key): value for key, value in grid_1d_fit.items()},
        },
    )
    _write_json(
        output / "feature_design_diagnostics.json",
        {
            "train": train_design,
            "train_plus_calibration": combined_design,
            "final_fit": final_fit,
            "final_1d_fit": final_1d_fit,
            "weight_audit": {
                "raw_train_sum_by_seed": _weights_by_seed(
                    train_weights, train_seed_ids
                ),
                "balanced_train_sum_by_seed": _weights_by_seed(
                    train_balanced_weights, train_seed_ids
                ),
                "raw_train_plus_calibration_sum_by_seed": _weights_by_seed(
                    train_weights + cal_weights, train_seed_ids + cal_seed_ids
                ),
                "balanced_train_plus_calibration_sum_by_seed": _weights_by_seed(
                    combined_balanced_weights, train_seed_ids + cal_seed_ids
                ),
                "normalization_rule": "each_outer_seed_sums_to_one",
            },
        },
    )
    _write_json(output / "fit_rng_registry.json", rng_registry)
    _write_json(output / "fit_decision.json", fit_decision)
    manifest.update(
        {
            "status": "fit_completed_evaluation_locked" if all(fit_checks.values()) else "fit_invalid_evaluation_forbidden",
            "frozen_predictor_path": str(predictor_path.relative_to(ROOT)),
            "frozen_predictor_sha256": predictor_sha,
            "fit_validity_pass": all(fit_checks.values()),
            "evaluation_trajectory_count_generated": 0,
        }
    )
    _write_json(output / "manifest.json", manifest)
    return fit_decision


def _validate_frozen_predictor(
    predictor: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    lock_sha256: str,
) -> None:
    expected_keys = {
        "schema_version",
        "campaign_id",
        "candidate",
        "feature_names",
        "feature_normalization_formula",
        "feature_scales",
        "coefficients",
        "selected_lambda",
        "one_dimensional_control",
        "influence_cap",
        "public_cohort_size",
        "fit_split",
        "lambda_selection_coefficients_fit_on",
        "observable_past_only_at_inference",
        "privileged_training_supervision",
        "config_sha256",
        "lock_sha256",
        "fit_rng_registry_sha256",
        "evaluation_generated_before_freeze",
        "holdout_opened",
    }
    if set(predictor) != expected_keys:
        raise RuntimeError("Frozen K5 predictor schema mismatch")
    if predictor["schema_version"] != 1 or predictor["campaign_id"] != config["campaign_id"]:
        raise RuntimeError("Frozen K5 predictor identity mismatch")
    if predictor["candidate"] != K5 or tuple(predictor["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Frozen K5 candidate or feature schema mismatch")
    if predictor["feature_normalization_formula"] != "sqrt(sum_h ||V_j,h||_2^2 / (H*d))":
        raise RuntimeError("Frozen K5 normalization formula mismatch")
    coefficients = [float(value) for value in predictor["coefficients"]]
    scales = [float(value) for value in predictor["feature_scales"]]
    if len(coefficients) != 6 or not all(math.isfinite(value) for value in coefficients):
        raise RuntimeError("Frozen K5 coefficients must be six finite scalars")
    if len(scales) != 6 or not all(math.isfinite(value) and value > 0.0 for value in scales):
        raise RuntimeError("Frozen K5 scales must be six positive finite scalars")
    grid = {float(value) for value in config["ridge"]["lambda_grid"]}
    if float(predictor["selected_lambda"]) not in grid:
        raise RuntimeError("Frozen K5 lambda is outside the preregistered grid")
    one = predictor["one_dimensional_control"]
    if not isinstance(one, Mapping) or set(one) != {
        "feature_name",
        "feature_scale",
        "coefficient",
        "selected_lambda",
    }:
        raise RuntimeError("Frozen K5 one-dimensional control schema mismatch")
    if one["feature_name"] != FEATURE_NAMES[0]:
        raise RuntimeError("Frozen K5 one-dimensional feature changed")
    if not math.isfinite(float(one["feature_scale"])) or float(one["feature_scale"]) <= 0.0:
        raise RuntimeError("Frozen K5 one-dimensional scale is invalid")
    if not math.isfinite(float(one["coefficient"])) or float(one["selected_lambda"]) not in grid:
        raise RuntimeError("Frozen K5 one-dimensional coefficient/lambda is invalid")
    if not math.isclose(
        float(predictor["influence_cap"]),
        float(config["references"]["total_client_influence_cap"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("Frozen K5 influence cap changed")
    if int(predictor["public_cohort_size"]) != int(config["cohort"]["num_clients"]):
        raise RuntimeError("Frozen K5 public denominator changed")
    if predictor["fit_split"] != "train_plus_calibration_after_lambda_selection_on_calibration":
        raise RuntimeError("Frozen K5 refit split changed")
    if predictor["lambda_selection_coefficients_fit_on"] != "train_only":
        raise RuntimeError("Frozen K5 lambda-selection fit split changed")
    if predictor["observable_past_only_at_inference"] is not True:
        raise RuntimeError("Frozen K5 predictor is not declared transcript-past-only")
    if predictor["privileged_training_supervision"] != "k4c_ch_synthetic_targets":
        raise RuntimeError("Frozen K5 supervision provenance changed")
    if predictor["config_sha256"] != config_sha256 or predictor["lock_sha256"] != lock_sha256:
        raise RuntimeError("Frozen K5 config/lock provenance mismatch")
    if not isinstance(predictor["fit_rng_registry_sha256"], str) or len(predictor["fit_rng_registry_sha256"]) != 64:
        raise RuntimeError("Frozen K5 fit RNG registry hash is malformed")
    if predictor["evaluation_generated_before_freeze"] is not False or predictor["holdout_opened"] is not False:
        raise RuntimeError("Frozen K5 phase-boundary declaration is invalid")


def _load_frozen_predictor(
    output: Path,
    published_sha256: str,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    fit_decision = json.loads((output / "fit_decision.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "fit_completed_evaluation_locked" or not fit_decision.get("all_validity_checks_pass"):
        raise RuntimeError("K5 evaluation requires a valid completed fit phase")
    if int(manifest.get("evaluation_trajectory_count_generated", -1)) != 0:
        raise RuntimeError("Evaluation data were generated before the phase boundary")
    path = output / "frozen_predictor.json"
    actual = _sha256(path)
    if actual != str(manifest.get("frozen_predictor_sha256")) or actual != str(fit_decision.get("frozen_predictor_sha256")):
        raise RuntimeError("Frozen K5 predictor changed after fit")
    publication = _attest(actual, published_sha256, name="predictor")
    predictor = json.loads(path.read_text(encoding="utf-8"))
    _validate_frozen_predictor(
        predictor,
        config=config,
        config_sha256=config_sha256,
        lock_sha256=lock_sha256,
    )
    return predictor, publication


def _candidate_references(
    config: Mapping[str, Any],
    k2_calibration: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    context: Mapping[str, Any],
    values: Mapping[str, Any],
    vectors: torch.Tensor,
    *,
    k5_predictor: torch.Tensor,
    one_dimensional_predictor: torch.Tensor,
    privileged_predictor: torch.Tensor,
) -> dict[str, torch.Tensor]:
    feature = context["feature"]
    rolling = feature[0]
    pointwise = pointwise_optimal_full_imputation_predictor(
        values["clipped"],
        values["gates"],
        values["history_gates"],
        target_direction=values["target_direction"],
        influence_cap=float(config["references"]["total_client_influence_cap"]),
    )
    aware_radii = k4._radii(
        config,
        context["components"]["variances"],
        k2_calibration,
        regime_name=str(context["cell"]["regime"]["name"]),
        blind=False,
    )
    k2_reference, _ = k4._current_reference(
        k4.AWARE,
        vectors,
        anchor=context["components"]["anchor"],
        aware_radii=aware_radii,
        blind_radii=aware_radii,
        config=config,
    )
    return {
        K2: k2_reference,
        K4: values["k4_reference"],
        K4B: _fixed_reference(context, values, rolling, config),
        K5_1D: _fixed_reference(context, values, one_dimensional_predictor, config),
        K5: _fixed_reference(context, values, k5_predictor, config),
        K4C: _fixed_reference(context, values, privileged_predictor, config),
        POINTWISE: _fixed_reference(context, values, pointwise, config),
    }


def _replace_one_audit(
    config: Mapping[str, Any],
    temporal_calibration: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cap = float(config["references"]["total_client_influence_cap"])
    n = int(config["cohort"]["num_clients"])
    rows: list[dict[str, Any]] = []
    for context in contexts:
        original = context["vectors"]
        for client in range(n):
            neighbour = original.clone()
            direction = torch.zeros(original.shape[1], dtype=original.dtype, device=original.device)
            direction[client % original.shape[1]] = cap
            neighbour[client] = context["anchor"] - direction
            common = {
                "anchor": context["anchor"],
                "aware_radii": context["aware_radii"],
                "history": context["history"],
                "enrollment_mean": context["enrollment_mean"],
                "predictor": context["predictor"],
                "predictor_role": "frozen_k5_transcript_past_predictor",
                "imputation_mode": FULL_TEMPORAL_MISSING_SLOT,
                "deployable": False,
                "privacy_claimed": True,
                "temporal_calibration": temporal_calibration,
                "config": config,
            }
            left, left_diag = k4b._k4b_reference(original, **common)
            right, _ = k4b._k4b_reference(neighbour, **common)
            difference = float(torch.linalg.vector_norm(left - right).item())
            bound = 2.0 * cap / float(n)
            rows.append(
                {
                    "seed": int(context["seed"]),
                    "noise_regime": context["noise_regime"],
                    "noise_permutation": context["noise_permutation"],
                    "replaced_client": client,
                    "history_id": context["history_id"],
                    "same_past": True,
                    "same_frozen_predictor": True,
                    "observed_difference": difference,
                    "theoretical_bound": bound,
                    "ratio_to_bound": difference / bound,
                    "history_gate_has_suppression": min(left_diag["temporal_gates_by_client"]) < 1.0,
                    "violation": difference > bound + 1e-6,
                }
            )
    return rows


def _evaluation_matrix_is_exact(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
) -> bool:
    seeds = [int(value) for value in config["randomness"]["evaluation_outer_seeds"]]
    expected_histories = {
        _history_id("evaluation", cell, int(round_index))
        for cell in _cells(config, seeds)
        for round_index in config["temporal"]["assessment_rounds"]
    }
    observed_histories = [str(row["history_id"]) for row in history_rows]
    if len(observed_histories) != len(expected_histories) or set(observed_histories) != expected_histories:
        return False
    children = int(config["nested_monte_carlo"]["evaluation_children"])
    expected_rows = {
        (history_id, candidate, child)
        for history_id in expected_histories
        for candidate in CANDIDATES
        for child in range(children)
    }
    observed_rows = [
        (str(row["history_id"]), str(row["candidate"]), int(row["evaluation_child"]))
        for row in child_rows
    ]
    return len(observed_rows) == len(expected_rows) and set(observed_rows) == expected_rows


def _evaluation_noise_composition_is_exact(
    config: Mapping[str, Any], history_rows: Sequence[Mapping[str, Any]]
) -> bool:
    expected_per_cell = 16
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        for regime, permutation in _noise_cells(config):
            count = sum(
                int(row["seed"]) == int(seed)
                and str(row["noise_regime"]) == str(regime["name"])
                and str(row["noise_permutation"]) == str(permutation)
                for row in history_rows
            )
            if count != expected_per_cell:
                return False
    return True


def _summarize_evaluation(
    config: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
    replace_rows: Sequence[Mapping[str, Any]],
    validity_extra: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_history: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta = {str(row["history_id"]): row for row in history_rows}
    for row in child_rows:
        by_history[str(row["history_id"])][str(row["candidate"])].append(float(row["squared_reference_error"]))
    seed_rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["evaluation_outer_seeds"]:
        ids = [key for key, row in meta.items() if int(row["seed"]) == int(seed)]
        sums: dict[str, float] = defaultdict(float)
        regime_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        split_sq = 0.0
        for history_id in ids:
            means = {candidate: _mean(by_history[history_id][candidate]) for candidate in CANDIDATES}
            regime = str(meta[history_id]["noise_regime"])
            for candidate, value in means.items():
                sums[candidate] += value
                regime_sums[regime][candidate] += value
            split_sq += float(meta[history_id]["privileged_split_aggregate_scale_distance"]) ** 2
        k4_headroom = sums[K4] - sums[K4C]
        seed_rows.append(
            {
                "seed": int(seed),
                "histories": len(ids),
                "k4_integrated_mse": sums[K4],
                "k4b_integrated_mse": sums[K4B],
                "one_dimensional_integrated_mse": sums[K5_1D],
                "k5_integrated_mse": sums[K5],
                "k4c_integrated_mse": sums[K4C],
                "pointwise_integrated_mse": sums[POINTWISE],
                "gain_vs_k4b": (sums[K4B] - sums[K5]) / sums[K4B],
                "gain_vs_k4": (sums[K4] - sums[K5]) / sums[K4],
                "gain_vs_one_dimensional": (sums[K5_1D] - sums[K5]) / sums[K5_1D],
                "ch_capture_fraction": (sums[K4] - sums[K5]) / k4_headroom,
                "k4_minus_k4c_positive": k4_headroom > 0.0,
                "homogeneous_gain_vs_k4b": (regime_sums["homogeneous"][K4B] - regime_sums["homogeneous"][K5]) / regime_sums["homogeneous"][K4B],
                "heteroscedastic_gain_vs_k4b": (regime_sums["heteroscedastic"][K4B] - regime_sums["heteroscedastic"][K5]) / regime_sums["heteroscedastic"][K4B],
                "privileged_target_split_disagreement_mse_ratio": split_sq / sums[K4],
            }
        )
    tcrit = float(config["statistical_analysis"]["t_critical_df11"])
    cis = {
        key: _ci([float(row[key]) for row in seed_rows], tcrit)
        for key in (
            "gain_vs_k4b",
            "gain_vs_k4",
            "gain_vs_one_dimensional",
            "ch_capture_fraction",
            "homogeneous_gain_vs_k4b",
            "heteroscedastic_gain_vs_k4b",
        )
    }
    gates = config["gates"]
    validity = {
        **validity_extra,
        "evaluation_matrix_exact": _evaluation_matrix_is_exact(
            config, history_rows, child_rows
        ),
        "evaluation_noise_cell_composition_exact": (
            _evaluation_noise_composition_is_exact(config, history_rows)
        ),
        "evaluation_outer_seed_count_exact": len(seed_rows) == int(
            config["statistical_analysis"]["evaluation_independent_units"]
        )
        and {int(row["seed"]) for row in seed_rows}
        == {int(value) for value in config["randomness"]["evaluation_outer_seeds"]},
        "evaluation_histories_exact": len(history_rows) == int(gates["evaluation_histories_exact"]),
        "evaluation_child_rows_exact": len(child_rows) == int(gates["evaluation_histories_exact"]) * int(config["nested_monte_carlo"]["evaluation_children"]) * len(CANDIDATES),
        "replace_one_trials_exact": len(replace_rows) == int(gates["replace_one_exact_trials"]),
        "replace_one_no_violation": sum(bool(row["violation"]) for row in replace_rows) <= int(gates["replace_one_violation_max"]),
        "finite_metrics": all(math.isfinite(float(row["squared_reference_error"])) for row in child_rows),
        "strict_past_features": all(int(row["feature_max_source_round"]) <= int(row["assessment_round"]) - 1 for row in history_rows),
        "forbidden_current_inference_fields_zero": max(
            int(row["forbidden_current_inference_field_count"])
            for row in history_rows
        )
        <= int(gates["forbidden_current_inference_field_count_max"]),
        "fixed_predictor_across_children": all(bool(row["frozen_predictor_fixed"]) for row in child_rows),
        "fixed_denominator_n": all(int(row["fixed_denominator_n"]) == int(config["cohort"]["num_clients"]) for row in child_rows),
        "no_gate_sum_normalization": all(not bool(row["normalization_by_gate_sum"]) for row in child_rows),
        "contribution_cap": all(bool(row["contribution_cap_respected"]) for row in child_rows if row["candidate"] == K5),
        "k4_manual_formula": max(float(row["k4_manual_formula_error"]) for row in child_rows) <= float(gates["k4_manual_formula_abs_error_max"]),
        "k4b_reproduction": max(float(row["k4b_reproduction_error"]) for row in child_rows) <= float(gates["k4b_reproduction_abs_error_max"]),
        "fixed_denominator_formula": max(float(row["fixed_denominator_formula_error"]) for row in child_rows) <= float(gates["fixed_denominator_formula_abs_error_max"]),
        "positive_headroom_denominators": sum(bool(row["k4_minus_k4c_positive"]) for row in seed_rows) == int(gates["positive_k4_minus_k4c_denominator_seed_count_exact"]),
        "privileged_target_mc_stability": max(float(row["privileged_target_split_disagreement_mse_ratio"]) for row in seed_rows) <= float(gates["privileged_target_split_disagreement_mse_ratio_max"]),
    }
    scientific = {
        "gain_vs_k4b_mean": float(cis["gain_vs_k4b"]["mean"]) >= float(gates["primary_gain_vs_k4b_mean_min"]),
        "gain_vs_k4b_ci": float(cis["gain_vs_k4b"]["low"]) > float(gates["primary_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "gain_vs_k4_mean": float(cis["gain_vs_k4"]["mean"]) >= float(gates["primary_gain_vs_k4_mean_min"]),
        "gain_vs_k4_ci": float(cis["gain_vs_k4"]["low"]) > float(gates["primary_gain_vs_k4_ci95_low_strictly_greater_than"]),
        "gain_vs_one_dimensional_mean": float(cis["gain_vs_one_dimensional"]["mean"]) >= float(gates["primary_gain_vs_one_dimensional_mean_min"]),
        "gain_vs_one_dimensional_ci": float(cis["gain_vs_one_dimensional"]["low"]) > float(gates["primary_gain_vs_one_dimensional_ci95_low_strictly_greater_than"]),
        "capture_mean": float(cis["ch_capture_fraction"]["mean"]) >= float(gates["ch_capture_fraction_mean_min"]),
        "capture_ci": float(cis["ch_capture_fraction"]["low"]) > float(gates["ch_capture_fraction_ci95_low_strictly_greater_than"]),
        "homogeneous_gain_ci": float(cis["homogeneous_gain_vs_k4b"]["low"]) > float(gates["homogeneous_gain_vs_k4b_ci95_low_strictly_greater_than"]),
        "heteroscedastic_gain_ci": float(cis["heteroscedastic_gain_vs_k4b"]["low"]) > float(gates["heteroscedastic_gain_vs_k4b_ci95_low_strictly_greater_than"]),
    }
    validity_pass = all(validity.values())
    scientific_pass = all(scientific.values()) if validity_pass else False
    decision = (
        "authorize_end_to_end_development_screen"
        if validity_pass and scientific_pass
        else "stop_this_linear_transcript_predictor_instance"
        if validity_pass
        else "invalid_or_inconclusive_screen"
    )
    return seed_rows, {
        "validity_pass": validity_pass,
        "scientific_checks_pass": scientific_pass,
        "all_gates_pass": validity_pass and scientific_pass,
        "decision": decision,
        "validity_checks": validity,
        "scientific_checks": scientific,
        "confidence_intervals": cis,
        "holdout_opened": False,
        "pass_authorizes_holdout_or_promotion": False,
        "pass_authorizes_end_to_end_development_screen": True,
    }


def evaluate_frozen(
    config_path: Path,
    lock_path: Path,
    output: Path,
    *,
    published_lock_sha256: str,
    published_predictor_sha256: str,
) -> dict[str, Any]:
    if output.resolve() != DEFAULT_OUTPUT.resolve():
        raise RuntimeError("K5 evaluation accepts only the preregistered output directory")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    lock = _verify_lock(lock_path, config_path)
    _attest(lock["sha256"], published_lock_sha256, name="lock")
    predictor_artifact, predictor_publication = _load_frozen_predictor(
        output,
        published_predictor_sha256,
        config=config,
        config_sha256=_sha256(config_path),
        lock_sha256=lock["sha256"],
    )
    evaluation_dir = output / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing K5 evaluation")
    oracle._configure_runtime("mps")
    if oracle._RUNTIME_DEVICE.type != "mps" or oracle._RUNTIME_DTYPE != torch.float32:
        raise RuntimeError("K5 evaluation refuses CPU or non-float32 runtime")
    k2_calibration, temporal_calibration, _ = _load_calibrations(config)
    coefficients = torch.tensor(predictor_artifact["coefficients"], device=oracle._RUNTIME_DEVICE, dtype=torch.float32)
    scales = torch.tensor(predictor_artifact["feature_scales"], device=oracle._RUNTIME_DEVICE, dtype=torch.float32)
    one = predictor_artifact["one_dimensional_control"]
    coefficients_1d = torch.tensor([one["coefficient"]], device=oracle._RUNTIME_DEVICE, dtype=torch.float32)
    scales_1d = torch.tensor([one["feature_scale"]], device=oracle._RUNTIME_DEVICE, dtype=torch.float32)
    predictor_hash = _sha256(output / "frozen_predictor.json")
    evaluation_dir.mkdir(parents=True)
    _write_json(
        evaluation_dir / "manifest.json",
        {
            "status": "evaluation_running",
            "device": "mps",
            "frozen_predictor_sha256": predictor_hash,
            "predictor_publication_attestation": predictor_publication,
            "holdout_opened": False,
        },
    )
    history_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    audit_contexts: list[dict[str, Any]] = []
    target_seeds: list[int] = []
    evaluation_seeds: list[int] = []
    outer_seeds = [int(v) for v in config["randomness"]["evaluation_outer_seeds"]]
    for context in _snapshot_contexts(
        config, k2_calibration, temporal_calibration, split="evaluation", seeds=outer_seeds
    ):
        target = _privileged_target(
            config,
            k2_calibration,
            temporal_calibration,
            context,
            stream="evaluation_target",
            children=int(config["nested_monte_carlo"]["evaluation_target_construction_children"]),
            split_diagnostic=True,
        )
        target_seeds.extend(target["child_seeds"])
        k5_predictor, k5_diag = transcript_past_predictor(
            context["feature"],
            coefficients,
            influence_cap=float(config["references"]["total_client_influence_cap"]),
            feature_scales=scales,
            return_diagnostics=True,
        )
        one_predictor = transcript_past_predictor(
            context["feature"][:1],
            coefficients_1d,
            influence_cap=float(config["references"]["total_client_influence_cap"]),
            feature_scales=scales_1d,
        )
        history_rows.append(
            {
                "history_id": context["history_id"],
                "seed": int(context["cell"]["seed"]),
                "noise_regime": str(context["cell"]["regime"]["name"]),
                "noise_permutation": str(context["cell"]["permutation"]),
                "outlier_geometry": str(context["cell"]["geometry"]),
                "honest_dynamics": str(context["cell"]["dynamics"]),
                "threat": str(context["cell"]["threat"]),
                "assessment_round": int(context["round_index"]),
                "feature_max_source_round": int(context["feature_max_source_round"]),
                "feature_hash": _tensor_hash(context["feature"]),
                "k5_predictor_hash": _tensor_hash(k5_predictor),
                "k5_predictor_norm": float(k5_diag["predictor_norm"]),
                "k5_projection_active": bool(k5_diag["projection_active"]),
                "one_dimensional_predictor_norm": float(torch.linalg.vector_norm(one_predictor).item()),
                "missing_slot_mass": float(target["missing_slot_mass"]),
                "privileged_predictor_norm": float(target["predictor_norm"]),
                "privileged_target_split_aggregate_scale_distance": float(target["split_aggregate_scale_distance"]),
                "frozen_predictor_sha256": predictor_hash,
                "forbidden_current_inference_field_count": 0,
            }
        )
        first_audit_added = False
        for child in range(int(config["nested_monte_carlo"]["evaluation_children"])):
            values, vectors, child_seed = _child_values(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                stream="evaluation",
                child=child,
            )
            if not math.isclose(
                float(values["missing_slot_mass"]),
                float(target["missing_slot_mass"]),
                abs_tol=1e-7,
            ):
                raise RuntimeError("Evaluation past missing-slot mass changed")
            evaluation_seeds.append(child_seed)
            references = _candidate_references(
                config,
                k2_calibration,
                temporal_calibration,
                context,
                values,
                vectors,
                k5_predictor=k5_predictor,
                one_dimensional_predictor=one_predictor,
                privileged_predictor=target["predictor"],
            )
            n = int(config["cohort"]["num_clients"])
            manual_k4 = context["components"]["anchor"] + values["direct_sum"] / float(n)
            k4_manual_error = float(
                torch.linalg.vector_norm(references[K4] - manual_k4).item()
            )
            aware_radii_for_reproduction = k4._radii(
                config,
                context["components"]["variances"],
                k2_calibration,
                regime_name=str(context["cell"]["regime"]["name"]),
                blind=False,
            )
            reproduced_k4b, _ = k4b._k4b_reference(
                vectors,
                anchor=context["components"]["anchor"],
                aware_radii=aware_radii_for_reproduction,
                history=context["history"],
                enrollment_mean=context["enrollment_mean"],
                predictor=context["feature"][0],
                predictor_role="frozen_k4b_rolling_past_imputation",
                imputation_mode=FULL_TEMPORAL_MISSING_SLOT,
                deployable=False,
                privacy_claimed=True,
                temporal_calibration=temporal_calibration,
                config=config,
            )
            k4b_reproduction_error = float(
                torch.linalg.vector_norm(references[K4B] - reproduced_k4b).item()
            )
            manual_k5 = context["components"]["anchor"] + (
                values["direct_sum"]
                + float(values["missing_slot_mass"]) * k5_predictor
            ) / float(n)
            fixed_denominator_error = float(
                torch.linalg.vector_norm(references[K5] - manual_k5).item()
            )
            contributions = values["gates"][:, None] * values["clipped"] + (
                1.0 - values["history_gates"]
            )[:, None] * k5_predictor[None, :]
            cap = float(config["references"]["total_client_influence_cap"])
            tolerance = 64.0 * torch.finfo(contributions.dtype).eps
            cap_ok = bool(
                (torch.linalg.vector_norm(contributions, dim=1) <= cap + tolerance).all()
            )
            max_contribution_norm = float(
                torch.max(torch.linalg.vector_norm(contributions, dim=1)).item()
            )
            if not first_audit_added and (
                str(context["cell"]["geometry"]) == "aligned"
                and str(context["cell"]["dynamics"]) == "stationary"
                and str(context["cell"]["threat"]) == "bitflip_x10"
                and int(context["round_index"]) == 17
            ):
                aware_radii = k4._radii(
                    config,
                    context["components"]["variances"],
                    k2_calibration,
                    regime_name=str(context["cell"]["regime"]["name"]),
                    blind=False,
                )
                audit_contexts.append(
                    {
                        "history_id": context["history_id"],
                        "seed": int(context["cell"]["seed"]),
                        "noise_regime": str(context["cell"]["regime"]["name"]),
                        "noise_permutation": str(context["cell"]["permutation"]),
                        "vectors": vectors.clone(),
                        "anchor": context["components"]["anchor"],
                        "aware_radii": aware_radii,
                        "history": context["history"],
                        "enrollment_mean": context["enrollment_mean"],
                        "predictor": k5_predictor.clone(),
                    }
                )
                first_audit_added = True
            for candidate, reference in references.items():
                child_rows.append(
                    {
                        "history_id": context["history_id"],
                        "seed": int(context["cell"]["seed"]),
                        "noise_regime": str(context["cell"]["regime"]["name"]),
                        "noise_permutation": str(context["cell"]["permutation"]),
                        "outlier_geometry": str(context["cell"]["geometry"]),
                        "honest_dynamics": str(context["cell"]["dynamics"]),
                        "threat": str(context["cell"]["threat"]),
                        "assessment_round": int(context["round_index"]),
                        "evaluation_child": child,
                        "evaluation_child_seed": child_seed,
                        "candidate": candidate,
                        "squared_reference_error": float(torch.sum((reference - values["target"]).square()).item()),
                        "reference_error_l2_descriptive": float(torch.linalg.vector_norm(reference - values["target"]).item()),
                        "frozen_predictor_sha256": predictor_hash,
                        "frozen_predictor_fixed": True,
                        "predictor_observable_past_only": candidate in {K4B, K5_1D, K5},
                        "fixed_denominator_n": int(config["cohort"]["num_clients"]),
                        "normalization_by_gate_sum": False,
                        "contribution_cap_respected": cap_ok if candidate == K5 else True,
                        "max_slot_contribution_norm": max_contribution_norm
                        if candidate == K5
                        else 0.0,
                        "k4_manual_formula_error": k4_manual_error,
                        "k4b_reproduction_error": k4b_reproduction_error,
                        "fixed_denominator_formula_error": fixed_denominator_error,
                    }
                )
    replace_rows = _replace_one_audit(config, temporal_calibration, audit_contexts)
    fit_registry = json.loads((output / "fit_rng_registry.json").read_text(encoding="utf-8"))
    fit_seeds = [int(value) for key in ("train_target", "calibration_target", "calibration_evaluation") for value in fit_registry[key]]
    all_generated = fit_seeds + target_seeds + evaluation_seeds
    validity_extra = {
        "device_mps": oracle._RUNTIME_DEVICE.type == "mps",
        "evaluation_target_seed_count_exact": len(target_seeds) == int(config["gates"]["evaluation_target_child_seeds_exact"]),
        "evaluation_seed_count_exact": len(evaluation_seeds) == int(config["gates"]["evaluation_child_seeds_exact"]),
        "global_child_seed_unique": len(all_generated) == len(set(all_generated)),
        "evaluation_target_and_evaluation_disjoint": not (set(target_seeds) & set(evaluation_seeds)),
        "evaluation_disjoint_from_fit": not (set(fit_seeds) & (set(target_seeds) | set(evaluation_seeds))),
        "frozen_predictor_hash_unchanged": _sha256(output / "frozen_predictor.json") == predictor_hash,
        "holdout_not_opened": True,
    }
    seed_rows, decision = _summarize_evaluation(
        config, history_rows, child_rows, replace_rows, validity_extra
    )
    _write_csv(evaluation_dir / "history_rows.csv", history_rows)
    _write_csv(evaluation_dir / "evaluation_child_rows.csv", child_rows)
    _write_csv(evaluation_dir / "seed_summary.csv", seed_rows)
    _write_csv(evaluation_dir / "replace_one_audit.csv", replace_rows)
    _write_json(
        evaluation_dir / "evaluation_rng_registry.json",
        {"evaluation_target": target_seeds, "evaluation": evaluation_seeds, "holdout": []},
    )
    _write_json(evaluation_dir / "decision.json", decision)
    _write_json(
        evaluation_dir / "manifest.json",
        {
            "status": "completed_development",
            "device": "mps",
            "frozen_predictor_sha256": predictor_hash,
            "predictor_publication_attestation": predictor_publication,
            "histories": len(history_rows),
            "child_rows": len(child_rows),
            "decision": decision["decision"],
            "all_gates_pass": decision["all_gates_pass"],
            "holdout_opened": False,
        },
    )
    if _sha256(output / "frozen_predictor.json") != predictor_hash:
        raise RuntimeError("Frozen K5 predictor changed during evaluation")
    root_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    root_manifest.update(
        {
            "status": "completed_development",
            "evaluation_trajectory_count_generated": len(history_rows),
            "evaluation_decision": decision["decision"],
            "all_gates_pass": decision["all_gates_pass"],
            "holdout_opened": False,
        }
    )
    _write_json(output / "manifest.json", root_manifest)
    return decision


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "fit-freeze", "evaluate-frozen"), default="validate")
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--published-lock-sha256")
    parser.add_argument("--published-predictor-sha256")
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if args.phase == "validate":
        print(
            json.dumps(
                {
                    "campaign_id": config["campaign_id"],
                    "phase": "validate",
                    "device_required": "mps",
                    "evaluation_generated": False,
                    "holdout_opened": False,
                },
                indent=2,
            )
        )
        return 0
    if not args.published_lock_sha256:
        parser.error("fit/evaluation requires --published-lock-sha256")
    if args.phase == "fit-freeze":
        decision = fit_freeze(
            config_path,
            lock_path,
            output,
            published_lock_sha256=args.published_lock_sha256,
        )
        print(json.dumps(_json_safe(decision), indent=2, sort_keys=True, allow_nan=False))
        return 0 if decision["all_validity_checks_pass"] else 2
    if not args.published_predictor_sha256:
        parser.error("evaluate-frozen requires --published-predictor-sha256")
    decision = evaluate_frozen(
        config_path,
        lock_path,
        output,
        published_lock_sha256=args.published_lock_sha256,
        published_predictor_sha256=args.published_predictor_sha256,
    )
    print(json.dumps(_json_safe(decision), indent=2, sort_keys=True, allow_nan=False))
    return 0 if decision["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
