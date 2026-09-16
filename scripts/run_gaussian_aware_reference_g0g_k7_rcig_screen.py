#!/usr/bin/env python3
r"""Locked synthetic screen for robust covariance-innovation fusion (K7).

K7 is evaluated only after K6/K6b failed against the natural Identity-Y/K4b
control.  It can change direction: a newer strictly-past view is accepted
unchanged when its innovation is statistically ordinary, and is shrunk toward
an older strictly-past view when the innovation is large relative to the
registered DP-plus-drift covariance.

The screen is fail-closed on MPS/float32 with CPU fallback disabled.  It is a
mechanistic development screen, not end-to-end federated-learning evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k7_rcig import (  # noqa: E402
    euclidean_innovation_fusion,
    robust_covariance_innovation_fusion,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k6_tp_eiv_microbench as m0,
)

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k7_rcig_screen_mps_v1"
DEFAULT_OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
ALGORITHM_PATH = ROOT / "algorithms/gaussian_aware_reference_k7_rcig.py"
PROTOCOL_PATH = ROOT / "output/analysis/Gaussian_Aware_G0g_K7_RCIG_Protocol_PreRun.md"
TEST_PATH = ROOT / "tests/test_gaussian_aware_reference_g0g_k7_rcig.py"

CALIBRATION_SEEDS = tuple(203011010001 + 37 * index for index in range(48))
NULL_VALIDATION_SEEDS_BY_REGIME = {
    "homogeneous": tuple(203011020003 + 37 * index for index in range(72)),
    "heteroscedastic": tuple(203011030007 + 37 * index for index in range(72)),
}
EVALUATION_SEEDS = tuple(203011040009 + 37 * index for index in range(24))

REGIMES = ("homogeneous", "heteroscedastic")
THREATS = ("none", "bitflip_x10", "model_replacement")
CANDIDATES = (
    "identity_y",
    "older_x",
    "midpoint_xy",
    "rcig_full",
    "rcig_isotropic",
    "euclidean_gate",
)

PROTOCOL: dict[str, Any] = {
    "campaign_id": CAMPAIGN_ID,
    "scope": "locked_synthetic_direction_changing_mechanistic_screen",
    "research_question": (
        "Can covariance-standardised innovation gating beat the recent-view "
        "Identity-Y control under attacks without harming the no-attack case?"
    ),
    "claims_excluded": [
        "end_to_end_accuracy_or_fairness",
        "universal_byzantine_robustness",
        "publication_confirmation",
        "exact_post_clipping_gaussian_covariance",
    ],
    "cohort": {"num_clients": 12, "num_byzantine": 2, "dimension": 8},
    "timeline": {
        "gate_source_rounds": list(range(1, 16)),
        "older_view_rounds": [16, 17, 18, 19],
        "newer_view_rounds": [20, 21, 22, 23],
        "deployment_round": 24,
        "attack_onset_round": 20,
        "assumption": "older_view_unattacked_at_attack_onset",
    },
    "noise_regimes": {
        "homogeneous": [0.03],
        "heteroscedastic": [0.012, 0.022, 0.040, 0.070],
    },
    "geometry": {
        "server_clip_norm": 0.75,
        "residual_influence_cap": 0.13,
        "minimum_accepted_mass": 6.0,
    },
    "innovation": {
        "process_variance_per_coordinate": 3.0e-5,
        "ridge": 1.0e-7,
        "threshold_rule": "q0.975_no_attack_calibration_separate_by_regime_and_metric",
        "threshold_quantile": 0.975,
        "full_covariance_primary": True,
        "isotropic_same_trace_control": True,
        "raw_euclidean_control": True,
    },
    "covariance_policy": {
        "source": "public_nominal_client_DP_noise_metadata",
        "retained_for_attacked_identities": True,
        "byzantine_mask_used_to_modify_covariance": False,
        "cross_covariance": "zero_for_disjoint_noise_windows",
        "approximation": "delta_method_after_clipping_calibrated_by_independent_gate_thresholds",
    },
    "counterbalancing": {
        "identity_rotation_period": 12,
        "attacked_clients_per_seed": 2,
        "each_client_attacked_four_times_across_24_evaluation_seeds": True,
        "heteroscedastic_tier_rule": "tier=(client+5*slot) mod 4",
        "attacker_tier_slots_balanced": True,
    },
    "randomness": {
        "calibration_outer_seeds": 48,
        "null_validation_outer_seeds_per_regime": 72,
        "evaluation_outer_seeds": 24,
        "all_registries_disjoint": True,
        "candidates_and_threat_counterfactuals_paired": True,
        "statistical_unit": "outer_seed",
    },
    "validity_gates": {
        "null_false_activation_CP975_upper_per_regime_max": 0.10,
        "all_predictor_norms_bounded": True,
        "no_oracle_covariance": True,
        "mps_float32_no_fallback": True,
    },
    "primary_science_gates": {
        "rcig_full_attacked_gain_vs_identity_y_mean_min_each_regime": 0.10,
        "rcig_full_attacked_gain_vs_identity_y_one_sided_ci_low_min_each_regime": 0.0,
        "rcig_full_no_attack_loss_vs_identity_y_one_sided_ci_high_max_each_regime": 0.02,
        "rcig_full_attack_gate_activation_mean_min_each_regime": 0.50,
    },
    "gaussian_specificity_gates": {
        "heteroscedastic_gain_vs_isotropic_one_sided_ci_low_min": 0.0,
        "heteroscedastic_gain_vs_euclidean_one_sided_ci_low_min": 0.0,
    },
    "decision_policy": {
        "invalid_if_any_validity_gate_fails": True,
        "stop_if_primary_science_gate_fails": True,
        "robust_gate_only_if_primary_passes_but_gaussian_specificity_fails": True,
        "advance_gaussian_aware_only_if_all_gates_pass": True,
    },
    "execution": {
        "required_device": "mps",
        "dtype": "float32",
        "pytorch_mps_fallback_required": "0",
        "overwrite": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_seeds() -> tuple[int, ...]:
    return (
        CALIBRATION_SEEDS
        + NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
        + EVALUATION_SEEDS
    )


def _validate_protocol() -> dict[str, Any]:
    timeline = PROTOCOL["timeline"]
    seeds = _all_seeds()
    checks = {
        "all_seeds_unique": len(seeds) == len(set(seeds)),
        "strictly_past_disjoint_views": (
            not set(timeline["older_view_rounds"]) & set(timeline["newer_view_rounds"])
            and max(timeline["newer_view_rounds"]) < int(timeline["deployment_round"])
        ),
        "attack_starts_at_newer_view": int(timeline["attack_onset_round"])
        == min(timeline["newer_view_rounds"]),
        "both_noise_regimes_registered": tuple(PROTOCOL["noise_regimes"]) == REGIMES,
        "identity_y_is_hard_control": CANDIDATES[0] == "identity_y",
        "candidate_changes_direction": "rcig_full" in CANDIDATES,
        "no_oracle_covariance": (
            bool(PROTOCOL["covariance_policy"]["retained_for_attacked_identities"])
            and not bool(
                PROTOCOL["covariance_policy"][
                    "byzantine_mask_used_to_modify_covariance"
                ]
            )
        ),
        "mps_fail_closed": PROTOCOL["execution"]
        == {
            "required_device": "mps",
            "dtype": "float32",
            "pytorch_mps_fallback_required": "0",
            "overwrite": False,
        },
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"Invalid K7 protocol: {failed}")
    return {
        "campaign_id": CAMPAIGN_ID,
        "checks": checks,
        "protocol_sha256": _canonical_hash(PROTOCOL),
        "expected_calibration_rows": 48 * 2,
        "expected_null_validation_rows": 72 * 2,
        "expected_evaluation_rows": 24 * 2 * 3,
    }


def _require_mps() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must equal 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    device = torch.device("mps")
    probe = torch.tensor([1.0, 2.0], device=device, dtype=torch.float32)
    if probe.device.type != "mps" or probe.dtype != torch.float32:
        raise RuntimeError("MPS float32 probe failed")
    return device


def _phase_slot(seed: int, phase: str, regime: str) -> int:
    if phase == "calibration":
        return CALIBRATION_SEEDS.index(seed) % 12
    if phase == "null_validation":
        return NULL_VALIDATION_SEEDS_BY_REGIME[regime].index(seed) % 12
    if phase == "evaluation":
        return EVALUATION_SEEDS.index(seed) % 12
    raise ValueError(f"Unknown phase: {phase}")


def _base_context(
    seed: int,
    regime: str,
    threat: str,
    *,
    phase: str,
    device: torch.device,
) -> dict[str, Any]:
    clean, base_honest = m0._clean_process(seed, device=device, null_signal=False)
    slot = _phase_slot(seed, phase, regime)
    clean = torch.roll(clean, shifts=slot, dims=1)
    honest = torch.roll(base_honest, shifts=slot, dims=0)
    n = int(PROTOCOL["cohort"]["num_clients"])
    dimension = int(PROTOCOL["cohort"]["dimension"])
    tiers = PROTOCOL["noise_regimes"][regime]
    if len(tiers) == 1:
        noise_std = torch.full(
            (n,), float(tiers[0]), device=device, dtype=torch.float32
        )
    else:
        noise_std = torch.tensor(
            [tiers[(client + 5 * slot) % len(tiers)] for client in range(n)],
            device=device,
            dtype=torch.float32,
        )
    noise = torch.randn(
        clean.shape,
        generator=m0._generator(device, CAMPAIGN_ID, "dp-noise", seed, regime),
        device=device,
        dtype=torch.float32,
    )
    uploads = clean + noise * noise_std[None, :, None]
    variances = noise_std.square()[None, :, None].expand_as(uploads).clone()
    active = torch.zeros_like(honest)
    if threat != "none":
        active = ~honest
        onset = int(PROTOCOL["timeline"]["attack_onset_round"]) - 1
        if threat == "bitflip_x10":
            uploads[onset:, active] = -10.0 * clean[onset:, active]
        elif threat == "model_replacement":
            direction = m0._unit(
                torch.randn(
                    dimension,
                    generator=m0._generator(device, CAMPAIGN_ID, "replacement", seed),
                    device=device,
                    dtype=torch.float32,
                )
            )
            uploads[onset:, active] = (
                10.0 * float(PROTOCOL["geometry"]["server_clip_norm"]) * direction
            )
        else:
            raise ValueError(f"Unknown threat: {threat}")
        # Deliberately retain nominal metadata: no simulator oracle is exposed.

    anchor = torch.zeros(dimension, device=device, dtype=torch.float32)
    gate_rounds = PROTOCOL["timeline"]["gate_source_rounds"]
    prehistory = uploads[min(gate_rounds) - 1 : max(gate_rounds)]
    clipped, _, _ = m0._clip_chain_with_jacobian(
        prehistory.reshape(-1, dimension), anchor
    )
    gate, gate_diagnostics = m0._predictable_gate(
        clipped.reshape(len(gate_rounds), n, dimension)
    )
    older_rounds = PROTOCOL["timeline"]["older_view_rounds"]
    newer_rounds = PROTOCOL["timeline"]["newer_view_rounds"]
    older, covariance_older, older_diagnostics = m0._view_and_covariance(
        uploads[min(older_rounds) - 1 : max(older_rounds)],
        variances[min(older_rounds) - 1 : max(older_rounds)],
        gate,
        anchor,
    )
    newer, covariance_newer, newer_diagnostics = m0._view_and_covariance(
        uploads[min(newer_rounds) - 1 : max(newer_rounds)],
        variances[min(newer_rounds) - 1 : max(newer_rounds)],
        gate,
        anchor,
    )
    target_round = int(PROTOCOL["timeline"]["deployment_round"]) - 1
    target = clip_l2(
        clean[target_round, honest].mean(dim=0),
        float(PROTOCOL["geometry"]["residual_influence_cap"]),
    )
    return {
        "seed": seed,
        "slot": slot,
        "phase": phase,
        "noise_regime": regime,
        "threat": threat,
        "older": older,
        "newer": newer,
        "covariance_older": covariance_older,
        "covariance_newer": covariance_newer,
        "target": target,
        "active": active,
        "gate": gate,
        "gate_hash": gate_diagnostics["gate_hash"],
        "older_diagnostics": older_diagnostics,
        "newer_diagnostics": newer_diagnostics,
    }


def _innovation_statistics(base: Mapping[str, Any]) -> dict[str, float]:
    common = {
        "process_variance": float(
            PROTOCOL["innovation"]["process_variance_per_coordinate"]
        ),
        "ridge": float(PROTOCOL["innovation"]["ridge"]),
        "innovation_threshold": 1.0e12,
        "influence_cap": float(PROTOCOL["geometry"]["residual_influence_cap"]),
        "return_diagnostics": True,
    }
    _, full = robust_covariance_innovation_fusion(
        base["older"],
        base["newer"],
        base["covariance_older"],
        base["covariance_newer"],
        covariance_mode="full",
        **common,
    )
    _, isotropic = robust_covariance_innovation_fusion(
        base["older"],
        base["newer"],
        base["covariance_older"],
        base["covariance_newer"],
        covariance_mode="isotropic",
        **common,
    )
    return {
        "full": float(full["standardized_innovation"]),
        "isotropic": float(isotropic["standardized_innovation"]),
        "euclidean": float(
            torch.linalg.vector_norm(base["newer"] - base["older"]).item()
        ),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(m0._quantile(values, probability))


def _calibrate(
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for seed in CALIBRATION_SEEDS:
        for regime in REGIMES:
            base = _base_context(
                seed, regime, "none", phase="calibration", device=device
            )
            stats = _innovation_statistics(base)
            rows.append(
                {
                    "seed": seed,
                    "noise_regime": regime,
                    "device": str(base["older"].device),
                    "dtype": str(base["older"].dtype),
                    **{f"{key}_innovation": value for key, value in stats.items()},
                    "target_used": False,
                }
            )
    probability = float(PROTOCOL["innovation"]["threshold_quantile"])
    thresholds = {
        regime: {
            key: _quantile(
                [
                    float(row[f"{key}_innovation"])
                    for row in rows
                    if row["noise_regime"] == regime
                ],
                probability,
            )
            for key in ("full", "isotropic", "euclidean")
        }
        for regime in REGIMES
    }
    return thresholds, rows


def _evaluate_base(
    base: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    regime = str(base["noise_regime"])
    cap = float(PROTOCOL["geometry"]["residual_influence_cap"])
    common = {
        "process_variance": float(
            PROTOCOL["innovation"]["process_variance_per_coordinate"]
        ),
        "ridge": float(PROTOCOL["innovation"]["ridge"]),
        "influence_cap": cap,
        "return_diagnostics": True,
    }
    full, full_diagnostics = robust_covariance_innovation_fusion(
        base["older"],
        base["newer"],
        base["covariance_older"],
        base["covariance_newer"],
        innovation_threshold=float(thresholds[regime]["full"]),
        covariance_mode="full",
        **common,
    )
    isotropic, isotropic_diagnostics = robust_covariance_innovation_fusion(
        base["older"],
        base["newer"],
        base["covariance_older"],
        base["covariance_newer"],
        innovation_threshold=float(thresholds[regime]["isotropic"]),
        covariance_mode="isotropic",
        **common,
    )
    euclidean, euclidean_diagnostics = euclidean_innovation_fusion(
        base["older"],
        base["newer"],
        innovation_threshold=float(thresholds[regime]["euclidean"]),
        influence_cap=cap,
        return_diagnostics=True,
    )
    candidates = {
        "identity_y": clip_l2(base["newer"], cap),
        "older_x": clip_l2(base["older"], cap),
        "midpoint_xy": clip_l2(0.5 * (base["older"] + base["newer"]), cap),
        "rcig_full": full,
        "rcig_isotropic": isotropic,
        "euclidean_gate": euclidean,
    }
    target = base["target"]
    row: dict[str, Any] = {
        "seed": int(base["seed"]),
        "slot": int(base["slot"]),
        "phase": str(base["phase"]),
        "noise_regime": regime,
        "threat": str(base["threat"]),
        "pairing_id": f"{base['seed']}|{regime}",
        "device": str(base["older"].device),
        "dtype": str(base["older"].dtype),
        "target_used_for_predictor": False,
        "byzantine_mask_used_for_covariance": False,
        "nominal_covariance_retained_for_attacked_identities": True,
        "full_standardized_innovation": full_diagnostics["standardized_innovation"],
        "isotropic_standardized_innovation": isotropic_diagnostics[
            "standardized_innovation"
        ],
        "euclidean_innovation": euclidean_diagnostics["innovation_norm"],
        "full_gate_active": full_diagnostics["gate_active"],
        "isotropic_gate_active": isotropic_diagnostics["gate_active"],
        "euclidean_gate_active": euclidean_diagnostics["gate_active"],
        "full_newer_trust": full_diagnostics["newer_view_trust"],
        "isotropic_newer_trust": isotropic_diagnostics["newer_view_trust"],
        "euclidean_newer_trust": euclidean_diagnostics["newer_view_trust"],
        "condition_number_upper_bound": full_diagnostics[
            "condition_number_upper_bound"
        ],
        "gate_hash": str(base["gate_hash"]),
        "attacker_indices": json.dumps(
            torch.nonzero(base["active"], as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        ),
        "minimum_clip_margin": min(
            float(
                base["older_diagnostics"]["minimum_piecewise_differentiability_margin"]
            ),
            float(
                base["newer_diagnostics"]["minimum_piecewise_differentiability_margin"]
            ),
        ),
    }
    for name, value in candidates.items():
        row[f"{name}_mse"] = float(torch.mean((value - target).square()).item())
        row[f"{name}_norm"] = float(torch.linalg.vector_norm(value).item())
    return row


def _null_rows(
    device: torch.device, thresholds: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, seeds in NULL_VALIDATION_SEEDS_BY_REGIME.items():
        for seed in seeds:
            base = _base_context(
                seed,
                regime,
                "none",
                phase="null_validation",
                device=device,
            )
            rows.append(_evaluate_base(base, thresholds))
    return rows


def _evaluation_rows(
    device: torch.device, thresholds: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    return [
        _evaluate_base(
            _base_context(
                seed,
                regime,
                threat,
                phase="evaluation",
                device=device,
            ),
            thresholds,
        )
        for seed in EVALUATION_SEEDS
        for regime in REGIMES
        for threat in THREATS
    ]


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes + 1)
    )


def _clopper_pearson_upper(
    successes: int, trials: int, *, one_sided_alpha: float
) -> float:
    if successes >= trials:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(successes, trials, midpoint) > one_sided_alpha:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _student_interval(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if len(numbers) != 24:
        raise RuntimeError("Expected 24 independent outer-seed values")
    mean = statistics.fmean(numbers)
    standard_error = statistics.stdev(numbers) / math.sqrt(len(numbers))
    return {
        "n": len(numbers),
        "mean": mean,
        "two_sided_low": mean - 2.0686576104190406 * standard_error,
        "two_sided_high": mean + 2.0686576104190406 * standard_error,
        "one_sided_low": mean - 1.7138715277470473 * standard_error,
        "one_sided_high": mean + 1.7138715277470473 * standard_error,
    }


def _contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    regime: str,
    candidate: str,
    baseline: str,
    threats: Sequence[str],
) -> dict[str, float]:
    values = []
    for seed in EVALUATION_SEEDS:
        cell = [
            row
            for row in rows
            if int(row["seed"]) == seed
            and row["noise_regime"] == regime
            and row["threat"] in threats
        ]
        if len(cell) != len(threats):
            raise RuntimeError("Incomplete paired seed contrast")
        baseline_total = sum(float(row[f"{baseline}_mse"]) for row in cell)
        candidate_total = sum(float(row[f"{candidate}_mse"]) for row in cell)
        values.append((baseline_total - candidate_total) / max(baseline_total, 1.0e-15))
    return _student_interval(values)


def _summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for regime in REGIMES:
        for threat in THREATS:
            cell = [
                row
                for row in rows
                if row["noise_regime"] == regime and row["threat"] == threat
            ]
            summary: dict[str, Any] = {
                "noise_regime": regime,
                "threat": threat,
                "seed_count": len(cell),
                "full_gate_activation_mean": statistics.fmean(
                    float(row["full_gate_active"]) for row in cell
                ),
                "full_newer_trust_mean": statistics.fmean(
                    float(row["full_newer_trust"]) for row in cell
                ),
            }
            for candidate in CANDIDATES:
                summary[f"{candidate}_mse_mean"] = statistics.fmean(
                    float(row[f"{candidate}_mse"]) for row in cell
                )
            result.append(summary)
    return result


def _decision(
    null_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    null_checks: dict[str, Any] = {}
    for regime in REGIMES:
        cell = [row for row in null_rows if row["noise_regime"] == regime]
        for mode, field in (
            ("full", "full_gate_active"),
            ("isotropic", "isotropic_gate_active"),
            ("euclidean", "euclidean_gate_active"),
        ):
            activations = sum(bool(row[field]) for row in cell)
            upper = _clopper_pearson_upper(
                activations, len(cell), one_sided_alpha=0.025
            )
            null_checks[f"{regime}_{mode}"] = {
                "activations": activations,
                "trials": len(cell),
                "rate": activations / len(cell),
                "CP97_5_upper": upper,
                "pass": upper
                <= float(
                    PROTOCOL["validity_gates"][
                        "null_false_activation_CP975_upper_per_regime_max"
                    ]
                ),
            }
    primary_contrasts: dict[str, Any] = {}
    primary_checks: dict[str, bool] = {}
    for regime in REGIMES:
        attacked = _contrast(
            evaluation_rows,
            regime=regime,
            candidate="rcig_full",
            baseline="identity_y",
            threats=("bitflip_x10", "model_replacement"),
        )
        benign = _contrast(
            evaluation_rows,
            regime=regime,
            candidate="rcig_full",
            baseline="identity_y",
            threats=("none",),
        )
        attacked_rows = [
            row
            for row in evaluation_rows
            if row["noise_regime"] == regime and row["threat"] != "none"
        ]
        activation = statistics.fmean(
            float(row["full_gate_active"]) for row in attacked_rows
        )
        primary_contrasts[regime] = {
            "attacked_gain_vs_identity_y": attacked,
            "no_attack_gain_vs_identity_y": benign,
            "attack_gate_activation_mean": activation,
        }
        primary_checks[f"{regime}_attacked_mean_gain"] = attacked["mean"] >= 0.10
        primary_checks[f"{regime}_attacked_one_sided_low"] = (
            attacked["one_sided_low"] >= 0.0
        )
        primary_checks[f"{regime}_no_attack_loss_one_sided_high"] = (
            -benign["one_sided_low"] <= 0.02
        )
        primary_checks[f"{regime}_attack_gate_activation"] = activation >= 0.50

    hetero_full_vs_iso = _contrast(
        evaluation_rows,
        regime="heteroscedastic",
        candidate="rcig_full",
        baseline="rcig_isotropic",
        threats=("bitflip_x10", "model_replacement"),
    )
    hetero_full_vs_euclidean = _contrast(
        evaluation_rows,
        regime="heteroscedastic",
        candidate="rcig_full",
        baseline="euclidean_gate",
        threats=("bitflip_x10", "model_replacement"),
    )
    specificity_checks = {
        "heteroscedastic_full_vs_isotropic_one_sided_low": (
            hetero_full_vs_iso["one_sided_low"] >= 0.0
        ),
        "heteroscedastic_full_vs_euclidean_one_sided_low": (
            hetero_full_vs_euclidean["one_sided_low"] >= 0.0
        ),
    }
    cap = float(PROTOCOL["geometry"]["residual_influence_cap"])
    validity_checks = {
        "all_null_CP_checks_pass": all(
            bool(value["pass"]) for value in null_checks.values()
        ),
        "all_rows_on_mps_float32": all(
            row["device"] == "mps:0" and row["dtype"] == "torch.float32"
            for row in (*null_rows, *evaluation_rows)
        ),
        "no_target_used_for_predictor": all(
            not bool(row["target_used_for_predictor"])
            for row in (*null_rows, *evaluation_rows)
        ),
        "no_covariance_oracle": all(
            not bool(row["byzantine_mask_used_for_covariance"])
            and bool(row["nominal_covariance_retained_for_attacked_identities"])
            for row in (*null_rows, *evaluation_rows)
        ),
        "all_predictors_within_cap": all(
            float(row[f"{candidate}_norm"]) <= cap + 1.0e-5
            for row in (*null_rows, *evaluation_rows)
            for candidate in CANDIDATES
        ),
    }
    validity_pass = all(validity_checks.values())
    primary_pass = all(primary_checks.values())
    specificity_pass = all(specificity_checks.values())
    if not validity_pass:
        verdict = "invalid_stop"
    elif not primary_pass:
        verdict = "stop_rcig_candidate"
    elif not specificity_pass:
        verdict = (
            "robust_temporal_gate_supported_but_gaussian_specificity_not_identified"
        )
    else:
        verdict = "advance_locked_gaussian_aware_rcig_confirmation"
    return {
        "verdict": verdict,
        "validity_checks": validity_checks,
        "null_false_activation": null_checks,
        "primary_contrasts": primary_contrasts,
        "primary_checks": primary_checks,
        "gaussian_specificity_contrasts": {
            "heteroscedastic_full_vs_isotropic": hetero_full_vs_iso,
            "heteroscedastic_full_vs_euclidean": hetero_full_vs_euclidean,
        },
        "gaussian_specificity_checks": specificity_checks,
        "validity_pass": validity_pass,
        "primary_pass": primary_pass,
        "gaussian_specificity_pass": specificity_pass,
        "claim_limit": (
            "synthetic onset-of-attack mechanism only; no end-to-end, fairness, "
            "or universal Byzantine robustness claim"
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(output: Path) -> dict[str, Any]:
    validation = _validate_protocol()
    device = _require_mps()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    thresholds, calibration_rows = _calibrate(device)
    null_rows = _null_rows(device, thresholds)
    evaluation_rows = _evaluation_rows(device, thresholds)
    summaries = _summaries(evaluation_rows)
    decision = _decision(null_rows, evaluation_rows)
    _write_json(output / "resolved_protocol.json", PROTOCOL)
    _write_json(output / "static_validation.json", validation)
    _write_json(output / "thresholds.json", thresholds)
    _write_csv(output / "calibration_rows.csv", calibration_rows)
    _write_csv(output / "null_validation_rows.csv", null_rows)
    _write_csv(output / "evaluation_rows.csv", evaluation_rows)
    _write_csv(output / "summary.csv", summaries)
    _write_json(output / "decision.json", decision)
    artifacts = (
        "resolved_protocol.json",
        "static_validation.json",
        "thresholds.json",
        "calibration_rows.csv",
        "null_validation_rows.csv",
        "evaluation_rows.csv",
        "summary.csv",
        "decision.json",
    )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "device": str(device),
        "dtype": "torch.float32",
        "mps_fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "verdict": decision["verdict"],
        "artifact_sha256": {name: _sha256(output / name) for name in artifacts},
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha256(
                Path(__file__).resolve()
            ),
            str(ALGORITHM_PATH.relative_to(ROOT)): _sha256(ALGORITHM_PATH),
            str(PROTOCOL_PATH.relative_to(ROOT)): _sha256(PROTOCOL_PATH),
            str(TEST_PATH.relative_to(ROOT)): _sha256(TEST_PATH),
            str(Path(m0.__file__).resolve().relative_to(ROOT)): _sha256(
                Path(m0.__file__).resolve()
            ),
        },
        "all_checks_pass": bool(decision["validity_pass"]),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validation = _validate_protocol()
    if args.validate:
        print(
            json.dumps(
                {
                    **validation,
                    "mode": "static_only_no_scientific_tensor_computation",
                    "required_environment": "PYTORCH_ENABLE_MPS_FALLBACK=0",
                    "output": str(args.output.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = _run(args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
