#!/usr/bin/env python3
r"""Independent M1 confirmation of the K6b temporal projection.

This synthetic, development-only study reuses the frozen M0 data generator
and clipping/covariance code while correcting two M0 design limitations:

* covariance is the nominal, public covariance attached to every authenticated
  upload; it is never zeroed using the simulator's Byzantine identity;
* Byzantine identities and heteroscedastic noise tiers are counterbalanced by
  a preregistered cyclic Latin-square schedule.

The primary contrast is the regularised temporal projection

    Proj_BG(((a-tau)_+/sqrt(max(b_X,b0))) h_r(Y))

using either raw or EIV-corrected moments.  Fixed-G rules and the older
square-root factorisation are controls only.  The latter is accompanied by an
explicit algebraic-equivalence audit because, away from its floors and caps,
its ``sqrt(b_Y)`` factor cancels the same term in the confidence.

``--run`` is fail-closed MPS/float32 with CPU fallback disabled.  ``--validate``
performs only static checks, dependency hashing and the seed-collision scan; it
does not run scientific tensor computation.  Passing M1 can authorize only a
new, locked n=25 K6b mechanistic confirmation, never an end-to-end claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference_k6_tp_eiv import (  # noqa: E402
    radial_confidence_predictor,
    temporal_eiv_confidence,
    temporal_eiv_corrected_moments,
)
from algorithms.gaussian_aware_reference_k6b_coupled import (  # noqa: E402
    coupled_magnitude_confidence_predictor,
    regularized_radial_direction,
    temporal_projection_predictor,
)
from robustness.aggregators import clip_l2  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k6_tp_eiv_microbench as m0,
)

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k6b_coupled_confirmation_mps_v1"
DEFAULT_OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
ALGORITHM_PATH = ROOT / "algorithms/gaussian_aware_reference_k6b_coupled.py"
M0_RUNNER_PATH = (
    ROOT / "scripts/run_gaussian_aware_reference_g0g_k6_tp_eiv_microbench.py"
)
M0_ALGORITHM_PATH = ROOT / "algorithms/gaussian_aware_reference_k6_tp_eiv.py"
TEST_PATHS = (
    ROOT / "tests/test_gaussian_aware_reference_g0g_k6b_coupled.py",
    ROOT / "tests/test_run_gaussian_aware_reference_g0g_k6b_coupled_confirmation.py",
)
PROTOCOL_DOCUMENT_PATH = (
    ROOT / "output/analysis/Gaussian_Aware_G0g_K6b_Coupled_Protocol_PreRun.md"
)

# Frozen dependency hashes are checked before any scientific run.  Changing M0
# requires a new campaign id and a new protocol rather than silently moving the
# data-generating process.
EXPECTED_M0_RUNNER_SHA256 = (
    "045b0f80cdea53864a843615cfc0da5c486f2721c7085037c539f55b4bf876a7"
)
EXPECTED_M0_ALGORITHM_SHA256 = (
    "5c0c43ce3dc4973786b04bb943c66bcedf8ae672821a1126d2e914d04a5d97ca"
)

CALIBRATION_SEEDS = tuple(202709210101 + 16 * index for index in range(12))
NULL_VALIDATION_SEEDS_BY_REGIME = {
    "homogeneous": tuple(202709215107 + 16 * index for index in range(36)),
    "heteroscedastic": tuple(202709216109 + 16 * index for index in range(36)),
}
EVALUATION_SEEDS = tuple(202709220103 + 16 * index for index in range(12))

NOISE_REGIMES = m0.NOISE_REGIMES
THREATS = m0.THREATS
STUDENT_T_95_ONE_SIDED_DF11 = 1.7958848187036691
STUDENT_T_975_TWO_SIDED_DF11 = 2.200985160091638
CANDIDATES = (
    "radial_y",
    "identity_y",
    "fixed_g_raw",
    "fixed_g_eiv",
    "pt_raw",
    "pt_eiv",
    "sqrt_rawmag_rawconf",
    "sqrt_rawmag_eivconf",
    "sqrt_eivmag_rawconf",
    "sqrt_eivmag_eivconf",
)

PROTOCOL: dict[str, Any] = {
    "campaign_id": CAMPAIGN_ID,
    "scope": "M1_independent_synthetic_mechanistic_confirmation",
    "claims_excluded": [
        "end_to_end_federated_learning_utility",
        "classification_accuracy_or_fairness",
        "universal_byzantine_robustness",
        "publication_confirmation",
        "exact_post_clipping_gaussian_moments_without_mc_validation",
    ],
    "frozen_m0_dependency": {
        "runner_sha256": EXPECTED_M0_RUNNER_SHA256,
        "algorithm_sha256": EXPECTED_M0_ALGORITHM_SHA256,
        "reuse": "data_generator_clipping_gate_and_block_covariance_only",
        "m0_oracle_covariance_policy_reused": False,
    },
    "cohort": {"num_clients": 12, "num_byzantine": 2, "dimension": 8},
    "timeline": {
        "gate_source_rounds": list(range(1, 16)),
        "gate_snapshot_round": 16,
        "older_view_rounds": [16, 17, 18, 19],
        "newer_view_rounds": [20, 21, 22, 23],
        "deployment_round": 24,
        "window_length": 4,
        "attack_onset_round": 20,
    },
    "geometry": {
        "server_clip_norm": 0.75,
        "residual_influence_cap": 0.13,
        "minimum_accepted_mass": 6.0,
        "minimum_direction_norm": 1.0e-6,
        "clip_boundary_margin_min": 1.0e-6,
    },
    "covariance_policy": {
        "source": "authenticated_public_nominal_client_noise_level",
        "retained_for_every_upload_including_simulated_byzantine_replacements": True,
        "byzantine_identity_used_to_zero_covariance": False,
        "cross_covariance": "zero_from_disjoint_noise_streams",
        "status": "server_information_policy_not_oracle_truth_about_attacker_randomness",
    },
    "counterbalancing": {
        "attacker_identity_rule": "cyclic_shift_of_two_base_attacker_slots_by_seed_slot",
        "heteroscedastic_tier_rule": "tier=(client+5*seed_slot) mod 4",
        "paired_across_threat_counterfactuals": True,
        "required_each_client_attacked_count_across_12_evaluation_seeds": 2,
        "required_each_tier_attacker_slot_count": 6,
    },
    "calibration": {
        "alignment_null_quantile": 0.975,
        "raw_and_eiv_thresholds_separate": True,
        "energy_floor_rule": (
            "max(1e-5,q975(max(corrected_older_energy,0))_null_calibration)"
        ),
        "energy_floor_minimum": 1.0e-5,
        "single_energy_floor_across_regimes": True,
        "evaluation_target_or_accuracy_used": False,
    },
    "null_validation": {
        "seeds_per_regime": 36,
        "distinct_seeds_across_regimes": True,
        "one_sided_family_confidence": 0.95,
        "bonferroni_strata_per_candidate": 2,
        "clopper_pearson_upper_max": 0.10,
        "require_positive_median_rejected_gate_mass_per_regime": True,
    },
    "transformed_covariance_mc": {
        "signal_norms": [0.0, 0.065, 0.1274, 0.1326, 0.26, 0.735, 0.765],
        "draws_per_outer_seed": 128,
        "outer_seeds": list(CALIBRATION_SEEDS),
        "noise_regimes": list(NOISE_REGIMES),
        "mean_bias_normalized_max": 0.25,
        "covariance_trace_ratio_min": 0.75,
        "covariance_trace_ratio_max": 1.25,
        "covariance_frobenius_relative_error_max": 0.35,
        "corrected_energy_bias_normalized_max": 0.20,
        "all_grid_cells_required": True,
    },
    "candidates": {
        "primary": "pt_eiv",
        "falsification_control": "pt_raw",
        "fixed_g_status": "stopped_primary_retained_descriptive_control_only",
        "sqrt_status": (
            "preannounced_ablation_and_algebraic_equivalence_diagnostic_only"
        ),
        "factorial_interaction_status": (
            "not_a_science_gate_due_to_algebraic_nonidentifiability"
        ),
    },
    "science_gates": {
        "heteroscedastic_pt_eiv_gain_vs_pt_raw_mean_min": 0.02,
        "heteroscedastic_pt_eiv_gain_vs_pt_raw_one_sided_ci_low_min": 0.0,
        "homogeneous_pt_eiv_loss_vs_pt_raw_one_sided_ci_high_max": 0.01,
        "attacked_pooled_gain_one_sided_ci_low_min": 0.0,
        "no_attack_pooled_loss_one_sided_ci_high_max": 0.02,
        "minimum_stratum_mean_gain": -0.05,
        "identity_y_no_attack_noninferiority_loss_ci_high_max": 0.01,
        "identity_y_attacked_gain_one_sided_ci_low_min": 0.0,
        "predictor_norm_cap_required": True,
        "sqrt_pt_algebraic_equivalence_abs_error_max": 2.0e-6,
        "magnitude_non_degenerate_positive_fraction_min": 0.25,
    },
    "power": {
        "outer_seed_count": 12,
        "one_sided_alpha": 0.05,
        "target_power": 0.80,
        "normal_approximation_paired_standardized_effect": 0.718,
        "finite_sample_planning_effect_rounded": 0.80,
        "warning": "M1_is_powered_only_for_large_mechanistic_effects",
    },
    "decision_policy": {
        "status": "M1_redesign_falsification_screen",
        "p0_original_fixed_g": "stopped",
        "p0b_amplitude_only_promotion": "forbidden",
        "identity_y_is_primary_hard_control": True,
        "recovery_claim_available": False,
        "next_required_redesign": (
            "candidate_that_can_change_direction_not_only_rescale_Y"
        ),
    },
    "randomness": {
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "null_validation_seeds_by_regime": {
            key: list(value) for key, value in NULL_VALIDATION_SEEDS_BY_REGIME.items()
        },
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "paired_candidates": True,
        "paired_threat_counterfactuals": True,
        "statistical_unit": "outer_seed",
        "child_cells_are_not_replicates": True,
    },
    "execution": {
        "required_device": "mps",
        "dtype": "float32",
        "pytorch_mps_fallback_required": "0",
        "overwrite": False,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    return m0._quantile(values, probability)


def _registry_seeds() -> tuple[int, ...]:
    return (
        CALIBRATION_SEEDS
        + NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
        + EVALUATION_SEEDS
    )


def _seed_collision_scan() -> dict[str, Any]:
    """Fail closed if a registered seed appears outside K6b-owned files."""

    allowed = {
        str(Path(__file__).resolve().relative_to(ROOT)),
        str(ALGORITHM_PATH.relative_to(ROOT)),
        *(str(path.relative_to(ROOT)) for path in TEST_PATHS),
        str(PROTOCOL_DOCUMENT_PATH.relative_to(ROOT)),
    }
    expression = "(?:" + "|".join(str(seed) for seed in _registry_seeds()) + ")"
    command = [
        "rg",
        "-l",
        "--hidden",
        "--glob",
        "!.git/**",
        "--glob",
        "!venv/**",
        "--glob",
        "!.cache/**",
        "--glob",
        f"!{DEFAULT_OUTPUT.relative_to(ROOT)}/**",
        "-e",
        expression,
        ".",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Seed collision scan could not complete") from exc
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"Seed collision scan failed: {completed.stderr.strip()}")
    matches = sorted(
        {
            line.removeprefix("./")
            for line in completed.stdout.splitlines()
            if line.strip()
        }
    )
    unexpected = sorted(set(matches) - allowed)
    if unexpected:
        raise RuntimeError(f"Registered K6b seed collision in: {unexpected}")
    evidence = {
        "scanner": "ripgrep_path_scan",
        "registered_seed_count": len(_registry_seeds()),
        "registered_seeds_unique": len(set(_registry_seeds()))
        == len(_registry_seeds()),
        "allowed_matching_paths": matches,
        "unexpected_matching_paths": unexpected,
        "registry_sha256": _canonical_hash(_registry_seeds()),
        "matching_file_sha256": {
            path: _sha256(ROOT / path) for path in matches if (ROOT / path).is_file()
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    return evidence


def _counterbalance_schedule() -> list[dict[str, Any]]:
    n = int(PROTOCOL["cohort"]["num_clients"])
    b = int(PROTOCOL["cohort"]["num_byzantine"])
    tiers = len(NOISE_REGIMES["heteroscedastic"])
    rows = []
    for slot, seed in enumerate(EVALUATION_SEEDS):
        attackers = sorted((n - b + slot + offset) % n for offset in range(b))
        tier_by_client = [(client + 5 * slot) % tiers for client in range(n)]
        rows.append(
            {
                "slot": slot,
                "seed": seed,
                "attacker_clients": attackers,
                "tier_by_client": tier_by_client,
                "attacker_tiers": [tier_by_client[index] for index in attackers],
            }
        )
    return rows


def _validate_protocol(protocol: Mapping[str, Any] = PROTOCOL) -> dict[str, Any]:
    all_null = (
        NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
    )
    schedule = _counterbalance_schedule()
    attacker_counts = [0] * int(protocol["cohort"]["num_clients"])
    tier_counts = [0] * len(NOISE_REGIMES["heteroscedastic"])
    for row in schedule:
        for client in row["attacker_clients"]:
            attacker_counts[int(client)] += 1
        for tier in row["attacker_tiers"]:
            tier_counts[int(tier)] += 1
    timeline = protocol["timeline"]
    checks = {
        "all_seed_registries_pairwise_disjoint": not (
            set(CALIBRATION_SEEDS) & set(all_null)
            or set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS)
            or set(all_null) & set(EVALUATION_SEEDS)
        ),
        "all_registered_seeds_unique": len(set(_registry_seeds()))
        == len(_registry_seeds()),
        "36_distinct_null_seeds_per_regime": all(
            len(values) == len(set(values)) == 36
            for values in NULL_VALIDATION_SEEDS_BY_REGIME.values()
        ),
        "null_regime_registries_disjoint": not (
            set(NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"])
            & set(NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"])
        ),
        "strictly_past_disjoint_views": (
            not set(timeline["older_view_rounds"]) & set(timeline["newer_view_rounds"])
            and max(timeline["newer_view_rounds"]) < int(timeline["deployment_round"])
            and max(timeline["gate_source_rounds"]) < min(timeline["older_view_rounds"])
        ),
        "attack_begins_at_newer_view": int(timeline["attack_onset_round"])
        == min(timeline["newer_view_rounds"]),
        "m0_dependency_hashes_frozen": _sha256(M0_RUNNER_PATH)
        == EXPECTED_M0_RUNNER_SHA256
        and _sha256(M0_ALGORITHM_PATH) == EXPECTED_M0_ALGORITHM_SHA256,
        "no_oracle_covariance_policy": (
            not bool(
                protocol["covariance_policy"][
                    "byzantine_identity_used_to_zero_covariance"
                ]
            )
            and bool(
                protocol["covariance_policy"][
                    "retained_for_every_upload_including_simulated_byzantine_replacements"
                ]
            )
        ),
        "each_client_attacked_exactly_twice": attacker_counts == [2] * 12,
        "attacker_noise_tiers_exactly_balanced": tier_counts == [6] * 4,
        "candidate_registry_exact": CANDIDATES
        == (
            "radial_y",
            "identity_y",
            "fixed_g_raw",
            "fixed_g_eiv",
            "pt_raw",
            "pt_eiv",
            "sqrt_rawmag_rawconf",
            "sqrt_rawmag_eivconf",
            "sqrt_eivmag_rawconf",
            "sqrt_eivmag_eivconf",
        ),
        "fixed_g_primary_stopped": "stopped_primary"
        in str(protocol["candidates"]["fixed_g_status"]),
        "mps_float32_fail_closed": protocol["execution"]
        == {
            "required_device": "mps",
            "dtype": "float32",
            "pytorch_mps_fallback_required": "0",
            "overwrite": False,
        },
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"Invalid K6b M1 protocol: {failed}")
    collision = _seed_collision_scan()
    return {
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": _canonical_hash(PROTOCOL),
        "checks": checks,
        "seed_collision_scan": collision,
        "counterbalance_schedule_sha256": _canonical_hash(schedule),
        "expected_calibration_rows": 24,
        "expected_null_validation_rows": 72,
        "expected_evaluation_rows": 72,
        "expected_mc_calibration_rows": (
            len(CALIBRATION_SEEDS)
            * len(NOISE_REGIMES)
            * len(PROTOCOL["transformed_covariance_mc"]["signal_norms"])
        ),
    }


def _require_mps() -> tuple[torch.device, torch.dtype]:
    return m0._require_mps()


def _seed_slot(seed: int, phase: str, regime: str | None = None) -> int:
    if phase == "calibration":
        return CALIBRATION_SEEDS.index(seed)
    if phase == "evaluation":
        return EVALUATION_SEEDS.index(seed)
    if phase == "null_validation" and regime is not None:
        return NULL_VALIDATION_SEEDS_BY_REGIME[regime].index(seed) % 12
    raise ValueError("Unknown seed/phase combination")


def _counterbalanced_clean(
    seed: int, *, phase: str, regime: str, device: torch.device, null_signal: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    clean, base_honest = m0._clean_process(seed, device=device, null_signal=null_signal)
    slot = _seed_slot(seed, phase, regime)
    clean = torch.roll(clean, shifts=slot, dims=1)
    honest = torch.roll(base_honest, shifts=slot, dims=0)
    return clean, honest, slot


def _counterbalanced_noise_std(
    regime: str, slot: int, *, device: torch.device
) -> torch.Tensor:
    n = int(PROTOCOL["cohort"]["num_clients"])
    tiers = NOISE_REGIMES[regime]
    if len(tiers) == 1:
        values = [tiers[0]] * n
    else:
        values = [tiers[(client + 5 * slot) % len(tiers)] for client in range(n)]
    return torch.tensor(values, device=device, dtype=torch.float32)


def _private_uploads(
    clean: torch.Tensor,
    noise_std: torch.Tensor,
    *,
    seed: int,
    regime: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(
        clean.shape,
        generator=m0._generator(device, CAMPAIGN_ID, "dp-noise", seed, regime),
        device=device,
        dtype=torch.float32,
    )
    uploads = clean + noise * noise_std[None, :, None]
    nominal_variances = noise_std.square()[None, :, None].expand_as(uploads).clone()
    return uploads, nominal_variances


def _apply_threat_without_covariance_oracle(
    uploads: torch.Tensor,
    clean: torch.Tensor,
    nominal_variances: torch.Tensor,
    honest: torch.Tensor,
    *,
    seed: int,
    threat: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if threat not in THREATS:
        raise ValueError(f"Unknown threat: {threat}")
    result = uploads.clone()
    covariance = nominal_variances.clone()
    active = ~honest
    if threat == "none":
        return result, covariance, torch.zeros_like(active)
    start = int(PROTOCOL["timeline"]["attack_onset_round"]) - 1
    if threat == "bitflip_x10":
        result[start:, active] = -10.0 * clean[start:, active]
    else:
        dimension = int(PROTOCOL["cohort"]["dimension"])
        attack_direction = m0._unit(
            torch.randn(
                dimension,
                generator=m0._generator(
                    device, CAMPAIGN_ID, "replacement-direction", seed
                ),
                device=device,
                dtype=torch.float32,
            )
        )
        result[start:, active] = (
            10.0 * float(PROTOCOL["geometry"]["server_clip_norm"]) * attack_direction
        )
    # Deliberately retain the public nominal covariance for every identity.
    # The predictor never receives the simulator's honest/Byzantine mask.
    return result, covariance, active


def _base_context(
    seed: int,
    regime: str,
    threat: str,
    *,
    phase: str,
    device: torch.device,
    null_signal: bool,
) -> dict[str, Any]:
    clean, honest, slot = _counterbalanced_clean(
        seed,
        phase=phase,
        regime=regime,
        device=device,
        null_signal=null_signal,
    )
    noise_std = _counterbalanced_noise_std(regime, slot, device=device)
    uploads, variances = _private_uploads(
        clean, noise_std, seed=seed, regime=regime, device=device
    )
    uploads, variances, active = _apply_threat_without_covariance_oracle(
        uploads,
        clean,
        variances,
        honest,
        seed=seed,
        threat=threat,
        device=device,
    )
    dimension = int(PROTOCOL["cohort"]["dimension"])
    n = int(PROTOCOL["cohort"]["num_clients"])
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
    cross = torch.zeros_like(covariance_older)
    corrected_a, corrected_bx, corrected_by = temporal_eiv_corrected_moments(
        older,
        newer,
        covariance_older=covariance_older,
        covariance_newer=covariance_newer,
        cross_covariance_older_newer=cross,
    )
    raw_a = torch.dot(older, newer)
    raw_bx = torch.dot(older, older)
    raw_by = torch.dot(newer, newer)
    target_round = int(PROTOCOL["timeline"]["deployment_round"]) - 1
    target = clip_l2(
        clean[target_round, honest].mean(dim=0),
        float(PROTOCOL["geometry"]["residual_influence_cap"]),
    )
    return {
        "seed": seed,
        "noise_regime": regime,
        "threat": threat,
        "phase": phase,
        "slot": slot,
        "clean": clean,
        "honest": honest,
        "active": active,
        "noise_std": noise_std,
        "gate": gate,
        "gate_diagnostics": gate_diagnostics,
        "older": older,
        "newer": newer,
        "covariance_older": covariance_older,
        "covariance_newer": covariance_newer,
        "older_diagnostics": older_diagnostics,
        "newer_diagnostics": newer_diagnostics,
        "corrected_a": corrected_a,
        "corrected_bx": corrected_bx,
        "corrected_by": corrected_by,
        "raw_a": raw_a,
        "raw_bx": raw_bx,
        "raw_by": raw_by,
        "target": target,
    }


def _mse(value: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((value - target).square()).item())


def _mse_reconstruction(
    *,
    radial_mse: float,
    radial_norm: float,
    target_norm: float,
    influence_cap: float,
    coefficient_on_h: float,
    h_norm: float,
    dimension: int,
) -> float:
    cap = float(influence_cap)
    dot_h_target = (
        radial_norm**2 + target_norm**2 - dimension * float(radial_mse)
    ) / (2.0 * cap)
    return (
        coefficient_on_h**2 * h_norm**2
        - 2.0 * coefficient_on_h * dot_h_target
        + target_norm**2
    ) / dimension


def _candidate_row(
    base: Mapping[str, Any], *, tau_raw: float, tau_eiv: float, b0: float
) -> dict[str, Any]:
    newer = base["newer"]
    target = base["target"]
    cap = float(PROTOCOL["geometry"]["residual_influence_cap"])
    r_min = float(PROTOCOL["geometry"]["minimum_direction_norm"])
    b_min = float(PROTOCOL["calibration"]["energy_floor_minimum"])
    raw_confidence = temporal_eiv_confidence(
        base["raw_a"],
        base["raw_bx"],
        base["raw_by"],
        b_min=b_min,
        tau_a=tau_raw,
    )
    eiv_confidence = temporal_eiv_confidence(
        base["corrected_a"],
        base["corrected_bx"],
        base["corrected_by"],
        b_min=b_min,
        tau_a=tau_eiv,
    )
    radial = radial_confidence_predictor(
        newer, 1.0, influence_cap=cap, minimum_direction_norm=r_min
    )
    identity_y = clip_l2(newer, cap)
    fixed_raw = radial_confidence_predictor(
        newer,
        raw_confidence,
        influence_cap=cap,
        minimum_direction_norm=r_min,
    )
    fixed_eiv = radial_confidence_predictor(
        newer,
        eiv_confidence,
        influence_cap=cap,
        minimum_direction_norm=r_min,
    )
    pt_raw, pt_raw_diagnostics = temporal_projection_predictor(
        newer,
        base["raw_a"],
        base["raw_bx"],
        alignment_threshold=tau_raw,
        energy_floor=b0,
        influence_cap=cap,
        minimum_direction_norm=r_min,
        return_diagnostics=True,
    )
    pt_eiv, pt_eiv_diagnostics = temporal_projection_predictor(
        newer,
        base["corrected_a"],
        base["corrected_bx"],
        alignment_threshold=tau_eiv,
        energy_floor=b0,
        influence_cap=cap,
        minimum_direction_norm=r_min,
        return_diagnostics=True,
    )
    factorial: dict[str, torch.Tensor] = {}
    factorial_diagnostics: dict[str, Mapping[str, Any]] = {}
    for magnitude_name, magnitude_energy in (
        ("rawmag", base["raw_by"]),
        ("eivmag", base["corrected_by"]),
    ):
        for confidence_name, confidence in (
            ("rawconf", raw_confidence),
            ("eivconf", eiv_confidence),
        ):
            name = f"sqrt_{magnitude_name}_{confidence_name}"
            value, diagnostics = coupled_magnitude_confidence_predictor(
                newer,
                confidence,
                magnitude_energy,
                influence_cap=cap,
                minimum_direction_norm=r_min,
                b0=b0,
                magnitude_rule="sqrt",
                return_diagnostics=True,
            )
            factorial[name] = value
            factorial_diagnostics[name] = diagnostics
    predictors = {
        "radial_y": radial,
        "identity_y": identity_y,
        "fixed_g_raw": fixed_raw,
        "fixed_g_eiv": fixed_eiv,
        "pt_raw": pt_raw,
        "pt_eiv": pt_eiv,
        **factorial,
    }
    mses = {name: _mse(value, target) for name, value in predictors.items()}
    direction = regularized_radial_direction(newer, minimum_direction_norm=r_min)
    h_norm = float(torch.linalg.vector_norm(direction).item())
    radial_norm = float(torch.linalg.vector_norm(radial).item())
    target_norm = float(torch.linalg.vector_norm(target).item())
    reconstruction_errors = []
    for name, value in predictors.items():
        if h_norm > 0.0:
            coefficient = float(torch.dot(value, direction).item()) / (h_norm**2)
        else:
            coefficient = 0.0
        reconstructed = _mse_reconstruction(
            radial_mse=mses["radial_y"],
            radial_norm=radial_norm,
            target_norm=target_norm,
            influence_cap=cap,
            coefficient_on_h=coefficient,
            h_norm=h_norm,
            dimension=int(PROTOCOL["cohort"]["dimension"]),
        )
        reconstruction_errors.append(abs(reconstructed - mses[name]))
    equivalence_conditions = (
        float(base["corrected_by"].item()) > b_min
        and math.sqrt(max(float(base["corrected_by"].item()), 0.0)) < cap
        and float(torch.linalg.vector_norm(newer).item()) > r_min
    )
    equivalence_error = float(
        torch.linalg.vector_norm(factorial["sqrt_eivmag_eivconf"] - pt_eiv).item()
    )
    gate = base["gate"]
    old_diag = base["older_diagnostics"]
    new_diag = base["newer_diagnostics"]
    active_indices = torch.nonzero(base["active"], as_tuple=False).flatten()
    attacker_tiers = [
        int((int(index) + 5 * int(base["slot"])) % 4)
        for index in active_indices.detach().cpu().tolist()
    ]
    row: dict[str, Any] = {
        "seed": int(base["seed"]),
        "noise_regime": str(base["noise_regime"]),
        "threat": str(base["threat"]),
        "phase": str(base["phase"]),
        "pairing_id": f"{base['seed']}|{base['noise_regime']}",
        "cell_id": f"{base['seed']}|{base['noise_regime']}|{base['threat']}",
        "device": str(newer.device),
        "dtype": str(newer.dtype),
        "tau_raw": float(tau_raw),
        "tau_eiv": float(tau_eiv),
        "b0": float(b0),
        "raw_alignment": float(base["raw_a"].item()),
        "eiv_alignment": float(base["corrected_a"].item()),
        "raw_energy_x": float(base["raw_bx"].item()),
        "eiv_energy_x": float(base["corrected_bx"].item()),
        "raw_energy_y": float(base["raw_by"].item()),
        "eiv_energy_y": float(base["corrected_by"].item()),
        "raw_confidence": float(raw_confidence.item()),
        "eiv_confidence": float(eiv_confidence.item()),
        "target_norm": target_norm,
        "newer_norm": float(torch.linalg.vector_norm(newer).item()),
        "gate_mean": float(gate.mean().item()),
        "gate_min": float(gate.min().item()),
        "rejected_gate_mass": float((1.0 - gate).sum().item()),
        "gate_hash": str(base["gate_diagnostics"]["gate_hash"]),
        "attacker_indices": json.dumps(active_indices.detach().cpu().tolist()),
        "attacker_tiers": json.dumps(attacker_tiers),
        "nominal_covariance_retained_for_attackers": True,
        "byzantine_identity_used_in_covariance": False,
        "covariance_psd": bool(
            old_diag["covariance_psd"] and new_diag["covariance_psd"]
        ),
        "minimum_clip_margin": min(
            float(old_diag["minimum_piecewise_differentiability_margin"]),
            float(new_diag["minimum_piecewise_differentiability_margin"]),
        ),
        "residual_clip_active_fraction": statistics.fmean(
            [
                float(old_diag["residual_clip_active_fraction"]),
                float(new_diag["residual_clip_active_fraction"]),
            ]
        ),
        "current_round_input_used": False,
        "target_used_for_predictor": False,
        "pt_raw_amplitude": float(pt_raw_diagnostics["predictor_norm"]),
        "pt_eiv_amplitude": float(pt_eiv_diagnostics["predictor_norm"]),
        "sqrt_eiv_magnitude": float(
            factorial_diagnostics["sqrt_eivmag_eivconf"]["magnitude"]
        ),
        "sqrt_equivalence_conditions_hold": bool(equivalence_conditions),
        "sqrt_pt_equivalence_l2_error": equivalence_error,
        "algebraic_mse_reconstruction_max_abs_error": max(reconstruction_errors),
    }
    for name in CANDIDATES:
        row[f"{name}_mse"] = mses[name]
        row[f"{name}_norm"] = float(torch.linalg.vector_norm(predictors[name]).item())
    return row


def _calibrate(
    device: torch.device,
) -> tuple[float, float, float, list[dict[str, Any]]]:
    bases = [
        _base_context(
            seed,
            regime,
            "none",
            phase="calibration",
            device=device,
            null_signal=True,
        )
        for seed in CALIBRATION_SEEDS
        for regime in NOISE_REGIMES
    ]
    probability = float(PROTOCOL["calibration"]["alignment_null_quantile"])
    tau_raw = max(0.0, _quantile([float(x["raw_a"]) for x in bases], probability))
    tau_eiv = max(0.0, _quantile([float(x["corrected_a"]) for x in bases], probability))
    b0 = max(
        float(PROTOCOL["calibration"]["energy_floor_minimum"]),
        _quantile([max(float(x["corrected_bx"]), 0.0) for x in bases], probability),
    )
    rows = [
        _candidate_row(base, tau_raw=tau_raw, tau_eiv=tau_eiv, b0=b0) for base in bases
    ]
    return tau_raw, tau_eiv, b0, rows


def _null_validation_rows(
    device: torch.device, *, tau_raw: float, tau_eiv: float, b0: float
) -> list[dict[str, Any]]:
    rows = []
    for regime, seeds in NULL_VALIDATION_SEEDS_BY_REGIME.items():
        for seed in seeds:
            base = _base_context(
                seed,
                regime,
                "none",
                phase="null_validation",
                device=device,
                null_signal=True,
            )
            rows.append(_candidate_row(base, tau_raw=tau_raw, tau_eiv=tau_eiv, b0=b0))
    return rows


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    total = 0.0
    for value in range(successes + 1):
        total += (
            math.comb(trials, value)
            * probability**value
            * (1.0 - probability) ** (trials - value)
        )
    return total


def _clopper_pearson_upper(
    successes: int, trials: int, *, one_sided_alpha: float
) -> float:
    """Exact one-sided CP upper bound via the binomial-tail equation."""

    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("Require 0 <= successes <= trials and trials >= 1")
    if not 0.0 < one_sided_alpha < 1.0:
        raise ValueError("one_sided_alpha must lie in (0,1)")
    if successes == trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _binomial_cdf(successes, trials, middle) > one_sided_alpha:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _student_interval(
    values: Sequence[float], *, one_sided: bool
) -> dict[str, float | int | str]:
    numbers = [float(value) for value in values]
    if len(numbers) != 12:
        raise ValueError("K6b inference requires exactly 12 outer seeds")
    mean = statistics.fmean(numbers)
    sd = statistics.stdev(numbers)
    critical = (
        STUDENT_T_95_ONE_SIDED_DF11 if one_sided else STUDENT_T_975_TWO_SIDED_DF11
    )
    half = critical * sd / math.sqrt(len(numbers))
    return {
        "n_independent_outer_seeds": len(numbers),
        "mean": mean,
        "standard_deviation": sd,
        "low": mean - half,
        "high": mean + half,
        "critical": critical,
        "interval": "one_sided_95" if one_sided else "two_sided_95",
    }


def _relative_gain(control: float, candidate: float) -> float:
    return (float(control) - float(candidate)) / max(float(control), 1.0e-15)


def _seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["noise_regime"]))].append(row)
    result = []
    for (seed, regime), group in sorted(grouped.items()):
        benign = [row for row in group if row["threat"] == "none"]
        attacked = [row for row in group if row["threat"] != "none"]
        if len(benign) != 1 or len(attacked) != 2:
            raise RuntimeError(
                "Each seed/regime must contain one benign and two attacks"
            )
        raw_all = sum(float(row["pt_raw_mse"]) for row in group)
        eiv_all = sum(float(row["pt_eiv_mse"]) for row in group)
        raw_attacked = sum(float(row["pt_raw_mse"]) for row in attacked)
        eiv_attacked = sum(float(row["pt_eiv_mse"]) for row in attacked)
        result.append(
            {
                "seed": seed,
                "noise_regime": regime,
                "pt_eiv_gain_vs_pt_raw_all": _relative_gain(raw_all, eiv_all),
                "pt_eiv_gain_vs_pt_raw_benign": _relative_gain(
                    float(benign[0]["pt_raw_mse"]),
                    float(benign[0]["pt_eiv_mse"]),
                ),
                "pt_eiv_gain_vs_pt_raw_attacked": _relative_gain(
                    raw_attacked, eiv_attacked
                ),
                "pt_eiv_gain_vs_fixed_g_raw_all": _relative_gain(
                    sum(float(row["fixed_g_raw_mse"]) for row in group), eiv_all
                ),
                "pt_eiv_gain_vs_identity_y_all": _relative_gain(
                    sum(float(row["identity_y_mse"]) for row in group), eiv_all
                ),
                "pt_eiv_gain_vs_identity_y_benign": _relative_gain(
                    float(benign[0]["identity_y_mse"]),
                    float(benign[0]["pt_eiv_mse"]),
                ),
                "pt_eiv_gain_vs_identity_y_attacked": _relative_gain(
                    sum(float(row["identity_y_mse"]) for row in attacked),
                    eiv_attacked,
                ),
            }
        )
    return result


def _contrast_intervals(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_regime = {
        regime: {
            metric: _student_interval(
                [
                    float(row[metric])
                    for row in seed_rows
                    if row["noise_regime"] == regime
                ],
                one_sided=True,
            )
            for metric in (
                "pt_eiv_gain_vs_pt_raw_all",
                "pt_eiv_gain_vs_pt_raw_benign",
                "pt_eiv_gain_vs_pt_raw_attacked",
                "pt_eiv_gain_vs_identity_y_all",
                "pt_eiv_gain_vs_identity_y_benign",
                "pt_eiv_gain_vs_identity_y_attacked",
            )
        }
        for regime in NOISE_REGIMES
    }
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_seed[int(row["seed"])].append(row)
    pooled = {
        metric: _student_interval(
            [
                statistics.fmean(float(row[metric]) for row in by_seed[seed])
                for seed in sorted(by_seed)
            ],
            one_sided=True,
        )
        for metric in (
            "pt_eiv_gain_vs_pt_raw_all",
            "pt_eiv_gain_vs_pt_raw_benign",
            "pt_eiv_gain_vs_pt_raw_attacked",
            "pt_eiv_gain_vs_fixed_g_raw_all",
            "pt_eiv_gain_vs_identity_y_all",
            "pt_eiv_gain_vs_identity_y_benign",
            "pt_eiv_gain_vs_identity_y_attacked",
        )
    }
    return {
        "independent_unit": "outer_seed",
        "child_cells_are_not_replicates": True,
        "by_noise_regime": by_regime,
        "pooled_equal_weight_regimes_within_seed": pooled,
    }


def _fixed_gate_view_batch(
    uploads: torch.Tensor, gate: torch.Tensor, anchor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised transformed views and random delta covariance estimates."""

    draws, length, n, dimension = uploads.shape
    residuals, jacobians, _ = m0._clip_chain_with_jacobian(
        uploads.reshape(-1, dimension), anchor
    )
    residuals = residuals.reshape(draws, length, n, dimension)
    jacobians = jacobians.reshape(draws, length, n, dimension, dimension)
    denominator = length * torch.maximum(
        gate.sum(),
        torch.as_tensor(
            float(PROTOCOL["geometry"]["minimum_accepted_mass"]),
            device=uploads.device,
            dtype=uploads.dtype,
        ),
    )
    coefficients = gate / denominator
    views = torch.sum(coefficients[None, None, :, None] * residuals, dim=(1, 2))
    cap = float(PROTOCOL["geometry"]["residual_influence_cap"])
    norms = torch.linalg.vector_norm(views, dim=1)
    if bool((norms >= cap - 1.0e-7).any()):
        raise RuntimeError("MC transformed view reached its final projection boundary")
    return views, jacobians


def _mc_base_uploads(
    signal_norm: float, *, seed: int, device: torch.device
) -> torch.Tensor:
    length = int(PROTOCOL["timeline"]["window_length"])
    n = int(PROTOCOL["cohort"]["num_clients"])
    d = int(PROTOCOL["cohort"]["dimension"])
    main = torch.zeros(d, device=device, dtype=torch.float32)
    main[0] = 1.0
    directions = []
    for client in range(n):
        vector = main.clone()
        vector[1] = 0.22 * math.sin(2.0 * math.pi * client / n)
        vector[2] = 0.16 * math.cos(2.0 * math.pi * client / n)
        directions.append(m0._unit(vector))
    direction_matrix = torch.stack(directions)
    temporal = torch.linspace(0.97, 1.03, length, device=device, dtype=torch.float32)
    return float(signal_norm) * temporal[:, None, None] * direction_matrix[None, :, :]


def _transformed_covariance_mc(
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    draws = int(PROTOCOL["transformed_covariance_mc"]["draws_per_outer_seed"])
    d = int(PROTOCOL["cohort"]["dimension"])
    anchor = torch.zeros(d, device=device, dtype=torch.float32)
    for seed in CALIBRATION_SEEDS:
        slot = _seed_slot(seed, "calibration")
        for regime in NOISE_REGIMES:
            std = _counterbalanced_noise_std(regime, slot, device=device)
            variance = std.square()[None, :, None].expand(
                int(PROTOCOL["timeline"]["window_length"]), -1, d
            )
            # Fixed, nontrivial and identity-agnostic gate for the transform audit.
            gate = torch.linspace(0.55, 1.0, 12, device=device, dtype=torch.float32)
            for signal_norm in PROTOCOL["transformed_covariance_mc"]["signal_norms"]:
                base = _mc_base_uploads(float(signal_norm), seed=seed, device=device)
                _, analytic_at_base, _ = m0._view_and_covariance(
                    base, variance, gate, anchor
                )
                noise = torch.randn(
                    draws,
                    *base.shape,
                    generator=m0._generator(
                        device,
                        CAMPAIGN_ID,
                        "transformed-covariance-mc",
                        seed,
                        regime,
                        signal_norm,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                samples = base[None, :, :, :] + noise * std[None, None, :, None]
                views, jacobians = _fixed_gate_view_batch(samples, gate, anchor)
                denominator = int(PROTOCOL["timeline"]["window_length"]) * max(
                    float(gate.sum().item()),
                    float(PROTOCOL["geometry"]["minimum_accepted_mass"]),
                )
                coefficients = gate / denominator
                factors = (
                    coefficients[None, None, :, None, None]
                    * jacobians
                    * std[None, None, :, None, None]
                )
                delta_covariances = torch.sum(
                    factors @ factors.transpose(-1, -2), dim=(1, 2)
                )
                mean_delta_covariance = delta_covariances.mean(dim=0)
                empirical_mean = views.mean(dim=0)
                centred = views - empirical_mean
                empirical_covariance = centred.T @ centred / float(draws - 1)
                noiseless_view, _, _ = m0._view_and_covariance(
                    base, torch.zeros_like(variance), gate, anchor
                )
                scale = max(
                    float(torch.linalg.vector_norm(noiseless_view).item()),
                    math.sqrt(
                        max(float(torch.trace(empirical_covariance).item()), 0.0)
                    ),
                    1.0e-12,
                )
                mean_bias = (
                    float(
                        torch.linalg.vector_norm(empirical_mean - noiseless_view).item()
                    )
                    / scale
                )
                empirical_trace = float(torch.trace(empirical_covariance).item())
                delta_trace = float(torch.trace(mean_delta_covariance).item())
                trace_ratio = delta_trace / max(empirical_trace, 1.0e-15)
                frobenius_error = float(
                    torch.linalg.matrix_norm(
                        mean_delta_covariance - empirical_covariance
                    ).item()
                ) / max(
                    float(torch.linalg.matrix_norm(empirical_covariance).item()),
                    1.0e-15,
                )
                corrected_energy_mean = float(
                    (
                        views.square().sum(dim=1)
                        - torch.diagonal(delta_covariances, dim1=-2, dim2=-1).sum(dim=1)
                    )
                    .mean()
                    .item()
                )
                latent_energy = float(empirical_mean.square().sum().item())
                energy_scale = max(latent_energy, empirical_trace, 1.0e-15)
                energy_bias = abs(corrected_energy_mean - latent_energy) / energy_scale
                rows.append(
                    {
                        "seed": seed,
                        "noise_regime": regime,
                        "signal_norm": float(signal_norm),
                        "draws": draws,
                        "device": str(views.device),
                        "dtype": str(views.dtype),
                        "mean_bias_normalized": mean_bias,
                        "empirical_covariance_trace": empirical_trace,
                        "analytic_at_base_covariance_trace": float(
                            torch.trace(analytic_at_base).item()
                        ),
                        "mean_random_delta_covariance_trace": delta_trace,
                        "random_delta_to_empirical_trace_ratio": trace_ratio,
                        "random_delta_covariance_frobenius_relative_error": frobenius_error,
                        "corrected_energy_mean": corrected_energy_mean,
                        "empirical_mean_energy": latent_energy,
                        "corrected_energy_bias_normalized": energy_bias,
                        "delta_covariance_random_and_correlated_with_view": True,
                    }
                )
    return rows


def _mc_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["noise_regime"]), float(row["signal_norm"]))].append(row)
    return [
        {
            "noise_regime": regime,
            "signal_norm": signal_norm,
            "outer_seeds": len(group),
            "mean_bias_normalized_mean": statistics.fmean(
                float(row["mean_bias_normalized"]) for row in group
            ),
            "trace_ratio_mean": statistics.fmean(
                float(row["random_delta_to_empirical_trace_ratio"]) for row in group
            ),
            "frobenius_relative_error_mean": statistics.fmean(
                float(row["random_delta_covariance_frobenius_relative_error"])
                for row in group
            ),
            "corrected_energy_bias_normalized_mean": statistics.fmean(
                float(row["corrected_energy_bias_normalized"]) for row in group
            ),
        }
        for (regime, signal_norm), group in sorted(grouped.items())
    ]


def _decision(
    calibration_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    mc_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seed_rows = _seed_summary(evaluation_rows)
    intervals = _contrast_intervals(seed_rows)
    alpha_stratum = (
        1.0 - float(PROTOCOL["null_validation"]["one_sided_family_confidence"])
    ) / int(PROTOCOL["null_validation"]["bonferroni_strata_per_candidate"])
    null_diagnostics: dict[str, Any] = {}
    null_pass = True
    gate_pass = True
    for candidate, alignment, tau in (
        ("raw", "raw_alignment", float(evaluation_rows[0]["tau_raw"])),
        ("eiv", "eiv_alignment", float(evaluation_rows[0]["tau_eiv"])),
    ):
        null_diagnostics[candidate] = {}
        for regime in NOISE_REGIMES:
            group = [row for row in null_rows if row["noise_regime"] == regime]
            successes = sum(float(row[alignment]) > tau for row in group)
            upper = _clopper_pearson_upper(
                successes, len(group), one_sided_alpha=alpha_stratum
            )
            rejected_median = statistics.median(
                float(row["rejected_gate_mass"]) for row in group
            )
            passed = upper <= float(
                PROTOCOL["null_validation"]["clopper_pearson_upper_max"]
            )
            nontrivial = rejected_median > 0.0
            null_pass = null_pass and passed
            gate_pass = gate_pass and nontrivial
            null_diagnostics[candidate][regime] = {
                "activations": successes,
                "trials": len(group),
                "empirical_rate": successes / len(group),
                "one_sided_clopper_pearson_upper_bonferroni": upper,
                "per_stratum_alpha": alpha_stratum,
                "passed": passed,
                "median_rejected_gate_mass": rejected_median,
                "gate_nontrivial": nontrivial,
            }
    mc_config = PROTOCOL["transformed_covariance_mc"]
    mc_cell_passes = [
        (
            float(row["mean_bias_normalized"])
            <= float(mc_config["mean_bias_normalized_max"])
            and float(mc_config["covariance_trace_ratio_min"])
            <= float(row["random_delta_to_empirical_trace_ratio"])
            <= float(mc_config["covariance_trace_ratio_max"])
            and float(row["random_delta_covariance_frobenius_relative_error"])
            <= float(mc_config["covariance_frobenius_relative_error_max"])
            and float(row["corrected_energy_bias_normalized"])
            <= float(mc_config["corrected_energy_bias_normalized_max"])
        )
        for row in mc_rows
    ]
    hetero = intervals["by_noise_regime"]["heteroscedastic"]
    homogeneous = intervals["by_noise_regime"]["homogeneous"]
    pooled = intervals["pooled_equal_weight_regimes_within_seed"]
    stratum_means = {
        f"{regime}|{threat}": statistics.fmean(
            _relative_gain(float(row["pt_raw_mse"]), float(row["pt_eiv_mse"]))
            for row in evaluation_rows
            if row["noise_regime"] == regime and row["threat"] == threat
        )
        for regime in NOISE_REGIMES
        for threat in THREATS
    }
    norms_bounded = all(
        float(row[f"{candidate}_norm"])
        <= float(PROTOCOL["geometry"]["residual_influence_cap"]) + 2.0e-6
        for row in evaluation_rows
        for candidate in CANDIDATES
    )
    positive_amplitudes = statistics.fmean(
        float(float(row["pt_eiv_norm"]) > 0.0) for row in evaluation_rows
    )
    equivalence_rows = [
        row for row in evaluation_rows if bool(row["sqrt_equivalence_conditions_hold"])
    ]
    equivalence_max = max(
        (float(row["sqrt_pt_equivalence_l2_error"]) for row in equivalence_rows),
        default=math.inf,
    )
    validity = {
        "calibration_rows_exact": len(calibration_rows) == 24,
        "null_rows_exact": len(null_rows) == 72,
        "evaluation_rows_exact": len(evaluation_rows) == 72,
        "mc_rows_exact": len(mc_rows) == 12 * 2 * len(mc_config["signal_norms"]),
        "all_mps_float32": all(
            row["device"] == "mps" and row["dtype"] == "torch.float32"
            for row in (*calibration_rows, *null_rows, *evaluation_rows, *mc_rows)
        ),
        "all_covariances_psd": all(
            bool(row["covariance_psd"])
            for row in (*calibration_rows, *null_rows, *evaluation_rows)
        ),
        "no_covariance_identity_oracle": all(
            bool(row["nominal_covariance_retained_for_attackers"])
            and not bool(row["byzantine_identity_used_in_covariance"])
            for row in evaluation_rows
        ),
        "past_only": all(
            not bool(row["current_round_input_used"])
            and not bool(row["target_used_for_predictor"])
            for row in evaluation_rows
        ),
        "algebraic_reconstruction_exact": max(
            float(row["algebraic_mse_reconstruction_max_abs_error"])
            for row in evaluation_rows
        )
        <= 2.0e-8,
        "norm_cap": norms_bounded,
        "sqrt_equivalence_has_support": len(equivalence_rows) > 0,
        "sqrt_equivalence_when_conditions_hold": equivalence_max
        <= float(
            PROTOCOL["science_gates"]["sqrt_pt_algebraic_equivalence_abs_error_max"]
        ),
    }
    scientific = {
        "null_fpr_cp_bonferroni_controlled": null_pass,
        "gate_nontrivial_per_regime": gate_pass,
        "transformed_covariance_mc_calibrated_on_entire_grid": all(mc_cell_passes),
        "primary_magnitude_non_degenerate": positive_amplitudes
        >= float(
            PROTOCOL["science_gates"]["magnitude_non_degenerate_positive_fraction_min"]
        ),
        "heteroscedastic_gain_mean_at_least_2pct": float(
            hetero["pt_eiv_gain_vs_pt_raw_all"]["mean"]
        )
        >= float(
            PROTOCOL["science_gates"]["heteroscedastic_pt_eiv_gain_vs_pt_raw_mean_min"]
        ),
        "heteroscedastic_gain_ci_low_positive": float(
            hetero["pt_eiv_gain_vs_pt_raw_all"]["low"]
        )
        > 0.0,
        "homogeneous_noninferiority_loss_below_1pct": -float(
            homogeneous["pt_eiv_gain_vs_pt_raw_all"]["low"]
        )
        < float(
            PROTOCOL["science_gates"][
                "homogeneous_pt_eiv_loss_vs_pt_raw_one_sided_ci_high_max"
            ]
        ),
        "attacked_pooled_gain_ci_low_positive": float(
            pooled["pt_eiv_gain_vs_pt_raw_attacked"]["low"]
        )
        > 0.0,
        "no_attack_pooled_loss_ci_high_below_2pct": -float(
            pooled["pt_eiv_gain_vs_pt_raw_benign"]["low"]
        )
        < float(
            PROTOCOL["science_gates"]["no_attack_pooled_loss_one_sided_ci_high_max"]
        ),
        "no_stratum_mean_loss_above_5pct": min(stratum_means.values())
        >= float(PROTOCOL["science_gates"]["minimum_stratum_mean_gain"]),
        "identity_y_no_attack_noninferiority": -float(
            pooled["pt_eiv_gain_vs_identity_y_benign"]["low"]
        )
        < float(
            PROTOCOL["science_gates"][
                "identity_y_no_attack_noninferiority_loss_ci_high_max"
            ]
        ),
        "identity_y_attacked_gain_ci_low_positive": float(
            pooled["pt_eiv_gain_vs_identity_y_attacked"]["low"]
        )
        > float(
            PROTOCOL["science_gates"]["identity_y_attacked_gain_one_sided_ci_low_min"]
        ),
    }
    all_valid = all(validity.values())
    all_science = all(scientific.values())
    if not scientific["transformed_covariance_mc_calibrated_on_entire_grid"]:
        decision = "requires_transformed_covariance_calibration"
    elif all_valid and all_science:
        decision = (
            "amplitude_only_pass_but_p0b_promotion_forbidden_pending_direction_redesign"
        )
    else:
        decision = "stop_amplitude_only_k6b_and_redesign_direction"
    return {
        "validity_checks": validity,
        "scientific_checks": scientific,
        "all_checks_pass": all_valid and all_science,
        "decision": decision,
        "null_validation": null_diagnostics,
        "transformed_covariance_mc_cells_passing": sum(mc_cell_passes),
        "transformed_covariance_mc_cells_total": len(mc_cell_passes),
        "primary_positive_amplitude_fraction": positive_amplitudes,
        "sqrt_equivalence_supported_rows": len(equivalence_rows),
        "sqrt_pt_equivalence_max_l2_error": equivalence_max,
        "stratum_mean_relative_gains_pt_eiv_vs_pt_raw": stratum_means,
        "paired_seed_level_intervals": intervals,
        "power_warning": PROTOCOL["power"],
        "scope_warning": (
            "M1 is a synthetic falsification harness. Even a pass cannot promote "
            "the amplitude-only K6b to n=25. A new preregistered candidate must "
            "change direction relative to Identity-Y before another campaign."
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    m0._write_json(path, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    m0._write_csv(path, rows)


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    static = _validate_protocol()
    device, _ = _require_mps()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tau_raw, tau_eiv, b0, calibration_rows = _calibrate(device)
    null_rows = _null_validation_rows(device, tau_raw=tau_raw, tau_eiv=tau_eiv, b0=b0)
    evaluation_rows = [
        _candidate_row(
            _base_context(
                seed,
                regime,
                threat,
                phase="evaluation",
                device=device,
                null_signal=False,
            ),
            tau_raw=tau_raw,
            tau_eiv=tau_eiv,
            b0=b0,
        )
        for seed in EVALUATION_SEEDS
        for regime in NOISE_REGIMES
        for threat in THREATS
    ]
    mc_rows = _transformed_covariance_mc(device)
    torch.mps.synchronize()
    seed_rows = _seed_summary(evaluation_rows)
    intervals = _contrast_intervals(seed_rows)
    mc_summary = _mc_summary(mc_rows)
    decision = _decision(calibration_rows, null_rows, evaluation_rows, mc_rows)
    artifacts: dict[str, Any] = {
        "null_calibration_rows.csv": calibration_rows,
        "independent_null_validation_rows.csv": null_rows,
        "evaluation_rows.csv": evaluation_rows,
        "seed_summary.csv": seed_rows,
        "transformed_covariance_mc_rows.csv": mc_rows,
        "transformed_covariance_mc_summary.csv": mc_summary,
    }
    for name, rows in artifacts.items():
        _write_csv(output / name, rows)
    _write_json(output / "contrast_intervals.json", intervals)
    _write_json(
        output / "calibration.json",
        {
            "tau_raw": tau_raw,
            "tau_eiv": tau_eiv,
            "b0": b0,
            "evaluation_target_or_accuracy_used": False,
            "thresholds_calibrated_separately": True,
            "single_b0_across_regimes": True,
        },
    )
    _write_json(output / "counterbalance_schedule.json", _counterbalance_schedule())
    _write_json(output / "decision.json", decision)
    _write_json(output / "resolved_protocol.json", PROTOCOL)
    artifact_names = tuple(artifacts) + (
        "contrast_intervals.json",
        "calibration.json",
        "counterbalance_schedule.json",
        "decision.json",
        "resolved_protocol.json",
    )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "scope": PROTOCOL["scope"],
        "device": "mps",
        "dtype": "float32",
        "mps_fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0",
        "protocol_sha256": static["protocol_sha256"],
        "seed_collision_scan": static["seed_collision_scan"],
        "counterbalance_schedule_sha256": static["counterbalance_schedule_sha256"],
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha256(
                Path(__file__).resolve()
            ),
            str(ALGORITHM_PATH.relative_to(ROOT)): _sha256(ALGORITHM_PATH),
            str(M0_RUNNER_PATH.relative_to(ROOT)): _sha256(M0_RUNNER_PATH),
            str(M0_ALGORITHM_PATH.relative_to(ROOT)): _sha256(M0_ALGORITHM_PATH),
            **{
                str(path.relative_to(ROOT)): _sha256(path)
                for path in TEST_PATHS
                if path.is_file()
            },
            **(
                {
                    str(PROTOCOL_DOCUMENT_PATH.relative_to(ROOT)): _sha256(
                        PROTOCOL_DOCUMENT_PATH
                    )
                }
                if PROTOCOL_DOCUMENT_PATH.is_file()
                else {}
            ),
        },
        "artifact_sha256": {name: _sha256(output / name) for name in artifact_names},
        "calibration_seed_registry_sha256": _canonical_hash(CALIBRATION_SEEDS),
        "null_seed_registry_sha256": _canonical_hash(NULL_VALIDATION_SEEDS_BY_REGIME),
        "evaluation_seed_registry_sha256": _canonical_hash(EVALUATION_SEEDS),
        "holdout_opened": False,
        "covariance_identity_oracle_used": False,
        "current_round_input_used": False,
        "all_checks_pass": bool(decision["all_checks_pass"]),
        "decision": decision["decision"],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _plan() -> dict[str, Any]:
    validation = _validate_protocol()
    return {
        **validation,
        "mode": "static_only_no_scientific_tensor_computation",
        "required_command_environment": "PYTORCH_ENABLE_MPS_FALLBACK=0",
        "output": str(DEFAULT_OUTPUT),
        "candidates": list(CANDIDATES),
        "primary_contrast": "pt_eiv_vs_pt_raw",
        "hard_control": "identity_y",
        "run_status": "implementation_only_do_not_launch_before_direction_redesign_review",
        "null_validation": "36 independent seeds per noise regime",
        "power": PROTOCOL["power"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output) if args.run else _plan()
    if not args.run:
        result["requested_mode"] = "validate" if args.validate else "dry-run"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
