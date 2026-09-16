#!/usr/bin/env python3
"""Independent, higher-powered K7b confirmation of the RCIG mechanism."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gaussian_aware_reference_g0g_k7_rcig_screen as k7  # noqa: E402

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k7b_rcig_confirmation_mps_v2"
DEFAULT_OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
PROTOCOL_PATH = (
    ROOT / "output/analysis/Gaussian_Aware_G0g_K7b_RCIG_Confirmation_Protocol_PreRun.md"
)
ALGORITHM_PATH = ROOT / "algorithms/gaussian_aware_reference_k7_rcig.py"
K7_RUNNER_PATH = ROOT / "scripts/run_gaussian_aware_reference_g0g_k7_rcig_screen.py"
TEST_PATH = (
    ROOT / "tests/test_run_gaussian_aware_reference_g0g_k7b_rcig_confirmation.py"
)
EXPECTED_K7_RUNNER_SHA256 = (
    "503f1fdf7fcf746eec5421b3557cf82241c4c66709b7890594110f56e77dd5b5"
)
EXPECTED_ALGORITHM_SHA256 = (
    "e8ad77c5c1ce19cfbcdd3908015a854d488457b7dbc473b48d9dbc4a1f76e6bc"
)

_CALIBRATION_REPLACEMENTS = {
    # Deterministic replacements after the MPS-only geometry preflight.  The
    # rejected seeds were never scored and produced no scientific artifact.
    60: 203012010001 + 41 * 96,
    63: 203012010001 + 41 * 97,
}
CALIBRATION_SEEDS = tuple(
    _CALIBRATION_REPLACEMENTS.get(index, 203012010001 + 41 * index)
    for index in range(96)
)
NULL_VALIDATION_SEEDS_BY_REGIME = {
    "homogeneous": tuple(203012020003 + 41 * index for index in range(120)),
    "heteroscedastic": tuple(203012030007 + 41 * index for index in range(120)),
}
EVALUATION_SEEDS = tuple(203012040009 + 41 * index for index in range(48))

PROTOCOL = copy.deepcopy(k7.PROTOCOL)
PROTOCOL.update(
    {
        "campaign_id": CAMPAIGN_ID,
        "scope": "locked_independent_higher_powered_synthetic_confirmation",
        "parent_screen": {
            "campaign_id": k7.CAMPAIGN_ID,
            "status": "invalid_null_precision_but_primary_signal_positive",
            "scientific_results_reused": False,
        },
    }
)
PROTOCOL["innovation"].update(
    {
        "threshold_rule": "q0.99_no_attack_calibration_separate_by_regime_and_metric",
        "threshold_quantile": 0.99,
    }
)
PROTOCOL["randomness"] = {
    "calibration_outer_seeds": 96,
    "null_validation_outer_seeds_per_regime": 120,
    "evaluation_outer_seeds": 48,
    "all_registries_disjoint": True,
    "all_seeds_fresh_relative_to_K7": True,
    "candidates_and_threat_counterfactuals_paired": True,
    "statistical_unit": "outer_seed",
}
PROTOCOL["counterbalancing"].update(
    {
        "each_client_attacked_eight_times_across_48_evaluation_seeds": True,
        "attacker_tier_slots_per_tier": 24,
    }
)
PROTOCOL["gaussian_specificity_gates"] = {
    "heteroscedastic_gain_vs_isotropic_one_sided_ci_low_min": 0.0,
    "heteroscedastic_gain_vs_euclidean_one_sided_ci_low_min": -0.02,
    "euclidean_status": "noninferiority_control_not_primary_orientation_contrast",
}


def _all_seeds() -> tuple[int, ...]:
    return (
        CALIBRATION_SEEDS
        + NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
        + EVALUATION_SEEDS
    )


def _configure_k7_dependency() -> None:
    """Inject only the locked K7b registry into the frozen K7 engine."""

    k7.CAMPAIGN_ID = CAMPAIGN_ID
    k7.CALIBRATION_SEEDS = CALIBRATION_SEEDS
    k7.NULL_VALIDATION_SEEDS_BY_REGIME = NULL_VALIDATION_SEEDS_BY_REGIME
    k7.EVALUATION_SEEDS = EVALUATION_SEEDS
    k7.PROTOCOL = PROTOCOL


def _validate_protocol() -> dict[str, Any]:
    seeds = _all_seeds()
    parent_seeds = set(
        k7.CALIBRATION_SEEDS
        + k7.NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"]
        + k7.NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"]
        + k7.EVALUATION_SEEDS
    )
    # The module has not yet been configured here, so parent_seeds are K7 seeds.
    identity_counts = [0] * 12
    tier_counts = [0] * 4
    for index, _seed in enumerate(EVALUATION_SEEDS):
        slot = index % 12
        attackers = [(10 + slot + offset) % 12 for offset in range(2)]
        for client in attackers:
            identity_counts[client] += 1
            tier_counts[(client + 5 * slot) % 4] += 1
    checks = {
        "dependencies_frozen": k7._sha256(K7_RUNNER_PATH) == EXPECTED_K7_RUNNER_SHA256
        and k7._sha256(ALGORITHM_PATH) == EXPECTED_ALGORITHM_SHA256,
        "all_k7b_seeds_unique": len(seeds) == len(set(seeds)),
        "all_k7b_seeds_fresh": not bool(set(seeds) & parent_seeds),
        "registered_counts": (
            len(CALIBRATION_SEEDS) == 96
            and all(
                len(values) == 120
                for values in NULL_VALIDATION_SEEDS_BY_REGIME.values()
            )
            and len(EVALUATION_SEEDS) == 48
        ),
        "identity_counterbalance_exact": identity_counts == [8] * 12,
        "noise_tier_counterbalance_exact": tier_counts == [24] * 4,
        "identity_y_remains_primary_control": k7.CANDIDATES[0] == "identity_y",
        "no_oracle_covariance": (
            PROTOCOL["covariance_policy"]["retained_for_attacked_identities"]
            and not PROTOCOL["covariance_policy"][
                "byzantine_mask_used_to_modify_covariance"
            ]
        ),
        "mps_float32_fail_closed": PROTOCOL["execution"]
        == {
            "required_device": "mps",
            "dtype": "float32",
            "pytorch_mps_fallback_required": "0",
            "overwrite": False,
        },
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"Invalid K7b protocol: {failed}")
    return {
        "campaign_id": CAMPAIGN_ID,
        "checks": checks,
        "protocol_sha256": k7._canonical_hash(PROTOCOL),
        "expected_calibration_rows": 192,
        "expected_null_validation_rows": 240,
        "expected_evaluation_rows": 288,
    }


def _student_interval(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if len(numbers) != 48:
        raise RuntimeError("Expected 48 independent K7b outer-seed values")
    mean = statistics.fmean(numbers)
    standard_error = statistics.stdev(numbers) / math.sqrt(len(numbers))
    return {
        "n": len(numbers),
        "mean": mean,
        "two_sided_low": mean - 2.0117405137297655 * standard_error,
        "two_sided_high": mean + 2.0117405137297655 * standard_error,
        "one_sided_low": mean - 1.6779267216418608 * standard_error,
        "one_sided_high": mean + 1.6779267216418608 * standard_error,
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
            raise RuntimeError("Incomplete paired K7b seed contrast")
        baseline_total = sum(float(row[f"{baseline}_mse"]) for row in cell)
        candidate_total = sum(float(row[f"{candidate}_mse"]) for row in cell)
        values.append((baseline_total - candidate_total) / max(baseline_total, 1.0e-15))
    return _student_interval(values)


def _decision(
    null_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    null_checks: dict[str, Any] = {}
    for regime in k7.REGIMES:
        cell = [row for row in null_rows if row["noise_regime"] == regime]
        for mode, field in (
            ("full", "full_gate_active"),
            ("isotropic", "isotropic_gate_active"),
            ("euclidean", "euclidean_gate_active"),
        ):
            activations = sum(bool(row[field]) for row in cell)
            upper = k7._clopper_pearson_upper(
                activations, len(cell), one_sided_alpha=0.025
            )
            null_checks[f"{regime}_{mode}"] = {
                "activations": activations,
                "trials": len(cell),
                "rate": activations / len(cell),
                "CP97_5_upper": upper,
                "pass": upper <= 0.10,
            }

    primary_contrasts: dict[str, Any] = {}
    primary_checks: dict[str, bool] = {}
    for regime in k7.REGIMES:
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

    full_vs_iso = _contrast(
        evaluation_rows,
        regime="heteroscedastic",
        candidate="rcig_full",
        baseline="rcig_isotropic",
        threats=("bitflip_x10", "model_replacement"),
    )
    full_vs_euclidean = _contrast(
        evaluation_rows,
        regime="heteroscedastic",
        candidate="rcig_full",
        baseline="euclidean_gate",
        threats=("bitflip_x10", "model_replacement"),
    )
    specificity_checks = {
        "full_vs_isotropic_one_sided_low_positive": full_vs_iso["one_sided_low"] >= 0.0,
        "full_vs_euclidean_noninferiority_at_2pct": full_vs_euclidean["one_sided_low"]
        >= -0.02,
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
            for candidate in k7.CANDIDATES
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
        verdict = "robust_temporal_gate_supported_gaussian_specificity_not_identified"
    else:
        verdict = "advance_to_real_gradient_mechanistic_screen"
    return {
        "verdict": verdict,
        "validity_checks": validity_checks,
        "null_false_activation": null_checks,
        "primary_contrasts": primary_contrasts,
        "primary_checks": primary_checks,
        "gaussian_specificity_contrasts": {
            "heteroscedastic_full_vs_isotropic": full_vs_iso,
            "heteroscedastic_full_vs_euclidean": full_vs_euclidean,
        },
        "gaussian_specificity_checks": specificity_checks,
        "validity_pass": validity_pass,
        "primary_pass": primary_pass,
        "gaussian_specificity_pass": specificity_pass,
        "claim_limit": (
            "synthetic onset-of-attack mechanism only; next authorized step is "
            "a locked real-gradient screen, not an end-to-end claim"
        ),
    }


def _run(output: Path) -> dict[str, Any]:
    validation = _validate_protocol()
    _configure_k7_dependency()
    device = k7._require_mps()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    thresholds, calibration_rows = k7._calibrate(device)
    null_rows = k7._null_rows(device, thresholds)
    evaluation_rows = k7._evaluation_rows(device, thresholds)
    summaries = k7._summaries(evaluation_rows)
    decision = _decision(null_rows, evaluation_rows)
    k7._write_json(output / "resolved_protocol.json", PROTOCOL)
    k7._write_json(output / "static_validation.json", validation)
    k7._write_json(output / "thresholds.json", thresholds)
    k7._write_csv(output / "calibration_rows.csv", calibration_rows)
    k7._write_csv(output / "null_validation_rows.csv", null_rows)
    k7._write_csv(output / "evaluation_rows.csv", evaluation_rows)
    k7._write_csv(output / "summary.csv", summaries)
    k7._write_json(output / "decision.json", decision)
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
        "artifact_sha256": {name: k7._sha256(output / name) for name in artifacts},
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): k7._sha256(
                Path(__file__).resolve()
            ),
            str(K7_RUNNER_PATH.relative_to(ROOT)): k7._sha256(K7_RUNNER_PATH),
            str(ALGORITHM_PATH.relative_to(ROOT)): k7._sha256(ALGORITHM_PATH),
            str(PROTOCOL_PATH.relative_to(ROOT)): k7._sha256(PROTOCOL_PATH),
            str(TEST_PATH.relative_to(ROOT)): k7._sha256(TEST_PATH),
        },
        "all_checks_pass": bool(
            decision["validity_pass"]
            and decision["primary_pass"]
            and decision["gaussian_specificity_pass"]
        ),
    }
    k7._write_json(output / "manifest.json", manifest)
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
                    "output": str(args.output.resolve()),
                    "required_environment": "PYTORCH_ENABLE_MPS_FALLBACK=0",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(_run(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
