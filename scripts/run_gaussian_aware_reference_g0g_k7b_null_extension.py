#!/usr/bin/env python3
"""Extend only K7b's independent null audit with frozen thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gaussian_aware_reference_g0g_k7_rcig_screen as k7  # noqa: E402
from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k7b_rcig_confirmation as k7b,
)

CAMPAIGN_ID = "gaussian_aware_reference_g0g_k7b_null_extension_mps_v1"
PARENT = (
    ROOT
    / "results/ldp_gradient_far"
    / "gaussian_aware_reference_g0g_k7b_rcig_confirmation_mps_v2"
)
DEFAULT_OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
PROTOCOL_PATH = (
    ROOT / "output/analysis/Gaussian_Aware_G0g_K7b_Null_Extension_Protocol_PreRun.md"
)
TEST_PATH = ROOT / "tests/test_run_gaussian_aware_reference_g0g_k7b_null_extension.py"

POOL_ROUNDS_PER_SLOT = 25
REQUIRED_PER_SLOT = 20
POOL_BASES = {
    "homogeneous": 203013010001,
    "heteroscedastic": 203013020003,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json(PARENT / "manifest.json")
    mismatches: list[str] = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = PARENT / relative
        if not path.is_file() or k7._sha256(path) != expected:
            mismatches.append(f"artifact:{relative}")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or k7._sha256(path) != expected:
            mismatches.append(f"source:{relative}")
    if mismatches:
        raise RuntimeError(f"Frozen K7b parent hash mismatch: {mismatches}")
    decision = _read_json(PARENT / "decision.json")
    thresholds = _read_json(PARENT / "thresholds.json")
    if not decision["primary_pass"] or not decision["gaussian_specificity_pass"]:
        raise RuntimeError("K7b parent did not pass its scientific gates")
    return manifest, decision, thresholds


def _candidate_pool(regime: str) -> tuple[int, ...]:
    base = POOL_BASES[regime]
    return tuple(
        base + 53 * (round_index * 12 + slot)
        for round_index in range(POOL_ROUNDS_PER_SLOT)
        for slot in range(12)
    )


def _static_validation() -> dict[str, Any]:
    parent_seeds = set(k7b._all_seeds())
    pools = {regime: _candidate_pool(regime) for regime in k7.REGIMES}
    all_pool = pools["homogeneous"] + pools["heteroscedastic"]
    checks = {
        "parent_exists": PARENT.is_dir(),
        "pool_sizes_are_300": all(len(values) == 300 for values in pools.values()),
        "pool_seeds_unique": len(all_pool) == len(set(all_pool)),
        "pool_fresh_relative_to_parent": not bool(set(all_pool) & parent_seeds),
        "fixed_20_of_25_per_slot": REQUIRED_PER_SLOT == 20
        and POOL_ROUNDS_PER_SLOT == 25,
        "target_total_is_360_per_regime": 120 + 12 * REQUIRED_PER_SLOT == 360,
        "mps_fail_closed": k7b.PROTOCOL["execution"]
        == {
            "required_device": "mps",
            "dtype": "float32",
            "pytorch_mps_fallback_required": "0",
            "overwrite": False,
        },
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"Invalid null-extension protocol: {failed}")
    return {"checks": checks, "candidate_pool_sha256": k7._canonical_hash(pools)}


def _configure_with_pools() -> dict[str, tuple[int, ...]]:
    pools = {regime: _candidate_pool(regime) for regime in k7.REGIMES}
    k7.CAMPAIGN_ID = CAMPAIGN_ID
    k7.PROTOCOL = k7b.PROTOCOL
    k7.NULL_VALIDATION_SEEDS_BY_REGIME = pools
    return pools


def _select_geometry_valid(
    pools: dict[str, tuple[int, ...]], device: Any
) -> tuple[dict[str, tuple[int, ...]], list[dict[str, Any]]]:
    selected: dict[str, tuple[int, ...]] = {}
    exclusions: list[dict[str, Any]] = []
    for regime, pool in pools.items():
        accepted_by_slot: dict[int, list[int]] = {slot: [] for slot in range(12)}
        for index, seed in enumerate(pool):
            slot = index % 12
            if len(accepted_by_slot[slot]) >= REQUIRED_PER_SLOT:
                continue
            try:
                k7._base_context(
                    seed,
                    regime,
                    "none",
                    phase="null_validation",
                    device=device,
                )
            except RuntimeError as exc:
                exclusions.append(
                    {
                        "noise_regime": regime,
                        "seed": seed,
                        "slot": slot,
                        "reason": str(exc),
                    }
                )
                continue
            accepted_by_slot[slot].append(seed)
        if any(
            len(values) != REQUIRED_PER_SLOT for values in accepted_by_slot.values()
        ):
            raise RuntimeError(f"Insufficient geometry-valid seeds for {regime}")
        selected[regime] = tuple(
            accepted_by_slot[slot][rank]
            for rank in range(REQUIRED_PER_SLOT)
            for slot in range(12)
        )
    return selected, exclusions


def _new_rows(
    selected: dict[str, tuple[int, ...]],
    thresholds: dict[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    k7.NULL_VALIDATION_SEEDS_BY_REGIME = selected
    rows: list[dict[str, Any]] = []
    for regime, seeds in selected.items():
        for seed in seeds:
            base = k7._base_context(
                seed,
                regime,
                "none",
                phase="null_validation",
                device=device,
            )
            stats = k7._innovation_statistics(base)
            rows.append(
                {
                    "seed": seed,
                    "noise_regime": regime,
                    "device": str(base["older"].device),
                    "dtype": str(base["older"].dtype),
                    "full_innovation": stats["full"],
                    "isotropic_innovation": stats["isotropic"],
                    "euclidean_innovation": stats["euclidean"],
                    "full_gate_active": stats["full"]
                    > float(thresholds[regime]["full"]),
                    "isotropic_gate_active": stats["isotropic"]
                    > float(thresholds[regime]["isotropic"]),
                    "euclidean_gate_active": stats["euclidean"]
                    > float(thresholds[regime]["euclidean"]),
                    "thresholds_frozen_from_parent": True,
                    "target_or_MSE_read": False,
                }
            )
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _decision(
    previous_rows: list[dict[str, str]],
    extension_rows: list[dict[str, Any]],
    parent_decision: dict[str, Any],
) -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for regime in k7.REGIMES:
        previous = [row for row in previous_rows if row["noise_regime"] == regime]
        extension = [row for row in extension_rows if row["noise_regime"] == regime]
        for mode, field in (
            ("full", "full_gate_active"),
            ("isotropic", "isotropic_gate_active"),
            ("euclidean", "euclidean_gate_active"),
        ):
            old_count = sum(_as_bool(row[field]) for row in previous)
            new_count = sum(_as_bool(row[field]) for row in extension)
            trials = len(previous) + len(extension)
            upper = k7._clopper_pearson_upper(
                old_count + new_count,
                trials,
                one_sided_alpha=0.025,
            )
            audits[f"{regime}_{mode}"] = {
                "parent_activations": old_count,
                "extension_activations": new_count,
                "combined_activations": old_count + new_count,
                "combined_trials": trials,
                "combined_rate": (old_count + new_count) / trials,
                "CP97_5_upper": upper,
                "pass": upper <= 0.10,
            }
    validity_checks = {
        "all_combined_CP_checks_pass": all(
            bool(value["pass"]) for value in audits.values()
        ),
        "exactly_360_trials_per_regime_and_mode": all(
            int(value["combined_trials"]) == 360 for value in audits.values()
        ),
        "all_extension_rows_mps_float32": all(
            row["device"] == "mps:0" and row["dtype"] == "torch.float32"
            for row in extension_rows
        ),
        "no_target_or_MSE_read": all(
            not bool(row["target_or_MSE_read"]) for row in extension_rows
        ),
        "parent_science_gates_frozen_and_passed": bool(
            parent_decision["primary_pass"]
            and parent_decision["gaussian_specificity_pass"]
        ),
    }
    passed = all(validity_checks.values())
    return {
        "verdict": (
            "advance_to_locked_real_gradient_mechanistic_screen"
            if passed
            else "stop_null_tolerance_not_certified"
        ),
        "validity_checks": validity_checks,
        "combined_null_audits": audits,
        "all_checks_pass": passed,
        "parent_evaluation_recomputed": False,
        "parent_thresholds_changed": False,
        "claim_limit": (
            "synthetic onset-of-attack mechanism only; no end-to-end, fairness, "
            "or universal Byzantine robustness claim"
        ),
    }


def _run(output: Path) -> dict[str, Any]:
    validation = _static_validation()
    parent_manifest, parent_decision, thresholds = _verify_parent()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must equal 0")
    device = k7._require_mps()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    pools = _configure_with_pools()
    selected, exclusions = _select_geometry_valid(pools, device)
    extension_rows = _new_rows(selected, thresholds, device)
    previous_rows = _read_csv(PARENT / "null_validation_rows.csv")
    decision = _decision(previous_rows, extension_rows, parent_decision)
    output.mkdir(parents=True)
    k7._write_json(output / "static_validation.json", validation)
    k7._write_json(output / "selected_seeds.json", selected)
    k7._write_json(output / "geometry_exclusions.json", exclusions)
    k7._write_csv(output / "null_extension_rows.csv", extension_rows)
    k7._write_json(output / "decision.json", decision)
    artifacts = (
        "static_validation.json",
        "selected_seeds.json",
        "geometry_exclusions.json",
        "null_extension_rows.csv",
        "decision.json",
    )
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "device": str(device),
        "dtype": "torch.float32",
        "mps_fallback_disabled": True,
        "parent_campaign_id": parent_manifest["campaign_id"],
        "parent_manifest_sha256": k7._sha256(PARENT / "manifest.json"),
        "parent_artifacts_verified_before_read": True,
        "thresholds_frozen": thresholds,
        "verdict": decision["verdict"],
        "all_checks_pass": decision["all_checks_pass"],
        "artifact_sha256": {name: k7._sha256(output / name) for name in artifacts},
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): k7._sha256(
                Path(__file__).resolve()
            ),
            str(PROTOCOL_PATH.relative_to(ROOT)): k7._sha256(PROTOCOL_PATH),
            str(TEST_PATH.relative_to(ROOT)): k7._sha256(TEST_PATH),
        },
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
    if args.validate:
        print(
            json.dumps(
                {
                    **_static_validation(),
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
