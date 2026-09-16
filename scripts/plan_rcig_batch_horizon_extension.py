#!/usr/bin/env python3
"""Print an independent RCIG extension plan; NEVER launch training or write files.

Default/``--plan`` prints the declarative design and task counts. ``--sigma-table``
performs only dependency-free privacy arithmetic, using the existing accountant.
No live campaign state or result is read. Deferred prerequisites are declarations,
not verified readiness. The recent-reference control still requires implementation.
"""

from __future__ import annotations

import sys

# Disable bytecode writes before loading any optional dependency or local module.
sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/ldp_gradient_far/rcig_batch_horizon_extension_v1.yaml"
SCHEMA = "rcig_batch_horizon_extension_plan/v1"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read and validate the fixed plan, without importing a campaign launcher."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Fail closed on changes that invalidate the predeclared design/counts."""
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA:
        raise ValueError("Expected a declarative extension plan, not a launch config")
    if config.get("training_executable") is not False:
        raise ValueError("This plan must remain non-executable")
    if config.get("artifact_kind") != "declarative_non_executable_plan":
        raise ValueError("The input must remain a declarative plan artifact")
    trigger = config["deferred_trigger"]
    if (
        trigger["predecessor_campaign"] != "rcig_r2_r3_exploratory_v1"
        or trigger["predecessor_result_root"]
        != "results/ldp_gradient_far/rcig_r2_r3_exploratory_v1"
        or trigger["predecessor_phase"] != "r3_e2e_confirmation"
        or trigger["required_valid_completed_runs"] != 384
        or trigger["required_active_campaign_processes"] != 0
    ):
        raise ValueError("The trigger must target the actual exploratory R3 campaign")
    scope, grid = config["scientific_scope"], config["grid"]
    if (scope["num_clients"], scope["fixed_local_dataset_size"]) != (25, 2400):
        raise ValueError("This extension requires n=25 and N=2400")
    fixed_scope = {
        "dataset": "fashionmnist",
        "model": "lenet5_tanh",
        "local_clip_norm": 4.0,
        "server_clip_norm": 16.0,
        "server_learning_rate": 0.2,
        "one_fixed_wor_step_and_release_per_client_round": True,
    }
    if any(scope.get(key) != value for key, value in fixed_scope.items()):
        raise ValueError("The fixed model/training scope cannot change silently")
    if grid["batch_sizes"] != [120, 240, 480] or grid["horizons"] != [40, 115, 268]:
        raise ValueError("The declared batch/horizon grid is fixed")
    expected_scales = {
        "homogeneous": [1] * 25,
        "heteroscedastic": [1 if i % 2 == 0 else 2 for i in range(25)],
    }
    if set(grid["noise_regimes"]) != set(expected_scales):
        raise ValueError("Exactly two declared noise regimes are required")
    for name, scales in expected_scales.items():
        if grid["noise_regimes"][name]["client_scales"] != scales:
            raise ValueError(f"Incorrect client scale vector for {name}")
    privacy = config["privacy"]
    if (
        privacy["adjacency"] != "replace_one"
        or privacy["sampling_scheme"] != "fixed_without_replacement"
        or privacy["sensitivity_multiplier"] != 2.0
        or privacy["private_releases_per_client_round"] != 1
        or privacy["target_epsilon"] != 4.0
        or privacy["delta"] != 1e-5
        or privacy["epsilon_tolerance"] != 1e-4
    ):
        raise ValueError("Privacy contract differs from the predeclared extension")
    references = {item["id"]: item for item in config["references"]}
    if set(references) != {"rcig_full", "recent_reference", "uniform", "fcc", "rfa"}:
        raise ValueError("Exactly the five paired references are required")
    for name, reference in references.items():
        if reference["far_alpha"] != (0.0 if name == "uniform" else 0.1):
            raise ValueError("All references except uniform must use alpha=0.1")
    for name in ("rcig_full", "recent_reference"):
        if references[name]["uniform_warmup_rounds"] != 12:
            raise ValueError("Temporal references require the same 12-round warmup")
    recent = references["recent_reference"]
    if (
        recent.get("implementation_required") is not True
        or recent.get("deployable") is not False
        or recent.get("proposed_reference_control") != "identity_new"
        or recent.get("robust_reference") is not None
    ):
        raise ValueError("identity_new is not deployable and must be flagged")
    contract = config["algorithm_contract"]
    fixed_contract = {
        "score_mode": "raw_distance",
        "tilt_bound_policy": "diagnostic",
        "far_alpha_except_uniform": 0.1,
        "rcig_covariance_mode": "full",
        "temporal_reference_warmup_rounds": 12,
        "temporal_reference_warmup_policy": "uniform",
    }
    if any(contract.get(key) != value for key, value in fixed_contract.items()):
        raise ValueError("The deployed algorithm contract must remain fixed")
    expected_attacks = [
        {"id": "bf_x10", "name": "bf", "scale": 10.0},
        {"id": "ipm", "name": "ipm", "scale": 1.0},
        {"id": "alie", "name": "alie", "scale": 1.0},
    ]
    expected_schedules = [
        {"id": "persistent", "first_round": 17, "last_round": "T"},
        {"id": "recovery", "first_round": 17, "last_round": "floor(0.6*T)"},
    ]
    if (
        config["scenarios"]["include_no_attack"] is not True
        or config["scenarios"]["attacks"] != expected_attacks
        or config["scenarios"]["schedules"] != expected_schedules
    ):
        raise ValueError("The seven declared attack scenarios must remain fixed")
    threshold = config["threshold_rule"]
    if (
        threshold["calibrated_mode"] != "deployed_full_only"
        or threshold["import_predecessor_r0_thresholds"] is not False
        or threshold["number_of_cells"] != 18
        or threshold["calibration_reference"] != "identity_new_no_activation"
        or threshold["first_activation_coupling_required"] is not True
        or threshold["post_activation_closed_loop_trajectory_equivalence_claimed"]
        is not False
    ):
        raise ValueError(
            "Fresh full-mode no-activation calibration/coupling is required"
        )
    seeds = seed_registries(config, 12)
    expected_seed_ranges = {
        "calibration": list(range(950001, 950064)),
        "null_validation": list(range(960001, 960037)),
        "evaluation": list(range(970001, 970013)),
    }
    if seeds != expected_seed_ranges:
        raise ValueError("Fresh, disjoint seed blocks must match the declared ranges")
    if config["randomness"]["evaluation"]["default_count"] != 4:
        raise ValueError("The user selected exactly four evaluation seed blocks")
    if 1 - 18 * threshold["population_null_coverage"] ** 63 < 0.975:
        raise ValueError("The simultaneous calibration confidence is insufficient")


def seed_registries(config: dict, evaluation_seeds: int = 4) -> dict[str, list[int]]:
    if evaluation_seeds not in (4, 12):
        raise ValueError("evaluation_seeds must be 4 or 12")
    registries = {}
    for name, registry in config["randomness"].items():
        if name not in ("calibration", "null_validation", "evaluation"):
            continue
        count = evaluation_seeds if name == "evaluation" else registry["count"]
        registries[name] = list(
            range(registry["first_seed"], registry["first_seed"] + count)
        )
    return registries


def grid_cells(config: dict) -> list[dict]:
    grid = config["grid"]
    return [
        {
            "cell_id": f"b{batch}_t{horizon}_{regime}",
            "batch_size": batch,
            "horizon": horizon,
            "noise_regime": regime,
            "sampling_rate": batch
            / config["scientific_scope"]["fixed_local_dataset_size"],
            "post_warmup_rounds": [13, horizon],
        }
        for batch, horizon, regime in itertools.product(
            grid["batch_sizes"], grid["horizons"], grid["noise_regimes"]
        )
    ]


def scenarios(config: dict, horizon: int) -> list[dict]:
    """Expand inclusive persistent/recovery intervals for one horizon."""
    expanded = [{"id": "no_attack", "name": "none", "enabled": False}]
    for attack, schedule in itertools.product(
        config["scenarios"]["attacks"], config["scenarios"]["schedules"]
    ):
        last = horizon if schedule["id"] == "persistent" else 3 * horizon // 5
        expanded.append(
            {
                "id": f"{attack['id']}_{schedule['id']}",
                "name": attack["name"],
                "scale": attack["scale"],
                "enabled": True,
                "first_round": schedule["first_round"],
                "last_round": last,
                "attack_client_ids": config["scientific_scope"]["attack_client_ids"],
            }
        )
    return expanded


def iter_planned_runs(config: dict, evaluation_seeds: int = 4) -> Iterator[dict]:
    """Enumerate labels only; no resolved launch configs or executable commands."""
    registries = seed_registries(config, evaluation_seeds)
    for phase in config["phases"]:
        evaluation = phase["id"] == "exploratory_comparison"
        references = (
            [r["id"] for r in config["references"]]
            if evaluation
            else phase["references"]
        )
        for cell in grid_cells(config):
            scenario_ids = (
                [s["id"] for s in scenarios(config, cell["horizon"])]
                if evaluation
                else ["no_attack"]
            )
            for seed, reference, scenario in itertools.product(
                registries[phase["seed_registry"]], references, scenario_ids
            ):
                yield {
                    "run_id": f"{phase['id']}/{cell['cell_id']}/s{seed}/{reference}/{scenario}",
                    "phase": phase["id"],
                    **cell,
                    "seed": seed,
                    "reference": reference,
                    "scenario": scenario,
                    "pairing_block": f"{phase['id']}/{cell['cell_id']}/s{seed}/{scenario}",
                    "training_executable": False,
                }


def build_plan(config: dict, evaluation_seeds: int = 4) -> dict:
    validate_config(config)
    registries = seed_registries(config, evaluation_seeds)
    counts: dict[str, int] = {}
    for run in iter_planned_runs(config, evaluation_seeds):
        counts[run["phase"]] = counts.get(run["phase"], 0) + 1
    expected = config["expected_task_counts"]
    if (
        counts["fresh_calibration"] != expected["fresh_calibration"]
        or counts["independent_null_validation"]
        != expected["independent_null_validation"]
        or counts["exploratory_comparison"]
        != expected[f"exploratory_comparison_{evaluation_seeds}_seeds"]
        or sum(counts.values()) != expected[f"total_{evaluation_seeds}_seeds"]
    ):
        raise ValueError("Expanded counts disagree with the declarative plan")
    return {
        "schema_version": SCHEMA,
        "artifact_kind": "read_only_plan_report",
        "training_executable": False,
        "readiness": "deferred_unverified_and_implementation_required",
        "evaluation_seed_count": evaluation_seeds,
        "evaluation_seed_choice": (
            "user_selected_exploratory_4"
            if evaluation_seeds == 4
            else "optional_12_requires_user_choice"
        ),
        "phase_counts": counts,
        "total_planned_runs": sum(counts.values()),
        "seed_registries": registries,
        "grid_cells": grid_cells(config),
        "scenarios_by_horizon": {
            str(t): scenarios(config, t) for t in config["grid"]["horizons"]
        },
        "simultaneous_calibration_confidence_lower": 1 - 18 * 0.9**63,
        "sigma_table_calculated": False,
        "sigma_table_instruction": "Use --sigma-table for read-only accountant recomputation.",
        "design": copy.deepcopy(config),
    }


@lru_cache(maxsize=1)
def _rdp_module():
    """Load the existing rdp.py only; privacy/__init__.py imports training code."""
    path = ROOT / "privacy/rdp.py"
    name = "_rcig_extension_existing_rdp"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Existing privacy accountant could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _epsilon(config: dict, batch: int, horizon: int, sigma: float) -> tuple[float, int]:
    privacy = config["privacy"]
    accountant = _rdp_module().RDPAccountant(orders=tuple(privacy["orders"]))
    accountant.add_sampled_without_replacement_gaussian(
        channel="model",
        sampling_rate=batch / config["scientific_scope"]["fixed_local_dataset_size"],
        noise_multiplier=sigma / privacy["sensitivity_multiplier"],
        steps=horizon * privacy["private_releases_per_client_round"],
    )
    return accountant.epsilon(privacy["delta"])


def build_sigma_table(config: dict) -> dict:
    """Calibrate each (B,T), then independently recompute both client budgets."""
    validate_config(config)
    privacy = config["privacy"]
    rows = []
    for batch, horizon in itertools.product(
        config["grid"]["batch_sizes"], config["grid"]["horizons"]
    ):
        q = batch / config["scientific_scope"]["fixed_local_dataset_size"]
        sigma = _rdp_module().calibrate_sampled_without_replacement_gaussian_noise(
            target_epsilon=privacy["target_epsilon"],
            delta=privacy["delta"],
            sampling_rate=q,
            steps=horizon,
            sensitivity_multiplier=privacy["sensitivity_multiplier"],
            orders=tuple(privacy["orders"]),
            tolerance=privacy["epsilon_tolerance"],
        )
        recomputed = {
            scale: _epsilon(config, batch, horizon, sigma * scale) for scale in (1, 2)
        }
        for regime, regime_config in config["grid"]["noise_regimes"].items():
            scales = regime_config["client_scales"]
            epsilons = [recomputed[scale][0] for scale in scales]
            sds = [
                config["scientific_scope"]["local_clip_norm"] * sigma * scale / batch
                for scale in scales
            ]
            error = abs(max(epsilons) - privacy["target_epsilon"])
            valid = (
                all(math.isfinite(value) for value in [sigma, *epsilons, *sds])
                and error <= privacy["epsilon_tolerance"]
            )
            if not valid:
                raise ValueError(
                    f"Recomputed privacy budget failed for B={batch}, T={horizon}, {regime}"
                )
            rows.append(
                {
                    "cell_id": f"b{batch}_t{horizon}_{regime}",
                    "batch_size": batch,
                    "horizon": horizon,
                    "noise_regime": regime,
                    "sampling_rate": q,
                    "private_steps": horizon,
                    "sigma": sigma,
                    "client_scales": scales,
                    "released_average_noise_sd_at_scale_1": sds[0],
                    "released_average_noise_sd_by_client": sds,
                    "accountant_noise_multiplier_by_client": [
                        sigma * scale / 2 for scale in scales
                    ],
                    "epsilon_by_client": epsilons,
                    "optimal_order_by_client": [
                        recomputed[scale][1] for scale in scales
                    ],
                    "minimum_client_epsilon": min(epsilons),
                    "maximum_client_epsilon": max(epsilons),
                    "max_epsilon_absolute_error": error,
                    "maximum_epsilon_within_tolerance": valid,
                    "epsilon_tolerance": privacy["epsilon_tolerance"],
                    "target_epsilon": privacy["target_epsilon"],
                    "delta": privacy["delta"],
                    "accountant_orders": privacy["orders"],
                }
            )
    return {
        "schema_version": SCHEMA,
        "artifact_kind": "read_only_privacy_arithmetic",
        "training_executable": False,
        "accountant_source": "privacy/rdp.py",
        "accountant_source_sha256": hashlib.sha256(
            (ROOT / "privacy/rdp.py").read_bytes()
        ).hexdigest(),
        "privacy_interpretation": privacy["interpretation"],
        "base_calibration_count": 9,
        "noise_regime_cell_count": 18,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print the plan (default)")
    mode.add_argument(
        "--sigma-table", action="store_true", help="Compute privacy arithmetic only"
    )
    parser.add_argument("--evaluation-seeds", type=int, choices=(4, 12), default=4)
    args = parser.parse_args(argv)
    config = load_config()
    report = (
        build_sigma_table(config)
        if args.sigma_table
        else build_plan(config, args.evaluation_seeds)
    )
    report["config_sha256"] = hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
