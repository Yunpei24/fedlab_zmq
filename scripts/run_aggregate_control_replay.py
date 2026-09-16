#!/usr/bin/env python3
"""Source-locked MPS replay of aggregate controls on six clean trajectories.

Only ``headroom_fit`` is executable in this runner.  Radius calibration and
untouched evaluation are preregistered in the matrix but intentionally remain
closed until a separate audited launcher is created.  ``--resume`` skips only
fully validated runs; partial or failed directories stop the campaign.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import yaml

from scripts import run_ldp_aggregation_role_ablation as historical
from scripts import run_rcig_batch_screen as shared
from scripts.run_rcig_ldp_gradient_far_v2 import (
    _metrics_path,
    _post_run_device_audit,
)


MATRIX = ROOT / "configs/ldp_gradient_far/aggregate_control_preregistered_v1.yaml"
CAMPAIGN = "aggregate_control_preregistered_v1"
LOG = ROOT / "logs/aggregate_control_preregistered_v1_headroom_fit_mps.log"
hash_file = shared.file_hash
write_json = shared.write_json


def load_matrix(path: Path = MATRIX) -> dict[str, Any]:
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError("aggregate-control matrix must be a mapping")
    expected = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN,
        "scientific_status": "exploratory_mechanistic_replay_not_confirmatory",
        "required_device": "mps",
        "mps_fallback": 0,
    }
    if any(matrix.get(key) != value for key, value in expected.items()):
        raise ValueError("aggregate-control campaign identity changed")
    contract = matrix["fixed_training_contract"]
    fixed = {
        "dataset": "fashionmnist",
        "model": "lenet5_tanh",
        "num_clients": 10,
        "rounds": 40,
        "public_local_size": 6000,
        "batch_size": 120,
        "local_steps_per_round": 1,
        "sampling_scheme": "fixed_without_replacement",
        "privacy_adjacency": "replace_one",
        "target_epsilon": 4.0,
        "delta": 1e-5,
        "local_clip_norm": 4.0,
        "server_clip_norm": 16.0,
        "far_alpha": 0.1,
        "model_driver": "uniform_rounds_1_to_12_then_far_rfa_rounds_13_to_40",
    }
    if any(contract.get(key) != value for key, value in fixed.items()):
        raise ValueError("fixed replay training contract changed")
    if matrix["noise_regimes"] != {
        "homogeneous": [1] * 10,
        "heteroscedastic": [1, 2] * 5,
    }:
        raise ValueError("authenticated ten-client noise registry changed")
    registry = matrix["seed_registry"]
    expected_seeds = {
        "headroom_fit": [930101, 930102, 930103],
        "radius_calibration": [930201, 930202, 930203],
        "untouched_evaluation": [930301, 930302, 930303],
    }
    if any(registry.get(name) != seeds for name, seeds in expected_seeds.items()):
        raise ValueError("preregistered seed registry changed")
    all_seeds = [seed for seeds in expected_seeds.values() for seed in seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("fit/calibration/evaluation seeds overlap")
    if matrix["stages"]["headroom_fit"].get("expected_runs") != 6:
        raise ValueError("headroom fit must contain exactly six runs")
    if matrix["first_phase_execution"] != {
        **matrix["first_phase_execution"],
        "only_phase": "headroom_fit",
        "only_six_clean_runs": True,
        "never_open_radius_or_evaluation_seeds": True,
    }:
        raise ValueError("first-phase lock changed")
    controls = matrix["shadow_controls"]
    if controls["ema_current_mixes"] != [0.2]:
        raise ValueError("EMA beta=0.2 control changed")
    if controls["isotropic_radius"]["uncalibrated_diagnostic_multipliers"] != [
        0.5,
        1.0,
        2.0,
    ]:
        raise ValueError("diagnostic isotropic radii changed")
    if controls["mahalanobis_radius"][
        "uncalibrated_diagnostic_multipliers"
    ] != [0.5, 1.0, 2.0]:
        raise ValueError("diagnostic Mahalanobis radii changed")
    return matrix


def output_root(matrix: dict[str, Any]) -> Path:
    return (MATRIX.parent / matrix["output_root"]).resolve()


def headroom_tasks(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        {"phase": "headroom_fit", "noise": noise, "seed": seed, "scenario": "none"}
        for seed, noise in itertools.product(
            matrix["seed_registry"]["headroom_fit"], matrix["noise_regimes"]
        )
    ]
    if len(tasks) != 6:
        raise RuntimeError("headroom task construction changed")
    used = {task["seed"] for task in tasks}
    if used != set(matrix["seed_registry"]["headroom_fit"]):
        raise RuntimeError("headroom task seeds are incomplete")
    if used & (
        set(matrix["seed_registry"]["radius_calibration"])
        | set(matrix["seed_registry"]["untouched_evaluation"])
    ):
        raise RuntimeError("closed seeds entered the headroom runner")
    return tasks


def run_id(task: dict[str, Any]) -> str:
    return f"{task['noise']}__seed{task['seed']}__clean"


def run_directory(matrix: dict[str, Any], task: dict[str, Any]) -> Path:
    return output_root(matrix) / "headroom_fit" / run_id(task)


def _privacy(matrix: dict[str, Any]) -> dict[str, Any]:
    contract = matrix["fixed_training_contract"]
    rdp = shared.rdp_module()
    rate = contract["batch_size"] / contract["public_local_size"]
    sigma = rdp.calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=contract["target_epsilon"],
        delta=contract["delta"],
        sampling_rate=rate,
        steps=contract["rounds"],
        sensitivity_multiplier=2.0,
    )
    ledger = {}
    for scale in (1, 2):
        accountant = rdp.RDPAccountant()
        accountant.add_sampled_without_replacement_gaussian(
            channel="gradient",
            sampling_rate=rate,
            noise_multiplier=sigma * scale / 2.0,
            steps=contract["rounds"],
        )
        epsilon, order = accountant.epsilon(contract["delta"])
        ledger[str(scale)] = {
            "epsilon": epsilon,
            "order": order,
            "noise_multiplier": sigma * scale,
            "per_coordinate_std": (
                sigma
                * scale
                * contract["local_clip_norm"]
                / contract["batch_size"]
            ),
        }
    if abs(ledger["1"]["epsilon"] - 4.0) > 1e-4:
        raise RuntimeError("base privacy calibration missed epsilon=4")
    return {
        "sampling_rate": rate,
        "base_noise_multiplier": sigma,
        "per_public_scale": ledger,
    }


def _historical_threshold(matrix: dict[str, Any]) -> Path:
    return (
        MATRIX.parent
        / matrix["historical_shadow_dependency"]["threshold_artifact"]
    ).resolve()


def provenance(matrix: dict[str, Any]) -> dict[str, Any]:
    base = (MATRIX.parent / matrix["base_config"]).resolve()
    threshold = _historical_threshold(matrix)
    if hash_file(threshold) != matrix["historical_shadow_dependency"][
        "threshold_artifact_sha256"
    ]:
        raise RuntimeError("historical shadow threshold artifact changed")
    sources = shared.source_closure(
        (
            Path(__file__).resolve(),
            ROOT / "algorithms/aggregate_radial_control.py",
            ROOT / "algorithms/aggregate_control_replay.py",
            ROOT / "run_experiment.py",
        )
    )
    return {
        "campaign_id": CAMPAIGN,
        "phase": "headroom_fit",
        "matrix_sha256": hash_file(MATRIX),
        "base_config_sha256": hash_file(base),
        "historical_matrix_sha256": hash_file(historical.MATRIX),
        "historical_threshold_sha256": hash_file(threshold),
        "privacy": _privacy(matrix),
        "sources": sources,
        "expected_runs": 6,
        "device": "mps",
        "mps_fallback": 0,
        "closed_seed_sets": {
            "radius_calibration": matrix["seed_registry"]["radius_calibration"],
            "untouched_evaluation": matrix["seed_registry"][
                "untouched_evaluation"
            ],
        },
    }


def verify_provenance(matrix: dict[str, Any], stamp: dict[str, Any]) -> None:
    if hash_file(MATRIX) != stamp["matrix_sha256"]:
        raise RuntimeError("aggregate-control matrix changed")
    base = (MATRIX.parent / matrix["base_config"]).resolve()
    threshold = _historical_threshold(matrix)
    if hash_file(base) != stamp["base_config_sha256"]:
        raise RuntimeError("base config changed")
    if hash_file(historical.MATRIX) != stamp["historical_matrix_sha256"]:
        raise RuntimeError("historical config-construction matrix changed")
    if hash_file(threshold) != stamp["historical_threshold_sha256"]:
        raise RuntimeError("historical threshold changed")
    changed = [
        name
        for name, digest in stamp["sources"].items()
        if hash_file(ROOT / name) != digest
    ]
    if changed:
        raise RuntimeError(f"source drift after campaign lock: {changed}")


def resolved_config(
    matrix: dict[str, Any], task: dict[str, Any], stamp: dict[str, Any]
) -> dict[str, Any]:
    old_matrix = historical.matrix()
    historical_task = {
        "noise": task["noise"],
        "seed": task["seed"],
        "scenario": "none",
        "arm": "far_rfa",
    }
    historical_stamp = {"privacy": historical.legacy.privacy(old_matrix)}
    config = historical.config_for(
        old_matrix, historical_task, historical_stamp
    )
    destination = run_directory(matrix, task)
    config.update(seed=task["seed"], device="mps", output_dir=str(destination))
    config["data"]["partition_seed"] = task["seed"]
    algo = config["training"]["algo_config"]
    state = matrix["predictable_state"]
    controls = matrix["shadow_controls"]
    algo.update(
        aggregation_role_arm="far_rfa",
        aggregation_role_campaign=CAMPAIGN,
        rcig_campaign_id=CAMPAIGN,
        rcig_n10_phase="headroom_fit",
        rcig_n10_scenario="none",
        rcig_n10_pairing_seed=task["seed"],
        rcig_n10_runtime_implementation=(
            "algorithms.aggregate_control_replay.AggregateControlReplay"
        ),
        aggregate_control_replay_phase="headroom_fit",
        aggregate_control_model_driver="uniform_t1_t12_then_far_rfa",
        aggregate_control_campaign=CAMPAIGN,
        aggregate_control_scientific_status=(
            "exploratory_headroom_only_no_promotion"
        ),
        aggregate_control_matrix_sha256=stamp["matrix_sha256"],
        aggregate_control_predictor_rate=state["predictor_rate"],
        aggregate_control_covariance_rate=state["covariance_rate"],
        aggregate_control_initial_variance_mode=(
            "public_pre_server_clip_uniform_mean_noise_proxy"
        ),
        aggregate_control_variance_ridge=state["variance_ridge"],
        aggregate_control_min_history=state["min_history"],
        aggregate_control_ema_current_mixes=controls["ema_current_mixes"],
        aggregate_control_isotropic_radius_multipliers=controls[
            "isotropic_radius"
        ]["uncalibrated_diagnostic_multipliers"],
        aggregate_control_mahalanobis_radius_multipliers=controls[
            "mahalanobis_radius"
        ]["uncalibrated_diagnostic_multipliers"],
        aggregate_control_projection_tolerance=controls[
            "euclidean_ellipsoid_projection"
        ]["tolerance"],
        aggregate_control_projection_max_iterations=controls[
            "euclidean_ellipsoid_projection"
        ]["max_iterations"],
        aggregate_control_radius_status="uncalibrated_diagnostic_only",
        aggregate_control_oracle_never_selects_model=True,
    )
    algo["attack"] = {
        "enabled": False,
        "name": "none",
        "scale": 1.0,
        "num_byzantine": 0,
        "client_ids": [],
    }
    expected_scale = matrix["noise_regimes"][task["noise"]]
    if algo["privacy_noise_multiplier_scale_by_client"] != expected_scale:
        raise RuntimeError("resolved public noise registry mismatch")
    if algo["robust_reference"] != "rcig_temporal":
        raise RuntimeError("oracle-detachment shadow path changed")
    if algo["rcig_n10_arm"] != "midpoint":
        raise RuntimeError("historical shadow state must remain midpoint")
    if algo["fixed_steps_per_round"] != 1 or algo["local_epochs"] != 1:
        raise RuntimeError("replay must release one fixed-WOR batch gradient")
    return config


def _expected_shadow_labels(matrix: dict[str, Any]) -> list[str]:
    controls = matrix["shadow_controls"]

    def label(value):
        return format(float(value), ".8g").replace("-", "m").replace(".", "p")

    result = ["unchanged"]
    result += [f"ema_mix_{label(x)}" for x in controls["ema_current_mixes"]]
    result += [
        f"isotropic_mult_{label(x)}"
        for x in controls["isotropic_radius"][
            "uncalibrated_diagnostic_multipliers"
        ]
    ]
    for value in controls["mahalanobis_radius"][
        "uncalibrated_diagnostic_multipliers"
    ]:
        result.extend(
            [
                f"radial_mult_{label(value)}",
                f"euclidean_projection_mult_{label(value)}",
            ]
        )
    return result


def _load_trace(matrix: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_directory(matrix, task) / "simulator_randomness_private_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return sorted(rows, key=lambda row: (row["round"], row["client_id"]))


def _validate_trace(matrix: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _load_trace(matrix, task)
    expected = set(itertools.product(range(1, 41), range(10)))
    keys = {(row["round"], row["client_id"]) for row in rows}
    if len(rows) != 400 or keys != expected:
        raise RuntimeError("private replay trace is incomplete")
    return rows


def _validate_noise_pairing(
    matrix: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    other_noise = (
        "heteroscedastic" if task["noise"] == "homogeneous" else "homogeneous"
    )
    other = {**task, "noise": other_noise}
    other_status = run_directory(matrix, other) / "orchestration_status.json"
    if not other_status.exists():
        return {"paired_noise_run_available": False}
    status = json.loads(other_status.read_text())
    if status.get("status") != "completed":
        return {"paired_noise_run_available": False}
    current = _validate_trace(matrix, task)
    counterpart = _validate_trace(matrix, other)
    for left, right in zip(current, counterpart, strict=True):
        if left["permutations"] != right["permutations"]:
            raise RuntimeError("fixed-WOR draws are not paired across noise regimes")
        if left["standard_gaussians"] != right["standard_gaussians"]:
            raise RuntimeError("standard Gaussian draws are not paired across regimes")
    return {
        "paired_noise_run_available": True,
        "paired_batch_and_standard_gaussian_draws": True,
        "paired_records": len(current),
    }


def validate_run(
    matrix: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
    stamp: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    destination = run_directory(matrix, task)
    metrics_path = _metrics_path(destination)
    if metrics_path is None:
        raise RuntimeError(f"missing, incomplete or non-unique metrics: {destination}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds")
    summary = payload.get("summary")
    if not isinstance(rounds, list) or len(rounds) != 40 or not isinstance(summary, dict):
        raise RuntimeError("replay metrics must contain forty rounds")
    expected_summary = (10, 40, task["seed"], "lenet5_tanh", "fashionmnist")
    observed_summary = (
        summary.get("num_clients"),
        summary.get("num_rounds"),
        summary.get("seed"),
        summary.get("model"),
        summary.get("dataset"),
    )
    if observed_summary != expected_summary:
        raise RuntimeError("replay summary identity mismatch")
    for key, value in config["training"]["algo_config"].items():
        if payload["config"].get(key) != value:
            raise RuntimeError(f"resolved algorithm config mismatch: {key}")
    _post_run_device_audit(metrics_path, config)
    shadow_labels = _expected_shadow_labels(matrix)
    for expected_round, row in enumerate(rounds, 1):
        if row.get("round_num") != expected_round or row.get("num_alive_clients") != 10:
            raise RuntimeError("round chronology/cohort mismatch")
        expected_arm = "uniform" if expected_round <= 12 else "far_rfa"
        checks = {
            "aggregation_role_effective_arm": expected_arm,
            "aggregate_control_model_driver": "uniform_t1_t12_then_far_rfa",
            "aggregate_control_all_alternatives_are_shadow": True,
            "aggregate_control_oracle_used_for_deployment": False,
            "aggregate_control_state_updated_from_uncorrected_candidate": True,
            "aggregate_control_headroom_fit_eligible": expected_round >= 13,
            "aggregate_control_same_cohort_verified": True,
        }
        if any(row.get(key) != value for key, value in checks.items()):
            raise RuntimeError(f"aggregate-control boundary mismatch at round {expected_round}")
        base_error = row.get("aggregate_control_base_error_sq")
        if not isinstance(base_error, (int, float)) or not math.isfinite(base_error):
            raise RuntimeError("missing finite base aggregate error")
        ready = expected_round >= 2
        if bool(row.get("aggregate_control_predictor_ready")) != ready:
            raise RuntimeError("predictable state readiness mismatch")
        if ready:
            required = [
                "aggregate_control_predictor_error_sq",
                "aggregate_control_innovation_sq",
                "aggregate_control_u_dot_predictor_minus_target",
                "aggregate_control_oracle_gamma_euclidean",
                "aggregate_control_oracle_relative_headroom",
                "aggregate_control_oracle_gamma_mahalanobis",
                "aggregate_control_oracle_mahalanobis_relative_headroom",
                "aggregate_control_initial_variance_proxy",
            ]
            for key in required:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RuntimeError(f"missing finite replay statistic {key}")
            reconstructed = (
                row["aggregate_control_predictor_error_sq"]
                + 2 * row["aggregate_control_u_dot_predictor_minus_target"]
                + row["aggregate_control_innovation_sq"]
            )
            if abs(reconstructed - base_error) > 1e-9 * max(1.0, base_error):
                raise RuntimeError("aggregate interpolation identity violated")
            oracle_error = row.get("aggregate_control_oracle_error_sq")
            if not isinstance(oracle_error, (int, float)) or not math.isfinite(oracle_error):
                raise RuntimeError("missing finite oracle error")
            if oracle_error > base_error + 1e-9 * max(1.0, base_error):
                raise RuntimeError("oracle interpolation is worse than its endpoint")
            for label in shadow_labels:
                for suffix in ("error_sq", "mahalanobis_error_sq", "correction_sq"):
                    key = f"aggregate_control_shadow_{label}_{suffix}"
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise RuntimeError(f"missing finite shadow statistic {key}")
    epsilon = rounds[-1].get("privacy_epsilon_max")
    if not isinstance(epsilon, (int, float)) or abs(epsilon - 4.0) > 1e-4:
        raise RuntimeError("privacy budget mismatch")
    trace = _validate_trace(matrix, task)
    runtime = json.loads((destination / "runtime_imports.json").read_text())
    if runtime.get("stage") != "after_training" or runtime.get("mps_fallback") != 0:
        raise RuntimeError("runtime MPS manifest is incomplete")
    unlocked = [
        name
        for name, digest in runtime.get("source_sha256", {}).items()
        if stamp["sources"].get(name) != digest
    ]
    if unlocked:
        raise RuntimeError(f"runtime imported unlocked sources: {unlocked}")
    return metrics_path, payload, {
        "trace_records": len(trace),
        **_validate_noise_pairing(matrix, task),
    }


def _summary(matrix: dict[str, Any], completed: list[tuple[dict, dict]]) -> None:
    groups = {}
    labels = _expected_shadow_labels(matrix)
    for task, payload in completed:
        # Model driver and replay fit are evaluated only after the historical
        # uniform warmup.  Per-run temporal means precede cross-seed summaries.
        rows = payload["rounds"][12:]
        item = {
            "seed": task["seed"],
            "oracle_gamma_euclidean": statistics.mean(
                row["aggregate_control_oracle_gamma_euclidean"] for row in rows
            ),
            "oracle_relative_headroom": statistics.mean(
                row["aggregate_control_oracle_relative_headroom"] for row in rows
            ),
            "oracle_gamma_mahalanobis": statistics.mean(
                row["aggregate_control_oracle_gamma_mahalanobis"] for row in rows
            ),
            "oracle_mahalanobis_relative_headroom": statistics.mean(
                row["aggregate_control_oracle_mahalanobis_relative_headroom"]
                for row in rows
            ),
            "base_error_sq": statistics.mean(
                row["aggregate_control_base_error_sq"] for row in rows
            ),
            "predictor_error_sq": statistics.mean(
                row["aggregate_control_predictor_error_sq"] for row in rows
            ),
            "test_accuracy_final": payload["rounds"][-1]["test_accuracy"],
            "test_loss_final": payload["rounds"][-1]["test_loss"],
            "shadows": {},
        }
        for label in labels:
            item["shadows"][label] = {
                "error_sq": statistics.mean(
                    row[f"aggregate_control_shadow_{label}_error_sq"] for row in rows
                ),
                "correction_fraction": statistics.mean(
                    bool(row[f"aggregate_control_shadow_{label}_correction_applied"])
                    for row in rows
                ),
                "target_coverage_fraction": statistics.mean(
                    1.0
                    if row[f"aggregate_control_shadow_{label}_target_covered"] is True
                    else 0.0
                    for row in rows
                    if row[f"aggregate_control_shadow_{label}_target_covered"]
                    is not None
                )
                if any(
                    row[f"aggregate_control_shadow_{label}_target_covered"] is not None
                    for row in rows
                )
                else None,
            }
        groups.setdefault(task["noise"], []).append(item)
    write_json(
        output_root(matrix) / "headroom_progress.json",
        {
            "completed": len(completed),
            "expected": 6,
            "scientific_status": "exploratory_headroom_only_no_promotion",
            "fit_window": "rounds_13_to_40",
            "groups": groups,
            "forbidden_inference": "lower_aggregate_mse_alone_does_not_promote_a_method",
        },
    )


@contextmanager
def execution_lock(matrix: dict[str, Any]):
    root = output_root(matrix)
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / f".{CAMPAIGN}.execution.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another aggregate-control launcher is active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_no_duplicate_process() -> None:
    shared.check_no_other_training_process()
    lines = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    current_pid = os.getpid()
    for line in lines:
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) == current_pid:
            continue
        try:
            args = shlex.split(fields[2])
        except ValueError:
            continue
        if any(Path(arg).name == Path(__file__).name for arg in args) and (
            "--run" in args or "--worker" in args
        ):
            raise RuntimeError(f"another aggregate-control process is active: {line}")


def _ensure_campaign_lock(matrix: dict[str, Any], stamp: dict[str, Any]) -> None:
    root = output_root(matrix)
    lock = root / "campaign_lock.json"
    if lock.exists():
        observed = json.loads(lock.read_text())
        if observed != stamp:
            raise RuntimeError("aggregate-control campaign lock mismatch")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("refusing to adopt an unlocked output directory")
    root.mkdir(parents=True, exist_ok=True)
    write_json(lock, stamp, exclusive=True)


def campaign_status(matrix: dict[str, Any]) -> dict[str, Any]:
    status = {"completed": 0, "total": 6, "active": [], "failed": [], "missing": 0}
    for task in headroom_tasks(matrix):
        path = run_directory(matrix, task) / "orchestration_status.json"
        if not path.exists():
            status["missing"] += 1
            continue
        row = json.loads(path.read_text())
        state = row.get("status")
        if state == "completed":
            status["completed"] += 1
        elif state == "running":
            status["active"].append(run_id(task))
        else:
            status["failed"].append({"run": run_id(task), "status": row})
    status["radius_calibration_opened"] = False
    status["untouched_evaluation_opened"] = False
    return status


def execute(
    matrix: dict[str, Any], stamp: dict[str, Any], *, max_new: int | None = None
) -> None:
    completed: list[tuple[dict, dict]] = []
    new = 0
    for index, task in enumerate(headroom_tasks(matrix), 1):
        verify_provenance(matrix, stamp)
        config = resolved_config(matrix, task, stamp)
        destination = run_directory(matrix, task)
        status_path = destination / "orchestration_status.json"
        config_path = destination / "resolved_config.yaml"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("status") != "completed":
                raise RuntimeError(f"incomplete run requires inspection: {destination}")
            if yaml.safe_load(config_path.read_text()) != config:
                raise RuntimeError(f"resolved config drift: {destination}")
            metrics_path, payload, pairing = validate_run(
                matrix, task, config, stamp
            )
            if status.get("metrics_sha256") != hash_file(metrics_path):
                raise RuntimeError("completed metrics hash changed")
        else:
            if max_new is not None and new >= max_new:
                break
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"unlocked partial run directory: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            with config_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            base_status = {
                "campaign_id": CAMPAIGN,
                "phase": "headroom_fit",
                "run_id": run_id(task),
                "task": task,
                "status": "running",
                "device": "mps",
                "mps_fallback": 0,
                "config_sha256": hash_file(config_path),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(status_path, base_status)
            print(f"START {index}/6 {run_id(task)}", flush=True)
            try:
                with (destination / "training.log").open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-u",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--config",
                            str(config_path),
                            "--output",
                            str(destination),
                            "--device",
                            "mps",
                        ],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                    write_json(status_path, {**base_status, "pid": child.pid})
                    code = child.wait()
                if code != 0:
                    raise RuntimeError(
                        f"training exited {code}: {destination / 'training.log'}"
                    )
                verify_provenance(matrix, stamp)
                metrics_path, payload, pairing = validate_run(
                    matrix, task, config, stamp
                )
                write_json(
                    status_path,
                    {
                        **base_status,
                        "status": "completed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "metrics_sha256": hash_file(metrics_path),
                        "trace_sha256": hash_file(
                            destination / "simulator_randomness_private_audit.jsonl"
                        ),
                        "pairing": pairing,
                    },
                )
                new += 1
                print(
                    f"DONE {index}/6 test_acc={payload['rounds'][-1]['test_accuracy']:.6f}",
                    flush=True,
                )
            except BaseException as exc:
                write_json(
                    status_path,
                    {
                        **base_status,
                        "status": "failed",
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc),
                    },
                )
                raise
        completed.append((task, payload))
        _summary(matrix, completed)
        write_json(
            output_root(matrix) / "progress.json",
            {
                "completed": len(completed),
                "total": 6,
                "status": "completed" if len(completed) == 6 else "active",
                "last_completed": run_id(task),
                "radius_calibration_opened": False,
                "untouched_evaluation_opened": False,
            },
        )


def worker(config_path: Path, output: Path, device: str) -> None:
    if device != "mps" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("aggregate-control private gradients require MPS without fallback")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no private training performed")
    config = yaml.safe_load(config_path.read_text())
    algo_config = config["training"]["algo_config"]
    if (
        config["clients"]["num_clients"] != 10
        or algo_config["aggregate_control_replay_phase"] != "headroom_fit"
        or algo_config["aggregation_role_arm"] != "far_rfa"
        or not algo_config["external_attack_diagnostics"]
        or not algo_config["enable_oracle_diagnostics"]
    ):
        raise RuntimeError("aggregate-control worker boundary mismatch")
    from algorithms.aggregate_control_replay import (
        AggregateControlReplay,
        evaluate_aggregate_control_replay,
    )
    from algorithms.base import register_algorithm
    from scripts.run_rcig_batch_screen_experiment import write_runtime_manifest
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    import run_experiment as harness

    register_algorithm("ldp_gradient_far")(AggregateControlReplay)
    output.mkdir(parents=True, exist_ok=True)
    before = write_runtime_manifest(
        output, algorithm="aggregate_control_replay", stage="before_training"
    )
    original_evaluator = harness.rcig_reference_oracle_metrics
    original_argv = sys.argv
    trace_path = output / "simulator_randomness_private_audit.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace:

        def sink(row):
            trace.write(json.dumps(row, sort_keys=True) + "\n")
            trace.flush()

        AggregateControlReplay.audit_sink = staticmethod(sink)
        harness.rcig_reference_oracle_metrics = evaluate_aggregate_control_replay
        sys.argv = [
            str(ROOT / "run_experiment.py"),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--device",
            "mps",
        ]
        try:
            with evaluation_rng_isolation(harness):
                harness.main()
        finally:
            harness.rcig_reference_oracle_metrics = original_evaluator
            AggregateControlReplay.audit_sink = None
            sys.argv = original_argv
    write_runtime_manifest(
        output,
        algorithm="aggregate_control_replay",
        stage="after_training",
        previous=before,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--launch", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["mps"])
    args = parser.parse_args()
    if args.worker:
        if args.config is None or args.output is None or args.device != "mps":
            raise ValueError("worker requires --config, --output and --device mps")
        worker(args.config, args.output, args.device)
        return

    matrix = load_matrix()
    if args.status:
        print(json.dumps(campaign_status(matrix), indent=2))
        return
    stamp = provenance(matrix)
    if args.plan:
        print(
            json.dumps(
                {
                    "campaign_id": CAMPAIGN,
                    "phase": "headroom_fit",
                    "runs": [run_id(task) for task in headroom_tasks(matrix)],
                    "total": 6,
                    "private_device": "mps",
                    "mps_fallback": 0,
                    "privacy": stamp["privacy"],
                    "model_driver": (
                        "uniform rounds 1-12, unchanged FAR(RFA) rounds 13-40"
                    ),
                    "radius_status": "uncalibrated_diagnostic_only",
                    "radius_calibration_opened": False,
                    "untouched_evaluation_opened": False,
                    "output": str(output_root(matrix)),
                },
                indent=2,
            )
        )
        return
    if not args.resume:
        raise ValueError("--resume is mandatory")
    if args.max_new is not None and args.max_new < 0:
        raise ValueError("--max-new must be non-negative")
    shared.require_working_mps()
    with execution_lock(matrix):
        _ensure_no_duplicate_process()
        if args.launch:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a", encoding="utf-8") as log:
                command = [
                    sys.executable,
                    "-B",
                    "-u",
                    str(Path(__file__).resolve()),
                    "--run",
                    "--resume",
                ]
                if args.max_new is not None:
                    command += ["--max-new", str(args.max_new)]
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
            print(json.dumps({"pid": child.pid, "log": str(LOG)}))
            return
        _ensure_campaign_lock(matrix, stamp)
        try:
            execute(matrix, stamp, max_new=args.max_new)
        except BaseException as exc:
            write_json(
                output_root(matrix) / "failure.json",
                {"error": str(exc), "at": datetime.now(timezone.utc).isoformat()},
            )
            raise


if __name__ == "__main__":
    main()
