#!/usr/bin/env python3
"""Isolated, resumable 36-run DMD-CB private screen; never mutates RCIG."""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.run_rcig_batch_screen import (
    canonical_hash, file_hash, read_json, write_json, source_closure,
    rdp_module, require_working_mps,
)

MATRIX = ROOT / "configs/ldp_gradient_far/dmd_cb_private_v1.yaml"
ENTRY = ROOT / "scripts/run_dmd_cb_private_experiment.py"
EXPECTED_ARMS = {
    "ce_uniform": (0.0, "uniform", 0.0),
    "dmd_uniform": (0.1875, "uniform", 0.0),
    "ce_rfa": (0.0, "direct_rfa", 0.0),
    "dmd_rfa": (0.1875, "direct_rfa", 0.0),
    "ce_far_rfa": (0.0, "far_rfa", 0.1),
    "dmd_far_rfa": (0.1875, "far_rfa", 0.1),
}


def load_matrix():
    matrix = yaml.safe_load(MATRIX.read_text())
    if matrix["campaign_id"] != "dmd_cb_private_v1" or matrix["expected_runs"] != 36:
        raise ValueError("Unexpected campaign identity/count")
    if matrix["seeds"] != [24, 42, 72] or set(matrix["arms"]) != set(EXPECTED_ARMS):
        raise ValueError("The six arms and three seeds are fixed")
    for arm, (mu, mode, alpha) in EXPECTED_ARMS.items():
        if matrix["arms"][arm] != {"dmd_mu": mu, "dmd_server_mode": mode, "far_alpha": alpha}:
            raise ValueError(f"Unexpected arm: {arm}")
    if matrix["noise_regimes"] != {"homogeneous": [1] * 25, "heteroscedastic": [1, 2] * 12 + [1]}:
        raise ValueError("Noise assignment must remain paired and public")
    a = matrix["base"]["training"]["algo_config"]
    fixed = {"fixed_batch_size": 240, "privacy_public_dataset_size": 2400,
             "target_epsilon": 3.75, "dmd_histogram_epsilon": 0.25,
             "dmd_total_epsilon": 4.0, "delta": 1e-5, "privacy_num_rounds": 40,
             "privacy_adjacency": "replace_one", "sampling_scheme": "fixed_without_replacement"}
    if any(a.get(k) != v for k, v in fixed.items()):
        raise ValueError("Privacy plan differs from the audited fixed screen")
    if matrix["base"]["training"]["num_rounds"] != 40:
        raise ValueError("This screen has exactly 40 rounds")
    return matrix


@lru_cache(maxsize=1)
def sigma():
    # Choose the conservative side if a bisection tolerance lands just above target.
    value = rdp_module().calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=3.75, delta=1e-5, sampling_rate=0.1, steps=40,
        sensitivity_multiplier=2.0, tolerance=1e-7,
    )
    while gradient_epsilon(40, value) > 3.75:
        value *= 1.0000001
    return value


@lru_cache(maxsize=256)
def gradient_epsilon(rounds, implementation_sigma):
    accountant = rdp_module().RDPAccountant()
    accountant.add_sampled_without_replacement_gaussian(
        channel="model", sampling_rate=0.1,
        noise_multiplier=implementation_sigma / 2, steps=rounds,
    )
    return accountant.epsilon(1e-5)[0]


def tasks(matrix):
    return [{"seed": seed, "noise": noise, "arm": arm,
             "id": f"{noise}__seed{seed}__{arm}"}
            for seed in matrix["seeds"] for noise in matrix["noise_regimes"]
            for arm in matrix["arms"]]


def resolved(matrix, task, output, rounds=40):
    config = copy.deepcopy(matrix["base"])
    config.update(seed=task["seed"], device="mps", output_dir=str(output))
    config["data"]["partition_seed"] = task["seed"]
    config["training"]["num_rounds"] = rounds
    a = config["training"]["algo_config"]
    a.update(matrix["arms"][task["arm"]])
    a.update(noise_multiplier=sigma(), dmd_frozen_base_noise_multiplier=sigma(), dmd_pairing_seed=task["seed"],
             privacy_noise_multiplier_scale_by_client=matrix["noise_regimes"][task["noise"]])
    return config


def provenance(matrix):
    paths = (Path(__file__), ENTRY, ROOT / "algorithms/ldp_gradient_dmd_cb.py",
             ROOT / "privacy/dmd_cb_private.py", ROOT / "privacy/rdp.py")
    return {"campaign_id": matrix["campaign_id"], "matrix_sha256": file_hash(MATRIX),
            "protocol_sha256": file_hash(ROOT / matrix["protocol"]),
            "source_sha256": source_closure(paths), "sigma": sigma(),
            "gradient_epsilon": gradient_epsilon(40, sigma()),
            "total_epsilon": .25 + gradient_epsilon(40, sigma()), "expected_runs": 36}


def ensure_lock(root, stamp):
    lock = root / "scientific_lock.json"
    if lock.exists():
        saved = read_json(lock)
        if saved != {"scientific_hash": canonical_hash(stamp), "provenance": stamp}:
            raise RuntimeError("Scientific provenance drift: refusing resume")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Refusing nonempty output root without scientific lock")
    root.mkdir(parents=True, exist_ok=True)
    write_json(lock, {"scientific_hash": canonical_hash(stamp), "provenance": stamp}, exclusive=True)


def verify_sources(matrix, stamp):
    if file_hash(MATRIX) != stamp["matrix_sha256"] or file_hash(ROOT / matrix["protocol"]) != stamp["protocol_sha256"]:
        raise RuntimeError("Matrix/protocol changed during campaign")
    changed = [rel for rel, digest in stamp["source_sha256"].items()
               if not (ROOT / rel).is_file() or file_hash(ROOT / rel) != digest]
    if changed:
        raise RuntimeError("Source drift: " + ", ".join(changed))


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # do not mistake a sandbox visibility restriction for a dead run
    except ProcessLookupError:
        return False


def number(row, key):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise RuntimeError(f"Missing/non-finite metric: {key}")
    return value


def validate_metrics(payload, config):
    a = config["training"]["algo_config"]
    T = config["training"]["num_rounds"]
    if payload.get("algorithm") != "ldp_gradient_dmd_cb":
        raise RuntimeError("Wrong algorithm in metrics")
    required = {"num_rounds": T, "seed": config["seed"], "partition_seed": config["seed"],
                "dataset": "fashionmnist", "model": "lenet5_tanh", "num_clients": 25}
    if any(payload.get("summary", {}).get(k) != v for k, v in required.items()):
        raise RuntimeError("Wrong dataset/model/seed/round count")
    actual = payload.get("config", {})
    if actual.get("device") != "mps" or any(actual.get(k) != v for k, v in a.items()):
        raise RuntimeError("Recorded algorithm configuration mismatch")
    rows = payload.get("rounds", [])
    if len(rows) != T:
        raise RuntimeError("Incomplete round metrics")
    scales = a["privacy_noise_multiplier_scale_by_client"]
    for t, row in enumerate(rows, 1):
        fixed = {"round_num": t, "num_clients": 25, "num_selected": 25,
                 "num_survivors": 25, "num_alive_clients": 25,
                 "privacy_sampling_scheme": "fixed_without_replacement",
                 "privacy_adjacency": "replace_one",
                 "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
                 "dmd_server_mode": a["dmd_server_mode"], "dmd_mu": a["dmd_mu"],
                 "dmd_far_enabled": a["dmd_server_mode"] == "far_rfa",
                 "privacy_dmd_histogram_calls_per_client_max": 1,
                 "privacy_dmd_histogram_calls_this_round": 25 if t == 1 else 0,
                 "privacy_dmd_budget_composition": "epsilon_histogram_plus_epsilon_gradient"}
        if any(row.get(k) != v for k, v in fixed.items()):
            raise RuntimeError(f"Round {t}: participation/privacy/device contract failed")
        if row.get("ldp_gradient_far_private_compute_device") not in {"mps", "mps:0"}:
            raise RuntimeError("Private gradients were not all computed on MPS")
        for key, low, high in (("test_accuracy", 0, 1), ("test_loss", 0, math.inf),
                               ("client_accuracy_mean", 0, 1),
                               ("client_accuracy_variance_pct2", 0, math.inf),
                               ("worst20_accuracy_pct", 0, 100),
                               ("best20_worst20_gap_pct", 0, 100)):
            if not low <= number(row, key) <= high:
                raise RuntimeError(f"Wrong units/range: {key}")
        epsilons = [.25 + gradient_epsilon(t, sigma() * scale) for scale in scales]
        expected = {"privacy_epsilon_max": max(epsilons),
                    "privacy_epsilon_mean": sum(epsilons) / 25,
                    "privacy_delta": 1e-5,
                    "privacy_model_noise_multiplier_min": sigma() * min(scales),
                    "privacy_model_noise_multiplier_max": sigma() * max(scales),
                    "privacy_model_noise_multiplier_mean": sigma() * sum(scales) / 25,
                    "privacy_model_steps_mean": 1.0, "privacy_dmd_histogram_epsilon": .25,
                    "privacy_target_epsilon": 4.0, "privacy_gradient_target_epsilon": 3.75,
                    "privacy_gradient_epsilon_max": max(epsilons) - .25,
                    "privacy_gradient_epsilon_mean": sum(epsilons) / 25 - .25}
        for key, value in expected.items():
            if not math.isclose(number(row, key), value, rel_tol=1e-7, abs_tol=1e-6):
                raise RuntimeError(f"Round {t}: recomputed accountant/mechanism mismatch: {key}")
        if row["privacy_epsilon_max"] > 4.0001:
            raise RuntimeError("Total epsilon exceeds the declared budget")


def metrics_path(output):
    # The unmodified harness creates a named run subdirectory below --output.
    found = sorted(output.glob("**/metrics.json"))
    if len(found) != 1:
        raise RuntimeError(f"Expected one unambiguous metrics artifact below {output}, found {len(found)}")
    return found[0]


def validate_result(output, config, stamp):
    validate_metrics(read_json(metrics_path(output)), config)
    runtime = read_json(output / "runtime_imports.json")
    if runtime.get("stage") != "after_training" or runtime.get("device") != "mps" or runtime.get("mps_fallback") != 0:
        raise RuntimeError("Runtime manifest is incomplete or not MPS")
    imported = runtime.get("source_sha256", {})
    if not imported:
        raise RuntimeError("Missing runtime source hashes")
    for rel, digest in imported.items():
        if stamp["source_sha256"].get(rel) != digest:
            raise RuntimeError(f"Runtime source not covered by scientific lock: {rel}")


def status(root, matrix, stamp):
    report = {"campaign_id": matrix["campaign_id"], "complete_valid": 0,
              "expected": 36, "active": [], "invalid": [], "missing": []}
    for task in tasks(matrix):
        output = root / "runs" / task["id"]
        statusfile = output / "orchestration_status.json"
        if not output.exists():
            report["missing"].append(task["id"])
            continue
        try:
            record = read_json(statusfile)
            if record.get("task") != task or record.get("scientific_hash") != canonical_hash(stamp):
                raise RuntimeError("Task/source identity mismatch")
            if record.get("state") == "running" and alive(record.get("child_pid")):
                report["active"].append({"task": task["id"], "pid": record["child_pid"]})
                continue
            if record.get("state") != "completed":
                raise RuntimeError(f"Noncompleted/stopped run: {record.get('state')}")
            config = resolved(matrix, task, output)
            if record.get("resolved_config_hash") != canonical_hash(config):
                raise RuntimeError("Resolved configuration changed")
            if yaml.safe_load((output / "config.yaml").read_text()) != config:
                raise RuntimeError("Saved configuration changed")
            validate_result(output, config, stamp)
            if record.get("metrics_sha256") != file_hash(metrics_path(output)):
                raise RuntimeError("Metrics changed after validation")
            if record.get("runtime_sha256") != file_hash(output / "runtime_imports.json"):
                raise RuntimeError("Runtime manifest changed after validation")
            report["complete_valid"] += 1
        except Exception as exc:
            report["invalid"].append({"task": task["id"], "error": str(exc)})
    report["missing_count"] = len(report["missing"])
    return report


def run_one(matrix, task, output, stamp, rounds=40):
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing run: {output}")
    verify_sources(matrix, stamp)
    output.mkdir(parents=True)
    config = resolved(matrix, task, output, rounds)
    with (output / "config.yaml").open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    record = {"task": task, "state": "starting", "parent_pid": os.getpid(),
              "scientific_hash": canonical_hash(stamp), "resolved_config_hash": canonical_hash(config)}
    statefile = output / "orchestration_status.json"
    write_json(statefile, record, exclusive=True)
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    try:
        with (output / "run.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-B", str(ENTRY), "--config", str(output / "config.yaml"),
                                      "--output", str(output), "--device", "mps"],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            record.update(state="running", child_pid=child.pid)
            write_json(statefile, record)
            print(json.dumps({"started": task["id"], "pid": child.pid}), flush=True)
            code = child.wait()
        if code != 0:
            raise RuntimeError(f"Training exit code {code}; inspect {output / 'run.log'}")
        verify_sources(matrix, stamp)
        validate_result(output, config, stamp)
        record.update(state="completed", metrics_sha256=file_hash(metrics_path(output)),
                      runtime_sha256=file_hash(output / "runtime_imports.json"))
        write_json(statefile, record)
        print(json.dumps({"completed_valid": task["id"]}), flush=True)
    except BaseException as exc:
        record.update(state="failed", error=str(exc))
        write_json(statefile, record)
        raise


def execute(root, matrix, stamp, allow_parallel):
    if not allow_parallel:
        raise RuntimeError("Parallel RCIG use requires --allow-parallel-rcig authorization")
    require_working_mps()
    ensure_lock(root, stamp)
    with (root / ".execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A DMD-CB controller is already active; refusing duplicate")
        write_json(root / "controller_status.json", {"state": "running", "pid": os.getpid(),
                    "allow_parallel_rcig": True, "device": "mps", "scientific_hash": canonical_hash(stamp)})
        try:
            report = status(root, matrix, stamp)
            if report["invalid"] or report["active"]:
                raise RuntimeError("Existing invalid/active run: " + json.dumps(report))
            missing = set(report["missing"])
            for task in tasks(matrix):
                if task["id"] in missing:
                    run_one(matrix, task, root / "runs" / task["id"], stamp)
            report = status(root, matrix, stamp)
            if report["complete_valid"] != 36 or report["invalid"]:
                raise RuntimeError("Final campaign validation failed")
            write_json(root / "final_validation.json", report)
            write_json(root / "controller_status.json", {"state": "completed", "pid": os.getpid(), "complete_valid": 36})
        except BaseException as exc:
            write_json(root / "controller_status.json", {"state": "failed", "pid": os.getpid(), "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-parallel-rcig", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix()
    root = ROOT / matrix["output_root"]
    stamp = provenance(matrix)
    if args.dry_run:
        print(json.dumps({"root": str(root), "tasks": tasks(matrix), "sigma": sigma(),
                          "gradient_epsilon": stamp["gradient_epsilon"], "total_epsilon": stamp["total_epsilon"],
                          "locked_source_count": len(stamp["source_sha256"])}, indent=2))
    elif args.smoke:
        if not args.allow_parallel_rcig:
            raise RuntimeError("Smoke parallelism must be explicitly authorized")
        require_working_mps()
        smoke_root = Path(tempfile.mkdtemp(prefix="dmd_cb_private_smoke_", dir="/private/tmp"))
        write_json(smoke_root / "scientific_lock.json", stamp)
        for arm in ("dmd_uniform", "dmd_rfa", "dmd_far_rfa"):
            task = next(t for t in tasks(matrix) if t["arm"] == arm and t["seed"] == 24 and t["noise"] == "homogeneous")
            run_one(matrix, task, smoke_root / arm, stamp, rounds=2)
        print(json.dumps({"smoke_valid": 3, "root": str(smoke_root)}), flush=True)
    elif args.run or args.detach:
        if not args.resume or not args.allow_parallel_rcig:
            raise RuntimeError("Use both --resume and --allow-parallel-rcig")
        if args.run:
            execute(root, matrix, stamp, True)
        else:
            require_working_mps()
            ensure_lock(root, stamp)
            # Holding the same lock serializes concurrent launch attempts.
            with (root / ".launch.lock").open("a+") as launch_lock:
                fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                controllerfile = root / "controller_status.json"
                if controllerfile.exists():
                    controller = read_json(controllerfile)
                    if controller.get("state") == "running" and alive(controller.get("pid")):
                        raise RuntimeError("DMD-CB campaign already running")
                report = status(root, matrix, stamp)
                if report["active"] or report["invalid"]:
                    raise RuntimeError("Existing active/invalid artifacts; refusing launch")
                with (root / "controller.log").open("a") as log:
                    child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--run", "--resume",
                                              "--allow-parallel-rcig"], cwd=ROOT, start_new_session=True,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
                for _ in range(100):
                    if child.poll() is not None:
                        raise RuntimeError(f"Controller stopped on startup: {child.returncode}")
                    if controllerfile.exists():
                        controller = read_json(controllerfile)
                        if controller.get("pid") == child.pid and controller.get("state") == "running":
                            write_json(root / "launch_receipt.json", {"pid": child.pid, "scientific_hash": canonical_hash(stamp),
                                        "parallel_rcig_authorized": True})
                            print(json.dumps({"launched_pid": child.pid, "root": str(root), "expected": 36}), flush=True)
                            return
                    time.sleep(.25)
                raise RuntimeError("Startup unconfirmed; inspect controller before retrying")
    else:
        if (root / "scientific_lock.json").exists():
            ensure_lock(root, stamp)
        print(json.dumps(status(root, matrix, stamp), indent=2))


if __name__ == "__main__":
    main()
