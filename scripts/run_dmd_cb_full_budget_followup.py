#!/usr/bin/env python3
"""MPS-only 18-run CE screen, then gate-controlled 48/96 follow-ups."""

from __future__ import annotations

import argparse
import copy
import fcntl
from functools import lru_cache
import itertools
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
from scripts import run_dmd_cb_private_screen as pilot
from scripts.dmd_cb_followup_gate import evaluate_gate
from scripts.run_rcig_batch_screen import (
    canonical_hash,
    file_hash,
    read_json,
    write_json,
    source_closure,
    rdp_module,
    require_working_mps,
)

MATRIX = ROOT / "configs/ldp_gradient_far/dmd_cb_full_budget_v1.yaml"
ENTRY = ROOT / "scripts/run_dmd_cb_full_budget_experiment.py"
PHASES = ("ce_full_budget_screen", "confirmation", "attacks")


def load_matrix():
    matrix = yaml.safe_load(MATRIX.read_text())
    if (
        matrix["campaign_id"] != "dmd_cb_full_budget_v1"
        or tuple(matrix["phases"]) != PHASES
    ):
        raise ValueError("Wrong scientific identity/phase order")
    expected_seeds = ([24, 42, 72], [101, 202, 303, 404], [101, 202, 303, 404])
    for phase, count, seeds in zip(PHASES, (18, 48, 96), expected_seeds):
        if (
            matrix["phases"][phase]["expected"] != count
            or matrix["phases"][phase]["seeds"] != seeds
        ):
            raise ValueError("Unexpected run count/seeds")
        if len(tasks(matrix, phase)) != count:
            raise ValueError("Factorial design has the wrong size")
    if matrix["noise_regimes"] != {
        "homogeneous": [1] * 25,
        "heteroscedastic": [1, 2] * 12 + [1],
    }:
        raise ValueError("Changed public noise assignment")
    if (
        matrix["execution"]["private_compute"] != "mps"
        or matrix["execution"]["fallback"] != 0
    ):
        raise ValueError("Only MPS without fallback is authorized")
    return matrix


def tasks(matrix, phase):
    p = matrix["phases"][phase]
    return [
        {
            "phase": phase,
            "seed": seed,
            "noise": noise,
            "objective": obj,
            "mode": mode,
            "scenario": scenario,
            "id": f"{noise}__seed{seed}__{obj}__{mode}__{scenario}",
        }
        for seed, noise, mode, scenario, obj in itertools.product(
            p["seeds"],
            matrix["noise_regimes"],
            p["modes"],
            p["scenarios"],
            p["objectives"],
        )
    ]


@lru_cache(maxsize=2)
def sigma(objective):
    if objective == "dmd":
        return pilot.sigma()  # Exactly the original DMD calibration, not a retune.
    if objective != "ce":
        raise ValueError("Unknown objective")
    value = rdp_module().calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=4,
        delta=1e-5,
        sampling_rate=0.1,
        steps=40,
        sensitivity_multiplier=2,
        tolerance=1e-8,
    )
    while pilot.gradient_epsilon(40, value) > 4:
        value *= 1.0000001
    return value


def resolved(matrix, task, output, rounds=40):
    base = yaml.safe_load((ROOT / matrix["base_matrix"]).read_text())
    config = copy.deepcopy(base["base"])
    config.update(seed=task["seed"], device="mps", output_dir=str(output))
    config["data"]["partition_seed"] = task["seed"]
    dmd = task["objective"] == "dmd"
    config["training"].update(
        num_rounds=rounds,
        algorithm="ldp_gradient_dmd_cb" if dmd else "ldp_gradient_ce_full_budget",
    )
    a = config["training"]["algo_config"]
    a.update(
        dmd_mu=0.1875 if dmd else 0.0,
        dmd_histogram_epsilon=0.25 if dmd else 0.0,
        target_epsilon=3.75 if dmd else 4.0,
        dmd_total_epsilon=4.0,
        dmd_server_mode=task["mode"],
        far_alpha=0.1 if task["mode"] == "far_rfa" else 0.0,
        noise_multiplier=sigma(task["objective"]),
        dmd_frozen_base_noise_multiplier=sigma(task["objective"]),
        dmd_pairing_seed=task["seed"],
        privacy_noise_multiplier_scale_by_client=matrix["noise_regimes"][task["noise"]],
    )
    if task["scenario"] != "none":
        spec = matrix["attacks"]
        a["num_byzantine"] = len(spec["client_ids"])
        a["attack"] = {
            "enabled": True,
            "name": task["scenario"],
            "scale": spec["scales"][task["scenario"]],
            "num_byzantine": len(spec["client_ids"]),
            "client_ids": spec["client_ids"],
            "active_round_start": spec["active_round_start"],
            "active_round_end": spec["active_round_end"],
        }
    return config


def provenance(matrix):
    old = pilot.load_matrix()
    oldstamp = pilot.provenance(old)
    pilot.ensure_lock(ROOT / matrix["pilot_root"], oldstamp)
    oldstatus = pilot.status(ROOT / matrix["pilot_root"], old, oldstamp)
    if oldstatus["complete_valid"] != 36 or oldstatus["invalid"] or oldstatus["active"]:
        raise RuntimeError(
            "Original 36 inputs must remain complete, validated and immutable"
        )
    inputs = {}
    for task in pilot.tasks(old):
        path = pilot.metrics_path(ROOT / matrix["pilot_root"] / "runs" / task["id"])
        inputs[str(path.relative_to(ROOT))] = file_hash(path)
    return {
        "campaign_id": matrix["campaign_id"],
        "matrix_sha256": file_hash(MATRIX),
        "protocol_sha256": file_hash(ROOT / matrix["protocol"]),
        "base_matrix_sha256": file_hash(ROOT / matrix["base_matrix"]),
        "source_sha256": source_closure(
            (
                Path(__file__),
                ENTRY,
                ROOT / "algorithms/ldp_gradient_ce_full_budget.py",
                ROOT / "algorithms/ldp_gradient_dmd_cb.py",
            )
        ),
        "original_metrics_sha256": inputs,
        "sigma": {obj: sigma(obj) for obj in ("ce", "dmd")},
    }


def verify_sources(matrix, stamp):
    if (
        file_hash(MATRIX) != stamp["matrix_sha256"]
        or file_hash(ROOT / matrix["protocol"]) != stamp["protocol_sha256"]
    ):
        raise RuntimeError("Matrix/protocol drift")
    if file_hash(ROOT / matrix["base_matrix"]) != stamp["base_matrix_sha256"]:
        raise RuntimeError("Base matrix changed")
    for relative, digest in {
        **stamp["source_sha256"],
        **stamp["original_metrics_sha256"],
    }.items():
        if file_hash(ROOT / relative) != digest:
            raise RuntimeError(f"Source/input drift: {relative}")


def validate_result(output, config, stamp):
    data = read_json(pilot.metrics_path(output))
    a = config["training"]["algo_config"]
    dmd = a["dmd_mu"] != 0
    if (
        data["algorithm"] != config["training"]["algorithm"]
        or data["config"].get("device") != "mps"
    ):
        raise RuntimeError("Wrong algorithm/device")
    if any(data["config"].get(k) != v for k, v in a.items()):
        raise RuntimeError("Recorded algorithm settings changed")
    expected_summary = {
        "num_rounds": config["training"]["num_rounds"],
        "num_clients": 25,
        "seed": config["seed"],
        "partition_seed": config["seed"],
        "model": "lenet5_tanh",
        "dataset": "fashionmnist",
    }
    if any(data["summary"].get(k) != v for k, v in expected_summary.items()):
        raise RuntimeError("Recorded model/data/seed/horizon changed")
    rows = data["rounds"]
    if len(rows) != config["training"]["num_rounds"]:
        raise RuntimeError("Incomplete metrics")
    ids = list(range(5, 25)) if a["attack"]["enabled"] else list(range(25))
    for t, row in enumerate(rows, 1):
        attacked = (
            bool(a["attack"]["enabled"])
            and a["attack"]["active_round_start"]
            <= t
            <= a["attack"]["active_round_end"]
        )
        exact = {
            "round_num": t,
            "num_clients": 25,
            "num_selected": 25,
            "num_survivors": 25,
            "num_alive_clients": 25,
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "dmd_server_mode": a["dmd_server_mode"],
            "dmd_mu": a["dmd_mu"],
            "privacy_dmd_histogram_calls_per_client_max": int(dmd),
            "privacy_dmd_histogram_calls_this_round": 25 if dmd and t == 1 else 0,
            "evaluated_client_ids_oracle": ids,
            "attack_window_active": attacked,
            "attack_enabled": attacked,
            "attack_schedule_phase": "attack" if attacked else "clean",
            "attack_name": a["attack"]["name"] if attacked else "none",
            "num_byzantine_oracle": 5 if attacked else 0,
            "byzantine_fraction_oracle": 0.2 if attacked else 0.0,
            "attack_scheduled_name": a["attack"]["name"],
            "far_attack_labels_visible_to_server_aggregate": False,
            "far_attack_config_visible_to_server_aggregate": False,
            "privacy_dmd_budget_composition": (
                "epsilon_histogram_plus_epsilon_gradient"
                if dmd
                else "gradient_rdp_only_no_histogram"
            ),
        }
        if any(row.get(k) != v for k, v in exact.items()):
            raise RuntimeError(f"Round {t}: execution/privacy/population contract")
        if row.get("ldp_gradient_far_private_compute_device") not in {"mps", "mps:0"}:
            raise RuntimeError("CPU private computation")
        grad = [
            pilot.gradient_epsilon(t, a["noise_multiplier"] * s)
            for s in a["privacy_noise_multiplier_scale_by_client"]
        ]
        hist = a["dmd_histogram_epsilon"]
        checks = {
            "privacy_epsilon_max": hist + max(grad),
            "privacy_epsilon_mean": hist + sum(grad) / 25,
            "privacy_gradient_epsilon_max": max(grad),
            "privacy_gradient_epsilon_mean": sum(grad) / 25,
            "privacy_delta": 1e-5,
            "privacy_dmd_histogram_epsilon": hist,
            "privacy_gradient_target_epsilon": a["target_epsilon"],
            "privacy_target_epsilon": 4.0,
            "privacy_model_noise_multiplier_min": a["noise_multiplier"]
            * min(a["privacy_noise_multiplier_scale_by_client"]),
            "privacy_model_noise_multiplier_max": a["noise_multiplier"]
            * max(a["privacy_noise_multiplier_scale_by_client"]),
        }
        for k, v in checks.items():
            if not math.isclose(pilot.number(row, k), v, rel_tol=1e-7, abs_tol=1e-6):
                raise RuntimeError(f"Round {t}: wrong privacy metric {k}")
        for k, lo, hi in (
            ("test_accuracy", 0, 1),
            ("test_loss", 0, math.inf),
            ("client_accuracy_mean", 0, 1),
            ("mean_client_balanced_accuracy_pct", 0, 100),
            ("worst20_accuracy_pct", 0, 100),
            ("best20_worst20_gap_pct", 0, 100),
            ("client_accuracy_variance_pct2", 0, math.inf),
        ):
            if not lo <= pilot.number(row, k) <= hi:
                raise RuntimeError(f"Round {t}: invalid metric {k}")
    runtime = read_json(output / "runtime_imports.json")
    if (
        runtime.get("stage") != "after_training"
        or runtime.get("device") != "mps"
        or runtime.get("mps_fallback") != 0
        or runtime.get("algorithm") != data["algorithm"]
    ):
        raise RuntimeError("Incomplete/non-MPS runtime manifest")
    if not runtime.get("source_sha256") or any(
        stamp["source_sha256"].get(k) != v for k, v in runtime["source_sha256"].items()
    ):
        raise RuntimeError("Runtime imports do not match source lock")


def status(root, matrix, stamp, phase):
    report = {
        "phase": phase,
        "complete_valid": 0,
        "expected": matrix["phases"][phase]["expected"],
        "active": [],
        "invalid": [],
        "missing": [],
    }
    for task in tasks(matrix, phase):
        output = root / phase / "runs" / task["id"]
        if not output.exists():
            report["missing"].append(task["id"])
            continue
        try:
            record = read_json(output / "orchestration_status.json")
            if record.get("task") != task or record.get(
                "scientific_hash"
            ) != canonical_hash(stamp):
                raise RuntimeError("Wrong task/provenance identity")
            if record.get("state") == "running" and pilot.alive(
                record.get("child_pid")
            ):
                report["active"].append(
                    {"task": task["id"], "pid": record["child_pid"]}
                )
                continue
            if record.get("state") != "completed":
                raise RuntimeError(f"Stopped/partial run: {record.get('state')}")
            cfg = resolved(matrix, task, output)
            if (
                record.get("resolved_config_hash") != canonical_hash(cfg)
                or yaml.safe_load((output / "config.yaml").read_text()) != cfg
            ):
                raise RuntimeError("Saved configuration changed")
            validate_result(output, cfg, stamp)
            if record["metrics_sha256"] != file_hash(
                pilot.metrics_path(output)
            ) or record["runtime_sha256"] != file_hash(output / "runtime_imports.json"):
                raise RuntimeError("Saved metrics/runtime changed")
            report["complete_valid"] += 1
        except Exception as exc:
            report["invalid"].append({"task": task["id"], "error": str(exc)})
    return report


def run_one(matrix, task, output, stamp, *, rounds=40):
    verify_sources(matrix, stamp)
    output.mkdir(parents=True, exist_ok=False)
    cfg = resolved(matrix, task, output, rounds)
    with (output / "config.yaml").open("x") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    rec = {
        "task": task,
        "state": "starting",
        "parent_pid": os.getpid(),
        "scientific_hash": canonical_hash(stamp),
        "resolved_config_hash": canonical_hash(cfg),
    }
    statefile = output / "orchestration_status.json"
    write_json(statefile, rec, exclusive=True)
    try:
        with (output / "run.log").open("x") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(ENTRY),
                    "--config",
                    str(output / "config.yaml"),
                    "--output",
                    str(output),
                    "--device",
                    "mps",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            rec.update(state="running", child_pid=child.pid)
            write_json(statefile, rec)
            print(json.dumps({"started": task}), flush=True)
            if child.wait() != 0:
                raise RuntimeError(f"Child failed; inspect {output/'run.log'}")
        verify_sources(matrix, stamp)
        validate_result(output, cfg, stamp)
        rec.update(
            state="completed",
            metrics_sha256=file_hash(pilot.metrics_path(output)),
            runtime_sha256=file_hash(output / "runtime_imports.json"),
        )
        write_json(statefile, rec)
        print(json.dumps({"completed": task}), flush=True)
    except BaseException as exc:
        rec.update(state="failed", error=str(exc))
        write_json(statefile, rec)
        raise


def execute(root, matrix, stamp):
    require_working_mps()
    pilot.ensure_lock(root, stamp)
    with (root / ".execution.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "controller_status.json").exists():
            previous = read_json(root / "controller_status.json")
            if previous.get("state") in {"stopped_by_gate", "failed", "completed"}:
                raise RuntimeError(
                    "Terminal controller state; evidence must not be rewritten by resume"
                )
            if previous.get("state") == "running" and pilot.alive(previous.get("pid")):
                raise RuntimeError("Another controller is active")
        state = {
            "state": "running",
            "pid": os.getpid(),
            "device": "mps",
            "scientific_hash": canonical_hash(stamp),
        }
        write_json(root / "controller_status.json", state)
        try:
            for phase in PHASES:
                verify_sources(matrix, stamp)
                report = status(root, matrix, stamp, phase)
                if report["invalid"] or report["active"]:
                    raise RuntimeError(
                        "Existing invalid/active run: " + json.dumps(report)
                    )
                state.update(phase=phase)
                write_json(root / "controller_status.json", state)
                missing = set(report["missing"])
                for task in tasks(matrix, phase):
                    if task["id"] in missing:
                        run_one(matrix, task, root / phase / "runs" / task["id"], stamp)
                report = status(root, matrix, stamp, phase)
                if report["complete_valid"] != report["expected"] or report["invalid"]:
                    raise RuntimeError("Final phase validation failed")
                write_json(root / phase / "final_validation.json", report)
                if phase != "attacks":
                    evidence = evaluate_gate(phase, root, matrix, ROOT)
                    evidence["scientific_hash"] = canonical_hash(stamp)
                    evidence["input_metrics_sha256"] = {
                        str(p.relative_to(ROOT)): file_hash(p)
                        for p in (root / phase).glob("runs/**/metrics.json")
                    }
                    if phase == "ce_full_budget_screen":
                        evidence["pilot_metrics_sha256"] = stamp[
                            "original_metrics_sha256"
                        ]
                    write_json(root / phase / "gate_evidence.json", evidence)
                    from scripts.dmd_cb_followup_gate import render_gate_report

                    (root / phase / "comparison.md").write_text(
                        render_gate_report(evidence)
                    )
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "gate": evidence["decision"],
                                "failed": evidence["failed_criteria"],
                            }
                        ),
                        flush=True,
                    )
                    if evidence["decision"] != "promote":
                        state.update(
                            state="stopped_by_gate",
                            completed_valid=report["complete_valid"],
                            gate=evidence["decision"],
                        )
                        write_json(root / "controller_status.json", state)
                        return
            state.update(state="completed")
            write_json(root / "controller_status.json", state)
        except BaseException as exc:
            state.update(state="failed", error=str(exc))
            write_json(root / "controller_status.json", state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    for name in ("status", "dry-run", "smoke", "run", "detach"):
        modes.add_argument("--" + name, action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    matrix = load_matrix()
    root = ROOT / matrix["output_root"]
    stamp = provenance(matrix)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phases": {k: len(tasks(matrix, k)) for k in PHASES},
                    "sigma": stamp["sigma"],
                    "source_count": len(stamp["source_sha256"]),
                    "root": str(root),
                },
                indent=2,
            )
        )
        return
    if args.smoke:
        require_working_mps()
        scratch = Path(tempfile.mkdtemp(prefix="dmd_ce4_smoke_", dir="/private/tmp"))
        for mode in ("uniform", "direct_rfa", "far_rfa"):
            task = next(
                t
                for t in tasks(matrix, PHASES[0])
                if t["seed"] == 24 and t["noise"] == "homogeneous" and t["mode"] == mode
            )
            run_one(matrix, task, scratch / mode, stamp, rounds=2)
        print(json.dumps({"smoke_valid": 3, "root": str(scratch)}))
        return
    if args.run or args.detach:
        if not args.resume:
            raise RuntimeError("--resume required; never overwrite partial results")
        if args.run:
            execute(root, matrix, stamp)
            return
        require_working_mps()
        pilot.ensure_lock(root, stamp)
        with (root / ".launch.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (root / "controller_status.json").exists():
                old = read_json(root / "controller_status.json")
                if old.get("state") == "running" and pilot.alive(old.get("pid")):
                    raise RuntimeError("Controller already active")
                if old.get("state") in {"stopped_by_gate", "failed", "completed"}:
                    raise RuntimeError(
                        "Chain is terminal; inspect evidence rather than relaunch"
                    )
            for phase in PHASES:
                current = status(root, matrix, stamp, phase)
                if current["invalid"] or current["active"]:
                    raise RuntimeError("Existing partial/invalid/active result")
            with (root / "controller.log").open("a") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "--run",
                        "--resume",
                    ],
                    cwd=ROOT,
                    start_new_session=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env={
                        **os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONUNBUFFERED": "1",
                    },
                )
            for _ in range(100):
                if child.poll() is not None:
                    raise RuntimeError(
                        f"Controller stopped at startup: {child.returncode}"
                    )
                if (root / "controller_status.json").exists():
                    rec = read_json(root / "controller_status.json")
                    if rec.get("pid") == child.pid and rec.get("state") == "running":
                        print(
                            json.dumps(
                                {
                                    "launched_pid": child.pid,
                                    "root": str(root),
                                    "first_phase": 18,
                                    "conditional_next": [48, 96],
                                }
                            )
                        )
                        return
                time.sleep(0.25)
            raise RuntimeError("Startup unconfirmed; inspect process before retrying")
    if (root / "scientific_lock.json").exists():
        pilot.ensure_lock(root, stamp)
    print(
        json.dumps(
            {phase: status(root, matrix, stamp, phase) for phase in PHASES}, indent=2
        )
    )


if __name__ == "__main__":
    main()
