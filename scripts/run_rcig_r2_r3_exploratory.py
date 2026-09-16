#!/usr/bin/env python3
"""Explicit, separately identified R2/R3 exploration after the v2 R1 stop.

No v2 gate is changed, copied as a promotion, or bypassed by the v2 runner.
This launcher consumes the frozen v2 task definitions under a NEW protocol.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_rcig_ldp_gradient_far_v2 as analysis  # noqa: E402
from scripts import run_rcig_ldp_gradient_far_v2 as v2  # noqa: E402

CAMPAIGN_ID = "rcig_r2_r3_exploratory_v1"
OUTPUT_ROOT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
PROTOCOL = ROOT / "output/analysis/RCIG_R2_R3_Exploratory_Continuation.md"
LOCK_NAME = "_exploratory_provenance.json"
PHASES = ("r2_attack_mechanism", "r3_e2e_confirmation")
PARENT_PHASES = ("r0_dynamic_calibration", "r1_dynamic_null")
EXPECTED_COUNTS = dict(zip(PHASES, (144, 384)))


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return result


def _parent_artifacts(parent: v2.RCIGV2Campaign) -> dict[str, str]:
    """Hash every parent run artifact, not just the self-hashed gate JSON."""
    result = {}
    for task in parent.tasks:
        if task.phase_id not in PARENT_PHASES:
            continue
        directory = v2.task_output_dir(parent, task)
        metrics = v2._metrics_path(directory)
        if metrics is None:
            raise RuntimeError(f"parent metrics absent/ambiguous: {directory}")
        for path in (
            directory / "orchestration_status.json",
            directory / "resolved_config.yaml",
            metrics,
        ):
            result[str(path.relative_to(parent.output_root))] = v2._file_sha256(path)
    return result


def provenance(parent: v2.RCIGV2Campaign) -> dict[str, Any]:
    """Read-only provenance check, repeated before and after every task."""
    v2.verify_existing_campaign_lock(parent)
    decisions = {}
    for phase, decision in zip(PARENT_PHASES, ("promote", "stop")):
        gate = v2._verified_gate_payload(parent, phase)
        if gate["decision"] != decision:
            raise RuntimeError(f"parent {phase} must retain decision={decision}")
        decisions[phase] = {
            "decision": decision,
            "sha256": v2._file_sha256(v2.gate_path(parent, phase)),
        }
    base_path = (parent.matrix_path.parent / parent.matrix["base_config"]).resolve()
    return {
        "campaign_id": CAMPAIGN_ID,
        "interpretation": "exploratory_after_failed_R1_not_confirmatory",
        "parent_lock": v2._campaign_lock_payload(parent),
        "parent_lock_file_sha256": v2._file_sha256(parent.output_root / v2.LOCK_NAME),
        "parent_base_file_sha256": v2._file_sha256(base_path),
        "parent_gates": decisions,
        "parent_run_artifacts": _parent_artifacts(parent),
        "launcher_sha256": v2._file_sha256(Path(__file__).resolve()),
        "new_protocol_sha256": v2._file_sha256(PROTOCOL),
        "output_root": str(OUTPUT_ROOT.resolve()),
        "phase_counts": EXPECTED_COUNTS,
        "private_compute_device": "mps",
        "mps_fallback": 0,
        "server_postprocessing": "cpu_float64",
    }


def verify_parent_evidence(parent: v2.RCIGV2Campaign) -> None:
    """Recompute both parent gates from their actual runs at every invocation."""
    restricted = replace(
        parent, tasks=tuple(t for t in parent.tasks if t.phase_id in PARENT_PHASES)
    )
    records = analysis.collect_runs(restricted)
    invalid = [
        f"{r.task.phase_id}/{r.task.run_id}: {r.reason}"
        for r in records
        if r.status != "complete"
    ]
    if invalid:
        raise RuntimeError("invalid parent artifacts: " + "; ".join(invalid))
    for record in records:
        v2._post_run_device_audit(record.metrics_path, record.expected)
    for phase, evaluator in zip(
        PARENT_PHASES, (analysis._calibration_evidence, analysis._null_evidence)
    ):
        subset = [r for r in records if r.task.phase_id == phase]
        expected_count = 106 if phase == PARENT_PHASES[0] else 72
        if len(subset) != expected_count:
            raise RuntimeError(f"incorrect number of parent runs in {phase}")
        observed = evaluator(subset)
        gate = v2._verified_gate_payload(parent, phase)
        if gate["evidence_sha256"] != v2._canonical_hash(observed):
            raise RuntimeError(f"parent gate stale against current runs: {phase}")


def verify_lock(expected: Mapping[str, Any]) -> None:
    observed = _json(OUTPUT_ROOT / LOCK_NAME)
    if observed.get("provenance") != expected or observed.get(
        "scientific_hash"
    ) != v2._canonical_hash(expected):
        raise RuntimeError("exploratory provenance changed; refusing resume")


def ensure_lock(expected: Mapping[str, Any]) -> None:
    path = OUTPUT_ROOT / LOCK_NAME
    if path.exists():
        verify_lock(expected)
        return
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("refusing to adopt pre-existing unlocked results")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": dict(expected),
        "scientific_hash": v2._canonical_hash(expected),
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def execution_lock():
    """One launcher across both phases, even if two terminals start together."""
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT.parent / f".{CAMPAIGN_ID}.execution.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another exploratory R2/R3 launcher is active") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "campaign_id": CAMPAIGN_ID}))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def expected_config(parent, task, stamp):
    directory = OUTPUT_ROOT / task.phase_id / task.run_id
    config = v2.resolved_config(parent, task, directory)
    algo = config["training"]["algo_config"]
    algo.update(
        {
            "rcig_campaign_id": CAMPAIGN_ID,
            "rcig_campaign_scientific_hash": v2._canonical_hash(stamp),
            "rcig_exploratory_after_r1_stop": True,
            "rcig_parent_campaign_id": parent.matrix["campaign_id"],
            "rcig_parent_scientific_hash": v2.campaign_scientific_hash(parent),
            "rcig_exploratory_protocol_sha256": stamp["new_protocol_sha256"],
        }
    )
    v2._validate_resolved_config(config, task)
    return config


def _numeric_outcomes(payload: Mapping[str, Any]) -> None:
    required = ("test_accuracy", "test_loss")
    optional = (
        "client_accuracy_mean",
        "client_accuracy_variance_pct2",
        "worst20_accuracy_pct",
        "best20_worst20_gap_pct",
    )
    for row in payload["rounds"]:
        for name in (*required, *(key for key in optional if key in row)):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise RuntimeError(
                    f"nonfinite/missing outcome {name}, round {row.get('round_num')}"
                )


def read_record(parent, task, stamp) -> analysis.RunRecord:
    config = expected_config(parent, task, stamp)
    directory = Path(config["output_dir"])
    if not directory.exists() or not any(directory.iterdir()):
        return analysis.RunRecord(
            task, "missing", "not started", None, None, config, {}
        )
    config_path = directory / "resolved_config.yaml"
    status_path = directory / "orchestration_status.json"
    metrics_path = v2._metrics_path(directory)
    if not config_path.is_file() or not status_path.is_file() or metrics_path is None:
        raise RuntimeError(f"partial artifacts (never auto-restarted): {directory}")
    status = _json(status_path)
    if status.get("status") != "completed":
        raise RuntimeError(
            f"{directory}: status={status.get('status')} (never auto-restarted)"
        )
    if status.get("interpretation") != "exploratory_after_failed_R1_not_confirmatory":
        raise RuntimeError(f"missing exploratory identity: {directory}")
    payload = _json(metrics_path)
    observed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    errors = analysis._critical_protocol_errors(
        payload, config, task, status, observed, config_path, metrics_path
    )
    for key in (
        "rcig_exploratory_after_r1_stop",
        "rcig_parent_campaign_id",
        "rcig_parent_scientific_hash",
        "rcig_exploratory_protocol_sha256",
    ):
        if payload.get("config", {}).get(key) != config["training"]["algo_config"][key]:
            errors.append(f"exploratory_metadata_{key}")
    if errors:
        raise RuntimeError(f"{directory}: " + "; ".join(errors))
    v2._post_run_device_audit(metrics_path, config)
    _numeric_outcomes(payload)
    hashes = {
        "metrics_sha256": v2._file_sha256(metrics_path),
        "resolved_config_sha256": v2._file_sha256(config_path),
        "status_sha256": v2._file_sha256(status_path),
    }
    return analysis.RunRecord(
        task, "complete", "", metrics_path, payload, config, hashes
    )


def active_task(parent, task, stamp) -> dict[str, Any] | None:
    """Read-only display of a live task; never counts it as valid/completed.

    Resume still uses read_record, which rejects running/partial artifacts.
    """
    directory = OUTPUT_ROOT / task.phase_id / task.run_id
    path = directory / "orchestration_status.json"
    if not path.is_file():
        return None
    status = _json(path)
    if status.get("status") != "running":
        return None
    config_path = directory / "resolved_config.yaml"
    expected = expected_config(parent, task, stamp)
    if (
        not config_path.is_file()
        or yaml.safe_load(config_path.read_text(encoding="utf-8")) != expected
        or status.get("resolved_config_sha256") != v2._file_sha256(config_path)
        or status.get("campaign_id") != CAMPAIGN_ID
        or status.get("scientific_hash") != v2._canonical_hash(stamp)
        or status.get("phase") != task.phase_id
        or status.get("run_id") != task.run_id
        or status.get("device") != "mps"
    ):
        raise RuntimeError(f"invalid running-task identity/configuration: {directory}")
    pid = status.get("launcher_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f"running task lacks a valid launcher PID: {directory}")
    try:
        os.kill(pid, 0)  # Existence check only; no signal is delivered.
    except ProcessLookupError:
        raise RuntimeError(
            f"stale running status; launcher {pid} is absent: {directory}"
        ) from None
    except PermissionError:
        raise RuntimeError(f"cannot verify whether launcher {pid} is alive") from None
    return {
        "run_id": task.run_id,
        "status": "running",
        "launcher_pid": pid,
        "log_path": str(directory / "run.log"),
        "final_artifact_validation": "pending",
    }


def run_task(parent, task, stamp):
    current = provenance(parent)
    if current != stamp:
        raise RuntimeError("parent/protocol/source drift before run")
    verify_lock(current)
    record = read_record(parent, task, stamp)
    if record.status == "complete":
        print(f"SKIP verified {task.phase_id}/{task.run_id}", flush=True)
        return record
    config = record.expected
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "resolved_config.yaml"
    with config_path.open("x", encoding="utf-8") as handle:
        handle.write(v2._serialized_config(config))
    algo = config["training"]["algo_config"]
    attack = algo["attack"]
    status = {
        "campaign_id": CAMPAIGN_ID,
        "phase": task.phase_id,
        "run_id": task.run_id,
        "status": "running",
        "device": "mps",
        "mps_fallback": 0,
        "server_postprocessing": "cpu_float64",
        "scientific_hash": v2._canonical_hash(stamp),
        "interpretation": stamp["interpretation"],
        "resolved_config_sha256": v2._file_sha256(config_path),
        "pairing_seed_block": task.seed,
        "pairing_design_sha256": algo["rcig_pairing_design_sha256"],
        "attack_ids": attack.get("client_ids", []),
        "attack_start": attack.get("active_round_start"),
        "attack_end": attack.get("active_round_end"),
        "launcher_pid": os.getpid(),
    }
    status_path = directory / "orchestration_status.json"
    v2._write_status(status_path, status)
    v2._write_status(OUTPUT_ROOT / "_status.json", status)
    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "run_experiment.py"),
        "--config",
        str(config_path),
        "--device",
        "mps",
        "--output",
        str(directory),
    ]
    print(
        f"RUN {task.phase_id} {task.phase_index + 1}/{EXPECTED_COUNTS[task.phase_id]} {task.run_id}",
        flush=True,
    )
    try:
        with (directory / "run.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        metrics_path = v2._metrics_path(directory)
        if metrics_path is None:
            raise RuntimeError(
                "training exited without exactly one 40-round metrics file"
            )
        if provenance(parent) != stamp:
            raise RuntimeError("parent/protocol/source drift during run")
        verify_lock(stamp)
        status.update(status="completed", metrics_sha256=v2._file_sha256(metrics_path))
        # Validate against the candidate completed status before publishing it.
        payload = _json(metrics_path)
        observed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        errors = analysis._critical_protocol_errors(
            payload, config, task, status, observed, config_path, metrics_path
        )
        if errors:
            raise RuntimeError("post-run protocol audit: " + "; ".join(errors))
        v2._post_run_device_audit(metrics_path, config)
        _numeric_outcomes(payload)
        v2._write_status(status_path, status)
        record = read_record(parent, task, stamp)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        v2._write_status(status_path, status)
        v2._write_status(OUTPUT_ROOT / "_status.json", status)
        raise
    v2._write_status(OUTPUT_ROOT / "_status.json", status)
    print(f"COMPLETE {task.phase_id}/{task.run_id}", flush=True)
    return record


def diagnostic(parent, phase, records, stamp):
    if len(records) != EXPECTED_COUNTS[phase] or any(
        r.status != "complete" for r in records
    ):
        raise RuntimeError(f"cannot analyze incomplete {phase}")
    evaluator = (
        analysis._mechanistic_evidence
        if phase == PHASES[0]
        else analysis._confirmation_evidence
    )
    evidence = evaluator(records)
    evaluations = []
    for criterion in v2.phase_definition(parent, phase)["gate_criteria"]:
        value = evidence.get(criterion["id"])
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError(f"invalid diagnostic criterion {criterion['id']}")
        threshold, op = criterion["threshold"], criterion["op"]
        passed = (
            math.isclose(value, threshold, abs_tol=1e-12)
            if op == "=="
            else (value <= threshold if op == "<=" else value >= threshold)
        )
        evaluations.append({**criterion, "observed": value, "passed": passed})
    return {
        "campaign_id": CAMPAIGN_ID,
        "phase": phase,
        "interpretation": stamp["interpretation"],
        "decision": "exploratory_complete_not_promoted",
        "diagnostic_criteria_all_pass": all(x["passed"] for x in evaluations),
        "parent_r1_decision": "stop",
        "evaluations": evaluations,
        "evidence": evidence,
        "evidence_sha256": v2._canonical_hash(evidence),
        "scientific_hash": v2._canonical_hash(stamp),
    }


def _write_diagnostic(path, payload):
    if path.exists():
        if _json(path) != payload:
            raise RuntimeError(f"immutable exploratory diagnostic differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_chain(parent, stamp):
    for phase in PHASES:
        tasks = v2.tasks_for_phase(parent, phase)
        if len(tasks) != EXPECTED_COUNTS[phase]:
            raise RuntimeError(f"unexpected task count for {phase}")
        for task in tasks:
            run_task(parent, task, stamp)
        # Revalidate every completed artifact, not just in-memory summaries.
        records = [read_record(parent, task, stamp) for task in tasks]
        result = diagnostic(parent, phase, records, stamp)
        _write_diagnostic(OUTPUT_ROOT / "_diagnostics" / f"{phase}.json", result)
        print(
            f"{phase}: {len(records)}/{len(tasks)} valid; exploratory diagnostic only, no promotion",
            flush=True,
        )
    v2._write_status(
        OUTPUT_ROOT / "_status.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "status": "completed",
            "phase_counts": EXPECTED_COUNTS,
            "interpretation": stamp["interpretation"],
            "parent_r1_decision": "stop",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--acknowledge-r1-stop", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="read-only status (default is read-only plan)",
    )
    parser.add_argument("--device", default="mps", choices=["mps"])
    args = parser.parse_args(argv)
    if args.run and args.status:
        parser.error("--run and --status are mutually exclusive")
    if args.run and not (args.resume and args.acknowledge_r1_stop):
        parser.error("execution requires --resume --acknowledge-r1-stop")
    parent = v2.load_campaign()
    stamp = provenance(parent)
    if (OUTPUT_ROOT / LOCK_NAME).exists():
        verify_lock(stamp)
    elif OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("pre-existing results have no exploratory provenance lock")
    if not args.run:
        report = {
            "campaign_id": CAMPAIGN_ID,
            "read_only": True,
            "parent_r1_decision": "stop",
            "output_root": str(OUTPUT_ROOT),
            "interpretation": stamp["interpretation"],
            "phases": {},
        }
        for phase in PHASES:
            complete, missing, invalid, active = 0, 0, [], []
            for task in v2.tasks_for_phase(parent, phase):
                try:
                    running = active_task(parent, task, stamp)
                    if running is not None:
                        active.append(running)
                        continue
                    record = read_record(parent, task, stamp)
                    complete += record.status == "complete"
                    missing += record.status == "missing"
                except RuntimeError as exc:
                    invalid.append({"run_id": task.run_id, "reason": str(exc)})
            report["phases"][phase] = {
                "valid_complete": complete,
                "not_started": missing,
                "expected": EXPECTED_COUNTS[phase],
                "active": active,
                "invalid": invalid,
            }
        print(json.dumps(report, indent=2))
        return 0
    with execution_lock():
        v2.require_working_mps()
        verify_parent_evidence(parent)
        # Re-read after the potentially expensive evidence check.
        if provenance(parent) != stamp:
            raise RuntimeError("parent changed during preflight")
        ensure_lock(stamp)
        # Fail before launching anything when any previous task is partial or
        # invalid; do not hide an old failure behind other newly completed jobs.
        for phase in PHASES:
            for task in v2.tasks_for_phase(parent, phase):
                read_record(parent, task, stamp)
        run_chain(parent, stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
