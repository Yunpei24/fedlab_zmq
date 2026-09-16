"""Small, no-training tests for the explicitly exploratory RCIG continuation."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_rcig_r2_r3_exploratory as runner


@pytest.fixture(scope="module")
def parent():
    return runner.v2.load_campaign()


@pytest.fixture
def stamp():
    return {
        "new_protocol_sha256": "test-protocol",
        "interpretation": "exploratory_after_failed_R1_not_confirmatory",
    }


def test_exact_tasks_and_configs_change_only_identity_and_output(
    parent, stamp, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "exploration")
    for phase, total in zip(runner.PHASES, (144, 384)):
        tasks = runner.v2.tasks_for_phase(parent, phase)
        assert len(tasks) == total
        for task in tasks:
            current = runner.expected_config(parent, task, stamp)
            original = runner.v2.resolved_config(
                parent, task, Path(current["output_dir"])
            )
            current_algo = current["training"]["algo_config"]
            original_algo = original["training"]["algo_config"]
            assert current_algo["rcig_campaign_id"] == runner.CAMPAIGN_ID
            assert current_algo["rcig_exploratory_after_r1_stop"] is True
            for key in ("rcig_campaign_id", "rcig_campaign_scientific_hash"):
                current_algo[key] = original_algo[key]
            for key in (
                "rcig_exploratory_after_r1_stop",
                "rcig_parent_campaign_id",
                "rcig_parent_scientific_hash",
                "rcig_exploratory_protocol_sha256",
            ):
                current_algo.pop(key)
            assert current == original


@pytest.mark.parametrize(
    "args",
    [
        ["--run"],
        ["--run", "--resume"],
        ["--run", "--acknowledge-r1-stop"],
        ["--run", "--status", "--resume", "--acknowledge-r1-stop"],
        ["--device", "cpu"],
    ],
)
def test_requires_explicit_acknowledgment_resume_and_mps(args):
    with pytest.raises(SystemExit):
        runner.main(args)


def test_lock_detects_drift_and_rejects_unlocked_existing_results(
    tmp_path, monkeypatch, stamp
):
    root = tmp_path / "out"
    monkeypatch.setattr(runner, "OUTPUT_ROOT", root)
    runner.ensure_lock(stamp)
    runner.verify_lock(stamp)
    changed = dict(stamp, source="changed")
    with pytest.raises(RuntimeError, match="provenance changed"):
        runner.verify_lock(changed)
    second = tmp_path / "existing"
    second.mkdir()
    (second / "metrics.json").write_text("{}")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", second)
    with pytest.raises(RuntimeError, match="pre-existing unlocked"):
        runner.ensure_lock(stamp)


def test_execution_lock_rejects_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    with runner.execution_lock():
        with pytest.raises(RuntimeError, match="another exploratory"):
            with runner.execution_lock():
                pytest.fail("duplicate launcher acquired lock")


def test_missing_is_distinct_from_partial_or_failed(
    parent, stamp, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    task = runner.v2.tasks_for_phase(parent, runner.PHASES[0])[0]
    assert runner.read_record(parent, task, stamp).status == "missing"
    directory = runner.OUTPUT_ROOT / task.phase_id / task.run_id
    directory.mkdir(parents=True)
    (directory / "orchestration_status.json").write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError, match="never auto-restarted"):
        runner.read_record(parent, task, stamp)


def test_source_drift_stops_before_subprocess(parent, stamp, monkeypatch):
    monkeypatch.setattr(runner, "provenance", lambda p: {"different": True})
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched training")
    )
    task = runner.v2.tasks_for_phase(parent, runner.PHASES[0])[0]
    with pytest.raises(RuntimeError, match="drift before run"):
        runner.run_task(parent, task, stamp)


def test_nan_outcomes_stop_but_finite_accuracy_degradation_is_valid():
    payload = {"rounds": [{"round_num": 1, "test_accuracy": 0.0, "test_loss": 100.0}]}
    runner._numeric_outcomes(payload)
    payload["rounds"][0]["test_loss"] = float("nan")
    with pytest.raises(RuntimeError, match="nonfinite"):
        runner._numeric_outcomes(payload)


def test_diagnostic_never_records_promotion(parent, stamp, monkeypatch):
    phase = runner.PHASES[0]
    tasks = runner.v2.tasks_for_phase(parent, phase)
    records = [
        runner.analysis.RunRecord(t, "complete", "", None, {}, {}, {}) for t in tasks
    ]
    evidence = {
        x["id"]: 0.0 for x in runner.v2.phase_definition(parent, phase)["gate_criteria"]
    }
    monkeypatch.setattr(runner.analysis, "_mechanistic_evidence", lambda rs: evidence)
    payload = runner.diagnostic(parent, phase, records, stamp)
    assert payload["decision"] == "exploratory_complete_not_promoted"
    assert payload["diagnostic_criteria_all_pass"] is False
    assert payload["parent_r1_decision"] == "stop"
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.diagnostic(parent, phase, records[:-1], stamp)


def test_r3_follows_all_valid_r2_even_negative_diagnostic(
    parent, stamp, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    tasks = tuple(
        t
        for phase in runner.PHASES
        for t in runner.v2.tasks_for_phase(parent, phase)[:2]
    )
    mini = replace(parent, tasks=tasks)
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", dict.fromkeys(runner.PHASES, 2))
    observed = []

    def record(parent, task, stamp):
        return runner.analysis.RunRecord(task, "complete", "", None, {}, {}, {})

    def run(parent, task, stamp):
        observed.append(task.phase_id)
        return record(parent, task, stamp)

    def diagnostic(parent, phase, records, stamp):
        assert len(records) == 2
        return {
            "decision": "exploratory_complete_not_promoted",
            "diagnostic_criteria_all_pass": False,
        }

    monkeypatch.setattr(runner, "run_task", run)
    monkeypatch.setattr(runner, "read_record", record)
    monkeypatch.setattr(runner, "diagnostic", diagnostic)
    runner.run_chain(mini, stamp)
    assert observed == [runner.PHASES[0]] * 2 + [runner.PHASES[1]] * 2


def test_invalid_r2_stops_before_any_r3(parent, stamp, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    observed = []

    def invalid_run(parent, task, stamp):
        observed.append(task.phase_id)
        raise RuntimeError("invalid protocol")

    monkeypatch.setattr(runner, "run_task", invalid_run)
    with pytest.raises(RuntimeError, match="invalid protocol"):
        runner.run_chain(parent, stamp)
    assert observed == [runner.PHASES[0]]


def test_read_only_status_creates_no_output(
    parent, stamp, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(runner.v2, "load_campaign", lambda: parent)
    monkeypatch.setattr(runner, "provenance", lambda p: stamp)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched training")
    )
    assert runner.main(["--status"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["read_only"] is True
    assert report["phases"][runner.PHASES[0]]["not_started"] == 144
    assert not runner.OUTPUT_ROOT.exists()


def test_parent_source_lock_is_unchanged_and_complete(parent):
    before = copy.deepcopy(runner.v2._campaign_lock_payload(parent))
    assert len(before["source_sha256"]) == 25
    runner.v2.verify_existing_campaign_lock(parent)
    for phase, decision in zip(runner.PARENT_PHASES, ("promote", "stop")):
        assert runner.v2._verified_gate_payload(parent, phase)["decision"] == decision
    after = runner.v2._campaign_lock_payload(parent)
    assert after == before


def test_diagnostics_are_immutable(tmp_path):
    path = tmp_path / "diagnostic.json"
    payload = {"decision": "exploratory_complete_not_promoted"}
    runner._write_diagnostic(path, payload)
    runner._write_diagnostic(path, payload)
    with pytest.raises(RuntimeError, match="immutable"):
        runner._write_diagnostic(path, {"decision": "promote"})


def test_mocked_execution_audits_and_resumes_without_duplicate(
    parent, stamp, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(runner, "provenance", lambda p: stamp)
    monkeypatch.setattr(runner, "verify_lock", lambda s: None)
    task = runner.v2.tasks_for_phase(parent, runner.PHASES[0])[0]
    calls = {"execute": 0, "critical": 0, "device": 0}

    def fake_training(command, **kwargs):
        calls["execute"] += 1
        assert kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        assert command[command.index("--device") + 1] == "mps"
        config_path = Path(command[command.index("--config") + 1])
        config = runner.yaml.safe_load(config_path.read_text())
        payload = {
            "config": config["training"]["algo_config"],
            "rounds": [
                {"round_num": t + 1, "test_accuracy": 0.4, "test_loss": 2.0}
                for t in range(40)
            ],
        }
        (config_path.parent / "metrics.json").write_text(json.dumps(payload))

    def critical(payload, expected, task, status, observed, config_path, metrics_path):
        calls["critical"] += 1
        assert expected == observed
        assert status["scientific_hash"] == runner.v2._canonical_hash(stamp)
        assert status["metrics_sha256"] == runner.v2._file_sha256(metrics_path)
        assert status["campaign_id"] == runner.CAMPAIGN_ID
        return []

    def device_audit(metrics_path, config):
        calls["device"] += 1

    monkeypatch.setattr(runner.subprocess, "run", fake_training)
    monkeypatch.setattr(runner.analysis, "_critical_protocol_errors", critical)
    monkeypatch.setattr(runner.v2, "_post_run_device_audit", device_audit)
    first = runner.run_task(parent, task, stamp)
    second = runner.run_task(parent, task, stamp)
    assert first.status == second.status == "complete"
    assert calls == {"execute": 1, "critical": 3, "device": 3}
    # A changed outcome cannot be silently skipped on resume.
    payload = json.loads(first.metrics_path.read_text())
    payload["rounds"][0]["test_accuracy"] = float("nan")
    first.metrics_path.write_text(json.dumps(payload))
    with pytest.raises((AssertionError, RuntimeError)):
        runner.run_task(parent, task, stamp)
    assert calls["execute"] == 1


def test_status_reports_live_running_without_adopting_it(
    parent, stamp, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "out")
    task = runner.v2.tasks_for_phase(parent, runner.PHASES[0])[0]
    config = runner.expected_config(parent, task, stamp)
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True)
    config_path = directory / "resolved_config.yaml"
    config_path.write_text(runner.v2._serialized_config(config))
    status = {
        "status": "running",
        "launcher_pid": 123,
        "campaign_id": runner.CAMPAIGN_ID,
        "device": "mps",
        "phase": task.phase_id,
        "run_id": task.run_id,
        "scientific_hash": runner.v2._canonical_hash(stamp),
        "resolved_config_sha256": runner.v2._file_sha256(config_path),
    }
    (directory / "orchestration_status.json").write_text(json.dumps(status))
    monkeypatch.setattr(runner.os, "kill", lambda pid, signal: None)
    assert runner.active_task(parent, task, stamp)["status"] == "running"
    with pytest.raises(RuntimeError, match="never auto-restarted"):
        runner.read_record(parent, task, stamp)

    def stale(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(runner.os, "kill", stale)
    with pytest.raises(RuntimeError, match="stale running"):
        runner.active_task(parent, task, stamp)
