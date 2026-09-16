"""No-training tests for the isolated 216-run screen and fail-closed artifacts."""

import copy
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import run_rcig_batch_screen as runner


@pytest.fixture(scope="module")
def original():
    return runner.load_campaign()


@pytest.fixture
def campaign(original, tmp_path):
    return replace(original, output_root=tmp_path / "screen")


@pytest.fixture
def stamp():
    return {
        "campaign_id": runner.CAMPAIGN_ID,
        "source_sha256": {
            "scripts/run_rcig_batch_screen_experiment.py": "wrapper",
            "algorithms/ldp_gradient_far_recent.py": "recent",
            "run_experiment.py": "experiment",
            "algorithms/__init__.py": "init",
            "privacy/rdp.py": "accountant",
        },
    }


def metrics_payload(config, task):
    """Synthetic artifact only; never imports or simulates a private model."""
    rounds = []
    for index in range(1, 41):
        active = task.scenario != "none" and index >= 17
        row = {
            "round_num": index,
            "test_accuracy": 0.0,
            "test_loss": 100.0,
            "client_accuracy_mean": 0.1,
            "client_accuracy_variance_pct2": 2.0,
            "worst20_accuracy_pct": 1.0,
            "best20_worst20_gap_pct": 20.0,
            "num_clients": 25,
            "num_selected": 25,
            "num_survivors": 25,
            "num_alive_clients": 25,
            "num_pre_training_dropouts": 0,
            "participation_rate": 1.0,
            "survival_ratio": 1.0,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "privacy_delta": 1e-5,
            "privacy_model_noise_multiplier_min": runner.calibrated_sigma(
                task.batch_size
            ),
            "privacy_model_noise_multiplier_max": runner.calibrated_sigma(
                task.batch_size
            )
            * (1 if task.noise_regime == "homogeneous" else 2),
            "privacy_model_steps_mean": 1.0,
            "privacy_noise_scale_public_min": 1.0,
            "privacy_noise_scale_public_max": (
                1.0 if task.noise_regime == "homogeneous" else 2.0
            ),
            "privacy_model_noise_multiplier_mean": runner.calibrated_sigma(
                task.batch_size
            )
            * (1.0 if task.noise_regime == "homogeneous" else 37.0 / 25.0),
            "privacy_epsilon_max": runner.epsilon_at(task.batch_size, index),
            "privacy_epsilon_mean": (
                runner.epsilon_at(task.batch_size, index)
                if task.noise_regime == "homogeneous"
                else (
                    13 * runner.epsilon_at(task.batch_size, index)
                    + 12 * runner.epsilon_at(task.batch_size, index, 2)
                )
                / 25
            ),
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "ldp_gradient_far_private_compute_device": "mps",
            "far_attack_labels_visible_to_server_aggregate": False,
            "far_attack_config_visible_to_server_aggregate": False,
            "far_external_attack_diagnostics": True,
            "far_external_attack_diagnostics_boundary": "posthoc_simulator_only",
            "attack_window_active": active,
            "num_byzantine_oracle": 5 if active else 0,
            "byzantine_weight_mass_oracle": 0.2,
            "ldp_gradient_far_effective_alpha": (
                0.0
                if task.reference == "uniform"
                or (task.reference == "recent" and index <= 12)
                else 0.1
            ),
        }
        if task.reference == "recent":
            row.update(
                {
                    "rcig_reference_mode": "recent_only",
                    "rcig_covariance_mode": "full",
                    "rcig_deployed_candidate": "identity_new",
                    "rcig_temporal_correction_enabled": False,
                    "rcig_innovation_test_enabled": False,
                    "ldp_gradient_far_reference_noise_aware": False,
                    "rcig_server_aggregation_device": "cpu",
                    "rcig_server_aggregation_dtype": "torch.float64",
                    "rcig_private_gradient_compute_device": "mps",
                    "rcig_private_gradient_mps_fraction": 1.0,
                    "rcig_history_ready": index > 12,
                    "rcig_reference_strictly_past": True,
                    "rcig_gate_source_round_min": index - 13,
                    "rcig_gate_source_round_max": index - 10,
                    "rcig_older_round_min": index - 9,
                    "rcig_older_round_max": index - 6,
                    "rcig_newer_round_min": index - 5,
                    "rcig_newer_round_max": index - 2,
                    "rcig_identity_new_squared_l2_error_to_clean_honest_center_oracle": 2.0,
                    "rcig_reference_squared_l2_error_to_clean_honest_center_oracle": 2.0,
                }
            )
        rounds.append(row)
    return {
        "algorithm": config["training"]["algorithm"],
        "config": {**copy.deepcopy(config["training"]["algo_config"]), "device": "mps"},
        "summary": {
            "num_rounds": 40,
            "seed": task.seed,
            "partition_seed": task.seed,
            "dataset": "fashionmnist",
            "model": "lenet5_tanh",
            "num_clients": 25,
        },
        "rounds": rounds,
    }


def write_complete(campaign, task, stamp):
    directory = runner.task_output_dir(campaign, task)
    directory.mkdir(parents=True)
    config = runner.resolved_config(campaign, task, stamp)
    config_path = directory / "resolved_config.yaml"
    config_path.write_text(runner.yaml.safe_dump(config))
    metrics = directory / "metrics.json"
    metrics.write_text(json.dumps(metrics_payload(config, task)))
    runtime = directory / "runtime_imports.json"
    runtime.write_text(
        json.dumps(
            {
                "stage": "after_training",
                "device": "mps",
                "mps_fallback": 0,
                "source_sha256": stamp["source_sha256"],
            }
        )
    )
    status = {
        **runner._status_identity(campaign, task, stamp, config_path),
        "status": "completed",
        "metrics_sha256": runner.file_hash(metrics),
        "runtime_imports_sha256": runner.file_hash(runtime),
    }
    runner.write_json(directory / "orchestration_status.json", status, exclusive=True)
    return directory


def test_exact_216_unique_cells_and_user_seeds(original):
    tasks = original.tasks
    assert len(tasks) == len({t.run_id for t in tasks}) == 216
    assert {t.seed for t in tasks} == {24, 42, 72, 121}
    assert Counter(t.batch_size for t in tasks) == {120: 72, 240: 72, 480: 72}
    assert Counter(t.reference for t in tasks) == {
        "uniform": 72,
        "rfa": 72,
        "recent": 72,
    }
    assert Counter(t.noise_regime for t in tasks) == {
        "homogeneous": 108,
        "heteroscedastic": 108,
    }
    assert Counter(t.scenario for t in tasks) == {
        "none": 72,
        "bf_x10_persistent": 72,
        "ipm_persistent": 72,
    }
    blocks = defaultdict(list)
    for task in tasks:
        blocks[task.pairing_block].append(task)
    assert len(blocks) == 72
    for block in blocks.values():
        assert [t.reference for t in block] == ["uniform", "rfa", "recent"]
        assert [t.global_index for t in block] == list(
            range(block[0].global_index, block[0].global_index + 3)
        )


def test_every_resolved_config_matches_B_noise_attack_and_pairing(campaign, stamp):
    fingerprints = defaultdict(set)
    for task in campaign.tasks:
        config = runner.resolved_config(campaign, task, stamp)
        runner.validate_resolved_config(config, task)
        algo = config["training"]["algo_config"]
        assert algo["privacy_sampling_rate_override"] == task.batch_size / 2400
        assert algo["batch_size"] == algo["fixed_batch_size"] == task.batch_size
        assert len(algo["privacy_noise_multiplier_scale_by_client"]) == 25
        assert config["seed"] == config["data"]["partition_seed"] == task.seed
        assert algo["client_metrics_every"] == 1
        assert algo["enable_oracle_diagnostics"] == (task.reference == "recent")
        assert algo["rcig_oracle_evaluation_only"] == (task.reference == "recent")
        assert algo["rcig_oracle_metrics_are_not_release"] == (
            task.reference == "recent"
        )
        assert "rcig_innovation_threshold" not in algo
        fingerprints[task.pairing_block].add(algo["rcig_pairing_design_sha256"])
        if task.scenario != "none":
            assert algo["attack"]["active_round_start"] == 17
            assert algo["attack"]["active_round_end"] == 40
            assert algo["attack"]["client_ids"] == list(range(5))
    assert all(len(values) == 1 for values in fingerprints.values())


def test_recalibrated_sigmas_and_replace_one_sensitivity():
    table = runner.privacy_table()
    assert len(table) == 3
    for row in table:
        assert row["max_epsilon_absolute_error"] <= 1e-4
        assert row["epsilon_scale_2"] < row["epsilon_scale_1"]
        assert row["sd_scale_1"] == 4 * row["sigma"] / row["batch_size"]
        accountant = runner.rdp_module().RDPAccountant()
        accountant.add_sampled_without_replacement_gaussian(
            channel="test",
            sampling_rate=row["q"],
            noise_multiplier=row["sigma"] / 2,
            steps=40,
        )
        assert accountant.epsilon(1e-5)[0] == row["epsilon_scale_1"]
    assert table[0]["sigma"] < table[1]["sigma"] < table[2]["sigma"]


@pytest.mark.parametrize("reference", ["uniform", "rfa", "recent"])
@pytest.mark.parametrize("batch", [120, 240, 480])
def test_valid_low_accuracy_is_a_valid_negative_finding(
    campaign, stamp, reference, batch
):
    task = next(
        t
        for t in campaign.tasks
        if t.reference == reference
        and t.batch_size == batch
        and t.noise_regime == "heteroscedastic"
        and t.scenario == "bf_x10_persistent"
    )
    config = runner.resolved_config(campaign, task, stamp)
    runner.validate_metrics(metrics_payload(config, task), config, task)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["rounds"].pop(),
        lambda p: p["rounds"][3].update(round_num=2),
        lambda p: p["rounds"][4].update(test_loss=float("nan")),
        lambda p: p["rounds"][5].update(ldp_gradient_far_private_compute_device="cpu"),
        lambda p: p["rounds"][6].update(
            ldp_gradient_far_private_gradient_mps_fraction=0.96
        ),
        lambda p: p["rounds"][7].update(num_alive_clients=24),
        lambda p: p["rounds"][8].update(privacy_epsilon_max=4.0),
        lambda p: p["rounds"][9].update(privacy_model_noise_multiplier_min=1.0),
        lambda p: p["rounds"][16].update(attack_window_active=False),
        lambda p: p["rounds"][17].update(
            far_attack_labels_visible_to_server_aggregate=True
        ),
        lambda p: p["rounds"][18].update(rcig_innovation_test_enabled=True),
        lambda p: p["rounds"][19].update(rcig_newer_round_max=20),
        lambda p: p["rounds"][20].update(ldp_gradient_far_effective_alpha=0.0),
        lambda p: p["rounds"][20].update(privacy_epsilon_mean=123.0),
        lambda p: p["rounds"][20].update(
            rcig_reference_squared_l2_error_to_clean_honest_center_oracle=3.0
        ),
        lambda p: p["config"].update(fixed_batch_size=120),
    ],
)
def test_invalid_device_privacy_rounds_population_or_recent_control_fails(
    campaign, stamp, mutation
):
    task = next(
        t
        for t in campaign.tasks
        if t.reference == "recent"
        and t.batch_size == 240
        and t.scenario == "bf_x10_persistent"
    )
    config = runner.resolved_config(campaign, task, stamp)
    payload = metrics_payload(config, task)
    mutation(payload)
    with pytest.raises(RuntimeError):
        runner.validate_metrics(payload, config, task)


def test_locked_sources_detect_drift_and_unlocked_artifacts_are_not_adopted(
    campaign, stamp
):
    runner.ensure_lock(campaign, stamp)
    runner.verify_lock(campaign, stamp)
    with pytest.raises(RuntimeError, match="provenance drift"):
        runner.verify_lock(campaign, {**stamp, "different": True})
    other = replace(campaign, output_root=campaign.output_root.parent / "unlocked")
    other.output_root.mkdir()
    (other.output_root / "foreign.json").write_text("{}")
    with pytest.raises(RuntimeError, match="unlocked"):
        runner.ensure_lock(other, stamp)


def test_one_global_execution_lock_also_covers_job_subsets(campaign):
    with runner.execution_lock(campaign):
        with pytest.raises(RuntimeError, match="global execution lock"):
            with runner.execution_lock(campaign):
                pytest.fail("Duplicate launcher acquired execution lock")


def test_resume_accepts_only_complete_valid_artifacts(campaign, stamp):
    task = campaign.tasks[0]
    assert runner.read_record(campaign, task, stamp)["status"] == "missing"
    directory = write_complete(campaign, task, stamp)
    assert runner.read_record(campaign, task, stamp)["status"] == "complete"
    metrics = directory / "metrics.json"
    metrics.write_text(metrics.read_text() + " ")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner.read_record(campaign, task, stamp)


def test_partial_results_are_not_restarted(campaign, stamp):
    directory = runner.task_output_dir(campaign, campaign.tasks[0])
    directory.mkdir(parents=True)
    (directory / "run.log").write_text("interrupted")
    with pytest.raises(RuntimeError, match="never automatically restarted"):
        runner.read_record(campaign, campaign.tasks[0], stamp)


def test_runtime_manifest_must_be_final_and_every_actual_import_locked(tmp_path, stamp):
    path = tmp_path / "runtime_imports.json"
    data = {
        "stage": "after_training",
        "device": "mps",
        "mps_fallback": 0,
        "source_sha256": stamp["source_sha256"],
    }
    path.write_text(json.dumps(data))
    runner.validate_runtime_imports(path, stamp)
    data["stage"] = "before_training"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        runner.validate_runtime_imports(path, stamp)
    data["stage"] = "after_training"
    data["source_sha256"] = {**data["source_sha256"], "unlocked.py": "hash"}
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Actual imported dependency"):
        runner.validate_runtime_imports(path, stamp)


def test_source_closure_includes_transitive_registry_and_energy_dependencies():
    sources = runner.source_closure((runner.ROOT / "run_experiment.py",))
    assert {
        "algorithms/__init__.py",
        "algorithms/ldp_gradient_far.py",
        "algorithms/far.py",
        "privacy/rdp.py",
        "hardware/energy_model.py",
        "core/seeding.py",
        "datasets/partitioner.py",
    }.issubset(sources)


def test_status_is_read_only_without_training_imports(
    campaign, stamp, monkeypatch, capsys
):
    monkeypatch.setattr(runner, "load_campaign", lambda: campaign)
    monkeypatch.setattr(runner, "provenance", lambda c: stamp)
    monkeypatch.setattr(
        runner, "require_working_mps", lambda: pytest.fail("MPS imported during status")
    )
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Training launched")
    )
    assert runner.main(["--status"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["missing"] == report["expected"] == 216
    assert report["valid_complete"] == 0
    assert report["active"] == report["invalid"] == []
    assert not campaign.output_root.exists()


def test_read_only_plan_does_not_import_torch_or_write():
    code = """
import os, sys
def audit(event, args):
    if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
        raise AssertionError('filesystem write')
sys.addaudithook(audit)
from scripts import run_rcig_batch_screen as runner
runner.main(['--plan'])
assert 'torch' not in sys.modules
assert 'privacy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=runner.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["expected"] == 216


@pytest.mark.parametrize(
    "args",
    [
        ["--run"],
        ["--run", "--device", "cpu"],
        ["--detach"],
        ["--status", "--run"],
        ["--run", "--resume", "--device", "cuda"],
    ],
)
def test_execution_is_explicit_resume_and_mps_only(args):
    with pytest.raises(SystemExit):
        runner.main(args)


def test_job_selection_never_changes_the_full_campaign(original):
    assert runner.selected_tasks(original, None, None) == original.tasks
    assert runner.selected_tasks(original, 4, None) == (original.tasks[4],)
    assert runner.selected_tasks(original, 4, 3) == original.tasks[4:7]
    with pytest.raises(ValueError):
        runner.selected_tasks(original, 216, None)


def test_source_drift_stops_before_any_training(campaign, stamp, monkeypatch):
    monkeypatch.setattr(runner, "provenance", lambda c: {"changed": True})
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Training launched")
    )
    with pytest.raises(RuntimeError, match="Source drift before run"):
        runner.run_task(campaign, campaign.tasks[0], stamp)
    assert not campaign.output_root.exists()


def test_completed_resume_never_launches_subprocess(campaign, stamp, monkeypatch):
    runner.ensure_lock(campaign, stamp)
    task = campaign.tasks[0]
    write_complete(campaign, task, stamp)
    monkeypatch.setattr(runner, "provenance", lambda c: stamp)
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Training launched")
    )
    assert runner.run_task(campaign, task, stamp)["status"] == "complete"


def test_mocked_training_child_audits_full_artifacts_and_resumes_once(
    campaign, stamp, monkeypatch
):
    task = next(
        t
        for t in campaign.tasks
        if t.batch_size == 480
        and t.reference == "recent"
        and t.noise_regime == "heteroscedastic"
    )
    runner.ensure_lock(campaign, stamp)
    monkeypatch.setattr(runner, "provenance", lambda c: stamp)
    launches = []

    class ArtifactOnlyChild:
        pid = 987655

        def __init__(self, command, **kwargs):
            launches.append(command)
            assert command[command.index("--device") + 1] == "mps"
            assert kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
            assert kwargs["env"]["PYTHONHASHSEED"] == str(task.seed)
            directory = runner.task_output_dir(campaign, task)
            config = runner.yaml.safe_load(
                (directory / "resolved_config.yaml").read_text()
            )
            (directory / "metrics.json").write_text(
                json.dumps(metrics_payload(config, task))
            )
            (directory / "runtime_imports.json").write_text(
                json.dumps(
                    {
                        "stage": "after_training",
                        "device": "mps",
                        "mps_fallback": 0,
                        "source_sha256": stamp["source_sha256"],
                    }
                )
            )

        def wait(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", ArtifactOnlyChild)
    assert runner.run_task(campaign, task, stamp)["status"] == "complete"
    assert runner.run_task(campaign, task, stamp)["status"] == "complete"
    assert len(launches) == 1
    status = runner.read_json(
        runner.task_output_dir(campaign, task) / "orchestration_status.json"
    )
    assert status["child_pid"] == ArtifactOnlyChild.pid
    assert status["metrics_sha256"]
    assert status["runtime_imports_sha256"]


def test_partial_unselected_run_stops_before_subset_launch(
    campaign, stamp, monkeypatch
):
    runner.ensure_lock(campaign, stamp)
    partial = runner.task_output_dir(campaign, campaign.tasks[-1])
    partial.mkdir(parents=True)
    (partial / "run.log").write_text("interrupted before status")
    monkeypatch.setattr(runner, "load_campaign", lambda: campaign)
    monkeypatch.setattr(runner, "provenance", lambda c: stamp)
    monkeypatch.setattr(runner, "check_no_other_training_process", lambda: None)
    monkeypatch.setattr(runner, "require_working_mps", lambda: None)
    monkeypatch.setattr(
        runner,
        "run_task",
        lambda *a, **k: pytest.fail("Any actual training is forbidden in this test"),
    )
    with pytest.raises(RuntimeError, match="never automatically restarted"):
        runner.main(["--run", "--resume", "--job-index", "0"])


def test_detach_never_reports_early_dead_child_as_started(campaign, stamp, monkeypatch):
    class DeadChild:
        pid = 987654
        returncode = 2

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner, "check_no_other_training_process", lambda: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: DeadChild())
    with pytest.raises(RuntimeError, match="failed before startup"):
        runner.detach(campaign, stamp, SimpleNamespace(job_index=None, max_jobs=None))


def test_background_command_is_detached_sequential_and_only_this_screen(
    campaign, stamp, monkeypatch
):
    captured = {}

    class StartedChild:
        pid = 987654

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        token = command[command.index("--launch-token") + 1]
        runner.write_json(
            campaign.output_root / "_launches" / f"{token}.started.json",
            {
                "launcher_pid": StartedChild.pid,
                "launch_token": token,
                "scientific_hash": runner.canonical_hash(stamp),
            },
            exclusive=True,
        )
        return StartedChild()

    monkeypatch.setattr(runner, "check_no_other_training_process", lambda: None)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    report = runner.detach(
        campaign, stamp, SimpleNamespace(job_index=None, max_jobs=None)
    )
    assert report["status"] == "started"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert "--run" in captured["command"] and "--resume" in captured["command"]
    assert "--detach" not in captured["command"]
    assert str(runner.ROOT / "scripts/run_rcig_batch_screen.py") in captured["command"]
