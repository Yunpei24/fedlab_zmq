"""MPS and causal/oracle contracts for the recent-only batch-screen control.

These tests deliberately fail rather than silently switch private work to CPU.
Run locally with PYTORCH_ENABLE_MPS_FALLBACK=0 and an available MPS device.
Server post-processing on CPU float64 is the unchanged RCIG contract.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import types

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.ldp_gradient_far_recent import (
    LDPGradientFARRecent,
    RecentTemporalReferenceState,
)
from algorithms.rcig_temporal_reference import TemporalRCIGReferenceState
from metrics.rcig_evaluation import detach_rcig_evaluation_oracles
from scripts import run_rcig_batch_screen_experiment as entry


@pytest.fixture(scope="module", autouse=True)
def require_mps():
    assert torch.backends.mps.is_available(), "Run these tests on local MPS, not CPU"


def _model():
    model = torch.nn.Linear(32, 1, bias=False, device="mps")
    with torch.no_grad():
        model.weight.zero_()
    return model


def _updates(round_num, shift=0.0):
    rows = []
    for client_id, value in enumerate((0.01, 0.02, -0.01, -0.02)):
        vector = torch.zeros(1, 32, dtype=torch.float32, device="mps")
        vector[0, 0] = value + shift
        rows.append(
            (
                # The genuine MPS private-gradient release transfers the
                # already-private upload to CPU float32 for transport.
                {"weight": vector.detach().cpu()},
                {
                    "client_id": client_id,
                    "round_num": round_num + 1,
                    "privacy_compute_device": "mps:0",
                    "privacy_gradient_release": True,
                    "privacy_upload_noise_variance_per_coordinate": 0.01,
                    "privacy_noise_multiplier_scale_public": 1.0,
                    "privacy_epsilon": 3.0,
                    "privacy_delta": 1e-5,
                    "privacy_sampling_scheme": "fixed_without_replacement",
                    "privacy_adjacency": "replace_one",
                    "privacy_accounting_assumption": "fixed_size_without_replacement_rdp_replace_one",
                    "privacy_noise_multiplier": 1.0,
                    "privacy_sensitivity_multiplier": 2.0,
                    "privacy_fixed_batch_size": 10,
                    "dataset_size": 200,
                    "model_steps": 1,
                    "far_update_mode": "single_step_gradient",
                    "bytes_sent": 128,
                    "energy_j_consumed": 0.0,
                    "local_loss": 0.0,
                },
                ClientState(client_id=client_id, battery_j=10.0),
            )
        )
    return rows


def _config(**overrides):
    config = {
        **LDPGradientFARRecent().get_default_config(),
        "device": "mps",
        "expected_num_clients": 4,
        "far_server_clip_norm": 1.0,
        "far_reference_output_clip_norm": 1.0,
        "rcig_reference_output_radius": 1.0,
        "reference_clip_radius": 1.0,
        "clip_norm": 1.0,
        "enable_dp": True,
        "noise_multiplier": 1.0,
        "target_epsilon": None,
        "fixed_batch_size": 10,
        "privacy_public_dataset_size": 200,
        "privacy_sampling_rate_override": 0.05,
        "fixed_steps_per_round": 1,
        "privacy_num_rounds": 40,
        "sampling_scheme": "fixed_without_replacement",
        "privacy_adjacency": "replace_one",
        "rcig_covariance_registry": "authenticated_public_mechanism",
        "far_alpha": 0.2,
        "kappa_w": 2.0,
        "rcig_gate_window": 4,
        "rcig_old_window": 4,
        "rcig_new_window": 4,
        "rcig_public_subspace_dimension": 32,
        "rcig_public_subspace_seed": 7,
        "rcig_process_variance": 1e-5,
        "rcig_covariance_ridge": 1e-6,
        "rcig_min_accepted_mass": 1.0,
        "far_server_lr": 0.1,
        "enable_oracle_diagnostics": True,
    }
    config.update(overrides)
    return config


def _prime(algorithm, model, config):
    return [
        algorithm.server_aggregate(model, _updates(t, 0.001 * t), t, config)
        for t in range(12)
    ]


def test_distinct_registration_and_no_calibrated_thresholds():
    assert isinstance(get_algorithm("ldp_gradient_far_recent"), LDPGradientFARRecent)
    assert type(get_algorithm("ldp_gradient_far")).__name__ == "LDPGradientFAR"
    defaults = LDPGradientFARRecent().get_default_config()
    assert defaults["rcig_persistent_policy"] == "recent_only"
    for key in (
        "rcig_innovation_threshold",
        "rcig_isotropic_innovation_threshold",
        "rcig_euclidean_innovation_threshold",
        "rcig_recovery_threshold",
    ):
        assert defaults[key] is None
        with pytest.raises(ValueError, match="no calibrated"):
            LDPGradientFARRecent._rcig_config(_config(**{key: 1.0}), 1.0)
    for bad in (
        {"rcig_persistent_policy": "freeze_hysteresis"},
        {"rcig_threshold_artifact_sha256": "ignored-but-invalid"},
        {"rcig_calibration_mode": True},
        {"rcig_new_window": 3},
        {"rcig_covariance_mode": "isotropic"},
    ):
        with pytest.raises(ValueError):
            LDPGradientFARRecent._rcig_config(_config(**bad), 1.0)


def test_twelve_uniform_rounds_then_genuine_recent_deployment():
    algorithm, model, config = LDPGradientFARRecent(), _model(), _config()
    results = _prime(algorithm, model, config)
    assert all(result.metrics["far_alpha"] == 0 for result in results)
    assert all(
        result.metrics["rcig_reference_mode"] == "recent_only" for result in results
    )
    assert all(
        result.metrics["rcig_deployed_candidate"] == "identity_new"
        for result in results
    )
    result = algorithm.server_aggregate(model, _updates(12), 12, config)
    metrics = result.metrics
    assert metrics["rcig_history_ready"] is True
    assert metrics["ldp_gradient_far_effective_alpha"] == pytest.approx(0.2)
    assert metrics["rcig_server_aggregation_dtype"] == "torch.float64"
    assert metrics["rcig_server_aggregation_device"] == "cpu"
    assert metrics["rcig_private_gradient_mps_fraction"] == 1.0
    assert metrics["ldp_gradient_far_reference_noise_aware"] is False
    assert metrics["rcig_temporal_correction_enabled"] is False
    assert metrics["rcig_innovation_test_enabled"] is False
    assert metrics["rcig_persistent_frozen_after_commit"] is False
    payload = metrics["_rcig_evaluation_payload"]
    assert set(payload["candidate_references"]) == {"identity_new"}
    assert torch.equal(
        payload["deployed_reference"], payload["candidate_references"]["identity_new"]
    )
    assert float(torch.linalg.vector_norm(payload["deployed_reference"])) <= 1.0


def test_exact_parent_identity_new_on_same_history_and_gate():
    recent, model, config = LDPGradientFARRecent(), _model(), _config()
    _prime(recent, model, config)
    state = recent._rcig_state
    parent = TemporalRCIGReferenceState(state.config)
    for snapshot in state._history:
        parent.commit_snapshot(copy.deepcopy(snapshot))
    kwargs = dict(round_num=12, dimension=32, device="cpu", dtype=torch.float64)
    expected = parent.reference_for_round(**kwargs)
    actual = state.reference_for_round(**kwargs)
    assert torch.equal(actual.reference, expected.candidate_references["identity_new"])
    assert torch.equal(actual.newer_view, expected.newer_view)
    assert torch.equal(actual.covariance_newer, expected.covariance_newer)
    assert (
        actual.diagnostics["rcig_gate_mass"] == expected.diagnostics["rcig_gate_mass"]
    )


def test_reference_calls_preserve_cpu_and_mps_rng_and_ignore_current_upload():
    one, two = LDPGradientFARRecent(), LDPGradientFARRecent()
    model_a, model_b, config = _model(), _model(), _config()
    _prime(one, model_a, config)
    _prime(two, model_b, config)
    rows_a, rows_b = _updates(12, 0.75), _updates(12, -0.75)
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone()
    one.server_aggregate(model_a, rows_a, 12, config)
    two.server_aggregate(model_b, rows_b, 12, config)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_mps, torch.mps.get_rng_state())
    assert torch.equal(
        one._rcig_last_deployed_reference, two._rcig_last_deployed_reference
    )


def test_no_fusion_freeze_or_threshold_dependency(monkeypatch):
    import algorithms.rcig_temporal_reference as temporal

    def forbidden(*args, **kwargs):
        raise AssertionError("recent-only reference must never run a fusion/test")

    monkeypatch.setattr(temporal, "robust_covariance_innovation_fusion", forbidden)
    monkeypatch.setattr(temporal, "euclidean_innovation_fusion", forbidden)
    algorithm, model, config = LDPGradientFARRecent(), _model(), _config()
    _prime(algorithm, model, config)
    state = algorithm._rcig_state
    alternate = RecentTemporalReferenceState(
        replace(
            state.config,
            innovation_threshold=1e12,
            isotropic_innovation_threshold=1e9,
            euclidean_innovation_threshold=1e6,
            process_variance=1e3,
        )
    )
    for snapshot in state._history:
        alternate.commit_snapshot(copy.deepcopy(snapshot))
    kwargs = dict(round_num=12, dimension=32, device="cpu", dtype=torch.float64)
    assert torch.equal(
        state.reference_for_round(**kwargs).reference,
        alternate.reference_for_round(**kwargs).reference,
    )
    for t in range(12, 21):
        result = algorithm.server_aggregate(
            model, _updates(t, 0.9 if t % 2 else -0.9), t, config
        )
        assert result.metrics["rcig_gate_active"] is False
        assert result.metrics["rcig_persistent_frozen_after_commit"] is False
    assert algorithm._rcig_state._recovery_count == 0
    assert algorithm._rcig_state._gate_source_range == (0, 3)


def test_rejects_forged_public_variance_cpu_metadata_and_oracle_leaks():
    for field, value, error in (
        ("privacy_upload_noise_variance_per_coordinate", 0.02, "variance"),
        ("privacy_compute_device", "cpu", "compute-device"),
        (
            "local_dp_noise_free_update_oracle",
            {"weight": torch.zeros(1, 32, device="mps")},
            "oracle",
        ),
    ):
        rows = _updates(0)
        rows[0][1][field] = value
        with pytest.raises(ValueError, match=error):
            LDPGradientFARRecent().server_aggregate(_model(), rows, 0, _config())


def test_public_config_signature_cannot_change():
    algorithm, model, config = LDPGradientFARRecent(), _model(), _config()
    algorithm.server_aggregate(model, _updates(0), 0, config)
    with pytest.raises(ValueError, match="public configuration changed"):
        algorithm.server_aggregate(
            model, _updates(1), 1, {**config, "rcig_process_variance": 0.3}
        )


def test_client_releases_real_private_gradient_on_mps_without_model_update():
    algorithm = LDPGradientFARRecent()
    model = torch.nn.Linear(2, 2, bias=False, device="mps")
    inputs = torch.arange(40, dtype=torch.float32).reshape(20, 2) / 10
    labels = torch.arange(20) % 2
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    config = _config(
        privacy_public_dataset_size=20,
        fixed_batch_size=4,
        privacy_sampling_rate_override=0.2,
        per_sample_backend="loop",
    )
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    gradient, metadata = algorithm.client_update(
        model, loader, ClientState(client_id=0, battery_j=100.0), config
    )
    assert metadata["privacy_compute_device"] in {"mps", "mps:0"}
    assert metadata["privacy_gradient_release"] is True
    assert metadata["privacy_fixed_batch_size"] == 4
    assert {
        "client_id",
        "round_num",
        "beta_actual",
        "battery_j_remaining",
        "energy_j_consumed",
        "bytes_sent",
        "bytes_received",
        "local_loss",
        "compression_ratio",
    } <= set(metadata)
    assert metadata["battery_j_remaining"] >= 0
    assert all(value.device.type == "cpu" for value in gradient.values())
    assert all(bool(torch.isfinite(value).all()) for value in gradient.values())
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )
    with pytest.raises(ValueError, match="model must be on MPS"):
        algorithm.client_update(model.cpu(), loader, ClientState(0), config)
    with pytest.raises(ValueError, match="require MPS"):
        algorithm.client_update(
            model, loader, ClientState(0), {**config, "device": "cpu"}
        )


def test_external_boundary_enables_detachment_and_restores_harness():
    original_evaluate = object()
    harness = types.SimpleNamespace(
        detach_rcig_evaluation_oracles=detach_rcig_evaluation_oracles,
        rcig_reference_oracle_metrics=original_evaluate,
    )
    rows = _updates(0)
    for update, metadata, _ in rows:
        metadata["local_dp_noise_free_update_oracle"] = {
            key: value.clone() for key, value in update.items()
        }
        metadata["is_byzantine"] = metadata["client_id"] == 0
        metadata["dp_noise_norm_mean"] = 100.0
        metadata["attack_type"] = "simulator-only"
    with entry.recent_evaluation_boundary(harness, enabled=True):
        safe, clean = harness.detach_rcig_evaluation_oracles(
            rows, enabled=False, strip_attack_oracles=True
        )
        assert len(clean) == 4
        for _, metadata, _ in safe:
            assert not set(metadata) & {
                "local_dp_noise_free_update_oracle",
                "is_byzantine",
                "dp_noise_norm_mean",
                "attack_type",
            }
        assert rows[0][1]["is_byzantine"] is True
        with pytest.raises(ValueError, match="stripping"):
            harness.detach_rcig_evaluation_oracles(rows, enabled=False)
    assert harness.detach_rcig_evaluation_oracles is detach_rcig_evaluation_oracles
    assert harness.rcig_reference_oracle_metrics is original_evaluate


def test_recent_external_oracle_metrics_and_no_fabricated_candidates():
    algorithm, model, config = LDPGradientFARRecent(), _model(), _config()
    _prime(algorithm, model, config)
    rows = _updates(12)
    for update, metadata, _ in rows:
        metadata["local_dp_noise_free_update_oracle"] = {
            key: value.clone() for key, value in update.items()
        }
        metadata["is_byzantine"] = metadata["client_id"] == 0
    safe, clean = detach_rcig_evaluation_oracles(
        rows, enabled=True, strip_attack_oracles=True
    )
    result = algorithm.server_aggregate(model, safe, 12, config)
    payload = result.metrics["_rcig_evaluation_payload"]
    metrics = entry.recent_reference_oracle_metrics(payload, clean, rows)
    clean_center = torch.stack(
        [row[0]["weight"].cpu().double().flatten() for row in rows[1:]]
    ).mean(0)
    expected = float((payload["deployed_reference"] - clean_center).square().sum())
    assert metrics[
        "rcig_reference_squared_l2_error_to_clean_honest_center_oracle"
    ] == pytest.approx(expected)
    assert metrics[
        "rcig_identity_new_squared_l2_error_to_clean_honest_center_oracle"
    ] == pytest.approx(expected)
    assert not any(name.startswith("rcig_full_") for name in metrics)
    bad = copy.deepcopy(payload)
    bad["candidate_references"]["rcig_full"] = bad["deployed_reference"]
    with pytest.raises(ValueError, match="only the deployed"):
        entry.recent_reference_oracle_metrics(bad, clean, rows)


def test_runtime_manifest_exclusive_final_and_source_tracking(tmp_path):
    before = entry.write_runtime_manifest(
        tmp_path, algorithm="ldp_gradient_far_recent", stage="before_training"
    )
    assert "algorithms/ldp_gradient_far_recent.py" in before
    assert "scripts/run_rcig_batch_screen_experiment.py" in before
    assert not any(name.startswith("venv/") for name in before)
    with pytest.raises(FileExistsError):
        entry.write_runtime_manifest(
            tmp_path, algorithm="ldp_gradient_far_recent", stage="before_training"
        )
    entry.write_runtime_manifest(
        tmp_path,
        algorithm="ldp_gradient_far_recent",
        stage="after_training",
        previous=before,
    )
    payload = json.loads((tmp_path / "runtime_imports.json").read_text())
    assert payload["stage"] == "after_training"
    assert payload["mps_fallback"] == 0
    corrupt = dict(before)
    corrupt["algorithms/ldp_gradient_far_recent.py"] = "bad"
    with pytest.raises(RuntimeError, match="sources changed"):
        entry.write_runtime_manifest(
            tmp_path,
            algorithm="ldp_gradient_far_recent",
            stage="after_training",
            previous=corrupt,
        )


def test_entry_config_rejects_cpu_fallback_or_missing_boundary(monkeypatch):
    config = {
        "training": {"algorithm": "ldp_gradient_far_recent", "algo_config": _config()}
    }
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert (
        entry.validate_entry_config(config, device="mps") == "ldp_gradient_far_recent"
    )
    with pytest.raises(ValueError, match="requires MPS"):
        entry.validate_entry_config(config, device="cpu")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback"):
        entry.validate_entry_config(config, device="mps")


def test_entrypoint_delegates_with_temporary_external_boundary(tmp_path, monkeypatch):
    """A mocked harness main proves entrypoint plumbing without any training."""
    import sys
    import yaml

    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "algorithm": "ldp_gradient_far_recent",
                    "algo_config": _config(),
                }
            }
        )
    )
    harness = types.ModuleType("run_experiment")
    harness.detach_rcig_evaluation_oracles = detach_rcig_evaluation_oracles
    sentinel = object()
    harness.rcig_reference_oracle_metrics = sentinel
    calls = []

    def fake_main():
        assert (
            harness.rcig_reference_oracle_metrics
            is entry.recent_reference_oracle_metrics
        )
        assert (
            harness.detach_rcig_evaluation_oracles is not detach_rcig_evaluation_oracles
        )
        assert sys.argv[-3:] == ["mps", "--output", str(tmp_path / "run")]
        calls.append(True)

    harness.main = fake_main
    monkeypatch.setitem(sys.modules, "run_experiment", harness)
    original_argv = sys.argv
    entry.main(
        [
            "--config",
            str(config_path),
            "--device",
            "mps",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert calls == [True]
    assert sys.argv is original_argv
    assert harness.rcig_reference_oracle_metrics is sentinel
    assert harness.detach_rcig_evaluation_oracles is detach_rcig_evaluation_oracles
    manifest = json.loads((tmp_path / "run" / "runtime_imports.json").read_text())
    assert manifest["stage"] == "after_training"
