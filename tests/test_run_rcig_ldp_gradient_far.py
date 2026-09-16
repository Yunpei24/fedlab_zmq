from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_rcig_ldp_gradient_far import (
    DEFAULT_MATRIX,
    campaign_scientific_hash,
    load_campaign,
    record_gate,
    resolved_config,
    tasks_for_phase,
    validate_requested_device,
)


def test_rcig_matrix_has_locked_gated_counts_and_disjoint_seeds() -> None:
    campaign = load_campaign(DEFAULT_MATRIX)
    counts = {
        phase["id"]: len(tasks_for_phase(campaign, phase["id"]))
        for phase in campaign.phases
    }
    assert (
        counts
        == campaign.matrix["expected_task_counts"]
        == {
            "r0_calibration_frozen": 24,
            "r1_null_validation_frozen": 72,
            "r2_attack_mechanism_frozen": 36,
            "r3_end_to_end_development": 80,
            "r4_end_to_end_confirmation": 210,
        }
    )
    assert len(campaign.tasks) == 422
    registries = campaign.matrix["randomness"]
    names = [name for name in registries if name.endswith("_seeds")]
    sets = {name: set(registries[name]) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert sets[left].isdisjoint(sets[right])


def test_all_arms_keep_the_private_gradient_and_n25_contract() -> None:
    campaign = load_campaign(DEFAULT_MATRIX)
    for task in campaign.tasks:
        config = resolved_config(campaign, task, inject_threshold=False)
        algo = config["training"]["algo_config"]
        assert config["device"] == "mps"
        assert config["data"]["dataset"] == "fashionmnist"
        assert config["model"]["architecture"] == "lenet5_tanh"
        assert config["clients"]["num_clients"] == 25
        assert algo["privacy_public_dataset_size"] == 2400
        assert algo["fixed_batch_size"] == algo["batch_size"] == 120
        assert algo["clip_norm"] == 4.0
        assert algo["fixed_steps_per_round"] == 1
        assert algo["sampling_scheme"] == "fixed_without_replacement"
        assert algo["privacy_adjacency"] == "replace_one"
        assert algo["enable_dp"] is True


def test_threshold_is_injected_from_immutable_calibration_artifact(
    tmp_path: Path,
) -> None:
    source = load_campaign(DEFAULT_MATRIX)
    campaign = replace(source, output_root=tmp_path / "results")
    thresholds = {
        "homogeneous": {"full": 3.0, "isotropic": 2.0, "euclidean": 1.0},
        "heteroscedastic": {"full": 6.0, "isotropic": 5.0, "euclidean": 4.0},
    }
    path = campaign.output_root / "_gates" / "r0_calibration_frozen.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "decision": "promote",
                "campaign_scientific_hash": campaign_scientific_hash(campaign),
                "evidence": {"thresholds_by_regime_and_mode": thresholds},
            }
        ),
        encoding="utf-8",
    )
    task = next(
        task
        for task in tasks_for_phase(campaign, "r3_end_to_end_development")
        if task.axis_values["reference"] == "rcig_isotropic"
    )
    algo = resolved_config(campaign, task)["training"]["algo_config"]
    assert algo["rcig_innovation_threshold"] == 6.0
    assert algo["rcig_isotropic_innovation_threshold"] == 5.0
    assert algo["rcig_euclidean_innovation_threshold"] == 4.0
    assert len(algo["rcig_threshold_artifact_sha256"]) == 64


def test_device_policy_refuses_cpu_and_cuda() -> None:
    validate_requested_device("mps")
    with pytest.raises(ValueError, match="CPU and CUDA"):
        validate_requested_device("cpu")
    with pytest.raises(ValueError, match="CPU and CUDA"):
        validate_requested_device("cuda")


def test_gate_record_is_fail_closed_and_immutable(tmp_path: Path) -> None:
    source = load_campaign(DEFAULT_MATRIX)
    campaign = replace(source, output_root=tmp_path / "results")
    evidence = {
        "complete_fraction": 1.0,
        "invalid_runs": 0.0,
        "mps_fraction": 1.0,
        "private_gradient_protocol_fraction": 1.0,
        "strict_past_fraction": 1.0,
        "covariance_psd_fraction": 1.0,
        "finite_threshold_fraction": 0.5,
        "thresholds_by_regime_and_mode": {},
    }
    path = record_gate(campaign, "r0_calibration_frozen", evidence)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["decision"] == "stop"
    changed = dict(evidence)
    changed["finite_threshold_fraction"] = 1.0
    with pytest.raises(RuntimeError, match="immutable"):
        record_gate(campaign, "r0_calibration_frozen", changed)
