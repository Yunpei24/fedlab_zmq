#!/usr/bin/env python3
"""Immutable, fail-closed runner for the RCIG end-to-end v2 protocol."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
# When this file is executed directly (``python scripts/...py``), Python adds
# ``scripts/`` rather than the repository root to ``sys.path``.  The gated
# resume path imports the companion analyzer through the ``scripts`` namespace,
# so make that import invariant to the launcher's current working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MATRIX = ROOT / "configs/ldp_gradient_far/rcig_end_to_end_v2.yaml"
PROTOCOL_PATH = ROOT / "output/analysis/RCIG_LDP_Gradient_FAR_Protocol_PreRun_v2.md"
LOCK_NAME = "_campaign_lock_v2.json"
RCIG_MODES = ("full", "isotropic", "euclidean")
RCIG_REFERENCES = {
    "rcig_temporal",
    "rcig_temporal_full",
    "rcig_temporal_isotropic",
    "rcig_temporal_euclidean",
}
ALLOWED_REFERENCES = {"centered_clipping", "rfa", *RCIG_REFERENCES}
ALLOWED_ATTACKS = {"none", "bf", "ipm", "alie"}
BYZANTINE_IDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class RCIGV2Task:
    global_index: int
    phase_index: int
    phase_id: str
    phase_role: str
    seed: int
    axis_values: dict[str, str]
    overrides: dict[str, Any]

    @property
    def variant_id(self) -> str:
        return "__".join(self.axis_values.values())

    @property
    def run_id(self) -> str:
        return f"{self.variant_id}_seed{self.seed}"


@dataclass(frozen=True)
class RCIGV2Campaign:
    matrix_path: Path
    matrix: dict[str, Any]
    base: dict[str, Any]
    output_root: Path
    tasks: tuple[RCIGV2Task, ...]

    @property
    def phases(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.matrix["phases"])


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialized_config(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(config), sort_keys=False)


def _expand_phase(
    phase: Mapping[str, Any], *, first_global_index: int
) -> list[RCIGV2Task]:
    phase_id = str(phase["id"])
    axes = phase.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError(f"phase {phase_id!r} must define non-empty axes")
    axis_names = [str(axis["name"]) for axis in axes]
    if len(axis_names) != len(set(axis_names)):
        raise ValueError(f"phase {phase_id!r} contains duplicate axis names")
    value_sets: list[list[Mapping[str, Any]]] = []
    for axis in axes:
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {axis['name']!r} has no values")
        ids = [str(item["id"]) for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"axis {axis['name']!r} has duplicate ids")
        value_sets.append(values)
    seeds = [int(seed) for seed in phase.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"phase {phase_id!r} needs unique non-empty seeds")

    tasks: list[RCIGV2Task] = []
    common = phase.get("common_overrides", {})
    for values in itertools.product(*value_sets):
        overrides = copy.deepcopy(common)
        axis_values: dict[str, str] = {}
        for axis_name, item in zip(axis_names, values):
            axis_values[axis_name] = str(item["id"])
            overrides = deep_merge(overrides, item.get("overrides", {}))
        for seed in seeds:
            tasks.append(
                RCIGV2Task(
                    global_index=first_global_index + len(tasks),
                    phase_index=len(tasks),
                    phase_id=phase_id,
                    phase_role=str(phase.get("role", "unspecified")),
                    seed=seed,
                    axis_values=axis_values,
                    overrides=overrides,
                )
            )
    return tasks


def load_campaign(matrix_path: Path = DEFAULT_MATRIX) -> RCIGV2Campaign:
    matrix_path = matrix_path.resolve()
    matrix = _read_yaml(matrix_path)
    if int(matrix.get("schema_version", 0)) != 2:
        raise ValueError("RCIG v2 requires schema_version: 2")
    if str(matrix.get("required_device", "")).lower() != "mps":
        raise ValueError("RCIG v2 must declare required_device: mps")
    base_path = (matrix_path.parent / str(matrix["base_config"])).resolve()
    base = _read_yaml(base_path)
    output_root = (matrix_path.parent / str(matrix["output_root"])).resolve()
    phases = matrix.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("RCIG v2 needs at least one phase")
    phase_ids = [str(phase["id"]) for phase in phases]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("RCIG v2 phase ids must be unique")
    seen: set[str] = set()
    tasks: list[RCIGV2Task] = []
    for phase in phases:
        unknown = [str(dep) for dep in phase.get("depends_on", []) if dep not in seen]
        if unknown:
            raise ValueError(f"phase {phase['id']!r} has unknown dependency {unknown}")
        tasks.extend(_expand_phase(phase, first_global_index=len(tasks)))
        seen.add(str(phase["id"]))
    campaign = RCIGV2Campaign(
        matrix_path=matrix_path,
        matrix=matrix,
        base=base,
        output_root=output_root,
        tasks=tuple(tasks),
    )
    validate_campaign(campaign)
    return campaign


def phase_definition(campaign: RCIGV2Campaign, phase_id: str) -> dict[str, Any]:
    for phase in campaign.phases:
        if str(phase["id"]) == phase_id:
            return phase
    raise KeyError(f"unknown RCIG v2 phase {phase_id!r}")


def tasks_for_phase(campaign: RCIGV2Campaign, phase_id: str) -> tuple[RCIGV2Task, ...]:
    phase_definition(campaign, phase_id)
    return tuple(task for task in campaign.tasks if task.phase_id == phase_id)


def task_output_dir(campaign: RCIGV2Campaign, task: RCIGV2Task) -> Path:
    return campaign.output_root / task.phase_id / task.run_id


def gate_path(campaign: RCIGV2Campaign, phase_id: str) -> Path:
    return campaign.output_root / "_gates" / f"{phase_id}.json"


def campaign_scientific_hash(campaign: RCIGV2Campaign) -> str:
    return _canonical_hash({"matrix": campaign.matrix, "base": campaign.base})


def _noise_regime(task: RCIGV2Task) -> str:
    return task.axis_values.get("noise_regime", "homogeneous")


def _load_calibrated_thresholds(
    campaign: RCIGV2Campaign, task: RCIGV2Task
) -> tuple[dict[str, float], str]:
    path = gate_path(campaign, "r0_dynamic_calibration")
    if not path.exists():
        raise RuntimeError("the immutable v2 calibration artifact is absent")
    payload = _verified_gate_payload(campaign, "r0_dynamic_calibration")
    if payload.get("decision") != "promote":
        raise RuntimeError("the v2 calibration gate did not promote")
    source = payload.get("evidence", {}).get("thresholds_by_regime_and_mode")
    try:
        thresholds = {
            mode: float(source[_noise_regime(task)][mode]) for mode in RCIG_MODES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("the v2 calibration artifact is incomplete") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds.values()):
        raise RuntimeError("all calibrated RCIG thresholds must be finite and > 0")
    return thresholds, _file_sha256(path)


def _r2_attack(task: RCIGV2Task) -> dict[str, Any]:
    attack_id = task.axis_values["attack"]
    schedule_id = task.axis_values["schedule"]
    names = {
        "bf_x10": ("bf", 10.0),
        "ipm": ("ipm", 1.0),
        "alie": ("alie", 1.0),
    }
    if attack_id not in names:
        raise ValueError(f"unknown R2 attack {attack_id!r}")
    if schedule_id == "persistent_after_clean_warmup":
        start = 17
    elif schedule_id == "persistent_from_round1":
        start = 1
    else:
        raise ValueError(f"unknown R2 schedule {schedule_id!r}")
    name, scale = names[attack_id]
    return {
        "enabled": True,
        "name": name,
        "scale": scale,
        "active_round_start": start,
        "active_round_end": 40,
        "num_byzantine": 5,
        "client_ids": list(BYZANTINE_IDS),
    }


def _pairing_design_hash(config: Mapping[str, Any]) -> str:
    """Fingerprint public inputs that determine paired stochastic streams.

    This certifies the requested design, not equality of realised tensors; the
    seed-controlled execution remains the reproducibility assumption.
    """

    algo = config["training"]["algo_config"]
    return _canonical_hash(
        {
            "seed": config["seed"],
            "partition_seed": config["data"]["partition_seed"],
            "dataset": config["data"]["dataset"],
            "partition": config["data"]["partition"],
            "partition_alpha": config["data"].get("alpha"),
            "model": config["model"]["architecture"],
            "num_clients": config["clients"]["num_clients"],
            "rounds": config["training"]["num_rounds"],
            "batch_size": algo["fixed_batch_size"],
            "sampling_scheme": algo["sampling_scheme"],
            "privacy_adjacency": algo["privacy_adjacency"],
            "noise_scale_by_client": algo.get(
                "privacy_noise_multiplier_scale_by_client"
            ),
            "attack": algo["attack"],
        }
    )


def resolved_config(
    campaign: RCIGV2Campaign,
    task: RCIGV2Task,
    output_dir: Path | None = None,
    *,
    inject_threshold: bool = True,
) -> dict[str, Any]:
    config = deep_merge(campaign.base, task.overrides)
    config["seed"] = task.seed
    config["device"] = "mps"
    config.setdefault("data", {})["partition_seed"] = task.seed
    config["output_dir"] = str(output_dir or task_output_dir(campaign, task))
    config["training"]["num_rounds"] = 40
    algo = config["training"]["algo_config"]
    algo["privacy_num_rounds"] = 40
    algo["expected_num_clients"] = 25
    algo["num_byzantine"] = 5
    oracle_phase = task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"}
    algo["enable_oracle_diagnostics"] = oracle_phase
    algo["rcig_oracle_evaluation_only"] = oracle_phase
    algo["rcig_oracle_metrics_are_not_release"] = oracle_phase
    if task.phase_id == "r2_attack_mechanism":
        algo["attack"] = _r2_attack(task)
    else:
        attack = dict(algo.get("attack") or {})
        if bool(attack.get("enabled", False)):
            attack.update({"num_byzantine": 5, "client_ids": list(BYZANTINE_IDS)})
        else:
            attack.update(
                {
                    "enabled": False,
                    "name": "none",
                    "scale": 1.0,
                    "num_byzantine": 0,
                    "client_ids": [],
                }
            )
            attack.pop("active_round_start", None)
            attack.pop("active_round_end", None)
        algo["attack"] = attack

    reference = str(algo.get("robust_reference", "")).lower()
    algo["rcig_oracle_separation_required"] = reference in RCIG_REFERENCES
    if (
        reference in RCIG_REFERENCES
        and not bool(algo.get("rcig_calibration_mode", False))
        and inject_threshold
    ):
        thresholds, artifact_hash = _load_calibrated_thresholds(campaign, task)
        algo["rcig_innovation_threshold"] = thresholds["full"]
        algo["rcig_isotropic_innovation_threshold"] = thresholds["isotropic"]
        algo["rcig_euclidean_innovation_threshold"] = thresholds["euclidean"]
        algo["rcig_threshold_artifact_sha256"] = artifact_hash
    algo.update(
        {
            "rcig_campaign_id": str(campaign.matrix["campaign_id"]),
            "rcig_campaign_phase": task.phase_id,
            "rcig_campaign_variant": task.variant_id,
            "rcig_campaign_scientific_hash": campaign_scientific_hash(campaign),
            "rcig_pairing_seed_block": task.seed,
            "rcig_pairing_design_sha256": _pairing_design_hash(config),
            "rcig_required_private_device": "mps",
            "rcig_required_server_postprocess_device": "cpu",
            "rcig_required_server_postprocess_dtype": "torch.float64",
            "rcig_mps_fallback": 0,
            "rcig_covariance_registry": "authenticated_public_mechanism",
            "external_attack_diagnostics": True,
        }
    )
    return config


def _validate_resolved_config(config: Mapping[str, Any], task: RCIGV2Task) -> None:
    if config.get("device") != "mps":
        raise ValueError("RCIG v2 training is MPS-only")
    if config["training"].get("algorithm") != "ldp_gradient_far":
        raise ValueError("RCIG v2 must run ldp_gradient_far")
    if config["data"].get("dataset") != "fashionmnist":
        raise ValueError("RCIG v2 is locked to Fashion-MNIST")
    if config["model"].get("architecture") != "lenet5_tanh":
        raise ValueError("RCIG v2 is locked to LeNet5-tanh")
    if config["data"].get("partition") != "client_dirichlet_balanced":
        raise ValueError("RCIG v2 requires fixed-size balanced client partitions")
    if int(config["training"].get("num_rounds", -1)) != 40:
        raise ValueError("every RCIG v2 arm requires T=40")
    clients = config["clients"]
    if (
        int(clients.get("num_clients", -1)) != 25
        or int(clients.get("min_clients", -1)) != 25
        or float(clients.get("sample_fraction", 0.0)) != 1.0
        or float(clients.get("dropout_rate", -1.0)) != 0.0
    ):
        raise ValueError("RCIG v2 requires full participation of 25 live clients")

    algo = config["training"]["algo_config"]
    integer_contract = {
        "batch_size": 120,
        "fixed_batch_size": 120,
        "privacy_public_dataset_size": 2400,
        "local_epochs": 1,
        "fixed_steps_per_round": 1,
        "expected_num_clients": 25,
        "privacy_num_rounds": 40,
    }
    for key, expected in integer_contract.items():
        if int(algo.get(key, -1)) != expected:
            raise ValueError(f"RCIG v2 requires {key}={expected}")
    scalar_contract = {
        "privacy_sampling_rate_override": 0.05,
        "target_epsilon": 4.0,
        "delta": 1.0e-5,
        "clip_norm": 4.0,
        "far_server_clip_norm": 16.0,
    }
    for key, expected in scalar_contract.items():
        if not math.isclose(float(algo.get(key, math.nan)), expected, abs_tol=1e-12):
            raise ValueError(f"RCIG v2 requires {key}={expected}")
    if algo.get("sampling_scheme") != "fixed_without_replacement":
        raise ValueError("RCIG v2 requires fixed sampling without replacement")
    if algo.get("privacy_adjacency") != "replace_one":
        raise ValueError("RCIG v2 requires replace-one adjacency")
    if algo.get("per_sample_backend") != "vectorized" or not bool(
        algo.get("enable_dp", False)
    ):
        raise ValueError("RCIG v2 requires vectorized private per-example gradients")
    if not bool(algo.get("external_attack_diagnostics", False)):
        raise ValueError(
            "RCIG v2 requires attack labels to be external to server_aggregate"
        )
    if algo.get("far_score_mode") != "raw_distance":
        raise ValueError("RCIG v2 forbids the bounded score transform")
    if algo.get("noise_score_standardization") != "none":
        raise ValueError("RCIG v2 places noise awareness in F, not the FAR score")
    reference = str(algo.get("robust_reference", "")).lower()
    if reference not in ALLOWED_REFERENCES:
        raise ValueError(f"unsupported RCIG v2 reference {reference!r}")
    if reference in RCIG_REFERENCES:
        if str(algo.get("rcig_covariance_mode")) not in RCIG_MODES:
            raise ValueError("invalid RCIG covariance mode")
        if int(algo.get("rcig_gate_window", -1)) != 4:
            raise ValueError("RCIG v2 requires a four-round gate window")
        if (
            int(algo.get("rcig_old_window", -1)) != 4
            or int(algo.get("rcig_new_window", -1)) != 4
        ):
            raise ValueError("RCIG v2 requires disjoint four-round old/new views")
        if int(algo.get("rcig_public_subspace_dimension", -1)) != 64:
            raise ValueError("RCIG v2 requires 64 public gate coordinates")
        if algo.get("rcig_warmup_policy") != "uniform":
            raise ValueError("RCIG v2 cold start must be uniform")
        if algo.get("rcig_persistent_policy") != "freeze_hysteresis":
            raise ValueError("RCIG v2 requires freeze_hysteresis")
        if not math.isclose(
            float(algo.get("rcig_reference_output_radius", math.nan)), 16.0
        ):
            raise ValueError("RCIG output radius must match server clipping")
        for key in (
            "rcig_innovation_threshold",
            "rcig_isotropic_innovation_threshold",
            "rcig_euclidean_innovation_threshold",
        ):
            value = float(algo.get(key, math.nan))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be finite and > 0")
    if task.phase_id in {
        "r0_dynamic_calibration",
        "r1_dynamic_null",
        "r2_attack_mechanism",
    }:
        if not math.isclose(float(algo.get("far_alpha", math.nan)), 0.1):
            raise ValueError("mechanism phases must match the final alpha=0.1")
        if not math.isclose(float(algo.get("far_server_lr", math.nan)), 0.2):
            raise ValueError("mechanism phases must retain dynamic model updates")
    if task.phase_id == "r3_e2e_confirmation":
        reference_id = task.axis_values["reference"]
        expected_reference = {
            "uniform": "centered_clipping",
            "fcc": "centered_clipping",
            "rfa": "rfa",
            "rcig_full": "rcig_temporal",
        }[reference_id]
        expected_alpha = 0.0 if reference_id == "uniform" else 0.1
        if reference != expected_reference or not math.isclose(
            float(algo.get("far_alpha", math.nan)), expected_alpha, abs_tol=1e-12
        ):
            raise ValueError("R3 reference/alpha arm does not match its declared id")
    oracle_phase = task.phase_id in {"r1_dynamic_null", "r2_attack_mechanism"}
    if bool(algo.get("enable_oracle_diagnostics", False)) != oracle_phase:
        raise ValueError("oracle diagnostics are permitted only in v2 R1/R2")
    if (
        bool(algo.get("rcig_oracle_evaluation_only", False)) != oracle_phase
        or bool(algo.get("rcig_oracle_metrics_are_not_release", False)) != oracle_phase
    ):
        raise ValueError("R1/R2 oracle data must be marked evaluation-only/non-release")
    if bool(algo.get("rcig_oracle_separation_required", False)) != (
        reference in RCIG_REFERENCES
    ):
        raise ValueError("every RCIG arm must require oracle/transcript separation")
    if algo.get("rcig_pairing_design_sha256") != _pairing_design_hash(config):
        raise ValueError(
            "pairing-design fingerprint does not match the resolved config"
        )
    attack = algo.get("attack", {})
    if str(attack.get("name", "none")) not in ALLOWED_ATTACKS:
        raise ValueError("unsupported RCIG v2 attack")
    if bool(attack.get("enabled", False)):
        if attack.get("client_ids") != BYZANTINE_IDS:
            raise ValueError("RCIG v2 Byzantine IDs are fixed to 0..4")
        if int(attack.get("num_byzantine", -1)) != 5:
            raise ValueError("RCIG v2 requires five Byzantine clients")
        start = int(attack.get("active_round_start", -1))
        end = int(attack.get("active_round_end", -1))
        if start not in {1, 17} or end != 40:
            raise ValueError("RCIG v2 attack schedule must be [1,40] or [17,40]")
    elif attack.get("client_ids") != []:
        raise ValueError("no-attack arms must have an empty Byzantine ID list")


def validate_campaign(campaign: RCIGV2Campaign) -> None:
    expected = {
        str(key): int(value)
        for key, value in campaign.matrix.get("expected_task_counts", {}).items()
    }
    observed = {
        str(phase["id"]): len(tasks_for_phase(campaign, str(phase["id"])))
        for phase in campaign.phases
    }
    if observed != expected:
        raise ValueError(f"unexpected v2 task counts {observed}; expected {expected}")
    if sum(observed.values()) != 706:
        raise ValueError("RCIG v2 must contain exactly 706 tasks")
    registries = campaign.matrix.get("randomness", {})
    registry_names = (
        "calibration_seeds",
        "null_validation_seeds",
        "mechanism_seeds",
        "confirmation_seeds",
    )
    registry_sets = {
        name: {int(value) for value in registries[name]} for name in registry_names
    }
    for index, left in enumerate(registry_names):
        for right in registry_names[index + 1 :]:
            if registry_sets[left] & registry_sets[right]:
                raise ValueError(f"v2 seed registries {left}/{right} overlap")
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        registry = str(phase["seed_registry"])
        if {task.seed for task in tasks_for_phase(campaign, phase_id)} != registry_sets[
            registry
        ]:
            raise ValueError(f"phase {phase_id!r} does not match {registry!r}")
        criteria = phase.get("gate_criteria", [])
        if bool(phase.get("gate_required", False)) and not criteria:
            raise ValueError(f"gated phase {phase_id!r} needs numeric criteria")
        ids = [str(item.get("id", "")) for item in criteria]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            raise ValueError(f"phase {phase_id!r} has invalid criterion ids")
        for criterion in criteria:
            if criterion.get("op") not in {"==", "<=", ">="}:
                raise ValueError("unsupported v2 gate operator")
            threshold = criterion.get("threshold")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("v2 gate thresholds must be numeric")
    for task in campaign.tasks:
        config = resolved_config(campaign, task, inject_threshold=False)
        algo = config["training"]["algo_config"]
        if str(
            algo.get("robust_reference", "")
        ).lower() in RCIG_REFERENCES and not bool(
            algo.get("rcig_calibration_mode", False)
        ):
            algo["rcig_innovation_threshold"] = 1.0
            algo["rcig_isotropic_innovation_threshold"] = 1.0
            algo["rcig_euclidean_innovation_threshold"] = 1.0
        _validate_resolved_config(config, task)


def validate_requested_device(device: str) -> None:
    if str(device).lower() != "mps":
        raise ValueError("RCIG v2 refuses CPU and CUDA for private-gradient compute")


def require_working_mps() -> None:
    validate_requested_device("mps")
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback not in {None, "0"}:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must equal 0")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("native MPS is unavailable; CPU fallback is forbidden")
    try:
        probe = torch.ones(2, device="mps")
        _ = probe.square().sum().cpu()
        torch.mps.synchronize()
    except Exception as exc:  # pragma: no cover - host dependent
        raise RuntimeError("native MPS execution probe failed") from exc


def _metrics_path(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("**/metrics.json"))
    if len(candidates) != 1:
        return None
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 40:
        return None
    return candidates[0]


def task_is_complete(campaign: RCIGV2Campaign, task: RCIGV2Task) -> bool:
    output_dir = task_output_dir(campaign, task)
    status_path = output_dir / "orchestration_status.json"
    config_path = output_dir / "resolved_config.yaml"
    metrics_path = _metrics_path(output_dir)
    if not status_path.exists() or not config_path.exists() or metrics_path is None:
        return False
    try:
        verify_existing_campaign_lock(campaign)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        observed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected_config = resolved_config(campaign, task, output_dir)
        expected_attack = expected_config["training"]["algo_config"]["attack"]
    except (OSError, json.JSONDecodeError, RuntimeError, yaml.YAMLError):
        return False
    expected_status = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": task.phase_id,
        "run_id": task.run_id,
        "status": "completed",
        "device": "mps",
        "mps_fallback": 0,
        "server_postprocessing": "cpu_float64",
        "scientific_hash": campaign_scientific_hash(campaign),
        "resolved_config_sha256": _file_sha256(config_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "pairing_seed_block": task.seed,
        "pairing_design_sha256": expected_config["training"]["algo_config"][
            "rcig_pairing_design_sha256"
        ],
        "attack_ids": expected_attack.get("client_ids", []),
        "attack_start": expected_attack.get("active_round_start"),
        "attack_end": expected_attack.get("active_round_end"),
    }
    if any(status.get(key) != value for key, value in expected_status.items()):
        return False
    if observed_config != expected_config:
        return False
    try:
        _post_run_device_audit(metrics_path, expected_config)
        from scripts.analyze_rcig_ldp_gradient_far_v2 import _critical_protocol_errors

        if _critical_protocol_errors(
            metrics_payload,
            expected_config,
            task,
            status,
            observed_config,
            config_path,
            metrics_path,
        ):
            return False
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return False
    return True


def phase_is_complete(campaign: RCIGV2Campaign, phase_id: str) -> bool:
    return all(
        task_is_complete(campaign, task) for task in tasks_for_phase(campaign, phase_id)
    )


def _gate_decision(campaign: RCIGV2Campaign, phase_id: str) -> str | None:
    path = gate_path(campaign, phase_id)
    if not path.exists():
        return None
    try:
        payload = _verified_gate_payload(campaign, phase_id)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return "invalid"
    return str(payload.get("decision"))


def _verified_gate_payload(campaign: RCIGV2Campaign, phase_id: str) -> dict[str, Any]:
    """Validate a gate and its evidence before it can unlock another phase."""

    path = gate_path(campaign, phase_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("gate artifact must be a JSON mapping")
    expected_header = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": phase_id,
        "campaign_scientific_hash": campaign_scientific_hash(campaign),
        "immutable": True,
    }
    if any(payload.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError("gate header does not match this immutable campaign")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or payload.get(
        "evidence_sha256"
    ) != _canonical_hash(evidence):
        raise RuntimeError("gate evidence hash is missing or invalid")
    criteria = phase_definition(campaign, phase_id).get("gate_criteria", [])
    if payload.get("criteria") != criteria:
        raise RuntimeError("gate criteria differ from the preregistration")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(criteria):
        raise RuntimeError("gate evaluations are incomplete")
    expected_evaluations = []
    for criterion in criteria:
        identifier = str(criterion["id"])
        value = evidence.get(identifier)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"gate evidence {identifier!r} is not numeric")
        observed = float(value)
        if not math.isfinite(observed):
            raise RuntimeError(f"gate evidence {identifier!r} is not finite")
        operator = str(criterion["op"])
        threshold = float(criterion["threshold"])
        if operator == "==":
            passed = math.isclose(observed, threshold, abs_tol=1e-12)
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == ">=":
            passed = observed >= threshold
        else:  # pragma: no cover - validate_campaign rejects this first
            raise RuntimeError(f"unsupported gate operator {operator!r}")
        expected_evaluations.append(
            {
                "id": identifier,
                "observed": observed,
                "op": operator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    if evaluations != expected_evaluations:
        raise RuntimeError("gate evaluations do not match the hashed evidence")
    expected_decision = (
        "promote"
        if expected_evaluations and all(row["passed"] for row in expected_evaluations)
        else "stop"
    )
    if payload.get("decision") != expected_decision:
        raise RuntimeError("gate decision does not follow its evaluations")
    return payload


def assert_dependencies_promoted(campaign: RCIGV2Campaign, phase_id: str) -> None:
    for dependency in phase_definition(campaign, phase_id).get("depends_on", []):
        dependency = str(dependency)
        if not phase_is_complete(campaign, dependency):
            raise RuntimeError(f"dependency {dependency!r} is incomplete")
        payload = _verified_gate_payload(campaign, dependency)
        if payload.get("decision") != "promote":
            raise RuntimeError(f"dependency {dependency!r} did not promote")
        # A gate is not merely self-authenticating: its hashed evidence must
        # still match a fresh evaluation of the immutable run artifacts.
        from scripts.analyze_rcig_ldp_gradient_far_v2 import evaluate_gate

        current = evaluate_gate(campaign, dependency)
        if payload.get("evidence_sha256") != _canonical_hash(current):
            raise RuntimeError(
                f"dependency {dependency!r} gate is stale against current artifacts"
            )


def _campaign_lock_payload(campaign: RCIGV2Campaign) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"pre-registered protocol is absent: {PROTOCOL_PATH}")
    sources = (
        ROOT / "algorithms/ldp_gradient_far.py",
        ROOT / "algorithms/__init__.py",
        ROOT / "algorithms/base.py",
        ROOT / "algorithms/far.py",
        ROOT / "algorithms/fedavg.py",
        ROOT / "algorithms/dp_references.py",
        ROOT / "algorithms/reference_utils.py",
        ROOT / "algorithms/rcig_temporal_reference.py",
        ROOT / "algorithms/gaussian_aware_reference_k7_rcig.py",
        ROOT / "core/seeding.py",
        ROOT / "datasets/partitioner.py",
        ROOT / "datasets/registry.py",
        ROOT / "models/registry.py",
        ROOT / "privacy/local_dpsgd.py",
        ROOT / "privacy/rdp.py",
        ROOT / "metrics/client_fairness.py",
        ROOT / "metrics/rcig_evaluation.py",
        ROOT / "metrics/robustness.py",
        ROOT / "robustness/aggregators.py",
        ROOT / "robustness/tensor_ops.py",
        ROOT / "attacks/byzantine.py",
        ROOT / "attacks/__init__.py",
        ROOT / "run_experiment.py",
        ROOT / "scripts/run_rcig_ldp_gradient_far_v2.py",
        ROOT / "scripts/analyze_rcig_ldp_gradient_far_v2.py",
    )
    missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"RCIG v2 lock source files are missing: {missing}")
    return {
        "campaign_id": campaign.matrix["campaign_id"],
        "scientific_hash": campaign_scientific_hash(campaign),
        "matrix_sha256": _file_sha256(campaign.matrix_path),
        "base_sha256": _canonical_hash(campaign.base),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "source_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in sources
        },
        "private_compute_device": "mps",
        "mps_fallback": 0,
        "server_postprocessing_device": "cpu",
        "server_postprocessing_dtype": "torch.float64",
    }


def ensure_campaign_lock(campaign: RCIGV2Campaign) -> Path:
    path = campaign.output_root / LOCK_NAME
    expected = _campaign_lock_payload(campaign)
    if path.exists():
        return verify_existing_campaign_lock(campaign)
    if campaign.output_root.exists() and any(campaign.output_root.iterdir()):
        raise RuntimeError(
            "refusing to create a v2 campaign lock above pre-existing artifacts"
        )
    campaign.output_root.mkdir(parents=True, exist_ok=True)
    expected["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    body = (json.dumps(expected, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return verify_existing_campaign_lock(campaign)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def verify_existing_campaign_lock(campaign: RCIGV2Campaign) -> Path:
    """Read-only validation used by resume/status and post-run checks."""

    path = campaign.output_root / LOCK_NAME
    if not path.is_file():
        raise RuntimeError("RCIG v2 campaign lock is absent")
    expected = _campaign_lock_payload(campaign)
    observed = json.loads(path.read_text(encoding="utf-8"))
    locked = {key: observed.get(key) for key in expected}
    if locked != expected:
        raise RuntimeError("RCIG v2 campaign lock no longer matches sources")
    return path


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    encoded = (json.dumps(body, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_task_lock(campaign: RCIGV2Campaign, task: RCIGV2Task):
    """Prevent two launchers from executing the same scientific task."""

    output_dir = task_output_dir(campaign, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".rcig_v2_execution.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another v2 launcher owns {task.phase_id}/{task.run_id}"
            ) from exc
        marker = json.dumps(
            {
                "campaign_id": campaign.matrix["campaign_id"],
                "scientific_hash": campaign_scientific_hash(campaign),
                "phase": task.phase_id,
                "run_id": task.run_id,
                "pid": os.getpid(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, marker)
        os.fsync(descriptor)
        yield path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _post_run_device_audit(metrics_path: Path, config: Mapping[str, Any]) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) != 40:
        raise RuntimeError("device audit requires exactly 40 recorded rounds")
    for row in rounds:
        if (
            float(row.get("ldp_gradient_far_private_gradient_mps_fraction", -1.0))
            != 1.0
        ):
            raise RuntimeError("generic private-gradient MPS audit failed")
        if str(row.get("ldp_gradient_far_private_compute_device", "")) not in {
            "mps",
            "mps:0",
        }:
            raise RuntimeError("generic private compute-device audit failed")
        if bool(row.get("far_attack_labels_visible_to_server_aggregate", True)):
            raise RuntimeError("attack-oracle labels crossed the aggregation boundary")
        if bool(row.get("far_attack_config_visible_to_server_aggregate", True)):
            raise RuntimeError("attack configuration crossed the aggregation boundary")
        if not bool(row.get("far_external_attack_diagnostics", False)):
            raise RuntimeError("external attack-diagnostic audit is missing")
        if row.get("far_external_attack_diagnostics_boundary") != (
            "posthoc_simulator_only"
        ):
            raise RuntimeError("external attack-diagnostic boundary is invalid")
        # The current metrics writer keeps scalar fields only. If the core's
        # device inventory is persisted by a future writer, validate it too.
        devices = row.get("ldp_gradient_far_private_compute_devices")
        if devices is not None:
            if isinstance(devices, str):
                devices = [devices]
            if (
                not isinstance(devices, list)
                or not devices
                or any(str(device) not in {"mps", "mps:0"} for device in devices)
            ):
                raise RuntimeError("generic private compute-device inventory failed")
    reference = str(config["training"]["algo_config"].get("robust_reference", ""))
    if reference in RCIG_REFERENCES:
        for row in rounds:
            if float(row.get("rcig_private_gradient_mps_fraction", -1.0)) != 1.0:
                raise RuntimeError("private-gradient MPS audit failed")
            if str(row.get("rcig_private_gradient_compute_device", "")) not in {
                "mps",
                "mps:0",
            }:
                raise RuntimeError("a private gradient was not computed on MPS")
            if str(row.get("rcig_server_aggregation_device", "")) != "cpu":
                raise RuntimeError("server post-processing is not on audited CPU")
            if str(row.get("rcig_server_aggregation_dtype", "")) not in {
                "torch.float64",
                "float64",
            }:
                raise RuntimeError("server post-processing is not audited float64")


def run_task(campaign: RCIGV2Campaign, task: RCIGV2Task, *, resume: bool) -> None:
    if not resume:
        raise ValueError("RCIG v2 execution requires --resume")
    # Deliberately rechecked before *every* task, including completed skips.
    ensure_campaign_lock(campaign)
    assert_dependencies_promoted(campaign, task.phase_id)
    with _exclusive_task_lock(campaign, task):
        # Recheck after acquiring the per-task lock so a concurrent process
        # cannot race a completed-skip decision.
        verify_existing_campaign_lock(campaign)
        output_dir = task_output_dir(campaign, task)
        if task_is_complete(campaign, task):
            print(f"SKIP completed {task.phase_id}/{task.run_id}")
            return
        config = resolved_config(campaign, task, output_dir)
        _validate_resolved_config(config, task)
        serialized = _serialized_config(config)
        config_path = output_dir / "resolved_config.yaml"
        if (
            config_path.exists()
            and config_path.read_text(encoding="utf-8") != serialized
        ):
            raise RuntimeError(f"refusing to overwrite changed config {config_path}")
        config_path.write_text(serialized, encoding="utf-8")
        config_hash = _file_sha256(config_path)
        algo = config["training"]["algo_config"]
        attack = algo["attack"]
        status_path = output_dir / "orchestration_status.json"
        status = {
            "campaign_id": campaign.matrix["campaign_id"],
            "phase": task.phase_id,
            "run_id": task.run_id,
            "status": "running",
            "device": "mps",
            "mps_fallback": 0,
            "server_postprocessing": "cpu_float64",
            "scientific_hash": campaign_scientific_hash(campaign),
            "resolved_config_sha256": config_hash,
            "pairing_seed_block": task.seed,
            "pairing_design_sha256": algo["rcig_pairing_design_sha256"],
            "attack_ids": attack.get("client_ids", []),
            "attack_start": attack.get("active_round_start"),
            "attack_end": attack.get("active_round_end"),
        }
        _write_status(status_path, status)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "run_experiment.py"),
            "--config",
            str(config_path),
            "--device",
            "mps",
            "--output",
            str(output_dir),
        ]
        environment = os.environ.copy()
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        print(f"RUN g={task.global_index:03d} {task.phase_id}/{task.run_id}")
        try:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            metrics_path = _metrics_path(output_dir)
            if metrics_path is None:
                raise RuntimeError(
                    "training returned without one complete metrics.json"
                )
            _post_run_device_audit(metrics_path, config)
            # Detect any source/config drift that happened while this sub-run
            # was executing, before it can be labelled completed.
            verify_existing_campaign_lock(campaign)
        except Exception as exc:
            status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            _write_status(status_path, status)
            raise
        status.update(
            {"status": "completed", "metrics_sha256": _file_sha256(metrics_path)}
        )
        _write_status(status_path, status)


def record_gate(
    campaign: RCIGV2Campaign, phase_id: str, evidence: Mapping[str, Any]
) -> Path:
    ensure_campaign_lock(campaign)
    criteria = phase_definition(campaign, phase_id).get("gate_criteria", [])
    evaluations = []
    for criterion in criteria:
        identifier = str(criterion["id"])
        value = evidence.get(identifier)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"gate evidence {identifier!r} must be numeric")
        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError(f"gate evidence {identifier!r} must be finite")
        operator = str(criterion["op"])
        threshold = float(criterion["threshold"])
        if operator == "==":
            passed = math.isclose(observed, threshold, abs_tol=1e-12)
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == ">=":
            passed = observed >= threshold
        else:
            raise ValueError(f"unsupported gate operator {operator!r}")
        evaluations.append(
            {
                "id": identifier,
                "observed": observed,
                "op": operator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    decision = (
        "promote"
        if evaluations and all(row["passed"] for row in evaluations)
        else "stop"
    )
    evidence_payload = dict(evidence)
    evidence_hash = _canonical_hash(evidence_payload)
    payload = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": phase_id,
        "decision": decision,
        "campaign_scientific_hash": campaign_scientific_hash(campaign),
        "criteria": copy.deepcopy(criteria),
        "evaluations": evaluations,
        "evidence": evidence_payload,
        "evidence_sha256": evidence_hash,
        "immutable": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = gate_path(campaign, phase_id)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = dict(previous)
        new = dict(payload)
        old.pop("recorded_at_utc", None)
        new.pop("recorded_at_utc", None)
        if old != new:
            raise RuntimeError(f"immutable RCIG v2 gate already exists at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = dict(previous)
        new = dict(payload)
        old.pop("recorded_at_utc", None)
        new.pop("recorded_at_utc", None)
        if old != new:
            raise RuntimeError(f"immutable RCIG v2 gate already exists at {path}")
        return path
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def evaluate_and_record_gate(campaign: RCIGV2Campaign, phase_id: str) -> str:
    ensure_campaign_lock(campaign)
    if not bool(phase_definition(campaign, phase_id).get("gate_required", False)):
        return "not_required"
    from scripts.analyze_rcig_ldp_gradient_far_v2 import evaluate_gate

    evidence = evaluate_gate(campaign, phase_id)
    path = record_gate(campaign, phase_id, evidence)
    decision = str(json.loads(path.read_text(encoding="utf-8"))["decision"])
    print(f"GATE {phase_id}: {decision} ({path})")
    return decision


def list_campaign(campaign: RCIGV2Campaign, selected: Iterable[RCIGV2Task]) -> None:
    print(
        f"campaign={campaign.matrix['campaign_id']} device=mps tasks={len(campaign.tasks)} "
        f"hash={campaign_scientific_hash(campaign)}"
    )
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        tasks = tasks_for_phase(campaign, phase_id)
        complete = sum(task_is_complete(campaign, task) for task in tasks)
        print(
            f"phase={phase_id} complete={complete}/{len(tasks)} gate={_gate_decision(campaign, phase_id)}"
        )
    for task in selected:
        print(
            f"g={task.global_index:03d} p={task.phase_index:03d} "
            f"{task.phase_id}/{task.run_id}"
        )


def _selected_tasks(
    campaign: RCIGV2Campaign, phase: str | None, job_index: int | None
) -> tuple[RCIGV2Task, ...]:
    pool = tasks_for_phase(campaign, phase) if phase else campaign.tasks
    if job_index is None:
        return tuple(pool)
    if not 0 <= job_index < len(pool):
        raise IndexError(f"job-index must lie in [0,{len(pool) - 1}]")
    return (pool[job_index],)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--phase")
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-phase", action="store_true")
    parser.add_argument("--run-chain", action="store_true")
    parser.add_argument("--evaluate-gate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    validate_requested_device(args.device)
    campaign = load_campaign(args.matrix)
    selected = _selected_tasks(campaign, args.phase, args.job_index)
    actions = (args.run, args.run_phase, args.run_chain, args.evaluate_gate)
    if args.list or args.status or not any(actions):
        list_campaign(campaign, selected)
        return
    if sum(bool(value) for value in actions) != 1:
        parser.error("choose exactly one execution action")
    if args.evaluate_gate:
        if not args.phase:
            parser.error("--evaluate-gate requires --phase")
        evaluate_and_record_gate(campaign, args.phase)
        return
    if not args.resume:
        parser.error("every RCIG v2 training action requires --resume")
    require_working_mps()
    if args.run:
        if len(selected) != 1:
            parser.error("--run requires --job-index")
        run_task(campaign, selected[0], resume=True)
        return
    if args.run_phase:
        if not args.phase or args.job_index is not None:
            parser.error("--run-phase requires --phase and no --job-index")
        for task in tasks_for_phase(campaign, args.phase):
            run_task(campaign, task, resume=True)
        evaluate_and_record_gate(campaign, args.phase)
        return
    if args.phase or args.job_index is not None:
        parser.error("--run-chain does not accept phase/job selection")
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        for task in tasks_for_phase(campaign, phase_id):
            run_task(campaign, task, resume=True)
        if evaluate_and_record_gate(campaign, phase_id) != "promote":
            print(f"STOP fail-closed after {phase_id}")
            return
    print("RCIG v2 gated chain completed")


if __name__ == "__main__":
    main()
