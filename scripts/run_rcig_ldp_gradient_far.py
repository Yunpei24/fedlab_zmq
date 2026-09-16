#!/usr/bin/env python3
"""Fail-closed runner for the gated RCIG/LDP-gradient-FAR campaign.

The scientific matrix is immutable once the first task starts.  Every training
action requires native Apple MPS, disables MPS CPU fallback, uses ``--resume``
semantics, and refuses to cross a phase boundary unless the preceding numeric
gate was evaluated and promoted automatically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "ldp_gradient_far" / "rcig_end_to_end_v1.yaml"
PROTOCOL_PATH = (
    ROOT / "output" / "analysis" / "RCIG_LDP_Gradient_FAR_Protocol_PreRun.md"
)
ALLOWED_REFERENCES = {
    "centered_clipping",
    "rfa",
    "trimmed_mean",
    "rcig_temporal",
}
ALLOWED_ATTACKS = {"none", "bf", "ipm", "alie"}
RCIG_MODES = {"full", "isotropic", "euclidean"}


@dataclass(frozen=True)
class RCIGTask:
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
class RCIGCampaign:
    matrix_path: Path
    matrix: dict[str, Any]
    base: dict[str, Any]
    output_root: Path
    tasks: tuple[RCIGTask, ...]

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
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_scientific_hash(campaign: RCIGCampaign) -> str:
    return _canonical_hash({"matrix": campaign.matrix, "base": campaign.base})


def _expand_phase(
    phase: Mapping[str, Any], *, first_global_index: int
) -> list[RCIGTask]:
    phase_id = str(phase["id"])
    axes = phase.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError(f"phase {phase_id!r} must define non-empty axes")
    names = [str(axis["name"]) for axis in axes]
    if len(names) != len(set(names)):
        raise ValueError(f"phase {phase_id!r} contains duplicate axis names")
    value_sets: list[list[Mapping[str, Any]]] = []
    for axis in axes:
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {axis['name']!r} has no values")
        identifiers = [str(value["id"]) for value in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"axis {axis['name']!r} has duplicate ids")
        value_sets.append(values)
    seeds = [int(seed) for seed in phase.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"phase {phase_id!r} needs unique non-empty seeds")

    common = phase.get("common_overrides", {})
    tasks: list[RCIGTask] = []
    for values in itertools.product(*value_sets):
        overrides = copy.deepcopy(common)
        axis_values: dict[str, str] = {}
        for name, value in zip(names, values):
            axis_values[name] = str(value["id"])
            overrides = deep_merge(overrides, value.get("overrides", {}))
        for seed in seeds:
            tasks.append(
                RCIGTask(
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


def load_campaign(matrix_path: Path = DEFAULT_MATRIX) -> RCIGCampaign:
    matrix_path = matrix_path.resolve()
    matrix = _read_yaml(matrix_path)
    if int(matrix.get("schema_version", 0)) != 1:
        raise ValueError("unsupported RCIG campaign schema")
    if str(matrix.get("required_device", "")).lower() != "mps":
        raise ValueError("RCIG campaign must declare required_device: mps")
    base_path = (matrix_path.parent / str(matrix["base_config"])).resolve()
    base = _read_yaml(base_path)
    output_root = (matrix_path.parent / str(matrix["output_root"])).resolve()

    phases = matrix.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("the RCIG matrix needs at least one phase")
    phase_ids = [str(phase["id"]) for phase in phases]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("RCIG phase ids must be unique")
    seen: set[str] = set()
    for phase in phases:
        dependencies = [str(item) for item in phase.get("depends_on", [])]
        unknown = [item for item in dependencies if item not in seen]
        if unknown:
            raise ValueError(
                f"phase {phase['id']!r} has forward/unknown dependencies {unknown}"
            )
        seen.add(str(phase["id"]))

    tasks: list[RCIGTask] = []
    for phase in phases:
        tasks.extend(_expand_phase(phase, first_global_index=len(tasks)))
    if len({(task.phase_id, task.run_id) for task in tasks}) != len(tasks):
        raise ValueError("expanded RCIG run ids are not unique")
    campaign = RCIGCampaign(
        matrix_path=matrix_path,
        matrix=matrix,
        base=base,
        output_root=output_root,
        tasks=tuple(tasks),
    )
    validate_campaign(campaign)
    return campaign


def phase_definition(campaign: RCIGCampaign, phase_id: str) -> dict[str, Any]:
    for phase in campaign.phases:
        if str(phase["id"]) == phase_id:
            return phase
    raise KeyError(f"unknown RCIG phase {phase_id!r}")


def tasks_for_phase(campaign: RCIGCampaign, phase_id: str) -> tuple[RCIGTask, ...]:
    phase_definition(campaign, phase_id)
    return tuple(task for task in campaign.tasks if task.phase_id == phase_id)


def task_output_dir(campaign: RCIGCampaign, task: RCIGTask) -> Path:
    return campaign.output_root / task.phase_id / task.run_id


def gate_path(campaign: RCIGCampaign, phase_id: str) -> Path:
    return campaign.output_root / "_gates" / f"{phase_id}.json"


def _noise_regime(task: RCIGTask) -> str:
    return task.axis_values.get("noise_regime", "homogeneous")


def _threshold_from_calibration(
    campaign: RCIGCampaign, task: RCIGTask, covariance_mode: str
) -> tuple[float, str]:
    thresholds, artifact_hash = _thresholds_from_calibration(campaign, task)
    try:
        threshold = thresholds[covariance_mode]
    except KeyError as exc:
        raise RuntimeError(
            f"missing calibrated threshold for {_noise_regime(task)}/{covariance_mode}"
        ) from exc
    return threshold, artifact_hash


def _thresholds_from_calibration(
    campaign: RCIGCampaign, task: RCIGTask
) -> tuple[dict[str, float], str]:
    """Load all paired RCIG thresholds for one public noise regime.

    Every RCIG run emits the full, isotropic and Euclidean counterfactual
    diagnostics.  Injecting only the deployed mode would silently leave the
    other two controls at algorithm defaults and invalidate paired gate
    comparisons.
    """

    path = gate_path(campaign, "r0_calibration_frozen")
    if not path.exists():
        raise RuntimeError("the immutable RCIG threshold artifact is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != "promote":
        raise RuntimeError("RCIG calibration did not promote")
    if payload.get("campaign_scientific_hash") != campaign_scientific_hash(campaign):
        raise RuntimeError("RCIG calibration artifact is stale")
    thresholds = payload.get("evidence", {}).get("thresholds_by_regime_and_mode")
    try:
        regime_thresholds = {
            mode: float(thresholds[_noise_regime(task)][mode])
            for mode in sorted(RCIG_MODES)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"missing calibrated thresholds for {_noise_regime(task)}"
        ) from exc
    if any(
        not math.isfinite(threshold) or threshold <= 0.0
        for threshold in regime_thresholds.values()
    ):
        raise RuntimeError("calibrated RCIG thresholds must be finite and positive")
    return regime_thresholds, _file_sha256(path)


def resolved_config(
    campaign: RCIGCampaign,
    task: RCIGTask,
    output_dir: Path | None = None,
    *,
    inject_threshold: bool = True,
) -> dict[str, Any]:
    config = deep_merge(campaign.base, task.overrides)
    config["seed"] = task.seed
    config["device"] = "mps"
    config.setdefault("data", {})["partition_seed"] = task.seed
    config["output_dir"] = str(output_dir or task_output_dir(campaign, task))
    rounds = int(config["training"]["num_rounds"])
    algo = config["training"]["algo_config"]
    algo["privacy_num_rounds"] = rounds
    algo["expected_num_clients"] = 25
    algo["num_byzantine"] = 5
    attack = dict(algo.get("attack") or {})
    if attack.get("enabled", False):
        attack.update({"num_byzantine": 5, "client_ids": [0, 1, 2, 3, 4]})
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
    algo["attack"] = attack
    if (
        algo.get("robust_reference") == "rcig_temporal"
        and not bool(algo.get("rcig_calibration_mode", False))
        and inject_threshold
    ):
        thresholds, artifact_hash = _thresholds_from_calibration(campaign, task)
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
            "rcig_required_device": "mps",
            "rcig_mps_fallback": 0,
        }
    )
    return config


def _validate_resolved_config(config: Mapping[str, Any], task: RCIGTask) -> None:
    if config.get("device") != "mps":
        raise ValueError("RCIG training is MPS-only")
    if config["training"].get("algorithm") != "ldp_gradient_far":
        raise ValueError("RCIG matrix must execute ldp_gradient_far")
    if config["data"].get("dataset") != "fashionmnist":
        raise ValueError("RCIG v1 is locked to Fashion-MNIST")
    if config["model"].get("architecture") != "lenet5_tanh":
        raise ValueError("RCIG v1 is locked to LeNet5-tanh")
    if config["data"].get("partition") != "client_dirichlet_balanced":
        raise ValueError("RCIG v1 requires fixed-size client partitions")
    clients = config["clients"]
    if int(clients.get("num_clients", -1)) != 25:
        raise ValueError("RCIG v1 requires n=25")
    if int(clients.get("min_clients", -1)) != 25:
        raise ValueError("RCIG v1 requires all clients")
    if float(clients.get("sample_fraction", 0.0)) != 1.0:
        raise ValueError("RCIG v1 requires complete participation")
    if float(clients.get("dropout_rate", -1.0)) != 0.0:
        raise ValueError("client dropout is excluded from RCIG v1")

    algo = config["training"]["algo_config"]
    fixed_protocol = {
        "batch_size": 120,
        "fixed_batch_size": 120,
        "privacy_public_dataset_size": 2400,
        "local_epochs": 1,
        "fixed_steps_per_round": 1,
        "expected_num_clients": 25,
    }
    for key, expected in fixed_protocol.items():
        if int(algo.get(key, -1)) != expected:
            raise ValueError(f"RCIG v1 requires {key}={expected}")
    if algo.get("sampling_scheme") != "fixed_without_replacement":
        raise ValueError("RCIG v1 requires fixed sampling without replacement")
    if algo.get("privacy_adjacency") != "replace_one":
        raise ValueError("RCIG v1 uses sample-level replace-one adjacency")
    if not math.isclose(
        float(algo.get("privacy_sampling_rate_override", -1.0)),
        0.05,
        abs_tol=1.0e-12,
    ):
        raise ValueError("RCIG v1 requires B/N_i=0.05")
    if not bool(algo.get("enable_dp", False)):
        raise ValueError("all RCIG v1 arms must release private gradients")
    if not math.isclose(float(algo.get("target_epsilon", -1.0)), 4.0):
        raise ValueError("RCIG v1 targets epsilon=4")
    if not math.isclose(float(algo.get("delta", -1.0)), 1.0e-5):
        raise ValueError("RCIG v1 targets delta=1e-5")
    if not math.isclose(float(algo.get("clip_norm", -1.0)), 4.0):
        raise ValueError("RCIG v1 requires local C=4")
    if algo.get("per_sample_backend") != "vectorized":
        raise ValueError("RCIG v1 requires true per-example gradient clipping")
    if algo.get("far_score_mode") != "raw_distance":
        raise ValueError("RCIG v1 does not use a bounded score transform")
    reference = str(algo.get("robust_reference"))
    if reference not in ALLOWED_REFERENCES:
        raise ValueError(f"unsupported RCIG comparator {reference!r}")
    if reference == "rcig_temporal":
        mode = str(algo.get("rcig_covariance_mode"))
        if mode not in RCIG_MODES:
            raise ValueError(f"unsupported RCIG covariance mode {mode!r}")
        if int(algo.get("rcig_old_window", -1)) != int(algo.get("rcig_new_window", -2)):
            raise ValueError("RCIG v1 requires equally sized old/new windows")
        if int(algo.get("rcig_public_subspace_dimension", 0)) not in {32, 64}:
            raise ValueError("RCIG v1 public gate subspace must have dimension 32/64")
        threshold = float(algo.get("rcig_innovation_threshold", math.nan))
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("RCIG threshold must be finite and positive")
        if algo.get("rcig_warmup_policy") != "uniform":
            raise ValueError("RCIG cold-start must use uniform aggregation")
    attack = algo.get("attack", {})
    if str(attack.get("name", "none")) not in ALLOWED_ATTACKS:
        raise ValueError("current end-to-end API does not support this attack")
    if attack.get("enabled", False):
        if attack.get("client_ids") != [0, 1, 2, 3, 4]:
            raise ValueError("RCIG attacks require five fixed Byzantine ids")
        start, end = attack.get("active_round_start"), attack.get("active_round_end")
        if (start is None) != (end is None):
            raise ValueError("attack schedule must provide both start and end")
        if start is not None and not (
            1 <= int(start) <= int(end) <= int(config["training"]["num_rounds"])
        ):
            raise ValueError("invalid RCIG attack interval")
    if task.phase_id in {
        "r0_calibration_frozen",
        "r1_null_validation_frozen",
        "r2_attack_mechanism_frozen",
    } and not math.isclose(float(algo.get("far_server_lr", math.nan)), 0.0):
        raise ValueError("mechanistic phases require a frozen global model")
    if task.phase_id.startswith("r3_") or task.phase_id.startswith("r4_"):
        if int(config["training"]["num_rounds"]) != 40:
            raise ValueError("end-to-end RCIG phases require T=40")


def validate_campaign(campaign: RCIGCampaign) -> None:
    expected = {
        str(key): int(value)
        for key, value in campaign.matrix.get("expected_task_counts", {}).items()
    }
    observed = {
        str(phase["id"]): len(tasks_for_phase(campaign, str(phase["id"])))
        for phase in campaign.phases
    }
    if observed != expected:
        raise ValueError(f"unexpected RCIG task counts {observed}; expected {expected}")
    registries = campaign.matrix.get("randomness", {})
    registry_names = (
        "calibration_seeds",
        "null_validation_seeds",
        "mechanistic_seeds",
        "development_seeds",
        "confirmation_seeds",
    )
    registry_sets = {
        name: {int(value) for value in registries[name]} for name in registry_names
    }
    for left_index, left in enumerate(registry_names):
        for right in registry_names[left_index + 1 :]:
            if registry_sets[left] & registry_sets[right]:
                raise ValueError(f"RCIG seed registries {left}/{right} overlap")
    for phase in campaign.phases:
        registry = str(phase["seed_registry"])
        phase_seeds = {
            task.seed for task in tasks_for_phase(campaign, str(phase["id"]))
        }
        if phase_seeds != registry_sets[registry]:
            raise ValueError(f"phase {phase['id']!r} does not match {registry!r}")
        criteria = phase.get("gate_criteria", [])
        if bool(phase.get("gate_required", False)) and not criteria:
            raise ValueError(f"gated phase {phase['id']!r} has no numeric criteria")
        ids = [str(item.get("id", "")) for item in criteria]
        if len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
            raise ValueError(f"invalid gate criterion ids in {phase['id']!r}")
        for criterion in criteria:
            if criterion.get("op") not in {"==", "<=", ">="}:
                raise ValueError("unsupported RCIG gate operator")
            threshold = criterion.get("threshold")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("RCIG gate thresholds must be numeric")
    for task in campaign.tasks:
        # Downstream thresholds do not exist during static validation.  A large
        # finite placeholder is used only to validate the rest of the contract.
        config = resolved_config(campaign, task, inject_threshold=False)
        algo = config["training"]["algo_config"]
        if algo.get("robust_reference") == "rcig_temporal" and not bool(
            algo.get("rcig_calibration_mode", False)
        ):
            algo["rcig_innovation_threshold"] = 1.0
        _validate_resolved_config(config, task)


def validate_requested_device(device: str) -> None:
    if str(device).lower() != "mps":
        raise ValueError("RCIG campaign refuses CPU and CUDA")


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
        _ = (probe.square().sum()).cpu()
        torch.mps.synchronize()
    except Exception as exc:  # pragma: no cover - depends on local Metal runtime
        raise RuntimeError("MPS execution probe failed") from exc


def _metrics_path(output_dir: Path) -> Path | None:
    for path in sorted(output_dir.glob("**/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rounds = payload.get("rounds")
        expected = int(payload.get("summary", {}).get("num_rounds", -1))
        if isinstance(rounds, list) and expected > 0 and len(rounds) == expected:
            return path
    return None


def task_is_complete(campaign: RCIGCampaign, task: RCIGTask) -> bool:
    output_dir = task_output_dir(campaign, task)
    status_path = output_dir / "orchestration_status.json"
    if not status_path.exists() or _metrics_path(output_dir) is None:
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "completed"


def phase_is_complete(campaign: RCIGCampaign, phase_id: str) -> bool:
    return all(
        task_is_complete(campaign, task) for task in tasks_for_phase(campaign, phase_id)
    )


def _gate_decision(campaign: RCIGCampaign, phase_id: str) -> str | None:
    path = gate_path(campaign, phase_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if payload.get("campaign_scientific_hash") != campaign_scientific_hash(campaign):
        return "stale"
    return str(payload.get("decision"))


def assert_dependencies_promoted(campaign: RCIGCampaign, phase_id: str) -> None:
    for dependency in phase_definition(campaign, phase_id).get("depends_on", []):
        dependency = str(dependency)
        if not phase_is_complete(campaign, dependency):
            raise RuntimeError(f"dependency {dependency!r} is incomplete")
        if _gate_decision(campaign, dependency) != "promote":
            raise RuntimeError(
                f"dependency {dependency!r} has not promoted; RCIG chain is stopped"
            )


def _campaign_lock_payload(campaign: RCIGCampaign) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError(f"pre-registered protocol is absent: {PROTOCOL_PATH}")
    source_candidates = (
        ROOT / "algorithms" / "ldp_gradient_far.py",
        ROOT / "algorithms" / "far.py",
        ROOT / "algorithms" / "dp_references.py",
        ROOT / "algorithms" / "gaussian_aware_reference_k7_rcig.py",
        ROOT / "algorithms" / "rcig_temporal_reference.py",
        ROOT / "privacy" / "local_dpsgd.py",
        ROOT / "attacks" / "byzantine.py",
        ROOT / "run_experiment.py",
        ROOT / "scripts" / "analyze_rcig_ldp_gradient_far.py",
    )
    payload = {
        "campaign_id": campaign.matrix["campaign_id"],
        "scientific_hash": campaign_scientific_hash(campaign),
        "matrix_sha256": _file_sha256(campaign.matrix_path),
        "base_sha256": _canonical_hash(campaign.base),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "source_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in source_candidates
            if path.exists()
        },
        "device": "mps",
        "mps_fallback": 0,
    }
    return payload


def ensure_campaign_lock(campaign: RCIGCampaign) -> Path:
    path = campaign.output_root / "_campaign_lock.json"
    expected = _campaign_lock_payload(campaign)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        locked = {key: observed.get(key) for key in expected}
        if locked != expected:
            raise RuntimeError("RCIG campaign lock does not match current sources")
        return path
    campaign.output_root.mkdir(parents=True, exist_ok=True)
    expected["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return path


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def run_task(campaign: RCIGCampaign, task: RCIGTask, *, resume: bool) -> None:
    if not resume:
        raise ValueError("RCIG execution requires --resume")
    assert_dependencies_promoted(campaign, task.phase_id)
    output_dir = task_output_dir(campaign, task)
    if task_is_complete(campaign, task):
        print(f"SKIP completed {task.phase_id}/{task.run_id}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolved_config(campaign, task, output_dir)
    _validate_resolved_config(config, task)
    resolved_path = output_dir / "resolved_config.yaml"
    serialized = yaml.safe_dump(config, sort_keys=False)
    if (
        resolved_path.exists()
        and resolved_path.read_text(encoding="utf-8") != serialized
    ):
        raise RuntimeError(f"refusing to overwrite changed config {resolved_path}")
    resolved_path.write_text(serialized, encoding="utf-8")
    status_path = output_dir / "orchestration_status.json"
    status = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": task.phase_id,
        "run_id": task.run_id,
        "status": "running",
        "device": "mps",
        "mps_fallback": 0,
        "scientific_hash": campaign_scientific_hash(campaign),
    }
    _write_status(status_path, status)
    command = [
        sys.executable,
        "-u",
        str(ROOT / "run_experiment.py"),
        "--config",
        str(resolved_path),
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
        if _metrics_path(output_dir) is None:
            raise RuntimeError("training returned without complete metrics.json")
    except Exception as exc:
        status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_status(status_path, status)
        raise
    status["status"] = "completed"
    _write_status(status_path, status)


def record_gate(
    campaign: RCIGCampaign, phase_id: str, evidence: Mapping[str, Any]
) -> Path:
    phase = phase_definition(campaign, phase_id)
    criteria = phase.get("gate_criteria", [])
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
            passed = math.isclose(observed, threshold, abs_tol=1.0e-12)
        elif operator == "<=":
            passed = observed <= threshold
        elif operator == ">=":
            passed = observed >= threshold
        else:  # statically validated
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
    path = gate_path(campaign, phase_id)
    payload = {
        "campaign_id": campaign.matrix["campaign_id"],
        "phase": phase_id,
        "decision": decision,
        "campaign_scientific_hash": campaign_scientific_hash(campaign),
        "criteria": copy.deepcopy(criteria),
        "evaluations": evaluations,
        "evidence": dict(evidence),
        "immutable": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            # Timestamps make byte equality impossible, so accept only the
            # same immutable scientific content.
            comparable = dict(payload)
            comparable.pop("recorded_at_utc", None)
            old = dict(existing)
            old.pop("recorded_at_utc", None)
            if old != comparable:
                raise RuntimeError(f"immutable RCIG gate already exists at {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def evaluate_and_record_gate(campaign: RCIGCampaign, phase_id: str) -> str:
    if not bool(phase_definition(campaign, phase_id).get("gate_required", False)):
        return "not_required"
    from scripts.analyze_rcig_ldp_gradient_far import evaluate_gate  # local import

    evidence = evaluate_gate(campaign, phase_id)
    path = record_gate(campaign, phase_id, evidence)
    decision = json.loads(path.read_text(encoding="utf-8"))["decision"]
    print(f"GATE {phase_id}: {decision} ({path})")
    return str(decision)


def list_campaign(campaign: RCIGCampaign, selected: Iterable[RCIGTask]) -> None:
    print(
        f"campaign={campaign.matrix['campaign_id']} device=mps "
        f"tasks={len(campaign.tasks)} hash={campaign_scientific_hash(campaign)}"
    )
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        tasks = tasks_for_phase(campaign, phase_id)
        complete = sum(task_is_complete(campaign, task) for task in tasks)
        print(
            f"phase={phase_id} complete={complete}/{len(tasks)} "
            f"gate={_gate_decision(campaign, phase_id)}"
        )
    for task in selected:
        print(
            f"g={task.global_index:03d} p={task.phase_index:03d} "
            f"{task.phase_id}/{task.run_id}"
        )


def _selected_tasks(
    campaign: RCIGCampaign, phase: str | None, job_index: int | None
) -> tuple[RCIGTask, ...]:
    pool = tasks_for_phase(campaign, phase) if phase else campaign.tasks
    if job_index is None:
        return tuple(pool)
    if not 0 <= job_index < len(pool):
        raise IndexError(f"job-index must be in [0,{len(pool) - 1}]")
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
    if (
        args.list
        or args.status
        or not any((args.run, args.run_phase, args.run_chain, args.evaluate_gate))
    ):
        list_campaign(campaign, selected)
        return
    if (
        sum(
            bool(value)
            for value in (args.run, args.run_phase, args.run_chain, args.evaluate_gate)
        )
        != 1
    ):
        parser.error("choose exactly one execution action")
    if args.evaluate_gate:
        if not args.phase:
            parser.error("--evaluate-gate requires --phase")
        evaluate_and_record_gate(campaign, args.phase)
        return
    if not args.resume:
        parser.error("every RCIG training action requires --resume")
    require_working_mps()
    ensure_campaign_lock(campaign)
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
        if phase_definition(campaign, args.phase).get("gate_required", False):
            evaluate_and_record_gate(campaign, args.phase)
        return

    if args.phase or args.job_index is not None:
        parser.error("--run-chain does not accept phase/job selection")
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        for task in tasks_for_phase(campaign, phase_id):
            run_task(campaign, task, resume=True)
        if bool(phase.get("gate_required", False)):
            decision = evaluate_and_record_gate(campaign, phase_id)
            if decision != "promote":
                print(f"STOP fail-closed after {phase_id}")
                return
    print("RCIG gated chain completed")


if __name__ == "__main__":
    main()
