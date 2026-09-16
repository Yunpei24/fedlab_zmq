#!/usr/bin/env python3
"""List, validate and run the gated LDP-gradient-FAR positioning campaign.

The harness deliberately refuses CPU and CUDA.  Listing and static validation
remain available on any host, but a training action requires a working Apple
Metal backend and probes it before creating an output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "ldp_gradient_far" / "positioning_v1.yaml"
TRAINSET_SIZES = {"mnist": 60_000, "fashionmnist": 60_000}
ALLOWED_REFERENCES = {
    "coordinate_median",
    "trimmed_mean",
    "rfa",
    "centered_clipping",
}
ALLOWED_ATTACKS = {"none", "bf", "ipm", "alie", "minmax", "minsum"}
GATE_OPERATORS = {"<=", ">=", "=="}


@dataclass(frozen=True)
class PositioningTask:
    global_index: int
    phase_index: int
    phase_id: str
    phase_role: str
    variant_id: str
    seed: int
    overrides: dict[str, Any]

    @property
    def run_id(self) -> str:
        return f"{self.variant_id}_seed{self.seed}"


@dataclass(frozen=True)
class Campaign:
    matrix_path: Path
    matrix: dict[str, Any]
    base: dict[str, Any]
    output_root: Path
    tasks: tuple[PositioningTask, ...]

    @property
    def phases(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.matrix["phases"])


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _expand_phase(
    phase: dict[str, Any], *, first_global_index: int
) -> list[PositioningTask]:
    phase_id = str(phase["id"])
    axes = phase.get("axes", [])
    if not axes:
        raise ValueError(f"phase {phase_id!r} must contain at least one axis")
    axis_names = [str(axis["name"]) for axis in axes]
    if len(axis_names) != len(set(axis_names)):
        raise ValueError(f"phase {phase_id!r} has duplicate axis names")
    value_sets = []
    for axis in axes:
        values = axis.get("values", [])
        value_ids = [str(value["id"]) for value in values]
        if not values or len(value_ids) != len(set(value_ids)):
            raise ValueError(
                f"axis {axis['name']!r} in {phase_id!r} needs unique values"
            )
        value_sets.append(values)
    seeds = [int(seed) for seed in phase.get("seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"phase {phase_id!r} needs unique non-empty seeds")

    tasks: list[PositioningTask] = []
    phase_index = 0
    common = phase.get("common_overrides", {})
    for values in itertools.product(*value_sets):
        overrides = copy.deepcopy(common)
        value_ids = []
        for value in values:
            value_ids.append(str(value["id"]))
            overrides = deep_merge(overrides, value.get("overrides", {}))
        variant_id = "__".join(value_ids)
        for seed in seeds:
            tasks.append(
                PositioningTask(
                    global_index=first_global_index + len(tasks),
                    phase_index=phase_index,
                    phase_id=phase_id,
                    phase_role=str(phase.get("role", "unspecified")),
                    variant_id=variant_id,
                    seed=seed,
                    overrides=overrides,
                )
            )
            phase_index += 1
    return tasks


def load_campaign(matrix_path: Path = DEFAULT_MATRIX) -> Campaign:
    matrix_path = matrix_path.resolve()
    matrix = _read_yaml(matrix_path)
    schema_version = int(matrix.get("schema_version", 0))
    if schema_version not in {1, 2}:
        raise ValueError("unsupported positioning matrix schema")
    if str(matrix.get("required_device", "")).lower() != "mps":
        raise ValueError("the positioning campaign must declare required_device: mps")
    base_path = (matrix_path.parent / str(matrix["base_config"])).resolve()
    base = _read_yaml(base_path)
    output_root = (matrix_path.parent / str(matrix["output_root"])).resolve()

    phases = matrix.get("phases", [])
    phase_ids = [str(phase["id"]) for phase in phases]
    if not phases or len(phase_ids) != len(set(phase_ids)):
        raise ValueError("phases must be non-empty and have unique ids")
    seen: set[str] = set()
    for phase in phases:
        dependencies = [str(item) for item in phase.get("depends_on", [])]
        unknown_or_forward = [item for item in dependencies if item not in seen]
        if unknown_or_forward:
            raise ValueError(
                f"phase {phase['id']!r} has unknown/forward dependencies: "
                f"{unknown_or_forward}"
            )
        seen.add(str(phase["id"]))

    tasks: list[PositioningTask] = []
    for phase in phases:
        tasks.extend(_expand_phase(phase, first_global_index=len(tasks)))
    run_keys = [(task.phase_id, task.run_id) for task in tasks]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("expanded positioning task ids are not unique")

    campaign = Campaign(
        matrix_path=matrix_path,
        matrix=matrix,
        base=base,
        output_root=output_root,
        tasks=tuple(tasks),
    )
    validate_campaign(campaign)
    return campaign


def _canonical_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def campaign_config_hash(campaign: Campaign) -> str:
    """Hash the immutable scientific matrix together with its base config."""

    return _canonical_hash({"matrix": campaign.matrix, "base": campaign.base})


def phase_config_hash(campaign: Campaign, phase_id: str) -> str:
    """Hash one phase in the context of the exact campaign configuration."""

    return _canonical_hash(
        {
            "campaign_config_hash": campaign_config_hash(campaign),
            "phase": _phase_definition(campaign, phase_id),
        }
    )


def _phase_definition(campaign: Campaign, phase_id: str) -> dict[str, Any]:
    for phase in campaign.phases:
        if str(phase["id"]) == phase_id:
            return phase
    raise KeyError(f"unknown phase {phase_id!r}")


def resolved_config(
    campaign: Campaign,
    task: PositioningTask,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    config = deep_merge(campaign.base, task.overrides)
    config["seed"] = int(task.seed)
    config["device"] = "mps"
    config.setdefault("data", {})["partition_seed"] = int(task.seed)
    config["output_dir"] = str(
        output_dir
        if output_dir is not None
        else campaign.output_root / task.phase_id / task.run_id
    )

    rounds = int(config["training"]["num_rounds"])
    algo = config["training"]["algo_config"]
    algo["privacy_num_rounds"] = rounds
    algo["tilt_bound_policy"] = "diagnostic"
    n = int(config["clients"]["num_clients"])
    assumed_byzantine = int(algo.get("num_byzantine", 0))
    attack = dict(algo.get("attack") or {})
    if attack.get("enabled", False):
        attack["num_byzantine"] = assumed_byzantine
        attack["client_ids"] = list(range(assumed_byzantine))
    else:
        attack.update({"enabled": False, "name": "none", "num_byzantine": 0})
        attack["client_ids"] = []
    algo["attack"] = attack
    algo["expected_num_clients"] = n
    public_n = int(algo["privacy_public_dataset_size"])
    fixed_batch = int(algo["fixed_batch_size"])
    algo.update(
        {
            "positioning_num_clients": n,
            "positioning_global_train_size": TRAINSET_SIZES[
                str(config["data"]["dataset"])
            ],
            "positioning_public_local_dataset_size": public_n,
            "positioning_fixed_batch_size": fixed_batch,
            "positioning_fixed_batch_fraction": fixed_batch / public_n,
            "positioning_cross_n_scale_confound": (
                "at_fixed_global_N_and_q, changing_n_changes_public_N_i_and_batch_B_i; "
                "primary causal comparisons are paired within_n"
            ),
        }
    )
    return config


def _validate_resolved_config(config: dict[str, Any]) -> None:
    if config.get("device") != "mps":
        raise ValueError("every resolved positioning config must use MPS")
    if config["training"].get("algorithm") != "ldp_gradient_far":
        raise ValueError("positioning configs must use ldp_gradient_far")
    data = config["data"]
    model = config["model"]
    dataset = str(data["dataset"])
    if dataset not in TRAINSET_SIZES:
        raise ValueError(f"unsupported positioning dataset {dataset!r}")
    if str(data.get("model")) != str(model.get("architecture")):
        raise ValueError("data.model and model.architecture must match")
    if str(model["architecture"]) not in {"lenet5_tanh", "lenet5_relu"}:
        raise ValueError("positioning uses only explicit LeNet-5 activation aliases")
    if data.get("partition") != "client_dirichlet_balanced":
        raise ValueError("fixed-cardinality privacy requires balanced client partitions")

    clients = config["clients"]
    n = int(clients["num_clients"])
    if n not in {10, 25}:
        raise ValueError("the positioning matrix must use n in {10,25}")
    if float(clients.get("sample_fraction", 0.0)) != 1.0:
        raise ValueError("paired positioning requires full client participation")
    if int(clients.get("min_clients", -1)) != n:
        raise ValueError("min_clients must equal num_clients")
    if float(clients.get("dropout_rate", -1.0)) != 0.0:
        raise ValueError("client deaths/dropout are excluded from positioning")
    if sum(int(item["count"]) for item in clients.get("fleet", [])) != n:
        raise ValueError("fleet counts must sum to num_clients")

    algo = config["training"]["algo_config"]
    public_n = int(algo["privacy_public_dataset_size"])
    expected_public_n = TRAINSET_SIZES[dataset] // n
    if public_n != expected_public_n:
        raise ValueError(
            f"public local size must be {expected_public_n} for {dataset}, n={n}"
        )
    q = float(algo["privacy_sampling_rate_override"])
    fixed_batch = int(algo["fixed_batch_size"])
    if not math.isclose(fixed_batch, q * public_n, abs_tol=1e-12):
        raise ValueError("fixed batch must equal q times the public local size")
    if int(algo.get("batch_size", fixed_batch)) != fixed_batch:
        raise ValueError("batch_size and fixed_batch_size must agree")
    if int(algo.get("positioning_num_clients", -1)) != n:
        raise ValueError("positioning metadata must record n")
    if int(algo.get("positioning_global_train_size", -1)) != TRAINSET_SIZES[dataset]:
        raise ValueError("positioning metadata must record the global train size")
    if int(algo.get("positioning_public_local_dataset_size", -1)) != public_n:
        raise ValueError("positioning metadata must record public N_i")
    if int(algo.get("positioning_fixed_batch_size", -1)) != fixed_batch:
        raise ValueError("positioning metadata must record fixed batch B_i")
    if not math.isclose(
        float(algo.get("positioning_fixed_batch_fraction", -1.0)),
        q,
        abs_tol=1e-12,
    ):
        raise ValueError("positioning metadata must record q=B_i/N_i")
    if algo.get("sampling_scheme") != "fixed_without_replacement":
        raise ValueError("positioning requires fixed sampling without replacement")
    if algo.get("privacy_adjacency") != "replace_one":
        raise ValueError("positioning requires replace-one sample adjacency")
    if int(algo.get("fixed_steps_per_round", -1)) != 1:
        raise ValueError("positioning requires one private-gradient step per round")
    if int(algo.get("local_epochs", -1)) != 1:
        raise ValueError("local_epochs must remain one in the private-gradient lane")
    if int(algo.get("expected_num_clients", -1)) != n:
        raise ValueError("expected_num_clients must match num_clients")
    public_noise_scales = algo.get("privacy_noise_multiplier_scale_by_client")
    if public_noise_scales is not None:
        if not isinstance(public_noise_scales, list) or len(public_noise_scales) != n:
            raise ValueError("public client noise scales must contain one value per client")
        scales = [float(value) for value in public_noise_scales]
        if min(scales) < 1.0:
            raise ValueError(
                "target-epsilon positioning requires every public noise scale >= 1"
            )
        if min(scales) != 1.0 or max(scales) <= 1.0:
            raise ValueError(
                "heteroscedastic screens require a scale-1 privacy anchor and "
                "at least one more-private client"
            )
    if not 0.0 < float(algo.get("clip_norm", 0.0)):
        raise ValueError("local clipping C must be positive")
    if not -5.0 <= float(algo.get("far_alpha", 0.0)) <= 5.0:
        raise ValueError("the registered alpha screen is restricted to [-5,5]")
    if str(algo.get("robust_reference")) not in ALLOWED_REFERENCES:
        raise ValueError("unexpected robust reference in positioning matrix")
    attack = algo.get("attack", {})
    if str(attack.get("name", "none")) not in ALLOWED_ATTACKS:
        raise ValueError("unexpected attack in positioning matrix")
    if attack.get("enabled", False):
        expected_f = 2 if n == 10 else 5
        if int(attack.get("num_byzantine", -1)) != expected_f:
            raise ValueError("Byzantine fraction must equal 20%")
        if attack.get("client_ids") != list(range(expected_f)):
            raise ValueError("Byzantine ids must be explicit and deterministic")
    if bool(algo.get("enable_dp", True)):
        epsilon = algo.get("target_epsilon")
        if epsilon is None or float(epsilon) not in {2.0, 4.0, 8.0}:
            raise ValueError("DP arms must target epsilon in {2,4,8}")
    else:
        if algo.get("target_epsilon") is not None:
            raise ValueError("no-DP arms must clear target_epsilon")
        if float(algo.get("noise_multiplier", 0.0)) != 0.0:
            raise ValueError("no-DP arms must use zero Gaussian noise")


def validate_campaign(campaign: Campaign) -> None:
    randomness = campaign.matrix.get("randomness", {})
    development = {int(seed) for seed in randomness.get("development_seeds", [])}
    confirmation = {int(seed) for seed in randomness.get("confirmation_seeds", [])}
    calibration = {int(seed) for seed in randomness.get("calibration_seeds", [])}
    if not development or not confirmation or development & confirmation:
        raise ValueError("development and confirmation seeds must be non-empty/disjoint")
    if calibration and (
        calibration & development or calibration & confirmation
    ):
        raise ValueError(
            "calibration, development and confirmation seeds must be disjoint"
        )
    task_cap = 300 if int(campaign.matrix.get("schema_version", 1)) == 1 else 400
    if len(campaign.tasks) > task_cap:
        raise ValueError(
            f"positioning campaign unexpectedly exceeds {task_cap} tasks"
        )
    for task in campaign.tasks:
        if task.phase_role == "confirmation" and task.seed not in confirmation:
            raise ValueError("confirmation task uses a development seed")
        if task.phase_role == "calibration" and task.seed not in calibration:
            raise ValueError("calibration task uses an unregistered calibration seed")
        if task.phase_role not in {"confirmation", "calibration"} and task.seed not in development:
            raise ValueError("development task uses a confirmation seed")
        _validate_resolved_config(resolved_config(campaign, task))

    # Locked scientific coverage: future edits cannot silently drop a requested
    # factor or inflate the screen into an uncontrolled full factorial.
    counts = {
        phase["id"]: sum(task.phase_id == phase["id"] for task in campaign.tasks)
        for phase in campaign.phases
    }
    if int(campaign.matrix.get("schema_version", 1)) == 1:
        expected_counts = {
            "a_activation_screen": 8,
            "b_local_clip_screen": 6,
            "c_alpha_reference_screen": 56,
            "d_privacy_screen": 24,
            "d2_noise_heterogeneity_screen": 16,
            "e_byzantine_screen": 48,
            "e2_far_stealth_attack_screen": 16,
            "f_confirmation": 120,
        }
    else:
        registered = campaign.matrix.get("expected_task_counts")
        if not isinstance(registered, Mapping) or not registered:
            raise ValueError("schema-v2 matrices must register expected_task_counts")
        expected_counts = {str(key): int(value) for key, value in registered.items()}
        inherited = campaign.matrix.get("inherited_lock")
        if not isinstance(inherited, Mapping):
            raise ValueError(
                "schema-v2 matrices must document their inherited v1 lock"
            )
        for phase in campaign.phases:
            if not phase.get("requires_approval_for_dependents", False):
                continue
            criteria = phase.get("gate_criteria")
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(
                    f"phase {phase['id']!r} needs preregistered numeric gate_criteria"
                )
            identifiers = [str(item.get("id", "")) for item in criteria]
            if any(not identifier for identifier in identifiers) or len(
                identifiers
            ) != len(set(identifiers)):
                raise ValueError(
                    f"phase {phase['id']!r} has missing/duplicate gate criterion ids"
                )
            for criterion in criteria:
                operator = str(criterion.get("op", ""))
                threshold = criterion.get("threshold")
                if operator not in GATE_OPERATORS:
                    raise ValueError(
                        f"phase {phase['id']!r} uses unsupported gate operator "
                        f"{operator!r}"
                    )
                if isinstance(threshold, bool) or not isinstance(
                    threshold, (int, float)
                ) or not math.isfinite(float(threshold)):
                    raise ValueError(
                        f"phase {phase['id']!r} gate threshold must be finite numeric"
                    )
    if counts != expected_counts:
        raise ValueError(f"unexpected task counts: {counts}; expected {expected_counts}")


def validate_requested_device(device: str) -> None:
    if str(device).lower() != "mps":
        raise ValueError("this campaign is MPS-only; CPU/CUDA execution is refused")


def require_working_mps() -> None:
    validate_requested_device("mps")
    if not torch.backends.mps.is_built():
        raise RuntimeError("PyTorch was not built with MPS support")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process. Run from the native macOS "
            "environment; no CPU fallback is permitted."
        )
    try:
        probe = torch.tensor([1.0], device="mps")
        _ = (probe + 1.0).cpu()
        torch.mps.synchronize()
    except Exception as exc:  # pragma: no cover - host-specific backend failure
        raise RuntimeError("MPS probe failed; refusing silent CPU fallback") from exc


def task_output_dir(campaign: Campaign, task: PositioningTask) -> Path:
    return campaign.output_root / task.phase_id / task.run_id


def completed_metrics(output_dir: Path) -> Path | None:
    for path in sorted(output_dir.glob("**/metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expected = int(payload.get("summary", {}).get("num_rounds", -1))
        if expected > 0 and len(payload.get("rounds", [])) == expected:
            return path
    return None


def tasks_for_phase(campaign: Campaign, phase_id: str) -> tuple[PositioningTask, ...]:
    _phase_definition(campaign, phase_id)
    return tuple(task for task in campaign.tasks if task.phase_id == phase_id)


def phase_is_complete(campaign: Campaign, phase_id: str) -> bool:
    return all(
        completed_metrics(task_output_dir(campaign, task)) is not None
        for task in tasks_for_phase(campaign, phase_id)
    )


def gate_path(campaign: Campaign, phase_id: str) -> Path:
    return campaign.output_root / "_gates" / f"{phase_id}.json"


def gate_decision(campaign: Campaign, phase_id: str) -> str | None:
    path = gate_path(campaign, phase_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    if int(campaign.matrix.get("schema_version", 1)) >= 2:
        if payload.get("campaign_config_hash") != campaign_config_hash(campaign):
            return "stale"
        if payload.get("phase_config_hash") != phase_config_hash(campaign, phase_id):
            return "stale"
    return str(payload.get("decision"))


def assert_dependencies_promoted(campaign: Campaign, phase_id: str) -> None:
    phase = _phase_definition(campaign, phase_id)
    for dependency in phase.get("depends_on", []):
        dependency = str(dependency)
        if not phase_is_complete(campaign, dependency):
            raise RuntimeError(
                f"dependency {dependency!r} is incomplete; {phase_id!r} is gated"
            )
        dependency_definition = _phase_definition(campaign, dependency)
        if dependency_definition.get("requires_approval_for_dependents", False):
            decision = gate_decision(campaign, dependency)
            if decision != "promote":
                raise RuntimeError(
                    f"dependency {dependency!r} needs a recorded promote decision "
                    f"before {phase_id!r} can run (current={decision!r})"
                )


def _evaluate_gate_criteria(
    criteria: Iterable[Mapping[str, Any]], evidence: Mapping[str, Any]
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for criterion in criteria:
        identifier = str(criterion["id"])
        operator = str(criterion["op"])
        threshold = float(criterion["threshold"])
        if identifier not in evidence:
            raise ValueError(f"gate evidence is missing {identifier!r}")
        value = evidence[identifier]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"gate evidence {identifier!r} must be numeric")
        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError(f"gate evidence {identifier!r} must be finite")
        if operator == "<=":
            passed = observed <= threshold
        elif operator == ">=":
            passed = observed >= threshold
        elif operator == "==":
            passed = math.isclose(observed, threshold, abs_tol=1e-12)
        else:  # validated when the campaign is loaded
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
    return evaluations


def approve_phase(
    campaign: Campaign,
    phase_id: str,
    decision: str,
    note: str | None,
    evidence: Mapping[str, Any] | None = None,
) -> Path:
    phase = _phase_definition(campaign, phase_id)
    if not phase_is_complete(campaign, phase_id):
        raise RuntimeError(f"cannot decide gate for incomplete phase {phase_id!r}")
    if decision not in {"promote", "stop"}:
        raise ValueError("gate decision must be promote or stop")
    path = gate_path(campaign, phase_id)
    if path.exists():
        raise RuntimeError(
            f"gate {path} already exists; gate records are immutable and cannot "
            "be overwritten"
        )
    criteria = phase.get("gate_criteria", [])
    evaluations: list[dict[str, Any]] = []
    if int(campaign.matrix.get("schema_version", 1)) >= 2:
        if decision == "promote" and evidence is None:
            raise ValueError("schema-v2 promote decisions require numeric evidence")
        if evidence is not None:
            evaluations = _evaluate_gate_criteria(criteria, evidence)
        if decision == "promote" and not all(row["passed"] for row in evaluations):
            failed = [row["id"] for row in evaluations if not row["passed"]]
            raise RuntimeError(
                f"cannot promote {phase_id!r}; numeric gate criteria failed: {failed}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign_id": campaign.matrix["campaign_id"],
        "schema_version": int(campaign.matrix.get("schema_version", 1)),
        "phase": phase_id,
        "decision": decision,
        "note": note or "",
        "all_registered_tasks_complete": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_config_hash": campaign_config_hash(campaign),
        "phase_config_hash": phase_config_hash(campaign, phase_id),
        "preregistered_criteria": copy.deepcopy(criteria),
        "evidence": dict(evidence or {}),
        "criterion_evaluations": evaluations,
        "immutable_record": True,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _selected_tasks(
    campaign: Campaign, phase_id: str | None, job_index: int | None
) -> tuple[PositioningTask, ...]:
    pool = tasks_for_phase(campaign, phase_id) if phase_id else campaign.tasks
    if job_index is None:
        return tuple(pool)
    if not 0 <= job_index < len(pool):
        scope = f"phase {phase_id}" if phase_id else "campaign"
        raise IndexError(f"job-index must lie in [0,{len(pool) - 1}] for {scope}")
    return (pool[job_index],)


def _format_task(task: PositioningTask, config: dict[str, Any]) -> str:
    algo = config["training"]["algo_config"]
    attack = algo.get("attack", {})
    epsilon = algo.get("target_epsilon") if algo.get("enable_dp") else "inf"
    return (
        f"g={task.global_index:03d} p={task.phase_index:03d} "
        f"{task.phase_id}/{task.run_id}: "
        f"data={config['data']['dataset']} model={config['model']['architecture']} "
        f"n={config['clients']['num_clients']} C={algo['clip_norm']:g} "
        f"eps={epsilon} alpha={algo['far_alpha']:g} "
        f"F={algo['robust_reference']} attack={attack.get('name', 'none')}"
    )


def list_campaign(campaign: Campaign, selected: Iterable[PositioningTask]) -> None:
    selected = tuple(selected)
    print(f"campaign={campaign.matrix['campaign_id']}")
    print(f"matrix={campaign.matrix_path}")
    print(f"required_device=mps tasks={len(campaign.tasks)} selected={len(selected)}")
    for phase in campaign.phases:
        phase_id = str(phase["id"])
        phase_tasks = tasks_for_phase(campaign, phase_id)
        complete = sum(
            completed_metrics(task_output_dir(campaign, task)) is not None
            for task in phase_tasks
        )
        print(
            f"phase={phase_id} role={phase.get('role')} "
            f"complete={complete}/{len(phase_tasks)} gate={gate_decision(campaign, phase_id)}"
        )
    for task in selected:
        print(_format_task(task, resolved_config(campaign, task)))


def run_task(campaign: Campaign, task: PositioningTask, resume: bool) -> None:
    assert_dependencies_promoted(campaign, task.phase_id)
    output_dir = task_output_dir(campaign, task)
    if resume and completed_metrics(output_dir) is not None:
        print(f"SKIP completed: {task.phase_id}/{task.run_id}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolved_config(campaign, task, output_dir)
    _validate_resolved_config(config)
    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
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
    print(f"RUN {_format_task(task, config)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--phase")
    parser.add_argument(
        "--job-index",
        type=int,
        help="phase-local index when --phase is set; otherwise global index",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--approve-phase")
    parser.add_argument("--decision", choices=("promote", "stop"))
    parser.add_argument("--note")
    parser.add_argument(
        "--evidence-json",
        type=Path,
        help=(
            "JSON object containing the preregistered numeric gate evidence; "
            "required for schema-v2 promote decisions"
        ),
    )
    args = parser.parse_args()

    validate_requested_device(args.device)
    campaign = load_campaign(args.matrix)
    if args.approve_phase:
        if args.run or args.job_index is not None or args.phase is not None:
            parser.error("--approve-phase cannot be combined with run selection")
        if args.decision is None:
            parser.error("--approve-phase requires --decision")
        evidence = None
        if args.evidence_json is not None:
            try:
                evidence = json.loads(args.evidence_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"cannot read --evidence-json: {exc}")
            if not isinstance(evidence, dict):
                parser.error("--evidence-json must contain a JSON object")
        path = approve_phase(
            campaign,
            args.approve_phase,
            args.decision,
            args.note,
            evidence=evidence,
        )
        print(f"gate recorded: {path}")
        return
    if args.decision is not None:
        parser.error("--decision requires --approve-phase")
    if args.evidence_json is not None:
        parser.error("--evidence-json requires --approve-phase")

    selected = _selected_tasks(campaign, args.phase, args.job_index)
    if args.list or not args.run:
        list_campaign(campaign, selected)
        return
    if args.phase is None and args.job_index is None:
        parser.error("refusing to run all phases at once; select --phase or --job-index")
    require_working_mps()
    for task in selected:
        run_task(campaign, task, resume=args.resume)
    if args.phase:
        print(f"phase_complete={phase_is_complete(campaign, args.phase)}")


if __name__ == "__main__":
    main()
