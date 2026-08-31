#!/usr/bin/env python3
"""Validate, expand and execute the frozen SC-FAR-DP paper-1 matrices.

The runner deliberately uses the standard in-process ``run_experiment.py``
entry point.  Every matrix cell becomes a standalone resolved YAML, output
directory and immutable sidecar manifest.  The main protocol is full-update,
full-participation user-level central DP with replace-one adjacency and no
sampling amplification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "scpfar" / "paper1" / "s1_reference_tradeoff.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "scfar_paper1"
SCHEMA_VERSION = 1
KNOWN_DATASETS = {"fashionmnist", "cifar10"}
KNOWN_MODELS = {"lenet5", "alexnet"}
KNOWN_PARTITIONS = {"client_dirichlet_balanced"}
KNOWN_ATTACKS = {"none", "bf", "ipm", "alie", "minmax", "minsum"}
KNOWN_AGGREGATION_RULES = {
    "controlled_tilt",
    "uniform",
    "reference",
    "far_raw_distance",
}


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alpha_max(n: int, kappa_w: float) -> float:
    if n < 2 or not 1.0 <= kappa_w < n:
        raise ValueError("Need n>=2 and 1<=kappa_w<n")
    return math.log(kappa_w * (n - 1) / (n - kappa_w))


@dataclass(frozen=True)
class Task:
    index: int
    matrix_id: str
    experiment_id: str
    scenario_id: str
    method_id: str
    reference_id: str
    anchor_id: str
    threat_id: str
    tilt_id: str
    privacy_id: str
    tau_over_c: float
    partition_seed: int
    training_seed: int
    config: dict[str, Any]
    output_dir: Path

    @property
    def task_id(self) -> str:
        return "__".join(
            (
                self.experiment_id,
                self.scenario_id,
                self.method_id,
                self.reference_id,
                self.anchor_id,
                self.threat_id,
                self.tilt_id,
                self.privacy_id,
                f"tau_{self.tau_over_c:g}",
                f"pseed_{self.partition_seed}",
                f"tseed_{self.training_seed}",
            )
        )

    @property
    def resolved_config_path(self) -> Path:
        return self.output_dir / "resolved_config.yaml"


def load_matrix(path: Path) -> dict[str, Any]:
    path = path.resolve()
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError(f"Matrix must contain a mapping: {path}")
    common_name = matrix.get("common")
    if not common_name:
        raise ValueError(f"Matrix does not declare common: {path}")
    common_path = (path.parent / str(common_name)).resolve()
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    if not isinstance(common, dict):
        raise ValueError(f"Common protocol must contain a mapping: {common_path}")
    return {"matrix": matrix, "common": common, "matrix_path": path, "common_path": common_path}


def _profile(common: dict[str, Any], family: str, profile_id: str) -> dict[str, Any]:
    profiles = common.get(family, {})
    if profile_id not in profiles:
        raise KeyError(f"Unknown {family} profile {profile_id!r}")
    value = profiles[profile_id]
    if not isinstance(value, dict):
        raise TypeError(f"{family}.{profile_id} must be a mapping")
    return copy.deepcopy(value)


def _outlier_ids(common: dict[str, Any], scenario_id: str, partition_seed: int) -> list[int]:
    mapping = common.get("honest_outliers", {}).get(
        "by_scenario_and_partition_seed", {}
    )
    scenario = mapping.get(scenario_id, {})
    values = scenario.get(partition_seed, scenario.get(str(partition_seed), []))
    return [int(value) for value in values]


def _resolved_config(
    common: dict[str, Any],
    *,
    scenario_id: str,
    method_id: str,
    reference_id: str,
    anchor_id: str,
    threat_id: str,
    tilt_id: str,
    privacy_id: str,
    tau_over_c: float,
    partition_seed: int,
    training_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(common["base_config"])
    scenario = _profile(common, "scenarios", scenario_id)
    method = _profile(common, "methods", method_id)
    reference = _profile(common, "references", reference_id)
    anchor = _profile(common, "anchors", anchor_id)
    threat = _profile(common, "threats", threat_id)
    tilt = _profile(common, "tilts", tilt_id)
    privacy = _profile(common, "privacy_profiles", privacy_id)

    config = _deep_merge(config, scenario)
    config["training"]["algorithm"] = str(method["algorithm"])
    algo = _deep_merge(config["training"]["algo_config"], method.get("algo_config"))
    algo = _deep_merge(algo, reference.get("algo_config"))
    algo = _deep_merge(algo, anchor.get("algo_config"))
    algo = _deep_merge(algo, {"num_byzantine": threat["num_byzantine"], "attack": threat["attack"]})

    if str(method.get("tilt_policy", "grid")) != "fixed":
        n = int(config["clients"]["num_clients"])
        kappa_w = float(tilt["kappa_w"])
        ratio = float(tilt["alpha_fraction"])
        algo["kappa_w"] = kappa_w
        algo["far_alpha"] = ratio * alpha_max(n, kappa_w)
        algo["alpha_fraction_of_max"] = ratio
    algo = _deep_merge(algo, privacy.get("algo_config"))

    clip_norm = float(algo["user_clip_norm"])
    algo["reference_clip_tau"] = float(tau_over_c) * clip_norm
    algo["privacy_num_rounds"] = int(config["training"]["num_rounds"])
    algo["honest_outlier_client_ids"] = _outlier_ids(
        common, scenario_id, partition_seed
    )

    config["training"]["algo_config"] = algo
    config["seed"] = int(training_seed)
    config["data"]["partition_seed"] = int(partition_seed)
    config["output_dir"] = str(output_dir)
    config["reproduction"] = {
        "protocol_id": str(common["protocol_id"]),
        "execution_scope": "full_protocol",
        "matrix_axes": {
            "scenario": scenario_id,
            "method": method_id,
            "reference": reference_id,
            "anchor": anchor_id,
            "threat": threat_id,
            "tilt": tilt_id,
            "privacy": privacy_id,
            "tau_over_c": float(tau_over_c),
            "partition_seed": int(partition_seed),
            "training_seed": int(training_seed),
        },
        "honest_outlier_rule": common["honest_outliers"]["rule"],
    }
    return config


def expand_tasks(
    document: dict[str, Any],
    *,
    output_root: Path,
    experiment_ids: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    method_ids: set[str] | None = None,
    threat_ids: set[str] | None = None,
    partition_seeds: set[int] | None = None,
    training_seeds: set[int] | None = None,
    pilot_rounds: int | None = None,
) -> list[Task]:
    matrix, common = document["matrix"], document["common"]
    matrix_id = str(matrix["matrix_id"])
    tasks: list[Task] = []
    for experiment in matrix.get("experiments", []):
        experiment_id = str(experiment["id"])
        if experiment_ids and experiment_id not in experiment_ids:
            continue
        axes = (
            experiment["scenario_ids"],
            experiment["method_ids"],
            experiment["reference_ids"],
            experiment["anchor_ids"],
            experiment["threat_ids"],
            experiment["tilt_ids"],
            experiment["privacy_ids"],
            experiment["tau_over_c"],
            common["partition_seeds"],
            common["training_seeds"],
        )
        for values in itertools.product(*axes):
            (
                scenario_id,
                method_id,
                reference_id,
                anchor_id,
                threat_id,
                tilt_id,
                privacy_id,
                tau_over_c,
                partition_seed,
                training_seed,
            ) = values
            if scenario_ids and scenario_id not in scenario_ids:
                continue
            if method_ids and method_id not in method_ids:
                continue
            if threat_ids and threat_id not in threat_ids:
                continue
            if partition_seeds and int(partition_seed) not in partition_seeds:
                continue
            if training_seeds and int(training_seed) not in training_seeds:
                continue
            parts = (
                matrix_id,
                experiment_id,
                scenario_id,
                method_id,
                reference_id,
                anchor_id,
                threat_id,
                tilt_id,
                privacy_id,
                f"tau_{float(tau_over_c):g}",
                f"partition_seed{partition_seed}",
                f"training_seed{training_seed}",
            )
            output_dir = output_root.joinpath(*(_slug(part) for part in parts))
            config = _resolved_config(
                common,
                scenario_id=str(scenario_id),
                method_id=str(method_id),
                reference_id=str(reference_id),
                anchor_id=str(anchor_id),
                threat_id=str(threat_id),
                tilt_id=str(tilt_id),
                privacy_id=str(privacy_id),
                tau_over_c=float(tau_over_c),
                partition_seed=int(partition_seed),
                training_seed=int(training_seed),
                output_dir=output_dir,
            )
            if pilot_rounds is not None:
                config["training"]["num_rounds"] = int(pilot_rounds)
                config["training"]["algo_config"]["privacy_num_rounds"] = int(pilot_rounds)
                config["reproduction"]["execution_scope"] = "pilot_not_full_protocol"
                output_dir = output_root.joinpath(
                    "pilots", f"rounds_{pilot_rounds}", *(_slug(part) for part in parts)
                )
                config["output_dir"] = str(output_dir)
            tasks.append(
                Task(
                    index=-1,
                    matrix_id=matrix_id,
                    experiment_id=str(experiment_id),
                    scenario_id=str(scenario_id),
                    method_id=str(method_id),
                    reference_id=str(reference_id),
                    anchor_id=str(anchor_id),
                    threat_id=str(threat_id),
                    tilt_id=str(tilt_id),
                    privacy_id=str(privacy_id),
                    tau_over_c=float(tau_over_c),
                    partition_seed=int(partition_seed),
                    training_seed=int(training_seed),
                    config=config,
                    output_dir=output_dir,
                )
            )
    return [replace(task, index=index) for index, task in enumerate(tasks)]


def _task_issues(task: Task) -> list[str]:
    cfg = task.config
    training = cfg["training"]
    algo = training["algo_config"]
    clients = cfg["clients"]
    issues: list[str] = []
    n = int(clients["num_clients"])
    if float(clients.get("sample_fraction", 0.0)) != 1.0:
        issues.append("sample_fraction must equal 1")
    if int(clients.get("min_clients", 0)) != n:
        issues.append("min_clients must equal num_clients")
    if float(clients.get("dropout_rate", 0.0)) != 0.0:
        issues.append("dropout_rate must equal 0")
    forbidden = {"num_layer_groups", "layer_selection", "rounds_per_layer"}
    present = sorted(forbidden.intersection(algo))
    if present:
        issues.append(f"partial-training keys are forbidden: {present}")
    if training["algorithm"] in {"scfar_dp", "sc_partial_far_dp"}:
        if training["algorithm"] != "scfar_dp":
            issues.append("paper 1 SC-FAR arms must use scfar_dp")
        if float(algo.get("user_clip_norm", 0.0)) <= 0:
            issues.append("user_clip_norm must be positive")
        if str(algo.get("alpha_bound_policy")) != "error":
            issues.append("alpha_bound_policy must be error in frozen runs")
        if str(algo.get("robust_reference")) not in {
            "centered_clipping",
            "regularized_huber",
        }:
            issues.append("SC-FAR reference must be F_CC or declared Huber ablation")
        aggregation_rule = str(algo.get("scfar_aggregation_rule", ""))
        if aggregation_rule not in KNOWN_AGGREGATION_RULES:
            issues.append(f"unsupported SC-FAR aggregation rule {aggregation_rule!r}")
        anchor_mode = str(algo.get("anchor_mode", ""))
        if anchor_mode not in {"fixed_zero", "ema_release", "previous_release"}:
            issues.append(f"unsupported anchor_mode {anchor_mode!r}")
        if int(algo.get("anchor_update_every", 0)) < 1:
            issues.append("anchor_update_every must be at least one")
        if not 0.0 <= float(algo.get("anchor_update_rate", -1.0)) <= 1.0:
            issues.append("anchor_update_rate must lie in [0,1]")
        if aggregation_rule == "controlled_tilt":
            requested_alpha = float(algo.get("far_alpha", -1.0))
            kappa_w = float(algo.get("kappa_w", 0.0))
            if requested_alpha < 0.0 or requested_alpha > alpha_max(n, kappa_w) + 1e-12:
                issues.append("controlled tilt lies outside the certified alpha region")
        if bool(algo.get("enable_central_dp")):
            if algo.get("target_epsilon") is None:
                issues.append("finite-DP arm requires target_epsilon")
            if int(algo.get("privacy_num_rounds", -1)) != int(training["num_rounds"]):
                issues.append("privacy_num_rounds must match training rounds")
        elif float(algo.get("central_noise_multiplier", 0.0)) != 0.0:
            issues.append("non-private arm must have zero central noise")
    attack = algo.get("attack", {})
    if int(attack.get("num_byzantine", 0)) != int(algo.get("num_byzantine", 0)):
        issues.append("attack and algorithm Byzantine counts differ")
    malicious = {int(value) for value in attack.get("client_ids", [])}
    outliers = {int(value) for value in algo.get("honest_outlier_client_ids", [])}
    if malicious.intersection(outliers):
        issues.append("honest-outlier and Byzantine IDs overlap")
    if any(value < 0 or value >= n for value in malicious.union(outliers)):
        issues.append("client IDs must lie in [0,n)")
    declared_byzantines = int(attack.get("num_byzantine", 0))
    if len(malicious) != declared_byzantines:
        issues.append("explicit Byzantine ID count differs from num_byzantine")
    if float(attack.get("scale", 1.0)) < 0.0:
        issues.append("attack intensity must be non-negative")
    return issues


def validate_matrix(document: dict[str, Any]) -> list[str]:
    matrix, common = document["matrix"], document["common"]
    errors: list[str] = []
    if int(matrix.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append("unsupported matrix schema_version")
    if int(common.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append("unsupported common schema_version")
    experiments = matrix.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        errors.append("matrix must define non-empty experiments")
        return errors
    for scenario_id, scenario in common.get("scenarios", {}).items():
        dataset = str(scenario.get("data", {}).get("dataset", ""))
        model = str(scenario.get("model", {}).get("architecture", ""))
        if dataset not in KNOWN_DATASETS:
            errors.append(
                f"{scenario_id}: unsupported paper-1 dataset {dataset!r}; "
                f"expected one of {sorted(KNOWN_DATASETS)}"
            )
        if model not in KNOWN_MODELS:
            errors.append(
                f"{scenario_id}: unsupported paper-1 model {model!r}; "
                f"expected one of {sorted(KNOWN_MODELS)}"
            )
    partition = str(common.get("base_config", {}).get("data", {}).get("partition", ""))
    if partition not in KNOWN_PARTITIONS:
        errors.append(
            f"paper-1 partition must be one of {sorted(KNOWN_PARTITIONS)}, got {partition!r}"
        )
    for threat_id, threat in common.get("threats", {}).items():
        attack_name = str(threat.get("attack", {}).get("name", "none")).lower()
        if attack_name not in KNOWN_ATTACKS:
            errors.append(f"{threat_id}: unsupported attack {attack_name!r}")
    for method_id, method in common.get("methods", {}).items():
        if str(method.get("algorithm")) == "scfar_dp":
            rule = str(
                method.get("algo_config", {}).get(
                    "scfar_aggregation_rule",
                    common["base_config"]["training"]["algo_config"].get(
                        "scfar_aggregation_rule", "controlled_tilt"
                    ),
                )
            )
            if rule not in KNOWN_AGGREGATION_RULES:
                errors.append(f"{method_id}: unsupported SC-FAR aggregation rule {rule!r}")
    for experiment in experiments:
        experiment_id = str(experiment.get("id", "missing"))
        for key in (
            "scenario_ids",
            "method_ids",
            "reference_ids",
            "anchor_ids",
            "threat_ids",
            "tilt_ids",
            "privacy_ids",
            "tau_over_c",
        ):
            if not isinstance(experiment.get(key), list) or not experiment[key]:
                errors.append(f"{experiment_id}: {key} must be a non-empty list")
        expected = int(experiment.get("expected_tasks", -1))
        actual = math.prod(
            len(experiment[key])
            for key in (
                "scenario_ids",
                "method_ids",
                "reference_ids",
                "anchor_ids",
                "threat_ids",
                "tilt_ids",
                "privacy_ids",
                "tau_over_c",
            )
        ) * len(common["partition_seeds"]) * len(common["training_seeds"])
        if expected != actual:
            errors.append(f"{experiment_id}: expected_tasks={expected}, expanded={actual}")
    if errors:
        return errors
    tasks = expand_tasks(document, output_root=Path("/tmp/scfar_matrix_validation"))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("expanded matrix contains duplicate task IDs")
    for task in tasks:
        for issue in _task_issues(task):
            errors.append(f"{task.task_id}: {issue}")
    return errors


def task_command(task: Task, *, python_bin: str, device: str, data_root: Path) -> list[str]:
    return [
        python_bin,
        str(ROOT / "run_experiment.py"),
        "--config",
        str(task.resolved_config_path),
        "--algo",
        str(task.config["training"]["algorithm"]),
        "--device",
        device,
        "--seed",
        str(task.training_seed),
        "--data-root",
        str(data_root),
        "--output",
        str(task.output_dir),
    ]


def write_config(task: Task, *, device: str, data_root: Path) -> Path:
    config = copy.deepcopy(task.config)
    config["device"] = device
    config["data"]["data_root"] = str(data_root)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    task.resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return task.resolved_config_path


def _metrics_path(task: Task) -> Path | None:
    paths = sorted(task.output_dir.glob("**/metrics.json"))
    return paths[-1] if paths else None


def is_complete(task: Task) -> bool:
    metrics = _metrics_path(task)
    if metrics is None:
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    rounds = payload.get("rounds", [])
    return len(rounds) == int(task.config["training"]["num_rounds"])


def verify_completed_run(task: Task) -> list[str]:
    metrics = _metrics_path(task)
    if metrics is None:
        return ["metrics.json is missing"]
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    rows = payload.get("rounds", [])
    n = int(task.config["clients"]["num_clients"])
    issues: list[str] = []
    if len(rows) != int(task.config["training"]["num_rounds"]):
        issues.append("run did not produce every declared round")
    dead_rounds = [
        int(row.get("round_num", -1))
        for row in rows
        if int(row.get("num_alive_clients", n)) != n
    ]
    if dead_rounds:
        issues.append(f"client death or missing client at rounds {dead_rounds[:10]}")
    if any(float(row.get("participation_rate", 1.0)) != 1.0 for row in rows):
        issues.append("full participation was not maintained")
    return issues


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or None


def write_manifest(
    task: Task,
    *,
    document: dict[str, Any],
    command: list[str],
    status: str,
    started: float,
    finished: float | None = None,
    compliance_issues: list[str] | None = None,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "task_index": task.index,
        "task_id": task.task_id,
        "matrix": str(document["matrix_path"]),
        "matrix_sha256": _sha256(document["matrix_path"]),
        "common": str(document["common_path"]),
        "common_sha256": _sha256(document["common_path"]),
        "config": task.config,
        "command": command,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": started,
        "finished_unix": finished,
        "compliance_issues": compliance_issues or [],
    }
    path = task.output_dir / "scfar_paper1_task_manifest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _parse_set(value: str | None, cast: type) -> set[Any] | None:
    if not value:
        return None
    return {cast(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--experiment")
    parser.add_argument("--scenario")
    parser.add_argument("--method")
    parser.add_argument("--threat")
    parser.add_argument("--partition-seed")
    parser.add_argument("--training-seed")
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--pilot-rounds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_matrix(args.matrix)
    errors = validate_matrix(document)
    if errors:
        raise SystemExit("Invalid SC-FAR-DP matrix:\n- " + "\n- ".join(errors))
    if args.pilot_rounds is not None and args.pilot_rounds <= 0:
        raise ValueError("pilot-rounds must be positive")
    tasks = expand_tasks(
        document,
        output_root=args.output_root.resolve(),
        experiment_ids=_parse_set(args.experiment, str),
        scenario_ids=_parse_set(args.scenario, str),
        method_ids=_parse_set(args.method, str),
        threat_ids=_parse_set(args.threat, str),
        partition_seeds=_parse_set(args.partition_seed, int),
        training_seeds=_parse_set(args.training_seed, int),
        pilot_rounds=args.pilot_rounds,
    )
    if not tasks:
        raise SystemExit("No task matches the requested filters")
    if args.validate:
        print(f"VALID: {len(tasks)} unique full-update tasks")
        print(f"matrix={document['matrix']['matrix_id']}")
        print(f"experiments={sorted({task.experiment_id for task in tasks})}")
        print(f"scenarios={sorted({task.scenario_id for task in tasks})}")
        return
    if args.list:
        for task in tasks:
            print(f"{task.index:05d}\t{task.task_id}\t{task.output_dir}")
        return
    if args.job_index is None or not 0 <= args.job_index < len(tasks):
        raise SystemExit(f"job-index must lie in [0,{len(tasks)-1}]")
    task = tasks[args.job_index]
    write_config(task, device=args.device, data_root=args.data_root.resolve())
    command = task_command(
        task,
        python_bin=args.python_bin,
        device=args.device,
        data_root=args.data_root.resolve(),
    )
    if args.dry_run:
        print(json.dumps({"task": task.task_id, "command": command}, indent=2))
        return
    started = time.time()
    if args.resume and is_complete(task):
        status = "complete_reused"
    else:
        write_manifest(
            task,
            document=document,
            command=command,
            status="running",
            started=started,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            write_manifest(
                task,
                document=document,
                command=command,
                status="failed",
                started=started,
                finished=time.time(),
            )
            raise SystemExit(completed.returncode)
        status = "complete"
    compliance = verify_completed_run(task)
    final_status = status if not compliance else "invalid_protocol_run"
    write_manifest(
        task,
        document=document,
        command=command,
        status=final_status,
        started=started,
        finished=time.time(),
        compliance_issues=compliance,
    )
    if compliance:
        raise RuntimeError("Protocol compliance failed: " + "; ".join(compliance))
    print(f"DONE {task.task_id}")


if __name__ == "__main__":
    main()
