#!/usr/bin/env python3
"""Expand and run the internship FAR/FedFDP reproduction matrices.

The matrix files deliberately keep the internship reproduction (faithful
lane) separate from the proposed paper-grade extension.  A finite target
epsilon is treated as a contract, not a label: the launcher refuses a run
unless the registered algorithm advertises target-epsilon calibration in its
default configuration.

Examples
--------
  python3 scripts/run_internship_far_fedfdp.py --validate --lane all
  python3 scripts/run_internship_far_fedfdp.py --list --lane faithful
  python3 scripts/run_internship_far_fedfdp.py --dry-run --job-index 0
  python3 scripts/run_internship_far_fedfdp.py --run --job-index 0 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = (
    ROOT / "configs" / "reproductions" / "internship_far_fedfdp"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "reproductions" / "internship_far_fedfdp"


@dataclass(frozen=True)
class Task:
    index: int
    lane: str
    source_file: Path
    scenario_id: str
    condition_id: str
    attack_id: str
    method_id: str
    algorithm: str
    seed: int
    privacy: dict[str, Any]
    requires: tuple[str, ...]
    resolved_config: dict[str, Any]
    output_dir: Path

    @property
    def task_id(self) -> str:
        return "/".join(
            (
                self.lane,
                self.scenario_id,
                self.condition_id,
                self.attack_id,
                self.method_id,
                f"seed{self.seed}",
            )
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_finite_epsilon(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().lower() in {
        "inf",
        "+inf",
        "infinity",
        "+infinity",
        "none",
    }:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)


def _load_documents(config_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"No matrix YAML found under {config_dir}")
    documents: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        if not isinstance(document, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        documents.append((path, document))
    return documents


def _validate_document(path: Path, document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append(f"{path}: schema_version must be 1")
    if document.get("lane") not in {"faithful", "paper-grade"}:
        errors.append(f"{path}: lane must be faithful or paper-grade")
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append(f"{path}: scenarios must be a non-empty list")
        return errors
    for scenario in scenarios:
        prefix = f"{path}:{scenario.get('id', '?')}"
        if not scenario.get("id"):
            errors.append(f"{prefix}: missing scenario id")
        if not scenario.get("seeds"):
            errors.append(f"{prefix}: seeds must be non-empty")
        base = scenario.get("base_config", {})
        for key in ("data", "model", "training", "clients"):
            if key not in base:
                errors.append(f"{prefix}: base_config.{key} is required")
        conditions = scenario.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            errors.append(f"{prefix}: conditions must be non-empty")
            continue
        for condition in conditions:
            cp = f"{prefix}:{condition.get('id', '?')}"
            if not condition.get("id"):
                errors.append(f"{cp}: missing condition id")
            if not condition.get("methods"):
                errors.append(f"{cp}: methods must be non-empty")
            if not condition.get("attacks"):
                errors.append(f"{cp}: attacks must be non-empty")
            for method in condition.get("methods", []):
                if not method.get("id") or not method.get("algorithm"):
                    errors.append(f"{cp}: each method needs id and algorithm")
    return errors


def _effective_privacy(condition: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    if "privacy_override" in method:
        return copy.deepcopy(method["privacy_override"])
    return copy.deepcopy(condition.get("privacy", {}))


def _resolved_config(
    *,
    document: dict[str, Any],
    source_path: Path,
    scenario: dict[str, Any],
    condition: dict[str, Any],
    attack: dict[str, Any],
    method: dict[str, Any],
    seed: int,
    output_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    lane = str(document["lane"])
    revision = document.get("revision")
    privacy = _effective_privacy(condition, method)
    path_components = [
        lane,
        _slug(str(scenario["id"])),
        _slug(str(condition["id"])),
        _slug(str(attack["id"])),
        _slug(str(method["id"])),
        f"seed{int(seed)}",
    ]
    if revision is not None:
        path_components.insert(1, _slug(str(revision)))
    run_dir = output_root.joinpath(*path_components)
    resolved = copy.deepcopy(scenario["base_config"])
    resolved["seed"] = int(seed)
    resolved["name"] = _slug(
        "__".join(
            (
                str(scenario["id"]),
                str(condition["id"]),
                str(attack["id"]),
                str(method["id"]),
            )
        )
    )
    resolved["description"] = str(document.get("description", ""))
    resolved["output_dir"] = str(run_dir)
    training = resolved.setdefault("training", {})
    training["algorithm"] = str(method["algorithm"])
    algo_config = copy.deepcopy(training.get("algo_config", {}))
    algo_config = _deep_merge(algo_config, condition.get("algo_config"))
    algo_config = _deep_merge(algo_config, method.get("algo_config"))

    attack_config = attack.get("config")
    if attack_config:
        algo_config["attack"] = copy.deepcopy(attack_config)
    else:
        algo_config.pop("attack", None)

    target_epsilon = privacy.get("target_epsilon")
    if _is_finite_epsilon(target_epsilon):
        algo_config["target_epsilon"] = float(target_epsilon)
        algo_config["privacy_num_rounds"] = int(training["num_rounds"])
        if privacy.get("delta") is not None:
            algo_config["delta"] = float(privacy["delta"])
    else:
        # Do not let a condition-level finite-epsilon override leak into a
        # deliberately non-private method such as FAR in the extension lane.
        algo_config.pop("target_epsilon", None)
        algo_config.pop("privacy_calibration", None)
        algo_config.pop("privacy_num_rounds", None)
    training["algo_config"] = algo_config

    resolved["reproduction"] = {
        "lane": lane,
        "revision": str(revision) if revision is not None else None,
        "source_matrix": str(source_path.relative_to(ROOT)),
        "source_claim": str(document.get("source", "unspecified")),
        "scenario": str(scenario["id"]),
        "condition": str(condition["id"]),
        "attack": str(attack["id"]),
        "method": str(method["id"]),
        "privacy": privacy,
        "requires": list(method.get("requires", [])),
        "scientific_note": (
            "Internship-report reproduction"
            if lane == "faithful"
            else "Proposed paper-grade extension; not in internship report"
        ),
    }
    return resolved, run_dir, privacy


def expand_tasks(
    documents: Iterable[tuple[Path, dict[str, Any]]],
    *,
    output_root: Path,
) -> list[Task]:
    tasks: list[Task] = []
    for source_path, document in documents:
        lane = str(document["lane"])
        for scenario in document["scenarios"]:
            for condition in scenario["conditions"]:
                for attack in condition["attacks"]:
                    for method in condition["methods"]:
                        for seed in scenario["seeds"]:
                            resolved, run_dir, privacy = _resolved_config(
                                document=document,
                                source_path=source_path,
                                scenario=scenario,
                                condition=condition,
                                attack=attack,
                                method=method,
                                seed=int(seed),
                                output_root=output_root,
                            )
                            tasks.append(
                                Task(
                                    index=-1,
                                    lane=lane,
                                    source_file=source_path,
                                    scenario_id=str(scenario["id"]),
                                    condition_id=str(condition["id"]),
                                    attack_id=str(attack["id"]),
                                    method_id=str(method["id"]),
                                    algorithm=str(method["algorithm"]),
                                    seed=int(seed),
                                    privacy=privacy,
                                    requires=tuple(method.get("requires", [])),
                                    resolved_config=resolved,
                                    output_dir=run_dir,
                                )
                            )
    tasks.sort(key=lambda task: task.task_id)
    return [
        Task(**{**task.__dict__, "index": index}) for index, task in enumerate(tasks)
    ]


def _registry_defaults() -> tuple[dict[str, dict[str, Any]], str | None]:
    try:
        sys.path.insert(0, str(ROOT))
        import algorithms  # noqa: F401
        from algorithms.base import get_algorithm, list_algorithms

        names = [str(item["name"]) for item in list_algorithms()]
        defaults = {name: get_algorithm(name).get_default_config() for name in names}
        return defaults, None
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        return {}, f"algorithm registry unavailable: {exc}"


def readiness_issues(
    task: Task, registry: dict[str, dict[str, Any]], registry_error: str | None
) -> list[str]:
    issues: list[str] = []
    if registry_error:
        issues.append(registry_error)
        return issues
    if task.algorithm not in registry:
        issues.append(f"algorithm {task.algorithm!r} is not registered")
        return issues
    if _is_finite_epsilon(task.privacy.get("target_epsilon")):
        defaults = registry[task.algorithm]
        if "target_epsilon" not in defaults:
            issues.append(
                f"algorithm {task.algorithm!r} does not advertise target_epsilon "
                "calibration in get_default_config()"
            )
        if (
            "private_loss_channel" in task.requires
            and defaults.get("target_epsilon_includes_auxiliary_channels") is not True
        ):
            issues.append(
                f"algorithm {task.algorithm!r} does not certify that target_epsilon "
                "includes its private auxiliary-loss channel"
            )
    return issues


def _filter_tasks(tasks: list[Task], args: argparse.Namespace) -> list[Task]:
    filtered = tasks
    if args.lane != "all":
        filtered = [task for task in filtered if task.lane == args.lane]
    if args.scenario:
        wanted = set(args.scenario)
        filtered = [task for task in filtered if task.scenario_id in wanted]
    if args.seed:
        wanted_seeds = set(args.seed)
        filtered = [task for task in filtered if task.seed in wanted_seeds]
    # Job indices are intentionally reassigned after lane/scenario filtering;
    # this makes SLURM arrays compact and deterministic for each submitted lane.
    return [
        Task(**{**task.__dict__, "index": index})
        for index, task in enumerate(filtered)
    ]


def _print_tasks(
    tasks: list[Task], registry: dict[str, dict[str, Any]], registry_error: str | None
) -> None:
    print(
        f"{'IDX':>4}  {'READY':<5}  {'LANE':<11}  {'SCENARIO':<30}  "
        f"{'CONDITION':<18}  {'ATTACK':<17}  {'METHOD':<24}  SEED"
    )
    for task in tasks:
        ready = "yes" if not readiness_issues(task, registry, registry_error) else "no"
        print(
            f"{task.index:>4}  {ready:<5}  {task.lane:<11}  "
            f"{task.scenario_id:<30}  {task.condition_id:<18}  "
            f"{task.attack_id:<17}  {task.method_id:<24}  {task.seed}"
        )


def _expected_count_errors(
    documents: list[tuple[Path, dict[str, Any]]], tasks: list[Task]
) -> list[str]:
    errors: list[str] = []
    for path, document in documents:
        actual = sum(1 for task in tasks if task.source_file == path)
        expected = document.get("expected_runs")
        if expected is not None and int(expected) != actual:
            errors.append(f"{path}: expected_runs={expected}, expanded={actual}")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("expanded matrix contains duplicate task IDs")
    return errors


def _command(task: Task, resolved_path: Path, args: argparse.Namespace) -> list[str]:
    data_root = args.data_root or task.resolved_config["data"].get("data_root", "./data")
    command = [
        sys.executable,
        str(ROOT / "run_experiment.py"),
        "--config",
        str(resolved_path),
        "--seed",
        str(task.seed),
        "--output",
        str(task.output_dir),
        "--data-root",
        str(data_root),
    ]
    if args.device:
        command.extend(("--device", args.device))
    if args.quiet:
        command.append("--quiet")
    return command


def _already_complete(task: Task) -> bool:
    return any(task.output_dir.glob("**/metrics.json"))


def _run_task(
    task: Task,
    args: argparse.Namespace,
    registry: dict[str, dict[str, Any]],
    registry_error: str | None,
) -> int:
    issues = readiness_issues(task, registry, registry_error)
    if issues:
        message = f"{task.task_id}: " + "; ".join(issues)
        if args.skip_unavailable:
            print(f"[skip-unavailable] {message}")
            return 0
        print(f"[blocked] {message}", file=sys.stderr)
        return 2
    if args.resume and _already_complete(task):
        print(f"[resume] already complete: {task.task_id}")
        return 0

    resolved = copy.deepcopy(task.resolved_config)
    if args.device:
        resolved["device"] = args.device
    if args.data_root:
        resolved.setdefault("data", {})["data_root"] = args.data_root
    if args.pilot_rounds is not None:
        resolved.setdefault("training", {})["num_rounds"] = int(args.pilot_rounds)
        resolved["training"].setdefault("algo_config", {})[
            "privacy_num_rounds"
        ] = int(args.pilot_rounds)
        resolved.setdefault("eval", {})["eval_every"] = 1
        resolved["reproduction"]["execution_scope"] = "pilot_not_full_protocol"
    if args.pilot_local_batches is not None:
        resolved.setdefault("training", {}).setdefault("algo_config", {})[
            "max_local_batches"
        ] = int(args.pilot_local_batches)
        resolved["training"]["algo_config"]["loss_eval_max_batches"] = int(
            args.pilot_local_batches
        )
        resolved["reproduction"]["execution_scope"] = "pilot_not_full_protocol"
    resolved_path = task.output_dir / "resolved_config.yaml"
    command = _command(task, resolved_path, args)
    if args.dry_run:
        print(f"[{task.index}] {task.task_id}")
        print("  " + " ".join(command))
        return 0

    task.output_dir.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)
    status_path = task.output_dir / "orchestration_status.json"
    status = {
        "task_id": task.task_id,
        "index": task.index,
        "command": command,
        "state": "running",
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[{task.index}] running {task.task_id}")
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    status["state"] = "completed" if completed.returncode == 0 else "failed"
    status["returncode"] = completed.returncode
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return int(completed.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="validate matrices")
    mode.add_argument("--list", action="store_true", help="list expanded task indices")
    mode.add_argument("--dry-run", action="store_true", help="print commands only")
    mode.add_argument("--run", action="store_true", help="execute selected tasks")
    parser.add_argument(
        "--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="matrix directory"
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="results root"
    )
    parser.add_argument(
        "--lane", choices=("all", "faithful", "paper-grade"), default="faithful"
    )
    parser.add_argument(
        "--scenario", action="append", help="scenario ID; repeat to select several"
    )
    parser.add_argument("--seed", action="append", type=int, help="seed filter")
    parser.add_argument("--job-index", type=int, help="one index after filtering")
    parser.add_argument("--max-runs", type=int, help="maximum selected tasks to process")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--data-root", type=str)
    parser.add_argument(
        "--pilot-rounds",
        type=int,
        help="override rounds for a software pilot; results are marked non-protocol",
    )
    parser.add_argument(
        "--pilot-local-batches",
        type=int,
        help="cap local batches for a software pilot; results are marked non-protocol",
    )
    parser.add_argument("--resume", action="store_true", help="skip completed run dirs")
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        help="skip unregistered or uncalibrated algorithms instead of failing",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    documents = _load_documents(args.config_dir.resolve())
    schema_errors = [
        error
        for path, document in documents
        for error in _validate_document(path, document)
    ]
    all_tasks = expand_tasks(documents, output_root=args.output_root.resolve())
    count_errors = _expected_count_errors(documents, all_tasks)
    if schema_errors or count_errors:
        for error in schema_errors + count_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    registry, registry_error = _registry_defaults()
    tasks = _filter_tasks(all_tasks, args)
    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(tasks):
            print(
                f"ERROR: --job-index {args.job_index} outside [0,{len(tasks)-1}]",
                file=sys.stderr,
            )
            return 2
        tasks = [tasks[args.job_index]]
    if args.max_runs is not None:
        tasks = tasks[: max(0, int(args.max_runs))]

    unavailable = sum(
        bool(readiness_issues(task, registry, registry_error)) for task in tasks
    )
    if args.validate:
        lane_counts: dict[str, int] = {}
        for task in all_tasks:
            lane_counts[task.lane] = lane_counts.get(task.lane, 0) + 1
        print(f"Schema OK: {len(documents)} files, {len(all_tasks)} total tasks")
        for lane, count in sorted(lane_counts.items()):
            print(f"  {lane}: {count}")
        print(f"Selected: {len(tasks)}; currently unavailable: {unavailable}")
        print(
            "Fidelity warning: the report's shared Dirichlet partition seed is "
            "not recoverable from the supplied report; seed 28 is used as an "
            "explicit shared partition seed for all training seeds."
        )
        return 0

    if args.list or not (args.run or args.dry_run):
        _print_tasks(tasks, registry, registry_error)
        print(f"\nSelected tasks: {len(tasks)}; currently unavailable: {unavailable}")
        return 0

    failures = 0
    for task in tasks:
        rc = _run_task(task, args, registry, registry_error)
        failures += int(rc != 0)
        if rc != 0 and not args.skip_unavailable:
            break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
