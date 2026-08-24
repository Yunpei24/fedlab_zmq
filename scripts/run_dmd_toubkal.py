#!/usr/bin/env python3
"""Validate, enumerate, and execute the DMD Toubkal experiment matrix.

Each array task owns exactly one method directory.  This prevents concurrent
jobs from overwriting the DMD prototype's shared manifest and aggregate CSVs.
The scientific runner remains non-private by construction; this launcher
refuses matrices that claim to be differentially private.
"""

from __future__ import annotations

import argparse
import hashlib
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

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "dmd" / "toubkal_phase_a.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "toubkal_dmd"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Task:
    index: int
    campaign: str
    stage_id: str
    scenario_id: str
    participation_id: str
    method_id: str
    runner_method: str
    seed: int
    alpha: float
    config: dict[str, Any]
    output_dir: Path

    @property
    def task_id(self) -> str:
        return "/".join(
            (
                self.stage_id,
                self.scenario_id,
                self.participation_id,
                _alpha_slug(self.alpha),
                f"seed{self.seed}",
                self.method_id,
            )
        )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)


def _alpha_slug(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return document


def validate_matrix(path: Path, document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{path}: schema_version must be {SCHEMA_VERSION}")
    if not document.get("campaign"):
        errors.append(f"{path}: campaign is required")
    if "private" not in str(document.get("campaign", "")).lower():
        errors.append(f"{path}: campaign name must explicitly contain 'non_private'")
    defaults = document.get("defaults")
    methods = document.get("methods")
    scenarios = document.get("scenarios")
    stages = document.get("stages")
    if not isinstance(defaults, dict):
        errors.append(f"{path}: defaults must be a mapping")
    if not isinstance(methods, list) or not methods:
        errors.append(f"{path}: methods must be a non-empty list")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append(f"{path}: scenarios must be a non-empty list")
    if not isinstance(stages, list) or not stages:
        errors.append(f"{path}: stages must be a non-empty list")
        return errors

    method_ids: set[str] = set()
    for method in methods or []:
        method_id = str(method.get("id", ""))
        if not method_id or not method.get("runner_method"):
            errors.append(f"{path}: each method needs id and runner_method")
        if method_id in method_ids:
            errors.append(f"{path}: duplicate method id {method_id}")
        method_ids.add(method_id)

    scenario_ids: set[str] = set()
    for scenario in scenarios or []:
        scenario_id = str(scenario.get("id", ""))
        if not scenario_id:
            errors.append(f"{path}: each scenario needs an id")
        if scenario_id in scenario_ids:
            errors.append(f"{path}: duplicate scenario id {scenario_id}")
        scenario_ids.add(scenario_id)
        dataset = scenario.get("dataset")
        model = scenario.get("model")
        classes = scenario.get("classes")
        valid = (
            dataset == "cifar10" and model == "resnet18_gn4" and classes == 10
        ) or (dataset == "emnist" and model == "cnn_gn" and classes == 62)
        if not valid:
            errors.append(
                f"{path}:{scenario_id}: unsupported dataset/model/classes contract"
            )

    stage_ids: set[str] = set()
    for stage in stages:
        stage_id = str(stage.get("id", ""))
        if not stage_id:
            errors.append(f"{path}: each stage needs an id")
        if stage_id in stage_ids:
            errors.append(f"{path}: duplicate stage id {stage_id}")
        stage_ids.add(stage_id)
        seeds = stage.get("seeds")
        alphas = stage.get("alphas")
        participations = stage.get("participations")
        if not isinstance(seeds, list) or not seeds:
            errors.append(f"{path}:{stage_id}: seeds must be non-empty")
        if not isinstance(alphas, list) or not alphas:
            errors.append(f"{path}:{stage_id}: alphas must be non-empty")
        if not isinstance(participations, list) or not participations:
            errors.append(f"{path}:{stage_id}: participations must be non-empty")
        for alpha in alphas or []:
            try:
                if float(alpha) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{path}:{stage_id}: alpha must be positive: {alpha}")
        for participation in participations or []:
            rate = float(participation.get("participation_rate", -1))
            dropout = float(participation.get("dropout_rate", -1))
            if not 0 < rate <= 1:
                errors.append(f"{path}:{stage_id}: participation_rate must be in (0,1]")
            if not 0 <= dropout < 1:
                errors.append(f"{path}:{stage_id}: dropout_rate must be in [0,1)")
        expected = stage.get("expected_tasks")
        actual = (
            len(scenarios or [])
            * len(participations or [])
            * len(alphas or [])
            * len(seeds or [])
            * len(methods or [])
        )
        if expected is not None and int(expected) != actual:
            errors.append(
                f"{path}:{stage_id}: expected_tasks={expected}, expanded={actual}"
            )
    return errors


def expand_tasks(
    document: dict[str, Any],
    *,
    output_root: Path,
    stage_ids: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    method_ids: set[str] | None = None,
    seeds: set[int] | None = None,
    alphas: set[float] | None = None,
    pilot_rounds: int | None = None,
) -> list[Task]:
    campaign = _slug(str(document["campaign"]))
    methods = document["methods"]
    scenarios = document["scenarios"]
    defaults = dict(document["defaults"])
    tasks: list[Task] = []
    for stage in document["stages"]:
        stage_id = str(stage["id"])
        if stage_ids and stage_id not in stage_ids:
            continue
        for scenario in scenarios:
            scenario_id = str(scenario["id"])
            if scenario_ids and scenario_id not in scenario_ids:
                continue
            for participation in stage["participations"]:
                participation_id = str(participation["id"])
                for alpha_value in stage["alphas"]:
                    alpha = float(alpha_value)
                    if alphas and not any(math.isclose(alpha, item) for item in alphas):
                        continue
                    for seed_value in stage["seeds"]:
                        seed = int(seed_value)
                        if seeds and seed not in seeds:
                            continue
                        for method in methods:
                            method_id = str(method["id"])
                            if method_ids and method_id not in method_ids:
                                continue
                            config = dict(defaults)
                            config.update(
                                {
                                    "rounds": int(stage.get("rounds", defaults["rounds"])),
                                    "dataset": str(scenario["dataset"]),
                                    "model": str(scenario["model"]),
                                    "classes": int(scenario["classes"]),
                                    "participation_rate": float(
                                        participation["participation_rate"]
                                    ),
                                    "dropout_rate": float(
                                        participation["dropout_rate"]
                                    ),
                                }
                            )
                            root = output_root / campaign
                            if pilot_rounds is not None:
                                config["rounds"] = int(pilot_rounds)
                                root = root / "pilots" / f"rounds_{pilot_rounds}"
                            output_dir = root.joinpath(
                                _slug(stage_id),
                                _slug(scenario_id),
                                _slug(participation_id),
                                _alpha_slug(alpha),
                                f"seed{seed}",
                                _slug(method_id),
                            )
                            tasks.append(
                                Task(
                                    index=-1,
                                    campaign=campaign,
                                    stage_id=stage_id,
                                    scenario_id=scenario_id,
                                    participation_id=participation_id,
                                    method_id=method_id,
                                    runner_method=str(method["runner_method"]),
                                    seed=seed,
                                    alpha=alpha,
                                    config=config,
                                    output_dir=output_dir,
                                )
                            )
    return [replace(task, index=index) for index, task in enumerate(tasks)]


def validate_data_root(data_root: Path, tasks: Iterable[Task]) -> list[str]:
    errors: list[str] = []
    datasets = {task.config["dataset"] for task in tasks}
    if "cifar10" in datasets:
        required = data_root / "cifar-10-batches-py" / "data_batch_1"
        if not required.exists():
            errors.append(f"missing CIFAR-10 under {data_root}: {required}")
    if "emnist" in datasets:
        required = (
            data_root
            / "EMNIST"
            / "raw"
            / "emnist-byclass-train-images-idx3-ubyte"
        )
        if not required.exists():
            errors.append(f"missing EMNIST/ByClass under {data_root}: {required}")
    return errors


def task_command(
    task: Task,
    *,
    python_bin: str,
    device: str,
    data_root: Path,
    resume: bool,
) -> list[str]:
    cfg = task.config
    command = [
        python_bin,
        str(ROOT / "research" / "dmd_prototype" / "run_cifar10_experiment.py"),
        "--output-dir",
        str(task.output_dir),
        "--methods",
        task.runner_method,
        "--data-root",
        str(data_root),
        "--dataset",
        str(cfg["dataset"]),
        "--model",
        str(cfg["model"]),
        "--classes",
        str(cfg["classes"]),
        "--device",
        device,
        "--seed",
        str(task.seed),
        "--partition-seed",
        str(task.seed),
        "--participation-seed",
        str(task.seed),
        "--clients",
        str(cfg["clients"]),
        "--participation-rate",
        str(cfg["participation_rate"]),
        "--dropout-rate",
        str(cfg["dropout_rate"]),
        "--rounds",
        str(cfg["rounds"]),
        "--warmup-rounds",
        str(cfg["warmup_rounds"]),
        "--dirichlet-alpha",
        str(task.alpha),
        "--anchor-fraction",
        str(cfg["anchor_fraction"]),
        "--max-train-samples",
        str(cfg["max_train_samples"]),
        "--max-anchor-samples",
        str(cfg["max_anchor_samples"]),
        "--min-train-samples",
        str(cfg["min_train_samples"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--eval-batch-size",
        str(cfg["eval_batch_size"]),
        "--test-eval-every",
        str(cfg["test_eval_every"]),
        "--local-epochs",
        str(cfg["local_epochs"]),
        "--learning-rate",
        str(cfg["learning_rate"]),
        "--momentum",
        str(cfg["momentum"]),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--gradient-clip",
        str(cfg["gradient_clip"]),
        "--dmd-mean-mu",
        str(cfg["dmd_mean_mu"]),
        "--dmd-tail-mu",
        str(cfg["dmd_tail_mu"]),
        "--dmd-dispersion-mu",
        str(cfg["dmd_dispersion_mu"]),
        "--dmd-cvar-tail-mass",
        str(cfg["dmd_cvar_tail_mass"]),
        "--dmd-client-weights",
        str(cfg["dmd_client_weights"]),
        "--loss-fair-mu",
        str(cfg["loss_fair_mu"]),
        "--term-tau",
        str(cfg["term_tau"]),
        "--term-weight-cap",
        str(cfg["term_weight_cap"]),
        "--min-reference-clients",
        str(cfg["min_reference_clients"]),
        "--reliability-power",
        str(cfg["reliability_power"]),
        "--calibration-bins",
        str(cfg["calibration_bins"]),
    ]
    if resume:
        command.append("--resume")
    return command


def is_complete(task: Task) -> bool:
    rounds = int(task.config["rounds"])
    stem = f"{task.runner_method}_seed{task.seed}"
    round_path = task.output_dir / f"round_metrics_{stem}.csv"
    client_path = task.output_dir / f"client_metrics_{stem}.csv"
    checkpoint = task.output_dir / f"checkpoint_{stem}.pt"
    model = task.output_dir / f"model_{stem}.pt"
    if not all(path.exists() for path in (round_path, client_path, checkpoint, model)):
        return False
    try:
        frame = pd.read_csv(round_path, usecols=["round"])
    except (ValueError, pd.errors.EmptyDataError):
        return False
    return bool(
        len(frame) == rounds
        and int(frame["round"].min()) == 1
        and int(frame["round"].max()) == rounds
        and not frame["round"].duplicated().any()
    )


def _write_task_manifest(
    task: Task,
    *,
    matrix_path: Path,
    command: list[str],
    device: str,
    status: str,
    started: float,
    finished: float | None = None,
) -> Path:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "task_index": task.index,
        "task_id": task.task_id,
        "campaign": task.campaign,
        "stage": task.stage_id,
        "scenario": task.scenario_id,
        "participation": task.participation_id,
        "method_id": task.method_id,
        "runner_method": task.runner_method,
        "seed": task.seed,
        "alpha": task.alpha,
        "config": task.config,
        "device": device,
        "command": command,
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": _sha256(matrix_path),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": started,
        "finished_unix": finished,
    }
    path = task.output_dir / "toubkal_task_manifest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def export_dashboard(task: Task, dashboard_root: Path, python_bin: str) -> None:
    destination = dashboard_root.joinpath(
        task.campaign,
        task.stage_id,
        task.scenario_id,
        task.participation_id,
        _alpha_slug(task.alpha),
        f"seed{task.seed}",
        task.method_id,
    )
    subprocess.run(
        [
            python_bin,
            str(ROOT / "scripts" / "export_dmd_to_dashboard.py"),
            "--input-root",
            str(task.output_dir),
            "--output-root",
            str(destination),
            "--overwrite",
        ],
        cwd=ROOT,
        check=True,
    )


def _parse_csv_set(value: str | None, cast: type) -> set[Any] | None:
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
    parser.add_argument("--stage", default=None, help="Comma-separated stage IDs")
    parser.add_argument("--scenario", default=None, help="Comma-separated scenario IDs")
    parser.add_argument("--method", default=None, help="Comma-separated method IDs")
    parser.add_argument("--seed", default=None, help="Comma-separated seeds")
    parser.add_argument("--alpha", default=None, help="Comma-separated Dirichlet alphas")
    parser.add_argument("--job-index", type=int, default=None)
    parser.add_argument("--pilot-rounds", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-data-check", action="store_true")
    parser.add_argument("--no-dashboard-export", action="store_true")
    parser.add_argument("--dashboard-output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix.resolve()
    document = load_matrix(matrix_path)
    errors = validate_matrix(matrix_path, document)
    if errors:
        raise SystemExit("Invalid DMD matrix:\n- " + "\n- ".join(errors))
    if args.pilot_rounds is not None and args.pilot_rounds <= 0:
        raise ValueError("pilot-rounds must be positive")
    tasks = expand_tasks(
        document,
        output_root=args.output_root.resolve(),
        stage_ids=_parse_csv_set(args.stage, str),
        scenario_ids=_parse_csv_set(args.scenario, str),
        method_ids=_parse_csv_set(args.method, str),
        seeds=_parse_csv_set(args.seed, int),
        alphas=_parse_csv_set(args.alpha, float),
        pilot_rounds=args.pilot_rounds,
    )
    if not tasks:
        raise SystemExit("No DMD task matches the requested filters")

    if (args.validate or args.run) and not args.skip_data_check:
        data_errors = validate_data_root(args.data_root.resolve(), tasks)
        if data_errors:
            raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(data_errors))

    if args.validate:
        print(f"VALID: {len(tasks)} unique tasks")
        print(f"stages={sorted({task.stage_id for task in tasks})}")
        print(f"datasets={sorted({task.config['dataset'] for task in tasks})}")
        return

    if args.list:
        for task in tasks:
            print(
                f"{task.index:04d}\t{task.task_id}\t{task.runner_method}\t{task.output_dir}"
            )
        return

    if args.job_index is None:
        raise SystemExit("--job-index is required for --dry-run and --run")
    if not 0 <= args.job_index < len(tasks):
        raise SystemExit(f"job-index must be between 0 and {len(tasks) - 1}")
    task = tasks[args.job_index]
    command = task_command(
        task,
        python_bin=args.python_bin,
        device=args.device,
        data_root=args.data_root.resolve(),
        resume=args.resume,
    )
    if args.dry_run:
        print(json.dumps({"task": task.task_id, "command": command}, indent=2))
        return

    started = time.time()
    if args.resume and is_complete(task):
        print(f"SKIP complete task: {task.task_id}")
        status = "complete-reused"
    else:
        _write_task_manifest(
            task,
            matrix_path=matrix_path,
            command=command,
            device=args.device,
            status="running",
            started=started,
        )
        print(f"RUN {task.index}: {task.task_id}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if not is_complete(task):
            raise RuntimeError(f"runner exited but task is incomplete: {task.task_id}")
        status = "complete"

    if not args.no_dashboard_export:
        dashboard_root = (
            args.dashboard_output_root.resolve()
            if args.dashboard_output_root
            else args.output_root.resolve() / "dashboard_exports"
        )
        export_dashboard(task, dashboard_root, args.python_bin)
    _write_task_manifest(
        task,
        matrix_path=matrix_path,
        command=command,
        device=args.device,
        status=status,
        started=started,
        finished=time.time(),
    )
    print(f"DONE {task.task_id}", flush=True)


if __name__ == "__main__":
    main()

