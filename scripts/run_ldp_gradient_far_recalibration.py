#!/usr/bin/env python3
"""Resolve and run the registered LDP-gradient-FAR recalibration screen."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT / "configs" / "ldp_gradient_far" / "fmnist_recalibration_v1.yaml"
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_jobs(
    matrix_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, list[int | None]]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if int(matrix.get("schema_version", 0)) != 1:
        raise ValueError("unsupported recalibration matrix schema")
    base_path = (matrix_path.parent / str(matrix["base_config"])).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    jobs = matrix.get("jobs", [])
    ids = [str(job["id"]) for job in jobs]
    if not jobs or len(ids) != len(set(ids)):
        raise ValueError("jobs must be non-empty and have unique ids")
    output_root = (matrix_path.parent / str(matrix["output_root"])).resolve()
    raw_seeds = matrix.get("seeds")
    seeds = [None] if raw_seeds is None else [int(seed) for seed in raw_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique when specified")
    return base, jobs, output_root, seeds


def resolved_job(
    base: dict[str, Any],
    job: dict[str, Any],
    output_dir: Path,
    seed: int | None,
    device: str = "mps",
) -> dict[str, Any]:
    config = deep_merge(base, job.get("overrides", {}))
    config["output_dir"] = str(output_dir)
    # Record the same explicit device that is passed to run_experiment.py so
    # the saved provenance cannot misleadingly say CPU for an MPS run.
    config["device"] = str(device)
    if seed is not None:
        config["seed"] = int(seed)
        config.setdefault("data", {})["partition_seed"] = int(seed)
    algo = config["training"]["algo_config"]
    rounds = int(config["training"]["num_rounds"])
    if bool(algo.get("enable_dp", True)):
        algo["privacy_num_rounds"] = rounds
    algo["tilt_bound_policy"] = "diagnostic"
    return config


def completed_metrics(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("**/metrics.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if len(payload.get("rounds", [])) == int(
            payload.get("summary", {}).get("num_rounds", -1)
        ):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    base, jobs, output_root, seeds = load_jobs(matrix_path)
    tasks = [(job, seed) for job in jobs for seed in seeds]
    indexed = list(enumerate(tasks))
    if args.job_index is not None:
        if not 0 <= args.job_index < len(indexed):
            raise SystemExit(f"job-index must lie in [0,{len(indexed) - 1}]")
        indexed = [indexed[args.job_index]]

    print(f"matrix={matrix_path}")
    print(
        f"jobs={len(jobs)} seeds={len(seeds)} tasks={len(tasks)} "
        f"selected={len(indexed)}"
    )
    for index, (job, seed) in indexed:
        base_job_id = str(job["id"])
        job_id = base_job_id if seed is None else f"{base_job_id}_seed{seed}"
        output_dir = output_root / job_id
        config = resolved_job(base, job, output_dir, seed, args.device)
        algo = config["training"]["algo_config"]
        print(
            f"[{index:02d}] {job_id}: dp={algo.get('enable_dp', True)} "
            f"U={algo['far_server_clip_norm']} "
            f"lr={algo['far_server_lr']} alpha={algo['far_alpha']}"
        )
        if not args.run:
            continue
        if args.resume and completed_metrics(output_dir) is not None:
            print("  completed; skipped")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
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
            args.device,
            "--output",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
