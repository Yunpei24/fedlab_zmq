#!/usr/bin/env python3
"""Isolated, fail-closed 216-run batch screen; --plan/--status never import torch.

Execution is explicitly MPS-only, sequential, source-locked and resume-only.
An incomplete or invalid run is never overwritten or silently retried.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shlex
import subprocess
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/ldp_gradient_far/rcig_batch_screen_v1.yaml"
CAMPAIGN_ID = "rcig_batch_screen_v1"
OUTPUT_ROOT = ROOT / "results/ldp_gradient_far" / CAMPAIGN_ID
ENTRYPOINT = ROOT / "scripts/run_rcig_batch_screen_experiment.py"
RECENT_MODULE = ROOT / "algorithms/ldp_gradient_far_recent.py"
LOCK_NAME = "_campaign_lock.json"
INTERPRETATION = "exploratory_screen_not_confirmatory"
SEEDS = (24, 42, 72, 121)
OUTCOMES = (
    "test_accuracy",
    "test_loss",
    "client_accuracy_mean",
    "client_accuracy_variance_pct2",
    "worst20_accuracy_pct",
    "best20_worst20_gap_pct",
)


@dataclass(frozen=True)
class ScreenTask:
    global_index: int
    batch_size: int
    noise_regime: str
    seed: int
    scenario: str
    reference: str

    @property
    def run_id(self) -> str:
        return f"b{self.batch_size}__{self.noise_regime}__seed{self.seed}__{self.scenario}__{self.reference}"

    @property
    def pairing_block(self) -> str:
        return (
            f"b{self.batch_size}__{self.noise_regime}__seed{self.seed}__{self.scenario}"
        )


@dataclass(frozen=True)
class ScreenCampaign:
    matrix_path: Path
    matrix: dict
    output_root: Path
    tasks: tuple[ScreenTask, ...]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def load_campaign(matrix_path: Path = DEFAULT_MATRIX) -> ScreenCampaign:
    matrix_path = matrix_path.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scientific_status": INTERPRETATION,
        "required_device": "mps",
        "expected_runs": 216,
        "batch_sizes": [120, 240, 480],
        "seeds": list(SEEDS),
        "references": ["uniform", "rfa", "recent"],
        "scenarios": ["none", "bf_x10_persistent", "ipm_persistent"],
    }
    if not isinstance(matrix, dict) or any(
        matrix.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "The screen must retain the exact user-authorized 216-run matrix"
        )
    scales = matrix["noise_regimes"]
    if scales != {"homogeneous": [1] * 25, "heteroscedastic": [1, 2] * 12 + [1]}:
        raise ValueError("Incorrect authenticated 25-client noise registry")
    if matrix["privacy"] != {
        "epsilon": 4.0,
        "delta": 1e-5,
        "epsilon_tolerance": 1e-4,
        "orders": [2, 3, 4, 5, 8, 10, 16, 20, 32, 64],
        "sigma_calibration": "fixed_wor_replace_one_per_B_frozen_across_references",
        "interpretation": "per_client_per_run_not_composed_across_experiments",
    }:
        raise ValueError("Screen privacy contract changed")
    configured_output = (matrix_path.parent / matrix["output_root"]).resolve()
    if (
        matrix_path == DEFAULT_MATRIX.resolve()
        and configured_output != OUTPUT_ROOT.resolve()
    ):
        raise ValueError("Screen must use its distinct result namespace")
    tasks = tuple(
        ScreenTask(index, batch, noise, seed, scenario, reference)
        for index, (batch, noise, seed, scenario, reference) in enumerate(
            itertools.product(
                matrix["batch_sizes"],
                scales,
                matrix["seeds"],
                matrix["scenarios"],
                matrix["references"],
            )
        )
    )
    campaign = ScreenCampaign(matrix_path, matrix, configured_output, tasks)
    if len(tasks) != 216 or len({t.run_id for t in tasks}) != 216:
        raise ValueError("Duplicate/missing screen cells")
    for task in tasks:
        validate_resolved_config(resolved_config(campaign, task, {}), task)
    return campaign


@lru_cache(maxsize=1)
def rdp_module():
    name = "_rcig_screen_existing_rdp"
    spec = importlib.util.spec_from_file_location(name, ROOT / "privacy/rdp.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Existing fixed-WOR accountant unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=3)
def calibrated_sigma(batch_size: int) -> float:
    if batch_size not in (120, 240, 480):
        raise ValueError("Undeclared batch size")
    return rdp_module().calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=4.0,
        delta=1e-5,
        sampling_rate=batch_size / 2400,
        steps=40,
        sensitivity_multiplier=2.0,
        tolerance=1e-4,
    )


@lru_cache(maxsize=512)
def epsilon_at(batch_size: int, rounds: int, scale: float = 1.0) -> float:
    accountant = rdp_module().RDPAccountant()
    accountant.add_sampled_without_replacement_gaussian(
        channel="model",
        sampling_rate=batch_size / 2400,
        noise_multiplier=calibrated_sigma(batch_size) * scale / 2.0,
        steps=rounds,
    )
    return accountant.epsilon(1e-5)[0]


def privacy_table() -> list[dict]:
    return [
        {
            "batch_size": batch,
            "q": batch / 2400,
            "sigma": calibrated_sigma(batch),
            "sd_scale_1": 4 * calibrated_sigma(batch) / batch,
            "epsilon_scale_1": epsilon_at(batch, 40),
            "epsilon_scale_2": epsilon_at(batch, 40, 2),
            "max_epsilon_absolute_error": abs(epsilon_at(batch, 40) - 4),
            "epsilon_tolerance": 1e-4,
        }
        for batch in (120, 240, 480)
    ]


def task_output_dir(campaign: ScreenCampaign, task: ScreenTask) -> Path:
    return campaign.output_root / "runs" / task.run_id


def attack_config(task: ScreenTask) -> dict:
    if task.scenario == "none":
        return {
            "enabled": False,
            "name": "none",
            "scale": 1.0,
            "num_byzantine": 0,
            "client_ids": [],
        }
    return {
        "enabled": True,
        "name": "bf" if task.scenario.startswith("bf_") else "ipm",
        "scale": 10.0 if task.scenario.startswith("bf_") else 1.0,
        "active_round_start": 17,
        "active_round_end": 40,
        "num_byzantine": 5,
        "client_ids": [0, 1, 2, 3, 4],
    }


def resolved_config(campaign: ScreenCampaign, task: ScreenTask, stamp: Mapping) -> dict:
    config = copy.deepcopy(campaign.matrix["base"])
    config.update(
        seed=task.seed, device="mps", output_dir=str(task_output_dir(campaign, task))
    )
    config["data"]["partition_seed"] = task.seed
    config["training"]["algorithm"] = (
        "ldp_gradient_far_recent" if task.reference == "recent" else "ldp_gradient_far"
    )
    algo = config["training"]["algo_config"]
    algo.update(
        {
            "batch_size": task.batch_size,
            "fixed_batch_size": task.batch_size,
            "privacy_sampling_rate_override": task.batch_size / 2400,
            "noise_multiplier": calibrated_sigma(task.batch_size),
            "privacy_noise_multiplier_scale_by_client": campaign.matrix[
                "noise_regimes"
            ][task.noise_regime],
            "far_alpha": 0.0 if task.reference == "uniform" else 0.1,
            "robust_reference": {
                "uniform": "centered_clipping",
                "rfa": "rfa",
                "recent": "rcig_temporal_full",
            }[task.reference],
            "rcig_oracle_separation_required": task.reference == "recent",
            "enable_oracle_diagnostics": task.reference == "recent",
            "rcig_oracle_evaluation_only": task.reference == "recent",
            "rcig_oracle_metrics_are_not_release": task.reference == "recent",
            "attack": attack_config(task),
            "rcig_campaign_id": CAMPAIGN_ID,
            "rcig_campaign_scientific_hash": canonical_hash(stamp),
            "rcig_screen_reference": task.reference,
            "rcig_screen_batch_size": task.batch_size,
            "rcig_screen_noise_regime": task.noise_regime,
            "rcig_screen_scenario": task.scenario,
            "rcig_pairing_seed_block": task.seed,
            "rcig_screen_pairing_block": task.pairing_block,
            "rcig_screen_interpretation": INTERPRETATION,
        }
    )
    algo["rcig_pairing_design_sha256"] = canonical_hash(
        {
            "seed": task.seed,
            "data": config["data"],
            "model": config["model"],
            "clients": config["clients"],
            "batch_size": task.batch_size,
            "rounds": 40,
            "noise_scales": algo["privacy_noise_multiplier_scale_by_client"],
            "sigma": algo["noise_multiplier"],
            "attack": algo["attack"],
        }
    )
    return config


def validate_resolved_config(config: Mapping, task: ScreenTask) -> None:
    if config.get("device") != "mps":
        raise ValueError("Private CPU/CUDA compute is forbidden; MPS only")
    training, data, clients = config["training"], config["data"], config["clients"]
    if (
        data.get("dataset"),
        data.get("partition"),
        data.get("alpha"),
        config["model"].get("architecture"),
    ) != ("fashionmnist", "client_dirichlet_balanced", 0.1, "lenet5_tanh"):
        raise ValueError("Dataset/model/partition contract changed")
    if config.get("seed") != task.seed or data.get("partition_seed") != task.seed:
        raise ValueError("Pairing seed mismatch")
    if (
        training.get("num_rounds") != 40
        or clients.get("num_clients") != 25
        or clients.get("min_clients") != 25
        or clients.get("sample_fraction") != 1.0
        or clients.get("dropout_rate") != 0.0
    ):
        raise ValueError("Exactly 40 rounds with 25 participating clients are required")
    algo = training["algo_config"]
    contract = {
        "batch_size": task.batch_size,
        "fixed_batch_size": task.batch_size,
        "privacy_public_dataset_size": 2400,
        "local_epochs": 1,
        "fixed_steps_per_round": 1,
        "expected_num_clients": 25,
        "privacy_num_rounds": 40,
        "privacy_sampling_rate_override": task.batch_size / 2400,
        "target_epsilon": 4.0,
        "delta": 1e-5,
        "clip_norm": 4.0,
        "far_server_clip_norm": 16.0,
        "far_server_lr": 0.2,
        "sampling_scheme": "fixed_without_replacement",
        "privacy_adjacency": "replace_one",
        "per_sample_backend": "vectorized",
        "enable_dp": True,
        "far_score_mode": "raw_distance",
        "noise_score_standardization": "none",
        "score_subspace_mode": "full",
        "tilt_bound_policy": "diagnostic",
        "client_metrics_every": 1,
        "fairness_tail_fraction": 0.2,
        "rcig_required_private_device": "mps",
        "rcig_mps_fallback": 0,
        "external_attack_diagnostics": True,
        "enable_oracle_diagnostics": task.reference == "recent",
        "rcig_oracle_evaluation_only": task.reference == "recent",
        "rcig_oracle_metrics_are_not_release": task.reference == "recent",
        "far_alpha": 0.0 if task.reference == "uniform" else 0.1,
        "robust_reference": {
            "uniform": "centered_clipping",
            "rfa": "rfa",
            "recent": "rcig_temporal_full",
        }[task.reference],
        "noise_multiplier": calibrated_sigma(task.batch_size),
        "attack": attack_config(task),
        "privacy_noise_multiplier_scale_by_client": (
            [1] * 25 if task.noise_regime == "homogeneous" else [1, 2] * 12 + [1]
        ),
    }
    for key, value in contract.items():
        if algo.get(key) != value:
            raise ValueError(f"Resolved screen contract mismatch: {key}")
    expected_algorithm = (
        "ldp_gradient_far_recent" if task.reference == "recent" else "ldp_gradient_far"
    )
    if training.get("algorithm") != expected_algorithm:
        raise ValueError("Reference algorithm identity mismatch")
    if any(
        algo.get(key) is not None
        for key in (
            "rcig_innovation_threshold",
            "rcig_isotropic_innovation_threshold",
            "rcig_euclidean_innovation_threshold",
            "rcig_threshold_artifact_sha256",
        )
    ):
        raise ValueError("This screen has no calibrated innovation thresholds")
    if task.reference == "recent":
        recent_contract = {
            "rcig_gate_window": 4,
            "rcig_old_window": 4,
            "rcig_new_window": 4,
            "rcig_warmup_policy": "uniform",
            "rcig_persistent_policy": "recent_only",
            "rcig_covariance_mode": "full",
            "rcig_reference_output_radius": 16.0,
        }
        if any(algo.get(key) != value for key, value in recent_contract.items()):
            raise ValueError("Recent-only control/warmup contract changed")
    if abs(epsilon_at(task.batch_size, 40) - 4.0) > 1e-4:
        raise ValueError("Privacy calibration outside epsilon tolerance")


def source_closure(entrypoints: tuple[Path, ...]) -> dict[str, str]:
    """Conservative transitive local import closure, without executing imports.

    Include package initializers and imported submodules, including imports inside
    functions. Runtime manifests must later be a hash-equal subset of this lock.
    """
    pending = list(entrypoints)
    seen: set[Path] = set()

    def enqueue_module(name: str):
        parts = name.split(".") if name else []
        for index in range(1, len(parts) + 1):
            prefix = ROOT.joinpath(*parts[:index])
            for candidate in (prefix.with_suffix(".py"), prefix / "__init__.py"):
                if candidate.is_file() and candidate not in seen:
                    pending.append(candidate)

    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise RuntimeError(
                f"Required local source missing/outside workspace: {path}"
            )
        seen.add(path)
        relative = path.relative_to(ROOT)
        package = list(relative.parts[:-1])
        for node in ast.walk(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        ):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join(
                    [*prefix, *(node.module.split(".") if node.module else [])]
                )
                enqueue_module(module)
                for alias in node.names:
                    if alias.name != "*":
                        enqueue_module(".".join(filter(None, (module, alias.name))))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(seen)}


def provenance(campaign: ScreenCampaign) -> dict:
    protocol = ROOT / campaign.matrix["protocol"]
    if not protocol.is_file():
        raise RuntimeError(f"Preregistered screen protocol missing: {protocol}")
    sources = source_closure(
        (
            Path(__file__).resolve(),
            ENTRYPOINT,
            RECENT_MODULE,
            ROOT / "run_experiment.py",
        )
    )
    # rdp.py is deliberately loaded by filename to avoid importing training.
    sources["privacy/rdp.py"] = file_hash(ROOT / "privacy/rdp.py")
    return {
        "campaign_id": CAMPAIGN_ID,
        "interpretation": INTERPRETATION,
        "matrix_sha256": file_hash(campaign.matrix_path),
        "protocol_sha256": file_hash(protocol),
        "source_sha256": dict(sorted(sources.items())),
        "output_root": str(campaign.output_root),
        "task_count": 216,
        "seeds": list(SEEDS),
        "privacy_table": privacy_table(),
        "private_compute_device": "mps",
        "mps_fallback": 0,
        "source_discovery": "static_local_import_closure_checked_against_runtime_imports",
    }


def verify_lock(campaign: ScreenCampaign, stamp: Mapping) -> None:
    observed = read_json(campaign.output_root / LOCK_NAME)
    if observed.get("provenance") != stamp or observed.get(
        "scientific_hash"
    ) != canonical_hash(stamp):
        raise RuntimeError(
            "Screen source/config/protocol provenance drift; refusing execution/resume"
        )


def write_json(path: Path, value: Mapping, *, exclusive: bool = False) -> None:
    destination = (
        path if exclusive else path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    )
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not exclusive:
        os.replace(destination, path)


def ensure_lock(campaign: ScreenCampaign, stamp: Mapping) -> None:
    if (campaign.output_root / LOCK_NAME).exists():
        verify_lock(campaign, stamp)
        return
    if campaign.output_root.exists() and any(campaign.output_root.iterdir()):
        raise RuntimeError("Refusing to adopt pre-existing unlocked screen artifacts")
    campaign.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        campaign.output_root / LOCK_NAME,
        {
            "provenance": dict(stamp),
            "scientific_hash": canonical_hash(stamp),
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        exclusive=True,
    )


@contextmanager
def execution_lock(campaign: ScreenCampaign):
    """One screen launcher globally, including all --job-index subsets."""
    campaign.output_root.parent.mkdir(parents=True, exist_ok=True)
    path = campaign.output_root.parent / f".{CAMPAIGN_ID}.execution.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another screen launcher already owns the global execution lock"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "campaign_id": CAMPAIGN_ID}))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_working_mps() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("MPS fallback must be disabled, never redirected to CPU")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Native MPS unavailable; private CPU fallback forbidden")
    torch.ones(2, device="mps").square().sum().cpu()
    torch.mps.synchronize()


def check_no_other_training_process() -> None:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    guarded = {
        "run_experiment.py",
        "run_rcig_batch_screen_experiment.py",
        "run_rcig_r2_r3_exploratory.py",
        "run_rcig_ldp_gradient_far_v2.py",
    }
    conflicts = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3 or int(fields[0]) == os.getpid():
            continue
        try:
            args = shlex.split(fields[2])
        except ValueError:
            continue
        found = [Path(arg).name for arg in args if Path(arg).name in guarded]
        if found and (
            found[0] in {"run_experiment.py", "run_rcig_batch_screen_experiment.py"}
            or "--run" in args
        ):
            conflicts.append({"pid": int(fields[0]), "command": fields[2]})
    if conflicts:
        raise RuntimeError(
            f"Other experiment process active; refusing concurrent private compute: {conflicts}"
        )


def numeric(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"Missing/nonfinite numeric metric: {label}")
    return float(value)


def validate_runtime_imports(path: Path, stamp: Mapping) -> None:
    imported = read_json(path)
    if (
        imported.get("device") != "mps"
        or imported.get("mps_fallback") != 0
        or imported.get("stage") != "after_training"
    ):
        raise RuntimeError("Runtime import manifest has wrong private device/fallback")
    sources = imported.get("source_sha256")
    required = {
        "scripts/run_rcig_batch_screen_experiment.py",
        "algorithms/ldp_gradient_far_recent.py",
        "run_experiment.py",
        "algorithms/__init__.py",
        "privacy/rdp.py",
    }
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise RuntimeError("Runtime manifest lacks required actually imported sources")
    if any(
        stamp["source_sha256"].get(name) != digest for name, digest in sources.items()
    ):
        raise RuntimeError(
            "Actual imported dependency is absent/different in source lock"
        )


def validate_metrics(payload: Mapping, expected: Mapping, task: ScreenTask) -> None:
    validate_resolved_config(expected, task)
    summary, actual = payload.get("summary", {}), payload.get("config", {})
    if payload.get("algorithm") != expected["training"]["algorithm"]:
        raise RuntimeError("Metrics algorithm identity mismatch")
    required_summary = {
        "num_rounds": 40,
        "seed": task.seed,
        "partition_seed": task.seed,
        "dataset": "fashionmnist",
        "model": "lenet5_tanh",
        "num_clients": 25,
    }
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise RuntimeError("Metrics summary dataset/model/seed/rounds mismatch")
    if actual.get("device") != "mps":
        raise RuntimeError("Metrics private device is not MPS")
    for key, value in expected["training"]["algo_config"].items():
        if actual.get(key) != value:
            raise RuntimeError(f"Metrics algorithm configuration differs: {key}")
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 40:
        raise RuntimeError("Exactly 40 recorded rounds are required")
    scale_max = 1 if task.noise_regime == "homogeneous" else 2
    for index, row in enumerate(rounds, 1):
        if row.get("round_num") != index:
            raise RuntimeError("Public rounds must be exactly 1 through 40")
        fixed = {
            "num_clients": 25,
            "num_selected": 25,
            "num_survivors": 25,
            "num_alive_clients": 25,
            "num_pre_training_dropouts": 0,
            "participation_rate": 1.0,
            "survival_ratio": 1.0,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "far_attack_labels_visible_to_server_aggregate": False,
            "far_attack_config_visible_to_server_aggregate": False,
            "far_external_attack_diagnostics": True,
            "far_external_attack_diagnostics_boundary": "posthoc_simulator_only",
        }
        if any(row.get(key) != value for key, value in fixed.items()):
            raise RuntimeError(
                f"Round {index}: participation/device/privacy/oracle-boundary contract failed"
            )
        if row.get("ldp_gradient_far_private_compute_device") not in ("mps", "mps:0"):
            raise RuntimeError(f"Round {index}: private CPU/mixed device forbidden")
        for key in OUTCOMES:
            numeric(row.get(key), f"round {index} {key}")
        if not (
            0 <= row["test_accuracy"] <= 1
            and 0 <= row["client_accuracy_mean"] <= 1
            and row["test_loss"] >= 0
            and row["client_accuracy_variance_pct2"] >= 0
            and 0 <= row["worst20_accuracy_pct"] <= 100
            and 0 <= row["best20_worst20_gap_pct"] <= 100
        ):
            raise RuntimeError(
                f"Round {index}: outcomes outside measurement units/ranges"
            )
        for key, value in {
            "privacy_delta": 1e-5,
            "privacy_model_noise_multiplier_min": calibrated_sigma(task.batch_size),
            "privacy_model_noise_multiplier_max": calibrated_sigma(task.batch_size)
            * scale_max,
            "privacy_model_steps_mean": 1.0,
            "privacy_noise_scale_public_min": 1.0,
            "privacy_noise_scale_public_max": float(scale_max),
            "privacy_model_noise_multiplier_mean": calibrated_sigma(task.batch_size)
            * (1.0 if scale_max == 1 else 37.0 / 25.0),
        }.items():
            if not math.isclose(
                numeric(row.get(key), key), value, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise RuntimeError(f"Round {index}: incorrect privacy mechanism {key}")
        if (
            abs(
                numeric(row.get("privacy_epsilon_max"), "epsilon_max")
                - epsilon_at(task.batch_size, index)
            )
            > 1e-4
        ):
            raise RuntimeError(f"Round {index}: recomputed privacy epsilon mismatch")
        epsilon_mean = (
            epsilon_at(task.batch_size, index)
            if scale_max == 1
            else (
                13 * epsilon_at(task.batch_size, index)
                + 12 * epsilon_at(task.batch_size, index, 2)
            )
            / 25
        )
        if (
            abs(numeric(row.get("privacy_epsilon_mean"), "epsilon_mean") - epsilon_mean)
            > 1e-4
        ):
            raise RuntimeError(f"Round {index}: heterogeneous mean epsilon mismatch")
        active = task.scenario != "none" and index >= 17
        if row.get("attack_window_active") != active or row.get(
            "num_byzantine_oracle"
        ) != (5 if active else 0):
            raise RuntimeError(f"Round {index}: attack window/Byzantine count mismatch")
        if active:
            mass = numeric(
                row.get("byzantine_weight_mass_oracle"), "Byzantine weight mass"
            )
            if not 0 <= mass <= 1:
                raise RuntimeError("Byzantine weight mass out of range")
        expected_alpha = (
            0.0
            if task.reference == "uniform"
            or (task.reference == "recent" and index <= 12)
            else 0.1
        )
        if not math.isclose(
            numeric(row.get("ldp_gradient_far_effective_alpha"), "effective alpha"),
            expected_alpha,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Round {index}: wrong deployed alpha/warmup")
        if task.reference == "recent":
            recent = {
                "rcig_reference_mode": "recent_only",
                "rcig_covariance_mode": "full",
                "rcig_deployed_candidate": "identity_new",
                "rcig_temporal_correction_enabled": False,
                "rcig_innovation_test_enabled": False,
                "ldp_gradient_far_reference_noise_aware": False,
                "rcig_server_aggregation_device": "cpu",
                "rcig_private_gradient_mps_fraction": 1.0,
                "rcig_history_ready": index > 12,
            }
            if any(row.get(key) != value for key, value in recent.items()):
                raise RuntimeError(
                    f"Round {index}: deployed recent-only control contract failed"
                )
            if row.get("rcig_server_aggregation_dtype") not in (
                "float64",
                "torch.float64",
            ) or row.get("rcig_private_gradient_compute_device") not in (
                "mps",
                "mps:0",
            ):
                raise RuntimeError(
                    "Recent reference private/server device audit failed"
                )
            if index > 12:
                recent_error = numeric(
                    row.get(
                        "rcig_identity_new_squared_l2_error_to_clean_honest_center_oracle"
                    ),
                    "identity_new reference quality oracle",
                )
                deployed_error = numeric(
                    row.get(
                        "rcig_reference_squared_l2_error_to_clean_honest_center_oracle"
                    ),
                    "deployed reference quality oracle",
                )
                if recent_error < 0 or not math.isclose(
                    recent_error, deployed_error, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        "Deployed reference quality does not match identity_new"
                    )
                names = (
                    "gate_source_round_min",
                    "gate_source_round_max",
                    "older_round_min",
                    "older_round_max",
                    "newer_round_min",
                    "newer_round_max",
                )
                g0, g1, o0, o1, n0, n1 = [
                    numeric(row.get("rcig_" + name), name) for name in names
                ]
                if row.get("rcig_reference_strictly_past") is not True or not (
                    g0 <= g1 < o0 <= o1 < n0 <= n1 < index - 1
                ):
                    raise RuntimeError(
                        "Recent reference used a non-past or overlapping window"
                    )
    if (
        abs(numeric(rounds[-1].get("privacy_epsilon_max"), "final epsilon") - 4.0)
        > 1e-4
    ):
        raise RuntimeError("Final maximum epsilon differs from 4 beyond 1e-4")


def _status_identity(
    campaign: ScreenCampaign, task: ScreenTask, stamp: Mapping, config_path: Path
) -> dict:
    return {
        "campaign_id": CAMPAIGN_ID,
        "run_id": task.run_id,
        "job_index": task.global_index,
        "interpretation": INTERPRETATION,
        "scientific_hash": canonical_hash(stamp),
        "device": "mps",
        "mps_fallback": 0,
        "pairing_block": task.pairing_block,
        "resolved_config_sha256": file_hash(config_path),
    }


def process_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeError("Permission prevents verifying launcher liveness") from exc
    return True


def read_record(
    campaign: ScreenCampaign,
    task: ScreenTask,
    stamp: Mapping,
    *,
    allow_running: bool = False,
) -> dict:
    directory = task_output_dir(campaign, task)
    if not directory.exists() or not any(directory.iterdir()):
        return {"run_id": task.run_id, "status": "missing"}
    status_path, config_path = (
        directory / "orchestration_status.json",
        directory / "resolved_config.yaml",
    )
    if not status_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Partial run, never automatically restarted: {task.run_id}")
    status = read_json(status_path)
    expected = resolved_config(campaign, task, stamp)
    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError(f"Changed resolved config: {task.run_id}")
    if any(
        status.get(key) != value
        for key, value in _status_identity(campaign, task, stamp, config_path).items()
    ):
        raise RuntimeError(f"Invalid run status identity/hash: {task.run_id}")
    if (
        status.get("status") == "running"
        and allow_running
        and process_alive(status.get("launcher_pid"))
    ):
        return {
            "run_id": task.run_id,
            "status": "running",
            "launcher_pid": status["launcher_pid"],
            "child_pid": status.get("child_pid"),
            "log_path": str(directory / "run.log"),
        }
    if status.get("status") != "completed":
        raise RuntimeError(
            f"Partial/failed/stale run ({status.get('status')}), never restarted: {task.run_id}"
        )
    metrics = sorted(directory.glob("**/metrics.json"))
    if len(metrics) != 1:
        raise RuntimeError(f"Missing/ambiguous metrics: {task.run_id}")
    runtime_path = directory / "runtime_imports.json"
    if (
        status.get("metrics_sha256") != file_hash(metrics[0])
        or not runtime_path.is_file()
        or status.get("runtime_imports_sha256") != file_hash(runtime_path)
    ):
        raise RuntimeError(f"Metrics/runtime manifest hash mismatch: {task.run_id}")
    validate_runtime_imports(runtime_path, stamp)
    validate_metrics(read_json(metrics[0]), expected, task)
    return {
        "run_id": task.run_id,
        "status": "complete",
        "metrics_path": str(metrics[0]),
    }


def status_report(campaign: ScreenCampaign, stamp: Mapping) -> dict:
    counts = {"valid_complete": 0, "missing": 0}
    active, invalid = [], []
    for task in campaign.tasks:
        try:
            record = read_record(campaign, task, stamp, allow_running=True)
            if record["status"] == "running":
                active.append(record)
            else:
                counts[
                    "valid_complete" if record["status"] == "complete" else "missing"
                ] += 1
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"run_id": task.run_id, "reason": str(exc)})
    return {
        "campaign_id": CAMPAIGN_ID,
        "read_only": True,
        "expected": 216,
        "interpretation": INTERPRETATION,
        "output_root": str(campaign.output_root),
        **counts,
        "active": active,
        "invalid": invalid,
    }


def run_task(campaign: ScreenCampaign, task: ScreenTask, stamp: Mapping) -> dict:
    if provenance(campaign) != stamp:
        raise RuntimeError("Source drift before run; no subprocess launched")
    verify_lock(campaign, stamp)
    existing = read_record(campaign, task, stamp)
    if existing["status"] == "complete":
        print(f"SKIP valid {task.run_id}", flush=True)
        return existing
    directory = task_output_dir(campaign, task)
    directory.mkdir(parents=True, exist_ok=True)
    config = resolved_config(campaign, task, stamp)
    validate_resolved_config(config, task)
    config_path = directory / "resolved_config.yaml"
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    status_path = directory / "orchestration_status.json"
    status = {
        **_status_identity(campaign, task, stamp, config_path),
        "status": "running",
        "launcher_pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(status_path, status, exclusive=True)
    environment = dict(
        os.environ,
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        PYTHONHASHSEED=str(task.seed),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    command = [
        sys.executable,
        "-B",
        str(ENTRYPOINT),
        "--config",
        str(config_path),
        "--device",
        "mps",
        "--output",
        str(directory),
    ]
    print(f"START {task.global_index + 1}/216 {task.run_id}", flush=True)
    try:
        with (directory / "run.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            status["child_pid"] = child.pid
            write_json(status_path, status)
            returncode = child.wait()
        if returncode:
            raise RuntimeError(
                f"Training child exited {returncode}; inspect {directory / 'run.log'}"
            )
        if provenance(campaign) != stamp:
            raise RuntimeError("Source drift during run")
        verify_lock(campaign, stamp)
        metrics = sorted(directory.glob("**/metrics.json"))
        if len(metrics) != 1:
            raise RuntimeError(
                "Training did not produce one unambiguous metrics artifact"
            )
        validate_runtime_imports(directory / "runtime_imports.json", stamp)
        validate_metrics(read_json(metrics[0]), config, task)
        status.update(
            status="completed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            metrics_sha256=file_hash(metrics[0]),
            runtime_imports_sha256=file_hash(directory / "runtime_imports.json"),
        )
        write_json(status_path, status)
    except BaseException as exc:
        status.update(
            status="failed",
            error=str(exc),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(status_path, status)
        raise
    return read_record(campaign, task, stamp)


def selected_tasks(
    campaign: ScreenCampaign, job_index: int | None, max_jobs: int | None
) -> tuple[ScreenTask, ...]:
    if job_index is not None and not 0 <= job_index < len(campaign.tasks):
        raise ValueError("--job-index is zero-based and must be in [0,215]")
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("--max-jobs must be positive")
    start = 0 if job_index is None else job_index
    count = (
        max_jobs
        if max_jobs is not None
        else (1 if job_index is not None else len(campaign.tasks))
    )
    return campaign.tasks[start : start + count]


def detach(campaign: ScreenCampaign, stamp: Mapping, args) -> dict:
    # Reserve/verify provenance first, then release before the child takes its
    # own global execution lock. Concurrent dispatches cannot train concurrently.
    with execution_lock(campaign):
        ensure_lock(campaign, stamp)
        check_no_other_training_process()
    launches = campaign.output_root / "_launches"
    launches.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    log_path = launches / f"{token}.log"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--run",
        "--resume",
        "--launch-token",
        token,
    ]
    if args.job_index is not None:
        command.extend(["--job-index", str(args.job_index)])
    if args.max_jobs is not None:
        command.extend(["--max-jobs", str(args.max_jobs)])
    environment = dict(
        os.environ,
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    with log_path.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    dispatch = {
        "campaign_id": CAMPAIGN_ID,
        "launcher_pid": child.pid,
        "launch_token": token,
        "log_path": str(log_path),
        "status": "starting",
    }
    write_json(launches / f"{token}.json", dispatch, exclusive=True)
    acknowledgment = launches / f"{token}.started.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if acknowledgment.is_file():
            started = read_json(acknowledgment)
            if (
                started.get("launcher_pid") != child.pid
                or started.get("launch_token") != token
                or started.get("scientific_hash") != canonical_hash(stamp)
            ):
                raise RuntimeError("Detached child startup acknowledgment is invalid")
            if child.poll() is not None:
                raise RuntimeError(
                    f"Detached child exited after startup; inspect {log_path}"
                )
            return {**dispatch, "status": "started", "startup_acknowledged": True}
        if child.poll() is not None:
            raise RuntimeError(
                f"Detached child failed before startup (exit {child.returncode}); inspect {log_path}"
            )
        time.sleep(0.1)
    return {
        **dispatch,
        "status": "startup_not_yet_confirmed",
        "startup_acknowledged": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--device", choices=["mps"], default="mps")
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--launch-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run and not args.resume:
        parser.error("Execution requires --run --resume")
    if (args.detach or args.launch_token) and not args.run:
        parser.error("Detached execution requires --run --resume")
    if args.launch_token and (
        args.detach
        or len(args.launch_token) != 32
        or any(c not in "0123456789abcdef" for c in args.launch_token)
    ):
        parser.error("Invalid internal detached-launch token")
    campaign = load_campaign()
    tasks = selected_tasks(campaign, args.job_index, args.max_jobs)
    if not args.run and not args.status:
        print(
            json.dumps(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "read_only": True,
                    "interpretation": INTERPRETATION,
                    "expected": 216,
                    "selected": len(tasks),
                    "seeds": list(SEEDS),
                    "output_root": str(campaign.output_root),
                    "by_batch": dict(Counter(t.batch_size for t in campaign.tasks)),
                    "by_reference": dict(Counter(t.reference for t in campaign.tasks)),
                    "privacy_table": privacy_table(),
                    "tasks": [
                        dict(vars(t), run_id=t.run_id, pairing_block=t.pairing_block)
                        for t in tasks
                    ],
                },
                indent=2,
            )
        )
        return 0
    stamp = provenance(campaign)
    if (campaign.output_root / LOCK_NAME).exists():
        verify_lock(campaign, stamp)
    elif campaign.output_root.exists() and any(campaign.output_root.iterdir()):
        raise RuntimeError("Screen artifacts exist without a provenance lock")
    if args.status:
        print(json.dumps(status_report(campaign, stamp), indent=2))
        return 0
    if args.detach:
        print(json.dumps(detach(campaign, stamp, args), indent=2))
        return 0
    with execution_lock(campaign):
        check_no_other_training_process()
        require_working_mps()
        ensure_lock(campaign, stamp)
        # All prior runs are checked, even when only a subset is requested.
        for task in campaign.tasks:
            read_record(campaign, task, stamp)
        if provenance(campaign) != stamp:
            raise RuntimeError("Source changed during preflight")
        if args.launch_token:
            write_json(
                campaign.output_root
                / "_launches"
                / f"{args.launch_token}.started.json",
                {
                    "launcher_pid": os.getpid(),
                    "launch_token": args.launch_token,
                    "scientific_hash": canonical_hash(stamp),
                    "status": "started",
                },
                exclusive=True,
            )
        for task in tasks:
            run_task(campaign, task, stamp)
        report = status_report(campaign, stamp)
        write_json(
            campaign.output_root / "_status.json",
            {
                **report,
                "read_only": False,
                "status": (
                    "completed"
                    if report["valid_complete"] == 216
                    else "selected_subset_completed"
                ),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
