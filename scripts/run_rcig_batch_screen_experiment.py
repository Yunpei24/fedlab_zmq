#!/usr/bin/env python3
"""Isolated MPS entrypoint for the independent RCIG batch screen.

The parent training harness and algorithm registry imports are hash-locked.
This entrypoint registers the new recent-only control explicitly and adapts
only its evaluation boundary: clean simulator gradients remain outside the
server call, while its genuinely deployed recent reference can be evaluated.
The original RCIG/RFA paths are otherwise delegated unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_source_hashes() -> dict[str, str]:
    """Hash actual imported project sources, excluding installed environments."""

    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, (str, os.PathLike)):
            continue
        path = Path(raw).resolve()
        if path.suffix == ".pyc":
            try:
                path = Path(importlib.util.source_from_cache(str(path))).resolve()
            except ValueError:
                continue
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts[0] in {"venv", ".venv", "env", ".env"}:
            continue
        if "site-packages" in relative.parts:
            continue
        paths.add(path)
    return {
        path.relative_to(ROOT).as_posix(): _source_hash(path) for path in sorted(paths)
    }


def write_runtime_manifest(
    output: Path,
    *,
    algorithm: str,
    stage: str,
    previous: dict[str, str] | None = None,
) -> dict[str, str]:
    current = runtime_source_hashes()
    if previous is not None:
        changed = [
            name
            for name, digest in previous.items()
            if _source_hash(ROOT / name) != digest
        ]
        if changed:
            raise RuntimeError("runtime project sources changed: " + ", ".join(changed))
        current = {**previous, **current}
    payload = {
        "source_sha256": current,
        "device": "mps",
        "mps_fallback": 0,
        "algorithm": algorithm,
        "stage": stage,
        "private_gradients_device": "mps",
        "server_postprocessing_device": "cpu_float64_for_rcig_temporal_paths",
    }
    path = output / "runtime_imports.json"
    # A process cannot silently adopt an existing run's execution manifest.
    mode = "x" if previous is None else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return current


def recent_reference_oracle_metrics(payload, clean_by_client, client_updates):
    """Evaluate only the deployed recent candidate, never fabricate RCIG views."""

    if payload is None:
        return {}
    if clean_by_client is None:
        raise ValueError("recent evaluation requires detached clean oracles")
    expected_fields = {
        "candidate_references",
        "deployed_reference",
        "server_clip_norm",
        "deployment_round",
        "contains_clean_data",
        "contains_realised_noise",
        "contains_attack_labels",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("recent server payload has unexpected fields")
    if any(
        payload[key] is not False
        for key in (
            "contains_clean_data",
            "contains_realised_noise",
            "contains_attack_labels",
        )
    ):
        raise ValueError("recent server payload contains forbidden oracle information")
    candidates = payload["candidate_references"]
    if not isinstance(candidates, dict) or set(candidates) != {"identity_new"}:
        raise ValueError("recent evaluation accepts only the deployed identity_new")

    import torch
    from robustness.tensor_ops import stack_updates

    ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
    if len(ids) != len(set(ids)) or set(ids) != set(clean_by_client):
        raise ValueError("recent evaluation requires one clean oracle per client")
    honest_ids = [
        int(metadata["client_id"])
        for _, metadata, _ in client_updates
        if not bool(metadata.get("is_byzantine", False))
    ]
    if not honest_ids:
        raise ValueError("recent evaluation requires at least one honest client")
    clean_vectors, _ = stack_updates([clean_by_client[cid] for cid in honest_ids])
    clean_vectors = clean_vectors.to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(clean_vectors).all()):
        raise ValueError("recent evaluation clean oracle is not finite")
    radius = float(payload["server_clip_norm"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("recent evaluation server radius must be positive and finite")
    factors = (
        radius / torch.linalg.vector_norm(clean_vectors, dim=1).clamp_min(1e-12)
    ).clamp(max=1.0)
    target = (clean_vectors * factors[:, None]).mean(dim=0)
    recent = torch.as_tensor(
        candidates["identity_new"], dtype=torch.float64, device="cpu"
    )
    deployed = torch.as_tensor(
        payload["deployed_reference"], dtype=torch.float64, device="cpu"
    )
    if recent.shape != target.shape or deployed.shape != target.shape:
        raise ValueError("recent reference/clean-center dimension mismatch")
    if not bool(torch.isfinite(recent).all()) or not bool(
        torch.isfinite(deployed).all()
    ):
        raise ValueError("recent reference is not finite")
    if not torch.equal(recent, deployed):
        raise ValueError("identity_new must be the actually deployed recent reference")
    squared_error = float((deployed - target).square().sum().item())
    metrics = {
        "rcig_oracle_evaluation_boundary": "offline_simulator_only",
        "rcig_oracle_was_visible_to_server_aggregate": False,
        "rcig_oracle_metric_is_squared_l2": True,
        "rcig_recent_control_oracle_evaluator": True,
    }
    for prefix in ("rcig_identity_new", "rcig_reference"):
        metrics[f"{prefix}_squared_l2_error_to_clean_honest_center_oracle"] = (
            squared_error
        )
        metrics[f"{prefix}_l2_error_to_clean_honest_center_oracle"] = math.sqrt(
            squared_error
        )
    return metrics


@contextmanager
def recent_evaluation_boundary(harness, *, enabled: bool):
    """Adapt the external harness, not inputs consumed by server_aggregate.

    The old harness recognises RCIG by the original algorithm name only. The
    recent arm has its own name, so the wrapper enables the same complete
    clean-oracle detachment here. Attacker identities and realised noise are
    stripped before the aggregation call, just as for the original RCIG arm.
    """

    original_detach = harness.detach_rcig_evaluation_oracles
    original_evaluate = harness.rcig_reference_oracle_metrics

    def detach(updates, *, enabled=False, strip_attack_oracles=False):
        if not strip_attack_oracles:
            raise ValueError("recent control requires attack-oracle stripping")
        return original_detach(
            updates,
            enabled=collect_oracles,
            strip_attack_oracles=True,
        )

    collect_oracles = bool(enabled)
    harness.detach_rcig_evaluation_oracles = detach
    harness.rcig_reference_oracle_metrics = recent_reference_oracle_metrics
    try:
        yield
    finally:
        harness.detach_rcig_evaluation_oracles = original_detach
        harness.rcig_reference_oracle_metrics = original_evaluate


def validate_entry_config(config: dict, *, device: str) -> str:
    if device != "mps" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("batch-screen private execution requires MPS with fallback=0")
    algorithm = config.get("training", {}).get("algorithm")
    if algorithm not in {"ldp_gradient_far", "ldp_gradient_far_recent"}:
        raise ValueError("unsupported batch-screen algorithm")
    algorithm_config = config["training"].get("algo_config", {})
    if not bool(algorithm_config.get("external_attack_diagnostics", False)):
        raise ValueError("batch screen requires external attack diagnostics")
    if algorithm == "ldp_gradient_far_recent":
        if not bool(algorithm_config.get("rcig_oracle_separation_required", False)):
            raise ValueError("recent screen requires strict oracle separation")
        if not bool(algorithm_config.get("enable_oracle_diagnostics", False)):
            raise ValueError("recent screen requires external clean-oracle evaluation")
    return algorithm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["mps"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    algorithm = validate_entry_config(config, device=args.device)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU fallback is permitted")
    # Explicit registration avoids any mutation of algorithms/__init__.py.
    import algorithms.ldp_gradient_far_recent
    import run_experiment as harness

    args.output.mkdir(parents=True, exist_ok=True)
    before = write_runtime_manifest(
        args.output, algorithm=algorithm, stage="before_training"
    )
    original_argv = sys.argv
    sys.argv = [
        str(ROOT / "run_experiment.py"),
        "--config",
        str(args.config),
        "--device",
        "mps",
        "--output",
        str(args.output),
    ]
    try:
        if algorithm == "ldp_gradient_far_recent":
            with recent_evaluation_boundary(harness, enabled=True):
                harness.main()
        else:
            harness.main()
    finally:
        sys.argv = original_argv
    write_runtime_manifest(
        args.output, algorithm=algorithm, stage="after_training", previous=before
    )


if __name__ == "__main__":
    main()
