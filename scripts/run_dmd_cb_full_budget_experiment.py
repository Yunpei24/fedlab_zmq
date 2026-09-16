"""Explicit registration boundary for CE-full-budget and immutable private DMD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=["mps"])
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS fallback forbidden")
    import torch
    import yaml

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; never fall back to CPU")
    import algorithms.ldp_gradient_ce_full_budget
    import algorithms.ldp_gradient_dmd_cb
    import run_experiment as harness
    from scripts.run_dmd_cb_private_experiment import source_hashes

    config = yaml.safe_load(args.config.read_text())
    name = config["training"]["algorithm"]
    if name not in {"ldp_gradient_ce_full_budget", "ldp_gradient_dmd_cb"}:
        raise ValueError("Unsupported algorithm")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "runtime_imports.json"
    before = source_hashes()
    payload = {
        "source_sha256": before,
        "device": "mps",
        "mps_fallback": 0,
        "algorithm": name,
        "stage": "before_training",
    }
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
    old_argv = sys.argv
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
        harness.main()
    finally:
        sys.argv = old_argv
    from scripts.run_rcig_batch_screen import file_hash

    for relative, digest in before.items():
        if file_hash(ROOT / relative) != digest:
            raise RuntimeError(f"Source changed during training: {relative}")
    payload.update(stage="after_training", source_sha256={**before, **source_hashes()})
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2)


if __name__ == "__main__":
    main()
