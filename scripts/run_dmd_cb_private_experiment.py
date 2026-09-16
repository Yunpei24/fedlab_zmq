#!/usr/bin/env python3
"""Register the new DMD private algorithm without editing active RCIG sources."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_hashes():
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
        if not path.is_file() or path.suffix != ".py" or not path.is_relative_to(ROOT):
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in {"venv", ".venv", "env", ".env"} and "site-packages" not in rel.parts:
            paths.add(path)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["mps"], required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS fallback must be disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import yaml
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: no CPU training is authorized")
    config = yaml.safe_load(args.config.read_text())
    if config["training"]["algorithm"] != "ldp_gradient_dmd_cb":
        raise ValueError("This entrypoint only accepts ldp_gradient_dmd_cb")
    import algorithms.ldp_gradient_dmd_cb  # explicit registration, isolated new file
    import run_experiment as harness

    args.output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    runtime_path = args.output / "runtime_imports.json"
    payload = {"source_sha256": before, "device": "mps", "mps_fallback": 0,
               "algorithm": "ldp_gradient_dmd_cb", "stage": "before_training"}
    with runtime_path.open("x") as handle:
        json.dump(payload, handle, indent=2)
    old_argv = sys.argv
    sys.argv = [str(ROOT / "run_experiment.py"), "--config", str(args.config),
                "--device", "mps", "--output", str(args.output)]
    try:
        harness.main()
    finally:
        sys.argv = old_argv
    for rel, digest in before.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed while training: {rel}")
    payload.update(source_sha256={**before, **source_hashes()}, stage="after_training")
    with runtime_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
