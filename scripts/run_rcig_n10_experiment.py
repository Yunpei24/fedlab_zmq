#!/usr/bin/env python3
"""MPS-only, isolated entrypoint for the N10 RCIG policy ablation."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


@contextmanager
def evaluation_rng_isolation(harness):
    """Ensure evaluation cannot consume training/model random streams."""
    import random
    import numpy as np
    import torch

    originals = {
        name: getattr(harness, name)
        for name in ("evaluate_global_model", "evaluate_client_loaders")
    }

    def wrap(function):
        def isolated(*args, **kwargs):
            cpu, mps = torch.get_rng_state(), torch.mps.get_rng_state()
            py, np_state = random.getstate(), np.random.get_state()
            try:
                return function(*args, **kwargs)
            finally:
                torch.set_rng_state(cpu)
                torch.mps.set_rng_state(mps)
                random.setstate(py)
                np.random.set_state(np_state)

        return isolated

    for name, function in originals.items():
        setattr(harness, name, wrap(function))
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(harness, name, function)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["mps"], required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("MPS CPU fallback forbidden")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    import yaml

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no training performed")
    cfg = yaml.safe_load(args.config.read_text())
    algo = cfg["training"]["algo_config"]
    if cfg["clients"]["num_clients"] != 10 or not algo.get(
        "external_attack_diagnostics"
    ):
        raise RuntimeError("N10/oracle boundary contract mismatch")
    from algorithms.base import register_algorithm
    from algorithms.rcig_n10_ablation import N10PolicyAblation
    import run_experiment as harness
    from scripts.run_rcig_batch_screen_experiment import write_runtime_manifest

    # Same harness ID preserves the audited oracle-detachment boundary. The
    # actual implementation is declared explicitly in config/runtime manifest.
    register_algorithm("ldp_gradient_far")(N10PolicyAblation)
    args.output.mkdir(parents=True, exist_ok=True)
    before = write_runtime_manifest(
        args.output, algorithm="rcig_n10_policy_ablation", stage="before_training"
    )
    trace = args.output / "simulator_randomness_private_audit.jsonl"
    with trace.open("x") as stream:

        def sink(row):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

        N10PolicyAblation.audit_sink = staticmethod(sink)
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
            with evaluation_rng_isolation(harness):
                harness.main()
        finally:
            N10PolicyAblation.audit_sink = None
            sys.argv = original_argv
    write_runtime_manifest(
        args.output,
        algorithm="rcig_n10_policy_ablation",
        stage="after_training",
        previous=before,
    )


if __name__ == "__main__":
    main()
