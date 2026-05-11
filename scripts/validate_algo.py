#!/usr/bin/env python3
"""
scripts/validate_algo.py
========================
Validates that a new algorithm correctly implements the FLAlgorithm interface.

Usage:
  python scripts/validate_algo.py --algo eceffl
  python scripts/validate_algo.py --algo my_new_algo
"""

import sys, argparse, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

REQUIRED_METADATA_KEYS = [
    "client_id", "round_num", "beta_actual",
    "battery_j_remaining", "energy_j_consumed",
    "bytes_sent", "bytes_received", "local_loss", "compression_ratio",
]

def validate(algo_name: str):
    results = []
    ok = True

    # Import all algorithms
    import algorithms.fedavg, algorithms.eceffl  # noqa
    try:
        mod = __import__(f"algorithms.{algo_name}", fromlist=[algo_name])
    except ImportError as e:
        print(f"FAIL  Import error: {e}"); return False

    from algorithms.base import get_algorithm, FLAlgorithm, ClientState
    try:
        algo = get_algorithm(algo_name)
    except ValueError as e:
        print(f"FAIL  @register_algorithm decorator missing: {e}"); return False
    results.append(f"PASS  Algorithm '{algo_name}' registered")

    # Check abstract methods
    for method in ["client_update", "server_aggregate", "get_default_config"]:
        if not hasattr(algo, method):
            print(f"FAIL  Missing method: {method}"); ok = False
        else:
            results.append(f"PASS  Method '{method}' present")

    # Check default config
    cfg = algo.get_default_config()
    for key in ["lr", "local_epochs", "batch_size", "device"]:
        if key not in cfg:
            print(f"FAIL  Default config missing key: '{key}'"); ok = False
        else:
            results.append(f"PASS  Config key '{key}' = {cfg[key]}")

    # Smoke test with a tiny model and dataset
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn as nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    X = torch.randn(20, 1, 2, 2)
    Y = torch.randint(0, 2, (20,))
    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=4)

    state = ClientState(client_id=0, battery_j=100000.0)
    config = {**cfg, "device": "cpu", "battery_max_j": 185400.0}

    try:
        update, metadata = algo.client_update(model, dl, state, config)
        results.append("PASS  client_update() returned without error")
    except Exception as e:
        print(f"FAIL  client_update() raised: {e}"); ok = False
        for r in results: print(r)
        return False

    # Check metadata keys
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            print(f"FAIL  Metadata missing key: '{key}'"); ok = False
        else:
            results.append(f"PASS  Metadata key '{key}' = {metadata[key]}")

    # Check battery non-negative
    if state.battery_j < 0:
        print(f"FAIL  battery_j went negative: {state.battery_j}"); ok = False
    else:
        results.append(f"PASS  Battery non-negative: {state.battery_j:.1f}J")

    # Check round_num incremented
    if state.round_num != 1:
        print(f"FAIL  round_num not incremented (got {state.round_num})"); ok = False
    else:
        results.append("PASS  round_num incremented to 1")

    # Print all results
    for r in results: print(r)

    if ok:
        print(f"\n✓ Algorithm '{algo_name}' passed all checks.")
    else:
        print(f"\n✗ Algorithm '{algo_name}' has issues. Fix the above FAILs.")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, help="Algorithm name to validate")
    args = p.parse_args()
    sys.exit(0 if validate(args.algo) else 1)
