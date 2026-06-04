"""
scripts/validate_flop_model.py
==============================
Compare the analytic FLOP model used by FedPart/FedPartBE against measured
FLOPs obtained from torch.utils.flop_counter.FlopCounterMode.

For ResNet-8 on 3x32x32 inputs with batch=32, for each layer group g:
    analytic[g] = full_flops_analytic * (1/3 + 2/3 * phi_g)
                  where phi_g = group_flops[g] / sum(group_flops)
    measured[g] = fwd_bwd_per_group[g] * steps
plus a "FedAvg full" row for reference.

Outputs:
    1. A human-readable table on stdout.
    2. A CSV at data_results/flop_model_validation.csv (parent created if missing).
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_fedpart_helpers():
    """Load algorithms/fedpart helpers without triggering algorithms/__init__.py.

    The __init__ imports every algorithm including fedpart_be_lora_gs.py, which
    in turn imports a now-removed symbol — completely unrelated to this script.
    Bypass it by hand-loading just base + fedpart.
    """
    pkg = types.ModuleType("algorithms")
    pkg.__path__ = [str(REPO_ROOT / "algorithms")]
    sys.modules.setdefault("algorithms", pkg)
    base_spec = importlib.util.spec_from_file_location(
        "algorithms.base", REPO_ROOT / "algorithms/base.py"
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    sys.modules["algorithms.base"] = base_mod
    base_spec.loader.exec_module(base_mod)
    fp_spec = importlib.util.spec_from_file_location(
        "algorithms.fedpart", REPO_ROOT / "algorithms/fedpart.py"
    )
    fp_mod = importlib.util.module_from_spec(fp_spec)
    sys.modules["algorithms.fedpart"] = fp_mod
    fp_spec.loader.exec_module(fp_mod)
    return fp_mod


def _group_label(group_keys: list[str], group_key_fn) -> str:
    if not group_keys:
        return "?"
    return group_key_fn(group_keys[0])


def main() -> None:
    fp_mod = _load_fedpart_helpers()
    from models.registry import get_model
    from hardware.profiles import DEVICE_PROFILES
    from hardware.flop_measure import (
        measure_fwd_bwd_full_flops,
        measure_fwd_bwd_per_group_flops,
        measure_fwd_flops,
    )

    # Setup
    model = get_model("resnet8", "cifar10").eval()
    input_shape = (1, 3, 32, 32)
    batch_size = 32
    local_epochs = 1
    dataset_size = 320  # 10 mini-batches
    profile = DEVICE_PROFILES["raspberry_pi_4"]

    # Steps formula (same convention as everywhere else in the repo)
    steps = (dataset_size // batch_size) * local_epochs

    # ── Analytic ────────────────────────────────────────────────────────────
    num_params = sum(p.numel() for p in model.parameters())
    full_flops_analytic = profile.flops_for_model(
        num_params, batch_size, local_epochs, dataset_size
    )
    groups = fp_mod._derive_layer_groups(model)
    gf = fp_mod._compute_group_flops(groups, model, input_shape=input_shape)
    total_gf = sum(gf)

    # ── Measured ────────────────────────────────────────────────────────────
    fwd_per_step = measure_fwd_flops(model, input_shape, batch_size)
    fwd_bwd_full_per_step = measure_fwd_bwd_full_flops(model, input_shape, batch_size)
    fwd_bwd_per_group_per_step = measure_fwd_bwd_per_group_flops(
        model, groups, input_shape, batch_size
    )

    # FedAvg full row
    measured_full = fwd_bwd_full_per_step * steps
    analytic_full = float(full_flops_analytic)
    full_rel_err = (measured_full - analytic_full) / max(analytic_full, 1.0)

    # Per-group rows
    rows: list[dict] = []
    rows.append({
        "label":           "FedAvg full (no freeze)",
        "group_idx":       -1,
        "phi":             1.0,
        "analytic_flops":  analytic_full,
        "measured_flops":  measured_full,
        "rel_err":         full_rel_err,
        "ratio_measured_over_analytic": measured_full / max(analytic_full, 1.0),
    })

    for g_idx, g_keys in enumerate(groups):
        phi = gf[g_idx] / max(total_gf, 1)
        ana = full_flops_analytic * (1.0 / 3.0 + 2.0 / 3.0 * phi)
        meas = fwd_bwd_per_group_per_step[g_idx] * steps
        rel = (meas - ana) / max(ana, 1.0)
        rows.append({
            "label":           _group_label(g_keys, fp_mod._param_group_key),
            "group_idx":       g_idx,
            "phi":             phi,
            "analytic_flops":  ana,
            "measured_flops":  meas,
            "rel_err":         rel,
            "ratio_measured_over_analytic": meas / max(ana, 1.0),
        })

    # ── Pretty print ────────────────────────────────────────────────────────
    print("=" * 100)
    print("FLOP model validation — ResNet-8 (3×32×32, batch=32, steps={})".format(steps))
    print("=" * 100)
    print(
        f"num_params={num_params}  full_flops_analytic={full_flops_analytic:.3e}  "
        f"sum(group_flops)={total_gf:.3e}"
    )
    print(
        f"per-step measured: fwd={fwd_per_step:.3e}  fwd_bwd_full={fwd_bwd_full_per_step:.3e}"
    )
    print()
    header = f"{'g':>3} {'label':<26} {'phi':>8} {'analytic':>14} {'measured':>14} {'meas/ana':>10} {'rel_err':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['group_idx']:>3} {r['label'][:25]:<26} {r['phi']:>8.4f} "
            f"{r['analytic_flops']:>14.3e} {r['measured_flops']:>14.3e} "
            f"{r['ratio_measured_over_analytic']:>10.2f} {r['rel_err']*100:>9.1f}%"
        )
    print()
    print("Interpretation:")
    print("  meas/ana >> 1  -> analytic under-counts in absolute terms (expected for CNNs).")
    print("  When ranking groups by 'measured', shallow groups should be the most expensive.")
    print("  When ranking by 'analytic', the same shallow groups appear cheap — this is the")
    print("  flaw that the 1/3 + 2/3*phi formula introduces, and the reason FedPartBE adds a")
    print("  corrected-cost formula on top of it for tier assignment.")
    print()
    print("WARNING: measured FLOPs are ~2 orders of magnitude > analytic. The energy_scale_factor")
    print("         (e.g. 12.6) was calibrated against the analytic estimator and will need a")
    print("         separate recalibration when use_measured_flops=True is enabled.")

    # ── CSV output ──────────────────────────────────────────────────────────
    csv_dir = REPO_ROOT / "data_results"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "flop_model_validation.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print()
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
