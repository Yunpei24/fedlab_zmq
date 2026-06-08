#!/usr/bin/env python3
"""
scripts/bench_fed_resonance_svd.py
==================================
Benchmark + equivalence check for the FedResonance single-pass SVD
optimization (algorithms/fed_resonance.py, svd_fast).

It compares, on ResNet-8 layer shapes:
  - OLD path: adaptive_rank() [svdvals, full spectrum] + svd_compress() [rSVD]
  - NEW path: compress_low_rank() [ONE decomposition]

and reports, per layer-size class (small vs large):
  - SVD time per layer (old vs new) and the speedup,
  - reconstruction equivalence (Frobenius relative error of NEW vs OLD),
  - rank agreement (|r_new - r_old|),
plus the total client_update() time with svd_fast on vs off, and a short
in-process smoke confirming eval accuracy is unchanged.

Usage:
    python scripts/bench_fed_resonance_svd.py [--reps 20] [--device cpu]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import algorithms.fed_resonance as fr  # noqa: E402
from algorithms.base import ClientState  # noqa: E402
from models.registry import get_model  # noqa: E402

EPS = 0.95  # spectral_energy_thresh
RANK_MIN, RANK_MAX = 4, 32
SIZE_THRESHOLD = 256


def _timeit(fn, reps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _old(G):
    r = fr.adaptive_rank(
        G, EPS, True, 10, 2, rank_criterion="energy", rank_min=RANK_MIN
    )
    r = max(RANK_MIN, min(RANK_MAX, r))
    U, S, Vt = fr.svd_compress(G, r, True, 10, 2)
    return fr.svd_decompress(U, S, Vt), r


def _new(G, power_iter=1):
    U, S, Vt, r = fr.compress_low_rank(
        G,
        EPS,
        RANK_MIN,
        RANK_MAX,
        size_threshold=SIZE_THRESHOLD,
        rsvd_oversample=10,
        rsvd_power_iter=power_iter,
    )
    return fr.svd_decompress(U, S, Vt), r


def _make_grad(m, n, true_rank=8, device="cpu"):
    """A decaying-spectrum matrix resembling a layer gradient."""
    g = torch.manual_seed
    g(0)
    A = torch.randn(m, true_rank, device=device) @ torch.randn(
        true_rank, n, device=device
    )
    return (A + 0.01 * torch.randn(m, n, device=device)).float()


def micro_bench(reps: int, device: str):
    print("\n=== Micro-bench: per-layer SVD (old double-pass vs new single-pass) ===")
    model = get_model("resnet8", "cifar10")
    # Collect the matrix shapes FedResonance would actually SVD (>=2-D weights).
    shapes = []
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            G = fr._matrix_form(p.detach())
            shapes.append((name, tuple(G.shape)))
    # De-duplicate by shape, keep one example.
    seen, uniq = set(), []
    for name, s in shapes:
        if s not in seen:
            seen.add(s)
            uniq.append((name, s))
    # Add a synthetic LARGE shape to exercise the rSVD (large) path too.
    uniq.append(("synthetic_large", (1024, 1024)))

    hdr = (
        f"{'layer/shape':>22} {'class':>6} {'r_old':>5} {'r_new':>5} "
        f"{'relF':>9} {'t_old_ms':>9} {'t_new_ms':>9} {'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    all_ok = True
    for name, (m, n) in uniq:
        G = _make_grad(m, n, device=device)
        rec_o, r_o = _old(G)
        rec_n, r_n = _new(G)
        relf = (rec_n - rec_o).norm().item() / max(rec_o.norm().item(), 1e-12)
        klass = "small" if min(m, n) <= SIZE_THRESHOLD else "large"
        t_o = _timeit(lambda: _old(G), reps)
        t_n = _timeit(lambda: _new(G), reps)
        ok = (abs(r_o - r_n) <= 2) and (relf < 1e-3)
        all_ok &= ok
        rows.append((klass, t_o, t_n))
        print(
            f"{name[:14] + ' ' + str((m, n)):>22} {klass:>6} {r_o:>5} {r_n:>5} "
            f"{relf:>9.1e} {t_o*1e3:>9.3f} {t_n*1e3:>9.3f} "
            f"{t_o/max(t_n,1e-12):>7.2f}x" + ("" if ok else "  <-- FAIL")
        )

    print("\n--- speedup by layer-size class ---")
    for klass in ("small", "large"):
        sel = [(o, n) for (k, o, n) in rows if k == klass]
        if sel:
            to = sum(o for o, _ in sel)
            tn = sum(n for _, n in sel)
            print(
                f"  {klass:>6}: old {to*1e3:7.2f} ms  new {tn*1e3:7.2f} ms  "
                f"-> {to/max(tn,1e-12):.2f}x  ({len(sel)} shapes)"
            )
    print(f"\nEQUIVALENCE (rank within +-2 AND Frobenius rel-err < 1e-3): {all_ok}")
    return all_ok


def macro_bench(reps: int, device: str):
    print("\n=== Macro-bench: full client_update (svd_fast on vs off) ===")
    algo = fr.FedResonance()
    cfg_base = algo.get_default_config()
    # Synthetic CIFAR-like loader (deterministic).
    torch.manual_seed(0)
    X = torch.randn(64, 3, 32, 32)
    Y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=32)

    def run_once(svd_fast):
        torch.manual_seed(0)
        m = get_model("resnet8", "cifar10").to(device)
        st = ClientState(client_id=0, battery_j=1e9)
        cfg = {
            **cfg_base,
            "device": device,
            "svd_fast": svd_fast,
            "local_epochs": 1,
            "lr": 0.01,
        }
        t0 = time.perf_counter()
        algo.client_update(model=m, dataloader=loader, state=st, config=cfg)
        return time.perf_counter() - t0

    t_off = statistics.median([run_once(False) for _ in range(max(3, reps // 4))])
    t_on = statistics.median([run_once(True) for _ in range(max(3, reps // 4))])
    print(f"  client_update  svd_fast=False: {t_off*1e3:8.1f} ms")
    print(
        f"  client_update  svd_fast=True : {t_on*1e3:8.1f} ms   "
        f"-> {t_off/max(t_on,1e-12):.2f}x"
    )
    return t_off, t_on


def smoke_accuracy(device: str):
    print("\n=== Smoke: eval accuracy unchanged (svd_fast on vs off, same seed) ===")
    # Deterministic learnable synthetic task: label = argmax of a fixed linear
    # teacher over the flattened image. resnet8 can fit it; both svd_fast modes
    # share the seed, so any accuracy gap reflects only the SVD path.
    torch.manual_seed(123)
    teacher = torch.randn(10, 3 * 32 * 32, device=device)

    def make_batch(n):
        x = torch.randn(n, 3, 32, 32, device=device)
        y = (teacher @ x.reshape(n, -1).t()).argmax(0)
        return x, y

    _xtr, _ytr = make_batch(128)
    train = DataLoader(TensorDataset(_xtr, _ytr), batch_size=32)
    xt, yt = make_batch(256)

    def run(svd_fast, rounds=8, clients=2):
        torch.manual_seed(7)
        algo = fr.FedResonance()
        cfg = {
            **algo.get_default_config(),
            "device": device,
            "svd_fast": svd_fast,
            "local_epochs": 1,
            "lr": 0.05,
        }
        gmodel = get_model("resnet8", "cifar10").to(device)
        states = [ClientState(client_id=i, battery_j=1e9) for i in range(clients)]
        for _ in range(rounds):
            updates = []
            gsd = {k: v.clone() for k, v in gmodel.state_dict().items()}
            for c in range(clients):
                cm = get_model("resnet8", "cifar10").to(device)
                cm.load_state_dict(gsd)
                upd, meta = algo.client_update(
                    model=cm, dataloader=train, state=states[c], config=cfg
                )
                updates.append((upd, meta, states[c]))
            agg = algo.server_aggregate(
                global_model=gmodel, client_updates=updates, round_num=_, config=cfg
            )
            gmodel.load_state_dict(
                {k: v.to(device) for k, v in agg.new_weights.items()}
            )
        gmodel.eval()
        with torch.no_grad():
            acc = (gmodel(xt).argmax(1) == yt).float().mean().item()
        return acc

    a_off = run(False)
    a_on = run(True)
    print(f"  eval accuracy  svd_fast=False: {a_off:.4f}")
    print(f"  eval accuracy  svd_fast=True : {a_on:.4f}")
    print(
        f"  |delta| = {abs(a_on - a_off):.4f}  "
        f"({'UNCHANGED' if abs(a_on - a_off) < 0.02 else 'CHANGED <-- check'})"
    )
    return a_off, a_on


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    eq = micro_bench(args.reps, args.device)
    macro_bench(args.reps, args.device)
    smoke_accuracy(args.device)

    print("\n" + "=" * 60)
    print(
        f"RESULT: per-layer equivalence {'PASS' if eq else 'FAIL'} "
        f"(rank +-2, Frobenius rel-err < 1e-3)."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
