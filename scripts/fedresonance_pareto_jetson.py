#!/usr/bin/env python
"""Pareto sweep accuracy vs upload-bytes for FedResonance on a Jetson Nano fleet.

Answers: "how far can we push compression while keeping good accuracy?"
Each config trains FedResonance for R rounds, evaluates global test accuracy, and
measures the post-warmup uplink bytes/client. Output: a Pareto scatter PNG + CSV.

FAST PROXY by default (small subset / few rounds-clients) so the *shape* of the
Pareto front is visible quickly; scale up via env vars for paper-final numbers:
  PAR_CLIENTS=40 PAR_ROUNDS=200 PAR_SUBSET=0 python scripts/fedresonance_pareto_jetson.py
(PAR_SUBSET=0 => full per-client data.)
"""
import os, sys, csv, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from algorithms.fed_resonance import FedResonance
from algorithms.base import ClientState
from models.registry import get_model
from datasets.registry import get_dataloader
from hardware.profiles import DEVICE_PROFILES

N        = int(os.environ.get("PAR_CLIENTS", 15))
ROUNDS   = int(os.environ.get("PAR_ROUNDS", 60))
SUBSET   = int(os.environ.get("PAR_SUBSET", 384))   # per-client train samples; 0 = all
WARMUP   = int(os.environ.get("PAR_WARMUP", 5))
OUT      = os.environ.get("PAR_OUT", "results/fedresonance/pareto_jetson")
DEV      = "mps" if torch.backends.mps.is_available() else "cpu"
PROFILE  = DEVICE_PROFILES["jetson_nano"]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

# ── data ────────────────────────────────────────────────────────────────────
def make_train(c):
    dl = get_dataloader("cifar10", "train", "dirichlet", client_id=c,
                        num_clients=N, alpha=0.5, batch_size=64, seed=42)
    if SUBSET and SUBSET < len(dl.dataset):
        ds = Subset(dl.dataset, list(range(SUBSET)))
    else:
        ds = dl.dataset
    return DataLoader(ds, batch_size=64, shuffle=True)

train_loaders = {c: make_train(c) for c in range(N)}
test_loader = get_dataloader("cifar10", "test", "iid", client_id=None,
                             num_clients=N, batch_size=256, seed=42)

@torch.no_grad()
def evaluate(model):
    model.eval(); correct = total = 0
    for x, y in test_loader:
        x, y = x.to(DEV), y.to(DEV)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / total

# ── one run ─────────────────────────────────────────────────────────────────
def run(label, overrides):
    torch.manual_seed(0)
    algo = FedResonance()
    base = dict(algo.get_default_config())
    base.update(dict(lr=0.003, local_epochs=1, optimizer="adam", weight_decay=1e-4,
                     batch_size=64, warmup_rounds=WARMUP, energy_scale_factor=1.5,
                     use_error_feedback=True, svd_fast=True,
                     energy_aware_selection=True, quant_stochastic=False,
                     ea_basis_refresh_period=5, ea_quantize_payloads=True,
                     ea_payload_bits=overrides.get("quant_bits", 4), device=DEV))
    base.update(overrides)
    g  = get_model("resnet8", "cifar10").to(DEV)
    cm = get_model("resnet8", "cifar10").to(DEV)
    st = {c: ClientState(client_id=c, battery_j=360_000.0) for c in range(N)}
    bytes_hist = []
    for r in range(ROUNDS):
        gsd = g.state_dict(); cups = []; rb = 0
        for c in range(N):
            cm.load_state_dict(gsd)
            cfg = dict(base); cfg["device_profile"] = PROFILE
            upd, meta = algo.client_update(cm, train_loaders[c], st[c], cfg)
            cups.append((upd, meta, st[c])); rb += meta["bytes_sent"]
        res = algo.server_aggregate(g, cups, r, cfg); g.load_state_dict(res.new_weights)
        if r >= WARMUP: bytes_hist.append(rb / N)        # bytes/client this round
    acc = evaluate(g)
    bpc = float(np.mean(bytes_hist[-10:])) if bytes_hist else float("nan")  # post-warmup avg
    print(f"  [{label:<28}] acc={acc:5.2f}%  bytes/client={bpc:8.0f}  ({(314892/bpc):.1f}x)", flush=True)
    return dict(label=label, acc=acc, bytes_per_client=bpc, ratio=314892/bpc)

# ── grid ────────────────────────────────────────────────────────────────────
GRID = [
    ("ref dense (budget 0)",       dict(error_budget=0.0,  quant_bits=8, rank_max=8)),
    ("q8 budget0.5",               dict(error_budget=0.50, quant_bits=8, rank_max=8)),
    ("q4 budget0.5",               dict(error_budget=0.50, quant_bits=4, rank_max=8)),
    ("q2 budget0.5",               dict(error_budget=0.50, quant_bits=2, rank_max=8)),
    ("q4 budget0.35",              dict(error_budget=0.35, quant_bits=4, rank_max=8)),
    ("q4 budget0.7",               dict(error_budget=0.70, quant_bits=4, rank_max=8)),
    ("q4 budget0.5 rank4",         dict(error_budget=0.50, quant_bits=4, rank_max=4)),
    ("q2 budget0.7 rank4 (aggr.)", dict(error_budget=0.70, quant_bits=2, rank_max=4)),
]

print(f"Pareto sweep | Jetson | N={N} clients | {ROUNDS} rounds | subset={SUBSET or 'full'} | dev={DEV}")
t0 = time.time(); rows = []
for label, ov in GRID:
    rows.append(run(label, ov))
print(f"\nsweep done in {time.time()-t0:.0f}s")

# ── save CSV + Pareto plot ───────────────────────────────────────────────────
csv_path = os.path.join(OUT, "pareto.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["label", "acc", "bytes_per_client", "ratio"])
    w.writeheader(); [w.writerow(r) for r in rows]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
xs = [r["bytes_per_client"]/1024 for r in rows]   # KB/client
ys = [r["acc"] for r in rows]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(xs, ys, s=70, zorder=3)
for r, x, y in zip(rows, xs, ys):
    ax.annotate(r["label"], (x, y), fontsize=7, xytext=(5, 4),
                textcoords="offset points")
ax.set_xlabel("Upload bytes / client / round (KB)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title(f"FedResonance Pareto — Jetson Nano (N={N}, {ROUNDS} rounds, "
             f"subset={SUBSET or 'full'})")
ax.grid(True, alpha=0.3)
png = os.path.join(OUT, "pareto_accuracy_vs_bytes.png")
fig.tight_layout(); fig.savefig(png, dpi=130)
print(f"saved: {png}\nsaved: {csv_path}")
