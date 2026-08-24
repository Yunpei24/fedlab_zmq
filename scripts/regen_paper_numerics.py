#!/usr/bin/env python3
"""
scripts/regen_paper_numerics.py
===============================
Regenerate ALL paper numerics from the REPO's ResNet-8 (models/registry.py), so
the LaTeX tables match the released code. Produces, for ResNet-8 / CIFAR-10 / B=32:

  (1) per-group byte cost b_g (trainable params x4, and incl. BN buffers x4),
  (2) Cost-Verification Table III: per-group FLOP share under phi / corrected /
      measured (FlopCounterMode), absolute totals, and the analytic->measured
      under-count factor,
  (3) Communication table C(M) for M=1..10 with the CORRECTED bucketing
      (empty buckets contribute 0; no phantom-singleton double-count),
      using sigma = groups sorted by corrected compute cost (ascending).

Prints human-readable values AND copy-paste LaTeX rows. Run:
    python scripts/regen_paper_numerics.py            # measured on CPU (slow-ish)
"""
from __future__ import annotations
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from models.registry import get_model
from algorithms.fedpart import _derive_layer_groups
from hardware.flop_cost import (
    compute_group_flops, compute_corrected_group_costs, measure_fwd_bwd_flops,
)

DEV = "cpu"; B = 32; SHAPE = (1, 3, 32, 32)
m = get_model("resnet8", "cifar10")
groups = _derive_layer_groups(m)                      # list[list[param_name]]
G = len(groups)
named = dict(m.named_parameters())
sd = m.state_dict()

# ---- (1) byte costs ----
params_g = [sum(named[p].numel() for p in g if p in named) for g in groups]
# BN buffers (running_mean/var/num_batches) belonging to each group's modules
def _mods(g): return {p.rsplit(".", 1)[0] for p in g}
buf_g = []
for g in groups:
    mods = _mods(g)
    buf_g.append(sum(v.numel() for k, v in sd.items()
                     if k.rsplit(".", 1)[0] in mods and k not in named))
bytes_train = [4 * p for p in params_g]               # uplink delta (trainable only)
bytes_full  = [4 * (p + b) for p, b in zip(params_g, buf_g)]  # incl BN buffers

print("="*70); print("(1) PER-GROUP BYTE COSTS  (ResNet-8, repo models/registry.py)")
print(f"{'g':>2} {'group (first param)':32} {'params':>8} {'b_g trainx4':>12} {'+BNbuf x4':>10}")
for i, g in enumerate(groups):
    print(f"{i:>2} {g[0][:32]:32} {params_g[i]:>8} {bytes_train[i]:>12} {bytes_full[i]:>10}")
print(f"{'':2} {'TOTAL':32} {sum(params_g):>8} {sum(bytes_train):>12} {sum(bytes_full):>10}")
print(f"mu_b (trainable x4) = {sum(bytes_train)/G:,.1f} bytes")
print(f"mu_b (incl BN buf)  = {sum(bytes_full)/G:,.1f} bytes")

# ---- (2) cost-verification: phi / corrected / measured ----
F = compute_group_flops(groups, m, SHAPE)             # per-group forward FLOPs
corr = compute_corrected_group_costs(F)
meas = [measure_fwd_bwd_flops(m, groups[i], SHAPE, B, DEV) for i in range(G)]
def shares(x): s = sum(x); return [100*v/s for v in x]
phi_pct, corr_pct, meas_pct = shares(F), shares(corr), shares(meas)
print("\n"+"="*70); print("(2) COST VERIFICATION  (shares %, ResNet-8, B=32)")
print(f"{'g':>2} {'group':22} {'phi %':>7} {'corr %':>7} {'measured %':>11}")
for i, g in enumerate(groups):
    print(f"{i:>2} {g[0][:22]:22} {phi_pct[i]:>7.2f} {corr_pct[i]:>7.2f} {meas_pct[i]:>11.2f}")
tot_phi, tot_meas = sum(F), sum(meas)
print(f"absolute totals: phi(forward)={tot_phi:,.0f}  corrected={sum(corr):,.0f}  measured={tot_meas:,.0f}")
print(f"measured / phi under-count factor = {tot_meas/tot_phi:.1f}x")
# rank agreement corrected vs measured
import statistics
def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda k: a[k]))}
    rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda k: b[k]))}
    da = [ra[i]-rb[i] for i in range(len(a))]
    n = len(a); return 1 - 6*sum(d*d for d in da)/(n*(n*n-1))
print(f"Spearman rho (corrected vs measured ordering) = {spearman(corr, meas):.3f}")

# ---- (3) communication C(M) ----
# NOTE (2026-07): algorithm renamed fedpart_be -> fedstep (registry alias kept);
# keys below stay "fedpart_be" to match the historical result dirs this script reads.
# Scheduler convention (algorithms/fedpart_be.py l.155-167): sort groups by
# corrected cost ASCENDING, then move the classifier head (fc) to the END so it
# always lands in the most-expensive bucket (Tier K-1).
head = next(i for i, g in enumerate(groups) if any("fc" in p for p in g))
sigma = [i for i in sorted(range(G), key=lambda i: corr[i]) if i != head] + [head]
b = [bytes_train[i] for i in sigma]                   # b_g in sigma-order (trainable bytes)
mu_b = sum(b)/G
print("\n"+"="*70); print("(3) COMMUNICATION TABLE C(M)  (sigma = corrected-cost ascending)")
print(f"sigma (group idx) = {sigma}")
print(f"b_g in sigma-order = {b}")
print(f"{'M':>2} {'s':>2} {'C(M)':>10} {'D% vs mu_b':>10} {'tmax':>5} {'C*(t+1)':>10} {'(G/M)mu_b':>10} {'M|G':>4}")
for M in range(1, G+1):
    s = math.ceil(G/M)
    means = []
    for t in range(M):
        idx = list(range(t*s, min((t+1)*s, G)))
        if idx: means.append(sum(b[i] for i in idx)/len(idx))   # empty bucket -> skip (contributes 0)
    CM = sum(means)/M
    tmax = s-1
    print(f"{M:>2} {s:>2} {CM:>10,.0f} {(CM-mu_b)/mu_b*100:>9.2f}% {tmax:>5} "
          f"{CM*(tmax+1):>10,.0f} {(G/M)*mu_b:>10,.0f} {'yes' if G%M==0 else 'no':>4}")
print(f"mu_b = {mu_b:,.1f} bytes;  M* (min C(M), no empty-tier) chosen among divisor-free feasible M")
