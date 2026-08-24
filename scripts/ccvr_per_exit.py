#!/usr/bin/env python3
"""CCVR per-exit: post-hoc, server-side recalibration of EVERY exit head.

Extends CCVR (Luo et al., NeurIPS 2021) to multi-exit models: for each exit d,
clients compute per-class Gaussian statistics (count, mean, covariance) of the
GAP boundary feature feeding head d — on their OWN local data only. The server
pools the moments (federated-legal: only aggregate statistics travel), samples
CLASS-BALANCED virtual features from N(mu_c, Sigma_c), and retrains the linear
head d on them. The trunk is never touched.

Usage:
  python3 scripts/ccvr_per_exit.py \
      --checkpoint results/E1_.../final_model.pt \
      --config configs/fedstep_ee_v31_e4_a01_gn4.yaml --seed 43 \
      [--samples-per-class 2000] [--ridge 0.01] [--steps 400]

Prints per-exit test accuracy BEFORE and AFTER calibration.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.registry import get_dataloader  # noqa: E402
from models.registry import get_model  # noqa: E402


def boundary_features(model, x):
    """GAP features at each exit boundary: {1: (B,c1), 2: (B,c2), 3: (B,c3)}."""
    f = {}
    h = model.stem(x)
    h = model.layer1(h)
    f[1] = model.avgpool(h).flatten(1)
    h = model.layer2(h)
    f[2] = model.avgpool(h).flatten(1)
    h = model.layer3(h)
    f[3] = model.avgpool(h).flatten(1)
    return f


@torch.no_grad()
def client_moments(model, loader, device, num_classes, num_exits=3):
    """Per-class feature moments on ONE client's data: N_c, sum_c, outer_c."""
    stats = {d: {} for d in range(1, num_exits + 1)}
    for x, y in loader:
        x = x.to(device)
        feats = boundary_features(model, x)
        for d in range(1, num_exits + 1):
            fd = feats[d].cpu().double()
            for c in y.unique().tolist():
                m = (y == c)
                fc = fd[m]
                s = stats[d].setdefault(c, [0, 0.0, 0.0])
                s[0] += int(m.sum())
                s[1] = s[1] + fc.sum(0)
                s[2] = s[2] + fc.T @ fc
    return stats


def pool_moments(all_client_stats, num_classes, num_exits=3):
    """Server-side pooling of per-client moments -> global (mu_c, Sigma_c)."""
    pooled = {}
    for d in range(1, num_exits + 1):
        pooled[d] = {}
        for c in range(num_classes):
            N, S, O = 0, None, None
            for st in all_client_stats:
                if c in st[d]:
                    n, s, o = st[d][c]
                    N += n
                    S = s if S is None else S + s
                    O = o if O is None else O + o
            if N >= 2:
                mu = S / N
                cov = O / N - torch.outer(mu, mu)
                pooled[d][c] = (N, mu, cov)
    return pooled


def recalibrate_head(head, pooled_d, dim, num_classes, samples_per_class,
                     ridge, steps, lr, device, seed):
    """Train a copy of `head` on class-balanced virtual features."""
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    for c, (N, mu, cov) in pooled_d.items():
        # Ridge for numerical stability (small-N classes, tiny dims)
        cov = cov + ridge * torch.eye(dim, dtype=cov.dtype)
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(samples_per_class, dim, generator=g, dtype=cov.dtype)
        zs.append(mu + eps @ L.T)
        ys.append(torch.full((samples_per_class,), c, dtype=torch.long))
    z = torch.cat(zs).float().to(device)
    y = torch.cat(ys).to(device)

    new_head = copy.deepcopy(head).to(device)
    opt = torch.optim.Adam(new_head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n = z.shape[0]
    bs = 256
    perm_g = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        idx = torch.randperm(n, generator=perm_g)[:bs].to(device)
        opt.zero_grad()
        loss = crit(new_head(z[idx]), y[idx])
        loss.backward()
        opt.step()
    return new_head


@torch.no_grad()
def eval_per_exit(model, loader, device):
    model.eval()
    correct = {1: 0, 2: 0, 3: 0}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outs = model.forward_all_exits(x)
        for d in (1, 2, 3):
            correct[d] += (outs[d].argmax(1) == y).sum().item()
        total += y.size(0)
    return {d: correct[d] / total for d in (1, 2, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--samples-per-class", type=int, default=2000)
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--scope", choices=["all", "heads"], default="all",
                    help="'heads' = calibrer les têtes 1..k-1 seulement, garder "
                         "la fc d'origine (recommandé à α=1: la finale est déjà "
                         "calibrée en IID, la recalibrer coûte ~0.6)")
    ap.add_argument("--device", default=None,
                    help="override du device de la config (ex: cpu pour ne pas "
                         "disputer le GPU à un run en cours)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["data"]["dataset"]
    model_name = cfg["data"]["model"]
    alpha = cfg["data"]["alpha"]
    num_clients = cfg["clients"]["num_clients"]
    data_root = cfg["data"].get("data_root", "./data")
    device = args.device or cfg.get("device", "cpu")
    num_classes = 10 if dataset == "cifar10" else 100
    exits_to_calibrate = (1, 2) if args.scope == "heads" else (1, 2, 3)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = get_model(model_name, dataset)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device).eval()

    test_loader = get_dataloader(dataset_name=dataset, split="test",
                                 partition="iid", client_id=None, num_clients=1,
                                 batch_size=256, data_root=data_root)

    print("Accuracy par exit AVANT calibration:")
    before = eval_per_exit(model, test_loader, device)
    print("  " + " / ".join(f"exit{d}={before[d]*100:.1f}" for d in (1, 2, 3)))

    # ── Phase client: moments par classe, client par client (fédéré-légal) ──
    print(f"Collecte des moments sur {num_clients} clients (partition Dirichlet α={alpha}, seed {args.seed})...")
    all_stats = []
    for cid in range(num_clients):
        loader = get_dataloader(dataset_name=dataset, split="train",
                                partition="dirichlet", client_id=cid,
                                num_clients=num_clients, batch_size=256,
                                alpha=alpha, data_root=data_root,
                                seed=args.seed)
        all_stats.append(client_moments(model, loader, device, num_classes))

    # ── Phase serveur: pooling + features virtuelles + ré-entraînement têtes ─
    pooled = pool_moments(all_stats, num_classes)
    dims = {1: model.aux_heads["1"].in_features,
            2: model.aux_heads["2"].in_features,
            3: model.fc.in_features}
    heads = {1: model.aux_heads["1"], 2: model.aux_heads["2"], 3: model.fc}
    for d in exits_to_calibrate:
        print(f"  exit {d}: {len(pooled[d])}/{num_classes} classes couvertes, dim={dims[d]}")
        new_head = recalibrate_head(heads[d], pooled[d], dims[d], num_classes,
                                    args.samples_per_class, args.ridge,
                                    args.steps, args.lr, device,
                                    seed=args.seed + d)
        if d < 3:
            model.aux_heads[str(d)] = new_head
        else:
            model.fc = new_head

    print("Accuracy par exit APRÈS calibration CCVR:")
    after = eval_per_exit(model, test_loader, device)
    print("  " + " / ".join(f"exit{d}={after[d]*100:.1f}" for d in (1, 2, 3)))
    print("Δ: " + " / ".join(f"exit{d}={100*(after[d]-before[d]):+.1f}" for d in (1, 2, 3)))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"before": before, "after": after, "scope": args.scope,
                       "config": args.config, "seed": args.seed}, f, indent=2)
        print(f"écrit: {args.out}")

    # ── Traçabilité dans les artefacts du run ────────────────────────────────
    # (1) modèle calibré à côté du checkpoint d'origine (jamais écrasé);
    # (2) bloc "ccvr_per_exit" ajouté au metrics.json du run, pour que le
    #     chiffre post-calibration soit visible depuis les artefacts du run
    #     (dashboard, agrégateurs) et pas seulement dans les logs CCVR.
    ckpt_dir = Path(args.checkpoint).parent
    torch.save(model.state_dict(), ckpt_dir / "final_model_ccvr.pt")
    print(f"modèle calibré: {ckpt_dir}/final_model_ccvr.pt")
    mpath = ckpt_dir / "metrics.json"
    if mpath.exists():
        with open(mpath) as f:
            metrics = json.load(f)
        metrics["ccvr_per_exit"] = {
            "before": before, "after": after,
            "delta": {d: after[d] - before[d] for d in before},
            "samples_per_class": args.samples_per_class,
            "ridge": args.ridge,
            "scope": args.scope,
        }
        with open(mpath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"metrics.json enrichi: {mpath}")


if __name__ == "__main__":
    main()
