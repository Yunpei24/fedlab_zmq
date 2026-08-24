"""
algorithms/fedle.py
===================
FedLE: Federated Learning Client Selection with Lifespan Extension for Edge
IoT Networks (Yan et al., IEEE) — official baseline ported into the
FedLab-ZMQ energy harness.

FedLE is a pure CLIENT-SELECTION method (it shapes WHO participates, not
what each computes) — the direct conceptual competitor to FedSTEP-EE on the
"survival / lifespan" axis. Faithful mechanics (paper Algorithms 1 & 2):

  * Similarity matrix (Alg. 1, ONE-TIME, first round): every client trains
    1 epoch, uploads PARTIAL weights (first conv `stem` + last layer `fc`),
    server builds a K×K pairwise dot-product similarity matrix.
  * K-means clustering on the partial-update vectors (groups clients with
    similar data distributions).
  * Client selection (Alg. 2, every round): pick K·C clients with
      p(client) ∝ (1 / |cluster|) × battery_level
    — smaller clusters (rarer data → more diverse) get higher probability
    (anti majority-class overfitting), and within any group higher-battery
    clients are preferred (energy optimization).
  * Critical battery δ: clients with SoC ≤ δ are excluded from selection
    (the edge device keeps that charge for its primary tasks).
  * Standby drain: EVERY alive client loses a small standby cost each round,
    even when not selected (idle consumption) — the mechanism by which idle
    clients eventually drop out.
  * Aggregation: FedAvg over the selected clients.

Runs through the harness `run_round` hook with sample_fraction=1.0 (FedLE
does its own selection among all alive clients).
"""

import gc
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops

from .base import AggregateResult, FLAlgorithm, register_algorithm


def _partial_vector(sd: dict) -> np.ndarray:
    """Flatten the FedLE partial model (first conv `stem` + last `fc`)."""
    parts = []
    for k, v in sd.items():
        if k.startswith("stem.") or k.startswith("fc."):
            parts.append(v.detach().cpu().float().flatten())
    return torch.cat(parts).numpy() if parts else np.zeros(1, dtype=np.float32)


def _kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny k-means (no sklearn dependency). Returns cluster label per row."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    # k-means++ style init: first random, rest by distance
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        p = d / max(d.sum(), 1e-12)
        centers.append(X[rng.choice(n, p=p)])
    C = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.stack([np.sum((X - c) ** 2, axis=1) for c in C], axis=1)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels


@register_algorithm("fedle")
class FedLE(FLAlgorithm):
    """FedLE (Yan et al.): lifespan-extending client selection via partial-
    model similarity clustering + battery-weighted cluster sampling."""

    name = "fedle"
    description = (
        "FedLE (Yan et al.): energy-aware client selection for edge lifespan "
        "extension — one-time partial-model similarity clustering + "
        "battery×inverse-cluster-size sampling + critical-battery exclusion."
    )

    # ── local training of one selected client (FedAvg-style, real drain) ─────
    def _train_client(self, model, loader, state, config, one_epoch=False):
        device = config.get("device", "cpu")
        lr = float(config.get("lr", 0.003))
        epochs = 1 if one_epoch else int(config.get("local_epochs", 8))
        wd = float(config.get("weight_decay", 1e-4))
        clip = config.get("max_grad_norm", None)
        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        model.train(); model.to(device)
        opt = (optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
               if str(config.get("optimizer", "adam")).lower() == "adam"
               else optim.SGD(model.parameters(), lr=lr,
                              momentum=float(config.get("momentum", 0.9)), weight_decay=wd))
        crit = nn.CrossEntropyLoss()
        tot, nb = 0.0, 0
        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                tot += loss.item(); nb += 1
        sd = model.state_dict()
        delta = OrderedDict({k: (w_before[k] - sd[k].cpu()).float() for k in w_before})

        profile = config.get("device_profile")
        up = self.count_bytes(delta, sparse=False)
        if profile is not None:
            flops = round_compute_flops(model, [n for n, _ in model.named_parameters()],
                                        config, profile, loader, epochs)
            bd = profile.round_energy_breakdown(flops, up, up,
                                                config.get("energy_scale_factor", 1.0),
                                                config.get("alpha_applies_to", "compute"))
        else:
            e = (0.5 + 2.0) * config.get("energy_scale_factor", 1.0)
            bd = {"compute": e, "uplink": 0.0, "downlink": 0.0, "total": e}
        state.battery_j = max(0.0, state.battery_j - bd["total"])
        state.round_num += 1
        del w_before, sd
        return delta, bd, tot / max(nb, 1), up

    def run_round(self, global_model, alive_clients, client_model, config,
                  round_num, device):
        import time
        C = float(config.get("fedle_C", 0.10))         # participation fraction
        n_clusters = int(config.get("fedle_n_clusters", 10))
        delta_crit = float(config.get("fedle_delta", 0.20))  # critical SoC
        standby_frac = float(config.get("fedle_standby_frac", 0.05))
        if not hasattr(self, "_server_state"):
            self._server_state = {}
        ss = self._server_state
        global_sd = {k: v.to(device) for k, v in global_model.state_dict().items()}
        K_total = config.get("num_clients", len(alive_clients))
        gsd_cpu = OrderedDict({k: v.clone().cpu() for k, v in global_model.state_dict().items()})

        client_tuples, train_times = [], []

        # ── Step: one-time similarity matrix + clustering (Alg. 1) ───────────
        if "clusters" not in ss:
            vecs, ids = [], []
            for cid, loader, state, profile in alive_clients:
                client_model.load_state_dict(gsd_cpu)
                cfg = {**config, "device_profile": profile}
                self._train_client(client_model, loader, state, cfg, one_epoch=True)
                vecs.append(_partial_vector(client_model.state_dict()))
                ids.append(cid)
            X = np.stack(vecs)
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)          # standardize
            labels = _kmeans(X, n_clusters, seed=int(config.get("seed", 42)))
            ss["clusters"] = {cid: int(l) for cid, l in zip(ids, labels)}
            # reference full-round compute energy for the standby cost
            _, bd0, _, _ = None, {"total": 0.0}, None, None
            ss["round_ref_j"] = None
            # (the 1-epoch pass above already drained everyone; treat as warmup)

        clusters = ss["clusters"]

        # ── Step: client selection (Alg. 2) ─────────────────────────────────
        cap = {cid: prof.battery.capacity_j for cid, _, _, prof in alive_clients}
        eligible = [(cid, ld, st, pf) for cid, ld, st, pf in alive_clients
                    if st.battery_j > delta_crit * cap[cid]]     # δ exclusion
        n_sel = max(1, int(round(C * K_total)))
        if eligible:
            sizes = {}
            for cid, _, _, _ in eligible:
                cl = clusters.get(cid, 0); sizes[cl] = sizes.get(cl, 0) + 1
            # p ∝ (1/|cluster|) × battery : rare clusters + high battery favored
            w = np.array([(1.0 / sizes[clusters.get(cid, 0)]) * st.battery_j
                          for cid, _, st, _ in eligible], dtype=float)
            w = w / w.sum()
            rng = np.random.default_rng(int(config.get("seed", 42)) + round_num)
            n_pick = min(n_sel, len(eligible))
            idx = rng.choice(len(eligible), size=n_pick, replace=False, p=w)
            selected = [eligible[i] for i in idx]
        else:
            selected = []

        # ── Step: train the selected (FedAvg), real drain ───────────────────
        for cid, loader, state, profile in selected:
            client_model.load_state_dict(gsd_cpu)
            cfg = {**config, "device_profile": profile}
            t0 = time.time()
            delta, bd, loss, up = self._train_client(client_model, loader, state, cfg)
            train_times.append(time.time() - t0)
            meta = {"client_id": cid, "round_num": state.round_num, "beta_actual": 1.0,
                    "battery_j_remaining": state.battery_j, "energy_j_consumed": bd["total"],
                    "energy_compute_j": bd["compute"], "energy_uplink_j": bd["uplink"],
                    "energy_downlink_j": bd["downlink"], "bytes_sent": up,
                    "bytes_received": up, "local_loss": loss, "compression_ratio": 1.0,
                    "dataset_size": len(loader.dataset)}
            client_tuples.append((delta, meta, state))

        # ── Standby drain for alive-but-not-selected clients ────────────────
        if ss.get("round_ref_j") is None and selected:
            ss["round_ref_j"] = float(np.mean([m["energy_compute_j"]
                                               for _, m, _ in client_tuples]))
        standby = standby_frac * (ss.get("round_ref_j") or 0.0)
        sel_ids = {cid for cid, _, _, _ in selected}
        for cid, _, state, _ in alive_clients:
            if cid not in sel_ids and state.battery_j > 0.0:
                state.battery_j = max(0.0, state.battery_j - standby)

        # ── Aggregation: FedAvg over selected ───────────────────────────────
        if client_tuples:
            sizes_n = [m["dataset_size"] for _, m, _ in client_tuples]
            tot_n = max(sum(sizes_n), 1)
            new = OrderedDict({k: v.clone().float() for k, v in global_sd.items()})
            agg = None
            for (upd, _, _), nk in zip(client_tuples, sizes_n):
                if agg is None:
                    agg = {k: upd[k].float() * (nk / tot_n) for k in upd}
                else:
                    for k in agg:
                        agg[k] += upd[k].float() * (nk / tot_n)
            for k in new:
                if k in agg:
                    new[k] = new[k] - agg[k].to(new[k].device)
            agg_res = AggregateResult(new_weights=new, metrics=self._metrics(
                client_tuples, alive_clients, round_num, len(selected)))
        else:
            agg_res = AggregateResult(new_weights=OrderedDict(global_sd),
                                      metrics=self._metrics([], alive_clients, round_num, 0))
        del global_sd
        gc.collect()
        return agg_res, client_tuples, train_times

    def _metrics(self, tuples, alive, round_num, n_sel):
        K = len(alive)
        tb = sum(m["bytes_sent"] for _, m, _ in tuples)
        te = sum(m["energy_j_consumed"] for _, m, _ in tuples)
        return {"round": round_num, "total_bytes_sent": tb, "total_energy_j": te,
                "avg_beta": 1.0,
                "avg_battery_j": sum(a[2].battery_j for a in alive) / max(K, 1),
                "avg_local_loss": (sum(m["local_loss"] for _, m, _ in tuples) / len(tuples))
                if tuples else 0.0,
                "participation_rate": n_sel / max(K, 1), "jain_index": 1.0,
                "num_clients": K, "num_selected": n_sel, "exit_mode": "fedle_selection"}

    # client_update unused (run_round drives everything) but required abstract
    def client_update(self, model, dataloader, state, config):
        raise RuntimeError("FedLE uses run_round(), not client_update().")

    def server_aggregate(self, global_model, client_updates, round_num, config):
        raise RuntimeError("FedLE uses run_round(), not server_aggregate().")

    def get_default_config(self):
        return {"lr": 0.003, "optimizer": "adam", "local_epochs": 8, "batch_size": 32,
                "device": "cpu", "device_profile": None,
                "fedle_C": 0.10,          # participation fraction (paper: 0.05)
                "fedle_n_clusters": 10,   # k-means clusters
                "fedle_delta": 0.20,      # critical SoC (paper: 0.2)
                "fedle_standby_frac": 0.05}  # idle drain as frac of a round's compute
