"""
algorithms/fedqgate.py
======================
FedQGate: Quality-Gated Federated Averaging via Loss-Improvement Weighting
Nikiema & Amhoud, UM6P 2026

Intuition
---------
In standard FedAvg each client's update is aggregated with equal weight
regardless of whether local training actually improved the model. Under
non-IID data, some clients may receive a global model that is poorly
aligned with their local distribution, causing their local loss to *increase*
after E SGD steps — yet that update is still aggregated at full weight.

FedQGate replaces uniform aggregation with a quality-weighted scheme:
each client computes a quality score

    q_k = max(0, (L_before^k − L_after^k) / L_before^k)  ∈ [0, 1]

where L_before / L_after are local losses evaluated on the same data
subset before and after local training. Clients whose local loss increased
contribute q_k = 0 (filtered out). Clients that improved contribute
proportionally to their improvement magnitude.

Aggregation
-----------
    Δ̄ = Σ_k (n_k · q_k · Δw_k) / Σ_k (n_k · q_k)

    If Σ q_k = 0 (all clients deteriorated):
        → fallback to uniform FedAvg (do not skip the round)

Server momentum (optional, γ ∈ [0, 1)):
    m_{t+1} = γ · m_t + Δ̄_t
    w_{t+1} = w_t − η_s · m_{t+1}

Connection to existing work
---------------------------
- FedProx   : prevents local drift via proximal penalty (per-step, client-side)
- Oort      : quality-aware *client selection* (server-side); FedQGate acts
              on aggregation weights, not selection
- SCAFFOLD  : corrects client drift via control variates (always aggregates)
- FedQGate  : soft quality gate at aggregation, no regularization overhead,
              compatible with any local optimizer

Parameters
----------
lr              : float  — local learning rate
local_epochs    : int    — local SGD epochs
val_fraction    : float  — fraction of local batches used for pre/post evaluation
                           (0.0 = skip evaluation, equivalent to FedAvg)
quality_floor   : float  — minimum q_k to participate in aggregation (default 0.0)
server_lr       : float  — server-side learning rate applied to the aggregated delta
server_momentum : float  — γ for server momentum buffer (0.0 = disabled)
fallback_uniform: bool   — if True, use uniform aggregation when all q_k = 0
beta_min        : float  — minimum compression ratio (energy-aware, unused here)
beta_max        : float  — maximum compression ratio (unused in this file, kept for API)
battery_max_j   : float  — battery capacity (J)
energy_scale_factor : float
max_grad_norm   : float | None
"""

import gc
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Optional

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_loss(model: nn.Module, dataloader, device: str,
               val_fraction: float = 0.2) -> float:
    """
    Compute mean cross-entropy loss over a fraction of the dataloader.
    Uses no_grad — does not affect model state.
    """
    if val_fraction <= 0.0:
        return 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    target_batches = max(1, math.ceil(len(dataloader) * val_fraction))
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= target_batches:
                break
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item()
            n += 1
    model.train()
    return total / max(n, 1)


def _quality_score(loss_before: float, loss_after: float) -> float:
    """
    q_k = max(0, (L_before - L_after) / L_before)

    Returns 0 if L_before <= 0 (degenerate case).
    Returns 0 if loss increased or stayed the same.
    Returns 1.0 if loss reached 0 (perfect local fit).
    """
    if loss_before <= 1e-9:
        return 0.0
    return max(0.0, (loss_before - loss_after) / loss_before)


# ─────────────────────────────────────────────────────────────────────────────
# FedQGate
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fedqgate")
class FedQGate(FLAlgorithm):
    """
    FedQGate — Quality-Gated Federated Averaging via Loss-Improvement Weighting.

    Soft-gates each client's contribution by its local loss improvement ratio.
    Clients that fail to improve locally contribute zero weight to aggregation.
    Server optionally applies momentum over the quality-weighted mean delta.
    """

    name = "fedqgate"
    description = (
        "FedQGate (Nikiema & Amhoud, 2026): quality-weighted aggregation "
        "based on per-client loss improvement ratio. Clients that worsen "
        "the local model are soft-filtered. Compatible with server momentum."
    )

    def __init__(self):
        super().__init__()
        # Server-side momentum buffer (persists across rounds)
        self._momentum_buf: Optional[dict] = None

    # ── Client ────────────────────────────────────────────────────────────────

    def client_update(self, model, dataloader, state: ClientState, config: dict):
        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        local_epochs  = config.get("local_epochs", 3)
        max_grad_norm = config.get("max_grad_norm", None)
        val_fraction  = config.get("val_fraction", 0.2)
        quality_floor = config.get("quality_floor", 0.0)

        model.train()
        model.to(device)

        # ── Snapshot weights before training ──────────────────────────────
        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        # ── Evaluate loss BEFORE local training ───────────────────────────
        loss_before = _eval_loss(model, dataloader, device, val_fraction)

        # ── Local SGD ─────────────────────────────────────────────────────
        optimizer = optim.SGD(
            model.parameters(), lr=lr,
            momentum=config.get("local_momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Evaluate loss AFTER local training ────────────────────────────
        loss_after = _eval_loss(model, dataloader, device, val_fraction)

        # ── Quality score ─────────────────────────────────────────────────
        q_k = _quality_score(loss_before, loss_after)

        # ── Compute delta (w_before − w_after, i.e. negative gradient) ───
        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })
        del w_before, current_sd
        gc.collect()

        # ── Energy accounting ─────────────────────────────────────────────
        profile        = config.get("device_profile")
        uplink_bytes   = self.count_bytes(delta, sparse=False)
        downlink_bytes = uplink_bytes

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size,
                local_epochs, len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0

        energy_j *= config.get("energy_scale_factor", 1.0)

        # If quality is below floor: still consume energy (already trained)
        # but signal 0 quality → server will discard
        effective_q = q_k if q_k >= quality_floor else 0.0

        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        metadata = {
            "client_id":          state.client_id,
            "round_num":          state.round_num,
            "loss_before":        loss_before,
            "loss_after":         loss_after,
            "quality_score":      effective_q,
            "beta_actual":        1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":  energy_j,
            "bytes_sent":         uplink_bytes,
            "bytes_received":     downlink_bytes,
            "local_loss":         total_loss / max(num_batches, 1),
            "compression_ratio":  1.0,
        }
        return dict(delta), metadata

    # ── Server ────────────────────────────────────────────────────────────────

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K          = len(client_updates)
        global_sd  = global_model.state_dict()
        server_lr  = config.get("server_lr", 1.0)
        gamma      = config.get("server_momentum", 0.0)
        fallback   = config.get("fallback_uniform", True)

        # Dataset sizes for weighted averaging
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = max(sum(sizes), 1)

        # ── Collect quality-weighted contributions ────────────────────────
        quality_scores = [m.get("quality_score", 1.0) for _, m, _ in client_updates]
        effective_weights = [n_k * q_k for n_k, q_k in zip(sizes, quality_scores)]
        sum_ew = sum(effective_weights)

        # Fallback to uniform if all quality scores are 0
        if sum_ew < 1e-9:
            if fallback:
                # Uniform FedAvg — don't skip the round
                effective_weights = [float(n_k) for n_k in sizes]
                sum_ew = float(total_n)
                fallback_used = True
            else:
                # Skip aggregation — return current weights unchanged
                return AggregateResult(
                    new_weights=OrderedDict(
                        {k: v.clone() for k, v in global_sd.items()}
                    ),
                    metrics={
                        "round": round_num,
                        "total_bytes_sent": sum(m["bytes_sent"] for _, m, _ in client_updates),
                        "total_energy_j":   sum(m["energy_j_consumed"] for _, m, _ in client_updates),
                        "avg_quality":      0.0,
                        "num_filtered":     K,
                        "fallback_used":    False,
                        "avg_beta":         1.0,
                        "avg_battery_j":    sum(s.battery_j for _, _, s in client_updates) / K,
                        "avg_local_loss":   sum(m["local_loss"] for _, m, _ in client_updates) / K,
                        "participation_rate": sum(1 for _, _, s in client_updates if s.battery_j > 0) / K,
                        "jain_index":       self._jain(client_updates),
                        "num_clients":      K,
                    }
                )
        else:
            fallback_used = False

        # ── Quality-weighted aggregation of deltas ─────────────────────────
        agg = None
        for (update, meta, _), ew in zip(client_updates, effective_weights):
            w_k = ew / sum_ew
            if agg is None:
                agg = {k: v.clone().float() * w_k for k, v in update.items()}
            else:
                for k in agg:
                    agg[k] += update[k].float() * w_k

        # ── Server momentum ───────────────────────────────────────────────
        if gamma > 0.0:
            if self._momentum_buf is None:
                self._momentum_buf = {k: v.clone() for k, v in agg.items()}
            else:
                for k in self._momentum_buf:
                    self._momentum_buf[k] = gamma * self._momentum_buf[k] + agg[k]
            update_direction = self._momentum_buf
        else:
            update_direction = agg

        # ── Apply to global model ─────────────────────────────────────────
        new_weights = OrderedDict({
            k: (global_sd[k].float() - server_lr * update_direction[k].to(global_sd[k].device))
            for k in global_sd
        })
        del agg
        gc.collect()

        # ── Metrics ───────────────────────────────────────────────────────
        num_filtered = sum(1 for q in quality_scores if q <= 0.0)
        avg_quality  = sum(quality_scores) / K

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round":             round_num,
                "total_bytes_sent":  sum(m["bytes_sent"] for _, m, _ in client_updates),
                "total_energy_j":    sum(m["energy_j_consumed"] for _, m, _ in client_updates),
                "avg_quality":       avg_quality,
                "num_filtered":      num_filtered,
                "fallback_used":     fallback_used,
                "avg_beta":          1.0,
                "avg_battery_j":     sum(s.battery_j for _, _, s in client_updates) / K,
                "avg_local_loss":    sum(m["local_loss"] for _, m, _ in client_updates) / K,
                "participation_rate": sum(1 for _, _, s in client_updates if s.battery_j > 0) / K,
                "jain_index":        self._jain(client_updates),
                "num_clients":       K,
            }
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _jain(client_updates) -> float:
        participations = [1.0 if s.battery_j > 0 else 0.0 for _, _, s in client_updates]
        K = len(participations)
        denom = K * sum(p ** 2 for p in participations)
        return (sum(participations) ** 2 / denom) if denom > 0 else 0.0

    def get_default_config(self):
        return {
            # ── Local training ──────────────────────────────────────────
            "lr":             0.01,
            "local_momentum": 0.9,
            "weight_decay":   1e-4,
            "local_epochs":   3,
            "batch_size":     32,
            "max_grad_norm":  10.0,
            # ── Quality gating ──────────────────────────────────────────
            # val_fraction: fraction of local batches evaluated before/after.
            # 0.2 = first 20% of batches. Lower → faster, noisier estimate.
            # 1.0 = full local dataset (expensive but accurate).
            "val_fraction":    0.2,
            # quality_floor: clients with q_k < floor are fully excluded.
            # 0.0 = soft gate (all clients with q>0 contribute proportionally).
            # 0.05 = exclude clients with < 5% improvement.
            "quality_floor":   0.0,
            # fallback_uniform: if True, use uniform FedAvg when all q_k=0.
            # Prevents round skipping at the cost of accepting bad updates.
            "fallback_uniform": True,
            # ── Server ──────────────────────────────────────────────────
            "server_lr":       1.0,
            "server_momentum": 0.9,
            # ── Energy ──────────────────────────────────────────────────
            "battery_max_j":        13320.0,
            "energy_scale_factor":  12.6,
            "device_profile":       None,
        }
