"""
algorithms/fedprox.py
=====================
FedProx: Federated Optimization in Heterogeneous Networks
Li et al., MLSys 2020 — https://arxiv.org/abs/1812.06127

Key idea:
  Add a proximal term to the local objective that pulls each client's
  solution towards the current global model:

    min_w  F_k(w) + (mu/2) * ||w - w_global||^2

  This bounds the amount each client drifts from the global model,
  which is critical in heterogeneous settings (systems heterogeneity:
  variable local epochs; statistical heterogeneity: non-IID data).

Algorithm:
  For each round t, each client k:
    1. Receive global weights w_t from server
    2. Run local optimization:
         w_k^{j+1} = w_k^j - lr * (grad F_k(w_k^j) + mu * (w_k^j - w_t))
       for E local epochs
    3. Send delta = w_t - w_k^E to server
  Server aggregates with FedAvg.

Key difference vs FedAvg:
  FedAvg allows unbounded client drift in non-IID settings.
  FedProx bounds drift via mu: larger mu → less drift, slower convergence
  but more stable in heterogeneous settings.

Key difference vs E-CEFFL:
  FedProx addresses statistical heterogeneity but does NOT model energy.
  It does not adapt communication cost to battery level.
  Full-precision updates (no compression).

Reference hyperparameters (Li et al. 2020):
  mu = 0.01 (CIFAR-10, FEMNIST benchmarks)
  lr = 0.01, local_epochs = 1, batch_size = 32

Convergence:
  Theorem 1 in Li et al.: FedProx converges to stationary point
  at rate O(1/T) under bounded gradient assumptions and partial participation.
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("fedprox")
class FedProx(FLAlgorithm):
    """
    FedProx: proximal regularization to bound client drift (Li et al., 2020).

    Key properties:
      - Proximal term (mu/2)||w - w_global||^2 limits drift
      - Full-precision updates (no compression)
      - Universal participation (no battery-based exclusion)
      - Standard FedAvg aggregation server-side
      - mu=0 reduces exactly to FedAvg
    """
    name = "fedprox"
    description = (
        "FedProx: proximal term (mu/2)||w-w_global||^2 bounds client drift "
        "(Li et al., MLSys 2020). Baseline for non-IID FL."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)
        mu = config.get("mu", 0.01)   # proximal coefficient
        profile = config.get("device_profile")

        # ── Step 1: Save global weights (w_global for proximal term) ──────────
        w_global = OrderedDict(
            {k: v.clone().detach().to(device) for k, v in model.state_dict().items()}
        )
        w_before = OrderedDict(
            {k: v.clone().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 2: Local training with proximal term ─────────────────────────
        model.train()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # Task loss
                out = model(x)
                loss = criterion(out, y)

                # Proximal term: (mu/2) * ||w - w_global||^2
                prox_loss = 0.0
                for name, param in model.named_parameters():
                    if name in w_global:
                        prox_loss += (param - w_global[name].detach()).pow(2).sum()
                loss = loss + (mu / 2.0) * prox_loss

                loss.backward()

                # Gradient clipping for stability (used in Li et al. experiments)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Step 3: Compute delta ─────────────────────────────────────────────
        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })

        # ── Step 4: Energy accounting ─────────────────────────────────────────
        uplink_bytes = self.count_bytes(delta, sparse=False)
        downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs,
                len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * 1.0  # full transmission

        del w_global
        del w_before
        del current_sd
        gc.collect()

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": 1.0,
            "mu": mu,
        }
        return dict(delta), metadata

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """Standard FedAvg aggregation (uniform weights). Li et al. use FedAvg server-side."""
        K = len(client_updates)
        global_sd = global_model.state_dict()

        agg = None
        for update, _, _ in client_updates:
            if agg is None:
                agg = {k: v.clone().float() for k, v in update.items()}
            else:
                for k in agg:
                    agg[k] += update[k].float()

        new_weights = OrderedDict({
            k: global_sd[k].float() - (agg[k] / K).to(global_sd[k].device) for k in global_sd
        })
        del agg
        gc.collect()

        total_bytes = sum(m.get("bytes_sent", 0) for _, m, _ in client_updates)
        total_energy = sum(m.get("energy_j_consumed", 0) for _, m, _ in client_updates)
        avg_loss = sum(m.get("local_loss", 0) for _, m, _ in client_updates) / K
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / K

        participations = [1.0 if s.battery_j >= 0 else 0.0 for _, _, s in client_updates]
        sum_p = sum(participations)
        jain = (sum_p ** 2) / (K * sum(p ** 2 for p in participations)) if sum_p > 0 else 0.0

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy,
                "avg_beta": 1.0,
                "avg_battery_j": avg_battery,
                "avg_local_loss": avg_loss,
                "participation_rate": sum_p / K,
                "jain_index": jain,
                "num_clients": K,
                "mu": config.get("mu", 0.01),
            },
        )

    def get_default_config(self) -> dict:
        return {
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            "mu": 0.01,            # proximal coefficient (Li et al. default)
            "device": "cpu",
            "device_profile": None,
        }
