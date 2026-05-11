"""
algorithms/fedbacys.py
======================
FedBacys: Federated Battery-Cyclic Scheduling
Jeong et al., 2025

Key idea:
  Clients are partitioned into groups based on battery levels.
  Only clients above a battery threshold T_active participate
  in a given round. Low-battery clients "skip and recharge."
  The threshold T_active decays over rounds to ensure
  long-run global participation.

  Participation decision:
    participate_t^k = 1  if B_t^k >= T_active(t)
                    = 0  otherwise (client skips, battery partially recovers)

  Recharge model:
    B_{t+1}^k = B_t^k + delta_recharge  if client skips
              = B_t^k - E_round^k       if client participates

  Threshold annealing:
    T_active(t) = T_0 * decay^t

Key difference vs E-CEFFL:
  FedBacys EXCLUDES low-battery clients → participation < 1 possible.
  In non-IID settings, systematic exclusion of certain clients
  introduces selection bias. Jain index J < 1 in heterogeneous batteries.
  E-CEFFL uses compression floor to guarantee J=1 always.

Reference:
  Jeong et al. (2025) "FedBacys: ..."
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("fedbacys")
class FedBacys(FLAlgorithm):
    """
    FedBacys: cyclic battery-aware scheduling with threshold annealing.

    Key properties:
      - Clients below T_active(t) are skipped, battery partially recovers
      - Threshold anneals over rounds to guarantee long-run participation
      - Full-precision updates for active clients
      - Jain index < 1 in heterogeneous scenarios (systematic exclusion risk)
    """
    name = "fedbacys"
    description = (
        "Cyclic battery-aware scheduling (Jeong et al., 2025). "
        "Low-battery clients skip rounds and recharge."
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
        t_init = config.get("t_active_init", 0.5)    # initial threshold as fraction of B_max
        decay = config.get("threshold_decay", 0.995)  # per-round decay
        recharge_rate = config.get("recharge_rate_j", 500.0)  # J recovered per skipped round
        battery_max_j = config.get("battery_max_j", 185400.0)

        # ── Step 1: Compute current threshold ─────────────────────────────
        t_active_j = t_init * battery_max_j * (decay ** state.round_num)
        t_active_j = max(0.0, t_active_j)   # floor at 0

        # ── Step 2: Participation decision ────────────────────────────────
        active = state.battery_j >= t_active_j

        if not active:
            # Client skips: partial recharge, send zero update
            state.battery_j = min(battery_max_j, state.battery_j + recharge_rate)
            state.round_num += 1
            empty_update = {k: torch.zeros_like(v.cpu())
                           for k, v in model.state_dict().items()}
            metadata = {
                "client_id": state.client_id,
                "round_num": state.round_num,
                "active": False,
                "t_active_j": t_active_j,
                "battery_j_remaining": state.battery_j,
                "energy_j_consumed": 0.0,
                "bytes_sent": 0,
                "bytes_received": 0,
                "local_loss": float("nan"),
                "compression_ratio": 0.0,
                "beta_actual": 0.0,
                "recharge_j": recharge_rate,
            }
            return empty_update, metadata

        # ── Step 3: Active client — standard local training ────────────────
        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        model.train(); model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        total_loss, steps = 0.0, 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); steps += 1

        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })

        # ── Step 4: Energy accounting ──────────────────────────────────────
        profile = config.get("device_profile")
        uplink_bytes = self.count_bytes(delta, sparse=False)
        downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs, len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0

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
            "active": True,
            "t_active_j": t_active_j,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(steps, 1),
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
        }
        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        """
        Aggregate only active clients (active=True in metadata).
        Inactive clients contribute a zero update (ignored in avg).
        """
        global_sd = global_model.state_dict()

        active_updates = [
            (upd, meta, state)
            for upd, meta, state in client_updates
            if meta.get("active", True)
        ]
        K_total = len(client_updates)
        K_active = len(active_updates)

        if K_active == 0:
            # No active clients this round — keep global model unchanged
            return AggregateResult(
                new_weights=OrderedDict({k: v.float() for k, v in global_sd.items()}),
                metrics={
                    "round": round_num,
                    "total_bytes_sent": 0,
                    "total_energy_j": 0,
                    "avg_beta": 0.0,
                    "avg_battery_j": sum(s.battery_j for _, _, s in client_updates) / K_total,
                    "avg_local_loss": float("nan"),
                    "participation_rate": 0.0,
                    "jain_index": 0.0,
                    "num_clients": K_total,
                },
            )

        agg = None
        for upd, _, _ in active_updates:
            if agg is None:
                agg = {k: v.float().clone() for k, v in upd.items()}
            else:
                for k in agg:
                    agg[k] += upd[k].float()

        new_weights = OrderedDict({
            k: global_sd[k].float() - (agg[k] / K_active).to(global_sd[k].device)
            for k in global_sd
        })
        del agg
        gc.collect()

        # Jain fairness: binary participation (1=active, 0=skipped)
        participations = [1.0 if m.get("active", True) else 0.0
                          for _, m, _ in client_updates]
        sum_p = sum(participations)
        jain = (sum_p ** 2) / (K_total * sum(p**2 for p in participations)) if sum_p > 0 else 0.0

        all_batteries = [s.battery_j for _, _, s in client_updates]

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": sum(m.get("bytes_sent", 0) for _, m, _ in active_updates),
                "total_energy_j": sum(m.get("energy_j_consumed", 0) for _, m, _ in active_updates),
                "avg_beta": 1.0,
                "avg_battery_j": sum(all_batteries) / K_total,
                "avg_local_loss": sum(m.get("local_loss", 0) for _, m, _ in active_updates) / K_active,
                "participation_rate": K_active / K_total,
                "jain_index": jain,
                "num_clients": K_total,
                "active_clients": K_active,
            },
        )

    def get_default_config(self):
        return {
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            "t_active_init": 0.5,        # initial threshold = 50% of B_max
            "threshold_decay": 0.995,    # annealing per round
            "recharge_rate_j": 500.0,    # Joules recovered per skipped round
            "battery_max_j": 185400.0,
            "device": "cpu",
            "device_profile": None,
        }
