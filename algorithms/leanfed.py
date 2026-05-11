"""
algorithms/leanfed.py
=====================
LeanFed: Lean Federated Learning for Resource-Constrained Devices
Pereira et al., 2025

Key idea:
  Instead of compressing the gradient, LeanFed reduces the LOCAL DATASET
  used for training proportionally to the residual battery.
  A client with 50% battery uses only 50% of its local data.

  subsample_ratio_t^k = max(rho_min, B_t^k / B_max)
  D_used^k = subsample(D^k, subsample_ratio_t^k)

  No gradient compression → full-precision updates.
  Battery cost = full model uplink + proportional compute.

Difference vs E-CEFFL:
  LeanFed adapts DATA VOLUME (information quality degrades with battery).
  E-CEFFL adapts GRADIENT SPARSITY (information completeness degrades,
  but error feedback recovers it). LeanFed therefore accumulates a
  "data starvation" bias in non-IID settings.

Reference:
  Pereira et al. (2025) "LeanFed: ..."
  Eq. (2): E_round = P_idle * T_comp + c_tx * |w| * 4
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
import numpy as np

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("leanfed")
class LeanFed(FLAlgorithm):
    """
    LeanFed: battery-proportional data subsampling.

    Key properties:
      - No gradient compression (full-precision updates)
      - Participation guaranteed via rho_min > 0
      - Data quality degrades as battery depletes
      - Simple: no error buffers, no state beyond battery
    """
    name = "leanfed"
    description = (
        "Battery-proportional local dataset subsampling (Pereira et al., 2025). "
        "Full-precision gradients, reduced data volume."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)
        rho_min = config.get("rho_min", 0.1)       # minimum subsample ratio

        # ── Resolve battery_max_j from device profile (same as E-CEFFL) ───
        # Priority: device_profile.battery.capacity_j → config override → RPi4 default
        profile = config.get("device_profile")
        if profile is not None:
            battery_max_j = profile.battery.capacity_j
        else:
            battery_max_j = config.get("battery_max_j", 185400.0)

        # ── Step 1: Compute subsample ratio from battery ───────────────────
        rho = max(rho_min, min(1.0, state.battery_j / battery_max_j))

        # ── Step 2: Subsample local dataset ───────────────────────────────
        dataset = dataloader.dataset
        n_total = len(dataset)
        n_use = max(1, int(rho * n_total))

        rng = np.random.default_rng(seed=state.round_num + state.client_id * 1000)
        selected = rng.choice(n_total, size=n_use, replace=False).tolist()
        subset = Subset(dataset, selected)
        sub_loader = DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # ── Step 3: Save weights before training ──────────────────────────
        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        # ── Step 4: Local SGD on subsampled data ───────────────────────────
        model.train(); model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss, steps = 0.0, 0
        for _ in range(local_epochs):
            for x, y in sub_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item(); steps += 1

        # ── Step 5: Compute delta (full precision, no compression) ─────────
        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })

        # ── Step 6: Energy accounting ──────────────────────────────────────
        # profile already resolved above (for battery_max_j)
        uplink_bytes = self.count_bytes(delta, sparse=False)
        downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            # Compute FLOPs proportional to subsampled data
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs, n_use
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * rho  # rho=1 → full cost

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
            "beta_actual": 1.0,          # no gradient compression
            "subsample_ratio": rho,
            "samples_used": n_use,
            "samples_total": n_total,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(steps, 1),
            "compression_ratio": 1.0,
        }
        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        """
        Weighted average by samples_used (not dataset_size),
        since each client contributed proportionally less data.
        """
        global_sd = global_model.state_dict()
        total_samples = sum(
            m.get("samples_used", 1) for _, m, _ in client_updates
        )
        agg = None
        for update, meta, _ in client_updates:
            w = meta.get("samples_used", 1) / max(total_samples, 1)
            if agg is None:
                agg = {k: v.float() * w for k, v in update.items()}
            else:
                for k in agg:
                    agg[k] += update[k].float() * w

        new_weights = OrderedDict({
            k: global_sd[k].float() - agg[k].to(global_sd[k].device) for k in global_sd
        })
        del agg
        gc.collect()

        K = len(client_updates)
        batteries = [s.battery_j for _, _, s in client_updates]
        avg_rho = sum(m.get("subsample_ratio", 1.0) for _, m, _ in client_updates) / K
        # Participation rate: clients with battery > 0 participate (with rho_min floor)
        participations = [1.0] * K
        jain = (sum(participations) ** 2) / (K * sum(p**2 for p in participations))

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": sum(m.get("bytes_sent", 0) for _, m, _ in client_updates),
                "total_energy_j": sum(m.get("energy_j_consumed", 0) for _, m, _ in client_updates),
                "avg_beta": avg_rho,              # reuse beta slot for subsample_ratio
                "avg_battery_j": sum(batteries) / K,
                "avg_local_loss": sum(m.get("local_loss", 0) for _, m, _ in client_updates) / K,
                "participation_rate": sum(participations) / K,
                "jain_index": jain,
                "num_clients": K,
            },
        )

    def get_default_config(self):
        return {
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            "rho_min": 0.1,            # minimum fraction of data to use
            # battery_max_j is auto-resolved from device_profile.battery.capacity_j
            # when a profile is provided. This fallback is only used when no
            # device_profile is set (e.g. unit tests without hardware context).
            "battery_max_j": 185400.0,  # fallback: RPi4 10Ah @ 5.1V
            "device": "cpu",
            "device_profile": None,
        }
