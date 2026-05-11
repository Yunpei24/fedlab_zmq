"""
algorithms/vaishnav.py
======================
Vaishnav: Channel-Adaptive Error-Feedback Sparsification
Vaishnav et al., 2024

Key idea:
  Like E-CEFFL, uses TopK sparsification + error feedback.
  BUT the sparsification ratio beta is driven by the estimated
  CHANNEL QUALITY (SNR / available bandwidth) rather than battery.

  beta_t^k = f(SNR_t^k, BW_t^k)
           = clip(BW_t^k / BW_ref, beta_min, 1.0)

  where BW_t^k is the measured uplink bandwidth at round t.

  This is equivalent to: "compress more when the channel is bad."

In FedLab, we simulate channel quality variation via the
CommSpec of the DeviceProfile + round-level noise.

Difference vs E-CEFFL:
  Vaishnav: beta driven by CHANNEL (communication bottleneck).
  E-CEFFL:  beta driven by BATTERY (energy bottleneck).
  Both use EF. Hybrid (battery AND channel) = future direction.

Reference:
  Vaishnav et al. (2024) — channel-adaptive gradient sparsification
  in wireless FL.
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


def topk_sparsify(tensor: torch.Tensor, beta: float) -> torch.Tensor:
    k = max(1, int(beta * tensor.numel()))
    flat = tensor.flatten()
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).reshape(tensor.shape)


@register_algorithm("vaishnav")
class Vaishnav(FLAlgorithm):
    """
    Channel-adaptive TopK sparsification with error feedback.

    Key properties:
      - beta_t^k driven by simulated channel SNR / bandwidth
      - Error feedback: no gradient information permanently lost
      - beta_min > 0: universal participation guarantee
      - Models wireless channel variation (fading, congestion)
    """
    name = "vaishnav"
    description = (
        "Channel-adaptive EF sparsification (Vaishnav et al., 2024). "
        "Compression ratio driven by wireless channel quality."
    )

    def _simulate_channel_beta(
        self,
        profile,
        round_num: int,
        client_id: int,
        beta_min: float,
        bw_ref_mbps: float,
    ) -> float:
        """
        Simulate time-varying channel: returns beta ∈ [beta_min, 1.0].

        Model: BW_t^k = BW_nominal * (1 + channel_noise)
               beta_t^k = clip(BW_t^k / BW_ref, beta_min, 1.0)

        The channel noise follows a simple Markov fading model:
          noise ~ N(0, sigma^2), sigma varies by technology.
        """
        if profile is None:
            # Fallback: random walk
            rng = np.random.default_rng(seed=round_num * 97 + client_id * 31)
            noise = rng.normal(0, 0.3)
            beta = float(np.clip(0.5 + noise, beta_min, 1.0))
            return beta

        nominal_mbps = profile.comm.uplink_mbps
        # Channel noise std: higher for mobile/wireless technologies
        tech = profile.comm.technology
        sigma = {"WiFi6": 0.1, "WiFi5": 0.15, "WiFi4": 0.2,
                 "LTE": 0.3, "5G": 0.15, "BLE5": 0.4,
                 "LoRa": 0.5, "Ethernet10G": 0.01}.get(tech, 0.2)

        rng = np.random.default_rng(seed=round_num * 97 + client_id * 31)
        noise = rng.normal(0, sigma)
        bw_actual = nominal_mbps * (1.0 + noise)
        beta = float(np.clip(bw_actual / bw_ref_mbps, beta_min, 1.0))
        return beta

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
        beta_min = config.get("beta_min", 0.01)
        bw_ref_mbps = config.get("bw_ref_mbps", 50.0)   # reference bandwidth
        profile = config.get("device_profile")

        # ── Step 1: Estimate channel quality → beta ────────────────────────
        beta = self._simulate_channel_beta(
            profile, state.round_num, state.client_id, beta_min, bw_ref_mbps
        )

        # ── Step 2: Local training ─────────────────────────────────────────
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
        del w_before
        del current_sd
        gc.collect()

        # ── Step 3: Error-feedback TopK compression ────────────────────────
        if state.error_buffer is None:
            state.error_buffer = {k: torch.zeros_like(v) for k, v in delta.items()}

        compressed = OrderedDict()
        new_error = OrderedDict()
        for k in delta:
            u = lr * delta[k] + state.error_buffer[k]
            v = topk_sparsify(u, beta)
            new_error[k] = u - v
            compressed[k] = v
        state.error_buffer = new_error

        # ── Step 4: Energy accounting ──────────────────────────────────────
        uplink_bytes = self.count_bytes(compressed, sparse=True)
        downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs, len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * beta

        del delta
        gc.collect()

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta,
            "channel_beta": beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(steps, 1),
            "compression_ratio": beta,
        }
        return dict(compressed), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()
        agg = None
        for upd, _, _ in client_updates:
            if agg is None:
                agg = {k: v.float().clone() for k, v in upd.items()}
            else:
                for k in agg:
                    agg[k] += upd[k].float()

        new_weights = OrderedDict({
            k: global_sd[k].float() - (agg[k] / K).to(global_sd[k].device) for k in global_sd
        })
        del agg
        gc.collect()
        participations = [1.0] * K
        jain = (sum(participations)**2) / (K * sum(p**2 for p in participations))
        avg_beta = sum(m.get("beta_actual", 1.0) for _, m, _ in client_updates) / K

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": sum(m.get("bytes_sent", 0) for _, m, _ in client_updates),
                "total_energy_j": sum(m.get("energy_j_consumed", 0) for _, m, _ in client_updates),
                "avg_beta": avg_beta,
                "avg_battery_j": sum(s.battery_j for _, _, s in client_updates) / K,
                "avg_local_loss": sum(m.get("local_loss", 0) for _, m, _ in client_updates) / K,
                "participation_rate": 1.0,
                "jain_index": jain,
                "num_clients": K,
            },
        )

    def get_default_config(self):
        return {
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            "beta_min": 0.01,
            "bw_ref_mbps": 50.0,     # reference bandwidth (Mbps)
            "device": "cpu",
            "device_profile": None,
        }
