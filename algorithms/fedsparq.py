"""
algorithms/fedsparq.py
======================
FedSparQ: Adaptive Sparse Quantization with Error Feedback
Medjadji et al., 2025

Key idea:
  Three-stage compression pipeline:
  1. Adaptive threshold sparsification:
       theta_t^k = EMA(|g_t^k|, alpha)   (layer-wise exponential mean)
       mask_t^k  = |g_t^k| > theta_t^k
       v_sparse  = g_t^k * mask_t^k
  2. Float16 quantization of retained values:
       v_quant  = quantize_fp16(v_sparse)
  3. Error feedback on residuals:
       e_{t+1}^k = g_t^k - v_quant + e_t^k   (accumulated error)

  The threshold adapts to the gradient distribution each round,
  independently of battery or channel state.

Communication cost:
  Bytes = nnz * (2 bytes fp16 + 4 bytes index)  [sparse fp16 format]
  vs FedAvg: num_params * 4 bytes [dense fp32]

Key difference vs E-CEFFL:
  FedSparQ adapts to GRADIENT DISTRIBUTION (signal-driven threshold).
  E-CEFFL adapts to BATTERY LEVEL (energy-driven ratio).
  FedSparQ does not model or bound device energy depletion.

Reference:
  Medjadji et al. (2025) "FedSparQ: Adaptive Sparse Quantization
  with Error Feedback for Federated Learning"
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


def _flatten_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all parameters into a single flat tensor."""
    return torch.cat([p.data.cpu().float().flatten() for p in model.parameters()])


def _unflatten_params(flat: torch.Tensor, reference: dict) -> dict:
    """Reshape flat tensor back into state_dict structure."""
    result = OrderedDict()
    offset = 0
    for k, v in reference.items():
        numel = v.numel()
        result[k] = flat[offset: offset + numel].reshape(v.shape)
        offset += numel
    return result


class LayerwiseEMAThreshold:
    """
    Layer-wise exponential moving average threshold for FedSparQ.
    theta_t = alpha * theta_{t-1} + (1 - alpha) * |g_t|.mean()
    """
    def __init__(self, model: nn.Module, alpha: float = 0.9):
        self.alpha = alpha
        self.thresholds = {
            name: torch.tensor(0.0)
            for name, _ in model.named_parameters()
        }

    def update_and_get_mask(
        self, named_grads: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Update EMA thresholds and return binary masks.
        Returns dict of {layer_name: mask_tensor}
        """
        masks = {}
        for name, grad in named_grads.items():
            grad_abs_mean = grad.abs().mean().item()
            old_theta = self.thresholds[name].item()
            new_theta = self.alpha * old_theta + (1 - self.alpha) * grad_abs_mean
            self.thresholds[name] = torch.tensor(new_theta)
            masks[name] = (grad.abs() > new_theta).float()
        return masks

    def state_dict(self) -> dict:
        return {k: v.item() for k, v in self.thresholds.items()}

    def load_state_dict(self, d: dict):
        for k, v in d.items():
            if k in self.thresholds:
                self.thresholds[k] = torch.tensor(v)


@register_algorithm("fedsparq")
class FedSparQ(FLAlgorithm):
    """
    FedSparQ: adaptive sparse-fp16 + error feedback.

    Key properties:
      - Layer-wise EMA threshold: adapts to gradient distribution
      - Float16 quantization of retained coordinates (2x memory saving)
      - Error feedback residuals prevent convergence bias
      - Up to 90% communication reduction on standard benchmarks
    """
    name = "fedsparq"
    description = (
        "Adaptive sparse fp16 quantization + EF (Medjadji et al., 2025). "
        "Gradient-distribution-driven compression threshold."
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
        ema_alpha = config.get("ema_alpha", 0.9)      # EMA smoothing factor
        profile = config.get("device_profile")

        # ── Dead-client guard ──────────────────────────────────────────────
        if state.battery_j <= 0:
            dataset_size = len(dataloader.dataset) if dataloader is not None else 1
            return {}, {
                "client_id":           state.client_id,
                "round_num":           state.round_num,
                "beta_actual":         0.0,
                "nnz_ratio":           0.0,
                "battery_j_remaining": 0.0,
                "energy_j_consumed":   0.0,
                "bytes_sent":          0,
                "bytes_received":      0,
                "local_loss":          0.0,
                "compression_ratio":   0.0,
                "quantization":        "fp16",
                "dataset_size":        dataset_size,
                "skipped":             True,
            }

        # ── Step 1: Initialize EMA threshold tracker ───────────────────────
        if "ema_thresholds" not in state.custom:
            state.custom["ema_thresholds"] = {
                name: 0.0 for name, _ in model.named_parameters()
            }
        ema_thresholds = state.custom["ema_thresholds"]

        # ── Step 2: Save weights before training ───────────────────────────
        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        # ── Step 3: Local SGD training ─────────────────────────────────────
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

        # ── Step 4: Compute gradient delta ────────────────────────────────
        w_after = model.state_dict()
        named_delta = OrderedDict({
            name: (w_before[name] - w_after[name].cpu()).float()
            for name in w_before
        })
        # Compute downlink size before deleting w_before (fix: was used after del)
        downlink_bytes = self.count_bytes(w_before, sparse=False)
        dataset_size = len(dataloader.dataset)
        del w_before
        del w_after
        gc.collect()

        # ── Step 5: Update EMA thresholds and compute masks ───────────────
        masks = OrderedDict()
        for name, delta in named_delta.items():
            g_abs_mean = delta.abs().mean().item()
            old_theta = ema_thresholds.get(name, 0.0)
            new_theta = ema_alpha * old_theta + (1 - ema_alpha) * g_abs_mean
            ema_thresholds[name] = new_theta
            masks[name] = (delta.abs() > new_theta).float()
        state.custom["ema_thresholds"] = ema_thresholds

        # ── Step 6: Apply error feedback + sparse mask ────────────────────
        if state.error_buffer is None:
            state.error_buffer = {k: torch.zeros_like(v) for k, v in named_delta.items()}

        compressed_fp16 = OrderedDict()
        new_error = OrderedDict()
        total_nnz = 0

        for name in named_delta:
            # Add error feedback to current gradient (no extra lr: named_delta already contains lr via SGD)
            u = named_delta[name] + state.error_buffer[name]
            # Apply threshold mask
            v_sparse = u * masks[name]
            # Quantize retained coordinates to fp16, keep as float32 for computation
            v_fp16 = v_sparse.half().float()   # fp16 round-trip → float32
            # Residual: quantization error on retained coords only (dropped coords = 0)
            # Matches paper Algorithm 1 line 10: r[I] = c[I] - float32(q), else 0
            new_error[name] = (u - v_fp16) * masks[name]
            compressed_fp16[name] = v_fp16
            total_nnz += masks[name].sum().item()

        state.error_buffer = new_error

        # ── Step 7: Compute sparsification ratio ──────────────────────────
        total_params = sum(v.numel() for v in named_delta.values())
        actual_beta = total_nnz / max(total_params, 1)

        # ── Step 8: Energy accounting ──────────────────────────────────────
        # FedSparQ bytes: each retained param uses 2B (fp16) + 4B (index)
        uplink_bytes = int(total_nnz * 6)
        # downlink_bytes already computed before del w_before

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs, len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * actual_beta

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        metadata = {
            "client_id":           state.client_id,
            "round_num":           state.round_num,
            "beta_actual":         actual_beta,
            "nnz_ratio":           actual_beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":   energy_j,
            "bytes_sent":          uplink_bytes,
            "bytes_received":      downlink_bytes,
            "local_loss":          total_loss / max(steps, 1),
            "compression_ratio":   actual_beta,
            "quantization":        "fp16",
            "dataset_size":        dataset_size,
        }
        return dict(compressed_fp16), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Separate alive from dead clients ──────────────────────────────
        active_updates = [
            (u, m, s) for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]
        K_act = max(len(active_updates), 1)

        # FedAvg weights over active clients only
        sizes = [m.get("dataset_size", 1) for _, m, _ in active_updates]
        total_n = max(sum(sizes), 1)
        weights = [n_k / total_n for n_k in sizes]

        agg = None
        for (upd, _, _), w_k in zip(active_updates, weights):
            if agg is None:
                agg = {k: v.float().clone() * w_k for k, v in upd.items()}
            else:
                for k in agg:
                    if k in upd:
                        agg[k] += upd[k].float() * w_k

        new_weights = OrderedDict()
        for k in global_sd:
            if agg is not None and k in agg:
                new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
            else:
                new_weights[k] = global_sd[k].float()
        del agg
        gc.collect()

        # Jain fairness index on bytes_sent (dead clients = 0 → penalisés)
        bytes_sent = [m["bytes_sent"] for _, m, _ in client_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b ** 2 for b in bytes_sent))
        else:
            jain = 1.0

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round":              round_num,
                "total_bytes_sent":   sum(m.get("bytes_sent", 0)        for _, m, _ in client_updates),
                "total_energy_j":     sum(m.get("energy_j_consumed", 0) for _, m, _ in client_updates),
                "avg_beta":           sum(m.get("beta_actual", 0.0)      for _, m, _ in active_updates) / K_act,
                "avg_battery_j":      sum(s.battery_j                   for _, _, s in client_updates) / K,
                "batt_min_j":         min(s.battery_j                   for _, _, s in client_updates),
                "avg_local_loss":     sum(m.get("local_loss", 0)        for _, m, _ in active_updates) / K_act,
                "compression_ratio":  sum(m.get("compression_ratio", 0) for _, m, _ in active_updates) / K_act,
                "participation_rate": len(active_updates) / K,
                "jain_index":         jain,
                "num_clients":        K,
                "skipped_clients":    K - len(active_updates),
            },
        )

    def get_default_config(self):
        return {
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            "ema_alpha": 0.1,          # EMA smoothing factor for threshold
            "device": "cpu",
            "device_profile": None,
        }
