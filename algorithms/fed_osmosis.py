"""
algorithms/fed_osmosis.py
==========================
Fed-Osmosis: Thermodynamic Distribution Alignment for Federated Learning
J. Nikiema, EL Amhoud, H. ELHAMMOUTI & I. KISSAMI — UM6P, 2026

Physical analogy
----------------
In osmosis, two solutions separated by a semi-permeable membrane exchange
solvent until their chemical potentials equalise. In FL with non-IID data,
each client has a local activation distribution p_k(z^l) that diverges from
the global reference p_global(z^l). The osmotic pressure Π_k^l measures this
divergence and penalises it during local training — pulling client activations
toward the global reference without sharing raw data.

Algorithm
---------
Server → Client (each round):
  - Global model weights w_t
  - Global activation statistics {μ_g^l, alpha_g^l} for each monitored layer l

Client update:
  - Register forward hooks on Linear / Conv2d layers
  - For each mini-batch (x, y):
      out = model(x)
      L_task  = CrossEntropy(out, y)
      L_osm   = gamma · Σ_l KL(N(μ_k^l, alpha_k²^l) ‖ N(μ_g^l, alpha_g²^l))
      L_total = L_task + L_osm
      L_total.backward()
  - Collect post-training activation statistics (μ_k^l, alpha_k^l) for each layer
  - Send: dense gradient update  +  local stats

Server aggregate:
  - FedAvg on weights (dataset-size weighted)
  - Weighted average of local stats → new global stats for next round

Osmotic pressure (Gaussian KL):
  Π_k^l = KL(N(μ_k,alpha_k²) ‖ N(μ_g,alpha_g²))
         = log(alpha_g/alpha_k) + (alpha_k² + (μ_k - μ_g)²) / (2alpha_g²)  -  1/2

Convergence sketch
------------------
Under L-smoothness and ‖∇Π‖ ≤ M:
  E[‖∇f(w_T)‖²] ≤ 2Δ₀/(ηT)  +  Lηalpha²/K  +  LηgammaM
For gamma → 0 this recovers the standard FedAvg rate O(1/√T).
gamma controls the accuracy - alignment tradeoff.

Communication overhead
----------------------
  Uplink  : d * 4 bytes (dense fp32 gradient) + 2·|L| * 4 bytes (local stats)
  Downlink: d * 4 bytes (model) + 2·|L| * 4 bytes (global stats)  ← negligible
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Optional

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


# ─────────────────────────────────────────────────────────────────────────────
# Osmotic pressure helpers
# ─────────────────────────────────────────────────────────────────────────────

_MONITORED_TYPES = (nn.Linear, nn.Conv2d)


def _osmotic_pressure(
    captured: dict[str, torch.Tensor],
    global_stats: dict[str, dict],
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the mean osmotic pressure over all monitored layers.

    Uses the Gaussian KL divergence between the per-batch activation
    distribution N(μ_k, alpha_k²) and the global reference N(μ_g, alpha_g²).

    The result is differentiable w.r.t. model parameters because `captured`
    holds the raw (non-detached) activation tensors from the forward pass.

    Args:
        captured    : {layer_name: activation_tensor} from forward hooks.
        global_stats: {layer_name: {"mu": float, "sigma": float}} from server.
        eps         : numerical floor for alpha to avoid log(0).

    Returns:
        Scalar tensor ≥ 0, differentiable.
    """
    pressure = torch.zeros(1, dtype=torch.float32)
    n_active = 0

    for name, act in captured.items():
        if name not in global_stats:
            continue

        gs      = global_stats[name]
        mu_g    = float(gs.get("mu", 0.0))
        sigma_g = max(float(gs.get("sigma", 1.0)), eps)

        # Local stats from current batch — differentiable
        flat    = act.float().reshape(act.shape[0], -1)   # (B, features)
        mu_k    = flat.mean()
        sigma_k = flat.std(unbiased=False).clamp(min=eps)

        # KL(N(μ_k,σ_k²) ‖ N(μ_g,σ_g²))
        kl = (
            torch.log(torch.tensor(sigma_g, dtype=torch.float32) / sigma_k)
            + (sigma_k ** 2 + (mu_k - mu_g) ** 2) / (2.0 * sigma_g ** 2)
            - 0.5
        ).clamp(min=0.0)

        pressure = pressure + kl
        n_active += 1

    return pressure / max(n_active, 1)


class _StatAccumulator:
    """
    Lightweight accumulator to collect per-layer activation statistics
    (mean, std) over an entire local training run (multiple batches/epochs).

    Uses Welford-style online update: accumulates sum and sum-of-squares,
    then computes μ and σ at the end.
    """

    def __init__(self):
        self._sums:    dict[str, torch.Tensor] = {}
        self._sum_sqs: dict[str, torch.Tensor] = {}
        self._counts:  dict[str, int]          = {}
        self._hooks = []
        self._names: list[str] = []

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, _MONITORED_TYPES) and name:
                self._names.append(name)
                self._sums[name]    = None
                self._sum_sqs[name] = None
                self._counts[name]  = 0

                def _hook(mod, inp, out, _n=name):
                    flat = out.detach().float().reshape(out.shape[0], -1)
                    batch_sum    = flat.sum(0)
                    batch_sum_sq = flat.pow(2).sum(0)
                    n = flat.shape[0]
                    if self._sums[_n] is None:
                        self._sums[_n]    = batch_sum
                        self._sum_sqs[_n] = batch_sum_sq
                    else:
                        self._sums[_n]    = self._sums[_n]    + batch_sum
                        self._sum_sqs[_n] = self._sum_sqs[_n] + batch_sum_sq
                    self._counts[_n] += n

                self._hooks.append(module.register_forward_hook(_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_stats(self) -> dict[str, dict]:
        result = {}
        for name in self._names:
            n = self._counts[name]
            if n > 0 and self._sums[name] is not None:
                mu_vec  = self._sums[name] / n
                sq_vec  = self._sum_sqs[name] / n
                var_vec = (sq_vec - mu_vec.pow(2)).clamp(min=0.0)
                result[name] = {
                    "mu":    mu_vec.mean().item(),
                    "sigma": var_vec.mean().sqrt().item(),
                    "n":     n,
                }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fed_osmosis")
class FedOsmosis(FLAlgorithm):
    """
    Fed-Osmosis: Thermodynamic Distribution Alignment for FL.

    Key properties:
      - Regularises local training via osmotic pressure (KL divergence
        between local and global activation distributions)
      - No architecture change: works with any nn.Module
      - Communication overhead over FedAvg: 2·|layers| floats per direction
        (negligible vs. the gradient payload)
      - gamma = 0 recovers exact FedAvg behaviour
      - Convergence: O(1/√T) for gamma small enough

    State stored in ClientState.custom:
      - "local_stats": dict[layer_name → {"mu": float, "sigma": float}]
        (stats from last round, sent to server for global aggregation)

    Config keys specific to Fed-Osmosis:
      - gamma             : osmotic pressure weight (default 0.1)
      - osmosis_global_stats : dict passed by server each round
      - monitored_layers  : list of layer names to monitor (None = all Linear/Conv2d)
    """

    name        = "fed_osmosis"
    description = (
        "Thermodynamic distribution alignment via osmotic pressure regularisation. "
        "Reduces non-IID bias by pulling local activation distributions toward "
        "the global reference. Nikiema, Amhoud, Elhammouti & Kissami, UM6P 2026."
    )

    # ── client_update ────────────────────────────────────────────────────────

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        Local training with osmotic pressure regularisation.

        Steps:
          1. Recover global stats from config (broadcast by server).
          2. Save w_before.
          3. Train with L = L_task + gamma * Π (hooks on Linear/Conv2d).
          4. Accumulate local activation stats (separate stat hooks).
          5. Compute dense gradient delta = w_before - w_after.
          6. Energy accounting.
          7. Return (delta, metadata) with local_stats in metadata.
        """

        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        local_epochs  = config.get("local_epochs", 1)
        gamma         = float(config.get("gamma", 0.1))
        battery_max_j = config.get("battery_max_j", 185400.0)
        profile       = config.get("device_profile")
        global_stats  = config.get("osmosis_global_stats", {})   # from server
        dataset_size  = len(dataloader.dataset)
        batch_size    = dataloader.batch_size

        # ── Step 1: Battery (informational, no compression in Osmosis) ──────
        battery_j = state.battery_j
        beta_proxy = battery_j / battery_max_j   # used only for energy model

        # ── Step 2: Save w_before ────────────────────────────────────────────
        w_before = OrderedDict(
            {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 3 & 4: Training with osmotic regularisation ─────────────────
        model.train()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        # Stat accumulator (detached, for server reporting)
        stat_acc = _StatAccumulator()
        stat_acc.register(model)

        total_loss  = 0.0
        osm_total   = 0.0
        num_batches = 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                # ── Osmotic hooks (non-detached, for differentiable KL) ──────
                captured: dict[str, torch.Tensor] = {}
                osm_hooks = []

                if global_stats:
                    for name, module in model.named_modules():
                        if isinstance(module, _MONITORED_TYPES) and name in global_stats:
                            def _make(n):
                                def _h(mod, inp, out, _n=n):
                                    captured[_n] = out
                                return _h
                            osm_hooks.append(
                                module.register_forward_hook(_make(name))
                            )

                # ── Forward pass ─────────────────────────────────────────────
                optimizer.zero_grad()
                out       = model(x)
                task_loss = criterion(out, y)

                # Remove osmotic hooks (captured already filled)
                for h in osm_hooks:
                    h.remove()

                # ── Osmotic pressure term ────────────────────────────────────
                if global_stats and captured:
                    o_loss = _osmotic_pressure(captured, global_stats)
                    loss   = task_loss + gamma * o_loss
                    osm_total += o_loss.item()
                else:
                    loss = task_loss

                loss.backward()
                optimizer.step()

                total_loss  += task_loss.item()
                num_batches += 1

        # ── Remove stat hooks and collect local stats ────────────────────────
        stat_acc.remove()
        local_stats = stat_acc.get_stats()
        state.custom["local_stats"] = local_stats

        # ── Step 5: Dense gradient delta ─────────────────────────────────────
        w_after = model.state_dict()
        delta   = OrderedDict({
            k: (w_before[k] - w_after[k].cpu()).float()
            for k in w_before
        })
        del w_before
        del w_after
        gc.collect()

        # ── Step 6: Energy accounting ─────────────────────────────────────────
        num_params     = sum(p.numel() for p in model.parameters())
        uplink_bytes   = num_params * 4    # dense fp32
        downlink_bytes = num_params * 4

        if profile is not None:
            flops    = profile.flops_for_model(
                num_params, batch_size, local_epochs, dataset_size
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * 1.0   # full dense transmission

        state.battery_j  = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        avg_osm = osm_total / max(num_batches, 1)

        metadata = {
            "client_id":           state.client_id,
            "round_num":           state.round_num,
            "beta_actual":         beta_proxy,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":   energy_j,
            "bytes_sent":          uplink_bytes,
            "bytes_received":      downlink_bytes,
            "local_loss":          total_loss / max(num_batches, 1),
            "osmotic_pressure":    avg_osm,
            "compression_ratio":   1.0,    # Osmosis sends dense updates
            "local_stats":         local_stats,
            "dataset_size":        dataset_size,
        }

        return dict(delta), metadata

    # ── server_aggregate ──────────────────────────────────────────────────────

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Server-side aggregation:
          1. FedAvg on gradients (dataset-size weighted).
          2. Compute new global activation stats = weighted average of local stats.
             These will be broadcast next round via config["osmosis_global_stats"].

        The new global_stats are stored in metrics["osmosis_global_stats"] so
        run_experiment.py / server.py can pass them back to clients next round.
        """

        K         = len(client_updates)
        global_sd = global_model.state_dict()

        sizes   = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = sum(sizes)
        weights = [n_k / total_n for n_k in sizes]

        # ── Weighted gradient accumulation ───────────────────────────────────
        agg: Optional[OrderedDict] = None
        for (update, _, _), w_k in zip(client_updates, weights):
            if agg is None:
                agg = OrderedDict({k: v.float() * w_k for k, v in update.items()})
            else:
                for k in update:
                    agg[k] = agg[k] + update[k].float() * w_k

        new_weights = OrderedDict()
        for k in global_sd:
            new_weights[k] = (
                global_sd[k].float() - agg[k].to(global_sd[k].device)
                if (agg is not None and k in agg)
                else global_sd[k].float()
            )
        del agg
        gc.collect()

        # ── Aggregate activation stats ────────────────────────────────────────
        # Weighted average: μ_g^l = Σ_k (n_k/n) μ_k^l
        #                   σ_g^l from pooled variance formula
        layer_mu_accum:    dict[str, float] = {}
        layer_var_accum:   dict[str, float] = {}
        layer_n_accum:     dict[str, int]   = {}

        for (_, metadata, _), w_k in zip(client_updates, weights):
            local_stats = metadata.get("local_stats", {})
            for layer, st in local_stats.items():
                mu_k    = st.get("mu", 0.0)
                sigma_k = st.get("sigma", 1.0)
                n_k     = st.get("n", 1)
                if layer not in layer_mu_accum:
                    layer_mu_accum[layer]  = 0.0
                    layer_var_accum[layer] = 0.0
                    layer_n_accum[layer]   = 0
                layer_mu_accum[layer]  += w_k * mu_k
                layer_var_accum[layer] += w_k * sigma_k ** 2
                layer_n_accum[layer]   += n_k

        new_global_stats = {
            layer: {
                "mu":    layer_mu_accum[layer],
                "sigma": max(layer_var_accum[layer], 1e-6) ** 0.5,
            }
            for layer in layer_mu_accum
        }

        # ── Round-level metrics ───────────────────────────────────────────────
        total_bytes  = sum(m["bytes_sent"]       for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery  = sum(s.battery_j           for _, _, s in client_updates) / K
        avg_loss     = sum(m["local_loss"]        for _, m, _ in client_updates) / K
        avg_pressure = sum(m.get("osmotic_pressure", 0.0) for _, m, _ in client_updates) / K

        jain = 1.0   # all clients participate

        metrics = {
            "round":                   round_num,
            "total_bytes_sent":        total_bytes,
            "total_energy_j":          total_energy,
            "avg_beta":                1.0,          # no compression
            "avg_battery_j":           avg_battery,
            "avg_local_loss":          avg_loss,
            "avg_osmotic_pressure":    avg_pressure,
            "osmosis_global_stats":    new_global_stats,  # broadcast next round
            "participation_rate":      1.0,
            "jain_index":              jain,
            "num_clients":             K,
        }

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    # ── get_default_config ────────────────────────────────────────────────────

    def get_default_config(self) -> dict:
        return {
            # Training
            "lr":                   0.01,
            "local_epochs":         1,
            "batch_size":           32,
            # Osmosis
            "gamma":                0.1,   # osmotic pressure weight
            "osmosis_global_stats": {},    # populated by server after round 0
            # Battery (informational — Osmosis sends dense, no compression)
            "battery_max_j":        185400.0,
            # Device
            "device":               "cpu",
            "device_profile":       None,
        }
