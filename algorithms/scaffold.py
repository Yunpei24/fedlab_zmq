"""
algorithms/scaffold.py
======================
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
Karimireddy et al., ICML 2020 — https://arxiv.org/abs/1910.06378

Key idea:
  FedAvg suffers "client drift" — each client's local gradient points in
  a different direction than the global gradient due to data heterogeneity.
  SCAFFOLD introduces control variates (c_i for each client, c for server)
  to correct this drift:

    Local update: w_k^{j+1} = w_k^j - lr * (grad F_k(w_k^j) - c_i + c)

  After local training, the control variates are updated:
    c_i+ = c_i - c + (1 / (K * lr)) * (w_global - w_k_new)

  Server update: c += (1/N) * sum_k (c_i_new - c_i_old)

Algorithm (Option II from paper — used in practice):
  Client k at round t:
    1. Receive w_t and c from server
    2. Compute: c_i+ = c_i - c + (1/(K*lr)) * (w_t - w_k_E)
    3. Update: delta_c_i = c_i+ - c_i
    4. Send (w_t - w_k_E, delta_c_i) to server
  Server:
    1. w_{t+1} = w_t - eta * (1/K) * sum_k (w_t - w_k_E)  [with global_lr]
    2. c += (1/N) * sum_k delta_c_i

Key invariant: the control variate c reduces the variance of the
  gradient aggregation, enabling convergence even with E > 1 local epochs
  and arbitrary data heterogeneity.

Key difference vs FedAvg:
  SCAFFOLD eliminates client drift via variance reduction.
  Converges in O(1/T) rounds, independent of the degree of heterogeneity.
  Cost: 2x communication (model delta + control variate delta).

Key difference vs FedProx:
  FedProx bounds drift implicitly via mu-regularization.
  SCAFFOLD explicitly corrects the gradient direction.
  SCAFFOLD is more communication-expensive but faster convergence.

Key difference vs E-CEFFL:
  SCAFFOLD does not model energy / battery.
  Full-precision control variates + gradient updates.
  E-CEFFL would benefit from SCAFFOLD's variance reduction (future work).

Reference hyperparameters (Karimireddy et al. 2020):
  global_lr = 1.0, local_lr = 0.01, local_epochs = 5
  CIFAR-10: 100 rounds, 10 clients (from fedprox_per.py in T_Correction)
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import copy

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("scaffold")
class SCAFFOLD(FLAlgorithm):
    """
    SCAFFOLD: variance-reduced federated learning via control variates.

    Key properties:
      - Control variates c_i (client) and c (server) correct gradient direction
      - Converges with arbitrary E local steps, any data heterogeneity
      - 2x communication overhead vs FedAvg (model + control variate)
      - Full-precision updates
      - Server maintains global control variate c (aggregated from clients)
    """
    name = "scaffold"
    description = (
        "SCAFFOLD: stochastic controlled averaging with variance reduction "
        "(Karimireddy et al., ICML 2020). Corrects client drift in non-IID FL."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        device = config.get("device", "cpu")
        local_lr = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)
        profile = config.get("device_profile")

        # ── Step 1: Retrieve or initialize control variates ───────────────────
        # c_i : client-side control variate (stored in state.custom)
        # c   : global control variate (passed via config from server)
        num_params_list = []
        for p in model.parameters():
            num_params_list.append(p.numel())

        # Initialize c_i (client control variate) if first round
        if "scaffold_c_i" not in state.custom:
            state.custom["scaffold_c_i"] = {
                k: torch.zeros_like(v.cpu())
                for k, v in model.state_dict().items()
                if v.dtype.is_floating_point
            }

        c_i = {k: v.to(device) for k, v in state.custom["scaffold_c_i"].items()}

        # Global control variate c (sent by server in config)
        c_global_raw = config.get("scaffold_c_global", None)
        if c_global_raw is not None and isinstance(c_global_raw, dict):
            c_global = {k: v.to(device) for k, v in c_global_raw.items()}
        else:
            # First round or server hasn't set it yet: use zeros
            c_global = {k: torch.zeros_like(v) for k, v in c_i.items()}

        # ── Step 2: Save w_t (initial global weights) ─────────────────────────
        w_t = OrderedDict(
            {k: v.clone().detach() for k, v in model.state_dict().items()}
        )

        # ── Step 3: Local SGD with control variate correction ─────────────────
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                # Manually compute gradient with control variate correction
                # We use gradient clipping for numerical stability
                model.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()

                # Apply control variate correction: grad -= c_i - c
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in c_i:
                            param.grad.add_(c_global[name] - c_i[name])

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Manual SGD step (no momentum to match SCAFFOLD theory)
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.sub_(local_lr * param.grad)

                total_loss += loss.item()
                num_batches += 1

        # ── Step 4: Compute model delta (w_t - w_k_E) ────────────────────────
        w_k_E = model.state_dict()
        delta_w = OrderedDict({
            k: (w_t[k] - w_k_E[k].cpu()).float()
            for k in w_t
        })
        del w_t
        del w_k_E
        gc.collect()

        # ── Step 5: Update client control variate c_i+ ────────────────────────
        # c_i+ = c_i - c + (1 / (K * lr)) * (w_t - w_k_E)
        # where K = local_epochs * steps_per_epoch (total local steps)
        total_steps = local_epochs * max(len(dataloader), 1)
        step_factor = 1.0 / (total_steps * local_lr)

        new_c_i = {}
        delta_c_i = {}
        for k in c_i:
            if k in delta_w:
                # c_i+ = c_i - c_global + step_factor * (w_t - w_k_E)
                c_i_new = c_i[k] - c_global[k] + step_factor * delta_w[k].to(device)
                new_c_i[k] = c_i_new.cpu()
                delta_c_i[k] = (c_i_new - c_i[k]).cpu()
            else:
                new_c_i[k] = c_i[k].cpu()
                delta_c_i[k] = torch.zeros_like(c_i[k]).cpu()

        # Store updated c_i in client state
        state.custom["scaffold_c_i"] = new_c_i

        # ── Step 6: Build combined update ────────────────────────────────────
        # We send both delta_w and delta_c_i concatenated with a special prefix
        # Server will split them using the model's state_dict keys
        update = OrderedDict()
        for k in delta_w:
            update[k] = delta_w[k]
        for k in delta_c_i:
            update[f"__scaffold_c__{k}"] = delta_c_i[k]

        # ── Step 7: Energy accounting ─────────────────────────────────────────
        # Communication cost is 2x (model + control variate)
        uplink_bytes = self.count_bytes(delta_w, sparse=False) * 2
        downlink_bytes = self.count_bytes(w_t, sparse=False) * 2  # model + c

        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(
                num_params, dataloader.batch_size, local_epochs,
                len(dataloader.dataset)
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * 2.0  # 2x full transmission

        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

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
            "comm_overhead": "2x (model + control variate)",
        }
        return dict(update), metadata

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        SCAFFOLD server aggregation:
          w_{t+1} = w_t - global_lr * (1/K) * sum_k delta_w_k
          c += (1/N) * sum_k delta_c_i_k
        """
        global_lr = config.get("global_lr", 1.0)
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # Separate model deltas from control variate deltas
        model_keys = [k for k in global_sd.keys()]
        scaffold_prefix = "__scaffold_c__"

        # Accumulate model deltas
        agg_delta_w = None
        agg_delta_c = None

        for update, _, _ in client_updates:
            # Model delta
            delta_w = {k: v for k, v in update.items() if not k.startswith(scaffold_prefix)}
            delta_c = {k[len(scaffold_prefix):]: v for k, v in update.items()
                       if k.startswith(scaffold_prefix)}

            if agg_delta_w is None:
                agg_delta_w = {k: v.float().clone() for k, v in delta_w.items()}
                agg_delta_c = {k: v.float().clone() for k, v in delta_c.items()} if delta_c else {}
            else:
                for k in agg_delta_w:
                    if k in delta_w:
                        agg_delta_w[k] += delta_w[k].float()
                for k in agg_delta_c:
                    if k in delta_c:
                        agg_delta_c[k] += delta_c[k].float()

        # w_{t+1} = w_t - global_lr * (1/K) * sum delta_w
        new_weights = OrderedDict()
        for k in global_sd:
            if k in agg_delta_w:
                new_weights[k] = global_sd[k].float() - global_lr * agg_delta_w[k] / K
            else:
                new_weights[k] = global_sd[k].float()

        # Update global control variate c (stored in config for next round)
        # c += (1/N) * sum_k delta_c_i  where N = total clients
        # NOTE: in this simplified version N = K (full participation)
        # For partial participation: c += (1/N) * sum_k delta_c_i

        # We store the updated c_global in metrics for the next round
        updated_c_global = {}
        if agg_delta_c:
            # Retrieve current c_global from config (or initialize to zeros)
            current_c = config.get("scaffold_c_global", {})
            for k, delta in agg_delta_c.items():
                if k in current_c:
                    updated_c_global[k] = current_c[k].float() + delta / K
                else:
                    updated_c_global[k] = delta / K

        del agg_delta_w
        del agg_delta_c
        gc.collect()

        total_bytes = sum(m.get("bytes_sent", 0) for _, m, _ in client_updates)
        total_energy = sum(m.get("energy_j_consumed", 0) for _, m, _ in client_updates)
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / K
        avg_loss = sum(m.get("local_loss", 0) for _, m, _ in client_updates) / K

        participations = [1.0] * K
        jain = (sum(participations) ** 2) / (K * sum(p ** 2 for p in participations))

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num,
                "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy,
                "avg_beta": 1.0,
                "avg_battery_j": avg_battery,
                "avg_local_loss": avg_loss,
                "participation_rate": 1.0,
                "jain_index": jain,
                "num_clients": K,
                # Pass updated c_global to next round via algo_config
                # NOTE: In ZMQ framework this would need to be broadcast
                # via TRAIN_REQ. In standalone mode, handled by run_experiment.py.
                "_scaffold_c_global_update": updated_c_global,
            },
        )

    def get_default_config(self) -> dict:
        return {
            "lr": 0.01,               # local learning rate
            "global_lr": 1.0,         # server global learning rate (1.0 from paper)
            "local_epochs": 1,
            "batch_size": 32,
            "device": "cpu",
            "device_profile": None,
            "scaffold_c_global": None,  # initialized to zeros on first use
        }
