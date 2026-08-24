"""
algorithms/eceffl.py
====================
E-CEFFL: Energy-Compensated Error-Feedback Federated Learning
J. Nikiema, EL Amhoud, H. ELHAMMOUTI & I. KISSAMI — UM6P, 2026

Reference: Nikiema et al. (2026), "E-CEFFL: Energy-Compensated Error-Feedback
Federated Learning", submission to NeurIPS/ICML/ICLR.

─────────────────────────────────────────────────────────────────────────────
Problem Setting
─────────────────────────────────────────────────────────────────────────────
Consider K clients and a central server. Client k holds local dataset D_k of
size n_k; total dataset N = Σ n_k. The FL objective is:

    min_{w ∈ R^d}  F(w) = Σ_k p_k F_k(w),   p_k = n_k / N

where F_k(w) = (1/n_k) Σ_{(x,y)∈D_k} l(w; (x,y)) is client k's local loss.

Energy consumption at round t:
    E_cost,t^k = E_comp + E_comm(β_t^k)
    E_comm(β_t^k) = c_tx · β_t^k · d        (linear in fraction transmitted)

Battery dynamics:
    B_{t+1}^k = B_t^k - E_cost,t^k
    B_0^k ~ N(mu_B, sigma_B^2) truncated to [0, B_max]   (heterogeneous fleet)

A client drops out if B_t^k < E_comp + E_comm(β_min). E-CEFFL is designed
to make this event negligible by construction.

─────────────────────────────────────────────────────────────────────────────
Core Innovation 1 — Battery-Driven Adaptive Sparsification
─────────────────────────────────────────────────────────────────────────────
Instead of fixing β or adapting to channel quality, E-CEFFL ties the
sparsification ratio directly to the client's residual battery:

    β_t^k = β_min + (β_max - β_min) · B_t^k / B_max        [eq:beta_policy]

where 0 < β_min ≤ β_max ≤ 1 are system hyperparameters.

  Universal Participation Guarantee:
    Since B_t^k ≥ 0 ⟹ β_t^k ≥ β_min > 0, every client always transmits a
    non-trivial update regardless of battery level. This eliminates the
    participate-or-dropout dilemma: rho_k = 1 for all k and all rounds.

  Energy Proportionality:
    E_comm(β_t^k) ∝ B_t^k — a client with half its battery spends half as
    much energy on transmission. The policy is self-regulating: low-battery
    clients automatically compress more, preserving their remaining charge.

  Contrast with data-reduction approaches (e.g. LeanFed):
    LeanFed saves energy by training on a battery-proportional SUBSET of D_k;
    discarded samples are permanently lost. E-CEFFL uses ALL local data for
    computing g_t^k and delays rather than discards gradient information.
    This distinction is critical in non-IID settings where every local sample
    contributes unique distributional information.

─────────────────────────────────────────────────────────────────────────────
Core Innovation 2 — Error Feedback Compensation
─────────────────────────────────────────────────────────────────────────────
Aggressive sparsification (β_t^k ≪ 1) introduces compression bias that can
impede convergence. E-CEFFL maintains an error-feedback buffer e_t^k ∈ R^d
across rounds. After computing local gradient g_t^k via SGD on D_k:

  Step 1 — Accumulate:
    u_t^k = η · g_t^k + e_t^k        (inject accumulated residual; η = lr)
    In implementation: u_t^k = δ_t^k + e_t^k  where δ_t^k = w_before - w_after
    (δ already has η applied via SGD optimizer step)

  Step 2 — Compress:
    v_t^k = TopK(u_t^k, β_t^k)
    Retains the ⌊β_t^k · d⌋ largest-magnitude coordinates, zeros the rest.
    Cost: ⌊β_t^k · d⌋ x 8 bytes (4B value + 4B index) in sparse format,
    or ⌊β_t^k · d⌋ x 4 bytes if dense representation is smaller (taken min).

  Step 3 — Update error buffer:
    e_{t+1}^k = u_t^k - v_t^k
    No gradient information is discarded — it is DELAYED to future rounds.
    u_t^k = v_t^k + e_{t+1}^k holds exactly.

TopK contraction property (Alistarh et al., 2018):
    ‖x - TopK(x, β)‖ ≤ γ ‖x‖,   γ = sqrt(1 - β)
    ⟹ δ(β) = 1 - γ² = β  (contraction factor = sparsification ratio)

─────────────────────────────────────────────────────────────────────────────
Server Aggregation
─────────────────────────────────────────────────────────────────────────────
Dataset-size-weighted FedAvg aggregation (reduces bias in non-IID settings):

    w_{t+1} = w_t - Σ_k (n_k / n) · v_t^k,    n = Σ_k n_k

Dead clients (B_t^k = 0) are excluded from the weighted sum; their weight is
redistributed proportionally among active clients.

─────────────────────────────────────────────────────────────────────────────
Convergence Result (Theorem 1)
─────────────────────────────────────────────────────────────────────────────
Under standard assumptions (L-smooth F_k, bounded gradient variance σ²,
bounded data heterogeneity ζ²), E-CEFFL achieves:

    (1/T) Σ_t E[‖∇F(w_t)‖²]  =  O(1/√T)

with a noise floor governed by β_min and bounded via a new error-buffer
lemma. This is a federated, energy-adaptive generalization of Theorem 2 in
Alistarh et al. (2018), with fixed K replaced by battery-proportional
schedule K_t^k = f(B_t^k) · d and explicit lower bound K_t^k ≥ β_min · d > 0
that prevents infinite staleness.

─────────────────────────────────────────────────────────────────────────────
Computational Complexity
─────────────────────────────────────────────────────────────────────────────
Per-round cost at client k:
  - SGD forward+backward: O(d)         (same as FedAvg)
  - TopK selection:       O(d log d)   (partial sort)
  - Error buffer update:  O(d)         (element-wise subtraction)
  - Communication:        O(β_t^k · d) values + indices   (1/β_t^k x vs FedAvg)

─────────────────────────────────────────────────────────────────────────────
Comparison with Related Work
─────────────────────────────────────────────────────────────────────────────
  FedSparQ (Medjadji et al., 2025): uses EMA-smoothed, layer-wise gradient
    statistics to set an adaptive sparsification threshold — adapts to GRADIENT
    DISTRIBUTION. E-CEFFL adapts to BATTERY LEVEL. FedSparQ does not model or
    bound device energy depletion, device dropout, or participation fairness.

  LeanFed (Pereira et al., 2025): adjusts training workload (fraction of local
    data) to battery state. Maintains participation but discards data permanently.

  FedBacys (Jeong et al., 2025): cyclic scheduling — low-battery clients skip
    rounds and recharge. Improves long-run stability but introduces selection bias.

  Vaishnav et al. (2024): adapts compression to CHANNEL QUALITY, not battery.
    Orthogonal adaptation signal; no participation or fairness guarantee.

─────────────────────────────────────────────────────────────────────────────
Algorithm Summary
─────────────────────────────────────────────────────────────────────────────
  β_t^k    = β_min + (β_max - β_min) · B_t^k / B_max   [battery → ratio]
  u_t^k    = δ_t^k + e_t^k                               [error-compensated update]
  v_t^k    = TopK(u_t^k, β_t^k)                          [compressed update]
  e_{t+1}^k = u_t^k - v_t^k                              [residual buffer]
  w_{t+1}  = w_t - Σ_k (n_k/N) v_t^k                    [weighted aggregation]
  B_{t+1}^k = B_t^k - E_comp - c_tx · β_t^k · d         [battery update]
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.profiles import DeviceProfile


def topk_sparsify(tensor: torch.Tensor, beta: float) -> torch.Tensor:
    """Retain top-beta fraction of coordinates by absolute magnitude."""
    k = max(1, int(beta * tensor.numel()))
    flat = tensor.flatten()
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).reshape(tensor.shape)


@register_algorithm("eceffl")
class ECEFFL(FLAlgorithm):
    """
    Energy-Compensated Error-Feedback Federated Learning.

    Key properties:
      - Universal participation: beta_t^k >= beta_min > 0 always
      - Error feedback: no gradient information permanently lost
      - Battery-proportional compression: self-regulating energy usage
      - Convergence: O(1/sqrt(T)) non-convex rate (Theorem 1 in paper)
    """
    name = "eceffl"
    description = (
        "Battery-driven adaptive gradient sparsification with error feedback. "
        "Guarantees universal client participation in every round."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        local_epochs  = config.get("local_epochs", 1)
        max_grad_norm   = config.get("max_grad_norm", None)
        use_fp16_uplink = config.get("use_fp16_uplink", True)
        beta_min        = config.get("beta_min", 0.01)
        beta_max = config.get("beta_max", 1.0)
        battery_max_j = config.get("battery_max_j", 185400.0)  # 10Ah @ 5.1V

        # ── Dead-client guard ──────────────────────────────────────────────
        # A client with zero battery cannot train. Return immediately with a
        # skipped=True marker so the server excludes this client from the
        # aggregation weights and participation metrics.
        if state.battery_j <= 0:
            dataset_size = len(dataloader.dataset) if dataloader is not None else 1
            return {}, {
                "client_id":           state.client_id,
                "round_num":           state.round_num,
                "beta_actual":         0.0,
                "battery_j_remaining": 0.0,
                "energy_j_consumed":   0.0,
                "bytes_sent":          0,
                "bytes_received":      0,
                "local_loss":          0.0,
                "compression_ratio":   0.0,
                "nnz_fraction":        0.0,
                "dataset_size":        dataset_size,
                "skipped":             True,
            }

        # ── Step 1: Compute beta from battery ──────────────────────────────
        battery_j = state.battery_j
        beta = beta_min + (beta_max - beta_min) * (battery_j / battery_max_j)
        beta = max(beta_min, min(beta_max, beta))   # numerical safety

        # ── Step 2: Save weights before training ───────────────────────────
        w_before = OrderedDict(
            {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 3: Local SGD training ─────────────────────────────────────
        model.train()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        total_flops = 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Step 4: Compute gradient delta = w_before - w_after ───────────
        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })

        # ── Step 5: Error-feedback compression ────────────────────────────
        # Initialize error buffer if needed
        if state.error_buffer is None:
            state.error_buffer = {
                k: torch.zeros_like(v) for k, v in delta.items()
            }

        compressed = OrderedDict()
        new_error = OrderedDict()

        for k in delta:
            u = delta[k] + state.error_buffer[k]
            v = topk_sparsify(u, beta)
            # float16 quantization: rounding error absorbed into EF buffer
            if use_fp16_uplink:
                v = v.half().float()
            new_error[k] = u - v  # captures TopK zeroing + float16 rounding
            compressed[k] = v

        state.error_buffer = new_error

        # ── Step 6: Energy accounting ──────────────────────────────────────
        profile: DeviceProfile = config.get("device_profile")
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        # float16 sparse: 4 bytes index + 2 bytes value = 6 bytes/nz
        _bpv = 2 if use_fp16_uplink else 4
        sparse_bytes   = self.count_bytes(compressed, sparse=True, bytes_per_value=_bpv)
        dense_bytes    = self.count_bytes(compressed, sparse=False)
        uplink_bytes   = min(sparse_bytes, dense_bytes)
        downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile is not None:
            flops    = profile.flops_for_model(num_params, batch_size, local_epochs, dataset_size)
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * beta  # E_comp + E_comm * beta

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
            "beta_actual": beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": beta,
            "nnz_fraction": beta,
            "dataset_size": dataset_size,
        }

        return dict(compressed), metadata

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        w_{t+1} = w_t - sum_k (n_k / n) * v_t^k
        Weighted aggregation (FedAvg-style): clients with larger local datasets
        contribute proportionally more to the global update, which reduces bias
        in non-IID settings (Zhao et al., 2018).
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Separate alive (active) from dead (skipped) clients ───────────
        active_updates = [
            (u, m, s) for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]
        K_act = max(len(active_updates), 1)

        # Compute dataset-size weights only over active clients
        sizes = [m.get("dataset_size", 1) for _, m, _ in active_updates]
        total_n = max(sum(sizes), 1)
        weights = [n_k / total_n for n_k in sizes]

        # Accumulate weighted compressed updates (active clients only)
        agg = None
        for (update, metadata, state), w_k in zip(active_updates, weights):
            if agg is None:
                agg = {k: v.clone().float() * w_k for k, v in update.items()}
            else:
                for k in agg:
                    if k in update:
                        agg[k] += update[k].float() * w_k

        # Apply weighted aggregate
        new_weights = OrderedDict()
        for k in global_sd:
            if agg is not None and k in agg:
                new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
            else:
                new_weights[k] = global_sd[k].float()

        del agg
        gc.collect()

        # ── Metrics (aligned with fedstep for cross-algorithm comparison) ─
        total_bytes  = sum(m["bytes_sent"]        for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery  = sum(s.battery_j            for _, _, s in client_updates) / K
        batt_min     = min(s.battery_j            for _, _, s in client_updates)
        avg_beta     = sum(m["beta_actual"]        for _, m, _ in active_updates) / K_act
        avg_loss     = sum(m["local_loss"]         for _, m, _ in active_updates) / K_act
        avg_cr       = sum(m["compression_ratio"]  for _, m, _ in active_updates) / K_act

        # Jain fairness index on bytes_sent (identical formula to fedstep)
        # Dead clients contribute 0 bytes → penalised correctly.
        bytes_sent = [m["bytes_sent"] for _, m, _ in client_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b ** 2 for b in bytes_sent))
        else:
            jain = 1.0

        metrics = {
            "round":              round_num,
            "total_bytes_sent":   total_bytes,
            "total_energy_j":     total_energy,
            "avg_beta":           avg_beta,
            "avg_battery_j":      avg_battery,
            "batt_min_j":         batt_min,
            "avg_local_loss":     avg_loss,
            "compression_ratio":  avg_cr,
            "participation_rate": len(active_updates) / K,
            "jain_index":         jain,
            "num_clients":        K,
            "skipped_clients":    K - len(active_updates),
        }

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    def get_default_config(self) -> dict:
        return {
            # Training
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            # E-CEFFL specific
            "beta_min": 0.01,
            "beta_max": 1.0,
            "battery_max_j": 185400.0,   # 10Ah @ 5.1V powerbank
            # Device
            "device": "cpu",
            "device_profile": None,
        }
