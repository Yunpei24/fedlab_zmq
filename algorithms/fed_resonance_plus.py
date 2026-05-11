"""
algorithms/fed_resonance_plus.py
=================================
Fed-Resonance+: Combined Weight-Space Filtering + Gradient-Space Compression.
J. Nikiema, EL Amhoud, H. ELHAMMOUTI & I. KISSAMI — UM6P, 2026

Algorithm overview
------------------
Fed-Resonance+ is a hybrid algorithm that combines BOTH mechanisms:
  1. WEIGHT-SPACE gradient filtering (from FedGradAlign)
  2. GRADIENT-SPACE compression (from FedResonance)

This provides:
  - Reduced gradient variance via weight-space alignment (filtering during training)
  - Reduced communication cost via gradient-space SVD/subspace compression (uplink)

Architecture per round:
  SERVER → CLIENT:
    - Full global model W_global (dense, float32)
    - Weight-space resonance bases {U_l^W, V_l^W} per layer (every K rounds)

  CLIENT:
    Step 1: Local SGD with gradient filtering
      - For each batch:
        - Backward pass → raw gradient g
        - Filter: g_filtered = U^W (U^W^T g V^W) V^W^T  (if weight bases available)
        - optimizer.step() with g_filtered

    Step 2: Compute weight delta
      - delta = w_before - w_after

    Step 3: Compress delta using gradient-space SVD/subspace
      - Use weight-space bases as warm start for subspace mode
      - Per-layer switch: SVD-truncated vs Subspace vs dense vs sparse
      - Error feedback maintained across mode switches

  CLIENT → SERVER:
    - Compressed gradient (SVD factors, subspace coefficients, or sparse/dense)

  SERVER:
    Step 1: Reconstruct and aggregate gradients (FedAvg)
    Step 2: Every K rounds: recompute weight-space SVD bases
    Step 3: Broadcast new model + bases

Key insight (warm start):
  The weight-space basis U^W from the server can be used as an initial
  gradient-space basis for subspace mode. If proj_energy(delta, U^W) >= tau_rank,
  clients can immediately use subspace mode without a cold SVD round. This
  eliminates the initial SVD rounds that standard FedResonance requires to
  build gradient-space bases.

Energy model:
  - Uplink: compressed gradient (SVD/subspace/sparse — same as FedResonance)
  - Downlink: full model + weight-space bases (amortized over basis_update_freq)
  - Energy: E_comp (training + compression) + E_comm (uplink + downlink)

Convergence:
  Under standard FL assumptions (L-smoothness, bounded variance), plus:
    - Weight-space bases remain sufficiently aligned with model structure
    - Gradient-space compression preserves at least alpha_c fraction of spectral energy
  Fed-Resonance+ achieves O(1/sqrt(T)) non-convex convergence with both:
    - Reduced gradient variance (from filtering)
    - Reduced communication cost (from compression)

Comparison table:
+------------------+----------------+------------------+------------------+
| Algorithm        | Filtering      | Compression      | Best for         |
+------------------+----------------+------------------+------------------+
| FedResonance     | None           | Gradient-space   | Comm-constrained |
| FedGradAlign     | Weight-space   | None             | Variance-limited |
| FedResonancePlus | Weight-space   | Gradient-space   | Both challenges  |
+------------------+----------------+------------------+------------------+

References:
  - FedResonance: Nikiema & Amhoud, UM6P 2026 (gradient-space SVD)
  - FedGradAlign: Nikiema & Amhoud, UM6P 2026 (weight-space filtering)
  - PowerSGD: Vogels et al., NeurIPS 2019 (gradient compression via low-rank)
  - E-CEFFL: Nikiema & Amhoud, 2025 (battery-adaptive compression + EF)
"""

import gc
import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from .fed_resonance import (
    randomized_svd,
    svd_compress,
    svd_decompress,
    _matrix_form,
    _count_svd_bytes,
    _detect_layer_type,
    sparse_topk_compress,
    sparse_topk_decompress,
    adaptive_rank,
    BasisMemoryManager,
)


@register_algorithm("fed_resonance_plus")
class FedResonancePlus(FLAlgorithm):
    """
    Fed-Resonance+: Combined Weight-Space Filtering + Gradient-Space Compression.

    Key properties:
      - Server computes weight-space SVD bases and broadcasts to clients
      - Clients filter gradients during training using weight-space bases
      - Clients compress gradients using gradient-space SVD/subspace
      - Weight-space bases provide warm start for gradient-space subspace mode
      - Achieves both variance reduction AND communication efficiency
      - Convergence: O(1/sqrt(T)) non-convex under standard assumptions

    Metadata returned per client:
      - All FedResonance fields: avg_rank, avg_drift, mode_per_layer, basis_evictions
      - All FedGradAlign fields: spectral_alignment_avg, basis_bytes_received
      - warmstart_layers: number of layers using weight-basis warm start
    """

    name = "fed_resonance_plus"
    description = (
        "Combined weight-space gradient filtering + gradient-space compression. "
        "Best of FedResonance and FedGradAlign. "
        "Nikiema & Amhoud, UM6P 2026."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        Local training with weight-space filtering + gradient-space compression.

        Steps:
          1. Load weight-space bases from config (broadcast by server)
          2. Local SGD with gradient filtering (FedGradAlign part)
          3. Compute weight delta
          4. Compress delta using gradient-space SVD/subspace (FedResonance part)
          5. Use weight-space bases as warm start for subspace mode (KEY INNOVATION)
          6. Energy accounting and battery update
        """
        device = config.get("device", "cpu")
        lr = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)
        beta_min = config.get("beta_min", 0.01)
        beta_max = config.get("beta_max", 1.0)
        battery_max_j = config.get("battery_max_j", 185400.0)
        rank_min = int(config.get("rank_min", 4))
        rank_max = int(config.get("rank_max", 32))
        eps = float(config.get("spectral_energy_thresh", 0.95))
        tau_drift = float(config.get("subspace_drift_thresh", 0.1))
        tau_rank = float(config.get("rank_coherence_thresh", 0.7))
        use_ef = bool(config.get("use_error_feedback", True))
        use_rsvd = bool(config.get("use_rsvd", True))
        rsvd_oversample = int(config.get("rsvd_oversample", 10))
        rsvd_power_iter = int(config.get("rsvd_power_iter", 2))
        profile = config.get("device_profile")

        # Weight-space filtering parameters (from FedGradAlign)
        weight_bases = config.get("weight_bases", {})
        grad_filter_alpha = config.get("grad_filter_alpha", 0.7)

        # Rank selection criterion
        rank_criterion = str(config.get("rank_criterion", "energy"))
        rank_budget_bytes = int(config.get("rank_budget_bytes", 0))

        # Soft projection parameters
        use_soft_proj = bool(config.get("use_soft_projection", False))
        soft_proj_adaptive = bool(config.get("soft_projection_adaptive", False))
        soft_proj_alpha = float(config.get("soft_projection_alpha", 0.8))
        soft_proj_alpha_start = float(config.get("soft_projection_alpha_start", 0.5))
        soft_proj_alpha_end = float(config.get("soft_projection_alpha_end", 0.95))
        total_rounds = int(config.get("num_rounds", 100))

        # Compute alpha for this round
        if use_soft_proj and soft_proj_adaptive:
            t_frac = min(1.0, state.round_num / max(total_rounds - 1, 1))
            soft_proj_alpha = (
                soft_proj_alpha_start
                + t_frac * (soft_proj_alpha_end - soft_proj_alpha_start)
            )

        # ── Step 1: Battery → beta ─────────────────────────────────────────────────
        battery_j = state.battery_j
        beta = beta_min + (beta_max - beta_min) * (battery_j / battery_max_j)
        beta = max(beta_min, min(beta_max, beta))
        battery_critical = beta < 0.1

        # ── Step 2: Save weights before training ───────────────────────────────────
        w_before = OrderedDict(
            {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 3: Local SGD with weight-space gradient filtering ─────────────────
        model.train()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0
        alignment_scores = []

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                total_loss += loss.item()
                num_batches += 1

                # ── Apply weight-space gradient filtering (FedGradAlign part) ──────
                if weight_bases:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.grad is None or name not in weight_bases:
                                continue

                            g = param.grad.data
                            B = weight_bases[name]
                            U = B["U"].to(g.device).float()
                            V = B["V"].to(g.device).float()
                            G = _matrix_form(g)

                            # Check shape compatibility
                            if U.shape[0] != G.shape[0] or V.shape[0] != G.shape[1]:
                                continue

                            # Project gradient: G_proj = U (U^T G V) V^T
                            temp = U.t() @ G @ V
                            G_proj = U @ temp @ V.t()

                            # Compute alignment
                            G_norm_sq = (G * G).sum().item()
                            if G_norm_sq > 1e-12:
                                G_proj_norm_sq = (G_proj * G_proj).sum().item()
                                alignment = G_proj_norm_sq / G_norm_sq
                                alignment_scores.append(min(1.0, alignment))

                            # Apply soft projection (use grad_filter_alpha, not soft_proj_alpha)
                            G_filtered = grad_filter_alpha * G_proj + (1.0 - grad_filter_alpha) * G
                            param.grad.data = G_filtered.reshape(g.shape)

                optimizer.step()

        # ── Step 4: Compute delta = w_before - w_after ─────────────────────────────
        w_after = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - w_after[k].cpu()).float()
            for k in w_before
        })
        del w_before
        del w_after
        gc.collect()

        # ── Step 5: Initialize persistent state ─────────────────────────────────────
        if state.error_buffer is None and use_ef:
            state.error_buffer = {k: torch.zeros_like(v) for k, v in delta.items()}

        if "subspace_bases" not in state.custom:
            state.custom["subspace_bases"] = {}
        if "basis_round" not in state.custom:
            state.custom["basis_round"] = {}

        bases = state.custom["subspace_bases"]
        b_rounds = state.custom["basis_round"]
        current_round = state.round_num

        # ── Step 6: Per-layer hybrid compression with warm start ───────────────────
        update = OrderedDict()
        new_error = OrderedDict()
        mode_per_layer: dict[str, str] = {}
        total_bytes_sent = 0
        total_bytes_dense = 0
        ranks_used: list[int] = []
        drifts: list[float] = []
        warmstart_layers = 0

        for name, g in delta.items():
            # Add error feedback
            if use_ef and state.error_buffer is not None:
                u = g + state.error_buffer[name]
            else:
                u = g

            # Layer type detection
            layer_type = _detect_layer_type(name, u)
            numel = u.numel()
            dense_bytes = numel * 4
            total_bytes_dense += dense_bytes

            # ── Handle embedding layers with sparse TopK ────────────────────────────
            if layer_type == "embedding":
                topk_ratio = config.get("embedding_topk_ratio", 0.01)
                values, indices = sparse_topk_compress(u, topk_ratio)
                k = values.numel()
                sparse_bytes = k * 8
                total_bytes_sent += sparse_bytes
                residual = u.flatten().clone()
                residual[indices.long()] = 0.0
                residual = residual.reshape(u.shape)
                update[name] = {
                    "mode": "sparse",
                    "values": values,
                    "indices": indices,
                    "shape": list(u.shape),
                }
                mode_per_layer[name] = "sparse"
                if use_ef:
                    new_error[name] = residual.clone()
                continue

            # ── Handle norm/bias layers with dense ──────────────────────────────────
            elif layer_type in ("norm", "bias"):
                update[name] = {"mode": "dense", "data": u.clone()}
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                if use_ef:
                    new_error[name] = torch.zeros_like(u)
                continue

            # ── Handle conv/linear layers with SVD/subspace ─────────────────────────
            if numel < rank_min * 2:
                mode = "dense"
            elif battery_critical:
                mode = "svd_forced"
            else:
                # ── WARM START: Try using weight-space basis as gradient-space basis ──
                # If weight-space basis U^W is available for this layer, check if
                # the gradient delta aligns well with it. If yes, use subspace mode
                # immediately without requiring a full SVD round.
                if name in weight_bases:
                    U_weight = weight_bases[name]["U"].float()
                    G_proxy = _matrix_form(u)

                    # Check if weight-space basis matches gradient dimensions
                    if U_weight.shape[0] == G_proxy.shape[0]:
                        # Project gradient onto weight-space basis
                        alpha_cols = U_weight.t() @ G_proxy
                        G_approx_proxy = U_weight @ alpha_cols
                        proj_energy = (
                            G_approx_proxy.pow(2).sum() /
                            (G_proxy.pow(2).sum() + 1e-12)
                        ).item()

                        if proj_energy >= tau_rank:
                            # Warm start successful: use weight-space basis as gradient-space basis
                            if name not in bases or bases[name] is None:
                                bases[name] = U_weight.clone().cpu()
                                b_rounds[name] = current_round
                                warmstart_layers += 1
                            mode = "subspace"
                            drifts.append(1.0 - proj_energy)
                        else:
                            # Weight-space basis doesn't align well — fall back to SVD
                            mode = "svd"
                    else:
                        # Shape mismatch — fall back to SVD
                        mode = "svd"

                # ── Standard subspace check (if no weight basis or warm start failed) ──
                elif name in bases and bases[name] is not None:
                    basis_old = bases[name]
                    G_proxy = _matrix_form(u)
                    B = basis_old.float()

                    if B.shape[0] != G_proxy.shape[0]:
                        mode = "svd"
                    else:
                        alpha_cols = B.t() @ G_proxy
                        G_approx_proxy = B @ alpha_cols
                        proj_energy = (
                            G_approx_proxy.pow(2).sum() /
                            (G_proxy.pow(2).sum() + 1e-12)
                        ).item()

                        if proj_energy >= tau_rank:
                            mode = "subspace"
                            drifts.append(1.0 - proj_energy)
                        else:
                            mode = "svd"
                else:
                    mode = "svd"

            # ── Compress according to mode ──────────────────────────────────────────
            if mode == "dense":
                v = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                update[name] = {"mode": "dense", "data": v}

            elif mode in ("svd", "svd_forced"):
                G = _matrix_form(u)
                m_dim, n_dim = G.shape

                if mode == "svd_forced":
                    r = rank_min
                else:
                    r = adaptive_rank(
                        G, eps, use_rsvd, rsvd_oversample, rsvd_power_iter,
                        rank_criterion=rank_criterion,
                        rank_budget_bytes=rank_budget_bytes,
                        rank_min=rank_min,
                    )
                    r = max(rank_min, min(rank_max, r))

                ranks_used.append(r)
                U, S, Vt = svd_compress(G, r, use_rsvd, rsvd_oversample, rsvd_power_iter)
                G_approx = svd_decompress(U, S, Vt)

                # Soft projection
                if use_soft_proj:
                    G_soft = soft_proj_alpha * G_approx + (1.0 - soft_proj_alpha) * G
                    v = G_soft.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape) * soft_proj_alpha
                else:
                    v = G_approx.reshape(u.shape)
                    residual = u - v

                # Update gradient-space basis
                new_basis = U.contiguous()
                bases[name] = new_basis
                b_rounds[name] = current_round

                svd_bytes = _count_svd_bytes(m_dim, n_dim, r)
                total_bytes_sent += svd_bytes
                mode_tag = "svd" if mode == "svd" else "svd_forced"
                mode_per_layer[name] = mode_tag
                update[name] = {
                    "mode": mode_tag,
                    "U": U, "S": S, "Vt": Vt,
                    "shape": list(u.shape),
                }

            elif mode == "subspace":
                B = bases[name].float()
                G = _matrix_form(u)
                alpha_cols = B.t() @ G
                G_approx = B @ alpha_cols

                # Soft projection
                if use_soft_proj:
                    G_soft = soft_proj_alpha * G_approx + (1.0 - soft_proj_alpha) * G
                    v = G_soft.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape) * soft_proj_alpha
                else:
                    v = G_approx.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape)

                r_used = B.shape[1]
                n_dim_sub = G.shape[1]
                ranks_used.append(r_used)
                sub_bytes = r_used * n_dim_sub * 4
                total_bytes_sent += sub_bytes
                mode_per_layer[name] = "subspace"
                update[name] = {
                    "mode": "subspace",
                    "alpha_cols": alpha_cols,
                    "basis_round": b_rounds[name],
                    "shape": list(u.shape),
                    "layer_name": name,
                }
            else:
                # Fallback
                v = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                update[name] = {"mode": "dense", "data": v}

            if use_ef:
                new_error[name] = residual.clone()

        if use_ef:
            state.error_buffer = new_error

        # ── Basis memory management ─────────────────────────────────────────────────
        max_basis_mb = float(config.get("max_basis_memory_mb", 128.0))
        basis_max_age = int(config.get("basis_max_age_rounds", 20))
        basis_mgr = BasisMemoryManager(max_basis_mb, basis_max_age)
        n_evicted = basis_mgr.update(bases, b_rounds, current_round)

        state.custom["subspace_bases"] = bases
        state.custom["basis_round"] = b_rounds

        # ── Energy accounting ───────────────────────────────────────────────────────
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        # Downlink: full model + weight-space bases
        basis_bytes = 0
        if weight_bases:
            for B in weight_bases.values():
                basis_bytes += (B["U"].numel() + B["V"].numel()) * 4
        downlink_bytes = total_bytes_dense + basis_bytes

        if profile is not None:
            flops = profile.flops_for_model(
                num_params, batch_size, local_epochs, dataset_size
            )
            energy_j = profile.round_energy_j(flops, total_bytes_sent, downlink_bytes)
        else:
            actual_beta_proxy = total_bytes_sent / max(total_bytes_dense, 1)
            energy_j = 0.5 + 2.0 * actual_beta_proxy

        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        compression_ratio = total_bytes_sent / max(total_bytes_dense, 1)
        avg_rank = float(sum(ranks_used) / len(ranks_used)) if ranks_used else 0.0
        avg_drift = float(sum(drifts) / len(drifts)) if drifts else 0.0
        avg_alignment = (
            sum(alignment_scores) / len(alignment_scores)
            if alignment_scores else 0.0
        )

        # ── Metadata (ALL mandatory + algo-specific) ────────────────────────────────
        metadata = {
            # Mandatory
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": total_bytes_sent,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            # FedResonance fields
            "mode_per_layer": mode_per_layer,
            "avg_rank": avg_rank,
            "avg_drift": avg_drift,
            "dataset_size": dataset_size,
            "battery_critical": battery_critical,
            "basis_evictions": n_evicted,
            # FedGradAlign fields
            "spectral_alignment_avg": avg_alignment,
            "basis_bytes_received": basis_bytes,
            "basis_updated_this_round": config.get("basis_updated_this_round", False),
            # FedResonancePlus-specific
            "warmstart_layers": warmstart_layers,
            "num_filtered_layers": len(alignment_scores) // max(num_batches, 1),
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
        Server-side aggregation with periodic weight-space SVD for basis updates.

        Steps:
          1. Reconstruct and aggregate gradients (FedAvg, same as FedResonance)
          2. Every K rounds: compute weight-space SVD bases (same as FedGradAlign)
          3. Return new bases in metrics for broadcast to clients next round
        """
        basis_update_freq = config.get("basis_update_freq", 5)
        rank = config.get("resonance_rank", 20)
        use_rsvd = config.get("use_rsvd", True)
        rsvd_oversample = config.get("rsvd_oversample", 10)
        rsvd_power_iter = config.get("rsvd_power_iter", 2)

        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── FedAvg aggregation (same as FedResonance) ───────────────────────────────
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = sum(sizes)
        weights = [n_k / total_n for n_k in sizes]

        agg: Optional[OrderedDict] = None

        for (update, metadata, cstate), w_k in zip(client_updates, weights):
            dense = OrderedDict()
            client_bases = cstate.custom.get("subspace_bases", {})

            for name, packed in update.items():
                mode = packed["mode"]

                if mode == "dense":
                    dense[name] = packed["data"].float()

                elif mode in ("svd", "svd_forced"):
                    U = packed["U"].float()
                    S = packed["S"].float()
                    Vt = packed["Vt"].float()
                    shape = packed["shape"]
                    G_approx = svd_decompress(U, S, Vt)
                    dense[name] = G_approx.reshape(shape).float()

                elif mode == "sparse":
                    values = packed["values"].float()
                    indices = packed["indices"].long()
                    shape = packed["shape"]
                    dense[name] = sparse_topk_decompress(values, indices, shape)

                elif mode == "subspace":
                    alpha_cols = packed["alpha_cols"].float()
                    shape = packed["shape"]
                    if name in client_bases and client_bases[name] is not None:
                        B = client_bases[name].float()
                        G_approx = B @ alpha_cols
                        dense[name] = G_approx.reshape(shape).float()
                    else:
                        dense[name] = torch.zeros(shape, dtype=torch.float32)

                else:
                    ref = global_sd.get(name)
                    if ref is not None:
                        dense[name] = torch.zeros_like(ref, dtype=torch.float32)

            if agg is None:
                agg = OrderedDict({k: v.clone() * w_k for k, v in dense.items()})
            else:
                for k in dense:
                    if k in agg:
                        agg[k] += dense[k] * w_k
                    else:
                        agg[k] = dense[k] * w_k

        # Apply aggregated gradient
        new_weights = OrderedDict()
        for k in global_sd:
            if agg is not None and k in agg:
                new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
            else:
                new_weights[k] = global_sd[k].float()
        del agg
        gc.collect()

        # Load into global model
        global_model.load_state_dict(new_weights)

        # ── Compute metrics ─────────────────────────────────────────────────────────
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_bytes_recv = sum(m["bytes_received"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_beta = sum(m["beta_actual"] for _, m, _ in client_updates) / K
        avg_battery = sum(s.battery_j for _, _, s in client_updates) / K
        avg_loss = sum(m["local_loss"] for _, m, _ in client_updates) / K
        avg_rank = sum(m.get("avg_rank", 0.0) for _, m, _ in client_updates) / K
        avg_drift = sum(m.get("avg_drift", 0.0) for _, m, _ in client_updates) / K
        avg_cr = sum(m.get("compression_ratio", 1.0) for _, m, _ in client_updates) / K
        avg_alignment = sum(
            m.get("spectral_alignment_avg", 0.0) for _, m, _ in client_updates
        ) / K
        total_warmstart = sum(m.get("warmstart_layers", 0) for _, m, _ in client_updates)

        # Mode distribution
        mode_counts: dict[str, int] = {
            "svd": 0, "svd_forced": 0, "subspace": 0, "dense": 0, "sparse": 0
        }
        for _, m, _ in client_updates:
            for mode in m.get("mode_per_layer", {}).values():
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

        jain = (sum([1.0] * K) ** 2) / (K * sum([1.0] * K))
        total_evictions = sum(m.get("basis_evictions", 0) for _, m, _ in client_updates)

        metrics = {
            "round": round_num,
            "total_bytes_sent": total_bytes,
            "total_bytes_received": total_bytes_recv,
            "total_energy_j": total_energy,
            "avg_beta": avg_beta,
            "avg_battery_j": avg_battery,
            "avg_local_loss": avg_loss,
            "avg_rank": avg_rank,
            "avg_drift": avg_drift,
            "avg_compression_ratio": avg_cr,
            "mode_counts": mode_counts,
            "participation_rate": 1.0,
            "jain_index": jain,
            "num_clients": K,
            "basis_evictions": total_evictions,
            # FedGradAlign metrics
            "avg_spectral_alignment": avg_alignment,
            "total_warmstart_layers": total_warmstart,
        }

        # ── Recompute weight-space bases every K rounds ─────────────────────────────
        should_update = (
            round_num % basis_update_freq == 0
            or "weight_bases" not in config
            or not config.get("weight_bases")
        )

        new_bases = config.get("weight_bases", {})
        basis_bytes_total = 0

        if should_update:
            new_bases = {}
            for name, param in global_model.named_parameters():
                W = _matrix_form(param.data)
                m, n = W.shape

                if m < 2 or n < 2:
                    continue

                r = min(rank, min(m, n) - 1)
                if r < 1:
                    continue

                try:
                    if use_rsvd:
                        U, S, Vt = randomized_svd(
                            W.float(), r, rsvd_oversample, rsvd_power_iter
                        )
                    else:
                        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
                        U = U[:, :r].contiguous()
                        Vt = Vt[:r, :].contiguous()

                    new_bases[name] = {
                        "U": U.cpu(),
                        "V": Vt.t().cpu(),
                    }
                    basis_bytes_total += (U.numel() + Vt.t().numel()) * 4

                except Exception:
                    pass

            metrics["basis_updated"] = True
            metrics["basis_layers"] = len(new_bases)
            metrics["basis_bytes_broadcast"] = basis_bytes_total
        else:
            metrics["basis_updated"] = False
            metrics["basis_layers"] = len(new_bases)
            metrics["basis_bytes_broadcast"] = 0

        metrics["weight_bases"] = new_bases
        metrics["basis_updated_this_round"] = should_update

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    def get_default_config(self) -> dict:
        return {
            # Training
            "lr": 0.01,
            "local_epochs": 1,
            "batch_size": 32,
            # SVD parameters (gradient-space compression from FedResonance)
            "rank_min": 4,
            "rank_max": 32,
            "spectral_energy_thresh": 0.95,
            "use_rsvd": True,
            "rsvd_oversample": 10,
            "rsvd_power_iter": 2,
            # Switch criteria (gradient-space)
            "subspace_drift_thresh": 0.1,
            "rank_coherence_thresh": 0.7,
            # Weight-space basis parameters (from FedGradAlign)
            "resonance_rank": 20,
            "basis_update_freq": 5,
            # Gradient filtering parameter (separate from compression projection)
            "grad_filter_alpha": 0.7,  # mixing for weight-space filtering during training
            # Battery model (E-CEFFL formula)
            "beta_min": 0.01,
            "beta_max": 1.0,
            "battery_max_j": 185400.0,
            # Error feedback
            "use_error_feedback": True,
            # Device
            "device": "cpu",
            "device_profile": None,
            # Hybrid compression parameters
            "embedding_topk_ratio": 0.01,
            # Basis memory management
            "max_basis_memory_mb": 128.0,
            "basis_max_age_rounds": 20,
            # Rank selection criterion
            "rank_criterion": "energy",
            "rank_budget_bytes": 0,
            # Soft projection (gradient-space compression)
            "use_soft_projection": False,
            "soft_projection_alpha": 0.8,
            "soft_projection_adaptive": False,
            "soft_projection_alpha_start": 0.5,
            "soft_projection_alpha_end": 0.95,
            "num_rounds": 100,
            # Server state (populated by server_aggregate)
            "weight_bases": {},  # dict[layer_name -> {"U": Tensor, "V": Tensor}]
            "basis_updated_this_round": False,
        }
