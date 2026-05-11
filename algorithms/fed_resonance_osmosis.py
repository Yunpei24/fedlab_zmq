"""
algorithms/fed_resonance_osmosis.py
=====================================
Fed-Resonance + Fed-Osmosis: Hybrid Spectral-Thermodynamic Federated Learning
J. Nikiema, EL Amhoud, H. ELHAMMOUTI & I. KISSAMI — UM6P, 2026

Motivation
----------
Fed-Resonance and Fed-Osmosis are complementary: they address non-IID bias
at different levels of the learning pipeline.

  Fed-Osmosis  (training time) : regularises LOCAL TRAINING by penalising
    divergence of activation distributions from the global reference.
    Acts on: the loss landscape during gradient computation.

  Fed-Resonance (transmission) : compresses the GRADIENT UPDATE using
    adaptive SVD-truncated or subspace projection per layer.
    Acts on: the communication payload.

Together they provide a double guarantee:
  1. Each client's model is trained to produce activations consistent with
     the global distribution (Osmosis).
  2. Only the most informative gradient directions are transmitted, with
     the residual preserved via error feedback (Resonance).

Pipeline (each round)
---------------------
Server → Client:
  w_t, global_stats {μ_g^l, σ_g^l}, resonance bases {B_l}

Client update:
  Step 1 : beta_t^k ← beta_min + (beta_max - beta_min) · B_t^k / B_max
  Step 2 : Register osmotic hooks on Linear/Conv2d layers
  Step 3 : Local SGD with L = L_task + γ · Σ_l KL(N(μ_k^l,σ_k²^l) ‖ N(μ_g^l,σ_g²^l))
  Step 4 : Compute delta = w_before − w_after
  Step 5 : Error-feedback: u = delta + e  (e = residual from last round)
  Step 6 : Per-layer hybrid compression (SVD / Subspace / dense)
           same switch logic as Fed-Resonance
  Step 7 : e ← u − v   (new residual)
  Step 8 : Energy accounting

Server aggregate:
  Step A : Reconstruct dense gradients from compressed updates
  Step B : Weighted FedAvg: w_{t+1} = w_t − Σ_k (n_k/n) v_k
  Step C : Update global activation stats: μ_g^l ← Σ_k (n_k/n) μ_k^l
  Step D : Update global resonance bases (optional SVD on aggregated gradient)

Convergence
-----------
Under L-smoothness, bounded variance, γ-Lipschitz osmotic gradient, and
spectral contraction α_c = spectral_energy_thresh:

  E[‖∇f(w_T)‖²] ≤ O(1/√T) + O(γ·M)

  The resonance compression introduces bias O(1 − α_c) which is absorbed
  by the error-feedback residual (same analysis as Fed-Resonance).
  The osmosis term adds O(γ·M) where M = sup_w ‖∇Π(w)‖ — controllable via γ.

References
----------
- Fed-Resonance : Nikiema & Amhoud, UM6P 2026 (see fed_resonance.py)
- Fed-Osmosis   : Nikiema & Amhoud, UM6P 2026 (see fed_osmosis.py)
- PowerSGD      : Vogels et al., NeurIPS 2019
- SCAFFOLD      : Karimireddy et al., ICML 2020
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import Optional

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm

# Reuse all spectral helpers from Fed-Resonance
from .fed_resonance import (
    adaptive_rank,
    svd_compress,
    svd_decompress,
    subspace_project,
    subspace_reconstruct,
    subspace_drift,
    _matrix_form,
    _count_svd_bytes,
    _count_subspace_bytes,
)

# Reuse osmotic helpers from Fed-Osmosis
from .fed_osmosis import (
    _osmotic_pressure,
    _StatAccumulator,
    _MONITORED_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fed_resonance_osmosis")
class FedResonanceOsmosis(FLAlgorithm):
    """
    Fed-Resonance + Fed-Osmosis Hybrid.

    Key properties:
      - Osmotic regularisation during training (distribution alignment)
      - Adaptive per-layer spectral compression at transmission
      - Battery-proportional compression budget (E-CEFFL formula)
      - Error feedback across compression modes
      - Dataset-size weighted aggregation (FedAvg-style)
      - All clients participate every round (universal participation)

    State stored in ClientState.custom:
      - "subspace_bases": dict[layer_name -> Tensor(d_l, r)]
      - "basis_round":    dict[layer_name -> int]
      - "local_stats":    dict[layer_name -> {"mu": float, "sigma": float}]

    Metadata keys (superset of Fed-Resonance + Fed-Osmosis):
      - beta_actual, battery_j_remaining, energy_j_consumed
      - bytes_sent, bytes_received, local_loss
      - osmotic_pressure, compression_ratio
      - avg_rank, avg_drift, mode_per_layer
      - local_stats, dataset_size
    """

    name        = "fed_resonance_osmosis"
    description = (
        "Hybrid: osmotic distribution alignment (training) + adaptive spectral "
        "compression (transmission) with error feedback and battery awareness. "
        "Nikiema & Amhoud, UM6P 2026."
    )

    # ── client_update ─────────────────────────────────────────────────────────

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
        beta_min      = config.get("beta_min", 0.01)
        beta_max      = config.get("beta_max", 1.0)
        battery_max_j = config.get("battery_max_j", 185400.0)
        rank_min      = int(config.get("rank_min", 4))
        rank_max      = int(config.get("rank_max", 32))
        eps           = float(config.get("spectral_energy_thresh", 0.95))
        tau_rank      = float(config.get("rank_coherence_thresh", 0.7))
        use_ef        = bool(config.get("use_error_feedback", True))
        gamma         = float(config.get("gamma", 0.1))
        global_stats  = config.get("osmosis_global_stats", {})
        profile       = config.get("device_profile")
        dataset_size  = len(dataloader.dataset)
        batch_size    = dataloader.batch_size

        # ── Step 1: Battery → beta ───────────────────────────────────────────
        battery_j        = state.battery_j
        beta             = beta_min + (beta_max - beta_min) * (battery_j / battery_max_j)
        beta             = max(beta_min, min(beta_max, beta))
        battery_critical = beta < 0.1

        # ── Step 2: Save w_before ────────────────────────────────────────────
        w_before = OrderedDict(
            {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 3: Local training with osmotic regularisation ───────────────
        model.train()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        # Stat accumulator for server reporting (detached)
        stat_acc = _StatAccumulator()
        stat_acc.register(model)

        total_loss  = 0.0
        osm_total   = 0.0
        num_batches = 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                # Register differentiable osmotic hooks for this batch
                captured: dict[str, torch.Tensor] = {}
                osm_hooks = []
                if global_stats:
                    for name_l, module in model.named_modules():
                        if (isinstance(module, _MONITORED_TYPES)
                                and name_l in global_stats):
                            def _make(n):
                                def _h(mod, inp, out, _n=n):
                                    captured[_n] = out
                                return _h
                            osm_hooks.append(
                                module.register_forward_hook(_make(name_l))
                            )

                optimizer.zero_grad()
                out       = model(x)
                task_loss = criterion(out, y)

                for h in osm_hooks:
                    h.remove()

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

        stat_acc.remove()
        local_stats = stat_acc.get_stats()
        state.custom["local_stats"] = local_stats

        # ── Step 4: Compute delta = w_before − w_after ───────────────────────
        w_after = model.state_dict()
        delta   = OrderedDict({
            k: (w_before[k] - w_after[k].cpu()).float()
            for k in w_before
        })
        del w_before
        del w_after
        gc.collect()

        # ── Step 5: Initialise error buffer and resonance state ───────────────
        if use_ef and state.error_buffer is None:
            state.error_buffer = {k: torch.zeros_like(v) for k, v in delta.items()}

        if "subspace_bases" not in state.custom:
            state.custom["subspace_bases"] = {}
        if "basis_round" not in state.custom:
            state.custom["basis_round"] = {}

        bases        = state.custom["subspace_bases"]
        b_rounds     = state.custom["basis_round"]
        current_round = state.round_num

        # ── Step 6: Per-layer hybrid spectral compression ────────────────────
        update          = OrderedDict()
        new_error       = OrderedDict()
        mode_per_layer: dict[str, str] = {}
        total_bytes_sent  = 0
        total_bytes_dense = 0
        ranks_used: list[int]   = []
        drifts:     list[float] = []

        for name_l, g in delta.items():
            # Error feedback
            if use_ef and state.error_buffer is not None:
                u = g + state.error_buffer[name_l]
            else:
                u = g

            numel       = u.numel()
            dense_bytes = numel * 4
            total_bytes_dense += dense_bytes

            # ── Mode selection (same logic as Fed-Resonance) ─────────────────
            if numel < rank_min * 2:
                mode = "dense"
            elif battery_critical:
                mode = "svd_forced"
            else:
                if name_l in bases and bases[name_l] is not None:
                    # Viability proxy check — FIXED: use matrix-space projection
                    # The basis B has shape (m_dim, r) — it lives in the row space
                    # of _matrix_form(u), which has shape (m_dim, n_dim).
                    # We must project in the same (m_dim, n_dim) space, NOT in
                    # the fully-flattened (d,) space, to avoid the dimension
                    # mismatch that occurs when numel(u) != m_dim.
                    G_proxy = _matrix_form(u)           # (m_dim, n_dim)
                    B = bases[name_l].float()           # (m_dim, r)
                    # Guard: basis rows must match current layer's m_dim.
                    # If the model was re-initialized or the layer shape changed
                    # between rounds the stored basis is stale — fall back to SVD.
                    if B.shape[0] != G_proxy.shape[0]:
                        # Basis is stale (shape mismatch) — fall back to SVD
                        mode = "svd"
                    else:
                        # Project each column of G onto the column space of B:
                        # alpha_cols = B^T G  (r, n_dim)
                        # G_approx   = B alpha_cols  (m_dim, n_dim)
                        alpha_cols = B.t() @ G_proxy            # (r, n_dim)
                        G_approx_proxy = B @ alpha_cols         # (m_dim, n_dim)
                        proj_energy = (
                            G_approx_proxy.pow(2).sum() /
                            (G_proxy.pow(2).sum() + 1e-12)
                        ).item()
                        # proj_energy >= tau_rank → basis explains gradient well → Subspace
                        if proj_energy >= tau_rank:
                            mode = "subspace"
                            drifts.append(1.0 - proj_energy)  # drift proxy = 1 - alignment
                        else:
                            mode = "svd"
                else:
                    mode = "svd"

            # ── Compress ─────────────────────────────────────────────────────
            if mode == "dense":
                v        = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name_l] = "dense"
                total_bytes_sent      += dense_bytes
                update[name_l]         = {"mode": "dense", "data": v}

            elif mode in ("svd", "svd_forced"):
                G          = _matrix_form(u)
                m_dim, n_dim = G.shape
                r          = rank_min if mode == "svd_forced" else max(
                    rank_min, min(rank_max, adaptive_rank(G, eps))
                )
                ranks_used.append(r)
                U, S, Vt   = svd_compress(G, r)
                G_approx   = svd_decompress(U, S, Vt)
                v          = G_approx.reshape(u.shape)
                residual   = u - v
                bases[name_l]   = U.contiguous()
                b_rounds[name_l] = current_round
                svd_bytes       = _count_svd_bytes(m_dim, n_dim, r)
                total_bytes_sent += svd_bytes
                mode_tag         = "svd" if mode == "svd" else "svd_forced"
                mode_per_layer[name_l] = mode_tag
                update[name_l] = {
                    "mode": mode_tag,
                    "U": U, "S": S, "Vt": Vt,
                    "shape": list(u.shape),
                }

            elif mode == "subspace":
                # FIXED: Project in matrix space instead of flattened space
                B = bases[name_l].float()   # (m_dim, r)
                G = _matrix_form(u)         # (m_dim, n_dim)

                # Project in matrix space: alpha_cols = B^T G  shape (r, n_dim)
                # Reconstruct:             G_approx   = B alpha_cols  (m_dim, n_dim)
                # Residual:                G - G_approx
                #
                # This is consistent with the stored basis shape (m_dim, r) and
                # avoids the dimension mismatch that arises when projecting the
                # fully-flattened gradient (length d = m_dim * n_dim) against B.
                alpha_cols = B.t() @ G                          # (r, n_dim)
                G_approx = B @ alpha_cols                       # (m_dim, n_dim)

                v = G_approx.reshape(u.shape)
                residual = (G - G_approx).reshape(u.shape)

                r_used = B.shape[1]
                n_dim_sub = G.shape[1]
                ranks_used.append(r_used)
                # Byte cost: r * n_dim floats (still << m_dim * n_dim for r << m_dim)
                sub_bytes = r_used * n_dim_sub * 4
                total_bytes_sent += sub_bytes
                mode_per_layer[name_l] = "subspace"
                update[name_l] = {
                    "mode": "subspace",
                    "alpha_cols": alpha_cols,       # (r, n_dim) — renamed from alpha
                    "basis_round": b_rounds[name_l],
                    "shape": list(u.shape),
                    "layer_name": name_l,
                }
            else:
                v        = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name_l] = "dense"
                total_bytes_sent      += dense_bytes
                update[name_l]         = {"mode": "dense", "data": v}

            if use_ef:
                new_error[name_l] = residual.clone()

        # ── Step 7: Update error buffer and resonance state ──────────────────
        if use_ef:
            state.error_buffer = new_error
        state.custom["subspace_bases"] = bases
        state.custom["basis_round"]    = b_rounds

        # ── Step 8: Energy accounting ─────────────────────────────────────────
        num_params     = sum(p.numel() for p in model.parameters())
        downlink_bytes = total_bytes_dense

        if profile is not None:
            flops    = profile.flops_for_model(
                num_params, batch_size, local_epochs, dataset_size
            )
            energy_j = profile.round_energy_j(flops, total_bytes_sent, downlink_bytes)
        else:
            actual_beta = total_bytes_sent / max(total_bytes_dense, 1)
            energy_j    = 0.5 + 2.0 * actual_beta

        state.battery_j  = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        compression_ratio = total_bytes_sent / max(total_bytes_dense, 1)
        avg_rank  = float(sum(ranks_used) / len(ranks_used)) if ranks_used else 0.0
        avg_drift = float(sum(drifts) / len(drifts))         if drifts     else 0.0

        metadata = {
            "client_id":           state.client_id,
            "round_num":           state.round_num,
            "beta_actual":         beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed":   energy_j,
            "bytes_sent":          total_bytes_sent,
            "bytes_received":      downlink_bytes,
            "local_loss":          total_loss / max(num_batches, 1),
            "osmotic_pressure":    osm_total / max(num_batches, 1),
            "compression_ratio":   compression_ratio,
            "mode_per_layer":      mode_per_layer,
            "avg_rank":            avg_rank,
            "avg_drift":           avg_drift,
            "local_stats":         local_stats,
            "dataset_size":        dataset_size,
            "battery_critical":    battery_critical,
        }

        return dict(update), metadata

    # ── server_aggregate ──────────────────────────────────────────────────────

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Server aggregation:
          A. Reconstruct dense gradients from resonance-compressed updates.
          B. Weighted FedAvg on reconstructed gradients.
          C. Aggregate activation stats → new osmosis_global_stats.
          D. Aggregate resonance metrics.
        """

        K         = len(client_updates)
        global_sd = global_model.state_dict()

        sizes   = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = sum(sizes)
        weights = [n_k / total_n for n_k in sizes]

        # ── A. Reconstruct + weighted accumulation ────────────────────────────
        agg: Optional[OrderedDict] = None

        for (update, metadata, cstate), w_k in zip(client_updates, weights):
            client_bases = cstate.custom.get("subspace_bases", {})
            dense        = OrderedDict()

            for name_l, packed in update.items():
                mode = packed["mode"]

                if mode == "dense":
                    dense[name_l] = packed["data"].float()

                elif mode in ("svd", "svd_forced"):
                    G_approx      = svd_decompress(
                        packed["U"].float(),
                        packed["S"].float(),
                        packed["Vt"].float(),
                    )
                    dense[name_l] = G_approx.reshape(packed["shape"]).float()

                elif mode == "subspace":
                    # FIXED: use alpha_cols instead of alpha
                    # alpha_cols has shape (r, n_dim); B has shape (m_dim, r).
                    # Reconstruction: G_approx = B @ alpha_cols  (m_dim, n_dim)
                    alpha_cols = packed["alpha_cols"].float()   # (r, n_dim)
                    shape  = packed["shape"]
                    # Retrieve basis from client state
                    if name_l in client_bases and client_bases[name_l] is not None:
                        B = client_bases[name_l].float()          # (m_dim, r)
                        G_approx = B @ alpha_cols                 # (m_dim, n_dim)
                        dense[name_l] = G_approx.reshape(shape).float()
                    else:
                        # Basis unavailable — treat as zero update (conservative)
                        dense[name_l] = torch.zeros(shape, dtype=torch.float32)
                else:
                    ref = global_sd.get(name_l)
                    dense[name_l] = (
                        torch.zeros_like(ref, dtype=torch.float32)
                        if ref is not None
                        else torch.zeros(1)
                    )

            if agg is None:
                agg = OrderedDict({k: v.clone() * w_k for k, v in dense.items()})
            else:
                for k in dense:
                    if k in agg:
                        agg[k] += dense[k] * w_k
                    else:
                        agg[k] = dense[k] * w_k

        # ── B. Apply aggregated gradient ──────────────────────────────────────
        new_weights = OrderedDict()
        for k in global_sd:
            new_weights[k] = (
                global_sd[k].float() - agg[k].to(global_sd[k].device)
                if (agg is not None and k in agg)
                else global_sd[k].float()
            )
        del agg
        gc.collect()

        # ── C. Aggregate activation stats ─────────────────────────────────────
        mu_acc:    dict[str, float] = {}
        var_acc:   dict[str, float] = {}

        for (_, metadata, _), w_k in zip(client_updates, weights):
            for layer, st in metadata.get("local_stats", {}).items():
                if layer not in mu_acc:
                    mu_acc[layer]  = 0.0
                    var_acc[layer] = 0.0
                mu_acc[layer]  += w_k * st.get("mu", 0.0)
                var_acc[layer] += w_k * st.get("sigma", 1.0) ** 2

        new_global_stats = {
            layer: {
                "mu":    mu_acc[layer],
                "sigma": max(var_acc[layer], 1e-6) ** 0.5,
            }
            for layer in mu_acc
        }

        # ── D. Round-level metrics ────────────────────────────────────────────
        total_bytes   = sum(m["bytes_sent"]          for _, m, _ in client_updates)
        total_energy  = sum(m["energy_j_consumed"]   for _, m, _ in client_updates)
        avg_beta      = sum(m["beta_actual"]          for _, m, _ in client_updates) / K
        avg_battery   = sum(s.battery_j              for _, _, s in client_updates) / K
        avg_loss      = sum(m["local_loss"]           for _, m, _ in client_updates) / K
        avg_pressure  = sum(m.get("osmotic_pressure", 0.0) for _, m, _ in client_updates) / K
        avg_rank      = sum(m.get("avg_rank", 0.0)   for _, m, _ in client_updates) / K
        avg_drift     = sum(m.get("avg_drift", 0.0)  for _, m, _ in client_updates) / K
        avg_cr        = sum(m.get("compression_ratio", 1.0) for _, m, _ in client_updates) / K

        mode_counts: dict[str, int] = {}
        for _, m, _ in client_updates:
            for mode in m.get("mode_per_layer", {}).values():
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

        jain = 1.0   # universal participation

        metrics = {
            "round":                   round_num,
            "total_bytes_sent":        total_bytes,
            "total_energy_j":          total_energy,
            "avg_beta":                avg_beta,
            "avg_battery_j":           avg_battery,
            "avg_local_loss":          avg_loss,
            "avg_osmotic_pressure":    avg_pressure,
            "avg_rank":                avg_rank,
            "avg_drift":               avg_drift,
            "avg_compression_ratio":   avg_cr,
            "mode_counts":             mode_counts,
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
            "lr":                     0.01,
            "local_epochs":           1,
            "batch_size":             32,
            # Osmosis
            "gamma":                  0.1,    # osmotic pressure weight (0 = pure Resonance)
            "osmosis_global_stats":   {},     # populated by server after round 0
            # Resonance / SVD
            "rank_min":               4,
            "rank_max":               32,
            "spectral_energy_thresh": 0.95,
            "rank_coherence_thresh":  0.7,
            # Battery model
            "beta_min":               0.01,
            "beta_max":               1.0,
            "battery_max_j":          185400.0,
            # Error feedback
            "use_error_feedback":     True,
            # Device
            "device":                 "cpu",
            "device_profile":         None,
        }
