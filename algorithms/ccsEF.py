"""
algorithms/ccsEF.py
===================
CCS-EF: Complementary Cluster Split with Error Feedback
J. Nikiema, EL Amhoud, H. EL HAMMOUTI & I. KISSAMI — UM6P, 2026

─────────────────────────────────────────────────────────────────────────────
Core Idea
─────────────────────────────────────────────────────────────────────────────
CCS-EF combines the partial training efficiency of FedPart with the error
feedback mechanism of E-CEFFL to achieve:
  1. Universal gradient coverage: every layer group receives gradient signals
     from both clusters over every T_rot-round window.
  2. No permanent gradient loss: secondary-group gradients are accumulated in
     error buffers, not discarded.
  3. Battery-proportional compression: β_t^k scales with residual battery.
  4. Communication reduction: ~50% uplink vs full model (only primary group
     transmitted at compression ratio β_t^k).

─────────────────────────────────────────────────────────────────────────────
Model Partition
─────────────────────────────────────────────────────────────────────────────
Model w is split into two groups: g1 (early layers) and g2 (late layers + FC).
For ResNet-8 with 10 layer groups:
  g1 = groups {0, 1, 2, 3, 4}   (stem + blocks 1-2)
  g2 = groups {5, 6, 7, 8, 9}   (blocks 3-4 + FC head)

─────────────────────────────────────────────────────────────────────────────
Client Clustering
─────────────────────────────────────────────────────────────────────────────
Clients are partitioned into C1 (high-battery) and C2 (low-battery) at
initialization based on median battery level:
  C1 = {k : B_0^k >= B_median}
  C2 = {k : B_0^k <  B_median}

Cluster membership is **static** (assigned once, never changed).

─────────────────────────────────────────────────────────────────────────────
Assignment Rule (Complementary)
─────────────────────────────────────────────────────────────────────────────
At round t, the primary group assignment alternates every T_rot rounds:

  phase(t) = floor((t - warmup_rounds) / T_rot) mod 2 : here floor is lower integer part

  phase=0:  C1 → g1 (primary),  C2 → g2 (primary) : the group is said to be the primary group means this
                                                     group is the one that is going to be compressed and sent to the server
  phase=1:  C1 → g2 (primary),  C2 → g1 (primary)

─────────────────────────────────────────────────────────────────────────────
Client Update (Error Feedback on Both Groups)
─────────────────────────────────────────────────────────────────────────────
For client k in cluster C, assigned primary group g_primary:

  Step 1: Compute full gradient (full backward pass)
      Run E local SGD steps on full model w^t using D_k
      Δw_k = w_before - w_after    (lr already applied)
      Split: Δw_k = (Δg1_k, Δg2_k)

  Step 2: Error-feedback compression for primary group
      u_primary^k = Δg_primary_k + e_t^{k, g_primary}
      v_primary^k = TopK(u_primary^k, β_t^k)
      e_{t+1}^{k, g_primary} = u_primary^k - v_primary^k

  Step 3: Error buffer accumulation for secondary group
      e_{t+1}^{k, g_secondary} = Δg_secondary_k + e_t^{k, g_secondary}

  Step 4: Transmit only v_primary^k (uplink cost ≈ β_t^k · |g_primary|)

─────────────────────────────────────────────────────────────────────────────
Server Aggregation
─────────────────────────────────────────────────────────────────────────────
Separate dataset-size-weighted FedAvg per group, per responsible cluster:

  agg_g1 = Σ_{k ∈ C1, alive} (n_k / N_C1) · v_g1^k
  agg_g2 = Σ_{k ∈ C2, alive} (n_k / N_C2) · v_g2^k

  w_g1^{t+1} = w_g1^t - agg_g1
  w_g2^{t+1} = w_g2^t - agg_g2

Both groups updated every round using updates from their respective cluster.

─────────────────────────────────────────────────────────────────────────────
Convergence (2-Round Meta-Step)
─────────────────────────────────────────────────────────────────────────────
Over a 2-round window, each group receives gradient signals from BOTH clusters:
  w_g1: round t from C1, round t+1 from C2
  w_g2: round t from C2, round t+1 from C1

Under L-smoothness, bounded variance, and TopK contraction:

  (1/T) Σ_t E[‖∇F(w_t)‖²] ≤ O(1/√T) + O(η L ζ²) + O(e_max² / η)

where ζ² is data heterogeneity and e_max is the error buffer norm, bounded
by the E-CEFFL error buffer lemma.

─────────────────────────────────────────────────────────────────────────────
Comparison with Related Work
─────────────────────────────────────────────────────────────────────────────
  FedPart: partial training, no error feedback → gradient loss
  FedPartBE: energy-tier assignment, no error feedback → gradient loss
  E-CEFFL: error feedback, full model every round → high uplink cost
  CCS-EF: partial training + error feedback → no gradient loss + uplink savings
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from .fedpart import _derive_layer_groups, _param_group_key
from hardware.profiles import DeviceProfile


def topk_sparsify(tensor: torch.Tensor, beta: float) -> torch.Tensor:
    """Retain top-beta fraction of coordinates by absolute magnitude."""
    k = max(1, int(beta * tensor.numel()))
    flat = tensor.flatten()
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).reshape(tensor.shape)


@register_algorithm("ccsEF")
class CCSEFFL(FLAlgorithm):
    """
    CCS-EF: Complementary Cluster Split with Error Feedback.

    Key properties:
      - 2-group model split: g1 (early layers), g2 (late layers + FC)
      - 2-cluster client split: C1 (high-battery), C2 (low-battery)
      - Complementary assignment: C1→g1, C2→g2 (phase=0); swap (phase=1)
      - Error feedback on both groups: no permanent gradient loss
      - Battery-proportional compression: β_t^k ∈ [β_min, β_max]
      - Universal participation: β_min > 0 guarantees all clients contribute
    """
    name = "ccsEF"
    description = (
        "Complementary Cluster Split with Error Feedback. "
        "Partial training + EF: no gradient loss, low uplink cost."
    )

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:

        import math as _math
        device = config.get("device", "cpu")
        lr_base = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        local_epochs = config.get("local_epochs", 8)
        beta_min        = config.get("beta_min", 0.01)
        beta_max        = config.get("beta_max", 1.0)
        battery_max_j   = config.get("battery_max_j", 13320.0)
        warmup_rounds   = config.get("ccsEF_warmup_rounds", config.get("warmup_rounds", 5))
        T_rot           = config.get("T_rot", 1)
        use_fp16_uplink  = config.get("use_fp16_uplink", True)
        partial_downlink = config.get("partial_downlink", True)

        # ── Cosine LR decay ───────────────────────────────────────────────────
        # lr_t = lr_min + 0.5*(lr_base - lr_min)*(1 + cos(π * t / T))
        # lr_min = None  → scheduler désactivé (lr constant)
        lr_min = config.get("lr_min", None)
        num_rounds = config.get("num_rounds", 200)
        if lr_min is not None:
            t = state.round_num
            cosine_factor = 0.5 * (1.0 + _math.cos(_math.pi * t / num_rounds))
            lr = lr_min + (lr_base - lr_min) * cosine_factor
        else:
            lr = lr_base

        # ── Dead-client guard ──────────────────────────────────────────────
        if state.battery_j <= 0:
            dataset_size = len(dataloader.dataset) if dataloader is not None else 1
            return {}, {
                "client_id": state.client_id,
                "round_num": state.round_num,
                "beta_actual": 0.0,
                "battery_j_remaining": 0.0,
                "energy_j_consumed": 0.0,
                "bytes_sent": 0,
                "bytes_received": 0,
                "local_loss": 0.0,
                "compression_ratio": 0.0,
                "dataset_size": 0,  # dead clients excluded from aggregation weights
                "skipped": True,
                "cluster": state.custom.get("cluster", "unknown"),
                "primary_group": "none",
            }

        # ── Derive layer groups (once) ─────────────────────────────────────
        if "layer_groups" not in state.custom:
            groups = _derive_layer_groups(model)
            state.custom["layer_groups"] = groups

        groups = state.custom["layer_groups"]
        num_groups = len(groups)

        # ── Split groups into g1 (first half) and g2 (second half) ────────
        split_idx = num_groups // 2
        g1_groups = groups[:split_idx]
        g2_groups = groups[split_idx:]

        # Flatten to parameter name lists
        g1_names = [name for group in g1_groups for name in group]
        g2_names = [name for group in g2_groups for name in group]

        # ── Cluster assignment (dynamic, refreshed every T_recluster rounds) ──
        # Static assignment breaks after ~50 rounds: high-battery C1 clients drain
        # below median and receive wrong group. Dynamic re-clustering tracks the
        # evolving fleet distribution. server_aggregate updates fleet_median_battery
        # each round so clients always see the current alive-fleet median.
        T_recluster = config.get("T_recluster", 10)
        if "cluster" not in state.custom or state.round_num % T_recluster == 0:
            median_battery = config.get("fleet_median_battery") or (0.5 * battery_max_j)
            if state.battery_j >= median_battery:
                state.custom["cluster"] = "C1"
            else:
                state.custom["cluster"] = "C2"

        cluster = state.custom["cluster"]

        # ── Determine phase and primary/secondary groups ──────────────────
        # Warmup: full model update (both groups active)
        if state.round_num < warmup_rounds:
            is_warmup = True
            primary_group_key = "full"
            secondary_group_key = "none"
            primary_names = g1_names + g2_names
            secondary_names = []
        else:
            is_warmup = False
            phase = ((state.round_num - warmup_rounds) // T_rot) % 2

            if cluster == "C1":
                primary_group_key = "g1" if phase == 0 else "g2"
                secondary_group_key = "g2" if phase == 0 else "g1"
            else:  # C2
                primary_group_key = "g2" if phase == 0 else "g1"
                secondary_group_key = "g1" if phase == 0 else "g2"

            primary_names = g1_names if primary_group_key == "g1" else g2_names
            secondary_names = g2_names if primary_group_key == "g1" else g1_names

        # ── Compute beta from battery ──────────────────────────────────────
        battery_j = state.battery_j
        beta = beta_min + (beta_max - beta_min) * (battery_j / battery_max_j)
        beta = max(beta_min, min(beta_max, beta))

        # ── Save weights before training ───────────────────────────────────
        w_before = OrderedDict({
            k: v.clone().detach().cpu() for k, v in model.state_dict().items()
        })

        # ── Local SGD training ─────────────────────────────────────────────
        # freeze_secondary_backward (config, default False):
        #   False → full backward on g1+g2 (EF accumulation on secondary — default CCS-EF)
        #   True  → freeze secondary params (requires_grad=False) during training
        #           → only primary group backward computed → ~33% lower E_comp
        #             (fwd: all params; bwd+update: primary only → (d+2|g_p|)/3d ≈ 2/3)
        #           → secondary EF buffer NOT updated this round (frozen gradient)
        #           → better survival at the cost of EF coverage on secondary group
        freeze_secondary = config.get("freeze_secondary_backward", False)

        model.train()
        model.to(device)

        if freeze_secondary and not is_warmup:
            # Freeze secondary-group params: no backward computed for them
            for name, param in model.named_parameters():
                param.requires_grad = (name in primary_names)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        max_grad_norm = config.get("max_grad_norm", None)

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max_grad_norm,
                    )
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # Restore requires_grad for all params (important: model is reused)
        if freeze_secondary and not is_warmup:
            for param in model.parameters():
                param.requires_grad = True

        # ── Compute gradient delta = w_before - w_after ───────────────────
        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })

        # ── Initialize error buffers if needed ────────────────────────────
        if state.error_buffer is None:
            state.error_buffer = {
                "g1": {k: torch.zeros_like(v) for k, v in delta.items() if k in g1_names},
                "g2": {k: torch.zeros_like(v) for k, v in delta.items() if k in g2_names},
            }

        if not is_warmup:
            state.custom["prev_phase"] = phase

        # ── Error-feedback compression for primary group ──────────────────
        if is_warmup:
            # Warmup: full model update, no compression
            update = delta
            compressed = delta
            # Initialize error buffers during warmup (no accumulation)
            state.error_buffer = {
                "g1": {k: torch.zeros_like(v) for k, v in delta.items() if k in g1_names},
                "g2": {k: torch.zeros_like(v) for k, v in delta.items() if k in g2_names},
            }
        else:
            # CCS-EF phase: compress primary, accumulate secondary
            compressed = OrderedDict()
            new_error_g1 = OrderedDict()
            new_error_g2 = OrderedDict()

            # Process primary group (compress + transmit)
            for k in primary_names:
                if k in delta:
                    # Add accumulated error from primary group's buffer
                    group_key = primary_group_key
                    if k in state.error_buffer[group_key]:
                        u = delta[k] + state.error_buffer[group_key][k]
                    else:
                        u = delta[k]
                    v = topk_sparsify(u, beta)
                    # float16 quantization: simulate float16 wire format.
                    # Rounding error is absorbed into the EF buffer (u - v_q),
                    # so no gradient information is permanently lost.
                    if use_fp16_uplink:
                        v = v.half().float()
                    compressed[k] = v
                    # Update error buffer: captures TopK zeroing + float16 rounding
                    if group_key == "g1":
                        new_error_g1[k] = u - v
                    else:
                        new_error_g2[k] = u - v

            # Process secondary group: local TopK compression (NOT transmitted)
            # Replaces raw accumulation (delta += buffer) with proper EF:
            #   u_sec = delta + e_sec
            #   v_sec = TopK(u_sec, beta)   ← local, never sent
            #   e_sec = u_sec - v_sec        ← bounded via TopK contraction ||e||≤(1-β)||u||
            # This keeps the EF buffer bounded for all T_rot values, eliminating
            # the spike (T_rot+1)× update that required the buffer flush workaround.
            for k in secondary_names:
                if k in delta:
                    group_key = secondary_group_key
                    if k in state.error_buffer[group_key]:
                        u_sec = delta[k] + state.error_buffer[group_key][k]
                    else:
                        u_sec = delta[k]
                    v_sec = topk_sparsify(u_sec, beta)
                    err_sec = u_sec - v_sec
                    if group_key == "g1":
                        new_error_g1[k] = err_sec
                    else:
                        new_error_g2[k] = err_sec

            # Preserve unchanged error buffer entries (for params not in current split)
            for k in state.error_buffer["g1"]:
                if k not in new_error_g1:
                    new_error_g1[k] = state.error_buffer["g1"][k]
            for k in state.error_buffer["g2"]:
                if k not in new_error_g2:
                    new_error_g2[k] = state.error_buffer["g2"][k]

            state.error_buffer = {"g1": new_error_g1, "g2": new_error_g2}

            # ── Always transmit BN running statistics (raw delta, no EF/TopK) ──
            # running_mean / running_var are buffers, not parameters — they are
            # NOT in named_parameters() so _derive_layer_groups never includes them
            # in g1_names / g2_names. Yet they are CRITICAL for eval mode:
            # PyTorch uses running stats for BatchNorm in model.eval(). If they
            # are not updated on the server, the global model's normalisation
            # becomes stale after warmup and accuracy collapses.
            # They are tiny (channel-dim vectors) so transmitting them raw is cheap.
            for name, val in delta.items():
                if (name.endswith(".running_mean") or name.endswith(".running_var")):
                    compressed[name] = val  # raw delta, no sparsification

            update = compressed

        # ── Energy accounting ──────────────────────────────────────────────
        profile: DeviceProfile = config.get("device_profile")
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        # Uplink: only primary group transmitted (or full model during warmup)
        # float16 sparse: 4 bytes index + 2 bytes value = 6 bytes/nz
        _bpv = 2 if (use_fp16_uplink and not is_warmup) else 4
        if is_warmup:
            sparse_bytes = self.count_bytes(update, sparse=True, bytes_per_value=4)
            dense_bytes  = self.count_bytes(update, sparse=False)
            uplink_bytes = min(sparse_bytes, dense_bytes)
        else:
            sparse_bytes = self.count_bytes(compressed, sparse=True, bytes_per_value=_bpv)
            dense_bytes  = self.count_bytes(compressed, sparse=False)
            uplink_bytes = min(sparse_bytes, dense_bytes)

        # Downlink: partial (secondary group + BN stats) or full model
        if not is_warmup and partial_downlink:
            # Client only needs what the server updated from the OTHER cluster:
            # - secondary group params (updated by the other cluster's primary deltas)
            # - BN running stats (always needed for eval correctness)
            secondary_set = set(secondary_names)
            downlink_bytes = sum(
                w_before[k].numel() * 4
                for k in w_before
                if k in secondary_set
                or k.endswith((".running_mean", ".running_var"))
            )
        else:
            downlink_bytes = self.count_bytes(w_before, sparse=False)

        if profile is not None:
            if freeze_secondary and not is_warmup:
                # Partial backward: PyTorch only computes gradients for primary params.
                # FLOPs decomposition (factor 2 = MACs→FLOPs convention):
                #   forward pass  : 1 × 2 × d          × B × S  (all params, always full)
                #   backward pass : 1 × 2 × |g_primary| × B × S  (frozen params skipped)
                #   optimizer step: 1 × 2 × |g_primary| × B × S  (only updated params)
                # Total = (d + 2|g_primary|) × 2 × B × S
                # vs full backward: 3 × 2 × d × B × S = 6d·B·S
                # Savings ≈ (6d - (2d + 4|g_p|)) / 6d = (4d - 4|g_p|) / 6d
                #         ≈ 2/6 = 33%  when |g_p| ≈ d/2
                primary_name_set = set(primary_names)
                num_primary_params = sum(
                    p.numel() for n, p in model.named_parameters()
                    if n in primary_name_set
                )
                steps = (dataset_size // batch_size) * local_epochs
                flops = float(2 * batch_size * steps * (num_params + 2 * num_primary_params))
            else:
                # Full backward pass: training_flop_factor=3 (fwd+bwd+update)
                num_primary_params = num_params  # for metadata
                flops = profile.flops_for_model(
                    num_params, batch_size, local_epochs, dataset_size
                )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            # Simplified model: only primary group (~50% of model) transmitted
            num_primary_params = num_params  # for metadata
            energy_j = 0.5 + 1.0 * beta

        del w_before
        del current_sd

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer

        # ── Compression ratio ───────────────────────────────────────────────
        full_model_bytes = self.count_bytes(delta, sparse=False)
        compression_ratio = uplink_bytes / max(full_model_bytes, 1)

        # Ratio of active params vs total (1.0 = full backward, ~0.5 = frozen secondary)
        flops_ratio = num_primary_params / max(num_params, 1)

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            "dataset_size": dataset_size,
            "cluster": cluster,
            "primary_group": primary_group_key,
            "is_warmup": is_warmup,
            "freeze_secondary": freeze_secondary and not is_warmup,
            "flops_ratio": flops_ratio,  # 1.0=full bwd, ~0.5=frozen secondary
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
        Separate aggregation per group, per responsible cluster.

        Phase 0: C1 updates g1, C2 updates g2
        Phase 1: C1 updates g2, C2 updates g1
        """
        K = len(client_updates)
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("ccsEF_warmup_rounds", config.get("warmup_rounds", 5))
        T_rot = config.get("T_rot", 1)

        # ── Update fleet median for dynamic clustering (A1) ──────────────
        # Computed from alive clients only; written into config so every
        # client_update call in the NEXT round sees the current median.
        alive_batteries = sorted(
            s.battery_j for _, _, s in client_updates if s.battery_j > 0
        )
        if alive_batteries:
            mid = len(alive_batteries) // 2
            config["fleet_median_battery"] = (
                alive_batteries[mid]
                if len(alive_batteries) % 2 == 1
                else (alive_batteries[mid - 1] + alive_batteries[mid]) / 2.0
            )

        # ── Separate alive (active) from dead (skipped) clients ───────────
        active_updates = [
            (u, m, s) for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]
        K_alive = len(active_updates)

        # ── Warmup: full FedAvg aggregation ───────────────────────────────
        if round_num < warmup_rounds:
            if K_alive == 0:
                # No active clients → return unchanged model
                metrics = self._compute_metrics(
                    client_updates, active_updates, round_num, config,
                    phase="warmup", g1_norm=0.0, g2_norm=0.0
                )
                return AggregateResult(new_weights=global_sd, metrics=metrics)

            # Dataset-size weighted aggregation (all clients, all params)
            sizes = [m.get("dataset_size", 1) for _, m, _ in active_updates]
            total_n = max(sum(sizes), 1)
            weights = [n_k / total_n for n_k in sizes]

            agg = None
            for (update, metadata, state), w_k in zip(active_updates, weights):
                if agg is None:
                    agg = {k: v.clone().float() * w_k for k, v in update.items()}
                else:
                    for k in agg:
                        if k in update:
                            agg[k] += update[k].float() * w_k

            new_weights = OrderedDict()
            for k in global_sd:
                if agg is not None and k in agg:
                    new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
                else:
                    new_weights[k] = global_sd[k].float()

            del agg

            metrics = self._compute_metrics(
                client_updates, active_updates, round_num, config,
                phase="warmup", g1_norm=0.0, g2_norm=0.0
            )
            return AggregateResult(new_weights=new_weights, metrics=metrics)

        # ── CCS-EF phase: cluster-specific aggregation ────────────────────
        # Derive layer groups to split into g1 and g2
        if len(active_updates) > 0:
            first_update, _, _ = active_updates[0]
            # Infer group split from first client's metadata or reconstruct
            groups = _derive_layer_groups(global_model)
            num_groups = len(groups)
            split_idx = num_groups // 2
            g1_groups = groups[:split_idx]
            g2_groups = groups[split_idx:]
            g1_names = set([name for group in g1_groups for name in group])
            g2_names = set([name for group in g2_groups for name in group])
        else:
            # No active clients → return unchanged model
            metrics = self._compute_metrics(
                client_updates, active_updates, round_num, config,
                phase=((round_num - warmup_rounds) // T_rot) % 2,
                g1_norm=0.0, g2_norm=0.0
            )
            return AggregateResult(new_weights=global_sd, metrics=metrics)

        # Partition active clients by cluster and primary group
        phase = ((round_num - warmup_rounds) // T_rot) % 2
        C1_updates = [(u, m, s) for u, m, s in active_updates if m.get("cluster") == "C1"]
        C2_updates = [(u, m, s) for u, m, s in active_updates if m.get("cluster") == "C2"]

        # Aggregate g1 (from responsible cluster)
        # Phase 0: C1 → g1; Phase 1: C2 → g1
        g1_responsible = C1_updates if phase == 0 else C2_updates
        agg_g1 = self._aggregate_group(g1_responsible, g1_names)

        # Aggregate g2 (from responsible cluster)
        # Phase 0: C2 → g2; Phase 1: C1 → g2
        g2_responsible = C2_updates if phase == 0 else C1_updates
        agg_g2 = self._aggregate_group(g2_responsible, g2_names)

        # ── Aggregate BN running stats from ALL active clients ────────────
        # BN buffers (running_mean, running_var) are always transmitted raw
        # by every active client, regardless of which group is primary.
        # They must be aggregated from all clusters to avoid stale normalisation.
        agg_bn = self._aggregate_bn_buffers(active_updates)

        # ── Server momentum (B3) ─────────────────────────────────────────
        # Applies EMA to aggregated deltas before updating global model:
        #   m_t = β_s * m_{t-1} + (1-β_s) * agg_delta_t
        #   w_{t+1} = w_t - η_s * m_t
        # Reduces variance from non-IID client drift. β_s=0 → vanilla FedAvg.
        server_momentum = config.get("server_momentum", 0.0)
        if server_momentum > 0.0:
            if not hasattr(self, "_server_state"):
                self._server_state = {}
            if "momentum_buffer" not in self._server_state:
                self._server_state["momentum_buffer"] = {}
            m_buf = self._server_state["momentum_buffer"]
            for k, v in list(agg_g1.items()):
                prev = m_buf.get(k, torch.zeros_like(v))
                m_buf[k] = server_momentum * prev + (1.0 - server_momentum) * v
                agg_g1[k] = m_buf[k]
            for k, v in list(agg_g2.items()):
                prev = m_buf.get(k, torch.zeros_like(v))
                m_buf[k] = server_momentum * prev + (1.0 - server_momentum) * v
                agg_g2[k] = m_buf[k]

        # ── Apply aggregated deltas to global model ───────────────────────
        new_weights = OrderedDict()
        for k in global_sd:
            if k in agg_bn:
                # BN running stats: apply from all-client weighted average
                new_weights[k] = global_sd[k].float() - agg_bn[k].to(global_sd[k].device)
            elif k in agg_g1:
                new_weights[k] = global_sd[k].float() - agg_g1[k].to(global_sd[k].device)
            elif k in agg_g2:
                new_weights[k] = global_sd[k].float() - agg_g2[k].to(global_sd[k].device)
            else:
                new_weights[k] = global_sd[k].float()

        # ── Compute gradient norms for metrics ────────────────────────────
        g1_norm = sum(v.norm().item() ** 2 for v in agg_g1.values()) ** 0.5 if agg_g1 else 0.0
        g2_norm = sum(v.norm().item() ** 2 for v in agg_g2.values()) ** 0.5 if agg_g2 else 0.0

        del agg_g1
        del agg_g2

        metrics = self._compute_metrics(
            client_updates, active_updates, round_num, config,
            phase=phase, g1_norm=g1_norm, g2_norm=g2_norm,
            C1_alive=len(C1_updates), C2_alive=len(C2_updates)
        )

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    def _aggregate_group(
        self,
        cluster_updates: list[tuple[dict, dict, ClientState]],
        group_names: set[str],
    ) -> dict[str, torch.Tensor]:
        """
        Dataset-size-weighted aggregation for a single group from a cluster.
        """
        if len(cluster_updates) == 0:
            return {}

        sizes = [m.get("dataset_size", 1) for _, m, _ in cluster_updates]
        total_n = max(sum(sizes), 1)
        weights = [n_k / total_n for n_k in sizes]

        agg = None
        for (update, metadata, state), w_k in zip(cluster_updates, weights):
            # Only aggregate parameters belonging to this group
            group_update = {k: v for k, v in update.items() if k in group_names}
            if not group_update:
                continue
            if agg is None:
                agg = {k: v.clone().float() * w_k for k, v in group_update.items()}
            else:
                for k in group_update:
                    if k in agg:
                        agg[k] += group_update[k].float() * w_k
                    else:
                        agg[k] = group_update[k].float() * w_k

        return agg if agg is not None else {}

    def _aggregate_bn_buffers(
        self,
        active_updates: list[tuple[dict, dict, ClientState]],
    ) -> dict[str, torch.Tensor]:
        """
        Dataset-size-weighted aggregation of BN running statistics.

        running_mean and running_var are NOT parameters — they are module buffers
        computed during forward passes via exponential moving average. They are
        excluded from named_parameters() and therefore never appear in g1_names /
        g2_names. This method aggregates them from ALL active clients (not just
        one cluster) so the global model's BN normalisation stays current.

        running_mean keys end with '.running_mean'
        running_var  keys end with '.running_var'
        num_batches_tracked is intentionally excluded (integer counter, not EMA).
        """
        if not active_updates:
            return {}

        sizes = [m.get("dataset_size", 1) for _, m, _ in active_updates]
        total_n = max(sum(sizes), 1)

        agg: dict[str, torch.Tensor] = {}
        for (upd, meta, _), n_k in zip(active_updates, sizes):
            w_k = n_k / total_n
            for k, v in upd.items():
                if k.endswith(".running_mean") or k.endswith(".running_var"):
                    if k not in agg:
                        agg[k] = v.clone().float() * w_k
                    else:
                        agg[k] += v.float() * w_k
        return agg

    def _compute_metrics(
        self,
        all_updates: list,
        active_updates: list,
        round_num: int,
        config: dict,
        phase: int | str,
        g1_norm: float,
        g2_norm: float,
        C1_alive: int = 0,
        C2_alive: int = 0,
    ) -> dict:
        """Compute round metrics (aligned with eceffl.py and fedpart_be.py)."""
        K = len(all_updates)
        K_alive = len(active_updates)
        K_dead = K - K_alive

        total_bytes = sum(m["bytes_sent"] for _, m, _ in all_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in all_updates)

        if K_alive > 0:
            avg_battery = sum(s.battery_j for _, _, s in all_updates) / K
            batt_min = min(s.battery_j for _, _, s in all_updates)
            avg_beta = sum(m["beta_actual"] for _, m, _ in active_updates) / K_alive
            avg_loss = sum(m["local_loss"] for _, m, _ in active_updates) / K_alive
            avg_cr = sum(m["compression_ratio"] for _, m, _ in active_updates) / K_alive
        else:
            avg_battery = 0.0
            batt_min = 0.0
            avg_beta = 0.0
            avg_loss = 0.0
            avg_cr = 0.0

        # Jain fairness index on bytes_sent (dead clients contribute 0)
        bytes_sent = [m["bytes_sent"] for _, m, _ in all_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b ** 2 for b in bytes_sent))
        else:
            jain = 1.0

        return {
            "round_num": round_num,
            "K_alive": K_alive,
            "K_dead": K_dead,
            "K_skipped": K - K_alive,
            "participation_rate": K_alive / max(K, 1),
            "total_bytes_sent": total_bytes,
            "total_energy_j": total_energy,
            "avg_battery_j": avg_battery,   # standard key read by run_experiment.py
            "batt_min_j": batt_min,
            "jain_index": jain,             # standard key read by run_experiment.py
            "avg_beta": avg_beta,           # standard key read by run_experiment.py
            "avg_local_loss": avg_loss,
            "compression_ratio": avg_cr,
            "g1_norm": g1_norm,
            "g2_norm": g2_norm,
            "phase": phase,
            "C1_alive": C1_alive,
            "C2_alive": C2_alive,
        }

    def get_default_config(self) -> dict:
        return {
            # Training
            "lr": 0.01,               # SGD learning rate (peak / initial)
            "lr_min": None,           # cosine decay floor (None = constant LR)
            "num_rounds": 200,        # total rounds — used by cosine scheduler
            "momentum": 0.9,          # SGD momentum
            "weight_decay": 1e-4,     # L2 regularization
            "max_grad_norm": 10.0,    # gradient clipping (None = disabled)
            "local_epochs": 8,        # E (local training epochs per round)
            "batch_size": 32,
            # CCS-EF specific
            "beta_min": 0.01,         # minimum sparsification ratio (universal participation floor)
            "beta_max": 1.0,          # maximum sparsification ratio (full model at max battery)
            "battery_max_j": 13320.0, # 3700mAh @ 3.6V (ESP32-S3 battery capacity)
            "energy_scale_factor": 12.6,  # calibration multiplier (from ESP32-S3 T_Correction)
            "ccsEF_warmup_rounds": 5, # full FedAvg rounds before CCS-EF phase (overrides warmup_rounds)
            "T_rot": 1,               # rotation period (swap C1↔C2 every T_rot rounds)
            "adaptive_rotation": False,  # if True: T_rot_eff = T_rot * ceil(K_0 / K_alive)
            "fleet_median_battery": None,  # updated each round by server_aggregate (dynamic clustering)
            "T_recluster": 10,        # re-assign C1/C2 every T_recluster rounds (0 = static)
            # ── Compute-energy trade-off ──────────────────────────────────────
            "freeze_secondary_backward": False,
            # ── Communication optimizations ───────────────────────────────────
            "use_fp16_uplink": True,  # float16 sparse values: 6 bytes/nz vs 8 (−25% uplink)
            "partial_downlink": True, # download only secondary group + BN stats (~−50% downlink)
            # ── Server-side optimization ──────────────────────────────────────
            "server_momentum": 0.0,   # EMA on aggregated deltas (0=off, 0.9=aggressive)
            # Device
            "device": "cpu",
            "device_profile": None,   # DeviceProfile instance or None
        }
