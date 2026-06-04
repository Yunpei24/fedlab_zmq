"""
algorithms/ccsEF.py
===================
CCS-EF: Complementary Cluster Split with Error Feedback
J. Nikiema, EL Amhoud, H. EL HAMMOUTI & I. KISSAMI — UM6P, 2026

─────────────────────────────────────────────────────────────────────────────
Core Idea
─────────────────────────────────────────────────────────────────────────────
CCS-EF combines the partial training efficiency of FedPart with the error
feedback mechanism of E-CEFFL.  The number of partitions M is a free parameter
(num_partitions, default 2):

  1. Universal gradient coverage: over every M-phase window, every partition
     receives gradient signals from all M clusters.
  2. No permanent gradient loss: secondary-partition gradients accumulate in
     error buffers and are deferred, never discarded.
  3. Battery-proportional compression: β_t^k ∈ [β_min, β_max] scales linearly
     with the residual battery of client k.
  4. Communication reduction: only the primary partition (~1/M of the model)
     is uploaded, further sparsified at ratio β_t^k.
     Expected uplink savings vs full-model: factor ≈ M / β_t^k.

─────────────────────────────────────────────────────────────────────────────
Model Partition  (M contiguous partitions)
─────────────────────────────────────────────────────────────────────────────
The model w is split into M contiguous partitions ordered from shallowest to
deepest layers:

  g0 (input layers) | g1 | … | g{M-1} (output + FC head)

The split is computed once at the start of training by _make_partitions(),
which divides the list of layer groups into M equal-size bins.  Batch-norm
running statistics are treated separately — they are always transmitted dense
by all clients regardless of partition assignment.

─────────────────────────────────────────────────────────────────────────────
Client Clustering  (M clusters, M-quantile battery split)
─────────────────────────────────────────────────────────────────────────────
Clients are sorted by residual battery and divided into M equal-quantile
clusters at each reclustering event:

  C0 = lowest-battery quantile … C{M-1} = highest-battery quantile

The server computes M-1 battery thresholds from the alive-fleet distribution
each round and broadcasts them so clients can self-assign.  Re-evaluation
occurs every T_recluster rounds (default 10) to track battery drain.

─────────────────────────────────────────────────────────────────────────────
Assignment Rule  (Complementary Rotation)
─────────────────────────────────────────────────────────────────────────────
After the warmup phase, a global phase counter advances every T_rot rounds:

  φ(t) = floor( (t − t_warmup) / T_rot ) mod M

Cluster C_i is assigned primary partition g_{(i+φ) mod M}.

Consequence: over any M consecutive phases (M·T_rot rounds), every cluster
trains every partition exactly once → universal gradient coverage guaranteed.
Server inverse: partition g is aggregated from cluster C_{(g−φ) mod M}.

─────────────────────────────────────────────────────────────────────────────
Client Update  (Error Feedback on All M Partitions)
─────────────────────────────────────────────────────────────────────────────
Client k ∈ C_i, primary partition index p = (i + φ) mod M:

  Step 1 — Local SGD  (E epochs on full model unless freeze_secondary=True)
      Δw^k = w_before − w_after

  Step 2 — EF + TopK on primary partition (transmitted)
      u_p^k   = Δg_p^k + e_t^{k,p}
      v_p^k   = TopK(u_p^k, β_t^k)       ← sparse, FP16 if use_fp16_uplink
      e_{t+1}^{k,p} = u_p^k − v_p^k

  Step 3 — EF accumulation on each secondary partition s ≠ p  (local only)
      [freeze_secondary=False]  full backward was computed:
          u_s^k = Δg_s^k + e_t^{k,s}
          v_s^k = TopK(u_s^k, β_t^k)     ← never transmitted
          e_{t+1}^{k,s} = u_s^k − v_s^k  ← ||e_s|| ≤ (1−β)||u_s||
      [freeze_secondary=True]  secondary params have requires_grad=False
          Δg_s^k = 0  →  buffer carries forward unchanged:
          e_{t+1}^{k,s} = e_t^{k,s}      ← no shrinkage, ~(M-1)/M FLOPs saved

  Step 4 — Transmit v_p^k only
      Uplink ≈ β_t^k · |g_p| / M  (sparse + FP16)

  β_t^k = β_min + (β_max − β_min) · B_t^k / B_max
  β_t^k ≥ β_min > 0  →  universal participation guaranteed

─────────────────────────────────────────────────────────────────────────────
Server Aggregation  (per-partition, per-responsible-cluster)
─────────────────────────────────────────────────────────────────────────────
At phase φ, partition g is aggregated from cluster C_{(g−φ) mod M}:

  agg_g = Σ_{k ∈ C_{(g−φ) mod M}, alive}  (n_k / N_C) · v_g^k
  w_g^{t+1} = w_g^t − agg_g

All M partitions are updated every round.  BN running stats are aggregated
globally (weighted average over all alive clients).  Optional server momentum
(server_momentum ∈ [0,1]) smooths the per-partition aggregated deltas.

─────────────────────────────────────────────────────────────────────────────
Warmup Phase  (rounds 0 … t_warmup − 1)
─────────────────────────────────────────────────────────────────────────────
During warmup the algorithm behaves as FedAvg: all clients send the full model
delta (dense, no sparsification, no partitioning).  Error buffers are
initialised to zero and remain untouched.  local_epochs is forced to 8
regardless of the config value, to ensure a stable starting point.

─────────────────────────────────────────────────────────────────────────────
Convergence  (M-Round Meta-Step)
─────────────────────────────────────────────────────────────────────────────
Over an M-round window, each partition receives gradient signals from all M
clusters.  Under L-smoothness, bounded heterogeneity, and TopK contraction:

  (1/T) Σ_t E[‖∇F(w_t)‖²] ≤ O(1/√T) + O(η L ζ²) + O(e_max² / η)

The error-buffer term O(e_max²/η) vanishes as β → 1 (low compression).

─────────────────────────────────────────────────────────────────────────────
Comparison with Related Work
─────────────────────────────────────────────────────────────────────────────
  FedPart      : partial training, no EF           → permanent gradient loss
  FedPartBE    : energy-tier assignment, no EF     → permanent gradient loss
  E-CEFFL      : EF, full model every round        → full uplink cost
  CCS-EF (M=2) : partial training + EF, 2 clusters → ~50% uplink, no loss
  CCS-EF (M≥3) : same, M clusters                 → ~1/M uplink, no loss,
                  finer battery-heterogeneity tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.flop_cost import round_compute_flops
from .fedpart import _derive_layer_groups
from hardware.profiles import DeviceProfile


def topk_sparsify(tensor: torch.Tensor, beta: float) -> torch.Tensor:
    """Retain top-beta fraction of coordinates by absolute magnitude."""
    k = max(1, int(beta * tensor.numel()))
    flat = tensor.flatten()
    _, idx = torch.topk(flat.abs(), k)
    mask = torch.zeros_like(flat)
    mask[idx] = 1.0
    return (flat * mask).reshape(tensor.shape)


def _make_partitions(
    layer_groups: list[list[str]], M: int
) -> tuple[list[list[str]], list[set[str]]]:
    """
    Split layer_groups into M contiguous partitions.

    Returns:
        partition_names: list of M name-lists (flat param names per partition)
        partition_name_sets: list of M sets for O(1) membership tests
    """
    num_groups = len(layer_groups)
    M = min(M, max(num_groups, 1))   # can't have more partitions than layer groups
    part_size = max(1, num_groups // M)
    partition_groups = [
        layer_groups[i * part_size : (i + 1) * part_size if i < M - 1 else num_groups]
        for i in range(M)
    ]
    partition_names = [
        [name for grp in part for name in grp]
        for part in partition_groups
    ]
    partition_name_sets = [set(names) for names in partition_names]
    return partition_names, partition_name_sets


@register_algorithm("ccsEF")
class CCSEFFL(FLAlgorithm):
    """
    CCS-EF: Complementary Cluster Split with Error Feedback (M-partition).

    Key properties:
      - M-partition model split (configurable via num_partitions, default 2)
      - M-cluster client split by battery quantile (C0=lowest … C{M-1}=highest)
      - Complementary rotation: cluster i → partition (i+phase)%M each T_rot rounds
      - Error feedback on all M partitions: no permanent gradient loss
      - Battery-proportional compression: β_t^k ∈ [β_min, β_max]
      - Universal participation: β_min > 0 guarantees all clients contribute
    """
    name = "ccsEF"
    description = (
        "Complementary Cluster Split with Error Feedback (M-partition). "
        "Partial training + EF: no gradient loss, ~1/M uplink cost."
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
        M               = config.get("num_partitions", 2)

        # ── Sync round number with server (prevents phase drift on skipped rounds) ──
        # If run_experiment.py injects config["round_num"], use it as authoritative.
        # Otherwise fall back to state.round_num and increment at the end of this call.
        if "round_num" in config:
            round_num = config["round_num"]
            state.round_num = round_num
        else:
            round_num = state.round_num

        # ── Cosine LR decay ───────────────────────────────────────────────────
        lr_min = config.get("lr_min", None)
        num_rounds = config.get("num_rounds", 200)
        if lr_min is not None:
            cosine_factor = 0.5 * (1.0 + _math.cos(_math.pi * round_num / num_rounds))
            lr = lr_min + (lr_base - lr_min) * cosine_factor
        else:
            lr = lr_base

        # ── Dead-client guard ──────────────────────────────────────────────
        if state.battery_j <= 0:
            dataset_size = len(dataloader.dataset) if dataloader is not None else 1
            return {}, {
                "client_id": state.client_id,
                "round_num": round_num,
                "beta_actual": 0.0,
                "battery_j_remaining": 0.0,
                "energy_j_consumed": 0.0,
                "bytes_sent": 0,
                "bytes_received": 0,
                "local_loss": 0.0,
                "compression_ratio": 0.0,
                "dataset_size": 0,
                "skipped": True,
                "cluster": f"C{state.custom.get('cluster_idx', -1)}"
                           if "cluster_idx" in state.custom
                           else state.custom.get("cluster", "unknown"),
                "cluster_idx": state.custom.get("cluster_idx", -1),
                "primary_group": "none",
                "num_partitions": M,
            }

        # ── Derive layer groups (once) ─────────────────────────────────────
        if "layer_groups" not in state.custom:
            state.custom["layer_groups"] = _derive_layer_groups(model)

        layer_groups = state.custom["layer_groups"]
        num_layer_groups = len(layer_groups)
        M = min(M, max(num_layer_groups, 1))  # clamp to valid range

        # ── Split into M contiguous partitions ────────────────────────────
        partition_names, partition_name_sets = _make_partitions(layer_groups, M)

        # ── Cluster assignment (dynamic, refreshed every T_recluster rounds) ──
        # C0 = lowest battery quantile, C{M-1} = highest battery quantile.
        # fleet_battery_thresholds: list of M-1 thresholds broadcast by the server.
        T_recluster = config.get("T_recluster", 10)
        if "cluster_idx" not in state.custom or round_num % T_recluster == 0:
            thresholds = config.get("fleet_battery_thresholds") or [
                battery_max_j * (i + 1) / M for i in range(M - 1)
            ]
            cidx = sum(1 for thr in sorted(thresholds) if state.battery_j >= thr)
            state.custom["cluster_idx"] = min(cidx, M - 1)

        cluster_idx = state.custom["cluster_idx"]
        cluster = f"C{cluster_idx}"

        # ── Detect cluster change (for Option 5 priming) ──────────────────
        _prev_cluster = state.custom.get("_prev_cluster_idx")
        state.custom["_prev_cluster_idx"] = cluster_idx

        # ── Determine phase and primary/secondary partitions ──────────────
        if round_num < warmup_rounds:
            is_warmup = True
            phase = 0
            primary_idx = None
            secondary_indices = []
            primary_names = [n for names in partition_names for n in names]
            secondary_names = []
        else:
            is_warmup = False
            phase = ((round_num - warmup_rounds) // T_rot) % M
            # Complementary rotation: cluster i trains partition (i + phase) % M
            primary_idx = (cluster_idx + phase) % M
            secondary_indices = [i for i in range(M) if i != primary_idx]
            primary_names = partition_names[primary_idx]
            secondary_names = [n for i in secondary_indices for n in partition_names[i]]

        # Warmup rounds always use 8 local epochs for stable initialisation.
        if is_warmup:
            local_epochs = 3

        # ── Option 5 — priming flag ───────────────────────────────────────
        # A priming round runs local SGD + EF to prime the buffer for the new
        # primary partition after a cluster change, but transmits nothing.
        # Cost: compute + downlink only (no uplink). Benefit: first real
        # transmission after cluster change uses a fresh, properly-primed buffer.
        is_priming = (
            config.get("buffer_priming", True)
            and not is_warmup
            and _prev_cluster is not None
            and _prev_cluster != cluster_idx
        )

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
        #   False → full backward on all partitions (EF accumulation on secondary)
        #   True  → freeze secondary params → only primary backward → ~(M-1)/M lower E_comp
        #           → secondary EF buffer carries forward unchanged this round
        freeze_secondary = config.get("freeze_secondary_backward", False)

        model.train()
        model.to(device)

        if freeze_secondary and not is_warmup:
            primary_name_set = set(primary_names)
            for name, param in model.named_parameters():
                param.requires_grad = (name in primary_name_set)

        optimizer_type = config.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay,
            )
        else:
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
                f"g{i}": {k: torch.zeros_like(v) for k, v in delta.items() if k in partition_name_sets[i]}
                for i in range(M)
            }
            # Initialise timestamps at current round so first-use age = 0.
            state.custom["buffer_last_update"] = {f"g{i}": round_num for i in range(M)}

        if not is_warmup:
            state.custom["prev_phase"] = phase

        # ── Error-feedback compression ────────────────────────────────────
        if is_warmup:
            # Warmup: full model, no compression. Error buffer already zeros; no re-init.
            update = delta
            compressed = delta
        else:
            compressed = OrderedDict()
            new_error = {f"g{i}": OrderedDict() for i in range(M)}
            primary_key = f"g{primary_idx}"

            # ── Buffer staleness check (Option 4) ─────────────────────────
            # Activatable via config: buffer_age_reset (default True).
            # The primary buffer must be updated every M*T_rot rounds.
            # A larger age means the client changed cluster (freeze_secondary
            # kept the buffer frozen while it was secondary in the old cluster).
            # Reset to zero so the stale accumulated error doesn't corrupt the
            # update — the next round will rebuild it from the fresh gradient.
            # For ccsEF_full, secondary buffers are updated every round so
            # their age is always ≤ 1: this check never fires.
            if config.get("buffer_age_reset", True):
                _last_updates = state.custom.get("buffer_last_update", {})
                _buf_age = round_num - _last_updates.get(primary_key, round_num)
                if _buf_age > M * T_rot and primary_key in state.error_buffer:
                    for _k in state.error_buffer[primary_key]:
                        state.error_buffer[primary_key][_k] = torch.zeros_like(
                            state.error_buffer[primary_key][_k]
                        )

            # Primary partition: EF + TopK → transmitted (or primed if is_priming)
            ef_clip_norm = config.get("ef_clip_norm", None)
            for k in primary_names:
                if k in delta:
                    e_buf = state.error_buffer.get(primary_key, {}).get(
                        k, torch.zeros_like(delta[k])
                    )
                    u = delta[k] + e_buf
                    # Clip u to prevent buffer explosion when β is small:
                    # without clipping, (1-β)>0.5 can cause ||e|| to grow without
                    # bound as gradients shrink late in training → catastrophic drops.
                    if ef_clip_norm is not None:
                        u_norm = u.norm().item()
                        if u_norm > ef_clip_norm:
                            u = u * (ef_clip_norm / u_norm)
                    v = topk_sparsify(u, beta)
                    # FP16 quantisation: rounding error absorbed into EF buffer.
                    if use_fp16_uplink:
                        v = v.half().float()
                    compressed[k] = v
                    new_error[primary_key][k] = u - v

            # Secondary partitions: carry forward (frozen) or local EF (full backward)
            if freeze_secondary:
                # delta[k] = 0 for secondary params (no backward) → carry buffer forward.
                for sec_idx in secondary_indices:
                    sec_key = f"g{sec_idx}"
                    for k in partition_names[sec_idx]:
                        if k in state.error_buffer.get(sec_key, {}):
                            new_error[sec_key][k] = state.error_buffer[sec_key][k]
            else:
                # Full backward was computed: local EF to bound each secondary buffer.
                # v_sec is never transmitted; ||e_sec|| ≤ (1-β)||u_sec|| (TopK contraction).
                for sec_idx in secondary_indices:
                    sec_key = f"g{sec_idx}"
                    for k in partition_names[sec_idx]:
                        if k in delta:
                            e_buf = state.error_buffer.get(sec_key, {}).get(
                                k, torch.zeros_like(delta[k])
                            )
                            u_sec = delta[k] + e_buf
                            if ef_clip_norm is not None:
                                u_norm = u_sec.norm().item()
                                if u_norm > ef_clip_norm:
                                    u_sec = u_sec * (ef_clip_norm / u_norm)
                            v_sec = topk_sparsify(u_sec, beta)
                            new_error[sec_key][k] = u_sec - v_sec

            # Preserve buffer entries not updated this round
            for i in range(M):
                gi_key = f"g{i}"
                for k in state.error_buffer.get(gi_key, {}):
                    if k not in new_error[gi_key]:
                        new_error[gi_key][k] = state.error_buffer[gi_key][k]

            state.error_buffer = new_error

            # ── Update buffer timestamps ───────────────────────────────────
            # Primary is always updated this round.
            # Secondaries are updated only in ccsEF_full (not frozen).
            if "buffer_last_update" not in state.custom:
                state.custom["buffer_last_update"] = {f"g{i}": round_num for i in range(M)}
            state.custom["buffer_last_update"][primary_key] = round_num
            if not freeze_secondary:
                for _sec_idx in secondary_indices:
                    state.custom["buffer_last_update"][f"g{_sec_idx}"] = round_num

            # BN running stats: always transmitted raw (no TopK) — same as fedpart_be.
            # They are buffers, not parameters, so not in any partition_names.
            for name, val in delta.items():
                if name.endswith((".running_mean", ".running_var")):
                    compressed[name] = val

            update = compressed

        # ── Option 5: priming round — suppress transmission ───────────────
        # Buffer has been primed (EF applied). Override the update to empty so
        # the server receives nothing from this client this round.
        # Energy: compute + downlink only (no uplink cost).
        if is_priming and not is_warmup:
            update = {}
            compressed = {}

        # ── Uplink bytes ───────────────────────────────────────────────────
        # BN stats are raw delta (no TopK) → always dense (4 bytes/elem), like fedpart_be.
        # Primary partition is TopK'd → take min(sparse_fp16, dense_fp32).
        _bn_suffixes = (".running_mean", ".running_var")
        _bpv = 2 if (use_fp16_uplink and not is_warmup) else 4
        if is_warmup:
            # Full model, no TopK → always dense (sparse >= dense for full delta)
            uplink_bytes = self.count_bytes(update, sparse=False)
        else:
            _primary_dict = {k: v for k, v in compressed.items() if not k.endswith(_bn_suffixes)}
            _bn_dict      = {k: v for k, v in compressed.items() if k.endswith(_bn_suffixes)}
            _sparse_primary = self.count_bytes(_primary_dict, sparse=True, bytes_per_value=_bpv)
            _dense_primary  = self.count_bytes(_primary_dict, sparse=False)
            _bn_bytes       = self.count_bytes(_bn_dict, sparse=False)
            uplink_bytes    = min(_sparse_primary, _dense_primary) + _bn_bytes

        # ── Downlink bytes ────────────────────────────────────────────────
        # With partial_downlink: download the M-1 secondary partitions + BN stats.
        # Without: full model.
        if not is_warmup and partial_downlink:
            secondary_set = set(secondary_names)
            downlink_bytes = sum(
                w_before[k].numel() * 4
                for k in w_before
                if k in secondary_set or k.endswith(_bn_suffixes)
            )
        else:
            downlink_bytes = self.count_bytes(w_before, sparse=False)

        # ── Energy accounting ──────────────────────────────────────────────
        profile: DeviceProfile = config.get("device_profile")
        num_params = sum(p.numel() for p in model.parameters())
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        if profile is not None:
            # Trainable set declaration drives the dispatcher: phi mode replays
            # the (d + 2|primary|)*2*B*S formula (via num_primary_params hint),
            # measured/corrected use the real backward through `primary_names`.
            if freeze_secondary and not is_warmup:
                primary_name_set = set(primary_names)
                num_primary_params = sum(
                    p.numel() for n, p in model.named_parameters()
                    if n in primary_name_set
                )
                trainable_names = list(primary_name_set)
                _np_hint = num_primary_params
            else:
                num_primary_params = num_params
                trainable_names = [n for n, _ in model.named_parameters()]
                _np_hint = None  # full path => phi falls back to full_flops
            flops = round_compute_flops(
                model, trainable_names, config,
                profile, dataloader, local_epochs,
                num_primary_params=_np_hint,
            )
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            num_primary_params = num_params
            energy_j = 0.5 + 1.0 * beta

        del w_before
        del current_sd

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)

        del optimizer

        # ── Compression ratio ──────────────────────────────────────────────
        full_model_bytes = self.count_bytes(delta, sparse=False)
        compression_ratio = uplink_bytes / max(full_model_bytes, 1)
        flops_ratio = num_primary_params / max(num_params, 1)

        metadata = {
            "client_id": state.client_id,
            "round_num": round_num,
            "beta_actual": beta,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": compression_ratio,
            "dataset_size": dataset_size,
            "cluster": cluster,
            "cluster_idx": cluster_idx,
            "primary_group": f"g{primary_idx}" if primary_idx is not None else "full",
            "is_warmup": is_warmup,
            "freeze_secondary": freeze_secondary and not is_warmup,
            "flops_ratio": flops_ratio,
            "num_partitions": M,
            "priming": is_priming and not is_warmup,
        }

        # Advance local round counter (only when server doesn't inject round_num)
        if "round_num" not in config:
            state.round_num = round_num + 1

        return dict(update), metadata

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Separate aggregation per partition, per responsible cluster.

        At phase p, partition g receives updates from cluster (g - p) % M.
        """
        global_sd = global_model.state_dict()
        warmup_rounds = config.get("ccsEF_warmup_rounds", config.get("warmup_rounds", 5))
        T_rot = config.get("T_rot", 1)
        M = config.get("num_partitions", 2)

        # ── Update fleet battery thresholds for dynamic clustering ────────
        # M-1 thresholds split the alive fleet into M equal-quantile clusters.
        alive_batteries = sorted(
            s.battery_j for _, _, s in client_updates if s.battery_j > 0
        )
        if alive_batteries and M > 1:
            N = len(alive_batteries)
            thresholds = [
                alive_batteries[max(0, min(int(N * i / M), N - 1))]
                for i in range(1, M)
            ]
            config["fleet_battery_thresholds"] = thresholds

        # ── Separate alive from dead clients ──────────────────────────────
        active_updates = [
            (u, m, s) for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]
        K_alive = len(active_updates)

        # ── Warmup: full FedAvg aggregation ───────────────────────────────
        if round_num < warmup_rounds:
            if K_alive == 0:
                metrics = self._compute_metrics(
                    client_updates, active_updates, round_num, config,
                    phase="warmup", partition_norms=[0.0] * M,
                    cluster_alive_counts={i: 0 for i in range(M)},
                )
                return AggregateResult(new_weights=global_sd, metrics=metrics)

            agg = None
            for update, metadata, state in active_updates:
                if agg is None:
                    agg = {k: v.clone().float() for k, v in update.items()}
                else:
                    for k in update:
                        if k in agg:
                            agg[k] += update[k].float()
                        else:
                            agg[k] = update[k].float()
            for k in agg:
                agg[k] /= K_alive

            new_weights = OrderedDict()
            for k in global_sd:
                if agg is not None and k in agg:
                    new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
                else:
                    new_weights[k] = global_sd[k].float()

            del agg

            metrics = self._compute_metrics(
                client_updates, active_updates, round_num, config,
                phase="warmup", partition_norms=[0.0] * M,
                cluster_alive_counts={i: 0 for i in range(M)},
            )
            return AggregateResult(new_weights=new_weights, metrics=metrics)

        # ── CCS-EF phase: cluster-specific per-partition aggregation ──────
        if len(active_updates) == 0:
            phase = ((round_num - warmup_rounds) // T_rot) % M
            metrics = self._compute_metrics(
                client_updates, active_updates, round_num, config,
                phase=phase, partition_norms=[0.0] * M,
                cluster_alive_counts={i: 0 for i in range(M)},
            )
            return AggregateResult(new_weights=global_sd, metrics=metrics)

        # Cache layer groups + M partitions (architecture is fixed)
        if not hasattr(self, "_server_state"):
            self._server_state = {}
        if "layer_groups" not in self._server_state:
            self._server_state["layer_groups"] = _derive_layer_groups(global_model)

        layer_groups = self._server_state["layer_groups"]
        num_layer_groups = len(layer_groups)
        M = min(M, max(num_layer_groups, 1))

        # Cache partition name sets (rebuild if M changes between runs)
        cached_M = self._server_state.get("cached_M", None)
        if cached_M != M:
            _, partition_name_sets = _make_partitions(layer_groups, M)
            self._server_state["partition_name_sets"] = partition_name_sets
            self._server_state["cached_M"] = M
        partition_name_sets = self._server_state["partition_name_sets"]

        # Phase and cluster buckets
        phase = ((round_num - warmup_rounds) // T_rot) % M
        cluster_buckets = {
            i: [(u, m, s) for u, m, s in active_updates if m.get("cluster_idx", -1) == i]
            for i in range(M)
        }

        # For partition g, responsible cluster = (g - phase) % M
        # (inverse of: cluster c → primary (c + phase) % M)
        agg_partitions = []
        for g_idx in range(M):
            responsible_cluster = (g_idx - phase) % M
            responsible_updates = cluster_buckets.get(responsible_cluster, [])
            agg = self._aggregate_group(responsible_updates, partition_name_sets[g_idx])
            agg_partitions.append(agg)

        # ── Aggregate BN running stats from ALL active clients ────────────
        agg_bn = self._aggregate_bn_buffers(active_updates)

        # ── Server momentum ───────────────────────────────────────────────
        server_momentum = config.get("server_momentum", 0.0)
        if server_momentum > 0.0:
            if "momentum_buffer" not in self._server_state:
                self._server_state["momentum_buffer"] = {}
            m_buf = self._server_state["momentum_buffer"]
            for agg in agg_partitions:
                for k, v in list(agg.items()):
                    prev = m_buf.get(k, torch.zeros_like(v))
                    m_buf[k] = server_momentum * prev + (1.0 - server_momentum) * v
                    agg[k] = m_buf[k]

        # ── Apply aggregated deltas to global model ───────────────────────
        new_weights = OrderedDict()
        for k in global_sd:
            if k in agg_bn:
                new_weights[k] = global_sd[k].float() - agg_bn[k].to(global_sd[k].device)
            else:
                applied = False
                for agg in agg_partitions:
                    if k in agg:
                        new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
                        applied = True
                        break
                if not applied:
                    new_weights[k] = global_sd[k].float()

        # ── Gradient norms and cluster counts ─────────────────────────────
        partition_norms = [
            sum(v.norm().item() ** 2 for v in agg.values()) ** 0.5 if agg else 0.0
            for agg in agg_partitions
        ]
        cluster_alive_counts = {i: len(cluster_buckets.get(i, [])) for i in range(M)}

        del agg_partitions

        metrics = self._compute_metrics(
            client_updates, active_updates, round_num, config,
            phase=phase, partition_norms=partition_norms,
            cluster_alive_counts=cluster_alive_counts,
        )

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    def _aggregate_group(
        self,
        cluster_updates: list[tuple[dict, dict, ClientState]],
        group_names: set[str],
    ) -> dict[str, torch.Tensor]:
        """Dataset-size-weighted aggregation for a single partition from a cluster."""
        if len(cluster_updates) == 0:
            return {}

        sizes = [m.get("dataset_size", 1) for _, m, _ in cluster_updates]
        total_n = max(sum(sizes), 1)
        weights = [n_k / total_n for n_k in sizes]

        agg = None
        for (update, metadata, state), w_k in zip(cluster_updates, weights):
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
        Dataset-size-weighted aggregation of BN running statistics from all clients.
        BN buffers are excluded from partition_names and must be aggregated globally.
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
        partition_norms: list[float] | None = None,
        cluster_alive_counts: dict[int, int] | None = None,
        # Legacy M=2 keyword args kept for internal warmup calls
        g1_norm: float = 0.0,
        g2_norm: float = 0.0,
        C1_alive: int = 0,
        C2_alive: int = 0,
    ) -> dict:
        """Compute round metrics. Generalised for M partitions."""
        M = config.get("num_partitions", 2)
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
            avg_battery = batt_min = avg_beta = avg_loss = avg_cr = 0.0

        bytes_sent = [m["bytes_sent"] for _, m, _ in all_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b ** 2 for b in bytes_sent))
        else:
            jain = 1.0

        _pnorms = partition_norms or [g1_norm, g2_norm]
        _calive = cluster_alive_counts or {0: C1_alive, 1: C2_alive}

        metrics = {
            "round_num": round_num,
            "K_alive": K_alive,
            "K_dead": K_dead,
            "K_skipped": K - K_alive,
            "participation_rate": K_alive / max(K, 1),
            "total_bytes_sent": total_bytes,
            "total_energy_j": total_energy,
            "avg_battery_j": avg_battery,
            "batt_min_j": batt_min,
            "jain_index": jain,
            "avg_beta": avg_beta,
            "avg_local_loss": avg_loss,
            "compression_ratio": avg_cr,
            "phase": phase,
            "partition_norms": _pnorms,
            "cluster_alive_counts": _calive,
            # Legacy M=2 keys for backward compatibility with analysis scripts
            "g1_norm": _pnorms[0] if len(_pnorms) > 0 else 0.0,
            "g2_norm": _pnorms[1] if len(_pnorms) > 1 else 0.0,
            "C1_alive": _calive.get(0, 0),
            "C2_alive": _calive.get(1, 0),
            "num_partitions": M,
        }
        return metrics

    def get_default_config(self) -> dict:
        return {
            # Training
            "optimizer": "sgd",   # "sgd" | "adam"
            "lr": 0.01,
            "lr_min": None,
            "num_rounds": 200,
            "momentum": 0.9,      # SGD only (ignored when optimizer=adam)
            "weight_decay": 1e-4,
            "max_grad_norm": 10.0,
            "local_epochs": 8,
            "batch_size": 32,
            # CCS-EF specific
            "num_partitions": 2,      # M: number of model partitions AND client clusters
            "beta_min": 0.01,
            "beta_max": 1.0,
            "battery_max_j": 13320.0,
            "energy_scale_factor": 12.6,
            "ccsEF_warmup_rounds": 5,
            "T_rot": 1,               # rotation period (each cluster shifts primary partition)
            "adaptive_rotation": False,
            "fleet_battery_thresholds": None,  # M-1 thresholds updated by server_aggregate
            "T_recluster": 10,
            # Compute-energy trade-off
            "freeze_secondary_backward": False,
            # Buffer age reset (Option 4): reset primary EF buffer when age > M*T_rot
            # (i.e. client changed cluster and buffer missed ≥1 update cycle).
            # Set False to disable and observe the staleness effect (ablation).
            "buffer_age_reset": True,
            # Option 5: priming round — run SGD + EF on cluster change but transmit
            # nothing. Primes the new primary buffer so first real transmission is clean.
            "buffer_priming": True,
            # EF buffer norm clipping: clips u = Δg + e before TopK.
            # Prevents buffer explosion when β is small (1-β > 0.5) and gradients
            # shrink late in training. None = disabled. Recommended: 1.0–5.0.
            "ef_clip_norm": None,
            # Communication optimizations
            "use_fp16_uplink": True,
            "partial_downlink": True,
            # Server-side optimization
            "server_momentum": 0.0,
            # Device
            "device": "cpu",
            "device_profile": None,
        }
