#!/usr/bin/env python3
"""
run_experiment.py
=================
Standalone FL experiment runner — NO ZMQ, NO subprocesses.

Simulates the full FL training loop in a single process:
  - All clients run sequentially (simulating distributed execution)
  - All algorithms in the registry are supported
  - Full energy tracking, battery simulation, Jain fairness
  - Results saved to JSON (same format as ZMQ framework)

This is the fastest way to run and compare algorithms.
Use the ZMQ framework when you need true distributed execution.

Usage:
  # Run E-CEFFL vs all baselines (main experiment for the paper):
  python run_experiment.py --config configs/comparison_eceffl_vs_baselines.yaml

  # Quick test run:
  python run_experiment.py --algo eceffl --rounds 10 --clients 5

  # Run a single algorithm:
  python run_experiment.py --algo fedprox --rounds 100 --dataset cifar10

  # Run all algorithms sequentially and save results:
  python run_experiment.py --benchmark --rounds 100 --output results/benchmark

Examples with all options:
  python run_experiment.py \\
      --algo eceffl \\
      --dataset cifar10 \\
      --model resnet18 \\
      --rounds 100 \\
      --clients 10 \\
      --alpha 0.5 \\
      --partition dirichlet \\
      --lr 0.01 \\
      --epochs 1 \\
      --batch-size 32 \\
      --beta-min 0.01 \\
      --device cpu \\
      --output results/eceffl_run1 \\
      --seed 42
"""

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import algorithms.fedavg           # noqa — trigger @register_algorithm
# import algorithms.eceffl           # noqa
# import algorithms.leanfed          # noqa
# import algorithms.fedbacys         # noqa
# import algorithms.vaishnav         # noqa
# import algorithms.fedsparq         # noqa
# import algorithms.fedprox          # noqa
# import algorithms.scaffold         # noqa
import algorithms.fedpart               # noqa
import algorithms.fedpart_be            # noqa
# import algorithms.fedpart_universal     # noqa
# import algorithms.heterofl              # noqa
# import algorithms.fjord                 # noqa
import algorithms.fed_resonance         # noqa
# import algorithms.fed_grad_align        # noqa
# import algorithms.fed_resonance_osmosis # noqa
# import algorithms.fed_resonance_plus    # noqa
import algorithms.server_mask_fl        # noqa
# import algorithms.fedmask               # noqa
# import algorithms.hermes                # noqa

from algorithms.base import get_algorithm, ClientState, list_algorithms
from models.registry import get_model
from datasets.registry import get_dataloader
from hardware.profiles import make_fleet
from diagnostics.layer_mismatch import LayerMismatchDiagnostic


# ─────────────────────────────────────────────────────────────────────────────
# Default fleet: 5 RPi4 + 3 smartphones + 2 Jetson (10 clients total)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FLEET = [
    ("raspberry_pi_4",      5),
    ("smartphone_midrange", 3),
    ("jetson_nano",         2),
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_global_model(model, test_loader, device: str) -> tuple[float, float]:
    """Evaluate accuracy and loss on test set."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, tloss = 0, 0, 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            tloss   += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total   += y.size(0)

    model.train()
    return correct / total, tloss / total


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def _select_clients(
    num_clients: int,
    sample_fraction: float,
    min_clients: int,
    strategy: str,
    client_states: list,
    round_num: int,
    rng: np.random.Generator,
) -> list[int]:
    """Return sorted list of client IDs selected for this round."""
    k = max(min_clients, int(np.ceil(sample_fraction * num_clients)))
    k = min(k, num_clients)

    if strategy == "battery_aware":
        # Prefer clients with most remaining battery
        sorted_ids = sorted(range(num_clients),
                            key=lambda i: client_states[i].battery_j,
                            reverse=True)
        return sorted(sorted_ids[:k])
    elif strategy == "round_robin":
        # Deterministic rotation: shift the window each round
        start = (round_num * k) % num_clients
        ids = [(start + i) % num_clients for i in range(k)]
        return sorted(ids)
    else:  # "random" (default)
        return sorted(rng.choice(num_clients, size=k, replace=False).tolist())


def run_single_experiment(
    algo_name: str,
    algo_config: dict,
    dataset: str,
    model_name: str,
    num_rounds: int,
    num_clients: int,
    alpha: float,
    partition: str,
    batch_size: int,
    fleet_spec: list,
    battery_noise_std: float,
    device: str,
    seed: int,
    data_root: str,
    verbose: bool = True,
    layer_mismatch: bool = False,
    layer_mismatch_layers: list = None,
    layer_mismatch_freq: int = 1,
    layer_mismatch_metrics: list = None,
    layer_mismatch_max_batches: int = 2,
    layer_mismatch_max_clients_cka: int = 8,
    sample_fraction: float = 1.0,
    sampling_strategy: str = "random",
    min_clients: int = 1,
    battery_dist: str = "gaussian",
    battery_params: Optional[dict] = None,
) -> dict:
    """
    Run a single FL experiment in-process (no ZMQ).

    Returns:
        dict with keys: config, rounds (list of per-round metrics), summary
    """
    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Algorithm ────────────────────────────────────────────────────────────
    algo = get_algorithm(algo_name)
    default_cfg = algo.get_default_config()
    merged_config = {**default_cfg, **algo_config, "device": device}

    # ── Global model ─────────────────────────────────────────────────────────
    global_model = get_model(model_name, dataset)
    global_model.to(device)

    # Single reusable client model — weights reset via load_state_dict() before
    # each client call. Avoids num_clients × num_rounds get_model() calls
    # (each does Kaiming init over all Conv2d/BN layers = ~5-10 ms each).
    _client_model = get_model(model_name, dataset)
    _client_model.to(device)

    # ── Test loader ──────────────────────────────────────────────────────────
    test_loader = get_dataloader(
        dataset_name=dataset, split="test",
        partition="iid", client_id=None,
        num_clients=1, batch_size=256, data_root=data_root,
    )

    # ── Layer mismatch diagnostic ────────────────────────────────────────────
    diagnostic = None
    _lm_freq = max(1, int(layer_mismatch_freq))
    _lm_active_metrics = set(layer_mismatch_metrics or ["drift", "loss_jump", "cka"])
    if layer_mismatch:
        try:
            diagnostic = LayerMismatchDiagnostic(
                model=global_model,
                ref_dataloader=test_loader,
                device=device,
                layer_names=layer_mismatch_layers or None,  # None = auto-detect
                max_batches_per_client=layer_mismatch_max_batches,
                max_clients_cka=layer_mismatch_max_clients_cka,
            )
            if verbose:
                print(f"  Layer mismatch diagnostic enabled.")
                print(f"    Layers       : {layer_mismatch_layers or 'auto (' + str(len(diagnostic.layer_names)) + ' layers)'}")
                print(f"    Frequency    : every {_lm_freq} round(s)")
                print(f"    Metrics      : {sorted(_lm_active_metrics)}")
                print(f"    Max batches  : {layer_mismatch_max_batches} / client (loss_jump)")
                print(f"    Max CKA clts : {layer_mismatch_max_clients_cka}")
        except Exception as e:
            print(f"Warning: Failed to initialize layer mismatch diagnostic: {e}")
            diagnostic = None

    # ── Fleet ────────────────────────────────────────────────────────────────
    import random as _random
    _random.seed(seed)
    fleet = make_fleet(
        fleet_spec,
        battery_noise_std=battery_noise_std,
        battery_dist=battery_dist,
        battery_params=battery_params,
        seed=seed,
    )
    # Pad or truncate to num_clients — cycle through base fleet to preserve heterogeneity
    base_len = len(fleet)
    while len(fleet) < num_clients:
        fleet.append(copy.deepcopy(fleet[len(fleet) % base_len]))
    fleet = fleet[:num_clients]

    # ── Client dataloaders ───────────────────────────────────────────────────
    client_loaders = []
    for cid in range(num_clients):
        loader = get_dataloader(
            dataset_name=dataset, split="train",
            partition=partition, client_id=cid,
            num_clients=num_clients, alpha=alpha,
            batch_size=batch_size, seed=seed,
            data_root=data_root,
        )
        client_loaders.append(loader)

    # ── Client states ─────────────────────────────────────────────────────────
    client_states = [
        ClientState(
            client_id=cid,
            battery_j=fleet[cid].battery.initial_energy_j,
        )
        for cid in range(num_clients)
    ]

    # ── Server-level algorithm state (e.g. SCAFFOLD c_global) ─────────────
    server_algo_state = {}

    # ── Sampling RNG (separate stream so seed is reproducible) ───────────────
    _sample_rng = np.random.default_rng(seed + 9999)
    _do_sampling = sample_fraction < 1.0

    if verbose and _do_sampling:
        k_expected = max(min_clients, int(np.ceil(sample_fraction * num_clients)))
        print(f"  Client sampling: {sampling_strategy} | "
              f"fraction={sample_fraction} → ~{k_expected}/{num_clients} per round")

    # ── Training loop ─────────────────────────────────────────────────────────
    rounds_log = []
    cumulative_bytes = 0.0
    cumulative_energy = 0.0
    # Simulated wall-clock time (parallel client assumption):
    # each round contributes max(client_train_times) rather than their sum,
    # mirroring real distributed execution where all clients train in parallel.
    # This produces a meaningful "time-to-accuracy" axis for paper figures.
    cumulative_sim_time_s = 0.0

    if verbose:
        print(f"\n{'='*64}")
        print(f"  {algo_name.upper()} | {dataset} | {model_name}")
        print(f"  {num_clients} clients | {num_rounds} rounds | α={alpha} | "
              f"partition={partition}")
        print(f"{'='*64}")

        # ── Client battery summary ─────────────────────────────────────────
        batteries = [cs.battery_j for cs in client_states]
        b_min, b_max, b_mean = min(batteries), max(batteries), sum(batteries) / len(batteries)
        full_cap = fleet[0].battery.capacity_j if fleet else b_max
        print(f"\n  Fleet: {num_clients} × {fleet[0].name if fleet else 'unknown'}")
        print(f"  Battery  min={b_min:.0f}J ({100*b_min/full_cap:.1f}% SOC)"
              f"  max={b_max:.0f}J ({100*b_max/full_cap:.1f}% SOC)"
              f"  mean={b_mean:.0f}J ({100*b_mean/full_cap:.1f}% SOC)")
        print(f"\n  {'ID':>3}  {'Battery (J)':>12}  {'SOC %':>6}  {'Bar':}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*20}")
        for cs in sorted(client_states, key=lambda c: c.battery_j):
            soc = 100.0 * cs.battery_j / full_cap if full_cap > 0 else 0.0
            bar_len = int(soc / 5)          # 1 char per 5% SOC → max 20 chars
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {cs.client_id:>3}  {cs.battery_j:>12.1f}  {soc:>5.1f}%  {bar}")
        print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*20}")
        print()

    for t in range(num_rounds):
        t0 = time.time()

        # ── Client updates ─────────────────────────────────────────────────
        # Use a shallow dict copy with cloned tensors (avoids keeping the full
        # deepcopy of the model object alive across the entire client loop).
        global_sd_snapshot = {k: v.clone() for k, v in global_model.state_dict().items()}

        # Save model before aggregation for diagnostic
        model_before_agg = None
        if diagnostic is not None:
            model_before_agg = get_model(model_name, dataset)
            model_before_agg.load_state_dict(
                {k: v.clone() for k, v in global_sd_snapshot.items()}
            )
            model_before_agg.to(device)

        # Build per-round algo config (include server state for SCAFFOLD etc.)
        round_config = {**merged_config, **server_algo_state}

        # ── Client selection ───────────────────────────────────────────────
        if _do_sampling:
            selected_ids = _select_clients(
                num_clients=num_clients,
                sample_fraction=sample_fraction,
                min_clients=min_clients,
                strategy=sampling_strategy,
                client_states=client_states,
                round_num=t,
                rng=_sample_rng,
            )
        else:
            selected_ids = list(range(num_clients))

        client_tuples = []
        client_models_before_agg = []   # for diagnostic
        participated_loaders = []       # loaders matching client_models_before_agg
        client_train_times = []         # per-client training durations (for sim time)
        agg_result = None

        # ── run_round() hook: algorithms with 2-phase / custom round logic ───
        # Algorithms implementing run_round() (e.g. FedPartBE-LoRA-GS with ASSC)
        # handle client training + aggregation internally.  The hook provides
        # alive_clients as a list of (cid, loader, state, profile) tuples and
        # receives back (AggregateResult, client_tuples, train_times).
        if hasattr(algo, "run_round"):
            alive_clients_info = [
                (cid, client_loaders[cid], client_states[cid], fleet[cid])
                for cid in selected_ids
                if client_states[cid].battery_j > 0.0
            ]
            if alive_clients_info:
                _client_model.load_state_dict(global_sd_snapshot)
                _rr_agg, client_tuples, client_train_times = algo.run_round(
                    global_model=global_model,
                    alive_clients=alive_clients_info,
                    client_model=_client_model,
                    config=round_config,
                    round_num=t,
                    device=device,
                )
                if client_tuples and _rr_agg.new_weights:
                    global_model.load_state_dict(
                        {k: v.to(device) for k, v in _rr_agg.new_weights.items()}
                    )
                    del _rr_agg.new_weights
                agg_result = _rr_agg
            del global_sd_snapshot

        else:
            # ── Normal client update loop ──────────────────────────────────
            for cid in selected_ids:
                # Permanent dropout: skip clients with no remaining battery
                if client_states[cid].battery_j <= 0.0:
                    continue

                # Reset reusable client model to current global weights.
                _client_model.load_state_dict(global_sd_snapshot)

                client_config = {**round_config, "device_profile": fleet[cid]}
                _battery_before = client_states[cid].battery_j

                _t_client = time.time()
                update, metadata = algo.client_update(
                    model=_client_model,
                    dataloader=client_loaders[cid],
                    state=client_states[cid],
                    config=client_config,
                )
                client_train_times.append(time.time() - _t_client)
                client_tuples.append((update, metadata, client_states[cid]))

                if _battery_before > 0.0 and client_states[cid].battery_j <= 0.0:
                    profile_name = getattr(fleet[cid], "name", f"device_{cid}")
                    alive_after = sum(1 for cs in client_states if cs.battery_j > 0)
                    # Record the death round for survival metrics — picked up by
                    # metrics.survival.derive_lifetimes().
                    client_states[cid].custom["death_round"] = t
                    print(f"  ☠  Round {t+1:3d} | Client {cid:2d} [{profile_name}] "
                          f"battery depleted — {alive_after}/{num_clients} clients alive")

                if diagnostic is not None:
                    client_model_copy = get_model(model_name, dataset)
                    client_model_copy.load_state_dict(
                        {k: v.clone() for k, v in _client_model.state_dict().items()}
                    )
                    client_models_before_agg.append(client_model_copy)
                    participated_loaders.append(client_loaders[cid])

                del client_config

            del global_sd_snapshot

        # ── Skip round if all clients have dropped out ─────────────────────
        if not client_tuples:
            if verbose:
                print(f"  Round {t+1:3d}: ALL clients dropped out — skipping aggregation.")
            rounds_log.append({
                "round_num":              t + 1,
                "test_accuracy":          rounds_log[-1]["test_accuracy"] if rounds_log else 0.0,
                "test_loss":              rounds_log[-1]["test_loss"] if rounds_log else 0.0,
                "train_loss":             0.0,
                "total_bytes":            0,
                "total_energy_j":         0.0,
                "cumulative_bytes":       cumulative_bytes,
                "cumulative_energy_j":    cumulative_energy,
                "avg_battery_j":          0.0,
                "avg_beta":               1.0,
                "jain_index":             0.0,
                "participation_rate":     0.0,
                "num_selected":           0,
                "num_clients":            num_clients,
                "elapsed_s":              time.time() - t0,
                "sim_round_time_s":       0.0,
                "cumulative_sim_time_s":  cumulative_sim_time_s,
                "num_alive_clients":      0,
                "survival_ratio":         0.0,
            })
            if model_before_agg is not None:
                del model_before_agg
            del client_models_before_agg
            gc.collect()
            # Batteries cannot recharge — all future rounds will also be empty.
            if all(cs.battery_j <= 0.0 for cs in client_states):
                if verbose:
                    print(f"  Fleet fully depleted at round {t+1}. Stopping early.")
                break
            continue

        # ── Server aggregation (skipped when run_round() already did it) ───
        if agg_result is None:
            agg_result = algo.server_aggregate(
                global_model=global_model,
                client_updates=client_tuples,
                round_num=t,
                config=round_config,
            )
            global_model.load_state_dict(
                {k: v.to(device) for k, v in agg_result.new_weights.items()}
            )
            del agg_result.new_weights

        # Save participant count before freeing client_tuples (used in metrics).
        _num_participated = len(client_tuples)
        # Free client updates: update dicts (partial deltas) are large and no
        # longer needed after server_aggregate has consumed them.
        # No gc.collect() here: Python ref-counting frees the dict of tensors
        # immediately (no circular refs). Calling gc.collect() every round adds
        # ~50-150 ms/round on a loaded heap (Python scans all live objects).
        del client_tuples

        # ── Layer mismatch diagnostic ──────────────────────────────────────
        # mismatch_score=None means "not computed this round" (freq skip).
        mismatch_score = None
        _run_diagnostic = (diagnostic is not None) and (t % _lm_freq == 0)
        if _run_diagnostic:
            try:
                criterion = torch.nn.CrossEntropyLoss()
                diagnostic_results = {}
                if "drift" in _lm_active_metrics and model_before_agg is not None:
                    diagnostic_results["representation_drift"] = diagnostic.representation_drift(
                        model_before_agg, global_model, diagnostic.ref_batch
                    )
                if "loss_jump" in _lm_active_metrics and client_models_before_agg:
                    diagnostic_results["loss_jump"] = diagnostic.loss_jump(
                        global_model, client_models_before_agg,
                        participated_loaders, criterion
                    )
                if "cka" in _lm_active_metrics and participated_loaders:
                    diagnostic_results["cka"] = diagnostic.cka_between_clients(
                        global_model, participated_loaders
                    )
                mismatch_score = diagnostic.summary_score(diagnostic_results)
            except Exception as e:
                if verbose:
                    print(f"Warning: Layer mismatch computation failed at round {t+1}: {e}")
                mismatch_score = None
            finally:
                if model_before_agg is not None:
                    del model_before_agg
                del client_models_before_agg
        elif diagnostic is not None:
            # Diagnostic active but freq skip — free memory, skip compute.
            if model_before_agg is not None:
                del model_before_agg
            del client_models_before_agg
        else:
            # Diagnostic disabled — lists were still allocated, free them.
            del client_models_before_agg

        # Update server-level state (SCAFFOLD c_global, Fed-Osmosis global stats, etc.)
        osmosis_stats = agg_result.metrics.pop("osmosis_global_stats", None)
        if osmosis_stats:
            # Propagate aggregated activation stats to next round's client configs
            server_algo_state["osmosis_global_stats"] = osmosis_stats

        c_global_update = agg_result.metrics.pop("_scaffold_c_global_update", None)
        if c_global_update:
            server_algo_state["scaffold_c_global"] = c_global_update

        # ── Evaluate ───────────────────────────────────────────────────────
        acc, loss = evaluate_global_model(global_model, test_loader, device)

        # ── Metrics ────────────────────────────────────────────────────────
        agg_m = agg_result.metrics
        total_bytes  = agg_m.get("total_bytes_sent", 0)
        total_energy = agg_m.get("total_energy_j", 0.0)
        cumulative_bytes  += total_bytes
        cumulative_energy += total_energy
        elapsed = time.time() - t0

        # Simulated parallel round time = max client training time.
        # In real distributed FL all clients train concurrently; the round
        # is gated by the slowest participating client, not their sum.
        sim_round_time = max(client_train_times) if client_train_times else 0.0
        cumulative_sim_time_s += sim_round_time

        # ── Jain fairness index — computed over ALL clients (alive + dead) ──
        # server_aggregate only receives alive clients → its internal Jain
        # is always 1.0 (all participants are alive by definition). The true
        # fairness metric must account for dead clients (x_i=0 if battery=0).
        #   J = (Σ x_i)² / (N × Σ x_i²)   where x_i ∈ {0,1}
        # With binary participation this simplifies to: J = alive / N.
        _alive = sum(1 for cs in client_states if cs.battery_j > 0)
        _jain_all = _alive / num_clients if num_clients > 0 else 0.0

        round_metrics = {
            "round_num":          t + 1,
            "test_accuracy":      acc,
            "test_loss":          loss,
            "train_loss":         agg_m.get("avg_local_loss", 0.0),
            "total_bytes":        total_bytes,
            "total_energy_j":     total_energy,
            "cumulative_bytes":   cumulative_bytes,
            "cumulative_energy_j": cumulative_energy,
            "avg_battery_j":      agg_m.get("avg_battery_j", 0.0),
            "avg_beta":           agg_m.get("avg_beta", 1.0),
            "jain_index":         _jain_all,
            "participation_rate": _num_participated / num_clients,
            "num_selected":       len(selected_ids),
            "num_clients":        num_clients,
            "elapsed_s":              elapsed,
            # ── Simulated distributed time (parallel client assumption) ───
            # sim_round_time_s  = max(client training times) this round
            # cumulative_sim_time_s = Σ sim_round_time over all rounds so far
            # Use these for "time-to-accuracy" figures in the paper — they
            # correctly reflect that FedPart_BE covers more groups per round,
            # reducing the number of rounds (and thus total simulated time)
            # needed to complete one full model update cycle.
            "sim_round_time_s":       sim_round_time,
            "cumulative_sim_time_s":  cumulative_sim_time_s,
            # ── Survival tracking (for FedPartBE paper) ───────────────────
            "num_alive_clients":  sum(1 for cs in client_states if cs.battery_j > 0),
            "survival_ratio":     sum(1 for cs in client_states if cs.battery_j > 0) / num_clients,
            "num_partitions":     agg_m.get("num_partitions", algo_config.get("num_partitions", 1)),
        }
        if diagnostic is not None:
            # None when freq > 1 and this round was skipped.
            round_metrics["layer_mismatch"] = mismatch_score
        rounds_log.append(round_metrics)

        # Periodic GC: Python's allocator does not return freed memory to the OS
        # automatically; on macOS this causes swap growth over long runs.
        if (t + 1) % 10 == 0:
            gc.collect()

        if verbose and ((t + 1) % 5 == 0 or t == 0 or t == num_rounds - 1):
            batteries = [cs.battery_j for cs in client_states]
            output_str = (f"  Round {t+1:3d}/{num_rounds} | "
                         f"Acc={acc*100:5.2f}% | Loss={loss:.4f} | "
                         f"Bytes={total_bytes/1e6:5.1f}MB | E={total_energy:6.1f}J | "
                         f"J={_jain_all:.3f}")
            if diagnostic is not None:
                lm_str = f"{mismatch_score:.3f}" if mismatch_score is not None else "—"
                output_str += f" | LM={lm_str}"
            output_str += f" | Batt_min={min(batteries):.0f}J | ⏱{elapsed:.1f}s"
            print(output_str)

    # ── Summary ────────────────────────────────────────────────────────────
    best_acc = max(r["test_accuracy"] for r in rounds_log)
    final = rounds_log[-1]
    summary = {
        "algorithm":          algo_name,
        "dataset":            dataset,
        "model":              model_name,
        "partition":          partition,
        "alpha":              alpha,
        "num_rounds":         num_rounds,
        "num_clients":        num_clients,
        "sample_fraction":    sample_fraction,
        "sampling_strategy":  sampling_strategy,
        "seed":               seed,
        "best_accuracy":      best_acc,
        "final_accuracy":     final["test_accuracy"],
        "final_test_loss":    final["test_loss"],
        "total_bytes_gb":     final["cumulative_bytes"] / 1e9,
        "total_energy_j":     final["cumulative_energy_j"],
        "final_jain_index":   final["jain_index"],
        "final_avg_battery_j": final["avg_battery_j"],
        "final_survival_ratio": final["survival_ratio"],
        # Round where survival drops to ≤50% (system lifetime proxy)
        "system_lifetime":    next(
            (r["round_num"] for r in rounds_log if r["survival_ratio"] <= 0.5),
            num_rounds
        ),
        # Accuracy when 50% clients have dropped out
        "accuracy_at_half_dropout": next(
            (r["test_accuracy"] for r in rounds_log if r["survival_ratio"] <= 0.5),
            final["test_accuracy"]
        ),
        # ── Simulated distributed timing ──────────────────────────────────
        # total_sim_time_s   : Σ max(client_times) over all rounds.
        #                      Models real distributed wall-clock time where
        #                      clients train in parallel.
        # rounds_to_best_acc : round index at which best accuracy was reached.
        # sim_time_to_best_acc: simulated time (s) to reach best accuracy —
        #                      primary "convergence speed" metric for paper.
        "total_sim_time_s":      final["cumulative_sim_time_s"],
        "rounds_to_best_acc":    next(
            r["round_num"] for r in rounds_log if r["test_accuracy"] == best_acc
        ),
        "sim_time_to_best_acc":  next(
            r["cumulative_sim_time_s"] for r in rounds_log
            if r["test_accuracy"] == best_acc
        ),
        "algo_config":        merged_config,
    }

    if verbose:
        print(f"\n  Summary: Best Acc={best_acc*100:.2f}% | "
              f"Total bytes={summary['total_bytes_gb']:.3f}GB | "
              f"Total energy={summary['total_energy_j']:.0f}J | "
              f"Final Jain={summary['final_jain_index']:.4f} | "
              f"SimTime={summary['total_sim_time_s']:.0f}s")

    # Compact, JSON-safe snapshot of final client states — consumed by the
    # survival metrics module post-run. Keeps the tensor-bearing ClientState
    # objects out of the returned (and serialized) result.
    _client_snapshots = [
        {
            "client_id":     cs.client_id,
            "battery_j":     float(cs.battery_j),
            "death_round":   int(cs.custom.get("death_round"))
                              if isinstance(cs.custom, dict) and "death_round" in cs.custom
                              else None,
        }
        for cs in client_states
    ]

    return {
        "algorithm": algo_name,
        "dataset":   dataset,
        "config": merged_config,
        "rounds": rounds_log,
        "summary": summary,
        "client_states": _client_snapshots,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: run all algorithms and compare
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    """Run the full comparison suite: Level-1 + Level-2 baselines + proposed algorithms."""
    # ── Level 1: unconstrained baselines (lower-bound anchors) ───────────────
    _lvl1 = [
        ("fedavg",   {}),
        ("fedprox",  {"mu": 0.01}),
        ("scaffold", {"global_lr": 1.0}),
    ]
    # ── Level 2: resource-constrained baselines (direct competitors) ─────────
    _lvl2 = [
        ("leanfed",    {"rho_min": 0.1}),
        ("fedbacys",   {"t_active_init": 0.5, "threshold_decay": 0.995,
                        "recharge_rate_j": 500.0}),
        ("vaishnav",   {"beta_min": 0.01, "bw_ref_mbps": 50.0}),
        ("fedsparq",   {"ema_alpha": 0.1}),
        ("heterofl",   {}),
        ("fjord",      {}),
        ("fedpart",    {"rounds_per_layer": 2, "training_cycles": 5,
                        "warmup_rounds": 5}),
        ("fedpart_be", {"num_tiers": 3, "mu_repr": 0.1, "warmup_rounds": 5}),
        ("fedmask",    {"beta_min": 0.1, "beta_max": 0.5, "warmup_rounds": 5}),
        ("hermes",     {"beta_min": 0.1, "beta_max": 0.5, "warmup_rounds": 5}),
    ]
    # ── Proposed algorithms ───────────────────────────────────────────────────
    _proposed = [
        ("eceffl",              {"beta_min": args.beta_min, "beta_max": 1.0}),
        ("fed_resonance",       {"resonance_rank": 16, "use_rsvd": True}),
        ("fed_grad_align",      {"resonance_rank": 20, "basis_update_freq": 5}),
        ("fed_resonance_osmosis", {}),
        ("fed_resonance_plus",  {"resonance_rank": 16, "use_rsvd": True,
                                  "basis_update_freq": 5}),
        ("server_mask",         {"beta_min": 0.1, "beta_max": 0.5, "warmup_rounds": 5,
                                  "ema_alpha": 0.3, "epsilon_explore": 0.05,
                                  "server_lr": 0.5, "mu_weight": 0.0}),
    ]

    _all = _lvl1 + _lvl2 + _proposed
    _level_map = {n: "L1" for n, _ in _lvl1}
    _level_map.update({n: "L2" for n, _ in _lvl2})
    _level_map.update({n: "Proposed" for n, _ in _proposed})

    # ── Optional filter: --algos fedavg,eceffl,fed_resonance ─────────────────
    filter_set = None
    if getattr(args, "algos", None):
        filter_set = {a.strip() for a in args.algos.split(",")}
        print(f"\n  [filter] Running only: {sorted(filter_set)}")

    algorithms_to_run = [
        (n, c) for n, c in _all
        if filter_set is None or n in filter_set
    ]
    if not algorithms_to_run:
        print(f"ERROR: --algos filter matched no algorithms. Available: "
              f"{[n for n, _ in _all]}")
        return

    all_results = {}
    for algo_name, extra_config in algorithms_to_run:
        print(f"\n{'#'*64}")
        print(f"  Running: {algo_name}")
        print(f"{'#'*64}")
        algo_config = {
            "lr": args.lr,
            "local_epochs": args.epochs if args.epochs is not None else 1,
            "batch_size": args.batch_size,
            **extra_config,
        }
        result = run_single_experiment(
            algo_name=algo_name,
            algo_config=algo_config,
            dataset=args.dataset,
            model_name=args.model,
            num_rounds=args.rounds,
            num_clients=args.clients,
            alpha=args.alpha,
            partition=args.partition,
            batch_size=args.batch_size,
            fleet_spec=DEFAULT_FLEET,
            battery_noise_std=0.15,
            device=args.device,
            seed=args.seed,
            data_root=args.data_root,
            verbose=True,
            layer_mismatch=getattr(args, "layer_mismatch", False),
            battery_dist=getattr(args, "battery_dist", "gaussian"),
            battery_params=getattr(args, "battery_params", None),
        )
        all_results[algo_name] = result

    # Save all results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for algo_name, result in all_results.items():
        algo_dir = out_dir / algo_name
        algo_dir.mkdir(exist_ok=True)
        with open(algo_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {algo_dir}/metrics.json")

    # Print comparison table
    print(f"\n{'='*105}")
    print(f"  BENCHMARK RESULTS — {args.dataset.upper()} | "
          f"Dir(α={args.alpha}) | {args.rounds} rounds")
    print(f"{'='*105}")
    print(f"  {'Lvl':<4} {'Algorithm':<18} {'Best Acc':>10} {'Final Acc':>10} "
          f"{'Total GB':>10} {'Total J':>12} {'Jain':>8}")
    print(f"  {'-'*88}")
    # ── Level 1 separator ────────────────────────────────────────────────────
    for lvl_key, lvl_label in [
        ("L1",       "Level 1: Unconstrained baselines (lower-bound anchors)"),
        ("L2",       "Level 2: Resource-constrained baselines (direct competitors)"),
        ("Proposed", "Proposed algorithms"),
    ]:
        entries = [(n, r) for n, r in all_results.items()
                   if _level_map.get(n) == lvl_key]
        if not entries:
            continue
        print(f"  {'─'*4}  {lvl_label}")
        for algo_name, result in entries:
            s = result["summary"]
            print(f"  {lvl_key:<9} {algo_name:<22} {s['best_accuracy']*100:>9.2f}% "
                  f"{s['final_accuracy']*100:>9.2f}% "
                  f"{s['total_bytes_gb']:>10.3f} "
                  f"{s['total_energy_j']:>12.0f} "
                  f"{s['final_jain_index']:>8.4f}")
    print(f"{'='*105}")

    # Save comparison summary
    comparison = {
        "benchmark_config": vars(args),
        "algorithms": {
            name: {**result["summary"], "baseline_level": _level_map.get(name, "L2")}
            for name, result in all_results.items()
        },
    }
    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved: {out_dir}/comparison_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="FedLab — Standalone FL Experiment Runner (no ZMQ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Main paper experiment: E-CEFFL vs all baselines
  python run_experiment.py --benchmark --rounds 100

  # Single algorithm test
  python run_experiment.py --algo eceffl --rounds 10 --clients 5

  # FedProx baseline
  python run_experiment.py --algo fedprox --mu 0.01 --rounds 100

  # SCAFFOLD baseline
  python run_experiment.py --algo scaffold --global-lr 1.0 --rounds 100

  # With GPU (MPS on Apple Silicon)
  python run_experiment.py --algo eceffl --device mps --rounds 100
"""
    )

    # Run mode
    p.add_argument("--benchmark", action="store_true",
                   help="Run the full comparison suite (L1 + L2 + proposed)")
    p.add_argument("--algos", type=str, default=None,
                   help="Comma-separated subset of algorithms to run in benchmark mode. "
                        "Example: --algos fedavg,fedprox,fed_resonance,eceffl "
                        "(default: all algorithms in the suite)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (overrides other args)")

    # Algorithm
    p.add_argument("--algo", type=str, default=None,
                   help="Algorithm name (fedavg, eceffl, leanfed, fedbacys, "
                        "vaishnav, fedsparq, fedprox, scaffold). "
                        "Overrides 'training.algorithm' in --config YAML.")
    p.add_argument("--list-algos", action="store_true",
                   help="List available algorithms and exit")

    # Dataset & model
    p.add_argument("--dataset",    type=str,   default="cifar10")
    p.add_argument("--model",      type=str,   default="resnet18")
    p.add_argument("--data-root",  type=str,   default="./data")

    # FL setup
    p.add_argument("--rounds",     type=int,   default=100)
    p.add_argument("--clients",    type=int,   default=10)
    p.add_argument("--alpha",      type=float, default=0.5,
                   help="Dirichlet alpha (lower = more non-IID)")
    p.add_argument("--partition",  type=str,   default="dirichlet",
                   choices=["iid", "dirichlet", "pathological"])

    # Training
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch-size", type=int,   default=32)

    # Algorithm-specific
    p.add_argument("--beta-min",   type=float, default=0.01,
                   help="[E-CEFFL/Vaishnav] min sparsification ratio")
    p.add_argument("--mu",         type=float, default=0.01,
                   help="[FedProx] proximal coefficient")
    p.add_argument("--global-lr",  type=float, default=1.0,
                   help="[SCAFFOLD] server global learning rate")
    p.add_argument("--rho-min",    type=float, default=0.1,
                   help="[LeanFed] min data subsample ratio")
    p.add_argument("--ema-alpha",  type=float, default=0.1,
                   help="[FedSparQ] EMA smoothing factor")
    p.add_argument("--freeze-secondary", action="store_true",
                   help="[ccsEF] freeze secondary-group backward (ccsEF frozen mode). "
                        "Sets freeze_secondary_backward=True in algo_config, "
                        "overriding the YAML value.")

    # Infrastructure
    p.add_argument("--device",     type=str,   default=None,
                   choices=["cpu", "cuda", "mps"],
                   help="Compute device. If omitted, reads from YAML 'device' key, "
                        "then auto-detects (MPS > CUDA > CPU).")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output",     type=str,   default=None,
                   help="Output directory. Overrides 'output_dir' in --config YAML.")
    p.add_argument("--quiet",      action="store_true")

    # Diagnostics
    p.add_argument("--layer-mismatch", action="store_true",
                   help="Enable layer mismatch diagnostic (representation drift, loss jump, CKA)")
    p.add_argument("--layer-mismatch-freq", type=int, default=1, metavar="N",
                   help="Compute layer mismatch every N rounds (default: 1 = every round)")
    p.add_argument("--layer-mismatch-layers", type=str, default=None, metavar="L1,L2,...",
                   help="Comma-separated layer names to monitor (default: auto-detect all Conv+Linear)")
    p.add_argument("--layer-mismatch-metrics", type=str, default=None, metavar="M1,M2,...",
                   help="Comma-separated metrics: drift,loss_jump,cka (default: all three)")

    # Compute-cost model (drives hardware/flop_cost.py)
    p.add_argument("--cost-model", type=str, default=None,
                   choices=["phi", "corrected", "measured"],
                   help="Per-round compute-FLOP cost model. Applied uniformly to "
                        "ALL algorithms in this run. 'phi' (legacy) reproduces "
                        "pre-refactor analytic formulas bit-for-bit. 'corrected' "
                        "uses the position-aware analytic model (groups only). "
                        "'measured' runs FlopCounterMode with the algo's actual "
                        "requires_grad mask (cached). Default: read from YAML "
                        "'cost_model' key, else 'phi' (backwards-compatible).")

    # Client sampling
    p.add_argument("--sample-fraction", type=float, default=1.0,
                   help="Fraction of clients selected per round (0<f<=1, default=1.0 = all)")
    p.add_argument("--sampling-strategy", type=str, default="random",
                   choices=["random", "battery_aware", "round_robin"],
                   help="Client selection strategy (default: random)")
    p.add_argument("--min-clients", type=int, default=1,
                   help="Minimum clients selected per round regardless of fraction (default: 1)")

    args = p.parse_args()

    # ── Resolve device: CLI > YAML 'device' key > auto-detect ────────────────
    # Save whether the user explicitly passed --device before we fill in the
    # default, so YAML can still override the auto-detected fallback.
    _cli_device_explicit = args.device is not None
    if not _cli_device_explicit:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    if args.list_algos:
        print("\nAvailable algorithms:")
        for info in list_algorithms():
            print(f"  {info['name']:<16} {info['description'][:60]}")
        return

    if args.benchmark:
        run_benchmark(args)
        return

    if args.config:
        # Load from YAML
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # CLI flags --algo and --output override YAML when explicitly provided.
        algo_name         = args.algo or cfg["training"].get("algorithm", "fedavg")
        algo_config       = cfg["training"].get("algo_config", {})
        dataset           = cfg["data"]["dataset"]
        model_name        = cfg["model"]["architecture"]
        num_rounds        = cfg["training"]["num_rounds"]
        num_clients       = cfg["clients"]["num_clients"]
        alpha             = cfg["data"].get("alpha", 0.5)
        partition         = cfg["data"].get("partition", "dirichlet")
        batch_size        = algo_config.get("batch_size", 32)
        seed              = cfg.get("seed", 42)
        output_dir        = args.output or cfg.get("output_dir", "./results")
        fleet_raw         = cfg["clients"].get("fleet", [])
        fleet_spec        = [(e["type"], e["count"]) for e in fleet_raw]
        if not fleet_spec:
            fleet_spec = DEFAULT_FLEET
        sample_fraction   = cfg["clients"].get("sample_fraction", 1.0)
        sampling_strategy = cfg["clients"].get("sampling_strategy", "random")
        min_clients       = cfg["clients"].get("min_clients", 1)
        battery_init      = cfg["clients"].get("battery_init", {})
        battery_dist      = battery_init.get("distribution", "gaussian")
        battery_params    = battery_init.get("params", None)
        layer_mismatch    = cfg.get("layer_mismatch", False)
        _lm_cfg = cfg.get("layer_mismatch_config", {})
        layer_mismatch_layers          = _lm_cfg.get("layers", None)
        layer_mismatch_freq            = int(_lm_cfg.get("freq", 1))
        layer_mismatch_metrics         = _lm_cfg.get("metrics", None)
        layer_mismatch_max_batches     = int(_lm_cfg.get("max_batches_per_client", 2))
        layer_mismatch_max_clients_cka = int(_lm_cfg.get("max_clients_cka", 8))
    else:
        algo_name         = args.algo or "fedavg"
        algo_config       = {
            "lr":            args.lr,
            "local_epochs":  args.epochs if args.epochs is not None else 1,
            "batch_size":    args.batch_size,
            "beta_min":      args.beta_min,
            "mu":            args.mu,
            "global_lr":     args.global_lr,
            "rho_min":       args.rho_min,
            "ema_alpha":     args.ema_alpha,
        }
        dataset           = args.dataset
        model_name        = args.model
        num_rounds        = args.rounds
        num_clients       = args.clients
        alpha             = args.alpha
        partition         = args.partition
        batch_size        = args.batch_size
        seed              = args.seed
        output_dir        = args.output or "./results"
        fleet_spec        = DEFAULT_FLEET
        sample_fraction   = args.sample_fraction
        sampling_strategy = args.sampling_strategy
        min_clients       = args.min_clients
        battery_dist      = "gaussian"
        battery_params    = None
        layer_mismatch         = args.layer_mismatch
        _raw_layers  = getattr(args, "layer_mismatch_layers", None)
        _raw_metrics = getattr(args, "layer_mismatch_metrics", None)
        layer_mismatch_layers          = [l.strip() for l in _raw_layers.split(",")] if _raw_layers else None
        layer_mismatch_freq            = int(getattr(args, "layer_mismatch_freq", 1))
        layer_mismatch_metrics         = [m.strip() for m in _raw_metrics.split(",")] if _raw_metrics else None
        layer_mismatch_max_batches     = int(getattr(args, "layer_mismatch_max_batches", 2))
        layer_mismatch_max_clients_cka = int(getattr(args, "layer_mismatch_max_clients_cka", 8))

    # ── CLI overrides for algo_config ────────────────────────────────────────
    if args.epochs is not None:
        algo_config["local_epochs"] = args.epochs
    if args.freeze_secondary:
        algo_config["freeze_secondary_backward"] = True

    # ── cost_model (single source of truth for compute FLOPs) ────────────────
    # Precedence: CLI > YAML training.cost_model > YAML top-level cost_model > "phi".
    # The same value is applied to every algorithm in the run; flop_cost reads
    # config["cost_model"] from algo_config.
    _yaml_cfg = locals().get("cfg", {}) or {}
    _yaml_train = _yaml_cfg.get("training") if isinstance(_yaml_cfg, dict) else None
    _yaml_cost_model = (
        (_yaml_train.get("cost_model") if isinstance(_yaml_train, dict) else None)
        or (_yaml_cfg.get("cost_model") if isinstance(_yaml_cfg, dict) else None)
    )
    _cost_model = args.cost_model or _yaml_cost_model or "phi"
    algo_config["cost_model"] = _cost_model
    if _cost_model != "phi":
        print(f"  cost_model: {_cost_model!r} (non-legacy — absolute energy will "
              f"differ from phi by ~10–200×; recalibrate energy_scale_factor)")

    # ── Final device: YAML 'device' key overrides auto-detect (not CLI) ────────
    _yaml_device = cfg.get("device", None)
    if not _cli_device_explicit and _yaml_device is not None:
        device = _yaml_device   # YAML explicit > auto-detected default
    else:
        device = args.device    # CLI explicit, or auto-detect (no YAML key)
    print(f"  Device: {device}")

    # Run single experiment
    result = run_single_experiment(
        algo_name=algo_name,
        algo_config=algo_config,
        dataset=dataset,
        model_name=model_name,
        num_rounds=num_rounds,
        num_clients=num_clients,
        alpha=alpha,
        partition=partition,
        batch_size=batch_size,
        fleet_spec=fleet_spec,
        battery_noise_std=0.15,
        device=device,
        seed=seed,
        data_root=getattr(args, "data_root", "./data"),
        verbose=not args.quiet,
        layer_mismatch=layer_mismatch,
        layer_mismatch_layers=layer_mismatch_layers,
        layer_mismatch_freq=layer_mismatch_freq,
        layer_mismatch_metrics=layer_mismatch_metrics,
        layer_mismatch_max_batches=layer_mismatch_max_batches,
        layer_mismatch_max_clients_cka=layer_mismatch_max_clients_cka,
        sample_fraction=sample_fraction,
        sampling_strategy=sampling_strategy,
        min_clients=min_clients,
        battery_dist=battery_dist,
        battery_params=battery_params,
    )

    # Save results
    out_dir = Path(output_dir) / f"{algo_name}_{dataset}_{model_name}_{partition}_ncl{num_clients}_r{num_rounds}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Survival / Pareto artifacts ─────────────────────────────────────────
    # Builds survival.csv next to metrics.json and folds a "survival" summary
    # block into the JSON for the ablation runner to read in one open().
    try:
        from metrics.survival import emit_survival_artifacts
        _rounds_log = result.get("rounds", [])
        _acc_hist = [r.get("test_accuracy") for r in _rounds_log]
        _client_states = result.get("client_states", [])  # populated by runner
        _survival = emit_survival_artifacts(
            output_dir=str(out_dir),
            rounds_metrics=_rounds_log,
            accuracy_history=_acc_hist,
            client_states=_client_states,
            num_clients_total=num_clients,
            num_rounds=num_rounds,
        )
        # Strip non-serializable side data before folding into JSON.
        result["survival"] = {
            k: v for k, v in _survival.items() if not k.startswith("_")
        }
        result["survival"]["csv_path"] = _survival.get("_csv_path")
        result["survival"]["lifetimes"] = _survival.get("_lifetimes", [])
        print(f"  Survival CSV: {out_dir}/survival.csv")
    except Exception as exc:
        print(f"  [warn] survival metrics skipped: {exc}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved: {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
