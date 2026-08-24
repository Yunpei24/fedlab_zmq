#!/usr/bin/env python3
"""
run_experiment_cv.py
====================
FL experiment runner for Computer Vision / Object Detection tasks.
Specifically designed for FedOD (WACV 2027).

Supports:
  - Dataset    : PASCAL VOC 2012 (20 classes, torchvision download)
  - Model      : SSDLite320 MobileNetV3-Large (torchvision detection)
  - Algorithms : fed_od  (ours)  |  fedavg  |  fedstep  (baselines)
  - Metric     : mAP@0.5 (VOC-style 11-point interpolation)
  - Fleet      : Raspberry Pi 4B (default) — realistic for edge OD

Key differences from run_experiment.py:
  1. Dataset: PASCAL VOC 2012 (not CIFAR-10)
  2. Model: SSDLite320 detection model (not ResNet classifier)
  3. Evaluation: mAP@0.5 + mAP@[.5:.95] (not accuracy)
  4. Loss: torchvision native detection loss (not CrossEntropyLoss)
  5. model_type="detection" passed to fed_od algorithm

Usage:
  # Run FedOD (ours):
  python run_experiment_cv.py --config configs/fed_od_voc.yaml --algo fed_od

  # Run baselines comparison:
  for ALGO in fedavg fedstep fed_od; do
    python run_experiment_cv.py \\
        --config configs/fed_od_voc.yaml \\
        --algo $ALGO \\
        --output results/fed_od_wacv/${ALGO}
  done

  # Quick smoke test (5 rounds, subset eval):
  python run_experiment_cv.py --algo fed_od --rounds 5 --clients 5 --quick

Config YAML structure (same as run_experiment.py + added fields):
  detection:
    num_classes: 21            # 20 VOC classes + background
    pretrained_backbone: true  # use ImageNet-pretrained MobileNetV3
    eval_freq: 5               # evaluate mAP every N rounds
    eval_max_batches: 50       # limit eval to N batches (speed up)
    score_threshold: 0.05      # min confidence for mAP eval
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import torch
import torchvision
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import algorithms.fed_od   # noqa — register fed_od
import algorithms.fedavg   # noqa
import algorithms.fedpart_be  # noqa
import algorithms.fedpart     # noqa
import algorithms.ccsEF       # noqa

from algorithms.base import get_algorithm, ClientState
from hardware.profiles import DEVICE_PROFILES, make_fleet
from datasets.voc_loader import (
    get_voc_dataloader, get_voc_test_loader, voc_dataset_sizes,
    VOC_CLASSES, NUM_CLASSES,
)
from metrics.map_eval import compute_map

# ─────────────────────────────────────────────────────────────────────────────
# Detection model factory
# ─────────────────────────────────────────────────────────────────────────────

def get_detection_model(
    model_name: str = "ssdlite320",
    num_classes: int = 21,
    pretrained_backbone: bool = True,
) -> torch.nn.Module:
    """
    Return a torchvision detection model.

    Args:
        model_name:          "ssdlite320" (default).
        num_classes:         Number of classes including background (21 for VOC).
        pretrained_backbone: Load ImageNet-pretrained MobileNetV3 backbone.
    """
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssdlite import SSDLiteHead
    from torchvision.models.detection._utils import retrieve_out_channels

    if model_name == "ssdlite320":
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1"
            if pretrained_backbone else None,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown detection model: {model_name}. Use 'ssdlite320'.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Client selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_clients(
    num_clients: int,
    sample_fraction: float,
    min_clients: int,
    client_states: list,
    rng: np.random.Generator,
) -> list[int]:
    k = max(min_clients, int(np.ceil(sample_fraction * num_clients)))
    k = min(k, num_clients)
    alive = [cid for cid in range(num_clients) if client_states[cid].battery_j > 0]
    if len(alive) <= k:
        return sorted(alive)
    chosen = rng.choice(alive, size=k, replace=False).tolist()
    return sorted(chosen)


def _round_energy_j(
    profile,
    model: torch.nn.Module,
    active_fraction: float,
    bytes_sent: int,
    config: dict,
) -> float:
    """
    Compute per-round energy in Joules using DeviceProfile API when available,
    falling back to a flat estimate otherwise.

    active_fraction: fraction of model trained (1.0 = full, 0.88 = backbone-only, etc.)
    """
    energy_scale_factor = config.get("energy_scale_factor", 1.0)
    batch_size    = config.get("batch_size", 8)
    local_epochs  = config.get("local_epochs", 3)
    dataset_size  = getattr(getattr(profile, "_ds_size", None), "__call__", None) or 500

    from hardware.profiles import DeviceProfile
    if isinstance(profile, DeviceProfile):
        num_params   = sum(p.numel() for p in model.parameters())
        active_params = int(num_params * active_fraction)
        full_flops   = profile.flops_for_model(active_params, batch_size, local_epochs, dataset_size)
        e_comp       = profile.compute_energy_j(full_flops)
        e_tx         = profile.comm_energy_j(bytes_sent, direction="uplink")
        return (e_comp + e_tx) * energy_scale_factor
    else:
        # flat fallback: 2.33 J/round base × scale factor
        e_flat = 2.33 * energy_scale_factor * active_fraction
        e_tx   = 1e-5 * bytes_sent * energy_scale_factor
        return e_flat + e_tx


# ─────────────────────────────────────────────────────────────────────────────
# Baseline training loop (FedAvg / FedStep adapted for detection)
# ─────────────────────────────────────────────────────────────────────────────

def _detection_client_update_fedavg(
    model: torch.nn.Module,
    dataloader,
    state: ClientState,
    config: dict,
) -> tuple[dict, dict]:
    """
    FedAvg-style full-model detection training (used as baseline).
    Full backward pass on backbone + head.
    """
    device       = next(model.parameters()).device
    lr           = config.get("lr", 0.005)
    local_epochs = config.get("local_epochs", 3)
    momentum     = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-4)
    optimizer_type = config.get("optimizer", "sgd").lower()
    energy_scale_factor = config.get("energy_scale_factor", 1.0)
    battery_max_j = config.get("battery_max_j", 183600.0)

    for param in model.parameters():
        param.requires_grad = True

    w_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            trainable, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    model.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        for images, targets in dataloader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    w_after = model.state_dict()
    delta = {k: (w_before[k] - w_after[k]).cpu() for k in w_before}

    # Energy (full model)
    profile = config.get("device_profile", {})
    model_size_bytes = sum(p.numel() * 4 for p in model.parameters())
    e_total = _round_energy_j(profile, model, 1.0, model_size_bytes, config)
    state.battery_j = max(0.0, state.battery_j - e_total)

    metadata = {
        "bytes_sent":       model_size_bytes,
        "energy_j":         e_total,
        "compression_ratio": 1.0,
        "local_loss":       total_loss / max(n_batches, 1),
        "tier":             1,
        "is_backbone_only": False,
        "is_warmup":        False,
    }
    return delta, metadata


# ─────────────────────────────────────────────────────────────────────────────
# FedStep-OD: battery-aware HEAD / full-model cycling for detection
# ─────────────────────────────────────────────────────────────────────────────

def _detection_client_update_fedstep(
    model: torch.nn.Module,
    dataloader,
    state: ClientState,
    config: dict,
) -> tuple[dict, dict]:
    """
    FedStep adapted for detection (SSDLite).

    Tier assignment for detection (contrast to FedOD):
      Tier 0 (β < beta_split) → HEAD ONLY training.
        Backbone frozen → cheap: only 11.9% params backward.
        Energy ratio ρ = (1 + α_h) / 2  where α_h = head fraction ≈ 0.119.
        ρ ≈ 0.56  → saves ~44% compute vs full model.
      Tier 1 (β ≥ beta_split) → full model (backbone + head).

    Server aggregation (caller):
      head   ← all clients (Tier 0 + Tier 1)
      backbone ← Tier-1 only

    This is the natural extension of FedStep's battery-tier strategy
    to SSDLite: assign the cheapest group (head) to low-battery clients.
    Contrast with FedOD which assigns the backbone (generalizable features)
    to low-battery clients — the key design choice under evaluation.
    """
    device         = next(model.parameters()).device
    lr             = config.get("lr", 0.005)
    local_epochs   = config.get("local_epochs", 3)
    momentum       = config.get("momentum", 0.9)
    weight_decay   = config.get("weight_decay", 1e-4)
    optimizer_type = config.get("optimizer", "sgd").lower()
    battery_max_j  = config.get("battery_max_j", 183600.0)
    beta_split     = config.get("beta_split", 0.50)
    warmup_rounds  = config.get("warmup_rounds", 5)

    battery_ratio  = state.battery_j / max(battery_max_j, 1.0)
    is_warmup      = state.round_num < warmup_rounds
    # Tier 0: low battery → head only (cheap). Tier 1: full model.
    is_head_only   = (not is_warmup) and (battery_ratio < beta_split)

    # Compute head fraction from model
    n_head  = sum(p.numel() for p in model.head.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    alpha_h = n_head / max(n_total, 1)   # ≈ 0.119 for SSDLite

    # Freeze backbone for head-only clients
    if is_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = not name.startswith("backbone.")
    else:
        for param in model.parameters():
            param.requires_grad = True

    w_before     = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainable    = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            trainable, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    model.train()
    total_loss, n_batches = 0.0, 0
    for _ in range(local_epochs):
        for images, targets in dataloader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

    w_after = model.state_dict()
    delta   = {k: (w_before[k] - w_after[k]).cpu() for k in w_before}

    # Zero-out backbone delta for head-only clients
    if is_head_only:
        for k in delta:
            if k.startswith("backbone."):
                delta[k] = torch.zeros_like(delta[k])

    # Energy: head-only ρ = (1 + α_h) / 2 ≈ 0.56
    profile          = config.get("device_profile", {})
    active_fraction  = (1.0 + alpha_h) / 2.0 if is_head_only else 1.0
    model_size_bytes = sum(p.numel() * 4 for p in model.parameters())
    bytes_sent       = int(model_size_bytes * alpha_h) if is_head_only else model_size_bytes
    e_total          = _round_energy_j(profile, model, active_fraction, bytes_sent, config)
    state.battery_j  = max(0.0, state.battery_j - e_total)

    state.custom["tier"]        = 0 if is_head_only else 1
    state.custom["is_head_only"] = is_head_only

    metadata = {
        "bytes_sent":        bytes_sent,
        "energy_j":          e_total,
        "compression_ratio": alpha_h if is_head_only else 1.0,
        "local_loss":        total_loss / max(n_batches, 1),
        "tier":              0 if is_head_only else 1,
        "is_backbone_only":  False,
        "is_head_only":      is_head_only,
        "is_warmup":         is_warmup,
    }
    return delta, metadata


def _fedstep_od_aggregate(
    global_model: torch.nn.Module,
    client_updates: list[tuple[dict, dict, ClientState]],
    server_lr: float = 1.0,
) -> dict:
    """
    FedStep-OD server aggregation:
      head params    ← all clients (Tier 0 head-only + Tier 1 full)
      backbone params ← Tier-1 clients only
    """
    new_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

    all_updates  = [(d, cs.custom.get("dataset_size", 1)) for d, _, cs in client_updates]
    full_updates = [
        (d, cs.custom.get("dataset_size", 1))
        for d, _, cs in client_updates
        if cs.custom.get("tier", 1) == 1
    ]
    if not full_updates:
        full_updates = all_updates  # fallback

    for name in new_weights:
        if name.startswith("backbone."):
            # backbone: Tier-1 only
            acc, w_sum = None, 0
            for delta, n in full_updates:
                if name in delta:
                    c = delta[name].float() * n
                    acc = c if acc is None else acc + c
                    w_sum += n
            if acc is not None and w_sum > 0:
                new_weights[name] = new_weights[name] - server_lr * (acc / w_sum).to(
                    new_weights[name].device
                )
        else:
            # head + other (transform, anchor_generator): all clients
            acc, w_sum = None, 0
            for delta, n in all_updates:
                if name in delta:
                    c = delta[name].float() * n
                    acc = c if acc is None else acc + c
                    w_sum += n
            if acc is not None and w_sum > 0:
                new_weights[name] = new_weights[name] - server_lr * (acc / w_sum).to(
                    new_weights[name].device
                )

    return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# Server aggregation for detection baselines (FedAvg-style weighted mean)
# ─────────────────────────────────────────────────────────────────────────────

def _fedavg_aggregate_detection(
    global_model: torch.nn.Module,
    client_updates: list[tuple[dict, dict, ClientState]],
    server_lr: float = 1.0,
) -> dict:
    """Dataset-size weighted average of model deltas (FedAvg on detection model)."""
    total_weight = sum(cs.custom.get("dataset_size", 1) for _, _, cs in client_updates)
    if total_weight == 0:
        return global_model.state_dict()

    new_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
    for name in new_weights:
        weighted_delta = None
        w_sum = 0
        for delta, _, cs in client_updates:
            n = cs.custom.get("dataset_size", 1)
            if name in delta:
                contrib = delta[name].float() * n
                weighted_delta = contrib if weighted_delta is None else weighted_delta + contrib
                w_sum += n
        if weighted_delta is not None and w_sum > 0:
            new_weights[name] = new_weights[name] - server_lr * (weighted_delta / w_sum).to(
                new_weights[name].device
            )
    return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# Centralized upper bound
# ─────────────────────────────────────────────────────────────────────────────

def run_centralized_experiment(
    model_name: str,
    num_rounds: int,
    alpha: float,
    device: str,
    seed: int,
    data_root: str,
    num_clients: int,
    num_classes: int = 21,
    pretrained_backbone: bool = True,
    eval_freq: int = 5,
    eval_max_batches: Optional[int] = 50,
    score_threshold: float = 0.05,
    batch_size: int = 8,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """
    Centralized upper bound: pool all client data and train a single SSDLite model.
    No battery constraints, no FL communication overhead.
    Reports mAP every eval_freq rounds (= epochs over pooled data).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = get_detection_model(model_name, num_classes, pretrained_backbone)
    model = model.to(device)
    model.train()

    # Pool all client data by loading each partition and concatenating
    from datasets.voc_loader import VOCDetectionFL, voc_collate_fn, _dirichlet_partition

    voc_train = torchvision.datasets.VOCDetection(
        root=data_root, year="2012", image_set="train",
        download=True, transform=None,
    )
    partitions = _dirichlet_partition(voc_train, num_clients, alpha, seed)
    all_indices = [idx for p in partitions for idx in p]

    from datasets.voc_loader import VOCDetectionFL
    pooled = VOCDetectionFL(data_root, "2012", "train", indices=all_indices)
    pooled_loader = torch.utils.data.DataLoader(
        pooled, batch_size=batch_size, shuffle=True,
        collate_fn=voc_collate_fn, num_workers=0, pin_memory=False,
    )
    test_loader = get_voc_test_loader(data_root, batch_size=batch_size)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )

    rounds_log = []
    best_map50  = 0.0
    map_results: dict = {}

    if verbose:
        print(f"\n  [Centralized] {len(pooled)} images pooled from {num_clients} clients")
        print(f"{'='*70}")
        print(f"  CENTRALIZED | VOC2012 | {model_name} | {num_rounds} rounds (epochs)")
        print(f"{'='*70}")

    for t in range(num_rounds):
        t0 = time.time()
        model.train()
        for images, targets in pooled_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if (t + 1) % eval_freq == 0 or t == 0 or t == num_rounds - 1:
            model.eval()
            try:
                map_results = compute_map(
                    model, test_loader, device,
                    max_batches=eval_max_batches,
                    score_threshold=score_threshold,
                    num_classes=num_classes,
                )
                if map_results["map_50"] > best_map50:
                    best_map50 = map_results["map_50"]
            except Exception as e:
                if verbose:
                    print(f"  Warning: mAP eval failed at round {t+1}: {e}")
            model.train()

        elapsed = time.time() - t0
        map50 = map_results.get("map_50", float("nan"))
        rounds_log.append({"round_num": t + 1, "map_50": map50,
                            "best_map_50": best_map50, "elapsed_s": elapsed})

        if verbose and ((t + 1) % eval_freq == 0 or t == 0 or t == num_rounds - 1):
            map_str = f"{map50*100:.2f}%" if not np.isnan(map50) else "  N/A "
            print(f"  Epoch {t+1:3d}/{num_rounds} | mAP@0.5={map_str} | ⏱{elapsed:.1f}s")

    valid_maps = [r["map_50"] for r in rounds_log if not np.isnan(r["map_50"])]
    summary = {
        "algorithm":      "centralized",
        "dataset":        "voc2012",
        "model":          model_name,
        "num_rounds":     num_rounds,
        "seed":           seed,
        "best_map_50":    max(valid_maps) if valid_maps else 0.0,
        "final_map_50":   rounds_log[-1].get("map_50", 0.0),
        "final_survival": 1.0,   # N/A — all "clients" always alive
        "system_lifetime": num_rounds,
    }
    return {"algorithm": "centralized", "summary": summary, "rounds": rounds_log}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_od_experiment(
    algo_name: str,
    algo_config: dict,
    model_name: str,
    num_rounds: int,
    num_clients: int,
    alpha: float,
    partition: str,
    fleet_spec: list,
    device: str,
    seed: int,
    data_root: str,
    sample_fraction: float = 1.0,
    min_clients: int = 3,
    battery_dist: str = "uniform_soc",
    battery_params: Optional[dict] = None,
    num_classes: int = 21,
    pretrained_backbone: bool = True,
    eval_freq: int = 5,
    eval_max_batches: Optional[int] = 50,
    score_threshold: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Run a single federated object detection experiment.

    Returns:
        dict with keys: algorithm, config, rounds (per-round metrics), summary
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Algorithm ────────────────────────────────────────────────────────────
    algo = get_algorithm(algo_name)
    default_cfg = algo.get_default_config()
    # Force detection mode for fed_od
    algo_config.setdefault("model_type", "detection")
    merged_config = {**default_cfg, **algo_config, "device": device}

    use_fed_od_algo      = (algo_name == "fed_od")
    use_fedstep_od    = (algo_name in ("fedstep", "fedpart_be", "fedpart"))

    # ── Detection model ───────────────────────────────────────────────────────
    global_model = get_detection_model(model_name, num_classes, pretrained_backbone)
    global_model.to(device)

    _client_model = get_detection_model(model_name, num_classes, pretrained_backbone)
    _client_model.to(device)

    # Log model size
    n_params  = sum(p.numel() for p in global_model.parameters()) / 1e6
    n_backbone = sum(p.numel() for p in global_model.backbone.parameters()) / 1e6
    if verbose:
        print(f"  Model: {model_name} | "
              f"Total={n_params:.2f}M | Backbone={n_backbone:.2f}M "
              f"({100*n_backbone/n_params:.1f}%) | Head={n_params-n_backbone:.2f}M "
              f"({100*(n_params-n_backbone)/n_params:.1f}%)")

    # ── Test loader (VOC val) ─────────────────────────────────────────────────
    batch_size = algo_config.get("batch_size", 8)
    test_loader = get_voc_test_loader(data_root=data_root, batch_size=batch_size)

    # ── Client dataloaders ────────────────────────────────────────────────────
    client_loaders = [
        get_voc_dataloader(
            data_root=data_root, split="train",
            client_id=cid, num_clients=num_clients,
            alpha=alpha, partition=partition,
            batch_size=batch_size, seed=seed,
        )
        for cid in range(num_clients)
    ]

    # Dataset sizes for weighted aggregation
    dataset_sizes = voc_dataset_sizes(data_root, num_clients, alpha, partition, seed)
    if verbose:
        print(f"  Dataset sizes: min={min(dataset_sizes)} max={max(dataset_sizes)} "
              f"mean={np.mean(dataset_sizes):.0f} images/client")

    # ── Fleet ────────────────────────────────────────────────────────────────
    fleet = make_fleet(
        fleet_spec,
        battery_noise_std=0.05,
        battery_dist=battery_dist,
        battery_params=battery_params,
        seed=seed,
    )
    while len(fleet) < num_clients:
        fleet.append(copy.deepcopy(fleet[len(fleet) % len(fleet)]))
    fleet = fleet[:num_clients]

    # ── Client states ─────────────────────────────────────────────────────────
    client_states = []
    for cid in range(num_clients):
        cs = ClientState(client_id=cid, battery_j=fleet[cid].battery.initial_energy_j)
        cs.custom["dataset_size"] = dataset_sizes[cid]
        client_states.append(cs)

    if verbose:
        batteries = [cs.battery_j for cs in client_states]
        full_cap  = fleet[0].battery.capacity_j
        print(f"\n  Fleet: {num_clients} × {fleet[0].name}")
        print(f"  Battery  min={min(batteries):.0f}J "
              f"({100*min(batteries)/full_cap:.1f}% SOC)  "
              f"max={max(batteries):.0f}J "
              f"({100*max(batteries)/full_cap:.1f}% SOC)")
        print(f"\n{'='*70}")
        print(f"  {algo_name.upper()} | VOC2012 | {model_name} | "
              f"{num_clients} clients | {num_rounds} rounds")
        print(f"{'='*70}")

    _sample_rng = np.random.default_rng(seed + 7777)
    rounds_log  = []
    best_map50  = 0.0

    for t in range(num_rounds):
        t0 = time.time()

        global_sd = {k: v.clone() for k, v in global_model.state_dict().items()}

        # ── Client selection ──────────────────────────────────────────────
        selected = _select_clients(
            num_clients, sample_fraction, min_clients, client_states, _sample_rng
        )

        client_tuples = []
        client_train_times = []

        for cid in selected:
            if client_states[cid].battery_j <= 0.0:
                continue

            _client_model.load_state_dict(global_sd)
            client_config = {
                **merged_config,
                "device_profile": fleet[cid],
            }
            _bat_before = client_states[cid].battery_j

            _t0_client = time.time()
            if use_fed_od_algo:
                # FedOD: backbone/head split via algo.client_update (detection mode)
                update, metadata = algo.client_update(
                    model=_client_model,
                    dataloader=client_loaders[cid],
                    state=client_states[cid],
                    config=client_config,
                )
            elif use_fedstep_od:
                # FedStep-OD: head-only for low-battery, full for high-battery
                update, metadata = _detection_client_update_fedstep(
                    model=_client_model,
                    dataloader=client_loaders[cid],
                    state=client_states[cid],
                    config=client_config,
                )
            else:
                # FedAvg: full model, no battery-aware split
                update, metadata = _detection_client_update_fedavg(
                    model=_client_model,
                    dataloader=client_loaders[cid],
                    state=client_states[cid],
                    config=client_config,
                )
            client_train_times.append(time.time() - _t0_client)
            client_states[cid].custom["tier"] = metadata.get("tier", 1)
            client_tuples.append((update, metadata, client_states[cid]))

            if _bat_before > 0.0 and client_states[cid].battery_j <= 0.0:
                alive_after = sum(1 for cs in client_states if cs.battery_j > 0)
                print(f"  ☠  Round {t+1:3d} | Client {cid:2d} [{fleet[cid].name}] "
                      f"depleted — {alive_after}/{num_clients} alive")

        del global_sd

        if not client_tuples:
            if verbose:
                print(f"  Round {t+1:3d}: all clients depleted — stopping.")
            break

        # ── Server aggregation ─────────────────────────────────────────────
        if use_fed_od_algo:
            agg_result = algo.server_aggregate(
                global_model=global_model,
                client_updates=client_tuples,
                round_num=t,
                config=merged_config,
            )
            global_model.load_state_dict(
                {k: v.to(device) for k, v in agg_result.new_weights.items()}
            )
            del agg_result.new_weights
        elif use_fedstep_od:
            # FedStep-OD: head from all clients, backbone from Tier-1 only
            server_lr = merged_config.get("server_lr", 1.0) or 1.0
            new_weights = _fedstep_od_aggregate(
                global_model, client_tuples, server_lr
            )
            global_model.load_state_dict(
                {k: v.to(device) for k, v in new_weights.items()}
            )
        else:
            # FedAvg: standard weighted mean over all clients
            server_lr = merged_config.get("server_lr", 1.0) or 1.0
            new_weights = _fedavg_aggregate_detection(
                global_model, client_tuples, server_lr
            )
            global_model.load_state_dict(
                {k: v.to(device) for k, v in new_weights.items()}
            )

        n_participated = len(client_tuples)
        del client_tuples
        gc.collect()

        # ── mAP evaluation (every eval_freq rounds) ──────────────────────
        map_results = {"map_50": float("nan"), "map": float("nan")}
        if (t + 1) % eval_freq == 0 or t == 0 or t == num_rounds - 1:
            try:
                map_results = compute_map(
                    model=global_model,
                    dataloader=test_loader,
                    device=device,
                    num_classes=num_classes,
                    iou_threshold=0.5,
                    score_threshold=score_threshold,
                    max_batches=eval_max_batches,
                )
                global_model.train()
                if map_results["map_50"] > best_map50:
                    best_map50 = map_results["map_50"]
            except Exception as e:
                if verbose:
                    print(f"  Warning: mAP evaluation failed at round {t+1}: {e}")

        # ── Round metrics ──────────────────────────────────────────────────
        alive_clients = sum(1 for cs in client_states if cs.battery_j > 0)
        survival_ratio = alive_clients / num_clients

        elapsed = time.time() - t0
        sim_round_time = max(client_train_times) if client_train_times else 0.0

        map50     = map_results.get("map_50", float("nan"))
        map_coco  = map_results.get("map",    float("nan"))
        n_bb_only = sum(
            1 for cs in client_states
            if cs.custom.get("tier", 1) == 0 and cs.battery_j > 0
        )

        round_metrics = {
            "round_num":         t + 1,
            "map_50":            map50,
            "map_coco":          map_coco,
            "best_map_50":       best_map50,
            "n_backbone_only":   n_bb_only,
            "n_full_training":   n_participated - n_bb_only,
            "num_alive_clients": alive_clients,
            "survival_ratio":    survival_ratio,
            "num_selected":      n_participated,
            "elapsed_s":         elapsed,
            "sim_round_time_s":  sim_round_time,
        }
        rounds_log.append(round_metrics)

        if verbose and ((t + 1) % eval_freq == 0 or t == 0 or t == num_rounds - 1):
            map_str = f"{map50*100:.2f}%" if not np.isnan(map50) else "  N/A "
            print(f"  Round {t+1:3d}/{num_rounds} | "
                  f"mAP@0.5={map_str} | "
                  f"Alive={alive_clients}/{num_clients} | "
                  f"BB-only={n_bb_only} | "
                  f"⏱{elapsed:.1f}s")

        if all(cs.battery_j <= 0.0 for cs in client_states):
            if verbose:
                print(f"  Fleet fully depleted at round {t+1}. Stopping.")
            break

    # ── Summary ────────────────────────────────────────────────────────────
    valid_maps = [r["map_50"] for r in rounds_log if not np.isnan(r["map_50"])]
    best_map   = max(valid_maps) if valid_maps else 0.0
    final_r    = rounds_log[-1]

    summary = {
        "algorithm":        algo_name,
        "dataset":          "voc2012",
        "model":            model_name,
        "partition":        partition,
        "alpha":            alpha,
        "num_rounds":       num_rounds,
        "num_clients":      num_clients,
        "seed":             seed,
        "best_map_50":      best_map,
        "final_map_50":     final_r.get("map_50", 0.0),
        "final_survival":   final_r["survival_ratio"],
        "system_lifetime":  next(
            (r["round_num"] for r in rounds_log if r["survival_ratio"] <= 0.5),
            num_rounds,
        ),
        "map_at_half_dropout": next(
            (r["map_50"] for r in rounds_log if r["survival_ratio"] <= 0.5),
            final_r.get("map_50", 0.0),
        ),
        "algo_config": merged_config,
    }

    if verbose:
        print(f"\n  Summary: best mAP@0.5={best_map*100:.2f}% | "
              f"final survival={final_r['survival_ratio']:.2f} | "
              f"system lifetime={summary['system_lifetime']} rounds")

    return {
        "algorithm": algo_name,
        "dataset":   "voc2012",
        "config":    merged_config,
        "rounds":    rounds_log,
        "summary":   summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FLEET_CV = [("raspberry_pi_4", 20)]


def main():
    p = argparse.ArgumentParser(
        description="FedOD — Federated Object Detection Experiment Runner (WACV 2027)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",   type=str, default=None,
                   help="YAML config file (same format as run_experiment.py)")
    p.add_argument("--algo",     type=str, default="fed_od",
                   choices=["fed_od", "fedavg", "fedstep", "fedpart", "centralized"],
                   help="FL algorithm to run")
    p.add_argument("--output",   type=str, default=None,
                   help="Output directory (overrides YAML output_dir)")
    p.add_argument("--rounds",   type=int, default=200)
    p.add_argument("--clients",  type=int, default=20)
    p.add_argument("--alpha",    type=float, default=0.5,
                   help="Dirichlet alpha for non-IID partitioning")
    p.add_argument("--partition", type=str, default="dirichlet",
                   choices=["dirichlet", "iid"])
    p.add_argument("--lr",       type=float, default=0.005)
    p.add_argument("--epochs",   type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device",   type=str, default=None,
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--eval-freq", type=int, default=5,
                   help="Evaluate mAP every N rounds (default: 5)")
    p.add_argument("--eval-batches", type=int, default=50,
                   help="Limit mAP eval to first N batches (None=all, default: 50)")
    p.add_argument("--quick",    action="store_true",
                   help="Quick test: 3 rounds, 10 eval batches, 3 clients")
    p.add_argument("--quiet",    action="store_true")
    p.add_argument("--benchmark", action="store_true",
                   help="Main benchmark: centralized (upper bound) + fedavg + fed_od")
    p.add_argument("--ablation",  action="store_true",
                   help="Layer-choice ablation: fedavg + fedstep + fed_od")

    args = p.parse_args()

    if args.quick:
        args.rounds      = 3
        args.clients     = 3
        args.eval_freq   = 1
        args.eval_batches = 5
        print("  [QUICK MODE] 3 rounds, 3 clients, 5 eval batches")

    # ── Device auto-detect ───────────────────────────────────────────────────
    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    # ── YAML config ──────────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        algo_name    = args.algo or cfg["training"].get("algorithm", "fed_od")
        algo_config  = cfg["training"].get("algo_config", {})
        num_rounds   = args.rounds  if args.rounds  != 200 else cfg["training"].get("num_rounds", 200)
        num_clients  = args.clients if args.clients != 20  else cfg["clients"]["num_clients"]
        alpha        = cfg["data"].get("alpha", 0.5)
        partition    = cfg["data"].get("partition", "dirichlet")
        seed         = cfg.get("seed", 42)
        data_root    = cfg["data"].get("data_root", "./data")
        output_dir   = args.output or cfg.get("output_dir", "./results/fed_od")
        model_name   = cfg.get("detection", {}).get("model", "ssdlite320")
        num_classes  = cfg.get("detection", {}).get("num_classes", 21)
        pretrained_bb = cfg.get("detection", {}).get("pretrained_backbone", True)
        eval_freq    = cfg.get("detection", {}).get("eval_freq", args.eval_freq)
        eval_batches = cfg.get("detection", {}).get("eval_max_batches", args.eval_batches)
        score_thr    = cfg.get("detection", {}).get("score_threshold", 0.05)
        fleet_raw    = cfg["clients"].get("fleet", [])
        fleet_spec   = [(e["type"], e["count"]) for e in fleet_raw] or DEFAULT_FLEET_CV
        bat_init     = cfg["clients"].get("battery_init", {})
        bat_dist     = bat_init.get("distribution", "uniform_soc")
        bat_params   = bat_init.get("params", None)
        device       = args.device or cfg.get("device", "cpu")
    else:
        algo_name    = args.algo
        algo_config  = {
            "lr":           args.lr,
            "local_epochs": args.epochs,
            "batch_size":   args.batch_size,
        }
        num_rounds   = args.rounds
        num_clients  = args.clients
        alpha        = args.alpha
        partition    = args.partition
        seed         = args.seed
        data_root    = args.data_root
        output_dir   = args.output or "./results/fed_od"
        model_name   = "ssdlite320"
        num_classes  = 21
        pretrained_bb = True
        eval_freq    = args.eval_freq
        eval_batches = args.eval_batches
        score_thr    = 0.05
        fleet_spec   = DEFAULT_FLEET_CV
        bat_dist     = "uniform_soc"
        bat_params   = {"min_soc": 0.10, "max_soc": 0.90}
        device       = args.device

    print(f"\n  Device: {device}")

    def _run_one(name):
        if name == "centralized":
            result = run_centralized_experiment(
                model_name=model_name,
                num_rounds=num_rounds,
                alpha=alpha,
                device=device,
                seed=seed,
                data_root=data_root,
                num_clients=num_clients,
                num_classes=num_classes,
                pretrained_backbone=pretrained_bb,
                eval_freq=eval_freq,
                eval_max_batches=eval_batches,
                score_threshold=score_thr,
                batch_size=algo_config.get("batch_size", 8),
                lr=algo_config.get("lr", 0.005),
                momentum=algo_config.get("momentum", 0.9),
                weight_decay=algo_config.get("weight_decay", 1e-4),
                verbose=not args.quiet,
            )
        else:
            cfg_copy = dict(algo_config)
            cfg_copy["model_type"] = "detection"
            result = run_od_experiment(
                algo_name=name,
                algo_config=cfg_copy,
                model_name=model_name,
                num_rounds=num_rounds,
                num_clients=num_clients,
                alpha=alpha,
                partition=partition,
                fleet_spec=fleet_spec,
                device=device,
                seed=seed,
                data_root=data_root,
                min_clients=3,
                battery_dist=bat_dist,
                battery_params=bat_params,
                num_classes=num_classes,
                pretrained_backbone=pretrained_bb,
                eval_freq=eval_freq,
                eval_max_batches=eval_batches,
                score_threshold=score_thr,
                verbose=not args.quiet,
            )
        out_path = Path(output_dir) / name / "metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Results saved → {out_path}")
        return result

    def _print_table(results: dict, title: str):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        print(f"  {'Algorithm':<16} {'Best mAP@0.5':>13} {'Final mAP@0.5':>14} "
              f"{'Lifetime':>10} {'Survival@end':>13}")
        print(f"  {'-'*68}")
        for name, r in results.items():
            s = r["summary"]
            lifetime = s["system_lifetime"]
            survival = s["final_survival"] * 100
            lifetime_str = f"{lifetime}" if name != "centralized" else "  N/A"
            survival_str = f"{survival:.1f}%" if name != "centralized" else "  N/A"
            print(f"  {name:<16} {s['best_map_50']*100:>12.2f}% "
                  f"{s['final_map_50']*100:>13.2f}% "
                  f"{lifetime_str:>10} {survival_str:>13}")
        print(f"{'='*70}")

    if args.benchmark:
        # Main benchmark: centralized (upper bound) + FedAvg + FedOD
        results = {}
        for name in ["centralized", "fedavg", "fed_od"]:
            print(f"\n{'#'*70}")
            print(f"  Running: {name.upper()}")
            print(f"{'#'*70}")
            results[name] = _run_one(name)
        _print_table(results, f"BENCHMARK — VOC2012 | Dir(α={alpha}) | {num_rounds} rounds")
        cmp_path = Path(output_dir) / "benchmark_summary.json"
        with open(cmp_path, "w") as f:
            json.dump({n: r["summary"] for n, r in results.items()}, f, indent=2, default=str)
        print(f"\n  Comparison saved → {cmp_path}")

    elif args.ablation:
        # Layer-choice ablation: FedAvg + FedStep-OD + FedOD
        results = {}
        for name in ["fedavg", "fedstep", "fed_od"]:
            print(f"\n{'#'*70}")
            print(f"  Running: {name.upper()}")
            print(f"{'#'*70}")
            results[name] = _run_one(name)
        _print_table(results, f"ABLATION — Layer choice | VOC2012 | Dir(α={alpha}) | {num_rounds} rounds")
        cmp_path = Path(output_dir) / "ablation_layer_summary.json"
        with open(cmp_path, "w") as f:
            json.dump({n: r["summary"] for n, r in results.items()}, f, indent=2, default=str)
        print(f"\n  Ablation saved → {cmp_path}")
    else:
        _run_one(algo_name)


if __name__ == "__main__":
    main()
