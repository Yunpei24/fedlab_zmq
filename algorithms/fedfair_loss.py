"""Efficient non-private FedFair loss-variance baseline.

This is the batch-level objective used as the loss-space control for DMD.  It
is deliberately distinct from :mod:`algorithms.fedfdp`, whose FedFair/FedFDP
implementation exposes per-example fair clipping and, optionally, DP noise.
"""

from __future__ import annotations

import gc
from collections import OrderedDict

import torch
from torch import nn, optim

from hardware.flop_cost import round_compute_flops

from .base import AggregateResult, FLAlgorithm, register_algorithm
from .reference_utils import apply_delta, common_round_metrics, empirical_loss


@register_algorithm("fedfair_loss")
class FedFairLoss(FLAlgorithm):
    """CE plus a delayed quadratic deviation from the global client loss."""

    description = "Non-private FedFair batch loss-variance objective."

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        model.to(device)
        profile_loader = config.get("anchor_dataloader") or dataloader
        loss_at_global = empirical_loss(model, profile_loader, device)
        before = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(config.get("lr", 0.03)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        criterion = nn.CrossEntropyLoss()
        reference = config.get("fedfair_loss_reference")
        server_round = int(config.get("_server_round", state.round_num))
        active = reference is not None and server_round >= int(
            config.get("warmup_rounds", 1)
        )
        fairness_lambda = float(config.get("fairness_lambda", 0.1))
        total_loss = total_ce = total_fair = 0.0
        batches = 0
        model.train()
        for _ in range(int(config.get("local_epochs", 1))):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                ce = criterion(model(x), y)
                fair = (
                    0.5 * fairness_lambda * (ce - float(reference)).square()
                    if active
                    else ce * 0.0
                )
                loss = ce + fair
                loss.backward()
                max_norm = config.get("max_grad_norm")
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
                optimizer.step()
                total_loss += float(loss.detach())
                total_ce += float(ce.detach())
                total_fair += float(fair.detach())
                batches += 1

        current = model.state_dict()
        delta = OrderedDict(
            (key, (before[key] - current[key].detach().cpu()).float())
            for key in before
        )
        uplink = self.count_bytes(delta, sparse=False) + 8
        downlink = self.count_bytes(delta, sparse=False) + 8
        profile = config.get("device_profile")
        if profile:
            flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                int(config.get("local_epochs", 1)),
            )
            energy = profile.round_energy_breakdown(
                flops,
                uplink,
                downlink,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            value = 2.5 * float(config.get("energy_scale_factor", 1.0))
            energy = {"compute": value, "uplink": 0.0, "downlink": 0.0, "total": value}
        state.battery_j = max(0.0, state.battery_j - float(energy["total"]))
        state.round_num += 1
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "dataset_size": len(dataloader.dataset),
            "fedfair_loss_at_global": float(loss_at_global),
            "fedfair_reference_applied": bool(active),
            "local_loss": total_loss / max(batches, 1),
            "local_ce": total_ce / max(batches, 1),
            "local_fedfair_addend": total_fair / max(batches, 1),
            "bytes_sent": uplink,
            "bytes_received": downlink,
            "energy_j_consumed": float(energy["total"]),
            "energy_compute_j": float(energy["compute"]),
            "energy_uplink_j": float(energy["uplink"]),
            "energy_downlink_j": float(energy["downlink"]),
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
        }
        del optimizer, before, current
        gc.collect()
        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        sizes = torch.tensor(
            [meta.get("dataset_size", 1) for _, meta, _ in client_updates],
            dtype=torch.float64,
        )
        weights = sizes / sizes.sum().clamp_min(1)
        aggregate = None
        for weight, (update, _, _) in zip(weights, client_updates):
            if aggregate is None:
                aggregate = {
                    key: value.clone().float() * weight for key, value in update.items()
                }
            else:
                for key in aggregate:
                    aggregate[key] += update[key].float() * weight
        reference = sum(
            float(weight) * float(meta["fedfair_loss_at_global"])
            for weight, (_, meta, _) in zip(weights, client_updates)
        )
        metrics = common_round_metrics(client_updates)
        metrics.update(
            {
                "round": round_num,
                "fedfair_loss_reference_next": reference,
                "fedfair_context_clients": sum(
                    bool(meta.get("fedfair_reference_applied", False))
                    for _, meta, _ in client_updates
                ),
                "avg_local_ce": sum(meta["local_ce"] for _, meta, _ in client_updates)
                / len(client_updates),
                "avg_local_fedfair_addend": sum(
                    meta["local_fedfair_addend"] for _, meta, _ in client_updates
                )
                / len(client_updates),
                "_server_state_updates": {"fedfair_loss_reference": reference},
            }
        )
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        return {
            "lr": 0.03,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "local_epochs": 1,
            "batch_size": 64,
            "device": "cpu",
            "fairness_lambda": 0.1,
            "warmup_rounds": 1,
            "client_metrics_every": 1,
        }


__all__ = ["FedFairLoss"]
