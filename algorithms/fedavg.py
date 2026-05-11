"""
algorithms/fedavg.py
====================
FedAvg: Communication-Efficient Learning of Deep Networks
McMahan et al., AISTATS 2017
"""

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("fedavg")
class FedAvg(FLAlgorithm):
    """
    Standard FedAvg baseline. Full-precision, no compression.
    Used as reference for accuracy and communication cost comparisons.
    """
    name = "fedavg"
    description = "Standard FedAvg (McMahan et al., 2017). No compression."

    def client_update(self, model, dataloader, state, config):
        device        = config.get("device", "cpu")
        lr            = config.get("lr", 0.01)
        local_epochs  = config.get("local_epochs", 1)
        max_grad_norm = config.get("max_grad_norm", None)

        w_before = OrderedDict({k: v.clone().cpu() for k, v in model.state_dict().items()})

        model.train(); model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        total_loss, num_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss.item(); num_batches += 1

        current_sd = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - current_sd[k].cpu()).float()
            for k in w_before
        })
        del w_before
        del current_sd
        gc.collect()

        profile = config.get("device_profile")
        uplink_bytes   = self.count_bytes(delta, sparse=False)
        downlink_bytes = self.count_bytes(delta, sparse=False)  # same size as uplink for FedAvg (full model)
        if profile:
            num_params = sum(p.numel() for p in model.parameters())
            flops = profile.flops_for_model(num_params, dataloader.batch_size,
                                             local_epochs, len(dataloader.dataset))
            energy_j = profile.round_energy_j(flops, uplink_bytes, downlink_bytes)
        else:
            energy_j = 0.5 + 2.0 * 1.0  # full transmission

        energy_j *= config.get("energy_scale_factor", 1.0)
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        metadata = {
            "client_id": state.client_id, "round_num": state.round_num,
            "beta_actual": 1.0, "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j, "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": total_loss / max(num_batches, 1),
            "compression_ratio": 1.0,
        }
        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        K = len(client_updates)
        global_sd = global_model.state_dict()

        # Dataset-size weights for BN aggregation (FedAvg proper weighting)
        sizes = [m.get("dataset_size", 1) for _, m, _ in client_updates]
        total_n = max(sum(sizes), 1)

        agg = None
        agg_bn: dict = {}
        for (update, meta, _), n_k in zip(client_updates, sizes):
            w_k = n_k / total_n
            if agg is None:
                agg = {k: v.clone().float() for k, v in update.items()}
            else:
                for k in agg:
                    agg[k] += update[k].float()
            # Weighted BN accumulation (separate pass)
            for k, v in update.items():
                if k.endswith((".running_mean", ".running_var")):
                    if k not in agg_bn:
                        agg_bn[k] = v.clone().float() * w_k
                    else:
                        agg_bn[k] += v.float() * w_k

        new_weights = OrderedDict({
            k: global_sd[k].float() - (agg[k] / K).to(global_sd[k].device) for k in global_sd
        })
        # Override BN running stats with dataset-size-weighted average
        for k, v in agg_bn.items():
            if k in new_weights:
                new_weights[k] = global_sd[k].float() - v.to(global_sd[k].device)
        del agg
        gc.collect()
        total_bytes = sum(m["bytes_sent"] for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)

        # Participation: clients with battery <= 0 drop out in FedAvg
        participations = [1.0 if s.battery_j > 0 else 0.0 for _, _, s in client_updates]
        jain = (sum(participations)**2 / (K * sum(p**2 for p in participations))
                if any(p > 0 for p in participations) else 0.0)

        return AggregateResult(
            new_weights=new_weights,
            metrics={
                "round": round_num, "total_bytes_sent": total_bytes,
                "total_energy_j": total_energy, "avg_beta": 1.0,
                "avg_battery_j": sum(s.battery_j for _,_,s in client_updates)/K,
                "avg_local_loss": sum(m["local_loss"] for _,m,_ in client_updates)/K,
                "participation_rate": sum(participations)/K,
                "jain_index": jain, "num_clients": K,
            }
        )

    def get_default_config(self):
        return {"lr": 0.01, "local_epochs": 1, "batch_size": 32,
                "device": "cpu", "device_profile": None}
