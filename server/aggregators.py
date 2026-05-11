"""
server/aggregators.py
=====================
Server-side aggregation strategies for FedLab ZMQ.
"""

from collections import OrderedDict
import torch


def fedavg_aggregate(
    global_sd: dict,
    client_updates: list[tuple[dict, float, dict]],
) -> dict:
    """
    Weighted FedAvg aggregation.
    w_{t+1} = w_t - sum_k( weight_k * update_k )

    Args:
        global_sd: current global model state_dict
        client_updates: list of (update_dict, weight, metadata)
    """
    agg = None
    total_weight = sum(w for _, w, _ in client_updates)

    for update, weight, _ in client_updates:
        norm_w = weight / max(total_weight, 1e-9)
        if agg is None:
            agg = {k: v.float() * norm_w for k, v in update.items()}
        else:
            for k in agg:
                agg[k] += update[k].float() * norm_w

    new_sd = OrderedDict()
    for k in global_sd:
        if k in agg:
            new_sd[k] = global_sd[k].float() - agg[k]
        else:
            new_sd[k] = global_sd[k].float()
    return new_sd


def uniform_aggregate(
    global_sd: dict,
    client_updates: list[tuple[dict, float, dict]],
) -> dict:
    """Equal-weight aggregation regardless of dataset sizes."""
    K = len(client_updates)
    uniform = [(upd, 1.0 / K, meta) for upd, _, meta in client_updates]
    return fedavg_aggregate(global_sd, uniform)
