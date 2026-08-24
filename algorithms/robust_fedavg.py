"""Standalone Byzantine-robust aggregation baselines.

The local training step is deliberately identical to :class:`FedAvg`.  Only
the server aggregation rule changes.  Keeping this separation makes an
ablation interpretable: a difference with FAR comes from the aggregation
rule, not from a different optimizer or client workload.
"""

from __future__ import annotations

import torch

from robustness.aggregators import aggregate_vectors
from robustness.tensor_ops import stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .reference_utils import apply_delta, common_round_metrics


@register_algorithm("robustfedavg")
@register_algorithm("robust_aggregate")
class RobustFedAvg(FedAvg):
    """FedAvg local SGD followed by a configurable robust server rule."""

    description = "FedAvg client training with CM(NNM), trMean(NNM), NBS, RFA or CMLS."

    def server_aggregate(self, global_model, client_updates, round_num, config):
        vectors, layout = stack_updates([update for update, _, _ in client_updates])
        method = str(config.get("robust_aggregator", "cm_nnm"))
        result = aggregate_vectors(
            vectors,
            method,
            num_byzantine=int(config.get("num_byzantine", 0)),
            screening_fraction=config.get("screening_fraction"),
            max_iter=int(config.get("rfa_max_iter", 100)),
            tol=float(config.get("rfa_tol", 1e-6)),
            smoothing=float(config.get("rfa_smoothing", 1e-8)),
            alpha_trusted=float(config.get("cmls_alpha_trusted", 1.0)),
            alpha_suspected=float(config.get("cmls_alpha_suspected", 1.0)),
        )
        aggregate = dict(unflatten_update(result, layout))
        metrics = common_round_metrics(client_updates)
        malicious = sum(
            bool(metadata.get("is_byzantine", False))
            for _, metadata, _ in client_updates
        )
        metrics.update(
            {
                "round": round_num,
                "robust_aggregator": method,
                "robust_aggregate_norm": float(torch.linalg.vector_norm(result).item()),
                "num_byzantine_oracle": int(malicious),
            }
        )
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "robust_aggregator": "cm_nnm",
                "num_byzantine": 0,
                "screening_fraction": None,
                "rfa_max_iter": 100,
                "rfa_tol": 1e-6,
                "rfa_smoothing": 1e-8,
                "cmls_alpha_trusted": 1.0,
                "cmls_alpha_suspected": 1.0,
                "client_metrics_every": 1,
            }
        )
        return cfg
