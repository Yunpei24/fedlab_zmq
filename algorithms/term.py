"""Client-level Tilted Empirical Risk Minimization for federated learning."""

from __future__ import annotations

import torch

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .reference_utils import apply_delta, common_round_metrics, empirical_loss


@register_algorithm("term")
class FederatedTERM(FedAvg):
    """Group-level TERM baseline using client losses as the tilted groups.

    Each client performs ordinary local training.  The server gives update
    ``i`` the stable softmax weight ``exp(t L_i) / sum_j exp(t L_j)``.
    Positive ``t`` emphasizes high-loss clients (fairness), negative ``t``
    attenuates them (outlier robustness), and ``t=0`` is uniform FedAvg.

    The client loss is visible to the server in this non-private reference.
    It must not be described as a private side channel.
    """

    description = "Client-level TERM with stable exponential loss tilting."

    def client_update(self, model, dataloader, state, config):
        profile_loader = config.get("anchor_dataloader") or dataloader
        loss = empirical_loss(
            model,
            profile_loader,
            config.get("device", "cpu"),
            max_batches=config.get("loss_eval_max_batches"),
        )
        update, metadata = super().client_update(model, dataloader, state, config)
        metadata["term_loss_at_global"] = loss
        return update, metadata

    @staticmethod
    def client_weights(
        losses: torch.Tensor, tilt: float, priors: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Stable TERM weights, exposed for tests and ablations."""

        if priors is None:
            priors = torch.full_like(losses, 1.0 / losses.numel())
        priors = priors / priors.sum()
        return torch.softmax(
            float(tilt) * losses + priors.clamp_min(1e-15).log(), dim=0
        )

    def server_aggregate(self, global_model, client_updates, round_num, config):
        losses = torch.tensor(
            [meta["term_loss_at_global"] for _, meta, _ in client_updates],
            dtype=torch.float64,
        )
        if config.get("client_prior", "uniform") == "dataset_size":
            priors = torch.tensor(
                [meta.get("dataset_size", 1) for _, meta, _ in client_updates],
                dtype=torch.float64,
            )
        else:
            priors = torch.ones_like(losses)
        weights = self.client_weights(losses, float(config.get("tilt", 0.1)), priors)
        aggregate = None
        for weight, (update, _, _) in zip(weights, client_updates):
            if aggregate is None:
                aggregate = {k: v.clone().float() * weight for k, v in update.items()}
            else:
                for key in aggregate:
                    aggregate[key] += update[key].float() * weight
        entropy = -(weights * weights.clamp_min(1e-15).log()).sum()
        metrics = common_round_metrics(client_updates)
        metrics.update(
            {
                "round": round_num,
                "term_tilt": float(config.get("tilt", 0.1)),
                "term_max_weight": float(weights.max().item()),
                "term_weight_entropy": float(entropy.item()),
                "term_mean_loss_at_global": float(losses.mean().item()),
            }
        )
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "tilt": 0.1,
                "loss_eval_max_batches": None,
                "client_metrics_every": 1,
                "client_prior": "uniform",
            }
        )
        return cfg
