"""q-FedAvg solver for q-Fair Federated Learning (q-FFL)."""

from __future__ import annotations

import math

import torch

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .reference_utils import apply_delta, common_round_metrics, empirical_loss


@register_algorithm("qffl")
@register_algorithm("qfedavg")
class QFedAvg(FedAvg):
    """Communication-efficient q-FFL reference implementation.

    The client first evaluates ``F_k(w_t)``, performs ordinary local SGD to
    obtain ``w_bar_k``, and returns the Algorithm-2 quantities

    ``Delta_k = F_k(w_t)^q L (w_t - w_bar_k)`` and
    ``h_k = q F_k^(q-1)||L(w_t-w_bar_k)||^2 + L F_k^q``.

    At ``q=0`` the paper rule reduces to an equally weighted FedAvg update.
    Algorithm 2 samples clients according to the external probabilities ``p_k``;
    it does not multiply the selected clients by their dataset sizes a second
    time inside the q-FedAvg ratio.

    ``aggregation_prior='dataset_size'`` is retained as an explicitly named
    extension for ablations, but is deliberately not the default.
    Losses are clipped only for numerical stability here; q-FFL itself does
    not provide differential privacy.
    """

    description = "q-FedAvg for q-Fair Federated Learning (Li et al., ICLR 2020)."

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        loss = empirical_loss(
            model,
            dataloader,
            device,
            max_batches=config.get("loss_eval_max_batches"),
        )
        raw_delta, metadata = super().client_update(model, dataloader, state, config)
        q = float(config.get("q", 1.0))
        if q < 0.0:
            raise ValueError("q-FedAvg requires q >= 0")
        configured_lipschitz = config.get("lipschitz")
        lipschitz = (
            float(configured_lipschitz)
            if configured_lipschitz is not None
            else 1.0 / float(config.get("lr", 0.01))
        )
        if not math.isfinite(lipschitz) or lipschitz <= 0.0:
            raise ValueError("q-FedAvg requires a finite, strictly positive L")
        floor = float(config.get("loss_floor", 1e-8))
        ceiling = float(config.get("loss_ceiling", 1e6))
        if floor <= 0.0 or ceiling < floor:
            raise ValueError("Require 0 < loss_floor <= loss_ceiling")
        stable_loss = min(max(loss, floor), ceiling)

        scaled_local = {
            key: value.float() * lipschitz for key, value in raw_delta.items()
        }
        norm_sq = sum(
            float(value.double().square().sum().item())
            for key, value in scaled_local.items()
            if not key.endswith("num_batches_tracked")
        )
        loss_q = stable_loss**q
        numerator = {key: loss_q * value for key, value in scaled_local.items()}
        curvature = q * stable_loss ** (q - 1.0) * norm_sq + lipschitz * loss_q
        metadata.update(
            {
                "qffl_loss_at_global": loss,
                "qffl_h": max(float(curvature), floor),
                "qffl_update_norm": math.sqrt(norm_sq),
                "qffl_q": q,
            }
        )
        return numerator, metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        if not client_updates:
            raise ValueError("q-FedAvg requires at least one client update")
        # ``client_prior`` was the name used by the first reconstruction.  Read
        # it only as a compatibility alias, while exposing the scientifically
        # clearer ``aggregation_prior`` in all new configurations.
        prior_mode = str(
            config.get(
                "aggregation_prior",
                config.get("client_prior", "uniform"),
            )
        ).lower()
        if prior_mode == "uniform":
            # Algorithm 2 uses unweighted sums.  A common constant such as 1/n
            # would cancel between numerator and denominator, so unit weights
            # make the equation most visible in code.
            priors = [1.0] * len(client_updates)
        elif prior_mode == "dataset_size":
            priors = [
                float(meta.get("dataset_size", 1)) for _, meta, _ in client_updates
            ]
        else:
            raise ValueError(
                "aggregation_prior must be 'uniform' (q-FFL Algorithm 2) or "
                "'dataset_size' (explicit extension)"
            )
        denominator = sum(
            prior * meta["qffl_h"]
            for prior, (_, meta, _) in zip(priors, client_updates)
        )
        if not math.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError(f"Invalid q-FedAvg denominator: {denominator}")
        numerator = None
        for prior, (update, _, _) in zip(priors, client_updates):
            if numerator is None:
                numerator = {k: v.clone().float() * prior for k, v in update.items()}
            else:
                for key in numerator:
                    numerator[key] += update[key].float() * prior
        aggregate = {key: value / denominator for key, value in numerator.items()}
        metrics = common_round_metrics(client_updates)
        metrics.update(
            {
                "round": round_num,
                "qffl_q": float(config.get("q", 1.0)),
                "qffl_aggregation_prior": prior_mode,
                "qffl_denominator": float(denominator),
                "qffl_mean_loss_at_global": sum(
                    meta["qffl_loss_at_global"] for _, meta, _ in client_updates
                )
                / len(client_updates),
            }
        )
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "q": 1.0,
                # None follows the q-FedAvg implementation convention L=1/lr.
                # Set an explicit smoothness estimate when it is available.
                "lipschitz": None,
                "loss_floor": 1e-8,
                "loss_ceiling": 1e6,
                "loss_eval_max_batches": None,
                "aggregation_prior": "uniform",
                "client_metrics_every": 1,
            }
        )
        return cfg
