"""Fairness-Aware Reweighting (FAR) around a robust reference aggregator."""

from __future__ import annotations

import gc
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops
from metrics.robustness import weight_diagnostics
from robustness.aggregators import aggregate_vectors
from robustness.tensor_ops import stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .reference_utils import apply_delta, common_round_metrics


@register_algorithm("far")
class FAR(FedAvg):
    """FAR reference implementation from the supplied manuscript.

    A configurable Byzantine-robust rule ``F`` first yields ``g_F``.  FAR then
    computes ``d_i = ||g_i-g_F||`` and positive-tilt weights

    ``lambda_i = softmax(alpha * d_i)``.

    The final output is the weighted sum of the *original* submissions, not
    the reference itself.  Therefore FAR's robustness depends on its attack
    regime and ``alpha``; a positive tilt is not a universal defense.
    """

    description = "FAR: distance-to-robust-reference exponential reweighting."

    _UPDATE_MODES = {"multi_epoch_delta", "single_step_gradient"}

    @staticmethod
    def _prox_mu(config: dict) -> float:
        """Resolve FAR's proximal coefficient while accepting the short alias ``mu``."""

        value = float(config.get("far_prox_mu", config.get("mu", 0.0)))
        if value < 0:
            raise ValueError("far_prox_mu must be non-negative")
        return value

    def client_update(self, model, dataloader, state, config):
        """Produce either a proximal local delta or one empirical gradient.

        ``multi_epoch_delta`` is the practical FedAvg/FedProx-style mode used
        by the internship protocol.  ``single_step_gradient`` evaluates one
        full local empirical gradient at the received global model and leaves
        the server step size explicit.  At that anchor the proximal gradient
        is mathematically zero; ``mu`` only affects trajectories containing
        more than one local optimisation step.
        """

        mode = str(config.get("far_update_mode", "multi_epoch_delta")).lower()
        if mode not in self._UPDATE_MODES:
            raise ValueError(
                f"far_update_mode must be one of {sorted(self._UPDATE_MODES)}, got {mode!r}"
            )
        if mode == "single_step_gradient":
            return self._single_step_gradient(model, dataloader, state, config)
        return self._multi_epoch_prox_delta(model, dataloader, state, config)

    def _multi_epoch_prox_delta(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        lr = float(config.get("lr", 0.01))
        local_epochs = int(config.get("local_epochs", 1))
        mu = self._prox_mu(config)
        max_grad_norm = config.get("max_grad_norm")

        before = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in model.state_dict().items()
        )
        anchor = {
            name: parameter.detach().to(device).clone()
            for name, parameter in model.named_parameters()
        }
        model.to(device).train()
        optimizer_type = str(config.get("optimizer", "sgd")).lower()
        momentum = float(config.get("momentum", 0.9))
        weight_decay = float(config.get("weight_decay", 1e-4))
        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        criterion = nn.CrossEntropyLoss()
        total_task_loss = 0.0
        total_objective = 0.0
        num_batches = 0
        max_local_batches = config.get("max_local_batches")
        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                if max_local_batches is not None and batch_idx >= int(max_local_batches):
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                task_loss = criterion(model(x), y)
                prox_sq = torch.zeros((), device=device)
                if mu > 0:
                    for name, parameter in model.named_parameters():
                        prox_sq = prox_sq + (parameter - anchor[name]).square().sum()
                objective = task_loss + 0.5 * mu * prox_sq
                objective.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                optimizer.step()
                total_task_loss += float(task_loss.detach().item())
                total_objective += float(objective.detach().item())
                num_batches += 1

        current = model.state_dict()
        update = OrderedDict(
            (name, (before[name] - current[name].detach().cpu()).float())
            for name in before
        )
        metadata = self._finalize_client_metadata(
            model=model,
            dataloader=dataloader,
            state=state,
            config=config,
            update=update,
            local_epochs=local_epochs,
            local_loss=total_task_loss / max(num_batches, 1),
            extra={
                "far_update_mode": "multi_epoch_delta",
                "far_prox_mu": mu,
                "far_prox_active": bool(mu > 0 and num_batches > 1),
                "far_local_objective": total_objective / max(num_batches, 1),
                "far_local_steps": num_batches,
            },
        )
        del optimizer, anchor, before, current
        gc.collect()
        return dict(update), metadata

    def _single_step_gradient(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        mu = self._prox_mu(config)
        model.to(device).train()
        parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradient_sums = {name: torch.zeros_like(p, device=device) for name, p in parameters}
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_examples = 0
        num_batches = 0
        max_local_batches = config.get("max_local_batches")
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_local_batches is not None and batch_idx >= int(max_local_batches):
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            loss_sum = criterion(model(x), y)
            loss_sum.backward()
            for name, parameter in parameters:
                gradient_sums[name].add_(parameter.grad.detach())
            total_loss += float(loss_sum.detach().item())
            total_examples += int(y.numel())
            num_batches += 1

        # FAR's paper notation aggregates gradients and applies one server
        # step. Buffers have no gradient, so they receive explicit zeros.
        update = OrderedDict()
        parameter_names = {name for name, _ in parameters}
        for name, value in model.state_dict().items():
            if name in parameter_names:
                update[name] = (gradient_sums[name] / max(total_examples, 1)).cpu().float()
            else:
                update[name] = torch.zeros_like(value, device="cpu").float()
        metadata = self._finalize_client_metadata(
            model=model,
            dataloader=dataloader,
            state=state,
            config=config,
            update=update,
            local_epochs=1,
            local_loss=total_loss / max(total_examples, 1),
            extra={
                "far_update_mode": "single_step_gradient",
                "far_prox_mu": mu,
                "far_prox_active": False,
                "far_prox_note": "zero_at_global_anchor_for_one_gradient_evaluation",
                "far_local_steps": 1,
                "far_gradient_batches": num_batches,
            },
        )
        return dict(update), metadata

    def _finalize_client_metadata(
        self, *, model, dataloader, state, config, update, local_epochs, local_loss, extra
    ):
        uplink_bytes = self.count_bytes(update, sparse=False)
        downlink_bytes = uplink_bytes
        profile = config.get("device_profile")
        if profile:
            flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                local_epochs,
            )
            breakdown = profile.round_energy_breakdown(
                flops,
                uplink_bytes,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            energy = 2.5 * float(config.get("energy_scale_factor", 1.0))
            breakdown = {"compute": energy, "uplink": 0.0, "downlink": 0.0, "total": energy}
        state.battery_j = max(0.0, state.battery_j - breakdown["total"])
        state.round_num += 1
        return {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": breakdown["total"],
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": float(local_loss),
            "compression_ratio": 1.0,
            "dataset_size": len(dataloader.dataset),
            **extra,
        }

    @staticmethod
    def far_weights(distances: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.softmax(float(alpha) * distances, dim=0)

    @staticmethod
    def _pearson_or_none(x: torch.Tensor, y: torch.Tensor) -> float | None:
        """Small aggregate-only diagnostic; individual values are not persisted."""

        if x.numel() < 2 or y.numel() != x.numel():
            return None
        x_centered = x.double() - x.double().mean()
        y_centered = y.double() - y.double().mean()
        denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(
            y_centered
        )
        if float(denominator.item()) <= 1e-15:
            return None
        return float((x_centered @ y_centered / denominator).item())

    def server_aggregate(self, global_model, client_updates, round_num, config):
        updates = [update for update, _, _ in client_updates]
        vectors, layout = stack_updates(updates)
        reference_name = str(config.get("robust_reference", "cm_nnm"))
        reference = aggregate_vectors(
            vectors,
            reference_name,
            num_byzantine=int(config.get("num_byzantine", 0)),
            screening_fraction=config.get("screening_fraction"),
            max_iter=int(config.get("rfa_max_iter", 100)),
            tol=float(config.get("rfa_tol", 1e-6)),
            alpha_trusted=float(config.get("cmls_alpha_trusted", 1.0)),
            alpha_suspected=float(config.get("cmls_alpha_suspected", 1.0)),
        )
        distances = torch.linalg.vector_norm(vectors - reference, dim=1)
        far_alpha = float(config.get("far_alpha", 0.1))
        weights = self.far_weights(distances, far_alpha)
        aggregate_vector = (weights[:, None] * vectors).sum(dim=0)
        aggregate = dict(unflatten_update(aggregate_vector, layout))
        modes = {
            str(metadata.get("far_update_mode", config.get("far_update_mode", "multi_epoch_delta")))
            for _, metadata, _ in client_updates
        }
        if len(modes) != 1:
            raise ValueError(f"FAR received mixed client update modes: {sorted(modes)}")
        update_mode = modes.pop()
        if update_mode == "single_step_gradient":
            configured_server_lr = config.get("far_server_lr")
            server_lr = float(
                config.get("lr", 0.01)
                if configured_server_lr is None
                else configured_server_lr
            )
            aggregate = {name: server_lr * value for name, value in aggregate.items()}
        else:
            server_lr = 1.0

        malicious_mask = torch.tensor(
            [bool(meta.get("is_byzantine", False)) for _, meta, _ in client_updates]
        )
        diagnostics = weight_diagnostics(weights, malicious_mask)
        diagnostics.update(
            {
                "far_mean_distance": float(distances.mean().item()),
                "far_max_distance": float(distances.max().item()),
                "far_reference_norm": float(torch.linalg.vector_norm(reference).item()),
                "far_alpha": far_alpha,
                # The softmax ratio between the largest and smallest weight is
                # exp(logit_range).  This directly exposes the amplification
                # that motivates sensitivity-controlled FAR.
                "far_logit_range": float(
                    (far_alpha * (distances.max() - distances.min())).item()
                ),
                "far_weight_ratio": float(
                    (weights.max() / weights.clamp_min(1e-15).min()).item()
                ),
                "far_num_byzantine_oracle": int(malicious_mask.sum().item()),
                "far_update_mode": update_mode,
                "far_server_lr": server_lr,
                "far_prox_mu": self._prox_mu(config),
            }
        )
        # Experimental diagnostic for the report's hypothesis that FAR may
        # amplify clients with larger realised DP perturbations.  The exact
        # client-level noise norms are simulation oracles and are never saved;
        # only the two cohort-level correlations are recorded.
        noise_norms = [
            metadata.get("dp_noise_norm_mean")
            for _, metadata, _ in client_updates
        ]
        if all(value is not None for value in noise_norms):
            noise_tensor = torch.tensor(noise_norms, dtype=torch.float64)
            diagnostics["far_weight_dp_noise_corr_oracle"] = self._pearson_or_none(
                weights, noise_tensor
            )
            diagnostics["far_distance_dp_noise_corr_oracle"] = self._pearson_or_none(
                distances, noise_tensor
            )
        metrics = common_round_metrics(client_updates)
        metrics.update({"round": round_num, "robust_reference": reference_name})
        metrics.update(diagnostics)
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "far_alpha": 0.1,
                "far_update_mode": "multi_epoch_delta",
                "far_prox_mu": 0.0,
                "far_server_lr": None,
                "robust_reference": "cm_nnm",
                "num_byzantine": 0,
                "screening_fraction": None,
                "rfa_max_iter": 100,
                "rfa_tol": 1e-6,
                "client_metrics_every": 1,
            }
        )
        return cfg
