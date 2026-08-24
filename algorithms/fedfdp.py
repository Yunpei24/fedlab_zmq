"""FedFair and FedFDP with explicit per-example clipping.

This implementation favours auditability over speed: per-example gradients
are computed in a small loop, making the fair-clipping equation visible in
code.  It is suitable for reference experiments and unit tests.  A vectorised
``torch.func.vmap`` backend can be added later without changing the contract.
"""

from __future__ import annotations

import gc
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops
from privacy.rdp import RDPAccountant, calibrate_composed_sampled_gaussian_noise

from .base import AggregateResult, FLAlgorithm, register_algorithm
from .reference_utils import common_round_metrics


class _FedFairCore(FLAlgorithm):
    """Shared client implementation for non-private FedFair and FedFDP."""

    def _private_enabled(self, config: dict) -> bool:
        return bool(config.get("enable_dp", True))

    def _resolved_noise_multipliers(self, dataloader, state, config):
        """Return model/loss multipliers, jointly calibrated when requested."""

        configured_model = float(config.get("noise_multiplier", 2.0))
        configured_loss = float(config.get("loss_noise_multiplier", 5.0))
        target = config.get("target_epsilon")
        if target is None or not self._private_enabled(config):
            return configured_model, configured_loss
        cache_key = "fedfdp_calibrated_noise_pair"
        if cache_key in state.custom:
            pair = state.custom[cache_key]
            return float(pair[0]), float(pair[1])
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError(
                "target_epsilon requires privacy_num_rounds for model+loss calibration"
            )
        dataset_size = max(len(dataloader.dataset), 1)
        batch_size = min(int(config.get("batch_size", 1)), dataset_size)
        q_override = config.get("privacy_sampling_rate_override")
        sampling_rate = (
            float(q_override) if q_override is not None else batch_size / dataset_size
        )
        batches_per_epoch = math.ceil(dataset_size / batch_size)
        max_local_batches = config.get("max_local_batches")
        if max_local_batches is not None:
            batches_per_epoch = min(batches_per_epoch, int(max_local_batches))
        model_steps = (
            int(total_rounds)
            * int(config.get("local_epochs", 1))
            * batches_per_epoch
        )
        ratio = float(config.get("loss_to_model_noise_ratio", 2.5))
        model_noise = calibrate_composed_sampled_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            channels=(
                (sampling_rate, model_steps, 1.0),
                (sampling_rate, int(total_rounds), ratio),
            ),
        )
        pair = (float(model_noise), float(model_noise * ratio))
        state.custom[cache_key] = pair
        return pair

    def client_update(self, model, dataloader, state, config):
        device = config.get("device", "cpu")
        model.to(device)
        model.train()
        before = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = optim.SGD(
            parameters,
            lr=float(config.get("lr", 0.01)),
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

        fairness_lambda = float(config.get("fairness_lambda", 0.1))
        clip_norm = float(config.get("clip_norm", 1.0))
        max_fair_scale = float(config.get("max_fair_scale", 10.0))
        noise_model, noise_loss = self._resolved_noise_multipliers(
            dataloader, state, config
        )
        local_epochs = int(config.get("local_epochs", 1))
        global_loss = float(
            config.get("fedfdp_global_loss", config.get("initial_global_loss", 1.0))
        )
        dp_enabled = self._private_enabled(config)

        total_loss = 0.0
        total_examples = 0
        model_steps = 0
        clipped_examples = 0
        last_losses = None
        last_batch_size = 0
        max_local_batches = config.get("max_local_batches")

        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                if max_local_batches is not None and batch_idx >= int(
                    max_local_batches
                ):
                    break
                x, y = x.to(device), y.to(device)
                summed = [torch.zeros_like(parameter) for parameter in parameters]
                sample_losses = []
                for sample_x, sample_y in zip(x, y):
                    logits = model(sample_x.unsqueeze(0))
                    loss = nn.functional.cross_entropy(logits, sample_y.unsqueeze(0))
                    grads = torch.autograd.grad(loss, parameters, retain_graph=False)
                    # MPS does not implement float64 tensors.  The gradients
                    # already use the model's training dtype (normally
                    # float32), which is sufficient for the clipping norm and
                    # avoids a backend-specific cast that used to stop the
                    # faithful FedFair/FedFDP campaign on Apple Silicon.
                    grad_norm = torch.sqrt(
                        sum(grad.detach().float().square().sum() for grad in grads)
                    ).item()
                    fair_scale = 1.0 + fairness_lambda * (
                        float(loss.item()) - global_loss
                    )
                    fair_scale = min(max(fair_scale, 0.0), max_fair_scale)
                    clip_scale = clip_norm / max(grad_norm, 1e-12)
                    scale = min(fair_scale, clip_scale)
                    clipped_examples += int(scale < fair_scale)
                    for accumulator, grad in zip(summed, grads):
                        accumulator.add_(grad.detach(), alpha=scale)
                    sample_losses.append(float(loss.item()))
                    total_loss += float(loss.item())
                    total_examples += 1

                optimizer.zero_grad(set_to_none=True)
                batch_size = max(len(sample_losses), 1)
                for parameter, accumulator in zip(parameters, summed):
                    if dp_enabled:
                        accumulator = accumulator + torch.randn_like(accumulator) * (
                            noise_model * clip_norm
                        )
                    parameter.grad = accumulator / batch_size
                optimizer.step()
                model_steps += 1
                last_losses = sample_losses
                last_batch_size = batch_size

        # A second, scalar release is needed because FedFDP uses a private loss
        # to construct the next round's global fairness reference.
        loss_release = total_loss / max(total_examples, 1)
        loss_clip = float(config.get("loss_clip", 5.0))
        if config.get("adaptive_loss_clip", True):
            loss_clip = max(float(config.get("min_loss_clip", 1e-3)), abs(global_loss))
        if last_losses:
            clipped_loss_sum = sum(
                min(max(value, 0.0), loss_clip) for value in last_losses
            )
            noise = (
                torch.randn((), device=device).item() * noise_loss * loss_clip
                if dp_enabled
                else 0.0
            )
            loss_release = (clipped_loss_sum + noise) / max(last_batch_size, 1)

        current = model.state_dict()
        delta = OrderedDict(
            (key, (before[key] - current[key].detach().cpu()).float()) for key in before
        )

        # Per-client cumulative accountant. Only actually released channels
        # are charged: model_steps local gradient releases and one loss release.
        accountant = RDPAccountant.from_state_dict(
            state.custom.get("fedfdp_accountant")
        )
        dataset_size = max(len(dataloader.dataset), 1)
        sampling_rate = min(
            1.0, float(config.get("batch_size", last_batch_size)) / dataset_size
        )
        epsilon = best_order = None
        if dp_enabled:
            accountant.add_sampled_gaussian(
                channel="model",
                sampling_rate=sampling_rate,
                noise_multiplier=noise_model,
                steps=model_steps,
            )
            accountant.add_sampled_gaussian(
                channel="loss",
                sampling_rate=sampling_rate,
                noise_multiplier=noise_loss,
                steps=1,
            )
            epsilon, best_order = accountant.epsilon(float(config.get("delta", 1e-5)))
            state.custom["fedfdp_accountant"] = accountant.state_dict()

        uplink_bytes = self.count_bytes(delta, sparse=False) + 8
        downlink_bytes = self.count_bytes(delta, sparse=False) + 8
        profile = config.get("device_profile")
        if profile:
            flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                local_epochs,
            ) * float(config.get("dp_compute_multiplier", 1.0))
            breakdown = profile.round_energy_breakdown(
                flops,
                uplink_bytes,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            energy = 2.5 * float(config.get("energy_scale_factor", 1.0))
            breakdown = {
                "compute": energy,
                "uplink": 0.0,
                "downlink": 0.0,
                "total": energy,
            }
        state.battery_j = max(0.0, state.battery_j - breakdown["total"])
        state.round_num += 1

        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "dataset_size": dataset_size,
            "local_loss": total_loss / max(total_examples, 1),
            "private_loss_release": float(loss_release),
            "loss_clip": loss_clip,
            "clip_rate": clipped_examples / max(total_examples, 1),
            "model_steps": model_steps,
            "sampling_rate": sampling_rate,
            "privacy_epsilon": epsilon,
            "privacy_best_order": best_order,
            "privacy_delta": float(config.get("delta", 1e-5)) if dp_enabled else None,
            "privacy_accounting_assumption": (
                "poisson_approximation_for_fixed_minibatches"
                if dp_enabled
                else "not_applicable"
            ),
            "privacy_model_noise_multiplier": noise_model if dp_enabled else None,
            "privacy_loss_noise_multiplier": noise_loss if dp_enabled else None,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "energy_j_consumed": breakdown["total"],
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
        }
        del optimizer, before, current
        gc.collect()
        return dict(delta), metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        sizes = [meta.get("dataset_size", 1) for _, meta, _ in client_updates]
        total_size = max(sum(sizes), 1)
        aggregate = None
        for (update, _, _), size in zip(client_updates, sizes):
            weight = size / total_size
            if aggregate is None:
                aggregate = {
                    key: value.clone().float() * weight for key, value in update.items()
                }
            else:
                for key in aggregate:
                    aggregate[key] += update[key].float() * weight
        global_sd = global_model.state_dict()
        new_weights = OrderedDict(
            (
                key,
                global_sd[key].float() - aggregate[key].to(global_sd[key].device),
            )
            for key in global_sd
        )
        next_global_loss = sum(
            (size / total_size) * meta["private_loss_release"]
            for (_, meta, _), size in zip(client_updates, sizes)
        )
        metrics = common_round_metrics(client_updates)
        epsilons = [
            meta["privacy_epsilon"]
            for _, meta, _ in client_updates
            if meta.get("privacy_epsilon") is not None
        ]
        best_orders = [
            float(meta["privacy_best_order"])
            for _, meta, _ in client_updates
            if meta.get("privacy_best_order") is not None
        ]
        sampling_rates = [
            float(meta["sampling_rate"])
            for _, meta, _ in client_updates
            if meta.get("sampling_rate") is not None
        ]
        model_steps = [
            float(meta["model_steps"])
            for _, meta, _ in client_updates
            if meta.get("model_steps") is not None
        ]
        loss_clips = [
            float(meta["loss_clip"])
            for _, meta, _ in client_updates
            if meta.get("loss_clip") is not None
        ]
        model_noises = [
            float(meta["privacy_model_noise_multiplier"])
            for _, meta, _ in client_updates
            if meta.get("privacy_model_noise_multiplier") is not None
        ]
        loss_noises = [
            float(meta["privacy_loss_noise_multiplier"])
            for _, meta, _ in client_updates
            if meta.get("privacy_loss_noise_multiplier") is not None
        ]
        metrics.update(
            {
                "round": round_num,
                "fedfdp_global_private_loss": float(next_global_loss),
                "fedfdp_clip_rate": sum(
                    meta["clip_rate"] for _, meta, _ in client_updates
                )
                / len(client_updates),
                "privacy_epsilon_max": max(epsilons) if epsilons else None,
                "privacy_epsilon_mean": (
                    sum(epsilons) / len(epsilons) if epsilons else None
                ),
                "privacy_delta": (
                    float(config.get("delta", 1e-5)) if epsilons else None
                ),
                "privacy_best_order_mean": (
                    sum(best_orders) / len(best_orders) if best_orders else None
                ),
                "privacy_sampling_rate_mean": (
                    sum(sampling_rates) / len(sampling_rates)
                    if sampling_rates
                    else None
                ),
                "privacy_model_steps_mean": (
                    sum(model_steps) / len(model_steps) if model_steps else None
                ),
                "privacy_model_noise_multiplier": (
                    sum(model_noises) / len(model_noises) if model_noises else None
                ),
                "privacy_loss_noise_multiplier": (
                    sum(loss_noises) / len(loss_noises) if loss_noises else None
                ),
                "fedfdp_loss_clip_mean": (
                    sum(loss_clips) / len(loss_clips) if loss_clips else None
                ),
                "privacy_accounting_assumption": (
                    "poisson_approximation_for_fixed_minibatches"
                    if epsilons
                    else "not_applicable"
                ),
                "_server_state_updates": {
                    "fedfdp_global_loss": float(next_global_loss)
                },
            }
        )
        return AggregateResult(new_weights, metrics)

    def get_default_config(self):
        return {
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "local_epochs": 1,
            "batch_size": 32,
            "device": "cpu",
            "fairness_lambda": 0.1,
            "clip_norm": 1.0,
            "max_fair_scale": 10.0,
            "noise_multiplier": 2.0,
            "loss_clip": 5.0,
            "adaptive_loss_clip": True,
            "min_loss_clip": 1e-3,
            "loss_noise_multiplier": 5.0,
            "loss_to_model_noise_ratio": 2.5,
            "target_epsilon": None,
            "privacy_num_rounds": None,
            "privacy_sampling_rate_override": None,
            "target_epsilon_includes_auxiliary_channels": True,
            "initial_global_loss": 1.0,
            "delta": 1e-5,
            "enable_dp": True,
            "dp_compute_multiplier": 1.0,
            "client_metrics_every": 1,
            # Optional diagnostic limit. Leave as None for real experiments.
            "max_local_batches": None,
        }


@register_algorithm("fedfdp")
class FedFDP(_FedFairCore):
    """FedFDP sample-level local DP with private model and loss channels."""

    description = "FedFDP fair clipping + Gaussian model/loss channels + RDP."


@register_algorithm("fedfair")
class FedFair(_FedFairCore):
    """Non-private fair-clipping ablation of FedFDP."""

    description = "FedFair fair-clipping ablation without Gaussian noise."

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "enable_dp": False,
                "noise_multiplier": 0.0,
                "loss_noise_multiplier": 0.0,
            }
        )
        return cfg
