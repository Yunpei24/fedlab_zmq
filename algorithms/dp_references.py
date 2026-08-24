"""Common DP-SGD baselines for the internship reproduction.

These classes answer a narrow experimental need: FedAvg, q-FFL and FAR must
receive the *same* local sample-level DP-SGD mechanism before their server
rules are compared.  They are deliberately distinct from
``sc_partial_far_dp``, whose privacy unit is an entire client and whose noise
is added centrally after user-level clipping.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hardware.flop_cost import round_compute_flops
from privacy.local_dpsgd import local_dpsgd_train, private_mean_release
from privacy.rdp import (
    RDPAccountant,
    calibrate_composed_sampled_gaussian_noise,
    calibrate_sampled_gaussian_noise,
)

from .base import register_algorithm
from .far import FAR
from .fedavg import FedAvg
from .qffl import QFedAvg


def _dp_defaults() -> dict:
    return {
        "enable_dp": True,
        "clip_norm": 1.0,
        "noise_multiplier": 1.0,
        "target_epsilon": None,
        "delta": 1e-5,
        "privacy_num_rounds": None,
        "privacy_sampling_rate_override": None,
        "per_sample_backend": "vectorized",
        "dp_compute_multiplier": 2.5,
        "private_aux_loss": True,
        "aux_loss_clip": 2.5,
        "aux_loss_noise_multiplier": 5.0,
        "aux_loss_to_model_noise_ratio": 2.5,
        "target_epsilon_includes_auxiliary_channels": False,
        "loss_eval_max_batches": None,
        "client_metrics_every": 2,
    }


class _LocalDPSGDMixin:
    """Client-side implementation shared by all DP reference baselines."""

    def _resolved_noise_multiplier(self, dataloader, state, config) -> float:
        if not bool(config.get("enable_dp", True)):
            return 0.0
        configured = float(config.get("noise_multiplier", 1.0))
        target = config.get("target_epsilon")
        if target is None:
            return configured
        cache_key = "local_dp_calibrated_noise"
        if cache_key in state.custom:
            return float(state.custom[cache_key])
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError(
                "target_epsilon requires privacy_num_rounds for pre-training calibration"
            )
        dataset_size = max(len(dataloader.dataset), 1)
        batch_size = min(int(config.get("batch_size", 1)), dataset_size)
        q = config.get("privacy_sampling_rate_override")
        q = float(q) if q is not None else batch_size / dataset_size
        batches_per_epoch = math.ceil(dataset_size / batch_size)
        max_local_batches = config.get("max_local_batches")
        if max_local_batches is not None:
            batches_per_epoch = min(batches_per_epoch, int(max_local_batches))
        steps_per_round = int(config.get("local_epochs", 1)) * batches_per_epoch
        noise = calibrate_sampled_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            sampling_rate=q,
            steps=int(total_rounds) * steps_per_round,
        )
        state.custom[cache_key] = float(noise)
        return float(noise)

    def _local_dp_update(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        dataset_size = max(len(dataloader.dataset), 1)
        noise_multiplier = self._resolved_noise_multiplier(dataloader, state, config)
        dp_enabled = bool(config.get("enable_dp", True))
        update, stats = local_dpsgd_train(
            model,
            dataloader,
            device=device,
            lr=float(config.get("lr", 0.01)),
            local_epochs=int(config.get("local_epochs", 1)),
            clip_norm=float(config.get("clip_norm", 1.0)),
            noise_multiplier=noise_multiplier if dp_enabled else 0.0,
            backend=str(config.get("per_sample_backend", "vectorized")),
            momentum=float(config.get("momentum", 0.0)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            max_local_batches=config.get("max_local_batches"),
            proximal_mu=(
                float(config.get("far_prox_mu", config.get("mu", 0.0)))
                if isinstance(self, FAR)
                else 0.0
            ),
        )

        sampling_override = config.get("privacy_sampling_rate_override")
        sampling_rate = (
            float(sampling_override)
            if sampling_override is not None
            else min(1.0, int(config.get("batch_size", 1)) / dataset_size)
        )
        accountant = RDPAccountant.from_state_dict(state.custom.get("local_dp_accountant"))
        epsilon = best_order = None
        if dp_enabled:
            accountant.add_sampled_gaussian(
                channel="model",
                sampling_rate=sampling_rate,
                noise_multiplier=noise_multiplier,
                steps=stats.steps,
            )
            epsilon, best_order = accountant.epsilon(float(config.get("delta", 1e-5)))
            state.custom["local_dp_accountant"] = accountant.state_dict()

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
                int(config.get("local_epochs", 1)),
            ) * float(config.get("dp_compute_multiplier", 2.5))
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
            "local_loss": stats.mean_loss,
            "clip_rate": stats.clip_rate,
            "model_steps": stats.steps,
            "dp_noise_norm_mean": stats.mean_noise_norm,
            "privacy_epsilon": epsilon,
            "privacy_best_order": best_order,
            "privacy_delta": float(config.get("delta", 1e-5)) if dp_enabled else None,
            "privacy_sampling_rate": sampling_rate,
            "privacy_noise_multiplier": noise_multiplier if dp_enabled else None,
            "privacy_level": "sample",
            "privacy_trust_model": "local",
            "privacy_accounting_assumption": (
                "poisson_approximation_for_fixed_minibatches"
                if dp_enabled
                else "not_applicable"
            ),
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "energy_j_consumed": breakdown["total"],
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
            "far_update_mode": "multi_epoch_delta" if isinstance(self, FAR) else None,
            "far_prox_mu": (
                float(config.get("far_prox_mu", config.get("mu", 0.0)))
                if isinstance(self, FAR)
                else None
            ),
        }
        return update, metadata

    def client_update(self, model, dataloader, state, config):
        return self._local_dp_update(model, dataloader, state, config)

    def server_aggregate(self, global_model, client_updates, round_num, config):
        result = super().server_aggregate(
            global_model, client_updates, round_num, config
        )
        epsilons = [
            float(metadata["privacy_epsilon"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_epsilon") is not None
        ]
        clip_rates = [
            float(metadata["clip_rate"])
            for _, metadata, _ in client_updates
            if metadata.get("clip_rate") is not None
        ]
        noise_multipliers = [
            float(metadata["privacy_noise_multiplier"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_noise_multiplier") is not None
        ]
        model_steps = [
            int(metadata["model_steps"])
            for _, metadata, _ in client_updates
            if metadata.get("model_steps") is not None
        ]
        realised_noise_norms = [
            float(metadata["dp_noise_norm_mean"])
            for _, metadata, _ in client_updates
            if metadata.get("dp_noise_norm_mean") is not None
        ]
        result.metrics.update(
            {
                "privacy_epsilon_max": max(epsilons) if epsilons else None,
                "privacy_epsilon_mean": (
                    sum(epsilons) / len(epsilons) if epsilons else None
                ),
                "privacy_delta": float(config.get("delta", 1e-5)) if epsilons else None,
                "privacy_level": "sample",
                "privacy_trust_model": "local",
                "privacy_clip_rate_mean": (
                    sum(clip_rates) / len(clip_rates) if clip_rates else None
                ),
                "privacy_model_noise_multiplier_mean": (
                    sum(noise_multipliers) / len(noise_multipliers)
                    if noise_multipliers
                    else None
                ),
                "privacy_model_steps_mean": (
                    sum(model_steps) / len(model_steps) if model_steps else None
                ),
                "privacy_realised_noise_norm_mean_oracle": (
                    sum(realised_noise_norms) / len(realised_noise_norms)
                    if realised_noise_norms
                    else None
                ),
                "privacy_accounting_assumption": (
                    "poisson_approximation_for_fixed_minibatches"
                    if epsilons
                    else "not_applicable"
                ),
            }
        )
        return result


@register_algorithm("dpfedavg")
class DPFedAvg(_LocalDPSGDMixin, FedAvg):
    """FedAvg with common sample-level local DP-SGD."""

    description = "DP-FedAvg reference with per-example clipping and local Gaussian DP."

    def get_default_config(self):
        return {**FedAvg.get_default_config(self), **_dp_defaults()}


@register_algorithm("dpfar")
class DPFAR(_LocalDPSGDMixin, FAR):
    """FAR applied as post-processing to sample-level DP client updates."""

    description = "DP-FAR reference: local DP-SGD followed by FAR aggregation."

    def get_default_config(self):
        return {**FAR.get_default_config(self), **_dp_defaults()}


@register_algorithm("dpqffl")
class DPQFedAvg(_LocalDPSGDMixin, QFedAvg):
    """q-FFL with DP model and, by default, a private scalar loss channel."""

    description = "DP-q-FFL with shared local DP-SGD and an accounted loss channel."

    def _resolved_noise_multiplier(self, dataloader, state, config) -> float:
        target = config.get("target_epsilon")
        private_aux = bool(config.get("private_aux_loss", True)) and bool(
            config.get("enable_dp", True)
        )
        if target is None or not private_aux:
            return super()._resolved_noise_multiplier(dataloader, state, config)
        cache_key = "local_dp_calibrated_noise"
        if cache_key in state.custom:
            return float(state.custom[cache_key])
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError(
                "target_epsilon requires privacy_num_rounds for joint calibration"
            )
        dataset_size = max(len(dataloader.dataset), 1)
        batch_size = min(int(config.get("batch_size", 1)), dataset_size)
        q_override = config.get("privacy_sampling_rate_override")
        q_model = float(q_override) if q_override is not None else batch_size / dataset_size
        batches_per_epoch = math.ceil(dataset_size / batch_size)
        max_local_batches = config.get("max_local_batches")
        if max_local_batches is not None:
            batches_per_epoch = min(batches_per_epoch, int(max_local_batches))
        model_steps = (
            int(total_rounds)
            * int(config.get("local_epochs", 1))
            * batches_per_epoch
        )
        ratio = float(config.get("aux_loss_to_model_noise_ratio", 2.5))
        model_noise = calibrate_composed_sampled_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            channels=((q_model, model_steps, 1.0), (1.0, int(total_rounds), ratio)),
        )
        state.custom[cache_key] = float(model_noise)
        state.custom["local_dp_calibrated_aux_loss_noise"] = float(
            model_noise * ratio
        )
        return float(model_noise)

    def _loss_values(self, model, dataloader, config) -> list[float]:
        device = str(config.get("device", "cpu"))
        was_training = model.training
        model.eval()
        values: list[float] = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                limit = config.get("loss_eval_max_batches")
                if limit is not None and batch_idx >= int(limit):
                    break
                logits = model(x.to(device))
                losses = F.cross_entropy(logits, y.to(device), reduction="none")
                values.extend(float(value) for value in losses.cpu())
        model.train(was_training)
        return values

    def client_update(self, model, dataloader, state, config):
        # Resolve the composed model+loss calibration before either release.
        self._resolved_noise_multiplier(dataloader, state, config)
        values = self._loss_values(model, dataloader, config)
        raw_loss = sum(values) / max(len(values), 1)
        private_aux = bool(config.get("private_aux_loss", True)) and bool(
            config.get("enable_dp", True)
        )
        if private_aux:
            aux_noise = float(
                state.custom.get(
                    "local_dp_calibrated_aux_loss_noise",
                    config.get("aux_loss_noise_multiplier", 5.0),
                )
            )
            released_loss = private_mean_release(
                values,
                clip=float(config.get("aux_loss_clip", 2.5)),
                noise_multiplier=aux_noise,
                device=str(config.get("device", "cpu")),
            )
        else:
            aux_noise = 0.0
            released_loss = raw_loss

        raw_delta, metadata = self._local_dp_update(model, dataloader, state, config)
        if private_aux:
            accountant = RDPAccountant.from_state_dict(state.custom.get("local_dp_accountant"))
            accountant.add_sampled_gaussian(
                channel="loss",
                sampling_rate=1.0,
                noise_multiplier=aux_noise,
                steps=1,
            )
            epsilon, best_order = accountant.epsilon(float(config.get("delta", 1e-5)))
            state.custom["local_dp_accountant"] = accountant.state_dict()
            metadata["privacy_epsilon"] = epsilon
            metadata["privacy_best_order"] = best_order

        q = float(config.get("q", 1.0))
        lipschitz_cfg = config.get("lipschitz")
        lipschitz = (
            float(lipschitz_cfg)
            if lipschitz_cfg is not None
            else 1.0 / float(config.get("lr", 0.01))
        )
        floor = float(config.get("loss_floor", 1e-8))
        ceiling = float(config.get("loss_ceiling", 1e6))
        stable_loss = min(max(released_loss, floor), ceiling)
        scaled = {key: value.float() * lipschitz for key, value in raw_delta.items()}
        norm_sq = sum(
            float(value.double().square().sum().item())
            for key, value in scaled.items()
            if not key.endswith("num_batches_tracked")
        )
        loss_q = stable_loss**q
        numerator = {key: loss_q * value for key, value in scaled.items()}
        curvature = q * stable_loss ** (q - 1.0) * norm_sq + lipschitz * loss_q
        metadata.update(
            {
                "qffl_loss_at_global": float(released_loss),
                "qffl_raw_loss_oracle": float(raw_loss),
                "qffl_loss_channel_private": private_aux,
                "privacy_aux_loss_noise_multiplier": aux_noise if private_aux else None,
                "qffl_h": max(float(curvature), floor),
                "qffl_update_norm": math.sqrt(norm_sq),
                "qffl_q": q,
            }
        )
        return numerator, metadata

    def get_default_config(self):
        return {
            **QFedAvg.get_default_config(self),
            **_dp_defaults(),
            "target_epsilon_includes_auxiliary_channels": True,
        }
