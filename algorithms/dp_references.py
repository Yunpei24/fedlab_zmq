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
from privacy.local_dpsgd import (
    local_dpsgd_train,
    private_gradient_release_fixed_without_replacement,
    private_mean_release,
)
from privacy.rdp import (
    RDPAccountant,
    calibrate_composed_sampled_gaussian_noise,
    calibrate_sampled_gaussian_noise,
    calibrate_sampled_without_replacement_gaussian_noise,
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
        # Legacy reference campaigns used shuffled fixed-size DataLoader
        # batches while accounting with a Poisson-SGM approximation.  The
        # DT-LDP-FAR paper lane overrides this with genuine Poisson sampling
        # and add/remove adjacency.
        "sampling_scheme": "fixed_minibatch",
        "privacy_adjacency": "unspecified",
        "poisson_steps_per_round": None,
        "fixed_batch_size": None,
        "fixed_steps_per_round": None,
        "per_sample_backend": "vectorized",
        "dp_compute_multiplier": 2.5,
        "private_aux_loss": True,
        "aux_loss_clip": 2.5,
        "aux_loss_noise_multiplier": 5.0,
        "aux_loss_to_model_noise_ratio": 2.5,
        "target_epsilon_includes_auxiliary_channels": False,
        "loss_eval_max_batches": None,
        "client_metrics_every": 2,
        "privacy_noise_multiplier_scale_by_client": None,
        "enable_noise_free_counterfactual_oracle": False,
    }


class _LocalDPSGDMixin:
    """Client-side implementation shared by all DP reference baselines."""

    @staticmethod
    def _client_noise_scale(state, config) -> float:
        """Return a public client-specific multiplier for heteroscedastic lanes.

        Scales are required to be at least one whenever a target epsilon is
        calibrated.  The least-noisy clients therefore attain the advertised
        epsilon and all other clients receive stronger privacy; the maximum
        realised epsilon remains bounded by the public target.
        """

        configured = config.get("privacy_noise_multiplier_scale_by_client")
        if configured is None:
            return 1.0
        client_id = int(state.client_id)
        if isinstance(configured, dict):
            value = configured.get(str(client_id), configured.get(client_id, 1.0))
        elif isinstance(configured, (list, tuple)):
            if client_id >= len(configured):
                raise ValueError(
                    "privacy_noise_multiplier_scale_by_client does not cover "
                    f"client {client_id}"
                )
            value = configured[client_id]
        else:
            raise TypeError(
                "privacy_noise_multiplier_scale_by_client must be a mapping or list"
            )
        scale = float(value)
        if scale <= 0.0:
            raise ValueError("Every public client noise scale must be positive")
        if config.get("target_epsilon") is not None and scale < 1.0:
            raise ValueError(
                "Target-epsilon heteroscedastic diagnostics require scales >= 1 "
                "so no client exceeds the advertised privacy budget"
            )
        return scale

    @staticmethod
    def _sampling_plan(dataloader, config) -> dict:
        """Resolve one sampling/accounting plan used by training and RDP.

        A Poisson step includes every local record independently with
        probability ``q``.  In that lane ``local_epochs`` is retained only as
        a backward-compatible configuration name for the number of local DP
        steps when ``poisson_steps_per_round`` is not explicitly provided.
        It does *not* denote complete passes over the local dataset.
        """

        dataset_size = max(len(dataloader.dataset), 1)
        batch_size = min(int(config.get("batch_size", 1)), dataset_size)
        q_override = config.get("privacy_sampling_rate_override")
        sampling_rate = (
            float(q_override) if q_override is not None else batch_size / dataset_size
        )
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("privacy sampling rate must lie in (0,1]")

        sampling_scheme = str(config.get("sampling_scheme", "fixed_minibatch")).lower()
        if sampling_scheme == "poisson":
            configured_steps = config.get("poisson_steps_per_round")
            steps_per_round = int(
                configured_steps
                if configured_steps is not None
                else config.get("local_epochs", 1)
            )
            max_local_batches = config.get("max_local_batches")
            if max_local_batches is not None:
                steps_per_round = min(steps_per_round, int(max_local_batches))
            if steps_per_round < 1:
                raise ValueError("Poisson sampling requires at least one DP step")
            adjacency = str(config.get("privacy_adjacency", "add_remove")).lower()
            if adjacency != "add_remove":
                raise ValueError(
                    "the primary Poisson-SGM lane requires add_remove adjacency"
                )
            public_dataset_size = config.get("privacy_public_dataset_size")
            if public_dataset_size is None:
                raise ValueError(
                    "Poisson add/remove DP requires a public fixed local dataset "
                    "capacity for normalization and server metadata"
                )
            public_dataset_size = int(public_dataset_size)
            if public_dataset_size < dataset_size:
                raise ValueError(
                    "privacy_public_dataset_size cannot be smaller than the "
                    "real local dataset"
                )
            sensitivity_multiplier = 1.0
            fixed_batch_size = None
        elif sampling_scheme == "fixed_without_replacement":
            adjacency = str(config.get("privacy_adjacency", "replace_one")).lower()
            if adjacency != "replace_one":
                raise ValueError(
                    "fixed-size sampling without replacement requires "
                    "replace_one adjacency"
                )
            public_dataset_size = config.get("privacy_public_dataset_size")
            if public_dataset_size is None:
                raise ValueError(
                    "fixed-size replace-one DP requires a public local dataset size"
                )
            public_dataset_size = int(public_dataset_size)
            if public_dataset_size != dataset_size:
                raise ValueError(
                    "replace-one fixed-size sampling requires the public local "
                    "dataset size to equal the realised fixed cardinality"
                )
            configured_batch_size = config.get("fixed_batch_size")
            if configured_batch_size is None:
                raw_batch_size = sampling_rate * public_dataset_size
                fixed_batch_size = int(round(raw_batch_size))
                if not math.isclose(
                    raw_batch_size,
                    fixed_batch_size,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "sampling rate times public dataset size must be an "
                        "integer fixed batch size"
                    )
            else:
                fixed_batch_size = int(configured_batch_size)
            if not 1 <= fixed_batch_size <= public_dataset_size:
                raise ValueError(
                    "fixed_batch_size must lie in [1, privacy_public_dataset_size]"
                )
            realised_rate = fixed_batch_size / public_dataset_size
            if q_override is not None and not math.isclose(
                realised_rate, sampling_rate, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "fixed_batch_size disagrees with privacy_sampling_rate_override"
                )
            sampling_rate = realised_rate
            configured_steps = config.get("fixed_steps_per_round")
            steps_per_round = int(
                configured_steps
                if configured_steps is not None
                else config.get("local_epochs", 1)
            )
            max_local_batches = config.get("max_local_batches")
            if max_local_batches is not None:
                steps_per_round = min(steps_per_round, int(max_local_batches))
            if steps_per_round < 1:
                raise ValueError("fixed-size sampling requires at least one DP step")
            # Replacing one clipped gradient can change the sum by 2C. The
            # implementation multiplier is relative to C, while the RDP
            # accountant multiplier must be relative to the 2C sensitivity.
            sensitivity_multiplier = 2.0
        elif sampling_scheme == "fixed_minibatch":
            batches_per_epoch = math.ceil(dataset_size / batch_size)
            max_local_batches = config.get("max_local_batches")
            if max_local_batches is not None:
                batches_per_epoch = min(batches_per_epoch, int(max_local_batches))
            steps_per_round = int(config.get("local_epochs", 1)) * batches_per_epoch
            adjacency = str(config.get("privacy_adjacency", "unspecified")).lower()
            public_dataset_size = dataset_size
            sensitivity_multiplier = 1.0
            fixed_batch_size = None
        else:
            raise ValueError(
                "sampling_scheme must be fixed_minibatch, poisson or "
                "fixed_without_replacement"
            )
        return {
            "scheme": sampling_scheme,
            "adjacency": adjacency,
            "sampling_rate": sampling_rate,
            "steps_per_round": steps_per_round,
            "expected_batch_size": (
                float(fixed_batch_size)
                if fixed_batch_size is not None
                else sampling_rate * public_dataset_size
            ),
            "public_dataset_size": public_dataset_size,
            "normalization_denominator": (
                float(fixed_batch_size)
                if fixed_batch_size is not None
                else sampling_rate * public_dataset_size
            ),
            "fixed_batch_size": fixed_batch_size,
            "sensitivity_multiplier": sensitivity_multiplier,
        }

    def _resolved_noise_multiplier(self, dataloader, state, config) -> float:
        if not bool(config.get("enable_dp", True)):
            return 0.0
        client_scale = self._client_noise_scale(state, config)
        configured = float(config.get("noise_multiplier", 1.0))
        target = config.get("target_epsilon")
        if target is None:
            return configured * client_scale
        cache_key = "local_dp_calibrated_base_noise"
        if cache_key in state.custom:
            return float(state.custom[cache_key]) * client_scale
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError(
                "target_epsilon requires privacy_num_rounds for pre-training calibration"
            )
        plan = self._sampling_plan(dataloader, config)
        if plan["scheme"] == "fixed_without_replacement":
            noise = calibrate_sampled_without_replacement_gaussian_noise(
                target_epsilon=float(target),
                delta=float(config.get("delta", 1e-5)),
                sampling_rate=float(plan["sampling_rate"]),
                steps=int(total_rounds) * int(plan["steps_per_round"]),
                sensitivity_multiplier=float(plan["sensitivity_multiplier"]),
            )
        else:
            noise = calibrate_sampled_gaussian_noise(
                target_epsilon=float(target),
                delta=float(config.get("delta", 1e-5)),
                sampling_rate=float(plan["sampling_rate"]),
                steps=int(total_rounds) * int(plan["steps_per_round"]),
            )
        state.custom[cache_key] = float(noise)
        return float(noise) * client_scale

    def _local_dp_update(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        plan = self._sampling_plan(dataloader, config)
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
            sampling_scheme=str(plan["scheme"]),
            poisson_sampling_rate=(
                float(plan["sampling_rate"]) if plan["scheme"] == "poisson" else None
            ),
            poisson_steps_per_round=(
                int(plan["steps_per_round"]) if plan["scheme"] == "poisson" else None
            ),
            poisson_normalization_denominator=(
                float(plan["normalization_denominator"])
                if plan["scheme"] == "poisson"
                else None
            ),
            fixed_batch_size=(
                int(plan["fixed_batch_size"])
                if plan["scheme"] == "fixed_without_replacement"
                else None
            ),
            fixed_steps_per_round=(
                int(plan["steps_per_round"])
                if plan["scheme"] == "fixed_without_replacement"
                else None
            ),
            fixed_normalization_denominator=(
                float(plan["normalization_denominator"])
                if plan["scheme"] == "fixed_without_replacement"
                else None
            ),
            track_noise_free_counterfactual=bool(
                config.get("enable_noise_free_counterfactual_oracle", False)
            ),
        )

        sampling_rate = float(plan["sampling_rate"])
        accountant = RDPAccountant.from_state_dict(
            state.custom.get("local_dp_accountant")
        )
        epsilon = best_order = None
        if dp_enabled:
            if plan["scheme"] == "fixed_without_replacement":
                accountant.add_sampled_without_replacement_gaussian(
                    channel="model",
                    sampling_rate=sampling_rate,
                    noise_multiplier=(
                        noise_multiplier / float(plan["sensitivity_multiplier"])
                    ),
                    steps=stats.steps,
                )
            else:
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
            # Under add/remove adjacency the exact cardinality is not part of
            # the transcript.  The server receives only the public capacity.
            "dataset_size": int(plan["public_dataset_size"]),
            "local_loss": stats.mean_loss,
            "clip_rate": stats.clip_rate,
            "model_steps": stats.steps,
            "dp_noise_norm_mean": stats.mean_noise_norm,
            "privacy_epsilon": epsilon,
            "privacy_best_order": best_order,
            "privacy_delta": float(config.get("delta", 1e-5)) if dp_enabled else None,
            "privacy_sampling_rate": sampling_rate,
            "privacy_sampling_scheme": str(plan["scheme"]),
            "privacy_adjacency": str(plan["adjacency"]),
            "privacy_expected_batch_size": float(plan["expected_batch_size"]),
            "privacy_fixed_batch_size": plan["fixed_batch_size"],
            "privacy_normalization_denominator": float(
                plan["normalization_denominator"]
            ),
            "privacy_empty_poisson_steps": int(stats.empty_steps),
            "privacy_noise_multiplier": noise_multiplier if dp_enabled else None,
            "privacy_accounting_noise_multiplier": (
                noise_multiplier / float(plan["sensitivity_multiplier"])
                if dp_enabled
                else None
            ),
            "privacy_sensitivity_multiplier": float(plan["sensitivity_multiplier"]),
            "privacy_noise_multiplier_scale_public": self._client_noise_scale(
                state, config
            ),
            "privacy_level": "sample" if dp_enabled else "none",
            "privacy_trust_model": "local" if dp_enabled else "not_applicable",
            "privacy_accounting_assumption": (
                "poisson_sampled_gaussian_add_remove"
                if dp_enabled and plan["scheme"] == "poisson"
                else (
                    "fixed_size_without_replacement_rdp_replace_one"
                    if dp_enabled and plan["scheme"] == "fixed_without_replacement"
                    else (
                        "poisson_approximation_for_fixed_minibatches"
                        if dp_enabled
                        else "not_applicable"
                    )
                )
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
            "far_update_mode": (
                "poisson_dp_steps"
                if isinstance(self, FAR) and plan["scheme"] == "poisson"
                else (
                    "fixed_without_replacement_dp_steps"
                    if isinstance(self, FAR)
                    and plan["scheme"] == "fixed_without_replacement"
                    else "multi_epoch_delta" if isinstance(self, FAR) else None
                )
            ),
            "far_prox_mu": (
                float(config.get("far_prox_mu", config.get("mu", 0.0)))
                if isinstance(self, FAR)
                else None
            ),
        }
        if stats.noise_free_delta_oracle is not None:
            metadata["local_dp_noise_free_update_oracle"] = (
                stats.noise_free_delta_oracle
            )
        return update, metadata

    def _local_private_gradient_update(self, model, dataloader, state, config):
        """Release one private mini-batch gradient without a local model step.

        This is the strict FedFDP-style *gradient channel* used by the new
        LDP-gradient-FAR lane.  It intentionally rejects the older Poisson and
        multi-step model-delta paths so the sampler, adjacency and accountant
        cannot drift apart silently.
        """

        plan = self._sampling_plan(dataloader, config)
        if plan["scheme"] != "fixed_without_replacement":
            raise ValueError(
                "private-gradient FAR requires fixed-size sampling without "
                "replacement"
            )
        if plan["adjacency"] != "replace_one":
            raise ValueError("private-gradient FAR requires replace_one adjacency")
        if int(plan["steps_per_round"]) != 1:
            raise ValueError(
                "private-gradient FAR releases exactly one sampled gradient per round; "
                "set fixed_steps_per_round=1"
            )
        fixed_batch_size = int(plan["fixed_batch_size"])
        noise_multiplier = self._resolved_noise_multiplier(dataloader, state, config)
        dp_enabled = bool(config.get("enable_dp", True))
        clip_norm = float(config.get("clip_norm", 1.0))
        gradient, stats = private_gradient_release_fixed_without_replacement(
            model,
            dataloader,
            device=str(config.get("device", "cpu")),
            batch_size=fixed_batch_size,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier if dp_enabled else 0.0,
            backend=str(config.get("per_sample_backend", "vectorized")),
            return_noise_free_oracle=bool(
                config.get("enable_oracle_diagnostics", False)
            ),
        )

        accountant = RDPAccountant.from_state_dict(
            state.custom.get("local_dp_accountant")
        )
        epsilon = best_order = None
        if dp_enabled:
            accounting_noise = noise_multiplier / float(plan["sensitivity_multiplier"])
            accountant.add_sampled_without_replacement_gaussian(
                channel="gradient",
                sampling_rate=float(plan["sampling_rate"]),
                noise_multiplier=accounting_noise,
                steps=1,
            )
            epsilon, best_order = accountant.epsilon(float(config.get("delta", 1e-5)))
            state.custom["local_dp_accountant"] = accountant.state_dict()
        else:
            accounting_noise = None

        uplink_bytes = self.count_bytes(gradient, sparse=False)
        downlink_bytes = uplink_bytes
        profile = config.get("device_profile")
        if profile:
            # The existing helper estimates one complete local epoch.  Scale
            # it by m/N because this mechanism evaluates exactly one public
            # fixed-size batch per round.
            full_epoch_flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                1,
            )
            flops = (
                full_epoch_flops
                * fixed_batch_size
                / float(plan["public_dataset_size"])
                * float(config.get("dp_compute_multiplier", 2.5))
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
            breakdown = {
                "compute": energy,
                "uplink": 0.0,
                "downlink": 0.0,
                "total": energy,
            }
        state.battery_j = max(0.0, state.battery_j - breakdown["total"])
        state.round_num += 1

        denominator = float(fixed_batch_size)
        gaussian_std = noise_multiplier * clip_norm / denominator if dp_enabled else 0.0
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            # Public execution diagnostic.  It depends only on the configured
            # runtime, not on any sampled record, and detects silent fallback
            # away from the requested accelerator.
            "privacy_compute_device": str(next(model.parameters()).device),
            "dataset_size": int(plan["public_dataset_size"]),
            # No unaccounted loss, clipping-rate or realised-noise diagnostic
            # is transmitted in this strict local-DP lane.
            "local_loss": 0.0,
            "local_loss_available": False,
            # The realised clipping rate is data-dependent.  It must not be
            # released by the strict local-DP mechanism without its own
            # accountant, but it is safe and useful in the no-DP calibration
            # control where no privacy claim is made.
            "clip_rate": (
                None
                if dp_enabled
                else float(stats.clipped_examples) / max(float(stats.examples), 1.0)
            ),
            "model_steps": 1,
            "privacy_epsilon": epsilon,
            "privacy_best_order": best_order,
            "privacy_delta": float(config.get("delta", 1e-5)) if dp_enabled else None,
            "privacy_sampling_rate": float(plan["sampling_rate"]),
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "privacy_expected_batch_size": denominator,
            "privacy_fixed_batch_size": fixed_batch_size,
            "privacy_normalization_denominator": denominator,
            "privacy_empty_poisson_steps": 0,
            "privacy_batch_nonempty_by_construction": True,
            "privacy_noise_multiplier": noise_multiplier if dp_enabled else None,
            "privacy_accounting_noise_multiplier": accounting_noise,
            "privacy_sensitivity_multiplier": 2.0,
            "privacy_query_sensitivity_l2": 2.0 * clip_norm / denominator,
            "privacy_gaussian_std_per_coordinate": gaussian_std,
            "privacy_upload_noise_variance_per_coordinate": gaussian_std**2,
            "privacy_noise_multiplier_scale_public": self._client_noise_scale(
                state, config
            ),
            "privacy_level": "sample" if dp_enabled else "none",
            "privacy_trust_model": "local" if dp_enabled else "not_applicable",
            "privacy_accounting_assumption": (
                "fixed_size_without_replacement_rdp_replace_one"
                if dp_enabled
                else "not_applicable"
            ),
            "privacy_release_object": "averaged_clipped_gradient",
            "privacy_gradient_release": True,
            "privacy_clip_before_noise": True,
            "privacy_post_noise_client_clip": False,
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "energy_j_consumed": breakdown["total"],
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
            "far_update_mode": "single_step_gradient",
            "far_local_steps": 1,
        }
        if stats.noise_free_delta_oracle is not None:
            # Simulator-only evaluation fields. FAR does not consume these to
            # choose its reference, scores, weights or aggregate.
            metadata["local_dp_noise_free_update_oracle"] = (
                stats.noise_free_delta_oracle
            )
            metadata["dp_noise_norm_mean"] = stats.mean_noise_norm / denominator
        return gradient, metadata

    def client_update(self, model, dataloader, state, config):
        return self._local_dp_update(model, dataloader, state, config)

    def _add_local_dp_round_metrics(self, result, client_updates, config):
        """Attach the common local-DP ledger diagnostics to any server rule.

        Keeping this step separate from ``server_aggregate`` lets research
        algorithms reuse the same client-side DP-SGD mechanism while changing
        only the post-processing rule at the server.  DT-LDP-FAR uses this to
        apply weights computed from the previous round rather than FAR's
        current-message weights.
        """

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
        noise_scales = [
            float(metadata["privacy_noise_multiplier_scale_public"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_noise_multiplier_scale_public") is not None
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
        assumptions = {
            str(metadata["privacy_accounting_assumption"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_accounting_assumption") is not None
        }
        sampling_schemes = {
            str(metadata["privacy_sampling_scheme"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_sampling_scheme") is not None
        }
        adjacencies = {
            str(metadata["privacy_adjacency"])
            for _, metadata, _ in client_updates
            if metadata.get("privacy_adjacency") is not None
        }
        result.metrics.update(
            {
                "privacy_epsilon_max": max(epsilons) if epsilons else None,
                "privacy_epsilon_mean": (
                    sum(epsilons) / len(epsilons) if epsilons else None
                ),
                "privacy_delta": float(config.get("delta", 1e-5)) if epsilons else None,
                "privacy_target_epsilon": (
                    float(config["target_epsilon"])
                    if config.get("target_epsilon") is not None
                    else None
                ),
                "privacy_profile": config.get("experiment_privacy_profile"),
                "privacy_level": "sample" if epsilons else "none",
                "privacy_trust_model": "local" if epsilons else "not_applicable",
                "privacy_clip_rate_mean": (
                    sum(clip_rates) / len(clip_rates) if clip_rates else None
                ),
                "privacy_model_noise_multiplier_mean": (
                    sum(noise_multipliers) / len(noise_multipliers)
                    if noise_multipliers
                    else None
                ),
                "privacy_model_noise_multiplier_min": (
                    min(noise_multipliers) if noise_multipliers else None
                ),
                "privacy_model_noise_multiplier_max": (
                    max(noise_multipliers) if noise_multipliers else None
                ),
                "privacy_noise_scale_public_min": (
                    min(noise_scales) if noise_scales else None
                ),
                "privacy_noise_scale_public_max": (
                    max(noise_scales) if noise_scales else None
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
                    next(iter(assumptions))
                    if len(assumptions) == 1
                    else "mixed_or_missing" if epsilons else "not_applicable"
                ),
                "privacy_sampling_scheme": (
                    next(iter(sampling_schemes))
                    if len(sampling_schemes) == 1
                    else "mixed_or_missing"
                ),
                "privacy_adjacency": (
                    next(iter(adjacencies))
                    if len(adjacencies) == 1
                    else "mixed_or_missing"
                ),
            }
        )
        return result

    def server_aggregate(self, global_model, client_updates, round_num, config):
        result = super().server_aggregate(
            global_model, client_updates, round_num, config
        )
        return self._add_local_dp_round_metrics(result, client_updates, config)


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

    def client_update(self, model, dataloader, state, config):
        """Optionally expose only the accounted locally-private model upload.

        Historical DP-FAR experiments retained local loss, realised clipping
        rate and realised noise norm as simulation diagnostics.  The matched
        current-round control for DT-LDP-FAR must have the same transcript
        policy as the delayed method, so those unaccounted auxiliary values
        are suppressed when ``suppress_private_client_diagnostics`` is true.
        """

        update, metadata = self._local_dp_update(model, dataloader, state, config)
        suppress = bool(config.get("suppress_private_client_diagnostics", False))
        if suppress:
            metadata.pop("local_loss", None)
            metadata.pop("clip_rate", None)
            metadata.pop("dp_noise_norm_mean", None)
            metadata.update(
                {
                    "local_loss": 0.0,
                    "local_loss_available": False,
                    "dpfar_transcript_policy": (
                        "no_unaccounted_loss_clip_or_noise_release"
                    ),
                }
            )
        else:
            metadata["dpfar_transcript_policy"] = (
                "non_private_simulation_oracles_enabled"
            )
        return update, metadata

    def get_default_config(self):
        return {
            **FAR.get_default_config(self),
            **_dp_defaults(),
            "suppress_private_client_diagnostics": False,
        }


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
        plan = self._sampling_plan(dataloader, config)
        q_model = float(plan["sampling_rate"])
        model_steps = int(total_rounds) * int(plan["steps_per_round"])
        ratio = float(config.get("aux_loss_to_model_noise_ratio", 2.5))
        model_noise = calibrate_composed_sampled_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            channels=((q_model, model_steps, 1.0), (1.0, int(total_rounds), ratio)),
        )
        state.custom[cache_key] = float(model_noise)
        state.custom["local_dp_calibrated_aux_loss_noise"] = float(model_noise * ratio)
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
            accountant = RDPAccountant.from_state_dict(
                state.custom.get("local_dp_accountant")
            )
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
