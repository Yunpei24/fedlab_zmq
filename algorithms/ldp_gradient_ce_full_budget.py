"""Pure CE control: the entire epsilon=4 budget funds private gradients.

This isolated registration is imported explicitly by its experiment wrapper.
It does not compute, store, or transmit a private histogram or class weights.
The existing mu=0 MPS primitive preserves the earlier CE/DMD fixed-WOR batch
and standard-Gaussian streams. Only sigma changes under full-budget calibration.
The inherited three server rules are postprocessing of private CPU uploads.

The reproducible simulation RNG is not a production secret client RNG; the
ledger describes the ideal sampled Gaussian mechanism, not seed-public privacy.
"""

from __future__ import annotations

import copy
import hashlib
import math
from functools import lru_cache

import torch

from hardware.flop_cost import round_compute_flops
from privacy.dmd_cb_private import (
    private_dmd_gradient_release,
    require_private_mps,
    simulation_seed,
)
from privacy.rdp import (
    RDPAccountant,
    calibrate_sampled_without_replacement_gaussian_noise,
)

from .base import register_algorithm
from .ldp_gradient_dmd_cb import LDPGradientDMDCB
from .ldp_gradient_far import LDPGradientFAR

_CONTEXT_KEY = "ce_full_budget_private_context"


def _reject_foreign_private_state(state):
    """ClientState is transported too: do not carry raw or histogram buffers."""
    if state.error_buffer is not None or state.momentum_buffer is not None:
        raise ValueError("CE does not transport gradient/error/momentum buffers")
    if any(
        "histogram" in key
        or "class_weights" in key
        or "oracle" in key
        or key
        in {"dmd_cb_private_context", "raw_counts", "class_counts", "raw_gradients"}
        for key in state.custom
    ):
        raise ValueError(
            "Histogram or raw private state is forbidden in full-budget CE"
        )


@lru_cache(maxsize=64)
def _calibrated_ce_sigma(delta, sampling_rate, rounds):
    """Public, conservative calibration; frozen campaign sigma takes priority."""
    return calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=4.0,
        delta=delta,
        sampling_rate=sampling_rate,
        steps=rounds,
        sensitivity_multiplier=2.0,
        tolerance=1e-8,
    ) * (1.0 + 1e-7)


@register_algorithm("ldp_gradient_ce_full_budget")
class LDPGradientCEFullBudget(LDPGradientDMDCB):
    """One MPS sample-level replace-one private CE gradient per client-round."""

    description = "Pure CE, no histogram, full epsilon=4 gradient budget; mean/direct RFA/FAR-RFA."

    def get_default_config(self):
        return {
            **LDPGradientFAR.get_default_config(
                self
            ),  # Shared gradient/server defaults.
            "device": "mps",  # Private computation is strictly MPS, without fallback.
            "dmd_mu": 0.0,  # Exact CE branch of the matched private primitive.
            "dmd_server_mode": "uniform",  # uniform, direct_rfa, or far_rfa.
            "dmd_pairing_seed": None,  # Required public experiment stream identity.
            "dmd_histogram_epsilon": 0.0,  # No histogram mechanism exists in this lane.
            "dmd_num_classes": 10,  # Fixed public class universe; public unit weights.
            "dmd_total_epsilon": 4.0,  # Entire sample-level local privacy budget.
            "target_epsilon": 4.0,  # Gradient channel receives all of that budget.
            "dmd_frozen_base_noise_multiplier": None,  # Optional exact certified sigma.
            "privacy_num_rounds": 40,  # Public horizon; one release per round.
            "privacy_public_dataset_size": 2400,  # Fixed local replace-one cardinality N.
            "privacy_sampling_rate_override": 0.1,  # Public sampling fraction B/N.
            "fixed_batch_size": 240,  # Uniform sample without replacement, size B.
            "clip_norm": 4.0,  # Joint per-example CE gradient clipping radius C.
            "far_server_clip_norm": 16.0,  # Post-release server clipping radius U.
            "far_server_lr": 0.2,  # Server subtracts this times aggregated gradients.
            "far_alpha": 0.0,  # Set to exactly 0.1 for the FAR-RFA control only.
            "robust_reference": "rfa",  # Same RFA rule as the matched DMD controls.
            "external_attack_diagnostics": True,  # Simulator labels stay outside server.
            "enable_oracle_diagnostics": False,  # Never expose private raw diagnostics.
            "rcig_required_private_device": "mps",  # Server audits MPS compute provenance.
        }

    def _validated_config(self, config):
        resolved = {**self.get_default_config(), **config}
        if not bool(resolved.get("enable_dp", True)):
            raise ValueError("Full-budget CE is a private-only lane")
        if resolved.get("dmd_pairing_seed") is None:
            raise ValueError(
                "Explicit dmd_pairing_seed is required for matched streams"
            )
        for name in (
            "enable_oracle_diagnostics",
            "private_aux_loss",
            "enable_noise_free_counterfactual_oracle",
        ):
            if bool(resolved.get(name, False)):
                raise ValueError(
                    "No auxiliary channel or raw diagnostics in full-budget CE"
                )
        if not bool(resolved.get("suppress_private_client_diagnostics", True)):
            raise ValueError("Private raw diagnostics must remain suppressed")
        for name, expected in (
            ("dmd_mu", 0.0),
            ("dmd_histogram_epsilon", 0.0),
            ("dmd_total_epsilon", 4.0),
            ("target_epsilon", 4.0),
        ):
            value = float(resolved[name])
            if not math.isfinite(value) or value != expected:
                raise ValueError(f"Full-budget CE requires {name}={expected}")
        if int(resolved["dmd_num_classes"]) != 10:
            raise ValueError("Full-budget CE fixes the public class universe to ten")
        if int(resolved["privacy_num_rounds"]) < 1:
            raise ValueError("A positive public gradient horizon is required")
        if not 0 < float(resolved["delta"]) < 1:
            raise ValueError("Privacy delta must lie in (0,1)")
        if (
            resolved.get("sampling_scheme") != "fixed_without_replacement"
            or resolved.get("privacy_adjacency") != "replace_one"
            or int(resolved.get("fixed_steps_per_round", 1)) != 1
        ):
            raise ValueError("Exactly one fixed-WOR replace-one gradient is required")
        if str(resolved.get("per_sample_backend")) != "vectorized":
            raise ValueError("The private CE gradient backend must be MPS vectorized")
        if str(resolved.get("device")) not in {"mps", "mps:0"}:
            raise ValueError("Private CE computations require MPS")
        if str(resolved.get("rcig_required_private_device")) != "mps":
            raise ValueError("Private gradient provenance must require MPS")
        if not bool(resolved.get("external_attack_diagnostics")):
            raise ValueError("Attack truth must remain outside the deployed server")
        mode = str(resolved["dmd_server_mode"])
        if mode not in {"uniform", "direct_rfa", "far_rfa"}:
            raise ValueError("Unknown full-budget CE server control")
        if float(resolved["far_alpha"]) != (0.1 if mode == "far_rfa" else 0.0):
            raise ValueError("Use alpha=0.1 for FAR-RFA and alpha=0 for other controls")
        if (
            str(resolved["robust_reference"]).lower() != "rfa"
            or resolved.get("far_score_mode") != "raw_distance"
            or resolved.get("tilt_bound_policy") != "diagnostic"
            or resolved.get("far_distance_clip") is not None
            or resolved.get("noise_score_standardization") != "none"
            or resolved.get("score_subspace_mode") != "full"
        ):
            raise ValueError(
                "Full-budget CE fixes RFA and unstandardized raw FAR distances"
            )
        for name in ("clip_norm", "far_server_clip_norm", "far_server_lr"):
            if not math.isfinite(float(resolved[name])) or float(resolved[name]) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return resolved

    def client_update(self, model, dataloader, state, config):
        config = self._validated_config(config)
        device = str(config["device"])
        require_private_mps(device)  # Fail before reading even a private dataset item.
        plan = self._sampling_plan(dataloader, config)
        horizon = int(config["privacy_num_rounds"])
        if plan["steps_per_round"] != 1 or not 0 <= state.round_num < horizon:
            raise ValueError("Client step exceeds the one-release public horizon")
        scale = self._registered_noise_scale(int(state.client_id), config)
        public_context = {
            "version": 1,
            "algorithm": self.name,
            "client_id": int(state.client_id),
            "N": int(plan["public_dataset_size"]),
            "B": int(plan["fixed_batch_size"]),
            "K": int(config["dmd_num_classes"]),
            "histogram_epsilon": 0.0,
            "gradient_target_epsilon": 4.0,
            "total_target_epsilon": 4.0,
            "delta": float(config["delta"]),
            "rounds": horizon,
            "C": float(config["clip_norm"]),
            "noise_scale": scale,
            "frozen_base_noise_multiplier": config["dmd_frozen_base_noise_multiplier"],
            "mu": 0.0,
            "adjacency": "replace_one",
            "sampling_scheme": "fixed_without_replacement",
            "pairing_identity_sha256": hashlib.sha256(
                str(int(config["dmd_pairing_seed"])).encode()
            ).hexdigest(),
            "server_mode": config["dmd_server_mode"],
            "server_clip_norm": float(config["far_server_clip_norm"]),
            "server_lr": float(config["far_server_lr"]),
            "far_alpha": float(config["far_alpha"]),
        }
        _reject_foreign_private_state(state)
        context = state.custom.get(_CONTEXT_KEY)
        if context is None:
            if (
                state.round_num != 0
                or state.local_steps != 0
                or "local_dp_accountant" in state.custom
                or "local_dp_calibrated_base_noise" in state.custom
            ):
                raise ValueError("Missing CE context: partial restart is forbidden")
            context = {"public_parameters": public_context, "gradient_steps": 0}
        if (
            not isinstance(context, dict)
            or set(context) != {"public_parameters", "gradient_steps"}
            or context.get("public_parameters") != public_context
            or context.get("gradient_steps") != state.round_num
            or state.local_steps != state.round_num
        ):
            raise ValueError("CE public context or once-step client state has changed")
        # Resolve calibration on a staged state: failed calls cannot mutate a
        # checkpoint or leave a half-initialized context behind.
        staged = copy.copy(state)
        staged.custom = copy.deepcopy(state.custom)
        if config["dmd_frozen_base_noise_multiplier"] is None:
            sigma = _calibrated_ce_sigma(
                public_context["delta"], float(plan["sampling_rate"]), horizon
            )
            cached = staged.custom.get("local_dp_calibrated_base_noise")
            if cached is not None and float(cached) != sigma:
                raise ValueError("Cached sigma disagrees with CE calibration")
            staged.custom["local_dp_calibrated_base_noise"] = sigma
            noise_multiplier = sigma * scale
        else:
            noise_multiplier = self._dmd_noise_multiplier(
                dataloader, staged, config, plan, scale
            )
        if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError("Private Gaussian sigma must be finite and positive")
        if state.round_num and "local_dp_calibrated_base_noise" not in state.custom:
            raise ValueError("Missing cached sigma in resumed CE state")
        persisted = state.custom.get("local_dp_accountant")
        accountant = RDPAccountant.from_state_dict(persisted)
        expected = RDPAccountant()
        if state.round_num:
            expected.add_sampled_without_replacement_gaussian(
                channel="gradient",
                sampling_rate=plan["sampling_rate"],
                noise_multiplier=noise_multiplier / 2.0,
                steps=state.round_num,
            )
        actual_rdp, expected_rdp = accountant.total_rdp(), expected.total_rdp()
        if (
            (state.round_num > 0 and not persisted)
            or accountant.orders != expected.orders
            or set(accountant.channels) != set(expected.channels)
            or any(
                set(values) != set(expected.orders)
                for values in accountant.channels.values()
            )
            or any(
                not math.isclose(actual_rdp[order], value, rel_tol=1e-10, abs_tol=1e-12)
                for order, value in expected_rdp.items()
            )
        ):
            raise ValueError(
                "Gradient accountant is missing or inconsistent with CE step"
            )
        # No local histogram access: these ones are public and never persisted.
        gradient = private_dmd_gradient_release(
            model,
            dataloader,
            class_weights=torch.ones(public_context["K"], device=device),
            mu=0.0,
            batch_size=public_context["B"],
            clip_norm=public_context["C"],
            noise_multiplier=noise_multiplier,
            batch_seed=simulation_seed(
                config["dmd_pairing_seed"], state.client_id, state.round_num, "batch"
            ),
            gaussian_seed=simulation_seed(
                config["dmd_pairing_seed"], state.client_id, state.round_num, "gaussian"
            ),
            device=device,
        )
        accountant.add_sampled_without_replacement_gaussian(
            channel="gradient",
            sampling_rate=plan["sampling_rate"],
            noise_multiplier=noise_multiplier / 2.0,
            steps=1,
        )
        epsilon, best_order = accountant.epsilon(public_context["delta"])
        if epsilon > 4.0 + 1e-7:
            raise ValueError("CE gradient privacy budget exceeds epsilon=4")
        bytes_sent = self.count_bytes(gradient, sparse=False)
        profile = config.get("device_profile")
        if profile is not None:
            flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                1,
            )
            flops *= (
                public_context["B"]
                / public_context["N"]
                * float(config.get("dp_compute_multiplier", 2.5))
            )
            breakdown = profile.round_energy_breakdown(
                flops,
                bytes_sent,
                bytes_sent,
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
        if any(
            not math.isfinite(float(value)) or value < 0 for value in breakdown.values()
        ):
            raise ValueError("Energy accounting must remain finite and nonnegative")
        state.custom["local_dp_calibrated_base_noise"] = staged.custom[
            "local_dp_calibrated_base_noise"
        ]
        state.custom["local_dp_accountant"] = accountant.state_dict()
        state.round_num += 1
        state.local_steps = state.round_num
        state.custom[_CONTEXT_KEY] = {
            "public_parameters": public_context,
            "gradient_steps": state.round_num,
        }
        energy_j = float(breakdown["total"])
        state.battery_j = max(0.0, state.battery_j - energy_j)
        denominator = float(public_context["B"])
        gaussian_std = noise_multiplier * public_context["C"] / denominator
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "privacy_compute_device": str(next(model.parameters()).device),
            "dataset_size": public_context["N"],
            "local_loss": 0.0,
            "local_loss_available": False,
            "clip_rate": None,
            "model_steps": 1,
            "privacy_epsilon": epsilon,
            "privacy_gradient_epsilon": epsilon,
            "privacy_target_epsilon": 4.0,
            "privacy_gradient_target_epsilon": 4.0,
            "privacy_best_order": best_order,
            "privacy_delta": public_context["delta"],
            "privacy_sampling_rate": plan["sampling_rate"],
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "privacy_expected_batch_size": denominator,
            "privacy_fixed_batch_size": public_context["B"],
            "privacy_normalization_denominator": denominator,
            "privacy_empty_poisson_steps": 0,
            "privacy_batch_nonempty_by_construction": True,
            "privacy_noise_multiplier": noise_multiplier,
            "privacy_accounting_noise_multiplier": noise_multiplier / 2.0,
            "privacy_sensitivity_multiplier": 2.0,
            "privacy_query_sensitivity_l2": 2 * public_context["C"] / denominator,
            "privacy_gaussian_std_per_coordinate": gaussian_std,
            "privacy_upload_noise_variance_per_coordinate": gaussian_std**2,
            "privacy_noise_multiplier_scale_public": scale,
            "privacy_level": "sample",
            "privacy_trust_model": "local",
            "privacy_accounting_assumption": "fixed_size_without_replacement_rdp_replace_one",
            "privacy_release_object": "averaged_clipped_gradient",
            "privacy_gradient_release": True,
            "privacy_clip_before_noise": True,
            "privacy_post_noise_client_clip": False,
            "privacy_dmd_histogram_epsilon": 0.0,
            "privacy_dmd_histogram_calls_total": 0,
            "privacy_dmd_histogram_calls_this_round": 0,
            "privacy_dmd_composition": "gradient_rdp_only_no_histogram",
            "dmd_mu": 0.0,
            "dmd_class_weights_source": "public_ones_no_histogram",
            "dmd_batch_weight_normalization": False,
            "dmd_clip_object": "per_example_ce_gradient",
            "dmd_randomness_policy": "separate_reproducible_simulation_streams_not_production_rng",
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_sent,
            "energy_j_consumed": energy_j,
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "battery_j_remaining": state.battery_j,
            "compression_ratio": 1.0,
            "beta_actual": 1.0,
            "far_update_mode": "single_step_gradient",
            "far_local_steps": 1,
        }
        return gradient, metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        config = self._validated_config(config)
        for _, metadata, state in client_updates:
            _reject_foreign_private_state(state)
            if any(
                metadata.get(key) != 0
                for key in (
                    "privacy_dmd_histogram_epsilon",
                    "privacy_dmd_histogram_calls_total",
                    "privacy_dmd_histogram_calls_this_round",
                )
            ):
                raise ValueError(
                    "Full-budget CE server rejects any histogram mechanism"
                )
            if (
                any(
                    key in metadata
                    for key in (
                        "raw_counts",
                        "class_counts",
                        "dp_class_weights",
                        "class_weights",
                        "dp_noise_norm_mean",
                        "dp_clipped_gradient_norm_mean",
                    )
                )
                or metadata.get("clip_rate") is not None
                or metadata.get("local_loss_available", False)
            ):
                raise ValueError(
                    "Raw private statistics or class weights must not reach server"
                )
            if (
                not math.isclose(
                    float(metadata["privacy_epsilon"]),
                    float(metadata["privacy_gradient_epsilon"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or not 0 <= float(metadata["privacy_epsilon"]) <= 4.0 + 1e-7
            ):
                raise ValueError(
                    "CE server privacy ledger must contain only gradients within epsilon=4"
                )
            if int(state.client_id) != int(metadata["client_id"]):
                raise ValueError("CE upload and authenticated client identity disagree")
        # Parent dispatch dynamically calls this lane's validation; no histogram
        # client code is invoked. All three actual aggregation paths stay exact.
        result = super().server_aggregate(
            global_model, client_updates, round_num, config
        )
        result.metrics.update(
            {
                "privacy_dmd_budget_composition": "gradient_rdp_only_no_histogram",
                "privacy_dmd_composition": "gradient_rdp_only_no_histogram",
                "dmd_class_weights_source": "public_ones_no_histogram",
                "dmd_clip_object": "per_example_ce_gradient",
                "ce_full_budget_no_histogram": True,
            }
        )
        return result


__all__ = ["LDPGradientCEFullBudget"]
