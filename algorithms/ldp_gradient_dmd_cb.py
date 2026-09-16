"""Private class-balanced DMD gradients with three explicit server controls.

This new registration is imported explicitly by its experiment wrapper.  No
active-campaign module or registry initializer is modified. The privacy ledger
composes a one-shot Laplace histogram (pure epsilon_hist-DP) with a fixed-WOR
replace-one Gaussian gradient channel. All private computations require MPS;
the server only postprocesses private CPU uploads.
"""

from __future__ import annotations

import math

import torch

from hardware.flop_cost import round_compute_flops
from metrics.robustness import weight_diagnostics
from privacy.dmd_cb_private import (
    private_class_weights,
    private_dmd_gradient_release,
    require_private_mps,
    simulation_seed,
)
from privacy.rdp import RDPAccountant
from robustness.tensor_ops import stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .ldp_gradient_far import LDPGradientFAR
from .reference_utils import apply_delta, common_round_metrics


def geometric_median_with_weights(vectors, *, max_iter=100, tol=1e-6, smoothing=1e-8):
    """Same finite Weiszfeld rule as the RFA baseline, plus its actual weights.

    These coefficients reconstruct the *final aggregate*. They are not FAR
    exponential tilting weights and do not have a kappa/n cap guarantee.
    """
    if vectors.ndim != 2 or len(vectors) < 1 or not torch.isfinite(vectors).all():
        raise ValueError("RFA requires a finite nonempty vector matrix")
    if (
        max_iter < 1
        or not math.isfinite(tol)
        or tol < 0
        or not math.isfinite(smoothing)
        or smoothing <= 0
    ):
        raise ValueError("Invalid public RFA iteration parameters")
    point = vectors.mean(dim=0)
    weights = torch.full(
        (len(vectors),), 1.0 / len(vectors), dtype=vectors.dtype, device=vectors.device
    )
    for _ in range(max_iter):
        distances = torch.linalg.vector_norm(vectors - point, dim=1).clamp_min(
            smoothing
        )
        raw_weights = distances.reciprocal()
        weights = raw_weights / raw_weights.sum()
        candidate = (raw_weights[:, None] * vectors).sum(dim=0) / raw_weights.sum()
        if torch.linalg.vector_norm(candidate - point) <= tol:
            return candidate, weights
        point = candidate
    return point, weights


@register_algorithm("ldp_gradient_dmd_cb")
class LDPGradientDMDCB(LDPGradientFAR):
    description = "One-shot DP class weights, jointly clipped CE+DMD private gradient; mean/direct RFA/FAR-RFA controls."

    def get_default_config(self):
        return {
            **super().get_default_config(),
            "dmd_mu": 0.1875,
            "dmd_server_mode": "uniform",
            "dmd_pairing_seed": None,
            "dmd_histogram_epsilon": 0.25,
            "dmd_num_classes": 10,
            "dmd_class_weight_cap": 20.0,
            "dmd_total_epsilon": 4.0,
            "dmd_frozen_base_noise_multiplier": None,
            "target_epsilon": 3.75,
            "privacy_num_rounds": 40,
            "privacy_public_dataset_size": 2400,
            "privacy_sampling_rate_override": 0.1,
            "fixed_batch_size": 240,
            "clip_norm": 4.0,
            "far_server_clip_norm": 16.0,
            "far_server_lr": 0.2,
            "far_alpha": 0.0,
            "robust_reference": "rfa",
            "external_attack_diagnostics": True,
            "enable_oracle_diagnostics": False,
            "rcig_required_private_device": "mps",
        }

    def _validated_config(self, config):
        resolved = {**self.get_default_config(), **config}
        mode = str(resolved["dmd_server_mode"])
        if mode not in {"uniform", "direct_rfa", "far_rfa"}:
            raise ValueError("Unknown DMD server control")
        if not bool(resolved.get("enable_dp", True)):
            raise ValueError("This DMD-CB experiment is the private lane only")
        if resolved.get("dmd_pairing_seed") is None:
            raise ValueError(
                "Explicit dmd_pairing_seed is required for matched streams"
            )
        if bool(resolved.get("enable_oracle_diagnostics", False)) or bool(
            resolved.get("private_aux_loss", False)
        ):
            raise ValueError(
                "No auxiliary raw diagnostics or extra loss channel in DMD-CB v1"
            )
        if int(resolved["dmd_num_classes"]) != 10:
            raise ValueError("DMD-CB v1 fixes the public class universe to ten classes")
        mu = float(resolved["dmd_mu"])
        target = float(resolved["target_epsilon"])
        histogram_epsilon = float(resolved["dmd_histogram_epsilon"])
        total = float(resolved["dmd_total_epsilon"])
        if (
            not all(math.isfinite(x) for x in (mu, target, histogram_epsilon, total))
            or mu < 0
            or min(target, histogram_epsilon) <= 0
        ):
            raise ValueError(
                "Finite nonnegative loss weight and positive privacy budgets required"
            )
        if not math.isclose(
            total, target + histogram_epsilon, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "Total epsilon must equal histogram epsilon plus gradient target"
            )
        if int(resolved["privacy_num_rounds"]) < 1:
            raise ValueError("A public positive horizon is required")
        if (
            resolved.get("sampling_scheme") != "fixed_without_replacement"
            or resolved.get("privacy_adjacency") != "replace_one"
            or int(resolved.get("fixed_steps_per_round", 1)) != 1
        ):
            raise ValueError(
                "DMD-CB v1 requires one fixed-WOR replace-one gradient per round"
            )
        if str(resolved.get("per_sample_backend", "vectorized")) != "vectorized":
            raise ValueError(
                "DMD-CB v1 requires the MPS vectorized per-example backend"
            )
        if not bool(resolved.get("external_attack_diagnostics", False)):
            raise ValueError("Attack truth must remain outside the deployed server")
        if str(resolved["robust_reference"]).lower() != "rfa":
            raise ValueError("The FAR control uses RFA as its reference")
        if mode != "far_rfa" and float(resolved["far_alpha"]) != 0:
            raise ValueError("Mean and direct RFA do not apply FAR; set far_alpha=0")
        if (
            str(resolved.get("far_score_mode")) != "raw_distance"
            or str(resolved.get("tilt_bound_policy")) != "diagnostic"
        ):
            raise ValueError(
                "This screen fixes raw FAR distances and a diagnostic alpha cap"
            )
        return resolved

    def _dmd_noise_multiplier(self, dataloader, state, config, plan, scale):
        """Honor the campaign's frozen sigma exactly, never silently recalibrate.

        The optional fallback exists for isolated unit tests. The preregistered
        runner requires and hashes a conservative frozen multiplier.
        """
        frozen = config.get("dmd_frozen_base_noise_multiplier")
        if frozen is None:
            return self._resolved_noise_multiplier(dataloader, state, config)
        frozen = float(frozen)
        if not math.isfinite(frozen) or frozen <= 0:
            raise ValueError("Frozen Gaussian multiplier must be finite and positive")
        if not math.isclose(
            float(config["noise_multiplier"]), frozen, rel_tol=0, abs_tol=1e-14
        ):
            raise ValueError("Public noise_multiplier disagrees with frozen base sigma")
        previously_cached = state.custom.get("local_dp_calibrated_base_noise")
        if previously_cached is not None and float(previously_cached) != frozen:
            raise ValueError("Cached sigma disagrees with the frozen campaign channel")
        # Public deterministic check; no private data or random draw involved.
        certificate = RDPAccountant()
        certificate.add_sampled_without_replacement_gaussian(
            channel="gradient",
            sampling_rate=float(plan["sampling_rate"]),
            noise_multiplier=frozen / 2.0,
            steps=int(config["privacy_num_rounds"]),
        )
        epsilon, _ = certificate.epsilon(float(config["delta"]))
        if epsilon > float(config["target_epsilon"]) + 1e-7:
            raise ValueError("Frozen sigma does not meet the gradient budget")
        state.custom["local_dp_calibrated_base_noise"] = frozen
        return frozen * scale

    def client_update(self, model, dataloader, state, config):
        config = self._validated_config(config)
        device = str(config.get("device", "cpu"))
        require_private_mps(device)
        plan = self._sampling_plan(dataloader, config)
        if int(plan["steps_per_round"]) != 1:
            raise ValueError("Exactly one private gradient release is required")
        horizon = int(config["privacy_num_rounds"])
        if not 0 <= state.round_num < horizon:
            raise ValueError(
                "Client step would exceed the pre-calibrated public horizon"
            )
        scale = self._client_noise_scale(state, config)
        if not math.isfinite(scale):
            raise ValueError("Noise scale must be finite")
        histogram_epsilon = float(config["dmd_histogram_epsilon"])
        clip_norm = float(config["clip_norm"])
        if not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("Gradient clipping radius must be finite and positive")
        # Public context signature only: no weights/count hash and no RNG seed.
        public_context = {
            "version": 1,
            "client_id": int(state.client_id),
            "N": int(plan["public_dataset_size"]),
            "B": int(plan["fixed_batch_size"]),
            "K": int(config["dmd_num_classes"]),
            "histogram_epsilon": histogram_epsilon,
            "histogram_l1_sensitivity": 2.0,
            "class_weight_cap": float(config["dmd_class_weight_cap"]),
            "gradient_target_epsilon": float(config["target_epsilon"]),
            "delta": float(config["delta"]),
            "rounds": horizon,
            "C": clip_norm,
            "noise_scale": scale,
            "frozen_base_noise_multiplier": config.get(
                "dmd_frozen_base_noise_multiplier"
            ),
            "mu": float(config["dmd_mu"]),
            "adjacency": "replace_one",
        }
        context = state.custom.get("dmd_cb_private_context")
        first_histogram = context is None
        if first_histogram:
            if (
                state.round_num != 0
                or state.custom.get("local_dp_accountant")
                or state.custom.get("local_dp_calibrated_base_noise")
            ):
                raise ValueError(
                    "Missing DMD histogram state: partial restart is forbidden"
                )
            weights = private_class_weights(
                dataloader.dataset,
                num_classes=public_context["K"],
                public_dataset_size=public_context["N"],
                epsilon=histogram_epsilon,
                weight_cap=public_context["class_weight_cap"],
                seed=simulation_seed(
                    config["dmd_pairing_seed"], state.client_id, 0, "histogram"
                ),
                device=device,
            )
            context = {
                "public_parameters": public_context,
                "dp_class_weights": weights.cpu().tolist(),
                "histogram_calls": 1,
                "gradient_steps": 0,
            }
            state.custom["dmd_cb_private_context"] = context
        if (
            context.get("public_parameters") != public_context
            or context.get("histogram_calls") != 1
            or context.get("gradient_steps") != state.round_num
        ):
            raise ValueError("DMD public context or once-only client state has changed")
        weights = torch.tensor(
            context["dp_class_weights"], dtype=torch.float32, device=device
        )
        if (
            weights.shape != (public_context["K"],)
            or not torch.isfinite(weights).all()
            or not (weights > 0).all()
            or (weights > public_context["class_weight_cap"]).any()
        ):
            raise ValueError("Invalid persisted DP class weights")
        noise_multiplier = self._dmd_noise_multiplier(
            dataloader, state, config, plan, scale
        )
        if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
            raise ValueError(
                "Calibrated Gaussian multiplier must be finite and positive"
            )
        accountant = RDPAccountant.from_state_dict(
            state.custom.get("local_dp_accountant")
        )
        expected_accountant = RDPAccountant()
        if state.round_num:
            expected_accountant.add_sampled_without_replacement_gaussian(
                channel="gradient",
                sampling_rate=plan["sampling_rate"],
                noise_multiplier=noise_multiplier / 2.0,
                steps=state.round_num,
            )
        actual_rdp, expected_rdp = (
            accountant.total_rdp(),
            expected_accountant.total_rdp(),
        )
        if set(actual_rdp) != set(expected_rdp) or any(
            not math.isclose(
                actual_rdp[order], expected_rdp[order], rel_tol=1e-10, abs_tol=1e-12
            )
            for order in expected_rdp
        ):
            raise ValueError(
                "Gradient accountant is missing or inconsistent with client step"
            )
        gradient = private_dmd_gradient_release(
            model,
            dataloader,
            class_weights=weights,
            mu=public_context["mu"],
            batch_size=public_context["B"],
            clip_norm=clip_norm,
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
        gradient_epsilon, best_order = accountant.epsilon(public_context["delta"])
        total_epsilon = histogram_epsilon + gradient_epsilon
        if total_epsilon > float(config["dmd_total_epsilon"]) + 1e-4 + 1e-10:
            raise ValueError(
                "Composed privacy budget exceeds its public target/tolerance"
            )
        state.custom["local_dp_accountant"] = accountant.state_dict()
        bytes_sent = self.count_bytes(gradient, sparse=False)
        profile = config.get("device_profile")
        if profile:
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
        state.battery_j = max(0.0, state.battery_j - breakdown["total"])
        state.round_num += 1
        context["gradient_steps"] = state.round_num
        denominator = float(public_context["B"])
        gaussian_std = noise_multiplier * clip_norm / denominator
        metadata = {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "privacy_compute_device": str(next(model.parameters()).device),
            "dataset_size": public_context["N"],
            "local_loss": 0.0,
            "local_loss_available": False,
            "clip_rate": None,
            "model_steps": 1,
            "privacy_epsilon": total_epsilon,
            "privacy_gradient_epsilon": gradient_epsilon,
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
            "privacy_query_sensitivity_l2": 2 * clip_norm / denominator,
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
            "privacy_dmd_histogram_epsilon": histogram_epsilon,
            "privacy_dmd_histogram_calls_total": 1,
            "privacy_dmd_histogram_calls_this_round": int(first_histogram),
            "privacy_dmd_histogram_l1_sensitivity": 2.0,
            "privacy_dmd_histogram_laplace_scale": 2.0 / histogram_epsilon,
            "privacy_dmd_composition": "one_shot_pure_dp_histogram_plus_conditional_gradient_rdp",
            "dmd_mu": public_context["mu"],
            "dmd_class_weights_source": "one_shot_dp_histogram_fixed_context",
            "dmd_batch_weight_normalization": False,
            "dmd_clip_object": "combined_ce_plus_dmd_gradient",
            "dmd_randomness_policy": "separate_reproducible_simulation_streams_not_production_rng",
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_sent,
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
        return gradient, metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        config = self._validated_config(config)
        if not client_updates:
            raise ValueError("DMD aggregation requires client uploads")
        ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
        if len(set(ids)) != len(ids) or len(ids) != int(
            config.get("expected_num_clients", len(ids))
        ):
            raise ValueError(
                "Expected unique authenticated full-cohort client identities"
            )
        for _, metadata, _ in client_updates:
            if not metadata.get("privacy_gradient_release") or metadata.get(
                "privacy_compute_device"
            ) not in {"mps", "mps:0"}:
                raise ValueError("Every private gradient must be computed on MPS")
            if any(
                "oracle" in key
                or key in {"is_byzantine", "attack_name", "attack_enabled"}
                for key in metadata
            ):
                raise ValueError(
                    "Raw simulator oracles must not enter server aggregation"
                )
        mode = config["dmd_server_mode"]
        if mode == "far_rfa":
            result = super().server_aggregate(
                global_model, client_updates, round_num, config
            )
        else:
            vectors, layout = stack_updates([update for update, _, _ in client_updates])
            if not torch.isfinite(vectors).all():
                raise ValueError("Nonfinite private client upload")
            radius = float(config["far_server_clip_norm"])
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Server clipping radius must be finite and positive")
            norms = vectors.norm(dim=1)
            factors = (radius / norms.clamp_min(1e-12)).clamp(max=1.0)
            vectors = vectors * factors[:, None]
            if mode == "direct_rfa":
                aggregate, weights = geometric_median_with_weights(
                    vectors,
                    max_iter=int(config.get("rfa_max_iter", 100)),
                    tol=float(config.get("rfa_tol", 1e-6)),
                )
            else:
                weights = torch.full(
                    (len(vectors),), 1.0 / len(vectors), dtype=vectors.dtype
                )
                aggregate = vectors.mean(dim=0)
            diagnostics = {
                **common_round_metrics(client_updates),
                **weight_diagnostics(weights),
                "far_alpha": 0.0,
                "far_server_clip_norm": radius,
                "far_server_clip_rate": float((factors < 1).double().mean()),
                "far_weight_l2_squared": float(weights.square().sum()),
                "far_max_weight": float(weights.max()),
                "far_noise_amplification_vs_uniform": float(
                    len(weights) * weights.square().sum()
                ),
                "far_weight_concentration_factor": float(
                    len(weights) * weights.square().sum()
                ),
                "far_attack_labels_visible_to_server_aggregate": False,
                "far_attack_config_visible_to_server_aggregate": False,
                "far_external_attack_diagnostics": True,
                "_far_external_weight_diagnostics_payload": {
                    "client_ids": ids,
                    "weights": weights.tolist(),
                    "contains_attack_labels": False,
                },
                "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
                "ldp_gradient_far_private_compute_device": "mps:0",
                "ldp_gradient_far_private_compute_devices": ["mps:0"],
                "ldp_gradient_far_server_post_clip_is_privacy_postprocessing": True,
                "ldp_gradient_far_effective_alpha": 0.0,
                "ldp_gradient_far_reference_noise_aware": False,
            }
            step = unflatten_update(float(config["far_server_lr"]) * aggregate, layout)
            result = AggregateResult(
                new_weights=apply_delta(global_model, step), metrics=diagnostics
            )
            result = self._add_local_dp_round_metrics(result, client_updates, config)
        histogram_eps = [
            float(metadata["privacy_dmd_histogram_epsilon"])
            for _, metadata, _ in client_updates
        ]
        gradient_eps = [
            float(metadata["privacy_gradient_epsilon"])
            for _, metadata, _ in client_updates
        ]
        result.metrics.update(
            {
                "dmd_server_mode": mode,
                "dmd_mu": float(config["dmd_mu"]),
                "dmd_far_enabled": mode == "far_rfa",
                "dmd_server_coefficient_kind": {
                    "uniform": "uniform",
                    "direct_rfa": "direct_rfa_irls",
                    "far_rfa": "far_softmax_raw_distance",
                }[mode],
                "privacy_target_epsilon": float(config["dmd_total_epsilon"]),
                "privacy_gradient_target_epsilon": float(config["target_epsilon"]),
                "privacy_gradient_epsilon_max": max(gradient_eps),
                "privacy_gradient_epsilon_mean": sum(gradient_eps) / len(gradient_eps),
                "privacy_dmd_histogram_epsilon": max(histogram_eps),
                "privacy_dmd_histogram_calls_per_client_max": max(
                    int(metadata["privacy_dmd_histogram_calls_total"])
                    for _, metadata, _ in client_updates
                ),
                "privacy_dmd_histogram_calls_this_round": sum(
                    int(metadata["privacy_dmd_histogram_calls_this_round"])
                    for _, metadata, _ in client_updates
                ),
                "privacy_dmd_budget_composition": "epsilon_histogram_plus_epsilon_gradient",
                "dmd_server_postprocessing_device": "cpu_float64",
            }
        )
        return result


__all__ = ["LDPGradientDMDCB", "geometric_median_with_weights"]
