"""Local-DP FAR built from a private gradient rather than a model delta.

This lane is deliberately separate from ``dpfar`` and ``dt_ldp_far``.  Each
client draws one public-cardinality mini-batch uniformly without replacement,
clips every per-example gradient, adds Gaussian noise to the clipped sum, and
releases the resulting average gradient.  The server clips the already-private
gradient, builds a robust noise-aware reference, and applies FAR to *raw*
distances.

Privacy is sample-level local DP with fixed-size replace-one adjacency.  The
server operations are deterministic post-processing of local-DP messages.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
from privacy.rdp import calibrate_sampled_without_replacement_gaussian_noise
from robustness.tensor_ops import stack_updates

from .base import register_algorithm
from .dp_references import _LocalDPSGDMixin, _dp_defaults
from .far import FAR
from .rcig_temporal_reference import (
    RCIGReferenceResult,
    RCIGTemporalConfig,
    TemporalRCIGReferenceState,
)


_RCIG_REFERENCE_NAMES = {
    "rcig_temporal",
    "rcig_temporal_full",
    "rcig_temporal_isotropic",
    "rcig_temporal_euclidean",
}


@lru_cache(maxsize=64)
def _calibrated_fixed_wor_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    sensitivity_multiplier: float,
) -> float:
    """Cache the same public fixed-WOR calibration used by every client."""

    return calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=target_epsilon,
        delta=delta,
        sampling_rate=sampling_rate,
        steps=steps,
        sensitivity_multiplier=sensitivity_multiplier,
    )


@register_algorithm("ldp_gradient_far")
class LDPGradientFAR(_LocalDPSGDMixin, FAR):
    """FedFDP-style private-gradient channel followed by raw-distance FAR."""

    description = (
        "Sample-level LDP private gradients with server clipping, a capped "
        "noise-aware robust reference, and raw-distance FAR."
    )

    def client_update(self, model, dataloader, state, config):
        return self._local_private_gradient_update(model, dataloader, state, config)

    @staticmethod
    def _uses_rcig(config: dict) -> bool:
        return str(config.get("robust_reference", "")).lower() in (
            _RCIG_REFERENCE_NAMES
        )

    @staticmethod
    def _rcig_mode(config: dict) -> str:
        reference = str(config.get("robust_reference", "")).lower()
        suffix_mode = {
            "rcig_temporal_full": "full",
            "rcig_temporal_isotropic": "isotropic",
            "rcig_temporal_euclidean": "euclidean",
        }.get(reference)
        configured = str(config.get("rcig_covariance_mode", suffix_mode or "full"))
        configured = configured.lower()
        if suffix_mode is not None and configured != suffix_mode:
            raise ValueError(
                "robust_reference suffix and rcig_covariance_mode disagree"
            )
        return configured

    @classmethod
    def _rcig_config(cls, config: dict, server_radius: float) -> RCIGTemporalConfig:
        old_window = int(config.get("rcig_old_window", 4))
        new_window = int(config.get("rcig_new_window", 4))
        if old_window != new_window:
            raise ValueError("RCIG v1 requires equal old and new window lengths")
        warmup_policy = str(config.get("rcig_warmup_policy", "uniform")).lower()
        if warmup_policy != "uniform":
            raise ValueError("RCIG v1 requires rcig_warmup_policy='uniform'")
        configured_influence_cap = config.get("rcig_reference_output_radius")
        influence_cap = (
            float(server_radius)
            if configured_influence_cap is None
            else float(configured_influence_cap)
        )
        mode = cls._rcig_mode(config)
        primary_threshold = float(config.get("rcig_innovation_threshold", 3.0))
        persistent_policy = str(config.get("rcig_persistent_policy", "rolling")).lower()
        recovery_threshold = config.get("rcig_recovery_threshold")
        if recovery_threshold is None and persistent_policy == "freeze_hysteresis":
            ratio = float(config.get("rcig_recovery_threshold_ratio", 0.75))
            if not 0.0 < ratio < 1.0:
                raise ValueError("rcig_recovery_threshold_ratio must lie in (0,1)")
            selected_threshold = {
                "full": primary_threshold,
                "isotropic": float(
                    config.get("rcig_isotropic_innovation_threshold", primary_threshold)
                ),
                "euclidean": float(
                    config.get("rcig_euclidean_innovation_threshold", primary_threshold)
                ),
            }[mode]
            recovery_threshold = ratio * selected_threshold
        return RCIGTemporalConfig(
            window_length=old_window,
            gate_window_length=int(config.get("rcig_gate_window", 8)),
            subspace_dimension=int(config.get("rcig_public_subspace_dimension", 64)),
            subspace_seed=int(config.get("rcig_public_subspace_seed", 0)),
            server_clip_norm=float(server_radius),
            influence_cap=influence_cap,
            minimum_accepted_mass=float(config.get("rcig_min_accepted_mass", 1.0)),
            gate_inner_mad_multiplier=float(config.get("rcig_gate_inner_mad", 2.5)),
            gate_outer_mad_multiplier=float(config.get("rcig_gate_outer_mad", 4.5)),
            gate_minimum_mad=float(config.get("rcig_gate_minimum_mad", 1e-8)),
            process_variance=float(config.get("rcig_process_variance", 0.0)),
            ridge=float(config.get("rcig_covariance_ridge", 1e-8)),
            innovation_threshold=primary_threshold,
            isotropic_innovation_threshold=float(
                config.get("rcig_isotropic_innovation_threshold", primary_threshold)
            ),
            euclidean_innovation_threshold=float(
                config.get("rcig_euclidean_innovation_threshold", primary_threshold)
            ),
            covariance_mode=mode,
            persistent_policy=persistent_policy,
            recovery_threshold=(
                None if recovery_threshold is None else float(recovery_threshold)
            ),
            recovery_patience=int(config.get("rcig_recovery_patience", 2)),
        )

    @staticmethod
    def _registered_noise_scale(client_id: int, config: dict) -> float:
        """Read one client channel from the immutable server registry."""

        configured = config.get("privacy_noise_multiplier_scale_by_client")
        if configured is None:
            value = 1.0
        elif isinstance(configured, dict):
            if str(client_id) in configured:
                value = configured[str(client_id)]
            elif client_id in configured:
                value = configured[client_id]
            else:
                raise ValueError(
                    "authenticated noise registry does not cover client " f"{client_id}"
                )
        elif isinstance(configured, (list, tuple)):
            if not 0 <= client_id < len(configured):
                raise ValueError(
                    "authenticated noise registry does not cover client " f"{client_id}"
                )
            value = configured[client_id]
        else:
            raise TypeError(
                "privacy_noise_multiplier_scale_by_client must be a mapping or list"
            )
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("authenticated client noise scales must be finite and > 0")
        if config.get("target_epsilon") is not None and scale < 1.0:
            raise ValueError("target-epsilon client noise scales must be at least one")
        return scale

    @classmethod
    def _public_rcig_variances(cls, client_updates, config: dict) -> torch.Tensor:
        """Reconstruct nominal channel variances from server-owned inputs.

        The returned values depend only on the immutable mechanism config and
        the authenticated ``client_id`` registry.  Client-supplied variance,
        noise-multiplier, and batch fields are *not* used to construct the
        covariance; they are checked afterwards and any disagreement fails
        closed.  Byzantine clients therefore receive their nominal registered
        channel covariance just like every other ID, without revealing or
        consulting an attack label.
        """

        if config.get("rcig_covariance_registry") != "authenticated_public_mechanism":
            raise ValueError(
                "RCIG covariance requires the authenticated public mechanism registry"
            )
        if str(config.get("sampling_scheme", "")).lower() != (
            "fixed_without_replacement"
        ):
            raise ValueError("RCIG v2 covariance requires fixed-WOR sampling")
        if str(config.get("privacy_adjacency", "")).lower() != "replace_one":
            raise ValueError("RCIG v2 covariance requires replace-one adjacency")
        if not bool(config.get("enable_dp", True)):
            raise ValueError("RCIG v2 covariance requires an enabled DP channel")

        clip_norm = float(config.get("clip_norm", 0.0))
        batch_size = int(config.get("fixed_batch_size", 0))
        public_size = int(config.get("privacy_public_dataset_size", 0))
        steps_per_round = int(config.get("fixed_steps_per_round", 0))
        total_rounds = int(config.get("privacy_num_rounds", 0))
        sensitivity_multiplier = 2.0
        if clip_norm <= 0.0 or batch_size <= 0 or public_size <= 0:
            raise ValueError("RCIG covariance requires positive C, B, and N_public")
        if batch_size > public_size:
            raise ValueError("RCIG fixed batch size cannot exceed N_public")
        if steps_per_round != 1 or total_rounds <= 0:
            raise ValueError(
                "RCIG v2 requires one fixed-WOR release per positive public round"
            )
        sampling_rate = batch_size / float(public_size)
        configured_rate = float(
            config.get("privacy_sampling_rate_override", sampling_rate)
        )
        if not math.isclose(configured_rate, sampling_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("RCIG public B/N sampling rate is inconsistent")

        target = config.get("target_epsilon")
        if target is None:
            base_noise = float(config.get("noise_multiplier", 0.0))
        else:
            base_noise = _calibrated_fixed_wor_noise_multiplier(
                float(target),
                float(config.get("delta", 1e-5)),
                sampling_rate,
                total_rounds * steps_per_round,
                sensitivity_multiplier,
            )
        if not math.isfinite(base_noise) or base_noise <= 0.0:
            raise ValueError("RCIG nominal noise multiplier must be finite and > 0")

        expected_clients = int(config.get("expected_num_clients", len(client_updates)))
        if expected_clients <= 0 or len(client_updates) != expected_clients:
            raise ValueError("RCIG authenticated client registry/cohort size mismatch")
        values: list[float] = []
        seen_ids: set[int] = set()
        for _, metadata, _ in client_updates:
            client_id = int(metadata["client_id"])
            if client_id in seen_ids or not 0 <= client_id < expected_clients:
                raise ValueError(
                    "RCIG covariance requires unique registered client ids in [0,n)"
                )
            seen_ids.add(client_id)
            scale = cls._registered_noise_scale(client_id, config)
            noise_multiplier = base_noise * scale
            variance = (noise_multiplier * clip_norm / float(batch_size)) ** 2

            # Fail closed if the authenticated transport metadata does not
            # match the server registry.  None of these fields determines the
            # value appended to ``values`` above.
            scalar_checks = {
                "privacy_upload_noise_variance_per_coordinate": variance,
                "privacy_noise_multiplier": noise_multiplier,
                "privacy_fixed_batch_size": float(batch_size),
                "privacy_noise_multiplier_scale_public": scale,
                "privacy_sensitivity_multiplier": sensitivity_multiplier,
                "dataset_size": float(public_size),
            }
            for name, expected_value in scalar_checks.items():
                observed = metadata.get(name)
                if observed is None or not math.isclose(
                    float(observed),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=max(1e-15, 1e-12 * max(1.0, abs(expected_value))),
                ):
                    raise ValueError(
                        f"RCIG client metadata {name} disagrees with the "
                        "authenticated server registry"
                    )
            if (
                metadata.get("privacy_sampling_scheme") != ("fixed_without_replacement")
                or metadata.get("privacy_adjacency") != "replace_one"
            ):
                raise ValueError(
                    "RCIG client sampler/adjacency metadata disagrees with the "
                    "authenticated server registry"
                )
            values.append(variance)
        return torch.tensor(values, dtype=torch.float64)

    def _far_reference_override(
        self,
        *,
        vectors,
        score_vectors,
        layout,
        client_updates,
        server_clip_factors,
        round_num,
        config,
        output_radius,
    ):
        if not self._uses_rcig(config):
            return None
        if score_vectors.shape != vectors.shape:
            raise ValueError(
                "RCIG uses a separate public gate subspace; FAR distances must "
                "remain full-dimensional"
            )
        if bool(config.get("rcig_oracle_separation_required", False)):
            forbidden = {
                "local_dp_noise_free_update_oracle",
                "dp_noise_norm_mean",
            }
            leaked = sorted(
                {
                    name
                    for _, metadata, _ in client_updates
                    for name in forbidden
                    if name in metadata
                }
            )
            if leaked:
                raise ValueError(
                    "RCIG server received simulator-only oracle fields: "
                    + ", ".join(leaked)
                )
        server_radius = float(config["far_server_clip_norm"])
        rcig_config = self._rcig_config(config, server_radius)
        if output_radius is None or not math.isclose(
            float(output_radius), rcig_config.influence_cap, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "far_reference_output_clip_norm and rcig_reference_output_radius "
                "must match"
            )
        signature = (
            tuple(layout.keys),
            tuple(tuple(shape) for shape in layout.shapes),
            len(client_updates),
            rcig_config,
        )
        if not hasattr(self, "_rcig_state"):
            self._rcig_state = TemporalRCIGReferenceState(rcig_config)
            self._rcig_signature = signature
        elif signature != self._rcig_signature:
            raise ValueError("RCIG public configuration changed during a run")

        client_ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
        compute_devices = {
            str(metadata.get("privacy_compute_device", "missing"))
            for _, metadata, _ in client_updates
        }
        if compute_devices not in ({"mps"}, {"mps:0"}):
            raise ValueError(
                "RCIG scientific runs require every private gradient to be "
                f"computed on MPS, got {sorted(compute_devices)}"
            )
        public_variances = self._public_rcig_variances(client_updates, config).to(
            vectors
        )
        snapshot = self._rcig_state.make_snapshot(
            round_num=int(round_num),
            client_ids=client_ids,
            clipped_vectors=vectors,
            server_clip_factors=server_clip_factors,
            public_noise_variances=public_variances,
        )
        reference_result = self._rcig_state.reference_for_round(
            round_num=int(round_num),
            dimension=int(score_vectors.shape[1]),
            device=score_vectors.device,
            dtype=score_vectors.dtype,
        )
        if not reference_result.ready:
            # FAR reads this local resolved config only after reference
            # construction.  Uniform weights make the deterministic cold-start
            # reference irrelevant without reading the current round into F_t.
            config["far_alpha"] = 0.0
        self._rcig_pending_snapshot = snapshot
        self._rcig_pending_reference_result = reference_result
        self._rcig_last_deployed_reference = reference_result.reference.detach().clone()
        diagnostics = dict(reference_result.diagnostics)
        selected_statistic = diagnostics.get("rcig_standardized_innovation")
        if selected_statistic is None:
            selected_statistic = diagnostics.get("rcig_innovation_norm")
        diagnostics.update(
            {
                "rcig_history_ready": bool(reference_result.ready),
                "rcig_warmup": not bool(reference_result.ready),
                "rcig_reference_mode": rcig_config.covariance_mode,
                "rcig_innovation_stat": selected_statistic,
                "rcig_reference_strictly_past": True,
                "rcig_public_subspace_dimension": rcig_config.subspace_dimension,
                "rcig_covariance_psd_certified": bool(
                    not reference_result.ready
                    or (
                        diagnostics["rcig_older_view"]["covariance_psd"]
                        and diagnostics["rcig_newer_view"]["covariance_psd"]
                    )
                ),
                "rcig_calibration_mode": bool(
                    config.get("rcig_calibration_mode", False)
                ),
                "rcig_private_gradient_mps_fraction": 1.0,
                "rcig_private_gradient_compute_device": next(iter(compute_devices)),
                "rcig_server_aggregation_device": str(vectors.device),
                "rcig_server_aggregation_dtype": str(vectors.dtype),
                "rcig_public_variance_registry_verified": True,
                "rcig_public_variance_source": (
                    "server_config_and_authenticated_client_id"
                ),
                "rcig_client_variance_metadata_used_for_construction": False,
                "rcig_client_variance_metadata_consistency_checked": True,
                "rcig_post_server_clip_covariance_is_delta_method_proxy": True,
            }
        )
        return reference_result.reference, diagnostics

    @staticmethod
    def _rcig_evaluation_payload(
        reference_result: RCIGReferenceResult,
        server_radius: float,
    ) -> dict | None:
        """Export private-transcript candidates to the evaluation harness.

        This payload contains no clean gradient, realised DP noise or attack
        label.  ``run_experiment`` removes it before metrics persistence and
        compares it with simulator-only oracles outside ``server_aggregate``.
        """

        if not reference_result.ready or not reference_result.candidate_references:
            return None
        return {
            "candidate_references": {
                name: value.detach().to(dtype=torch.float64, device="cpu").clone()
                for name, value in reference_result.candidate_references.items()
            },
            "deployed_reference": reference_result.reference.detach()
            .to(dtype=torch.float64, device="cpu")
            .clone(),
            "server_clip_norm": float(server_radius),
            "deployment_round": int(reference_result.deployment_round),
            "contains_clean_data": False,
            "contains_realised_noise": False,
            "contains_attack_labels": False,
        }

    @staticmethod
    def _resolved_tilt(config: dict, num_clients: int, server_radius: float):
        if num_clients < 2:
            raise ValueError("LDP-gradient-FAR requires at least two clients")
        kappa_w = float(config.get("kappa_w", 2.0))
        if not 1.0 <= kappa_w < float(num_clients):
            raise ValueError("kappa_w must lie in [1,n)")
        # Both the server-clipped upload and the projected reference belong to
        # B(0,U), hence every raw distance is in [0,2U].
        public_distance_range = 2.0 * float(server_radius)
        alpha_cap = (
            math.log(kappa_w * (num_clients - 1) / (num_clients - kappa_w))
            / public_distance_range
        )
        requested = float(config.get("far_alpha", alpha_cap))
        policy = str(config.get("tilt_bound_policy", "diagnostic")).lower()
        if policy not in {"diagnostic", "error", "clip"}:
            raise ValueError(
                "tilt_bound_policy must be 'diagnostic', 'error', or 'clip'"
            )
        if abs(requested) > alpha_cap + 1e-12:
            if policy == "error":
                raise ValueError(
                    f"|far_alpha|={abs(requested):.6g} exceeds the certified "
                    f"raw-distance cap {alpha_cap:.6g}"
                )
            effective = (
                math.copysign(alpha_cap, requested) if policy == "clip" else requested
            )
        else:
            effective = requested
        return requested, effective, alpha_cap, public_distance_range

    def server_aggregate(self, global_model, client_updates, round_num, config):
        if not client_updates:
            raise ValueError("LDP-gradient-FAR received no client update")
        uses_rcig = self._uses_rcig(config)
        if uses_rcig and int(round_num) == 0:
            # One algorithm instance can be reused by a test harness.  A new
            # round-zero invocation must never inherit a previous transcript.
            for attribute in (
                "_rcig_state",
                "_rcig_signature",
                "_rcig_pending_snapshot",
                "_rcig_pending_reference_result",
                "_rcig_last_deployed_reference",
            ):
                if hasattr(self, attribute):
                    delattr(self, attribute)
        if str(config.get("far_score_mode", "raw_distance")).lower() != "raw_distance":
            raise ValueError(
                "LDP-gradient-FAR intentionally uses raw distances; bounded score "
                "transformations are not accepted in this lane"
            )
        if str(config.get("noise_score_standardization", "none")).lower() != "none":
            raise ValueError(
                "Noise awareness belongs to F in this lane, not to a transformed "
                "or bounded FAR score"
            )
        if bool(config.get("noise_score_include_server_contraction", False)):
            raise ValueError(
                "The certified noise-aware reference must use public pre-server-"
                "clip variances; data-dependent clipping factors cannot define its "
                "weights"
            )
        if str(config.get("score_subspace_mode", "full")).lower() != "full":
            raise ValueError(
                "The first private-gradient FAR protocol fixes the reference and "
                "distances in the full released-gradient space"
            )
        server_radius = config.get("far_server_clip_norm")
        if server_radius is None:
            raise ValueError(
                "far_server_clip_norm is required to bound Byzantine influence and "
                "the raw FAR distance range"
            )
        server_radius = float(server_radius)
        if server_radius <= 0.0:
            raise ValueError("far_server_clip_norm must be positive")
        expected = config.get("expected_num_clients")
        if expected is not None and len(client_updates) != int(expected):
            raise ValueError(
                f"expected {int(expected)} clients, received {len(client_updates)}"
            )
        releases = {
            bool(metadata.get("privacy_gradient_release", False))
            for _, metadata, _ in client_updates
        }
        if releases != {True}:
            raise ValueError("every client message must be a private gradient release")
        private_compute_devices = {
            str(metadata.get("privacy_compute_device", "missing"))
            for _, metadata, _ in client_updates
        }
        required_private_device = config.get("rcig_required_private_device")
        if required_private_device is not None:
            required_private_device = str(required_private_device).lower()
            allowed_devices = (
                {"mps", "mps:0"}
                if required_private_device == "mps"
                else {required_private_device}
            )
            if not private_compute_devices or not private_compute_devices.issubset(
                allowed_devices
            ):
                raise ValueError(
                    "private-gradient compute-device audit failed: required "
                    f"{required_private_device}, got {sorted(private_compute_devices)}"
                )

        requested, effective, alpha_cap, public_range = self._resolved_tilt(
            config, len(client_updates), server_radius
        )
        resolved = dict(config)
        resolved.update(
            {
                "far_alpha": effective,
                "far_score_mode": "raw_distance",
                "noise_score_standardization": "none",
                "noise_score_include_server_contraction": False,
                "far_public_score_range": public_range,
                "far_reference_output_clip_norm": server_radius,
                "anchor_clip_norm": server_radius,
                "reference_clip_radius": float(
                    config.get("reference_clip_radius", server_radius)
                ),
            }
        )
        try:
            result = FAR.server_aggregate(
                self, global_model, client_updates, round_num, resolved
            )
            if uses_rcig:
                reference_result = self._rcig_pending_reference_result
                if bool(config.get("enable_oracle_diagnostics", False)):
                    evaluation_payload = self._rcig_evaluation_payload(
                        reference_result, server_radius
                    )
                    if evaluation_payload is not None:
                        result.metrics["_rcig_evaluation_payload"] = evaluation_payload
                self._rcig_state.commit_snapshot(
                    self._rcig_pending_snapshot,
                    reference_result=(
                        reference_result if reference_result.ready else None
                    ),
                )
        finally:
            if uses_rcig:
                for attribute in (
                    "_rcig_pending_snapshot",
                    "_rcig_pending_reference_result",
                ):
                    if hasattr(self, attribute):
                        delattr(self, attribute)
        actual_effective = float(result.metrics.get("far_alpha", effective))
        result.metrics.update(
            {
                "ldp_gradient_far_protocol": "fixed_wor_private_gradient_v1",
                "ldp_gradient_far_score": "raw_distance",
                "ldp_gradient_far_requested_alpha": requested,
                "ldp_gradient_far_effective_alpha": actual_effective,
                "ldp_gradient_far_post_warmup_alpha": effective,
                "ldp_gradient_far_alpha_cap": alpha_cap,
                "ldp_gradient_far_alpha_cap_exceeded": (
                    abs(requested) > alpha_cap + 1e-12
                ),
                "ldp_gradient_far_tilt_bound_policy": str(
                    config.get("tilt_bound_policy", "diagnostic")
                ).lower(),
                "ldp_gradient_far_alpha_cap_role": (
                    "optional_robustness_guardrail_not_ldp_requirement"
                ),
                "ldp_gradient_far_public_distance_range": public_range,
                "ldp_gradient_far_server_post_clip_is_privacy_postprocessing": True,
                "ldp_gradient_far_private_compute_devices": sorted(
                    private_compute_devices
                ),
                "ldp_gradient_far_private_compute_device": (
                    next(iter(private_compute_devices))
                    if len(private_compute_devices) == 1
                    else "mixed:" + ",".join(sorted(private_compute_devices))
                ),
                "ldp_gradient_far_private_gradient_mps_fraction": float(
                    all(
                        device in {"mps", "mps:0"} for device in private_compute_devices
                    )
                ),
                "ldp_gradient_far_reference_noise_aware": str(
                    resolved.get("robust_reference", "")
                ).lower()
                in {
                    "noise_aware_centered_clipping",
                    "noise_aware_cc",
                    "na_cc",
                    "f_na_cc",
                    *_RCIG_REFERENCE_NAMES,
                },
            }
        )
        return self._add_local_dp_round_metrics(result, client_updates, resolved)

    def get_default_config(self):
        config = {
            **FAR.get_default_config(self),
            **_dp_defaults(),
        }
        config.update(
            {
                "sampling_scheme": "fixed_without_replacement",
                "privacy_adjacency": "replace_one",
                "privacy_public_dataset_size": None,
                "privacy_sampling_rate_override": 0.05,
                "fixed_batch_size": None,
                "fixed_steps_per_round": 1,
                "local_epochs": 1,
                "private_aux_loss": False,
                "far_update_mode": "single_step_gradient",
                "far_server_lr": 0.01,
                "far_score_mode": "raw_distance",
                "far_distance_clip": None,
                "noise_score_standardization": "none",
                "noise_score_include_server_contraction": False,
                "score_subspace_mode": "full",
                "far_server_clip_norm": 0.1,
                "robust_reference": "noise_aware_centered_clipping",
                "reference_clip_radius": 0.1,
                "noise_aware_reference_kappa": 2.0,
                "noise_aware_reference_variance_floor": 1e-12,
                "kappa_w": 2.0,
                "far_alpha": 0.1,
                # In local DP the server-side weighting is post-processing of
                # already-private messages.  The kappa-derived alpha bound is
                # therefore an optional robustness/variance guardrail, not a
                # privacy requirement.  Keep it as a diagnostic by default.
                "tilt_bound_policy": "diagnostic",
                "anchor_update_rate": 0.1,
                "anchor_clip_norm": 0.1,
                "suppress_private_client_diagnostics": True,
                "rcig_covariance_mode": "full",
                "rcig_calibration_mode": False,
                "rcig_gate_window": 8,
                "rcig_old_window": 4,
                "rcig_new_window": 4,
                "rcig_public_subspace_dimension": 64,
                "rcig_public_subspace_seed": 20260911,
                "rcig_gate_inner_mad": 2.5,
                "rcig_gate_outer_mad": 4.5,
                "rcig_gate_minimum_mad": 1e-8,
                "rcig_min_accepted_mass": 1.0,
                "rcig_covariance_ridge": 1e-8,
                "rcig_process_variance": 0.0,
                "rcig_innovation_threshold": 3.0,
                "rcig_isotropic_innovation_threshold": 3.0,
                "rcig_euclidean_innovation_threshold": 0.1,
                "rcig_warmup_policy": "uniform",
                "rcig_reference_output_radius": None,
                "rcig_persistent_policy": "rolling",
                "rcig_recovery_threshold_ratio": 0.75,
                "rcig_recovery_patience": 2,
                "rcig_covariance_registry": "authenticated_public_mechanism",
                "external_attack_diagnostics": False,
            }
        )
        return config


__all__ = ["LDPGradientFAR"]
