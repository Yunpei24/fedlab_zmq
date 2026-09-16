"""Deployed recent-window control for the independent RCIG batch screen.

The initial client gate, disjoint past windows, twelve-round uniform warmup,
and output projection are exactly those of RCIG.  This control deploys the
recent view itself: no innovation test, fusion, freeze, or recovery is run.
The covariance computations retained below are diagnostics only.  Import this
module explicitly from the screen entrypoint; the locked registry imports and
RCIG implementation are deliberately unchanged.
"""

from __future__ import annotations

import torch

from .base import register_algorithm
from .gaussian_aware_reference_k7_rcig import clip_l2
from .ldp_gradient_far import LDPGradientFAR
from .rcig_temporal_reference import (
    RCIGReferenceResult,
    RCIGTemporalConfig,
    TemporalRCIGReferenceState,
    _coordinate_hash,
    _public_coordinates,
)

_THRESHOLD_KEYS = (
    "rcig_innovation_threshold",
    "rcig_isotropic_innovation_threshold",
    "rcig_euclidean_innovation_threshold",
    "rcig_recovery_threshold",
)


class RecentTemporalReferenceState(TemporalRCIGReferenceState):
    """Reuse RCIG's authenticated snapshots and gate, never its correction."""

    def __init__(self, config: RCIGTemporalConfig):
        if config.persistent_policy != "rolling":
            raise ValueError("recent-only state must have no persistent freeze policy")
        super().__init__(config)

    def reference_for_round(self, *, round_num, dimension, device, dtype):
        # Same causal preconditions as TemporalRCIGReferenceState, without
        # calling its reference_for_round (which would run innovation fusion).
        if not isinstance(round_num, int) or isinstance(round_num, bool):
            raise TypeError("round_num must be an integer")
        if round_num < 0:
            raise ValueError("round_num must be non-negative")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        dimension = int(dimension)
        if dimension < self.config.subspace_dimension:
            raise ValueError("dimension is smaller than the public subspace")
        if self._dimension is not None and dimension != self._dimension:
            raise ValueError("requested reference dimension changed during the run")
        if self._history and round_num != self._history[-1].round_num + 1:
            raise ValueError("reference must target the next uncommitted round")
        if self._persistent_frozen or self._recovery_count:
            raise RuntimeError("recent-only control cannot hold frozen/recovery state")

        common = {
            "rcig_ready": self.ready,
            "rcig_cold_start": not self.ready,
            "rcig_past_only": True,
            "rcig_current_round_read": False,
            "rcig_history_length": len(self._history),
            "rcig_required_history": self.config.required_history,
            "rcig_window_length": self.config.window_length,
            "rcig_gate_window_length": self.config.gate_window_length,
            "rcig_covariance_mode": self.config.covariance_mode,
            "rcig_subspace_dimension": self.config.subspace_dimension,
            "rcig_full_dimension": dimension,
            "rcig_subspace_seed": self.config.subspace_seed,
            "rcig_coordinate_hash": _coordinate_hash(
                _public_coordinates(
                    dimension,
                    self.config.subspace_dimension,
                    self.config.subspace_seed,
                )
            ),
            "rcig_public_variance_provenance": self.config.public_variance_provenance,
            "rcig_diagnostics_use_realised_noise": False,
            "rcig_diagnostics_use_attack_labels": False,
            "rcig_diagnostics_use_clean_gradients": False,
            "rcig_temporal_correction_enabled": False,
            "rcig_innovation_test_enabled": False,
            "rcig_calibrated_threshold_required": False,
            "rcig_covariance_used_for_reference": False,
            "rcig_reference_construction": "projected_recent_view_with_frozen_client_gate",
            "rcig_persistent_policy": "recent_only",
            "rcig_persistent_frozen_before_round": False,
            "rcig_persistent_frozen_after_commit": False,
            "rcig_recovery_count_before_round": 0,
            "rcig_recovery_count_after_commit": 0,
            "rcig_gate_active": False,
            "rcig_persistent_action": "recent_only",
            "rcig_deployed_candidate": "identity_new",
            "rcig_selected_candidate_drives_decision": True,
            "rcig_counterfactual_candidates_drive_decision": False,
            "rcig_paired_counterfactual_metrics_drive_deployment": False,
            "rcig_newer_view_trust": 1.0,
        }
        if not self.ready:
            return RCIGReferenceResult(
                reference=torch.zeros(dimension, device=device, dtype=dtype),
                ready=False,
                diagnostics={
                    **common,
                    "rcig_cold_start_policy": "zero_reference_uniform_weights_required",
                },
                deployment_round=round_num,
            )

        if self._frozen_gate is None:
            source = self._history[
                -self.config.required_history : -2 * self.config.window_length
            ]
            gate, gate_metrics = self._compute_gate(source)
            gate_rounds = [item.round_num for item in source]
        else:
            gate = self._frozen_gate
            gate_metrics = {
                "gate_mass": float(gate.sum().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "gate_mean": float(gate.mean().item()),
                "gate_zero_count": int((gate == 0.0).sum().item()),
            }
            gate_rounds = list(self._gate_source_range or ())

        window = self.config.window_length
        old_snapshots = self._history[-2 * window : -window]
        new_snapshots = self._history[-window:]
        # Older view/covariances remain available for auditing, but neither
        # determines the deployed reference or any accept/reject decision.
        older, old_covariance, old_metrics = self._view_and_covariance(
            old_snapshots, gate
        )
        newer, new_covariance, new_metrics = self._view_and_covariance(
            new_snapshots, gate
        )
        reference = clip_l2(newer, float(self.config.influence_cap)).to(
            device=device, dtype=dtype
        )
        diagnostics = {
            **common,
            **{f"rcig_{key}": value for key, value in gate_metrics.items()},
            "rcig_gate_frozen": self._frozen_gate is not None,
            "rcig_gate_source_round_min": min(gate_rounds),
            "rcig_gate_source_round_max": max(gate_rounds),
            "rcig_older_round_min": old_snapshots[0].round_num,
            "rcig_older_round_max": old_snapshots[-1].round_num,
            "rcig_newer_round_min": new_snapshots[0].round_num,
            "rcig_newer_round_max": new_snapshots[-1].round_num,
            "rcig_windows_disjoint": True,
            "rcig_reference_norm": float(torch.linalg.vector_norm(reference).item()),
            "rcig_reference_norm_cap": float(self.config.influence_cap),
            "rcig_reference_norm_cap_respected": True,
            "rcig_older_view": old_metrics,
            "rcig_newer_view": new_metrics,
            "rcig_max_covariance_anisotropy_ratio": max(
                old_metrics["covariance_anisotropy_ratio"],
                new_metrics["covariance_anisotropy_ratio"],
            ),
            "rcig_local_dp_effect": "unchanged_post_processing_of_private_transcript",
        }
        return RCIGReferenceResult(
            reference=reference,
            ready=True,
            diagnostics=diagnostics,
            older_view=older.to(device=device, dtype=dtype),
            newer_view=newer.to(device=device, dtype=dtype),
            covariance_older=old_covariance,
            covariance_newer=new_covariance,
            candidate_references={"identity_new": reference.detach().clone()},
            deployment_round=round_num,
            _persistent_transition=(reference.detach().clone(), False, 0),
        )


@register_algorithm("ldp_gradient_far_recent")
class LDPGradientFARRecent(LDPGradientFAR):
    """Private-gradient FAR with a genuinely deployed recent-only reference."""

    description = "LDP-gradient FAR, frozen client gate and uncorrected recent view"

    @classmethod
    def _rcig_config(cls, config, server_radius):
        if config.get("robust_reference") not in {
            "rcig_temporal",
            "rcig_temporal_full",
        }:
            raise ValueError("recent control requires the temporal full reference path")
        if config.get("rcig_persistent_policy", "recent_only") != "recent_only":
            raise ValueError(
                "recent control requires rcig_persistent_policy=recent_only"
            )
        if cls._rcig_mode(config) != "full":
            raise ValueError("recent control requires full covariance diagnostics")
        if any(config.get(key) is not None for key in _THRESHOLD_KEYS):
            raise ValueError(
                "recent control has no calibrated innovation/recovery threshold"
            )
        if config.get("rcig_threshold_artifact_sha256") is not None:
            raise ValueError("recent control cannot consume a threshold artifact")
        if bool(config.get("rcig_calibration_mode", False)):
            raise ValueError("recent control is not an RCIG calibration run")
        for key in ("rcig_gate_window", "rcig_old_window", "rcig_new_window"):
            if int(config.get(key, 4)) != 4:
                raise ValueError("recent screen requires gate4 + old4 + new4 warmup")
        adapted = dict(config)
        adapted.update(
            {
                "rcig_gate_window": 4,
                "rcig_old_window": 4,
                "rcig_new_window": 4,
                # RCIGTemporalConfig validates these legacy fields. They are
                # inert structural placeholders; the recent state NEVER
                # reads a threshold or calls an innovation/fusion routine.
                "rcig_innovation_threshold": 1.0,
                "rcig_isotropic_innovation_threshold": 1.0,
                "rcig_euclidean_innovation_threshold": 1.0,
                "rcig_recovery_threshold": None,
                "rcig_persistent_policy": "rolling",
            }
        )
        return super()._rcig_config(adapted, server_radius)

    def client_update(self, model, dataloader, state, config):
        if str(config.get("device", "")) not in {"mps", "mps:0"}:
            raise ValueError("recent private gradients require MPS")
        if next(model.parameters()).device.type != "mps":
            raise ValueError("recent private-gradient model must be on MPS")
        return super().client_update(model, dataloader, state, config)

    def _far_reference_override(self, **kwargs):
        config = kwargs["config"]
        radius = float(config["far_server_clip_norm"])
        temporal_config = self._rcig_config(config, radius)
        if not hasattr(self, "_rcig_state"):
            layout = kwargs["layout"]
            self._rcig_state = RecentTemporalReferenceState(temporal_config)
            self._rcig_signature = (
                tuple(layout.keys),
                tuple(tuple(shape) for shape in layout.shapes),
                len(kwargs["client_updates"]),
                temporal_config,
            )
        elif not isinstance(self._rcig_state, RecentTemporalReferenceState):
            raise RuntimeError("recent control cannot reuse a corrected RCIG state")
        return super()._far_reference_override(**kwargs)

    def server_aggregate(self, global_model, client_updates, round_num, config):
        # Validate even if an unsupported reference would bypass the override.
        self._rcig_config(config, float(config["far_server_clip_norm"]))
        if not bool(config.get("rcig_oracle_separation_required", False)):
            raise ValueError(
                "recent control requires the strict external oracle boundary"
            )
        if not bool(config.get("external_attack_diagnostics", False)):
            raise ValueError("recent control requires external attack diagnostics")
        result = super().server_aggregate(
            global_model, client_updates, round_num, config
        )
        result.metrics.update(
            {
                "ldp_gradient_far_reference_noise_aware": False,
                "ldp_gradient_far_reference_control": "recent_only",
                "rcig_reference_mode": "recent_only",
                "rcig_temporal_correction_enabled": False,
                "rcig_innovation_test_enabled": False,
                "rcig_calibrated_threshold_required": False,
            }
        )
        return result

    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            {
                "robust_reference": "rcig_temporal_full",
                "rcig_gate_window": 4,
                "rcig_persistent_policy": "recent_only",
                "rcig_oracle_separation_required": True,
                "external_attack_diagnostics": True,
                "rcig_required_private_device": "mps",
            }
        )
        for key in _THRESHOLD_KEYS:
            config[key] = None
        return config


__all__ = ["LDPGradientFARRecent", "RecentTemporalReferenceState"]
