"""Sensitivity-Controlled FAR and its partial-training extension.

This module implements the current conceptual framework without the two
mechanisms that were removed from the research design: there is no capped-
simplex projection and no FedAvg shrinkage.  Influence is controlled by:

1. trusted server-side user-level clipping;
2. a bounded distance score in ``[0, 1]``;
3. an analytical bound on the softmax tilt ``alpha``;
4. an explicit sensitivity mode for calibrating central Gaussian noise.

The improved ``O(kappa_w C / n)`` sensitivity is conditional on a proved
replace-one stability bound for the chosen robust reference ``F``.  Until
that proof is available, ``conservative_2C`` is the safe default.
"""

from __future__ import annotations

import math

import torch

from metrics.robustness import weight_diagnostics
from privacy.rdp import RDPAccountant, calibrate_sampled_gaussian_noise
from robustness.aggregators import aggregate_vectors
from robustness.tensor_ops import stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .fedpart import FedPart
from .reference_utils import apply_delta, common_round_metrics


def alpha_max_for_weight_factor(n: int, kappa_w: float) -> float:
    """Largest non-negative tilt guaranteeing ``max_i q_i <= kappa_w/n``.

    The guarantee uses scores in ``[0,1]``.  ``kappa_w=1`` gives uniform
    weights, while larger values permit controlled non-uniformity.
    """

    if n < 2:
        raise ValueError("At least two clients are required")
    if not 1.0 <= kappa_w < n:
        raise ValueError("kappa_w must satisfy 1 <= kappa_w < n")
    return math.log(kappa_w * (n - 1) / (n - kappa_w))


def bounded_distance_scores(
    distances: torch.Tensor,
    distance_clip: float,
    mode: str = "hard_clip",
) -> torch.Tensor:
    """Map robust-reference distances to a public range ``[0,1]``."""

    if distance_clip <= 0:
        raise ValueError("distance_clip must be positive")
    normalized = distances / float(distance_clip)
    if mode == "hard_clip":
        return normalized.clamp(0.0, 1.0)
    if mode == "tanh":
        return torch.tanh(normalized).clamp(0.0, 1.0)
    raise ValueError("score_mode must be 'hard_clip' or 'tanh'")


def clip_rows(vectors: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Trusted user-level clipping of every received client vector."""

    if clip_norm <= 0:
        raise ValueError("user_clip_norm must be positive")
    norms = torch.linalg.vector_norm(vectors, dim=1)
    factors = (float(clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)
    return vectors * factors[:, None], factors


class _SCFARServerMixin:
    """Sensitivity-controlled server rule shared by full and partial variants."""

    @staticmethod
    def controlled_weights(scores: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.softmax(float(alpha) * scores, dim=0)

    def _alpha(self, n: int, config: dict) -> tuple[float, float, bool]:
        requested = float(config.get("far_alpha", 0.1))
        if requested < 0:
            raise ValueError("Sensitivity-Controlled FAR currently assumes alpha >= 0")
        kappa_w = float(config.get("kappa_w", 2.0))
        maximum = alpha_max_for_weight_factor(n, kappa_w)
        policy = str(config.get("alpha_bound_policy", "clip"))
        clipped = requested > maximum
        if clipped and policy == "error":
            raise ValueError(
                f"far_alpha={requested} exceeds alpha_max={maximum:.6g} "
                f"for n={n}, kappa_w={kappa_w}"
            )
        if policy not in {"clip", "error", "diagnostic_only"}:
            raise ValueError("alpha_bound_policy must be clip, error or diagnostic_only")
        effective = min(requested, maximum) if policy == "clip" else requested
        return effective, maximum, clipped

    def _sensitivity(
        self,
        *,
        n: int,
        clip_norm: float,
        distance_clip: float,
        alpha: float,
        config: dict,
    ) -> tuple[float, str]:
        mode = str(config.get("sensitivity_mode", "conservative_2C"))
        if mode == "conservative_2C":
            return 2.0 * clip_norm, mode
        if mode != "proved_reference_bound":
            raise ValueError(
                "sensitivity_mode must be conservative_2C or proved_reference_bound"
            )
        stability = config.get("reference_stability_constant")
        if stability is None:
            raise ValueError(
                "proved_reference_bound requires reference_stability_constant "
                "from a proved bound delta_F <= L_ref*C/n"
            )
        kappa_w = float(config.get("kappa_w", 2.0))
        rho = distance_clip / clip_norm
        sensitivity = (2.0 * clip_norm / n) * (
            kappa_w * (1.0 + alpha) + alpha * float(stability) / rho
        )
        return float(sensitivity), mode

    def _noise_multiplier(self, config: dict) -> float:
        target = config.get("target_epsilon")
        if target is None:
            return float(config.get("central_noise_multiplier", 1.0))
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError("target_epsilon requires privacy_num_rounds")
        return calibrate_sampled_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            sampling_rate=1.0,
            steps=int(total_rounds),
        )

    def server_aggregate(self, global_model, client_updates, round_num, config):
        updates = [update for update, _, _ in client_updates]
        vectors, layout = stack_updates(updates)
        n = int(vectors.shape[0])
        clip_norm = float(config.get("user_clip_norm", 1.0))
        clipped_vectors, clip_factors = clip_rows(vectors, clip_norm)

        reference_name = str(config.get("robust_reference", "cm_nnm"))
        reference = aggregate_vectors(
            clipped_vectors,
            reference_name,
            num_byzantine=int(config.get("num_byzantine", 0)),
            screening_fraction=config.get("screening_fraction"),
            max_iter=int(config.get("rfa_max_iter", 100)),
            tol=float(config.get("rfa_tol", 1e-6)),
            smoothing=float(config.get("rfa_smoothing", 1e-8)),
            alpha_trusted=float(config.get("cmls_alpha_trusted", 1.0)),
            alpha_suspected=float(config.get("cmls_alpha_suspected", 1.0)),
        )
        distances = torch.linalg.vector_norm(clipped_vectors - reference, dim=1)
        configured_distance_clip = config.get("distance_clip")
        distance_clip = (
            float(configured_distance_clip)
            if configured_distance_clip is not None
            else float(config.get("distance_clip_multiple", 2.0)) * clip_norm
        )
        scores = bounded_distance_scores(
            distances, distance_clip, str(config.get("score_mode", "hard_clip"))
        )
        alpha, alpha_max, alpha_was_clipped = self._alpha(n, config)
        weights = self.controlled_weights(scores, alpha)
        aggregate_vector = (weights[:, None] * clipped_vectors).sum(0)

        sensitivity, sensitivity_mode = self._sensitivity(
            n=n,
            clip_norm=clip_norm,
            distance_clip=distance_clip,
            alpha=alpha,
            config=config,
        )
        dp_enabled = bool(config.get("enable_central_dp", True))
        noise_multiplier = self._noise_multiplier(config) if dp_enabled else 0.0
        noise = torch.zeros_like(aggregate_vector)
        if dp_enabled:
            noise = torch.randn_like(aggregate_vector) * (
                noise_multiplier * sensitivity
            )
            aggregate_vector = aggregate_vector + noise
        aggregate = dict(unflatten_update(aggregate_vector, layout))

        accountant = RDPAccountant.from_state_dict(config.get("scfar_accountant"))
        epsilon = best_order = None
        if dp_enabled:
            # Conservative initial protocol: no privacy amplification is claimed
            # for fixed-size client sampling.
            accountant.add_sampled_gaussian(
                channel="central_model",
                sampling_rate=1.0,
                noise_multiplier=noise_multiplier,
                steps=1,
            )
            epsilon, best_order = accountant.epsilon(float(config.get("delta", 1e-5)))

        malicious_mask = torch.tensor(
            [bool(meta.get("is_byzantine", False)) for _, meta, _ in client_updates]
        )
        diagnostics = weight_diagnostics(weights, malicious_mask)
        diagnostics.update(
            {
                "scfar_requested_alpha": float(config.get("far_alpha", 0.1)),
                "scfar_effective_alpha": alpha,
                "scfar_alpha_max": alpha_max,
                "scfar_alpha_was_clipped": alpha_was_clipped,
                "scfar_kappa_w": float(config.get("kappa_w", 2.0)),
                "scfar_user_clip_norm": clip_norm,
                "scfar_user_clip_rate": float((clip_factors < 1.0).double().mean()),
                "scfar_distance_clip": distance_clip,
                "scfar_score_saturation_rate": float((scores >= 1.0).double().mean()),
                "scfar_mean_distance": float(distances.mean()),
                "scfar_max_distance": float(distances.max()),
                "scfar_reference_norm": float(torch.linalg.vector_norm(reference)),
                "scfar_sensitivity": sensitivity,
                "scfar_sensitivity_mode": sensitivity_mode,
                "central_noise_multiplier": noise_multiplier if dp_enabled else None,
                "central_noise_norm": float(torch.linalg.vector_norm(noise)),
                "privacy_epsilon": epsilon,
                "privacy_delta": float(config.get("delta", 1e-5)) if dp_enabled else None,
                "privacy_best_order": best_order,
                "privacy_level": "user",
                "privacy_trust_model": "central",
                "privacy_adjacency": "replace_one",
                "privacy_sampling_assumption": "no_amplification_q_equals_1",
                "robust_reference": reference_name,
            }
        )
        metrics = common_round_metrics(client_updates)
        metrics.update({"round": round_num})
        metrics.update(diagnostics)
        first_meta = client_updates[0][1]
        for key in ("active_group_idx", "is_warmup", "num_layer_groups"):
            if key in first_meta:
                metrics[key] = first_meta[key]
        if dp_enabled:
            metrics["_server_state_updates"] = {
                "scfar_accountant": accountant.state_dict()
            }
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def _scfar_defaults(self) -> dict:
        return {
            "user_clip_norm": 1.0,
            "distance_clip_multiple": 2.0,
            "distance_clip": None,
            "score_mode": "hard_clip",
            "far_alpha": 0.1,
            "kappa_w": 2.0,
            "alpha_bound_policy": "clip",
            "robust_reference": "cm_nnm",
            "num_byzantine": 0,
            "enable_central_dp": True,
            "central_noise_multiplier": 1.0,
            "target_epsilon": None,
            "privacy_num_rounds": None,
            "delta": 1e-5,
            "sensitivity_mode": "conservative_2C",
            "reference_stability_constant": None,
            "client_metrics_every": 1,
        }


@register_algorithm("scfar_dp")
class SensitivityControlledFAR(_SCFARServerMixin, FedAvg):
    """Full-model Sensitivity-Controlled FAR ablation."""

    description = "Sensitivity-Controlled FAR with trusted user clipping and central DP."

    def get_default_config(self):
        return {**FedAvg.get_default_config(self), **self._scfar_defaults()}


@register_algorithm("fairpartfar_dp")
@register_algorithm("sc_partial_far_dp")
class SensitivityControlledPartialFAR(_SCFARServerMixin, FedPart):
    """One active layer group per round plus Sensitivity-Controlled FAR."""

    description = "Partial training + SC-FAR + central user-level Gaussian DP."

    def get_default_config(self):
        return {**FedPart.get_default_config(self), **self._scfar_defaults()}
