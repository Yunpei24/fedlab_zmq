"""Sensitivity-Controlled FAR and its partial-training extension.

This module implements the current conceptual framework without the two
mechanisms that were removed from the research design: there is no capped-
simplex projection and no FedAvg shrinkage.  Influence is controlled by:

1. trusted server-side user-level clipping;
2. a bounded distance score in ``[0, 1]``;
3. an analytical bound on the softmax tilt ``alpha``;
4. an explicit sensitivity mode for calibrating central Gaussian noise.

Two references carry implementation-level replace-one certificates:

``centered_clipping`` (primary)
    One public-anchor centered-clipping step, with
    ``delta_F <= 2*min(C,tau)/n``.

``regularized_huber`` (ablation)
    A public, fixed number of iterations for a regularized vector-Huber
    objective.  Its certificate includes the finite-solver factor and thus
    applies to the actual returned iterate.

Other robust references remain available as empirical baselines, but the safe
privacy calibration then defaults to the global ``2C`` sensitivity bound.
"""

from __future__ import annotations

import math

import torch

from metrics.robustness import weight_diagnostics
from privacy.rdp import RDPAccountant, calibrate_gaussian_noise
from robustness.aggregators import (
    aggregate_vectors,
    centered_clipping,
    clip_l2,
    regularized_huber_reference,
)
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


def clip_rows(
    vectors: torch.Tensor, clip_norm: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trusted user-level clipping of every received client vector."""

    if clip_norm <= 0:
        raise ValueError("user_clip_norm must be positive")
    norms = torch.linalg.vector_norm(vectors, dim=1)
    factors = (float(clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)
    return vectors * factors[:, None], factors


def softmax_weight_factor_bound(n: int, alpha: float) -> float:
    """Return the certified factor ``kappa(alpha)`` in ``q_i <= kappa/n``.

    For scores in ``[0,1]``, the largest possible weight is attained when one
    score is one and all others are zero:

    ``q_max = exp(alpha) / (exp(alpha) + n - 1)``.
    """

    if n < 2:
        raise ValueError("At least two clients are required")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    # This algebraically equivalent form avoids overflow for large alpha.
    q_max = 1.0 / (1.0 + (n - 1) * math.exp(-float(alpha)))
    return float(n * q_max)


def certified_scfar_sensitivity(
    *,
    n: int,
    clip_norm: float,
    distance_clip: float,
    alpha: float,
    kappa_bound: float,
    reference_stability: float,
) -> float:
    r"""Transport a reference-stability certificate through SC-FAR.

    ``reference_stability`` is the *absolute* replace-one bound ``delta_F``.
    The certified result is

    .. math::

        \min\left\{2C,\frac{2C\kappa}{n}(1+\alpha)
        +\frac{2C\alpha}{D_{\max}}\delta_F\right\}.

    The final ``2C`` cap is always valid because SC-FAR returns a convex
    combination of vectors in the radius-``C`` ball.
    """

    if n < 2 or clip_norm <= 0 or distance_clip <= 0:
        raise ValueError("Need n>=2 and positive clipping constants")
    if alpha < 0 or kappa_bound < 1 or reference_stability < 0:
        raise ValueError("Invalid sensitivity-certificate parameter")
    transported = (2.0 * clip_norm * kappa_bound / n) * (1.0 + alpha) + (
        2.0 * clip_norm * alpha / distance_clip
    ) * reference_stability
    return float(min(2.0 * clip_norm, transported))


class _SCFARServerMixin:
    """Sensitivity-controlled server rule shared by full and partial variants."""

    @staticmethod
    def controlled_weights(scores: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.softmax(float(alpha) * scores, dim=0)

    @staticmethod
    def _reference_name(config: dict) -> str:
        aliases = {
            "cc": "centered_clipping",
            "f_cc": "centered_clipping",
            "huber": "regularized_huber",
            "huber_regularized": "regularized_huber",
        }
        requested = str(config.get("robust_reference", "centered_clipping")).lower()
        return aliases.get(requested, requested)

    def _anchor_for_round(
        self,
        *,
        vectors: torch.Tensor,
        round_num: int,
        group_key: str,
    ) -> torch.Tensor:
        """Return an anchor independent of the current cohort.

        Full training has one anchor.  Partial training keeps one anchor per
        active layer group so that changing the transmitted subspace does not
        silently reset or dimension-mismatch the history.
        """

        if round_num == 0 or not hasattr(self, "_scfar_reference_anchors"):
            self._scfar_reference_anchors = {}
        stored = self._scfar_reference_anchors.get(group_key)
        if stored is None or stored.numel() != vectors.shape[1]:
            return torch.zeros(
                vectors.shape[1], device=vectors.device, dtype=vectors.dtype
            )
        return stored.to(device=vectors.device, dtype=vectors.dtype)

    def _compute_reference(
        self,
        *,
        vectors: torch.Tensor,
        anchor: torch.Tensor,
        clip_norm: float,
        config: dict,
    ) -> tuple[torch.Tensor, float | None, dict]:
        """Compute ``F`` and, when proved, its absolute stability certificate."""

        n = int(vectors.shape[0])
        name = self._reference_name(config)
        tau = float(
            config.get("reference_clip_tau")
            if config.get("reference_clip_tau") is not None
            else float(config.get("reference_clip_multiple", 1.0)) * clip_norm
        )
        if tau <= 0:
            raise ValueError("reference_clip_tau must be positive")
        effective_radius = min(clip_norm, tau)

        if name == "centered_clipping":
            reference = centered_clipping(vectors, anchor=anchor, tau=tau)
            stability = 2.0 * effective_radius / n
            return (
                reference,
                stability,
                {
                    "scfar_reference_certificate": "centered_clipping_global",
                    "scfar_reference_clip_tau": tau,
                    "scfar_reference_effective_radius": effective_radius,
                },
            )

        if name == "regularized_huber":
            gamma = float(config.get("huber_gamma", 1.0))
            num_steps = int(config.get("huber_num_steps", 10))
            reference, huber_metrics = regularized_huber_reference(
                vectors,
                anchor=anchor,
                tau=tau,
                gamma=gamma,
                num_steps=num_steps,
                return_diagnostics=True,
            )
            contraction = float(huber_metrics["huber_contraction"])
            stability = (
                2.0 * effective_radius / (gamma * n) * (1.0 - contraction**num_steps)
            )
            return (
                reference,
                stability,
                {
                    "scfar_reference_certificate": "regularized_huber_fixed_steps",
                    "scfar_reference_clip_tau": tau,
                    "scfar_reference_effective_radius": effective_radius,
                    "scfar_huber_gamma": gamma,
                    **{f"scfar_{key}": value for key, value in huber_metrics.items()},
                },
            )

        reference = aggregate_vectors(
            vectors,
            name,
            num_byzantine=int(config.get("num_byzantine", 0)),
            screening_fraction=config.get("screening_fraction"),
            max_iter=int(config.get("rfa_max_iter", 100)),
            tol=float(config.get("rfa_tol", 1e-6)),
            smoothing=float(config.get("rfa_smoothing", 1e-8)),
            alpha_trusted=float(config.get("cmls_alpha_trusted", 1.0)),
            alpha_suspected=float(config.get("cmls_alpha_suspected", 1.0)),
        )
        return (
            reference,
            None,
            {
                "scfar_reference_certificate": "none_use_conservative_sensitivity",
                "scfar_reference_clip_tau": None,
                "scfar_reference_effective_radius": None,
            },
        )

    def _update_reference_anchor(
        self,
        *,
        anchor: torch.Tensor,
        released_aggregate: torch.Tensor,
        round_num: int,
        group_key: str,
        clip_norm: float,
        config: dict,
    ) -> tuple[torch.Tensor, float]:
        """Build next round's anchor only from the current public release."""

        mode = str(config.get("anchor_mode", "previous_release")).lower()
        if mode == "fixed_zero":
            rate = 0.0
        elif mode == "previous_release":
            rate = 1.0
        elif mode == "ema_release":
            rate = float(config.get("anchor_update_rate", 0.1))
        else:
            raise ValueError(
                "anchor_mode must be fixed_zero, previous_release or ema_release"
            )
        if not 0.0 <= rate <= 1.0:
            raise ValueError("anchor_update_rate must lie in [0,1]")
        update_every = int(config.get("anchor_update_every", 1))
        if update_every < 1:
            raise ValueError("anchor_update_every must be at least one")
        configured_radius = config.get("anchor_clip_norm")
        radius = clip_norm if configured_radius is None else float(configured_radius)
        if radius <= 0:
            raise ValueError("anchor_clip_norm must be positive")
        should_update = (round_num + 1) % update_every == 0
        effective_rate = rate if should_update else 0.0
        candidate = (
            (1.0 - effective_rate) * anchor
            + effective_rate * released_aggregate
        )
        next_anchor = clip_l2(candidate, radius).detach().cpu()
        self._scfar_reference_anchors[group_key] = next_anchor
        drift = torch.linalg.vector_norm(next_anchor.to(anchor) - anchor)
        return next_anchor, float(drift)

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
            raise ValueError(
                "alpha_bound_policy must be clip, error or diagnostic_only"
            )
        effective = min(requested, maximum) if policy == "clip" else requested
        return effective, maximum, clipped

    def _sensitivity(
        self,
        *,
        n: int,
        clip_norm: float,
        distance_clip: float,
        alpha: float,
        kappa_bound: float,
        reference_stability: float | None,
        config: dict,
    ) -> tuple[float, str]:
        mode = str(config.get("sensitivity_mode", "automatic_certified"))
        if mode == "conservative_2C":
            return 2.0 * clip_norm, mode
        if mode not in {"automatic_certified", "proved_reference_bound"}:
            raise ValueError(
                "sensitivity_mode must be automatic_certified, conservative_2C "
                "or proved_reference_bound"
            )
        if reference_stability is None and bool(
            config.get("allow_manual_reference_certificate", False)
        ):
            coefficient = config.get("reference_stability_constant")
            if coefficient is not None:
                # Backward-compatible research escape hatch.  It is disabled
                # by default because the caller, not this implementation,
                # bears the proof obligation delta_F <= L_ref*C/n.
                reference_stability = float(coefficient) * clip_norm / n
                mode = "manual_reference_certificate"
        if reference_stability is None and mode == "automatic_certified":
            return 2.0 * clip_norm, "automatic_fallback_conservative_2C"
        if reference_stability is None:
            raise ValueError(
                "proved_reference_bound requires a certified reference. Use "
                "centered_clipping or regularized_huber, or explicitly enable "
                "allow_manual_reference_certificate with a proved constant."
            )
        sensitivity = certified_scfar_sensitivity(
            n=n,
            clip_norm=clip_norm,
            distance_clip=distance_clip,
            alpha=alpha,
            kappa_bound=kappa_bound,
            reference_stability=float(reference_stability),
        )
        return sensitivity, "proved_reference_bound"

    def _noise_multiplier(self, config: dict) -> float:
        target = config.get("target_epsilon")
        if target is None:
            return float(config.get("central_noise_multiplier", 1.0))
        total_rounds = config.get("privacy_num_rounds")
        if total_rounds is None:
            raise ValueError("target_epsilon requires privacy_num_rounds")
        return calibrate_gaussian_noise(
            target_epsilon=float(target),
            delta=float(config.get("delta", 1e-5)),
            steps=int(total_rounds),
        )

    def server_aggregate(self, global_model, client_updates, round_num, config):
        updates = [update for update, _, _ in client_updates]
        vectors, layout = stack_updates(updates)
        n = int(vectors.shape[0])
        clip_norm = float(config.get("user_clip_norm", 1.0))
        clipped_vectors, clip_factors = clip_rows(vectors, clip_norm)

        first_meta = client_updates[0][1]
        group_key = str(first_meta.get("active_group_idx", "full"))
        anchor = self._anchor_for_round(
            vectors=clipped_vectors,
            round_num=round_num,
            group_key=group_key,
        )
        reference_name = self._reference_name(config)
        reference, reference_stability, reference_metrics = self._compute_reference(
            vectors=clipped_vectors,
            anchor=anchor,
            clip_norm=clip_norm,
            config=config,
        )
        distances = torch.linalg.vector_norm(clipped_vectors - reference, dim=1)
        configured_distance_clip = config.get("distance_clip")
        distance_clip = (
            float(configured_distance_clip)
            if configured_distance_clip is not None
            else float(config.get("distance_clip_multiple", 2.0)) * clip_norm
        )
        bounded_scores = bounded_distance_scores(
            distances, distance_clip, str(config.get("score_mode", "hard_clip"))
        )
        aggregation_rule = str(
            config.get("scfar_aggregation_rule", "controlled_tilt")
        ).lower()
        if aggregation_rule == "controlled_tilt":
            alpha, alpha_max, alpha_was_clipped = self._alpha(n, config)
            scores = bounded_scores
            weights = self.controlled_weights(scores, alpha)
            clean_aggregate_vector = (weights[:, None] * clipped_vectors).sum(0)
        elif aggregation_rule == "uniform":
            alpha, alpha_max, alpha_was_clipped = 0.0, 0.0, False
            scores = torch.zeros_like(bounded_scores)
            weights = torch.full_like(scores, 1.0 / n)
            clean_aggregate_vector = clipped_vectors.mean(dim=0)
        elif aggregation_rule == "reference":
            alpha, alpha_max, alpha_was_clipped = 0.0, 0.0, False
            scores = torch.zeros_like(bounded_scores)
            weights = torch.full_like(scores, 1.0 / n)
            clean_aggregate_vector = reference
        elif aggregation_rule == "far_raw_distance":
            alpha = float(config.get("far_alpha", 0.1))
            if alpha < 0:
                raise ValueError("far_raw_distance requires far_alpha >= 0")
            alpha_max, alpha_was_clipped = None, False
            scores = distances
            weights = self.controlled_weights(scores, alpha)
            clean_aggregate_vector = (weights[:, None] * clipped_vectors).sum(0)
        else:
            raise ValueError(
                "scfar_aggregation_rule must be controlled_tilt, uniform, "
                "reference or far_raw_distance"
            )

        configured_kappa = float(config.get("kappa_w", 2.0))
        analytical_kappa = float(n * weights.max().detach().cpu())
        weight_bound_holds = (
            analytical_kappa <= configured_kappa + 1e-12
            if aggregation_rule != "far_raw_distance"
            else None
        )
        # Privacy calibration must use a public, data-independent bound.  The
        # observed factor ``n * max_i q_i`` is useful telemetry, but using it
        # to set the noise scale would make that scale depend on the private
        # cohort.  For bounded scores in [0, 1], kappa(alpha) below is the
        # deterministic worst-case factor implied by the public alpha.  It is
        # no larger than the pre-registered design ceiling ``kappa_w``.
        if aggregation_rule in {"uniform", "reference"}:
            certified_kappa = 1.0
        elif aggregation_rule == "controlled_tilt":
            certified_kappa = softmax_weight_factor_bound(n, alpha)
        else:
            # Raw-distance FAR always uses the conservative 2C branch below;
            # this value is diagnostic only.
            certified_kappa = configured_kappa

        if aggregation_rule == "reference":
            sensitivity = (
                float(reference_stability)
                if reference_stability is not None
                else 2.0 * clip_norm
            )
            sensitivity_mode = (
                "reference_certificate"
                if reference_stability is not None
                else "reference_fallback_conservative_2C"
            )
        elif aggregation_rule == "far_raw_distance":
            sensitivity, sensitivity_mode = 2.0 * clip_norm, "conservative_2C"
        else:
            sensitivity, sensitivity_mode = self._sensitivity(
                n=n,
                clip_norm=clip_norm,
                distance_clip=distance_clip,
                alpha=alpha,
                kappa_bound=certified_kappa,
                reference_stability=reference_stability,
                config=config,
            )

        dp_enabled = bool(config.get("enable_central_dp", True))
        privacy_horizon = config.get("privacy_num_rounds")
        if (
            dp_enabled
            and privacy_horizon is not None
            and round_num >= int(privacy_horizon)
        ):
            raise RuntimeError(
                "Refusing to release SC-FAR-DP beyond privacy_num_rounds"
            )
        noise_multiplier = self._noise_multiplier(config) if dp_enabled else 0.0
        noise = torch.zeros_like(clean_aggregate_vector)
        if dp_enabled:
            noise = torch.randn_like(clean_aggregate_vector) * (
                noise_multiplier * sensitivity
            )
        aggregate_vector = clean_aggregate_vector + noise
        _, anchor_drift = self._update_reference_anchor(
            anchor=anchor,
            released_aggregate=aggregate_vector,
            round_num=round_num,
            group_key=group_key,
            clip_norm=clip_norm,
            config=config,
        )
        aggregate = dict(unflatten_update(aggregate_vector, layout))

        accountant = RDPAccountant.from_state_dict(config.get("scfar_accountant"))
        epsilon = best_order = None
        if dp_enabled:
            # Conservative initial protocol: no privacy amplification is claimed
            # for fixed-size client sampling.
            accountant.add_gaussian(
                channel="central_model",
                noise_multiplier=noise_multiplier,
                steps=1,
            )
            epsilon, best_order = accountant.epsilon(
                float(config.get("delta", 1e-5))
            )
            target_epsilon = config.get("target_epsilon")
            if target_epsilon is not None and epsilon > float(target_epsilon) + 5e-4:
                raise RuntimeError(
                    "Refusing to release SC-FAR-DP after exhausting target_epsilon"
                )

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
                "scfar_aggregation_rule": aggregation_rule,
                "scfar_kappa_w": configured_kappa,
                "scfar_certified_kappa": certified_kappa,
                "scfar_certified_kappa_source": "public_score_range_and_alpha",
                "scfar_analytical_kappa": analytical_kappa,
                "scfar_weight_bound_holds": weight_bound_holds,
                "scfar_user_clip_norm": clip_norm,
                "scfar_user_clip_rate": float((clip_factors < 1.0).double().mean()),
                "scfar_distance_clip": distance_clip,
                "scfar_score_saturation_rate": float(
                    (bounded_scores >= 1.0).double().mean()
                ),
                "scfar_mean_distance": float(distances.mean()),
                "scfar_max_distance": float(distances.max()),
                "scfar_reference_norm": float(torch.linalg.vector_norm(reference)),
                "scfar_reference_anchor_norm": float(torch.linalg.vector_norm(anchor)),
                "scfar_reference_anchor_drift": anchor_drift,
                "scfar_reference_group_key": group_key,
                "scfar_reference_stability": reference_stability,
                "scfar_sensitivity": sensitivity,
                "scfar_sensitivity_mode": sensitivity_mode,
                "central_noise_multiplier": noise_multiplier if dp_enabled else None,
                "central_noise_std": (
                    noise_multiplier * sensitivity if dp_enabled else None
                ),
                "central_noise_norm": float(torch.linalg.vector_norm(noise)),
                "privacy_epsilon": epsilon,
                "privacy_delta": (
                    float(config.get("delta", 1e-5)) if dp_enabled else None
                ),
                "privacy_best_order": best_order,
                "privacy_level": "user",
                "privacy_trust_model": "central",
                "privacy_adjacency": "replace_one",
                "privacy_sampling_assumption": "no_amplification_q_equals_1",
                "privacy_accountant_mechanism": "ordinary_gaussian_rdp",
                "privacy_num_rounds": (
                    int(privacy_horizon) if privacy_horizon is not None else None
                ),
                "robust_reference": reference_name,
            }
        )
        diagnostics.update(reference_metrics)

        client_ids = [
            int(meta.get("client_id", -1)) for _, meta, _ in client_updates
        ]
        registered_outliers = {
            int(value) for value in config.get("honest_outlier_client_ids", [])
        }
        if registered_outliers:
            outlier_mask = torch.tensor(
                [
                    client_id in registered_outliers and not bool(is_malicious)
                    for client_id, is_malicious in zip(client_ids, malicious_mask)
                ],
                dtype=torch.bool,
            )
            outlier_count = int(outlier_mask.sum().item())
            diagnostics["honest_outlier_count_oracle"] = outlier_count
            diagnostics["honest_outlier_weight_mass_oracle"] = (
                float(weights.detach().cpu()[outlier_mask].sum().item())
                if outlier_count
                else 0.0
            )
            diagnostics["honest_outlier_mean_weight_oracle"] = (
                float(weights.detach().cpu()[outlier_mask].mean().item())
                if outlier_count
                else 0.0
            )
            diagnostics["honest_outlier_above_uniform_count_oracle"] = (
                int((weights.detach().cpu()[outlier_mask] > 1.0 / n).sum().item())
                if outlier_count
                else 0
            )

        honest = ~malicious_mask
        if bool(honest.any()):
            honest_vectors = clipped_vectors[honest.to(clipped_vectors.device)]
            honest_mean = honest_vectors.mean(dim=0)
            diagnostics["scfar_reference_error_honest_mean"] = float(
                torch.linalg.vector_norm(reference - honest_mean)
            )
            diagnostics["scfar_anchor_error_honest_mean"] = float(
                torch.linalg.vector_norm(anchor - honest_mean)
            )
            if reference_name == "centered_clipping":
                tau = float(reference_metrics["scfar_reference_clip_tau"])
                honest_residuals = torch.linalg.vector_norm(
                    honest_vectors - anchor, dim=1
                )
                honest_tail = (honest_residuals - tau).clamp_min(0.0).mean()
                byzantine_fraction = float(malicious_mask.double().mean())
                # Conditional finite-sample bound from the theory document.
                diagnostics["scfar_cc_honest_tail_bias"] = float(honest_tail)
                diagnostics["scfar_cc_byzantine_fraction"] = byzantine_fraction
                diagnostics["scfar_cc_conditional_robustness_bound"] = float(
                    byzantine_fraction
                    * (torch.linalg.vector_norm(anchor - honest_mean) + tau)
                    + honest_tail
                )
        metrics = common_round_metrics(client_updates)
        metrics.update({"round": round_num})
        metrics.update(diagnostics)
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
            "scfar_aggregation_rule": "controlled_tilt",
            "far_alpha": 0.1,
            "kappa_w": 2.0,
            "alpha_bound_policy": "clip",
            # Primary theory-aligned F.  CM(NNM), trMean(NNM), RFA, etc. remain
            # available as empirical baselines but do not receive the proved
            # O(C/n) privacy calibration automatically.
            "robust_reference": "centered_clipping",
            "reference_clip_multiple": 1.0,
            "reference_clip_tau": None,
            "anchor_mode": "previous_release",
            "anchor_update_rate": 1.0,
            "anchor_update_every": 1,
            "anchor_clip_norm": None,
            # Huber is an ablation, not the default reference.
            "huber_gamma": 1.0,
            "huber_num_steps": 10,
            "num_byzantine": 0,
            "enable_central_dp": True,
            "central_noise_multiplier": 1.0,
            "target_epsilon": None,
            "privacy_num_rounds": None,
            "delta": 1e-5,
            "sensitivity_mode": "automatic_certified",
            "allow_manual_reference_certificate": False,
            "reference_stability_constant": None,
            "honest_outlier_client_ids": [],
            "client_metrics_every": 1,
        }


@register_algorithm("scfar_dp")
class SensitivityControlledFAR(_SCFARServerMixin, FedAvg):
    """Full-model Sensitivity-Controlled FAR ablation."""

    description = (
        "Sensitivity-Controlled FAR with trusted user clipping and central DP."
    )

    def get_default_config(self):
        return {**FedAvg.get_default_config(self), **self._scfar_defaults()}


@register_algorithm("fairpartfar_dp")
@register_algorithm("sc_partial_far_dp")
class SensitivityControlledPartialFAR(_SCFARServerMixin, FedPart):
    """One active layer group per round plus Sensitivity-Controlled FAR."""

    description = "Partial training + SC-FAR + central user-level Gaussian DP."

    def get_default_config(self):
        return {**FedPart.get_default_config(self), **self._scfar_defaults()}
