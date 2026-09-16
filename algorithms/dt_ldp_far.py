"""Delayed-Tilting FAR under client-side sample-level differential privacy.

This module implements the experimental algorithm studied in
``output/pdf/Analyse_Theorique_DT_LDP_FAR.pdf``.  It is intentionally
different from both existing FAR privacy paths:

``dpfar``
    computes FAR scores and weights from the *current* locally-private
    uploads.  A fresh noise realisation can therefore affect both its update
    and the coefficient multiplying that update.

``scfar_dp``
    clips whole-client updates and adds one central Gaussian perturbation to
    a sensitivity-certified server release (central user-level DP).

``dt_ldp_far`` (this file)
    runs sample-level DP-SGD at every honest client, clips the already-private
    uploads at the server, and aggregates round ``t`` with weights computed
    solely from round ``t-1`` scores.  The current reference and scores are
    used only to construct weights for the next round.

The reference ``F`` is configurable.  Centered clipping is the main candidate
because its quality and temporal drift are analyzable, not because local DP
requires its replace-one stability.  CM(NNM), trimmed mean, RFA and the other
registered robust aggregators remain valid post-processing ablations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from attacks import scheduled_attack_phase
from metrics.robustness import weight_diagnostics
from robustness.aggregators import (
    aggregate_vectors,
    centered_clipping,
    clip_l2,
    guarded_aggregate,
    regularized_huber_reference,
)
from robustness.tensor_ops import score_subspace, stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .dp_references import _dp_defaults, _LocalDPSGDMixin
from .far import FAR
from .noise_aware_scores import (
    DIRECT_SCORE_MODES,
    EFFECTIVE_NULL_MOMENT_SCORE_MODE,
    EFFECTIVE_NULL_SCORE_MODES,
    ENERGY_SCORE_MODE,
    ROBUST_DIRECT_SCORE_MODES,
    calibrated_independence_trust_scores,
    debiased_distance_scores,
    directional_trust_scores,
    effective_null_fcc_loo_scores,
    effective_upload_noise_variances,
    excess_energy_scores,
    lagged_descent_alignment_scores,
    mean_reference_residual_variances,
    multi_krum_admissibility_trust_scores,
    public_noise_scales,
    ranked_directional_peer_support_scores,
    separate_novelty_trust_scores,
    standardize_distances,
)
from .reference_utils import apply_delta, common_round_metrics
from .sc_partial_far_dp import bounded_distance_scores, clip_rows


def tilt_tau_max(n: int, kappa_w: float, *, active_clients: int | None = None) -> float:
    r"""Largest tilt guaranteeing ``max_i omega_i <= kappa_w / n``.

    The certificate assumes scores in ``[0, 1]``.  For ``n=1`` the only
    possible weight is one and the tilt is immaterial.
    """

    if n < 1:
        raise ValueError("At least one client is required")
    if n == 1:
        return math.inf
    if not 1.0 <= float(kappa_w) < n:
        raise ValueError("kappa_w must satisfy 1 <= kappa_w < n")
    active = n if active_clients is None else int(active_clients)
    if not 1 <= active <= n:
        raise ValueError("active_clients must lie in [1,n]")
    ratio = float(kappa_w) * (active - 1) / (n - float(kappa_w))
    if ratio < 1.0:
        raise ValueError(
            "No non-negative tilt can guarantee kappa_w/n with this many "
            "active clients"
        )
    return math.log(ratio)


def calibrated_tilt_tau(
    n: int, config: dict, *, public_score_range: float = 1.0
) -> tuple[float, float, bool]:
    """Resolve the tilt and enforce a cap for the declared public score range."""

    requested = float(config.get("tilt_tau", config.get("far_alpha", 0.1)))
    if requested < 0:
        raise ValueError("DT-LDP-FAR requires a non-negative tilt_tau")
    if n == 1:
        return 0.0, math.inf, requested > 0
    active_clients = None
    if bool(config.get("dt_admissibility_filter_enabled", False)):
        excluded = int(
            config.get(
                "dt_admissibility_excluded_clients",
                config.get("dt_admissibility_max_byzantine", 0),
            )
        )
        active_clients = n - excluded
    if not math.isfinite(float(public_score_range)) or public_score_range <= 0:
        raise ValueError("public_score_range must be finite and positive")
    maximum = tilt_tau_max(
        n,
        float(config.get("kappa_w", 2.0)),
        active_clients=active_clients,
    ) / float(public_score_range)
    policy = str(config.get("tilt_bound_policy", "clip")).lower()
    if policy not in {"clip", "error", "diagnostic_only"}:
        raise ValueError("tilt_bound_policy must be clip, error or diagnostic_only")
    requested_exceeds_cap = requested > maximum
    if requested_exceeds_cap and policy == "error":
        raise ValueError(
            f"tilt_tau={requested} exceeds tau_max={maximum:.6g} for "
            f"n={n}, kappa_w={config.get('kappa_w', 2.0)}"
        )
    effective = min(requested, maximum) if policy == "clip" else requested
    return float(effective), float(maximum), bool(requested_exceeds_cap)


def delayed_weights_from_scores(
    client_ids: Sequence[int],
    previous_scores: dict[int, float],
    *,
    tau: float,
    unseen_score: float = 0.0,
    score_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weights aligned to ``client_ids`` using only stored scores.

    This pure helper is deliberately independent of the current vectors.  It
    is the central causal invariant of DT-LDP-FAR and is tested directly.
    """

    if len(set(int(cid) for cid in client_ids)) != len(client_ids):
        raise ValueError("DT-LDP-FAR received duplicate client_id values")
    if not math.isfinite(float(unseen_score)):
        raise ValueError("unseen_client_score must be finite")
    if score_bounds is not None:
        lower, upper = map(float, score_bounds)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("score_bounds must be finite and ordered")
        if not lower <= float(unseen_score) <= upper:
            raise ValueError("unseen_client_score lies outside score_bounds")
    scores = torch.tensor(
        [float(previous_scores.get(int(cid), unseen_score)) for cid in client_ids],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Stored DT-LDP-FAR scores must be finite")
    if score_bounds is not None and bool(
        ((scores < lower) | (scores > upper)).any()
    ):
        raise ValueError("Stored DT-LDP-FAR scores lie outside score_bounds")
    weights = torch.softmax(float(tau) * scores, dim=0)
    return weights, scores


def delayed_filtered_weights_from_scores(
    client_ids: Sequence[int],
    previous_scores: dict[int, float],
    previous_trust: dict[int, float],
    *,
    tau: float,
    max_excluded: int,
    unseen_score: float = 0.0,
    score_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    r"""Apply delayed FAR only inside a delayed robust admissible set.

    When the preceding round contains one trust value for every current
    client, the ``n-max_excluded`` largest trust values are retained.  Ties
    are resolved by the public client identifier, so the rule is fully
    deterministic.  The softmax is then normalised over retained clients;
    excluded clients receive exactly zero weight.  If the delayed trust state
    is incomplete (notably at cold start), no client is excluded.

    The trust values and FAR scores are both functions of the already-private
    transcript.  This filtering is therefore local-DP post-processing, not an
    additional privacy mechanism and not a Byzantine-identification theorem.
    """

    n = len(client_ids)
    if n < 1 or len(set(int(cid) for cid in client_ids)) != n:
        raise ValueError("client_ids must be a non-empty sequence of unique IDs")
    if not 0 <= int(max_excluded) < n:
        raise ValueError("max_excluded must lie in [0,n)")
    weights, scores = delayed_weights_from_scores(
        client_ids,
        previous_scores,
        tau=tau,
        unseen_score=unseen_score,
        score_bounds=score_bounds,
    )
    complete = all(int(cid) in previous_trust for cid in client_ids)
    if not complete or int(max_excluded) == 0:
        return weights, scores, torch.ones(n, dtype=torch.bool), False

    trust_values = {int(cid): float(previous_trust[int(cid)]) for cid in client_ids}
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in trust_values.values()
    ):
        raise ValueError("Stored admissibility trust values must lie in [0,1]")
    active_count = n - int(max_excluded)
    ranked_ids = sorted(client_ids, key=lambda cid: (-trust_values[int(cid)], int(cid)))
    retained = {int(cid) for cid in ranked_ids[:active_count]}
    eligible = torch.tensor([int(cid) in retained for cid in client_ids])
    logits = float(tau) * scores
    logits = logits.masked_fill(~eligible, float("-inf"))
    filtered_weights = torch.softmax(logits, dim=0)
    return filtered_weights, scores, eligible, True


def robust_anchor_perturbation(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    anchor: torch.Tensor,
    *,
    residual_radius: float,
    gain: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Add a certified FAR correction around a robust anchor.

    The correction is centred relative to uniform weights,

    ``gain * sum_i (weights_i - 1/n) Clip_radius(vectors_i - anchor)``.

    Hence a uniform weighting returns the anchor exactly.  Since every clipped
    residual has norm at most ``residual_radius`` and the L1 distance between
    two probability vectors is at most two, the deterministic certificate

    ``||output - anchor|| <= 2 * gain * residual_radius``

    holds for every cohort, independently of score quality or the attack.
    This containment certificate does not itself certify the quality of the
    anchor; that remains the responsibility of the robust reference rule.
    """

    if vectors.ndim != 2 or vectors.shape[0] < 1:
        raise ValueError("vectors must have shape (n,d) with n >= 1")
    n = vectors.shape[0]
    if weights.shape != (n,):
        raise ValueError("weights must contain one value per vector")
    if anchor.shape != (vectors.shape[1],):
        raise ValueError("anchor must have the vector feature dimension")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("vectors must be finite")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("weights must be finite and non-negative")
    if not math.isfinite(float(residual_radius)) or float(residual_radius) <= 0.0:
        raise ValueError("residual_radius must be finite and positive")
    if not math.isfinite(float(gain)) or not 0.0 <= float(gain) <= 1.0:
        raise ValueError("gain must lie in [0,1]")
    weight_sum = float(weights.double().sum().item())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("weights must sum to one")

    residuals, _ = clip_rows(vectors - anchor, float(residual_radius))
    uniform = torch.full_like(weights, 1.0 / n)
    centred_weights = weights - uniform
    correction = float(gain) * (centred_weights[:, None] * residuals).sum(dim=0)
    output = anchor + correction
    correction_norm = float(torch.linalg.vector_norm(correction).item())
    weight_l1 = float(torch.linalg.vector_norm(centred_weights, ord=1).item())
    data_dependent_bound = float(gain) * float(residual_radius) * weight_l1
    universal_bound = 2.0 * float(gain) * float(residual_radius)
    tolerance = 1e-6 * max(1.0, universal_bound)
    if correction_norm > data_dependent_bound + tolerance:
        raise RuntimeError("Robust-anchor perturbation violated its L1 certificate")
    return output, {
        "dtldp_anchor_correction_norm": correction_norm,
        "dtldp_anchor_correction_weight_l1": weight_l1,
        "dtldp_anchor_correction_data_bound": data_dependent_bound,
        "dtldp_anchor_correction_universal_bound": universal_bound,
        "dtldp_anchor_correction_certificate_respected": bool(
            correction_norm <= data_dependent_bound + tolerance
        ),
    }


def _pearson_or_none(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2 or y.numel() != x.numel():
        return None
    x = x.detach().double().cpu()
    y = y.detach().double().cpu()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator.item()) <= 1e-15:
        return None
    return float((x @ y / denominator).item())


@register_algorithm("dt_ldp_far")
class DTLDPFAR(_LocalDPSGDMixin, FAR):
    """Delayed, bounded FAR tilting over locally-private client updates."""

    description = (
        "DT-LDP-FAR: client-side sample-DP SGD with one-round-delayed, "
        "concentration-controlled FAR weights."
    )

    def _reset_server_state(self) -> None:
        self._dt_previous_scores: dict[int, float] = {}
        self._dt_score_history: dict[int, dict[int, float]] = {}
        self._dt_anchor: torch.Tensor | None = None
        self._dt_full_anchor: torch.Tensor | None = None
        self._dt_previous_reference: torch.Tensor | None = None
        self._dt_trust_history: dict[int, dict[int, float]] = {}

    def client_update(self, model, dataloader, state, config):
        """Run the shared local DP-SGD mechanism and police auxiliary leaks.

        Local loss, realised clipping rate and realised noise norm are not
        needed by the algorithm.  They are therefore suppressed by default so
        the claimed transcript contains no unaccounted auxiliary data channel.
        Simulation-only oracle diagnostics can be enabled explicitly.
        """

        update, metadata = self._local_dp_update(model, dataloader, state, config)
        raw_loss = metadata.pop("local_loss", None)
        raw_clip_rate = metadata.pop("clip_rate", None)
        raw_noise_norm = metadata.pop("dp_noise_norm_mean", None)
        raw_noise_free_update = metadata.pop("local_dp_noise_free_update_oracle", None)
        expose_private_client_oracles = bool(
            config.get("enable_private_client_oracle_diagnostics", False)
        )
        if expose_private_client_oracles:
            metadata.update(
                {
                    "dtldp_local_loss_oracle": raw_loss,
                    "dtldp_clip_rate_oracle": raw_clip_rate,
                    "dtldp_realised_noise_norm_oracle": raw_noise_norm,
                    "dtldp_noise_free_update_oracle": raw_noise_free_update,
                }
            )
        elif raw_noise_free_update is not None:
            raise ValueError(
                "Noise-free counterfactuals require explicit private-client "
                "oracle diagnostics"
            )
        # ``local_loss`` is part of FedLab's generic algorithm contract.  Zero
        # is a sentinel here, not a data-dependent release; the explicit flag
        # prevents it from being mistaken for a measured loss.
        metadata.update(
            {
                "local_loss": 0.0,
                "local_loss_available": False,
                "dtldp_auxiliary_diagnostics_oracle": expose_private_client_oracles,
                "dtldp_transcript_policy": (
                    "non_private_simulation_oracles_enabled"
                    if expose_private_client_oracles
                    else "no_unaccounted_loss_clip_or_noise_release"
                ),
            }
        )
        return update, metadata

    @staticmethod
    def _client_ids(client_updates) -> list[int]:
        ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
        if len(set(ids)) != len(ids):
            raise ValueError("DT-LDP-FAR requires unique client_id metadata")
        return ids

    @staticmethod
    def _reference_name(config: dict) -> str:
        aliases = {
            "cc": "centered_clipping",
            "f_cc": "centered_clipping",
            "huber": "regularized_huber",
        }
        requested = str(
            config.get(
                "dt_reference", config.get("robust_reference", "centered_clipping")
            )
        ).lower()
        return aliases.get(requested, requested)

    def _reference_proposal(
        self,
        vectors: torch.Tensor,
        anchor: torch.Tensor,
        config: dict,
        *,
        name: str | None = None,
        radius_key: str = "reference_clip_radius",
    ) -> tuple[torch.Tensor, dict]:
        """Compute one robust proposal around an anchor fixed before the call."""

        name = name or self._reference_name(config)
        if name == "centered_clipping":
            rho = float(config.get(radius_key, config.get("server_clip_norm", 1.0)))
            proposal = centered_clipping(vectors, anchor=anchor, tau=rho)
            reference_metrics = {
                "dtldp_reference_role": "main_quality_and_drift_candidate",
                "dtldp_reference_clip_radius": rho,
            }
        elif name == "regularized_huber":
            rho = float(config.get(radius_key, config.get("server_clip_norm", 1.0)))
            gamma = float(config.get("huber_gamma", 1.0))
            num_steps = int(config.get("huber_num_steps", 10))
            proposal, huber_metrics = regularized_huber_reference(
                vectors,
                anchor=anchor,
                tau=rho,
                gamma=gamma,
                num_steps=num_steps,
                return_diagnostics=True,
            )
            reference_metrics = {
                "dtldp_reference_role": "finite_step_huber_ablation",
                "dtldp_reference_clip_radius": rho,
                "dtldp_huber_gamma": gamma,
                **{f"dtldp_{key}": value for key, value in huber_metrics.items()},
            }
        else:
            proposal = aggregate_vectors(
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
            reference_metrics = {
                "dtldp_reference_role": "robust_aggregation_ablation",
                "dtldp_reference_clip_radius": None,
            }
        return proposal, reference_metrics

    def _reference(
        self, vectors: torch.Tensor, config: dict, *, round_num: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return the score reference and update its causal temporal state.

        ``current`` preserves the historical implementation: a robust
        proposal is recomputed from the current private cohort and immediately
        used for current scores.  ``lagged_ema`` is the strict temporal lane:
        scores use ``H_{t-1}``, then the current proposal updates ``H_t`` only
        for future rounds.
        """

        dimension = vectors.shape[1]
        initialized = (
            self._dt_anchor is not None and self._dt_anchor.numel() == dimension
        )
        if initialized:
            anchor = self._dt_anchor.to(vectors)
        else:
            anchor = torch.zeros(dimension, dtype=vectors.dtype, device=vectors.device)

        proposal, reference_metrics = self._reference_proposal(vectors, anchor, config)

        configured_rate = float(config.get("anchor_update_rate", 0.1))
        if not 0.0 <= configured_rate <= 1.0:
            raise ValueError("anchor_update_rate must lie in [0,1]")
        anchor_radius = float(
            config.get("anchor_clip_norm", config.get("server_clip_norm", 1.0))
        )
        step_radius = float(
            config.get(
                "temporal_reference_step_radius",
                config.get("reference_clip_radius", anchor_radius),
            )
        )
        attack_phase = scheduled_attack_phase(
            config.get("attack"), int(config.get("_server_round", round_num))
        )
        freeze_on_attack = bool(
            config.get("temporal_reference_freeze_during_attack", False)
        )
        anchor_was_frozen = freeze_on_attack and attack_phase == "attack"
        effective_rate = 0.0 if anchor_was_frozen else configured_rate
        bounded_step = clip_l2(proposal - anchor, step_radius)
        next_anchor = clip_l2(anchor + effective_rate * bounded_step, anchor_radius)
        time_mode = str(config.get("dt_reference_time_mode", "current")).lower()
        if time_mode == "current":
            reference = proposal
        elif time_mode == "lagged_ema":
            reference = anchor
        else:
            raise ValueError("dt_reference_time_mode must be current or lagged_ema")
        self._dt_anchor = next_anchor.detach().cpu()
        reference_metrics.update(
            {
                "dtldp_reference_time_mode": time_mode,
                "dtldp_reference_is_strictly_lagged": time_mode == "lagged_ema",
                "dtldp_reference_state_initialized": bool(initialized),
                "dtldp_reference_proposal_norm": float(
                    torch.linalg.vector_norm(proposal).item()
                ),
                "dtldp_reference_proposal_from_prior_distance": float(
                    torch.linalg.vector_norm(proposal - anchor).item()
                ),
                "dtldp_reference_state_update_norm": float(
                    torch.linalg.vector_norm(next_anchor - anchor).item()
                ),
                "dtldp_reference_state_next_norm": float(
                    torch.linalg.vector_norm(next_anchor).item()
                ),
                "dtldp_reference_update_rate": configured_rate,
                "dtldp_reference_effective_update_rate": effective_rate,
                "dtldp_reference_step_radius": step_radius,
                "dtldp_reference_attack_phase": attack_phase,
                "dtldp_reference_freeze_during_attack": freeze_on_attack,
                "dtldp_reference_was_frozen": anchor_was_frozen,
            }
        )
        return reference, anchor, reference_metrics

    def _guard_reference(
        self, vectors: torch.Tensor, config: dict
    ) -> tuple[torch.Tensor, dict]:
        """Build the guard centre in the complete model-update space."""

        dimension = vectors.shape[1]
        if self._dt_full_anchor is None or self._dt_full_anchor.numel() != dimension:
            anchor = torch.zeros(dimension, dtype=vectors.dtype, device=vectors.device)
        else:
            anchor = self._dt_full_anchor.to(vectors)
        name = str(
            config.get("dt_guard_reference", self._reference_name(config))
        ).lower()
        aliases = {
            "fcc": "centered_clipping",
            "f_cc": "centered_clipping",
            "huber": "regularized_huber",
        }
        name = aliases.get(name, name)
        proposal, _ = self._reference_proposal(
            vectors,
            anchor,
            config,
            name=name,
            radius_key="dt_guard_reference_clip_radius",
        )
        rate = float(config.get("dt_guard_anchor_update_rate", 1.0))
        if not 0.0 <= rate <= 1.0:
            raise ValueError("dt_guard_anchor_update_rate must lie in [0,1]")
        anchor_radius = float(
            config.get("anchor_clip_norm", config.get("server_clip_norm", 1.0))
        )
        next_anchor = clip_l2((1.0 - rate) * anchor + rate * proposal, anchor_radius)
        self._dt_full_anchor = next_anchor.detach().cpu()
        return proposal, {
            "dtldp_guard_reference": name,
            "dtldp_guard_reference_norm": float(
                torch.linalg.vector_norm(proposal).item()
            ),
            "dtldp_guard_reference_drift_from_anchor": float(
                torch.linalg.vector_norm(proposal - anchor).item()
            ),
        }

    def server_aggregate(self, global_model, client_updates, round_num, config):
        if not client_updates:
            raise ValueError("DT-LDP-FAR needs at least one client update")
        if round_num == 0 or not hasattr(self, "_dt_previous_scores"):
            self._reset_server_state()

        client_ids = self._client_ids(client_updates)
        expected = config.get("expected_num_clients")
        if (
            bool(config.get("require_full_participation", True))
            and expected is not None
        ):
            if len(client_ids) != int(expected):
                raise ValueError(
                    "DT-LDP-FAR theoretical lane requires full participation: "
                    f"received {len(client_ids)}, expected {expected}"
                )

        updates = [update for update, _, _ in client_updates]
        vectors, layout = stack_updates(updates)
        server_clip = float(config.get("server_clip_norm", 1.0))
        upload_norms_pre_server = torch.linalg.vector_norm(vectors, dim=1)
        clipped, server_clip_factors = clip_rows(vectors, server_clip)
        upload_norms_post_server = torch.linalg.vector_norm(clipped, dim=1)
        score_vectors, score_space_metrics = score_subspace(
            clipped,
            layout,
            mode=str(config.get("score_subspace_mode", "full")),
            dimension=config.get("score_subspace_dimension"),
            seed=int(config.get("score_subspace_seed", 0)),
        )
        lagged_descent_reference = None
        if (
            self._dt_previous_reference is not None
            and self._dt_previous_reference.numel() == score_vectors.shape[1]
        ):
            lagged_descent_reference = self._dt_previous_reference.to(score_vectors)

        score_transform = str(
            config.get("dt_score_transform", "bounded_normalized")
        ).lower()
        if score_transform not in {"bounded_normalized", "raw_distance"}:
            raise ValueError(
                "dt_score_transform must be bounded_normalized or raw_distance"
            )
        noise_score_mode = str(
            config.get("noise_score_standardization", "none")
        ).lower()
        if score_transform == "raw_distance":
            if noise_score_mode != "none":
                raise ValueError(
                    "The raw_distance transform currently applies only to the "
                    "geometric FAR distance. Noise-aware direct scores retain "
                    "their own explicitly bounded semantics."
                )
            if self._reference_name(config) != "centered_clipping":
                raise ValueError(
                    "The certified raw-distance DT lane currently requires "
                    "the centered_clipping reference"
                )
            anchor_bound = float(config.get("anchor_clip_norm", server_clip))
            if anchor_bound > server_clip + 1e-12:
                raise ValueError(
                    "raw_distance requires anchor_clip_norm <= server_clip_norm "
                    "to certify d_i in [0,2U]"
                )
            public_score_range = 2.0 * server_clip
            delayed_score_bounds: tuple[float, float] | None = (
                0.0,
                public_score_range,
            )
        else:
            public_score_range = 1.0
            delayed_score_bounds = (0.0, 1.0)

        tau, tau_maximum, tau_requested_exceeds_cap = calibrated_tilt_tau(
            len(client_ids), config, public_score_range=public_score_range
        )
        tau_policy = str(config.get("tilt_bound_policy", "clip")).lower()
        delay = int(config.get("tilt_delay_rounds", 1))
        if delay < 1:
            raise ValueError("tilt_delay_rounds must be at least one")
        source_round = int(round_num) - delay
        source_scores = self._dt_score_history.get(source_round, {})
        filter_enabled = bool(config.get("dt_admissibility_filter_enabled", False))
        source_trust = self._dt_trust_history.get(source_round, {})
        if filter_enabled:
            excluded_clients = int(
                config.get(
                    "dt_admissibility_excluded_clients",
                    config.get("dt_admissibility_max_byzantine", 0),
                )
            )
            (
                delayed_weights,
                delayed_scores,
                delayed_eligible_mask,
                admissibility_filter_active,
            ) = delayed_filtered_weights_from_scores(
                client_ids,
                source_scores,
                source_trust,
                tau=tau,
                max_excluded=excluded_clients,
                unseen_score=float(config.get("unseen_client_score", 0.0)),
                score_bounds=delayed_score_bounds,
            )
        else:
            delayed_weights, delayed_scores = delayed_weights_from_scores(
                client_ids,
                source_scores,
                tau=tau,
                unseen_score=float(config.get("unseen_client_score", 0.0)),
                score_bounds=delayed_score_bounds,
            )
            delayed_eligible_mask = torch.ones(len(client_ids), dtype=torch.bool)
            admissibility_filter_active = False
        delayed_weights = delayed_weights.to(clipped)
        aggregate_mode = str(config.get("dt_aggregate_mode", "weighted_mean")).lower()
        aggregate_anchor_vector = None
        aggregate_mode_metrics: dict[str, float | str | bool | None] = {
            "dtldp_aggregate_mode": aggregate_mode,
            "dtldp_aggregate_anchor": None,
            "dtldp_aggregate_anchor_norm": None,
        }
        if aggregate_mode == "weighted_mean":
            aggregate_vector_raw = (delayed_weights[:, None] * clipped).sum(dim=0)
        elif aggregate_mode == "robust_anchor_perturbation":
            aggregate_anchor_name = str(
                config.get("dt_aggregate_anchor", "rfa")
            ).lower()
            aggregate_anchor_vector = aggregate_vectors(
                clipped,
                aggregate_anchor_name,
                num_byzantine=int(config.get("dt_support_assumed_byzantine", 0)),
                max_iter=int(config.get("dt_aggregate_anchor_max_iter", 100)),
                tol=float(config.get("dt_aggregate_anchor_tol", 1e-6)),
                smoothing=float(config.get("dt_aggregate_anchor_smoothing", 1e-8)),
            )
            residual_fraction = float(
                config.get("dt_anchor_residual_radius_fraction", 1.0)
            )
            if not 0.0 < residual_fraction <= 2.0:
                raise ValueError("dt_anchor_residual_radius_fraction must lie in (0,2]")
            residual_radius = residual_fraction * server_clip
            correction_gain = float(config.get("dt_anchor_correction_gain", 0.25))
            aggregate_vector_raw, correction_metrics = robust_anchor_perturbation(
                clipped,
                delayed_weights,
                aggregate_anchor_vector,
                residual_radius=residual_radius,
                gain=correction_gain,
            )
            aggregate_mode_metrics.update(
                {
                    "dtldp_aggregate_anchor": aggregate_anchor_name,
                    "dtldp_aggregate_anchor_norm": float(
                        torch.linalg.vector_norm(aggregate_anchor_vector).item()
                    ),
                    "dtldp_anchor_residual_radius_fraction": residual_fraction,
                    "dtldp_anchor_residual_radius": residual_radius,
                    "dtldp_anchor_correction_gain": correction_gain,
                    **correction_metrics,
                }
            )
        else:
            raise ValueError(
                "dt_aggregate_mode must be weighted_mean or "
                "robust_anchor_perturbation"
            )

        # The raw FAR aggregate is fixed before current references and scores
        # are computed.  Hence fresh scores never choose their own softmax
        # coefficient.  An optional robust output guard is a separate,
        # explicitly diagnosed post-processing layer.
        reference, anchor, reference_metrics = self._reference(
            score_vectors, config, round_num=round_num
        )
        raw_distances = torch.linalg.vector_norm(score_vectors - reference, dim=1)
        if noise_score_mode in ROBUST_DIRECT_SCORE_MODES:
            raise ValueError(
                "Generation-4 robust scores are currently implemented only "
                "for current-round DPFAR. Their delayed-score semantics must "
                "be specified separately before use in DT-LDP-FAR."
            )
        score_public_scales = (
            public_noise_scales(
                client_updates,
                device=raw_distances.device,
                dtype=raw_distances.dtype,
            )
            if noise_score_mode != "none"
            else None
        )
        configured_distance_clip = config.get("distance_clip")
        distance_clip = (
            float(configured_distance_clip)
            if configured_distance_clip is not None
            else float(config.get("distance_clip_multiple", 2.0)) * server_clip
        )
        residual_noise_variances = None
        upload_noise_variances = None
        utility_trust_scores: torch.Tensor | None = None
        if noise_score_mode in DIRECT_SCORE_MODES | EFFECTIVE_NULL_SCORE_MODES:
            upload_noise_variances, covariance_metrics = (
                effective_upload_noise_variances(
                    client_updates,
                    config=config,
                    server_clip_factors=server_clip_factors,
                    device=raw_distances.device,
                    dtype=raw_distances.dtype,
                )
            )
            residual_noise_variances = mean_reference_residual_variances(
                upload_noise_variances
            )
            score_dimension = int(score_space_metrics["score_subspace_dimension"])
            if noise_score_mode in EFFECTIVE_NULL_SCORE_MODES:
                calibration_seed = int(
                    config.get("noise_score_null_mc_seed", 20260906)
                ) + 100003 * int(round_num)
                null_mode = (
                    "moment"
                    if noise_score_mode == EFFECTIVE_NULL_MOMENT_SCORE_MODE
                    else "quantile"
                )
                reference_time_mode = (
                    "lagged_anchor"
                    if str(config.get("dt_reference_time_mode", "current")).lower()
                    == "lagged_ema"
                    else "current_loo"
                )
                (
                    novelty_scores,
                    score_references,
                    calibration_energies,
                    direct_score_metrics,
                ) = effective_null_fcc_loo_scores(
                    score_vectors,
                    upload_noise_variances,
                    anchor=anchor,
                    reference_radius=float(
                        config.get(
                            "reference_clip_radius",
                            config.get("server_clip_norm", 1.0),
                        )
                    ),
                    calibration_draws=int(config.get("noise_score_null_mc_draws", 64)),
                    calibration_seed=calibration_seed,
                    mode=null_mode,
                    reference_time_mode=reference_time_mode,
                    z_clip=float(config.get("noise_score_z_clip", 4.0)),
                    tail_probability=float(
                        config.get("noise_score_tail_probability", 0.90)
                    ),
                    individual_calibration_weight=float(
                        config.get("noise_score_individual_calibration_weight", 1.0)
                    ),
                    variance_ridge=float(
                        config.get("noise_score_variance_ridge", 1e-12)
                    ),
                )
                trust_profile = str(config.get("dt_trust_profile", "none")).lower()
                trust_metrics: dict = {"noise_score_trust_profile": trust_profile}
                if trust_profile == "none":
                    current_scores = novelty_scores
                elif trust_profile == "lagged_descent_alignment":
                    neutral_score = float(
                        config.get("noise_score_lagged_descent_neutral", 0.50)
                    )
                    if lagged_descent_reference is None:
                        utility_trust_scores = torch.full_like(
                            novelty_scores, neutral_score
                        )
                        current_scores = torch.full_like(novelty_scores, neutral_score)
                        trust_metrics.update(
                            {
                                "noise_score_lagged_descent_direction_available": False,
                                "noise_score_lagged_descent_direction_norm": 0.0,
                                "noise_score_lagged_descent_cold_start_uniform": True,
                                "noise_score_lagged_descent_trust_min": neutral_score,
                                "noise_score_lagged_descent_trust_mean": neutral_score,
                                "noise_score_lagged_descent_trust_max": neutral_score,
                                "noise_score_lagged_descent_is_loss_decrease_certificate": False,
                                "noise_score_lagged_descent_is_private_postprocessing": True,
                            }
                        )
                    else:
                        utility_trust_scores, utility_metrics = (
                            lagged_descent_alignment_scores(
                                score_vectors,
                                lagged_descent_reference,
                                reject_cosine=float(
                                    config.get(
                                        "noise_score_lagged_descent_reject_cosine",
                                        0.0,
                                    )
                                ),
                                full_support_cosine=float(
                                    config.get(
                                        "noise_score_lagged_descent_full_cosine",
                                        0.50,
                                    )
                                ),
                                neutral_score=neutral_score,
                            )
                        )
                        current_scores, channel_metrics = separate_novelty_trust_scores(
                            novelty_scores,
                            utility_trust_scores,
                            trust_fraction=float(
                                config.get("noise_score_trust_logit_fraction", 0.75)
                            ),
                        )
                        trust_metrics.update(
                            {
                                **utility_metrics,
                                **channel_metrics,
                                "noise_score_lagged_descent_cold_start_uniform": False,
                                "noise_score_combined_trust_rule": (
                                    "lagged_robust_direction_alignment"
                                ),
                            }
                        )
                elif trust_profile in {
                    "directional_independence",
                    "ranked_directional_peer_support",
                    "ranked_directional_peer_support_filter",
                }:
                    if trust_profile == "directional_independence":
                        directional_trust, directional_metrics = (
                            directional_trust_scores(
                                score_vectors,
                                reference,
                                reject_cosine=float(
                                    config.get("noise_score_reject_cosine", -0.10)
                                ),
                                full_trust_cosine=float(
                                    config.get("noise_score_full_trust_cosine", 0.25)
                                ),
                            )
                        )
                    else:
                        assumed_byzantine = int(
                            config.get(
                                "dt_support_assumed_byzantine",
                                config.get("num_byzantine", 0),
                            )
                        )
                        directional_trust, directional_metrics = (
                            ranked_directional_peer_support_scores(
                                score_vectors,
                                reference,
                                assumed_byzantine=assumed_byzantine,
                                reference_reject_cosine=float(
                                    config.get("noise_score_reject_cosine", -0.10)
                                ),
                                reference_full_support_cosine=float(
                                    config.get("noise_score_full_trust_cosine", 0.25)
                                ),
                                peer_reject_cosine=float(
                                    config.get("noise_score_peer_reject_cosine", 0.00)
                                ),
                                peer_full_support_cosine=float(
                                    config.get(
                                        "noise_score_peer_full_support_cosine", 0.20
                                    )
                                ),
                                extra_honest_supporters=int(
                                    config.get(
                                        "noise_score_peer_extra_honest_supporters",
                                        1,
                                    )
                                ),
                            )
                        )
                    independence_trust, independence_metrics = (
                        calibrated_independence_trust_scores(
                            score_vectors,
                            upload_noise_variances,
                            calibration_draws=int(
                                config.get("noise_score_null_mc_draws", 64)
                            ),
                            calibration_seed=calibration_seed + 7919,
                            low_null_quantile=float(
                                config.get(
                                    "noise_score_independence_low_quantile", 0.05
                                )
                            ),
                            full_trust_null_quantile=float(
                                config.get(
                                    "noise_score_independence_full_quantile", 0.50
                                )
                            ),
                            trust_floor=float(
                                config.get("noise_score_trust_floor", 0.05)
                            ),
                        )
                    )
                    combined_trust = torch.minimum(
                        directional_trust, independence_trust
                    )
                    utility_trust_scores = combined_trust
                    if trust_profile == "ranked_directional_peer_support_filter":
                        current_scores = novelty_scores
                        channel_metrics = {
                            "noise_score_channels_separated": True,
                            "noise_score_novelty_logit_fraction": 1.0,
                            "noise_score_trust_logit_fraction": 0.0,
                            "noise_score_trust_used_only_for_admissibility": True,
                        }
                    else:
                        current_scores, channel_metrics = separate_novelty_trust_scores(
                            novelty_scores,
                            combined_trust,
                            trust_fraction=float(
                                config.get("noise_score_trust_logit_fraction", 0.25)
                            ),
                        )
                    trust_metrics.update(
                        {
                            **directional_metrics,
                            **independence_metrics,
                            **channel_metrics,
                            "noise_score_combined_trust_rule": (
                                "minimum_of_ranked_directional_peer_and_independence"
                                if trust_profile
                                in {
                                    "ranked_directional_peer_support",
                                    "ranked_directional_peer_support_filter",
                                }
                                else "minimum"
                            ),
                        }
                    )
                elif trust_profile == "multi_krum_admissibility":
                    assumed_byzantine = int(
                        config.get(
                            "dt_support_assumed_byzantine",
                            config.get("num_byzantine", 0),
                        )
                    )
                    utility_trust_scores, krum_metrics = (
                        multi_krum_admissibility_trust_scores(
                            score_vectors,
                            assumed_byzantine=assumed_byzantine,
                        )
                    )
                    current_scores = novelty_scores
                    trust_metrics.update(
                        {
                            **krum_metrics,
                            "noise_score_channels_separated": True,
                            "noise_score_novelty_logit_fraction": 1.0,
                            "noise_score_trust_logit_fraction": 0.0,
                            "noise_score_trust_used_only_for_admissibility": True,
                            "noise_score_combined_trust_rule": (
                                "multi_krum_neighbourhood_ranking"
                            ),
                        }
                    )
                else:
                    raise ValueError(
                        "dt_trust_profile must be none, directional_independence "
                        "ranked_directional_peer_support, "
                        "ranked_directional_peer_support_filter, "
                        "multi_krum_admissibility or "
                        "lagged_descent_alignment"
                    )
                direct_score_metrics.update(trust_metrics)
                distances = torch.linalg.vector_norm(
                    score_vectors - score_references, dim=1
                )
                direct_score_metrics.update(
                    {
                        "noise_score_effective_null_calibration_energy_mean": float(
                            calibration_energies.mean().item()
                        ),
                        "noise_score_effective_null_novelty_min": float(
                            novelty_scores.min().item()
                        ),
                        "noise_score_effective_null_novelty_mean": float(
                            novelty_scores.mean().item()
                        ),
                        "noise_score_effective_null_novelty_max": float(
                            novelty_scores.max().item()
                        ),
                    }
                )
            elif noise_score_mode == ENERGY_SCORE_MODE:
                current_scores, direct_score_metrics = excess_energy_scores(
                    raw_distances,
                    residual_noise_variances,
                    score_dimension=score_dimension,
                    z_clip=float(config.get("noise_score_z_clip", 5.0)),
                )
                distances = raw_distances
            else:
                current_scores, direct_score_metrics = debiased_distance_scores(
                    raw_distances,
                    residual_noise_variances,
                    score_dimension=score_dimension,
                    distance_clip=distance_clip,
                )
                distances = raw_distances
            noise_score_divisors = residual_noise_variances.sqrt()
            noise_score_metrics = {
                "noise_score_standardization": noise_score_mode,
                "noise_score_residual_covariance_model": "mean_reference_proxy",
                "noise_score_residual_variance_min": float(
                    residual_noise_variances.min().item()
                ),
                "noise_score_residual_variance_mean": float(
                    residual_noise_variances.mean().item()
                ),
                "noise_score_residual_variance_max": float(
                    residual_noise_variances.max().item()
                ),
                **covariance_metrics,
                **direct_score_metrics,
            }
        else:
            distances, noise_score_divisors, noise_score_metrics = (
                standardize_distances(
                    raw_distances,
                    score_public_scales,
                    mode=noise_score_mode,
                    reference_variance_factor=config.get(
                        "noise_score_reference_variance_factor"
                    ),
                    variance_ridge=float(
                        config.get("noise_score_variance_ridge", 1e-12)
                    ),
                )
            )
            if score_transform == "raw_distance":
                current_scores = distances
            else:
                current_scores = bounded_distance_scores(
                    distances,
                    distance_clip,
                    str(config.get("score_mode", "hard_clip")),
                )
        current_score_span = float((current_scores.max() - current_scores.min()).item())
        if score_transform == "raw_distance" and float(current_scores.max().item()) > (
            public_score_range + 1e-6
        ):
            raise RuntimeError(
                "Raw DT-LDP-FAR score exceeded the certified public 2U range"
            )
        score_saturation_rate = (
            float((current_scores >= 1.0 - 1e-7).float().mean().item())
            if score_transform == "bounded_normalized"
            else None
        )
        informative_span = float(config.get("pilot_minimum_score_span", 0.02))
        current_weights = torch.softmax(tau * current_scores.double(), dim=0).to(
            clipped
        )
        current_weight_l2_squared = float(
            current_weights.double().square().sum().item()
        )

        guard_setting = config.get("dt_output_guard_radius_fraction")
        guard_metrics: dict[str, float | str | bool | None] = {
            "dtldp_output_guard_enabled": False,
            "dtldp_output_guard_radius_fraction": None,
            "dtldp_output_guard_radius": None,
            "dtldp_output_guard_active": False,
            "dtldp_output_guard_shift_norm": 0.0,
        }
        guard_reference = None
        if guard_setting is None or str(guard_setting).lower() == "raw":
            aggregate_vector = aggregate_vector_raw
        else:
            guard_fraction = float(guard_setting)
            if not 0.0 <= guard_fraction <= 1.0:
                raise ValueError(
                    "dt_output_guard_radius_fraction must lie in [0,1] or be raw"
                )
            guard_reference, full_reference_metrics = self._guard_reference(
                clipped, config
            )
            guard_radius = guard_fraction * server_clip
            aggregate_vector = guarded_aggregate(
                aggregate_vector_raw,
                guard_reference,
                radius=guard_radius,
            )
            guard_shift = torch.linalg.vector_norm(
                aggregate_vector - aggregate_vector_raw
            )
            raw_distance_to_guard = torch.linalg.vector_norm(
                aggregate_vector_raw - guard_reference
            )
            guard_metrics.update(
                {
                    **full_reference_metrics,
                    "dtldp_output_guard_enabled": True,
                    "dtldp_output_guard_radius_fraction": guard_fraction,
                    "dtldp_output_guard_radius": guard_radius,
                    "dtldp_output_guard_active": bool(
                        float(raw_distance_to_guard.item()) > guard_radius + 1e-12
                    ),
                    "dtldp_output_guard_raw_distance": float(
                        raw_distance_to_guard.item()
                    ),
                    "dtldp_output_guard_shift_norm": float(guard_shift.item()),
                }
            )
        aggregate = dict(unflatten_update(aggregate_vector, layout))

        previous_reference = self._dt_previous_reference
        reference_drift = 0.0
        if (
            previous_reference is not None
            and previous_reference.numel() == reference.numel()
        ):
            reference_drift = float(
                torch.linalg.vector_norm(
                    reference.detach().cpu() - previous_reference
                ).item()
            )
        self._dt_previous_reference = reference.detach().cpu()

        overlap_drifts = [
            abs(float(current_scores[index].item()) - self._dt_previous_scores[cid])
            for index, cid in enumerate(client_ids)
            if cid in self._dt_previous_scores
        ]
        score_drift_linf = max(overlap_drifts, default=0.0)
        previous_coverage = sum(cid in source_scores for cid in client_ids)
        new_scores = {
            cid: float(current_scores[index].item())
            for index, cid in enumerate(client_ids)
        }
        self._dt_score_history[int(round_num)] = new_scores
        self._dt_previous_scores = new_scores
        new_trust = (
            {
                cid: float(utility_trust_scores[index].item())
                for index, cid in enumerate(client_ids)
            }
            if utility_trust_scores is not None
            else {}
        )
        self._dt_trust_history[int(round_num)] = new_trust

        malicious_mask = torch.tensor(
            [
                bool(metadata.get("is_byzantine", False))
                for _, metadata, _ in client_updates
            ],
            device=clipped.device,
        )
        honest_mask = ~malicious_mask
        delayed_eligible_mask = delayed_eligible_mask.to(malicious_mask.device)
        server_clipped_mask = server_clip_factors < 1.0

        def masked_mean(values, mask):
            return (
                float(values[mask].mean().item()) if bool(mask.any().item()) else None
            )

        def masked_max(values, mask):
            return float(values[mask].max().item()) if bool(mask.any().item()) else None

        weighted_rows = delayed_weights[:, None] * clipped
        honest_contribution_norm = float(
            torch.linalg.vector_norm(weighted_rows[honest_mask].sum(dim=0)).item()
        )
        byzantine_contribution_norm = (
            float(
                torch.linalg.vector_norm(
                    weighted_rows[malicious_mask].sum(dim=0)
                ).item()
            )
            if bool(malicious_mask.any().item())
            else 0.0
        )
        honest_center_full = clipped[honest_mask].mean(dim=0)
        raw_aggregate_honest_error = torch.linalg.vector_norm(
            aggregate_vector_raw - honest_center_full
        )
        released_aggregate_honest_error = torch.linalg.vector_norm(
            aggregate_vector - honest_center_full
        )
        guard_reference_honest_error = (
            torch.linalg.vector_norm(guard_reference - honest_center_full)
            if guard_reference is not None
            else None
        )
        aggregate_anchor_honest_error = (
            torch.linalg.vector_norm(aggregate_anchor_vector - honest_center_full)
            if aggregate_anchor_vector is not None
            else None
        )
        aggregate_mode_metrics["dtldp_aggregate_anchor_honest_center_error_oracle"] = (
            float(aggregate_anchor_honest_error.item())
            if aggregate_anchor_honest_error is not None
            else None
        )
        diagnostics = weight_diagnostics(delayed_weights, malicious_mask)
        if utility_trust_scores is not None:
            trust_honest = masked_mean(utility_trust_scores, honest_mask)
            trust_byzantine = masked_mean(utility_trust_scores, malicious_mask)
            trust_margin = (
                trust_honest - trust_byzantine
                if bool(malicious_mask.any().item())
                else None
            )
            diagnostics.update(
                {
                    "dtldp_admissibility_trust_honest_oracle": trust_honest,
                    "dtldp_admissibility_trust_byzantine_oracle": trust_byzantine,
                    "dtldp_admissibility_trust_margin_oracle": trust_margin,
                    "dtldp_current_trust_by_client": {
                        str(cid): float(utility_trust_scores[index].item())
                        for index, cid in enumerate(client_ids)
                    },
                }
            )
            if trust_profile == "lagged_descent_alignment":
                diagnostics.update(
                    {
                        "dtldp_lagged_descent_trust_honest_oracle": trust_honest,
                        "dtldp_lagged_descent_trust_byzantine_oracle": trust_byzantine,
                        "dtldp_lagged_descent_trust_margin_oracle": trust_margin,
                    }
                )
        l2_weight_sq = float(delayed_weights.double().square().sum().item())
        kappa_w = float(config.get("kappa_w", 2.0))
        diagnostics.update(
            {
                "dtldp_algorithm_variant": "delayed_tilting_local_sample_dp",
                "dtldp_weights_source_round": source_round,
                "dtldp_tilt_delay_rounds": delay,
                "dtldp_initial_weights_uniform": bool(source_round < 0),
                "dtldp_previous_score_coverage": previous_coverage / len(client_ids),
                "dtldp_tilt_tau_requested": float(
                    config.get("tilt_tau", config.get("far_alpha", 0.1))
                ),
                "dtldp_tilt_tau": tau,
                "dtldp_tilt_tau_max": tau_maximum,
                "dtldp_tilt_public_score_range": public_score_range,
                "dtldp_tilt_tau_requested_exceeds_cap": tau_requested_exceeds_cap,
                "dtldp_tilt_tau_was_clipped": bool(
                    tau_requested_exceeds_cap and tau_policy == "clip"
                ),
                "dtldp_tilt_influence_certificate_claimed": bool(
                    tau <= tau_maximum + 1e-12
                ),
                "dtldp_kappa_w": kappa_w,
                "dtldp_weight_cap": kappa_w / len(client_ids),
                "dtldp_weight_cap_respected": bool(
                    float(delayed_weights.max().item())
                    <= kappa_w / len(client_ids) + 1e-10
                ),
                "dtldp_admissibility_filter_enabled": filter_enabled,
                "dtldp_admissibility_filter_active": admissibility_filter_active,
                "dtldp_admissibility_max_byzantine": int(
                    config.get("dt_admissibility_max_byzantine", 0)
                ),
                "dtldp_admissibility_excluded_clients": int(
                    config.get(
                        "dt_admissibility_excluded_clients",
                        config.get("dt_admissibility_max_byzantine", 0),
                    )
                ),
                "dtldp_admissible_client_count": int(
                    delayed_eligible_mask.sum().item()
                ),
                "dtldp_admissible_byzantine_count_oracle": int(
                    (delayed_eligible_mask & malicious_mask).sum().item()
                ),
                "dtldp_admissible_honest_count_oracle": int(
                    (delayed_eligible_mask & honest_mask).sum().item()
                ),
                "dtldp_admissible_by_client": {
                    str(cid): bool(delayed_eligible_mask[index].item())
                    for index, cid in enumerate(client_ids)
                },
                "dtldp_weight_l2_squared": l2_weight_sq,
                "dtldp_noise_amplification_vs_uniform": len(client_ids) * l2_weight_sq,
                "dtldp_delayed_score_min": float(delayed_scores.min().item()),
                "dtldp_delayed_score_max": float(delayed_scores.max().item()),
                "dtldp_client_ids": list(client_ids),
                "dtldp_delayed_scores_by_client": {
                    str(cid): float(delayed_scores[index].item())
                    for index, cid in enumerate(client_ids)
                },
                "dtldp_delayed_weights_by_client": {
                    str(cid): float(delayed_weights[index].item())
                    for index, cid in enumerate(client_ids)
                },
                "dtldp_current_score_min": float(current_scores.min().item()),
                "dtldp_current_score_mean": float(current_scores.mean().item()),
                "dtldp_current_score_max": float(current_scores.max().item()),
                "dtldp_current_score_span": current_score_span,
                "dtldp_current_logit_span": tau * current_score_span,
                "dtldp_current_score_saturation_rate": score_saturation_rate,
                "dtldp_current_score_all_saturated": bool(
                    score_saturation_rate is not None
                    and score_saturation_rate >= 1.0 - 1e-12
                ),
                "dtldp_pilot_minimum_score_span": informative_span,
                "dtldp_pilot_tilting_informative": bool(
                    current_score_span >= informative_span
                    and (
                        score_saturation_rate is None
                        or score_saturation_rate < 1.0 - 1e-12
                    )
                ),
                "dtldp_score_transform": score_transform,
                "dtldp_score_is_unit_bounded": (
                    score_transform == "bounded_normalized"
                ),
                "dtldp_score_public_range_respected": bool(
                    float(current_scores.min().item()) >= -1e-7
                    and float(current_scores.max().item())
                    <= public_score_range + 1e-6
                ),
                "dtldp_current_scores_by_client": {
                    str(cid): float(current_scores[index].item())
                    for index, cid in enumerate(client_ids)
                },
                "dtldp_next_weights_by_client": {
                    str(cid): float(current_weights[index].item())
                    for index, cid in enumerate(client_ids)
                },
                "dtldp_score_drift_linf": score_drift_linf,
                "dtldp_delayed_current_weight_l1": float(
                    torch.linalg.vector_norm(
                        delayed_weights - current_weights, ord=1
                    ).item()
                ),
                "dtldp_current_max_client_weight": float(current_weights.max().item()),
                "dtldp_current_weight_l2_squared": current_weight_l2_squared,
                "dtldp_current_noise_amplification_vs_uniform": (
                    len(client_ids) * current_weight_l2_squared
                ),
                "dtldp_staleness_aggregate_norm": float(
                    torch.linalg.vector_norm(
                        ((delayed_weights - current_weights)[:, None] * clipped).sum(0)
                    ).item()
                ),
                "dtldp_raw_aggregate_norm": float(
                    torch.linalg.vector_norm(aggregate_vector_raw).item()
                ),
                "dtldp_released_aggregate_norm": float(
                    torch.linalg.vector_norm(aggregate_vector).item()
                ),
                "dtldp_raw_aggregate_honest_center_error_oracle": float(
                    raw_aggregate_honest_error.item()
                ),
                "dtldp_released_aggregate_honest_center_error_oracle": float(
                    released_aggregate_honest_error.item()
                ),
                "dtldp_guard_reference_honest_center_error_oracle": (
                    float(guard_reference_honest_error.item())
                    if guard_reference_honest_error is not None
                    else None
                ),
                "dtldp_guard_error_reduction_vs_raw_oracle": float(
                    1.0
                    - released_aggregate_honest_error.item()
                    / max(raw_aggregate_honest_error.item(), 1e-30)
                ),
                "dtldp_reference": self._reference_name(config),
                "dtldp_reference_norm": float(
                    torch.linalg.vector_norm(reference).item()
                ),
                "dtldp_reference_drift": reference_drift,
                "dtldp_anchor_norm": float(torch.linalg.vector_norm(anchor).item()),
                "dtldp_server_clip_norm": server_clip,
                "dtldp_server_clip_rate": float(
                    server_clipped_mask.float().mean().item()
                ),
                # Oracle-only group labels are used for evaluation, never to
                # construct the reference, scores, weights, or aggregate.
                "dtldp_server_clip_rate_honest_oracle": masked_mean(
                    server_clipped_mask.float(), honest_mask
                ),
                "dtldp_server_clip_rate_byzantine_oracle": masked_mean(
                    server_clipped_mask.float(), malicious_mask
                ),
                "dtldp_upload_norm_pre_server_min": float(
                    upload_norms_pre_server.min().item()
                ),
                "dtldp_upload_norm_pre_server_mean": float(
                    upload_norms_pre_server.mean().item()
                ),
                "dtldp_upload_norm_pre_server_max": float(
                    upload_norms_pre_server.max().item()
                ),
                "dtldp_upload_norm_post_server_mean": float(
                    upload_norms_post_server.mean().item()
                ),
                "dtldp_upload_norm_post_server_max": float(
                    upload_norms_post_server.max().item()
                ),
                "dtldp_upload_norm_pre_server_mean_honest_oracle": masked_mean(
                    upload_norms_pre_server, honest_mask
                ),
                "dtldp_upload_norm_pre_server_max_honest_oracle": masked_max(
                    upload_norms_pre_server, honest_mask
                ),
                "dtldp_upload_norm_post_server_mean_honest_oracle": masked_mean(
                    upload_norms_post_server, honest_mask
                ),
                "dtldp_upload_norm_post_server_max_honest_oracle": masked_max(
                    upload_norms_post_server, honest_mask
                ),
                "dtldp_upload_norm_pre_server_mean_byzantine_oracle": masked_mean(
                    upload_norms_pre_server, malicious_mask
                ),
                "dtldp_upload_norm_pre_server_max_byzantine_oracle": masked_max(
                    upload_norms_pre_server, malicious_mask
                ),
                "dtldp_upload_norm_post_server_mean_byzantine_oracle": masked_mean(
                    upload_norms_post_server, malicious_mask
                ),
                "dtldp_upload_norm_post_server_max_byzantine_oracle": masked_max(
                    upload_norms_post_server, malicious_mask
                ),
                "dtldp_honest_weighted_contribution_norm_oracle": (
                    honest_contribution_norm
                ),
                "dtldp_byzantine_weighted_contribution_norm_oracle": (
                    byzantine_contribution_norm
                ),
                # Ratio above one means at least one upload crosses the
                # server clipping radius U. This is derived only from already
                # private uploads and is safe server-side post-processing.
                "dtldp_server_clip_utilization_max": float(
                    upload_norms_pre_server.max().item() / server_clip
                ),
                "dtldp_distance_clip": distance_clip,
                # Preferred scientific name. Keep dtldp_distance_clip as a
                # backward-compatible metric alias for existing dashboards.
                "dtldp_score_scale": distance_clip,
                "dtldp_distance_min": float(distances.min().item()),
                "dtldp_distance_mean": float(distances.mean().item()),
                "dtldp_distance_max": float(distances.max().item()),
                "dtldp_raw_distance_min": float(raw_distances.min().item()),
                "dtldp_raw_distance_mean": float(raw_distances.mean().item()),
                "dtldp_raw_distance_max": float(raw_distances.max().item()),
                "dtldp_current_score_weight_corr": _pearson_or_none(
                    current_scores, current_weights
                ),
                **{f"dtldp_{key}": value for key, value in noise_score_metrics.items()},
                **{f"dtldp_{key}": value for key, value in score_space_metrics.items()},
                **aggregate_mode_metrics,
                **reference_metrics,
                **guard_metrics,
            }
        )

        if bool(config.get("enable_oracle_diagnostics", False)):
            loss_values = [
                metadata.get("dtldp_local_loss_oracle")
                for _, metadata, _ in client_updates
            ]
            clip_values = [
                metadata.get("dtldp_clip_rate_oracle")
                for _, metadata, _ in client_updates
            ]
            if all(value is not None for value in loss_values):
                diagnostics["dtldp_local_loss_mean_oracle"] = float(
                    sum(float(value) for value in loss_values) / len(loss_values)
                )
            if all(value is not None for value in clip_values):
                diagnostics["dtldp_client_clip_rate_mean_oracle"] = float(
                    sum(float(value) for value in clip_values) / len(clip_values)
                )
            noise_values = [
                metadata.get("dtldp_realised_noise_norm_oracle")
                for _, metadata, _ in client_updates
            ]
            if all(value is not None for value in noise_values):
                noise = torch.tensor(noise_values, dtype=torch.float64)
                diagnostics.update(
                    {
                        "dtldp_realised_noise_norm_mean_oracle": float(
                            noise.mean().item()
                        ),
                        "dtldp_delayed_weight_noise_corr_oracle": _pearson_or_none(
                            delayed_weights, noise
                        ),
                        "dtldp_current_score_noise_corr_oracle": _pearson_or_none(
                            current_scores, noise
                        ),
                    }
                )
            clean_updates = [
                metadata.get("dtldp_noise_free_update_oracle")
                for _, metadata, _ in client_updates
            ]
            if all(isinstance(value, dict) for value in clean_updates):
                clean_vectors, clean_layout = stack_updates(clean_updates)
                if (
                    clean_layout.keys != layout.keys
                    or clean_layout.shapes != layout.shapes
                ):
                    raise ValueError("Noise-free counterfactual layout mismatch")
                clean_clipped, _ = clip_rows(clean_vectors, server_clip)
                clean_score_vectors, clean_score_metrics = score_subspace(
                    clean_clipped,
                    clean_layout,
                    mode=str(config.get("score_subspace_mode", "full")),
                    dimension=config.get("score_subspace_dimension"),
                    seed=int(config.get("score_subspace_seed", 0)),
                )
                if (
                    clean_score_metrics["score_subspace_dimension"]
                    != score_space_metrics["score_subspace_dimension"]
                ):
                    raise ValueError("Counterfactual score-space dimension mismatch")
                effective_full = clipped - clean_clipped
                effective_score = score_vectors - clean_score_vectors
                effective_full_norm = torch.linalg.vector_norm(effective_full, dim=1)
                effective_score_norm = torch.linalg.vector_norm(effective_score, dim=1)
                public_scales = torch.tensor(
                    [
                        float(
                            metadata.get("privacy_noise_multiplier_scale_public", 1.0)
                        )
                        for _, metadata, _ in client_updates
                    ],
                    dtype=torch.float64,
                )
                normalized_effective_score_norm = (
                    effective_score_norm.double() / public_scales.clamp_min(1e-12)
                )
                delayed_noise_vector = (delayed_weights[:, None] * effective_full).sum(
                    dim=0
                )
                current_noise_vector = (current_weights[:, None] * effective_full).sum(
                    dim=0
                )
                reference_name = self._reference_name(config)
                if reference_name == "centered_clipping":
                    clean_reference = centered_clipping(
                        clean_score_vectors,
                        anchor=anchor,
                        tau=float(
                            config.get(
                                "reference_clip_radius",
                                config.get("server_clip_norm", 1.0),
                            )
                        ),
                    )
                elif reference_name == "regularized_huber":
                    clean_reference = regularized_huber_reference(
                        clean_score_vectors,
                        anchor=anchor,
                        tau=float(
                            config.get(
                                "reference_clip_radius",
                                config.get("server_clip_norm", 1.0),
                            )
                        ),
                        gamma=float(config.get("huber_gamma", 1.0)),
                        num_steps=int(config.get("huber_num_steps", 10)),
                    )
                else:
                    clean_reference = aggregate_vectors(
                        clean_score_vectors,
                        reference_name,
                        num_byzantine=int(config.get("num_byzantine", 0)),
                        screening_fraction=config.get("screening_fraction"),
                        max_iter=int(config.get("rfa_max_iter", 100)),
                        tol=float(config.get("rfa_tol", 1e-6)),
                        smoothing=float(config.get("rfa_smoothing", 1e-8)),
                        alpha_trusted=float(config.get("cmls_alpha_trusted", 1.0)),
                        alpha_suspected=float(config.get("cmls_alpha_suspected", 1.0)),
                    )
                clean_raw_distances = torch.linalg.vector_norm(
                    clean_score_vectors - clean_reference, dim=1
                )
                if noise_score_mode in DIRECT_SCORE_MODES:
                    if residual_noise_variances is None:
                        raise RuntimeError(
                            "Missing residual covariance for direct score"
                        )
                    clean_noise_divisors = residual_noise_variances.sqrt()
                    clean_distances = clean_raw_distances
                    if noise_score_mode == ENERGY_SCORE_MODE:
                        clean_scores, _ = excess_energy_scores(
                            clean_raw_distances,
                            residual_noise_variances,
                            score_dimension=int(
                                score_space_metrics["score_subspace_dimension"]
                            ),
                            z_clip=float(config.get("noise_score_z_clip", 5.0)),
                            subtract_noise_floor=False,
                        )
                    else:
                        clean_scores, _ = debiased_distance_scores(
                            clean_raw_distances,
                            residual_noise_variances,
                            score_dimension=int(
                                score_space_metrics["score_subspace_dimension"]
                            ),
                            distance_clip=distance_clip,
                            subtract_noise_floor=False,
                        )
                else:
                    clean_distances, clean_noise_divisors, _ = standardize_distances(
                        clean_raw_distances,
                        public_scales.to(clean_raw_distances),
                        mode=noise_score_mode,
                        reference_variance_factor=config.get(
                            "noise_score_reference_variance_factor"
                        ),
                        variance_ridge=float(
                            config.get("noise_score_variance_ridge", 1e-12)
                        ),
                    )
                    if score_transform == "raw_distance":
                        clean_scores = clean_distances
                    else:
                        clean_scores = bounded_distance_scores(
                            clean_distances,
                            distance_clip,
                            str(config.get("score_mode", "hard_clip")),
                        )
                if not torch.allclose(
                    noise_score_divisors.double().cpu(),
                    clean_noise_divisors.double().cpu(),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Public noise-score divisors changed in oracle path"
                    )
                clean_current_weights = torch.softmax(
                    tau * clean_scores.double(), dim=0
                ).to(clipped)
                raw_clean_scores = (
                    clean_raw_distances
                    if score_transform == "raw_distance"
                    else bounded_distance_scores(
                        clean_raw_distances,
                        distance_clip,
                        str(config.get("score_mode", "hard_clip")),
                    )
                )
                raw_clean_current_weights = torch.softmax(
                    tau * raw_clean_scores.double(), dim=0
                ).to(clipped)
                diagnostics.update(
                    {
                        "dtldp_effective_fresh_noise_norm_mean_oracle": float(
                            effective_full_norm.mean().item()
                        ),
                        "dtldp_effective_fresh_score_noise_norm_mean_oracle": float(
                            effective_score_norm.mean().item()
                        ),
                        "dtldp_delayed_weight_effective_noise_corr_oracle": (
                            _pearson_or_none(delayed_weights, effective_score_norm)
                        ),
                        "dtldp_current_weight_effective_noise_corr_oracle": (
                            _pearson_or_none(current_weights, effective_score_norm)
                        ),
                        "dtldp_delayed_weight_normalized_effective_noise_corr_oracle": (
                            _pearson_or_none(
                                delayed_weights,
                                normalized_effective_score_norm,
                            )
                        ),
                        "dtldp_current_weight_normalized_effective_noise_corr_oracle": (
                            _pearson_or_none(
                                current_weights,
                                normalized_effective_score_norm,
                            )
                        ),
                        "dtldp_delayed_fixed_weight_fresh_noise_sq_error_oracle": float(
                            delayed_noise_vector.square().sum().item()
                        ),
                        "dtldp_current_fixed_weight_fresh_noise_sq_error_oracle": float(
                            current_noise_vector.square().sum().item()
                        ),
                        "dtldp_current_vs_delayed_fresh_noise_sq_error_ratio_oracle": float(
                            current_noise_vector.square().sum().item()
                            / max(delayed_noise_vector.square().sum().item(), 1e-30)
                        ),
                        "dtldp_current_noisy_clean_score_corr_oracle": (
                            _pearson_or_none(current_scores, clean_scores)
                        ),
                        "dtldp_current_noisy_clean_score_mae_oracle": float(
                            (current_scores - clean_scores).abs().mean().item()
                        ),
                        "dtldp_current_noisy_clean_score_rmse_oracle": float(
                            (current_scores - clean_scores)
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "dtldp_current_noisy_clean_weight_l1_oracle": float(
                            torch.linalg.vector_norm(
                                current_weights - clean_current_weights, ord=1
                            ).item()
                        ),
                        "dtldp_noisy_clean_raw_distance_corr_oracle": (
                            _pearson_or_none(raw_distances, clean_raw_distances)
                        ),
                        "dtldp_current_score_public_noise_scale_corr_oracle": (
                            _pearson_or_none(
                                current_scores, public_scales.to(current_scores)
                            )
                        ),
                        "dtldp_clean_score_public_noise_scale_corr_oracle": (
                            _pearson_or_none(
                                clean_scores, public_scales.to(clean_scores)
                            )
                        ),
                        "dtldp_current_noisy_raw_clean_score_corr_oracle": (
                            _pearson_or_none(current_scores, raw_clean_scores)
                        ),
                        "dtldp_current_noisy_raw_clean_score_mae_oracle": float(
                            (current_scores - raw_clean_scores).abs().mean().item()
                        ),
                        "dtldp_current_noisy_raw_clean_score_rmse_oracle": float(
                            (current_scores - raw_clean_scores)
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "dtldp_current_noisy_raw_clean_weight_l1_oracle": float(
                            torch.linalg.vector_norm(
                                current_weights - raw_clean_current_weights, ord=1
                            ).item()
                        ),
                        "dtldp_raw_clean_score_public_noise_scale_corr_oracle": (
                            _pearson_or_none(
                                raw_clean_scores,
                                public_scales.to(raw_clean_scores),
                            )
                        ),
                    }
                )
            honest = ~malicious_mask
            if bool(honest.any()):
                # ``reference`` is built in the configured score subspace.  The
                # oracle centre must live in that same space (not in the full
                # model-update space) before their distance can be evaluated.
                honest_center = score_vectors[honest].mean(dim=0)
                diagnostics["dtldp_reference_honest_center_error_oracle"] = float(
                    torch.linalg.vector_norm(reference - honest_center).item()
                )

        metrics = common_round_metrics(client_updates)
        metrics.update({"round": round_num, **diagnostics})
        result = AggregateResult(apply_delta(global_model, aggregate), metrics)
        return self._add_local_dp_round_metrics(result, client_updates, config)

    def get_default_config(self):
        return {
            **FAR.get_default_config(self),
            **_dp_defaults(),
            "enable_dp": True,
            "tilt_tau": 0.1,
            "kappa_w": 2.0,
            "tilt_bound_policy": "clip",
            "server_clip_norm": 1.0,
            "distance_clip_multiple": 2.0,
            "distance_clip": None,
            "dt_score_transform": "bounded_normalized",
            "score_mode": "hard_clip",
            "unseen_client_score": 0.0,
            "tilt_delay_rounds": 1,
            "dt_reference": "centered_clipping",
            "dt_reference_time_mode": "current",
            "reference_clip_radius": 1.0,
            "huber_gamma": 1.0,
            "huber_num_steps": 10,
            "anchor_update_rate": 0.1,
            "anchor_clip_norm": 1.0,
            "temporal_reference_step_radius": 1.0,
            "temporal_reference_freeze_during_attack": False,
            "dt_output_guard_radius_fraction": None,
            "dt_guard_reference": "centered_clipping",
            "dt_guard_reference_clip_radius": 1.0,
            "dt_guard_anchor_update_rate": 1.0,
            "score_subspace_mode": "full",
            "score_subspace_dimension": None,
            "score_subspace_seed": 0,
            "noise_score_standardization": "none",
            "noise_score_reference_variance_factor": None,
            "noise_score_variance_ridge": 1e-12,
            "noise_score_z_clip": 5.0,
            "noise_score_tail_probability": 0.90,
            "noise_score_individual_calibration_weight": 1.0,
            "noise_score_null_mc_draws": 64,
            "noise_score_null_mc_seed": 20260906,
            "dt_trust_profile": "none",
            "noise_score_trust_logit_fraction": 0.25,
            "noise_score_reject_cosine": -0.10,
            "noise_score_full_trust_cosine": 0.25,
            "noise_score_independence_low_quantile": 0.05,
            "noise_score_independence_full_quantile": 0.50,
            "noise_score_trust_floor": 0.05,
            "dt_admissibility_filter_enabled": False,
            "dt_admissibility_max_byzantine": 0,
            "dt_aggregate_mode": "weighted_mean",
            "dt_aggregate_anchor": "rfa",
            "dt_aggregate_anchor_max_iter": 100,
            "dt_aggregate_anchor_tol": 1e-6,
            "dt_aggregate_anchor_smoothing": 1e-8,
            "dt_anchor_residual_radius_fraction": 1.0,
            "dt_anchor_correction_gain": 0.25,
            "noise_score_include_server_contraction": True,
            "require_full_participation": True,
            "expected_num_clients": None,
            "enable_oracle_diagnostics": False,
            "enable_private_client_oracle_diagnostics": False,
            "client_metrics_every": 5,
            "fairness_tail_fraction": 0.2,
            "pilot_minimum_score_span": 0.02,
        }
