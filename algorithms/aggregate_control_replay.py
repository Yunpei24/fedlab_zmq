"""Shadow replay of causal aggregate controls on real private gradients.

The deployed model preserves the historical comparator exactly: uniform
aggregation during rounds 1--12, followed by the unchanged FAR rule using an
RFA reference.  Every aggregate controller is evaluated on that exact same
private cohort but remains a shadow: it cannot select the model update.  Clean
simulator gradients are joined only by
:func:`evaluate_aggregate_control_replay`, after aggregation, and only scalar
diagnostics are returned to the experiment log.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from algorithms.aggregate_radial_control import (
    AggregateControlConfig,
    PredictableAggregateState,
    apply_aggregate_control,
    diagonal_mahalanobis_norm,
    stable_l2_norm,
)
from algorithms.ldp_aggregation_role_ablation import (
    LDPAggregationRoleAblation,
)
from metrics.aggregation_role_evaluation import evaluate_aggregation_roles
from robustness.tensor_ops import stack_updates


REPLAY_KIND = "aggregate_control_replay_v1"


def _label_float(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def _finite_scalar(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _positive_list(config: dict, key: str) -> list[float]:
    values = config.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty list")
    resolved = [_finite_scalar(value, name=key) for value in values]
    if any(value <= 0.0 for value in resolved):
        raise ValueError(f"{key} entries must be positive")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{key} entries must be unique")
    return resolved


def _control_config(
    config: dict, *, public_initial_variance: float
) -> AggregateControlConfig:
    if config.get("aggregate_control_initial_variance_mode") != (
        "public_pre_server_clip_uniform_mean_noise_proxy"
    ):
        raise ValueError("unexpected aggregate-control initial variance mode")
    return AggregateControlConfig(
        predictor_rate=float(config["aggregate_control_predictor_rate"]),
        covariance_rate=float(config["aggregate_control_covariance_rate"]),
        initial_variance=float(public_initial_variance),
        variance_ridge=float(config["aggregate_control_variance_ridge"]),
        min_history=int(config["aggregate_control_min_history"]),
        projection_tolerance=float(config["aggregate_control_projection_tolerance"]),
        projection_max_iterations=int(
            config["aggregate_control_projection_max_iterations"]
        ),
    )


def diagnostic_control_candidates(
    candidate: torch.Tensor,
    snapshot,
    *,
    config: dict,
    num_clients: int,
    server_clip_norm: float,
) -> dict[str, dict[str, Any]]:
    """Construct the preregistered, explicitly uncalibrated shadow grid."""

    if int(num_clients) < 2:
        raise ValueError("aggregate controls require at least two clients")
    dimension = int(candidate.numel())
    euclidean_nominal = float(server_clip_norm) / math.sqrt(num_clients)
    mahalanobis_nominal = math.sqrt(dimension)
    if not math.isfinite(euclidean_nominal) or euclidean_nominal <= 0.0:
        raise ValueError("invalid public Euclidean nominal radius")
    candidates: dict[str, dict[str, Any]] = {}

    base, diagnostics = apply_aggregate_control("unchanged", candidate, snapshot)
    candidates["unchanged"] = {"vector": base, "diagnostics": diagnostics}

    for mix in _positive_list(config, "aggregate_control_ema_current_mixes"):
        if mix > 1.0:
            raise ValueError("EMA current mixes must not exceed one")
        label = f"ema_mix_{_label_float(mix)}"
        output, diagnostics = apply_aggregate_control(
            "ema", candidate, snapshot, current_mix=mix
        )
        candidates[label] = {"vector": output, "diagnostics": diagnostics}

    for multiplier in _positive_list(
        config, "aggregate_control_isotropic_radius_multipliers"
    ):
        radius = multiplier * euclidean_nominal
        label = f"isotropic_mult_{_label_float(multiplier)}"
        output, diagnostics = apply_aggregate_control(
            "isotropic_clip", candidate, snapshot, radius=radius
        )
        diagnostics["radius_multiplier"] = multiplier
        diagnostics["radius_nominal"] = euclidean_nominal
        candidates[label] = {"vector": output, "diagnostics": diagnostics}

    for multiplier in _positive_list(
        config, "aggregate_control_mahalanobis_radius_multipliers"
    ):
        radius = multiplier * mahalanobis_nominal
        radial_label = f"radial_mult_{_label_float(multiplier)}"
        radial, radial_diagnostics = apply_aggregate_control(
            "radial_ellipsoid", candidate, snapshot, radius=radius
        )
        radial_diagnostics["radius_multiplier"] = multiplier
        radial_diagnostics["radius_nominal"] = mahalanobis_nominal
        candidates[radial_label] = {
            "vector": radial,
            "diagnostics": radial_diagnostics,
        }

        projection_label = f"euclidean_projection_mult_{_label_float(multiplier)}"
        projected, projection_diagnostics = apply_aggregate_control(
            "euclidean_ellipsoid_projection",
            candidate,
            snapshot,
            radius=radius,
            projection_tolerance=float(
                config["aggregate_control_projection_tolerance"]
            ),
            projection_max_iterations=int(
                config["aggregate_control_projection_max_iterations"]
            ),
        )
        projection_diagnostics["radius_multiplier"] = multiplier
        projection_diagnostics["radius_nominal"] = mahalanobis_nominal
        candidates[projection_label] = {
            "vector": projected,
            "diagnostics": projection_diagnostics,
        }

    if len(candidates) != 1 + len(
        config["aggregate_control_ema_current_mixes"]
    ) + len(config["aggregate_control_isotropic_radius_multipliers"]) + 2 * len(
        config["aggregate_control_mahalanobis_radius_multipliers"]
    ):
        raise RuntimeError("aggregate control labels collided")
    return candidates


class AggregateControlReplay(LDPAggregationRoleAblation):
    """Historical uniform/FAR(RFA) path plus same-cohort control shadows."""

    def server_aggregate(self, global_model, client_updates, round_num, config):
        if config.get("aggregate_control_replay_phase") != "headroom_fit":
            raise ValueError("this runtime implements only headroom_fit")
        if config.get("aggregate_control_model_driver") != (
            "uniform_t1_t12_then_far_rfa"
        ):
            raise ValueError("model driver must preserve the historical warmup")
        if config.get("aggregation_role_arm") != "far_rfa":
            raise ValueError("parent aggregation arm must be far_rfa")
        result = super().server_aggregate(
            global_model, client_updates, round_num, config
        )
        parent_payload = result.metrics.pop("_rcig_evaluation_payload", None)
        if not isinstance(parent_payload, dict) or parent_payload.get("kind") != (
            "aggregation_role_v1"
        ):
            raise RuntimeError("missing same-cohort aggregation payload")
        if any(
            parent_payload.get(key) is not False
            for key in (
                "contains_clean_data",
                "contains_attack_labels",
                "contains_realised_noise",
            )
        ):
            raise ValueError("oracle data crossed the replay boundary")

        rules = parent_payload["rules"]
        deployed = str(parent_payload["deployed"])
        expected = "uniform" if round_num < 12 else "far_rfa"
        if deployed != expected:
            raise RuntimeError("historical warmup/FAR(RFA) chronology changed")
        base_candidate = torch.as_tensor(
            rules["aggregates"][deployed], dtype=torch.float64, device="cpu"
        )

        public_client_variances = self._public_rcig_variances(client_updates, config)
        public_initial_variance = float(
            public_client_variances.sum() / len(client_updates) ** 2
        )
        if not math.isfinite(public_initial_variance) or public_initial_variance <= 0:
            raise ValueError("public aggregate noise proxy must be positive and finite")
        state_config = _control_config(
            config, public_initial_variance=public_initial_variance
        )
        signature = (int(base_candidate.numel()), state_config)
        if not hasattr(self, "_aggregate_control_state"):
            self._aggregate_control_state = PredictableAggregateState(
                signature[0], state_config
            )
            self._aggregate_control_signature = signature
        elif self._aggregate_control_signature != signature:
            raise ValueError("aggregate control state signature changed")
        snapshot = self._aggregate_control_state.snapshot(round_num)
        shadows = diagnostic_control_candidates(
            base_candidate,
            snapshot,
            config=config,
            num_clients=len(client_updates),
            server_clip_norm=float(config["far_server_clip_norm"]),
        )
        # All controls use the same pre-current snapshot.  Only after they have
        # been computed is the common state updated from the uncorrected A_t.
        self._aggregate_control_state.observe_uncorrected(base_candidate, round_num)

        for key in list(result.metrics):
            if key.startswith("aggregate_control_"):
                result.metrics.pop(key)
        diagnostics = dict(
            aggregate_control_replay_phase="headroom_fit",
            aggregate_control_model_driver="uniform_t1_t12_then_far_rfa",
            aggregate_control_all_alternatives_are_shadow=True,
            aggregate_control_oracle_used_for_deployment=False,
            aggregate_control_calibration_status=(
                "uncalibrated_diagnostic_grid_no_promotion"
            ),
            aggregate_control_predictor_strictly_past=True,
            aggregate_control_state_updated_from_uncorrected_candidate=True,
            aggregate_control_persistent_contamination_possible=True,
            aggregate_control_predictor_ready=bool(snapshot.ready),
            aggregate_control_history_observations=int(snapshot.observations),
            aggregate_control_headroom_fit_eligible=round_num >= 12,
            aggregate_control_initial_variance_proxy=public_initial_variance,
            aggregate_control_initial_variance_is_exact_far_covariance=False,
            aggregate_control_variance_semantics=(
                "ema_predictor_residual_second_moment_not_exact_covariance"
            ),
        )
        for label, value in shadows.items():
            for name, item in value["diagnostics"].items():
                if isinstance(item, (bool, int, float)) and not isinstance(item, str):
                    diagnostics[f"aggregate_control_{label}_{name}"] = item
        result.metrics.update(diagnostics)

        ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
        result.metrics["_rcig_evaluation_payload"] = {
            "kind": REPLAY_KIND,
            "round": round_num + 1,
            "client_ids": ids,
            "server_clip_norm": float(config["far_server_clip_norm"]),
            "base_candidate": base_candidate,
            "predictor": snapshot.predictor,
            "variance": snapshot.variance,
            "predictor_ready": bool(snapshot.ready),
            "controls": shadows,
            "aggregation_payload": parent_payload,
            "contains_clean_data": False,
            "contains_attack_labels": False,
            "contains_realised_noise": False,
            "oracle_used_for_deployment": False,
            "calibration_status": "uncalibrated_diagnostic_grid_no_promotion",
        }
        return result


def _squared(vector: torch.Tensor) -> float:
    norm = stable_l2_norm(vector)
    value = norm * norm
    if not math.isfinite(value):
        raise ValueError("squared norm overflowed")
    return value


def _mahalanobis_squared(vector: torch.Tensor, variance: torch.Tensor) -> float:
    norm = diagonal_mahalanobis_norm(vector, variance)
    value = norm * norm
    if not math.isfinite(value):
        raise ValueError("Mahalanobis squared norm overflowed")
    return value


def _clean_target(payload, clean_by_client, client_updates) -> torch.Tensor:
    ids = [int(metadata["client_id"]) for _, metadata, _ in client_updates]
    if ids != payload["client_ids"] or len(ids) != len(set(ids)):
        raise ValueError("replay client order or identity mismatch")
    if clean_by_client is None or set(ids) != set(clean_by_client):
        raise ValueError("replay requires a complete detached clean oracle")
    honest = torch.tensor(
        [not bool(metadata.get("is_byzantine", False)) for _, metadata, _ in client_updates],
        dtype=torch.bool,
    )
    if not bool(honest.any()):
        raise ValueError("replay has no honest clean target")
    clean, _ = stack_updates([clean_by_client[client_id] for client_id in ids])
    clean = clean.to(dtype=torch.float64, device="cpu")
    radius = _finite_scalar(payload["server_clip_norm"], name="server_clip_norm")
    if radius <= 0.0:
        raise ValueError("server_clip_norm must be positive")
    norms = torch.linalg.vector_norm(clean, dim=1).clamp_min(1e-12)
    clipped = clean * (radius / norms).clamp(max=1.0)[:, None]
    target = clipped[honest].mean(dim=0)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("clean target is non-finite")
    return target


def evaluate_aggregate_control_replay(payload, clean_by_client, client_updates):
    """Offline scalar audit; neither targets nor oracle gamma reach deployment."""

    if not isinstance(payload, dict) or payload.get("kind") != REPLAY_KIND:
        raise ValueError("missing aggregate-control replay payload")
    for key in (
        "contains_clean_data",
        "contains_attack_labels",
        "contains_realised_noise",
        "oracle_used_for_deployment",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"forbidden replay boundary flag: {key}")
    aggregation_payload = payload.get("aggregation_payload")
    base_metrics = evaluate_aggregation_roles(
        aggregation_payload, clean_by_client, client_updates
    )
    target = _clean_target(payload, clean_by_client, client_updates)
    base = torch.as_tensor(
        payload["base_candidate"], dtype=torch.float64, device="cpu"
    )
    predictor = torch.as_tensor(
        payload["predictor"], dtype=torch.float64, device="cpu"
    )
    variance = torch.as_tensor(
        payload["variance"], dtype=torch.float64, device="cpu"
    )
    if base.shape != target.shape or predictor.shape != target.shape:
        raise ValueError("replay candidate/predictor target shape mismatch")
    if variance.shape != target.shape or bool((variance <= 0.0).any()):
        raise ValueError("replay variance is invalid")
    if not all(bool(torch.isfinite(value).all()) for value in (base, predictor, variance)):
        raise ValueError("replay payload contains a non-finite vector")

    output: dict[str, Any] = dict(base_metrics)
    output.update(
        aggregate_control_oracle_boundary="offline_simulator_only",
        aggregate_control_oracle_used_for_deployment=False,
        aggregate_control_same_cohort_verified=True,
        aggregate_control_calibration_status=payload["calibration_status"],
        aggregate_control_predictor_ready=bool(payload["predictor_ready"]),
        aggregate_control_target_definition=(
            "honest_clean_per_example_clipped_gradient_then_server_clipped_mean"
        ),
    )
    if not bool(payload["predictor_ready"]):
        output.update(
            aggregate_control_base_error_sq=_squared(base - target),
            aggregate_control_predictor_error_sq=None,
            aggregate_control_innovation_sq=None,
            aggregate_control_u_dot_predictor_minus_target=None,
            aggregate_control_oracle_gamma_euclidean=None,
            aggregate_control_oracle_error_sq=None,
            aggregate_control_oracle_relative_headroom=None,
            aggregate_control_oracle_gamma_mahalanobis=None,
            aggregate_control_oracle_mahalanobis_error_sq=None,
            aggregate_control_oracle_mahalanobis_relative_headroom=None,
        )
        return output

    innovation = base - predictor
    predictor_error = predictor - target
    base_error = base - target
    u_sq = _squared(innovation)
    cross = float(torch.dot(innovation, predictor_error))
    gamma = 0.0 if u_sq <= 1e-30 else min(1.0, max(0.0, -cross / u_sq))
    oracle = predictor + gamma * innovation
    base_error_sq = _squared(base_error)
    predictor_error_sq = _squared(predictor_error)
    identity_residual = base_error_sq - (predictor_error_sq + 2.0 * cross + u_sq)
    identity_scale = max(
        1.0, base_error_sq, predictor_error_sq, abs(2.0 * cross), u_sq
    )
    if abs(identity_residual) > 1e-10 * identity_scale:
        raise ValueError("Euclidean interpolation sufficient statistics disagree")
    oracle_error_sq = _squared(oracle - target)
    if oracle_error_sq > base_error_sq + 1e-10 * max(1.0, base_error_sq):
        raise ValueError("clamped Euclidean oracle is worse than the base candidate")
    relative_headroom = (
        0.0
        if base_error_sq <= 1e-30
        else (base_error_sq - oracle_error_sq) / base_error_sq
    )

    inv_variance = variance.reciprocal()
    mahalanobis_u_sq = float(torch.dot(innovation * inv_variance, innovation))
    mahalanobis_cross = float(
        torch.dot(innovation * inv_variance, predictor_error)
    )
    if not math.isfinite(mahalanobis_u_sq) or not math.isfinite(mahalanobis_cross):
        raise ValueError("Mahalanobis oracle sufficient statistics are non-finite")
    gamma_mahalanobis = (
        0.0
        if mahalanobis_u_sq <= 1e-30
        else min(1.0, max(0.0, -mahalanobis_cross / mahalanobis_u_sq))
    )
    oracle_mahalanobis = predictor + gamma_mahalanobis * innovation
    base_mahalanobis_error_sq = _mahalanobis_squared(base_error, variance)
    predictor_mahalanobis_error_sq = _mahalanobis_squared(
        predictor_error, variance
    )
    mahalanobis_identity_residual = base_mahalanobis_error_sq - (
        predictor_mahalanobis_error_sq
        + 2.0 * mahalanobis_cross
        + mahalanobis_u_sq
    )
    mahalanobis_identity_scale = max(
        1.0,
        base_mahalanobis_error_sq,
        predictor_mahalanobis_error_sq,
        abs(2.0 * mahalanobis_cross),
        mahalanobis_u_sq,
    )
    if abs(mahalanobis_identity_residual) > 1e-10 * mahalanobis_identity_scale:
        raise ValueError("Mahalanobis interpolation sufficient statistics disagree")
    oracle_mahalanobis_error_sq = _mahalanobis_squared(
        oracle_mahalanobis - target, variance
    )
    if oracle_mahalanobis_error_sq > base_mahalanobis_error_sq + 1e-10 * max(
        1.0, base_mahalanobis_error_sq
    ):
        raise ValueError("clamped Mahalanobis oracle is worse than the base candidate")
    mahalanobis_headroom = (
        0.0
        if base_mahalanobis_error_sq <= 1e-30
        else (
            base_mahalanobis_error_sq - oracle_mahalanobis_error_sq
        )
        / base_mahalanobis_error_sq
    )
    output.update(
        aggregate_control_base_error_sq=base_error_sq,
        aggregate_control_predictor_error_sq=predictor_error_sq,
        aggregate_control_innovation_sq=u_sq,
        aggregate_control_u_dot_predictor_minus_target=cross,
        aggregate_control_euclidean_identity_residual=identity_residual,
        aggregate_control_oracle_gamma_euclidean=gamma,
        aggregate_control_oracle_error_sq=oracle_error_sq,
        aggregate_control_oracle_relative_headroom=relative_headroom,
        aggregate_control_base_mahalanobis_error_sq=base_mahalanobis_error_sq,
        aggregate_control_predictor_mahalanobis_error_sq=(
            predictor_mahalanobis_error_sq
        ),
        aggregate_control_innovation_mahalanobis_sq=mahalanobis_u_sq,
        aggregate_control_u_dot_vinv_predictor_minus_target=mahalanobis_cross,
        aggregate_control_mahalanobis_identity_residual=(
            mahalanobis_identity_residual
        ),
        aggregate_control_oracle_gamma_mahalanobis=gamma_mahalanobis,
        aggregate_control_oracle_mahalanobis_error_sq=oracle_mahalanobis_error_sq,
        aggregate_control_oracle_mahalanobis_relative_headroom=mahalanobis_headroom,
        aggregate_control_target_mahalanobis_from_predictor=diagonal_mahalanobis_norm(
            predictor_error, variance
        ),
        aggregate_control_target_euclidean_from_predictor=stable_l2_norm(
            predictor_error
        ),
    )

    controls = payload.get("controls")
    if not isinstance(controls, dict) or "unchanged" not in controls:
        raise ValueError("replay controls are incomplete")
    for label, entry in controls.items():
        if set(entry) != {"vector", "diagnostics"}:
            raise ValueError("unexpected replay control payload")
        candidate = torch.as_tensor(
            entry["vector"], dtype=torch.float64, device="cpu"
        )
        if candidate.shape != target.shape or not bool(torch.isfinite(candidate).all()):
            raise ValueError("invalid replay shadow candidate")
        diagnostics = entry["diagnostics"]
        prefix = f"aggregate_control_shadow_{label}_"
        output[prefix + "error_sq"] = _squared(candidate - target)
        output[prefix + "mahalanobis_error_sq"] = _mahalanobis_squared(
            candidate - target, variance
        )
        output[prefix + "correction_sq"] = _squared(candidate - base)
        output[prefix + "correction_applied"] = bool(
            diagnostics["correction_applied"]
        )
        radius = diagnostics.get("radius")
        mode = diagnostics["mode"]
        if radius is None:
            output[prefix + "target_covered"] = None
        elif mode == "isotropic_clip":
            output[prefix + "target_covered"] = bool(
                stable_l2_norm(predictor_error) <= float(radius) + 1e-12
            )
        else:
            output[prefix + "target_covered"] = bool(
                diagonal_mahalanobis_norm(predictor_error, variance)
                <= float(radius) + 1e-12
            )
        output[prefix + "gamma"] = diagnostics.get("gamma")
        output[prefix + "lagrange_multiplier"] = diagnostics.get(
            "lagrange_multiplier"
        )
        output[prefix + "radius"] = radius

    for value in output.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite replay scalar diagnostic")
    return output


__all__ = [
    "AggregateControlReplay",
    "REPLAY_KIND",
    "diagnostic_control_candidates",
    "evaluate_aggregate_control_replay",
]
