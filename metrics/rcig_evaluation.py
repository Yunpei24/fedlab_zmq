"""Evaluation-only RCIG oracle utilities.

The deployed LDP-Gradient-FAR server must receive only locally private client
messages and public mechanism metadata.  Simulation-only clean gradients are
therefore detached from the server input and retained by the experiment
harness.  After aggregation, the harness may compare private-transcript RCIG
candidate references with the clean honest center.  These diagnostics are
never inputs to a model update, reference, score, weight, gate, or threshold.
"""

from __future__ import annotations

from typing import Any

import torch

from metrics.robustness import weight_diagnostics
from robustness.tensor_ops import stack_updates


_SIMULATOR_ONLY_FIELDS = (
    "local_dp_noise_free_update_oracle",
    "dp_noise_norm_mean",
)


def _is_attack_oracle_field(name: str) -> bool:
    """Return whether one metadata key reveals simulator attack truth."""

    normalized = str(name).lower()
    return normalized == "is_byzantine" or normalized.startswith("attack_")


def detach_rcig_evaluation_oracles(
    client_updates: list[tuple[dict, dict, object]],
    *,
    enabled: bool,
    strip_attack_oracles: bool = False,
) -> tuple[list[tuple[dict, dict, object]], dict[int, dict[str, torch.Tensor]] | None]:
    """Return server-safe tuples and a separately held clean-gradient oracle.

    Metadata dictionaries are copied, while uploaded tensors and client states
    are shared read-only.  When evaluation is enabled, every client must expose
    exactly one clean-gradient oracle; a partial oracle set fails closed.
    """

    sanitized: list[tuple[dict, dict, object]] = []
    clean_by_client: dict[int, dict[str, torch.Tensor]] = {}
    for update, metadata, state in client_updates:
        server_metadata = dict(metadata)
        clean = server_metadata.pop("local_dp_noise_free_update_oracle", None)
        # Realised noise is never part of the server mechanism either.  It is
        # intentionally discarded here because RCIG v2 has no pre-registered
        # diagnostic that requires it.
        server_metadata.pop("dp_noise_norm_mean", None)
        if strip_attack_oracles:
            for name in tuple(server_metadata):
                if _is_attack_oracle_field(name):
                    server_metadata.pop(name)
        sanitized.append((update, server_metadata, state))
        if enabled:
            if not isinstance(clean, dict) or not clean:
                raise ValueError(
                    "RCIG evaluation requires a complete clean-gradient oracle"
                )
            client_id = int(metadata["client_id"])
            if client_id in clean_by_client:
                raise ValueError("duplicate client id in RCIG evaluation oracle")
            clean_by_client[client_id] = {
                name: value.detach().to(device="cpu").clone()
                for name, value in clean.items()
            }

    if not enabled:
        return sanitized, None
    if len(clean_by_client) != len(client_updates):
        raise ValueError("RCIG evaluation oracle is incomplete")
    return sanitized, clean_by_client


def external_byzantine_weight_metrics(
    payload: dict[str, Any] | None,
    evaluated_client_updates: list[tuple[dict, dict, object]],
) -> dict[str, float | int | bool | str]:
    """Join server weights with attack truth strictly outside aggregation.

    The server payload contains only authenticated client identifiers and the
    weights it actually used.  Simulator attack labels remain in the harness
    copy of ``evaluated_client_updates`` and are joined here *after* the model
    update has been chosen.  This prevents Byzantine identities from becoming
    an accidental input to any reference, score, gate, or weight.
    """

    if not isinstance(payload, dict):
        raise ValueError("external attack diagnostics require a weight payload")
    allowed = {"client_ids", "weights", "contains_attack_labels"}
    if set(payload) != allowed:
        raise ValueError("external weight payload contains unexpected fields")
    if bool(payload.get("contains_attack_labels", True)):
        raise ValueError("external weight payload must not contain attack labels")

    try:
        client_ids = [int(value) for value in payload["client_ids"]]
        weights = torch.as_tensor(payload["weights"], dtype=torch.float64, device="cpu")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("malformed external weight payload") from exc
    if weights.ndim != 1 or len(client_ids) != int(weights.numel()) or not client_ids:
        raise ValueError("external weight payload has inconsistent dimensions")
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("external weight payload contains duplicate client ids")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise ValueError("external weights must be finite and non-negative")
    if not torch.isclose(
        weights.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("external weights must sum to one")

    attack_truth: dict[int, bool] = {}
    for _, metadata, _ in evaluated_client_updates:
        client_id = int(metadata["client_id"])
        if client_id in attack_truth:
            raise ValueError("duplicate client id in external attack truth")
        attack_truth[client_id] = bool(metadata.get("is_byzantine", False))
    if set(client_ids) != set(attack_truth):
        raise ValueError("external weight payload/client cohort mismatch")

    malicious_mask = torch.tensor(
        [attack_truth[client_id] for client_id in client_ids], dtype=torch.bool
    )
    diagnostics = weight_diagnostics(weights, malicious_mask)
    return {
        "byzantine_weight_mass_oracle": diagnostics["byzantine_weight_mass_oracle"],
        "far_num_byzantine_oracle": int(malicious_mask.sum().item()),
        "far_attack_labels_visible_to_server_aggregate": False,
        "far_external_attack_diagnostics": True,
        "far_external_attack_diagnostics_boundary": "posthoc_simulator_only",
    }


def rcig_reference_oracle_metrics(
    payload: dict[str, Any] | None,
    clean_by_client: dict[int, dict[str, torch.Tensor]] | None,
    evaluated_client_updates: list[tuple[dict, dict, object]],
) -> dict[str, float | bool | str]:
    """Compute post-hoc squared-L2 reference errors outside the mechanism."""

    if payload is None:
        return {}
    if clean_by_client is None:
        raise ValueError("an RCIG evaluation payload requires detached clean oracles")
    if any(
        bool(payload.get(key, True))
        for key in (
            "contains_clean_data",
            "contains_realised_noise",
            "contains_attack_labels",
        )
    ):
        raise ValueError("RCIG server payload contains forbidden oracle information")

    honest_ids = [
        int(metadata["client_id"])
        for _, metadata, _ in evaluated_client_updates
        if not bool(metadata.get("is_byzantine", False))
    ]
    if not honest_ids:
        raise ValueError("RCIG oracle evaluation requires at least one honest client")
    try:
        clean_updates = [clean_by_client[client_id] for client_id in honest_ids]
    except KeyError as exc:
        raise ValueError("missing clean oracle for an evaluated client") from exc

    clean_vectors, _ = stack_updates(clean_updates)
    radius = float(payload["server_clip_norm"])
    if radius <= 0.0:
        raise ValueError("RCIG evaluation server clip norm must be positive")
    clean_norms = torch.linalg.vector_norm(clean_vectors, dim=1)
    factors = (radius / clean_norms.clamp_min(1e-12)).clamp(max=1.0)
    clean_center = (clean_vectors * factors[:, None]).mean(dim=0)

    prefixes = {
        "identity_new": "rcig_identity_new",
        "identity_old": "rcig_identity_old",
        "midpoint": "rcig_midpoint",
        "rcig_full": "rcig_full",
        "rcig_isotropic": "rcig_isotropic",
        "rcig_euclidean": "rcig_euclidean",
    }
    candidates = payload.get("candidate_references")
    if not isinstance(candidates, dict) or set(candidates) != set(prefixes):
        raise ValueError("RCIG evaluation payload has incomplete candidates")

    metrics: dict[str, float | bool | str] = {
        "rcig_oracle_evaluation_boundary": "offline_simulator_only",
        "rcig_oracle_was_visible_to_server_aggregate": False,
        "rcig_oracle_metric_is_squared_l2": True,
    }
    for name, prefix in prefixes.items():
        candidate = torch.as_tensor(candidates[name], dtype=torch.float64, device="cpu")
        if candidate.shape != clean_center.shape:
            raise ValueError("RCIG candidate/clean-center dimension mismatch")
        squared_error = float((candidate - clean_center).square().sum().item())
        metrics[f"{prefix}_squared_l2_error_to_clean_honest_center_oracle"] = (
            squared_error
        )
        # Transitional compatibility for older result readers.  The name is
        # explicit about L2 and is never labelled MSE.
        metrics[f"{prefix}_l2_error_to_clean_honest_center_oracle"] = squared_error**0.5

    deployed = torch.as_tensor(
        payload["deployed_reference"], dtype=torch.float64, device="cpu"
    )
    if deployed.shape != clean_center.shape:
        raise ValueError("deployed RCIG reference/clean-center dimension mismatch")
    deployed_squared_error = float((deployed - clean_center).square().sum().item())
    metrics["rcig_reference_squared_l2_error_to_clean_honest_center_oracle"] = (
        deployed_squared_error
    )
    metrics["rcig_reference_l2_error_to_clean_honest_center_oracle"] = (
        deployed_squared_error**0.5
    )
    return metrics


__all__ = [
    "detach_rcig_evaluation_oracles",
    "external_byzantine_weight_metrics",
    "rcig_reference_oracle_metrics",
]
