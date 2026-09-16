"""Tests for the evaluation-only RCIG clean-gradient boundary."""

from __future__ import annotations

import pytest
import torch

from algorithms.base import ClientState
from metrics.rcig_evaluation import (
    detach_rcig_evaluation_oracles,
    external_byzantine_weight_metrics,
    rcig_reference_oracle_metrics,
)


def _tuples():
    rows = []
    for client_id, value in enumerate((1.0, 2.0, 8.0)):
        rows.append(
            (
                {"weight": torch.tensor([value], dtype=torch.float32)},
                {
                    "client_id": client_id,
                    "is_byzantine": client_id == 2,
                    "dp_noise_norm_mean": 123.0,
                    "local_dp_noise_free_update_oracle": {
                        "weight": torch.tensor([value], dtype=torch.float32)
                    },
                },
                ClientState(client_id=client_id),
            )
        )
    return rows


def _payload(candidate: float = 1.5):
    names = (
        "identity_new",
        "identity_old",
        "midpoint",
        "rcig_full",
        "rcig_isotropic",
        "rcig_euclidean",
    )
    return {
        "candidate_references": {
            name: torch.tensor([candidate], dtype=torch.float64) for name in names
        },
        "deployed_reference": torch.tensor([candidate], dtype=torch.float64),
        "server_clip_norm": 10.0,
        "deployment_round": 4,
        "contains_clean_data": False,
        "contains_realised_noise": False,
        "contains_attack_labels": False,
    }


def test_oracles_are_removed_from_server_input_and_retained_outside():
    original = _tuples()
    sanitized, clean = detach_rcig_evaluation_oracles(original, enabled=True)
    assert clean is not None and set(clean) == {0, 1, 2}
    for _, metadata, _ in sanitized:
        assert "local_dp_noise_free_update_oracle" not in metadata
        assert "dp_noise_norm_mean" not in metadata
    # The evaluation view, not the server view, keeps attack labels.
    assert original[2][1]["is_byzantine"] is True


def test_attack_oracles_are_stripped_and_weights_joined_only_posthoc():
    original = _tuples()
    original[2][1].update(
        {
            "attack_name": "bf",
            "attack_window_active": True,
            "attack_schedule_phase": "attack",
        }
    )
    sanitized, _ = detach_rcig_evaluation_oracles(
        original, enabled=False, strip_attack_oracles=True
    )
    for _, metadata, _ in sanitized:
        assert "is_byzantine" not in metadata
        assert all(not key.startswith("attack_") for key in metadata)

    metrics = external_byzantine_weight_metrics(
        {
            "client_ids": [0, 1, 2],
            "weights": [0.2, 0.3, 0.5],
            "contains_attack_labels": False,
        },
        original,
    )
    assert metrics["byzantine_weight_mass_oracle"] == pytest.approx(0.5)
    assert metrics["far_num_byzantine_oracle"] == 1
    assert metrics["far_attack_labels_visible_to_server_aggregate"] is False


def test_external_weight_payload_fails_closed_on_label_or_cohort_leak():
    original = _tuples()
    with pytest.raises(ValueError, match="unexpected fields"):
        external_byzantine_weight_metrics(
            {
                "client_ids": [0, 1, 2],
                "weights": [0.2, 0.3, 0.5],
                "contains_attack_labels": False,
                "is_byzantine": [False, False, True],
            },
            original,
        )
    with pytest.raises(ValueError, match="cohort mismatch"):
        external_byzantine_weight_metrics(
            {
                "client_ids": [0, 1, 9],
                "weights": [0.2, 0.3, 0.5],
                "contains_attack_labels": False,
            },
            original,
        )


def test_posthoc_metric_is_squared_l2_and_uses_only_honest_oracles():
    evaluated = _tuples()
    _, clean = detach_rcig_evaluation_oracles(evaluated, enabled=True)
    metrics = rcig_reference_oracle_metrics(_payload(), clean, evaluated)
    # Honest clean center is (1 + 2) / 2 = 1.5.
    assert metrics["rcig_full_squared_l2_error_to_clean_honest_center_oracle"] == 0.0
    assert metrics["rcig_oracle_metric_is_squared_l2"] is True
    assert metrics["rcig_oracle_was_visible_to_server_aggregate"] is False
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


def test_oracle_changes_only_posthoc_metrics_not_server_payload():
    evaluated = _tuples()
    sanitized, clean = detach_rcig_evaluation_oracles(evaluated, enabled=True)
    assert clean is not None
    first_payload = _payload(candidate=1.5)
    first = rcig_reference_oracle_metrics(first_payload, clean, evaluated)
    clean[0]["weight"].fill_(-10.0)
    second = rcig_reference_oracle_metrics(first_payload, clean, evaluated)
    assert first_payload["deployed_reference"].item() == 1.5
    assert sanitized[0][0]["weight"].item() == 1.0
    assert (
        first["rcig_full_squared_l2_error_to_clean_honest_center_oracle"]
        != second["rcig_full_squared_l2_error_to_clean_honest_center_oracle"]
    )


def test_enabled_boundary_fails_closed_on_missing_oracle():
    rows = _tuples()
    rows[1][1].pop("local_dp_noise_free_update_oracle")
    with pytest.raises(ValueError, match="complete clean-gradient oracle"):
        detach_rcig_evaluation_oracles(rows, enabled=True)
