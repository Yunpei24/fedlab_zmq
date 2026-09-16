"""Unit tests for the strictly causal temporal RCIG state machine."""

from __future__ import annotations

import pytest
import torch

from algorithms.rcig_temporal_reference import (
    RCIGTemporalConfig,
    TemporalRCIGReferenceState,
)


def _config(**overrides) -> RCIGTemporalConfig:
    values = {
        "window_length": 1,
        "gate_window_length": 2,
        "subspace_dimension": 32,
        "subspace_seed": 17,
        "server_clip_norm": 1.0,
        "influence_cap": 1.0,
        "minimum_accepted_mass": 1.0,
        "process_variance": 1.0e-5,
        "ridge": 1.0e-6,
        "innovation_threshold": 3.0,
    }
    values.update(overrides)
    return RCIGTemporalConfig(**values)


def _snapshot(
    state: TemporalRCIGReferenceState,
    round_num: int,
    rows: list[list[float]] | torch.Tensor,
    *,
    client_ids: list[int] | None = None,
    factors: list[float] | torch.Tensor | None = None,
    variances: list[float] | torch.Tensor | None = None,
):
    vectors = torch.as_tensor(rows, dtype=torch.float64)
    n = int(vectors.shape[0])
    return state.make_snapshot(
        round_num=round_num,
        client_ids=list(range(n)) if client_ids is None else client_ids,
        clipped_vectors=vectors,
        server_clip_factors=(
            torch.ones(n, dtype=torch.float64) if factors is None else factors
        ),
        public_noise_variances=(
            torch.full((n,), 0.01, dtype=torch.float64)
            if variances is None
            else variances
        ),
    )


def _constant_rows(values: list[float], dimension: int = 32) -> torch.Tensor:
    rows = torch.zeros(len(values), dimension, dtype=torch.float64)
    rows[:, 0] = torch.tensor(values, dtype=torch.float64)
    return rows


def test_snapshot_is_sorted_by_client_id_without_mutating_state() -> None:
    state = TemporalRCIGReferenceState(_config())
    rows = _constant_rows([0.1, 0.2, 0.3])
    snapshot = _snapshot(
        state,
        0,
        rows,
        client_ids=[30, 10, 20],
        factors=[1.0, 1.0, 1.0],
        variances=[0.3, 0.1, 0.2],
    )
    assert state.history_length == 0
    assert snapshot.client_ids == (10, 20, 30)
    assert snapshot.vectors[:, 0].tolist() == pytest.approx([0.2, 0.3, 0.1])
    assert snapshot.public_noise_variances.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_cold_start_is_deterministic_zero_and_requests_uniform_weights() -> None:
    state = TemporalRCIGReferenceState(_config())
    result = state.reference_for_round(
        round_num=0, dimension=64, device="cpu", dtype=torch.float32
    )
    assert result.ready is False
    assert torch.equal(result.reference, torch.zeros(64))
    assert result.diagnostics["rcig_current_round_read"] is False
    assert (
        result.diagnostics["rcig_cold_start_policy"]
        == "zero_reference_uniform_weights_required"
    )


def test_reference_is_strictly_causal_and_windows_are_disjoint() -> None:
    state = TemporalRCIGReferenceState(_config())
    for round_num, value in enumerate((0.00, 0.01, 0.02, 0.03)):
        state.commit_snapshot(_snapshot(state, round_num, _constant_rows([value] * 4)))

    first = state.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    # Constructing two incompatible current-round snapshots cannot affect F_4:
    # make_snapshot is pure and neither snapshot is committed yet.
    _snapshot(state, 4, _constant_rows([0.9] * 4))
    _snapshot(state, 4, _constant_rows([-0.9] * 4))
    second = state.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    assert first.ready is True
    assert torch.equal(first.reference, second.reference)
    assert first.diagnostics["rcig_gate_source_round_min"] == 0
    assert first.diagnostics["rcig_gate_source_round_max"] == 1
    assert first.diagnostics["rcig_older_round_min"] == 2
    assert first.diagnostics["rcig_newer_round_min"] == 3
    assert first.diagnostics["rcig_windows_disjoint"] is True
    assert first.diagnostics["rcig_newer_round_max"] < 4


def test_all_paired_candidate_references_are_exposed_but_not_deployed_controls() -> (
    None
):
    state = TemporalRCIGReferenceState(_config())
    for round_num, value in enumerate((0.00, 0.00, 0.10, 0.12)):
        state.commit_snapshot(_snapshot(state, round_num, _constant_rows([value] * 4)))
    result = state.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    assert set(result.candidate_references or {}) == {
        "identity_new",
        "identity_old",
        "midpoint",
        "rcig_full",
        "rcig_isotropic",
        "rcig_euclidean",
    }
    assert result.diagnostics["rcig_selected_candidate_drives_decision"] is True
    assert result.diagnostics["rcig_counterfactual_candidates_drive_decision"] is False
    assert result.diagnostics["rcig_max_covariance_anisotropy_ratio"] >= 1.0
    assert torch.equal(
        result.reference, result.candidate_references["rcig_full"]  # type: ignore[index]
    )


def test_predictable_gate_removes_a_persistent_prehistory_outlier() -> None:
    state = TemporalRCIGReferenceState(
        _config(gate_window_length=3, innovation_threshold=1.0e6)
    )
    for round_num in range(3):
        state.commit_snapshot(
            _snapshot(state, round_num, _constant_rows([0.0, 0.0, 0.0, 0.9]))
        )
    state.commit_snapshot(_snapshot(state, 3, _constant_rows([0.0, 0.0, 0.0, 0.8])))
    state.commit_snapshot(_snapshot(state, 4, _constant_rows([0.0, 0.0, 0.0, 0.8])))
    result = state.reference_for_round(
        round_num=5, dimension=32, device="cpu", dtype=torch.float64
    )
    assert result.diagnostics["rcig_gate_zero_count"] == 1
    assert result.diagnostics["rcig_gate_mass"] == pytest.approx(3.0)
    assert result.older_view is not None
    assert result.newer_view is not None
    assert torch.linalg.vector_norm(result.older_view) == pytest.approx(0.0)
    assert torch.linalg.vector_norm(result.newer_view) == pytest.approx(0.0)


def test_post_clip_covariance_matches_restricted_radial_jacobian() -> None:
    state = TemporalRCIGReferenceState(
        _config(gate_window_length=1, process_variance=0.0)
    )
    zeros = torch.zeros(2, 32, dtype=torch.float64)
    state.commit_snapshot(_snapshot(state, 0, zeros, variances=[0.0, 0.0]))

    old = zeros.clone()
    old[0, 0] = 1.0
    state.commit_snapshot(
        _snapshot(
            state,
            1,
            old,
            factors=[0.5, 1.0],
            variances=[0.04, 0.04],
        )
    )
    state.commit_snapshot(_snapshot(state, 2, zeros, variances=[0.04, 0.04]))
    result = state.reference_for_round(
        round_num=3, dimension=32, device="cpu", dtype=torch.float64
    )
    covariance = result.covariance_older
    assert covariance is not None
    # Each client coefficient is 1/2.  For the clipped first client,
    # f^2(I-u u^T) has diagonal 0 on e1 and 1/4 elsewhere.  The second
    # client's multiplier is I.
    assert float(covariance[0, 0]) == pytest.approx(0.01)
    assert torch.diagonal(covariance)[1:].tolist() == pytest.approx([0.0125] * 31)
    assert result.diagnostics["rcig_older_view"]["covariance_psd"] is True
    assert (
        result.diagnostics["rcig_older_view"][
            "dense_full_model_covariance_materialized"
        ]
        is False
    )


def test_freeze_hysteresis_uses_only_innovation_and_recovers_after_patience() -> None:
    state = TemporalRCIGReferenceState(
        _config(
            gate_window_length=1,
            innovation_threshold=0.1,
            recovery_threshold=0.05,
            recovery_patience=2,
            process_variance=0.0,
            ridge=1.0e-6,
            persistent_policy="freeze_hysteresis",
        )
    )
    state.commit_snapshot(_snapshot(state, 0, _constant_rows([0.0, 0.0])))
    state.commit_snapshot(_snapshot(state, 1, _constant_rows([0.0, 0.0])))
    state.commit_snapshot(_snapshot(state, 2, _constant_rows([0.5, 0.5])))
    activated = state.reference_for_round(
        round_num=3, dimension=32, device="cpu", dtype=torch.float64
    )
    assert activated.diagnostics["rcig_persistent_action"] == "freeze_on_activation"
    assert torch.equal(
        activated.reference,
        activated.candidate_references["identity_old"],  # type: ignore[index]
    )
    state.commit_snapshot(
        _snapshot(state, 3, _constant_rows([0.5, 0.5])),
        reference_result=activated,
    )

    persistent = state.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    assert persistent.diagnostics["rcig_persistent_action"] == "remain_frozen"
    assert torch.equal(persistent.reference, activated.reference)
    state.commit_snapshot(
        _snapshot(state, 4, _constant_rows([0.0, 0.0])),
        reference_result=persistent,
    )

    pending = state.reference_for_round(
        round_num=5, dimension=32, device="cpu", dtype=torch.float64
    )
    assert pending.diagnostics["rcig_persistent_action"] == "recovery_pending"
    assert torch.equal(pending.reference, activated.reference)
    state.commit_snapshot(
        _snapshot(state, 5, _constant_rows([0.0, 0.0])),
        reference_result=pending,
    )

    released = state.reference_for_round(
        round_num=6, dimension=32, device="cpu", dtype=torch.float64
    )
    assert released.diagnostics["rcig_persistent_action"] == "release_after_recovery"
    assert float(released.reference[0]) == pytest.approx(0.0)


def test_state_rejects_unstable_roster_round_gaps_and_invalid_provenance() -> None:
    state = TemporalRCIGReferenceState(_config())
    state.commit_snapshot(_snapshot(state, 0, _constant_rows([0.0, 0.0])))
    with pytest.raises(ValueError, match="stable full-participation"):
        state.commit_snapshot(
            _snapshot(
                state,
                1,
                _constant_rows([0.0, 0.0]),
                client_ids=[0, 2],
            )
        )
    with pytest.raises(ValueError, match="consecutive"):
        state.commit_snapshot(_snapshot(state, 2, _constant_rows([0.0, 0.0])))
    with pytest.raises(ValueError, match="authenticated public"):
        _config(public_variance_provenance="client_declared")


def test_snapshot_rejects_inconsistent_radial_clip_factor() -> None:
    state = TemporalRCIGReferenceState(_config())
    with pytest.raises(ValueError, match="clip sphere"):
        _snapshot(
            state,
            0,
            _constant_rows([0.2, 0.0]),
            factors=[0.5, 1.0],
        )


@pytest.mark.parametrize("mode", ["full", "isotropic", "euclidean"])
def test_all_deployment_modes_are_finite_bounded_and_oracle_free(mode: str) -> None:
    state = TemporalRCIGReferenceState(_config(covariance_mode=mode))
    for round_num, value in enumerate((0.0, 0.0, 0.01, 0.02)):
        state.commit_snapshot(_snapshot(state, round_num, _constant_rows([value] * 3)))
    result = state.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    assert bool(torch.isfinite(result.reference).all())
    assert float(torch.linalg.vector_norm(result.reference)) <= 1.0
    diagnostics_text = repr(result.diagnostics).lower()
    assert "byzantine_mask" not in diagnostics_text
    assert "noise_free" not in diagnostics_text
    assert result.diagnostics["rcig_diagnostics_use_realised_noise"] is False
    assert result.diagnostics["rcig_diagnostics_use_attack_labels"] is False
    assert result.diagnostics["rcig_diagnostics_use_clean_gradients"] is False
