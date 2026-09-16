from __future__ import annotations

import pytest
import torch

from scripts import audit_gaussian_aware_reference_g0g_k5_tp as audit


def test_independent_lambda_selection_recomputes_equal_seed_mean() -> None:
    rows = [
        {
            "ridge_lambda": 0.1,
            "equal_seed_mean_projected_aggregate_mse": 2.0,
            "calibration_seed_mse": [1.0, 3.0],
        },
        {
            "ridge_lambda": 1.0,
            "equal_seed_mean_projected_aggregate_mse": 1.5,
            "calibration_seed_mse": [1.5, 1.5],
        },
    ]
    assert audit._select_lambda(rows, 1e-12) == 1.0


def test_independent_lambda_selection_checks_recorded_mean() -> None:
    rows = [
        {
            "ridge_lambda": 0.1,
            "equal_seed_mean_projected_aggregate_mse": 9.0,
            "calibration_seed_mse": [1.0, 3.0],
        }
    ]
    with pytest.raises(RuntimeError, match="equal-seed mean"):
        audit._select_lambda(rows, 1e-12)


def test_sufficient_statistics_solver_is_independent() -> None:
    diagnostics = {
        "normalized_gram": [[2.0, 0.0], [0.0, 3.0]],
        "normalized_rhs": [4.0, 9.0],
        "ridge_lambda": 1.0,
    }
    result = audit._solve_sufficient_statistics(diagnostics, torch.device("cpu"))
    assert torch.allclose(result, torch.tensor([4.0 / 3.0, 9.0 / 4.0]))


def test_expected_evaluation_matrix_has_576_histories() -> None:
    import yaml

    config = yaml.safe_load(audit.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert len(audit._expected_history_ids(config, "evaluation")) == 576


def test_student_interval_uses_outer_units() -> None:
    result = audit._ci([1.0, 2.0, 3.0], 2.0)
    assert result["n"] == 3
    assert result["mean"] == 2.0
    assert float(result["low"]) < 2.0 < float(result["high"])
