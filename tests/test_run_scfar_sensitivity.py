"""Fast integration tests for the synthetic S0.3 SC-FAR audit."""

from __future__ import annotations

import json
import math

import torch

from scripts.run_scfar_sensitivity import (
    audit_replace_one,
    build_parser,
    build_public_anchor,
    write_outputs,
)


def _arguments(*extra: str):
    return build_parser().parse_args(
        [
            "--n",
            "5",
            "--dimension",
            "4",
            "--trials",
            "3",
            "--methods",
            "f_cc,huber,mean",
            "--clip",
            "1.0",
            "--distance-clip",
            "2.0",
            "--tau",
            "0.4",
            "--huber-gamma",
            "0.8",
            "--huber-num-steps",
            "3",
            "--alpha",
            "0.2",
            "--kappa-w",
            "2.0",
            *extra,
        ]
    )


def test_public_random_anchor_is_reproducible_and_data_independent():
    first = build_public_anchor(dimension=7, mode="fixed_random", norm=0.3, seed=91)
    second = build_public_anchor(dimension=7, mode="fixed_random", norm=0.3, seed=91)
    different_seed = build_public_anchor(
        dimension=7, mode="fixed_random", norm=0.3, seed=92
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)
    assert math.isclose(float(torch.linalg.vector_norm(first)), 0.3, rel_tol=1e-12)


def test_s0_3_audit_uses_certified_primary_and_huber_ablation():
    records, summary = audit_replace_one(
        _arguments("--anchor-mode", "fixed_random", "--anchor-norm", "0.2")
    )
    assert len(records) == 3 * 3
    assert summary["public_anchor"]["independent_of_current_cohort"] is True
    assert summary["method_roles"] == {
        "centered_clipping": "primary_certified_reference",
        "regularized_huber": "certified_finite_step_ablation",
        "mean": "empirical_comparator",
    }

    centered = [row for row in records if row["method"] == "centered_clipping"]
    huber = [row for row in records if row["method"] == "regularized_huber"]
    mean = [row for row in records if row["method"] == "mean"]
    assert {row["reference_certificate"] for row in centered} == {
        "centered_clipping_global"
    }
    assert {row["reference_certificate"] for row in huber} == {
        "regularized_huber_fixed_steps"
    }
    assert all(row["reference_bound"] is not None for row in centered + huber)
    assert {row["reference_certificate"] for row in mean} == {
        "arithmetic_mean_global_not_byzantine_robust"
    }
    assert all(math.isclose(row["reference_bound"], 2.0 / 5.0) for row in mean)

    for row in centered + huber + mean:
        for check in (
            "reference_certificate_holds",
            "unchanged_score_linf_certificate_holds",
            "replaced_score_certificate_holds",
            "score_l1_certificate_holds",
            "weight_l1_certificate_holds",
            "aggregate_certificate_holds",
            "aggregate_refined_certificate_holds",
        ):
            assert row[check] is True


def test_s0_3_writes_csv_and_json(tmp_path):
    records, summary = audit_replace_one(_arguments())
    write_outputs(records, summary, tmp_path)
    csv_path = tmp_path / "replace_one_trials.csv"
    json_path = tmp_path / "summary.json"
    assert csv_path.exists()
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["audit"] == "S0.3_replace_one_F_scores_weights_aggregate"
    assert loaded["schema_version"] == 2
    assert "n=5/centered_clipping" in loaded["groups"]
