"""Tests for the strictly paired DT-LDP-FAR mechanism stress."""

from __future__ import annotations

import math

import torch

from scripts.run_dt_ldp_far_delay_mechanistic_stress import (
    alpha_max,
    evaluate_cell,
    make_paired_inputs,
)


def _paired(seed: int = 28):
    return make_paired_inputs(
        n=25,
        dimension=4,
        draws=64,
        seed=seed,
        server_clip=0.42,
        noise_l2_scale=0.16,
        drift_l2_scale=0.0,
        heterogeneity_l2_scale=0.0,
        heteroscedastic_noise=True,
    )


def test_pairing_is_exactly_reproducible():
    first = _paired()
    second = _paired()
    assert first.randomness_pair_id == second.randomness_pair_id
    assert torch.equal(first.clean, second.clean)
    assert torch.equal(first.current_noisy, second.current_noisy)
    assert torch.equal(first.past_noisy, second.past_noisy)
    assert torch.equal(
        first.effective_fresh_perturbation,
        second.effective_fresh_perturbation,
    )


def test_different_seed_changes_the_paired_randomness():
    first = _paired(28)
    second = _paired(36)
    assert first.randomness_pair_id != second.randomness_pair_id
    assert not torch.equal(first.current_noisy, second.current_noisy)


def test_certified_cell_respects_analytic_weight_cap():
    kappa_w = 4.0
    alpha = alpha_max(25, kappa_w)
    row = evaluate_cell(
        paired=_paired(),
        n=25,
        dimension=4,
        draws=64,
        seed=28,
        server_clip=0.42,
        d_score=0.21,
        rho=0.21,
        noise_l2_scale=0.16,
        drift_l2_scale=0.0,
        heterogeneity_l2_scale=0.0,
        heteroscedastic_noise=True,
        alpha_profile="kappa4_boundary",
        alpha=alpha,
        certificate_kappa_w=kappa_w,
    )
    assert row["influence_certificate_claimed"] is True
    assert math.isclose(float(row["analytic_weight_cap"]), 4.0 / 25.0)
    assert float(row["current_max_weight_median"]) <= 4.0 / 25.0 + 1e-12
    assert 0.0 <= float(row["score_span_median"]) <= 1.0
    assert math.isclose(
        float(row["logit_span_median"]),
        alpha * float(row["score_span_median"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_uncertified_alpha_is_labelled_and_can_make_logit_span_exceed_one():
    row = evaluate_cell(
        paired=_paired(),
        n=25,
        dimension=4,
        draws=64,
        seed=28,
        server_clip=0.42,
        d_score=0.105,
        rho=0.21,
        noise_l2_scale=0.16,
        drift_l2_scale=0.0,
        heterogeneity_l2_scale=0.0,
        heteroscedastic_noise=True,
        alpha_profile="diagnostic_alpha8",
        alpha=8.0,
        certificate_kappa_w=None,
    )
    assert row["influence_certificate_claimed"] is False
    assert row["analytic_weight_cap"] is None
    assert float(row["logit_span_median"]) > 1.0

