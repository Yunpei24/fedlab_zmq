"""Policy isolation, chronology, real MPS RNG pairing and new matrix tests."""

import math
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.rcig_n10_ablation import (
    AblationTemporalState,
    isolated_simulator_rng,
    stream_seed,
)
from algorithms.rcig_temporal_reference import (
    RCIGTemporalConfig,
    TemporalRCIGReferenceState,
)
from privacy.local_dpsgd import private_gradient_release_fixed_without_replacement
from scripts import run_rcig_n10_policy_ablation as runner
from scripts.run_rcig_n10_experiment import evaluation_rng_isolation


def make_state(arm, threshold=1):
    cfg = RCIGTemporalConfig(
        window_length=1,
        gate_window_length=2,
        subspace_dimension=32,
        server_clip_norm=1,
        influence_cap=1,
        minimum_accepted_mass=0.6,
        process_variance=1e-5,
        ridge=1e-6,
        innovation_threshold=threshold,
        persistent_policy="freeze_hysteresis" if arm == "freeze" else "rolling",
        recovery_threshold=threshold * 0.8 if arm == "freeze" else None,
    )
    return AblationTemporalState(cfg, arm)


def snapshot(state, t, value):
    x = torch.zeros(10, 32, dtype=torch.float64)
    x[:, 0] = value
    return state.make_snapshot(
        round_num=t,
        client_ids=list(range(10)),
        clipped_vectors=x,
        server_clip_factors=torch.ones(10),
        public_noise_variances=torch.full((10,), 0.001),
    )


@pytest.mark.parametrize(
    "arm,selected",
    [("recent", "identity_new"), ("midpoint", "midpoint"), ("rolling", "rcig_full")],
)
def test_actual_deployed_policy_not_oracle_choice(arm, selected):
    s = make_state(arm)
    for t, v in enumerate([0.0, 0.0, 0.1, 0.6]):
        s.commit_snapshot(snapshot(s, t, v))
    result = s.reference_for_round(
        round_num=4, dimension=32, device="cpu", dtype=torch.float64
    )
    assert torch.equal(result.reference, result.candidate_references[selected])
    assert result.diagnostics["rcig_deployed_candidate"] == selected
    assert result.diagnostics["rcig_n10_innovation_drives_deployment"] == (
        arm == "rolling"
    )
    assert result.diagnostics["rcig_newer_round_max"] == 3
    assert result.diagnostics["rcig_n10_older_cov_trace"] > 0
    s.commit_snapshot(snapshot(s, 4, 0.8), reference_result=result)
    assert not s._persistent_frozen


def test_controls_ignore_threshold_for_deployment():
    for arm in ("recent", "midpoint"):
        outputs = []
        for threshold in (0.01, 100):
            s = make_state(arm, threshold)
            for t, v in enumerate([0.0, 0.0, 0.1, 0.6]):
                s.commit_snapshot(snapshot(s, t, v))
            outputs.append(
                s.reference_for_round(
                    round_num=4, dimension=32, device="cpu", dtype=torch.float64
                ).reference
            )
        assert torch.equal(*outputs)


def test_freeze_exactly_matches_historical_policy():
    state = make_state("freeze")
    original = TemporalRCIGReferenceState(state.config)
    for t, v in enumerate([0.0, 0.0, 0.1, 0.6, 0.6, 0.6, 0.1, 0.1, 0.1]):
        a = state.reference_for_round(
            round_num=t, dimension=32, device="cpu", dtype=torch.float64
        )
        b = original.reference_for_round(
            round_num=t, dimension=32, device="cpu", dtype=torch.float64
        )
        assert torch.equal(a.reference, b.reference)
        for s, r in ((state, a), (original, b)):
            s.commit_snapshot(
                snapshot(s, t, v), reference_result=r if r.ready else None
            )
        if a.ready:
            assert state._persistent_frozen == original._persistent_frozen
            assert state._recovery_count == original._recovery_count


def test_new_matrix_counts_and_no_old_outputs():
    m = runner.matrix()
    tasks = runner.tasks(m)
    assert sum(t["phase"] == "calibration" for t in tasks) == 12
    assert sum(t["phase"] == "comparison" for t in tasks) == 120
    assert len({str(runner.directory(t)) for t in tasks}) == 132
    assert runner.OUTPUT.name == "rcig_n10_policy_ablation_v1"
    assert set(m["comparison_seeds"]).isdisjoint(m["calibration_seeds"])


def test_privacy_is_recalibrated_at_n10():
    p = runner.privacy(runner.matrix())
    assert p["q"] == 0.02
    assert abs(p["per_scale"]["1"]["epsilon"] - 4) < 1e-4
    assert p["per_scale"]["2"]["epsilon"] < 4
    assert p["per_scale"]["2"]["std"] == 2 * p["per_scale"]["1"]["std"]


def test_seed_domains_distinct():
    assert (
        len(
            {
                stream_seed(s, c, t, p)
                for s in (1, 2)
                for c in range(10)
                for t in range(40)
                for p in ("batch", "noise")
            }
        )
        == 1600
    )


def test_real_private_gradient_mps_pairing_despite_extra_evaluation():
    assert (
        torch.backends.mps.is_available()
    ), "Run this regression on local MPS, never CPU fallback"
    model = torch.nn.Linear(4, 2).to("mps")
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
    dl = DataLoader(TensorDataset(x, torch.arange(8) % 2), batch_size=2)
    traces = []
    uploads = []
    before_cpu = torch.get_rng_state().clone()
    before_mps = torch.mps.get_rng_state().clone()
    for extra_loaders in (2, 10):
        for _ in range(extra_loaders):
            list(dl)
        torch.randn(50, device="mps")
        ambient_cpu = torch.get_rng_state().clone()
        ambient_mps = torch.mps.get_rng_state().clone()
        audit = {}
        with isolated_simulator_rng(42, 3, 5, device="mps", audit=audit):
            y, _ = private_gradient_release_fixed_without_replacement(
                model,
                dl,
                device="mps",
                batch_size=2,
                clip_norm=1,
                noise_multiplier=1,
                backend="vectorized",
                return_noise_free_oracle=True,
            )
        assert torch.equal(ambient_cpu, torch.get_rng_state())
        assert torch.equal(ambient_mps, torch.mps.get_rng_state())
        traces.append(audit)
        uploads.append(y)
    assert traces[0] == traces[1]
    assert all(torch.equal(uploads[0][k], uploads[1][k]) for k in uploads[0])
    torch.set_rng_state(before_cpu)
    torch.mps.set_rng_state(before_mps)


def test_evaluation_wrapper_restores_rng_after_error():
    assert torch.backends.mps.is_available()

    def consume(*args, **kwargs):
        torch.randperm(100)
        torch.randn(10, device="mps")
        raise RuntimeError("fixture")

    harness = SimpleNamespace(
        evaluate_global_model=consume, evaluate_client_loaders=consume
    )
    cpu, mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    with evaluation_rng_isolation(harness):
        with pytest.raises(RuntimeError, match="fixture"):
            harness.evaluate_global_model()
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(
        mps, torch.mps.get_rng_state()
    )
    assert harness.evaluate_global_model is consume
