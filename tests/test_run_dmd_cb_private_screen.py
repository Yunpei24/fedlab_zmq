"""Read-only orchestration/ledger checks, no training and no campaign writes."""
import math
from pathlib import Path

import pytest

from scripts import run_dmd_cb_private_screen as runner


def fixture_payload(config):
    a = config["training"]["algo_config"]
    rows = []
    for t in range(1, config["training"]["num_rounds"] + 1):
        scales = a["privacy_noise_multiplier_scale_by_client"]
        eps = [.25 + runner.gradient_epsilon(t, runner.sigma() * s) for s in scales]
        rows.append({
            "round_num": t, "num_clients": 25, "num_selected": 25,
            "num_survivors": 25, "num_alive_clients": 25,
            "privacy_sampling_scheme": "fixed_without_replacement",
            "privacy_adjacency": "replace_one",
            "ldp_gradient_far_private_gradient_mps_fraction": 1.0,
            "ldp_gradient_far_private_compute_device": "mps",
            "dmd_server_mode": a["dmd_server_mode"], "dmd_mu": a["dmd_mu"],
            "dmd_far_enabled": a["dmd_server_mode"] == "far_rfa",
            "privacy_dmd_histogram_calls_per_client_max": 1,
            "privacy_dmd_histogram_calls_this_round": 25 if t == 1 else 0,
            "privacy_dmd_budget_composition": "epsilon_histogram_plus_epsilon_gradient",
            "test_accuracy": .5, "test_loss": 1., "client_accuracy_mean": .5,
            "client_accuracy_variance_pct2": 100., "worst20_accuracy_pct": 20.,
            "best20_worst20_gap_pct": 30.,
            "privacy_epsilon_max": max(eps), "privacy_epsilon_mean": sum(eps) / 25,
            "privacy_delta": 1e-5,
            "privacy_model_noise_multiplier_min": runner.sigma() * min(scales),
            "privacy_model_noise_multiplier_max": runner.sigma() * max(scales),
            "privacy_model_noise_multiplier_mean": runner.sigma() * sum(scales) / 25,
            "privacy_model_steps_mean": 1.0,
            "privacy_dmd_histogram_epsilon": .25, "privacy_target_epsilon": 4.,
            "privacy_gradient_target_epsilon": 3.75,
            "privacy_gradient_epsilon_max": max(eps) - .25,
            "privacy_gradient_epsilon_mean": sum(eps) / 25 - .25,
        })
    return {"algorithm": "ldp_gradient_dmd_cb", "config": {**a, "device": "mps"},
            "summary": {"num_rounds": len(rows), "seed": config["seed"], "partition_seed": config["seed"],
                        "dataset": "fashionmnist", "model": "lenet5_tanh", "num_clients": 25},
            "rounds": rows}


def test_36_unique_tasks_and_adjacent_six_arms():
    matrix = runner.load_matrix()
    tasks = runner.tasks(matrix)
    assert len(tasks) == len({t["id"] for t in tasks}) == 36
    for start in range(0, 36, 6):
        block = tasks[start:start + 6]
        assert len({(t["seed"], t["noise"]) for t in block}) == 1
        assert {t["arm"] for t in block} == set(runner.EXPECTED_ARMS)


def test_paired_common_configuration():
    matrix = runner.load_matrix()
    configs = [runner.resolved(matrix, t, Path("/private/tmp/dmd_test_placeholder"))
               for t in runner.tasks(matrix)[:6]]
    for c in configs:
        for field in ("dmd_mu", "dmd_server_mode", "far_alpha"):
            c["training"]["algo_config"].pop(field)
    assert all(c == configs[0] for c in configs)


def test_calibration_is_conservative_and_sensitivity_is_two():
    assert 3.7499 <= runner.gradient_epsilon(40, runner.sigma()) <= 3.75
    assert .25 + runner.gradient_epsilon(40, runner.sigma()) <= 4
    acc = runner.rdp_module().RDPAccountant()
    acc.add_sampled_without_replacement_gaussian(channel="test", sampling_rate=.1,
                                                noise_multiplier=runner.sigma() / 2, steps=40)
    assert math.isclose(acc.epsilon(1e-5)[0], runner.gradient_epsilon(40, runner.sigma()))


@pytest.mark.parametrize("noise", ["homogeneous", "heteroscedastic"])
def test_recomputed_ledger_accepts_valid_fixture(noise):
    matrix = runner.load_matrix()
    task = next(t for t in runner.tasks(matrix) if t["noise"] == noise)
    config = runner.resolved(matrix, task, Path("/private/tmp/dmd_test_placeholder"))
    runner.validate_metrics(fixture_payload(config), config)


@pytest.mark.parametrize("field,value", [
    ("privacy_epsilon_max", 3.75),  # omission of the histogram channel
    ("privacy_epsilon_max", 4.01),
    ("privacy_model_noise_multiplier_min", 1.),
    ("ldp_gradient_far_private_compute_device", "cpu"),
    ("ldp_gradient_far_private_gradient_mps_fraction", .96),
    ("num_survivors", 24), ("test_accuracy", float("nan")),
    ("privacy_sampling_scheme", "poisson"),
    ("privacy_dmd_histogram_calls_this_round", 25),
    ("privacy_dmd_histogram_calls_per_client_max", 40),
])
def test_validation_rejects_bad_results(field, value):
    matrix = runner.load_matrix()
    config = runner.resolved(matrix, runner.tasks(matrix)[0], Path("/private/tmp/dmd_test_placeholder"))
    payload = fixture_payload(config)
    payload["rounds"][-1][field] = value
    with pytest.raises(RuntimeError):
        runner.validate_metrics(payload, config)


def test_completed_evidence_cannot_silently_change(tmp_path):
    stamp = {"source_sha256": {}, "test": 1}
    root = tmp_path / "new"
    runner.ensure_lock(root, stamp)
    runner.ensure_lock(root, stamp)
    with pytest.raises(RuntimeError, match="provenance drift"):
        runner.ensure_lock(root, {**stamp, "test": 2})


def test_refuses_unlocked_partial_outputs(tmp_path):
    with pytest.raises(RuntimeError, match="nonempty"):
        runner.ensure_lock(Path(__file__).parent, {"test": 1})


def test_entrypoint_and_new_sources_are_in_closure():
    closure = runner.provenance(runner.load_matrix())["source_sha256"]
    for name in ("algorithms/ldp_gradient_dmd_cb.py", "privacy/dmd_cb_private.py",
                 "scripts/run_dmd_cb_private_experiment.py", "privacy/rdp.py", "run_experiment.py"):
        assert name in closure
