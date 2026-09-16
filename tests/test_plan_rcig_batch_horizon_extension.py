"""Read-only RCIG extension checks: no model, torch, MPS, or training is loaded."""

import copy
import json
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from scripts import plan_rcig_batch_horizon_extension as planner


@pytest.fixture(scope="module")
def config():
    return planner.load_config()


@pytest.fixture(scope="module")
def sigma_report(config):
    return planner.build_sigma_table(config)


@pytest.mark.parametrize("count,total,evaluation", [(4, 4302, 2520), (12, 9342, 7560)])
def test_expanded_counts_and_unique_nonexecutable_run_ids(
    config, count, total, evaluation
):
    plan = planner.build_plan(config, count)
    runs = list(planner.iter_planned_runs(config, count))
    assert plan["total_planned_runs"] == len(runs) == total
    assert len({run["run_id"] for run in runs}) == total
    assert Counter(run["phase"] for run in runs) == {
        "fresh_calibration": 1134,
        "independent_null_validation": 648,
        "exploratory_comparison": evaluation,
    }
    assert not any(run["training_executable"] for run in runs)
    assert plan["readiness"] == "deferred_unverified_and_implementation_required"


def test_pairing_is_exactly_five_references_per_evaluation_block(config):
    blocks = defaultdict(set)
    for run in planner.iter_planned_runs(config):
        if run["phase"] == "exploratory_comparison":
            blocks[run["pairing_block"]].add(run["reference"])
    assert len(blocks) == 18 * 7 * 4
    assert all(
        refs == {"rcig_full", "recent_reference", "uniform", "fcc", "rfa"}
        for refs in blocks.values()
    )


def test_fresh_phase_seed_blocks_are_disjoint(config):
    seeds = planner.seed_registries(config, 12)
    assert seeds["calibration"] == list(range(950001, 950064))
    assert seeds["null_validation"] == list(range(960001, 960037))
    assert seeds["evaluation"] == list(range(970001, 970013))
    assert len(set(sum(seeds.values(), []))) == 63 + 36 + 12
    assert planner.seed_registries(config)["evaluation"] == [
        970001,
        970002,
        970003,
        970004,
    ]


@pytest.mark.parametrize("horizon,recovery_end", [(40, 24), (115, 69), (268, 160)])
def test_seven_attack_scenarios_with_inclusive_recovery(config, horizon, recovery_end):
    scenarios = planner.scenarios(config, horizon)
    assert len(scenarios) == 7
    assert scenarios[0] == {"id": "no_attack", "name": "none", "enabled": False}
    for scenario in scenarios[1:]:
        assert scenario["first_round"] == 17
        assert scenario["last_round"] == (
            horizon if scenario["id"].endswith("persistent") else recovery_end
        )
        assert scenario["attack_client_ids"] == [0, 1, 2, 3, 4]
        assert scenario["scale"] == (10.0 if scenario["name"] == "bf" else 1.0)


def test_threshold_scope_and_predecessor_completion_are_not_success_gates(config):
    rule = config["threshold_rule"]
    assert rule["calibrated_mode"] == "deployed_full_only"
    assert rule["import_predecessor_r0_thresholds"] is False
    assert rule["number_of_cells"] == 18
    assert rule["statistic_across_seeds"] == "maximum_of_63_seed_maxima"
    assert 1 - 18 * 0.9**63 >= 0.975
    assert (
        planner.build_plan(config)["simultaneous_calibration_confidence_lower"]
        == 1 - 18 * 0.9**63
    )
    assert all(
        cell["post_warmup_rounds"] == [13, cell["horizon"]]
        for cell in planner.grid_cells(config)
    )
    trigger = config["deferred_trigger"]
    assert trigger["predecessor_campaign"] == "rcig_r2_r3_exploratory_v1"
    assert (
        trigger["predecessor_result_root"]
        == "results/ldp_gradient_far/rcig_r2_r3_exploratory_v1"
    )
    assert trigger["predecessor_phase"] == "r3_e2e_confirmation"
    assert trigger["required_valid_completed_runs"] == 384
    assert trigger["required_active_campaign_processes"] == 0
    assert trigger["completion_is_not_scientific_success"] is True
    assert config["analysis_plan"]["negative_results_must_remain_negative"] is True


def test_calibration_uses_no_activation_control_and_requires_coupling_validation(
    config,
):
    rule = config["threshold_rule"]
    assert rule["calibration_reference"] == "identity_new_no_activation"
    assert rule["deployed_reference"] == "rcig_full"
    assert rule["first_activation_coupling_required"] is True
    assert (
        rule["first_activation_coupling_status"]
        == "requires_implementation_and_validation"
    )
    assert rule["post_activation_closed_loop_trajectory_equivalence_claimed"] is False
    references = defaultdict(set)
    for run in planner.iter_planned_runs(config):
        references[run["phase"]].add(run["reference"])
    assert references["fresh_calibration"] == {"identity_new_no_activation"}
    assert references["independent_null_validation"] == {"rcig_full"}


def test_model_metrics_define_loss_variance_and_best20_minus_worst20(config):
    analysis = config["analysis_plan"]
    assert "test_loss" in analysis["model_metrics"]
    assert "honest_client_mean_accuracy" in analysis["model_metrics"]
    assert "honest_client_accuracy_variance_pp2" in analysis["model_metrics"]
    assert "honest_best20_minus_worst20_gap_pp" in analysis["model_metrics"]
    assert analysis["fairness_gap_is_not_best_single_client_minus_worst_single_client"]


def test_identity_new_is_explicitly_unsupported_with_matched_warmup(config):
    refs = {item["id"]: item for item in config["references"]}
    assert refs["recent_reference"]["implementation_required"] is True
    assert refs["recent_reference"]["deployable"] is False
    assert refs["recent_reference"]["robust_reference"] is None
    assert refs["recent_reference"]["same_client_gate_as_rcig"] is True
    assert refs["recent_reference"]["temporal_fusion"] is False
    assert refs["recent_reference"]["freeze_policy"] is False
    assert refs["recent_reference"]["recovery_policy"] is False
    assert (
        refs["recent_reference"]["definition"]
        == "strict_past_robust_gated_recent_window_view_projected_to_output_radius"
    )
    assert (
        refs["rcig_full"]["uniform_warmup_rounds"]
        == refs["recent_reference"]["uniform_warmup_rounds"]
        == 12
    )
    assert refs["uniform"]["far_alpha"] == 0.0
    assert all(
        ref["far_alpha"] == 0.1 for name, ref in refs.items() if name != "uniform"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(training_executable=True),
        lambda c: c["grid"].update(batch_sizes=[120, 240]),
        lambda c: c["privacy"].update(sensitivity_multiplier=1.0),
        lambda c: c["threshold_rule"].update(import_predecessor_r0_thresholds=True),
        lambda c: c["references"][1].update(deployable=True),
        lambda c: c["randomness"]["evaluation"].update(first_seed=950001),
        lambda c: c["scientific_scope"].update(local_clip_norm=1.0),
        lambda c: c["algorithm_contract"].update(tilt_bound_policy="strict"),
        lambda c: c["scenarios"]["schedules"][1].update(first_round=1),
        lambda c: c["randomness"]["evaluation"].update(default_count=3),
        lambda c: c["deferred_trigger"].update(
            predecessor_campaign="rcig_ldp_gradient_far_end_to_end_v2"
        ),
        lambda c: c["threshold_rule"].update(
            post_activation_closed_loop_trajectory_equivalence_claimed=True
        ),
    ],
)
def test_invalid_design_fails_closed(config, mutation):
    changed = copy.deepcopy(config)
    mutation(changed)
    with pytest.raises(ValueError):
        planner.validate_config(changed)


def test_sigma_table_uses_existing_wor_accountant_and_recomputes_all_clients(
    config, sigma_report
):
    assert sigma_report["base_calibration_count"] == 9
    assert sigma_report["noise_regime_cell_count"] == len(sigma_report["rows"]) == 18
    assert len(sigma_report["accountant_source_sha256"]) == 64
    for row in sigma_report["rows"]:
        assert row["sampling_rate"] == row["batch_size"] / 2400
        assert row["private_steps"] == row["horizon"]
        assert row["maximum_epsilon_within_tolerance"] is True
        assert abs(row["maximum_client_epsilon"] - 4.0) <= 1e-4
        assert len(row["epsilon_by_client"]) == len(row["client_scales"]) == 25
        expected = {}
        for scale in set(row["client_scales"]):
            accountant = planner._rdp_module().RDPAccountant(
                orders=tuple(config["privacy"]["orders"])
            )
            accountant.add_sampled_without_replacement_gaussian(
                channel="independent_recomputation",
                sampling_rate=row["batch_size"] / 2400,
                noise_multiplier=row["sigma"] * scale / 2.0,
                steps=row["horizon"],
            )
            expected[scale] = accountant.epsilon(1e-5)[0]
        assert row["epsilon_by_client"] == [expected[s] for s in row["client_scales"]]
        assert row["released_average_noise_sd_by_client"] == [
            4.0 * row["sigma"] * s / row["batch_size"] for s in row["client_scales"]
        ]
        if row["noise_regime"] == "heteroscedastic":
            assert row["minimum_client_epsilon"] < row["maximum_client_epsilon"]
        else:
            assert row["minimum_client_epsilon"] == row["maximum_client_epsilon"]


def test_plans_are_deterministic_and_do_not_mutate_config(config):
    before = copy.deepcopy(config)
    assert planner.build_plan(config) == planner.build_plan(config)
    assert config == before


def test_cli_default_is_json_plan_and_has_no_training_imports():
    code = (
        "import sys; from scripts import plan_rcig_batch_horizon_extension as p; "
        "p.main([]); assert 'torch' not in sys.modules; "
        "assert 'privacy' not in sys.modules; assert sys.dont_write_bytecode"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=planner.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["total_planned_runs"] == 4302
    assert report["training_executable"] is False
    assert len(report["config_sha256"]) == 64


def test_direct_accountant_loading_does_not_import_training_framework():
    code = (
        "import sys; from scripts import plan_rcig_batch_horizon_extension as p; "
        "p._rdp_module(); assert 'torch' not in sys.modules; "
        "assert 'privacy' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=planner.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_sigma_cli_is_read_only_and_does_not_import_training_framework():
    # An audit hook makes *any* filesystem mutation a failing operation, rather
    # than relying on an inventory that could miss a transient file write.
    code = """
import os
import sys

def reject_filesystem_writes(event, args):
    if event == 'open':
        flags = args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('Filesystem write attempted: ' + str(args[0]))
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink'}:
        raise AssertionError('Filesystem mutation attempted: ' + event)

sys.addaudithook(reject_filesystem_writes)
from scripts import plan_rcig_batch_horizon_extension as planner
planner.main(['--sigma-table'])
assert 'torch' not in sys.modules
assert 'privacy' not in sys.modules
assert not any(name.startswith('scripts.run_') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=planner.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert len(report["rows"]) == 18
    assert report["training_executable"] is False


def test_cli_rejects_launch_or_output_arguments():
    for option in ("--launch", "--run", "--output"):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(planner.ROOT / "scripts/plan_rcig_batch_horizon_extension.py"),
                option,
            ],
            cwd=planner.ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
