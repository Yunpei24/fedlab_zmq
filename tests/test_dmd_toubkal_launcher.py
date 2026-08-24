from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_dmd_toubkal.py"
SPEC = importlib.util.spec_from_file_location("run_dmd_toubkal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
MATRIX = ROOT / "configs" / "dmd" / "toubkal_phase_a.yaml"


def test_matrix_counts_and_unique_method_directories(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    assert MODULE.validate_matrix(MATRIX, document) == []

    expected = {
        "smoke": 24,
        "phase_a_full": 180,
        "phase_a_partial_no_dropout": 120,
        "phase_a_partial_dropout": 120,
    }
    for stage, count in expected.items():
        tasks = MODULE.expand_tasks(
            document, output_root=tmp_path, stage_ids={stage}
        )
        assert len(tasks) == count
        assert [task.index for task in tasks] == list(range(count))
        assert len({task.output_dir for task in tasks}) == count
        assert all(task.method_id in task.output_dir.parts for task in tasks)


def test_command_is_single_method_cuda_and_paired_seed(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    task = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_partial_dropout"},
        scenario_ids={"emnist_byclass_cnngn"},
        method_ids={"dmd_cb_usv025"},
        seeds={101},
        alphas={0.1},
    )[0]
    command = MODULE.task_command(
        task,
        python_bin="python3",
        device="cuda",
        data_root=tmp_path / "data",
        resume=True,
    )

    assert command[command.index("--methods") + 1] == (
        "margin_mean_cb_fixed_zero_stale_usv_r025"
    )
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--seed") + 1] == "101"
    assert command[command.index("--partition-seed") + 1] == "101"
    assert command[command.index("--participation-seed") + 1] == "101"
    assert command[command.index("--participation-rate") + 1] == "0.5"
    assert command[command.index("--dropout-rate") + 1] == "0.2"
    assert command[command.index("--dataset") + 1] == "emnist"
    assert command[command.index("--model") + 1] == "cnn_gn"
    assert "--resume" in command


def test_pilot_rounds_are_isolated_from_confirmatory_outputs(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    pilot = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"fedavg"},
        seeds={101},
        alphas={0.1},
        pilot_rounds=2,
    )[0]
    full = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"fedavg"},
        seeds={101},
        alphas={0.1},
    )[0]

    assert pilot.config["rounds"] == 2
    assert full.config["rounds"] == 150
    assert pilot.output_dir != full.output_dir
    assert "pilots" in pilot.output_dir.parts

