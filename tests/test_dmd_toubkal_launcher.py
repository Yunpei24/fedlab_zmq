from __future__ import annotations

import importlib.util
import json
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
        "smoke": 28,
        "phase_a_full": 630,
        "phase_a_partial_no_dropout": 420,
        "phase_a_partial_dropout": 420,
    }
    for stage, count in expected.items():
        tasks = MODULE.expand_tasks(
            document, output_root=tmp_path, stage_ids={stage}
        )
        assert len(tasks) == count
        assert [task.index for task in tasks] == list(range(count))
        assert len({task.output_dir for task in tasks}) == count
        assert all(task.method_id in task.output_dir.parts for task in tasks)

    smoke = MODULE.expand_tasks(
        document, output_root=tmp_path, stage_ids={"smoke"}
    )
    assert {task.method_id for task in smoke} == {
        "fedavg",
        "fedfair",
        "fedfair_loss",
        "term",
        "margin_mean_sample",
        "dmd_cb",
        "dmd_cb_usv025",
    }


def test_command_separates_partition_and_training_seeds(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    task = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_partial_dropout"},
        scenario_ids={"emnist_byclass_cnngn"},
        method_ids={"dmd_cb_usv025"},
        partition_seeds={101},
        training_seeds={201},
        alphas={0.1},
    )[0]
    command = MODULE.task_command(
        task,
        python_bin="python3",
        device="cuda",
        data_root=tmp_path / "data",
        resume=True,
    )

    MODULE.write_native_config(task, data_root=tmp_path / "data", device="cuda")
    assert command[command.index("--algo") + 1] == "dmd_usv"
    assert command[1].endswith("run_experiment.py")
    assert command[command.index("--config") + 1] == str(task.resolved_config_path)
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--seed") + 1] == "201"
    resolved = MODULE.native_config(task, data_root=tmp_path / "data", device="cuda")
    assert resolved["seed"] == 201
    assert resolved["data"]["partition_seed"] == 101
    assert resolved["clients"]["sample_fraction"] == 0.5
    assert resolved["clients"]["dropout_rate"] == 0.2
    assert resolved["data"]["dataset"] == "emnist"
    assert resolved["model"]["architecture"] == "cnn_gn"
    assert resolved["training"]["algo_config"]["num_classes"] == 62
    assert resolved["training"]["algo_config"]["reference_mode"] == "fixed_zero"
    assert "--resume" not in command


def test_fedfair_arm_uses_faithful_algorithm_one_adapter(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    task = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"fedfair"},
        partition_seeds={101},
        training_seeds={201},
        alphas={0.1},
    )[0]

    resolved = MODULE.native_config(
        task, data_root=tmp_path / "data", device="cpu"
    )
    algo = resolved["training"]["algo_config"]
    assert resolved["training"]["algorithm"] == "fedfair"
    assert algo["fairness_lambda"] == 0.1
    assert algo["initial_global_loss"] == 1.0
    assert algo["fedfair_scale_policy"] == "error"
    assert algo["loss_eval_max_batches"] is None


def test_pilot_rounds_are_isolated_from_confirmatory_outputs(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    pilot = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"fedavg"},
        partition_seeds={101},
        training_seeds={201},
        alphas={0.1},
        pilot_rounds=2,
    )[0]
    full = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"fedavg"},
        partition_seeds={101},
        training_seeds={201},
        alphas={0.1},
    )[0]

    assert pilot.config["rounds"] == 2
    assert full.config["rounds"] == 150
    assert pilot.output_dir != full.output_dir
    assert "pilots" in pilot.output_dir.parts


def test_native_completion_contract_uses_metrics_json(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    task = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"smoke"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"dmd_cb"},
        partition_seeds={101},
        training_seeds={201},
        alphas={0.1},
        pilot_rounds=2,
    )[0]
    task.result_dir.mkdir(parents=True)
    (task.result_dir / "metrics.json").write_text(
        json.dumps({"rounds": [{"round_num": 1}, {"round_num": 2}]}),
        encoding="utf-8",
    )
    (task.result_dir / "final_model.pt").write_bytes(b"model")
    (task.result_dir / "manifest.json").write_text("{}", encoding="utf-8")
    assert MODULE.is_complete(task)


def test_factorial_seed_grid_has_unique_paths_and_fixed_partition(tmp_path: Path) -> None:
    document = MODULE.load_matrix(MATRIX)
    tasks = MODULE.expand_tasks(
        document,
        output_root=tmp_path,
        stage_ids={"phase_a_full"},
        scenario_ids={"cifar10_resnet18gn"},
        method_ids={"dmd_cb"},
        partition_seeds={101},
        alphas={0.1},
    )
    assert [task.training_seed for task in tasks] == [201, 202, 203]
    assert {task.partition_seed for task in tasks} == {101}
    assert len({task.output_dir for task in tasks}) == 3
    for task in tasks:
        resolved = MODULE.native_config(task, data_root=tmp_path / "data", device="cpu")
        assert resolved["data"]["partition_seed"] == 101
        assert resolved["seed"] == task.training_seed
