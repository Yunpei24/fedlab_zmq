from __future__ import annotations

import run_experiment


def test_shared_partition_seed_is_independent_of_training_seed() -> None:
    data_config = {"partition_seed": 28}

    assert run_experiment.resolve_partition_seed(data_config, training_seed=28) == 28
    assert run_experiment.resolve_partition_seed(data_config, training_seed=36) == 28
    assert run_experiment.resolve_partition_seed(data_config, training_seed=54) == 28


def test_partition_seed_defaults_to_training_seed_when_unspecified() -> None:
    assert run_experiment.resolve_partition_seed({}, training_seed=36) == 36
