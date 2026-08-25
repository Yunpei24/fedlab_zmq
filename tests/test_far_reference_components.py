"""Scientific smoke tests for FAR reference algorithms and components."""

from __future__ import annotations

import math
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import ClientState, get_algorithm
from algorithms.reference_utils import empirical_loss
from attacks import apply_attack
from datasets.partitioner import matched_dirichlet_partition
from metrics.client_fairness import (
    evaluate_client_loaders,
    summarize_client_performance,
)
from metrics.robustness import attack_diagnostics
from models.registry import get_model
from privacy.rdp import RDPAccountant, sampled_gaussian_rdp
from robustness.aggregators import aggregate_vectors, nearest_neighbor_mixing
from robustness.tensor_ops import flatten_update


def _updates(values):
    return [{"weight": torch.tensor([float(value)])} for value in values]


def test_coordinate_rules_and_nnm_are_deterministic():
    vectors = torch.tensor([[0.0], [0.1], [0.2], [20.0]], dtype=torch.float64)
    assert torch.allclose(
        aggregate_vectors(vectors, "cm"), torch.tensor([0.15], dtype=torch.float64)
    )
    assert torch.allclose(
        aggregate_vectors(vectors, "trmean", f=1),
        torch.tensor([0.15], dtype=torch.float64),
    )
    mixed = nearest_neighbor_mixing(vectors, f=1)
    assert mixed.shape == vectors.shape
    assert torch.allclose(mixed, nearest_neighbor_mixing(vectors, f=1))


def test_rfa_and_nbs_resist_one_large_outlier():
    vectors = torch.tensor([[0.0], [0.1], [-0.1], [100.0]], dtype=torch.float64)
    assert abs(float(aggregate_vectors(vectors, "rfa"))) < 0.2
    assert abs(float(aggregate_vectors(vectors, "nbs", screening_fraction=0.25))) < 0.1


def test_attacks_do_not_change_honest_updates():
    source = _updates([1.0, 1.1, 0.9, 1.05])
    for name in ("alie", "ipm", "minmax", "minsum", "bf"):
        attacked = apply_attack(source, [0], name=name, scale=2.0)
        for honest_idx in (1, 2, 3):
            assert torch.equal(
                attacked[honest_idx]["weight"], source[honest_idx]["weight"]
            )
        assert not torch.equal(attacked[0]["weight"], source[0]["weight"])


def test_attack_diagnostics_are_oracle_reporting_only():
    states = [ClientState(client_id=i, battery_j=1.0) for i in range(3)]
    tuples = [
        (
            {"weight": torch.tensor([float(i)])},
            {
                "is_byzantine": i == 0,
                "attack_name": "alie" if i == 0 else "none",
            },
            states[i],
        )
        for i in range(3)
    ]
    diagnostics = attack_diagnostics(tuples)
    assert diagnostics["attack_enabled"] is True
    assert diagnostics["attack_name"] == "alie"
    assert diagnostics["num_byzantine_oracle"] == 1
    assert math.isclose(diagnostics["byzantine_fraction_oracle"], 1 / 3)


def test_term_tilt_zero_and_positive_monotonicity():
    term = get_algorithm("term")
    losses = torch.tensor([0.2, 1.0, 2.0], dtype=torch.float64)
    zero = term.client_weights(losses, 0.0)
    positive = term.client_weights(losses, 1.0)
    assert torch.allclose(zero, torch.ones(3, dtype=torch.float64) / 3)
    assert positive[2] > positive[1] > positive[0]
    # Softmax weights are invariant when all losses get the same offset.
    assert torch.allclose(positive, term.client_weights(losses + 100.0, 1.0))


def test_far_alpha_zero_is_uniform():
    far = get_algorithm("far")
    distances = torch.tensor([0.1, 2.0, 10.0], dtype=torch.float64)
    assert torch.allclose(
        far.far_weights(distances, 0.0), torch.ones(3, dtype=torch.float64) / 3
    )


def test_far_proximal_delta_and_single_gradient_modes_are_distinct():
    torch.manual_seed(19)
    source = torch.nn.Linear(2, 2, bias=False)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 1.0]]),
            torch.tensor([0, 1, 0, 1]),
        ),
        batch_size=2,
        shuffle=False,
    )
    far = get_algorithm("far")
    common = {
        **far.get_default_config(),
        "device": "cpu",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "local_epochs": 2,
    }

    no_prox_model = copy.deepcopy(source)
    prox_model = copy.deepcopy(source)
    no_prox, no_prox_meta = far.client_update(
        no_prox_model,
        loader,
        ClientState(client_id=0, battery_j=100.0),
        {**common, "far_update_mode": "multi_epoch_delta", "far_prox_mu": 0.0},
    )
    prox, prox_meta = far.client_update(
        prox_model,
        loader,
        ClientState(client_id=1, battery_j=100.0),
        {**common, "far_update_mode": "multi_epoch_delta", "far_prox_mu": 1.0},
    )
    no_prox_vec, _ = flatten_update(no_prox)
    prox_vec, _ = flatten_update(prox)
    assert not torch.allclose(no_prox_vec, prox_vec)
    assert no_prox_meta["far_prox_active"] is False
    assert prox_meta["far_prox_active"] is True

    gradient_model = copy.deepcopy(source)
    before = copy.deepcopy(gradient_model.state_dict())
    gradient, gradient_meta = far.client_update(
        gradient_model,
        loader,
        ClientState(client_id=2, battery_j=100.0),
        {
            **common,
            "far_update_mode": "single_step_gradient",
            "far_prox_mu": 1.0,
        },
    )
    assert gradient_meta["far_update_mode"] == "single_step_gradient"
    assert gradient_meta["far_prox_active"] is False
    # Computing a paper-style gradient must not locally mutate the model.
    for name, value in gradient_model.state_dict().items():
        assert torch.equal(value, before[name])

    result = far.server_aggregate(
        gradient_model,
        [
            (
                gradient,
                gradient_meta,
                ClientState(client_id=2, battery_j=100.0),
            )
        ],
        round_num=0,
        config={
            **common,
            "far_update_mode": "single_step_gradient",
            "far_server_lr": 0.05,
            "robust_reference": "mean",
            "num_byzantine": 0,
        },
    )
    assert result.metrics["far_update_mode"] == "single_step_gradient"
    assert result.metrics["far_server_lr"] == 0.05
    assert not torch.equal(result.new_weights["weight"], before["weight"])


def test_qffl_server_uses_curvature_denominator():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    state0 = ClientState(client_id=0, battery_j=1.0)
    state1 = ClientState(client_id=1, battery_j=1.0)
    updates = [
        (
            {"weight": torch.tensor([[1.0]])},
            {"qffl_h": 1.0, "qffl_loss_at_global": 1.0, "dataset_size": 1},
            state0,
        ),
        (
            {"weight": torch.tensor([[3.0]])},
            {"qffl_h": 1.0, "qffl_loss_at_global": 1.0, "dataset_size": 1},
            state1,
        ),
    ]
    result = get_algorithm("qffl").server_aggregate(
        model,
        updates,
        round_num=0,
        config={"q": 0.0, "aggregation_prior": "uniform"},
    )
    assert torch.allclose(result.new_weights["weight"], torch.tensor([[-2.0]]))
    assert result.metrics["qffl_aggregation_prior"] == "uniform"


def test_qffl_paper_default_does_not_reweight_selected_clients_by_dataset_size():
    algo = get_algorithm("qffl")
    assert algo.get_default_config()["aggregation_prior"] == "uniform"
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    updates = [
        (
            {"weight": torch.tensor([[1.0]])},
            {"qffl_h": 1.0, "qffl_loss_at_global": 1.0, "dataset_size": 1},
            ClientState(client_id=0, battery_j=1.0),
        ),
        (
            {"weight": torch.tensor([[3.0]])},
            {"qffl_h": 1.0, "qffl_loss_at_global": 1.0, "dataset_size": 9},
            ClientState(client_id=1, battery_j=1.0),
        ),
    ]
    paper = algo.server_aggregate(model, updates, 0, algo.get_default_config())
    extension = algo.server_aggregate(
        model, updates, 0, {"q": 0.0, "aggregation_prior": "dataset_size"}
    )
    assert torch.allclose(paper.new_weights["weight"], torch.tensor([[-2.0]]))
    assert torch.allclose(extension.new_weights["weight"], torch.tensor([[-2.8]]))


def test_robustfedavg_is_an_executable_baseline():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    tuples = []
    for client_id, value in enumerate((0.0, 0.1, -0.1, 100.0)):
        state = ClientState(client_id=client_id, battery_j=1.0)
        tuples.append(
            (
                {"weight": torch.tensor([[value]])},
                {"dataset_size": 1, "local_loss": 0.0},
                state,
            )
        )
    result = get_algorithm("robustfedavg").server_aggregate(
        model,
        tuples,
        round_num=0,
        config={"robust_aggregator": "rfa", "num_byzantine": 1},
    )
    assert abs(float(result.new_weights["weight"])) < 0.2


def test_fairness_summary_has_fedfdp_weighting_and_far_units():
    result = summarize_client_performance([0.0, 1.0], [1.0, 3.0], sample_counts=[1, 3])
    assert math.isclose(result["client_accuracy_variance"], 0.25)
    assert math.isclose(result["client_accuracy_variance_pct2"], 2500.0)
    # Weighted loss mean is 2.5, hence weighted variance is 0.75.
    assert math.isclose(result["balanced_performance_fairness"], 0.75)


def test_client_fairness_can_exclude_byzantine_clients_for_oracle_eval():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    honest = DataLoader(
        TensorDataset(torch.tensor([[2.0, 0.0]]), torch.tensor([0])), batch_size=1
    )
    byzantine = DataLoader(
        TensorDataset(torch.tensor([[2.0, 0.0]]), torch.tensor([1])), batch_size=1
    )
    result = evaluate_client_loaders(
        model,
        [honest, byzantine],
        "cpu",
        client_ids=[0, 1],
        exclude_client_ids={1},
    )
    assert result["num_evaluated_clients"] == 1
    assert result["worst_client_accuracy"] == 1.0
    assert result["evaluated_client_ids_oracle"] == [0]
    assert result["client_accuracy_values_oracle"] == [1.0]


def test_matched_dirichlet_preserves_client_class_proportions():
    class LabelOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, per_class):
            self.targets = torch.arange(3).repeat_interleave(per_class)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.tensor([float(index)]), self.targets[index]

    train = LabelOnlyDataset(1000)
    test = LabelOnlyDataset(200)
    train_parts = matched_dirichlet_partition(train, 5, alpha=0.5, seed=11)
    test_parts = matched_dirichlet_partition(test, 5, alpha=0.5, seed=11)
    for class_id in range(3):
        train_counts = torch.tensor(
            [
                sum(train.targets[idx] == class_id for idx in part)
                for part in train_parts
            ],
            dtype=torch.float64,
        )
        test_counts = torch.tensor(
            [sum(test.targets[idx] == class_id for idx in part) for part in test_parts],
            dtype=torch.float64,
        )
        assert torch.allclose(
            train_counts / train_counts.sum(),
            test_counts / test_counts.sum(),
            atol=0.01,
        )


def test_sampled_gaussian_rdp_composes_channels():
    one = sampled_gaussian_rdp(4, 0.1, 1.2)
    assert one > 0
    accountant = RDPAccountant(orders=(2, 4, 8))
    accountant.add_sampled_gaussian(
        channel="model", sampling_rate=0.1, noise_multiplier=1.2, steps=3
    )
    accountant.add_sampled_gaussian(
        channel="loss", sampling_rate=0.1, noise_multiplier=2.0, steps=1
    )
    epsilon, order = accountant.epsilon(1e-5)
    assert epsilon > 0 and order in accountant.orders
    assert accountant.total_rdp()[4] > 3 * one


def test_alexnet_supports_grayscale_and_rgb():
    assert get_model("alexnet", "fashionmnist")(torch.randn(2, 1, 28, 28)).shape == (
        2,
        10,
    )
    assert get_model("alexnet_gn", "cifar10")(torch.randn(2, 3, 32, 32)).shape == (
        2,
        10,
    )


def test_fedfdp_runs_true_per_example_step():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    loader = DataLoader(
        TensorDataset(torch.randn(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])),
        batch_size=4,
    )
    algo = get_algorithm("fedfdp")
    state = ClientState(client_id=0, battery_j=100.0)
    update, metadata = algo.client_update(
        model,
        loader,
        state,
        {
            **algo.get_default_config(),
            "batch_size": 4,
            "noise_multiplier": 2.0,
            "loss_noise_multiplier": 2.0,
            "adaptive_loss_clip": False,
        },
    )
    vector, _ = flatten_update(update)
    assert vector.numel() == 10
    assert metadata["model_steps"] == 1
    assert metadata["privacy_epsilon"] > 0
    assert set(state.custom["fedfdp_accountant"]["channels"]) == {"model", "loss"}


def test_fedfair_is_dynamic_lr_without_hidden_dp_clipping():
    torch.manual_seed(31)
    source = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    loader = DataLoader(
        TensorDataset(torch.randn(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])),
        batch_size=4,
        shuffle=False,
    )
    algo = get_algorithm("fedfair")
    assert "target_epsilon" not in algo.get_default_config()
    common = {
        **algo.get_default_config(),
        "lr": 0.05,
        "fairness_lambda": 0.1,
        "fair_global_loss": 1.0,
    }
    tiny_clip_model = copy.deepcopy(source)
    huge_clip_model = copy.deepcopy(source)
    tiny_update, tiny_meta = algo.client_update(
        tiny_clip_model,
        loader,
        ClientState(client_id=0, battery_j=100.0),
        {**common, "clip_norm": 1e-12},
    )
    huge_update, huge_meta = algo.client_update(
        huge_clip_model,
        loader,
        ClientState(client_id=1, battery_j=100.0),
        {**common, "clip_norm": 1e12},
    )
    tiny_vec, _ = flatten_update(tiny_update)
    huge_vec, _ = flatten_update(huge_update)
    assert torch.allclose(tiny_vec, huge_vec)
    assert tiny_meta["clip_rate"] == huge_meta["clip_rate"] == 0.0
    assert tiny_meta["privacy_epsilon"] is None
    assert math.isclose(
        tiny_meta["fedfair_reported_loss"],
        empirical_loss(tiny_clip_model, loader, "cpu"),
        rel_tol=1e-6,
    )


def test_fedfdp_without_noise_remains_a_fair_clipping_ablation():
    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    loader = DataLoader(
        TensorDataset(torch.randn(4, 1, 2, 2), torch.tensor([0, 1, 0, 1])),
        batch_size=4,
        shuffle=False,
    )
    algo = get_algorithm("fedfdp")
    _, metadata = algo.client_update(
        model,
        loader,
        ClientState(client_id=0, battery_j=100.0),
        {
            **algo.get_default_config(),
            "enable_dp": False,
            "clip_norm": 1e-8,
            "adaptive_loss_clip": False,
            "batch_size": 4,
        },
    )
    assert metadata["clip_rate"] == 1.0
    assert metadata["privacy_epsilon"] is None
    assert metadata["privacy_accounting_assumption"] == "not_applicable"
