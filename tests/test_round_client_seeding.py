import numpy as np
import torch

from core.seeding import round_client_seed, seed_torch


def test_round_client_seed_depends_only_on_its_arguments() -> None:
    assert round_client_seed(1, 5, 3) == round_client_seed(1, 5, 3)
    assert len({round_client_seed(1, t, c) for t in range(20) for c in range(10)}) == 200
    assert round_client_seed(1, 5, 3) != round_client_seed(2, 5, 3)


def test_reseeding_gives_the_same_draws_whatever_was_consumed_before() -> None:
    seed_torch(round_client_seed(7, 4, 2))
    reference = torch.rand(16)
    torch.rand(1000)  # an arm that draws more before this client's update
    seed_torch(round_client_seed(7, 4, 2))
    assert torch.equal(torch.rand(16), reference)


def test_seed_torch_leaves_the_numpy_stream_alone() -> None:
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    seed_torch(123)
    assert np.random.rand() == expected
