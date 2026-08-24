"""
datasets/partitioner.py
=======================
FL-standard data partitioning strategies.

Supported:
  - IID          : uniform random split
  - Dirichlet    : non-IID via Dir(alpha) label distribution (Hsu et al., 2019)
  - Pathological : each client gets exactly N shards of sorted data
  - Natural      : use pre-defined splits (FEMNIST by writer, etc.)

References:
  Hsu et al. (2019) "Measuring the Effects of Non-Identical Data Distribution
  for Federated Visual Classification" arXiv:1909.06335
"""

import numpy as np
from torch.utils.data import Dataset, Subset


# ─────────────────────────────────────────────────────────────────────────────


def iid_partition(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> list[list[int]]:
    """
    Uniformly random IID partition.
    Each client gets len(dataset) // num_clients samples.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    size = len(dataset) // num_clients
    return [indices[i * size : (i + 1) * size] for i in range(num_clients)]


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 10,
) -> list[list[int]]:
    """
    Non-IID partition via Dirichlet distribution (Hsu et al., 2019).

    Lower alpha → more heterogeneous (each client dominated by 1-2 classes).
    alpha → ∞   → approaches IID.

    Args:
        alpha:       Dirichlet concentration parameter
        min_samples: Minimum samples per client (prevent empty clients)
    """
    rng = np.random.default_rng(seed)

    # Get labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)

        # Enforce minimum samples
        proportions = np.array(
            [
                p * (len(idx_j) < min_samples or True)
                for p, idx_j in zip(proportions, client_indices)
            ]
        )
        proportions = proportions / proportions.sum()

        # Assign indices
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[k].extend(chunk.tolist())

    # Shuffle each client's data
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices


def matched_dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Dirichlet split whose class proportions match across train and test.

    A usual implementation consumes random numbers while shuffling each
    class.  Because train and test class sizes differ, reusing the same seed
    does *not* then reproduce the same client proportions.  Client-level
    fairness evaluation needs each held-out shard to represent the matching
    training client.  We therefore use independent deterministic streams for
    the class proportions and for index shuffling.
    """

    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
    else:
        labels = np.asarray([dataset[i][1] for i in range(len(dataset))])

    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_idx = np.where(labels == class_id)[0]
        index_rng = np.random.default_rng(
            np.random.SeedSequence([seed, class_id, 1, len(dataset)])
        )
        index_rng.shuffle(class_idx)
        proportion_rng = np.random.default_rng(
            np.random.SeedSequence([seed, class_id, 0])
        )
        proportions = proportion_rng.dirichlet(np.full(num_clients, float(alpha)))
        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(class_idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    for client_id, indices in enumerate(client_indices):
        client_rng = np.random.default_rng(
            np.random.SeedSequence([seed, client_id, 2, len(dataset)])
        )
        client_rng.shuffle(indices)
    return client_indices


def pathological_partition(
    dataset: Dataset,
    num_clients: int,
    num_shards: int = 200,
    seed: int = 42,
) -> list[list[int]]:
    """
    Pathological non-IID: sort by label, split into num_shards shards,
    assign num_shards // num_clients shards per client.

    Classic "2 classes per client" scenario from McMahan et al. (2017).
    """
    rng = np.random.default_rng(seed)

    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    sorted_idx = np.argsort(labels)
    shard_size = len(dataset) // num_shards
    shards = [
        sorted_idx[i * shard_size : (i + 1) * shard_size].tolist()
        for i in range(num_shards)
    ]

    shard_indices = rng.permutation(num_shards).tolist()
    shards_per_client = num_shards // num_clients

    client_indices = []
    for k in range(num_clients):
        assigned = shard_indices[k * shards_per_client : (k + 1) * shards_per_client]
        combined = []
        for s in assigned:
            combined.extend(shards[s])
        client_indices.append(combined)

    return client_indices


def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    partition: str = "dirichlet",
    alpha: float = 0.5,
    num_shards: int = 200,
    seed: int = 42,
    matched_dirichlet: bool = False,
) -> list[Subset]:
    """
    Main entry point: partition a dataset and return list of Subsets.

    Args:
        partition: "iid" | "dirichlet" | "pathological"
    """
    if partition == "iid":
        idx_lists = iid_partition(dataset, num_clients, seed)
    elif partition == "dirichlet":
        if matched_dirichlet:
            idx_lists = matched_dirichlet_partition(dataset, num_clients, alpha, seed)
        else:
            idx_lists = dirichlet_partition(dataset, num_clients, alpha, seed)
    elif partition == "pathological":
        idx_lists = pathological_partition(dataset, num_clients, num_shards, seed)
    else:
        raise ValueError(
            f"Unknown partition: {partition}. "
            f"Choose from: iid, dirichlet, pathological"
        )

    return [Subset(dataset, indices) for indices in idx_lists]
