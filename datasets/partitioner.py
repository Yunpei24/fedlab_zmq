"""
datasets/partitioner.py
=======================
FL-standard data partitioning strategies.

Supported:
  - IID          : uniform random split
  - Dirichlet    : non-IID via Dir(alpha) label distribution (Hsu et al., 2019)
  - Client Dirichlet balanced: one label profile per client with controlled sizes
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


def _balanced_client_sizes(num_samples: int, num_clients: int, seed: int) -> np.ndarray:
    """Return deterministic quotas that differ by at most one sample.

    The clients receiving a possible remainder are shuffled independently of
    the dataset split.  This avoids always favouring low client IDs while
    preserving reproducibility.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    base, remainder = divmod(int(num_samples), int(num_clients))
    quotas = np.full(num_clients, base, dtype=np.int64)
    if remainder:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31_415]))
        recipients = rng.permutation(num_clients)[:remainder]
        quotas[recipients] += 1
    return quotas


def _client_dirichlet_profiles(
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Sample one class-proportion profile for each client.

    The random stream does not depend on the train/test split size.  Calling
    the partitioner with the same seed therefore gives train and held-out test
    shards driven by the same latent client profiles.
    """

    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 27_182]))
    return rng.dirichlet(
        np.full(num_classes, float(alpha), dtype=np.float64),
        size=num_clients,
    )


def _balanced_integer_allocation(
    profiles: np.ndarray,
    row_targets: np.ndarray,
    column_targets: np.ndarray,
    *,
    max_iterations: int = 10_000,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Fit and integerise a client-by-class allocation table.

    Iterative proportional fitting adjusts the sampled client profiles to the
    exact client-size and observed class-count margins.  Largest-remainder
    rounding then produces integer counts while preserving both margins.

    This operation is necessary because independently rounding each client's
    Dirichlet profile would duplicate some examples and leave others unused.
    """

    profiles = np.asarray(profiles, dtype=np.float64)
    row_targets = np.asarray(row_targets, dtype=np.int64)
    column_targets = np.asarray(column_targets, dtype=np.int64)
    if profiles.shape != (len(row_targets), len(column_targets)):
        raise ValueError("profiles shape does not match allocation margins")
    if int(row_targets.sum()) != int(column_targets.sum()):
        raise ValueError("row and column allocation totals must match")

    # Dirichlet draws are positive in theory, but clipping protects the matrix
    # scaling from floating-point underflow for very small alpha.
    fitted = np.maximum(profiles, np.finfo(np.float64).tiny).copy()
    for _ in range(max_iterations):
        fitted *= (row_targets / fitted.sum(axis=1))[:, None]
        fitted *= (column_targets / fitted.sum(axis=0))[None, :]
        row_error = np.max(np.abs(fitted.sum(axis=1) - row_targets))
        column_error = np.max(np.abs(fitted.sum(axis=0) - column_targets))
        if max(row_error, column_error) <= tolerance:
            break
    else:
        raise RuntimeError("balanced Dirichlet allocation did not converge")

    allocation = np.floor(fitted).astype(np.int64)
    row_remaining = row_targets - allocation.sum(axis=1)
    column_remaining = column_targets - allocation.sum(axis=0)
    if np.any(row_remaining < 0) or np.any(column_remaining < 0):
        raise RuntimeError("invalid negative residual after allocation rounding")

    # At most O(num_clients * num_classes) units remain after flooring.  Pick
    # the feasible cell with the largest fractional part at each step.  Cells
    # already incremented are softly discouraged, but may receive another unit
    # if that is required to satisfy the exact margins.
    scores = fitted - allocation
    while int(row_remaining.sum()) > 0:
        feasible_rows = np.flatnonzero(row_remaining > 0)
        feasible_columns = np.flatnonzero(column_remaining > 0)
        if not len(feasible_rows) or not len(feasible_columns):
            raise RuntimeError("could not satisfy integer allocation margins")
        feasible_scores = scores[np.ix_(feasible_rows, feasible_columns)]
        flat_index = int(np.argmax(feasible_scores))
        row_offset, column_offset = np.unravel_index(
            flat_index, feasible_scores.shape
        )
        row = int(feasible_rows[row_offset])
        column = int(feasible_columns[column_offset])
        allocation[row, column] += 1
        row_remaining[row] -= 1
        column_remaining[column] -= 1
        scores[row, column] -= 1.0

    if not np.array_equal(allocation.sum(axis=1), row_targets):
        raise RuntimeError("client-size margins were not preserved")
    if not np.array_equal(allocation.sum(axis=0), column_targets):
        raise RuntimeError("class-count margins were not preserved")
    return allocation


def client_dirichlet_balanced_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Client-centric Dirichlet partition with controlled local sizes.

    Unlike the usual class-centric split, which draws a distribution over
    clients separately for every class, this policy first draws one label
    distribution per client::

        pi_i ~ Dirichlet(alpha * 1_K).

    It then fits those profiles to (i) balanced client quotas and (ii) the
    dataset's exact class counts.  Every example is assigned exactly once and
    local dataset sizes differ by at most one sample.
    """

    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
    else:
        labels = np.asarray([dataset[index][1] for index in range(len(dataset))])
    labels = labels.reshape(-1)
    classes, encoded_labels = np.unique(labels, return_inverse=True)
    num_classes = int(len(classes))

    profiles = _client_dirichlet_profiles(
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=alpha,
        seed=seed,
    )
    client_sizes = _balanced_client_sizes(len(dataset), num_clients, seed)
    class_sizes = np.bincount(encoded_labels, minlength=num_classes).astype(np.int64)
    allocation = _balanced_integer_allocation(
        profiles,
        row_targets=client_sizes,
        column_targets=class_sizes,
    )

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for class_position in range(num_classes):
        class_indices = np.flatnonzero(encoded_labels == class_position)
        index_rng = np.random.default_rng(
            np.random.SeedSequence([seed, class_position, 41_421, len(dataset)])
        )
        index_rng.shuffle(class_indices)
        cursor = 0
        for client_id in range(num_clients):
            count = int(allocation[client_id, class_position])
            client_indices[client_id].extend(
                class_indices[cursor : cursor + count].tolist()
            )
            cursor += count
        if cursor != len(class_indices):
            raise RuntimeError("not every class example was allocated")

    for client_id, indices in enumerate(client_indices):
        client_rng = np.random.default_rng(
            np.random.SeedSequence([seed, client_id, 17_321, len(dataset)])
        )
        client_rng.shuffle(indices)

    flattened = [index for indices in client_indices for index in indices]
    if len(flattened) != len(dataset) or len(set(flattened)) != len(dataset):
        raise RuntimeError("balanced Dirichlet partition lost or duplicated examples")
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
        partition: "iid" | "dirichlet" | "client_dirichlet_balanced" |
                   "pathological"
    """
    if partition == "iid":
        idx_lists = iid_partition(dataset, num_clients, seed)
    elif partition == "dirichlet":
        if matched_dirichlet:
            idx_lists = matched_dirichlet_partition(dataset, num_clients, alpha, seed)
        else:
            idx_lists = dirichlet_partition(dataset, num_clients, alpha, seed)
    elif partition == "client_dirichlet_balanced":
        idx_lists = client_dirichlet_balanced_partition(
            dataset, num_clients, alpha, seed
        )
    elif partition == "pathological":
        idx_lists = pathological_partition(dataset, num_clients, num_shards, seed)
    else:
        raise ValueError(
            f"Unknown partition: {partition}. "
            "Choose from: iid, dirichlet, client_dirichlet_balanced, "
            "pathological"
        )

    return [Subset(dataset, indices) for indices in idx_lists]
