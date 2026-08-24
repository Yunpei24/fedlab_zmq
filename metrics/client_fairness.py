"""Client-level fairness metrics used by q-FFL, FedFDP and FAR."""

from __future__ import annotations

import math

import numpy as np
import torch


def summarize_client_performance(
    accuracies: list[float],
    losses: list[float],
    tail_fraction: float = 0.2,
    sample_counts: list[int] | None = None,
) -> dict[str, float]:
    """Summarise the empirical performance distribution across clients.

    FAR reports variance and the Best-20/Worst-20 gap. q-FFL additionally
    reports worst and best tails. FedFDP's balanced-performance fairness is
    the (weighted) variance of client losses; the equal-client version is
    returned here because every client is one fairness unit.
    """

    if not accuracies or len(accuracies) != len(losses):
        raise ValueError("Need equally-sized non-empty accuracy and loss lists")
    acc = np.asarray(accuracies, dtype=float)
    loss = np.asarray(losses, dtype=float)
    if sample_counts is None:
        weights = np.full(len(acc), 1.0 / len(acc))
    else:
        counts = np.asarray(sample_counts, dtype=float)
        if counts.shape != acc.shape or counts.sum() <= 0:
            raise ValueError("sample_counts must match clients and have positive sum")
        weights = counts / counts.sum()
    k = max(1, int(math.ceil(tail_fraction * len(acc))))
    ordered = np.sort(acc)
    worst = float(ordered[:k].mean())
    best = float(ordered[-k:].mean())
    weighted_loss_mean = float(np.sum(weights * loss))
    weighted_loss_variance = float(np.sum(weights * (loss - weighted_loss_mean) ** 2))
    weighted_acc_mean = float(np.sum(weights * acc))
    weighted_acc_variance = float(np.sum(weights * (acc - weighted_acc_mean) ** 2))
    return {
        "client_accuracy_mean": float(acc.mean()),
        "client_accuracy_variance": float(acc.var()),
        "client_accuracy_std": float(acc.std()),
        "client_loss_mean": float(loss.mean()),
        "client_loss_variance": float(loss.var()),
        # FedFDP's balanced-performance fairness uses client data weights p_i.
        "balanced_performance_fairness": weighted_loss_variance,
        "weighted_client_loss_variance": weighted_loss_variance,
        "weighted_client_accuracy_variance": weighted_acc_variance,
        "worst_client_accuracy": float(acc.min()),
        "worst20_accuracy": worst,
        "best20_accuracy": best,
        "best20_worst20_gap": best - worst,
        # FAR tables report accuracy in percentage points. Keep both units so
        # a value such as 120 means 120 percentage-points squared, not 1.2.
        "client_accuracy_variance_pct2": float(acc.var() * 10_000.0),
        "worst20_accuracy_pct": worst * 100.0,
        "best20_accuracy_pct": best * 100.0,
        "best20_worst20_gap_pct": (best - worst) * 100.0,
    }


def evaluate_client_loaders(
    model: torch.nn.Module,
    loaders: list[torch.utils.data.DataLoader],
    device: str,
    *,
    max_batches: int | None = None,
    tail_fraction: float = 0.2,
    client_ids: list[int] | None = None,
    exclude_client_ids: set[int] | None = None,
    client_sample_counts: list[int] | None = None,
) -> dict[str, float | list[float] | list[int]]:
    """Evaluate one common model on client-specific held-out loaders.

    ``exclude_client_ids`` is an oracle *evaluation* mask for Byzantine
    experiments.  It must never be consumed by the training algorithm.  FAR's
    Worst-20/Best-20 metrics are defined over honest clients.
    """

    was_training = model.training
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    accuracies, losses, counts, evaluated_ids = [], [], [], []
    if client_ids is None:
        client_ids = list(range(len(loaders)))
    if len(client_ids) != len(loaders):
        raise ValueError("client_ids must have one entry per loader")
    if client_sample_counts is not None and len(client_sample_counts) != len(loaders):
        raise ValueError("client_sample_counts must have one entry per loader")
    excluded = exclude_client_ids or set()
    with torch.no_grad():
        for position, (client_id, loader) in enumerate(zip(client_ids, loaders)):
            if client_id in excluded:
                continue
            correct = total = 0
            loss_sum = 0.0
            for batch_idx, (x, y) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += float(criterion(logits, y).item())
                correct += int((logits.argmax(1) == y).sum().item())
                total += int(y.numel())
            if total:
                accuracies.append(correct / total)
                losses.append(loss_sum / total)
                evaluated_ids.append(int(client_id))
                counts.append(
                    int(client_sample_counts[position])
                    if client_sample_counts is not None
                    else total
                )
    model.train(was_training)
    result = summarize_client_performance(
        accuracies, losses, tail_fraction, sample_counts=counts
    )
    result["num_evaluated_clients"] = float(len(accuracies))
    result["num_excluded_byzantine_clients"] = float(len(excluded))
    # Oracle evaluation traces enable per-client distribution plots.  They are
    # research diagnostics, not privacy-preserving telemetry for deployment.
    result["evaluated_client_ids_oracle"] = evaluated_ids
    result["client_accuracy_values_oracle"] = [float(x) for x in accuracies]
    result["client_loss_values_oracle"] = [float(x) for x in losses]
    return result
