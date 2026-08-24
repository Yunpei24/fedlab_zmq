"""Shared, explicit helpers for fairness-oriented reference algorithms."""

from __future__ import annotations

from collections import OrderedDict

import torch


def empirical_loss(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    *,
    max_batches: int | None = None,
) -> float:
    """Mean cross-entropy at the model received at the start of a round."""

    was_training = model.training
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            total_loss += float(criterion(model(x), y).item())
            total_examples += int(y.numel())
    model.train(was_training)
    return total_loss / max(total_examples, 1)


def apply_delta(
    global_model: torch.nn.Module, delta: dict[str, torch.Tensor]
) -> OrderedDict:
    """Apply FedLab's ``old - local`` delta convention to a global model."""

    global_sd = global_model.state_dict()
    result = OrderedDict()
    for key, value in global_sd.items():
        if key in delta:
            result[key] = value.float() - delta[key].to(value.device).float()
        else:
            result[key] = value.clone()
    return result


def common_round_metrics(client_updates) -> dict[str, float]:
    n = len(client_updates)
    if not n:
        return {}
    return {
        "total_bytes_sent": sum(m.get("bytes_sent", 0) for _, m, _ in client_updates),
        "total_energy_j": sum(
            m.get("energy_j_consumed", 0.0) for _, m, _ in client_updates
        ),
        "avg_battery_j": sum(state.battery_j for _, _, state in client_updates) / n,
        "avg_local_loss": sum(m.get("local_loss", 0.0) for _, m, _ in client_updates)
        / n,
        "participation_rate": 1.0,
        "jain_index": 1.0,
        "num_clients": n,
        "avg_beta": 1.0,
    }
