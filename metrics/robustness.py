"""Oracle diagnostics for Byzantine-robust experiments.

These metrics may use Byzantine labels for evaluation, but algorithms must
never consume them to choose an update.
"""

from __future__ import annotations

import math

import torch


def attack_diagnostics(client_updates) -> dict[str, float | int | str | bool]:
    """Summarise oracle attack labels for experiment reporting only.

    These values must never be consumed by an aggregation algorithm.  They are
    persisted so the dashboard can show which threat was active and how many
    received updates were Byzantine in each round.
    """

    n = len(client_updates)
    if not n:
        return {
            "attack_enabled": False,
            "attack_name": "none",
            "num_byzantine_oracle": 0,
            "byzantine_fraction_oracle": 0.0,
        }
    malicious = [
        bool(metadata.get("is_byzantine", False))
        for _, metadata, _ in client_updates
    ]
    names = sorted(
        {
            str(metadata.get("attack_name", "none"))
            for _, metadata, _ in client_updates
            if str(metadata.get("attack_name", "none")).lower() != "none"
        }
    )
    count = sum(malicious)
    first_metadata = client_updates[0][1]
    return {
        "attack_enabled": bool(count),
        "attack_name": "+".join(names) if names else "none",
        "num_byzantine_oracle": int(count),
        "byzantine_fraction_oracle": float(count / n),
        "attack_schedule_phase": str(
            first_metadata.get("attack_schedule_phase", "attack" if count else "clean")
        ),
        "attack_window_active": bool(
            first_metadata.get("attack_window_active", bool(count))
        ),
        "attack_scheduled_name": str(
            first_metadata.get(
                "attack_scheduled_name", "+".join(names) if names else "none"
            )
        ),
    }


def weight_diagnostics(
    weights: torch.Tensor, malicious_mask: torch.Tensor | None = None
) -> dict[str, float]:
    weights = weights.detach().to(dtype=torch.float64, device="cpu")
    entropy = -(weights.clamp_min(1e-15) * weights.clamp_min(1e-15).log()).sum()
    result = {
        "max_client_weight": float(weights.max().item()),
        "min_client_weight": float(weights.min().item()),
        "weight_entropy": float(entropy.item()),
        "effective_num_clients": float(torch.exp(entropy).item()),
    }
    if malicious_mask is not None:
        mask = malicious_mask.to(dtype=torch.bool, device="cpu")
        result["byzantine_weight_mass_oracle"] = float(weights[mask].sum().item())
    return result


def update_norm_diagnostics(
    client_updates, *, prefix: str = "received_update"
) -> dict[str, float]:
    """Summarise whole-update L2 norms without flattening the cohort.

    The input follows the algorithm contract ``(update, metadata, state)``.
    Computing each norm from tensorwise squared norms avoids allocating a
    second full cohort matrix merely for instrumentation.
    """

    norms: list[float] = []
    for update, _, _ in client_updates:
        squared_norm = 0.0
        for tensor in update.values():
            squared_norm += float(
                tensor.detach().double().square().sum().cpu().item()
            )
        norm = math.sqrt(max(squared_norm, 0.0))
        if not math.isfinite(norm):
            raise ValueError("Client update norm must be finite")
        norms.append(norm)
    if not norms:
        return {}

    values = torch.tensor(norms, dtype=torch.float64)

    def quantile(probability: float) -> float:
        return float(torch.quantile(values, probability).item())

    return {
        f"{prefix}_norm_mean": float(values.mean().item()),
        f"{prefix}_norm_p10": quantile(0.10),
        f"{prefix}_norm_p50": quantile(0.50),
        f"{prefix}_norm_p90": quantile(0.90),
        f"{prefix}_norm_p95": quantile(0.95),
        f"{prefix}_norm_max": float(values.max().item()),
    }
