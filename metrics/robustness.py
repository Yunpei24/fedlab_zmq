"""Oracle diagnostics for Byzantine-robust experiments.

These metrics may use Byzantine labels for evaluation, but algorithms must
never consume them to choose an update.
"""

from __future__ import annotations

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
    return {
        "attack_enabled": bool(count),
        "attack_name": "+".join(names) if names else "none",
        "num_byzantine_oracle": int(count),
        "byzantine_fraction_oracle": float(count / n),
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
