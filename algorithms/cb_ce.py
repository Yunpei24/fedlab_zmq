"""FedAvg with a class-balanced cross-entropy local objective.

This is the control the DMD-CB study was missing. ``Margin-Mean`` isolates
"margin, no class balancing"; nothing isolated "class balancing, no margin".
That asymmetry matters, because the reported gap between DMD-CB and
Margin-Mean (+3.90 points of Worst-20 balanced accuracy) says the class
balancing is what carries the effect -- and class-balanced cross-entropy is a
one-line baseline. If it matches DMD-CB on the tail metrics, the margin term
adds nothing and the contribution is a reweighting, not a decision-space
signal.

Weights are inverse local class frequency, renormalised to mean one over the
classes the client actually observes, so the loss keeps the same overall scale
as plain cross-entropy and the learning rate stays comparable across arms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import register_algorithm
from .fedavg import FedAvg


def inverse_frequency_class_weights(
    counts, num_classes: int, *, device: str, dtype=torch.float32
) -> torch.Tensor:
    """Inverse-frequency weights, mean-one over observed classes.

    Unobserved classes get weight zero: they contribute no examples locally, and
    giving them the (infinite) inverse of a zero count would be meaningless.
    """

    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.numel() != num_classes:
        raise ValueError("client_class_counts must contain one value per class")
    observed = counts > 0
    weights = torch.zeros_like(counts)
    if not bool(observed.any()):
        return torch.ones(num_classes, device=device, dtype=dtype)
    weights[observed] = counts[observed].reciprocal()
    weights[observed] = weights[observed] / weights[observed].mean()
    return weights.to(device=device, dtype=dtype)


@register_algorithm("cb_ce")
class ClassBalancedCE(FedAvg):
    """FedAvg whose local loss weights every observed class equally."""

    name = "cb_ce"
    description = "FedAvg with inverse-frequency class-balanced cross-entropy."

    def build_criterion(self, config, device):
        counts = config.get("client_class_counts")
        if counts is None:
            raise ValueError(
                "cb_ce requires per-client class counts; run_experiment.py "
                "injects client_class_counts when num_classes is configured"
            )
        weights = inverse_frequency_class_weights(
            counts, int(config["num_classes"]), device=device
        )
        return nn.CrossEntropyLoss(weight=weights)

    def get_default_config(self):
        config = super().get_default_config()
        config["num_classes"] = None
        return config


__all__ = ["ClassBalancedCE", "inverse_frequency_class_weights"]
