"""Established client-side losses for label-distribution skew.

The baselines CB-CE (algorithms/cb_ce.py) has to be measured against.  Each one
reshapes the local cross-entropy from the client's training-class counts n_y:

- CB-loss (Cui et al., CVPR 2019): class weights (1 - beta) / (1 - beta^n_y).
  beta -> 1 recovers inverse-frequency weighting, i.e. CB-CE; beta = 0 is CE.
- Balanced Softmax (Ren et al., NeurIPS 2020), the tau = 1 case of the logit
  adjustment loss (Menon et al., ICLR 2021): cross-entropy on z + tau * log n.
- FedLC (Zhang et al., ICML 2022): cross-entropy on z - tau * n^(-1/4), the
  softmax form used by the public FL-bench and PFLlib implementations.

A class absent from the client (n_y = 0) never occurs as a local target.  The
weighting losses give it weight zero; the two logit adjustments floor its count
at COUNT_FLOOR, which pushes its logit out of the local softmax, as those public
implementations do.  Weights are renormalised to mean one over the observed
classes, like CB-CE, so every loss keeps the scale of plain cross-entropy and a
single learning rate serves all arms.

LDAM (Cao et al., NeurIPS 2019) is deliberately absent: its margins are defined
on a normalised cosine classifier with a scale, and applying them to raw logits
would be a different method.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import register_algorithm
from .cb_ce import inverse_frequency_class_weights
from .fedavg import FedAvg

LABEL_SKEW_LOSSES = ("ce", "inverse_frequency", "effective_number",
                     "balanced_softmax", "fedlc")
COUNT_FLOOR = 1e-8


def _class_counts(counts, num_classes: int) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.numel() != num_classes:
        raise ValueError("client_class_counts must contain one value per class")
    return counts


def effective_number_class_weights(
    counts, num_classes: int, beta: float, *, device, dtype=torch.float32
) -> torch.Tensor:
    """CB-loss weights (1 - beta) / (1 - beta^n), mean one over observed classes."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0, 1)")
    counts = _class_counts(counts, num_classes)
    observed = counts > 0
    if not bool(observed.any()):
        return torch.ones(num_classes, device=device, dtype=dtype)
    weights = torch.zeros_like(counts)
    effective = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64),
                                 counts[observed])) / (1.0 - beta)
    weights[observed] = effective.reciprocal()
    weights[observed] = weights[observed] / weights[observed].mean()
    return weights.to(device=device, dtype=dtype)


class AdjustedLogitCrossEntropy(nn.Module):
    """Cross-entropy on logits shifted by a fixed per-class offset."""

    def __init__(self, offsets: torch.Tensor):
        super().__init__()
        self.register_buffer("offsets", offsets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.offsets.to(logits.dtype), targets)


def label_skew_criterion(
    kind: str, counts, num_classes: int, *, device, beta: float = 0.999,
    tau: float = 1.0,
) -> nn.Module:
    """The local loss ``kind`` for a client with training-class ``counts``."""

    if kind not in LABEL_SKEW_LOSSES:
        raise ValueError(f"unknown label-skew loss {kind!r}; choose from {LABEL_SKEW_LOSSES}")
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if counts is None:
        raise ValueError(
            f"{kind} needs client_class_counts; run_experiment.py injects them "
            "when num_classes is configured"
        )
    if kind == "inverse_frequency":
        return nn.CrossEntropyLoss(
            weight=inverse_frequency_class_weights(counts, num_classes, device=device)
        )
    if kind == "effective_number":
        return nn.CrossEntropyLoss(
            weight=effective_number_class_weights(counts, num_classes, beta, device=device)
        )
    if tau < 0:
        raise ValueError("tau must be non-negative")
    floored = _class_counts(counts, num_classes).clamp_min(COUNT_FLOOR)
    if kind == "balanced_softmax":
        offsets = tau * torch.log(floored)
    else:
        offsets = -tau * torch.pow(floored, -0.25)
    return AdjustedLogitCrossEntropy(offsets.to(device=device, dtype=torch.float32))


class _LabelSkewFedAvg(FedAvg):
    """FedAvg whose local loss is one of LABEL_SKEW_LOSSES."""

    loss_kind = "ce"

    def build_criterion(self, config, device):
        return label_skew_criterion(
            self.loss_kind,
            config.get("client_class_counts"),
            int(config["num_classes"]),
            device=device,
            beta=float(config.get("label_skew_beta", 0.999)),
            tau=float(config.get("label_skew_tau", 1.0)),
        )

    def get_default_config(self):
        config = super().get_default_config()
        config["num_classes"] = None
        return config


@register_algorithm("cb_loss")
class ClassBalancedLoss(_LabelSkewFedAvg):
    name = "cb_loss"
    description = "FedAvg with the effective-number class-balanced loss (Cui et al., 2019)."
    loss_kind = "effective_number"


@register_algorithm("balanced_softmax")
class BalancedSoftmax(_LabelSkewFedAvg):
    name = "balanced_softmax"
    description = "FedAvg with Balanced Softmax / logit adjustment on the local prior."
    loss_kind = "balanced_softmax"


@register_algorithm("fedlc")
class FedLC(_LabelSkewFedAvg):
    name = "fedlc"
    description = "FedLC: logits calibrated by tau * n^(-1/4) (Zhang et al., 2022)."
    loss_kind = "fedlc"


__all__ = [
    "LABEL_SKEW_LOSSES",
    "COUNT_FLOOR",
    "effective_number_class_weights",
    "AdjustedLogitCrossEntropy",
    "label_skew_criterion",
    "ClassBalancedLoss",
    "BalancedSoftmax",
    "FedLC",
]
