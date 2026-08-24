"""Reusable local DP-SGD primitives for the reference experiments.

The internship reproduction applies the same per-example clipping and
Gaussian mechanism to FedAvg, q-FFL and FAR.  Keeping that operation in one
module prevents the three baselines from silently using different privacy
mechanisms.

The implementation exposes two backends:

``vectorized``
    Uses :mod:`torch.func` to compute one gradient per example.  This is the
    default for experiments.

``loop``
    Computes examples one by one.  It is slow but deliberately simple and is
    useful as a parity oracle in tests.

Neither backend changes the sampling performed by the DataLoader.  Therefore
the RDP accountant's sampled-Gaussian statement is labelled a Poisson
approximation when ordinary shuffled mini-batches are used.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


@dataclass
class DPSGDStats:
    """Diagnostics produced by one local DP-SGD training invocation."""

    steps: int = 0
    examples: int = 0
    clipped_examples: int = 0
    loss_sum: float = 0.0
    noise_norm_sum: float = 0.0

    @property
    def clip_rate(self) -> float:
        return self.clipped_examples / max(self.examples, 1)

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(self.examples, 1)

    @property
    def mean_noise_norm(self) -> float:
        return self.noise_norm_sum / max(self.steps, 1)


def _has_batch_norm(model: nn.Module) -> bool:
    return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


def _per_sample_grads_loop(
    model: nn.Module,
    parameters: list[nn.Parameter],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    proximal_mu: float = 0.0,
    proximal_anchor: dict[str, torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Transparent per-example gradients used as the reference backend."""

    by_parameter: list[list[torch.Tensor]] = [[] for _ in parameters]
    losses = []
    for sample_x, sample_y in zip(x, y):
        logits = model(sample_x.unsqueeze(0))
        task_loss = F.cross_entropy(logits, sample_y.unsqueeze(0))
        prox = torch.zeros((), device=task_loss.device)
        if proximal_mu > 0 and proximal_anchor is not None:
            for name, parameter in model.named_parameters():
                prox = prox + (parameter - proximal_anchor[name]).square().sum()
        loss = task_loss + 0.5 * float(proximal_mu) * prox
        sample_grads = torch.autograd.grad(loss, parameters, retain_graph=False)
        for bucket, sample_grad in zip(by_parameter, sample_grads):
            bucket.append(sample_grad.detach())
        losses.append(loss.detach())
    return [torch.stack(bucket, dim=0) for bucket in by_parameter], torch.stack(losses)


def _per_sample_grads_vectorized(
    model: nn.Module,
    parameter_names: list[str],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    proximal_mu: float = 0.0,
    proximal_anchor: dict[str, torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Vectorised per-example gradients using ``torch.func.vmap``."""

    if _has_batch_norm(model):
        raise ValueError(
            "The vectorized DP-SGD backend does not support training BatchNorm "
            "with per-example batches. Use GroupNorm/LayerNorm or backend='loop'."
        )
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())

    def loss_one(current_params, current_buffers, sample_x, sample_y):
        logits = functional_call(
            model,
            (current_params, current_buffers),
            (sample_x.unsqueeze(0),),
        )
        task_loss = F.cross_entropy(logits, sample_y.unsqueeze(0))
        if proximal_mu <= 0 or proximal_anchor is None:
            return task_loss
        prox = torch.zeros((), device=task_loss.device)
        for name, parameter in current_params.items():
            prox = prox + (parameter - proximal_anchor[name]).square().sum()
        return task_loss + 0.5 * float(proximal_mu) * prox

    grad_fn = grad(loss_one)
    gradients = vmap(
        grad_fn,
        in_dims=(None, None, 0, 0),
        randomness="different",
    )(params, buffers, x, y)
    with torch.no_grad():
        losses = vmap(
            loss_one,
            in_dims=(None, None, 0, 0),
            randomness="different",
        )(params, buffers, x, y)
    return [gradients[name].detach() for name in parameter_names], losses.detach()


def local_dpsgd_train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str,
    lr: float,
    local_epochs: int,
    clip_norm: float,
    noise_multiplier: float,
    backend: str = "vectorized",
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    max_local_batches: int | None = None,
    proximal_mu: float = 0.0,
) -> tuple[dict[str, torch.Tensor], DPSGDStats]:
    """Train locally with per-example clipping and return ``old - new``.

    Gaussian noise is added to the *sum* of clipped gradients with standard
    deviation ``noise_multiplier * clip_norm`` and the result is divided by
    the realised batch size, matching the DP-SGD convention used by FedFDP.
    """

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative")
    if proximal_mu < 0:
        raise ValueError("proximal_mu cannot be negative")
    if backend not in {"vectorized", "loop"}:
        raise ValueError("backend must be 'vectorized' or 'loop'")

    model.to(device)
    model.train()
    before = OrderedDict(
        (key, value.detach().cpu().clone()) for key, value in model.state_dict().items()
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    proximal_anchor = {
        name: parameter.detach().clone() for name, parameter in named_parameters
    }
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    stats = DPSGDStats()

    for _ in range(int(local_epochs)):
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_local_batches is not None and batch_idx >= int(max_local_batches):
                break
            x, y = x.to(device), y.to(device)
            if backend == "vectorized":
                per_sample, losses = _per_sample_grads_vectorized(
                    model,
                    names,
                    x,
                    y,
                    proximal_mu=float(proximal_mu),
                    proximal_anchor=proximal_anchor,
                )
            else:
                per_sample, losses = _per_sample_grads_loop(
                    model,
                    parameters,
                    x,
                    y,
                    proximal_mu=float(proximal_mu),
                    proximal_anchor=proximal_anchor,
                )

            norm_sq = torch.zeros(x.shape[0], dtype=torch.float64, device=device)
            for sample_grad in per_sample:
                norm_sq += sample_grad.reshape(x.shape[0], -1).double().square().sum(1)
            norms = norm_sq.sqrt()
            factors = (float(clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)

            optimizer.zero_grad(set_to_none=True)
            step_noise_norm_sq = 0.0
            for parameter, sample_grad in zip(parameters, per_sample):
                view_shape = (x.shape[0],) + (1,) * (sample_grad.ndim - 1)
                clipped_sum = (sample_grad * factors.to(sample_grad.dtype).view(view_shape)).sum(0)
                if noise_multiplier > 0:
                    noise = torch.randn_like(clipped_sum) * (
                        float(noise_multiplier) * float(clip_norm)
                    )
                    clipped_sum = clipped_sum + noise
                    step_noise_norm_sq += float(noise.double().square().sum().item())
                parameter.grad = clipped_sum / max(int(x.shape[0]), 1)
            optimizer.step()

            stats.steps += 1
            stats.examples += int(x.shape[0])
            stats.clipped_examples += int((factors < 1.0).sum().item())
            stats.loss_sum += float(losses.sum().item())
            stats.noise_norm_sum += step_noise_norm_sq**0.5

    current = model.state_dict()
    delta = OrderedDict(
        (key, (before[key] - current[key].detach().cpu()).float()) for key in before
    )
    return dict(delta), stats


def private_mean_release(
    values: Iterable[float],
    *,
    clip: float,
    noise_multiplier: float,
    device: str = "cpu",
) -> float:
    """Release a clipped scalar mean through a Gaussian mechanism."""

    values = list(values)
    if not values:
        return 0.0
    if clip <= 0:
        raise ValueError("clip must be positive")
    clipped = torch.tensor(values, dtype=torch.float64, device=device).clamp(0.0, clip)
    noise = torch.randn((), dtype=torch.float64, device=device) * (
        float(noise_multiplier) * float(clip)
    )
    return float(((clipped.sum() + noise) / len(values)).item())
