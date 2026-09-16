"""Isolated MPS primitives for the CE versus private DMD-CB experiment.

The histogram is a *single* pure-DP release, reused as fixed context for every
per-example gradient.  Conditional on that release, changing one example
changes one clipped contribution, hence replace-one sensitivity 2C.

The experimental random streams are reproducible, not cryptographic.  The
accountant describes the ideal mechanism with independent secret randomness;
publishing simulation seeds is not a production privacy guarantee.  As in the
existing Gaussian implementation, finite-precision sampling is not a hardened
implementation of mathematical differential privacy.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import math
import os

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from torch.utils.data import Subset, default_collate


def require_private_mps(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise ValueError("DMD-CB private computations require available local MPS")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("DMD-CB requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    return device


def simulation_seed(
    pairing_seed: int, client_id: int, round_num: int, domain: str
) -> int:
    """Public experiment identity, deliberately independent of method/loss."""
    raw = f"dmd-cb-v1:{int(pairing_seed)}:{int(client_id)}:{int(round_num)}:{domain}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big") % (
        2**63 - 1
    )


@contextmanager
def paired_mps_randomness(seed: int):
    """Restore global CPU/MPS streams even when an operation fails.

    Each experiment runs in its own process. This context is not thread-safe
    and should not wrap concurrently executed clients within one process.
    """
    require_private_mps("mps")
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        torch.mps.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        torch.mps.set_rng_state(mps_state)


def _dataset_labels(dataset) -> list[int]:
    # Reading labels is private local dataset I/O, not a released statistic.
    if isinstance(dataset, Subset):
        parent_labels = _dataset_labels(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    if hasattr(dataset, "tensors") and len(dataset.tensors) == 2:
        return [int(value) for value in dataset.tensors[1]]
    return [int(dataset[index][1]) for index in range(len(dataset))]


def private_class_weights(
    dataset,
    *,
    num_classes: int,
    public_dataset_size: int,
    epsilon: float,
    weight_cap: float,
    seed: int,
    device: str = "mps",
) -> torch.Tensor:
    """One replace-one Laplace histogram release, followed by postprocessing.

    Only the DP weights are returned. Raw counts, labels and random variates
    must never be stored in a transcript or in ClientState (which may itself
    be transferred by the simulator). Re-sending these *same* DP weights does
    not spend an additional budget; drawing a new histogram would.
    """
    target = require_private_mps(device)
    if num_classes < 2 or public_dataset_size != len(dataset):
        raise ValueError("Public class universe and fixed local cardinality required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Histogram epsilon must be finite and positive")
    if not math.isfinite(weight_cap) or weight_cap <= 0:
        raise ValueError("Class-weight cap must be finite and positive")
    with paired_mps_randomness(seed), torch.no_grad():
        labels = _dataset_labels(dataset)
        if len(labels) != public_dataset_size or any(
            y < 0 or y >= num_classes for y in labels
        ):
            raise ValueError("Labels must belong to the fixed public class universe")
        labels_mps = torch.tensor(labels, dtype=torch.long, device=target)
        classes = torch.arange(num_classes, device=target)
        counts = (labels_mps[:, None] == classes[None, :]).float().sum(dim=0)
        # Inverse CDF of Laplace(0, 2/epsilon); finite-precision research RNG.
        unit = torch.rand(num_classes, device=target)
        unit = unit.clamp(
            min=torch.finfo(torch.float32).eps, max=1.0 - torch.finfo(torch.float32).eps
        )
        centered = unit - 0.5
        laplace = (
            -(2.0 / epsilon) * centered.sign() * torch.log1p(-2.0 * centered.abs())
        )
        noisy_counts = (counts + laplace).clamp_min(1.0)
        weights = (public_dataset_size / (num_classes * noisy_counts)).clamp(
            max=weight_cap
        )
        return weights.detach()


def dmd_cb_losses(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, mu: float
) -> torch.Tensor:
    """Separable loss; fixed DP class weights, no batch normalization of weights."""
    if logits.ndim != 2 or logits.shape[1] != class_weights.numel():
        raise ValueError("Model outputs must match the public class universe")
    ce = F.cross_entropy(logits, labels, reduction="none")
    if mu == 0.0:
        return ce  # Exact CE computational path for the matched control.
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    mask = (
        torch.arange(logits.shape[1], device=logits.device)[None, :] == labels[:, None]
    )
    competitor = logits.masked_fill(mask, -torch.inf).max(dim=1).values
    deficit = torch.relu(competitor - correct)
    selected_weights = class_weights.gather(0, labels).detach()
    return ce + float(mu) * selected_weights * 0.5 * deficit.square()


def per_example_dmd_gradients(model, x, y, class_weights, mu):
    """All private differentiable operations execute on MPS, one-example loss."""
    require_private_mps(x.device)
    if any(parameter.device.type != "mps" for parameter in model.parameters()):
        raise ValueError("Model must already reside on MPS")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("This MPS implementation fixes private parameters to float32")
    if any(
        isinstance(
            layer, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)
        )
        for layer in model.modules()
    ):
        raise ValueError("DMD-CB v1 uses deterministic, batch-independent models")
    params = OrderedDict(model.named_parameters())
    buffers = OrderedDict(model.named_buffers())

    def loss_one(parameters, fixed_buffers, one_x, one_y):
        logits = functional_call(
            model, (parameters, fixed_buffers), (one_x.unsqueeze(0),)
        )
        return dmd_cb_losses(logits, one_y.unsqueeze(0), class_weights, mu).sum()

    derivative = grad(loss_one)
    gradients = vmap(derivative, in_dims=(None, None, 0, 0), randomness="error")(
        params, buffers, x, y
    )
    return OrderedDict(
        (key, value) for key, value in gradients.items() if params[key].requires_grad
    )


def clip_combined_gradients(gradients: dict[str, torch.Tensor], clip_norm: float):
    """Clip concatenated CE+DMD per-example gradients, never the components."""
    if not gradients or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError(
            "Nonempty gradients and a positive finite clipping radius required"
        )
    first = next(iter(gradients.values()))
    require_private_mps(first.device)
    norms_squared = torch.zeros(first.shape[0], device=first.device)
    for value in gradients.values():
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite private per-example gradient")
        norms_squared += value.reshape(value.shape[0], -1).square().sum(dim=1)
    factors = (clip_norm / norms_squared.sqrt().clamp_min(1e-12)).clamp(max=1.0)
    return OrderedDict(
        (key, value * factors.reshape((-1,) + (1,) * (value.ndim - 1)))
        for key, value in gradients.items()
    )


def private_dmd_gradient_release(
    model,
    dataloader,
    *,
    class_weights: torch.Tensor,
    mu: float,
    batch_size: int,
    clip_norm: float,
    noise_multiplier: float,
    batch_seed: int,
    gaussian_seed: int,
    device: str = "mps",
) -> OrderedDict:
    """Sample one fresh fixed-WOR batch; send (sum Clip_C(g_j)+sigma*C*Z)/B.

    No local model step, no post-noise client clipping, and no raw diagnostic
    accompanies this private release. The transport copy is CPU float32.
    """
    target = require_private_mps(device)
    if not (1 <= batch_size <= len(dataloader.dataset)):
        raise ValueError("Batch size must be between one and public cardinality")
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("DMD mu must be finite and nonnegative")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("The private DMD-CB lane requires positive Gaussian noise")
    if (
        class_weights.ndim != 1
        or not torch.isfinite(class_weights).all()
        or not (class_weights > 0).all()
    ):
        raise ValueError("Fixed class weights must be finite and positive")
    model.to(target)
    model.train()
    with paired_mps_randomness(batch_seed):
        selected = (
            torch.randperm(len(dataloader.dataset), device=target)[:batch_size]
            .cpu()
            .tolist()
        )
        x, y = default_collate([dataloader.dataset[index] for index in selected])
        x, y = x.to(target), y.to(target)
        weights = class_weights.detach().to(target)
        gradients = per_example_dmd_gradients(model, x, y, weights, mu)
        clipped = clip_combined_gradients(gradients, clip_norm)
    released = OrderedDict()
    with paired_mps_randomness(gaussian_seed), torch.no_grad():
        for key, value in model.state_dict().items():
            if key in clipped:
                clipped_sum = clipped[key].sum(dim=0)
                noisy_sum = clipped_sum + torch.randn_like(clipped_sum) * (
                    noise_multiplier * clip_norm
                )
                averaged = noisy_sum / batch_size
                if not torch.isfinite(averaged).all():
                    raise FloatingPointError("Nonfinite private released gradient")
                released[key] = averaged.detach().cpu().float()
            else:
                # Only trainable gradients change the model; floating buffers
                # are carried as zero deltas just like the parent mechanism.
                released[key] = torch.zeros_like(value, device="cpu")
    return released
