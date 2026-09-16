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

Sampling is independent of the gradient backend. ``fixed_minibatch`` retains
the legacy DataLoader behaviour. ``poisson`` performs a genuine Bernoulli
inclusion trial for every local record at every DP step, which is the primary
add/remove-DP lane for DT-LDP-FAR. ``fixed_without_replacement`` draws an
independent uniformly random subset of exactly ``m`` records at each step and
is accounted under public-size replace-one adjacency.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.utils.data import default_collate


@dataclass
class DPSGDStats:
    """Diagnostics produced by one local DP-SGD training invocation."""

    steps: int = 0
    examples: int = 0
    clipped_examples: int = 0
    loss_sum: float = 0.0
    noise_norm_sum: float = 0.0
    sampling_scheme: str = "fixed_minibatch"
    sampling_rate: float | None = None
    expected_batch_size: float | None = None
    normalization_denominator: float | None = None
    empty_steps: int = 0
    # Simulation-only counterfactual.  It is never part of the claimed DP
    # transcript and is populated only when explicitly requested.
    noise_free_delta_oracle: dict[str, torch.Tensor] | None = field(
        default=None, repr=False
    )

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
    return any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )


def _diagnostic_dtype(device: str | torch.device) -> torch.dtype:
    """Use the most accurate accumulation dtype supported by the backend.

    Apple MPS has no float64 kernels.  CPU and CUDA retain the previous
    float64 diagnostic accumulation, while MPS uses float32.  This helper is
    only used for clipping/noise norms and scalar releases; it does not alter
    the model parameter dtype or the DP mechanism.
    """

    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def _capture_rng_state(device: str | torch.device) -> dict[str, torch.Tensor]:
    """Capture RNG state so a diagnostic shadow can reuse stochastic masks."""

    states = {"cpu": torch.get_rng_state()}
    device_type = torch.device(device).type
    if device_type == "cuda" and torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state(torch.device(device))
    if device_type == "mps" and hasattr(torch.mps, "get_rng_state"):
        states["mps"] = torch.mps.get_rng_state()
    return states


def _restore_rng_state(
    states: dict[str, torch.Tensor], device: str | torch.device
) -> None:
    torch.set_rng_state(states["cpu"])
    device_type = torch.device(device).type
    if device_type == "cuda" and "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"], torch.device(device))
    if device_type == "mps" and "mps" in states:
        torch.mps.set_rng_state(states["mps"])


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
    sampling_scheme: str = "fixed_minibatch",
    poisson_sampling_rate: float | None = None,
    poisson_steps_per_round: int | None = None,
    poisson_normalization_denominator: float | None = None,
    fixed_batch_size: int | None = None,
    fixed_steps_per_round: int | None = None,
    fixed_normalization_denominator: float | None = None,
    track_noise_free_counterfactual: bool = False,
) -> tuple[dict[str, torch.Tensor], DPSGDStats]:
    """Train locally with per-example clipping and return ``old - new``.

    Gaussian noise is added to the *sum* of clipped gradients with standard
    deviation ``noise_multiplier * clip_norm``. Fixed batches divide by their
    realised size. Genuine Poisson steps divide by the public expected batch
    size ``q * N_public``; this keeps the normalization independent of the
    sampled subset and also defines the empty-sample step unambiguously. A
    fixed-size without-replacement step divides by its public fixed batch size.
    """

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier cannot be negative")
    if proximal_mu < 0:
        raise ValueError("proximal_mu cannot be negative")
    if backend not in {"vectorized", "loop"}:
        raise ValueError("backend must be 'vectorized' or 'loop'")
    sampling_scheme = str(sampling_scheme).lower()
    supported_schemes = {
        "fixed_minibatch",
        "poisson",
        "fixed_without_replacement",
    }
    if sampling_scheme not in supported_schemes:
        raise ValueError(
            "sampling_scheme must be fixed_minibatch, poisson or "
            "fixed_without_replacement"
        )
    dataset_size = max(len(dataloader.dataset), 1)
    if sampling_scheme == "poisson":
        if poisson_sampling_rate is None:
            raise ValueError("poisson_sampling_rate is required for Poisson sampling")
        poisson_sampling_rate = float(poisson_sampling_rate)
        if not 0.0 < poisson_sampling_rate <= 1.0:
            raise ValueError("poisson_sampling_rate must lie in (0,1]")
        poisson_steps = int(
            poisson_steps_per_round
            if poisson_steps_per_round is not None
            else local_epochs
        )
        if poisson_steps < 1:
            raise ValueError("poisson_steps_per_round must be at least one")
        if max_local_batches is not None:
            poisson_steps = min(poisson_steps, int(max_local_batches))
        if poisson_steps < 1:
            raise ValueError("max_local_batches removed every Poisson step")
        if poisson_normalization_denominator is None:
            poisson_normalization_denominator = poisson_sampling_rate * dataset_size
        poisson_normalization_denominator = float(poisson_normalization_denominator)
        if poisson_normalization_denominator <= 0.0:
            raise ValueError("poisson_normalization_denominator must be positive")
        fixed_steps = 0
    elif sampling_scheme == "fixed_without_replacement":
        if fixed_batch_size is None:
            raise ValueError("fixed_batch_size is required for fixed-size sampling")
        fixed_batch_size = int(fixed_batch_size)
        if not 1 <= fixed_batch_size <= dataset_size:
            raise ValueError("fixed_batch_size must lie in [1, dataset_size]")
        fixed_steps = int(
            fixed_steps_per_round if fixed_steps_per_round is not None else local_epochs
        )
        if fixed_steps < 1:
            raise ValueError("fixed_steps_per_round must be at least one")
        if max_local_batches is not None:
            fixed_steps = min(fixed_steps, int(max_local_batches))
        if fixed_steps < 1:
            raise ValueError("max_local_batches removed every fixed-size step")
        if fixed_normalization_denominator is None:
            fixed_normalization_denominator = float(fixed_batch_size)
        fixed_normalization_denominator = float(fixed_normalization_denominator)
        if not math.isclose(
            fixed_normalization_denominator,
            float(fixed_batch_size),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "fixed-size sampling must normalize by the public batch size"
            )
        poisson_steps = 0
    else:
        poisson_steps = 0
        fixed_steps = 0

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
    clean_model = None
    clean_parameters: list[nn.Parameter] = []
    clean_names: list[str] = []
    clean_proximal_anchor: dict[str, torch.Tensor] | None = None
    clean_optimizer = None
    if track_noise_free_counterfactual:
        # The shadow follows the same samples and stochastic model
        # masks but omits Gaussian perturbations.  It isolates the effective
        # fresh perturbation of the complete local optimisation trajectory.
        clean_model = copy.deepcopy(model).to(device)
        clean_model.train()
        clean_named_parameters = [
            (name, parameter)
            for name, parameter in clean_model.named_parameters()
            if parameter.requires_grad
        ]
        clean_names = [name for name, _ in clean_named_parameters]
        clean_parameters = [parameter for _, parameter in clean_named_parameters]
        if clean_names != names:
            raise ValueError("Noise-free shadow parameter layout mismatch")
        clean_proximal_anchor = {
            name: parameter.detach().clone()
            for name, parameter in clean_named_parameters
        }
        clean_optimizer = torch.optim.SGD(
            clean_parameters,
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
    stats = DPSGDStats(
        sampling_scheme=sampling_scheme,
        sampling_rate=(
            float(poisson_sampling_rate)
            if sampling_scheme == "poisson"
            else (
                float(fixed_batch_size) / dataset_size
                if sampling_scheme == "fixed_without_replacement"
                else None
            )
        ),
        expected_batch_size=(
            float(poisson_sampling_rate) * dataset_size
            if sampling_scheme == "poisson"
            else (
                float(fixed_batch_size)
                if sampling_scheme == "fixed_without_replacement"
                else None
            )
        ),
        normalization_denominator=(
            float(poisson_normalization_denominator)
            if sampling_scheme == "poisson"
            else (
                float(fixed_normalization_denominator)
                if sampling_scheme == "fixed_without_replacement"
                else None
            )
        ),
    )

    if sampling_scheme == "poisson":
        batches = []
        for _ in range(poisson_steps):
            selected = (
                torch.nonzero(
                    torch.rand(dataset_size) < float(poisson_sampling_rate),
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            if selected:
                batches.append(
                    default_collate([dataloader.dataset[index] for index in selected])
                )
            else:
                batches.append(None)
    elif sampling_scheme == "fixed_without_replacement":
        batches = []
        for _ in range(fixed_steps):
            selected = torch.randperm(dataset_size)[: int(fixed_batch_size)].tolist()
            batches.append(
                default_collate([dataloader.dataset[index] for index in selected])
            )
    else:
        batches = []
        for _ in range(int(local_epochs)):
            for batch_idx, batch in enumerate(dataloader):
                if max_local_batches is not None and batch_idx >= int(
                    max_local_batches
                ):
                    break
                batches.append(batch)

    for batch in batches:
        empty_sample_step = batch is None
        if empty_sample_step:
            batch_examples = 0
            per_sample = [
                torch.empty(
                    (0,) + tuple(parameter.shape),
                    dtype=parameter.dtype,
                    device=device,
                )
                for parameter in parameters
            ]
            losses = torch.empty(0, device=device)
            clean_per_sample = [tensor.clone() for tensor in per_sample]
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            batch_examples = int(x.shape[0])
            rng_before = (
                _capture_rng_state(device) if track_noise_free_counterfactual else None
            )
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
            if track_noise_free_counterfactual:
                if clean_model is None or rng_before is None:
                    raise RuntimeError("Noise-free shadow was not initialized")
                rng_after_noisy = _capture_rng_state(device)
                _restore_rng_state(rng_before, device)
                if backend == "vectorized":
                    clean_per_sample, _ = _per_sample_grads_vectorized(
                        clean_model,
                        clean_names,
                        x,
                        y,
                        proximal_mu=float(proximal_mu),
                        proximal_anchor=clean_proximal_anchor,
                    )
                else:
                    clean_per_sample, _ = _per_sample_grads_loop(
                        clean_model,
                        clean_parameters,
                        x,
                        y,
                        proximal_mu=float(proximal_mu),
                        proximal_anchor=clean_proximal_anchor,
                    )
                # The diagnostic branch must not perturb subsequent Poisson,
                # DP-noise or model-randomness draws.
                _restore_rng_state(rng_after_noisy, device)
            else:
                clean_per_sample = []

        diagnostic_dtype = _diagnostic_dtype(device)
        norm_sq = torch.zeros(batch_examples, dtype=diagnostic_dtype, device=device)
        if batch_examples:
            for sample_grad in per_sample:
                norm_sq += (
                    sample_grad.reshape(batch_examples, -1)
                    .to(dtype=diagnostic_dtype)
                    .square()
                    .sum(1)
                )
        norms = norm_sq.sqrt()
        factors = (float(clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)

        clean_factors = None
        if track_noise_free_counterfactual:
            clean_norm_sq = torch.zeros(
                batch_examples, dtype=diagnostic_dtype, device=device
            )
            if batch_examples:
                for sample_grad in clean_per_sample:
                    clean_norm_sq += (
                        sample_grad.reshape(batch_examples, -1)
                        .to(dtype=diagnostic_dtype)
                        .square()
                        .sum(1)
                    )
            clean_factors = (
                float(clip_norm) / clean_norm_sq.sqrt().clamp_min(1e-12)
            ).clamp(max=1.0)

        optimizer.zero_grad(set_to_none=True)
        if clean_optimizer is not None:
            clean_optimizer.zero_grad(set_to_none=True)
        step_noise_norm_sq = 0.0
        for index, (parameter, sample_grad) in enumerate(zip(parameters, per_sample)):
            view_shape = (batch_examples,) + (1,) * (sample_grad.ndim - 1)
            clipped_sum = (
                sample_grad * factors.to(sample_grad.dtype).view(view_shape)
            ).sum(0)
            if noise_multiplier > 0:
                noise = torch.randn_like(clipped_sum) * (
                    float(noise_multiplier) * float(clip_norm)
                )
                clipped_sum = clipped_sum + noise
                step_noise_norm_sq += float(
                    noise.to(dtype=diagnostic_dtype).square().sum().item()
                )
            if sampling_scheme == "poisson":
                denominator = float(poisson_normalization_denominator)
            elif sampling_scheme == "fixed_without_replacement":
                denominator = float(fixed_normalization_denominator)
            else:
                denominator = max(batch_examples, 1)
            parameter.grad = clipped_sum / denominator
            if clean_optimizer is not None and clean_factors is not None:
                clean_sample_grad = clean_per_sample[index]
                clean_view_shape = (batch_examples,) + (1,) * (
                    clean_sample_grad.ndim - 1
                )
                clean_clipped_sum = (
                    clean_sample_grad
                    * clean_factors.to(clean_sample_grad.dtype).view(clean_view_shape)
                ).sum(0)
                clean_parameters[index].grad = clean_clipped_sum / denominator
        optimizer.step()
        if clean_optimizer is not None:
            clean_optimizer.step()

        stats.steps += 1
        stats.examples += batch_examples
        stats.clipped_examples += int((factors < 1.0).sum().item())
        stats.loss_sum += float(losses.sum().item())
        stats.noise_norm_sum += step_noise_norm_sq**0.5
        stats.empty_steps += int(empty_sample_step)

    current = model.state_dict()
    delta = OrderedDict(
        (key, (before[key] - current[key].detach().cpu()).float()) for key in before
    )
    if clean_model is not None:
        clean_current = clean_model.state_dict()
        stats.noise_free_delta_oracle = dict(
            OrderedDict(
                (
                    key,
                    (before[key] - clean_current[key].detach().cpu()).float(),
                )
                for key in before
            )
        )
    return dict(delta), stats


def private_gradient_release_fixed_without_replacement(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str,
    batch_size: int,
    clip_norm: float,
    noise_multiplier: float,
    backend: str = "vectorized",
    return_noise_free_oracle: bool = False,
) -> tuple[dict[str, torch.Tensor], DPSGDStats]:
    r"""Release one FedFDP-style private empirical gradient.

    A uniformly random subset ``B`` of exactly ``batch_size`` records is drawn
    without replacement.  For every selected record, the complete model
    gradient is clipped *before* the Gaussian perturbation.  The released
    gradient is

    .. math::

        Y = \frac{1}{|B|}\left(
            \sum_{z\in B}\operatorname{Clip}_{C}(\nabla\ell(w;z))
            + \sigma C Z
        \right),\qquad Z\sim\mathcal N(0,I).

    The division applies to both the clipped sum and the noise, exactly as in
    the gradient channel of FedFDP.  Unlike :func:`local_dpsgd_train`, this
    primitive performs no optimizer step and does not mutate ``model``: the
    dictionary returned to the server is a private gradient, not a model
    delta.  Fixed cardinality makes an empty batch impossible.

    Under fixed-size replace-one adjacency, the average clipped-gradient query
    has L2 sensitivity at most ``2 * clip_norm / batch_size``.  The companion
    accountant therefore interprets the implementation multiplier relative to
    ``2C`` (equivalently it accounts with ``noise_multiplier / 2``).
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier cannot be negative")
    if backend not in {"vectorized", "loop"}:
        raise ValueError("backend must be 'vectorized' or 'loop'")
    dataset_size = len(dataloader.dataset)
    if batch_size > dataset_size:
        raise ValueError("batch_size cannot exceed the local dataset size")

    model.to(device)
    model.train()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise ValueError("the model has no trainable parameters")

    selected = torch.randperm(dataset_size)[: int(batch_size)].tolist()
    x, y = default_collate([dataloader.dataset[index] for index in selected])
    x, y = x.to(device), y.to(device)
    if backend == "vectorized":
        per_sample, losses = _per_sample_grads_vectorized(model, names, x, y)
    else:
        per_sample, losses = _per_sample_grads_loop(model, parameters, x, y)

    diagnostic_dtype = _diagnostic_dtype(device)
    norm_sq = torch.zeros(batch_size, dtype=diagnostic_dtype, device=device)
    for sample_grad in per_sample:
        norm_sq += (
            sample_grad.reshape(batch_size, -1)
            .to(dtype=diagnostic_dtype)
            .square()
            .sum(dim=1)
        )
    norms = norm_sq.sqrt()
    factors = (float(clip_norm) / norms.clamp_min(1e-12)).clamp(max=1.0)

    released_parameters: dict[str, torch.Tensor] = {}
    noise_free_parameters: dict[str, torch.Tensor] = {}
    noise_norm_sq = torch.zeros((), dtype=diagnostic_dtype, device=device)
    for name, sample_grad in zip(names, per_sample):
        view_shape = (batch_size,) + (1,) * (sample_grad.ndim - 1)
        clipped_sum = (
            sample_grad * factors.to(sample_grad.dtype).view(view_shape)
        ).sum(dim=0)
        if return_noise_free_oracle:
            # Simulation-only counterfactual on exactly the same sampled
            # records and clipping factors. It is never a protocol output.
            noise_free_parameters[name] = (
                clipped_sum / float(batch_size)
            ).detach().cpu().float()
        if noise_multiplier > 0.0:
            noise = torch.randn_like(clipped_sum) * (
                float(noise_multiplier) * float(clip_norm)
            )
            clipped_sum = clipped_sum + noise
            noise_norm_sq += noise.to(dtype=diagnostic_dtype).square().sum()
        released_parameters[name] = (
            clipped_sum / float(batch_size)
        ).detach().cpu().float()

    # FAR's tensor dispatcher expects a complete state-dict-shaped mapping.
    # Non-trainable buffers carry no gradient and are represented by zeros.
    release: OrderedDict[str, torch.Tensor] = OrderedDict()
    noise_free_release: OrderedDict[str, torch.Tensor] | None = (
        OrderedDict() if return_noise_free_oracle else None
    )
    for name, value in model.state_dict().items():
        release[name] = released_parameters.get(
            name, torch.zeros_like(value, device="cpu").float()
        )
        if noise_free_release is not None:
            noise_free_release[name] = noise_free_parameters.get(
                name, torch.zeros_like(value, device="cpu").float()
            )

    stats = DPSGDStats(
        steps=1,
        examples=int(batch_size),
        clipped_examples=int((factors < 1.0).sum().item()),
        loss_sum=float(losses.sum().item()),
        noise_norm_sum=float(noise_norm_sq.sqrt().item()),
        sampling_scheme="fixed_without_replacement",
        sampling_rate=float(batch_size) / float(dataset_size),
        expected_batch_size=float(batch_size),
        normalization_denominator=float(batch_size),
        empty_steps=0,
        noise_free_delta_oracle=(
            dict(noise_free_release) if noise_free_release is not None else None
        ),
    )
    return dict(release), stats


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
    diagnostic_dtype = _diagnostic_dtype(device)
    clipped = torch.tensor(values, dtype=diagnostic_dtype, device=device).clamp(
        0.0, clip
    )
    noise = torch.randn((), dtype=diagnostic_dtype, device=device) * (
        float(noise_multiplier) * float(clip)
    )
    return float(((clipped.sum() + noise) / len(values)).item())
