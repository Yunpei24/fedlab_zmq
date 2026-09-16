"""Public, architecture-only parameter masks for partial-model training.

The selectors in this module depend only on the public model architecture and
on a predeclared configuration.  They never inspect client updates, gradients,
losses, or private dataset sizes.  This distinction matters for experiments
where only the selected coordinates are allowed to depend on client data and
are subsequently protected by a central-DP mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

SUPPORTED_ACTIVE_PARAMETER_MODES = {
    "full",
    "last_layer",
    "classifier_head",
    "classifier_tail",
    "bias_only",
    "explicit_prefixes",
}


@dataclass(frozen=True)
class ActiveParameterSelection:
    """Resolved public trainable-parameter selection."""

    mode: str
    names: tuple[str, ...]
    active_count: int
    full_count: int

    @property
    def fraction(self) -> float:
        return self.active_count / max(self.full_count, 1)


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _last_parameter_prefix(names: list[str]) -> str:
    if not names:
        raise ValueError("The model has no named parameters")
    return names[-1].rsplit(".", 1)[0] if "." in names[-1] else names[-1]


def _classifier_prefix(model: nn.Module, names: list[str]) -> str:
    for candidate in ("classifier", "head", "fc"):
        if any(_matches_prefix(name, (candidate,)) for name in names):
            return candidate
    raise ValueError(
        "classifier_head requires a public classifier/head/fc module prefix; "
        "use explicit_prefixes for this architecture"
    )


def _last_linear_prefixes(model: nn.Module, count: int) -> tuple[str, ...]:
    prefixes = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    if len(prefixes) < count:
        raise ValueError(
            f"classifier_tail requires at least {count} Linear modules, found "
            f"{len(prefixes)}"
        )
    return tuple(prefixes[-count:])


def resolve_active_parameters(
    model: nn.Module,
    config: dict,
) -> ActiveParameterSelection:
    """Resolve a data-independent trainable set from ``config``.

    ``classifier_tail`` means the last two public ``nn.Linear`` modules.  The
    mode is intentionally structural rather than update-dependent, so the same
    mask is used for every client, neighboring cohort, and training round.
    """

    mode = str(config.get("active_parameter_mode", "full")).strip().lower()
    if mode not in SUPPORTED_ACTIVE_PARAMETER_MODES:
        raise ValueError(
            "active_parameter_mode must be one of "
            f"{sorted(SUPPORTED_ACTIVE_PARAMETER_MODES)}, got {mode!r}"
        )

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    full_count = int(sum(parameter.numel() for _, parameter in named))

    if mode == "full":
        active_names = names
    elif mode == "last_layer":
        prefix = _last_parameter_prefix(names)
        active_names = [name for name in names if _matches_prefix(name, (prefix,))]
    elif mode == "classifier_head":
        prefix = _classifier_prefix(model, names)
        active_names = [name for name in names if _matches_prefix(name, (prefix,))]
    elif mode == "classifier_tail":
        prefixes = _last_linear_prefixes(model, count=2)
        active_names = [name for name in names if _matches_prefix(name, prefixes)]
    elif mode == "bias_only":
        active_names = [
            name for name in names if name.endswith(".bias") or name == "bias"
        ]
    else:
        raw_prefixes = config.get("active_parameter_prefixes")
        if not isinstance(raw_prefixes, (list, tuple)) or not raw_prefixes:
            raise ValueError(
                "explicit_prefixes requires a non-empty active_parameter_prefixes list"
            )
        prefixes = tuple(str(prefix).strip() for prefix in raw_prefixes)
        if any(not prefix for prefix in prefixes):
            raise ValueError("active_parameter_prefixes cannot contain empty values")
        active_names = [name for name in names if _matches_prefix(name, prefixes)]

    if not active_names:
        raise ValueError(f"Public mask {mode!r} selected no model parameters")
    active_set = set(active_names)
    active_count = int(
        sum(parameter.numel() for name, parameter in named if name in active_set)
    )
    return ActiveParameterSelection(
        mode=mode,
        names=tuple(active_names),
        active_count=active_count,
        full_count=full_count,
    )


def configure_active_parameters(
    model: nn.Module,
    config: dict,
) -> ActiveParameterSelection:
    """Apply the resolved mask through ``requires_grad`` and return metadata."""

    selection = resolve_active_parameters(model, config)
    active = set(selection.names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in active)
    return selection


def active_parameters(
    model: nn.Module,
    selection: ActiveParameterSelection,
) -> list[torch.nn.Parameter]:
    """Return active parameters in deterministic model order."""

    active = set(selection.names)
    parameters = [
        parameter for name, parameter in model.named_parameters() if name in active
    ]
    if not parameters:
        raise ValueError("The active parameter selection is empty")
    return parameters
