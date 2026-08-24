"""Validated configuration for deployable DMD algorithm adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import DMDVariant


@dataclass(frozen=True)
class DMDConfig:
    variant: DMDVariant = "mean"
    lr: float = 0.03
    local_epochs: int = 1
    batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 1e-4
    device: str = "cpu"
    num_classes: int = 10
    mean_mu: float = 0.15
    dispersion_mu: float = 0.0375
    cvar_tail_mass: float = 0.2
    class_weight_mode: str = "uniform"
    min_profile_count: int = 1
    reference_method: str = "median"
    reference_mode: str = "robust"
    trim_fraction: float = 0.1
    min_reference_clients: int = 2
    context_policy: str = "one_round_stale"
    warmup_rounds: int = 1

    def validate(self) -> "DMDConfig":
        if self.variant not in {"mean", "upper_semivariance", "cvar"}:
            raise ValueError(f"unsupported DMD variant: {self.variant}")
        if self.lr <= 0 or self.local_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("lr, local_epochs and batch_size must be positive")
        if self.num_classes <= 1 or self.min_profile_count <= 0:
            raise ValueError("invalid profile configuration")
        if self.mean_mu < 0 or self.dispersion_mu < 0:
            raise ValueError("DMD coefficients must be non-negative")
        if not 0.0 < self.cvar_tail_mass <= 1.0:
            raise ValueError("cvar_tail_mass must lie in (0, 1]")
        if self.class_weight_mode not in {"uniform", "frequency"}:
            raise ValueError("class_weight_mode must be uniform or frequency")
        if self.reference_method not in {"median", "trimmed_mean"}:
            raise ValueError("reference_method must be median or trimmed_mean")
        if self.reference_mode not in {"robust", "fixed_zero"}:
            raise ValueError("reference_mode must be robust or fixed_zero")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0, 0.5)")
        if self.min_reference_clients <= 0:
            raise ValueError("min_reference_clients must be positive")
        if self.context_policy != "one_round_stale":
            raise ValueError("only one_round_stale context is currently supported")
        if self.warmup_rounds < 1:
            raise ValueError("warmup_rounds must be at least one")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        variant: DMDVariant | None = None,
    ) -> "DMDConfig":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in mapping.items() if key in fields}
        if variant is not None:
            values["variant"] = variant
        return cls(**values).validate()


def default_algorithm_config(variant: DMDVariant) -> dict[str, Any]:
    config = DMDConfig(variant=variant).validate().to_dict()
    config["device_profile"] = None
    config["dmd_round_context"] = None
    config["anchor_dataloader"] = None
    return config
