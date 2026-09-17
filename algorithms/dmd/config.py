"""Validated configuration for deployable DMD algorithm adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import DMDVariant
from .profiles import MARGIN_BOUND, MARGIN_SPACES


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
    # "empirical" reproduces the historical behaviour: eta is the exact weighted
    # upper-VaR of the previous cohort.  With |A| survivors and equal weights
    # that order statistic collapses onto the cohort maximum whenever
    # tail_mass <= 1/|A| (4 survivors, tail_mass=0.2 -> eta = max), which leaves
    # the hinge inert.  "dual" instead carries eta across rounds and moves it by
    # Rockafellar-Uryasev dual ascent, so the threshold converges to the
    # (1 - tail_mass) quantile by averaging over rounds rather than by ranking
    # four points.
    cvar_eta_mode: str = "empirical"
    # Dimensionless: the dual step is scaled by the cohort mean deficit so the
    # same value works whatever the magnitude of the margins.
    cvar_eta_lr: float = 0.1
    # Decision-margin space; see algorithms/dmd/profiles.py.  "logit" is the
    # historical, unbounded definition used by the published tables.
    margin_space: str = "logit"
    # Satisficing target: the penalty is [margin_target - m]_+^2, so the class
    # must clear the boundary by this much before its gradient switches off.
    # 0.0 reproduces the published "reference zero" behaviour.  A positive value
    # only makes sense in a bounded space, where it is on a comparable scale.
    margin_target: float = 0.0
    # Weighting of the cross-entropy the penalty is added to.  "none" is the
    # plain CE of every published DMD run.  "inverse_frequency" is the CB-CE
    # objective of algorithms/cb_ce.py, so a margin arm built on it differs from
    # a mean_mu=0 control by the penalty alone.
    ce_class_weighting: str = "none"
    # Local loss the penalty is added to, one of algorithms.label_skew's
    # LABEL_SKEW_LOSSES.  At mean_mu=0 the client trains exactly like the matching
    # standalone baseline, so baselines compared through this client share the
    # anchor pass and the random draws.  ce_class_weighting=inverse_frequency is
    # the earlier spelling of base_loss=inverse_frequency and stays valid.
    base_loss: str = "ce"
    label_skew_beta: float = 0.999
    label_skew_tau: float = 1.0
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
        if self.cvar_eta_mode not in {"empirical", "dual"}:
            raise ValueError("cvar_eta_mode must be empirical or dual")
        if self.cvar_eta_lr < 0:
            raise ValueError("cvar_eta_lr must be non-negative")
        if self.margin_space not in MARGIN_SPACES:
            raise ValueError(f"margin_space must be one of {MARGIN_SPACES}")
        bound = MARGIN_BOUND[self.margin_space]
        if bound is not None and not 0.0 <= self.margin_target < bound:
            raise ValueError(
                f"margin_target must lie in [0, {bound}) for {self.margin_space}"
            )
        if self.margin_target < 0:
            raise ValueError("margin_target must be non-negative")
        if self.ce_class_weighting not in {"none", "inverse_frequency"}:
            raise ValueError("ce_class_weighting must be none or inverse_frequency")
        from algorithms.label_skew import LABEL_SKEW_LOSSES

        if self.base_loss not in LABEL_SKEW_LOSSES:
            raise ValueError(f"base_loss must be one of {LABEL_SKEW_LOSSES}")
        if self.ce_class_weighting == "inverse_frequency" and self.base_loss not in {
            "ce", "inverse_frequency"
        }:
            raise ValueError(
                f"ce_class_weighting=inverse_frequency conflicts with base_loss={self.base_loss}"
            )
        if not 0.0 <= self.label_skew_beta < 1.0:
            raise ValueError("label_skew_beta must lie in [0, 1)")
        if self.label_skew_tau < 0:
            raise ValueError("label_skew_tau must be non-negative")
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

    @property
    def effective_base_loss(self) -> str:
        if self.ce_class_weighting == "inverse_frequency":
            return "inverse_frequency"
        return self.base_loss

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
