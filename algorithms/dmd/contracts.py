"""Typed, serialization-safe contracts shared by DMD clients and server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

DMDVariant = Literal["mean", "upper_semivariance", "cvar"]


@dataclass(frozen=True)
class MarginProfile:
    """Per-class mean margins and their local support."""

    values: Tensor
    counts: Tensor

    @property
    def observed(self) -> Tensor:
        return torch.isfinite(self.values) & (self.counts > 0)


@dataclass(frozen=True)
class TemporalMarginReference:
    """Class-wise temporal reference and its auditable support statistics."""

    values: Tensor
    scale: Tensor
    raw_support: Tensor
    effective_support: Tensor
    reliability: Tensor
    mean_age: Tensor
    max_age: Tensor


@dataclass(frozen=True)
class WeightedUpperCvar:
    """Exact finite-cohort upper-tail CVaR state."""

    eta: Tensor
    cvar: Tensor
    tail_fraction: Tensor
    tail_weights: Tensor


@dataclass(frozen=True)
class ObjectiveTerms:
    """Auditable decomposition of one DMD fairness addend."""

    total: Tensor
    mean: Tensor
    dispersion: Tensor


@dataclass(frozen=True)
class DMDClientReport:
    """Client report that can be embedded in existing metadata messages.

    Only primitive Python values are emitted by :meth:`to_wire`; tensors never
    enter msgpack directly.  Margins are evaluated after local training and
    become eligible for the one-round-stale reference of the next round.
    """

    client_id: int
    round_num: int
    margins: tuple[float | None, ...]
    counts: tuple[int, ...]
    dataset_size: int
    deficit: float | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "client_id": int(self.client_id),
            "round_num": int(self.round_num),
            "margins": list(self.margins),
            "counts": [int(value) for value in self.counts],
            "dataset_size": int(self.dataset_size),
            "deficit": self.deficit,
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "DMDClientReport":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("unsupported DMD client-report schema")
        margins = tuple(
            None if item is None else float(item) for item in value["margins"]
        )
        counts = tuple(int(item) for item in value["counts"])
        if len(margins) != len(counts):
            raise ValueError("DMD margin values and counts must align")
        deficit = value.get("deficit")
        return cls(
            client_id=int(value["client_id"]),
            round_num=int(value["round_num"]),
            margins=margins,
            counts=counts,
            dataset_size=int(value["dataset_size"]),
            deficit=None if deficit is None else float(deficit),
        )


@dataclass(frozen=True)
class DMDRoundContext:
    """Frozen context consumed by local optimization in a later round.

    The current ZMQ server does not yet propagate algorithm-generated state to
    the next ``TRAIN_REQ``.  Consequently this contract is deployable only
    when an orchestrator injects ``dmd_round_context`` explicitly.  Absence of
    a context safely falls back to cross-entropy-only local training.
    """

    source_round: int
    variant: DMDVariant
    reference: tuple[float | None, ...]
    reliability: tuple[float, ...]
    cohort_mean_deficit: float = 0.0
    cvar_eta: float = 0.0
    cvar_tail_mass: float = 0.2

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_round": int(self.source_round),
            "variant": self.variant,
            "reference": list(self.reference),
            "reliability": [float(value) for value in self.reliability],
            "cohort_mean_deficit": float(self.cohort_mean_deficit),
            "cvar_eta": float(self.cvar_eta),
            "cvar_tail_mass": float(self.cvar_tail_mass),
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "DMDRoundContext":
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("unsupported DMD round-context schema")
        variant = str(value["variant"])
        if variant not in {"mean", "upper_semivariance", "cvar"}:
            raise ValueError(f"unsupported DMD variant: {variant}")
        reference = tuple(
            None if item is None else float(item) for item in value["reference"]
        )
        reliability = tuple(float(item) for item in value["reliability"])
        if len(reference) != len(reliability):
            raise ValueError("DMD reference and reliability must align")
        return cls(
            source_round=int(value["source_round"]),
            variant=variant,  # type: ignore[arg-type]
            reference=reference,
            reliability=reliability,
            cohort_mean_deficit=float(value.get("cohort_mean_deficit", 0.0)),
            cvar_eta=float(value.get("cvar_eta", 0.0)),
            cvar_tail_mass=float(value.get("cvar_tail_mass", 0.2)),
        )

    def reference_tensor(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> Tensor:
        return torch.tensor(
            [float("nan") if value is None else value for value in self.reference],
            device=device,
            dtype=dtype,
        )

    def reliability_tensor(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> Tensor:
        return torch.tensor(self.reliability, device=device, dtype=dtype)
