r"""Strictly causal, scalable temporal RCIG reference for private FAR.

This module turns already-private, server-clipped client uploads into the two
strictly-past views required by robust covariance-innovation gating (RCIG).
It deliberately separates two spaces:

* complete update vectors are retained for the deployed reference; and
* a fixed public coordinate subspace (32 or 64 coordinates) is used for the
  innovation test and its covariance.

The separation avoids a dense covariance in model dimension.  For a clipped
upload with radial clipping factor ``f`` and unit direction ``u``, the
selected-coordinate covariance multiplier is

.. math::

   I_S,\quad f=1,
   \qquad\text{or}\qquad
   f^2(I_S-u_Su_S^\top),\quad f<1.

The second expression is the selected-row block of ``J J^T`` for the radial
projection Jacobian ``J=f(I-uu^T)``.  Public per-coordinate DP variances are
propagated through this multiplier.  No realised noise, clean-gradient,
attack-label, or Byzantine-mask quantity is accepted by the API.

The state enforces causality structurally.  ``reference_for_round(t)`` only
reads snapshots that were committed at rounds strictly smaller than ``t``.
The caller must commit round ``t`` only *after* aggregation succeeds.  During
the deterministic cold start the returned reference is zero; the integrating
algorithm is expected to use uniform FAR weights until ``ready`` is true.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import torch

from algorithms.gaussian_aware_reference_k7_rcig import (
    euclidean_innovation_fusion,
    robust_covariance_innovation_fusion,
)
from robustness.aggregators import clip_l2

RCIGMode = Literal["full", "isotropic", "euclidean"]
PersistentPolicy = Literal["rolling", "freeze_hysteresis"]

_PUBLIC_VARIANCE_PROVENANCE = "authenticated_public_mechanism"


def _finite_scalar(value: float, *, name: str, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class RCIGTemporalConfig:
    """Public parameters of the temporal RCIG construction."""

    window_length: int = 4
    gate_window_length: int = 8
    subspace_dimension: int = 64
    subspace_seed: int = 0
    server_clip_norm: float = 0.1
    influence_cap: float = 0.1
    minimum_accepted_mass: float = 1.0
    gate_inner_mad_multiplier: float = 2.5
    gate_outer_mad_multiplier: float = 4.5
    gate_minimum_mad: float = 1.0e-8
    process_variance: float = 0.0
    ridge: float = 1.0e-8
    innovation_threshold: float = 3.0
    isotropic_innovation_threshold: float | None = None
    euclidean_innovation_threshold: float | None = None
    covariance_mode: RCIGMode = "full"
    persistent_policy: PersistentPolicy = "rolling"
    recovery_threshold: float | None = None
    recovery_patience: int = 2
    public_variance_provenance: str = _PUBLIC_VARIANCE_PROVENANCE

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_length, Integral)
            or isinstance(self.window_length, bool)
            or self.window_length < 1
        ):
            raise ValueError("window_length must be a positive integer")
        if (
            not isinstance(self.gate_window_length, Integral)
            or isinstance(self.gate_window_length, bool)
            or self.gate_window_length < 1
        ):
            raise ValueError("gate_window_length must be a positive integer")
        if (
            not isinstance(self.subspace_dimension, Integral)
            or isinstance(self.subspace_dimension, bool)
            or self.subspace_dimension not in {32, 64}
        ):
            raise ValueError("subspace_dimension must be exactly 32 or 64")
        if not isinstance(self.subspace_seed, Integral) or isinstance(
            self.subspace_seed, bool
        ):
            raise ValueError("subspace_seed must be an integer")
        server_cap = _finite_scalar(
            self.server_clip_norm, name="server_clip_norm", positive=True
        )
        influence_cap = _finite_scalar(
            self.influence_cap, name="influence_cap", positive=True
        )
        if influence_cap > server_cap:
            raise ValueError("influence_cap cannot exceed server_clip_norm")
        _finite_scalar(
            self.minimum_accepted_mass,
            name="minimum_accepted_mass",
            positive=True,
        )
        inner = _finite_scalar(
            self.gate_inner_mad_multiplier,
            name="gate_inner_mad_multiplier",
        )
        outer = _finite_scalar(
            self.gate_outer_mad_multiplier,
            name="gate_outer_mad_multiplier",
        )
        if inner < 0.0 or outer <= inner:
            raise ValueError("gate multipliers must satisfy 0 <= inner < outer")
        _finite_scalar(self.gate_minimum_mad, name="gate_minimum_mad", positive=True)
        process_variance = _finite_scalar(
            self.process_variance, name="process_variance"
        )
        if process_variance < 0.0:
            raise ValueError("process_variance must be non-negative")
        _finite_scalar(self.ridge, name="ridge", positive=True)
        _finite_scalar(
            self.innovation_threshold,
            name="innovation_threshold",
            positive=True,
        )
        for name, value in (
            ("isotropic_innovation_threshold", self.isotropic_innovation_threshold),
            ("euclidean_innovation_threshold", self.euclidean_innovation_threshold),
        ):
            if value is not None:
                _finite_scalar(value, name=name, positive=True)
        if self.covariance_mode not in {"full", "isotropic", "euclidean"}:
            raise ValueError(
                "covariance_mode must be 'full', 'isotropic', or 'euclidean'"
            )
        if self.persistent_policy not in {"rolling", "freeze_hysteresis"}:
            raise ValueError(
                "persistent_policy must be 'rolling' or 'freeze_hysteresis'"
            )
        if (
            not isinstance(self.recovery_patience, Integral)
            or isinstance(self.recovery_patience, bool)
            or self.recovery_patience < 1
        ):
            raise ValueError("recovery_patience must be a positive integer")
        primary_threshold = self.threshold_for_mode(self.covariance_mode)
        if self.recovery_threshold is not None:
            recovery = _finite_scalar(
                self.recovery_threshold, name="recovery_threshold", positive=True
            )
            if recovery >= primary_threshold:
                raise ValueError(
                    "recovery_threshold must be lower than the activation threshold"
                )
        if self.public_variance_provenance != _PUBLIC_VARIANCE_PROVENANCE:
            raise ValueError(
                "public_noise_variances must come from authenticated public "
                "mechanism parameters"
            )

    @property
    def required_history(self) -> int:
        """Number of committed past rounds needed for the first deployment."""

        return self.gate_window_length + 2 * self.window_length

    def threshold_for_mode(self, mode: RCIGMode) -> float:
        """Return the public threshold registered for one candidate."""

        if mode == "isotropic" and self.isotropic_innovation_threshold is not None:
            return float(self.isotropic_innovation_threshold)
        if mode == "euclidean" and self.euclidean_innovation_threshold is not None:
            return float(self.euclidean_innovation_threshold)
        return float(self.innovation_threshold)


@dataclass(frozen=True)
class RCIGSnapshot:
    """One immutable, already-private and already-server-clipped round."""

    round_num: int
    client_ids: tuple[int, ...]
    vectors: torch.Tensor
    selected_vectors: torch.Tensor
    server_clip_factors: torch.Tensor
    public_noise_variances: torch.Tensor
    public_coordinate_indices: torch.Tensor


@dataclass(frozen=True)
class RCIGReferenceResult:
    """Reference and audit data returned for one deployment round."""

    reference: torch.Tensor
    ready: bool
    diagnostics: dict[str, Any]
    older_view: torch.Tensor | None = None
    newer_view: torch.Tensor | None = None
    covariance_older: torch.Tensor | None = None
    covariance_newer: torch.Tensor | None = None
    candidate_references: dict[str, torch.Tensor] | None = None
    deployment_round: int | None = None
    _persistent_transition: tuple[torch.Tensor | None, bool, int] | None = None


def _public_coordinates(dimension: int, count: int, seed: int) -> torch.Tensor:
    if dimension < count:
        raise ValueError(
            f"full dimension {dimension} is smaller than the public "
            f"RCIG subspace dimension {count}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(dimension, generator=generator)[:count].sort().values


def _coordinate_hash(indices: torch.Tensor) -> str:
    payload = ",".join(str(int(value)) for value in indices.cpu().tolist())
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class TemporalRCIGReferenceState:
    """Causal state machine for an RCIG reference.

    The initial gate is learned once from a disjoint, strictly-past window and
    then frozen.  Only the two view windows roll.  This keeps memory bounded by
    ``gate_window_length + 2*window_length`` before deployment and by
    ``2*window_length`` afterwards.
    """

    def __init__(self, config: RCIGTemporalConfig):
        if not isinstance(config, RCIGTemporalConfig):
            raise TypeError("config must be an RCIGTemporalConfig")
        self.config = config
        self._history: list[RCIGSnapshot] = []
        self._frozen_gate: torch.Tensor | None = None
        self._client_ids: tuple[int, ...] | None = None
        self._dimension: int | None = None
        self._coordinate_indices: torch.Tensor | None = None
        self._snapshot_device: torch.device | None = None
        self._snapshot_dtype: torch.dtype | None = None
        self._last_accepted_reference: torch.Tensor | None = None
        self._persistent_frozen = False
        self._recovery_count = 0
        self._gate_source_range: tuple[int, int] | None = None

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def last_committed_round(self) -> int | None:
        return self._history[-1].round_num if self._history else None

    @property
    def ready(self) -> bool:
        return self._frozen_gate is not None or (
            len(self._history) >= self.config.required_history
        )

    def reset(self) -> None:
        """Discard all transcript state, for a fresh independent run."""

        self._history.clear()
        self._frozen_gate = None
        self._client_ids = None
        self._dimension = None
        self._coordinate_indices = None
        self._snapshot_device = None
        self._snapshot_dtype = None
        self._last_accepted_reference = None
        self._persistent_frozen = False
        self._recovery_count = 0
        self._gate_source_range = None

    def make_snapshot(
        self,
        *,
        round_num: int,
        client_ids: list[int] | tuple[int, ...],
        clipped_vectors: torch.Tensor,
        server_clip_factors: torch.Tensor,
        public_noise_variances: torch.Tensor,
    ) -> RCIGSnapshot:
        """Validate and copy one round without mutating temporal state."""

        if not isinstance(round_num, int) or isinstance(round_num, bool):
            raise TypeError("round_num must be an integer")
        if round_num < 0:
            raise ValueError("round_num must be non-negative")
        if not isinstance(clipped_vectors, torch.Tensor) or clipped_vectors.ndim != 2:
            raise ValueError("clipped_vectors must have shape (n,d)")
        if not clipped_vectors.is_floating_point() or not bool(
            torch.isfinite(clipped_vectors).all()
        ):
            raise ValueError("clipped_vectors must be finite floating point")
        n, dimension = (int(value) for value in clipped_vectors.shape)
        if n < 2 or dimension < self.config.subspace_dimension:
            raise ValueError(
                "RCIG requires at least two clients and enough model coordinates"
            )
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in client_ids
        ):
            raise TypeError("client_ids must contain integers")
        resolved_ids = tuple(int(value) for value in client_ids)
        if len(resolved_ids) != n or len(set(resolved_ids)) != n:
            raise ValueError("client_ids must be unique and aligned with vectors")
        factors = torch.as_tensor(
            server_clip_factors,
            device=clipped_vectors.device,
            dtype=clipped_vectors.dtype,
        )
        variances = torch.as_tensor(
            public_noise_variances,
            device=clipped_vectors.device,
            dtype=clipped_vectors.dtype,
        )
        if factors.shape != (n,) or variances.shape != (n,):
            raise ValueError(
                "server_clip_factors and public_noise_variances must have shape (n,)"
            )
        if not bool(torch.isfinite(factors).all()) or bool(
            ((factors <= 0.0) | (factors > 1.0)).any()
        ):
            raise ValueError("server_clip_factors must be finite and lie in (0,1]")
        if not bool(torch.isfinite(variances).all()) or bool((variances < 0.0).any()):
            raise ValueError("public_noise_variances must be finite and non-negative")
        norms = torch.linalg.vector_norm(clipped_vectors, dim=1)
        cap = float(self.config.server_clip_norm)
        tolerance = 256.0 * torch.finfo(clipped_vectors.dtype).eps * max(1.0, cap)
        if bool((norms > cap + tolerance).any()):
            raise ValueError("clipped_vectors violate server_clip_norm")
        clipped_mask = factors < 1.0 - 64.0 * torch.finfo(factors.dtype).eps
        if bool((torch.abs(norms[clipped_mask] - cap) > tolerance).any()):
            raise ValueError(
                "a factor below one must correspond to a vector on the clip sphere"
            )

        indices = _public_coordinates(
            dimension, self.config.subspace_dimension, self.config.subspace_seed
        ).to(clipped_vectors.device)
        order = torch.tensor(
            sorted(range(n), key=lambda index: resolved_ids[index]),
            device=clipped_vectors.device,
            dtype=torch.long,
        )
        sorted_ids = tuple(resolved_ids[int(index)] for index in order.cpu().tolist())
        vectors = clipped_vectors.detach().index_select(0, order).clone()
        factors = factors.detach().index_select(0, order).clone()
        variances = variances.detach().index_select(0, order).clone()
        selected = vectors.index_select(1, indices).clone()
        return RCIGSnapshot(
            round_num=round_num,
            client_ids=sorted_ids,
            vectors=vectors,
            selected_vectors=selected,
            server_clip_factors=factors,
            public_noise_variances=variances,
            public_coordinate_indices=indices.detach().clone(),
        )

    def commit_snapshot(
        self,
        snapshot: RCIGSnapshot,
        *,
        reference_result: RCIGReferenceResult | None = None,
    ) -> None:
        """Commit a completed round; snapshots must be consecutive."""

        if not isinstance(snapshot, RCIGSnapshot):
            raise TypeError("snapshot must be an RCIGSnapshot")
        if snapshot.vectors.ndim != 2 or snapshot.selected_vectors.ndim != 2:
            raise ValueError("RCIG snapshot vectors must be matrices")
        if not bool(torch.isfinite(snapshot.vectors).all()) or not bool(
            torch.isfinite(snapshot.selected_vectors).all()
        ):
            raise ValueError("RCIG snapshot vectors must be finite")
        if snapshot.selected_vectors.shape != (
            snapshot.vectors.shape[0],
            self.config.subspace_dimension,
        ):
            raise ValueError("RCIG snapshot selected-vector shape is invalid")
        if self._history and snapshot.round_num != self._history[-1].round_num + 1:
            raise ValueError("RCIG snapshots must be committed in consecutive rounds")
        if self._client_ids is None:
            next_client_ids = snapshot.client_ids
            next_dimension = int(snapshot.vectors.shape[1])
            next_coordinates = snapshot.public_coordinate_indices.cpu().clone()
            next_device = snapshot.vectors.device
            next_dtype = snapshot.vectors.dtype
        else:
            if snapshot.client_ids != self._client_ids:
                raise ValueError(
                    "RCIG requires a stable full-participation client roster"
                )
            if int(snapshot.vectors.shape[1]) != self._dimension:
                raise ValueError("RCIG full update dimension changed during the run")
            if not torch.equal(
                snapshot.public_coordinate_indices.cpu(), self._coordinate_indices
            ):
                raise ValueError(
                    "RCIG public coordinate subspace changed during the run"
                )
            if (
                snapshot.vectors.device != self._snapshot_device
                or snapshot.vectors.dtype != self._snapshot_dtype
            ):
                raise ValueError("RCIG snapshot device or dtype changed during the run")
            next_client_ids = self._client_ids
            next_dimension = self._dimension
            next_coordinates = self._coordinate_indices
            next_device = self._snapshot_device
            next_dtype = self._snapshot_dtype

        next_accepted = self._last_accepted_reference
        next_persistent_frozen = self._persistent_frozen
        next_recovery_count = self._recovery_count
        if reference_result is not None:
            if not isinstance(reference_result, RCIGReferenceResult):
                raise TypeError("reference_result must be an RCIGReferenceResult")
            if reference_result.deployment_round != snapshot.round_num:
                raise ValueError(
                    "reference_result and snapshot must describe the same round"
                )
            transition = reference_result._persistent_transition
            if transition is not None:
                accepted, frozen, recovery_count = transition
                next_accepted = None if accepted is None else accepted.detach().clone()
                if next_accepted is not None:
                    if next_accepted.shape != (next_dimension,) or not bool(
                        torch.isfinite(next_accepted).all()
                    ):
                        raise ValueError("invalid RCIG persistent-state transition")
                next_persistent_frozen = bool(frozen)
                next_recovery_count = int(recovery_count)
        elif self.config.persistent_policy == "freeze_hysteresis" and self.ready:
            raise ValueError(
                "freeze_hysteresis requires the deployed reference_result at commit"
            )

        # Freeze the gate from the strictly older, disjoint gate window before
        # incorporating the new round.  This operation is deterministic and
        # uses only already committed snapshots.
        next_gate = self._frozen_gate
        next_gate_source_range = self._gate_source_range
        if next_gate is None and len(self._history) >= self.config.required_history:
            source = self._history[
                -self.config.required_history : -2 * self.config.window_length
            ]
            next_gate, _ = self._compute_gate(source)
            next_gate_source_range = (source[0].round_num, source[-1].round_num)
        next_history = [*self._history, snapshot]
        keep = (
            2 * self.config.window_length
            if next_gate is not None
            else self.config.required_history
        )
        if len(next_history) > keep:
            next_history = next_history[-keep:]

        # Commit atomically after every validation and deterministic
        # construction has succeeded.
        self._client_ids = next_client_ids
        self._dimension = next_dimension
        self._coordinate_indices = next_coordinates
        self._snapshot_device = next_device
        self._snapshot_dtype = next_dtype
        self._last_accepted_reference = next_accepted
        self._persistent_frozen = next_persistent_frozen
        self._recovery_count = next_recovery_count
        self._frozen_gate = next_gate
        self._gate_source_range = next_gate_source_range
        self._history = next_history

    def _compute_gate(
        self, snapshots: list[RCIGSnapshot]
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        if len(snapshots) != self.config.gate_window_length:
            raise RuntimeError("internal RCIG gate window has the wrong length")
        values = torch.stack([item.selected_vectors for item in snapshots])
        centres = values.median(dim=1).values
        deviations = torch.linalg.vector_norm(values - centres[:, None, :], dim=2)
        statistic = deviations.median(dim=0).values
        median = statistic.median()
        mad = (
            torch.abs(statistic - median)
            .median()
            .clamp_min(float(self.config.gate_minimum_mad))
        )
        inner = median + float(self.config.gate_inner_mad_multiplier) * mad
        outer = median + float(self.config.gate_outer_mad_multiplier) * mad
        gate = ((outer - statistic) / (outer - inner)).clamp(0.0, 1.0)
        return gate, {
            "gate_mass": float(gate.sum().item()),
            "gate_min": float(gate.min().item()),
            "gate_max": float(gate.max().item()),
            "gate_mean": float(gate.mean().item()),
            "gate_zero_count": int((gate == 0.0).sum().item()),
            "gate_statistic_median": float(median.item()),
            "gate_statistic_mad": float(mad.item()),
            "gate_inner_threshold": float(inner.item()),
            "gate_outer_threshold": float(outer.item()),
        }

    def _view_and_covariance(
        self,
        snapshots: list[RCIGSnapshot],
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        length = self.config.window_length
        if len(snapshots) != length:
            raise RuntimeError("internal RCIG view window has the wrong length")
        n = len(snapshots[0].client_ids)
        if float(self.config.minimum_accepted_mass) > n:
            raise ValueError("minimum_accepted_mass cannot exceed the client count")
        gate = gate.to(snapshots[0].vectors)
        accepted_mass = gate.sum()
        denominator = length * torch.maximum(
            accepted_mass,
            torch.as_tensor(
                float(self.config.minimum_accepted_mass),
                device=gate.device,
                dtype=gate.dtype,
            ),
        )
        coefficients = gate / denominator
        full = torch.stack([item.vectors for item in snapshots])
        view = torch.sum(coefficients[None, :, None] * full, dim=(0, 1))

        selected_dimension = self.config.subspace_dimension
        identity = torch.eye(selected_dimension, device=view.device, dtype=view.dtype)
        covariance = torch.zeros_like(identity)
        clipped_blocks = 0
        for snapshot in snapshots:
            norms = torch.linalg.vector_norm(snapshot.vectors, dim=1)
            for client_index in range(n):
                coefficient = coefficients[client_index]
                variance = snapshot.public_noise_variances[client_index]
                factor = snapshot.server_clip_factors[client_index]
                if float(factor.item()) >= 1.0 - 64.0 * torch.finfo(factor.dtype).eps:
                    multiplier = identity
                else:
                    unit_selected = snapshot.selected_vectors[client_index] / norms[
                        client_index
                    ].clamp_min(torch.finfo(snapshot.vectors.dtype).tiny)
                    multiplier = factor.square() * (
                        identity - unit_selected[:, None] * unit_selected[None, :]
                    )
                    clipped_blocks += 1
                covariance = covariance + coefficient.square() * variance * multiplier
        covariance = 0.5 * (covariance + covariance.T)
        tolerance = (
            512.0
            * torch.finfo(covariance.dtype).eps
            * selected_dimension
            * max(1.0, float(covariance.abs().max().item()))
        )
        eigenvalues = torch.linalg.eigvalsh(covariance)
        if float(eigenvalues.min().item()) < -tolerance:
            raise RuntimeError(
                "RCIG propagated covariance is not positive semidefinite"
            )
        cap_tolerance = (
            256.0
            * torch.finfo(view.dtype).eps
            * max(1.0, float(self.config.server_clip_norm))
        )
        if float(torch.linalg.vector_norm(view).item()) > (
            float(self.config.server_clip_norm) + cap_tolerance
        ):
            raise RuntimeError("RCIG robust view violated the server norm bound")
        covariance_trace = float(torch.trace(covariance).item())
        covariance_max_eigenvalue = float(eigenvalues.max().item())
        covariance_mean_eigenvalue = covariance_trace / float(covariance.shape[0])
        covariance_anisotropy_ratio = max(
            1.0,
            covariance_max_eigenvalue
            / max(covariance_mean_eigenvalue, float(self.config.ridge)),
        )
        return (
            view,
            covariance,
            {
                "accepted_mass": float(accepted_mass.item()),
                "denominator": float(denominator.item()),
                "denominator_floor_active": bool(
                    float(accepted_mass.item())
                    < float(self.config.minimum_accepted_mass)
                ),
                "view_norm": float(torch.linalg.vector_norm(view).item()),
                "covariance_trace": covariance_trace,
                "covariance_min_eigenvalue": float(eigenvalues.min().item()),
                "covariance_max_eigenvalue": covariance_max_eigenvalue,
                "covariance_mean_eigenvalue": covariance_mean_eigenvalue,
                "covariance_anisotropy_ratio": covariance_anisotropy_ratio,
                "covariance_psd": True,
                "covariance_block_count": length * n,
                "radially_clipped_covariance_block_count": clipped_blocks,
                "dense_full_model_covariance_materialized": False,
                "covariance_model": (
                    "delta_method_radial_clip_restricted_to_public_coordinates"
                ),
                "covariance_status": (
                    "local_delta_method_not_exact_across_radial_clip_boundary"
                ),
            },
        )

    def reference_for_round(
        self,
        *,
        round_num: int,
        dimension: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RCIGReferenceResult:
        """Return ``F_t`` from committed rounds ``< t`` only."""

        if not isinstance(round_num, int) or isinstance(round_num, bool):
            raise TypeError("round_num must be an integer")
        if round_num < 0:
            raise ValueError("round_num must be non-negative")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        resolved_dimension = int(dimension)
        if resolved_dimension < self.config.subspace_dimension:
            raise ValueError("dimension is smaller than the RCIG public subspace")
        if self._dimension is not None and resolved_dimension != self._dimension:
            raise ValueError("requested reference dimension changed during the run")
        if self._history and round_num != self._history[-1].round_num + 1:
            raise ValueError(
                "reference_for_round must target the next uncommitted round"
            )
        common = {
            "rcig_ready": self.ready,
            "rcig_cold_start": not self.ready,
            "rcig_past_only": True,
            "rcig_current_round_read": False,
            "rcig_history_length": len(self._history),
            "rcig_required_history": self.config.required_history,
            "rcig_window_length": self.config.window_length,
            "rcig_gate_window_length": self.config.gate_window_length,
            "rcig_covariance_mode": self.config.covariance_mode,
            "rcig_subspace_dimension": self.config.subspace_dimension,
            "rcig_full_dimension": resolved_dimension,
            "rcig_subspace_seed": self.config.subspace_seed,
            "rcig_coordinate_hash": _coordinate_hash(
                _public_coordinates(
                    resolved_dimension,
                    self.config.subspace_dimension,
                    self.config.subspace_seed,
                )
            ),
            "rcig_public_variance_provenance": (self.config.public_variance_provenance),
            "rcig_diagnostics_use_realised_noise": False,
            "rcig_diagnostics_use_attack_labels": False,
            "rcig_diagnostics_use_clean_gradients": False,
        }
        if not self.ready:
            return RCIGReferenceResult(
                reference=torch.zeros(resolved_dimension, device=device, dtype=dtype),
                ready=False,
                diagnostics={
                    **common,
                    "rcig_cold_start_policy": "zero_reference_uniform_weights_required",
                },
                candidate_references=None,
                deployment_round=round_num,
            )

        if self._frozen_gate is None:
            gate_snapshots = self._history[
                -self.config.required_history : -2 * self.config.window_length
            ]
            gate, gate_metrics = self._compute_gate(gate_snapshots)
            gate_rounds = [item.round_num for item in gate_snapshots]
        else:
            gate = self._frozen_gate
            gate_metrics = {
                "gate_mass": float(gate.sum().item()),
                "gate_min": float(gate.min().item()),
                "gate_max": float(gate.max().item()),
                "gate_mean": float(gate.mean().item()),
                "gate_zero_count": int((gate == 0.0).sum().item()),
            }
            gate_rounds = (
                []
                if self._gate_source_range is None
                else [self._gate_source_range[0], self._gate_source_range[1]]
            )

        past_views = self._history[-2 * self.config.window_length :]
        older_snapshots = past_views[: self.config.window_length]
        newer_snapshots = past_views[self.config.window_length :]
        older, covariance_older, older_metrics = self._view_and_covariance(
            older_snapshots, gate
        )
        newer, covariance_newer, newer_metrics = self._view_and_covariance(
            newer_snapshots, gate
        )
        coordinates = older_snapshots[0].public_coordinate_indices.to(older.device)
        selected_older = older.index_select(0, coordinates)
        selected_newer = newer.index_select(0, coordinates)
        mode_diagnostics: dict[str, dict[str, Any]] = {}
        mode_trust: dict[str, float] = {}
        for mode in ("full", "isotropic"):
            _, candidate_metrics = robust_covariance_innovation_fusion(
                selected_older,
                selected_newer,
                covariance_older,
                covariance_newer,
                process_variance=float(self.config.process_variance),
                ridge=float(self.config.ridge),
                innovation_threshold=self.config.threshold_for_mode(mode),
                influence_cap=float(self.config.server_clip_norm),
                covariance_mode=mode,
                return_diagnostics=True,
            )
            mode_diagnostics[mode] = candidate_metrics
            mode_trust[mode] = float(candidate_metrics["newer_view_trust"])
        _, euclidean_metrics = euclidean_innovation_fusion(
            selected_older,
            selected_newer,
            innovation_threshold=self.config.threshold_for_mode("euclidean"),
            influence_cap=float(self.config.server_clip_norm),
            return_diagnostics=True,
        )
        mode_diagnostics["euclidean"] = euclidean_metrics
        mode_trust["euclidean"] = float(euclidean_metrics["newer_view_trust"])

        def projected_fusion(trust: float) -> torch.Tensor:
            return clip_l2(
                older + trust * (newer - older),
                float(self.config.influence_cap),
            )

        candidate_references = {
            "identity_new": clip_l2(newer, float(self.config.influence_cap)),
            "identity_old": clip_l2(older, float(self.config.influence_cap)),
            "midpoint": projected_fusion(0.5),
            "rcig_full": projected_fusion(mode_trust["full"]),
            "rcig_isotropic": projected_fusion(mode_trust["isotropic"]),
            "rcig_euclidean": projected_fusion(mode_trust["euclidean"]),
        }
        configured_candidate = {
            "full": "rcig_full",
            "isotropic": "rcig_isotropic",
            "euclidean": "rcig_euclidean",
        }[self.config.covariance_mode]
        rolling_reference = candidate_references[configured_candidate]
        fusion_metrics = mode_diagnostics[self.config.covariance_mode]
        trust = mode_trust[self.config.covariance_mode]

        proposed_accepted = self._last_accepted_reference
        proposed_frozen = self._persistent_frozen
        proposed_recovery = self._recovery_count
        persistent_action = "rolling"
        if self.config.persistent_policy == "rolling":
            reference = rolling_reference
            proposed_accepted = rolling_reference
            proposed_frozen = False
            proposed_recovery = 0
        elif not self._persistent_frozen:
            if bool(fusion_metrics["gate_active"]):
                reference = (
                    candidate_references["identity_old"]
                    if self._last_accepted_reference is None
                    else self._last_accepted_reference.to(rolling_reference)
                )
                proposed_accepted = reference
                proposed_frozen = True
                proposed_recovery = 0
                persistent_action = "freeze_on_activation"
            else:
                reference = rolling_reference
                proposed_accepted = rolling_reference
                proposed_frozen = False
                proposed_recovery = 0
                persistent_action = "accept"
        else:
            if self._last_accepted_reference is None:
                raise RuntimeError("frozen RCIG state has no accepted reference")
            accepted_selected = self._last_accepted_reference.to(older).index_select(
                0, coordinates
            )
            if self.config.covariance_mode == "euclidean":
                _, recovery_metrics = euclidean_innovation_fusion(
                    accepted_selected,
                    selected_newer,
                    innovation_threshold=self.config.threshold_for_mode("euclidean"),
                    influence_cap=float(self.config.server_clip_norm),
                    return_diagnostics=True,
                )
            else:
                _, recovery_metrics = robust_covariance_innovation_fusion(
                    accepted_selected,
                    selected_newer,
                    covariance_older,
                    covariance_newer,
                    process_variance=float(self.config.process_variance),
                    ridge=float(self.config.ridge),
                    innovation_threshold=self.config.threshold_for_mode(
                        self.config.covariance_mode
                    ),
                    influence_cap=float(self.config.server_clip_norm),
                    covariance_mode=self.config.covariance_mode,
                    return_diagnostics=True,
                )
            statistic = float(
                recovery_metrics.get(
                    "standardized_innovation", recovery_metrics["innovation_norm"]
                )
            )
            release_threshold = (
                float(self.config.recovery_threshold)
                if self.config.recovery_threshold is not None
                else 0.75 * self.config.threshold_for_mode(self.config.covariance_mode)
            )
            proposed_recovery = (
                self._recovery_count + 1 if statistic <= release_threshold else 0
            )
            if proposed_recovery >= self.config.recovery_patience:
                reference = rolling_reference
                proposed_accepted = rolling_reference
                proposed_frozen = False
                proposed_recovery = 0
                persistent_action = "release_after_recovery"
            else:
                reference = self._last_accepted_reference.to(rolling_reference)
                proposed_frozen = True
                persistent_action = (
                    "recovery_pending" if proposed_recovery else "remain_frozen"
                )
        reference = reference.to(device=device, dtype=dtype)
        older_rounds = [item.round_num for item in older_snapshots]
        newer_rounds = [item.round_num for item in newer_snapshots]
        if set(older_rounds) & set(newer_rounds):
            raise RuntimeError("RCIG old and new windows unexpectedly overlap")
        if max(newer_rounds) >= round_num:
            raise RuntimeError("RCIG attempted to read the deployment round")
        diagnostics: dict[str, Any] = {
            **common,
            **{f"rcig_{key}": value for key, value in gate_metrics.items()},
            "rcig_gate_frozen": self._frozen_gate is not None,
            "rcig_gate_source_round_min": min(gate_rounds) if gate_rounds else None,
            "rcig_gate_source_round_max": max(gate_rounds) if gate_rounds else None,
            "rcig_older_round_min": min(older_rounds),
            "rcig_older_round_max": max(older_rounds),
            "rcig_newer_round_min": min(newer_rounds),
            "rcig_newer_round_max": max(newer_rounds),
            "rcig_windows_disjoint": True,
            "rcig_newer_view_trust": trust,
            "rcig_gate_active": bool(fusion_metrics["gate_active"]),
            "rcig_innovation_norm": float(fusion_metrics["innovation_norm"]),
            "rcig_standardized_innovation": fusion_metrics.get(
                "standardized_innovation"
            ),
            "rcig_reference_norm": float(torch.linalg.vector_norm(reference).item()),
            "rcig_reference_norm_cap": float(self.config.influence_cap),
            "rcig_reference_norm_cap_respected": True,
            "rcig_older_view": older_metrics,
            "rcig_newer_view": newer_metrics,
            "rcig_older_covariance_anisotropy_ratio": float(
                older_metrics["covariance_anisotropy_ratio"]
            ),
            "rcig_newer_covariance_anisotropy_ratio": float(
                newer_metrics["covariance_anisotropy_ratio"]
            ),
            "rcig_max_covariance_anisotropy_ratio": float(
                max(
                    older_metrics["covariance_anisotropy_ratio"],
                    newer_metrics["covariance_anisotropy_ratio"],
                )
            ),
            "rcig_local_dp_effect": ("unchanged_post_processing_of_private_transcript"),
            # Exactly one preconfigured candidate drives deployment.  The
            # remaining candidates are evaluated only as paired, post-hoc
            # counterfactuals and never enter the deployed decision.
            "rcig_selected_candidate_drives_decision": True,
            "rcig_counterfactual_candidates_drive_decision": False,
            "rcig_deployed_candidate": configured_candidate,
            "rcig_paired_counterfactual_metrics_drive_deployment": False,
            "rcig_candidate_trust_full": mode_trust["full"],
            "rcig_candidate_trust_isotropic": mode_trust["isotropic"],
            "rcig_candidate_trust_euclidean": mode_trust["euclidean"],
            "rcig_full_newer_trust": mode_trust["full"],
            "rcig_isotropic_newer_trust": mode_trust["isotropic"],
            "rcig_euclidean_newer_trust": mode_trust["euclidean"],
            "rcig_full_innovation_stat": float(
                mode_diagnostics["full"]["standardized_innovation"]
            ),
            "rcig_isotropic_innovation_stat": float(
                mode_diagnostics["isotropic"]["standardized_innovation"]
            ),
            "rcig_euclidean_innovation_stat": float(
                mode_diagnostics["euclidean"]["innovation_norm"]
            ),
            "rcig_candidate_gate_active_full": bool(
                mode_diagnostics["full"]["gate_active"]
            ),
            "rcig_candidate_gate_active_isotropic": bool(
                mode_diagnostics["isotropic"]["gate_active"]
            ),
            "rcig_candidate_gate_active_euclidean": bool(
                mode_diagnostics["euclidean"]["gate_active"]
            ),
            "rcig_full_gate_active": bool(mode_diagnostics["full"]["gate_active"]),
            "rcig_isotropic_gate_active": bool(
                mode_diagnostics["isotropic"]["gate_active"]
            ),
            "rcig_euclidean_gate_active": bool(
                mode_diagnostics["euclidean"]["gate_active"]
            ),
            "rcig_persistent_policy": self.config.persistent_policy,
            "rcig_persistent_frozen_before_round": self._persistent_frozen,
            "rcig_persistent_frozen_after_commit": proposed_frozen,
            "rcig_recovery_count_before_round": self._recovery_count,
            "rcig_recovery_count_after_commit": proposed_recovery,
            "rcig_persistent_action": persistent_action,
            "rcig_recovery_compares_newer_to_last_accepted": True,
            "rcig_recovery_covariance_proxy": ("older_plus_newer_public_covariance"),
        }
        return RCIGReferenceResult(
            reference=reference,
            ready=True,
            diagnostics=diagnostics,
            older_view=older.to(device=device, dtype=dtype),
            newer_view=newer.to(device=device, dtype=dtype),
            covariance_older=covariance_older,
            covariance_newer=covariance_newer,
            candidate_references={
                name: value.to(device=device, dtype=dtype)
                for name, value in candidate_references.items()
            },
            deployment_round=round_num,
            _persistent_transition=(
                (
                    None
                    if proposed_accepted is None
                    else proposed_accepted.detach().clone()
                ),
                proposed_frozen,
                proposed_recovery,
            ),
        )


__all__ = [
    "RCIGReferenceResult",
    "RCIGSnapshot",
    "RCIGTemporalConfig",
    "TemporalRCIGReferenceState",
]
