"""Fairness-Aware Reweighting (FAR) around a robust reference aggregator."""

from __future__ import annotations

import gc
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from hardware.flop_cost import round_compute_flops
from metrics.robustness import weight_diagnostics
from robustness.aggregators import (
    aggregate_vectors,
    centered_clipping_leave_one_out,
    clip_l2,
    noise_aware_centered_clipping,
)
from robustness.tensor_ops import score_subspace, stack_updates, unflatten_update

from .base import AggregateResult, register_algorithm
from .fedavg import FedAvg
from .noise_aware_scores import (
    DEBIASED_DISTANCE_SCORE_MODE,
    DIRECT_SCORE_MODES,
    ENERGY_SCORE_MODE,
    LOO_EXCESS_ROBUST_SCORE_MODE,
    NULL_MC_ROBUST_SCORE_MODE,
    NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
    NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
    ROBUST_DIRECT_SCORE_MODES,
    TEMPORAL_EMA_PROJECTION_SCORE_MODE,
    TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE,
    TEMPORAL_PROJECTION_SCORE_MODES,
    calibrated_independence_trust_scores,
    debiased_distance_scores,
    directional_trust_scores,
    effective_upload_noise_variances,
    excess_energy_scores,
    leave_one_out_mean_residual_variances,
    mean_reference_residual_variances,
    null_mc_moment_scores,
    orthogonalize_scores_against_public_scale,
    public_noise_scales,
    robust_novelty_scores,
    separate_novelty_trust_scores,
    standardize_distances,
    temporal_projection_scores,
    theil_sen_orthogonalize_scores_against_public_scale,
    tierwise_midrank_scores,
)
from .reference_utils import apply_delta, common_round_metrics


@register_algorithm("far")
class FAR(FedAvg):
    """FAR reference implementation from the supplied manuscript.

    A configurable Byzantine-robust rule ``F`` first yields ``g_F``.  FAR then
    computes ``d_i = ||g_i-g_F||`` and positive-tilt weights

    ``lambda_i = softmax(alpha * d_i)``.

    The final output is the weighted sum of the *original* submissions, not
    the reference itself.  Therefore FAR's robustness depends on its attack
    regime and ``alpha``; a positive tilt is not a universal defense.
    """

    description = "FAR: distance-to-robust-reference exponential reweighting."

    _UPDATE_MODES = {"multi_epoch_delta", "single_step_gradient"}

    def _far_reference_override(
        self,
        *,
        vectors: torch.Tensor,
        score_vectors: torch.Tensor,
        layout,
        client_updates,
        server_clip_factors: torch.Tensor,
        round_num: int,
        config: dict,
        output_radius: float | None,
    ) -> tuple[torch.Tensor, dict] | None:
        """Optional causal reference supplied by a specialised FAR variant.

        The hook runs after the current uploads have been flattened and after
        the optional server clipping.  Implementations may inspect those
        tensors only to prepare state that will be committed *after* the
        aggregate succeeds.  A causal implementation must construct the
        returned reference exclusively from state committed before this call.

        Returning ``None`` preserves FAR's historical reference dispatcher.
        Keeping this narrow hook avoids reimplementing FAR's score, weight,
        aggregation and diagnostic paths in every research variant.
        """

        del (
            vectors,
            score_vectors,
            layout,
            client_updates,
            server_clip_factors,
            round_num,
            config,
            output_radius,
        )
        return None

    @staticmethod
    def _prox_mu(config: dict) -> float:
        """Resolve FAR's proximal coefficient while accepting the short alias ``mu``."""

        value = float(config.get("far_prox_mu", config.get("mu", 0.0)))
        if value < 0:
            raise ValueError("far_prox_mu must be non-negative")
        return value

    def client_update(self, model, dataloader, state, config):
        """Produce either a proximal local delta or one empirical gradient.

        ``multi_epoch_delta`` is the practical FedAvg/FedProx-style mode used
        by the internship protocol.  ``single_step_gradient`` evaluates one
        full local empirical gradient at the received global model and leaves
        the server step size explicit.  At that anchor the proximal gradient
        is mathematically zero; ``mu`` only affects trajectories containing
        more than one local optimisation step.
        """

        mode = str(config.get("far_update_mode", "multi_epoch_delta")).lower()
        if mode not in self._UPDATE_MODES:
            raise ValueError(
                f"far_update_mode must be one of {sorted(self._UPDATE_MODES)}, got {mode!r}"
            )
        if mode == "single_step_gradient":
            return self._single_step_gradient(model, dataloader, state, config)
        return self._multi_epoch_prox_delta(model, dataloader, state, config)

    def _multi_epoch_prox_delta(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        lr = float(config.get("lr", 0.01))
        local_epochs = int(config.get("local_epochs", 1))
        mu = self._prox_mu(config)
        max_grad_norm = config.get("max_grad_norm")

        before = OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in model.state_dict().items()
        )
        anchor = {
            name: parameter.detach().to(device).clone()
            for name, parameter in model.named_parameters()
        }
        model.to(device).train()
        optimizer_type = str(config.get("optimizer", "sgd")).lower()
        momentum = float(config.get("momentum", 0.9))
        weight_decay = float(config.get("weight_decay", 1e-4))
        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        criterion = nn.CrossEntropyLoss()
        total_task_loss = 0.0
        total_objective = 0.0
        num_batches = 0
        max_local_batches = config.get("max_local_batches")
        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                if max_local_batches is not None and batch_idx >= int(
                    max_local_batches
                ):
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                task_loss = criterion(model(x), y)
                prox_sq = torch.zeros((), device=device)
                if mu > 0:
                    for name, parameter in model.named_parameters():
                        prox_sq = prox_sq + (parameter - anchor[name]).square().sum()
                objective = task_loss + 0.5 * mu * prox_sq
                objective.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(max_grad_norm)
                    )
                optimizer.step()
                total_task_loss += float(task_loss.detach().item())
                total_objective += float(objective.detach().item())
                num_batches += 1

        current = model.state_dict()
        update = OrderedDict(
            (name, (before[name] - current[name].detach().cpu()).float())
            for name in before
        )
        metadata = self._finalize_client_metadata(
            model=model,
            dataloader=dataloader,
            state=state,
            config=config,
            update=update,
            local_epochs=local_epochs,
            local_loss=total_task_loss / max(num_batches, 1),
            extra={
                "far_update_mode": "multi_epoch_delta",
                "far_prox_mu": mu,
                "far_prox_active": bool(mu > 0 and num_batches > 1),
                "far_local_objective": total_objective / max(num_batches, 1),
                "far_local_steps": num_batches,
            },
        )
        del optimizer, anchor, before, current
        gc.collect()
        return dict(update), metadata

    def _single_step_gradient(self, model, dataloader, state, config):
        device = str(config.get("device", "cpu"))
        mu = self._prox_mu(config)
        model.to(device).train()
        parameters = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        gradient_sums = {
            name: torch.zeros_like(p, device=device) for name, p in parameters
        }
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_examples = 0
        num_batches = 0
        max_local_batches = config.get("max_local_batches")
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_local_batches is not None and batch_idx >= int(max_local_batches):
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            loss_sum = criterion(model(x), y)
            loss_sum.backward()
            for name, parameter in parameters:
                gradient_sums[name].add_(parameter.grad.detach())
            total_loss += float(loss_sum.detach().item())
            total_examples += int(y.numel())
            num_batches += 1

        # FAR's paper notation aggregates gradients and applies one server
        # step. Buffers have no gradient, so they receive explicit zeros.
        update = OrderedDict()
        parameter_names = {name for name, _ in parameters}
        for name, value in model.state_dict().items():
            if name in parameter_names:
                # application of clipping and server step size is left to the server; the client only computes the empirical gradient
                update[name] = (
                    (gradient_sums[name] / max(total_examples, 1)).cpu().float()
                )
            else:
                update[name] = torch.zeros_like(value, device="cpu").float()
        metadata = self._finalize_client_metadata(
            model=model,
            dataloader=dataloader,
            state=state,
            config=config,
            update=update,
            local_epochs=1,
            local_loss=total_loss / max(total_examples, 1),
            extra={
                "far_update_mode": "single_step_gradient",
                "far_prox_mu": mu,
                "far_prox_active": False,
                "far_prox_note": "zero_at_global_anchor_for_one_gradient_evaluation",
                "far_local_steps": 1,
                "far_gradient_batches": num_batches,
            },
        )
        return dict(update), metadata

    def _finalize_client_metadata(
        self,
        *,
        model,
        dataloader,
        state,
        config,
        update,
        local_epochs,
        local_loss,
        extra,
    ):
        uplink_bytes = self.count_bytes(update, sparse=False)
        downlink_bytes = uplink_bytes
        profile = config.get("device_profile")
        if profile:
            flops = round_compute_flops(
                model,
                [name for name, _ in model.named_parameters()],
                config,
                profile,
                dataloader,
                local_epochs,
            )
            breakdown = profile.round_energy_breakdown(
                flops,
                uplink_bytes,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            energy = 2.5 * float(config.get("energy_scale_factor", 1.0))
            breakdown = {
                "compute": energy,
                "uplink": 0.0,
                "downlink": 0.0,
                "total": energy,
            }
        state.battery_j = max(0.0, state.battery_j - breakdown["total"])
        state.round_num += 1
        return {
            "client_id": state.client_id,
            "round_num": state.round_num,
            "beta_actual": 1.0,
            "battery_j_remaining": state.battery_j,
            "energy_j_consumed": breakdown["total"],
            "energy_compute_j": breakdown["compute"],
            "energy_uplink_j": breakdown["uplink"],
            "energy_downlink_j": breakdown["downlink"],
            "bytes_sent": uplink_bytes,
            "bytes_received": downlink_bytes,
            "local_loss": float(local_loss),
            "compression_ratio": 1.0,
            "dataset_size": len(dataloader.dataset),
            **extra,
        }

    @staticmethod
    def far_weights(distances: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.softmax(float(alpha) * distances, dim=0)

    @staticmethod
    def far_scores(
        distances: torch.Tensor, config: dict
    ) -> tuple[torch.Tensor, float | None]:
        """Resolve FAR's score while preserving the manuscript default.

        ``raw_distance`` is the historical/original FAR rule.  The explicit
        ``bounded_normalized`` diagnostic uses scores in [0,1] and exists so
        current-round FAR can be compared with DT-LDP-FAR without confounding
        causal timing with a different distance scale.  It is never enabled
        silently.
        """

        mode = str(config.get("far_score_mode", "raw_distance")).lower()
        if mode == "raw_distance":
            return distances, None
        if mode != "bounded_normalized":
            raise ValueError(
                "far_score_mode must be 'raw_distance' or 'bounded_normalized'"
            )
        distance_clip = float(
            config.get("far_distance_clip", config.get("distance_clip", 0.0))
        )
        if distance_clip <= 0.0:
            raise ValueError(
                "bounded_normalized FAR scores require far_distance_clip > 0"
            )
        return (distances / distance_clip).clamp(min=0.0, max=1.0), distance_clip

    @staticmethod
    def _pearson_or_none(x: torch.Tensor, y: torch.Tensor) -> float | None:
        """Small aggregate-only diagnostic; individual values are not persisted."""

        if x.numel() < 2 or y.numel() != x.numel():
            return None
        x_centered = x.double() - x.double().mean()
        y_centered = y.double() - y.double().mean()
        denominator = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(
            y_centered
        )
        if float(denominator.item()) <= 1e-15:
            return None
        return float((x_centered @ y_centered / denominator).item())

    @staticmethod
    def _top_fraction_recall(
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        fraction: float,
    ) -> float | None:
        """Recall of the target's largest-score set under predicted scores."""

        if predicted.ndim != 1 or target.shape != predicted.shape:
            raise ValueError("Top-fraction recall requires aligned score vectors")
        if predicted.numel() == 0:
            return None
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Top-fraction recall fraction must lie in (0,1]")
        count = min(predicted.numel(), max(1, math.ceil(fraction * predicted.numel())))
        predicted_top = set(torch.topk(predicted, count).indices.cpu().tolist())
        target_top = set(torch.topk(target, count).indices.cpu().tolist())
        return len(predicted_top & target_top) / count

    def server_aggregate(self, global_model, client_updates, round_num, config):
        updates = [update for update, _, _ in client_updates]
        vectors, layout = stack_updates(updates)
        configured_server_clip = config.get("far_server_clip_norm")
        far_server_clip_rate = 0.0
        factors = torch.ones(
            vectors.shape[0], device=vectors.device, dtype=vectors.dtype
        )
        if configured_server_clip is not None:
            server_clip = float(configured_server_clip)
            if server_clip <= 0.0:
                raise ValueError("far_server_clip_norm must be positive when set")
            norms = torch.linalg.vector_norm(vectors, dim=1)
            factors = (server_clip / norms.clamp_min(1e-12)).clamp(max=1.0)
            vectors = vectors * factors[:, None]
            far_server_clip_rate = float((factors < 1.0).float().mean().item())
        score_vectors, score_space_metrics = score_subspace(
            vectors,
            layout,
            mode=str(config.get("score_subspace_mode", "full")),
            dimension=config.get("score_subspace_dimension"),
            seed=int(config.get("score_subspace_seed", 0)),
        )
        reference_name = str(config.get("robust_reference", "cm_nnm"))
        # Centered clipping and the regularised Huber ablation are anchored
        # references, unlike CM/NNM, trimmed mean or RFA.  The original FAR
        # dispatcher did not pass their mandatory public anchor/radius and
        # therefore crashed as soon as either reference was selected.  Keep
        # the anchor causal: round t reads only the anchor committed after
        # round t-1, then updates it after F_t has been evaluated.
        if round_num == 0 or not hasattr(self, "_far_reference_anchor"):
            self._far_reference_anchor = torch.zeros(
                score_vectors.shape[1], dtype=vectors.dtype, device="cpu"
            )
        if self._far_reference_anchor.numel() != score_vectors.shape[1]:
            raise ValueError("FAR score subspace changed during a run")
        anchor = self._far_reference_anchor.to(score_vectors)
        reference_radius = float(
            config.get("reference_clip_radius", config.get("server_clip_norm", 1.0))
        )
        reference_output_radius = config.get("far_reference_output_clip_norm")
        if reference_output_radius is not None:
            reference_output_radius = float(reference_output_radius)
            if reference_output_radius <= 0.0:
                raise ValueError("far_reference_output_clip_norm must be positive")
        reference_override = self._far_reference_override(
            vectors=vectors,
            score_vectors=score_vectors,
            layout=layout,
            client_updates=client_updates,
            server_clip_factors=factors,
            round_num=round_num,
            config=config,
            output_radius=reference_output_radius,
        )
        override_reference_metrics: dict = {}
        if reference_override is not None:
            reference, override_reference_metrics = reference_override
            if not isinstance(reference, torch.Tensor) or reference.ndim != 1:
                raise ValueError("A FAR reference override must return a vector")
            if reference.shape != score_vectors.shape[1:]:
                raise ValueError(
                    "A FAR reference override must match the score-space dimension"
                )
            reference = reference.to(score_vectors)
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("A FAR reference override must be finite")
            if not isinstance(override_reference_metrics, dict):
                raise TypeError("FAR reference override diagnostics must be a mapping")
        noise_aware_reference_names = {
            "noise_aware_centered_clipping",
            "noise_aware_cc",
            "na_cc",
            "f_na_cc",
        }
        reference_noise_variances = None
        reference_noise_metrics = {}
        if (
            reference_override is None
            and reference_name.lower() in noise_aware_reference_names
        ):
            reference_noise_variances, reference_noise_metrics = (
                effective_upload_noise_variances(
                    client_updates,
                    config=config,
                    server_clip_factors=factors,
                    device=score_vectors.device,
                    dtype=score_vectors.dtype,
                )
            )
        reference_kwargs = {
            "num_byzantine": int(config.get("num_byzantine", 0)),
            "screening_fraction": config.get("screening_fraction"),
            "max_iter": int(config.get("rfa_max_iter", 100)),
            "tol": float(config.get("rfa_tol", 1e-6)),
            "alpha_trusted": float(config.get("cmls_alpha_trusted", 1.0)),
            "alpha_suspected": float(config.get("cmls_alpha_suspected", 1.0)),
            "anchor": anchor,
            "tau": reference_radius,
            "gamma": float(config.get("huber_gamma", 1.0)),
            "num_steps": int(config.get("huber_num_steps", 10)),
            "noise_variances": reference_noise_variances,
            "max_weight_ratio": float(config.get("noise_aware_reference_kappa", 2.0)),
            "variance_floor": float(
                config.get("noise_aware_reference_variance_floor", 1e-12)
            ),
            "output_radius": reference_output_radius,
        }

        def build_score_reference(candidate_vectors: torch.Tensor) -> torch.Tensor:
            if reference_override is not None:
                # A temporal/causal reference is fixed by the already-private
                # past transcript.  In particular, the simulator-only clean
                # counterfactual must not recompute it from current uploads.
                return reference
            candidate = aggregate_vectors(
                candidate_vectors,
                reference_name,
                **reference_kwargs,
            )
            return (
                clip_l2(candidate, reference_output_radius)
                if reference_output_radius is not None
                else candidate
            )

        if reference_override is not None:
            noise_aware_reference_metrics = {}
        elif reference_name.lower() in noise_aware_reference_names:
            reference, noise_aware_reference_metrics = noise_aware_centered_clipping(
                score_vectors,
                anchor=anchor,
                tau=reference_radius,
                noise_variances=reference_noise_variances,
                max_weight_ratio=float(config.get("noise_aware_reference_kappa", 2.0)),
                variance_floor=float(
                    config.get("noise_aware_reference_variance_floor", 1e-12)
                ),
                output_radius=reference_output_radius,
                return_diagnostics=True,
            )
        else:
            reference = build_score_reference(score_vectors)
            noise_aware_reference_metrics = {}
        anchor_rate = float(config.get("anchor_update_rate", 0.1))
        if not 0.0 <= anchor_rate <= 1.0:
            raise ValueError("anchor_update_rate must lie in [0,1]")
        anchor_norm = float(
            config.get("anchor_clip_norm", config.get("server_clip_norm", 1.0))
        )
        if reference_override is None:
            self._far_reference_anchor = (
                clip_l2(
                    (1.0 - anchor_rate) * anchor + anchor_rate * reference,
                    anchor_norm,
                )
                .detach()
                .cpu()
            )
        noise_score_mode = str(
            config.get("noise_score_standardization", "none")
        ).lower()
        score_references: torch.Tensor = reference
        if noise_score_mode in {
            LOO_EXCESS_ROBUST_SCORE_MODE,
            *TEMPORAL_PROJECTION_SCORE_MODES,
        }:
            if reference_name.lower() not in {"centered_clipping", "cc", "f_cc"}:
                raise ValueError(
                    "The leave-one-out robust score requires centered_clipping"
                )
            score_references = centered_clipping_leave_one_out(
                score_vectors,
                anchor=anchor,
                tau=reference_radius,
            )
        raw_distances = torch.linalg.vector_norm(
            score_vectors - score_references,
            dim=1,
        )
        score_public_scales = (
            public_noise_scales(
                client_updates,
                device=raw_distances.device,
                dtype=raw_distances.dtype,
            )
            if noise_score_mode != "none"
            else None
        )
        residual_noise_variances = None
        if noise_score_mode in DIRECT_SCORE_MODES | ROBUST_DIRECT_SCORE_MODES:
            upload_noise_variances, covariance_metrics = (
                effective_upload_noise_variances(
                    client_updates,
                    config=config,
                    server_clip_factors=factors,
                    device=raw_distances.device,
                    dtype=raw_distances.dtype,
                )
            )
            residual_noise_variances = (
                leave_one_out_mean_residual_variances(upload_noise_variances)
                if noise_score_mode
                in {LOO_EXCESS_ROBUST_SCORE_MODE, *TEMPORAL_PROJECTION_SCORE_MODES}
                else mean_reference_residual_variances(upload_noise_variances)
            )
            score_dimension = int(score_space_metrics["score_subspace_dimension"])
            if noise_score_mode == ENERGY_SCORE_MODE:
                score_distance_clip = float(config.get("noise_score_z_clip", 5.0))
                scores, direct_score_metrics = excess_energy_scores(
                    raw_distances,
                    residual_noise_variances,
                    score_dimension=score_dimension,
                    z_clip=score_distance_clip,
                )
            elif noise_score_mode == DEBIASED_DISTANCE_SCORE_MODE:
                score_distance_clip = float(
                    config.get("far_distance_clip", config.get("distance_clip", 0.0))
                )
                scores, direct_score_metrics = debiased_distance_scores(
                    raw_distances,
                    residual_noise_variances,
                    score_dimension=score_dimension,
                    distance_clip=score_distance_clip,
                )
            elif noise_score_mode in TEMPORAL_PROJECTION_SCORE_MODES:
                if round_num == 0 or not hasattr(self, "_far_temporal_history"):
                    self._far_temporal_history: dict[int, torch.Tensor] = {}
                client_ids = [
                    int(metadata["client_id"]) for _, metadata, _ in client_updates
                ]
                if len(set(client_ids)) != len(client_ids):
                    raise ValueError(
                        "Temporal FAR scoring requires unique client_id metadata"
                    )
                residual_vectors = score_vectors - score_references
                predictable_rows = []
                for client_id in client_ids:
                    previous = self._far_temporal_history.get(client_id)
                    if previous is None:
                        predictable_rows.append(torch.zeros_like(residual_vectors[0]))
                    else:
                        if previous.numel() != residual_vectors.shape[1]:
                            raise ValueError(
                                "Temporal FAR score subspace changed during a run"
                            )
                        predictable_rows.append(previous.to(residual_vectors))
                predictable_vectors = torch.stack(predictable_rows)
                scores, direct_score_metrics = temporal_projection_scores(
                    residual_vectors,
                    residual_noise_variances,
                    predictable_vectors,
                    lower_z=float(config.get("noise_score_temporal_lower_z", 0.0)),
                    upper_z=float(config.get("noise_score_temporal_upper_z", 3.0)),
                )
                decay = float(config.get("noise_score_temporal_ema_decay", 0.5))
                if not 0.0 <= decay < 1.0:
                    raise ValueError("noise_score_temporal_ema_decay must lie in [0,1)")
                next_history = dict(self._far_temporal_history)
                for index, client_id in enumerate(client_ids):
                    current = residual_vectors[index].detach().cpu()
                    previous = self._far_temporal_history.get(client_id)
                    if (
                        noise_score_mode == TEMPORAL_EMA_PROJECTION_SCORE_MODE
                        and previous is not None
                    ):
                        current = decay * previous + (1.0 - decay) * current
                    next_history[client_id] = current
                self._far_temporal_history = next_history
                direct_score_metrics.update(
                    {
                        "noise_score_temporal_history_rule": (
                            "previous_round"
                            if noise_score_mode
                            == TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE
                            else "exponential_moving_average"
                        ),
                        "noise_score_temporal_ema_decay": (
                            decay
                            if noise_score_mode == TEMPORAL_EMA_PROJECTION_SCORE_MODE
                            else 0.0
                        ),
                        "noise_score_temporal_history_size": float(
                            len(self._far_temporal_history)
                        ),
                    }
                )
                score_distance_clip = 1.0
            else:
                calibration_draws = int(config.get("noise_score_null_mc_draws", 64))
                calibration_seed = int(
                    config.get("noise_score_null_mc_seed", 20260906)
                ) + 1009 * int(round_num)
                trust_floor = float(config.get("noise_score_trust_floor", 0.05))
                null_mc_modes = {
                    NULL_MC_ROBUST_SCORE_MODE,
                    NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE,
                    NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE,
                }
                if noise_score_mode in null_mc_modes:
                    base_scores, score_metrics = null_mc_moment_scores(
                        raw_distances,
                        upload_noise_variances,
                        score_dimension=score_dimension,
                        reference_builder=build_score_reference,
                        null_center=reference,
                        calibration_draws=calibration_draws,
                        calibration_seed=calibration_seed,
                        z_clip=float(config.get("noise_score_z_clip", 4.0)),
                    )
                    independence, trust_metrics = calibrated_independence_trust_scores(
                        score_vectors,
                        upload_noise_variances,
                        calibration_draws=calibration_draws,
                        calibration_seed=calibration_seed + 1,
                        low_null_quantile=float(
                            config.get("noise_score_independence_low_quantile", 0.05)
                        ),
                        full_trust_null_quantile=float(
                            config.get("noise_score_independence_full_quantile", 0.50)
                        ),
                        trust_floor=trust_floor,
                    )
                    if noise_score_mode == NULL_MC_ROBUST_SCORE_MODE:
                        scores = robust_novelty_scores(
                            base_scores,
                            independence,
                            trust_floor=0.0,
                        )
                        direct_score_metrics = {**score_metrics, **trust_metrics}
                    else:
                        if (
                            noise_score_mode
                            == NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE
                        ):
                            novelty, calibration_metrics = (
                                theil_sen_orthogonalize_scores_against_public_scale(
                                    base_scores,
                                    score_public_scales,
                                    lower_quantile=float(
                                        config.get(
                                            "noise_score_robust_lower_quantile",
                                            0.05,
                                        )
                                    ),
                                    upper_quantile=float(
                                        config.get(
                                            "noise_score_robust_upper_quantile",
                                            0.95,
                                        )
                                    ),
                                )
                            )
                        else:
                            novelty, calibration_metrics = tierwise_midrank_scores(
                                base_scores,
                                score_public_scales,
                            )
                        direction, direction_metrics = directional_trust_scores(
                            score_vectors,
                            score_references,
                            reject_cosine=float(
                                config.get("noise_score_reject_cosine", -0.10)
                            ),
                            full_trust_cosine=float(
                                config.get("noise_score_full_trust_cosine", 0.25)
                            ),
                        )
                        combined_trust = torch.minimum(direction, independence)
                        scores, composition_metrics = separate_novelty_trust_scores(
                            novelty,
                            combined_trust,
                            trust_fraction=float(
                                config.get("noise_score_trust_logit_fraction", 0.25)
                            ),
                        )
                        direct_score_metrics = {
                            **score_metrics,
                            **trust_metrics,
                            **direction_metrics,
                            **calibration_metrics,
                            **composition_metrics,
                            "noise_score_combined_trust_rule": "minimum",
                        }
                else:
                    base_scores, score_metrics = excess_energy_scores(
                        raw_distances,
                        residual_noise_variances,
                        score_dimension=score_dimension,
                        z_clip=float(config.get("noise_score_z_clip", 5.0)),
                    )
                    direction, direction_metrics = directional_trust_scores(
                        score_vectors,
                        score_references,
                        reject_cosine=float(
                            config.get("noise_score_reject_cosine", -0.10)
                        ),
                        full_trust_cosine=float(
                            config.get("noise_score_full_trust_cosine", 0.25)
                        ),
                    )
                    scores = robust_novelty_scores(
                        base_scores,
                        direction,
                        trust_floor=trust_floor,
                    )
                    independence, trust_metrics = calibrated_independence_trust_scores(
                        score_vectors,
                        upload_noise_variances,
                        calibration_draws=calibration_draws,
                        calibration_seed=calibration_seed + 1,
                        low_null_quantile=float(
                            config.get("noise_score_independence_low_quantile", 0.05)
                        ),
                        full_trust_null_quantile=float(
                            config.get("noise_score_independence_full_quantile", 0.50)
                        ),
                        trust_floor=trust_floor,
                    )
                    scores = robust_novelty_scores(
                        scores,
                        independence,
                        trust_floor=0.0,
                    )
                    direct_score_metrics = {
                        **score_metrics,
                        **direction_metrics,
                        **trust_metrics,
                    }
                if noise_score_mode in {
                    NULL_MC_ROBUST_SCORE_MODE,
                    LOO_EXCESS_ROBUST_SCORE_MODE,
                }:
                    scores, orthogonalization_metrics = (
                        orthogonalize_scores_against_public_scale(
                            scores,
                            score_public_scales,
                        )
                    )
                    direct_score_metrics.update(orthogonalization_metrics)
                score_distance_clip = 1.0
            distances = raw_distances
            noise_score_divisors = residual_noise_variances.sqrt()
            noise_score_metrics = {
                "noise_score_standardization": noise_score_mode,
                "noise_score_residual_covariance_model": (
                    "leave_one_out_mean_reference_proxy"
                    if noise_score_mode
                    in {LOO_EXCESS_ROBUST_SCORE_MODE, *TEMPORAL_PROJECTION_SCORE_MODES}
                    else "mean_reference_proxy"
                ),
                "noise_score_residual_variance_min": float(
                    residual_noise_variances.min().item()
                ),
                "noise_score_residual_variance_mean": float(
                    residual_noise_variances.mean().item()
                ),
                "noise_score_residual_variance_max": float(
                    residual_noise_variances.max().item()
                ),
                **covariance_metrics,
                **direct_score_metrics,
            }
        else:
            distances, noise_score_divisors, noise_score_metrics = (
                standardize_distances(
                    raw_distances,
                    score_public_scales,
                    mode=noise_score_mode,
                    reference_variance_factor=config.get(
                        "noise_score_reference_variance_factor"
                    ),
                    variance_ridge=float(
                        config.get("noise_score_variance_ridge", 1e-12)
                    ),
                )
            )
            scores, score_distance_clip = self.far_scores(distances, config)
        far_alpha = float(config.get("far_alpha", 0.1))
        weights = self.far_weights(scores, far_alpha)
        aggregate_vector = (weights[:, None] * vectors).sum(dim=0)
        aggregate = dict(unflatten_update(aggregate_vector, layout))
        modes = {
            str(
                metadata.get(
                    "far_update_mode",
                    config.get("far_update_mode", "multi_epoch_delta"),
                )
            )
            for _, metadata, _ in client_updates
        }
        if len(modes) != 1:
            raise ValueError(f"FAR received mixed client update modes: {sorted(modes)}")
        update_mode = modes.pop()
        if update_mode == "single_step_gradient":
            configured_server_lr = config.get("far_server_lr")
            server_lr = float(
                config.get("lr", 0.01)
                if configured_server_lr is None
                else configured_server_lr
            )
            aggregate = {name: server_lr * value for name, value in aggregate.items()}
        else:
            server_lr = 1.0

        external_attack_diagnostics = bool(
            config.get("external_attack_diagnostics", False)
        )
        if external_attack_diagnostics:
            if "attack" in config:
                raise ValueError("FAR server received external attack configuration")
            leaked_attack_fields = sorted(
                {
                    str(name)
                    for _, metadata, _ in client_updates
                    for name in metadata
                    if str(name).lower() == "is_byzantine"
                    or str(name).lower().startswith("attack_")
                }
            )
            if leaked_attack_fields:
                raise ValueError(
                    "FAR server received external attack-oracle fields: "
                    + ", ".join(leaked_attack_fields)
                )
            diagnostic_client_ids = [
                int(metadata["client_id"]) for _, metadata, _ in client_updates
            ]
            if len(diagnostic_client_ids) != len(set(diagnostic_client_ids)):
                raise ValueError(
                    "external FAR diagnostics require unique authenticated client ids"
                )
            malicious_mask = None
            diagnostics = weight_diagnostics(weights)
            # This transient payload contains no attack label.  The experiment
            # harness removes it immediately after aggregation, then joins the
            # weights with simulator truth outside the deployed mechanism.
            diagnostics["_far_external_weight_diagnostics_payload"] = {
                "client_ids": diagnostic_client_ids,
                "weights": weights.detach()
                .to(dtype=torch.float64, device="cpu")
                .tolist(),
                "contains_attack_labels": False,
            }
            diagnostics.update(
                {
                    "far_attack_labels_visible_to_server_aggregate": False,
                    "far_attack_config_visible_to_server_aggregate": False,
                    "far_external_attack_diagnostics": True,
                    "far_external_attack_diagnostics_boundary": (
                        "posthoc_simulator_only"
                    ),
                }
            )
        else:
            malicious_mask = torch.tensor(
                [bool(meta.get("is_byzantine", False)) for _, meta, _ in client_updates]
            )
            diagnostics = weight_diagnostics(weights, malicious_mask)
        weight_l2_squared = float(weights.double().square().sum().item())
        kappa_w = float(config.get("kappa_w", 0.0))
        bounded_score_mode = score_distance_clip is not None
        valid_kappa = 1.0 <= kappa_w < len(weights)
        configured_public_range = config.get("far_public_score_range")
        public_score_range = (
            1.0
            if bounded_score_mode
            else (
                float(configured_public_range)
                if configured_public_range is not None
                else None
            )
        )
        if public_score_range is not None and public_score_range <= 0.0:
            raise ValueError("far_public_score_range must be positive when set")
        alpha_cap = (
            math.log(kappa_w * (len(weights) - 1) / (len(weights) - kappa_w))
            / public_score_range
            if public_score_range is not None and valid_kappa and len(weights) > 1
            else None
        )
        certificate_claimed = bool(
            alpha_cap is not None and abs(far_alpha) <= alpha_cap + 1e-12
        )
        score_mode_label = {
            ENERGY_SCORE_MODE: "bounded_excess_energy",
            DEBIASED_DISTANCE_SCORE_MODE: "bounded_debiased_distance",
            NULL_MC_ROBUST_SCORE_MODE: "bounded_null_mc_robust",
            NULL_MC_THEIL_SEN_SEPARATE_TRUST_SCORE_MODE: (
                "bounded_null_mc_theil_sen_separate_trust"
            ),
            NULL_MC_TIER_ROBUST_SEPARATE_TRUST_SCORE_MODE: (
                "bounded_null_mc_tier_robust_separate_trust"
            ),
            LOO_EXCESS_ROBUST_SCORE_MODE: "bounded_loo_excess_robust",
            TEMPORAL_PREVIOUS_PROJECTION_SCORE_MODE: (
                "bounded_temporal_previous_projection"
            ),
            TEMPORAL_EMA_PROJECTION_SCORE_MODE: "bounded_temporal_ema_projection",
        }.get(noise_score_mode, str(config.get("far_score_mode", "raw_distance")))
        diagnostics.update(
            {
                "far_mean_distance": float(distances.mean().item()),
                "far_min_distance": float(distances.min().item()),
                "far_max_distance": float(distances.max().item()),
                "far_raw_distance_min": float(raw_distances.min().item()),
                "far_raw_distance_mean": float(raw_distances.mean().item()),
                "far_raw_distance_max": float(raw_distances.max().item()),
                "far_score_mode": score_mode_label,
                "far_score_min": float(scores.min().item()),
                "far_score_max": float(scores.max().item()),
                "far_score_span": float((scores.max() - scores.min()).item()),
                "far_score_saturation_rate": float(
                    (scores >= 1.0 - 1e-7).float().mean().item()
                    if score_distance_clip is not None
                    else 0.0
                ),
                "far_distance_clip": score_distance_clip,
                "far_score_scale": score_distance_clip,
                "far_public_score_range": public_score_range,
                "far_reference_norm": float(torch.linalg.vector_norm(reference).item()),
                "far_reference_anchor_norm": float(
                    torch.linalg.vector_norm(anchor).item()
                ),
                "far_reference_clip_radius": (
                    reference_radius
                    if reference_name.lower()
                    in {
                        "centered_clipping",
                        "cc",
                        "f_cc",
                        "noise_aware_centered_clipping",
                        "noise_aware_cc",
                        "na_cc",
                        "f_na_cc",
                        "regularized_huber",
                        "huber",
                        "huber_regularized",
                        "rcig_temporal",
                        "rcig_temporal_isotropic",
                        "rcig_temporal_euclidean",
                    }
                    else None
                ),
                "far_server_clip_norm": (
                    float(configured_server_clip)
                    if configured_server_clip is not None
                    else None
                ),
                "far_server_clip_rate": far_server_clip_rate,
                "far_alpha": far_alpha,
                "far_kappa_w": kappa_w if valid_kappa else None,
                "far_alpha_cap": alpha_cap,
                "far_weight_cap": kappa_w / len(weights) if valid_kappa else None,
                "far_weight_cap_respected": (
                    bool(float(weights.max().item()) <= kappa_w / len(weights) + 1e-10)
                    if valid_kappa
                    else None
                ),
                "far_tilt_influence_certificate_claimed": certificate_claimed,
                "far_weight_l2_squared": weight_l2_squared,
                "far_max_weight": float(weights.max().item()),
                "far_noise_amplification_vs_uniform": (
                    len(weights) * weight_l2_squared
                ),
                # The softmax ratio between the largest and smallest weight is
                # exp(logit_range).  This directly exposes the amplification
                # that motivates sensitivity-controlled FAR.
                # A range is non-negative even for negative alpha.  Its value
                # is exactly log(q_max/q_min), whereas the previous signed
                # expression mislabeled negative-alpha runs.
                "far_logit_range": float(
                    (abs(far_alpha) * (scores.max() - scores.min())).item()
                ),
                "far_weight_ratio": float(
                    (weights.max() / weights.clamp_min(1e-15).min()).item()
                ),
                "far_update_mode": update_mode,
                "far_server_lr": server_lr,
                "far_prox_mu": self._prox_mu(config),
                **{f"far_{key}": value for key, value in noise_score_metrics.items()},
                **{f"far_{key}": value for key, value in score_space_metrics.items()},
                **reference_noise_metrics,
                **noise_aware_reference_metrics,
                **override_reference_metrics,
            }
        )
        if malicious_mask is not None:
            diagnostics["far_num_byzantine_oracle"] = int(malicious_mask.sum().item())
        # Experimental diagnostic for the report's hypothesis that FAR may
        # amplify clients with larger realised DP perturbations.  The exact
        # client-level noise norms are simulation oracles and are never saved;
        # only the two cohort-level correlations are recorded.
        noise_norms = [
            metadata.get("dp_noise_norm_mean") for _, metadata, _ in client_updates
        ]
        if all(value is not None for value in noise_norms):
            noise_tensor = torch.tensor(noise_norms, dtype=torch.float64)
            diagnostics["far_weight_dp_noise_corr_oracle"] = self._pearson_or_none(
                weights, noise_tensor
            )
            diagnostics["far_distance_dp_noise_corr_oracle"] = self._pearson_or_none(
                distances, noise_tensor
            )
            diagnostics["far_raw_distance_dp_noise_corr_oracle"] = (
                self._pearson_or_none(raw_distances, noise_tensor)
            )
        clean_updates = [
            metadata.get("local_dp_noise_free_update_oracle")
            for _, metadata, _ in client_updates
        ]
        if all(isinstance(value, dict) for value in clean_updates):
            if malicious_mask is None:
                raise ValueError(
                    "clean-gradient oracle fields reached FAR while attack diagnostics "
                    "were externalized"
                )
            clean_vectors, clean_layout = stack_updates(clean_updates)
            if clean_layout.keys != layout.keys or clean_layout.shapes != layout.shapes:
                raise ValueError("Noise-free counterfactual layout mismatch")
            if configured_server_clip is not None:
                clean_norms = torch.linalg.vector_norm(clean_vectors, dim=1)
                clean_factors = (
                    float(configured_server_clip) / clean_norms.clamp_min(1e-12)
                ).clamp(max=1.0)
                clean_vectors = clean_vectors * clean_factors[:, None]
            clean_score_vectors, clean_score_metrics = score_subspace(
                clean_vectors,
                clean_layout,
                mode=str(config.get("score_subspace_mode", "full")),
                dimension=config.get("score_subspace_dimension"),
                seed=int(config.get("score_subspace_seed", 0)),
            )
            if (
                clean_score_metrics["score_subspace_dimension"]
                != score_space_metrics["score_subspace_dimension"]
            ):
                raise ValueError("Counterfactual score-space dimension mismatch")
            effective_full = vectors - clean_vectors
            effective_score = score_vectors - clean_score_vectors
            effective_full_norm = torch.linalg.vector_norm(effective_full, dim=1)
            effective_score_norm = torch.linalg.vector_norm(effective_score, dim=1)
            public_scales = torch.tensor(
                [
                    float(metadata.get("privacy_noise_multiplier_scale_public", 1.0))
                    for _, metadata, _ in client_updates
                ],
                dtype=torch.float64,
            )
            normalized_effective_score_norm = (
                effective_score_norm.double() / public_scales.clamp_min(1e-12)
            )
            # Reuse the exact same public reference rule and parameters. This
            # matters for noise-aware references whose public variance vector
            # is not part of the legacy aggregate_vectors argument list below.
            clean_reference = build_score_reference(clean_score_vectors)
            clean_score_references: torch.Tensor = clean_reference
            if noise_score_mode in {
                LOO_EXCESS_ROBUST_SCORE_MODE,
                *TEMPORAL_PROJECTION_SCORE_MODES,
            }:
                clean_score_references = centered_clipping_leave_one_out(
                    clean_score_vectors,
                    anchor=anchor,
                    tau=reference_radius,
                )
            clean_raw_distances = torch.linalg.vector_norm(
                clean_score_vectors - clean_score_references, dim=1
            )
            if noise_score_mode in DIRECT_SCORE_MODES | ROBUST_DIRECT_SCORE_MODES:
                if residual_noise_variances is None:
                    raise RuntimeError("Missing residual covariance for direct score")
                clean_noise_divisors = residual_noise_variances.sqrt()
                clean_distances = clean_raw_distances
                if noise_score_mode == ENERGY_SCORE_MODE:
                    clean_scores, _ = excess_energy_scores(
                        clean_raw_distances,
                        residual_noise_variances,
                        score_dimension=int(
                            score_space_metrics["score_subspace_dimension"]
                        ),
                        z_clip=float(config.get("noise_score_z_clip", 5.0)),
                        subtract_noise_floor=False,
                    )
                elif noise_score_mode == DEBIASED_DISTANCE_SCORE_MODE:
                    clean_scores, _ = debiased_distance_scores(
                        clean_raw_distances,
                        residual_noise_variances,
                        score_dimension=int(
                            score_space_metrics["score_subspace_dimension"]
                        ),
                        distance_clip=score_distance_clip,
                        subtract_noise_floor=False,
                    )
                elif noise_score_mode in ROBUST_DIRECT_SCORE_MODES:
                    # Target the bounded FAR geometry of noise-free uploads.
                    # The clean oracle intentionally applies no DP-noise
                    # correction: it is the signal the deployable score should
                    # recover, not a replay of the noisy calibration pipeline.
                    clean_scores, _ = self.far_scores(
                        clean_raw_distances,
                        config,
                    )
                else:  # pragma: no cover - guarded by the mode set above.
                    raise RuntimeError(
                        f"Unhandled direct score mode {noise_score_mode!r}"
                    )
            else:
                clean_distances, clean_noise_divisors, _ = standardize_distances(
                    clean_raw_distances,
                    public_scales.to(clean_raw_distances),
                    mode=noise_score_mode,
                    reference_variance_factor=config.get(
                        "noise_score_reference_variance_factor"
                    ),
                    variance_ridge=float(
                        config.get("noise_score_variance_ridge", 1e-12)
                    ),
                )
                clean_scores, _ = self.far_scores(clean_distances, config)
            if not torch.allclose(
                noise_score_divisors.double().cpu(),
                clean_noise_divisors.double().cpu(),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise RuntimeError("Public noise-score divisors changed in oracle path")
            clean_weights = self.far_weights(clean_scores, far_alpha).to(vectors)
            raw_clean_scores, _ = self.far_scores(clean_raw_distances, config)
            raw_clean_weights = self.far_weights(raw_clean_scores, far_alpha).to(
                vectors
            )
            noisy_aggregate = (weights[:, None] * vectors).sum(dim=0)
            fixed_weight_clean = (weights[:, None] * clean_vectors).sum(dim=0)
            counterfactual_clean = (clean_weights[:, None] * clean_vectors).sum(dim=0)
            fixed_weight_noise = noisy_aggregate - fixed_weight_clean
            total_fresh_effect = noisy_aggregate - counterfactual_clean
            reweighting_component = fixed_weight_clean - counterfactual_clean
            diagnostics.update(
                {
                    "far_effective_fresh_noise_norm_mean_oracle": float(
                        effective_full_norm.mean().item()
                    ),
                    "far_effective_fresh_score_noise_norm_mean_oracle": float(
                        effective_score_norm.mean().item()
                    ),
                    "far_weight_effective_noise_corr_oracle": self._pearson_or_none(
                        weights, effective_score_norm
                    ),
                    "far_weight_normalized_effective_noise_corr_oracle": (
                        self._pearson_or_none(weights, normalized_effective_score_norm)
                    ),
                    "far_fixed_weight_fresh_noise_sq_error_oracle": float(
                        fixed_weight_noise.square().sum().item()
                    ),
                    "far_total_fresh_noise_sq_error_oracle": float(
                        total_fresh_effect.square().sum().item()
                    ),
                    "far_reweighting_component_sq_norm_oracle": float(
                        reweighting_component.square().sum().item()
                    ),
                    "far_noisy_clean_weight_l1_oracle": float(
                        torch.linalg.vector_norm(weights - clean_weights, ord=1).item()
                    ),
                    "far_noisy_clean_score_corr_oracle": self._pearson_or_none(
                        scores, clean_scores
                    ),
                    "far_noisy_clean_score_mae_oracle": float(
                        (scores - clean_scores).abs().mean().item()
                    ),
                    "far_noisy_clean_score_rmse_oracle": float(
                        (scores - clean_scores).square().mean().sqrt().item()
                    ),
                    "far_noisy_clean_raw_distance_corr_oracle": (
                        self._pearson_or_none(raw_distances, clean_raw_distances)
                    ),
                    "far_score_public_noise_scale_corr_oracle": (
                        self._pearson_or_none(scores, public_scales.to(scores))
                    ),
                    "far_clean_score_public_noise_scale_corr_oracle": (
                        self._pearson_or_none(
                            clean_scores, public_scales.to(clean_scores)
                        )
                    ),
                    "far_noisy_raw_clean_score_corr_oracle": (
                        self._pearson_or_none(scores, raw_clean_scores)
                    ),
                    "far_noisy_raw_clean_score_mae_oracle": float(
                        (scores - raw_clean_scores).abs().mean().item()
                    ),
                    "far_noisy_raw_clean_score_rmse_oracle": float(
                        (scores - raw_clean_scores).square().mean().sqrt().item()
                    ),
                    "far_noisy_raw_clean_weight_l1_oracle": float(
                        torch.linalg.vector_norm(
                            weights - raw_clean_weights, ord=1
                        ).item()
                    ),
                    "far_raw_clean_score_public_noise_scale_corr_oracle": (
                        self._pearson_or_none(
                            raw_clean_scores, public_scales.to(raw_clean_scores)
                        )
                    ),
                }
            )
            honest_oracle_mask = (~malicious_mask).to(scores.device)
            if bool(honest_oracle_mask.any()):
                honest_noisy_scores = scores[honest_oracle_mask]
                honest_clean_scores = clean_scores[honest_oracle_mask]
                honest_noisy_weights = weights[honest_oracle_mask]
                honest_clean_weights = clean_weights[honest_oracle_mask]
                honest_public_scales = public_scales.to(scores)[honest_oracle_mask]
                # Exact post-server-clipping decomposition used only for
                # evaluation.  Let H be the honest cohort, q_H its total
                # weight, and mu_H the uniform mean of the noise-free honest
                # uploads.  Then
                #
                #   A - mu_H = q_H * B_tilt + B_byz + Z_H,
                #
                # where B_tilt is the clean honest reweighting displacement,
                # B_byz is the contribution of Byzantine uploads relative to
                # mu_H, and Z_H is the honest DP perturbation under the noisy
                # run's fixed weights.  This identity connects the empirical
                # diagnostics to the convergence-error decomposition; none of
                # these oracle quantities is available to the algorithm.
                honest_count = int(honest_oracle_mask.sum().item())
                honest_weight_mass = honest_noisy_weights.sum()
                conditioned_honest_weights = (
                    honest_noisy_weights
                    / honest_weight_mass.clamp_min(torch.finfo(weights.dtype).tiny)
                )
                uniform_honest_weights = torch.full_like(
                    conditioned_honest_weights,
                    1.0 / honest_count,
                )
                clean_honest_center = clean_vectors[honest_oracle_mask].mean(dim=0)
                honest_clean_tilting_bias = (
                    conditioned_honest_weights[:, None]
                    * (clean_vectors[honest_oracle_mask] - clean_honest_center[None, :])
                ).sum(dim=0)
                honest_fixed_weight_dp_noise = (
                    honest_noisy_weights[:, None] * effective_full[honest_oracle_mask]
                ).sum(dim=0)
                byzantine_oracle_mask = malicious_mask.to(vectors.device)
                if bool(byzantine_oracle_mask.any()):
                    byzantine_displacement = (
                        weights[byzantine_oracle_mask, None]
                        * (
                            vectors[byzantine_oracle_mask]
                            - clean_honest_center[None, :]
                        )
                    ).sum(dim=0)
                else:
                    byzantine_displacement = torch.zeros_like(clean_honest_center)
                aggregate_error_to_clean_honest_center = (
                    noisy_aggregate - clean_honest_center
                )
                decomposition_residual = (
                    aggregate_error_to_clean_honest_center
                    - honest_weight_mass * honest_clean_tilting_bias
                    - byzantine_displacement
                    - honest_fixed_weight_dp_noise
                )
                diagnostics.update(
                    {
                        "far_honest_noisy_clean_score_corr_oracle": (
                            self._pearson_or_none(
                                honest_noisy_scores,
                                honest_clean_scores,
                            )
                        ),
                        "far_honest_noisy_clean_score_mae_oracle": float(
                            (honest_noisy_scores - honest_clean_scores)
                            .abs()
                            .mean()
                            .item()
                        ),
                        "far_honest_noisy_clean_score_rmse_oracle": float(
                            (honest_noisy_scores - honest_clean_scores)
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "far_honest_clean_top_tail_recall_oracle": (
                            self._top_fraction_recall(
                                honest_noisy_scores,
                                honest_clean_scores,
                                fraction=float(
                                    config.get("fairness_tail_fraction", 0.2)
                                ),
                            )
                        ),
                        "far_honest_noisy_clean_weight_l1_oracle": float(
                            torch.linalg.vector_norm(
                                honest_noisy_weights - honest_clean_weights,
                                ord=1,
                            ).item()
                        ),
                        "far_honest_score_public_noise_scale_corr_oracle": (
                            self._pearson_or_none(
                                honest_noisy_scores,
                                honest_public_scales,
                            )
                        ),
                        "far_honest_clean_score_public_noise_scale_corr_oracle": (
                            self._pearson_or_none(
                                honest_clean_scores,
                                honest_public_scales,
                            )
                        ),
                        "far_honest_oracle_count": honest_count,
                        "far_honest_weight_mass_oracle": float(
                            honest_weight_mass.item()
                        ),
                        "far_honest_conditioned_weight_l1_from_uniform_oracle": float(
                            torch.linalg.vector_norm(
                                conditioned_honest_weights - uniform_honest_weights,
                                ord=1,
                            ).item()
                        ),
                        "far_honest_clean_tilting_bias_norm_oracle": float(
                            torch.linalg.vector_norm(honest_clean_tilting_bias).item()
                        ),
                        "far_honest_fixed_weight_dp_noise_norm_oracle": float(
                            torch.linalg.vector_norm(
                                honest_fixed_weight_dp_noise
                            ).item()
                        ),
                        "far_byzantine_displacement_norm_oracle": float(
                            torch.linalg.vector_norm(byzantine_displacement).item()
                        ),
                        "far_aggregate_error_to_clean_honest_center_norm_oracle": float(
                            torch.linalg.vector_norm(
                                aggregate_error_to_clean_honest_center
                            ).item()
                        ),
                        "far_error_decomposition_residual_norm_oracle": float(
                            torch.linalg.vector_norm(decomposition_residual).item()
                        ),
                    }
                )
        if (
            bool(config.get("enable_oracle_diagnostics", False))
            and malicious_mask is not None
        ):
            honest_mask = ~malicious_mask.to(score_vectors.device)
            if bool(honest_mask.any()):
                honest_center = score_vectors[honest_mask].mean(dim=0)
                diagnostics["far_reference_honest_center_error_oracle"] = float(
                    torch.linalg.vector_norm(reference - honest_center).item()
                )
        metrics = common_round_metrics(client_updates)
        metrics.update({"round": round_num, "robust_reference": reference_name})
        metrics.update(diagnostics)
        return AggregateResult(apply_delta(global_model, aggregate), metrics)

    def get_default_config(self):
        cfg = super().get_default_config()
        cfg.update(
            {
                "far_alpha": 0.1,
                "far_score_mode": "raw_distance",
                "far_distance_clip": None,
                "far_public_score_range": None,
                "far_reference_output_clip_norm": None,
                "far_server_clip_norm": None,
                "far_update_mode": "multi_epoch_delta",
                "far_prox_mu": 0.0,
                "far_server_lr": None,
                "robust_reference": "cm_nnm",
                "num_byzantine": 0,
                "screening_fraction": None,
                "rfa_max_iter": 100,
                "rfa_tol": 1e-6,
                "reference_clip_radius": 1.0,
                "noise_aware_reference_kappa": 2.0,
                "noise_aware_reference_variance_floor": 1e-12,
                "huber_gamma": 1.0,
                "huber_num_steps": 10,
                "anchor_update_rate": 0.1,
                "anchor_clip_norm": 1.0,
                "score_subspace_mode": "full",
                "score_subspace_dimension": None,
                "score_subspace_seed": 0,
                "noise_score_standardization": "none",
                "noise_score_reference_variance_factor": None,
                "noise_score_variance_ridge": 1e-12,
                "noise_score_z_clip": 5.0,
                "noise_score_null_mc_draws": 64,
                "noise_score_null_mc_seed": 20260906,
                "noise_score_trust_floor": 0.05,
                "noise_score_independence_low_quantile": 0.05,
                "noise_score_independence_full_quantile": 0.50,
                "noise_score_reject_cosine": -0.10,
                "noise_score_full_trust_cosine": 0.25,
                "noise_score_include_server_contraction": True,
                "noise_score_temporal_lower_z": 0.0,
                "noise_score_temporal_upper_z": 3.0,
                "noise_score_temporal_ema_decay": 0.5,
                "client_metrics_every": 1,
            }
        )
        return cfg
