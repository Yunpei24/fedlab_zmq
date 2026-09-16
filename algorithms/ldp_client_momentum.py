"""Isolated clean-screen client momentum; no change to the private channel.

The evaluator below is never called by the deployed aggregation. Its clean
gradients live outside ClientState and cannot choose weights or hyperparameters.
"""
import hashlib
import math
import torch

from algorithms.base import AggregateResult
from algorithms.ldp_gradient_far import LDPGradientFAR
from algorithms.ldp_aggregation_role_ablation import rfa_with_weights
from algorithms.rcig_n10_ablation import isolated_simulator_rng, model_digest, tensor_digest
from algorithms.reference_utils import apply_delta
from robustness.tensor_ops import stack_updates, unflatten_update

ARMS = ("uniform", "rfa_direct", "far_rfa")


def dict_digest(values):
    return hashlib.sha256("".join(k + tensor_digest(v) for k, v in values.items()).encode()).hexdigest()


def momentum_step(current, previous, beta):
    if not 0 <= beta < 1:
        raise ValueError("momentum beta must be in [0,1)")
    if previous is None or beta == 0:
        return {k: v.detach().clone() for k, v in current.items()}
    if current.keys() != previous.keys():
        raise ValueError("momentum layout changed")
    return {k: beta * previous[k] + (1-beta) * v for k, v in current.items()}


def aggregate_rules(vectors, arm, alpha, radius, max_iter=100, tol=1e-6):
    if arm not in ARMS or not math.isfinite(alpha) or radius <= 0:
        raise ValueError("invalid rule")
    norms = torch.linalg.vector_norm(vectors, dim=1)
    factors = (radius / norms.clamp_min(1e-12)).clamp(max=1)
    x = vectors * factors[:, None]
    reference, rw, iterations, residual = rfa_with_weights(x, max_iter=max_iter, tol=tol)
    distances = torch.linalg.vector_norm(x-reference, dim=1)
    weights = (torch.full_like(distances, 1/len(x)) if arm == "uniform" else
               rw if arm == "rfa_direct" else torch.softmax(alpha*distances, dim=0))
    aggregate = reference if arm == "rfa_direct" else (weights[:, None]*x).sum(0)
    return x, factors, reference, distances, weights, aggregate, iterations, residual


class LDPClientMomentum(LDPGradientFAR):
    audit_sink = None

    def client_update(self, model, dataloader, state, config):
        device = str(config["device"])
        if torch.device(device).type != "mps":
            raise RuntimeError("private gradients and client momentum require MPS")
        t = int(config["_server_round"])
        beta = float(config["client_momentum_beta"])
        if state.custom.get("momentum_next_round", 0) != t:
            raise RuntimeError("client momentum lost or repeated a round")
        if t > 0 and state.momentum_buffer is None:
            raise RuntimeError("client memory was reset between rounds")
        row = dict(client_id=int(state.client_id), round=t+1, beta=beta,
                   model_before=model_digest(model),
                   memory_before=None if state.momentum_buffer is None else dict_digest(state.momentum_buffer))
        with isolated_simulator_rng(config["momentum_pairing_seed"], state.client_id, t,
                                    device=device, audit=row):
            gradient, metadata = super().client_update(model, dataloader, state, config)
        if len(row.get("permutations", [])) != 1 or not row.get("standard_gaussians"):
            raise RuntimeError("missing actual batch/Gaussian draw fingerprints")
        raw = {k: v.to(device) for k, v in gradient.items()}
        state.momentum_buffer = momentum_step(raw, state.momentum_buffer, beta)
        state.custom["momentum_next_round"] = t+1
        factor = 1.0 if t == 0 else beta**2 * state.custom["momentum_noise_factor"] + (1-beta)**2
        state.custom["momentum_noise_factor"] = factor
        upload = {k: v.detach().cpu().clone() for k, v in state.momentum_buffer.items()}
        row.update(raw_private_gradient=dict_digest(gradient), private_upload=dict_digest(upload),
                   memory_after=dict_digest(state.momentum_buffer), momentum_device=device)
        metadata.update(client_momentum_beta=beta, client_momentum_history_length=t+1,
                        client_momentum_device=device, client_momentum_linear_noise_factor=factor,
                        privacy_release_object="momentum_of_private_batch_gradients",
                        privacy_variance_semantics="fresh_gradient_before_momentum_not_effective_upload_covariance")
        # This copy is consumed only by the evaluator; the boundary removes it.
        metadata["momentum_raw_private_gradient_oracle"] = gradient
        if self.audit_sink is not None:
            self.audit_sink(row)
        return upload, metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        if round_num != getattr(self, "_momentum_server_round", 0):
            raise RuntimeError("server chronology mismatch")
        if config.get("attack") is not None or len(client_updates) != 10:
            raise RuntimeError("clean full-cohort screen contract violated")
        for _, metadata, _ in client_updates:
            if any("oracle" in k or k == "is_byzantine" or k.startswith("attack_") for k in metadata):
                raise RuntimeError("evaluation information crossed the server boundary")
            if not str(metadata["privacy_compute_device"]).startswith("mps") or not str(metadata["client_momentum_device"]).startswith("mps"):
                raise RuntimeError("CPU private computation detected")
        vectors, layout = stack_updates([u for u, _, _ in client_updates])
        arm, beta = config["momentum_arm"], float(config["client_momentum_beta"])
        alpha = float(config["far_alpha"]) if arm == "far_rfa" else 0.0
        x, factors, ref, distances, weights, aggregate, iterations, residual = aggregate_rules(
            vectors, arm, alpha, float(config["far_server_clip_norm"]),
            int(config["rfa_max_iter"]), float(config["rfa_tol"]))
        if not torch.isfinite(aggregate).all():
            raise RuntimeError("nonfinite aggregate")
        lr = float(config["far_server_lr"])
        new_weights = apply_delta(global_model, {k: lr*v for k, v in unflatten_update(aggregate, layout).items()})
        metrics = dict(momentum_arm=arm, client_momentum_beta=beta,
            momentum_history_length=round_num+1, momentum_warmup_rounds=0,
            momentum_private_gradient_device="mps", momentum_buffer_device="mps",
            momentum_server_device="cpu", momentum_server_dtype="torch.float64",
            momentum_weight_semantics="weiszfeld_effective_coefficients" if arm == "rfa_direct" else "far_softmax" if arm == "far_rfa" else "uniform",
            momentum_linear_noise_factor=client_updates[0][1]["client_momentum_linear_noise_factor"],
            far_alpha=alpha, far_server_lr=lr, far_server_clip_norm=config["far_server_clip_norm"],
            far_server_clip_rate=float((factors < 1).double().mean()),
            far_max_weight=float(weights.max()), far_weight_l2_squared=float(weights.square().sum()),
            far_noise_amplification_vs_uniform=float(len(weights)*weights.square().sum()),
            far_weight_entropy=float(-(weights * weights.clamp_min(1e-30).log()).sum()),
            far_distance_span=float(distances.max()-distances.min()),
            far_logit_span=float(alpha*(distances.max()-distances.min())),
            far_score_mode="raw_distance", far_tilt_influence_certificate_claimed=False,
            momentum_rfa_iterations=iterations, momentum_rfa_last_displacement=residual,
            far_attack_labels_visible_to_server_aggregate=False,
            far_attack_config_visible_to_server_aggregate=False,
            far_external_attack_diagnostics=True,
            far_external_attack_diagnostics_boundary="posthoc_simulator_only",
            ldp_gradient_far_private_gradient_mps_fraction=1.0,
            ldp_gradient_far_private_gradient_compute_device="mps",
            _far_external_weight_diagnostics_payload=dict(
                client_ids=[int(m["client_id"]) for _, m, _ in client_updates],
                weights=weights.detach().clone(), contains_attack_labels=False),
            _rcig_evaluation_payload=dict(round=round_num, beta=beta, aggregate=aggregate,
                reference=ref, vectors=vectors, clipped=x, weights=weights,
                ids=[int(m["client_id"]) for _, m, _ in client_updates]))
        self._momentum_server_round = round_num+1
        return self._add_local_dp_round_metrics(AggregateResult(new_weights, metrics), client_updates, config)


class MomentumEvaluator:
    """Offline diagnostics on each rule's own trajectory, never counterfactual training."""
    def __init__(self):
        self.clean_history = {}

    def __call__(self, payload, clean, original):
        if clean is None or set(clean) != set(payload["ids"]):
            raise RuntimeError("incomplete offline clean cohort")
        beta = payload["beta"]
        raw = {int(m["client_id"]): m["momentum_raw_private_gradient_oracle"] for _, m, _ in original}
        history = []
        for cid in payload["ids"]:
            # Same float32 arithmetic/device as deployment; strictly evaluator-owned.
            c = {k: v.to("mps") for k, v in clean[cid].items()}
            self.clean_history[cid] = momentum_step(c, self.clean_history.get(cid), beta)
            # Explicit transfer before float64 conversion (MPS has no float64).
            history.append({k: v.cpu() for k, v in self.clean_history[cid].items()})
        h, _ = stack_updates([clean[i] for i in payload["ids"]])
        z, _ = stack_updates([raw[i] for i in payload["ids"]])
        hm, _ = stack_updates(history)
        def mse(a, b):
            return float((a-b).square().sum(-1).mean())
        return dict(momentum_oracle_evaluation_only=True,
            momentum_oracle_applied_mse_current_clean_mean=mse(payload["aggregate"], h.mean(0)),
            momentum_oracle_applied_mse_filtered_clean_mean=mse(payload["aggregate"], hm.mean(0)),
            momentum_oracle_reference_mse_current_clean_mean=mse(payload["reference"], h.mean(0)),
            momentum_oracle_fresh_noise_energy=mse(z, h),
            momentum_oracle_filtered_noise_energy=mse(payload["vectors"], hm),
            momentum_oracle_clean_lag_energy=mse(hm, h),
            momentum_oracle_server_clip_displacement=mse(payload["vectors"], payload["clipped"]))
