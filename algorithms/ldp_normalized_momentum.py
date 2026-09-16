"""Bias-corrected client EMA, isolated from the source-locked first screen."""
import math
import torch

from algorithms.ldp_client_momentum import LDPClientMomentum, dict_digest, aggregate_rules
from algorithms.ldp_gradient_far import LDPGradientFAR
from algorithms.rcig_n10_ablation import isolated_simulator_rng, model_digest
from robustness.tensor_ops import stack_updates

INITIALIZATION = "zero_buffer_with_bias_correction"


def normalized_step(current, previous_z, beta, step):
    """Return the unnormalized state z and the released normalized momentum.

    Step is one-based. The first upload is exactly the current private gradient,
    avoiding a gratuitous float32 multiply/divide round-trip at step one.
    """
    if not 0 <= beta < 1 or not isinstance(step, int) or step < 1:
        raise ValueError("invalid normalized EMA parameters")
    if (previous_z is None) != (step == 1):
        raise ValueError("EMA state/step chronology mismatch")
    if previous_z is not None and previous_z.keys() != current.keys():
        raise ValueError("EMA tensor layout changed")
    denominator = 1-beta**step
    z = {k: (1-beta)*v if previous_z is None else beta*previous_z[k]+(1-beta)*v
         for k, v in current.items()}
    momentum = {k: v.detach().clone() for k, v in current.items()} if step == 1 else {
        k: v/denominator for k, v in z.items()}
    return z, momentum, denominator


def linear_noise_factor(beta, step):
    """Variance of explicit filtered iid noise / variance of one fresh draw."""
    return (1-beta)/(1+beta)*(1+beta**step)/(1-beta**step)


class LDPNormalizedMomentum(LDPClientMomentum):
    def client_update(self, model, dataloader, state, config):
        device = str(config["device"])
        if torch.device(device).type != "mps":
            raise RuntimeError("private gradient and normalized EMA require MPS")
        if config.get("momentum_initialization") != INITIALIZATION:
            raise RuntimeError("wrong initialization")
        t = int(config["_server_round"])
        beta = float(config["client_momentum_beta"])
        if state.custom.get("normalized_momentum_next_round", 0) != t:
            raise RuntimeError("normalized EMA repeated/skipped a round")
        if t > 0 and state.momentum_buffer is None:
            raise RuntimeError("normalized EMA memory reset")
        row = dict(client_id=int(state.client_id), round=t+1, beta=beta,
            initialization=INITIALIZATION, memory_kind="unnormalized_z",
            model_before=model_digest(model),
            memory_before=None if state.momentum_buffer is None else dict_digest(state.momentum_buffer))
        with isolated_simulator_rng(config["momentum_pairing_seed"], state.client_id, t,
                                    device=device, audit=row):
            # Directly reuse the original private channel, not the old EMA.
            gradient, metadata = LDPGradientFAR.client_update(self, model, dataloader, state, config)
        if len(row.get("permutations", [])) != 1 or not row.get("standard_gaussians"):
            raise RuntimeError("missing actual random-draw trace")
        raw = {k: v.to(device) for k, v in gradient.items()}
        z, momentum, denominator = normalized_step(raw, state.momentum_buffer, beta, t+1)
        state.momentum_buffer = z
        state.custom["normalized_momentum_next_round"] = t+1
        upload = {k: v.detach().cpu().clone() for k, v in momentum.items()}
        row.update(raw_private_gradient=dict_digest(gradient), private_upload=dict_digest(upload),
            memory_after=dict_digest(z), normalization_denominator=denominator, momentum_device=device)
        metadata.update(client_momentum_beta=beta, client_momentum_history_length=t+1,
            client_momentum_device=device, client_momentum_linear_noise_factor=linear_noise_factor(beta,t+1),
            momentum_initialization=INITIALIZATION, momentum_normalization_denominator=denominator,
            privacy_release_object="normalized_momentum_of_private_batch_gradients",
            privacy_variance_semantics="fresh_gradient_before_momentum_not_effective_upload_covariance",
            momentum_raw_private_gradient_oracle=gradient)
        if self.audit_sink is not None:
            self.audit_sink(row)
        return upload, metadata

    def server_aggregate(self, global_model, client_updates, round_num, config):
        for _, metadata, _ in client_updates:
            if metadata["momentum_initialization"] != INITIALIZATION:
                raise RuntimeError("wrong client implementation")
        result = super().server_aggregate(global_model, client_updates, round_num, config)
        result.metrics.update(momentum_initialization=INITIALIZATION,
            momentum_normalization_denominator=1-float(config["client_momentum_beta"])**(round_num+1))
        result.metrics["_rcig_evaluation_payload"].update(
            arm=config["momentum_arm"], alpha=result.metrics["far_alpha"],
            server_lr=float(config["far_server_lr"]), rfa_max_iter=int(config["rfa_max_iter"]),
            rfa_tol=float(config["rfa_tol"]))
        return result


class NormalizedMomentumEvaluator:
    """All clean oracles/counterfactuals are post-hoc, outside deployment."""
    def __init__(self):
        self.clean_z = {}
        self.cumulative_error = None
        self.next_round = 0

    def __call__(self, p, clean, original):
        if clean is None or set(clean) != set(p["ids"]) or p["round"] != self.next_round:
            raise RuntimeError("incomplete/misordered offline oracle")
        raw = {int(m["client_id"]): m["momentum_raw_private_gradient_oracle"] for _,m,_ in original}
        history=[]
        for cid in p["ids"]:
            c={k:v.to("mps") for k,v in clean[cid].items()}
            z, momentum, _ = normalized_step(c, self.clean_z.get(cid), p["beta"], p["round"]+1)
            self.clean_z[cid]=z
            history.append({k:v.cpu() for k,v in momentum.items()})
        h,_=stack_updates([clean[i] for i in p["ids"]])
        g,_=stack_updates([raw[i] for i in p["ids"]])
        hm,_=stack_updates(history)
        def mse(a,b):return float((a-b).square().sum(-1).mean())
        # Same received vectors, but recompute both reference AND weights with
        # no server clipping. This is not a separately trained no-clip model.
        unclipped = aggregate_rules(p["vectors"], p["arm"], p["alpha"], math.inf,
                                   p["rfa_max_iter"], p["rfa_tol"])[5]
        err = p["server_lr"]*(p["aggregate"]-h.mean(0))
        self.cumulative_error = err.clone() if self.cumulative_error is None else self.cumulative_error+err
        self.next_round += 1
        return dict(momentum_oracle_evaluation_only=True,
            momentum_oracle_applied_mse_current_clean_mean=mse(p["aggregate"],h.mean(0)),
            momentum_oracle_applied_mse_filtered_clean_mean=mse(p["aggregate"],hm.mean(0)),
            momentum_oracle_reference_mse_current_clean_mean=mse(p["reference"],h.mean(0)),
            momentum_oracle_fresh_noise_energy=mse(g,h),
            momentum_oracle_filtered_noise_energy=mse(p["vectors"],hm),
            momentum_oracle_clean_lag_energy=mse(hm,h),
            momentum_oracle_server_clip_displacement=mse(p["vectors"],p["clipped"]),
            momentum_oracle_clip_aggregate_displacement=mse(p["aggregate"],unclipped),
            momentum_oracle_no_clip_same_messages_mse=mse(unclipped,h.mean(0)),
            momentum_oracle_cumulative_applied_error_squared=float(self.cumulative_error.square().sum()),
            momentum_no_clip_counterfactual_is_end_to_end=False)
