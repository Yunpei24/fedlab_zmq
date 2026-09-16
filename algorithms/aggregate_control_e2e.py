"""Actually deployed EMA aggregate correction; private client stays unchanged.

The first pilot is deliberately clean-only. The old RCIG midpoint machinery is
a compatibility shadow, never the predictor or the applied reference. All
oracle calculations are in a separate evaluator called after aggregation.
"""
from __future__ import annotations

import math
import torch

from algorithms.ldp_aggregation_role_ablation import LDPAggregationRoleAblation
from algorithms.reference_utils import apply_delta
from algorithms.aggregate_radial_control import ema_smooth_about_predictor
from algorithms.aggregate_control_replay import _clean_target
from metrics.aggregation_role_evaluation import evaluate_aggregation_roles
from robustness.tensor_ops import stack_updates, unflatten_update

ARMS = ("far_rfa", "ema_far_rfa", "uniform", "rfa_direct")
KIND = "aggregate_control_e2e_pilot_v1"


class AggregateControlEndToEnd(LDPAggregationRoleAblation):
    def server_aggregate(self, global_model, client_updates, round_num, config):
        arm = config["aggregate_e2e_arm"]
        if arm not in ARMS:
            raise ValueError("undeclared aggregate-control arm")
        if config.get("aggregate_e2e_scenario") != "none":
            raise ValueError("this pilot is clean-only; attacked evaluation is not implemented")
        beta = float(config["aggregate_e2e_predictor_rate"])
        mix = float(config["aggregate_e2e_current_mix"])
        if beta != 0.25 or mix != 0.2:
            raise ValueError("fixed pilot coefficients changed")
        parent_arm = "far_rfa" if arm == "ema_far_rfa" else arm
        if config["aggregation_role_arm"] != parent_arm:
            raise ValueError("parent rule mismatch")
        result = super().server_aggregate(global_model, client_updates, round_num, config)
        parent = result.metrics.pop("_rcig_evaluation_payload")
        if parent["kind"] != "aggregation_role_v1":
            raise ValueError("missing detached private-cohort payload")
        ready = round_num >= 12
        rules = parent["rules"]
        raw_far = rules["aggregates"]["far_rfa" if ready else "uniform"]
        if not hasattr(self, "_e2e_observations"):
            self._e2e_observations = 0
            self._e2e_predictor = None
        if self._e2e_observations != round_num:
            raise ValueError("predictor must be advanced exactly once per consecutive round")
        predictor = None if self._e2e_predictor is None else self._e2e_predictor.clone()
        effective = arm if ready else "uniform"
        if effective == "ema_far_rfa":
            if predictor is None:
                raise ValueError("past predictor unavailable after warmup")
            applied = ema_smooth_about_predictor(raw_far, predictor, current_mix=mix)
        else:
            applied = rules["aggregates"][effective].clone()
        _, layout = stack_updates([u for u, _, _ in client_updates])
        lr = float(config["far_server_lr"])
        new_weights = apply_delta(global_model, {
            k: lr*v for k, v in unflatten_update(applied, layout).items()
        })
        if effective != "ema_far_rfa" and any(
            not torch.equal(new_weights[k], result.new_weights[k]) for k in new_weights
        ):
            raise ValueError("uncorrected controls no longer reproduce parent updates")
        result.new_weights = new_weights
        # ONLY now commit A_t. The output of the smoother does not feed its EMA.
        self._e2e_predictor = raw_far.clone() if predictor is None else (
            (1.0-beta)*predictor + beta*raw_far
        )
        self._e2e_observations += 1
        if not torch.isfinite(self._e2e_predictor).all():
            raise ValueError("nonfinite predictor")
        result.metrics.update(
            aggregate_e2e_arm=arm,
            aggregate_e2e_effective_arm=effective,
            aggregate_e2e_correction_applied_to_model=effective == "ema_far_rfa",
            aggregate_e2e_predictor_strictly_past=True,
            aggregate_e2e_history_observations=round_num,
            aggregate_e2e_predictor_rate=beta,
            aggregate_e2e_current_mix=mix if effective == "ema_far_rfa" else 1.0,
            aggregate_e2e_oracle_used_for_deployment=False,
            aggregate_e2e_rcig_predictor_used=False,
            aggregate_e2e_predictor_input="uncorrected_far_rfa_on_own_model",
            aggregate_e2e_applied_norm=float(torch.linalg.vector_norm(applied)),
            aggregate_e2e_correction_norm=float(torch.linalg.vector_norm(applied-raw_far))
                if effective == "ema_far_rfa" else 0.0,
            aggregate_e2e_raw_far_diagnostics_are_not_corrected_weight_diagnostics=True,
        )
        result.metrics["_rcig_evaluation_payload"] = dict(
            kind=KIND, round=round_num+1, arm=arm, effective=effective,
            applied=applied, raw_far=raw_far, predictor=predictor,
            client_ids=parent["client_ids"], server_clip_norm=parent["server_clip_norm"],
            parent=parent, contains_clean_data=False, contains_attack_labels=False,
            contains_realised_noise=False, oracle_used_for_deployment=False,
        )
        return result


class EndToEndEvaluator:
    """Separate state; stores cumulative diagnostic error, never exposed to server."""
    def __init__(self):
        self.last_round = 0
        self.error_sum = None
        self.error_sq_sum = 0.0
        self.count = 0

    def __call__(self, payload, clean_by_client, client_updates):
        if not isinstance(payload, dict) or payload.get("kind") != KIND:
            raise ValueError("unexpected deployed aggregate evaluator payload")
        if any(payload.get(k) is not False for k in (
            "contains_clean_data", "contains_attack_labels", "contains_realised_noise",
            "oracle_used_for_deployment",
        )):
            raise ValueError("oracle boundary violation")
        if any(m.get("is_byzantine", False) for _,m,_ in client_updates):
            raise ValueError("clean-only target cannot be used for attacked runs")
        t = int(payload["round"])
        if t != self.last_round+1:
            raise ValueError("duplicate or out-of-order evaluation")
        self.last_round = t
        output = evaluate_aggregation_roles(payload["parent"], clean_by_client, client_updates)
        # Parent 'deployed' refers to the input rule, not the corrected aggregate.
        for key in list(output):
            if key.startswith("deployed_"):
                output["uncorrected_parent_"+key] = output.pop(key)
        target = _clean_target(payload, clean_by_client, client_updates)
        applied, raw, predictor = payload["applied"], payload["raw_far"], payload["predictor"]
        error = applied-target
        error_sq = float(error.square().sum())
        if t > 12:
            self.error_sum = error.clone() if self.error_sum is None else self.error_sum+error
            self.error_sq_sum += error_sq
            self.count += 1
        output.update(
            aggregate_e2e_oracle_boundary="offline_only_no_feedback",
            aggregate_e2e_applied_mse=error_sq,
            aggregate_e2e_uncorrected_far_mse=float((raw-target).square().sum()),
            aggregate_e2e_predictor_mse=None if predictor is None else float((predictor-target).square().sum()),
            aggregate_e2e_target_norm=float(torch.linalg.vector_norm(target)),
            aggregate_e2e_cumulative_error_norm_sq=None if self.error_sum is None else float(self.error_sum.square().sum()),
            aggregate_e2e_cumulative_mean_error_norm_sq=None if self.error_sum is None else float((self.error_sum/self.count).square().sum()),
            aggregate_e2e_cumulative_mean_mse=None if not self.count else self.error_sq_sum/self.count,
            aggregate_e2e_target_definition="all_ten_clean_per_example_clipped_batch_gradients",
        )
        if not all(v is None or not isinstance(v,float) or math.isfinite(v) for v in output.values()):
            raise ValueError("nonfinite diagnostic")
        return output
