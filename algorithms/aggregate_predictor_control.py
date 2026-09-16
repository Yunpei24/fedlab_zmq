"""Past EMA/RCIG predictors with independently calibrated aggregate controls.

Calibration observes both predictors on one uncorrected FAR model. Evaluation
applies one frozen rule. Attack labels and clean-gradient oracles are available
only to the separate evaluator, which uses a fixed set of eight honest IDs.
"""
from __future__ import annotations
import math
import torch

from algorithms.ldp_aggregation_role_ablation import LDPAggregationRoleAblation
from algorithms.rcig_temporal_reference import TemporalRCIGReferenceState
from algorithms.aggregate_radial_control import (
    PredictableSnapshot, apply_aggregate_control, stable_l2_norm,
    diagonal_mahalanobis_norm,
)
from algorithms.reference_utils import apply_delta
from robustness.tensor_ops import stack_updates, unflatten_update

KIND = "aggregate_predictor_control_v1"
POLICIES = {
    "far_rfa": ("ema", "unchanged"),
    "rfa_direct": ("ema", "unchanged"),
    "ema_smooth": ("ema", "ema"),
    "ema_iso": ("ema", "isotropic_clip"),
    "ema_radial": ("ema", "radial_ellipsoid"),
    "ema_projection": ("ema", "euclidean_ellipsoid_projection"),
    "rcig_iso": ("rcig", "isotropic_clip"),
    "rcig_radial": ("rcig", "radial_ellipsoid"),
    "rcig_projection": ("rcig", "euclidean_ellipsoid_projection"),
}


def radius_key(mode):
    return "isotropic" if mode == "isotropic_clip" else "mahalanobis"


class AggregatePredictorControl(LDPAggregationRoleAblation):
    def server_aggregate(self, global_model, client_updates, round_num, config):
        phase, arm = config["apc_phase"], config["apc_arm"]
        if phase not in {"calibration", "evaluation"} or arm not in POLICIES:
            raise ValueError("undeclared predictor-control phase/arm")
        if phase == "calibration" and arm != "far_rfa":
            raise ValueError("calibration host must remain unchanged FAR")
        if config["aggregation_role_arm"] != ("rfa_direct" if arm=="rfa_direct" else "far_rfa"):
            raise ValueError("parent aggregate mismatch")
        if not hasattr(self, "_apc_next"):
            self._apc_next=0
            self._apc_ema=None
            self._apc_variances=None
            # This is actual rolling RCIG, not the parent's midpoint shadow.
            rcfg=self._rcig_config(dict(config,rcig_persistent_policy="rolling"),float(config["far_server_clip_norm"]))
            self._apc_rcig=TemporalRCIGReferenceState(rcfg)
        if self._apc_next != round_num:
            raise ValueError("non-sequential predictor state")
        vectors,layout=stack_updates([u for u,_,_ in client_updates])
        rcig_result=self._apc_rcig.reference_for_round(round_num=round_num,
            dimension=vectors.shape[1],device="cpu",dtype=torch.float64)
        past={"ema":None if self._apc_ema is None else self._apc_ema.clone(),
              "rcig":rcig_result.reference.clone() if rcig_result.ready else None}
        result=super().server_aggregate(global_model,client_updates,round_num,config)
        parent=result.metrics.pop("_rcig_evaluation_payload")
        if parent["kind"] != "aggregation_role_v1":raise ValueError("cohort payload mismatch")
        ready=round_num>=12
        if ready and any(v is None for v in past.values()):raise ValueError("predictor not ready after warmup")
        raw=parent["rules"]["aggregates"]["far_rfa" if ready else "uniform"]
        public_var=self._public_rcig_variances(client_updates,config)
        if self._apc_variances is None:
            init=float(public_var.sum()/len(client_updates)**2)
            self._apc_variances={p:torch.full_like(raw,init) for p in past}
        snapshots={}
        diagnostics={}
        for p in past:
            predictor=past[p] if past[p] is not None else torch.zeros_like(raw)
            v=self._apc_variances[p].clone()+1e-8
            snap=PredictableSnapshot(round_num,round_num,past[p] is not None,predictor,v)
            snapshots[p]=snap
            diagnostics.update({
                f"apc_{p}_residual_l2":stable_l2_norm(raw-predictor) if snap.ready else None,
                f"apc_{p}_residual_mahalanobis":diagonal_mahalanobis_norm(raw-predictor,v) if snap.ready else None,
                f"apc_{p}_variance_trace":float(v.sum()),
                f"apc_{p}_variance_condition":float(v.max()/v.min()),
                f"apc_{p}_predictor_norm":stable_l2_norm(predictor) if snap.ready else None,
            })
        predictor_name,mode=POLICIES[arm]
        radius=None
        if mode in {"isotropic_clip","radial_ellipsoid","euclidean_ellipsoid_projection"}:
            radii=config.get("apc_frozen_radii")
            if not radii:raise ValueError("evaluation controller requires frozen out-of-test calibration")
            radius=float(radii[predictor_name][radius_key(mode)])
        if ready and mode!="unchanged":
            applied,control=apply_aggregate_control(mode,raw,snapshots[predictor_name],
                current_mix=.2 if mode=="ema" else None,radius=radius)
        else:
            applied=parent["rules"]["aggregates"][parent["deployed"]].clone()
            control={"gamma":1.0}
        lr=float(config["far_server_lr"])
        wanted=apply_delta(global_model,{k:lr*v for k,v in unflatten_update(applied,layout).items()})
        if (not ready or mode=="unchanged") and any(not torch.equal(wanted[k],result.new_weights[k]) for k in wanted):
            raise ValueError("identity/warmup controller changed parent model")
        result.new_weights=wanted
        # Current data are committed only after all predictor/control decisions.
        for p in past:
            if past[p] is not None:
                self._apc_variances[p]=.9*self._apc_variances[p]+.1*(raw-past[p]).square()
        self._apc_ema=raw.clone() if past["ema"] is None else .75*past["ema"]+.25*raw
        radius_server=float(config["far_server_clip_norm"])
        factors=(radius_server/torch.linalg.vector_norm(vectors,dim=1).clamp_min(1e-12)).clamp(max=1)
        snapshot=self._apc_rcig.make_snapshot(round_num=round_num,
            client_ids=parent["client_ids"],clipped_vectors=parent["vectors"],
            server_clip_factors=factors,public_noise_variances=public_var)
        self._apc_rcig.commit_snapshot(snapshot,reference_result=rcig_result)
        self._apc_next+=1
        correction=raw-applied
        diagnostics.update(apc_phase=phase,apc_arm=arm,
            apc_predictor_name=predictor_name,apc_controller=mode,
            apc_predictors_strictly_past=True,apc_history_observations=round_num,
            apc_current_round_used_for_predictor=False,apc_oracle_used_for_deployment=False,
            apc_correction_triggered=bool(ready and mode!="unchanged" and not torch.equal(raw,applied)),
            apc_correction_norm=stable_l2_norm(correction) if mode!="unchanged" else 0.0,
            apc_gamma=control.get("gamma"),apc_radius=radius,
            apc_rcig_inner_gate_active=bool(rcig_result.diagnostics.get("rcig_gate_active",False)),
            apc_rcig_inner_trust=rcig_result.diagnostics.get("rcig_newer_view_trust"),
            apc_rcig_is_midpoint=False,apc_rcig_state_policy="rolling",
            apc_rcig_latest_round_used=round_num-1 if ready else None,
            apc_applied_aggregate_norm=stable_l2_norm(applied))
        result.metrics.update(diagnostics)
        result.metrics["_rcig_evaluation_payload"]=dict(kind=KIND,round=round_num+1,
            parent=parent,raw=raw,applied=applied,snapshots=snapshots,
            contains_clean_data=False,contains_attack_labels=False,contains_realised_noise=False,
            oracle_used_for_deployment=False,radii=config.get("apc_frozen_radii"),
            gamma=control.get("gamma"),mode=mode)
        return result


class PredictorControlEvaluator:
    def __init__(self):
        self.last_round=0;self.error_sum=None;self.error_sq_sum=0.;self.count=0

    def __call__(self,payload,clean_by_client,client_updates):
        if payload.get("kind")!=KIND or any(payload.get(k) is not False for k in (
            "contains_clean_data","contains_attack_labels","contains_realised_noise","oracle_used_for_deployment")):
            raise ValueError("oracle boundary violation")
        t=payload["round"]
        if t!=self.last_round+1:raise ValueError("non-sequential evaluation")
        self.last_round=t
        parent=payload["parent"];ids=[int(m["client_id"]) for _,m,_ in client_updates]
        if ids!=parent["client_ids"] or set(ids)!=set(clean_by_client):raise ValueError("oracle identity mismatch")
        actual,_=stack_updates([u for u,_,_ in client_updates])
        U=parent["server_clip_norm"]
        factor=(U/torch.linalg.vector_norm(actual,dim=1).clamp_min(1e-12)).clamp(max=1)
        if not torch.equal(actual*factor[:,None],parent["vectors"]):raise ValueError("cohort changed")
        clean,_=stack_updates([clean_by_client[i] for i in ids])
        mask=torch.tensor([i in range(2,10) for i in ids])
        if int(mask.sum())!=8:raise ValueError("fixed eight-client honest target missing")
        h=clean[mask].mean(0)
        raw,applied=payload["raw"],payload["applied"]
        error=applied-h;correction=raw-applied
        mse=float(error.square().sum())
        if t>12:
            self.error_sum=error.clone() if self.error_sum is None else self.error_sum+error
            self.error_sq_sum+=mse;self.count+=1
        w=parent["rules"]["weights"][parent["deployed"]]
        byz=(w[~mask,None]*parent["vectors"][~mask]).sum(0)
        metrics=dict(apc_same_cohort_verified=True,apc_fixed_honest_count=8,
            apc_oracle_boundary="posthoc_only_no_feedback",
            apc_applied_mse=mse,apc_raw_far_mse=float((raw-h).square().sum()),
            apc_target_norm=stable_l2_norm(h),apc_target_ids="2,3,4,5,6,7,8,9",
            apc_removed_alignment_with_clean_gradient=float(correction@h) if payload["mode"]!="unchanged" else 0.,
            apc_designated_two_client_mass=float(w[~mask].sum()),
            apc_designated_current_contribution_norm=stable_l2_norm(byz),
            apc_attack_active=any(m.get("is_byzantine",False) for _,m,_ in client_updates),
            apc_cumulative_error_norm_sq=None if self.error_sum is None else float(self.error_sum.square().sum()),
            apc_time_mean_error_norm_sq=None if self.error_sum is None else float((self.error_sum/self.count).square().sum()),
            apc_time_mean_mse=None if not self.count else self.error_sq_sum/self.count)
        for p,snapshot in payload["snapshots"].items():
            d=h-snapshot.predictor
            metrics[f"apc_{p}_predictor_mse"]=float(d.square().sum()) if snapshot.ready else None
            metrics[f"apc_{p}_target_mahalanobis"]=diagonal_mahalanobis_norm(d,snapshot.variance) if snapshot.ready else None
            if payload["radii"] and snapshot.ready:
                rad=payload["radii"][p]
                metrics[f"apc_{p}_target_inside_isotropic"]=stable_l2_norm(d)<=rad["isotropic"]
                metrics[f"apc_{p}_target_inside_ellipsoid"]=diagonal_mahalanobis_norm(d,snapshot.variance)<=rad["mahalanobis"]
        if not all(v is None or not isinstance(v,float) or math.isfinite(v) for v in metrics.values()):raise ValueError("nonfinite audit")
        return metrics
