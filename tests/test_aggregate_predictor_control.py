"""Server algebra, temporal causality, calibration and recovery contract tests."""
import copy
import math
import pytest
import torch

from algorithms.aggregate_predictor_control import AggregatePredictorControl, PredictorControlEvaluator, POLICIES
from algorithms.aggregate_radial_control import apply_aggregate_control
from algorithms.reference_utils import apply_delta
from robustness.tensor_ops import stack_updates, unflatten_update
from tests.test_ldp_aggregation_role_ablation import toy_config, toy_updates
from scripts import run_aggregate_predictor_calibration as runner
from scripts.aggregate_control_attack_schedule import scheduled_attack


def configuration(arm="far_rfa", phase="evaluation", radius=.00001):
    cfg=toy_config("rfa_direct" if arm=="rfa_direct" else "far_rfa")
    cfg.update(apc_phase=phase,apc_arm=arm,apc_frozen_radii={p:{"isotropic":radius,"mahalanobis":radius} for p in ("ema","rcig")})
    return cfg


@pytest.mark.parametrize("arm",POLICIES)
def test_actual_applied_aggregate_and_honest_oracle(arm):
    alg=AggregatePredictorControl();model=torch.nn.Linear(32,1,bias=False)
    cfg=configuration(arm);evaluator=PredictorControlEvaluator();previous=None
    for t in range(14):
        updates=toy_updates(t);before=model.weight.detach().clone()
        result=alg.server_aggregate(model,updates,t,cfg);payload=result.metrics["_rcig_evaluation_payload"]
        assert torch.equal(before,model.weight)
        if previous is not None:assert torch.equal(payload["snapshots"]["ema"].predictor,previous)
        p,mode=POLICIES[arm]
        if t>=12 and mode!="unchanged":
            expected,_=apply_aggregate_control(mode,payload["raw"],payload["snapshots"][p],
                current_mix=.2 if mode=="ema" else None,radius=.00001)
            assert result.metrics["apc_correction_triggered"]
        else:expected=payload["parent"]["rules"]["aggregates"][payload["parent"]["deployed"]]
        assert torch.equal(expected,payload["applied"])
        _,layout=stack_updates([u for u,_,_ in updates])
        wanted=apply_delta(model,{k:.2*v for k,v in unflatten_update(expected,layout).items()})
        assert torch.equal(wanted["weight"],result.new_weights["weight"])
        previous=payload["raw"].clone() if previous is None else .75*previous+.25*payload["raw"]
        clean={i:{"weight":torch.full((1,32),float(i)/100)} for i in range(10)}
        measured=evaluator(payload,clean,updates)
        target=torch.full((32,),.055,dtype=torch.float64)
        assert measured["apc_applied_mse"]==pytest.approx(float((expected-target).square().sum()),abs=1e-7)
        assert measured["apc_fixed_honest_count"]==8
        assert result.metrics["apc_rcig_is_midpoint"] is False
        if t>=12:assert payload["snapshots"]["rcig"].ready
        model.load_state_dict(result.new_weights)


def test_future_upload_does_not_change_either_current_predictor():
    algs=[AggregatePredictorControl(),AggregatePredictorControl()]
    model=torch.nn.Linear(32,1,bias=False)
    for t in range(13):
        outputs=[]
        for j,alg in enumerate(algs):
            updates=toy_updates(t)
            if t==12 and j==1:
                for u,_,_ in updates:u["weight"].add_(3)
            outputs.append(alg.server_aggregate(model,updates,t,configuration()).metrics["_rcig_evaluation_payload"])
        if t==12:
            assert not torch.equal(outputs[0]["raw"],outputs[1]["raw"])
            for p in ("ema","rcig"):
                assert torch.equal(outputs[0]["snapshots"][p].predictor,outputs[1]["snapshots"][p].predictor)
                assert torch.equal(outputs[0]["snapshots"][p].variance,outputs[1]["snapshots"][p].variance)


@pytest.mark.parametrize("arm",["ema_iso","ema_radial","ema_projection","rcig_iso","rcig_radial","rcig_projection"])
def test_first_trigger_coupling_identity_before_trigger(arm):
    a,b=AggregatePredictorControl(),AggregatePredictorControl()
    model=torch.nn.Linear(32,1,bias=False)
    for t in range(14):
        r=a.server_aggregate(model,toy_updates(t),t,configuration("far_rfa"))
        s=b.server_aggregate(model,toy_updates(t),t,configuration(arm,radius=1e12))
        assert torch.equal(r.new_weights["weight"],s.new_weights["weight"])
        assert not s.metrics["apc_correction_triggered"]
        model.load_state_dict(r.new_weights)


def test_invalid_deployment_and_oracle_boundaries():
    model=torch.nn.Linear(32,1,bias=False)
    with pytest.raises(ValueError,match="unchanged FAR"):
        AggregatePredictorControl().server_aggregate(model,toy_updates(0),0,configuration("ema_smooth","calibration"))
    cfg=configuration("ema_iso");cfg["apc_frozen_radii"]=None
    with pytest.raises(ValueError,match="frozen"):
        AggregatePredictorControl().server_aggregate(model,toy_updates(0),0,cfg)
    alg=AggregatePredictorControl();updates=toy_updates(0)
    p=alg.server_aggregate(model,updates,0,configuration()).metrics["_rcig_evaluation_payload"]
    untouched=p["applied"].clone();state=alg._apc_ema.clone()
    clean={i:{"weight":torch.zeros(1,32)} for i in range(10)}
    m1=PredictorControlEvaluator()(p,clean,updates)
    for u in clean.values():u["weight"].add_(1)
    m2=PredictorControlEvaluator()(p,clean,updates)
    assert m1["apc_applied_mse"]!=m2["apc_applied_mse"]
    assert torch.equal(state,alg._apc_ema) and torch.equal(untouched,p["applied"])
    p["contains_clean_data"]=True
    with pytest.raises(ValueError,match="boundary"):PredictorControlEvaluator()(p,clean,updates)


def test_gradual_schedule_and_recovery_leave_honest_uploads_unchanged():
    updates=toy_updates(0)
    cfg=dict(enabled=True,name="ipm",num_byzantine=2,client_ids=[0,1],scale=1,
        active_round_start=17,active_round_end=28,ramp_end=24)
    h=torch.stack([u["weight"] for u,_,_ in updates[2:]]).mean(0)
    for public_round,frac in ((16,0),(17,1/8),(24,1),(28,1),(29,0)):
        output=scheduled_attack(updates,cfg,round_num=public_round-1)
        for i,((u,meta,_),(original,_,_)) in enumerate(zip(output,updates)):
            expected=original["weight"]+frac*(-h-original["weight"]) if i<2 and frac else original["weight"]
            assert torch.allclose(u["weight"],expected)
            assert bool(meta["is_byzantine"])==(i<2 and bool(frac))


def test_rank_calibration_and_recovery_censoring():
    assert runner.calibration_threshold(list(range(1,20)))==19
    for vals in (list(range(1,19)),[1.]*18+[float("nan")],[0.]*19):
        with pytest.raises(RuntimeError):runner.calibration_threshold(vals)
    m=runner.matrix();clean=[dict(test_accuracy=.6,worst20_accuracy_pct=35.,best20_worst20_gap_pct=30.) for _ in range(40)]
    rows=copy.deepcopy(clean)
    for r in rows[24:]:r["test_accuracy"]=.3
    assert runner.recovery_delay(rows,clean,24,m) is None
    rows[27:30]=copy.deepcopy(clean[27:30])
    assert runner.recovery_delay(rows,clean,24,m)==4
    rows[28]["worst20_accuracy_pct"]=20.
    assert runner.recovery_delay(rows,clean,24,m) is None


def test_matrix_separation_private_channel_and_actual_scenario_grid(monkeypatch):
    m=runner.matrix();stamp=runner.provenance(m)
    cal,ev=runner.tasks(m,"calibration"),runner.tasks(m,"evaluation")
    assert len(cal)==38 and len(ev)==216
    assert not {t["seed"] for t in cal}&{t["seed"] for t in ev}
    assert len({runner.directory(t) for t in cal+ev})==254
    artifact={"radii":{n:{p:{"isotropic":3.,"mahalanobis":10.} for p in ("ema","rcig")} for n in m["noise_regimes"]}}
    first=runner.config_for(m,cal[0],stamp)["training"]["algo_config"]
    assert first["attack"]["client_ids"]==[0,1] and not first["attack"]["enabled"]
    assert first["privacy_adjacency"]=="replace_one" and first["fixed_batch_size"]==120
    assert first["fixed_steps_per_round"]==1
    assert first["clip_norm"]==4 and first["far_server_clip_norm"]==16
    with pytest.raises(RuntimeError,match="before calibration"):
        runner.config_for(m,ev[0],stamp)
    real_hash=runner.hash_file
    monkeypatch.setattr(runner,"hash_file",lambda p:"test-radius-hash" if p==runner.ARTIFACT else real_hash(p))
    configs=[runner.config_for(m,t,stamp,artifact) for t in ev]
    assert len(configs)==216
    for t,cfg in zip(ev,configs):
        a=cfg["training"]["algo_config"]
        assert a["attack"]["client_ids"]==[0,1]
        assert a["apc_frozen_radii"]==artifact["radii"][t["noise"]]
        assert a["noise_multiplier"]==first["noise_multiplier"]
        if t["scenario"]!="clean":
            spec=m["scenarios"][t["scenario"]]
            assert a["attack"]["active_round_start"]==spec["start"]
            assert a["attack"]["active_round_end"]==spec["end"]
            counterpart=dict(t,scenario="clean")
            assert ev.index(counterpart)<ev.index(t)
    runner.verify(stamp)
