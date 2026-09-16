import copy
import pytest
import torch

from algorithms.aggregate_control_e2e import AggregateControlEndToEnd, EndToEndEvaluator, ARMS
from algorithms.reference_utils import apply_delta
from robustness.tensor_ops import stack_updates, unflatten_update
from tests.test_ldp_aggregation_role_ablation import toy_config, toy_updates
from scripts import run_aggregate_control_e2e as runner


def cfg_for(arm):
    cfg=toy_config("far_rfa" if arm=="ema_far_rfa" else arm)
    cfg.update(aggregate_e2e_arm=arm,aggregate_e2e_scenario="none",
               aggregate_e2e_predictor_rate=.25,aggregate_e2e_current_mix=.2)
    return cfg


@pytest.mark.parametrize("arm", ARMS)
def test_actual_model_update_and_causal_ema(arm):
    alg=AggregateControlEndToEnd();cfg=cfg_for(arm)
    model=torch.nn.Linear(32,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    prediction=None
    for t in range(14):
        updates=toy_updates(t);before=model.weight.detach().clone()
        result=alg.server_aggregate(model,updates,t,cfg)
        p=result.metrics["_rcig_evaluation_payload"]
        assert torch.equal(before,model.weight)
        if prediction is None:assert p["predictor"] is None
        else:assert torch.equal(prediction,p["predictor"])
        if t>=12 and arm=="ema_far_rfa":
            expected=prediction+.2*(p["raw_far"]-prediction)
            assert not torch.equal(expected,p["raw_far"])
        else:
            expected=p["parent"]["rules"]["aggregates"]["uniform" if t<12 else arm]
        assert torch.equal(expected,p["applied"])
        _,layout=stack_updates([u for u,_,_ in updates])
        wanted=apply_delta(model,{k:.2*v for k,v in unflatten_update(expected,layout).items()})
        assert torch.equal(wanted["weight"],result.new_weights["weight"])
        prediction=p["raw_far"].clone() if prediction is None else .75*prediction+.25*p["raw_far"]
        assert torch.equal(prediction,alg._e2e_predictor)
        model.load_state_dict(result.new_weights)


def test_duplicate_round_is_rejected():
    alg=AggregateControlEndToEnd();model=torch.nn.Linear(32,1,bias=False)
    alg.server_aggregate(model,toy_updates(0),0,cfg_for("far_rfa"))
    with pytest.raises((ValueError,RuntimeError)):
        alg.server_aggregate(model,toy_updates(0),0,cfg_for("far_rfa"))


def test_no_oracle_controls_predictor_or_model():
    alg=AggregateControlEndToEnd();model=torch.nn.Linear(32,1,bias=False)
    updates=toy_updates(0);result=alg.server_aggregate(model,updates,0,cfg_for("ema_far_rfa"))
    p=result.metrics["_rcig_evaluation_payload"]
    predictor=alg._e2e_predictor.clone();applied=p["applied"].clone()
    clean={i:{"weight":torch.zeros(1,32)} for i in range(10)}
    metrics=EndToEndEvaluator()(p,clean,updates)
    assert metrics["aggregate_e2e_applied_mse"]>0
    assert torch.equal(predictor,alg._e2e_predictor) and torch.equal(applied,p["applied"])
    changed=copy.deepcopy(p);changed["contains_clean_data"]=True
    with pytest.raises(ValueError,match="boundary"):
        EndToEndEvaluator()(changed,clean,updates)
    attack=copy.deepcopy(updates);attack[0][1]["is_byzantine"]=True
    with pytest.raises(ValueError,match="clean-only"):
        EndToEndEvaluator()(p,clean,attack)


def test_frozen_matrix_and_private_channel():
    m=runner.matrix();stamp=runner.provenance(m);ts=runner.tasks(m)
    assert len(ts)==24 and len({runner.run_id(t) for t in ts})==24
    assert {t["seed"] for t in ts}=={930401,930402,930403}
    configs=[runner.config_for(m,t,stamp) for t in ts[:4]]
    for key in ("clip_norm","far_server_clip_norm","far_alpha","far_server_lr","fixed_batch_size",
                "sampling_scheme","privacy_adjacency","noise_multiplier","privacy_noise_multiplier_scale_by_client"):
        assert all(c["training"]["algo_config"][key]==configs[0]["training"]["algo_config"][key] for c in configs)
    assert not configs[0]["training"]["algo_config"]["attack"]["enabled"]
    assert abs(stamp["privacy"]["per_scale"]["1"]["epsilon"]-4)<1e-4
    runner.verify(stamp)
