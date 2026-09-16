"""Numeric, deployed-rule, oracle-isolation and matrix regression tests."""
import copy
import pytest
import torch
from algorithms.base import ClientState
from algorithms.ldp_aggregation_role_ablation import LDPAggregationRoleAblation,ARMS,cohort_rules,rfa_with_weights
from metrics.aggregation_role_evaluation import evaluate_aggregation_roles
from robustness.aggregators import geometric_median
from scripts import run_ldp_aggregation_role_ablation as runner


@pytest.mark.parametrize('seed',[0,1,2,42])
def test_rfa_matches_existing_solver_and_coefficients(seed):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(10,64,generator=g,dtype=torch.float64)
    x[0]*=10
    a,w,it,res=rfa_with_weights(x)
    assert torch.equal(a,geometric_median(x))
    assert torch.allclose(a,(w[:,None]*x).sum(0),atol=1e-12,rtol=1e-12)
    assert 1<=it<=100 and res>=0 and abs(float(w.sum())-1)<1e-12


def test_rfa_degenerate_cohort():
    x=torch.ones(10,32,dtype=torch.float64)
    a,w,_,_=rfa_with_weights(x)
    assert torch.equal(a,x[0]) and torch.allclose(w,torch.full_like(w,.1))


def toy_config(arm):
    return {**LDPAggregationRoleAblation().get_default_config(),
        'aggregation_role_arm':arm,'rcig_n10_arm':'midpoint',
        'robust_reference':'rcig_temporal','rcig_persistent_policy':'rolling',
        'expected_num_clients':10,'far_server_clip_norm':16.,'rcig_reference_output_radius':16.,
        'clip_norm':1.,'enable_dp':True,'noise_multiplier':1.,'target_epsilon':None,
        'fixed_batch_size':10,'privacy_public_dataset_size':200,'privacy_sampling_rate_override':.05,
        'fixed_steps_per_round':1,'privacy_num_rounds':40,
        'sampling_scheme':'fixed_without_replacement','privacy_adjacency':'replace_one',
        'rcig_covariance_registry':'authenticated_public_mechanism',
        'far_alpha':.1,'kappa_w':2.,'rcig_gate_window':4,'rcig_old_window':4,'rcig_new_window':4,
        'rcig_public_subspace_dimension':32,'rcig_public_subspace_seed':7,
        'rcig_process_variance':1e-6,'rcig_covariance_ridge':1e-6,
        'rcig_innovation_threshold':9.5,'rcig_min_accepted_mass':.6,'far_server_lr':.2,
        'enable_oracle_diagnostics':True,'external_attack_diagnostics':True,
        'rcig_oracle_separation_required':True}


def toy_updates(t):
    result=[]
    for i in range(10):
        vector=(torch.arange(32).float().reshape(1,32)/1000+(i-4.5)/20+t*.001)
        if i==0:vector=-8*vector
        result.append(({'weight':vector},dict(client_id=i,round_num=t+1,
            privacy_compute_device='mps:0',privacy_gradient_release=True,
            privacy_upload_noise_variance_per_coordinate=.01,privacy_noise_multiplier_scale_public=1.,
            privacy_epsilon=3.,privacy_delta=1e-5,privacy_noise_multiplier=1.,privacy_sensitivity_multiplier=2.,
            privacy_fixed_batch_size=10,dataset_size=200,model_steps=1,
            privacy_sampling_scheme='fixed_without_replacement',privacy_adjacency='replace_one',
            privacy_accounting_assumption='fixed_size_without_replacement_rdp_replace_one',
            far_update_mode='single_step_gradient'),ClientState(client_id=i,battery_j=100)))
    return result


@pytest.mark.parametrize('arm',ARMS)
def test_only_declared_rule_changes_model_and_warmup_is_shared(arm):
    a=LDPAggregationRoleAblation();cfg=toy_config(arm)
    model=torch.nn.Linear(32,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    for t in range(14):
        before=model.weight.detach().clone()
        result=a.server_aggregate(model,toy_updates(t),t,cfg)
        assert torch.equal(model.weight,before),'aggregator mutated its input model'
        p=result.metrics['_rcig_evaluation_payload'];deployed='uniform' if t<12 else arm
        expected=before-.2*p['rules']['aggregates'][deployed].reshape_as(before).float()
        assert torch.equal(expected,result.new_weights['weight'])
        assert p['deployed']==deployed and len(p['rules']['aggregates'])==4
        assert result.metrics['aggregation_role_weight_semantics']==('weiszfeld_effective_coefficients' if deployed=='rfa_direct' else 'far_softmax' if deployed.startswith('far_') else 'uniform')
        model.load_state_dict(result.new_weights)


def evaluation_fixture():
    gen=torch.Generator().manual_seed(123)
    x=torch.randn(10,32,dtype=torch.float64,generator=gen)*.1
    clean={i:{'weight':(x[i]*.3).reshape(1,32)} for i in range(10)}
    updates=[({'weight':x[i].reshape(1,32)},dict(client_id=i,is_byzantine=i<2),None) for i in range(10)]
    rules=cohort_rules(x,torch.zeros(32,dtype=torch.float64),alpha=.1)
    payload=dict(kind='aggregation_role_v1',client_ids=list(range(10)),server_clip_norm=16.,vectors=x,
        rules=rules,ready=True,deployed='far_rfa',contains_clean_data=False,contains_realised_noise=False,
        contains_attack_labels=False,public_noise_scales=[1,2]*5)
    return payload,clean,updates


def test_offline_exact_error_decomposition_and_undefined_correlations():
    payload,clean,updates=evaluation_fixture()
    result=evaluate_aggregation_roles(payload,clean,updates)
    for name in ARMS:
        p='cohort_'+name+'_'
        assert result[p+'decomposition_residual_sq']<1e-20
        terms=['honest_effective_noise_sq','honest_tilt_sq','byzantine_centered_sq','cross_noise_tilt','cross_noise_byzantine','cross_tilt_byzantine']
        assert sum(result[p+k] for k in terms)==pytest.approx(result[p+'aggregate_error_sq'],abs=1e-12)
    assert result['cohort_uniform_weight_effective_noise_corr_honest'] is None
    assert result['cohort_uniform_byzantine_mass']==pytest.approx(.2)


def test_oracle_changes_do_not_change_candidates():
    payload,clean,updates=evaluation_fixture()
    before={k:v.clone() for k,v in payload['rules']['aggregates'].items()}
    r1=evaluate_aggregation_roles(payload,clean,updates)
    changed={i:{'weight':torch.zeros_like(v['weight'])} for i,v in clean.items()}
    r2=evaluate_aggregation_roles(payload,changed,updates)
    assert r1['deployed_aggregate_error_sq']!=r2['deployed_aggregate_error_sq']
    assert all(torch.equal(v,payload['rules']['aggregates'][k]) for k,v in before.items())


def test_oracle_contamination_and_cohort_tampering_fail_closed():
    p,c,u=evaluation_fixture();p['contains_attack_labels']=True
    with pytest.raises(ValueError,match='forbidden'):evaluate_aggregation_roles(p,c,u)
    p,c,u=evaluation_fixture();p['vectors']=p['vectors']+1
    with pytest.raises(ValueError,match='cohort differs'):evaluate_aggregation_roles(p,c,u)
    a=LDPAggregationRoleAblation();updates=toy_updates(0);updates[0][1]['is_byzantine']=True
    with pytest.raises(ValueError,match='oracle crossed'):a.server_aggregate(torch.nn.Linear(32,1,bias=False),updates,0,toy_config('uniform'))


def test_matrix_and_private_channel_identical_across_arms():
    m=runner.matrix();stamp=runner.provenance(m);ts=runner.tasks(m)
    assert len(ts)==96 and len({str(runner.directory(t)) for t in ts})==96
    assert all('aggregation_role_n10_v1' in str(runner.directory(t)) for t in ts)
    cfgs=[runner.config_for(m,dict(ts[0],arm=arm),stamp) for arm in ARMS]
    for key in ['noise_multiplier','privacy_sampling_rate_override','fixed_batch_size','clip_norm','far_server_clip_norm','far_alpha','privacy_noise_multiplier_scale_by_client']:
        assert all(c['training']['algo_config'][key]==cfgs[0]['training']['algo_config'][key] for c in cfgs)
    assert abs(stamp['privacy']['per_scale']['1']['epsilon']-4)<1e-4
