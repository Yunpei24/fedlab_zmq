import copy
from fractions import Fraction as Q
import inspect
import json
import math
import mpmath as mp
import pytest
import torch
from privacy import full_population_private_risk_confirmation_v29 as m
from privacy import full_population_private_risk_v28 as full
from privacy.fair_objective import require_mps, per_example, release
from scripts import run_full_population_private_risk_confirmation_v29 as runner


def test_frozen_matrix_prerequisites_and_endpoint():
    c,p,stamp=runner.inputs()
    assert len(runner.jobs())==len({runner.identifier(j) for j in runner.jobs()})==24
    assert c['seeds']==list(m.SEEDS) and c['primary_method']=='risk_rfa' and c['primary_batch']==4800
    assert c['minimum_worst20_advantage_pp']==c['maximum_accuracy_loss_pp']==1
    assert c['test_rounds']==[120] and not c['automatic_attacks'] and not c['automatic_retry']
    assert p['local_optimizer_steps']==0
    code=inspect.getsource(runner.train)
    assert "test=base.evaluate(model,data,'test') if t+1==120 else None" in code
    assert 'optimizer.step' not in code and 'pre_round_model=before' in code
    assert 'last_private_messages=' in code and 'checkpoint_sha256' in code
    assert any('Confirmation_Ledger_V29.json' in f for f in stamp)
    runner.base.verify_stamp(stamp)


def test_plan_matches_prior_arms_and_all_prefixes():
    for batch,method in m.ARMS:
        p=m.plan(batch,method)
        if batch==4800: assert p==full.ledger(method)
        es=[m.prefix(p,t)[0] for t in range(121)]
        assert es[0]==0 and all(a<=b<=4 for a,b in zip(es,es[1:]))
        assert math.isclose(es[-1],p['epsilon_realized'],abs_tol=1e-12)
        assert math.isclose(p['gradient_std']/p['gradient_z'],4/batch,rel_tol=1e-15)
        assert p['gradient_releases']==120 and p['risk_releases']==(120 if method.startswith('risk_') else 0)
        assert m.prefix(p,120)==m.prefix(json.loads(json.dumps(p)),120)
    for b,k in ((240,'risk_mean'),(240,'risk_rfa'),(960,'erm_mean'),(4800,'unregistered')):
        with pytest.raises(ValueError): m.plan(b,k)
    for t in (-1,121,1.5,True):
        with pytest.raises(ValueError): m.prefix(m.plan(4800,'risk_rfa'),t)


def sample(n):
    require_mps(); torch.manual_seed(291); torch.mps.manual_seed(291)
    model=torch.nn.Linear(3,2).to('mps').eval()
    x=10*torch.randn(n,3,device='mps'); y=torch.arange(n,device='mps')%2
    return model,x,y


def test_small_query_is_original_v19_reduction_and_one_release(monkeypatch):
    model,x,y=sample(240); before={k:v.clone() for k,v in model.state_dict().items()}
    _,rows,norms,_=per_example(model,x,y,kind='brier',clip_norm=2.)
    calls=[]
    def spy(q,*,noise_std,seed):
        calls.append(q.clone()); return release(q,noise_std=noise_std,seed=seed)
    monkeypatch.setattr(m,'release',spy)
    out,q,d=m.private_message(model,x,y,batch=240,noise_std=.1,seed=552)
    assert len(calls)==1 and torch.equal(q,rows.mean(0)) and torch.equal(out,release(q,noise_std=.1,seed=552))
    assert d['per_example_clipped_count']==int((norms>2).sum()) and d['gaussian_releases']==1
    assert d['replace_one_sensitivity']==4/240 and d['local_optimizer_steps']==0
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    assert all(p.grad is None for p in model.parameters())


def test_full_query_keeps_single_release_and_fixed_model(monkeypatch):
    model,x,y=sample(4800); before={k:v.clone() for k,v in model.state_dict().items()}; calls=[]
    def spy(q,*,noise_std,seed):
        calls.append(q.clone()); return release(q,noise_std=noise_std,seed=seed)
    monkeypatch.setattr(full,'release',spy)
    out,q,d=m.private_message(model,x,y,batch=4800,noise_std=.1,seed=553)
    assert len(calls)==1 and d['accumulation_blocks']==20 and d['gaussian_releases']==1
    assert d['replace_one_sensitivity']==4/4800
    assert torch.equal(out,release(q,noise_std=.1,seed=553))
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])


def test_invalid_query_and_cpu_inputs_rejected():
    model,x,y=sample(240)
    with pytest.raises(ValueError): m.private_message(model,x.cpu(),y,batch=240,noise_std=.1,seed=1)
    for b,s in ((4800,.1),(240,0.),(240,float('nan'))):
        with pytest.raises(ValueError): m.private_message(model,x,y,batch=b,noise_std=s,seed=1)
    model.train()
    with pytest.raises(ValueError): m.private_message(model,x,y,batch=240,noise_std=.1,seed=1)


def test_full_and_small_indices_are_paired_without_changing_rng():
    require_mps(); state=torch.mps.get_rng_state()
    all_=runner.base.draw_indices(4800,4800,282); small=runner.base.draw_indices(4800,240,282)
    assert torch.equal(all_[:240],small) and len(torch.unique(all_))==4800
    assert torch.equal(torch.mps.get_rng_state(),state)


def test_t_threshold_independent_integral_and_wave_budget():
    with mp.workdps(60):
        x=mp.mpf(m.TCRIT); tail=mp.betainc(mp.mpf('1.5'),mp.mpf('.5'),0,3/(3+x*x),regularized=True)/2
        assert abs(tail-mp.mpf('.0125'))<mp.mpf('1e-17')
        # V25 critical value must not accidentally be reused at the stricter wave.
        assert m.TCRIT>3.182446305284263


def differences():
    return [dict(zip(m.KEYS,(Q(0),Q(2),Q(-1),Q(-2)))) for _ in range(4)]


def test_all_seed_margins_and_statistical_lower_bounds():
    ds=differences(); assert m.compare(ds)['passed']
    ds[0]['worst20_pct']=Q(99,100); assert not m.compare(ds)['passed']
    ds=differences(); ds[0]['accuracy_pct']=Q(-101,100); assert not m.compare(ds)['passed']
    ds=differences()
    for i,w in enumerate((1,1,1,100)): ds[i]['worst20_pct']=Q(w)
    r=m.compare(ds); assert r['gates']['all_seed_gates'] and not r['gates']['worst20_lower_positive']
    for key in ('gap_best20_worst20_pp','variance_pp2'):
        ds=differences(); ds[0][key]=Q(100); assert not m.compare(ds)['passed']
    with pytest.raises(ValueError): m.compare(differences()[:3])


def metrics(candidate):
    hits=[650,650,800,800,800,800,800,800,900,900] if candidate else [600,600,800,800,800,800,800,800,900,900]
    aa=[Q(h,10) for h in hits]; mean=sum(aa)/10; ss=sorted(aa); worst=sum(ss[:2])/2
    return dict(accuracy_pct=float(mean),worst20_pct=float(worst),gap_best20_worst20_pp=float(sum(ss[-2:])/2-worst),
        variance_pp2=float(sum((v-mean)**2 for v in aa)/10),
        clients=[dict(N=1000,class_count=[1000]+[0]*9,class_hits=[h]+[0]*9) for h in hits])


def records():
    return [dict(job=j,device='mps',test_evaluated=True,test_evaluation_rounds=[120],
        final=dict(round=120,test=metrics(j['batch']==4800 and j['method']=='risk_rfa'))) for j in runner.jobs()]


def test_fixed_primary_matrix_and_no_partial_or_checkpoint_selection():
    rs=records(); assert m.decide(rs)['clean_confirmation_passed']
    for bad in (rs[:-1],rs[:-1]+[rs[0]]):
        with pytest.raises(ValueError): m.decide(bad)
    bad=copy.deepcopy(rs); bad[0]['final']['round']=90
    with pytest.raises(ValueError): m.decide(bad)
    bad=copy.deepcopy(rs); bad[0]['test_evaluation_rounds']=[90,120]
    with pytest.raises(ValueError): m.decide(bad)
    bad=copy.deepcopy(rs)
    for r in bad:
        if r['job']['method']=='risk_mean': r['final']['test']=metrics(True)
        if r['job']['method']=='risk_rfa': r['final']['test']=metrics(False)
    assert not m.decide(bad)['clean_confirmation_passed']


def test_exact_counts_invalid_or_rounded_metrics_fail():
    v=metrics(True); r=m.exact_test_metrics(v); assert all(isinstance(x,Q) for x in r.values())
    v['accuracy_pct']+=.01
    with pytest.raises(ValueError): m.exact_test_metrics(v)
    v=metrics(True); v['clients'][0]['class_hits'][0]=1001
    with pytest.raises(ValueError): m.exact_test_metrics(v)
