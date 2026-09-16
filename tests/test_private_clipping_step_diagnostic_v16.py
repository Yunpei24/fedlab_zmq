import math
import torch
from privacy.fair_objective import require_mps, per_example
from scripts import run_private_clipping_step_diagnostic_v16 as run


def test_reclipping_matches_direct_per_example_gradients():
    require_mps();torch.mps.manual_seed(816)
    model=torch.nn.Linear(4,3).to('mps')
    x=20*torch.randn(12,4,device='mps');y=torch.arange(12,device='mps')%3
    _,g8,_,_=per_example(model,x,y,clip_norm=8.)
    for C in (2.,4.,8.):
        _,direct,_,_=per_example(model,x,y,clip_norm=C)
        torch.testing.assert_close(run.reclipped(g8,C),direct,rtol=1e-5,atol=2e-6)


def test_identical_applied_message_noise_at_constant_product():
    for rule in run.RULES:
        values=[]
        for C,eta in run.COUPLES[:3]:
            p=run.ledger(C,rule);assert p['epsilon_realized']<=4
            values.append(eta*p['gradient_std'])
        assert max(values)-min(values)<1e-15
    assert run.TOTAL==160 and run.SEEDS==[170501,170502]


def test_gate_requires_both_states_and_distinguishes_step_only():
    gs=[]
    for seed in run.SEEDS:
        for C,eta in run.COUPLES:
            for rule in run.RULES:
                fair=rule=='risk_rfa' and C>2
                gs.append(dict(seed=seed,C=C,eta=eta,rule=rule,n=4,
                    J_gain=.01 if fair else .005,accuracy_delta_pp=0.,worst20_delta_pp=1. if fair else 0.))
    assert run.decision(gs)['selected_for_end_to_end_calibration']==dict(C=4.,eta=.25)
    for g in gs:
        if g['seed']==run.SEEDS[1] and g['rule']=='risk_rfa' and g['C']>2:g['J_gain']=0.
    assert run.decision(gs)['selected_for_end_to_end_calibration'] is None


def test_sources_exclude_confirmation_test_checkpoints():
    profile,stamp=run.inputs()
    assert profile['model']=='lenet5_tanh'
    assert all('17060' not in p for p in stamp if p.endswith('checkpoint.pt'))
