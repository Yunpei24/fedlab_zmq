import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.aggregate_radial_step_v21 import controlled_step


def test_three_controls_and_radial_cap():
    require_mps();a=torch.tensor([3.,4.],device='mps')
    for mode,expected in [('unchanged',[6.,8.]),('half',[3.,4.]),('global_clip',[1.2,1.6])]:
        s,d=controlled_step(a,2.,mode)
        torch.testing.assert_close(s,torch.tensor(expected,device='mps'))
        if mode=='global_clip':assert d['global_clipped'] and d['step_norm']<=2.000002


def test_zero_and_inactive_clip():
    require_mps()
    for a in [torch.zeros(4,device='mps'),torch.ones(4,device='mps')*.1]:
        x,d=controlled_step(a,.5,'global_clip')
        assert not d['global_clipped'];torch.testing.assert_close(x,.5*a,rtol=0,atol=0)


def test_does_not_clip_each_client_before_aggregation():
    require_mps();messages=torch.tensor([[10.,0.],[-8.,0.]],device='mps')
    s,d=controlled_step(messages.mean(0),1.,'global_clip')
    torch.testing.assert_close(s,torch.tensor([1.,0.],device='mps'))
    # Clipping each row first would instead give zero, a different algorithm.


def test_invalid_data_rejected():
    require_mps();a=torch.ones(3,device='mps')
    for eta in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):controlled_step(a,eta,'unchanged')
    with pytest.raises(ValueError):controlled_step(a,1.,'unknown')
    with pytest.raises(ValueError):controlled_step(a*float('nan'),1.,'global_clip')


def test_smaller_mse_does_not_override_gate():
    from scripts.diagnose_prisma_global_step_v21 import decide,SEEDS,ROUNDS,BRANCHES
    records=[]
    for seed in SEEDS:
        for k in ROUNDS:
            rows=[]
            for branch in BRANCHES:
                candidate=branch=='recursive__global_clip'
                rows.append(dict(branch=branch,J_gain=.1 if candidate else 0.,
                    validation=dict(worst20_pct=51. if candidate else 50.,accuracy_pct=70.)))
            records.append(dict(seed=seed,round=k,rows=rows))
    assert decide(records)['admitted_to_end_to_end_screen']
    records[0]['rows'][2]['validation']['accuracy_pct']=60.
    assert not decide(records)['admitted_to_end_to_end_screen']
