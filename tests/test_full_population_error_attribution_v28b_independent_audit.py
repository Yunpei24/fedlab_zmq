import torch
from privacy.fair_objective import require_mps,per_example,losses
from scripts.audit_full_population_error_attribution_v28b import polynomial_brier,population


def test_polynomial_brier_values_and_gradient_match_onehot_expression():
    require_mps();torch.mps.manual_seed(281728)
    logits=torch.randn(17,10,device='mps',requires_grad=True);y=torch.arange(17,device='mps')%10
    a=polynomial_brier(logits,y);b=losses(logits,y,'brier')
    torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-7)
    ga=torch.autograd.grad(a.sum(),logits,retain_graph=True)[0];gb=torch.autograd.grad(b.sum(),logits)[0]
    torch.testing.assert_close(ga,gb,rtol=3e-5,atol=3e-7)


def test_direct_batched_population_gradient_matches_individual_gradient_mean():
    require_mps();torch.manual_seed(2828);torch.mps.manual_seed(2828)
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2)).to('mps').eval()
    x=torch.randn(17,3,device='mps');y=torch.arange(17,device='mps')%2
    state={k:v.clone() for k,v in model.state_dict().items()}
    r,_,_,raw=per_example(model,x,y,kind='brier',clip_norm=.01)
    rr,g=population(model,x,y,block_size=5)
    torch.testing.assert_close(rr,r.mean(),rtol=3e-5,atol=3e-7)
    torch.testing.assert_close(g,raw,rtol=3e-5,atol=3e-7)
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])
    assert all(p.grad is None for p in model.parameters())
