import itertools
import torch
from privacy.fair_objective import require_mps,per_example,losses
from privacy.capped_private_risk import potential,weights
from privacy.complete_recursive_query import queries


def test_full_risk_chain_rule_agrees_with_direct_autograd():
    require_mps();torch.mps.manual_seed(88020)
    model=torch.nn.Linear(3,2).to('mps');xs=[torch.randn(7,3,device='mps') for _ in range(3)]
    ys=[torch.arange(7,device='mps')%2 for _ in range(3)]
    rs=[];gs=[]
    for x,y in zip(xs,ys):
        r,_,_,g=per_example(model,x,y,clip_norm=2.)
        rs.append(r.mean());gs.append(g)
    rs=torch.stack(rs);gs=torch.stack(gs);lam,coeff=weights(rs,.5)
    derived=(coeff[:,None]*gs).mean(0)
    objective=potential(torch.stack([losses(model(x),y,'brier').mean() for x,y in zip(xs,ys)]),.5).mean()
    objective.backward();direct=torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(derived,direct,rtol=3e-5,atol=2e-7)
    torch.testing.assert_close(derived/coeff.mean(),(lam[:,None]*gs).sum(0),rtol=2e-5,atol=2e-7)


def test_conditional_bias_sampling_and_private_memory_identity():
    require_mps();torch.mps.manual_seed(88021)
    current=torch.randn(6,4,device='mps')*.25;previous=torch.randn_like(current)*.25
    exact,old,projected,_=queries(current,previous,C=2.,D=.5,theta=.25)
    mem=torch.randn(4,device='mps');noise=.03*torch.randn_like(mem);a=.75
    c=current.mean(0);cp=previous.mean(0);idx=torch.tensor([0,3],device='mps')
    for q in (old,projected):
        pop=q.mean(0);batch=q[idx].mean(0);message=a*mem+batch+noise
        past=a*(mem-cp);bias=pop-(c-a*cp);sampling=batch-pop
        torch.testing.assert_close(message-c,past+bias+sampling+noise,rtol=2e-5,atol=2e-7)
        possible=[]
        for indices in itertools.combinations(range(6),2):
            possible.append(q[torch.tensor(indices,device='mps')].mean(0)-pop)
        torch.testing.assert_close(torch.stack(possible).mean(0),torch.zeros_like(pop),rtol=0,atol=1e-7)
