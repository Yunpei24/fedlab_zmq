import math
import torch
from privacy.fair_objective import require_mps
from scripts import run_private_recursive_query_diagnostic_v17 as run


def test_exact_tracking_without_increment_clipping_and_theta_one_identity():
    require_mps();torch.mps.manual_seed(817)
    before=.2*torch.randn(12,5,device='mps');now=before+.1*torch.randn_like(before)
    delta,_=run.clipped_increment(now,before,10.)
    for theta in (.1,.2,.5,1.):
        result=(1-theta)*before.mean(0)+theta*now.mean(0)+(1-theta)*delta.mean(0)
        torch.testing.assert_close(result,now.mean(0),rtol=1e-5,atol=1e-6)
    delta,_=run.clipped_increment(now,before,.01)
    assert float(torch.linalg.vector_norm(delta,dim=1).max())<=.010001


def test_clipping_bias_identity_and_replace_one_sensitivity():
    require_mps();torch.mps.manual_seed(818)
    now=torch.randn(20,6,device='mps');before=torch.randn_like(now)
    now*=torch.minimum(torch.ones(20,device='mps'),2/torch.linalg.vector_norm(now,dim=1))[:,None]
    before*=torch.minimum(torch.ones(20,device='mps'),2/torch.linalg.vector_norm(before,dim=1))[:,None]
    for theta in run.THETAS:
        for ratio in run.RATIOS:
            D=2*ratio;delta,_=run.clipped_increment(now,before,D)
            q=theta*now+(1-theta)*delta
            result=(1-theta)*before.mean(0)+q.mean(0)
            bias=(1-theta)*(delta-(now-before)).mean(0)
            torch.testing.assert_close(result-now.mean(0),bias,rtol=2e-5,atol=2e-6)
            assert float(torch.linalg.vector_norm(q,dim=1).max())<=theta*2+(1-theta)*D+1e-6
            changed=q.clone();changed[0]=-q[0]
            assert float(torch.linalg.vector_norm(changed.mean(0)-q.mean(0)))<=2*(theta*2+(1-theta)*D)/20+1e-6


def test_variance_recursion_and_quantiles():
    for theta in run.THETAS+[1.]:
        for ratio in run.RATIOS:
            v=1.;r=theta+(1-theta)*ratio
            for _ in range(20):v=(1-theta)**2*v+r*r
            assert math.isclose(v,run.public_ratios(theta,ratio)['finite_noise_variance_ratio'],rel_tol=1e-12)
    assert run.quantile([0.,10.],.9)==9.


def test_only_calibration_states_and_exact_screen_size():
    _,stamp=run.inputs()
    assert run.SEEDS==[170501,170502] and len(run.THETAS)*len(run.RATIOS)==15
    assert all('17060' not in p for p in stamp if p.endswith('.pt'))
