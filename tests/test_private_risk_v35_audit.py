import torch
from privacy.fair_objective import require_mps
from scripts.audit_private_risk_spectral_v35 import power_bounds


def test_independent_power_enclosures_known_spectrum():
    require_mps()
    m=torch.tensor([[[2.,1.],[1.,2.]],[[4.,-2.],[-2.,1.]],[[0.,0.],[0.,0.]],[[3.,0.],[0.,3.]]],device='mps')
    lo,up=power_bounds(m);target=torch.tensor([3.,5.,0.,3.],device='mps')
    assert bool((lo<=target+2e-5).all())
    assert bool((up>=target-2e-5).all())
    torch.testing.assert_close(lo,target,rtol=2e-5,atol=2e-5)
    assert bool((up-target<=.005).all())
