import pytest
import torch
from scripts.diagnose_private_risk_batch_noise_v31 import components, validate_confusion
from scripts.analyze_v28_terminal_aggregation_counterfactual import controls
from privacy.fair_objective import require_mps


def test_cross_term_is_kept_not_variance_added_blindly():
    require_mps()
    t=lambda x:torch.tensor(x,device='mps',dtype=torch.float32)
    result=components(t([3,0]),t([1,0]),t([0,0]))
    assert result==dict(error_squared=9.,clean_error_squared=1.,noise_displacement_squared=4.,cross_term=4.)


def test_same_noise_fixed_mean_weights_separates_sampling():
    require_mps()
    x=torch.arange(30,device='mps',dtype=torch.float32).reshape(10,3)/10
    y=x.flip(0)
    z=torch.cos(x)
    w=torch.arange(1,11,device='mps',dtype=torch.float32);w=w/w.sum()
    a,_=controls(x+.03*z,w);b,_=controls(y+.03*z,w)
    torch.testing.assert_close(a['mean']-b['mean'],(w[:,None]*(x-y)).sum(0),rtol=1e-5,atol=1e-6)


def test_confusion_requires_all_classes_and_correct_diagonal():
    matrix=[[0]*10 for _ in range(10)]
    for k in range(10):matrix[k][k]=2
    evaluation=dict(clients=[dict(class_count=[2]*10,class_hits=[2]*10)])
    validate_confusion(matrix,evaluation)
    matrix[2][2]=1
    with pytest.raises(AssertionError):validate_confusion(matrix,evaluation)
