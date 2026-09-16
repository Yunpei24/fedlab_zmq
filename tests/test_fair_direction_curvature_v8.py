import pytest
import torch
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import potential
from scripts import run_fair_direction_curvature_v8 as run


def test_protocol_never_trains_or_selects_on_test():
    m=run.config()
    assert len(m['calibration_seeds'])*m['replays_per_seed']*len(m['etas'])*len(m['conditions'])==192
    assert m['source_arm']=='erm_mean_C2' and not m['automatic_promotion']


def test_report_objective_matches_differentiable_potential():
    require_mps()
    r=torch.tensor([.1,.2,.6,.8],device='mps')
    assert run.objective(r.cpu().tolist(),.5)==pytest.approx(float(potential(r,.5).mean()),abs=1e-7)
