from copy import deepcopy
import pytest
from scripts.analyze_private_risk_batch_noise_v31 import check_row,close,summarize


def toy():
    matrix=[[0]*10 for _ in range(10)]
    for k in range(10):matrix[k][k]=900;matrix[k][(k+1)%10]=300
    clients=[dict(N=1200,class_count=[120]*10,class_hits=[90]*10) for _ in range(10)]
    return dict(metrics=dict(clients=clients,accuracy_pct=75.,worst20_pct=75.,
        gap_best20_worst20_pp=0.,variance_pp2=0.,balanced_accuracy_pct=75.),
        confusion=matrix,noise='zero_oracle',std=0.,
        decomposition=dict(error_squared=1.,clean_error_squared=1.,noise_displacement_squared=0.,cross_term=0.))


def test_counts_and_zero_noise_identity():check_row(toy())


def test_diagonal_tampering_is_detected():
    r=deepcopy(toy());r['confusion'][2][2]-=1;r['confusion'][2][3]+=1
    with pytest.raises(AssertionError):check_row(r)


def test_no_nonfinite_or_pseudoreplication():
    with pytest.raises(ValueError):close(float('nan'),0)
    with pytest.raises(ValueError):summarize([1]*96)


def test_wrong_error_decomposition_is_detected():
    r=toy();r['decomposition']['cross_term']=.1
    with pytest.raises(ValueError):check_row(r)
