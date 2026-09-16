import math
import pytest
from scripts.diagnose_private_risk_weights_v25 import (
    weights,round_diagnostic,top_two_membership,summarize,
)


def test_without_noise_the_weight_distortion_vanishes():
    risks=[.05*i for i in range(10)]
    r=round_diagnostic(risks,risks,weights(risks))
    assert r['weight_total_variation_noise']==0
    assert r['risk_targeting_ideal']>0
    assert r['risk_targeting_private']==r['risk_targeting_ideal']
    assert r['top_two_true_risk_mass_private']>.2
    assert math.isclose(r['correlation_private_weight_true_risk'],1.,abs_tol=1e-13)


def test_inversion_of_private_reports_is_identified_not_hidden():
    risks=[.05*i for i in range(10)]; private=list(reversed(risks))
    r=round_diagnostic(risks,private,weights(private))
    assert r['risk_targeting_private']<0 and r['risk_targeting_ideal']>0
    assert r['weight_total_variation_noise']>0
    assert r['top_two_true_risk_mass_private']<.2
    summary=summarize([dict(round=1,**r)])
    assert summary['negative_true_risk_targeting_round_fraction']==1.


def test_ties_do_not_arbitrarily_select_clients():
    assert top_two_membership([.3]*10)==[.2]*10
    high=top_two_membership([.4]+[.3]*3+[.1]*6)
    assert high==[1.]+[1/3]*3+[0.]*6
    r=round_diagnostic([.3]*10,[.3]*10,[.1]*10)
    assert r['correlation_private_weight_true_risk'] is None
    assert math.isclose(r['top_two_true_risk_mass_private'],.2,abs_tol=1e-15)


def test_solver_weights_are_not_silently_treated_as_exact():
    risks=[.05*i for i in range(10)]; objective=weights(risks)
    solver=dict(stationary_weights=[.1]*10,stationary_reconstruction_error=.123)
    r=round_diagnostic(risks,risks,objective,solver)
    assert r['stationary_reconstruction_error']==.123
    assert r['effective_weight_total_variation_vs_private']>0
    assert math.isclose(r['effective_weight_concentration'],1.,abs_tol=1e-15)
    with pytest.raises(AssertionError):
        round_diagnostic(risks,risks,[.2]*10)
