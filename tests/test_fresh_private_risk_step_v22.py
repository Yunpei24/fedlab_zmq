from scripts.run_fresh_private_risk_step_v22 import jobs,contrast_passes


def test_matrix_has_two_controls_for_every_method_seed():
    js=jobs(dict(seeds=[170501,170502],step_controls=['half','global_clip']))
    assert len(js)==16 and len({tuple(sorted(j.items())) for j in js})==16
    assert all('recursive' not in str(j) for j in js)


def test_predefined_margins_not_average_only():
    good=dict(accuracy_pct=-.8,worst20_pct=1.2,gap_best20_worst20_pp=-1.,variance_pp2=-2.)
    assert contrast_passes([good,good])
    assert not contrast_passes([good,dict(good,worst20_pct=.99)])
    assert not contrast_passes([good,dict(good,accuracy_pct=-1.01)])
    assert not contrast_passes([good,dict(good,variance_pp2=3.)])
