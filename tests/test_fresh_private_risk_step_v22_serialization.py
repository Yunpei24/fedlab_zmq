import json
from scripts.run_fresh_private_risk_step_v22_r1 import jobs,contrast_passes


def test_matrix_and_gate_unchanged():
    assert len(jobs(dict(seeds=[170501,170502],step_controls=['half','global_clip'])))==16
    good=dict(accuracy_pct=-.8,worst20_pct=1.2,gap_best20_worst20_pp=-1.,variance_pp2=-2.)
    assert contrast_passes([good,good])
    assert not contrast_passes([good,dict(good,worst20_pct=.99)])
    assert not contrast_passes([good,dict(good,accuracy_pct=-1.01)])


def test_json_order_keys_preserve_every_numeric_field():
    from scripts.run_private_clipping_step_diagnostic_v16 import ledger
    from scripts.run_fresh_private_risk_step_v22 import prior
    for kind in ('erm_mean','erm_rfa','risk_mean','risk_rfa'):
        calculated=ledger(2.,kind)
        stored=json.loads((prior.OUT/f'seed170501__fresh__{kind}/metrics.json').read_text())['privacy']
        assert json.loads(json.dumps(calculated))==stored
        wrong=json.loads(json.dumps(calculated));wrong['gradient_std']+=1e-8
        assert wrong!=stored
        if kind.startswith('risk_'):
            assert calculated!=stored
            assert calculated['rdp']=={int(k):v for k,v in stored['rdp'].items()}
