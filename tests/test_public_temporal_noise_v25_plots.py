"""Plot construction tests on explicitly synthetic fixtures, outside real outputs."""
import json
import pytest
from scripts import plot_public_temporal_noise_confirmation_v25 as plot


def test_partial_audit_is_not_plotted_as_confirmation(tmp_path,monkeypatch):
    source=tmp_path/'synthetic_partial.json'
    source.write_text(json.dumps(dict(audit_passed=True,runs=16,snapshot_partial=True)))
    monkeypatch.setattr(plot,'SOURCE',source); monkeypatch.setattr(plot,'OUT',tmp_path/'figures')
    with pytest.raises(AssertionError):
        plot.main()
    assert not (tmp_path/'figures').exists()


def test_two_readable_figures_from_complete_synthetic_fixture(tmp_path,monkeypatch):
    rows=[]
    for seed in range(4):
        for mode in (0,13):
            for method in ('erm_mean','erm_rfa','risk_mean','risk_rfa'):
                value=70.+seed+.1*mode
                rows.append(dict(job=dict(seed=seed,grid_index=mode,method=method),test=dict(
                    accuracy_pct=value,worst20_pct=value-10,gap_best20_worst20_pp=15.+seed,variance_pp2=30.+seed)))
    contrasts=[]
    for mode in (0,13):
        for method in ('erm_mean','erm_rfa'):
            contrasts.append(dict(control_grid_index=mode,control_method=method,
                summaries={k:dict(mean=0.,ci95=[-1.,1.]) for k in ('accuracy_pct','worst20_pct')},
                pairs=[dict(delta=dict(accuracy_pct=.1*i,worst20_pct=.2*i)) for i in range(4)]))
    audit=dict(audit_passed=True,runs=32,snapshot_partial=False,final_test_records=rows,
               decision=dict(contrasts=contrasts,clean_confirmation_passed=False))
    source=tmp_path/'SYNTHETIC_NOT_EXPERIMENTAL.json'; source.write_text(json.dumps(audit))
    monkeypatch.setattr(plot,'SOURCE',source); monkeypatch.setattr(plot,'OUT',tmp_path/'figures')
    plot.main()
    for name in ('test_metrics.png','paired_primary_intervals.png'):
        path=tmp_path/'figures'/name
        assert path.exists() and path.stat().st_size>10000
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
