from scripts.analyze_scfar_fmnist_step1b import (
    RunSummary,
    _percentile,
    build_report,
)


def test_percentile_uses_linear_interpolation_and_ignores_missing_values():
    rows = [{"x": 0.0}, {"x": None}, {"x": 1.0}, {"x": 2.0}]
    assert _percentile(rows, "x", 0.5) == 1.0
    assert _percentile(rows, "x", 0.9) == 1.8


def _summary(*, method: str, anchor: str, tau: float) -> RunSummary:
    return RunSummary(
        task_id=f"{method}-{anchor}-{tau}",
        method=method,
        anchor=anchor,
        tau_over_c=tau,
        test_accuracy_pct=75.0,
        worst20_pct=50.0,
        gap_pct=30.0,
        variance_pct2=100.0,
        median_user_clip=0.14,
        median_score_span=0.4,
        score_saturation_p90=0.1,
        median_qmax=0.06,
        median_concentration=1.05,
        median_entropy=3.1,
        median_reference_error=0.1,
        median_anchor_drift=0.1,
    )


def test_report_contains_one_control_and_nine_tilted_configurations():
    summaries = [_summary(method="central_dp_fedavg_exact", anchor="fixed_zero", tau=1.0)]
    for anchor in ("fixed_zero", "ema_release_0p1", "previous_release"):
        for tau in (0.25, 0.5, 1.0):
            summaries.append(_summary(method="scfar_no_dp", anchor=anchor, tau=tau))
    report = build_report(summaries)
    assert "Gate géométrique partiel" in report
    assert report.count("**oui**") == 9
    assert "2 époques locales" in report
