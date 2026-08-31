"""Fast integration tests for the S0.5 reference-comparator audit."""

from __future__ import annotations

import json
import math

from scripts.run_scfar_reference_comparators import (
    build_parser,
    run_audit,
    write_outputs,
)


def _args(*extra: str):
    return build_parser().parse_args(
        [
            "--n",
            "5",
            "--dimension",
            "4",
            "--trials",
            "3",
            "--clip",
            "1.0",
            "--distance-clip",
            "2.0",
            "--alpha",
            "0.2",
            "--kappa-w",
            "2.0",
            *extra,
        ]
    )


def test_s0_5_covers_required_comparators_and_keeps_claims_separate():
    records, summary = run_audit(_args())
    assert summary["methods"] == ["mean", "cm", "trmean", "rfa"]
    assert len(records) == 4 * (3 + 2)
    assert summary["all_mean_certificates_hold"] is True
    assert summary["all_aggregate_2C_bounds_hold"] is True

    for method in ("cm", "trmean", "rfa"):
        rows = [row for row in records if row["method"] == method]
        assert rows
        assert all(row["reference_bound"] is None for row in rows)
        assert all(row["reference_certificate_holds"] is None for row in rows)


def test_s0_5_stress_families_expose_non_vanishing_robust_reference_shift():
    records, _ = run_audit(_args("--alpha", "0"))
    cm = next(
        row
        for row in records
        if row["method"] == "cm" and row["scenario"] == "majority_axis_stress"
    )
    trmean = next(
        row
        for row in records
        if row["method"] == "trmean" and row["scenario"] == "trim_boundary_stress"
    )
    assert math.isclose(cm["reference_shift"], 2.0, rel_tol=1e-12)
    assert math.isclose(trmean["reference_shift"], 2.0 / 3.0, rel_tol=1e-12)


def test_s0_5_writes_reproducible_csv_and_json(tmp_path):
    records, summary = run_audit(_args())
    write_outputs(records, summary, tmp_path)
    assert (tmp_path / "replace_one_comparators.csv").exists()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["audit"] == "S0.5_reference_comparators"
    assert loaded["status"] == "empirical_falsification_audit_not_dp_certificate"
