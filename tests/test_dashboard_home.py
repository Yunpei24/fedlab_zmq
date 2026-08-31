from pathlib import Path

from dashboard.home import _experiment_families, _track_counts


def test_home_counts_research_tracks_without_mixing_runs(tmp_path: Path) -> None:
    run_dirs = [
        tmp_path / "algorithm_fidelity_v4" / "exp1_fairness_no_attack" / "run_a",
        tmp_path / "algorithm_fidelity_v4" / "exp1_fairness_no_attack" / "run_b",
        tmp_path / "algorithm_fidelity_v4" / "exp2_fairness_robustness" / "run_c",
        tmp_path / "unrelated_benchmark" / "run_d",
    ]

    counts = _track_counts(run_dirs)

    assert counts == {"R1": 2, "R2": 1}
    assert _experiment_families(run_dirs) == 2

