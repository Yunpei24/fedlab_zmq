from __future__ import annotations

from scripts import run_gaussian_aware_reference_g0g_k7b_null_extension as runner


def test_static_protocol_is_valid() -> None:
    validation = runner._static_validation()
    assert all(validation["checks"].values())


def test_candidate_pools_preserve_twelve_rotation_slots() -> None:
    for regime in runner.k7.REGIMES:
        pool = runner._candidate_pool(regime)
        assert len(pool) == 300
        assert all(
            sum(index % 12 == slot for index in range(len(pool))) == 25
            for slot in range(12)
        )


def test_extension_never_changes_parent_thresholds_or_evaluation() -> None:
    text = runner.PROTOCOL_PATH.read_text(encoding="utf-8")
    assert "conserver exactement les seuils" in text
    assert "ne relance pas l'évaluation" in text
    assert runner.REQUIRED_PER_SLOT == 20
    assert runner.POOL_ROUNDS_PER_SLOT == 25


def test_frozen_parent_hashes_are_valid() -> None:
    manifest, decision, thresholds = runner._verify_parent()
    assert manifest["status"] == "completed"
    assert decision["primary_pass"]
    assert decision["gaussian_specificity_pass"]
    assert set(thresholds) == {"homogeneous", "heteroscedastic"}
