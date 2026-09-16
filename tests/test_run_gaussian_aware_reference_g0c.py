from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.run_gaussian_aware_reference_g0c import (
    _materialize_reference_config,
    _reference_specs,
    _select_reference,
    _validate_g0c,
    _weight_specs_and_configs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0c.yaml"


def _load():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_g0c_preregisters_disjoint_fresh_splits_and_full_reference_grid():
    config = _load()
    _validate_g0c(config)
    specs = _reference_specs(config)
    gaussian = [spec for spec in specs.values() if spec["method"] == "f_sigma_huber"]
    grid = config["references"]["reference_grid"]
    assert len(gaussian) == (
        len(grid["radial_scales"])
        * len(grid["influence_cap_totals"])
        * len(grid["regularizations"])
    )
    random = config["randomness"]
    groups = [
        {random["null_calibration_seed"]},
        set(random["reference_development_seeds"]),
        set(random["weight_development_seeds"]),
        set(random["holdout_seeds"]),
        set(config["excluded_prior_seeds"]),
    ]
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            assert groups[left].isdisjoint(groups[right])


def test_g0c_materializes_scaled_radial_threshold_without_precision_weighting():
    config = _load()
    specs = _reference_specs(config)
    low = next(
        spec
        for spec in specs.values()
        if spec.get("radial_scale") == 0.45
        and spec.get("influence_cap_total") == 0.22
        and spec.get("regularization") == 0.10
    )
    high = dict(low, radial_scale=0.75)
    low_config = _materialize_reference_config(config, low)
    high_config = _materialize_reference_config(config, high)
    low_huber = low_config["references"]["f_sigma_huber"]
    high_huber = high_config["references"]["f_sigma_huber"]
    assert all(
        left < right
        for left, right in zip(
            low_huber["standardized_threshold"],
            high_huber["standardized_threshold"],
            strict=True,
        )
    )
    assert (
        abs(sum(value * value for value in low_huber["influence_cap"]) ** 0.5 - 0.22)
        < 1e-12
    )
    # Reference covariance remains absent from the optimiser geometry: the
    # Gaussian parameters only determine transition radii in the shared
    # equal-client objective.
    assert config["scientific_contract"]["inverse_variance_estimand_forbidden"]


def test_g0c_weight_profiles_change_only_score_rule_after_reference_lock():
    config = _load()
    specs = _reference_specs(config)
    locked = next(spec for spec in specs.values() if spec["method"] == "f_sigma_huber")
    locked_config = _materialize_reference_config(config, locked)
    weight_specs, weight_configs = _weight_specs_and_configs(
        config, locked, locked_config
    )
    selectable = [name for name in weight_specs if name != "uniform_mean"]
    assert len(selectable) == len(config["score"]["weight_profiles"])
    reference_settings = {
        name: candidate["references"]["f_sigma_huber"]
        for name, candidate in weight_configs.items()
        if name != "uniform_mean"
    }
    first = reference_settings[selectable[0]]
    assert all(settings == first for settings in reference_settings.values())
    assert len({weight_configs[name]["score"]["alpha"] for name in selectable}) > 1


def test_g0c_rejects_any_reuse_of_g0b_holdout():
    config = _load()
    config["randomness"]["holdout_seeds"][0] = 503
    with pytest.raises(ValueError, match="already inspected"):
        _validate_g0c(config)


def test_reference_selection_is_deterministic_and_uses_only_supplied_development_rows():
    config = _load()
    identifiers = ["candidate_a", "candidate_b"]

    def row(identifier: str, *, recall: float, covariance_tail: float):
        result = {
            "candidate": identifier,
            "weight_mode": "reference_only",
            "clean_reference_error_ratio": 0.9,
            "attacked_reference_error_ratio_worst_group": 0.8,
            "false_outlier_rate": 0.1,
            "honest_outlier_recall": recall,
            "abs_correlation_null_excess_ci95_high": 0.01,
            "tier_range_null_excess_ci95_high": 0.01,
            "fraction_covariance_limited": 0.5,
            "fraction_covariance_limited_and_tail": covariance_tail,
            "covariance_counterfactual_ratio": 0.02,
            "replace_one_bound": 0.08,
        }
        result.update(
            {
                "gate_clean_reference_error": True,
                "gate_attacked_reference_error": True,
                "gate_false_outlier_rate": True,
                "gate_honest_outlier_recall": recall
                >= config["gates"]["honest_outlier_recall_min"],
                "gate_abs_correlation_null_excess": True,
                "gate_tier_range_null_excess": True,
                "gate_covariance_branch_active": True,
                "gate_covariance_branch_effective_on_huber_influence": covariance_tail
                >= config["gates"][
                    "gaussian_candidate_fraction_covariance_limited_and_tail_min"
                ],
                "gate_covariance_changes_returned_reference": True,
            }
        )
        return result

    selected = _select_reference(
        [
            row("candidate_a", recall=0.3, covariance_tail=0.01),
            row("candidate_b", recall=0.7, covariance_tail=0.10),
        ],
        identifiers,
        config,
    )
    assert selected["candidate"] == "candidate_b"
    assert selected["reference_gate_fail_count"] == 0


def test_g0c_protocol_has_no_cpu_execution_escape_hatch():
    config = _load()
    altered = copy.deepcopy(config)
    altered["execution"]["allow_cpu_fallback"] = True
    with pytest.raises(ValueError, match="MPS-only"):
        _validate_g0c(altered)
