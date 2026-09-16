#!/usr/bin/env python3
"""G0c: two-stage, MPS-only audit of Gaussian-aware FAR references.

G0c is a new experiment, not a re-analysis of the failed G0b holdout.  It
uses fresh identities and separates three roles:

1. reference-development selects the radial threshold, Euclidean influence
   cap and regularisation using reference/score metrics only;
2. weight-development freezes that reference and selects one pre-registered
   FAR weighting profile on new seeds;
3. a final holdout evaluates the two locks without changing either one.

The common equal-client quadratic curvature of ``F_{Sigma,Hub}`` is retained:
public covariance changes only the Huber transition radius, never a client's
central-zone coefficient.  No accuracy is generated or used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_gaussian_aware_reference_g0b as g0b  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402


REFERENCE_GATE_FIELDS = (
    "gate_clean_reference_error",
    "gate_attacked_reference_error",
    "gate_false_outlier_rate",
    "gate_honest_outlier_recall",
    "gate_abs_correlation_null_excess",
    "gate_tier_range_null_excess",
    "gate_covariance_branch_active",
    "gate_covariance_branch_effective_on_huber_influence",
    "gate_covariance_changes_returned_reference",
)

WEIGHT_GATE_FIELDS = (
    "gate_clean_aggregate_error",
    "gate_attacked_aggregate_error",
    "gate_false_outlier_rate",
    "gate_honest_outlier_recall",
    "gate_honest_outlier_weight_mass",
    "gate_byzantine_weight_mass",
    "gate_evasive_aggregate_error",
    "gate_evasive_byzantine_weight_mass",
    "gate_abs_correlation_null_excess",
    "gate_tier_range_null_excess",
)


def _candidate_id(scale: float, cap: float, regularization: float) -> str:
    return (
        f"sigma_huber_rs{round(100 * scale):03d}"
        f"_g{round(100 * cap):03d}_r{round(100 * regularization):03d}"
    )


def _radial_thresholds(
    block_sizes: Sequence[int], *, tail_probability: float, scale: float
) -> list[float]:
    """Scaled Laurent--Massart radii, computed from public dimensions only."""

    x = math.log(1.0 / float(tail_probability))
    return [
        float(scale) * math.sqrt(float(d) + 2.0 * math.sqrt(float(d) * x) + 2.0 * x)
        for d in block_sizes
    ]


def _reference_specs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {str(raw["id"]): dict(raw) for raw in config["references"]["comparators"]}
    grid = config["references"]["reference_grid"]
    for scale in grid["radial_scales"]:
        for cap in grid["influence_cap_totals"]:
            for regularization in grid["regularizations"]:
                identifier = _candidate_id(
                    float(scale), float(cap), float(regularization)
                )
                if identifier in specs:
                    raise ValueError(f"Duplicate candidate ID {identifier!r}")
                specs[identifier] = {
                    "id": identifier,
                    "method": "f_sigma_huber",
                    "radial_scale": float(scale),
                    "radial_base_tail_probability": float(
                        grid["radial_base_tail_probability"]
                    ),
                    "influence_cap_total": float(cap),
                    "regularization": float(regularization),
                    "num_steps": int(grid["num_steps"]),
                }
    return specs


def _materialize_reference_config(
    base: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    method = str(spec["method"])
    config["references"]["candidates"] = [method]
    if method == "f_sigma_huber":
        blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
        total_cap = float(spec["influence_cap_total"])
        config["references"]["f_sigma_huber"].update(
            {
                "standardized_threshold": _radial_thresholds(
                    blocks,
                    tail_probability=float(spec["radial_base_tail_probability"]),
                    scale=float(spec["radial_scale"]),
                ),
                "influence_cap": [total_cap / math.sqrt(len(blocks))] * len(blocks),
                "regularization": float(spec["regularization"]),
                "num_steps": int(spec["num_steps"]),
            }
        )
    oracle._validate_config(config)
    return config


def _stage_config(
    base: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["references"]["candidates"] = [dict(spec) for spec in specs.values()]
    # The shared G0b engine only reads this list while parsing candidate IDs.
    config["selection"]["candidate_references"] = [
        identifier
        for identifier, spec in specs.items()
        if spec["method"] == "f_sigma_huber"
    ]
    config["selection"]["accuracy_used"] = False
    return config


def _validate_g0c(config: dict[str, Any]) -> None:
    contract = config["scientific_contract"]
    if contract["estimand"] != "equal_client_mean_of_clean_honest_updates":
        raise ValueError("G0c must retain the equal-client estimand")
    if not bool(contract["inverse_variance_estimand_forbidden"]):
        raise ValueError("Inverse-variance redefinition of the estimand is forbidden")
    if bool(contract["accuracy_used_for_selection"]):
        raise ValueError("Accuracy cannot be used for G0c selection")
    if config["execution"]["required_device"] != "mps" or bool(
        config["execution"]["allow_cpu_fallback"]
    ):
        raise ValueError("G0c is MPS-only and forbids CPU fallback")

    random = config["randomness"]
    calibration = [int(random["null_calibration_seed"])]
    reference_dev = [int(value) for value in random["reference_development_seeds"]]
    weight_dev = [int(value) for value in random["weight_development_seeds"]]
    holdout = [int(value) for value in random["holdout_seeds"]]
    fresh = calibration + reference_dev + weight_dev + holdout
    if len(fresh) != len(set(fresh)):
        raise ValueError(
            "Calibration, both development splits and holdout must be disjoint"
        )
    if set(fresh) & set(int(value) for value in config["excluded_prior_seeds"]):
        raise ValueError("G0c reuses an identity already inspected in G0/G0b")
    if min(len(reference_dev), len(weight_dev)) < 3 or len(holdout) < 5:
        raise ValueError("Need >=3 seeds per development split and >=5 holdout seeds")
    if bool(config["selection"]["holdout_used_for_selection"]):
        raise ValueError("Holdout use in selection is forbidden")

    grid = config["references"]["reference_grid"]
    if not all(float(value) > 0.0 for value in grid["radial_scales"]):
        raise ValueError("radial_scales must be positive")
    if not all(float(value) > 0.0 for value in grid["influence_cap_totals"]):
        raise ValueError("influence caps must be positive")
    if not all(float(value) > 0.0 for value in grid["regularizations"]):
        raise ValueError("regularizations must be positive")
    if int(grid["num_steps"]) < 1:
        raise ValueError("The public solver must execute at least one fixed step")
    profiles = config["score"]["weight_profiles"]
    if len({str(profile["id"]) for profile in profiles}) != len(profiles):
        raise ValueError("Weight profile IDs must be unique")
    for profile in profiles:
        if profile["mode"] not in {
            "reference_only",
            "novelty_only",
            "novelty_confidence",
        }:
            raise ValueError(f"Unknown weight mode {profile['mode']!r}")
        if not (
            float(profile["novelty_start_z"])
            < float(profile["novelty_full_z"])
            <= float(profile["rejection_start_z"])
            < float(profile["rejection_full_z"])
        ):
            raise ValueError(f"Invalid thresholds in weight profile {profile['id']}")

    # Validate the base tensor/statistical schema through the shared oracle.
    first = next(
        spec
        for spec in _reference_specs(config).values()
        if spec["method"] == "f_sigma_huber"
    )
    _materialize_reference_config(config, first)


def _violation_upper(value: float, limit: float) -> float:
    return max(0.0, value / limit - 1.0)


def _violation_lower(value: float, limit: float) -> float:
    return max(0.0, 1.0 - value / limit)


def _reference_checks(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {name: bool(summary[name]) for name in REFERENCE_GATE_FIELDS}
    checks["gate_replace_one_bound"] = float(summary["replace_one_bound"]) <= float(
        config["gates"]["gaussian_candidate_replace_one_bound_max"]
    )
    return checks


def _reference_penalty(summary: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    gates = config["gates"]
    penalty = 0.0
    penalty += _violation_upper(
        float(summary["clean_reference_error_ratio"]),
        float(gates["clean_reference_error_ratio_to_uniform_max"]),
    )
    penalty += _violation_upper(
        float(summary["attacked_reference_error_ratio_worst_group"]),
        float(gates["separated_att_reference_error_ratio_to_uniform_max"]),
    )
    false_rate = float(summary["false_outlier_rate"])
    if false_rate < float(gates["false_outlier_rate_min"]):
        penalty += _violation_lower(false_rate, float(gates["false_outlier_rate_min"]))
    else:
        penalty += _violation_upper(false_rate, float(gates["false_outlier_rate_max"]))
    penalty += _violation_lower(
        float(summary["honest_outlier_recall"]),
        float(gates["honest_outlier_recall_min"]),
    )
    penalty += _violation_upper(
        max(0.0, float(summary["abs_correlation_null_excess_ci95_high"])),
        float(gates["abs_correlation_null_excess_ci95_upper_max"]),
    )
    penalty += _violation_upper(
        max(0.0, float(summary["tier_range_null_excess_ci95_high"])),
        float(gates["tier_range_null_excess_ci95_upper_max"]),
    )
    penalty += _violation_lower(
        float(summary["fraction_covariance_limited"]),
        float(gates["gaussian_candidate_fraction_covariance_limited_min"]),
    )
    penalty += _violation_lower(
        float(summary["fraction_covariance_limited_and_tail"]),
        float(gates["gaussian_candidate_fraction_covariance_limited_and_tail_min"]),
    )
    penalty += _violation_lower(
        float(summary["covariance_counterfactual_ratio"]),
        float(gates["gaussian_candidate_covariance_counterfactual_ratio_min"]),
    )
    penalty += _violation_upper(
        float(summary["replace_one_bound"]),
        float(gates["gaussian_candidate_replace_one_bound_max"]),
    )
    return penalty


def _select_reference(
    summaries: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    order = {identifier: index for index, identifier in enumerate(candidate_ids)}
    candidates = [
        dict(row)
        for row in summaries
        if row["candidate"] in order and row["weight_mode"] == "reference_only"
    ]
    if len(candidates) != len(candidate_ids):
        raise RuntimeError("Reference-development did not evaluate every grid cell")
    for row in candidates:
        checks = _reference_checks(row, config)
        row["reference_gate_checks"] = checks
        row["reference_gate_fail_count"] = sum(not value for value in checks.values())
        row["reference_gate_penalty"] = _reference_penalty(row, config)
    candidates.sort(
        key=lambda row: (
            int(row["reference_gate_fail_count"]),
            float(row["reference_gate_penalty"]),
            order[str(row["candidate"])],
        )
    )
    return candidates[0]


def _weight_specs_and_configs(
    config: Mapping[str, Any],
    locked_spec: Mapping[str, Any],
    locked_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    specs: dict[str, dict[str, Any]] = {
        "uniform_mean": {"id": "uniform_mean", "method": "uniform_mean"}
    }
    configs = {
        "uniform_mean": _materialize_reference_config(
            config, {"id": "uniform_mean", "method": "uniform_mean"}
        )
    }
    for profile in config["score"]["weight_profiles"]:
        identifier = f"{locked_spec['id']}__w_{profile['id']}"
        spec = dict(locked_spec)
        spec.update(
            {
                "id": identifier,
                "weight_profile": str(profile["id"]),
                "weight_mode": str(profile["mode"]),
            }
        )
        candidate = copy.deepcopy(dict(locked_config))
        for field in (
            "alpha",
            "novelty_start_z",
            "novelty_full_z",
            "rejection_start_z",
            "rejection_full_z",
            "trust_floor",
        ):
            candidate["score"][field] = float(profile[field])
        specs[identifier] = spec
        configs[identifier] = candidate
    return specs, configs


def _weight_checks(summary: Mapping[str, Any]) -> dict[str, bool]:
    return {name: bool(summary[name]) for name in WEIGHT_GATE_FIELDS}


def _weight_penalty(summary: Mapping[str, Any]) -> float:
    # The shared summary already computes a dimensionless sum of normalized
    # gate violations.  Reference-only gate terms are constant after locking
    # F, so they do not leak any holdout information into this ordering.
    return float(summary["normalized_gate_penalty"])


def _select_weight(
    summaries: Sequence[Mapping[str, Any]],
    selectable_ids: Sequence[str],
) -> dict[str, Any]:
    order = {identifier: index for index, identifier in enumerate(selectable_ids)}
    candidates = [dict(row) for row in summaries if row["candidate"] in order]
    if len(candidates) != len(selectable_ids):
        raise RuntimeError("Weight-development did not evaluate every profile")
    for row in candidates:
        checks = _weight_checks(row)
        row["weight_gate_checks"] = checks
        row["weight_gate_fail_count"] = sum(not value for value in checks.values())
        row["weight_gate_penalty"] = _weight_penalty(row)
    candidates.sort(
        key=lambda row: (
            int(row["weight_gate_fail_count"]),
            float(row["weight_gate_penalty"]),
            order[str(row["candidate"])],
        )
    )
    return candidates[0]


def _tag_runtime(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["resolved_device"] = "mps"
        row["tensor_dtype"] = "float32"


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    reference_lock: Mapping[str, Any],
    weight_lock: Mapping[str, Any],
    holdout_locked: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    lines = [
        "# G0c — audit Gaussian-aware MPS à deux verrous",
        "",
        "## Verdict",
        "",
        (
            "**Promotion synthétique autorisée.** Les gates de référence et de "
            "pondération passent sur leurs développements respectifs et sur le "
            "holdout final."
            if decision["promote"]
            else "**Aucune promotion.** Le meilleur réglage préenregistré ne "
            "satisfait pas toute la chaîne développement puis holdout."
        ),
        "",
        "G0c n'a lu aucun résultat G0b. Les anciennes seeds, y compris le "
        "holdout G0b, sont interdites. Le premier verrou porte uniquement sur "
        "la référence; le second ajuste les poids sur de nouvelles seeds; le "
        "holdout ne participe à aucun choix.",
        "",
        "## Verrous",
        "",
        f"- référence : `{reference_lock['candidate']}`; "
        f"échecs développement = `{reference_lock['reference_gate_fail_count']}`;",
        f"- poids : `{weight_lock['candidate']}` / `{weight_lock['weight_mode']}`; "
        f"échecs développement = `{weight_lock['weight_gate_fail_count']}`;",
        f"- échecs sur holdout, référence = "
        f"`{decision['holdout_reference_gate_fail_count']}`, poids = "
        f"`{decision['holdout_weight_gate_fail_count']}`.",
        "",
        "## Métriques du candidat verrouillé sur holdout",
        "",
        "| Réf. propre / uniforme | Réf. attaquée / uniforme | Agrégat propre / "
        "uniforme | Agrégat attaqué / uniforme | Rappel outliers | Gain masse "
        "outliers | Masse byzantine | Covariance active | Covariance & tail | "
        "Contre-factuel cov./cap | IC95 corr. haut | IC95 tiers haut |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {cr:.3f} | {ar:.3f} | {ca:.3f} | {aa:.3f} | {rec:.3f} | "
        "{gain:.3f} | {byz:.3f} | {cov:.3f} | {covtail:.3f} | {cf:.3f} | "
        "{corr:.3f} | {tier:.3f} |".format(
            cr=float(holdout_locked["clean_reference_error_ratio"]),
            ar=float(holdout_locked["attacked_reference_error_ratio_worst_group"]),
            ca=float(holdout_locked["clean_aggregate_error_ratio"]),
            aa=float(holdout_locked["attacked_aggregate_error_ratio_worst_group"]),
            rec=float(holdout_locked["honest_outlier_recall"]),
            gain=float(holdout_locked["honest_outlier_weight_mass_gain"]),
            byz=float(holdout_locked["byzantine_weight_mass_worst_group"]),
            cov=float(holdout_locked["fraction_covariance_limited"]),
            covtail=float(holdout_locked["fraction_covariance_limited_and_tail"]),
            cf=float(holdout_locked["covariance_counterfactual_ratio"]),
            corr=float(holdout_locked["abs_correlation_null_excess_ci95_high"]),
            tier=float(holdout_locked["tier_range_null_excess_ci95_high"]),
        ),
        "",
        "## Interprétation correcte",
        "",
        "`fraction_covariance_limited_and_tail` exige que la covariance ne soit "
        "pas seulement présente dans une formule : elle doit effectivement "
        "déterminer un rayon pour un résidu situé dans la branche d'influence "
        "bornée. Le contre-factuel remplace ces rayons par le cap fixe et exige "
        "que la référence retournée change de façon mesurable.",
        "",
        "Le score conserve l'estimand equal-client : dans la zone quadratique, "
        "chaque client a la même courbure. La covariance publique règle le "
        "point de transition, elle ne réalise jamais une moyenne par précision.",
        "",
        "## Reproductibilité",
        "",
        f"- seeds référence : `{config['randomness']['reference_development_seeds']}`;",
        f"- seeds poids : `{config['randomness']['weight_development_seeds']}`;",
        f"- seeds holdout : `{config['randomness']['holdout_seeds']}`;",
        f"- sorties : `{output_dir.resolve()}`;",
        "- exécution : `mps`, tenseurs `float32`, sans fallback CPU.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_g0c(config)
    runtime_device, runtime_dtype = oracle._configure_runtime("mps")
    if runtime_device.type != "mps" or runtime_dtype != torch.float32:
        raise RuntimeError("G0c requires real MPS float32 execution")

    config["execution"].update(
        {
            "requested_device": "mps",
            "resolved_device": str(runtime_device),
            "tensor_dtype": "float32",
            "silent_cpu_fallback_observed": False,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    reference_specs = _reference_specs(config)
    reference_configs = {
        identifier: _materialize_reference_config(config, spec)
        for identifier, spec in reference_specs.items()
    }
    reference_stage = _stage_config(config, reference_specs)
    calibrations, calibration_rows = g0b._calibrations(
        reference_stage, reference_configs
    )
    _tag_runtime(calibration_rows)
    g0b._write_csv(output_dir / "null_calibration.csv", calibration_rows)

    reference_rows = g0b._phase_rows(
        phase="reference_development",
        config=reference_stage,
        specs=reference_specs,
        candidate_configs=reference_configs,
        calibrations=calibrations,
        candidates=list(reference_specs),
        modes_by_candidate={name: ["reference_only"] for name in reference_specs},
        seeds=[
            int(value) for value in config["randomness"]["reference_development_seeds"]
        ],
        draws_per_seed=int(
            config["randomness"]["reference_development_draws_per_seed"]
        ),
        severities=[
            float(value)
            for value in config["threats"]["reference_development_severities"]
        ],
    )
    _tag_runtime(reference_rows)
    reference_summaries = g0b._summaries(reference_rows, reference_stage)
    grid_ids = [
        identifier
        for identifier, spec in reference_specs.items()
        if spec["method"] == "f_sigma_huber"
    ]
    reference_lock = _select_reference(reference_summaries, grid_ids, config)
    reference_lock_payload = {
        "candidate": reference_lock["candidate"],
        "selected_on_phase": "reference_development_only",
        "holdout_used_for_selection": False,
        "reference_gate_fail_count": reference_lock["reference_gate_fail_count"],
        "reference_gate_penalty": reference_lock["reference_gate_penalty"],
        "reference_gate_checks": reference_lock["reference_gate_checks"],
    }
    g0b._write_csv(output_dir / "reference_development_detail.csv", reference_rows)
    g0b._write_csv(
        output_dir / "reference_development_summary.csv", reference_summaries
    )
    (output_dir / "reference_development_lock.json").write_text(
        json.dumps(reference_lock_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    locked_reference_id = str(reference_lock["candidate"])
    weight_specs, weight_configs = _weight_specs_and_configs(
        config,
        reference_specs[locked_reference_id],
        reference_configs[locked_reference_id],
    )
    weight_stage = _stage_config(config, weight_specs)
    weight_calibrations: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for candidate, spec in weight_specs.items():
        source = (
            "uniform_mean" if spec["method"] == "uniform_mean" else locked_reference_id
        )
        for regime in config["privacy_noise"]["regimes"]:
            regime_name = str(regime["name"])
            for permutation in regime["permutations"]:
                key = (source, regime_name, str(permutation))
                weight_calibrations[(candidate, regime_name, str(permutation))] = (
                    calibrations[key]
                )

    weight_modes = {
        candidate: [str(spec.get("weight_mode", "reference_only"))]
        for candidate, spec in weight_specs.items()
    }
    weight_rows = g0b._phase_rows(
        phase="weight_development",
        config=weight_stage,
        specs=weight_specs,
        candidate_configs=weight_configs,
        calibrations=weight_calibrations,
        candidates=list(weight_specs),
        modes_by_candidate=weight_modes,
        seeds=[
            int(value) for value in config["randomness"]["weight_development_seeds"]
        ],
        draws_per_seed=int(config["randomness"]["weight_development_draws_per_seed"]),
        severities=[
            float(value) for value in config["threats"]["weight_development_severities"]
        ],
    )
    _tag_runtime(weight_rows)
    weight_summaries = g0b._summaries(weight_rows, weight_stage)
    selectable_weights = [name for name in weight_specs if name != "uniform_mean"]
    weight_lock = _select_weight(weight_summaries, selectable_weights)
    weight_lock_payload = {
        "candidate": weight_lock["candidate"],
        "weight_mode": weight_lock["weight_mode"],
        "weight_profile": weight_specs[str(weight_lock["candidate"])]["weight_profile"],
        "selected_on_phase": "weight_development_only_after_reference_lock",
        "holdout_used_for_selection": False,
        "weight_gate_fail_count": weight_lock["weight_gate_fail_count"],
        "weight_gate_penalty": weight_lock["weight_gate_penalty"],
        "weight_gate_checks": weight_lock["weight_gate_checks"],
    }
    g0b._write_csv(output_dir / "weight_development_detail.csv", weight_rows)
    g0b._write_csv(output_dir / "weight_development_summary.csv", weight_summaries)
    (output_dir / "weight_development_lock.json").write_text(
        json.dumps(weight_lock_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    locked_weight_id = str(weight_lock["candidate"])
    holdout_specs = {
        name: reference_specs[name] for name in ("uniform_mean", "fcc", "fna_cc", "rfa")
    }
    holdout_specs[locked_weight_id] = weight_specs[locked_weight_id]
    holdout_configs = {
        name: reference_configs[name]
        for name in ("uniform_mean", "fcc", "fna_cc", "rfa")
    }
    holdout_configs[locked_weight_id] = weight_configs[locked_weight_id]
    holdout_calibrations: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for candidate, spec in holdout_specs.items():
        source = locked_reference_id if candidate == locked_weight_id else candidate
        for regime in config["privacy_noise"]["regimes"]:
            regime_name = str(regime["name"])
            for permutation in regime["permutations"]:
                key = (source, regime_name, str(permutation))
                holdout_calibrations[(candidate, regime_name, str(permutation))] = (
                    calibrations[key]
                )
    holdout_stage = _stage_config(config, holdout_specs)
    holdout_rows = g0b._phase_rows(
        phase="holdout",
        config=holdout_stage,
        specs=holdout_specs,
        candidate_configs=holdout_configs,
        calibrations=holdout_calibrations,
        candidates=list(holdout_specs),
        modes_by_candidate={
            candidate: [
                (
                    str(spec.get("weight_mode", "reference_only"))
                    if candidate == locked_weight_id
                    else "reference_only"
                )
            ]
            for candidate, spec in holdout_specs.items()
        },
        seeds=[int(value) for value in config["randomness"]["holdout_seeds"]],
        draws_per_seed=int(config["randomness"]["holdout_draws_per_seed"]),
        severities=[float(value) for value in config["threats"]["holdout_severities"]],
    )
    _tag_runtime(holdout_rows)
    holdout_summaries = g0b._summaries(holdout_rows, holdout_stage)
    holdout_locked = next(
        row for row in holdout_summaries if row["candidate"] == locked_weight_id
    )
    holdout_reference_checks = _reference_checks(holdout_locked, config)
    holdout_weight_checks = _weight_checks(holdout_locked)
    development_reference_pass = reference_lock["reference_gate_fail_count"] == 0
    development_weight_pass = weight_lock["weight_gate_fail_count"] == 0
    holdout_reference_pass = all(holdout_reference_checks.values())
    holdout_weight_pass = all(holdout_weight_checks.values())
    decision = {
        "campaign_id": config["campaign_id"],
        "requested_device": "mps",
        "resolved_device": "mps",
        "tensor_dtype": "float32",
        "silent_cpu_fallback_allowed": False,
        "accuracy_used_for_selection": False,
        "g0b_outputs_read": False,
        "holdout_used_for_selection": False,
        "locked_reference": locked_reference_id,
        "locked_weight_candidate": locked_weight_id,
        "locked_weight_mode": weight_lock["weight_mode"],
        "development_reference_passes": development_reference_pass,
        "development_weight_passes": development_weight_pass,
        "holdout_reference_passes": holdout_reference_pass,
        "holdout_weight_passes": holdout_weight_pass,
        "holdout_reference_gate_fail_count": sum(
            not value for value in holdout_reference_checks.values()
        ),
        "holdout_weight_gate_fail_count": sum(
            not value for value in holdout_weight_checks.values()
        ),
        "promote": bool(
            development_reference_pass
            and development_weight_pass
            and holdout_reference_pass
            and holdout_weight_pass
        ),
        "promotion_rule": "both_development_locks_and_both_holdout_gate_sets_pass",
        "reference_grid_size": len(grid_ids),
        "weight_grid_size": len(selectable_weights),
        "reference_development_seeds": config["randomness"][
            "reference_development_seeds"
        ],
        "weight_development_seeds": config["randomness"]["weight_development_seeds"],
        "holdout_seeds": config["randomness"]["holdout_seeds"],
    }
    g0b._write_csv(output_dir / "holdout_detail.csv", holdout_rows)
    g0b._write_csv(output_dir / "holdout_summary.csv", holdout_summaries)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        config=config,
        output_dir=output_dir,
        reference_lock=reference_lock,
        weight_lock=weight_lock,
        holdout_locked=holdout_locked,
        decision=decision,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0c.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_g0c_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_Robust_Reference_G0c_MPS.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
