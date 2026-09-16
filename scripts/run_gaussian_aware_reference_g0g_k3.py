#!/usr/bin/env python3
"""Run the preregistered G0g-K3 dual-gate/global-cap screen on MPS."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_fixed_anchor_dual_gated_reference,
    gaussian_aware_fixed_anchor_scalar_gated_reference,
)
from robustness.aggregators import centered_clipping, clip_l2  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k1 as base  # noqa: E402
from scripts import run_gaussian_aware_reference_g0g_k2 as k2  # noqa: E402
from scripts import run_gaussian_aware_reference_oracle as oracle  # noqa: E402

PRIMARY = "g0g_k3_dual_gate"
AWARE = "g0g_k2"
BLIND = "g0g_k2_sigma_blind"
NO_GATE = "fixed_global_cap_no_gate"
CANDIDATES = ("fcc", NO_GATE, BLIND, AWARE, PRIMARY)
TIERS = (1.0, 1.5, 2.0)
FROZEN_CALIBRATION_SHA256 = (
    "98a30cad8a2f92e6843346e9a9512bd321e8302b9b6e17b447b2641fffb544a5"
)


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "campaign_id",
        "scope",
        "scientific_contract",
        "frozen_calibration",
        "excluded_prior_seeds",
        "cohort",
        "privacy_noise",
        "references",
        "aggregation",
        "threats",
        "randomness",
        "candidates",
        "gates",
        "execution",
    }
    if set(config) != expected:
        raise ValueError("The frozen G0g-K3 top-level schema changed")
    if config["campaign_id"] != "gaussian_aware_reference_g0g_k3_mps_v1":
        raise ValueError("Unexpected G0g-K3 campaign id")
    required_contract = {
        "estimand": "equal_client_mean_of_clean_honest_updates",
        "reference_only": True,
        "far_weights_used": False,
        "accuracy_used": False,
        "base_anchor": "fixed_public_or_prior_transcript",
        "covariance_role": "statistical_tolerance_only",
        "common_gate_role": "identity_blind_acceptance_restriction",
        "dual_gate_rule": "elementwise_minimum_aware_and_common",
        "influence_cap_depends_on_covariance": False,
        "inverse_variance_weighting": False,
        "gate_sum_normalization": False,
        "global_l2_cap": True,
        "no_gate_is_exactly_fcc": True,
        "homogeneous_dual_is_exactly_aware": True,
        "covariance_provenance": "public_authenticated",
        "client_declared_covariance_forbidden": True,
        "calibration_is_frozen_from_k2": True,
        "recalibration_for_k3_forbidden": True,
        "development_only_screen": True,
    }
    if config["scientific_contract"] != required_contract:
        raise ValueError("The frozen G0g-K3 scientific contract changed")
    frozen = config["frozen_calibration"]
    if frozen != {
        "path": (
            "results/ldp_gradient_far/"
            "gaussian_aware_reference_g0g_k2_mps_v1/calibration.json"
        ),
        "sha256": FROZEN_CALIBRATION_SHA256,
        "source_campaign_id": "gaussian_aware_reference_g0g_k2_mps_v1",
        "reuse_thresholds_exactly": True,
        "recalibrate": False,
    }:
        raise ValueError("The frozen K2 calibration provenance changed")
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    b = int(cohort["num_byzantine"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    if n < 3 or not 0 <= b < n / 2:
        raise ValueError("G0g-K3 needs n>=3 and 0<=b<n/2")
    if sum(blocks) != int(cohort["dimension"]) or any(width <= 0 for width in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity standard deviation is required per block")
    if tuple(str(value) for value in config["candidates"]["names"]) != CANDIDATES:
        raise ValueError("The frozen G0g-K3 candidate set changed")
    if config["candidates"]["primary"] != PRIMARY:
        raise ValueError("Unexpected G0g-K3 primary candidate")
    if config["execution"] != {
        "required_device": "mps",
        "tensor_dtype": "float32",
        "allow_cpu_fallback": False,
    }:
        raise ValueError("Production G0g-K3 must use MPS float32 without fallback")
    randomness = config["randomness"]
    if set(randomness) != {
        "development_seeds",
        "holdout_seeds",
        "holdout_rule",
        "replace_one_trials_per_seed_cell",
        "pair_standard_noise_across_regimes",
    }:
        raise ValueError("The frozen G0g-K3 randomness schema changed")
    if randomness["holdout_rule"] != "open_only_if_all_development_gates_pass":
        raise ValueError("The frozen G0g-K3 holdout rule changed")
    development = {int(value) for value in randomness["development_seeds"]}
    holdout = {int(value) for value in randomness["holdout_seeds"]}
    excluded = {int(value) for value in config["excluded_prior_seeds"]}
    if development & excluded or holdout & excluded or development & holdout:
        raise ValueError("G0g-K3 development, holdout and prior seeds must be disjoint")
    if len(development) != len(randomness["development_seeds"]):
        raise ValueError("G0g-K3 development seeds must be unique")
    if len(holdout) != 7 or len(holdout) != len(randomness["holdout_seeds"]):
        raise ValueError("G0g-K3 must reserve exactly seven unique holdout seeds")
    if not math.isclose(
        float(config["references"]["total_client_influence_cap"]),
        float(config["references"]["fcc_radius"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("No-gate/FCC identity requires influence cap == FCC radius")


def _load_frozen_calibration(
    config: Mapping[str, Any], *, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = config["frozen_calibration"]
    path = (root / str(frozen["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "Frozen calibration must remain inside the repository"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Frozen K2 calibration not found: {path}")
    observed_hash = base._sha256(path)
    expected_hash = str(frozen["sha256"])
    if observed_hash != expected_hash or expected_hash != FROZEN_CALIBRATION_SHA256:
        raise RuntimeError(
            "Frozen K2 calibration hash mismatch: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    expected_thresholds = {
        "aware": {"homogeneous", "heteroscedastic"},
        "blind": {"homogeneous", "heteroscedastic"},
    }
    thresholds = calibration.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Frozen K2 calibration has no threshold registry")
    for mode, regimes in expected_thresholds.items():
        if set(thresholds.get(mode, {})) != regimes:
            raise ValueError(f"Frozen K2 {mode} threshold registry changed")
        if any(
            not math.isfinite(float(thresholds[mode][regime]))
            or float(thresholds[mode][regime]) <= 0.0
            for regime in regimes
        ):
            raise ValueError("Frozen K2 thresholds must be finite and positive")
    source_seeds = {int(value) for value in calibration.get("calibration_seeds", [])}
    current_seeds = {int(value) for value in config["randomness"]["development_seeds"]}
    if source_seeds & current_seeds:
        raise ValueError("Frozen calibration and K3 development seeds overlap")
    provenance = {
        "source_path": str(path.relative_to(root.resolve())),
        "source_campaign_id": str(frozen["source_campaign_id"]),
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "sha256_verified": True,
        "recalibrated_for_k3": False,
        "thresholds_reused_exactly": thresholds,
        "source_protocol": calibration.get("protocol"),
        "source_calibration_seeds": sorted(source_seeds),
    }
    return calibration, provenance


def _radii(
    config: Mapping[str, Any],
    variances: torch.Tensor,
    calibration: Mapping[str, Any],
    *,
    regime_name: str,
    blind: bool,
) -> torch.Tensor:
    return k2._radii(
        config,
        variances,
        calibration,
        regime_name=regime_name,
        blind=blind,
    )


def _reference(
    method: str,
    vectors: torch.Tensor,
    *,
    anchor: torch.Tensor,
    aware_radii: torch.Tensor,
    blind_radii: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "fcc":
        return (
            centered_clipping(
                vectors,
                anchor=anchor,
                tau=float(config["references"]["fcc_radius"]),
            ),
            {},
        )
    cap = float(config["references"]["total_client_influence_cap"])
    width = float(config["references"]["gate_transition_width"])
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    if method == NO_GATE:
        return gaussian_aware_fixed_anchor_scalar_gated_reference(
            vectors,
            anchor=anchor,
            statistical_radii=torch.full_like(aware_radii, 1.0e12),
            block_sizes=blocks,
            influence_cap=cap,
            gate_transition_width=width,
            return_diagnostics=True,
        )
    if method == BLIND:
        return gaussian_aware_fixed_anchor_scalar_gated_reference(
            vectors,
            anchor=anchor,
            statistical_radii=blind_radii,
            block_sizes=blocks,
            influence_cap=cap,
            gate_transition_width=width,
            return_diagnostics=True,
        )
    if method == AWARE:
        return gaussian_aware_fixed_anchor_scalar_gated_reference(
            vectors,
            anchor=anchor,
            statistical_radii=aware_radii,
            block_sizes=blocks,
            influence_cap=cap,
            gate_transition_width=width,
            return_diagnostics=True,
        )
    if method == PRIMARY:
        return gaussian_aware_fixed_anchor_dual_gated_reference(
            vectors,
            anchor=anchor,
            statistical_radii=aware_radii,
            common_statistical_radii=blind_radii,
            block_sizes=blocks,
            influence_cap=cap,
            gate_transition_width=width,
            return_diagnostics=True,
        )
    raise ValueError(f"Unknown G0g-K3 candidate: {method}")


def _tier_fields(
    *,
    normalized: torch.Tensor,
    gates: torch.Tensor,
    tiers: torch.Tensor,
    regular: torch.Tensor,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for tier in TIERS:
        selected = regular & torch.isclose(
            tiers, torch.tensor(tier, dtype=tiers.dtype, device=tiers.device)
        )
        count = int(selected.sum().item())
        key = str(tier).replace(".", "p")
        result[f"regular_count_tier_{key}"] = count
        result[f"tail_count_tier_{key}"] = (
            int((normalized[selected] > 1.0).sum().item()) if count else 0
        )
        result[f"gate_sum_tier_{key}"] = (
            float(gates[selected].sum().item()) if count else 0.0
        )
    return result


def _evaluate_cell(
    config: dict[str, Any],
    calibration: Mapping[str, Any],
    *,
    seed: int,
    regime: dict[str, Any],
    permutation: str,
    geometry: str,
    threat: str,
) -> list[dict[str, Any]]:
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    clean, outliers, centre, anchor = oracle._honest_clean_vectors(
        config,
        seed=seed,
        draw=0,
        geometry=geometry,
        include_outliers=True,
    )
    variances, tiers = oracle._noise_variances(config, regime, permutation)
    observed = base._paired_private_noise(
        clean, variances, blocks, seed=seed, draw=0, geometry=geometry
    )
    attacked, byzantine = oracle._replace_with_attack(
        observed,
        config,
        threat=threat,
        severity=float(config["threats"]["severity"]),
        seed=oracle._seed(
            "g0g-k3-attack", seed, regime["name"], permutation, geometry, threat
        ),
    )
    server_clip = float(config["aggregation"]["server_clip_norm"])
    vectors = clip_l2(attacked, server_clip)
    aware_radii = _radii(
        config,
        variances,
        calibration,
        regime_name=str(regime["name"]),
        blind=False,
    )
    blind_radii = _radii(
        config,
        variances,
        calibration,
        regime_name=str(regime["name"]),
        blind=True,
    )
    honest = ~byzantine
    regular = honest & ~outliers
    honest_outliers = honest & outliers
    target = clean[honest].mean(dim=0)
    pair_id = f"{seed}|{regime['name']}|{permutation}|{geometry}|{threat}"
    outputs: dict[str, torch.Tensor] = {}
    diagnostics_by_method: dict[str, dict[str, Any]] = {}
    for method in CANDIDATES:
        outputs[method], diagnostics_by_method[method] = _reference(
            method,
            vectors,
            anchor=anchor,
            aware_radii=aware_radii,
            blind_radii=blind_radii,
            config=config,
        )

    result: list[dict[str, Any]] = []
    empty_tiers = {
        f"{prefix}_tier_{str(tier).replace('.', 'p')}": 0
        for tier in TIERS
        for prefix in ("regular_count", "tail_count", "gate_sum")
    }
    for method in CANDIDATES:
        reference = outputs[method]
        diagnostics = diagnostics_by_method[method]
        if method == PRIMARY:
            aware_norm = torch.tensor(
                diagnostics["aware_normalized_residuals_by_client"],
                dtype=vectors.dtype,
                device=vectors.device,
            )
            common_norm = torch.tensor(
                diagnostics["common_normalized_residuals_by_client"],
                dtype=vectors.dtype,
                device=vectors.device,
            )
            normalized = torch.maximum(aware_norm, common_norm)
            gates = torch.tensor(
                diagnostics["gates_by_client"],
                dtype=vectors.dtype,
                device=vectors.device,
            )
        elif method in {NO_GATE, BLIND, AWARE}:
            normalized = torch.tensor(
                diagnostics["normalized_residuals_by_client"],
                dtype=vectors.dtype,
                device=vectors.device,
            )
            gates = torch.tensor(
                diagnostics["gates_by_client"],
                dtype=vectors.dtype,
                device=vectors.device,
            )
        else:
            normalized = None
            gates = None
        tier_fields = dict(empty_tiers)
        tail_rate = float("nan")
        tier_gap = float("nan")
        gate_mean = float("nan")
        honest_outlier_gate_mean = float("nan")
        if normalized is not None and gates is not None:
            tail_rate = float((normalized[regular] > 1.0).float().mean().item())
            gate_mean = float(gates.mean().item())
            if bool(honest_outliers.any()):
                honest_outlier_gate_mean = float(gates[honest_outliers].mean().item())
            tier_fields = _tier_fields(
                normalized=normalized,
                gates=gates,
                tiers=tiers,
                regular=regular,
            )
            rates = []
            for tier in TIERS:
                key = str(tier).replace(".", "p")
                count = int(tier_fields[f"regular_count_tier_{key}"])
                if count:
                    rates.append(float(tier_fields[f"tail_count_tier_{key}"]) / count)
            tier_gap = max(rates) - min(rates) if len(rates) > 1 else 0.0
        contribution_norms = diagnostics.get("client_contribution_norms")
        byzantine_share = float("nan")
        if contribution_norms is not None and bool(byzantine.any()):
            norms = torch.tensor(
                contribution_norms, dtype=vectors.dtype, device=vectors.device
            )
            total = float(norms.sum().item())
            byzantine_share = (
                float(norms[byzantine].sum().item()) / total if total else 0.0
            )
        result.append(
            {
                "pairing_id": pair_id,
                "seed": seed,
                "noise_regime": str(regime["name"]),
                "noise_permutation": permutation,
                "outlier_geometry": geometry,
                "threat": threat,
                "candidate": method,
                "reference_error": float(
                    torch.linalg.vector_norm(reference - target).item()
                ),
                "error_to_population_centre": float(
                    torch.linalg.vector_norm(reference - centre).item()
                ),
                "exact_difference_to_fcc_reference": float(
                    torch.linalg.vector_norm(reference - outputs["fcc"]).item()
                ),
                "exact_difference_to_aware_reference": float(
                    torch.linalg.vector_norm(reference - outputs[AWARE]).item()
                ),
                "regular_combined_false_tail_rate": tail_rate,
                "regular_false_tail_tier_gap_cell": tier_gap,
                "gate_mean": gate_mean,
                "honest_outlier_gate_mean": honest_outlier_gate_mean,
                "aware_gate_mean": float(
                    diagnostics.get("aware_gate_mean", float("nan"))
                ),
                "common_gate_mean": float(
                    diagnostics.get("common_gate_mean", float("nan"))
                ),
                "aware_gate_limiting_fraction": float(
                    diagnostics.get("aware_gate_limiting_fraction", float("nan"))
                ),
                "common_gate_limiting_fraction": float(
                    diagnostics.get("common_gate_limiting_fraction", float("nan"))
                ),
                "hard_reject_fraction": float(
                    diagnostics.get(
                        "final_hard_reject_fraction",
                        diagnostics.get("hard_reject_fraction", float("nan")),
                    )
                ),
                "global_cap_active_fraction": float(
                    diagnostics.get("global_cap_active_fraction", float("nan"))
                ),
                "byzantine_realized_contribution_share": byzantine_share,
                "max_client_contribution_norm": (
                    max(float(value) for value in contribution_norms)
                    if contribution_norms is not None
                    else float("nan")
                ),
                "complete_client_influence_cap": float(
                    diagnostics.get("complete_client_influence_cap", float("nan"))
                ),
                "contribution_cap_respected": bool(
                    diagnostics.get("client_contribution_cap_respected", True)
                ),
                "replace_one_bound": float(
                    diagnostics.get("replace_one_bound", float("nan"))
                ),
                "server_clip_rate_honest": float(
                    (torch.linalg.vector_norm(attacked[honest], dim=1) > server_clip)
                    .float()
                    .mean()
                    .item()
                ),
                "server_clip_rate_byzantine": (
                    float(
                        (
                            torch.linalg.vector_norm(attacked[byzantine], dim=1)
                            > server_clip
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    if bool(byzantine.any())
                    else float("nan")
                ),
                "aware_statistical_radius_min": float(aware_radii.min().item()),
                "aware_statistical_radius_max": float(aware_radii.max().item()),
                "common_statistical_radius_min": float(blind_radii.min().item()),
                "common_statistical_radius_max": float(blind_radii.max().item()),
                "all_finite": bool(torch.isfinite(reference).all()),
                "gate_sum_normalized": bool(
                    diagnostics.get("normalization_by_gate_sum", False)
                ),
                **tier_fields,
            }
        )
    return result


def _attach_baselines(rows: list[dict[str, Any]]) -> None:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    for row in rows:
        fcc = lookup[(row["pairing_id"], "fcc")]
        blind = lookup[(row["pairing_id"], BLIND)]
        aware = lookup[(row["pairing_id"], AWARE)]
        error = float(row["reference_error"])
        row["difference_vs_fcc"] = error - float(fcc["reference_error"])
        row["ratio_to_fcc"] = error / max(float(fcc["reference_error"]), 1.0e-12)
        row["difference_vs_sigma_blind"] = error - float(blind["reference_error"])
        row["ratio_to_sigma_blind"] = error / max(
            float(blind["reference_error"]), 1.0e-12
        )
        row["difference_vs_aware"] = error - float(aware["reference_error"])
        row["ratio_to_aware"] = error / max(float(aware["reference_error"]), 1.0e-12)


def _replace_one_audit(
    config: dict[str, Any], calibration: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = tuple(int(value) for value in config["cohort"]["block_sizes"])
    trials = int(config["randomness"]["replace_one_trials_per_seed_cell"])
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in base._noise_cells(config):
            clean, _, _, anchor = oracle._honest_clean_vectors(
                config,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
                include_outliers=True,
            )
            variances, _ = oracle._noise_variances(config, regime, permutation)
            observed = base._paired_private_noise(
                clean,
                variances,
                blocks,
                seed=int(seed),
                draw=0,
                geometry="orthogonal",
            )
            vectors = clip_l2(
                observed, float(config["aggregation"]["server_clip_norm"])
            )
            aware = _radii(
                config,
                variances,
                calibration,
                regime_name=str(regime["name"]),
                blind=False,
            )
            blind = _radii(
                config,
                variances,
                calibration,
                regime_name=str(regime["name"]),
                blind=True,
            )
            for trial in range(trials):
                client = (
                    oracle._seed(
                        "g0g-k3-replace-client",
                        seed,
                        regime["name"],
                        permutation,
                        trial,
                    )
                    % vectors.shape[0]
                )
                neighbour = vectors.clone()
                replacement = torch.randn(
                    vectors.shape[1],
                    generator=oracle._generator(
                        "g0g-k3-replacement",
                        seed,
                        regime["name"],
                        permutation,
                        trial,
                    ),
                    dtype=vectors.dtype,
                    device=vectors.device,
                )
                neighbour[client] = 100.0 * replacement
                left, diagnostics = _reference(
                    PRIMARY,
                    vectors,
                    anchor=anchor,
                    aware_radii=aware,
                    blind_radii=blind,
                    config=config,
                )
                right, _ = _reference(
                    PRIMARY,
                    neighbour,
                    anchor=anchor,
                    aware_radii=aware,
                    blind_radii=blind,
                    config=config,
                )
                difference = float(torch.linalg.vector_norm(left - right).item())
                bound = float(diagnostics["replace_one_bound"])
                rows.append(
                    {
                        "seed": int(seed),
                        "noise_regime": str(regime["name"]),
                        "noise_permutation": permutation,
                        "trial": trial,
                        "replaced_client": int(client),
                        "observed_replace_one_difference": difference,
                        "theoretical_replace_one_bound": bound,
                        "ratio_observed_to_bound": difference / bound,
                        "violation": difference > bound + 1.0e-6,
                    }
                )
    return rows


def _selected(
    rows: list[dict[str, Any]], candidate: str, predicate
) -> list[dict[str, Any]]:
    return [row for row in rows if row["candidate"] == candidate and predicate(row)]


def _ratio(
    rows: list[dict[str, Any]], candidate: str, baseline: str, predicate
) -> float:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    selected = _selected(rows, candidate, predicate)
    numerator = base._finite_mean(row["reference_error"] for row in selected)
    denominator = base._finite_mean(
        lookup[(row["pairing_id"], baseline)]["reference_error"] for row in selected
    )
    return numerator / max(denominator, 1.0e-12)


def _seed_contrasts(
    rows: list[dict[str, Any]], competitor: str, predicate
) -> list[float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["candidate"] in {PRIMARY, competitor} and predicate(row):
            grouped[(int(row["seed"]), str(row["candidate"]))].append(
                float(row["reference_error"])
            )
    seeds = sorted(
        seed
        for seed, candidate in grouped
        if candidate == PRIMARY and (seed, competitor) in grouped
    )
    return [
        base._finite_mean(grouped[(seed, PRIMARY)])
        - base._finite_mean(grouped[(seed, competitor)])
        for seed in seeds
    ]


def _worst_group_ratio(rows: list[dict[str, Any]], *, threats: set[str]) -> float:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["candidate"] == PRIMARY
            and row["noise_regime"] == "heteroscedastic"
            and row["threat"] in threats
        ):
            grouped[
                (
                    str(row["noise_permutation"]),
                    str(row["outlier_geometry"]),
                    str(row["threat"]),
                )
            ].append(float(row["ratio_to_fcc"]))
    return max(base._finite_mean(values) for values in grouped.values())


def _pooled_tier_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _selected(
        rows,
        PRIMARY,
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] == "none"
        ),
    )
    tail_rates: dict[str, float] = {}
    gate_means: dict[str, float] = {}
    for tier in TIERS:
        key = str(tier).replace(".", "p")
        count = sum(int(row[f"regular_count_tier_{key}"]) for row in selected)
        tails = sum(int(row[f"tail_count_tier_{key}"]) for row in selected)
        gates = sum(float(row[f"gate_sum_tier_{key}"]) for row in selected)
        if count:
            tail_rates[str(tier)] = tails / float(count)
            gate_means[str(tier)] = gates / float(count)
    return {
        "tail_rates": tail_rates,
        "gate_means": gate_means,
        "tail_rate_gap": max(tail_rates.values()) - min(tail_rates.values()),
        "gate_mean_gap": max(gate_means.values()) - min(gate_means.values()),
    }


def _clean_ratios_by_regime(
    rows: list[dict[str, Any]], baseline: str
) -> dict[str, float]:
    return {
        regime: _ratio(
            rows,
            PRIMARY,
            baseline,
            lambda row, selected_regime=regime: (
                row["noise_regime"] == selected_regime and row["threat"] == "none"
            ),
        )
        for regime in ("homogeneous", "heteroscedastic")
    }


def _honest_outlier_gate_drop(rows: list[dict[str, Any]]) -> dict[str, float]:
    lookup = {(row["pairing_id"], row["candidate"]): row for row in rows}
    result: dict[str, float] = {}
    for regime in ("homogeneous", "heteroscedastic"):
        selected = _selected(
            rows,
            PRIMARY,
            lambda row, selected_regime=regime: (
                row["noise_regime"] == selected_regime and row["threat"] == "none"
            ),
        )
        aware = base._finite_mean(
            lookup[(row["pairing_id"], AWARE)]["honest_outlier_gate_mean"]
            for row in selected
        )
        dual = base._finite_mean(row["honest_outlier_gate_mean"] for row in selected)
        result[regime] = max(0.0, aware - dual)
    return result


def _evaluate_gates(
    rows: list[dict[str, Any]],
    replace_rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    separated = {
        str(value) for value in config["threats"]["separated_for_primary_gate"]
    }
    primary = [row for row in rows if row["candidate"] == PRIMARY]
    clean = [row for row in primary if row["threat"] == "none"]
    fixed_identity = max(
        float(row["exact_difference_to_fcc_reference"])
        for row in rows
        if row["candidate"] == NO_GATE
    )
    homogeneous_identity = max(
        float(row["exact_difference_to_aware_reference"])
        for row in primary
        if row["noise_regime"] == "homogeneous"
    )
    clean_fcc = _clean_ratios_by_regime(rows, "fcc")
    clean_aware = _clean_ratios_by_regime(rows, AWARE)
    outlier_drop = _honest_outlier_gate_drop(rows)
    tail_rate = base._finite_mean(
        row["regular_combined_false_tail_rate"] for row in clean
    )
    pooled = _pooled_tier_diagnostics(rows)

    def separated_predicate(row: Mapping[str, Any]) -> bool:
        return bool(
            row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
        )

    fcc_contrasts = _seed_contrasts(rows, "fcc", separated_predicate)
    blind_contrasts = _seed_contrasts(rows, BLIND, separated_predicate)
    fcc_ci = base._ci95(fcc_contrasts)
    blind_ci = base._ci95(blind_contrasts)
    attacked = _selected(rows, PRIMARY, separated_predicate)
    mean_primary = base._finite_mean(row["reference_error"] for row in attacked)
    mean_fcc = mean_primary - float(fcc_ci["mean"])
    mean_blind = mean_primary - float(blind_ci["mean"])
    gain_fcc = (mean_fcc - mean_primary) / max(mean_fcc, 1.0e-12)
    gain_blind = (mean_blind - mean_primary) / max(mean_blind, 1.0e-12)

    def byzantine_high_predicate(row: Mapping[str, Any]) -> bool:
        return bool(
            row["noise_regime"] == "heteroscedastic"
            and row["noise_permutation"] == "byzantine_high"
            and row["threat"] in separated
        )

    byzantine_high_ratio = _ratio(rows, PRIMARY, BLIND, byzantine_high_predicate)
    byzantine_high_ci = base._ci95(
        _seed_contrasts(rows, BLIND, byzantine_high_predicate)
    )
    global_attacked_ratio_to_aware = _ratio(
        rows,
        PRIMARY,
        AWARE,
        lambda row: row["threat"] != "none",
    )
    worst_fcc = _worst_group_ratio(rows, threats=separated)
    worst_alie = _worst_group_ratio(rows, threats={"alie"})
    byzantine_share = max(
        base._finite_mean(
            row["byzantine_realized_contribution_share"]
            for row in attacked
            if (row["noise_permutation"], row["outlier_geometry"], row["threat"])
            == group
        )
        for group in {
            (row["noise_permutation"], row["outlier_geometry"], row["threat"])
            for row in attacked
        }
    )
    expected = (
        len(config["randomness"]["development_seeds"])
        * len(base._noise_cells(config))
        * len(config["cohort"]["honest_outliers"]["geometries"])
        * len(config["threats"]["names"])
        * len(CANDIDATES)
    )
    complete_fraction = len(rows) / float(expected)
    finite_fraction = base._finite_mean(
        1.0 if row["all_finite"] else 0.0 for row in rows
    )
    certified = [row for row in rows if row["candidate"] != "fcc"]
    cap_violations = sum(
        not bool(row["contribution_cap_respected"]) for row in certified
    )
    replace_violations = sum(bool(row["violation"]) for row in replace_rows)
    max_clean_fcc = max(clean_fcc.values())
    max_clean_aware = max(clean_aware.values())
    max_outlier_drop = max(outlier_drop.values())
    checks = {
        "complete": complete_fraction >= float(gates["complete_fraction_min"]),
        "finite": finite_fraction >= float(gates["finite_metric_fraction_min"]),
        "fixed_global_cap_identity": fixed_identity
        <= float(gates["fixed_global_cap_identity_abs_error_max"]),
        "homogeneous_dual_aware_identity": homogeneous_identity
        <= float(gates["homogeneous_dual_aware_identity_abs_error_max"]),
        "contribution_cap": cap_violations
        <= int(gates["contribution_cap_violation_max"]),
        "replace_one": replace_violations <= int(gates["replace_one_violation_max"]),
        "combined_false_tail_rate": tail_rate
        <= float(gates["regular_combined_false_tail_rate_max"]),
        "pooled_false_tail_tier_gap": float(pooled["tail_rate_gap"])
        <= float(gates["pooled_false_tail_tier_gap_max"]),
        "pooled_gate_mean_tier_gap": float(pooled["gate_mean_gap"])
        <= float(gates["pooled_gate_mean_tier_gap_max"]),
        "honest_outlier_gate_drop": max_outlier_drop
        <= float(gates["honest_outlier_gate_drop_vs_aware_max"]),
        "clean_noninferiority_vs_fcc": max_clean_fcc
        <= float(gates["clean_error_ratio_to_fcc_max"]),
        "clean_noninferiority_vs_aware": max_clean_aware
        <= float(gates["clean_error_ratio_to_aware_max"]),
        "attacked_ci_vs_fcc": float(fcc_ci["high"])
        <= float(gates["heteroscedastic_attacked_difference_ci95_high_max"]),
        "attacked_gain_vs_fcc": gain_fcc
        >= float(gates["heteroscedastic_attacked_relative_gain_vs_fcc_min"]),
        "attacked_ci_vs_sigma_blind": float(blind_ci["high"])
        <= float(gates["heteroscedastic_attacked_blind_difference_ci95_high_max"]),
        "attacked_gain_vs_sigma_blind": gain_blind
        >= float(gates["heteroscedastic_attacked_relative_gain_vs_sigma_blind_min"]),
        "byzantine_high_ratio_vs_sigma_blind": byzantine_high_ratio
        <= float(gates["byzantine_high_error_ratio_to_sigma_blind_max"]),
        "byzantine_high_ci_vs_sigma_blind": float(byzantine_high_ci["high"])
        <= float(gates["byzantine_high_difference_vs_sigma_blind_ci95_high_max"]),
        "global_attacked_noninferiority_vs_aware": global_attacked_ratio_to_aware
        <= float(gates["global_attacked_error_ratio_to_aware_max"]),
        "attacked_worst_group_ratio": worst_fcc
        <= float(gates["attacked_error_ratio_to_fcc_max"]),
        "alie_ratio": worst_alie <= float(gates["alie_error_ratio_to_fcc_max"]),
        "byzantine_contribution_share": byzantine_share
        <= float(gates["byzantine_realized_contribution_share_max"]),
    }
    return {
        "decision": (
            "promote_to_holdout" if all(checks.values()) else "stop_after_development"
        ),
        "all_gates_pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "development_rows": len(rows),
            "expected_development_rows": expected,
            "complete_fraction": complete_fraction,
            "finite_fraction": finite_fraction,
            "fixed_global_cap_max_abs_difference_vs_fcc": fixed_identity,
            "homogeneous_dual_max_abs_difference_vs_aware": homogeneous_identity,
            "contribution_cap_violations": cap_violations,
            "replace_one_trials": len(replace_rows),
            "replace_one_violations": replace_violations,
            "replace_one_max_ratio_observed_to_bound": max(
                float(row["ratio_observed_to_bound"]) for row in replace_rows
            ),
            "regular_combined_false_tail_rate": tail_rate,
            "pooled_tier_diagnostics": pooled,
            "honest_outlier_gate_drop_vs_aware_by_regime": outlier_drop,
            "clean_error_ratio_to_fcc_by_regime": clean_fcc,
            "clean_error_ratio_to_aware_by_regime": clean_aware,
            "heteroscedastic_attacked_difference_vs_fcc_seed_ci95": fcc_ci,
            "heteroscedastic_attacked_relative_gain_vs_fcc": gain_fcc,
            "heteroscedastic_attacked_difference_vs_sigma_blind_seed_ci95": blind_ci,
            "heteroscedastic_attacked_relative_gain_vs_sigma_blind": gain_blind,
            "byzantine_high_error_ratio_to_sigma_blind": byzantine_high_ratio,
            "byzantine_high_difference_vs_sigma_blind_seed_ci95": byzantine_high_ci,
            "global_attacked_error_ratio_to_aware": global_attacked_ratio_to_aware,
            "heteroscedastic_attacked_worst_group_ratio_to_fcc": worst_fcc,
            "heteroscedastic_alie_worst_group_ratio_to_fcc": worst_alie,
            "heteroscedastic_attacked_byzantine_contribution_share_worst_group": (
                byzantine_share
            ),
        },
    }


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["candidate"], row["noise_regime"], row["threat"])].append(row)
    result: list[dict[str, Any]] = []
    for (candidate, regime, threat), selected in sorted(grouped.items()):
        result.append(
            {
                "candidate": candidate,
                "noise_regime": regime,
                "threat": threat,
                "n_rows": len(selected),
                "reference_error_mean": base._finite_mean(
                    row["reference_error"] for row in selected
                ),
                "reference_error_std": base._finite_std(
                    row["reference_error"] for row in selected
                ),
                "ratio_to_fcc_mean": base._finite_mean(
                    row["ratio_to_fcc"] for row in selected
                ),
                "ratio_to_sigma_blind_mean": base._finite_mean(
                    row["ratio_to_sigma_blind"] for row in selected
                ),
                "ratio_to_aware_mean": base._finite_mean(
                    row["ratio_to_aware"] for row in selected
                ),
                "regular_combined_false_tail_rate_mean": base._finite_mean(
                    row["regular_combined_false_tail_rate"] for row in selected
                ),
                "gate_mean": base._finite_mean(row["gate_mean"] for row in selected),
                "honest_outlier_gate_mean": base._finite_mean(
                    row["honest_outlier_gate_mean"] for row in selected
                ),
                "byzantine_contribution_share_mean": base._finite_mean(
                    row["byzantine_realized_contribution_share"] for row in selected
                ),
            }
        )
    return result


def _make_figures(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    key_methods = ["fcc", BLIND, AWARE, PRIMARY]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, regime in zip(axes, ("homogeneous", "heteroscedastic"), strict=True):
        threats = list(dict.fromkeys(str(row["threat"]) for row in rows))
        x = range(len(threats))
        width = 0.19
        for index, method in enumerate(key_methods):
            means = [
                base._finite_mean(
                    row["reference_error"]
                    for row in rows
                    if row["candidate"] == method
                    and row["noise_regime"] == regime
                    and row["threat"] == threat
                )
                for threat in threats
            ]
            axis.bar(
                [value + (index - 1.5) * width for value in x],
                means,
                width,
                label=method,
            )
        axis.set_xticks(list(x), threats, rotation=25, ha="right")
        axis.set_title(
            "Bruit homogène" if regime == "homogeneous" else "Bruit hétéroscédastique"
        )
        axis.set_ylabel("Erreur L2 de référence")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    path = figure_dir / "k3_reference_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    separated = {"ipm", "bitflip_x10", "model_replacement"}
    selected = _selected(
        rows,
        PRIMARY,
        lambda row: (
            row["noise_regime"] == "heteroscedastic" and row["threat"] in separated
        ),
    )
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in selected:
        by_seed[int(row["seed"])].append(float(row["difference_vs_sigma_blind"]))
    seeds = sorted(by_seed)
    differences = [base._finite_mean(by_seed[seed]) for seed in seeds]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.bar(
        [str(seed) for seed in seeds],
        differences,
        color=["#188977" if value <= 0.0 else "#d95f02" for value in differences],
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Erreur K3 − erreur K2 sigma-blind")
    axis.set_title("Contraste apparié : attaques séparées hétéroscédastiques")
    axis.grid(axis="y", alpha=0.25)
    path = figure_dir / "k3_paired_difference_vs_blind.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))
    return paths


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    provenance: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    decision: Mapping[str, Any],
    figures: Sequence[str],
) -> None:
    checks = decision["checks"]
    observed = decision["observed"]
    gates = config["gates"]
    pooled = observed["pooled_tier_diagnostics"]
    gate_rows = [
        (
            "Complétude",
            observed["complete_fraction"],
            f">= {gates['complete_fraction_min']}",
            "complete",
        ),
        (
            "Valeurs finies",
            observed["finite_fraction"],
            f">= {gates['finite_metric_fraction_min']}",
            "finite",
        ),
        (
            "Identité sans gate/FCC",
            observed["fixed_global_cap_max_abs_difference_vs_fcc"],
            f"<= {gates['fixed_global_cap_identity_abs_error_max']}",
            "fixed_global_cap_identity",
        ),
        (
            "Identité K3/K2 en bruit homogène",
            observed["homogeneous_dual_max_abs_difference_vs_aware"],
            f"<= {gates['homogeneous_dual_aware_identity_abs_error_max']}",
            "homogeneous_dual_aware_identity",
        ),
        (
            "Violations du cap",
            observed["contribution_cap_violations"],
            f"<= {gates['contribution_cap_violation_max']}",
            "contribution_cap",
        ),
        (
            "Violations replace-one",
            observed["replace_one_violations"],
            f"<= {gates['replace_one_violation_max']}",
            "replace_one",
        ),
        (
            "Faux-tail combiné régulier (%)",
            100.0 * observed["regular_combined_false_tail_rate"],
            f"<= {100.0 * gates['regular_combined_false_tail_rate_max']:.2f}",
            "combined_false_tail_rate",
        ),
        (
            "Gap de faux-tail entre tiers (pp)",
            100.0 * pooled["tail_rate_gap"],
            f"<= {100.0 * gates['pooled_false_tail_tier_gap_max']:.2f}",
            "pooled_false_tail_tier_gap",
        ),
        (
            "Gap de gate moyenne entre tiers (pp)",
            100.0 * pooled["gate_mean_gap"],
            f"<= {100.0 * gates['pooled_gate_mean_tier_gap_max']:.2f}",
            "pooled_gate_mean_tier_gap",
        ),
        (
            "Baisse de gate honest-outlier (pp)",
            100.0
            * max(observed["honest_outlier_gate_drop_vs_aware_by_regime"].values()),
            f"<= {100.0 * gates['honest_outlier_gate_drop_vs_aware_max']:.2f}",
            "honest_outlier_gate_drop",
        ),
        (
            "Pire ratio propre K3/FCC",
            max(observed["clean_error_ratio_to_fcc_by_regime"].values()),
            f"<= {gates['clean_error_ratio_to_fcc_max']}",
            "clean_noninferiority_vs_fcc",
        ),
        (
            "Pire ratio propre K3/K2 aware",
            max(observed["clean_error_ratio_to_aware_by_regime"].values()),
            f"<= {gates['clean_error_ratio_to_aware_max']}",
            "clean_noninferiority_vs_aware",
        ),
        (
            "Borne haute IC95 : K3 - FCC",
            observed["heteroscedastic_attacked_difference_vs_fcc_seed_ci95"]["high"],
            f"<= {gates['heteroscedastic_attacked_difference_ci95_high_max']}",
            "attacked_ci_vs_fcc",
        ),
        (
            "Gain attaques vs FCC (%)",
            100.0 * observed["heteroscedastic_attacked_relative_gain_vs_fcc"],
            f">= {100.0 * gates['heteroscedastic_attacked_relative_gain_vs_fcc_min']:.2f}",
            "attacked_gain_vs_fcc",
        ),
        (
            "Borne haute IC95 : K3 - sigma-blind",
            observed["heteroscedastic_attacked_difference_vs_sigma_blind_seed_ci95"][
                "high"
            ],
            f"<= {gates['heteroscedastic_attacked_blind_difference_ci95_high_max']}",
            "attacked_ci_vs_sigma_blind",
        ),
        (
            "Gain attaques vs sigma-blind (%)",
            100.0 * observed["heteroscedastic_attacked_relative_gain_vs_sigma_blind"],
            f">= {100.0 * gates['heteroscedastic_attacked_relative_gain_vs_sigma_blind_min']:.2f}",
            "attacked_gain_vs_sigma_blind",
        ),
        (
            "Ratio `byzantine_high` K3/sigma-blind",
            observed["byzantine_high_error_ratio_to_sigma_blind"],
            f"<= {gates['byzantine_high_error_ratio_to_sigma_blind_max']}",
            "byzantine_high_ratio_vs_sigma_blind",
        ),
        (
            "Borne haute IC95 `byzantine_high`",
            observed["byzantine_high_difference_vs_sigma_blind_seed_ci95"]["high"],
            f"<= {gates['byzantine_high_difference_vs_sigma_blind_ci95_high_max']}",
            "byzantine_high_ci_vs_sigma_blind",
        ),
        (
            "Ratio global attaqué K3/K2 aware",
            observed["global_attacked_error_ratio_to_aware"],
            f"<= {gates['global_attacked_error_ratio_to_aware_max']}",
            "global_attacked_noninferiority_vs_aware",
        ),
        (
            "Pire ratio de groupe K3/FCC",
            observed["heteroscedastic_attacked_worst_group_ratio_to_fcc"],
            f"<= {gates['attacked_error_ratio_to_fcc_max']}",
            "attacked_worst_group_ratio",
        ),
        (
            "Pire ratio ALIE K3/FCC",
            observed["heteroscedastic_alie_worst_group_ratio_to_fcc"],
            f"<= {gates['alie_error_ratio_to_fcc_max']}",
            "alie_ratio",
        ),
        (
            "Part Byzantine maximale par groupe (%)",
            100.0
            * observed[
                "heteroscedastic_attacked_byzantine_contribution_share_worst_group"
            ],
            f"<= {100.0 * gates['byzantine_realized_contribution_share_max']:.2f}",
            "byzantine_contribution_share",
        ),
    ]
    lines = [
        "# G0g-K3 — décision expérimentale",
        "",
        "## Verdict",
        "",
        "**"
        + (
            "PROMOTION AU HOLDOUT"
            if decision["all_gates_pass"]
            else "ARRÊT AU DÉVELOPPEMENT"
        )
        + "**",
        "",
        "K3 intersecte la gate covariance-aware de K2 avec une gate commune à "
        "toutes les identités, sans changer le cap global ni les seuils K2.",
        "",
        "## Provenance gelée",
        "",
        f"- fichier : `{provenance['source_path']}`;",
        f"- SHA-256 vérifié : `{provenance['observed_sha256']}`;",
        "- recalibration K3 : **non**;",
        f"- seuils réutilisés : `{provenance['thresholds_reused_exactly']}`.",
        "",
        "## Gates préenregistrés",
        "",
        "| Test | Observation | Seuil | Verdict |",
        "|---|---:|---:|:---:|",
    ]
    for label, value, threshold, check in gate_rows:
        lines.append(
            f"| {label} | {_fmt(value, 6)} | {threshold} | "
            f"{'✓' if checks[check] else '**✗**'} |"
        )
    lines.extend(
        [
            "",
            "Les ratios de décision sont des ratios de moyennes sur l'estimand "
            "préenregistré. Ils ne doivent pas être confondus avec la moyenne "
            "des ratios cellule par cellule publiée dans `summary.csv`.",
            "",
            "## Certificat",
            "",
            f"- cap : {observed['contribution_cap_violations']} violation;",
            f"- replace-one : {observed['replace_one_violations']} violation sur {observed['replace_one_trials']} essais;",
            f"- maximum observé / borne 2G/n : {_fmt(observed['replace_one_max_ratio_observed_to_bound'])};",
            "- borne analytique : 2 × 0,13 / 25 = 0,0104;",
            "- coût local-DP additionnel : zéro, par post-traitement.",
            "",
            "## Table synthétique",
            "",
            "| Référence | Bruit | Attaque | Erreur L2 moyenne ± écart-type | Ratio/FCC | Ratio/K2 aware |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        if row["candidate"] not in {"fcc", BLIND, AWARE, PRIMARY}:
            continue
        lines.append(
            f"| {row['candidate']} | {row['noise_regime']} | {row['threat']} | "
            f"{_fmt(row['reference_error_mean'])} ± {_fmt(row['reference_error_std'])} | "
            f"{_fmt(row['ratio_to_fcc_mean'])} | {_fmt(row['ratio_to_aware_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Les écarts-types ci-dessus décrivent la dispersion poolée entre "
            "seeds, permutations et géométries corrélées. Ils ne sont pas des "
            "incertitudes inter-seeds. Les intervalles de confiance des gates "
            "sont calculés sur les cinq contrastes appariés par seed.",
            "",
            "## Portée",
            "",
            "Cette campagne isole une erreur de référence synthétique. Elle ne suffit "
            "pas à conclure sur l'accuracy, la fairness ou la convergence end-to-end.",
            "",
        ]
    )
    if figures:
        lines.extend(["## Figures", ""])
        for figure in figures:
            lines.extend([f"![{Path(figure).stem}]({ROOT / figure})", ""])
    failed = [name for name, passed in checks.items() if not passed]
    lines.extend(
        [
            "## Étape suivante",
            "",
            (
                "Tous les gates passent : ouvrir un holdout gelé sans modifier K3."
                if not failed
                else "Gates échoués : **" + ", ".join(failed) + "**. Aucun "
                "holdout et aucun résultat Fashion-MNIST ne doivent être ouverts."
            ),
            "",
            f"Configuration : `{config_path.relative_to(ROOT)}`. Résultats : `{output_dir.relative_to(ROOT)}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_dir: Path, report_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    calibration, provenance = _load_frozen_calibration(config)
    oracle._configure_runtime("mps")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        output_dir / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": base._sha256(config_path),
            "source_sha256": {
                "runner": base._sha256(Path(__file__).resolve()),
                "algorithm": base._sha256(
                    ROOT / "algorithms/gaussian_aware_reference.py"
                ),
                "shared_k2_runner": base._sha256(
                    ROOT / "scripts/run_gaussian_aware_reference_g0g_k2.py"
                ),
            },
            "frozen_calibration_sha256": provenance["observed_sha256"],
            "reserved_holdout_seeds": [
                int(value) for value in config["randomness"]["holdout_seeds"]
            ],
            "holdout_rule": config["randomness"]["holdout_rule"],
            "device": str(oracle._RUNTIME_DEVICE),
            "dtype": str(oracle._RUNTIME_DTYPE),
            "holdout_opened": False,
        },
    )
    base._atomic_json(output_dir / "calibration_provenance.json", provenance)
    rows: list[dict[str, Any]] = []
    for seed in config["randomness"]["development_seeds"]:
        for regime, permutation in base._noise_cells(config):
            for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                for threat in config["threats"]["names"]:
                    rows.extend(
                        _evaluate_cell(
                            config,
                            calibration,
                            seed=int(seed),
                            regime=regime,
                            permutation=str(permutation),
                            geometry=str(geometry),
                            threat=str(threat),
                        )
                    )
    _attach_baselines(rows)
    base._write_csv(output_dir / "development_rows.csv", rows)
    replace_rows = _replace_one_audit(config, calibration)
    base._write_csv(output_dir / "replace_one_rows.csv", replace_rows)
    summaries = _summaries(rows)
    base._write_csv(output_dir / "summary.csv", summaries)
    decision = _evaluate_gates(rows, replace_rows, config)
    decision["holdout_opened"] = False
    decision["privacy_claim"] = "zero additional local-DP cost by post-processing"
    decision["frozen_calibration_sha256"] = provenance["observed_sha256"]
    base._atomic_json(output_dir / "decision.json", decision)
    figures = _make_figures(rows, output_dir)
    _write_report(
        report_path,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        provenance=provenance,
        summaries=summaries,
        decision=decision,
        figures=figures,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_g0g_k3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/ldp_gradient_far/gaussian_aware_reference_g0g_k3_mps_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "output/analysis/Gaussian_Aware_G0g_K3_Monday_Report.md",
    )
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    run(args.config.resolve(), args.output_dir.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
