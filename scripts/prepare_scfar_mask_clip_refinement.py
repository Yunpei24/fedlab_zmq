#!/usr/bin/env python3
"""Materialize a one-point-per-mask clipping refinement after Step 1I."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "results/scfar_paper1_fmnist_step1i_mask_clip_calibration_v1/selection.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/scpfar/paper1/s0_step1ir_fmnist_mask_clip_refinement.yaml"
)

MODE_TO_METHOD = {
    "classifier_tail": "clipref_mask_classifier_tail",
    "last_layer": "clipref_mask_last_layer",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_matrix(selection_path: Path) -> dict[str, Any]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = dict(payload.get("selected_clip_norm_by_mode", {}))
    candidates.update(payload.get("refinement_candidate_by_mode", {}))
    if not candidates:
        raise RuntimeError("Step 1I produced no selectable or refinable mask")
    unknown = sorted(set(candidates).difference(MODE_TO_METHOD))
    if unknown:
        raise ValueError(f"Unsupported active modes: {unknown}")

    methods: dict[str, Any] = {}
    experiments: list[dict[str, Any]] = []
    for mode, clip_norm in sorted(candidates.items()):
        method_id = MODE_TO_METHOD[mode]
        methods[method_id] = {
            "algorithm": "scfar_dp",
            "role": "masked_nodp_quantile_clip_refinement",
            "tilt_policy": "fixed",
            "algo_config": {
                "active_parameter_mode": mode,
                "scfar_aggregation_rule": "uniform",
                "far_alpha": 0.0,
                "kappa_w": 1.0,
                "sensitivity_mode": "automatic_certified",
            },
        }
        experiments.append(
            {
                "id": f"step1ir_{mode}",
                "expected_tasks": 1,
                "scenario_ids": ["fmnist_lenet5_b01"],
                "method_ids": [method_id],
                "reference_ids": ["fcc"],
                "anchor_ids": ["previous_release"],
                "threat_ids": ["clean"],
                "tilt_ids": ["uniform_k1"],
                "privacy_ids": ["no_dp"],
                "tau_over_c": [1.0],
                "user_clip_norms": [float(clip_norm)],
                "seed_pairs": [{"partition_seed": 104, "training_seed": 17}],
            }
        )

    return {
        "schema_version": 1,
        "matrix_id": "s0_step1ir_fmnist_mask_clip_refinement",
        "common": "common.yaml",
        "description": (
            "One deterministic validation run per active mask. C is the "
            "median over rounds of the per-round median pre-clipping update "
            "norm measured by the essentially-unclipped Step 1I probe."
        ),
        "common_overrides": {
            "protocol_id": "scfar_dp_mask_clip_refinement_v1",
            "execution_scope": "public_static_parameter_mask_clip_refinement",
            "invariants": {
                "parameter_mask_selection": "public_architecture_only",
                "candidate_selection_source": _portable_path(selection_path),
                "candidate_selection_sha256": _sha256(selection_path),
                "candidate_statistic": "median_t_of_preclip_norm_p50_i",
                "frozen_coordinates_depend_on_private_data": False,
                "frozen_coordinates_transmitted": False,
                "frozen_coordinates_noised": False,
            },
            "base_config": {
                "training": {
                    "num_rounds": 20,
                    "algo_config": {"privacy_num_rounds": 20},
                }
            },
            "methods": methods,
        },
        "preregistration": {
            "upstream_selection": {
                "source": _portable_path(selection_path),
                "sha256": _sha256(selection_path),
            },
            "development_seed_only": {
                "partition_seed": 104,
                "training_seed": 17,
            },
            "horizon_rounds": 20,
            "public_grid": {"user_clip_norm_by_mode": candidates},
            "feasibility_gate": payload["decision_rule"] | {
                "median_user_clip_rate_min_exclusive": 0.05,
                "median_user_clip_rate_max_exclusive": 0.95,
            },
            "selection_rule": (
                "A candidate is retained only if its validation run passes all "
                "utility and clipping-health gates."
            ),
        },
        "experiments": experiments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_path = args.selection.resolve()
    output_path = args.output.resolve()
    matrix = build_matrix(selection_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(matrix, sort_keys=False, width=88), encoding="utf-8"
    )
    print(f"WROTE {output_path}")
    print(
        "CANDIDATES "
        f"{matrix['preregistration']['public_grid']['user_clip_norm_by_mode']}"
    )


if __name__ == "__main__":
    main()
