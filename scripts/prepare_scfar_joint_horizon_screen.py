#!/usr/bin/env python3
"""Materialize Step 1J from the immutable Step 1I clipping selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT
    / "results/scfar_paper1_fmnist_step1ir_mask_clip_refinement_v1/selection.json"
)
DEFAULT_OUTPUT = (
    ROOT / "configs/scpfar/paper1/s0_step1j_fmnist_joint_mask_clip_horizon.yaml"
)

MODE_TO_METHOD = {
    "classifier_tail": "joint_mask_classifier_tail",
    "last_layer": "joint_mask_last_layer",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_matrix(selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("selected_clip_norm_by_mode", {})
    unknown = sorted(set(chosen).difference(MODE_TO_METHOD))
    if unknown:
        raise ValueError(f"Unsupported selected active modes: {unknown}")
    if not chosen:
        raise RuntimeError("Step 1I selected no mask; Step 1J must not be run")

    methods: dict[str, Any] = {}
    experiments: list[dict[str, Any]] = []
    for mode, clip_norm in sorted(chosen.items()):
        method_id = MODE_TO_METHOD[mode]
        methods[method_id] = {
            "algorithm": "scfar_dp",
            "role": "joint_mask_clip_horizon_dpfedavg_screen",
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
                "id": f"step1j_{mode}",
                "expected_tasks": 2,
                "scenario_ids": ["fmnist_lenet5_b01"],
                "method_ids": [method_id],
                "reference_ids": ["fcc"],
                "anchor_ids": ["previous_release"],
                "threat_ids": ["clean"],
                "tilt_ids": ["uniform_k1"],
                "privacy_ids": ["no_dp", "eps10"],
                "tau_over_c": [1.0],
                "user_clip_norms": [float(clip_norm)],
                "seed_pairs": [{"partition_seed": 104, "training_seed": 17}],
            }
        )

    return {
        "schema_version": 1,
        "matrix_id": "s0_step1j_fmnist_joint_mask_clip_horizon",
        "common": "common.yaml",
        "description": (
            "Matched no-DP/epsilon=10 horizon screen generated mechanically "
            "from the Step 1I clipping selection. Each task is executed at "
            "T in {5,10,20} through the runner's pilot-rounds lock."
        ),
        "common_overrides": {
            "protocol_id": "scfar_dp_joint_mask_clip_horizon_v1",
            "execution_scope": "public_static_mask_clip_horizon_screen",
            "description": (
                "Development screen for the interaction between active update "
                "dimension, calibrated clipping and number of central releases."
            ),
            "invariants": {
                "parameter_mask_selection": "public_architecture_only",
                "clip_selection_source": _portable_path(selection_path),
                "clip_selection_sha256": _sha256(selection_path),
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
                "selected_clip_norm_by_mode": chosen,
            },
            "development_seed_only": {
                "partition_seed": 104,
                "training_seed": 17,
            },
            "horizon_rounds": [5, 10, 20],
            "privacy_profiles": ["no_dp", "eps10"],
            "private_feasibility_gate": {
                "final_test_accuracy_min": 0.50,
                "final_worst20_accuracy_min": 0.20,
                "median_noise_to_clean_aggregate_ratio_max": 20.0,
            },
            "matched_control_gate": {
                "final_test_accuracy_min": 0.50,
                "final_worst20_accuracy_min": 0.20,
                "median_user_clip_rate_min_exclusive": 0.05,
                "median_user_clip_rate_max_exclusive": 0.95,
            },
            "selection_rule": (
                "For each mask, select the smallest T for which both its "
                "matched no-DP arm and epsilon=10 arm pass every corresponding "
                "gate. Exclude the mask if no horizon passes."
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
    print(f"SELECTED {matrix['preregistration']['upstream_selection']['selected_clip_norm_by_mode']}")


if __name__ == "__main__":
    main()
