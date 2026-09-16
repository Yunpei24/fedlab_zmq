#!/usr/bin/env python3
"""Validate, expand, dry-run and execute the DT-LDP-FAR E1--E8 protocol.

The protocol intentionally has two matrices: a small pilot and a bounded full
campaign.  Every Cartesian cell is resolved into its own YAML file and output
directory.  Array jobs can therefore be resumed independently without
silently changing another task's configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "dt_ldp_far" / "pilot_e1_e8.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "dt_ldp_far"
SCHEMA_VERSION = 1
KNOWN_ALGORITHMS = {
    "fedavg",
    "far",
    "dpfedavg",
    "dpqffl",
    "fedfdp",
    "dpfar",
    "dt_ldp_far",
}
KNOWN_REFERENCES = {
    "centered_clipping",
    "regularized_huber",
    "cm_nnm",
    "trmean_nnm",
    "rfa",
}
KNOWN_ATTACKS = {"none", "bf", "ipm", "alie", "minmax", "minsum"}


def _slug(value: object) -> str:
    return "".join(
        character if character.isalnum() else "_"
        for character in str(value).strip().lower()
    ).strip("_")


def _deep_merge(
    base: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tilt_tau_max(
    num_clients: int,
    kappa_w: float,
    *,
    active_clients: int | None = None,
) -> float:
    """Return the analytic logit-range cap ensuring q_max <= kappa_w / n.

    ``active_clients`` is the public size of the support of the softmax.  It
    differs from ``num_clients`` when a deterministic admissibility filter
    assigns zero weight to a fixed number of clients.
    """

    if num_clients < 2 or not 1.0 <= kappa_w < num_clients:
        raise ValueError("Need n>=2 and 1<=kappa_w<n")
    active = num_clients if active_clients is None else int(active_clients)
    if not 1 <= active <= num_clients:
        raise ValueError("active_clients must lie in [1,n]")
    ratio = kappa_w * (active - 1) / (num_clients - kappa_w)
    if ratio < 1.0:
        raise ValueError(
            "No non-negative tilt can guarantee kappa_w/n with this many "
            "active clients"
        )
    return math.log(ratio)


@dataclass(frozen=True)
class Task:
    index: int
    campaign_id: str
    experiment_id: str
    scenario_id: str
    method_id: str
    reference_id: str
    threat_id: str
    privacy_id: str
    tilt_id: str
    geometry_id: str
    local_epochs: int
    delay: int
    partition_seed: int
    training_seed: int
    config: dict[str, Any]
    output_dir: Path

    @property
    def task_id(self) -> str:
        return "__".join(
            (
                self.experiment_id,
                self.scenario_id,
                self.method_id,
                self.reference_id,
                self.threat_id,
                self.privacy_id,
                self.tilt_id,
                self.geometry_id,
                f"epochs_{self.local_epochs}",
                f"delay_{self.delay}",
                f"pseed_{self.partition_seed}",
                f"tseed_{self.training_seed}",
            )
        )

    @property
    def resolved_config_path(self) -> Path:
        return self.output_dir / "resolved_config.yaml"


def load_protocol(matrix_path: Path) -> dict[str, Any]:
    matrix_path = matrix_path.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError(f"Matrix must be a YAML mapping: {matrix_path}")
    common_name = matrix.get("common")
    if not common_name:
        raise ValueError(f"Matrix does not declare common: {matrix_path}")
    common_path = (matrix_path.parent / str(common_name)).resolve()
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    if not isinstance(common, dict):
        raise ValueError(f"Common protocol must be a YAML mapping: {common_path}")
    return {
        "matrix": matrix,
        "common": common,
        "matrix_path": matrix_path,
        "common_path": common_path,
    }


def _profile(common: dict[str, Any], family: str, profile_id: str) -> dict[str, Any]:
    profiles = common.get(family, {})
    if profile_id not in profiles:
        raise KeyError(f"Unknown {family} profile {profile_id!r}")
    profile = profiles[profile_id]
    if not isinstance(profile, dict):
        raise TypeError(f"{family}.{profile_id} must be a mapping")
    return copy.deepcopy(profile)


def _seed_pairs(matrix: dict[str, Any], common: dict[str, Any]) -> list[dict[str, int]]:
    all_pairs = common.get("seed_pairs", [])
    indices = matrix.get("seed_pair_indices", [])
    pairs: list[dict[str, int]] = []
    for index in indices:
        if not isinstance(index, int) or not 0 <= index < len(all_pairs):
            raise ValueError(f"Invalid seed_pair_indices entry: {index!r}")
        pair = all_pairs[index]
        pairs.append(
            {
                "partition_seed": int(pair["partition_seed"]),
                "training_seed": int(pair["training_seed"]),
            }
        )
    return pairs


def _resolve_config(
    common: dict[str, Any],
    *,
    scenario_id: str,
    method_id: str,
    reference_id: str,
    threat_id: str,
    privacy_id: str,
    tilt_id: str,
    geometry_id: str,
    local_epochs: int,
    delay: int,
    partition_seed: int,
    training_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = _deep_merge(
        common["base_config"], _profile(common, "scenarios", scenario_id)
    )
    method = _profile(common, "methods", method_id)
    reference = _profile(common, "references", reference_id)
    threat = _profile(common, "threats", threat_id)
    privacy = _profile(common, "privacy", privacy_id)
    tilt = _profile(common, "tilts", tilt_id)
    geometry = _profile(common, "geometries", geometry_id)

    config["training"]["algorithm"] = str(method["algorithm"])
    algo = _deep_merge(
        config["training"].get("algo_config", {}), method.get("algo_config")
    )
    algo = _deep_merge(algo, reference.get("algo_config"))
    algo = _deep_merge(algo, privacy.get("algo_config"))
    algo = _deep_merge(algo, geometry.get("algo_config"))
    algo = _deep_merge(algo, tilt.get("algo_config"))
    algo = _deep_merge(
        algo,
        {
            "num_byzantine": int(threat["num_byzantine"]),
            "attack": threat["attack"],
            "local_epochs": int(local_epochs),
            "tilt_delay_rounds": int(delay),
        },
    )

    n = int(config["clients"]["num_clients"])
    kappa_w = float(tilt["kappa_w"])
    tau_fraction = float(tilt["tau_fraction_of_max"])
    active_clients = None
    if bool(algo.get("dt_admissibility_filter_enabled", False)):
        excluded = int(
            algo.get(
                "dt_admissibility_excluded_clients",
                algo.get("dt_admissibility_max_byzantine", 0),
            )
        )
        active_clients = n - excluded
    score_transform = str(algo.get("dt_score_transform", "bounded_normalized"))
    if str(method["algorithm"]) == "dt_ldp_far" and score_transform == "raw_distance":
        server_clip = float(algo["server_clip_norm"])
        anchor_clip = float(algo.get("anchor_clip_norm", server_clip))
        if anchor_clip > server_clip + 1e-12:
            raise ValueError(
                "raw_distance DT-LDP-FAR requires anchor_clip_norm <= "
                "server_clip_norm"
            )
        if str(algo.get("dt_reference", "centered_clipping")) not in {
            "centered_clipping",
            "fcc",
            "f_cc",
        }:
            raise ValueError(
                "raw_distance DT-LDP-FAR currently requires centered_clipping"
            )
        public_score_range = 2.0 * server_clip
    else:
        public_score_range = 1.0
    tau_maximum = (
        tilt_tau_max(n, kappa_w, active_clients=active_clients)
        / public_score_range
    )
    resolved_tau = tau_fraction * tau_maximum
    algo["kappa_w"] = kappa_w
    algo["tilt_tau"] = resolved_tau
    algo["tilt_tau_fraction_of_max"] = tau_fraction
    algo["tilt_public_score_range"] = public_score_range
    # FAR calls its tilt ``far_alpha``.  Matched current-round profiles opt in
    # to the same bounded-normalised scores as DT-LDP-FAR; native FAR profiles
    # retain the manuscript's raw-distance score.
    if str(method["algorithm"]) in {"far", "dpfar"}:
        algo["far_alpha"] = resolved_tau

    rounds = int(config["training"]["num_rounds"])
    algo["privacy_num_rounds"] = rounds
    algo["expected_num_clients"] = n
    algo["experiment_privacy_profile"] = privacy_id
    algo["experiment_geometry_profile"] = geometry_id
    config["training"]["algo_config"] = algo
    config["data"]["partition_seed"] = int(partition_seed)
    config["seed"] = int(training_seed)
    config["output_dir"] = str(output_dir)
    config["reproduction"] = {
        "protocol_id": str(common["protocol_id"]),
        "algorithmic_scope": "DT-LDP-FAR full-update, full-participation lane",
        "comparison_lane": str(method.get("comparison_lane", "method_native")),
        "privacy_label_policy": "report_realised_epsilon_not_profile_name",
        "server_oracle_diagnostics_are_not_published": bool(
            algo.get("enable_oracle_diagnostics", False)
        ),
        "private_client_oracles_make_run_non_private_diagnostic": bool(
            algo.get("enable_private_client_oracle_diagnostics", False)
        ),
        "privacy_adjacency": str(algo.get("privacy_adjacency", "unspecified")),
        "local_sampling_scheme": str(algo.get("sampling_scheme", "fixed_minibatch")),
        "poisson_sampling_rate": (
            float(algo["privacy_sampling_rate_override"])
            if algo.get("sampling_scheme") == "poisson"
            and algo.get("privacy_sampling_rate_override") is not None
            else None
        ),
        "fixed_without_replacement_batch_size": (
            int(algo["fixed_batch_size"])
            if algo.get("sampling_scheme") == "fixed_without_replacement"
            and algo.get("fixed_batch_size") is not None
            else None
        ),
        "local_sampling_rate": (
            float(algo["privacy_sampling_rate_override"])
            if algo.get("privacy_sampling_rate_override") is not None
            else None
        ),
        "privacy_public_dataset_size": algo.get("privacy_public_dataset_size"),
        "tilt_influence_certificate_claimed": bool(
            tau_fraction <= 1.0
            and str(algo.get("tilt_bound_policy", "error")) == "error"
        ),
        "uncertified_tilt_is_local_dp_postprocessing_diagnostic": bool(
            algo.get("allow_uncertified_tilt_diagnostic", False)
        ),
        "local_steps_semantics": (
            "local_epochs axis denotes independent Poisson DP steps, not full data passes"
            if algo.get("sampling_scheme") == "poisson"
            else (
                "local_epochs axis denotes independent fixed-size WOR DP steps, not full data passes"
                if algo.get("sampling_scheme") == "fixed_without_replacement"
                else "complete fixed-minibatch local epochs"
            )
        ),
        # Current-round and delayed arms with this same key are required to
        # use identical initialization, partition, local samples, client
        # order, local optimiser randomness, DP noise and attack randomness.
        # Only method/delay are excluded from the key.
        "randomness_pair_key": hashlib.sha256(
            "|".join(
                (
                    str(scenario_id),
                    str(reference_id),
                    str(threat_id),
                    str(privacy_id),
                    str(geometry_id),
                    str(local_epochs),
                    str(partition_seed),
                    str(training_seed),
                )
            ).encode("utf-8")
        ).hexdigest()[:16],
        "randomness_pairing_contract": (
            "same key means identical client-side randomness; method, tilt "
            "and server weighting timestamp may differ"
        ),
        "axes": {
            "scenario": scenario_id,
            "method": method_id,
            "reference": reference_id,
            "threat": threat_id,
            "privacy": privacy_id,
            "tilt": tilt_id,
            "geometry": geometry_id,
            "local_epochs": int(local_epochs),
            "delay": int(delay),
            "partition_seed": int(partition_seed),
            "training_seed": int(training_seed),
        },
    }
    return config


def expand_tasks(
    document: dict[str, Any],
    *,
    output_root: Path,
    pilot_rounds: int | None = None,
) -> list[Task]:
    matrix, common = document["matrix"], document["common"]
    pairs = _seed_pairs(matrix, common)
    campaign_id = str(matrix["campaign_id"])
    tasks: list[Task] = []
    for experiment in matrix.get("experiments", []):
        experiment_id = str(experiment["id"])
        axes = (
            experiment["scenarios"],
            experiment["methods"],
            experiment["references"],
            experiment["threats"],
            experiment["privacy"],
            experiment["tilts"],
            experiment["geometries"],
            experiment["local_epochs"],
            experiment["delays"],
            pairs,
        )
        for values in itertools.product(*axes):
            (
                scenario_id,
                method_id,
                reference_id,
                threat_id,
                privacy_id,
                tilt_id,
                geometry_id,
                local_epochs,
                delay,
                pair,
            ) = values
            components = (
                campaign_id,
                experiment_id,
                scenario_id,
                method_id,
                reference_id,
                threat_id,
                privacy_id,
                tilt_id,
                geometry_id,
                f"epochs_{local_epochs}",
                f"delay_{delay}",
                f"pseed_{pair['partition_seed']}",
                f"tseed_{pair['training_seed']}",
            )
            output_dir = output_root.joinpath(*(_slug(item) for item in components))
            config = _resolve_config(
                common,
                scenario_id=str(scenario_id),
                method_id=str(method_id),
                reference_id=str(reference_id),
                threat_id=str(threat_id),
                privacy_id=str(privacy_id),
                tilt_id=str(tilt_id),
                geometry_id=str(geometry_id),
                local_epochs=int(local_epochs),
                delay=int(delay),
                partition_seed=int(pair["partition_seed"]),
                training_seed=int(pair["training_seed"]),
                output_dir=output_dir,
            )
            if pilot_rounds is not None:
                if pilot_rounds <= 0:
                    raise ValueError("pilot_rounds must be positive")
                output_dir = (
                    output_root
                    / "short_runs"
                    / f"rounds_{pilot_rounds}"
                    / Path(*(_slug(item) for item in components))
                )
                config["training"]["num_rounds"] = int(pilot_rounds)
                config["training"]["algo_config"]["privacy_num_rounds"] = int(
                    pilot_rounds
                )
                config["output_dir"] = str(output_dir)
                config["reproduction"][
                    "algorithmic_scope"
                ] = "short-run infrastructure check, not paper evidence"
            tasks.append(
                Task(
                    index=-1,
                    campaign_id=campaign_id,
                    experiment_id=experiment_id,
                    scenario_id=str(scenario_id),
                    method_id=str(method_id),
                    reference_id=str(reference_id),
                    threat_id=str(threat_id),
                    privacy_id=str(privacy_id),
                    tilt_id=str(tilt_id),
                    geometry_id=str(geometry_id),
                    local_epochs=int(local_epochs),
                    delay=int(delay),
                    partition_seed=int(pair["partition_seed"]),
                    training_seed=int(pair["training_seed"]),
                    config=config,
                    output_dir=output_dir,
                )
            )
    return [replace(task, index=index) for index, task in enumerate(tasks)]


def _task_issues(task: Task) -> list[str]:
    cfg = task.config
    training = cfg["training"]
    clients = cfg["clients"]
    algo = training["algo_config"]
    n = int(clients["num_clients"])
    issues: list[str] = []
    if training["algorithm"] not in KNOWN_ALGORITHMS:
        issues.append(f"unsupported algorithm {training['algorithm']!r}")
    if float(clients.get("sample_fraction", 0.0)) != 1.0:
        issues.append("sample_fraction must equal one in the primary lane")
    if int(clients.get("min_clients", 0)) != n:
        issues.append("min_clients must equal num_clients")
    if float(clients.get("dropout_rate", 0.0)) != 0.0:
        issues.append("dropout_rate must equal zero")
    if int(algo.get("local_epochs", 0)) < 1:
        issues.append("local_epochs must be at least one")
    if int(algo.get("privacy_num_rounds", -1)) != int(training["num_rounds"]):
        issues.append("privacy_num_rounds must equal training.num_rounds")
    attack = algo.get("attack", {})
    if str(attack.get("name", "none")).lower() not in KNOWN_ATTACKS:
        issues.append(f"unsupported attack {attack.get('name')!r}")
    if int(attack.get("num_byzantine", 0)) != int(algo.get("num_byzantine", 0)):
        issues.append("attack and algorithm Byzantine counts differ")
    if len(attack.get("client_ids", [])) != int(attack.get("num_byzantine", 0)):
        issues.append("explicit Byzantine client IDs do not match num_byzantine")

    if training["algorithm"] == "dt_ldp_far":
        if int(algo.get("tilt_delay_rounds", 0)) < 1:
            issues.append("DT-LDP-FAR delay must be at least one round")
        if not bool(algo.get("require_full_participation", False)):
            issues.append("DT-LDP-FAR primary lane must require full participation")
        if int(algo.get("expected_num_clients", -1)) != n:
            issues.append("expected_num_clients must equal num_clients")
        reference = str(algo.get("dt_reference", ""))
        if reference not in KNOWN_REFERENCES:
            issues.append(f"unsupported DT-LDP-FAR reference {reference!r}")
        kappa_w = float(algo.get("kappa_w", 0.0))
        requested_tau = float(algo.get("tilt_tau", -1.0))
        active_clients = None
        if bool(algo.get("dt_admissibility_filter_enabled", False)):
            excluded = int(
                algo.get(
                    "dt_admissibility_excluded_clients",
                    algo.get("dt_admissibility_max_byzantine", 0),
                )
            )
            active_clients = n - excluded
        public_score_range = float(algo.get("tilt_public_score_range", 1.0))
        certified_tau_max = (
            tilt_tau_max(n, kappa_w, active_clients=active_clients)
            / public_score_range
        )
        tau_is_uncertified = requested_tau > certified_tau_max + 1e-12
        explicit_stress_lane = bool(
            algo.get("allow_uncertified_tilt_diagnostic", False)
        )
        if requested_tau < 0.0:
            issues.append("tilt_tau must be non-negative")
        if tau_is_uncertified and not explicit_stress_lane:
            issues.append("tilt_tau lies outside the analytic cap")
        if explicit_stress_lane:
            if not tau_is_uncertified:
                issues.append(
                    "uncertified tilt diagnostic must actually exceed the analytic cap"
                )
            if str(algo.get("tilt_bound_policy")) != "diagnostic_only":
                issues.append(
                    "uncertified tilt diagnostic requires diagnostic_only policy"
                )
            if "alpha_stress" not in task.experiment_id.lower():
                issues.append(
                    "uncertified tilt is permitted only in an alpha_stress experiment"
                )
        elif str(algo.get("tilt_bound_policy")) != "error":
            issues.append(
                "certified DT-LDP-FAR runs must reject, not silently clip, invalid tau"
            )
        if float(algo.get("server_clip_norm", 0.0)) <= 0.0:
            issues.append("server_clip_norm must be positive")
        if float(algo.get("distance_clip", 0.0)) <= 0.0:
            issues.append("distance_clip must be positive")
        if bool(algo.get("enable_dp", True)):
            scheme = str(algo.get("sampling_scheme", "")).lower()
            adjacency = str(algo.get("privacy_adjacency", "")).lower()
            if scheme == "poisson" and adjacency != "add_remove":
                issues.append("private Poisson runs require add_remove adjacency")
            elif scheme == "fixed_without_replacement":
                if adjacency != "replace_one":
                    issues.append("fixed-size WOR runs require replace_one adjacency")
                if int(algo.get("fixed_batch_size", 0)) < 1:
                    issues.append(
                        "fixed-size WOR runs require a positive fixed_batch_size"
                    )
            elif scheme != "poisson":
                issues.append(
                    "private DT-LDP-FAR runs require genuine Poisson or "
                    "fixed-size without-replacement sampling"
                )
            sampling_rate = float(algo.get("privacy_sampling_rate_override", 0.0))
            if not 0.0 < sampling_rate <= 1.0:
                issues.append(
                    "private DT-LDP-FAR runs require an explicit sampling rate in (0,1]"
                )
            if int(algo.get("privacy_public_dataset_size", 0)) < 1:
                issues.append(
                    "private DT-LDP-FAR runs require a positive public local "
                    "dataset capacity"
                )
        if (
            bool(algo.get("enable_private_client_oracle_diagnostics", False))
            and task.experiment_id != "E2_noise_decoupling"
            and not (
                bool(algo.get("non_private_diagnostic_transcript", False))
                and "oracle_diagnostic" in task.experiment_id.lower()
            )
        ):
            issues.append(
                "private client oracles require an explicitly labelled non-private "
                "oracle diagnostic experiment"
            )
    return issues


def validate_protocol(document: dict[str, Any]) -> list[str]:
    matrix, common = document["matrix"], document["common"]
    errors: list[str] = []
    if int(matrix.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append("unsupported matrix schema_version")
    if int(common.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append("unsupported common schema_version")
    experiments = matrix.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return errors + ["matrix must contain non-empty experiments"]

    # Some confirmatory matrices depend on a profile selected by an earlier
    # public calibration. Make that dependency executable: a checked-in
    # placeholder cannot be launched until the gate evidence explicitly
    # records the selected profile.
    selection = matrix.get("requires_geometry_selection")
    if selection is not None:
        if not isinstance(selection, dict):
            errors.append("requires_geometry_selection must be a mapping")
        else:
            status = str(selection.get("status", "pending")).lower()
            winner = selection.get("selected_geometry")
            evidence_value = selection.get("evidence")
            required_evidence_status = str(
                selection.get("required_evidence_status", "selected")
            ).lower()
            if status != "selected" or not isinstance(winner, str) or not winner:
                errors.append(
                    "matrix is blocked until the n=25 geometry gate records "
                    "status=selected and selected_geometry"
                )
            if not isinstance(evidence_value, str) or not evidence_value:
                errors.append("geometry selection requires an evidence JSON path")
            else:
                evidence_path = (ROOT / evidence_value).resolve()
                if not evidence_path.exists():
                    errors.append(
                        f"geometry selection evidence is missing: {evidence_path}"
                    )
                else:
                    try:
                        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        errors.append(
                            f"invalid geometry selection evidence {evidence_path}: {error}"
                        )
                    else:
                        if winner and evidence.get("selected_geometry") != winner:
                            errors.append(
                                "matrix selected_geometry disagrees with gate evidence"
                            )
                        if (
                            str(evidence.get("status", "")).lower()
                            != required_evidence_status
                        ):
                            errors.append(
                                "geometry gate evidence has status "
                                f"{evidence.get('status')!r}, expected "
                                f"{required_evidence_status!r}"
                            )
            if isinstance(winner, str) and winner:
                for experiment in experiments:
                    if experiment.get("geometries") != [winner]:
                        errors.append(
                            f"{experiment.get('id', 'missing')}: geometries must equal "
                            f"the selected profile [{winner!r}]"
                        )

    score_confirmation = matrix.get("requires_score_confirmation")
    if score_confirmation is not None:
        if not isinstance(score_confirmation, dict):
            errors.append("requires_score_confirmation must be a mapping")
        else:
            evidence_value = score_confirmation.get("evidence")
            required_profile = score_confirmation.get("profile")
            required_reference = score_confirmation.get("reference")
            if not isinstance(evidence_value, str) or not evidence_value:
                errors.append("score confirmation requires an evidence JSON path")
            else:
                evidence_path = (ROOT / evidence_value).resolve()
                if not evidence_path.exists():
                    errors.append(
                        f"score confirmation evidence is missing: {evidence_path}"
                    )
                else:
                    try:
                        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        errors.append(
                            f"invalid score confirmation evidence {evidence_path}: {error}"
                        )
                    else:
                        candidate = evidence.get("candidate", {})
                        if not bool(evidence.get("confirmed", False)):
                            errors.append(
                                "matrix is blocked until score evidence records "
                                "confirmed=true"
                            )
                        if (
                            required_profile
                            and candidate.get("profile") != required_profile
                        ):
                            errors.append(
                                "matrix score profile disagrees with confirmation evidence"
                            )
                        if (
                            required_reference
                            and candidate.get("robust_reference") != required_reference
                        ):
                            errors.append(
                                "matrix score reference disagrees with confirmation evidence"
                            )
    try:
        pairs = _seed_pairs(matrix, common)
    except (KeyError, TypeError, ValueError) as error:
        return errors + [str(error)]
    if not pairs:
        errors.append("matrix selects no seed pairs")

    required_axes = (
        "scenarios",
        "methods",
        "references",
        "threats",
        "privacy",
        "tilts",
        "geometries",
        "local_epochs",
        "delays",
    )
    total_expected = 0
    experiment_ids: set[str] = set()
    for experiment in experiments:
        experiment_id = str(experiment.get("id", "missing"))
        if experiment_id in experiment_ids:
            errors.append(f"duplicate experiment id {experiment_id!r}")
        experiment_ids.add(experiment_id)
        for axis in required_axes:
            if not isinstance(experiment.get(axis), list) or not experiment[axis]:
                errors.append(f"{experiment_id}: {axis} must be a non-empty list")
        if errors:
            continue
        actual = math.prod(len(experiment[axis]) for axis in required_axes) * len(pairs)
        expected = int(experiment.get("expected_tasks", -1))
        total_expected += expected
        if actual != expected:
            errors.append(
                f"{experiment_id}: expected_tasks={expected}, expanded={actual}"
            )
    if total_expected != int(matrix.get("expected_tasks", -1)):
        errors.append(
            f"campaign expected_tasks={matrix.get('expected_tasks')}, experiment sum={total_expected}"
        )
    if errors:
        return errors
    try:
        tasks = expand_tasks(document, output_root=Path("/tmp/dt_ldp_far_validation"))
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]
    if len(tasks) != int(matrix["expected_tasks"]):
        errors.append(
            f"campaign expands to {len(tasks)}, expected {matrix['expected_tasks']}"
        )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("expanded campaign contains duplicate task IDs")
    for task in tasks:
        errors.extend(f"{task.task_id}: {issue}" for issue in _task_issues(task))
    return errors


def filter_tasks(
    tasks: list[Task],
    *,
    experiments: set[str] | None = None,
    methods: set[str] | None = None,
    scenarios: set[str] | None = None,
    tilts: set[str] | None = None,
    geometries: set[str] | None = None,
) -> list[Task]:
    return [
        task
        for task in tasks
        if (not experiments or task.experiment_id in experiments)
        and (not methods or task.method_id in methods)
        and (not scenarios or task.scenario_id in scenarios)
        and (not tilts or task.tilt_id in tilts)
        and (not geometries or task.geometry_id in geometries)
    ]


def write_config(task: Task, *, device: str, data_root: Path) -> Path:
    config = copy.deepcopy(task.config)
    config["device"] = device
    config["data"]["data_root"] = str(data_root)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    task.resolved_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return task.resolved_config_path


def task_command(
    task: Task, *, python_bin: str, device: str, data_root: Path
) -> list[str]:
    return [
        python_bin,
        str(ROOT / "run_experiment.py"),
        "--config",
        str(task.resolved_config_path),
        "--algo",
        str(task.config["training"]["algorithm"]),
        "--device",
        device,
        "--seed",
        str(task.training_seed),
        "--data-root",
        str(data_root),
        "--output",
        str(task.output_dir),
    ]


def _metrics_path(task: Task) -> Path | None:
    paths = sorted(task.output_dir.glob("**/metrics.json"))
    return paths[-1] if paths else None


def is_complete(task: Task) -> bool:
    metrics = _metrics_path(task)
    if metrics is None:
        return False
    try:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return len(payload.get("rounds", [])) == int(task.config["training"]["num_rounds"])


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def write_manifest(
    task: Task,
    document: dict[str, Any],
    command: list[str],
    *,
    status: str,
    started: float,
    finished: float | None = None,
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "task_index": task.index,
        "task_id": task.task_id,
        "matrix_path": str(document["matrix_path"]),
        "matrix_sha256": _sha256(document["matrix_path"]),
        "common_path": str(document["common_path"]),
        "common_sha256": _sha256(document["common_path"]),
        "command": command,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_unix": started,
        "finished_unix": finished,
    }
    path = task.output_dir / "dt_ldp_far_task_manifest.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--experiment")
    parser.add_argument("--method")
    parser.add_argument("--scenario")
    parser.add_argument("--tilt")
    parser.add_argument("--geometry")
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--pilot-rounds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    document = load_protocol(args.matrix)
    errors = validate_protocol(document)
    if errors:
        raise SystemExit("Invalid DT-LDP-FAR protocol:\n- " + "\n- ".join(errors))
    all_tasks = expand_tasks(
        document,
        output_root=args.output_root.resolve(),
        pilot_rounds=args.pilot_rounds,
    )
    if len(all_tasks) > args.max_tasks:
        raise SystemExit(
            f"Safety limit: matrix expands to {len(all_tasks)} tasks, above --max-tasks={args.max_tasks}"
        )
    tasks = filter_tasks(
        all_tasks,
        experiments=_parse_csv(args.experiment),
        methods=_parse_csv(args.method),
        scenarios=_parse_csv(args.scenario),
        tilts=_parse_csv(args.tilt),
        geometries=_parse_csv(args.geometry),
    )
    if not tasks:
        raise SystemExit("No task matches the requested filters")
    if args.validate:
        print(
            f"VALID campaign={document['matrix']['campaign_id']} tasks={len(all_tasks)}"
        )
        print(f"selected_tasks={len(tasks)} scope={document['matrix'].get('scope')}")
        return 0
    if args.list:
        for task in tasks:
            print(f"{task.index:05d}\t{task.task_id}\t{task.output_dir}")
        return 0
    # The job index is the stable index in the unfiltered matrix.  This avoids
    # silently remapping SLURM array indices when a display filter is used.
    if args.job_index is None:
        raise SystemExit("--job-index is required for --dry-run and --run")
    matching = [task for task in tasks if task.index == args.job_index]
    if not matching:
        raise SystemExit(
            f"job-index {args.job_index} is not selected or does not exist"
        )
    task = matching[0]
    write_config(task, device=args.device, data_root=args.data_root.resolve())
    command = task_command(
        task,
        python_bin=args.python_bin,
        device=args.device,
        data_root=args.data_root.resolve(),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "task_index": task.index,
                    "task_id": task.task_id,
                    "resolved_config": str(task.resolved_config_path),
                    "command": command,
                },
                indent=2,
            )
        )
        return 0

    started = time.time()
    if args.resume and is_complete(task):
        status = "complete_reused"
    else:
        write_manifest(task, document, command, status="running", started=started)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            write_manifest(
                task,
                document,
                command,
                status="failed",
                started=started,
                finished=time.time(),
            )
            return int(completed.returncode)
        if not is_complete(task):
            write_manifest(
                task,
                document,
                command,
                status="incomplete",
                started=started,
                finished=time.time(),
            )
            raise RuntimeError(
                "run exited successfully but did not produce all declared rounds"
            )
        status = "complete"
    write_manifest(
        task,
        document,
        command,
        status=status,
        started=started,
        finished=time.time(),
    )
    print(f"DONE {task.task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
