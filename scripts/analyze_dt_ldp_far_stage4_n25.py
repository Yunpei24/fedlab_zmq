#!/usr/bin/env python3
"""Build reproducible comparison tables for the DT-LDP-FAR stage-4 campaign.

The script deliberately keeps run-level observations separate from paired
contrasts.  Current-round and delayed variants are paired only when their
reference, threat, geometry, partition seed, training seed and public privacy
configuration coincide.  The same rule is used for the active/inactive server
clipping ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

plt.switch_backend("Agg")


CAMPAIGNS = {
    "screen": "dt_ldp_far_decisive_stage4_mechanistic_transfer_screen_n25_v1",
    "server_clip": "dt_ldp_far_decisive_stage4_server_clip_ablation_n25_v1",
    "current_delay_refs": "dt_ldp_far_decisive_stage4_current_delay_references_n25_v1",
}

EXPECTED = {"screen": 26, "server_clip": 24, "current_delay_refs": 72}


def _last_fairness_round(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in rounds if r.get("client_accuracy_mean") is not None]
    if not candidates:
        raise ValueError("No round contains client-level fairness metrics")
    return max(candidates, key=lambda r: int(r["round_num"]))


def _median(rounds: Iterable[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in rounds:
        value = None
        for key in keys:
            if row.get(key) is not None:
                value = row[key]
                break
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.median(values)) if values else None


def _p90(rounds: Iterable[dict[str, Any]], *keys: str) -> float | None:
    values: list[float] = []
    for row in rounds:
        value = None
        for key in keys:
            if row.get(key) is not None:
                value = row[key]
                break
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.quantile(values, 0.9)) if values else None


def _resolved_config(metrics_path: Path) -> dict[str, Any]:
    candidate = metrics_path.parent.parent / "resolved_config.yaml"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing resolved config for {metrics_path}")
    return yaml.safe_load(candidate.read_text(encoding="utf-8"))


def _threat_label(config: dict[str, Any]) -> str:
    attack = config.get("attack") or {}
    if not attack.get("enabled", False):
        return "None"
    name = str(attack.get("name", "unknown")).lower()
    scale = float(attack.get("scale", 1.0))
    if name in {"bit_flip", "sign_flip", "bf"}:
        return f"Bit-Flip x{scale:g}"
    return name.upper()


def _reference_label(config: dict[str, Any]) -> str:
    name = str(config.get("dt_reference", config.get("robust_reference", "unknown")))
    return {
        "centered_clipping": "F_CC",
        "fcc": "F_CC",
        "regularized_huber": "Huber",
        "huber": "Huber",
        "rfa": "RFA",
    }.get(name.lower(), name)


def _method_label(algorithm: str) -> str:
    if algorithm == "dt_ldp_far":
        return "Delayed"
    if algorithm == "dpfar":
        return "Current"
    return algorithm


def load_campaign(root: Path, block: str) -> pd.DataFrame:
    metrics_paths = sorted(root.glob("**/metrics.json"))
    if len(metrics_paths) != EXPECTED[block]:
        raise RuntimeError(
            f"{block}: expected {EXPECTED[block]} metrics, found {len(metrics_paths)}"
        )

    rows: list[dict[str, Any]] = []
    for path in metrics_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rounds = payload["rounds"]
        final = _last_fairness_round(rounds)
        summary = payload["summary"]
        resolved = _resolved_config(path)
        training = resolved["training"]
        config = training["algo_config"]
        algorithm = str(training["algorithm"])
        reproduction = resolved.get("reproduction", {})
        rows.append(
            {
                "block": block,
                "path": str(path),
                "algorithm": algorithm,
                "method": _method_label(algorithm),
                "reference": _reference_label(config),
                "threat": _threat_label(config),
                "geometry": str(config.get("experiment_geometry_profile", "")),
                "tilt": float(config.get("tilt_tau", config.get("far_alpha", 0.0))),
                "kappa_w": float(config.get("kappa_w", np.nan)),
                "seed": int(summary["training_seed"]),
                "partition_seed": int(summary["partition_seed"]),
                "randomness_pair_key": str(reproduction.get("randomness_pair_key", "")),
                "certificate_claimed": bool(
                    reproduction.get("tilt_influence_certificate_claimed", False)
                ),
                "server_clip_norm": float(
                    config.get(
                        "server_clip_norm", config.get("far_server_clip_norm", np.nan)
                    )
                ),
                "test_acc_pp": float(final["test_accuracy"]) * 100.0,
                "client_acc_pp": float(final["client_accuracy_mean"]) * 100.0,
                "variance_pp2": float(final["client_accuracy_variance_pct2"]),
                "worst20_pp": float(final["worst20_accuracy_pct"]),
                "gap_pp": float(final["best20_worst20_gap_pct"]),
                "test_loss": float(final["test_loss"]),
                "epsilon": float(final.get("privacy_epsilon_max", np.nan)),
                "delta": float(final.get("privacy_delta", np.nan)),
                "noise_multiplier": float(
                    final.get("privacy_model_noise_multiplier_mean", np.nan)
                ),
                "score_span_median": _median(
                    rounds, "dtldp_current_score_span", "far_score_span"
                ),
                "logit_span_median": _median(
                    rounds, "dtldp_current_logit_span", "far_logit_range"
                ),
                "score_saturation_p90": _p90(
                    rounds,
                    "dtldp_current_score_saturation_rate",
                    "far_score_saturation_rate",
                ),
                "max_weight_median": _median(rounds, "max_client_weight"),
                "concentration_median": _median(
                    rounds,
                    "dtldp_noise_amplification_vs_uniform",
                    "far_noise_amplification_vs_uniform",
                ),
                "weight_entropy_median": _median(rounds, "weight_entropy"),
                "byzantine_weight_mass_median": _median(
                    rounds, "byzantine_weight_mass_oracle"
                ),
                "server_clip_rate_median": _median(
                    rounds, "dtldp_server_clip_rate", "far_server_clip_rate"
                ),
                "server_clip_honest_median": _median(
                    rounds, "dtldp_server_clip_rate_honest_oracle"
                ),
                "server_clip_byzantine_median": _median(
                    rounds, "dtldp_server_clip_rate_byzantine_oracle"
                ),
                "byzantine_contribution_norm_median": _median(
                    rounds, "dtldp_byzantine_weighted_contribution_norm_oracle"
                ),
                "reference_drift_median": _median(rounds, "dtldp_reference_drift"),
                "reference_honest_error_median": _median(
                    rounds, "dtldp_reference_honest_center_error_oracle"
                ),
                "score_drift_median": _median(rounds, "dtldp_score_drift_linf"),
                "staleness_norm_median": _median(
                    rounds, "dtldp_staleness_aggregate_norm"
                ),
                "current_weight_noise_corr_median": _median(
                    rounds, "far_weight_dp_noise_corr_oracle"
                ),
                "delayed_weight_noise_corr_median": _median(
                    rounds, "dtldp_delayed_weight_noise_corr_oracle"
                ),
                "weight_cap_respected_all": all(
                    bool(
                        r.get(
                            "dtldp_weight_cap_respected",
                            r.get("far_weight_cap_respected", True),
                        )
                    )
                    for r in rounds
                ),
                "minimum_survivors": min(int(r["num_survivors"]) for r in rounds),
                "maximum_pre_training_dropouts": max(
                    int(r["num_pre_training_dropouts"]) for r in rounds
                ),
                "full_participation_all_rounds": all(
                    int(r["num_selected"]) == 25 for r in rounds
                ),
                "num_rounds": len(rounds),
            }
        )
    return pd.DataFrame(rows)


def paired_difference(
    frame: pd.DataFrame,
    *,
    arm_col: str,
    left: str,
    right: str,
    pair_cols: list[str],
) -> pd.DataFrame:
    index_cols = pair_cols
    measures = [
        "test_acc_pp",
        "client_acc_pp",
        "variance_pp2",
        "worst20_pp",
        "gap_pp",
        "test_loss",
    ]
    left_frame = frame[frame[arm_col] == left].set_index(index_cols)
    right_frame = frame[frame[arm_col] == right].set_index(index_cols)
    if set(left_frame.index) != set(right_frame.index):
        raise RuntimeError(f"Unpaired {arm_col}: {left} versus {right}")
    rows: list[dict[str, Any]] = []
    for index in sorted(set(left_frame.index)):
        lhs = left_frame.loc[index]
        rhs = right_frame.loc[index]
        if isinstance(lhs, pd.DataFrame) or isinstance(rhs, pd.DataFrame):
            raise RuntimeError(f"Non-unique paired index: {index}")
        row = dict(zip(index_cols, index, strict=True))
        row.update(
            {
                "left_arm": left,
                "right_arm": right,
                "pair_key_matches": lhs["randomness_pair_key"]
                == rhs["randomness_pair_key"],
            }
        )
        for measure in measures:
            row[f"delta_{measure}"] = float(rhs[measure] - lhs[measure])
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_summary(
    frame: pd.DataFrame, group_cols: list[str], measures: list[str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key, strict=True))
        record["n"] = len(group)
        for measure in measures:
            values = group[measure].astype(float)
            record[f"{measure}_mean"] = float(values.mean())
            record[f"{measure}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/dt_ldp_far/decisive"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/analysis/dt_ldp_far_n25_stage4"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        name: load_campaign(args.root / campaign, name)
        for name, campaign in CAMPAIGNS.items()
    }
    all_runs = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in frames.values()],
        ignore_index=True,
    )
    all_runs.to_csv(args.output_dir / "run_level.csv", index=False)

    refs = frames["current_delay_refs"]
    delay_pairs = paired_difference(
        refs,
        arm_col="method",
        left="Current",
        right="Delayed",
        pair_cols=["reference", "threat", "seed", "partition_seed", "geometry", "tilt"],
    )
    delay_pairs.to_csv(args.output_dir / "current_vs_delayed_paired.csv", index=False)

    clip = frames["server_clip"].copy()
    clip["clip_arm"] = np.where(clip["server_clip_norm"] < 1.0, "Active", "Inactive")
    clip_pairs = paired_difference(
        clip,
        arm_col="clip_arm",
        left="Inactive",
        right="Active",
        pair_cols=["reference", "threat", "seed", "partition_seed", "tilt"],
    )
    clip_pairs.to_csv(args.output_dir / "server_clip_paired.csv", index=False)

    final_measures = [
        "test_acc_pp",
        "client_acc_pp",
        "variance_pp2",
        "worst20_pp",
        "gap_pp",
    ]
    diagnostic_measures = [
        "score_span_median",
        "logit_span_median",
        "max_weight_median",
        "concentration_median",
        "byzantine_weight_mass_median",
        "server_clip_rate_median",
        "server_clip_honest_median",
        "server_clip_byzantine_median",
        "byzantine_contribution_norm_median",
        "reference_drift_median",
        "reference_honest_error_median",
        "score_drift_median",
        "staleness_norm_median",
    ]
    grouped_summary(
        refs, ["method", "threat", "reference"], final_measures + diagnostic_measures
    ).to_csv(args.output_dir / "method_threat_reference_summary.csv", index=False)
    grouped_summary(
        delay_pairs,
        ["threat", "reference"],
        [f"delta_{m}" for m in final_measures],
    ).to_csv(args.output_dir / "delay_effect_summary.csv", index=False)
    grouped_summary(
        clip.assign(
            clip_arm=np.where(clip["server_clip_norm"] < 1.0, "Active", "Inactive")
        ),
        ["clip_arm", "threat"],
        final_measures + diagnostic_measures,
    ).to_csv(args.output_dir / "server_clip_summary.csv", index=False)
    grouped_summary(
        clip_pairs,
        ["threat"],
        [f"delta_{m}" for m in final_measures],
    ).to_csv(args.output_dir / "server_clip_effect_summary.csv", index=False)

    screen_gate_path = Path(
        "output/analysis/dt_ldp_far_n25_mechanistic_transfer_v1/transfer_gate.csv"
    )
    screen_gate = pd.read_csv(screen_gate_path)
    screen_gate.to_csv(args.output_dir / "mechanistic_transfer_gate.csv", index=False)

    # Compact, paper-audit-friendly figures.  Error bars are sample standard
    # deviations over the three paired seeds, not confidence intervals.
    delay_summary = grouped_summary(
        delay_pairs,
        ["threat", "reference"],
        ["delta_test_acc_pp"],
    )
    delay_summary["label"] = (
        delay_summary["threat"].astype(str)
        + "\n"
        + delay_summary["reference"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(delay_summary))
    ax.errorbar(
        x,
        delay_summary["delta_test_acc_pp_mean"],
        yerr=delay_summary["delta_test_acc_pp_sd"],
        fmt="o",
        capsize=4,
        color="#0f766e",
    )
    ax.axhline(0.0, color="#475569", linewidth=1)
    ax.set_xticks(x, delay_summary["label"], rotation=45, ha="right")
    ax.set_ylabel("Accuracy test : retardé − courant (pp)")
    ax.set_title("DT-LDP-FAR n=25 : effet apparié du retard d’un tour")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "delay_effect_test_accuracy.png", dpi=180)
    plt.close(fig)

    clip_summary = grouped_summary(
        clip,
        ["clip_arm", "threat"],
        ["test_acc_pp"],
    )
    threats = ["None", "Bit-Flip x10", "IPM", "ALIE"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(threats))
    width = 0.36
    for offset, arm, color in [
        (-width / 2, "Inactive", "#94a3b8"),
        (width / 2, "Active", "#ea580c"),
    ]:
        rows = clip_summary.set_index(["clip_arm", "threat"])
        means = [rows.loc[(arm, threat), "test_acc_pp_mean"] for threat in threats]
        sds = [rows.loc[(arm, threat), "test_acc_pp_sd"] for threat in threats]
        ax.bar(
            positions + offset,
            means,
            width,
            yerr=sds,
            capsize=4,
            label=arm,
            color=color,
        )
    ax.set_xticks(positions, threats)
    ax.set_ylabel("Accuracy test (%)")
    ax.set_title("Ablation du clipping serveur (U=0,42 contre U=10)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "server_clip_test_accuracy.png", dpi=180)
    plt.close(fig)

    delayed = refs[refs["method"] == "Delayed"]
    reference_summary = grouped_summary(
        delayed,
        ["threat", "reference"],
        ["test_acc_pp"],
    ).set_index(["threat", "reference"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    colors = {"F_CC": "#0f766e", "Huber": "#ea580c", "RFA": "#7c3aed"}
    for index, reference in enumerate(["F_CC", "Huber", "RFA"]):
        means = [
            reference_summary.loc[(threat, reference), "test_acc_pp_mean"]
            for threat in threats
        ]
        sds = [
            reference_summary.loc[(threat, reference), "test_acc_pp_sd"]
            for threat in threats
        ]
        ax.bar(
            positions + (index - 1) * width,
            means,
            width,
            yerr=sds,
            capsize=3,
            label=reference,
            color=colors[reference],
        )
    ax.set_xticks(positions, threats)
    ax.set_ylabel("Accuracy test (%)")
    ax.set_title("DT-LDP-FAR retardé : ablation de la référence")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "reference_ablation_test_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    valid_corr = screen_gate["current_weight_noise_corr_median"].notna()
    plotted_gate = screen_gate[valid_corr]
    scatter = ax.scatter(
        plotted_gate["current_logit_span_median"],
        plotted_gate["current_weight_noise_corr_median"],
        s=45 + 220 * (plotted_gate["current_concentration_median"] - 1.0),
        c=plotted_gate["alpha"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(1.0, color="#ea580c", linestyle="--", label="seuil R = 1")
    ax.axhline(0.1, color="#dc2626", linestyle="--", label="seuil corr. = 0,1")
    ax.set_xlabel("Logit-span médian R")
    ax.set_ylabel("Corrélation médiane poids courant–bruit")
    ax.set_title("Gate de transfert mécanistique end-to-end")
    missing_count = int((~valid_corr).sum())
    if missing_count:
        ax.text(
            0.02,
            0.03,
            f"{missing_count} cellules saturées : corrélation non définie",
            transform=ax.transAxes,
            fontsize=9,
            color="#475569",
        )
    ax.legend(loc="lower right")
    fig.colorbar(scatter, ax=ax, label="alpha")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "mechanistic_transfer_gate.png", dpi=180)
    plt.close(fig)

    integrity = {
        "schema_version": 1,
        "run_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "total_runs": int(len(all_runs)),
        "delay_pairs": int(len(delay_pairs)),
        "delay_pair_keys_all_match": bool(delay_pairs["pair_key_matches"].all()),
        "server_clip_pairs": int(len(clip_pairs)),
        "server_clip_pair_keys_all_match": bool(clip_pairs["pair_key_matches"].all()),
        "weight_cap_respected_all_publishable_blocks": bool(
            all_runs.loc[
                all_runs["block"].isin(["server_clip", "current_delay_refs"]),
                "weight_cap_respected_all",
            ].all()
        ),
        "uncertified_oracle_screen_runs": int(
            (
                (all_runs["block"] == "screen")
                & (~all_runs["certificate_claimed"].fillna(False))
            ).sum()
        ),
        "epsilon_range": [
            float(all_runs["epsilon"].min()),
            float(all_runs["epsilon"].max()),
        ],
        "delta_values": sorted(float(v) for v in all_runs["delta"].dropna().unique()),
        "noise_multiplier_values": sorted(
            float(v) for v in all_runs["noise_multiplier"].dropna().unique()
        ),
        "mechanistic_pairs": int(len(screen_gate)),
        "mechanistic_promoted": int(screen_gate["promoted"].sum()),
        "mechanistic_pair_keys_all_match": bool(screen_gate["pair_key_matches"].all()),
        "mechanistic_noise_draws_all_match": bool(
            screen_gate["noise_draws_match"].all()
        ),
        "all_runs_have_20_rounds": bool((all_runs["num_rounds"] == 20).all()),
        "all_rounds_have_25_survivors": bool(
            (all_runs["minimum_survivors"] == 25).all()
        ),
        "no_pre_training_dropouts": bool(
            (all_runs["maximum_pre_training_dropouts"] == 0).all()
        ),
        "full_participation_all_rounds": bool(
            all_runs["full_participation_all_rounds"].all()
        ),
    }
    (args.output_dir / "integrity_summary.json").write_text(
        json.dumps(integrity, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
