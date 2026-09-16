#!/usr/bin/env python3
"""Build the reproducible LDP-gradient-FAR positioning analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "LDP_Gradient_FAR_Positioning_Analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# LDP-Gradient-FAR — analyse reproductible des campagnes

Ce notebook découvre automatiquement les campagnes sous
`results/ldp_gradient_far`, notamment `positioning_v1`, `positioning_v2` et
`positioning_v3`. Il ne sélectionne ni ne promeut aucune configuration.

Principes de lecture :

- les fichiers incomplets ou incohérents sont séparés des runs valides ;
- les trajectoires par round et les métriques finales sont deux tables distinctes ;
- aucune moyenne ne mélange des campagnes, phases ou horizons différents ;
- une cellule à une seule seed est marquée **exploratoire / non identifiable** ;
- les métriques suffixées `_oracle` servent au diagnostic du simulateur et ne
  sont jamais présentées comme une information disponible à l'algorithme ;
- lorsque `suppress_private_client_diagnostics=true`, `train_loss` et
  `avg_local_loss` sont des placeholders nuls : ni la vraie accuracy ni la
  vraie loss d'entraînement ne sont disponibles. Le notebook utilise alors
  `test_accuracy`, `test_loss`, `client_accuracy_mean` et `client_loss_mean`
  calculées sur les évaluations held-out.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import itertools
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fedlab_zmq_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import Markdown, display

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 80)
plt.style.use("seaborn-v0_8-whitegrid")


def locate_repo(start: Path | None = None) -> Path:
    candidates = [Path(start or Path.cwd()).resolve(), Path.cwd().resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "results" / "ldp_gradient_far").exists():
                return parent
    raise FileNotFoundError("Impossible de localiser results/ldp_gradient_far")


ROOT = locate_repo()
RESULTS_ROOT = ROOT / "results" / "ldp_gradient_far"
EXPORT_ROOT = ROOT / "output" / "analysis" / "ldp_gradient_far_positioning_notebook"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Repository : {ROOT}")
print(f"Résultats  : {RESULTS_ROOT}")
"""
    ),
    md(
        r"""
## 1. Découverte, validation et mise en forme tidy

Un run est valide si le JSON est lisible, si l'algorithme est
`ldp_gradient_far`, si `summary.num_rounds` est positif, si le nombre de lignes
de round correspond à cet horizon et si les indices sont exactement
`1, …, T`. La validation vérifie aussi la présence d'une seed, du nombre de
clients et de la configuration.
"""
    ),
    code(
        r"""
REFERENCE_LABELS = {
    "coordinate_median": "CM",
    "trimmed_mean": "trMean",
    "rfa": "RFA",
    "centered_clipping": "F_CC",
    "noise_aware_centered_clipping": "F_NA-CC",
}

ROUND_METRICS = {
    "train_accuracy_pct": ("train_accuracy", 100.0),
    "train_loss": ("train_loss", 1.0),
    "client_loss_heldout": ("client_loss_mean", 1.0),
    "test_loss": ("test_loss", 1.0),
    "test_accuracy_pct": ("test_accuracy", 100.0),
    "client_accuracy_pct": ("client_accuracy_mean", 100.0),
    "variance_pp2": ("client_accuracy_variance_pct2", 1.0),
    "worst20_pct": ("worst20_accuracy_pct", 1.0),
    "gap_pp": ("best20_worst20_gap_pct", 1.0),
    "balanced_accuracy_pct": ("mean_client_balanced_accuracy_pct", 1.0),
    "balanced_variance_pp2": ("client_balanced_accuracy_variance_pct2", 1.0),
    "balanced_worst20_pct": ("worst20_balanced_accuracy_pct", 1.0),
    "balanced_gap_pp": ("best_worst_balanced_accuracy_gap_pct", 1.0),
    "balanced_performance_fairness": ("balanced_performance_fairness", 1.0),
    "max_weight": ("max_client_weight", 1.0),
    "min_weight": ("min_client_weight", 1.0),
    "weight_concentration": ("far_noise_amplification_vs_uniform", 1.0),
    "weight_entropy": ("weight_entropy", 1.0),
    "effective_clients": ("effective_num_clients", 1.0),
    "score_span": ("far_score_span", 1.0),
    "logit_range": ("far_logit_range", 1.0),
    "byzantine_mass_oracle": ("byzantine_weight_mass_oracle", 1.0),
    "reference_error_oracle": ("far_reference_honest_center_error_oracle", 1.0),
    "weight_noise_corr_oracle": ("far_weight_effective_noise_corr_oracle", 1.0),
    "score_clean_corr_oracle": ("far_honest_noisy_clean_score_corr_oracle", 1.0),
    "top_tail_recall_oracle": ("far_honest_clean_top_tail_recall_oracle", 1.0),
    "tilting_bias_norm_oracle": ("far_honest_clean_tilting_bias_norm_oracle", 1.0),
    "fixed_weight_dp_noise_norm_oracle": ("far_honest_fixed_weight_dp_noise_norm_oracle", 1.0),
    "byzantine_displacement_norm_oracle": ("far_byzantine_displacement_norm_oracle", 1.0),
    "aggregate_error_oracle": ("far_aggregate_error_to_clean_honest_center_norm_oracle", 1.0),
    "decomposition_residual_oracle": ("far_error_decomposition_residual_norm_oracle", 1.0),
    "epsilon": ("privacy_epsilon_max", 1.0),
    "delta": ("privacy_delta", 1.0),
    "sigma_min": ("privacy_model_noise_multiplier_min", 1.0),
    "sigma_mean": ("privacy_model_noise_multiplier_mean", 1.0),
    "sigma_max": ("privacy_model_noise_multiplier_max", 1.0),
    "local_clip_rate": ("privacy_clip_rate_mean", 1.0),
    "server_clip_rate": ("far_server_clip_rate", 1.0),
}


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def metric_value(row, key, scale=1.0):
    value = finite(row.get(key))
    return value * scale if np.isfinite(value) else np.nan


def infer_location(path: Path):
    rel = path.relative_to(RESULTS_ROOT)
    parts = rel.parts
    campaign = parts[0]
    if campaign.startswith("positioning_v") and len(parts) >= 4:
        phase, task_id = parts[1], parts[2]
    else:
        phase = "unphased"
        task_id = "/".join(parts[1:-1]) or path.parent.name
    return campaign, phase, task_id, str(rel)


def noise_profile(config):
    scales = config.get("privacy_noise_multiplier_scale_by_client")
    if not scales:
        return "homogeneous", 1.0, 1.0
    values = [finite(x) for x in scales]
    if not values or not np.isfinite(values).all():
        return "unknown", np.nan, np.nan
    lo, hi = min(values), max(values)
    return ("homogeneous" if np.isclose(lo, hi) else "heteroscedastic", lo, hi)


def noise_assignment(task_id, config):
    scales = config.get("privacy_noise_multiplier_scale_by_client")
    if not scales:
        return "homogeneous"
    level = "strong" if "strong" in task_id else "mild" if "mild" in task_id else "custom"
    order = "reverse" if "reverse" in task_id else "identity" if "identity" in task_id else "unspecified"
    return f"{level}_{order}"


def validate_payload(path, payload):
    errors = []
    if not isinstance(payload, dict):
        return ["top-level JSON is not an object"]
    if payload.get("algorithm") != "ldp_gradient_far":
        errors.append("algorithm != ldp_gradient_far")
    summary, rounds, config = payload.get("summary"), payload.get("rounds"), payload.get("config")
    if not isinstance(summary, dict): errors.append("summary missing")
    if not isinstance(rounds, list): errors.append("rounds missing")
    if not isinstance(config, dict): errors.append("config missing")
    if errors:
        return errors
    try: expected = int(summary.get("num_rounds", -1))
    except (TypeError, ValueError): expected = -1
    if expected <= 0: errors.append("invalid summary.num_rounds")
    if len(rounds) != expected: errors.append(f"round count {len(rounds)} != {expected}")
    observed = [r.get("round_num") for r in rounds]
    if observed != list(range(1, expected + 1)): errors.append("round indices are not 1..T")
    if summary.get("seed") is None and summary.get("training_seed") is None: errors.append("seed missing")
    if summary.get("num_clients") is None: errors.append("num_clients missing")
    if config.get("device") != "mps": errors.append("device != mps")
    if rounds:
        final = rounds[-1]
        for key in (
            "test_accuracy", "test_loss", "client_accuracy_variance_pct2",
            "worst20_accuracy_pct", "best20_worst20_gap_pct",
        ):
            if not np.isfinite(finite(final.get(key))):
                errors.append(f"non-finite final {key}")
    if bool(config.get("enable_dp", False)) and rounds:
        if config.get("sampling_scheme") != "fixed_without_replacement":
            errors.append("DP sampling_scheme != fixed_without_replacement")
        if config.get("privacy_adjacency") != "replace_one":
            errors.append("DP privacy_adjacency != replace_one")
        target_epsilon = finite(config.get("target_epsilon"))
        realised_epsilon = finite(rounds[-1].get("privacy_epsilon_max"))
        if not (np.isfinite(target_epsilon) and np.isfinite(realised_epsilon)):
            errors.append("DP epsilon missing")
        elif abs(realised_epsilon - target_epsilon) > 0.05:
            errors.append("realised epsilon differs from target by > 0.05")
        realised_delta = finite(rounds[-1].get("privacy_delta"))
        if not np.isfinite(realised_delta) or abs(realised_delta - 1e-5) > 1e-12:
            errors.append("DP delta != 1e-5")
    return errors


def read_all_metrics():
    inventory, tidy = [], []
    for path in sorted(RESULTS_ROOT.glob("**/metrics.json")):
        campaign, phase, task_id, relative = infer_location(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_payload(path, payload)
        except Exception as exc:
            payload, errors = None, [f"JSON unreadable: {exc}"]
        inventory.append({
            "campaign": campaign, "phase": phase, "task_id": task_id,
            "metrics_path": str(path), "relative_path": relative,
            "status": "valid" if not errors else "invalid", "errors": "; ".join(errors),
        })
        if errors:
            continue
        summary, config, rounds = payload["summary"], payload["config"], payload["rounds"]
        profile, scale_min, scale_max = noise_profile(config)
        attack_cfg = config.get("attack") or {}
        n = int(summary["num_clients"])
        byz = int(attack_cfg.get("num_byzantine", 0) or 0)
        seed = int(summary.get("training_seed", summary.get("seed")))
        metadata = {
            "campaign": campaign, "phase": phase, "task_id": task_id,
            "run_uid": relative.removesuffix("/metrics.json"), "metrics_path": str(path),
            "seed": seed, "partition_seed": summary.get("partition_seed"),
            "dataset": summary.get("dataset", payload.get("dataset")),
            "model": summary.get("model"), "partition": summary.get("partition"),
            "n_clients": n, "horizon": int(summary["num_rounds"]),
            "alpha": finite(config.get("far_alpha")),
            "C_local": finite(config.get("clip_norm")),
            "U_server": finite(config.get("far_server_clip_norm")),
            "reference_raw": config.get("robust_reference"),
            "reference": REFERENCE_LABELS.get(config.get("robust_reference"), config.get("robust_reference")),
            "dp_enabled": bool(config.get("enable_dp", False)),
            "target_epsilon": finite(config.get("target_epsilon")),
            "target_delta": finite(config.get("delta")),
            "sampling_scheme": config.get("sampling_scheme"),
            "adjacency": config.get("privacy_adjacency"),
            "update_mode": config.get("far_update_mode"),
            "private_train_metrics_suppressed": bool(config.get("suppress_private_client_diagnostics", False)),
            "noise_profile": profile, "noise_assignment": noise_assignment(task_id, config),
            "noise_signature": tuple(config.get("privacy_noise_multiplier_scale_by_client") or (1.0,)),
            "noise_scale_min": scale_min, "noise_scale_max": scale_max,
            "attack": attack_cfg.get("name", "none"), "num_byzantine": byz,
            "byzantine_fraction": byz / n if n else np.nan,
            "attack_client_ids": tuple(sorted(int(x) for x in (
                attack_cfg.get("client_ids") or range(byz)
            ))),
        }
        metadata["strict_stratum"] = " | ".join(map(str, [
            campaign, phase, metadata["dataset"], metadata["model"], n,
            metadata["horizon"], metadata["partition"], metadata["sampling_scheme"],
            metadata["adjacency"], metadata["update_mode"], profile,
            metadata["noise_assignment"],
        ]))
        for row in rounds:
            output = dict(metadata)
            output["round"] = int(row["round_num"])
            for out_name, (source, scale) in ROUND_METRICS.items():
                if out_name in {"train_accuracy_pct", "train_loss"} and metadata["private_train_metrics_suppressed"]:
                    output[out_name] = np.nan
                else:
                    output[out_name] = metric_value(row, source, scale)
            output["evaluated_client_ids_oracle"] = row.get("evaluated_client_ids_oracle")
            output["client_accuracy_values_oracle"] = row.get("client_accuracy_values_oracle")
            output["client_balanced_accuracy_values_oracle"] = row.get("client_balanced_accuracy_values_oracle")
            tidy.append(output)
    tidy_df = pd.DataFrame(tidy)
    if not tidy_df.empty:
        # For an attacked run, fairness must exclude Byzantine clients.  For its
        # no-attack control, exclude the same client IDs found in the matched
        # attacked arm; otherwise the comparison changes the evaluated cohort.
        pairing = [
            "campaign", "phase", "seed", "round", "dataset", "model", "horizon",
            "n_clients", "partition", "dp_enabled", "target_epsilon", "noise_profile",
            "noise_assignment",
            "C_local", "U_server", "alpha", "reference",
        ]
        attacked = tidy_df[tidy_df.attack != "none"]
        matched_ids = {}
        for keys, group in attacked.groupby(pairing, dropna=False):
            matched_ids[keys] = tuple(sorted({
                client_id for ids in group.attack_client_ids for client_id in ids
            }))

        def honest_fairness(row):
            excluded = row.attack_client_ids
            if row.attack == "none":
                excluded = matched_ids.get(tuple(row[key] for key in pairing), ())
            ids = row.evaluated_client_ids_oracle
            acc = row.client_accuracy_values_oracle
            bal = row.client_balanced_accuracy_values_oracle
            if not isinstance(ids, list) or not isinstance(acc, list) or len(ids) != len(acc):
                return pd.Series([np.nan] * 7)
            keep = [idx for idx, client_id in enumerate(ids) if int(client_id) not in excluded]
            values = np.asarray([acc[idx] for idx in keep], dtype=float) * 100.0
            if not len(values):
                return pd.Series([np.nan] * 7)
            k = max(1, int(np.ceil(0.2 * len(values))))
            ordered = np.sort(values)
            balanced = np.nan
            if isinstance(bal, list) and len(bal) == len(ids):
                bal_values = np.asarray([bal[idx] for idx in keep], dtype=float) * 100.0
                balanced = float(np.mean(bal_values))
            return pd.Series([
                float(np.mean(values)), float(np.var(values)), float(np.mean(ordered[:k])),
                float(np.mean(ordered[-k:]) - np.mean(ordered[:k])),
                balanced, len(values), ",".join(map(str, excluded)),
            ])

        tidy_df[[
            "honest_client_accuracy_pct", "honest_variance_pp2", "honest_worst20_pct",
            "honest_gap_pp", "honest_balanced_accuracy_pct", "honest_client_count",
            "honest_excluded_client_ids",
        ]] = tidy_df.apply(honest_fairness, axis=1)
    return pd.DataFrame(inventory), tidy_df


inventory_df, rounds_df = read_all_metrics()
valid_runs_df = rounds_df.drop_duplicates("run_uid") if not rounds_df.empty else pd.DataFrame()
print(f"Fichiers découverts : {len(inventory_df)}")
print(f"Runs valides        : {(inventory_df.status == 'valid').sum() if len(inventory_df) else 0}")
print(f"Runs invalides      : {(inventory_df.status == 'invalid').sum() if len(inventory_df) else 0}")
print(f"Lignes round tidy   : {len(rounds_df)}")
"""
    ),
    code(
        r"""
inventory_summary = (
    inventory_df.groupby(["campaign", "phase", "status"], dropna=False)
    .size().rename("files").reset_index()
    .sort_values(["campaign", "phase", "status"])
)
display(inventory_summary)
invalid_df = inventory_df[inventory_df.status != "valid"]
if len(invalid_df):
    display(Markdown("### Fichiers invalides ou partiels"))
    display(invalid_df[["campaign", "phase", "task_id", "errors"]])
else:
    display(Markdown("**Aucun fichier invalide parmi les fichiers présents.**"))


def expected_positioning_tasks():
    rows = []
    config_root = ROOT / "configs" / "ldp_gradient_far"
    for matrix_path in sorted(config_root.glob("positioning_v*.yaml")):
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        campaign = Path(matrix["output_root"]).name
        output_root = (matrix_path.parent / matrix["output_root"]).resolve()
        for phase in matrix.get("phases", []):
            value_sets = [axis.get("values", []) for axis in phase.get("axes", [])]
            for values in itertools.product(*value_sets):
                variant = "__".join(str(value["id"]) for value in values)
                for seed in phase.get("seeds", []):
                    run_id = f"{variant}_seed{int(seed)}"
                    metrics = sorted((output_root / phase["id"] / run_id).glob("**/metrics.json"))
                    valid = False
                    reason = "metrics absent"
                    for metrics_path in metrics:
                        try:
                            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                            errs = validate_payload(metrics_path, payload)
                        except Exception as exc:
                            errs = [str(exc)]
                        if not errs:
                            valid, reason = True, "complete"
                            break
                        reason = "; ".join(errs)
                    rows.append({
                        "campaign": campaign, "phase": phase["id"], "run_id": run_id,
                        "expected": True, "status": "complete" if valid else "missing_or_invalid",
                        "reason": reason,
                    })
    return pd.DataFrame(rows)


expected_df = expected_positioning_tasks()
if len(expected_df):
    expected_summary = (expected_df.groupby(["campaign", "phase", "status"])
                        .size().rename("runs").reset_index())
    display(Markdown("### Complétude par rapport aux matrices positioning v1/v2/v3"))
    display(expected_summary)
    missing_expected_df = expected_df[expected_df.status != "complete"]
    print(f"Tâches attendues : {len(expected_df)}; complètes : {(expected_df.status == 'complete').sum()}; "
          f"manquantes/invalides : {len(missing_expected_df)}")
    if len(missing_expected_df):
        display(missing_expected_df.head(100))
else:
    missing_expected_df = pd.DataFrame()
    display(Markdown("**Matrices positioning introuvables : complétude non identifiable.**"))
"""
    ),
    md(
        r"""
## 2. Tables finales et règle de comparabilité

`final_df` contient uniquement le dernier round de chaque run valide.
`rounds_df` conserve toutes les trajectoires. Le regroupement statistique garde
toujours `campaign`, `phase` et `horizon`; il ne mélange donc jamais v1/v2/v3,
ni 20 et 40 rounds. Les traitements (`alpha`, `C`, `n`, attaque, référence,
DP/bruit) restent des colonnes explicites.
"""
    ),
    code(
        r"""
FINAL_METRICS = [
    "train_accuracy_pct", "train_loss", "client_loss_heldout", "test_loss",
    "test_accuracy_pct", "client_accuracy_pct",
    "variance_pp2", "worst20_pct", "gap_pp", "balanced_accuracy_pct",
    "honest_client_accuracy_pct", "honest_variance_pp2", "honest_worst20_pct",
    "honest_gap_pp", "honest_balanced_accuracy_pct",
    "balanced_variance_pp2", "balanced_worst20_pct", "balanced_gap_pp",
    "balanced_performance_fairness",
    "max_weight", "weight_concentration", "weight_entropy", "effective_clients",
    "score_span", "logit_range", "byzantine_mass_oracle", "reference_error_oracle",
    "weight_noise_corr_oracle", "score_clean_corr_oracle", "top_tail_recall_oracle",
    "tilting_bias_norm_oracle", "fixed_weight_dp_noise_norm_oracle",
    "byzantine_displacement_norm_oracle", "aggregate_error_oracle",
    "decomposition_residual_oracle", "epsilon", "delta", "sigma_min", "sigma_mean",
    "sigma_max", "local_clip_rate", "server_clip_rate",
]
IDENTITY = [
    "campaign", "phase", "dataset", "model", "horizon", "n_clients",
    "partition", "sampling_scheme", "adjacency", "update_mode", "dp_enabled",
    "target_epsilon", "noise_profile", "noise_assignment", "noise_signature",
    "noise_scale_min", "noise_scale_max",
    "C_local", "U_server", "alpha", "reference", "attack", "byzantine_fraction",
]

final_df = (
    rounds_df.sort_values(["run_uid", "round"])
    .groupby("run_uid", as_index=False, dropna=False).tail(1).copy()
)


def summarize_final(frame, metrics=FINAL_METRICS):
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(IDENTITY, dropna=False)
    pieces = []
    for keys, group in grouped:
        row = dict(zip(IDENTITY, keys if isinstance(keys, tuple) else (keys,)))
        seeds = sorted(group.seed.dropna().astype(int).unique().tolist())
        row["n_seeds"] = len(seeds)
        row["seeds"] = ",".join(map(str, seeds))
        row["identifiability"] = (
            "multi-seed" if len(seeds) >= 3 else
            "limited (2 seeds)" if len(seeds) == 2 else
            "exploratory / non-identifiable (1 seed)"
        )
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_n"] = len(values)
        pieces.append(row)
    return pd.DataFrame(pieces)


summary_df = summarize_final(final_df)
display(Markdown(f"**Runs finaux valides : {len(final_df)} — cellules statistiques : {len(summary_df)}.**"))
display(summary_df[[
    "campaign", "phase", "horizon", "n_clients", "dp_enabled", "target_epsilon",
    "C_local", "alpha", "reference", "attack", "n_seeds", "identifiability",
    "test_accuracy_pct_mean", "test_accuracy_pct_sd", "worst20_pct_mean", "gap_pp_mean",
]].head(60))
"""
    ),
    code(
        r"""
# Exports réutilisables par le dashboard ou un autre notebook.
inventory_df.to_csv(EXPORT_ROOT / "inventory.csv", index=False)
rounds_df.to_csv(EXPORT_ROOT / "rounds_tidy.csv", index=False)
final_df.to_csv(EXPORT_ROOT / "final_runs.csv", index=False)
summary_df.to_csv(EXPORT_ROOT / "final_mean_sd.csv", index=False)
print("Exports :", EXPORT_ROOT)
"""
    ),
    md(
        r"""
## 3. Trajectoires : accuracy et losses held-out

Les bandes représentent moyenne ± un écart-type seulement lorsqu'au moins deux
seeds existent. Une ligne sans bande est une trajectoire exploratoire. Les
sorties ne contiennent pas de vraie `train_accuracy`; lorsque les diagnostics
privés sont supprimés, `train_loss=0` est aussi un placeholder et n'est jamais
tracé. Nous affichons `test_loss` et `client_loss_mean` held-out.
"""
    ),
    code(
        r"""
def best_trajectory_stratum(frame):
    if frame.empty:
        return None
    candidates = []
    for stratum, group in frame.groupby("strict_stratum"):
        treatment_seed_counts = group.groupby("trajectory_treatment").seed.nunique()
        candidates.append((
            int(treatment_seed_counts.max()),
            int((treatment_seed_counts >= 2).sum()),
            int(group.run_uid.nunique()),
            stratum,
        ))
    candidates.sort(reverse=True)
    return candidates[0][-1]


def plot_trajectories(frame, metrics, title, hue="reference"):
    if frame.empty:
        display(Markdown(f"**Non identifiable : aucune donnée pour {title}.**"))
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for label, group in frame.groupby(hue, dropna=False):
            stats = group.groupby("round")[metric].agg(["mean", "std", "count"]).reset_index()
            ax.plot(stats["round"], stats["mean"], label=str(label))
            mask = stats["count"] >= 2
            if mask.any():
                ax.fill_between(
                    stats.loc[mask, "round"],
                    stats.loc[mask, "mean"] - stats.loc[mask, "std"],
                    stats.loc[mask, "mean"] + stats.loc[mask, "std"], alpha=0.16,
                )
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Round")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


rounds_df["trajectory_treatment"] = (
    "C=" + rounds_df.C_local.astype(str)
    + " | alpha=" + rounds_df.alpha.astype(str)
    + " | F=" + rounds_df.reference.astype(str)
    + " | eps=" + rounds_df.target_epsilon.fillna("noDP").astype(str)
    + " | bruit=" + rounds_df.noise_assignment.astype(str)
    + " | attaque=" + rounds_df.attack.astype(str)
)
chosen = best_trajectory_stratum(rounds_df)
trajectory_sample = rounds_df[rounds_df.strict_stratum == chosen].copy() if chosen else pd.DataFrame()
if chosen:
    print("Strate affichée (aucun mélange campagne/horizon/protocole) :", chosen)
plot_trajectories(
    trajectory_sample,
    ["test_accuracy_pct", "test_loss", "client_loss_heldout"],
    "Trajectoires de la strate compatible la plus renseignée",
    hue="trajectory_treatment",
)
metric_availability = pd.DataFrame([
    {
        "métrique demandée": "Train accuracy",
        "champ utilisé": "train_accuracy",
        "valeurs disponibles": int(rounds_df.train_accuracy_pct.notna().sum()),
        "statut": "indisponible" if not rounds_df.train_accuracy_pct.notna().any() else "disponible",
    },
    {
        "métrique demandée": "Train loss",
        "champ utilisé": "train_loss (hors placeholders privés)",
        "valeurs disponibles": int(rounds_df.train_loss.notna().sum()),
        "statut": "indisponible" if not rounds_df.train_loss.notna().any() else "disponible",
    },
    {
        "métrique demandée": "Client accuracy",
        "champ utilisé": "client_accuracy_mean (held-out)",
        "valeurs disponibles": int(rounds_df.client_accuracy_pct.notna().sum()),
        "statut": "disponible",
    },
    {
        "métrique demandée": "Client loss",
        "champ utilisé": "client_loss_mean (held-out)",
        "valeurs disponibles": int(rounds_df.client_loss_heldout.notna().sum()),
        "statut": "disponible",
    },
    {
        "métrique demandée": "Test accuracy / loss",
        "champ utilisé": "test_accuracy / test_loss",
        "valeurs disponibles": int(rounds_df.test_accuracy_pct.notna().sum()),
        "statut": "disponible",
    },
])
display(Markdown(
    "**Disponibilité des métriques.** Les zéros techniques masqués par le "
    "transcript local-DP ne sont jamais interprétés comme des observations."
))
display(metric_availability)
"""
    ),
    md(
        r"""
## 4. Accuracy et fairness finales

Les tableaux suivants conservent le nombre de seeds. Un faible gap ou une faible
variance n'est pas une amélioration si toutes les accuracies s'effondrent.

`balanced_accuracy_pct` est une accuracy équilibrée entre classes. La métrique
`balanced_performance_fairness` (BPF) est la variance des pertes clientes
held-out pondérée par les tailles clientes. Ce sont deux objets distincts :
le notebook ne les fusionne jamais et n'impute pas BPF lorsqu'elle est absente.
"""
    ),
    code(
        r"""
PERF_COLS = [
    "campaign", "phase", "horizon", "n_clients", "dp_enabled", "target_epsilon",
    "noise_profile", "noise_assignment", "C_local", "alpha", "reference", "attack", "n_seeds",
    "identifiability", "test_accuracy_pct_mean", "test_accuracy_pct_sd",
    "client_accuracy_pct_mean", "variance_pp2_mean", "worst20_pct_mean", "gap_pp_mean",
    "balanced_accuracy_pct_mean", "balanced_worst20_pct_mean", "balanced_gap_pp_mean",
    "balanced_performance_fairness_mean",
]
display(summary_df[PERF_COLS].sort_values(["campaign", "phase", "n_clients", "alpha"], na_position="last").head(100))

available_balanced = final_df["balanced_accuracy_pct"].notna().sum() if len(final_df) else 0
display(Markdown(
    f"Balanced accuracy disponible dans **{available_balanced}/{len(final_df)}** runs finaux. "
    "Les cellules absentes restent `NaN` et ne sont pas imputées."
))
"""
    ),
    code(
        r"""
# Courbes alpha : on choisit la phase compatible contenant le plus de valeurs distinctes de alpha.
alpha_candidates = []
for (campaign, phase, horizon, n), group in final_df.groupby(
    ["campaign", "phase", "horizon", "n_clients"], dropna=False
):
    alpha_candidates.append((group.alpha.nunique(), len(group), campaign, phase, horizon, n))
alpha_candidates.sort(reverse=True)
if alpha_candidates and alpha_candidates[0][0] > 1:
    _, _, campaign, phase, horizon, n = alpha_candidates[0]
    alpha_view = final_df[
        (final_df.campaign == campaign) & (final_df.phase == phase) &
        (final_df.horizon == horizon) & (final_df.n_clients == n)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), ["test_accuracy_pct", "variance_pp2", "worst20_pct", "gap_pp"]):
        for ref, group in alpha_view.groupby("reference"):
            stats = group.groupby("alpha")[metric].agg(["mean", "std", "count"]).reset_index()
            ax.plot(stats.alpha, stats["mean"], marker="o", label=ref)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("alpha")
        ax.legend(fontsize=8)
    fig.suptitle(f"Effet de alpha sans mélange : {campaign}/{phase}, T={horizon}, n={n}")
    fig.tight_layout(); plt.show()
else:
    display(Markdown("**Non identifiable : aucune strate ne contient plusieurs valeurs de alpha.**"))
"""
    ),
    md(
        r"""
## 5. Poids FAR, concentration et entropie

`nΣq² = 1` correspond aux poids uniformes. Une valeur supérieure signifie que
le bruit et les contributions sont davantage concentrés. Une entropie plus
faible et un poids maximal plus élevé indiquent également une softmax plus
concentrée.
"""
    ),
    code(
        r"""
WEIGHT_COLS = [
    "campaign", "phase", "horizon", "n_clients", "alpha", "reference", "attack",
    "n_seeds", "identifiability", "max_weight_mean", "weight_concentration_mean",
    "weight_entropy_mean", "effective_clients_mean", "score_span_mean", "logit_range_mean",
]
display(summary_df[WEIGHT_COLS].dropna(subset=["max_weight_mean"]).head(100))

weight_view = final_df[final_df.max_weight.notna()].copy()
if len(weight_view) and weight_view.alpha.nunique() > 1:
    # The plot remains stratified by campaign/phase/horizon through facets in the label.
    weight_view["panel"] = weight_view.campaign + "/" + weight_view.phase + "/T" + weight_view.horizon.astype(str)
    panel = weight_view.groupby("panel").run_uid.nunique().idxmax()
    selected = weight_view[weight_view.panel == panel]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["max_weight", "weight_concentration", "weight_entropy"]):
        for ref, group in selected.groupby("reference"):
            means = group.groupby("alpha")[metric].mean().reset_index()
            ax.plot(means.alpha, means[metric], marker="o", label=ref)
        ax.set_title(metric.replace("_", " ")); ax.set_xlabel("alpha"); ax.legend(fontsize=8)
    fig.suptitle(f"Concentration des poids — {panel}"); fig.tight_layout(); plt.show()
else:
    display(Markdown("**Non identifiable : pas assez de valeurs de alpha pour tracer la concentration.**"))
"""
    ),
    md(
        r"""
## 6. Confidentialité, bruit et clipping

Le tableau rapporte le budget réalisé, le profil de bruit public, ainsi que les
taux de clipping local et serveur. DP/no-DP et les niveaux de bruit restent des
traitements distincts; ils ne sont jamais moyennés ensemble.
"""
    ),
    code(
        r"""
DP_COLS = [
    "campaign", "phase", "horizon", "n_clients", "dp_enabled", "target_epsilon",
    "noise_profile", "noise_assignment", "noise_scale_min", "noise_scale_max", "C_local", "U_server",
    "alpha", "reference", "attack", "n_seeds", "identifiability", "epsilon_mean",
    "epsilon_sd", "delta_mean", "sigma_min_mean", "sigma_mean_mean", "sigma_max_mean",
    "local_clip_rate_mean", "server_clip_rate_mean",
]
display(summary_df[DP_COLS].sort_values(["campaign", "phase", "n_clients", "target_epsilon"], na_position="last").head(120))
"""
    ),
    md(
        r"""
## 7. Attaques byzantines et références robustes

Les comparaisons CM/trMean/RFA/F_CC ne sont interprétables que dans une même
campagne, phase, horizon, nombre de clients, budget DP et attaque. La masse
byzantine et les erreurs de centre sont des oracles de simulation.

Les matrices prévoient des contrôles à 0 % et des attaques à 20 %. Le snapshot
exécuté ci-dessous ne contient encore que 0 % : aucun effet byzantin ne peut
donc être estimé à ce stade, et deux niveaux ne permettraient de toute façon
pas une dose-réponse. De même, le passage de 10 à 25 clients change aussi la
taille locale (N_i), le batch et parfois (C); il s'agit d'un diagnostic de
transfert d'échelle, pas d'un effet causal pur du nombre de clients.

Sous attaque, les colonnes `honest_*` excluent les identifiants byzantins. Le
contrôle `none` apparié exclut exactement les mêmes identifiants : les écarts de
fairness ne viennent donc pas d'un changement artificiel de cohorte évaluée.
"""
    ),
    code(
        r"""
ATTACK_COLS = [
    "campaign", "phase", "horizon", "n_clients", "dp_enabled", "target_epsilon",
    "attack", "byzantine_fraction", "alpha", "reference", "n_seeds", "identifiability",
    "test_accuracy_pct_mean", "worst20_pct_mean", "gap_pp_mean",
    "honest_client_accuracy_pct_mean", "honest_variance_pp2_mean",
    "honest_worst20_pct_mean", "honest_gap_pp_mean",
    "byzantine_mass_oracle_mean", "reference_error_oracle_mean",
    "byzantine_displacement_norm_oracle_mean", "aggregate_error_oracle_mean",
]
attack_summary = summary_df[(summary_df.attack != "none") | (summary_df.byzantine_fraction > 0)]
if len(attack_summary):
    display(attack_summary[ATTACK_COLS].head(120))
else:
    display(Markdown("**Non identifiable actuellement : aucun run d'attaque valide découvert.**"))

if len(attack_summary) and attack_summary.reference.nunique() > 1:
    plot_data = attack_summary.dropna(subset=["test_accuracy_pct_mean"])
    if len(plot_data):
        labels = plot_data["attack"].astype(str) + " | " + plot_data["reference"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(np.arange(len(plot_data)), plot_data.test_accuracy_pct_mean)
        ax.set_xticks(np.arange(len(plot_data)), labels, rotation=70, ha="right")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Runs attaqués disponibles — cellules non fusionnées")
        plt.tight_layout(); plt.show()
print("Fractions byzantines observées :", sorted(final_df.byzantine_fraction.dropna().unique().tolist()))
print("Interprétation n=10/25 : transfert d'échelle confondu avec N_i, batch et C.")

positioning_phases = set(final_df.loc[final_df.campaign.str.startswith("positioning", na=False), "phase"])
for expected_phase in (
    "e_byzantine_identification_screen", "f_confirmation_t20",
    "f2_horizon_extension_t40",
):
    if expected_phase not in positioning_phases:
        print(f"Robustesse {expected_phase}: données en attente (aucun run valide); aucun verdict.")
"""
    ),
    md(
        r"""
## 8. Diagnostics oracle : bruit, tilting et robustesse

Ces colonnes décomposent l'erreur observée en biais de tilting honnête, bruit DP
à poids fixes, déplacement byzantin et erreur totale par rapport au centre
honnête propre. Elles ne sont disponibles que lorsque les diagnostics oracle
ont été activés.
"""
    ),
    code(
        r"""
ORACLE_COLS = [
    "campaign", "phase", "horizon", "n_clients", "target_epsilon", "noise_profile",
    "noise_assignment",
    "alpha", "reference", "attack", "n_seeds", "identifiability",
    "weight_noise_corr_oracle_mean", "score_clean_corr_oracle_mean",
    "top_tail_recall_oracle_mean", "tilting_bias_norm_oracle_mean",
    "fixed_weight_dp_noise_norm_oracle_mean", "byzantine_displacement_norm_oracle_mean",
    "reference_error_oracle_mean", "aggregate_error_oracle_mean",
    "decomposition_residual_oracle_mean",
]
oracle_summary = summary_df[summary_df[[c for c in ORACLE_COLS if c.endswith("_mean")]].notna().any(axis=1)]
if len(oracle_summary):
    display(oracle_summary[ORACLE_COLS].head(120))
else:
    display(Markdown("**Non identifiable : aucun diagnostic oracle disponible.**"))
"""
    ),
    md(
        r"""
## 9. Explorateur de cellules strictement comparables

La fonction suivante évite les comparaisons accidentelles. Elle exige un seul
`campaign`, une seule `phase` et un seul `horizon`; elle signale sinon la liste
des strates qu'il faut analyser séparément.
"""
    ),
    code(
        r"""
def comparable_cell(frame, *, campaign, phase, horizon, n_clients=None, dp_enabled=None,
                    target_epsilon=None, attack=None):
    view = frame[
        (frame.campaign == campaign) & (frame.phase == phase) & (frame.horizon == horizon)
    ].copy()
    if n_clients is not None: view = view[view.n_clients == n_clients]
    if dp_enabled is not None: view = view[view.dp_enabled == dp_enabled]
    if target_epsilon is not None: view = view[np.isclose(view.target_epsilon, target_epsilon, equal_nan=False)]
    if attack is not None: view = view[view.attack == attack]
    return view


if len(final_df):
    example = final_df.groupby(["campaign", "phase", "horizon"]).size().sort_values(ascending=False).index[0]
    example_view = comparable_cell(
        final_df, campaign=example[0], phase=example[1], horizon=example[2]
    )
    print("Exemple sélectionné :", example, "—", len(example_view), "runs")
    display(example_view[[
        "seed", "n_clients", "dp_enabled", "target_epsilon", "C_local", "alpha",
        "reference", "attack", "test_accuracy_pct", "worst20_pct", "gap_pp",
    ]].head(50))
"""
    ),
    md(
        r"""
## 10. Vues explicites par axe expérimental

Chaque panneau sélectionne automatiquement une cellule où l'axe varie tout en
gardant fixes campagne, phase et horizon ainsi que les autres facteurs connus.
S'il n'existe pas au moins deux valeurs de l'axe, le panneau affiche « non
identifiable ». Les vues couvrent performance/loss, fairness et, lorsqu'ils
existent, les oracles de robustesse.
"""
    ),
    code(
        r"""
AXIS_METRICS = [
    "test_accuracy_pct", "test_loss", "variance_pp2", "worst20_pct", "gap_pp",
    "reference_error_oracle", "byzantine_mass_oracle", "aggregate_error_oracle",
]


def select_axis_cell(frame, axis, invariants):
    candidates = []
    for keys, group in frame.groupby(invariants, dropna=False):
        distinct = group[axis].dropna().nunique()
        if distinct >= 2:
            seed_counts = group.groupby(axis, dropna=False).seed.nunique()
            candidates.append((
                int(seed_counts.min()), distinct, group.run_uid.nunique(), keys, group,
            ))
    if not candidates:
        return pd.DataFrame(), None
    # A confirmation multi-seeds is preferred to a wider one-seed screen.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][4].copy(), candidates[0][3]


def plot_axis_panel(view, axis, title):
    if view.empty:
        display(Markdown(f"**Non identifiable : pas de cellule compatible pour {title}.**"))
        return
    usable = [metric for metric in AXIS_METRICS if view[metric].notna().any()]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    for ax, metric in zip(axes.ravel(), usable):
        stats = view.groupby(axis, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
        labels = stats[axis].map(
            lambda value: f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
        )
        positions = np.arange(len(stats))
        ax.plot(positions, stats["mean"], marker="o")
        multi = stats["count"] >= 2
        if multi.any():
            ax.errorbar(positions[multi], stats.loc[multi, "mean"],
                        yerr=stats.loc[multi, "std"], fmt="none", capsize=3)
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel(axis)
    for ax in axes.ravel()[len(usable):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(); plt.show()


common = ["campaign", "phase", "horizon", "dataset", "model", "n_clients",
          "partition", "sampling_scheme", "adjacency", "update_mode", "dp_enabled",
          "target_epsilon", "noise_profile", "noise_assignment", "noise_signature",
          "U_server", "alpha", "reference", "attack"]
c_invariants = [key for key in common if key != "C_local"]
c_view, c_key = select_axis_cell(final_df, "C_local", c_invariants)
print("Cellule C_local :", c_key)
plot_axis_panel(c_view, "C_local", "Clipping local C — cellule compatible")
"""
    ),
    code(
        r"""
# Transfert des deux profils calibrés de B3 : C=8 à n=10 et C=4 à n=25.
# On évite ainsi de moyenner plusieurs valeurs de C comme si elles étaient des seeds.
n_view = final_df[
    (final_df.campaign == "positioning_v3")
    & (final_df.phase == "b3_local_clip_dp_confirmation")
    & (
        ((final_df.n_clients == 10) & np.isclose(final_df.C_local, 8.0))
        | ((final_df.n_clients == 25) & np.isclose(final_df.C_local, 4.0))
    )
].copy()
n_key = "positioning_v3/B3; profils calibrés n10:C8 et n25:C4"
display(Markdown(
    "**Attention — transfert confondu :** une différence entre n=10 et n=25 "
    "ne peut pas être attribuée causalement à n, car N_i, le batch et parfois C/U changent."
))
print("Cellule de transfert n :", n_key)
plot_axis_panel(n_view, "n_clients", "Transfert n=10/25 — non causal")
"""
    ),
    code(
        r"""
# Deux vues privacy : epsilon réalisé, puis profil de bruit/sigma. Elles restent
# stratifiées par campagne/phase/horizon et configuration algorithmique.
privacy_invariants = [
    "campaign", "phase", "horizon", "dataset", "model", "n_clients", "partition",
    "sampling_scheme", "adjacency", "update_mode", "C_local", "alpha", "reference", "attack",
    "noise_assignment",
]
eps_view, eps_key = select_axis_cell(final_df, "epsilon", privacy_invariants)
print("Cellule epsilon réalisé :", eps_key)
plot_axis_panel(eps_view, "epsilon", "Budget réalisé epsilon — cellule compatible")

noise_invariants = [key for key in privacy_invariants if key != "noise_assignment"] + ["target_epsilon"]
noise_view, noise_key = select_axis_cell(final_df, "noise_assignment", noise_invariants)
print("Cellule bruit multi-seeds :", noise_key)
plot_axis_panel(noise_view, "noise_assignment", "Bruit homogène/hétéroscédastique — vue multi-seeds disponible")
if len(noise_view):
    display(noise_view[["noise_profile", "noise_assignment", "sigma_min", "sigma_mean", "sigma_max", "epsilon",
                        "test_accuracy_pct", "test_loss", "worst20_pct", "gap_pp"]]
            .sort_values(["noise_assignment", "sigma_mean"]))

# D2 is deliberately shown separately: identity/reverse are paired assignments,
# not independent seeds, and each arm is still a development result.
d2_view = final_df[
    (final_df.campaign == "positioning_v3")
    & (final_df.phase == "d2_noise_assignment_dev")
    & (final_df.n_clients == 10)
    & np.isclose(final_df.C_local, 8.0)
    & np.isclose(final_df.alpha, 2.0)
    & (final_df.reference == "F_CC")
].copy()
display(Markdown(
    "**D2 — assignations mild/strong × identity/reverse :** une seule seed de "
    "développement; les deux ordres sont une nuisance appariée, pas des réplications."
))
plot_axis_panel(d2_view, "noise_assignment", "D2 : niveau et assignation du bruit — seed de développement")
if len(d2_view):
    display(d2_view[["noise_profile", "noise_assignment", "sigma_min", "sigma_mean", "sigma_max", "epsilon",
                     "test_accuracy_pct", "test_loss", "variance_pp2", "worst20_pct", "gap_pp"]]
            .sort_values("noise_assignment"))
"""
    ),
    code(
        r"""
byz_invariants = [
    "campaign", "phase", "horizon", "dataset", "model", "n_clients", "partition",
    "sampling_scheme", "adjacency", "update_mode", "dp_enabled", "target_epsilon",
    "noise_profile", "C_local", "alpha", "reference",
]
byz_view, byz_key = select_axis_cell(final_df, "byzantine_fraction", byz_invariants)
observed_byz = sorted(final_df.byzantine_fraction.dropna().unique().tolist())
print("Fractions disponibles dans tout le snapshot :", observed_byz)
if observed_byz == [0.0]:
    display(Markdown(
        "**Aucun run attaqué n'est encore achevé : la robustesse et l'effet de "
        "la fraction byzantine sont non identifiables dans ce snapshot.**"
    ))
elif observed_byz == [0.0, 0.2]:
    display(Markdown(
        "**Seulement deux points (0 % et 20 %) : présence/absence d'attaque, "
        "aucune conclusion de dose-réponse.**"
    ))
print("Cellule fraction byzantine :", byz_key)
plot_axis_panel(byz_view, "byzantine_fraction", "Fraction byzantine — disponibilité actuelle")
"""
    ),
    md(
        r"""
## 11. Bilan de disponibilité, sans verdict automatique

Cette cellule décrit uniquement ce qui est identifiable dans l'état courant des
fichiers. Elle ne transforme pas la complétude en décision scientifique.
"""
    ),
    code(
        r"""
lines = []
for (campaign, phase), group in final_df.groupby(["campaign", "phase"]):
    seeds = group.seed.nunique()
    horizons = sorted(group.horizon.unique().tolist())
    attacks = sorted(group.attack.dropna().unique().tolist())
    lines.append(
        f"- **{campaign}/{phase}** : {group.run_uid.nunique()} runs, {seeds} seed(s), "
        f"horizons {horizons}, attaques {attacks}. "
        + ("Comparaisons multi-seeds possibles." if seeds >= 3 else "Résultat exploratoire/non identifiable inter-seeds.")
    )
display(Markdown("\n".join(lines) if lines else "Aucun run valide."))
print("\nAucun gate n'est calculé ou modifié par ce notebook.")
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "fedlab venv",
            "language": "python",
            "name": "fedlab-venv",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
