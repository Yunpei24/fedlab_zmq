#!/usr/bin/env python3
"""Generate the reproducible DT-LDP-FAR results-analysis notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "DT_LDP_FAR_Results_Analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Analyse des résultats DT-LDP-FAR

Ce notebook détecte récursivement les runs sous **results/dt_ldp_far** et produit :

- l'inventaire et l'état de complétude des runs ;
- un tableau final par seed et une synthèse moyenne ± écart-type ;
- les courbes d'accuracy, de loss et de fairness par round ;
- les diagnostics de confidentialité et de tilting ;
- une heatmap E7 croisant le nombre de pas Poisson K et l'horizon T.

Les métriques oracle servent uniquement au diagnostic expérimental. Elles ne constituent pas des
sorties publiables d'un mécanisme LDP.
"""
    ),
    code(
        """
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 180)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.figsize": (11, 5.5), "axes.titlesize": 13})


def locate_repo_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "run_experiment.py").exists() and (candidate / "results").exists():
            return candidate
    raise FileNotFoundError(
        "Lancez le notebook depuis fedlab_zmq ou un de ses sous-dossiers."
    )


REPO_ROOT = locate_repo_root()
RESULTS_ROOT = REPO_ROOT / "results" / "dt_ldp_far"
EXPORT_ROOT = REPO_ROOT / "output" / "analysis" / "dt_ldp_far_notebook"
print("Dépôt     :", REPO_ROOT)
print("Résultats :", RESULTS_ROOT)
"""
    ),
    markdown(
        """
## 1. Filtres

Modifiez ces variables puis réexécutez les cellules suivantes. Une valeur **None** conserve toutes
les valeurs. Exemple : CAMPAIGN_CONTAINS = "local_pilot_v2".
"""
    ),
    code(
        """
CAMPAIGN_CONTAINS = None
METHOD_FILTER = None
SCENARIO_FILTER = None
PRIVACY_FILTER = None
REFERENCE_FILTER = None
THREAT_FILTER = None
ONLY_COMPLETE = True
INCLUDE_METRICS_ONLY = False

# Dimensions utilisées pour agréger les seeds.
GROUP_BY = [
    "scenario", "method", "reference", "threat", "privacy",
    "tilt", "geometry", "K", "T", "delay",
]
"""
    ),
    code(
        """
def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"JSON ignoré: {path} ({exc})")
        return {}


def read_yaml(path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        warnings.warn(f"YAML ignoré: {path} ({exc})")
        return {}


def find_upwards(start, filename, stop):
    current = Path(start).resolve()
    stop = Path(stop).resolve()
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current == stop or current.parent == current:
            return None
        current = current.parent


def percent(value):
    if value is None or pd.isna(value):
        return np.nan
    value = float(value)
    return 100.0 * value if abs(value) <= 1.5 else value


def variance_pp2(row):
    if row.get("client_accuracy_variance_pct2") is not None:
        return float(row["client_accuracy_variance_pct2"])
    raw = row.get("client_accuracy_variance")
    return np.nan if raw is None else 10000.0 * float(raw)


def first_present(mapping, keys, default=np.nan):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def scalar_items(mapping):
    scalar_types = (str, int, float, bool, type(None), np.integer, np.floating)
    return {
        key: value
        for key, value in (mapping or {}).items()
        if isinstance(value, scalar_types)
    }


def load_results(results_root):
    run_rows, round_rows, client_rows, error_rows = [], [], [], []

    for metrics_path in sorted(Path(results_root).rglob("metrics.json")):
        payload = read_json(metrics_path)
        rounds = payload.get("rounds") or []
        if not rounds:
            error_rows.append({"path": str(metrics_path), "reason": "aucun round"})
            continue

        manifest_path = find_upwards(
            metrics_path.parent, "dt_ldp_far_task_manifest.json", results_root
        )
        config_path = find_upwards(metrics_path.parent, "resolved_config.yaml", results_root)
        manifest = read_json(manifest_path) if manifest_path else {}
        config = read_yaml(config_path) if config_path else {}
        summary = payload.get("summary") or {}
        training = config.get("training") or {}
        algo = training.get("algo_config") or summary.get("algo_config") or {}
        axes = (config.get("reproduction") or {}).get("axes") or {}
        data = config.get("data") or {}

        expected_rounds = int(
            training.get("num_rounds") or summary.get("num_rounds") or len(rounds)
        )
        status = manifest.get("status", "metrics_only")
        complete = len(rounds) == expected_rounds and status in {
            "complete", "complete_reused", "metrics_only"
        }
        task_id = manifest.get("task_id") or str(metrics_path.parent)
        final = max(rounds, key=lambda row: row.get("round_num", row.get("round", -1)))

        try:
            output_family = metrics_path.relative_to(results_root).parts[0]
        except (ValueError, IndexError):
            output_family = "unknown"

        meta = {
            "run_key": task_id,
            "task_id": task_id,
            "output_family": output_family,
            "matrix": (
                Path(manifest["matrix_path"]).stem
                if manifest.get("matrix_path")
                else "unknown"
            ),
            "status": status,
            "complete": complete,
            "rounds_recorded": len(rounds),
            "T": expected_rounds,
            "scenario": axes.get("scenario", "unknown"),
            "method": axes.get(
                "method", summary.get("algorithm", payload.get("algorithm", "unknown"))
            ),
            "reference": axes.get(
                "reference", algo.get("dt_reference", algo.get("robust_reference", "unknown"))
            ),
            "threat": axes.get("threat", final.get("attack_name", "none")),
            "privacy": axes.get("privacy", final.get("privacy_profile", "unknown")),
            "tilt": axes.get("tilt", "unknown"),
            "geometry": axes.get("geometry", "unknown"),
            "K": int(axes.get("local_epochs", algo.get("local_epochs", 1))),
            "delay": int(axes.get("delay", algo.get("tilt_delay_rounds", 1))),
            "partition_seed": int(
                axes.get(
                    "partition_seed",
                    summary.get("partition_seed", data.get("partition_seed", -1)),
                )
            ),
            "training_seed": int(
                axes.get(
                    "training_seed",
                    summary.get("training_seed", summary.get("seed", -1)),
                )
            ),
            "dataset": summary.get("dataset", data.get("dataset", "unknown")),
            "model": summary.get("model", data.get("model", "unknown")),
            "partition": summary.get("partition", data.get("partition", "unknown")),
            "dirichlet_alpha": float(
                summary.get("alpha", data.get("alpha", np.nan))
            ),
            "metrics_path": str(metrics_path),
        }

        run_rows.append(
            {
                **meta,
                "train_accuracy_pct": percent(
                    first_present(final, ["train_accuracy", "training_accuracy"])
                ),
                "client_accuracy_pct": percent(final.get("client_accuracy_mean")),
                "test_accuracy_pct": percent(final.get("test_accuracy")),
                "train_loss": first_present(final, ["train_loss", "avg_local_loss"]),
                "client_loss": final.get("client_loss_mean", np.nan),
                "test_loss": final.get("test_loss", np.nan),
                "variance_pp2": variance_pp2(final),
                "worst20_pct": percent(final.get("worst20_accuracy")),
                "gap_pct": percent(final.get("best20_worst20_gap")),
                "balanced_accuracy_pct": percent(
                    final.get("mean_client_balanced_accuracy")
                ),
                "epsilon": final.get("privacy_epsilon_max", np.nan),
                "delta": final.get("privacy_delta", np.nan),
                "noise_multiplier": final.get(
                    "privacy_model_noise_multiplier_mean", np.nan
                ),
                "privacy_clip_rate": final.get("privacy_clip_rate_mean", np.nan),
                "max_weight": final.get("max_client_weight", np.nan),
                "weight_entropy": final.get("weight_entropy", np.nan),
                "effective_clients": final.get("effective_num_clients", np.nan),
                "logit_span": final.get("dtldp_current_logit_span", np.nan),
                "score_saturation_rate": final.get(
                    "dtldp_current_score_saturation_rate", np.nan
                ),
                "reference_drift": final.get("dtldp_reference_drift", np.nan),
                "score_drift_linf": final.get("dtldp_score_drift_linf", np.nan),
                "noise_amplification": final.get(
                    "dtldp_noise_amplification_vs_uniform", np.nan
                ),
                "total_energy_j": final.get(
                    "cumulative_energy_j", summary.get("total_energy_j", np.nan)
                ),
                "total_bytes": final.get("cumulative_bytes", np.nan),
            }
        )

        for row in rounds:
            record = {**meta, **scalar_items(row)}
            record["round_num"] = int(row.get("round_num", row.get("round", 0)))
            record["test_accuracy_pct"] = percent(row.get("test_accuracy"))
            record["client_accuracy_pct"] = percent(row.get("client_accuracy_mean"))
            record["train_accuracy_pct"] = percent(
                first_present(row, ["train_accuracy", "training_accuracy"])
            )
            record["worst20_pct"] = percent(row.get("worst20_accuracy"))
            record["gap_pct"] = percent(row.get("best20_worst20_gap"))
            record["variance_pp2"] = variance_pp2(row)
            round_rows.append(record)

        values = final.get("client_accuracy_values_oracle") or []
        client_ids = final.get("evaluated_client_ids_oracle") or range(len(values))
        for client_id, accuracy in zip(client_ids, values):
            client_rows.append(
                {
                    **meta,
                    "client_id": client_id,
                    "client_accuracy_pct": percent(accuracy),
                }
            )

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(round_rows),
        pd.DataFrame(client_rows),
        pd.DataFrame(error_rows),
    )


runs_df, rounds_df, clients_df, load_errors_df = load_results(RESULTS_ROOT)
print(
    f"Runs: {len(runs_df)} | lignes round: {len(rounds_df)} | "
    f"observations clientes: {len(clients_df)}"
)
if not load_errors_df.empty:
    display(load_errors_df)
"""
    ),
    code(
        """
def matches_filter(series, values):
    if values is None:
        return pd.Series(True, index=series.index)
    if isinstance(values, str):
        values = [values]
    return series.astype(str).isin([str(value) for value in values])


mask = pd.Series(True, index=runs_df.index)
if CAMPAIGN_CONTAINS:
    mask &= runs_df["metrics_path"].str.contains(
        str(CAMPAIGN_CONTAINS), case=False, regex=False
    )
mask &= matches_filter(runs_df["method"], METHOD_FILTER)
mask &= matches_filter(runs_df["scenario"], SCENARIO_FILTER)
mask &= matches_filter(runs_df["privacy"], PRIVACY_FILTER)
mask &= matches_filter(runs_df["reference"], REFERENCE_FILTER)
mask &= matches_filter(runs_df["threat"], THREAT_FILTER)
if ONLY_COMPLETE:
    mask &= runs_df["complete"]
if not INCLUDE_METRICS_ONLY:
    mask &= runs_df["status"] != "metrics_only"

selected_runs = runs_df.loc[mask].copy()
selected_keys = set(selected_runs["run_key"])
selected_rounds = rounds_df[rounds_df["run_key"].isin(selected_keys)].copy()
selected_clients = (
    clients_df[clients_df["run_key"].isin(selected_keys)].copy()
    if not clients_df.empty
    else clients_df.copy()
)


def add_series_label(frame):
    if frame.empty:
        return frame
    columns = ["method", "reference", "privacy", "scenario", "K", "T"]
    frame["series"] = frame[columns].astype(str).agg(" | ".join, axis=1)
    return frame


selected_runs = add_series_label(selected_runs)
selected_rounds = add_series_label(selected_rounds)
selected_clients = add_series_label(selected_clients)

print("Runs sélectionnés:", len(selected_runs))
if selected_runs.empty:
    display(
        runs_df[
            ["output_family", "scenario", "method", "reference", "threat", "privacy"]
        ].drop_duplicates()
    )
else:
    display(
        selected_runs.groupby(["status", "complete"])
        .size()
        .rename("runs")
        .reset_index()
    )
"""
    ),
    markdown(
        """
## 2. Tableaux finaux

**Client Acc.** est la moyenne non pondérée des accuracies sur les partitions clientes de test.
**Test Acc.** est l'accuracy du modèle global sur le jeu de test global. **Var (pp²)** est la
variance inter-clients exprimée en points de pourcentage carrés. **Worst-20** est la moyenne des
20 % de clients les moins performants et **Gap** l'écart Best-20 moins Worst-20.

Les runs actuels n'enregistrent généralement pas l'accuracy sur les données d'entraînement. La
colonne Train Acc. reste alors vide ; elle n'est jamais remplacée par Client Acc.
"""
    ),
    code(
        """
FINAL_COLUMNS = [
    "scenario", "method", "reference", "threat", "privacy", "tilt", "geometry",
    "K", "T", "delay", "partition_seed", "training_seed",
    "train_accuracy_pct", "client_accuracy_pct", "test_accuracy_pct",
    "train_loss", "client_loss", "test_loss", "variance_pp2",
    "worst20_pct", "gap_pct", "epsilon", "delta", "noise_multiplier",
    "max_weight", "weight_entropy", "effective_clients", "logit_span",
    "score_saturation_rate", "total_energy_j", "complete",
]

if not selected_runs.empty:
    final_table = selected_runs[FINAL_COLUMNS].sort_values(
        ["scenario", "method", "reference", "privacy", "K", "T", "partition_seed"]
    )
    display(
        final_table.style.format(
            {
                "train_accuracy_pct": "{:.2f}",
                "client_accuracy_pct": "{:.2f}",
                "test_accuracy_pct": "{:.2f}",
                "train_loss": "{:.4f}",
                "client_loss": "{:.4f}",
                "test_loss": "{:.4f}",
                "variance_pp2": "{:.2f}",
                "worst20_pct": "{:.2f}",
                "gap_pct": "{:.2f}",
                "epsilon": "{:.4f}",
                "delta": "{:.1e}",
                "noise_multiplier": "{:.4f}",
                "max_weight": "{:.4f}",
                "weight_entropy": "{:.4f}",
                "effective_clients": "{:.2f}",
                "logit_span": "{:.4f}",
                "score_saturation_rate": "{:.3f}",
                "total_energy_j": "{:.1f}",
            },
            na_rep="—",
        )
    )
"""
    ),
    code(
        """
SUMMARY_METRICS = {
    "client_accuracy_pct": "Client Acc. (%)",
    "test_accuracy_pct": "Test Acc. (%)",
    "variance_pp2": "Var (pp²)",
    "worst20_pct": "Worst-20 (%)",
    "gap_pct": "Gap (pp)",
    "test_loss": "Test Loss",
    "epsilon": "Epsilon réalisé",
    "noise_multiplier": "Noise multiplier",
    "max_weight": "Poids maximal",
    "effective_clients": "Clients effectifs",
}


def mean_std_text(values, digits=2):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "—"
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f} ± {std:.{digits}f}"


if not selected_runs.empty:
    groups = [column for column in GROUP_BY if column in selected_runs.columns]
    aggregate_rows = []
    for values, group in selected_runs.groupby(groups, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(groups, values))
        row["Seeds"] = group["training_seed"].nunique()
        for metric, label in SUMMARY_METRICS.items():
            digits = 4 if metric in {"epsilon", "noise_multiplier", "max_weight"} else 2
            row[label] = mean_std_text(group[metric], digits)
        aggregate_rows.append(row)
    aggregate_table = pd.DataFrame(aggregate_rows).sort_values(groups)
    display(aggregate_table)
"""
    ),
    markdown(
        """
## 3. Courbes par round

Chaque ligne représente la moyenne sur les seeds disponibles. La zone colorée représente ± un
écart-type.
"""
    ),
    code(
        """
def plot_round_metric(metric, ylabel, title, max_series=16):
    frame = selected_rounds
    if frame.empty or metric not in frame or frame[metric].dropna().empty:
        print("Métrique indisponible:", metric)
        return
    labels = frame["series"].drop_duplicates().tolist()
    if len(labels) > max_series:
        print(
            f"{len(labels)} séries sélectionnées; affichage des {max_series} premières. "
            "Affinez les filtres."
        )
        labels = labels[:max_series]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for label in labels:
        subset = frame[frame["series"] == label]
        curve = subset.groupby("round_num")[metric].agg(["mean", "std"]).reset_index()
        x = curve["round_num"].to_numpy(float)
        y = curve["mean"].to_numpy(float)
        std = curve["std"].fillna(0).to_numpy(float)
        line, = ax.plot(x, y, linewidth=2, label=label)
        ax.fill_between(x, y - std, y + std, color=line.get_color(), alpha=0.14)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


for metric, ylabel, title in [
    ("test_accuracy_pct", "Test accuracy (%)", "Accuracy globale de test"),
    ("client_accuracy_pct", "Client accuracy (%)", "Accuracy moyenne des clients"),
    ("test_loss", "Test loss", "Loss globale de test"),
    ("train_loss", "Train loss enregistrée", "Loss d'entraînement"),
    ("variance_pp2", "Variance (pp²)", "Variance inter-clients"),
    ("worst20_pct", "Worst-20 (%)", "Performance des 20 % moins performants"),
    ("gap_pct", "Best-20 − Worst-20 (pp)", "Écart de performance"),
]:
    plot_round_metric(metric, ylabel, title)
"""
    ),
    markdown("## 4. Diagnostics DP et tilting"),
    code(
        """
for metric, ylabel, title in [
    ("privacy_epsilon_max", "Epsilon réalisé", "Composition de la confidentialité"),
    ("privacy_model_noise_multiplier_mean", "Noise multiplier", "Multiplicateur de bruit"),
    ("max_client_weight", "Poids maximal", "Concentration maximale des poids"),
    ("weight_entropy", "Entropie", "Entropie des poids"),
    ("dtldp_current_logit_span", "Plage des logits", "Intensité effective du tilting"),
    ("dtldp_score_drift_linf", "Drift L-infini", "Drift des scores"),
    (
        "dtldp_noise_amplification_vs_uniform",
        "Facteur",
        "Amplification du bruit par rapport aux poids uniformes",
    ),
]:
    plot_round_metric(metric, ylabel, title)
"""
    ),
    markdown(
        """
## 5. Distribution finale des accuracies clientes

Cette cellule utilise les diagnostics oracle enregistrés. Elle sert à visualiser la dispersion, pas à
définir une sortie LDP publiable.
"""
    ),
    code(
        """
if selected_clients.empty:
    print("Aucune distribution cliente oracle disponible.")
else:
    labels = selected_clients["series"].drop_duplicates().tolist()[:14]
    values = [
        selected_clients.loc[
            selected_clients["series"] == label, "client_accuracy_pct"
        ].dropna().to_numpy()
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(max(11, 0.75 * len(labels)), 5.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Accuracy cliente finale (%)")
    ax.set_title("Distribution inter-clients par configuration")
    ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    ),
    markdown(
        """
## 6. E7 : nombre de pas Poisson K × horizon T

Chaque cellule est la moyenne sur les seeds. À epsilon fixé, l'accountant augmente normalement le
multiplicateur de bruit lorsque le nombre total de mécanismes T × K augmente.
"""
    ),
    code(
        """
def plot_heatmap(pivot, title, value_label, fmt=".1f", cmap="viridis"):
    if pivot.empty:
        return
    matrix = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Pas Poisson K par round")
    ax.set_title(title)
    threshold = np.nanmean(matrix)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                color = "white" if matrix[row, col] < threshold else "black"
                ax.text(
                    col, row, format(matrix[row, col], fmt),
                    ha="center", va="center", color=color,
                )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_label)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


e7 = selected_runs[
    selected_runs["method"].astype(str).str.contains(
        "dt_ldp_far", case=False, regex=False
    )
].copy()
if e7.empty:
    print("Aucun run DT-LDP-FAR sélectionné.")
else:
    for scenario in e7["scenario"].drop_duplicates():
        subset = e7[e7["scenario"] == scenario]
        accuracy = subset.pivot_table(
            index="K", columns="T", values="test_accuracy_pct", aggfunc="mean"
        ).sort_index()
        sigma = subset.pivot_table(
            index="K", columns="T", values="noise_multiplier", aggfunc="mean"
        ).sort_index()
        plot_heatmap(
            accuracy, f"E7 — Test accuracy — {scenario}", "Test accuracy (%)"
        )
        plot_heatmap(
            sigma,
            f"E7 — Multiplicateur de bruit — {scenario}",
            "Noise multiplier",
            fmt=".3f",
            cmap="magma",
        )
"""
    ),
    markdown("## 7. Export CSV"),
    code(
        """
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
selected_runs.to_csv(EXPORT_ROOT / "dt_ldp_far_final_per_run.csv", index=False)
selected_rounds.to_csv(EXPORT_ROOT / "dt_ldp_far_round_metrics.csv", index=False)
if "aggregate_table" in globals():
    aggregate_table.to_csv(
        EXPORT_ROOT / "dt_ldp_far_mean_std_summary.csv", index=False
    )
print("Exports écrits dans:", EXPORT_ROOT)
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (fedlab-zmq)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, TARGET)
print(TARGET)
