#!/usr/bin/env python3
"""Replace only notebook sections 3--8 with compact, readable analyses.

The updater is deliberately surgical and idempotent.  It replaces the cells
between the ``## 3.`` and ``## 9.`` headings, owns the inserted cells through
one dedicated tag, and preserves every cell outside that interval (including
the user-edited introduction and the G0d/G0e sections).  It never executes the
notebook and never changes experimental artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "LDP_Gradient_FAR_Positioning_Analysis.ipynb"
SECTION_TAG = "ldp-positioning-readability-sections-3-8-2026-09"


def markdown(source: str) -> nbformat.NotebookNode:
    normalized = source.strip()
    cell = nbformat.v4.new_markdown_cell(
        normalized, metadata={"tags": [SECTION_TAG]}
    )
    cell.id = hashlib.sha256(
        f"markdown\0{SECTION_TAG}\0{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return cell


def code(source: str) -> nbformat.NotebookNode:
    normalized = source.strip()
    cell = nbformat.v4.new_code_cell(
        normalized, metadata={"tags": [SECTION_TAG]}
    )
    cell.id = hashlib.sha256(
        f"code\0{SECTION_TAG}\0{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return cell


def replacement_cells() -> list[nbformat.NotebookNode]:
    return [
        markdown(
            r"""
## 3. Trajectoires sur les jeux d'évaluation (held-out)

Les graphiques de cette section utilisent la confirmation principale
`positioning_v3 / f_confirmation_t20`, soit la campagne *positioning* version
3, sa phase confirmatoire multi-seeds, et un horizon de **20 tours de
communication**. Ce libellé décrit la provenance des données ; ce n'est ni un
algorithme ni un hyperparamètre.

Pour éviter une légende illisible, nous fixons ici la référence à FCC et
l'absence d'attaque. Chaque panneau contient au plus trois conditions :

- sans DP, $\alpha=2$ ;
- avec DP, $\alpha=0$ (pondération uniforme) ;
- avec DP, $\alpha=2$ (tilting FAR).

Les bandes représentent moyenne $\pm$ un écart-type entre seeds. **Held-out**
signifie ici « évalué sur des exemples mis de côté et jamais utilisés pour
calculer les gradients locaux ». L'accuracy/loss test évalue le modèle global
sur le jeu de test global ; la loss cliente held-out est d'abord calculée sur
la partition d'évaluation de chaque client, puis moyennée entre clients. Ce
terme ne désigne pas le holdout confirmatoire scellé des audits G0f.

**Point important.** Dans ces données, la courbe sans DP à $\alpha=2$ est
effectivement sous les deux courbes DP. C'est une observation de cette
optimisation à 20 tours, pas une preuve que la confidentialité améliore
l'apprentissage. La vue ne contient notamment pas le contrôle sans DP à
$\alpha=0$ et ne constitue donc pas un plan factoriel complet DP $\times$
$\alpha$. Le tableau suivant donne les valeurs exactes avant interprétation.
"""
        ),
        code(
            r"""
READABLE_CAMPAIGN = "positioning_v3"
READABLE_PHASE = "f_confirmation_t20"


def _mean_sd_by_round(frame, metric, group):
    return (
        frame.groupby([group, "round"], dropna=False)[metric]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )


def _plot_one_trajectory(frame, metric, ylabel, title):
    if frame.empty or not frame[metric].notna().any():
        display(Markdown(f"**{title} : données indisponibles.**"))
        return
    for n_clients in sorted(frame.n_clients.dropna().unique()):
        view = frame[frame.n_clients == n_clients]
        stats = _mean_sd_by_round(view, metric, "condition")
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        for label, group in stats.groupby("condition", sort=False):
            group = group.sort_values("round")
            ax.plot(group["round"], group["mean"], linewidth=2, label=label)
            mask = group["n"] >= 2
            if mask.any():
                ax.fill_between(
                    group.loc[mask, "round"],
                    group.loc[mask, "mean"] - group.loc[mask, "sd"],
                    group.loc[mask, "mean"] + group.loc[mask, "sd"],
                    alpha=0.14,
                )
        ax.set(title=f"{title} — n={int(n_clients)} clients", xlabel="Tour", ylabel=ylabel)
        ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        plt.show()


trajectory_focus = rounds_df[
    (rounds_df.campaign == READABLE_CAMPAIGN)
    & (rounds_df.phase == READABLE_PHASE)
    & (rounds_df.reference == "F_CC")
    & (rounds_df.attack == "none")
].copy()

trajectory_focus["condition"] = np.select(
    [
        ~trajectory_focus.dp_enabled & np.isclose(trajectory_focus.alpha, 2.0),
        trajectory_focus.dp_enabled & np.isclose(trajectory_focus.alpha, 0.0),
        trajectory_focus.dp_enabled & np.isclose(trajectory_focus.alpha, 2.0),
    ],
    ["Sans DP · α=2", "DP · α=0", "DP · α=2"],
    default="hors vue",
)
trajectory_focus = trajectory_focus[trajectory_focus.condition != "hors vue"]

trajectory_scope = (
    trajectory_focus.groupby(["n_clients", "condition"], dropna=False)
    .agg(seeds=("seed", "nunique"), runs=("run_uid", "nunique"))
    .reset_index()
    .rename(columns={"n_clients": "Clients", "condition": "Condition", "seeds": "Seeds", "runs": "Runs"})
)
display(Markdown("**Périmètre exact des courbes ci-dessous.**"))
display(trajectory_scope)

trajectory_final = trajectory_focus[
    trajectory_focus["round"] == trajectory_focus["horizon"]
].copy()
trajectory_final_summary = (
    trajectory_final.groupby(["n_clients", "condition"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        accuracy_mean=("test_accuracy_pct", "mean"),
        accuracy_sd=("test_accuracy_pct", "std"),
        loss_mean=("test_loss", "mean"),
        loss_sd=("test_loss", "std"),
        worst20_mean=("worst20_pct", "mean"),
        gap_mean=("gap_pp", "mean"),
        score_span=("score_span", "mean"),
        concentration=("weight_concentration", "mean"),
        qmax=("max_weight", "mean"),
    )
    .reset_index()
)
trajectory_final_summary["Accuracy test finale"] = trajectory_final_summary.apply(
    lambda row: f"{row.accuracy_mean:.2f} ± {row.accuracy_sd:.2f} %", axis=1
)
trajectory_final_summary["Loss test finale"] = trajectory_final_summary.apply(
    lambda row: f"{row.loss_mean:.3f} ± {row.loss_sd:.3f}", axis=1
)
trajectory_final_summary["Worst-20 final"] = trajectory_final_summary["worst20_mean"].map(
    lambda value: f"{value:.2f} %"
)
trajectory_final_summary["Gap final"] = trajectory_final_summary["gap_mean"].map(
    lambda value: f"{value:.2f} pp"
)
trajectory_final_summary["kappa_threshold"] = 2.0 / trajectory_final_summary.n_clients
display(Markdown("### Valeurs au tour 20"))
display(trajectory_final_summary[[
    "n_clients", "condition", "Seeds", "Accuracy test finale",
    "Loss test finale", "Worst-20 final", "Gap final",
]].rename(columns={"n_clients": "Clients", "condition": "Condition"}))
display(Markdown("### Pourquoi le bruit change ici le comportement de FAR"))
display(trajectory_final_summary[[
    "n_clients", "condition", "score_span", "concentration", "qmax",
    "kappa_threshold",
]].rename(columns={
    "n_clients": "Clients", "condition": "Condition",
    "score_span": "Score-span final",
    "concentration": "n × Σ poids² final",
    "qmax": "Poids maximal final",
    "kappa_threshold": "Seuil diagnostique 2/n",
}).round(3))
display(Markdown(
    r"**Lecture correcte.** Oui, le contrôle sans DP à $\alpha=2$ termine "
    "plus bas dans cette campagne. Le clipping par exemple est appliqué dans "
    "les deux branches : le changement principal est l'ajout du bruit. Mais "
    "le bruit homogène aplatit ici les distances FAR et rend les poids à "
    r"$\alpha=2$ presque uniformes, alors que sans DP le même $\alpha$ les "
    r"concentre fortement. Le seuil $\kappa_w/n=2/n$ est seulement diagnostiqué "
    "dans ces runs, pas imposé : sans DP, le poids maximal le dépasse. De plus, "
    "les tirages de batch et de bruit utilisent "
    "le même générateur aléatoire : après le premier bruit, les batchs futurs "
    "ne sont plus strictement appariés. Cette observation ne mesure donc pas "
    "un effet causal pur du bruit et ne prouve pas que la DP améliore "
    "l'apprentissage. Il manque notamment le contrôle multi-seed sans DP à "
    r"$\alpha=0$."
))
"""
        ),
        code(
            r"""
_plot_one_trajectory(
    trajectory_focus,
    "test_accuracy_pct",
    "Accuracy test (%)",
    "Accuracy du modèle global sur le jeu de test",
)
"""
        ),
        code(
            r"""
_plot_one_trajectory(
    trajectory_focus,
    "test_loss",
    "Loss test",
    "Loss du modèle global sur le jeu de test",
)
"""
        ),
        code(
            r"""
_plot_one_trajectory(
    trajectory_focus,
    "client_loss_heldout",
    "Loss cliente moyenne",
    "Loss moyenne sur les jeux d'évaluation clients",
)
"""
        ),
        code(
            r"""
metric_availability = pd.DataFrame([
    {
        "Métrique": "Train accuracy",
        "Champ": "train_accuracy",
        "Valeurs": int(rounds_df.train_accuracy_pct.notna().sum()),
        "Lecture": "Indisponible" if not rounds_df.train_accuracy_pct.notna().any() else "Disponible",
    },
    {
        "Métrique": "Train loss",
        "Champ": "train_loss hors placeholders privés",
        "Valeurs": int(rounds_df.train_loss.notna().sum()),
        "Lecture": "Indisponible" if not rounds_df.train_loss.notna().any() else "Disponible",
    },
    {
        "Métrique": "Test accuracy / loss",
        "Champ": "test_accuracy / test_loss",
        "Valeurs": int(rounds_df.test_accuracy_pct.notna().sum()),
        "Lecture": "Évaluation held-out",
    },
    {
        "Métrique": "Client loss",
        "Champ": "client_loss_mean",
        "Valeurs": int(rounds_df.client_loss_heldout.notna().sum()),
        "Lecture": "Moyenne held-out sur les clients",
    },
])
display(Markdown(
    "**Disponibilité.** Un zéro technique masqué par le transcript local-DP "
    "n'est jamais interprété comme une observation d'entraînement."
))
display(metric_availability)
"""
        ),
        markdown(
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
display(summary_df[PERF_COLS].sort_values(
    ["campaign", "phase", "n_clients", "alpha"], na_position="last"
).head(100))

available_balanced = final_df["balanced_accuracy_pct"].notna().sum() if len(final_df) else 0
display(Markdown(
    f"Balanced accuracy disponible dans **{available_balanced}/{len(final_df)}** runs finaux. "
    "Les cellules absentes restent `NaN` et ne sont pas imputées."
))
"""
        ),
        code(
            r"""
# Effet de alpha : une cellule compatible, choisie sans mélanger campagne,
# phase, horizon ou nombre de clients.
alpha_candidates = []
for (campaign, phase, horizon, n), group in final_df.groupby(
    ["campaign", "phase", "horizon", "n_clients"], dropna=False
):
    alpha_candidates.append((group.alpha.nunique(), len(group), campaign, phase, horizon, n))
alpha_candidates.sort(reverse=True)
if alpha_candidates and alpha_candidates[0][0] > 1:
    _, _, campaign, phase, horizon, n = alpha_candidates[0]
    alpha_view = final_df[
        (final_df.campaign == campaign) & (final_df.phase == phase)
        & (final_df.horizon == horizon) & (final_df.n_clients == n)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, metric, ylabel in zip(
        axes.ravel(),
        ["test_accuracy_pct", "variance_pp2", "worst20_pct", "gap_pp"],
        ["Accuracy test (%)", "Variance (pp²)", "Worst-20 (%)", "Gap (pp)"],
    ):
        for ref, group in alpha_view.groupby("reference"):
            stats = group.groupby("alpha")[metric].agg(["mean", "std", "count"]).reset_index()
            ax.plot(stats.alpha, stats["mean"], marker="o", label=ref)
        ax.set(xlabel="α", ylabel=ylabel)
    axes[0, 0].legend(title="Référence", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Effet de α — {campaign}/{phase}, T={horizon}, n={n}")
    fig.tight_layout()
    plt.show()
else:
    display(Markdown("**Non identifiable : aucune strate ne contient plusieurs valeurs de α.**"))
"""
        ),
        markdown(
            r"""
## 5. Poids FAR : poids maximal, concentration et entropie

Le libellé `positioning_v3/f_confirmation_t20/T20` signifie simplement :

1. **`positioning_v3`** : troisième version de la matrice expérimentale ;
2. **`f_confirmation_t20`** : phase F de confirmation multi-seeds ;
3. **`T20`** : 20 tours de communication.

Trois métriques complémentaires décrivent la distribution des poids
$q_1,\ldots,q_n$ :

| Métrique | Valeur uniforme | Information fournie |
|---|---:|---|
| $nq_{\max}$ | 1 | surcharge relative du client le plus pondéré ; 1,5 signifie 50 % au-dessus de $1/n$ |
| $n\sum_i q_i^2$ | 1 | concentration quadratique et amplification de variance relativement aux poids uniformes |
| $H(q)/\log n$ | 1 | dispersion normalisée ; plus elle diminue, plus les poids se concentrent |
| $N_{\mathrm{eff}}/n$ | 1 | fraction effective de clients ; plus elle diminue, moins de clients portent réellement l'agrégat |

Les vues suivantes fixent la phase F, DP avec $\varepsilon=4$, sans attaque.
Elles ne mélangent donc pas des protocoles incompatibles.
"""
        ),
        code(
            r"""
POSITIONING_RUN_LEVEL = (
    ROOT / "output" / "analysis" / "ldp_gradient_far_positioning_v3" / "run_level.csv"
)
if not POSITIONING_RUN_LEVEL.exists():
    raise FileNotFoundError(
        f"Artefact agrégé introuvable : {POSITIONING_RUN_LEVEL}. "
        "Exécuter d'abord l'analyse positioning_v3."
    )
positioning_run_level = pd.read_csv(POSITIONING_RUN_LEVEL)
reference_labels = {
    "coordinate_median": "CM", "centered_clipping": "F_CC",
    "rfa": "RFA", "trimmed_mean": "trMean",
}

# Les métriques de poids sont d'abord résumées par la médiane des tours dans
# chaque run, puis moyennées entre seeds. Cela évite qu'un dernier tour isolé
# représente toute la dynamique.
weights_focus = positioning_run_level[
    (positioning_run_level.phase == READABLE_PHASE)
    & positioning_run_level.dp_enabled.astype(str).str.lower().eq("true")
    & (positioning_run_level.attack == "none")
    & positioning_run_level.far_max_weight_median.notna()
].copy()
weights_focus["reference"] = weights_focus.robust_reference.map(reference_labels).fillna(
    weights_focus.robust_reference
)
weights_focus["n_qmax"] = weights_focus.num_clients * weights_focus.far_max_weight_median
weights_focus["entropy_normalized"] = (
    weights_focus.far_weight_entropy_median / np.log(weights_focus.num_clients)
)
weights_focus["effective_fraction"] = (
    weights_focus.far_effective_clients_median / weights_focus.num_clients
)

weight_summary = (
    weights_focus.groupby(["num_clients", "far_alpha", "reference"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        n_qmax=("n_qmax", "mean"),
        concentration=("far_weight_concentration_median", "mean"),
        entropy_normalized=("entropy_normalized", "mean"),
        effective_fraction=("effective_fraction", "mean"),
        score_span=("far_score_span_median", "mean"),
        logit_range=("far_logit_range_median", "mean"),
    )
    .reset_index()
    .rename(columns={
        "num_clients": "Clients", "far_alpha": "α", "reference": "Référence",
        "n_qmax": "n × poids max", "concentration": "n × Σ poids²",
        "entropy_normalized": "Entropie / log(n)",
        "effective_fraction": "Clients effectifs / n",
        "score_span": "Score-span médian", "logit_range": "Logit-range médian",
    })
)
display(Markdown(
    "### Concentration des poids\n\n"
    "Médiane sur les 20 tours de chaque run, puis moyenne entre seeds."
))
display(weight_summary[[
    "Clients", "α", "Référence", "Seeds", "n × poids max", "n × Σ poids²",
    "Entropie / log(n)", "Clients effectifs / n",
]].round({
    "n × poids max": 3, "n × Σ poids²": 3,
    "Entropie / log(n)": 3, "Clients effectifs / n": 3,
}))
display(Markdown(
    "### Géométrie des scores et logits\n\n"
    "Le score-span mesure l'étendue des distances utilisées par FAR ; le "
    "logit-range mesure l'étendue après multiplication par α."
))
display(weight_summary[[
    "Clients", "α", "Référence", "Seeds", "Score-span médian", "Logit-range médian",
]].round({"Score-span médian": 3, "Logit-range médian": 3}))
display(Markdown(
    "**Contrôle d'identité.** `Logit-range = |α| × score-span = "
    "log(q_max/q_min)`. Il mesure donc directement le contraste exponentiel "
    "entre le poids maximal et le poids minimal."
))
"""
        ),
        code(
            r"""
def _plot_weight_metric(column, ylabel, uniform_value):
    view = weight_summary[weight_summary["α"] == 2.0].copy()
    if view.empty:
        display(Markdown(f"**{ylabel} : données indisponibles.**"))
        return
    pivot = view.pivot(index="Référence", columns="Clients", values=column)
    ax = pivot.plot(kind="bar", figsize=(8.2, 4.5), width=0.72)
    ax.axhline(uniform_value, color="black", linestyle="--", linewidth=1.2, label="Uniforme")
    ax.set(xlabel="Référence robuste", ylabel=ylabel, title=f"{ylabel} à α=2, DP ε=4, sans attaque")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Clients", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


_plot_weight_metric("n × poids max", "n × poids maximal", 1.0)
"""
        ),
        code(
            r"""
_plot_weight_metric("n × Σ poids²", "Facteur de concentration nΣq²", 1.0)
"""
        ),
        code(
            r"""
_plot_weight_metric("Entropie / log(n)", "Entropie normalisée H(q)/log(n)", 1.0)
"""
        ),
        code(
            r"""
if len(weight_summary):
    tilted = weight_summary[weight_summary["α"] > 0]
    display(Markdown(
        "**Lecture automatique.** Dans cette vue, le client le plus pondéré reçoit "
        f"entre **{tilted['n × poids max'].min():.2f}×** et "
        f"**{tilted['n × poids max'].max():.2f}×** le poids uniforme. "
        "Le facteur quadratique varie entre "
        f"**{tilted['n × Σ poids²'].min():.3f}** et "
        f"**{tilted['n × Σ poids²'].max():.3f}** ; l'entropie normalisée varie entre "
        f"**{tilted['Entropie / log(n)'].min():.3f}** et "
        f"**{tilted['Entropie / log(n)'].max():.3f}**. "
        "Ces valeurs quantifient la concentration ; elles ne prouvent pas à elles seules "
        "un gain ou une perte d'accuracy."
    ))
"""
        ),
        markdown(
            r"""
## 6. Confidentialité, niveau de bruit et clipping

Cette section montre directement comment le budget de confidentialité, le
niveau de bruit et le seuil de clipping sont associés aux métriques finales.
Elle sépare deux expériences :

1. **écran D** : $\varepsilon\in\{2,4,8\}$, donc le bruit change alors que
   $C$ reste fixé pour chaque nombre de clients ;
2. **confirmation B3** : $C\in\{4,8,16\}$ à $\varepsilon=4$, sur trois
   seeds. Changer $C$ change simultanément le clipping et l'écart-type absolu
   du bruit $\sigma C/m$ : B3 mesure donc un effet conjoint, pas l'effet pur
   du clipping.

Chaque figure contient une seule métrique de résultat.

| Quantité | Unité | Question à laquelle elle répond |
|---|---:|---|
| $\varepsilon$ réalisé | sans unité | le budget comptabilisé correspond-il à la cible ? Plus petit signifie davantage de confidentialité |
| multiplicateur $\sigma$ | sans unité | combien de bruit est ajouté relativement au seuil de clipping local ? Plus grand signifie davantage de bruit |
| clipping local | % de gradients individuels | quelle part dépasse la borne locale avant bruit ? `NaN` signifie non publié, jamais zéro |
| clipping serveur | % d'uploads | quelle part des messages privés dépasse la borne serveur et est ramenée sur la boule ? |

Le multiplicateur $\sigma$ et $\varepsilon$ ne sont pas deux axes
indépendants : à protocole fixé, davantage de bruit produit généralement un
$\varepsilon$ plus faible. Dans les figures de l'écran D, l'axe horizontal
est $\varepsilon$ et chaque point est annoté par l'écart-type effectif
$\sigma C/m$. L'écran D n'a qu'une seed : ses courbes sont exploratoires.
"""
        ),
        code(
            r"""
privacy_screen = final_df[
    (final_df.campaign == READABLE_CAMPAIGN)
    & (final_df.phase == "d_privacy_screen_v3")
    & (final_df.reference == "F_CC")
    & np.isclose(final_df.alpha, 2.0)
    & (final_df.attack == "none")
].copy()

# Les paramètres sont lus dans le config exact de chaque metrics.json plutôt
# que réinférés depuis le nom du run.
def _privacy_config_fields(metrics_path):
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    config = payload["config"]
    batch = int(config.get("fixed_batch_size", config.get("batch_size")))
    return pd.Series({
        "batch_size": batch,
        "C_config": float(config["clip_norm"]),
        "U_config": float(config["far_server_clip_norm"]),
    })


privacy_screen[["batch_size", "C_config", "U_config"]] = privacy_screen.metrics_path.apply(
    _privacy_config_fields
)
privacy_screen["coord_noise_std"] = (
    privacy_screen.sigma_mean * privacy_screen.C_config / privacy_screen.batch_size
)
privacy_screen["replace_one_sensitivity"] = (
    2.0 * privacy_screen.C_config / privacy_screen.batch_size
)

privacy_table = (
    privacy_screen.groupby(["n_clients", "target_epsilon"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        C_local=("C_config", "mean"),
        batch_size=("batch_size", "mean"),
        U_server=("U_config", "mean"),
        epsilon_realise=("epsilon", "mean"),
        delta=("delta", "mean"),
        sigma_min=("sigma_min", "mean"),
        sigma_moyen=("sigma_mean", "mean"),
        sigma_max=("sigma_max", "mean"),
        coord_noise_std=("coord_noise_std", "mean"),
        replace_one_sensitivity=("replace_one_sensitivity", "mean"),
        clipping_serveur=("server_clip_rate", "mean"),
        accuracy=("test_accuracy_pct", "mean"),
        test_loss=("test_loss", "mean"),
        variance=("variance_pp2", "mean"),
        worst20=("worst20_pct", "mean"),
        gap=("gap_pp", "mean"),
    )
    .reset_index()
)
privacy_table["clipping_serveur"] *= 100.0
privacy_table = privacy_table.rename(columns={
    "n_clients": "Clients", "target_epsilon": "ε cible",
    "C_local": "C local", "batch_size": "Batch fixe",
    "U_server": "U serveur",
    "epsilon_realise": "ε réalisé", "delta": "δ",
    "sigma_min": "σ min", "sigma_moyen": "σ moyen", "sigma_max": "σ max",
    "coord_noise_std": "É.-t. bruit/coord. σC/m",
    "replace_one_sensitivity": "Sensibilité 2C/m",
    "clipping_serveur": "Uploads clippés (%)",
    "accuracy": "Accuracy test (%)", "test_loss": "Loss test",
    "variance": "Variance (pp²)", "worst20": "Worst-20 (%)",
    "gap": "Gap (pp)",
})
display(Markdown(
    "**Écran privacy D.** Les cellules sont exploratoires lorsqu'une seule seed "
    "est présente ; elles servent ici à lire le calibrage, pas à estimer une incertitude."
))
display(Markdown("### Ledger du mécanisme DP"))
display(privacy_table[[
    "Clients", "ε cible", "Seeds", "C local", "Batch fixe", "U serveur",
    "σ min", "σ moyen", "σ max", "É.-t. bruit/coord. σC/m",
    "Sensibilité 2C/m", "ε réalisé", "δ",
]].round({
    "ε cible": 3, "ε réalisé": 4, "δ": 7, "σ min": 3,
    "σ moyen": 3, "σ max": 3, "É.-t. bruit/coord. σC/m": 4,
    "Sensibilité 2C/m": 4,
}))
display(Markdown("### Résultats et clipping serveur"))
display(privacy_table[[
    "Clients", "ε cible", "Seeds", "Accuracy test (%)", "Loss test",
    "Variance (pp²)", "Worst-20 (%)", "Gap (pp)",
    "Uploads clippés (%)",
]].round({
    "ε cible": 3, "Accuracy test (%)": 2, "Loss test": 3,
    "Variance (pp²)": 1, "Worst-20 (%)": 2,
    "Gap (pp)": 2, "Uploads clippés (%)": 1,
}))
display(Markdown(
    "**Clipping local sous DP.** Le taux data-dependent n'est volontairement "
    "pas publié par ces runs (`NaN`) ; il n'est donc ni affiché ni remplacé par zéro. "
    "Pour le mécanisme à batch fixe, le runner ajoute au gradient moyen un bruit "
    "par coordonnée d'écart-type $\\sigma C/m$, et la sensibilité replace-one "
    "certifiée du gradient moyen vaut $2C/m$."
))
"""
        ),
        code(
            r"""
if len(privacy_table):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n_clients, group in privacy_table.groupby("Clients"):
        ax.plot(group["ε cible"], group["ε réalisé"], marker="o", label=f"n={int(n_clients)}")
    lo = float(min(privacy_table["ε cible"].min(), privacy_table["ε réalisé"].min()))
    hi = float(max(privacy_table["ε cible"].max(), privacy_table["ε réalisé"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="cible = réalisé")
    ax.set(xlabel="ε cible", ylabel="ε réalisé", title="Vérification du budget de confidentialité")
    ax.legend()
    fig.tight_layout()
    plt.show()
"""
        ),
        code(
            r"""
def _plot_privacy_outcome(metric, ylabel, better):
    if privacy_table.empty or not privacy_table[metric].notna().any():
        display(Markdown(f"**{metric} : données indisponibles.**"))
        return
    client_values = sorted(privacy_table["Clients"].dropna().unique())
    fig, axes = plt.subplots(
        1, len(client_values), figsize=(6.0 * len(client_values), 4.5),
        squeeze=False,
    )
    for ax, n_clients in zip(axes.ravel(), client_values):
        group = privacy_table[privacy_table["Clients"] == n_clients].copy()
        group = group.sort_values("ε cible")
        ax.plot(
            group["ε cible"], group[metric], marker="o", linewidth=2,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"bruit={row['É.-t. bruit/coord. σC/m']:.3f}",
                (row["ε cible"], row[metric]), xytext=(4, 7),
                textcoords="offset points", fontsize=8,
            )
        ax.set(
            xlabel="ε (petit = plus privé)", ylabel=ylabel,
            title=f"n={int(n_clients)} clients",
        )
    fig.suptitle(f"{ylabel} selon le budget de confidentialité — {better} (exploratoire, 1 seed)")
    fig.tight_layout(); plt.show()


_plot_privacy_outcome("Accuracy test (%)", "Accuracy test (%)", "plus haut est meilleur")
_plot_privacy_outcome("Loss test", "Loss test", "plus bas est meilleur")
_plot_privacy_outcome("Variance (pp²)", "Variance inter-clients (pp²)", "plus bas est plus homogène")
_plot_privacy_outcome("Worst-20 (%)", "Worst-20 (%)", "plus haut est meilleur")
_plot_privacy_outcome("Gap (pp)", "Gap best–worst (pp)", "plus bas est meilleur")

display(Markdown(
    "**Lecture.** Pour chaque nombre de clients, $C$, le batch, la référence "
    r"et $\alpha$ sont fixes. Lorsque $\varepsilon$ augmente de 2 à 8, "
    r"$\sigma$ et $\sigma C/m$ diminuent. Les annotations donnent le niveau "
    "de bruit effectivement ajouté par coordonnée. Comme l'écran ne contient "
    "qu'une seed, une pente ne doit pas être présentée comme une loi générale."
))
"""
        ),
        code(
            r"""
if len(privacy_table):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n_clients, group in privacy_table.groupby("Clients"):
        group = group.sort_values("ε cible")
        ax.plot(group["ε cible"], group["σ moyen"], marker="o", label=f"n={int(n_clients)}")
    ax.set(
        xlabel="ε cible (plus petit = plus privé)",
        ylabel="Multiplicateur de bruit σ",
        title="Bruit nécessaire pour chaque budget cible",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
"""
        ),
        code(
            r"""
if len(privacy_table):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n_clients, group in privacy_table.groupby("Clients"):
        group = group.sort_values("ε cible")
        ax.plot(
            group["ε cible"], group["Uploads clippés (%)"], marker="o",
            label=f"n={int(n_clients)}",
        )
    ax.set(
        xlabel="ε cible", ylabel="Uploads clippés au serveur (%)", ylim=(-3, 103),
        title="Clipping serveur dans l'écran privacy D",
    )
    ax.legend()
    fig.tight_layout(); plt.show()

f_clip = final_df[
    (final_df.campaign == READABLE_CAMPAIGN)
    & (final_df.phase == READABLE_PHASE)
    & final_df.dp_enabled
    & (final_df.reference == "F_CC")
    & (final_df.attack == "none")
]
f_server = 100.0 * f_clip.server_clip_rate.dropna()
f_note = (
    f"Dans F-confirmation, le taux serveur observé vaut "
    f"{f_server.min():.1f}–{f_server.max():.1f} %. " if len(f_server) else
    "Dans F-confirmation, le taux serveur n'est pas publié. "
)
display(Markdown(
    "**Note F-confirmation.** " + f_note
    + "Le taux local sous DP reste non publié. Une courbe plate à zéro ne "
    "démontrerait pas l'utilité du clipping serveur ; elle indique seulement "
    "que ce seuil n'est pas actif dans cette cellule."
))

if len(privacy_table):
    deviation = (privacy_table["ε réalisé"] - privacy_table["ε cible"]).abs().max()
    server_values = privacy_table["Uploads clippés (%)"].dropna()
    clip_sentence = (
        f"Dans l'écran D, le clipping serveur varie de {server_values.min():.1f} % à "
        f"{server_values.max():.1f} %." if len(server_values) else
        "Le clipping serveur n'est pas publié dans l'écran D."
    )
    display(Markdown(
        f"**Lecture automatique.** L'écart maximal $|\\varepsilon_{{réalisé}}-"
        f"\\varepsilon_{{cible}}|$ vaut **{deviation:.4g}**. {clip_sentence} "
        "Un taux de 100 % indique que tous les uploads dépassent le seuil serveur ; "
        "cela signale un calibrage géométrique très contraignant, pas un gain d'utilité."
    ))

display(Markdown("## 6.2 Impact conjoint du seuil local C et du bruit absolu"))
b3_clip = final_df[
    (final_df.campaign == READABLE_CAMPAIGN)
    & (final_df.phase == "b3_local_clip_dp_confirmation")
    & (final_df.reference == "F_CC")
    & np.isclose(final_df.alpha, 0.0)
    & (final_df.attack == "none")
].copy()
b3_clip[["batch_size", "C_config", "U_config"]] = b3_clip.metrics_path.apply(
    _privacy_config_fields
)
b3_clip["coord_noise_std"] = (
    b3_clip.sigma_mean * b3_clip.C_config / b3_clip.batch_size
)
b3_summary = (
    b3_clip.groupby(["n_clients", "C_config"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        noise_std=("coord_noise_std", "mean"),
        server_clip_mean=("server_clip_rate", "mean"),
        accuracy_mean=("test_accuracy_pct", "mean"),
        accuracy_sd=("test_accuracy_pct", "std"),
        loss_mean=("test_loss", "mean"),
        loss_sd=("test_loss", "std"),
        variance_mean=("variance_pp2", "mean"),
        variance_sd=("variance_pp2", "std"),
        worst20_mean=("worst20_pct", "mean"),
        worst20_sd=("worst20_pct", "std"),
        gap_mean=("gap_pp", "mean"),
        gap_sd=("gap_pp", "std"),
    )
    .reset_index()
)
b3_summary["server_clip_pct"] = 100.0 * b3_summary.server_clip_mean
display(Markdown(
    r"**B3-confirmation :** trois seeds, $\varepsilon=4$, FCC, $\alpha=0$, "
    "sans attaque. Le taux de clipping local dépendant des données n'est pas "
    r"publié. Le taux serveur et $\sigma C/m$ permettent néanmoins de voir "
    "quand la géométrie du mécanisme change."
))
display(b3_summary.rename(columns={
    "n_clients": "Clients", "C_config": "C local",
    "noise_std": "É.-t. bruit/coord.",
    "server_clip_pct": "Uploads serveur clippés (%)",
    "accuracy_mean": "Accuracy moyenne", "loss_mean": "Loss moyenne",
    "variance_mean": "Variance moyenne",
    "worst20_mean": "Worst-20 moyen", "gap_mean": "Gap moyen",
})[[
    "Clients", "C local", "Seeds", "É.-t. bruit/coord.",
    "Uploads serveur clippés (%)", "Accuracy moyenne", "Loss moyenne",
    "Variance moyenne", "Worst-20 moyen", "Gap moyen",
]].round(3))


def _plot_clipping_outcome(mean_col, sd_col, ylabel, better):
    if b3_summary.empty or not b3_summary[mean_col].notna().any():
        display(Markdown(f"**{ylabel} : données indisponibles dans B3.**"))
        return
    client_values = sorted(b3_summary.n_clients.unique())
    fig, axes = plt.subplots(
        1, len(client_values), figsize=(5.8 * len(client_values), 4.4),
        squeeze=False,
    )
    for ax, n_clients in zip(axes.ravel(), client_values):
        group = b3_summary[b3_summary.n_clients == n_clients].sort_values("C_config")
        ax.errorbar(
            group.C_config, group[mean_col], yerr=group[sd_col], marker="o",
            linewidth=2, capsize=4,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"bruit={row.noise_std:.3f}\nclip srv={row.server_clip_pct:.0f}%",
                (row.C_config, row[mean_col]), xytext=(4, 7),
                textcoords="offset points", fontsize=8,
            )
        ax.set(xlabel="Seuil local C", ylabel=ylabel, title=f"n={int(n_clients)} clients")
    fig.suptitle(f"{ylabel} selon C — {better} (3 seeds, ε=4)")
    fig.tight_layout(); plt.show()


_plot_clipping_outcome("accuracy_mean", "accuracy_sd", "Accuracy test (%)", "plus haut est meilleur")
_plot_clipping_outcome("loss_mean", "loss_sd", "Loss test", "plus bas est meilleur")
_plot_clipping_outcome("variance_mean", "variance_sd", "Variance inter-clients (pp²)", "plus bas est plus homogène")
_plot_clipping_outcome("worst20_mean", "worst20_sd", "Worst-20 (%)", "plus haut est meilleur")
_plot_clipping_outcome("gap_mean", "gap_sd", "Gap best–worst (pp)", "plus bas est meilleur")

for metric, ylabel in [("noise_std", "É.-t. du bruit par coordonnée σC/m"),
                         ("server_clip_pct", "Uploads clippés au serveur (%)")]:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), squeeze=False)
    for ax, n_clients in zip(axes.ravel(), sorted(b3_summary.n_clients.unique())):
        group = b3_summary[b3_summary.n_clients == n_clients].sort_values("C_config")
        ax.plot(group.C_config, group[metric], marker="o", linewidth=2)
        ax.set(xlabel="Seuil local C", ylabel=ylabel, title=f"n={int(n_clients)} clients")
        if metric == "server_clip_pct":
            ax.set_ylim(-3, 103)
    fig.suptitle(ylabel + " selon le seuil local C")
    fig.tight_layout(); plt.show()

display(Markdown(
    r"**Interprétation causale.** À $\varepsilon=4$, le multiplicateur "
    r"$\sigma$ reste fixé mais $\sigma C/m$ augmente avec $C$. Une mauvaise "
    "métrique à grand $C$ peut donc venir du bruit absolu plus fort, de "
    "l'activation du clipping serveur, ou de leur interaction. Cette campagne "
    "compare les mécanismes complets associés à chaque $C$ ; elle n'identifie "
    "pas l'effet pur du clipping local. Une ablation causale demanderait des "
    "streams aléatoires séparés et un facteur maintenu fixe."
))
"""
        ),
        markdown(
            r"""
## 7. Attaques byzantines et références robustes

La comparaison principale utilise uniquement la confirmation F : DP avec
$\varepsilon=4$, $\alpha=2$, trois seeds, 20 % de clients byzantins dans les
bras attaqués. Les références CM, FCC, RFA et trMean sont évaluées sur les
mêmes cellules expérimentales.

Sens de lecture :

- **accuracy test** et **Worst-20 honnête** : plus grand est meilleur ;
- **gap honnête**, **erreur de référence oracle** et **masse byzantine oracle** :
  plus petit est meilleur ;
- les métriques `honest_*` retirent les identifiants byzantins de l'évaluation
  de fairness ;
- les métriques oracle servent à expliquer le simulateur et ne sont pas
  disponibles au serveur en situation réelle.
"""
        ),
        code(
            r"""
attack_focus = final_df[
    (final_df.campaign == READABLE_CAMPAIGN)
    & (final_df.phase == READABLE_PHASE)
    & final_df.dp_enabled
    & np.isclose(final_df.target_epsilon, 4.0)
    & np.isclose(final_df.alpha, 2.0)
].copy()

attack_summary_readable = (
    attack_focus.groupby(["n_clients", "attack", "reference"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        accuracy=("test_accuracy_pct", "mean"),
        accuracy_sd=("test_accuracy_pct", "std"),
        worst20=("honest_worst20_pct", "mean"),
        gap=("honest_gap_pp", "mean"),
        reference_error=("reference_error_oracle", "mean"),
        byzantine_mass=("byzantine_mass_oracle", "mean"),
        byzantine_displacement=("byzantine_displacement_norm_oracle", "mean"),
        aggregate_error=("aggregate_error_oracle", "mean"),
    )
    .reset_index()
)

for n_clients in sorted(attack_summary_readable.n_clients.unique()):
    table = attack_summary_readable[attack_summary_readable.n_clients == n_clients].copy()
    table = table.rename(columns={
        "attack": "Attaque", "reference": "Référence",
        "accuracy": "Accuracy test (%)", "accuracy_sd": "Écart-type accuracy",
        "worst20": "Worst-20 honnête (%)", "gap": "Gap honnête (pp)",
        "reference_error": "Erreur référence (oracle)",
        "byzantine_mass": "Masse byzantine (oracle)",
        "byzantine_displacement": "Déplacement byzantin (oracle)",
        "aggregate_error": "Erreur agrégat (oracle)",
    })
    display(Markdown(f"### Utilité et fairness — n={int(n_clients)} clients"))
    display(table[[
        "Attaque", "Référence", "Seeds", "Accuracy test (%)", "Écart-type accuracy",
        "Worst-20 honnête (%)", "Gap honnête (pp)",
    ]].round(3))
    display(Markdown(f"### Diagnostics oracle de robustesse — n={int(n_clients)} clients"))
    display(table[[
        "Attaque", "Référence", "Seeds", "Erreur référence (oracle)",
        "Masse byzantine (oracle)", "Déplacement byzantin (oracle)",
        "Erreur agrégat (oracle)",
    ]].round(4))
"""
        ),
        code(
            r"""
# Deltas strictement appariés à FCC au niveau seed. Une valeur positive de
# Δ accuracy/Worst-20 favorise la référence; une valeur négative de Δ gap la favorise.
paired_keys = ["n_clients", "attack", "seed"]
paired_source = attack_focus[[
    *paired_keys, "reference", "test_accuracy_pct", "honest_worst20_pct", "honest_gap_pp",
]].copy()
paired_wide = paired_source.pivot_table(
    index=paired_keys, columns="reference",
    values=["test_accuracy_pct", "honest_worst20_pct", "honest_gap_pp"],
    aggfunc="first",
)
paired_rows = []
for reference in ["CM", "RFA", "trMean"]:
    required = [(metric, label) for metric, label in [
        ("test_accuracy_pct", "Δ accuracy (pp)"),
        ("honest_worst20_pct", "Δ Worst-20 (pp)"),
        ("honest_gap_pp", "Δ gap (pp)"),
    ] if (metric, reference) in paired_wide and (metric, "F_CC") in paired_wide]
    if not required:
        continue
    for keys, row in paired_wide.iterrows():
        output = {"Clients": int(keys[0]), "Attaque": keys[1], "Seed": int(keys[2]), "Référence": reference}
        for metric, label in required:
            output[label] = row[(metric, reference)] - row[(metric, "F_CC")]
        paired_rows.append(output)
paired_delta_runs = pd.DataFrame(paired_rows)
if len(paired_delta_runs):
    paired_delta_summary = (
        paired_delta_runs.groupby(["Clients", "Attaque", "Référence"], dropna=False)
        .agg(
            Seeds=("Seed", "nunique"),
            delta_accuracy=("Δ accuracy (pp)", "mean"),
            delta_worst20=("Δ Worst-20 (pp)", "mean"),
            delta_gap=("Δ gap (pp)", "mean"),
        )
        .reset_index()
        .rename(columns={
            "delta_accuracy": "Δ accuracy vs FCC (pp)",
            "delta_worst20": "Δ Worst-20 vs FCC (pp)",
            "delta_gap": "Δ gap vs FCC (pp)",
        })
    )
    display(Markdown(
        "### Différences appariées à FCC\n\n"
        "Chaque différence est d'abord calculée pour la même seed, puis moyennée. "
        "Ainsi, une différence de seed ou de partition ne peut pas expliquer le signe."
    ))
    for n_clients in sorted(paired_delta_summary.Clients.unique()):
        display(Markdown(f"**n={int(n_clients)} clients**"))
        display(paired_delta_summary[paired_delta_summary.Clients == n_clients].drop(columns="Clients").round(3))
"""
        ),
        code(
            r"""
def _plot_attack_reference(metric, ylabel, higher_is_better):
    if attack_summary_readable.empty or not attack_summary_readable[metric].notna().any():
        display(Markdown(f"**{ylabel} : données indisponibles.**"))
        return
    for n_clients in sorted(attack_summary_readable.n_clients.unique()):
        view = attack_summary_readable[attack_summary_readable.n_clients == n_clients]
        pivot = view.pivot(index="attack", columns="reference", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9, 4.8), width=0.78)
        direction = "plus grand = meilleur" if higher_is_better else "plus petit = meilleur"
        ax.set(
            xlabel="Attaque", ylabel=ylabel,
            title=f"{ylabel} — n={int(n_clients)} ({direction})",
        )
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Référence", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure(); fig.tight_layout(); plt.show()


_plot_attack_reference("accuracy", "Accuracy test (%)", True)
"""
        ),
        code(
            r"""
_plot_attack_reference("worst20", "Worst-20 honnête (%)", True)
"""
        ),
        code(
            r"""
_plot_attack_reference("gap", "Gap honnête (pp)", False)
"""
        ),
        code(
            r"""
_plot_attack_reference("reference_error", "Erreur de référence oracle", False)
"""
        ),
        code(
            r"""
_plot_attack_reference("byzantine_mass", "Masse de poids byzantine oracle", False)

if len(attack_summary_readable):
    attacked_only = attack_summary_readable[attack_summary_readable.attack != "none"]
    winners = []
    for (n_clients, attack), group in attacked_only.groupby(["n_clients", "attack"]):
        winners.append({
            "Clients": int(n_clients),
            "Attaque": attack,
            "Meilleure accuracy": group.loc[group.accuracy.idxmax(), "reference"],
            "Meilleur Worst-20": group.loc[group.worst20.idxmax(), "reference"],
            "Plus petite erreur de référence": group.loc[group.reference_error.idxmin(), "reference"],
        })
    display(Markdown(
        "**Lecture automatique, critère par critère.** Des gagnants différents "
        "montrent qu'une petite erreur de centre ne garantit pas automatiquement "
        "la meilleure accuracy ou fairness end-to-end."
    ))
    display(pd.DataFrame(winners))
"""
        ),
        markdown(
            r"""
## 8. Diagnostics oracle : signal propre, bruit DP et déplacement byzantin

Ces diagnostics ne sont connus que dans le simulateur :

| Diagnostic | Définition opérationnelle | Lecture |
|---|---|---|
| corrélation poids–bruit | corrélation entre poids FAR et amplitude de bruit effectif par client | proche de 0 : pas d'alignement linéaire ; positive : les clients plus bruités tendent à être plus pondérés |
| corrélation score bruité–score propre | fidélité du classement géométrique après ajout du bruit | proche de 1 : le score bruité conserve mieux la géométrie propre |
| rappel de la queue honnête | part des honest outliers propres retrouvés par le score bruité | plus grand est meilleur pour l'objectif FAR d'inclusion |
| biais de tilting honnête | déplacement dû à la repondération des seuls messages honnêtes propres | coût géométrique de fairness |
| bruit DP à poids fixes | contribution du bruit lorsque les poids sont figés | bruit incompressible à pondération donnée |
| déplacement byzantin | contribution vectorielle des messages byzantins | effet direct de l'attaque |
| erreur totale | distance de l'agrégat au centre honnête propre | résultat combiné, donc non égal en général à la somme des normes précédentes |

Pour garder les figures lisibles, cette section fixe FCC, DP
$\varepsilon=4$, $\alpha=2$ et la confirmation F. Chaque graphique varie
seulement l'attaque et le nombre de clients.
"""
        ),
        code(
            r"""
# D2-confirmation isole le problème score/bruit sous bruit hétéroscédastique,
# sans attaque. Les quantités sont des médianes sur les tours de chaque run,
# ensuite moyennées entre les trois seeds.
d2_oracle = positioning_run_level[
    (positioning_run_level.phase == "d2_noise_assignment_confirmation")
    & (positioning_run_level.robust_reference == "centered_clipping")
    & (positioning_run_level.attack == "none")
].copy()
d2_oracle["niveau_bruit"] = np.where(
    d2_oracle.variant_id.str.contains("strong", na=False), "fort", "modéré"
)
d2_oracle["assignation"] = np.where(
    d2_oracle.variant_id.str.contains("reverse", na=False), "inversée", "directe"
)
d2_oracle["condition_bruit"] = d2_oracle.niveau_bruit + " · " + d2_oracle.assignation

d2_oracle_summary = (
    d2_oracle.groupby(["num_clients", "condition_bruit"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        corr_weight_noise=("weight_dp_noise_corr_median_oracle", "mean"),
        corr_distance_noise=("distance_dp_noise_corr_median_oracle", "mean"),
        corr_score_clean=("honest_noisy_clean_score_corr_median_oracle", "mean"),
        top_tail_recall=("honest_clean_top_tail_recall_median_oracle", "mean"),
    )
    .reset_index()
)
d2_table = d2_oracle_summary.rename(columns={
    "num_clients": "Clients", "condition_bruit": "Bruit / assignation",
    "corr_weight_noise": "Corr. poids–bruit DP",
    "corr_distance_noise": "Corr. distance–bruit DP",
    "corr_score_clean": "Corr. score bruité–propre",
    "top_tail_recall": "Rappel honest outliers",
})
d2_table["Rappel honest outliers"] *= 100.0
display(Markdown("### D2-confirmation — fidélité du score sous bruit hétéroscédastique"))
display(d2_table.round(3))
"""
        ),
        code(
            r"""
if len(d2_oracle_summary):
    for n_clients in sorted(d2_oracle_summary.num_clients.unique()):
        view = d2_oracle_summary[d2_oracle_summary.num_clients == n_clients].set_index("condition_bruit")
        ax = view[["corr_score_clean", "top_tail_recall"]].rename(columns={
            "corr_score_clean": "Corr. score bruité–propre",
            "top_tail_recall": "Rappel honest outliers",
        }).plot(kind="bar", figsize=(8.5, 4.5))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Niveau / assignation du bruit", ylabel="Corrélation ou rappel",
            ylim=(-1, 1), title=f"Le score retrouve-t-il la géométrie propre ? — n={int(n_clients)}",
        )
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Oracle", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure(); fig.tight_layout(); plt.show()
"""
        ),
        code(
            r"""
if len(d2_oracle_summary):
    for n_clients in sorted(d2_oracle_summary.num_clients.unique()):
        view = d2_oracle_summary[d2_oracle_summary.num_clients == n_clients].set_index("condition_bruit")
        ax = view[["corr_weight_noise", "corr_distance_noise"]].rename(columns={
            "corr_weight_noise": "Poids–bruit DP",
            "corr_distance_noise": "Distance–bruit DP",
        }).plot(kind="bar", figsize=(8.5, 4.5))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Niveau / assignation du bruit", ylabel="Corrélation",
            ylim=(-1, 1), title=f"Le score brut suit-il le niveau de bruit ? — n={int(n_clients)}",
        )
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Oracle", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure(); fig.tight_layout(); plt.show()

    median_weight_noise = d2_oracle_summary.corr_weight_noise.median()
    median_score_clean = d2_oracle_summary.corr_score_clean.median()
    median_recall = d2_oracle_summary.top_tail_recall.median()
    display(Markdown(
        f"**Lecture automatique D2.** La corrélation poids–bruit médiane vaut "
        f"**{median_weight_noise:.3f}**, tandis que la corrélation score bruité–propre "
        f"médiane vaut **{median_score_clean:.3f}** et le rappel médian "
        f"**{100*median_recall:.1f} %**. Cela quantifie séparément l'alignement au "
        "bruit et la conservation du signal propre ; ce n'est pas une preuve causale "
        "hors du simulateur."
    ))
"""
        ),
        code(
            r"""
# F/T20 fournit ensuite la décomposition sous attaques. Le terme de tilting
# est q_H ||B_tilt|| : la masse honnête q_H est nécessaire pour retrouver la
# contribution réellement présente dans l'identité vectorielle.
f_oracle = positioning_run_level[
    (positioning_run_level.phase == READABLE_PHASE)
    & positioning_run_level.dp_enabled.astype(str).str.lower().eq("true")
    & np.isclose(positioning_run_level.target_epsilon, 4.0)
    & np.isclose(positioning_run_level.far_alpha, 2.0)
    & (positioning_run_level.robust_reference == "centered_clipping")
].copy()
f_oracle["weighted_tilting"] = (
    f_oracle.honest_weight_mass_median_oracle
    * f_oracle.honest_clean_tilting_bias_norm_median_oracle
)
oracle_readable = (
    f_oracle.groupby(["num_clients", "attack"], dropna=False)
    .agg(
        Seeds=("seed", "nunique"),
        weighted_tilting=("weighted_tilting", "mean"),
        dp_noise=("honest_fixed_weight_dp_noise_norm_median_oracle", "mean"),
        byzantine_shift=("byzantine_displacement_norm_median_oracle", "mean"),
        aggregate_error=("aggregate_error_to_clean_honest_center_median_oracle", "mean"),
        residual=("error_decomposition_residual_norm_median_oracle", "max"),
    )
    .reset_index()
)
oracle_table = oracle_readable.rename(columns={
    "num_clients": "Clients", "attack": "Attaque",
    "weighted_tilting": "q_H × biais tilting",
    "dp_noise": "Bruit DP à poids fixes",
    "byzantine_shift": "Déplacement byzantin",
    "aggregate_error": "Erreur agrégat",
    "residual": "Résidu décomposition",
})
display(Markdown("### F-confirmation — décomposition des contributions"))
display(oracle_table.round(4))
"""
        ),
        code(
            r"""
if len(oracle_readable):
    component_labels = {
        "weighted_tilting": "q_H × biais tilting",
        "dp_noise": "Bruit DP (poids fixes)",
        "byzantine_shift": "Déplacement byzantin",
    }
    for n_clients in sorted(oracle_readable.num_clients.unique()):
        view = oracle_readable[oracle_readable.num_clients == n_clients].set_index("attack")
        ax = view[list(component_labels)].rename(columns=component_labels).plot(
            kind="bar", figsize=(9.2, 4.8), width=0.72
        )
        x = np.arange(len(view))
        ax.scatter(
            x, view.aggregate_error, color="black", marker="D", s=42,
            label="Erreur totale", zorder=5,
        )
        ax.set(
            xlabel="Attaque", ylabel="Norme ℓ2",
            title=f"Contributions vectorielles et erreur totale — FCC, n={int(n_clients)}",
        )
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Composante", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig = ax.get_figure(); fig.tight_layout(); plt.show()

    max_residual = oracle_readable.residual.max()
    dominant = oracle_readable[["weighted_tilting", "dp_noise", "byzantine_shift"]].median().idxmax()
    dominant_label = {
        "weighted_tilting": "la contribution de tilting honnête",
        "dp_noise": "le bruit DP à poids fixes",
        "byzantine_shift": "le déplacement byzantin",
    }[dominant]
    display(Markdown(
        f"**Lecture automatique F.** La composante médiane la plus grande est "
        f"**{dominant_label}**. Le résidu maximal vaut **{max_residual:.2e}** : "
        "il contrôle l'identité numérique, mais n'est pas une métrique de qualité. "
        "Les barres ne sont pas empilées, car les vecteurs peuvent se compenser et "
        "leurs normes ne s'additionnent pas."
    ))
else:
    display(Markdown("**Décomposition oracle indisponible dans F-confirmation.**"))
"""
        ),
    ]


def _first_line(cell: nbformat.NotebookNode) -> str:
    return str(cell.get("source", "")).lstrip().splitlines()[0] if cell.get("source") else ""


def backup_notebook(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("/tmp") / f"{path.name}.{stamp}.readability.bak"
    shutil.copy2(path, destination)
    return destination


def update(notebook_path: Path) -> tuple[Path, int, int]:
    notebook = nbformat.read(notebook_path, as_version=4)
    before_g0e = [
        cell.get("source", "")
        for cell in notebook.cells
        if "ldp-gaussian-aware-g0e-2026-09"
        in set(cell.get("metadata", {}).get("tags", []))
    ]

    starts = [i for i, cell in enumerate(notebook.cells) if _first_line(cell).startswith("## 3.")]
    ends = [i for i, cell in enumerate(notebook.cells) if _first_line(cell).startswith("## 9.")]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise RuntimeError(
            f"Expected one ordered section 3/9 boundary, got starts={starts}, ends={ends}"
        )
    start, end = starts[0], ends[0]
    inserted = replacement_cells()
    notebook.cells = notebook.cells[:start] + inserted + notebook.cells[end:]

    after_g0e = [
        cell.get("source", "")
        for cell in notebook.cells
        if "ldp-gaussian-aware-g0e-2026-09"
        in set(cell.get("metadata", {}).get("tags", []))
    ]
    if before_g0e != after_g0e:
        raise AssertionError("G0e cells changed during readability update")

    owned = [
        cell for cell in notebook.cells
        if SECTION_TAG in set(cell.get("metadata", {}).get("tags", []))
    ]
    if len(owned) != len(inserted):
        raise AssertionError("Readability cells were duplicated")

    backup = backup_notebook(notebook_path)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    return backup, len(notebook.cells), len(inserted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    backup, total, inserted = update(notebook_path)
    print(f"Backup: {backup}")
    print(f"Notebook updated without execution: {notebook_path}")
    print(f"Inserted/replaced cells: {inserted}; total cells: {total}")


if __name__ == "__main__":
    main()
