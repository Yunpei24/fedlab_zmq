#!/usr/bin/env python3
"""Add the alpha-cap and DP-noise-scale diagnostics to the decisive notebook.

The update is idempotent: generated cells are tagged and replaced on every run.
The notebook remains the executable artifact; this script records how the two
scientific sections were constructed so they are not lost during maintenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "DT_LDP_FAR_N10_Decisive_Analysis.ipynb"
GENERATED_TAG = "dtldp-alpha-noise-scale-2026-09"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        source.strip(), metadata={"tags": [GENERATED_TAG]}
    )


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(
        source.strip(), metadata={"tags": [GENERATED_TAG]}
    )


def starts_with(cell: nbformat.NotebookNode, prefix: str) -> bool:
    return str(cell.get("source", "")).lstrip().startswith(prefix)


def insert_before(
    cells: list[nbformat.NotebookNode],
    prefix: str,
    additions: list[nbformat.NotebookNode],
) -> None:
    index = next(i for i, cell in enumerate(cells) if starts_with(cell, prefix))
    cells[index:index] = additions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute every notebook cell after updating the generated sections.",
    )
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cells = [
        cell
        for cell in notebook.cells
        if GENERATED_TAG not in cell.get("metadata", {}).get("tags", [])
    ]

    intro = next(cell for cell in cells if starts_with(cell, "# DT-LDP-FAR"))
    if "5. le score FAR confond-il" not in str(intro.source):
        intro.source = str(intro.source).replace(
            "4. F_CC est-elle empiriquement la meilleure référence ?",
            "4. F_CC est-elle empiriquement la meilleure référence ?\n"
            "5. le score FAR confond-il hétérogénéité utile et niveau de bruit DP ?",
        )
    if "La cinquième question utilise" not in str(intro.source):
        intro.source = str(intro.source).replace(
            "Elles ne doivent pas être attribuées à la campagne n=10.",
            "Elles ne doivent pas être attribuées à la campagne n=10. La cinquième "
            "question utilise le stress mécanistique Stage 5 à 25 clients et trois seeds.",
        )

    alpha_cells = [
        markdown(
            r"""
### Dégradation explicite au-delà de $\alpha_{\max}$

Les figures précédentes donnent les valeurs absolues. La cellule suivante prend
la lane certifiée $\alpha=\alpha_{\max}$ comme référence et calcule, pour
$2\alpha_{\max}$ et $4\alpha_{\max}$ :

\[
\Delta M(r)=M(r\alpha_{\max})-M(\alpha_{\max}),\qquad r\in\{2,4\}.
\]

Une baisse de Test Accuracy ou de Worst-20 est défavorable. Une hausse de la
variance, du gap, du poids maximal ou de $H=n\sum_i\omega_i^2$ est également
défavorable. Les lanes $r>1$ sont des diagnostics non certifiés : elles montrent
ce que le cap empêche, sans revendiquer la garantie $\max_i\omega_i\leq\kappa_w/n$.
"""
        ),
        code(
            r"""
comparison_keys = ["partition_label", "variant_label"]
cap_rows = n10[np.isclose(n10.tilt_multiple, 1.0)].copy()
over_cap = n10[n10.tilt_multiple > 1.0].copy()

cap_metrics = [
    "test_acc", "worst20", "variance_pp2", "gap", "max_weight", "H"
]
alpha_degradation = over_cap.merge(
    cap_rows[comparison_keys + cap_metrics],
    on=comparison_keys,
    how="inner",
    suffixes=("", "_at_alpha_max"),
)

for metric in cap_metrics:
    alpha_degradation[f"delta_{metric}"] = (
        alpha_degradation[metric]
        - alpha_degradation[f"{metric}_at_alpha_max"]
    )

alpha_degradation["degradation_indicators"] = (
    (alpha_degradation.delta_test_acc < 0).astype(int)
    + (alpha_degradation.delta_worst20 < 0).astype(int)
    + (alpha_degradation.delta_variance_pp2 > 0).astype(int)
    + (alpha_degradation.delta_gap > 0).astype(int)
    + (alpha_degradation.delta_max_weight > 0).astype(int)
    + (alpha_degradation.delta_H > 0).astype(int)
)

alpha_degradation_table = alpha_degradation.rename(columns={
    "partition_label": "Partition",
    "variant_label": "Variante",
    "tilt_multiple": "α / α_max",
    "delta_test_acc": "Δ Test Acc. (pp)",
    "delta_worst20": "Δ Worst-20 (pp)",
    "delta_variance_pp2": "Δ variance (pp²)",
    "delta_gap": "Δ gap (pp)",
    "delta_max_weight": "Δ poids maximal",
    "delta_H": "Δ H",
    "degradation_indicators": "Critères dégradés / 6",
})[[
    "Partition", "Variante", "α / α_max", "Δ Test Acc. (pp)",
    "Δ Worst-20 (pp)", "Δ variance (pp²)", "Δ gap (pp)",
    "Δ poids maximal", "Δ H", "Critères dégradés / 6",
]].sort_values(["Partition", "Variante", "α / α_max"])

display(alpha_degradation_table.round(4))
alpha_degradation_table.to_csv(
    EXPORT_ROOT / "n10_alpha_above_cap_degradation.csv", index=False
)

delta_panels = [
    ("delta_test_acc", "Δ Test Accuracy (pp)"),
    ("delta_worst20", "Δ Worst-20 (pp)"),
    ("delta_variance_pp2", "Δ variance (pp²)"),
    ("delta_gap", "Δ gap (pp)"),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (column, title) in zip(axes.ravel(), delta_panels):
    for (partition, variant), group in alpha_degradation.groupby(
        ["partition_label", "variant_label"]
    ):
        group = group.sort_values("tilt_multiple")
        ax.plot(
            group.tilt_multiple,
            group[column],
            marker=markers[partition],
            color=colors[variant],
            linestyle="-" if "contrôlées" in partition else "--",
            label=f"{variant} — {partition}",
        )
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_xticks([2, 4])
    ax.set_xlabel("α / α_max")
    ax.set_title(title)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
    ncol=2, fontsize=9
)
fig.suptitle(
    "Dégradation par rapport à la frontière certifiée α = α_max",
    y=1.01,
    fontsize=15,
)
save_show(fig, "n10_alpha_above_cap_degradation.png", rect=(0, 0.10, 1, 0.98));
"""
        ),
        markdown(
            r"""
**Lecture.** Le tableau ne suppose pas que toutes les métriques doivent être
monotones pour chaque seed. Il montre directement quelles métriques se détériorent
par rapport à la frontière certifiée. Le résultat le plus net apparaît sous la
partition Dirichlet par classe : lorsque $\alpha$ passe de $\alpha_{\max}$ à
$4\alpha_{\max}$, Worst-20 chute fortement, tandis que la variance, le gap et la
concentration des poids augmentent. Le contrôle de $\alpha$ est donc à la fois un
certificat d'influence et un garde-fou empirique contre un tilting agressif.
"""
        ),
    ]
    insert_before(cells, "## 3. Le retard", alpha_cells)

    for cell in cells:
        if starts_with(cell, "## 6. Tableau de synthèse"):
            cell.source = str(cell.source).replace("## 6.", "## 7.", 1)
        elif starts_with(cell, "## 7. Limites"):
            cell.source = str(cell.source).replace("## 7.", "## 8.", 1)

    noise_cells = [
        markdown(
            r"""
## 6. Le score FAR confond-il signal utile et niveau de bruit DP ?

Le stress Stage 5 à $n=25$ compare 12 paires courant/retardé, avec les mêmes
partitions et les mêmes tirages aléatoires. Pour chaque client, le diagnostic
mesure d'abord la corrélation brute entre le poids courant et la norme de la
perturbation DP effective. Il recalcule ensuite cette corrélation après division
de la perturbation par son échelle DP publique.

Dans le cas isotrope, une première correction de score serait

\[
\widetilde d_{i,t}
=
\frac{\lVert X_{i,t}-F_t\rVert_2}
{\sqrt{\nu_{i,t}^2+\nu_{F,t}^2+\lambda}},
\]

et sa généralisation covariance-aware serait

\[
\widetilde d_{i,t}^{,2}
=
(X_{i,t}-F_t)^\top
(\Sigma_{i,t}+\Sigma_{F,t}+\lambda I)^{-1}
(X_{i,t}-F_t).
\]

La campagne ne teste pas encore cette nouvelle pondération. Elle teste le
diagnostic qui la motive : une corrélation brute disparaissant ou changeant de
signe après correction indique que le score euclidien confond l'échelle de bruit
avec l'écart utile du client.
"""
        ),
        code(
            r"""
STAGE5_DIR = REPO_ROOT / "output" / "analysis" / "dt_ldp_far_n25_stage5_stress"
candidate_path = STAGE5_DIR / "candidate_summary.csv"
pair_path = STAGE5_DIR / "paired_seed_gate.csv"
if not candidate_path.exists() or not pair_path.exists():
    raise FileNotFoundError(
        "Exécuter scripts/analyze_dt_ldp_far_stage5_stress.py avant cette cellule."
    )

stage5_candidates = pd.read_csv(candidate_path)
stage5_pairs = pd.read_csv(pair_path)

def stage5_label(row):
    if "hetero" in str(row["privacy"]):
        return "Score complet — bruit hétéroscédastique"
    if "coord256" in str(row["geometry"]):
        return "256 coordonnées publiques"
    if "last" in str(row["geometry"]):
        return "Dernière couche"
    return "Score complet — bruit homogène"

stage5_candidates["Configuration"] = stage5_candidates.apply(stage5_label, axis=1)
stage5_candidates["mean_delta_test_pp"] = (
    100 * stage5_candidates.mean_delta_test_delayed_minus_current
)

noise_hypothesis_table = stage5_candidates.rename(columns={
    "score_subspace_mode": "Sous-espace",
    "score_subspace_dimension": "Dimension",
    "seeds_passing_core_gate": "Seeds passant le gate / 3",
    "mean_logit_span": "Plage moyenne des logits",
    "mean_effective_noise_corr": "Corrélation brute poids–bruit",
    "mean_normalized_effective_noise_corr": "Corrélation après correction d'échelle",
    "mean_concentration": "Concentration H",
    "mean_current_delay_noise_ratio": "Erreur bruit courant / retardé",
    "mean_delta_test_pp": "Δ accuracy retardé−courant (pp)",
})[[
    "Configuration", "Sous-espace", "Dimension",
    "Plage moyenne des logits", "Corrélation brute poids–bruit",
    "Corrélation après correction d'échelle", "Concentration H",
    "Erreur bruit courant / retardé", "Δ accuracy retardé−courant (pp)",
    "Seeds passant le gate / 3",
]].sort_values("Configuration")

display(noise_hypothesis_table.round(4))
noise_hypothesis_table.to_csv(
    EXPORT_ROOT / "n25_noise_scale_hypothesis_summary.csv", index=False
)

stage5_pairs["Configuration"] = stage5_pairs.apply(stage5_label, axis=1)
plot_pairs = stage5_pairs.sort_values(["Configuration", "training_seed"]).copy()
positions = np.arange(len(plot_pairs))
width = 0.38
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(
    positions - width / 2,
    plot_pairs.effective_noise_corr_median,
    width,
    label="Corrélation brute",
    color="#D95F02",
)
ax.bar(
    positions + width / 2,
    plot_pairs.normalized_effective_noise_corr_median,
    width,
    label="Après correction par l'échelle DP publique",
    color="#1B9E77",
)
ax.axhline(0.0, color="black", linewidth=0.9)
ax.set_xticks(positions)
ax.set_xticklabels(
    [f"{label}\nseed {seed}" for label, seed in zip(
        plot_pairs.Configuration, plot_pairs.training_seed
    )],
    rotation=35,
    ha="right",
)
ax.set_ylabel("Corrélation médiane poids–perturbation effective")
ax.set_title("La correction d'échelle isole l'effet du tirage frais du bruit")
ax.legend()
save_show(fig, "n25_raw_vs_noise_scale_corrected_correlation.png");
"""
        ),
        markdown(
            r"""
### Lecture et décision

Dans le régime hétéroscédastique, la corrélation brute est très forte et positive
($\simeq0{,}947$), mais elle devient fortement négative après correction par
l'échelle publique ($\simeq-0{,}771$). Le score FAR euclidien réagit donc surtout
au **niveau de bruit assigné au client**, et non à une réalisation fraîche
accidentellement grande. Cette observation motive un score standardisé ; elle ne
démontre pas encore que ce score améliorera accuracy ou fairness.

Les trois mécanismes restent complémentaires :

- la standardisation par $\Sigma_i$ rend les distances comparables entre clients ;
- le clipping serveur borne la norme des uploads déjà privés et l'influence des
  messages de grande norme ;
- le cap sur $\alpha$ borne la concentration produite par la softmax, même si les
  scores standardisés sont imparfaits ou attaqués.

La standardisation ne remplace donc ni le clipping serveur ni le contrôle de
$\alpha$. La prochaine expérience doit comparer, à randomness appariée, le score
euclidien brut, le score corrigé par l'échelle isotrope et la distance de
Mahalanobis, sans sélectionner la variante d'après son accuracy finale.
"""
        ),
    ]
    insert_before(cells, "## 7. Tableau de synthèse", noise_cells)

    verdict_cell = next(
        cell for cell in cells if starts_with(cell, "verdicts = pd.DataFrame")
    )
    source = str(verdict_cell.source)
    marker = "    {\n        \"Question\": \"F_CC est-elle empiriquement la meilleure ?\""
    insertion = """    {
        \"Question\": \"Le score doit-il être standardisé par le bruit DP ?\",
        \"Périmètre\": \"Stage 5, n=25, 4 configurations, 3 seeds\",
        \"Verdict\": \"Hypothèse fortement motivée, méthode pas encore testée\",
        \"Réserve\": \"Le diagnostic utilise un oracle ; comparer les scores dans une campagne dédiée\",
    },
"""
    if insertion.strip() not in source:
        position = source.index(marker)
        source = source[:position] + insertion + source[position:]
    verdict_cell.source = source

    limits_cell = next(cell for cell in cells if starts_with(cell, "## 8. Limites"))
    limits_cell.source = str(limits_cell.source).rstrip() + (
        "\n- Stage 5 soutient un **confondant d'échelle DP**, pas encore la supériorité "
        "d'une pondération covariance-aware.\n"
        "- Le diagnostic normalisé emploie la perturbation contrefactuelle oracle ; "
        "il sert à l'analyse mécanistique et ne doit pas être publié dans le transcript LDP.\n"
    )

    notebook.cells = cells
    if args.execute:
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
    nbformat.write(notebook, NOTEBOOK)
    action = "mis à jour et exécuté" if args.execute else "mis à jour"
    print(f"Notebook {action} : {NOTEBOOK}")


if __name__ == "__main__":
    main()
