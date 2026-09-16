#!/usr/bin/env python3
"""Surgical, repeatable update of the existing positioning notebook.

Retains all unrelated cell sources/IDs, including user comments. The notebook
is an output artifact; this script does not change any experiment or gate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis.ipynb"
TAG = "positioning-rcig-explanation-20260912"


def cell(kind, source, key):
    result = getattr(nbformat.v4, f"new_{kind}_cell")(source.strip())
    result.id = "rcig-review-" + key
    result.metadata["tags"] = [TAG]
    return result


def md(source, key):
    return cell("markdown", source, key)


def code(source, key):
    return cell("code", source, key)


TRAJECTORY_INTRO = r"""
## 3. Trajectoires sur les jeux d'évaluation (held-out)

**Held-out** signifie que les exemples d'évaluation sont mis de côté : ils ne
servent pas au calcul des gradients locaux. L'accuracy/loss test évalue le
modèle global ; la loss cliente moyenne résume ses évaluations par client.
Ce terme ne désigne pas le holdout scellé des audits synthétiques G0f.

La confirmation F à 20 tours contient seulement α = 0 et 2. Ce choix vient
de la matrice exécutée, et ne décrit pas toutes les valeurs testées. Nous
conservons ce résultat à trois seeds, puis ajoutons trois vues séparées :

- **E, sans attaque** : les quatre contrôles DP oui/non × α = 0/2, sur une seed ;
- **C, sans DP** : toutes les valeurs α = −5, −2, −1, 0, 1, 2, 5 ;
- **D, avec DP** : α = −2 et 2, avec un budget fixé dans chaque figure.

Les campagnes, horizons, seuils de clipping et nombres de clients restent
séparés. Une seule seed donne une courbe exploratoire sans intervalle ; avec
plusieurs seeds, les bandes indiquent moyenne ± écart-type (ce ne sont pas
des IC95). Chaque figure porte sur une seule métrique et au plus quatre α.

### 3.1 Confirmation F : trois conditions sur trois seeds

La phrase « il manque le contrôle sans DP à α = 0 » concerne **cette phase F**.
α = 0 donne λᵢ = 1/n : tous les clients ont le même poids. Sans DP signifie
que le bruit gaussien est désactivé ; les autres opérations de ce protocole,
dont le clipping, sont conservées. Ce contrôle compléterait la comparaison
DP × α pour l'agrégation uniforme. Isoler l'effet du bruit demanderait aussi
des batches strictement appariés entre les deux branches.

F contient **DP/α=0, DP/α=2 et sans-DP/α=2**, mais pas **sans-DP/α=0**.
Le tableau E ci-dessous contient les quatre cases, à titre exploratoire.
"""

MORE_TRAJECTORIES = [
    md(r"""
### 3.2 Le contrôle sans DP à α = 0 : présent dans E, absent dans F

Ce tableau compte les seeds réellement disponibles **sans attaque, à référence
FCC et à protocole fixé dans chaque phase**. Une case absente ne vaut pas zéro
accuracy. E utilise la seed 137 ; F utilise les seeds 28, 36 et 54. Leurs
estimations ne sont pas fusionnées. Dans E, les configurations sont identiques
hors α et mécanisme DP, mais une même seed ne garantit pas les mêmes batches
après chaque tirage de bruit : ce n'est pas un appariement aléatoire strict.
""", "controls-intro"),
    code(r"""
CONTROL_PHASES = ["e_byzantine_identification_screen", "f_confirmation_t20"]
control_data = final_df[
    (final_df.campaign == "positioning_v3")
    & final_df.phase.isin(CONTROL_PHASES)
    & (final_df.reference == "F_CC") & (final_df.attack == "none")
    & final_df.alpha.isin([0.0, 2.0])
].copy()
control_rows = []
for (phase, n), group in control_data.groupby(["phase", "n_clients"]):
    for dp in [False, True]:
        for alpha in [0.0, 2.0]:
            part = group[(group.dp_enabled == dp) & (group.alpha == alpha)]
            control_rows.append({
                "Phase": "E : exploratoire" if phase.startswith("e_") else "F : confirmation",
                "Clients": n, "Bruit DP": "oui" if dp else "non", "α": alpha,
                "Seeds": ", ".join(map(str, sorted(part.seed.unique()))),
                "Disponible": "oui" if len(part) else "absent",
                "Accuracy test (%)": "—" if part.empty else f"{part.test_accuracy_pct.mean():.2f}",
            })
control_table = pd.DataFrame(control_rows)
display(control_table)
control_table.to_csv(EXPORT_ROOT / "dp_alpha_control_coverage.csv", index=False)

alpha_figure_root = EXPORT_ROOT / "alpha_trajectories"
alpha_figure_root.mkdir(exist_ok=True)

def _save_alpha_figure(fig, name):
    fig.savefig(alpha_figure_root / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def _plot_controls_e(metric, ylabel):
    view = rounds_df[
        (rounds_df.campaign == "positioning_v3")
        & (rounds_df.phase == "e_byzantine_identification_screen")
        & (rounds_df.reference == "F_CC") & (rounds_df.attack == "none")
        & rounds_df.alpha.isin([0.0, 2.0])
    ].copy()
    for n, group in view.groupby("n_clients"):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for (dp, alpha), part in group.groupby(["dp_enabled", "alpha"]):
            values = part.groupby("round")[metric].mean()
            ax.plot(values.index, values, linewidth=2,
                    label=f"{'DP ε=4' if dp else 'Sans DP'} · α={alpha:g}",
                    linestyle="-" if dp else "--")
        c = group.C_local.unique()
        assert len(c) == 1
        ax.set(xlabel="Tour", ylabel=ylabel,
               title=f"Écran E · n={n} · C={c[0]:g} · FCC · sans attaque · seed 137")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        _save_alpha_figure(fig, f"e_controls_{metric}_n{n}")

_plot_controls_e("test_accuracy_pct", "Accuracy test (%)")
""", "controls-table"),
    md(r"""
### 3.3 Toutes les valeurs d'α : écran C sans DP

L'écran C a exécuté **−5, −2, −1, 0, 1, 2 et 5**, pour FCC, CM, trMean et RFA.
Les listes de sélection ci-dessous fixent les références et nombres de clients affichés ;
FCC est affichée par défaut. Les α négatifs et positifs ont chacun leur figure,
avec α = 0 comme repère commun. Toutes les valeurs sont ainsi visibles.

Il s'agit de **positioning_v1, C = 8, T = 20, seed 137, sans DP**. Les figures
ne sont pas des confirmations multi-seeds et ne doivent pas être superposées
aux courbes v3 à C = 4 pour n = 25.
""", "alpha-intro"),
    code(r"""
ALPHA_REFERENCES_TO_PLOT = ["F_CC"]  # Ajouter "CM", "trMean", "RFA" si souhaité.
ALPHA_CLIENTS_TO_PLOT = [10, 25]
alpha_all = rounds_df[
    (rounds_df.campaign == "positioning_v1")
    & (rounds_df.phase == "c_alpha_reference_screen")
    & (rounds_df.attack == "none") & ~rounds_df.dp_enabled
].copy()
alpha_availability = (alpha_all.drop_duplicates("run_uid")
    .groupby(["reference", "n_clients", "C_local", "horizon"])
    .agg(Alphas=("alpha", lambda s: ", ".join(f"{v:g}" for v in sorted(s.unique()))),
         Seeds=("seed", lambda s: ", ".join(map(str, sorted(s.unique())))))
    .reset_index().rename(columns={"reference":"Référence", "n_clients":"Clients",
                                  "C_local":"C", "horizon":"Tours"}))
display(alpha_availability)
alpha_availability.to_csv(EXPORT_ROOT / "alpha_trajectory_availability.csv", index=False)

def plot_all_alpha_trajectories(metric, ylabel):
    if alpha_all.empty:
        display(Markdown("**Écran C absent : aucune courbe inventée.**"))
        return
    chosen = alpha_all[alpha_all.reference.isin(ALPHA_REFERENCES_TO_PLOT)
                       & alpha_all.n_clients.isin(ALPHA_CLIENTS_TO_PLOT)]
    for (reference, n, c, horizon), group in chosen.groupby(
        ["reference", "n_clients", "C_local", "horizon"]
    ):
        for side, allowed in [("négatifs", [-5., -2., -1., 0.]),
                              ("positifs", [0., 1., 2., 5.])]:
            fig, ax = plt.subplots(figsize=(8.4, 4.8))
            for alpha, part in group[group.alpha.isin(allowed)].groupby("alpha"):
                stats = part.groupby("round")[metric].agg(["mean", "std", "count"])
                ax.plot(stats.index, stats["mean"], label=f"α = {alpha:g}",
                        color="black" if alpha == 0 else None,
                        linestyle="--" if alpha == 0 else "-", linewidth=2)
                mask = stats["count"] > 1
                if mask.any():
                    ax.fill_between(stats.index[mask],
                                    (stats["mean"]-stats["std"])[mask],
                                    (stats["mean"]+stats["std"])[mask], alpha=.12)
            ax.set(xlabel="Tour", ylabel=ylabel,
                   title=f"Sans DP · {reference} · n={n} · C={c:g}\n"
                         f"Écran C · T={horizon} · α {side} · seed 137")
            ax.legend(loc="best", fontsize=10)
            fig.tight_layout()
            _save_alpha_figure(fig, f"c_{metric}_{reference}_n{n}_{side}")

plot_all_alpha_trajectories("test_accuracy_pct", "Accuracy test (%)")
""", "alpha-accuracy"),
    code('plot_all_alpha_trajectories("test_loss", "Loss test")', "alpha-loss"),
    code('plot_all_alpha_trajectories("worst20_pct", "Worst-20 (%) · plus élevé = meilleur")', "alpha-worst"),
    code('plot_all_alpha_trajectories("gap_pp", "Gap Best-20 − Worst-20 (pp) · plus faible = meilleur")', "alpha-gap"),
    md(r"""
### 3.4 α négatif et positif avec DP : écran D

Les autres α de l'écran C n'ont pas tous été entraînés sous DP dans la
campagne positioning. D contient α = −2 et 2 aux budgets ε = 2, 4, 8.
Le budget choisi ci-dessous est fixé dans chaque figure ; il peut être changé
sans mélanger les runs. Les courbes sont exploratoires (seed 137).
""", "alpha-dp-intro"),
    code(r"""
DP_ALPHA_EPSILON_TO_PLOT = 4.0  # Autres valeurs disponibles : 2.0 et 8.0.
dp_alpha_view = rounds_df[
    (rounds_df.campaign == "positioning_v3")
    & (rounds_df.phase == "d_privacy_screen_v3")
    & (rounds_df.reference == "F_CC") & (rounds_df.attack == "none")
    & np.isclose(rounds_df.target_epsilon, DP_ALPHA_EPSILON_TO_PLOT)
].copy()
for n, group in dp_alpha_view.groupby("n_clients"):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for alpha, part in group.groupby("alpha"):
        stats = part.groupby("round").test_accuracy_pct.mean()
        ax.plot(stats.index, stats, linewidth=2, label=f"α = {alpha:g}")
    ax.set(xlabel="Tour", ylabel="Accuracy test (%)",
           title=f"Écran D · DP ε={DP_ALPHA_EPSILON_TO_PLOT:g} · FCC · n={n}\n"
                 f"C={group.C_local.iloc[0]:g} · sans attaque · seed 137")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_alpha_figure(fig, f"d_accuracy_eps{DP_ALPHA_EPSILON_TO_PLOT:g}_n{n}")
""", "alpha-dp"),
]

WEIGHTS_INTRO = r"""
## 5. Poids FAR : poids maximal, concentration et entropie

**Cette section sert à expliquer comment FAR agrège les gradients.** L'accuracy
et le Worst-20 donnent le résultat final ; les poids indiquent si FAR utilise
une pondération presque uniforme ou privilégie quelques clients. Ils aident
à comprendre pourquoi changer α ou la référence a parfois très peu d'effet.

Nous notons λᵢ les poids de FAR (les champs historiques du code utilisent parfois
`q`). Ils sont positifs et leur somme vaut 1. Ce λ ne désigne pas un budget DP.

| Indicateur | Uniforme | Information et exemple |
|---|---:|---|
| n × λ max | 1 | Domination du plus gros poids. À n = 10, une valeur 3 signifie λ max = 0,30, contre 0,10 uniforme. |
| Q = n × Σ λ² | 1 | Concentration de tous les poids. Q = 2 donne n/Q = n/2 clients effectifs au sens quadratique. |
| H / log(n) | 1 | Dispersion des poids : proche de 1, poids répartis ; proche de 0, un client domine. |
| exp(H) / n | 1 | Nombre effectif entropique, celui enregistré par le code. C'est une transformation de H, pas une quatrième preuve indépendante. |

$$
H(\lambda)=-\sum_i\lambda_i\log\lambda_i,\qquad
N_{2}=\frac{1}{\sum_i\lambda_i^2}=\frac{n}{Q},\qquad
N_H=\exp(H(\lambda)).
$$

Les deux nombres effectifs N₂ et N_H ne sont généralement pas égaux. Par exemple,
à n = 10, un poids de 0,30 et neuf poids de 0,70/9 donnent Q ≈ 1,444,
N₂ ≈ 6,92 et H/log(n) ≈ 0,933. Une entropie assez haute peut donc coexister avec
un client trois fois plus pondéré que sous uniforme.

**Lien précis avec le bruit.** Pour des poids fixés avant le bruit, et des
bruits centrés indépendants de même covariance Σ, on a

$$
\operatorname{tr}\operatorname{Cov}\left(\sum_i\lambda_i Z_i\right)
=\operatorname{tr}(\Sigma)\sum_i\lambda_i^2.
$$

Le rapport à l'uniforme vaut alors Q. Avec des covariances différentes,
le numérateur devient Σᵢ λᵢ² tr(Σᵢ). Dans FAR courant, les poids dépendent
des gradients bruités et le clipping serveur est non linéaire : **Q reste un
diagnostic de concentration, pas une mesure exacte de la variance DP réalisée**.

**Ce qu'il faut croiser.** Q proche de 1 explique un FAR presque uniforme ;
Q grand motive l'audit de l'identité des clients favorisés. Les corrélations
bruit–poids (section 8), la masse Byzantine (section 7), l'accuracy et le
Worst-20 permettent ensuite d'interpréter cette concentration. Aucun des
trois indicateurs ne distingue seul un honest outlier d'un attaquant et aucun
ne démontre la nécessité d'une référence temporelle RCIG.

Les vues ci-dessous concernent la phase F, T = 20, DP ε = 4, sans attaque.
Une médiane est d'abord calculée sur les tours de chaque run, puis ces médianes
sont moyennées entre seeds. Ce n'est pas une médiane sur un scalaire.
"""

WEIGHT_BRIDGE = [
    md(r"""
### 5.1 Ce que ces diagnostics expliquent dans nos propres entraînements

Nous remettons ici côte à côte le contrôle uniforme, FAR sous DP et FAR sans
DP, avec l'accuracy et le Worst-20. Les poids sont résumés sur les tours de
chaque seed ; les performances sont celles du dernier tour. Cette lecture
évite de conclure « le bruit aide » simplement parce que DP/α=2 dépasse
sans-DP/α=2 : les deux trajectoires peuvent utiliser des poids très différents.
""", "weight-bridge-intro"),
    code(r"""
weight_bridge_source = trajectory_focus.copy()
weight_bridge_source["n_lambda_max"] = weight_bridge_source.n_clients * weight_bridge_source.max_weight
weight_bridge_source["entropy_fraction"] = weight_bridge_source.weight_entropy / np.log(weight_bridge_source.n_clients)
weight_per_run = (weight_bridge_source.groupby(["n_clients", "condition", "seed", "run_uid"])
    .agg(qmax=("n_lambda_max", "median"), concentration=("weight_concentration", "median"),
         entropy=("entropy_fraction", "median")).reset_index())
weight_bridge_final = trajectory_focus[trajectory_focus["round"] == trajectory_focus.horizon][
    ["run_uid", "test_accuracy_pct", "worst20_pct", "gap_pp"]]
weight_per_run = weight_per_run.merge(weight_bridge_final, on="run_uid", validate="one_to_one")
weight_bridge_rows = []
for (n, condition), part in weight_per_run.groupby(["n_clients", "condition"]):
    row = {"Clients":n, "Condition":condition, "Seeds":part.seed.nunique()}
    for source, label, digits in [
        ("qmax", "n × λ max", 3), ("concentration", "Q = n × Σ λ²", 3),
        ("entropy", "H / log(n)", 3), ("test_accuracy_pct", "Accuracy test (%)", 2),
        ("worst20_pct", "Worst-20 (%)", 2), ("gap_pp", "Gap (pp)", 2)]:
        row[label] = f"{part[source].mean():.{digits}f} ± {part[source].std(ddof=1):.{digits}f}"
    weight_bridge_rows.append(row)
weight_bridge = pd.DataFrame(weight_bridge_rows)
for n, part in weight_bridge.groupby("Clients"):
    display(Markdown(f"**n = {n} — moyenne ± écart-type entre seeds.**"))
    display(part.drop(columns="Clients"))
weight_bridge.to_csv(EXPORT_ROOT / "weight_mechanism_and_performance.csv", index=False)
display(Markdown(
    "**Lecture pour RCIG.** Une référence mieux estimée peut ne presque rien changer "
    "à l'agrégat lorsque Q ≈ 1. Si Q augmente, il reste à vérifier que les clients "
    "favorisés portent un signal honnête et pas surtout du bruit ou une attaque. "
    "La section 2.1 relie cette difficulté aux essais mécanistiques temporels."
))
""", "weight-bridge"),
]


def main():
    notebook = nbformat.read(NB, as_version=4)
    original_hash = hashlib.sha256(NB.read_bytes()).hexdigest()
    original_sources = {c.id: c.source for c in notebook.cells}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_root = ROOT / "output/analysis/ldp_gradient_far_positioning_notebook/revision_rcig"
    audit_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NB, audit_root / f"before_{stamp}.ipynb")
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get("tags", [])
                      and "rcig-motivation-20260912" not in c.metadata.get("tags", [])]
    by_id = {c.id: c for c in notebook.cells}
    changed_ids = {"51a8d33cea6d1b93", "180afe8c731ef31f", "228f01f8bee42d78", "29876ebf"}
    by_id["51a8d33cea6d1b93"].source = TRAJECTORY_INTRO.strip()
    by_id["228f01f8bee42d78"].source = WEIGHTS_INTRO.strip()
    current = by_id["180afe8c731ef31f"].source
    current = current.replace(
        "Il manque notamment le contrôle multi-seed sans DP à ",
        "Dans la confirmation F à trois seeds, il manque le contrôle sans DP à "
    ).replace(
        'r"$\\alpha=0$."',
        'r"$\\alpha=0$. Ce contrôle existe dans l’écran E, sur la seule seed 137; "\n'
        '    "la section 3.2 l’affiche séparément."'
    )
    by_id["180afe8c731ef31f"].source = current
    # An invalidated RCIG root must never appear as a valid final campaign.
    discovery = by_id["29876ebf"].source
    discovery = discovery.replace(
        'if campaign.startswith("positioning_v") and len(parts) >= 4:',
        'if campaign.startswith(("positioning_v", "rcig_end_to_end")) and len(parts) >= 4:'
    )
    insertion = '''        if "invalidated" in campaign.lower():
            inventory.append({
                "campaign": campaign, "phase": phase, "task_id": task_id,
                "metrics_path": str(path), "relative_path": relative,
                "status": "excluded", "errors": "Archived invalidated campaign; never pooled",
            })
            continue
'''
    anchor = '        campaign, phase, task_id, relative = infer_location(path)\n'
    if insertion not in discovery:
        discovery = discovery.replace(anchor, anchor + insertion, 1)
    by_id["29876ebf"].source = discovery
    for identifier in changed_ids:
        target = by_id[identifier]
        if target.cell_type == "code":
            target.outputs = []
            target.execution_count = None

    index = next(i for i,c in enumerate(notebook.cells)
                 if c.cell_type == "markdown" and c.source.startswith("## 4."))
    notebook.cells[index:index] = MORE_TRAJECTORIES
    index = next(i for i,c in enumerate(notebook.cells)
                 if c.cell_type == "markdown" and c.source.startswith("## 6."))
    notebook.cells[index:index] = WEIGHT_BRIDGE
    from notebook_rcig_motivation_cells import build_cells, build_evaluation_cells
    index = next(i for i,c in enumerate(notebook.cells)
                 if c.cell_type == "markdown" and c.source.startswith("## 3."))
    notebook.cells[index:index] = build_cells()
    notebook.cells.extend(build_evaluation_cells())
    # Preserve the user's later layout decision and the sparse-round plot fix
    # even if this surgical updater is rerun. Never restore removed sections.
    from notebook_positioning_layout_repair import apply_repairs
    layout_audit = apply_repairs(notebook)
    changed_ids.update(layout_audit["edited_cell_ids"])
    removed_ids = set(layout_audit["removed_cell_ids"])
    new_by_id = {c.id:c for c in notebook.cells}
    preserved_ids = []
    for identifier, source in original_sources.items():
        if identifier in changed_ids or identifier in removed_ids or identifier.startswith("rcig-"):
            continue
        if identifier not in new_by_id or new_by_id[identifier].source != source:
            raise AssertionError(f"Unexpected modification of user cell {identifier}")
        preserved_ids.append(identifier)
    assert len(new_by_id) == len(notebook.cells), "Duplicate cell IDs"
    nbformat.validate(notebook)
    nbformat.write(notebook, NB)
    audit = {"input_sha256":original_hash, "output_sha256":hashlib.sha256(NB.read_bytes()).hexdigest(),
             "preserved_source_cell_ids":preserved_ids, "edited_existing_cell_ids":sorted(changed_ids),
             "layout_and_round_coverage_repair":layout_audit,
             "cells":len(notebook.cells), "timestamp_utc":stamp}
    (audit_root / f"source_preservation_{stamp}.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"cells":len(notebook.cells), "preserved":len(preserved_ids),
                      "backup":str(audit_root / f"before_{stamp}.ipynb")}, indent=2))


if __name__ == "__main__":
    main()
