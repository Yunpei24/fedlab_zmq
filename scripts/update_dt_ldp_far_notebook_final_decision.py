#!/usr/bin/env python3
"""Add an executable, presentation-ready final DT-LDP-FAR decision section.

The update preserves every existing cell, including user edits.  Only cells
carrying ``GENERATED_TAG`` are replaced, which makes the operation idempotent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "DT_LDP_FAR_N10_Decisive_Analysis.ipynb"
GENERATED_TAG = "dtldp-final-negative-decision-2026-09"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        source.strip(), metadata={"tags": [GENERATED_TAG]}
    )


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(
        source.strip(), metadata={"tags": [GENERATED_TAG]}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cells = [
        cell
        for cell in notebook.cells
        if GENERATED_TAG not in cell.get("metadata", {}).get("tags", [])
    ]

    clipping_code_index = next(
        index
        for index, cell in enumerate(cells)
        if "CLIP_CAMPAIGN =" in str(cell.get("source", ""))
    )
    clipping_certificate_cells = [
        markdown(
            r"""
### Certificat d'influence associé au clipping serveur

L'ablation précédente montre l'effet empirique du clipping sous Bit-Flip.
La propriété « influence bornée » est distincte : elle découle de la norme
serveur et du cap des poids. Si

\[
\lVert X_{i,t}\rVert_2\leq U,
\qquad
\omega_{i,t}\leq\frac{\kappa_w}{n},
\]

alors un client contribue au plus

\[
\lVert\omega_{i,t}X_{i,t}\rVert_2
\leq\frac{\kappa_w U}{n},
\]

et l'ensemble de $b$ clients byzantins contribue au plus

\[
\left\lVert\sum_{i\in\mathcal B}\omega_{i,t}X_{i,t}\right\rVert_2
\leq\frac{b\kappa_w U}{n}.
\]

La cellule suivante instancie ce certificat pour l'ablation à 25 clients.
"""
        ),
        code(
            r"""
certificate_n = 25
certificate_b = 5
certificate_kappa_w = 2.0
certificate_U = 0.28

individual_bound = certificate_kappa_w * certificate_U / certificate_n
byzantine_cohort_bound = (
    certificate_b * certificate_kappa_w * certificate_U / certificate_n
)
bitflip_clipped = clip_df[
    (clip_df.threat == "bf20_n25_s10") & np.isclose(clip_df.U, certificate_U)
].iloc[0]
observed_byzantine_contribution = float(bitflip_clipped.byz_contrib)

certificate_table = pd.DataFrame([
    {
        "Quantité": "Contribution maximale d'un client",
        "Valeur ou borne": individual_bound,
        "Interprétation": "borne analytique κ_w U / n",
    },
    {
        "Quantité": "Contribution maximale des 5 Byzantins",
        "Valeur ou borne": byzantine_cohort_bound,
        "Interprétation": "borne analytique b κ_w U / n",
    },
    {
        "Quantité": "Contribution byzantine observée — Bit-Flip ×10",
        "Valeur ou borne": observed_byzantine_contribution,
        "Interprétation": "doit rester sous la borne de cohorte",
    },
])
certificate_table["Ratio observé / borne de cohorte"] = np.nan
certificate_table.loc[2, "Ratio observé / borne de cohorte"] = (
    observed_byzantine_contribution / byzantine_cohort_bound
)
display(certificate_table.round(5))
print(
    "Certificat respecté :",
    observed_byzantine_contribution <= byzantine_cohort_bound + 1e-12,
)
"""
        ),
        markdown(
            r"""
**Conclusion.** L'expérience illustre la réduction effective d'une attaque de
grande norme ; les inégalités établissent la borne d'influence. Ici, la
contribution byzantine observée reste sous la borne de cohorte. Cela ne montre
pas que le seuil $U=0{,}28$ est optimal : ALIE peut rester sous le seuil et les
uploads honnêtes peuvent être excessivement clippés.
"""
        ),
    ]
    cells[clipping_code_index + 1:clipping_code_index + 1] = (
        clipping_certificate_cells
    )

    summary_index = next(
        index
        for index, cell in enumerate(cells)
        if "Tableau de synthèse" in str(cell.get("source", ""))
    )

    additions = [
        markdown(
            r"""
## Décision finale : arrêter cette instanciation, pas la local-DP

La conclusion négative porte sur **l'hypothèse d'utilité de l'instanciation
DT-LDP-FAR étudiée**. Elle ne dit pas que la confidentialité locale est inutile
ou impossible. Les expériences séparent deux questions :

1. le mécanisme respecte-t-il ses contrôles de confidentialité et de sécurité ?
2. ces contrôles produisent-ils un gain reproductible d'accuracy ou de fairness ?

La première réponse est positive ; la seconde est négative. Les cellules
suivantes reconstruisent ce verdict depuis le fichier de décision cumulatif,
sans sélectionner une seed ou une attaque favorable.
"""
        ),
        code(
            r"""
FINAL_DECISION_PATH = (
    REPO_ROOT / "results" / "dt_ldp_far"
    / "final_decision_stage22_to_stage26.json"
)
if not FINAL_DECISION_PATH.exists():
    raise FileNotFoundError(FINAL_DECISION_PATH)

final_decision = read_json(FINAL_DECISION_PATH)
stage26b = final_decision["stage26b"]

decision_table = pd.DataFrame([
    {
        "Question": "Budget DP apparié ?",
        "Résultat": "Oui",
        "Preuve enregistrée": (
            f"écart maximal d'epsilon = "
            f"{stage26b['privacy_epsilon_difference_max']:.1f}"
        ),
        "Verdict": "validé",
    },
    {
        "Question": "Cap des poids respecté ?",
        "Résultat": "Oui",
        "Preuve enregistrée": "aucune violation dans les campagnes",
        "Verdict": "validé",
    },
    {
        "Question": "Appariement aléatoire strict ?",
        "Résultat": "Oui",
        "Preuve enregistrée": (
            f"taux = {stage26b['strict_randomness_pair_rate']:.1f}"
        ),
        "Verdict": "validé",
    },
    {
        "Question": "Confinement déterministe respecté ?",
        "Résultat": "Oui",
        "Preuve enregistrée": (
            f"norme médiane {stage26b['candidate_correction_norm_median']:.6f} "
            f"≤ borne {stage26b['candidate_correction_universal_bound']:.3f}"
        ),
        "Verdict": "validé",
    },
    {
        "Question": "Gain d'utilité reproductible ?",
        "Résultat": "Non",
        "Preuve enregistrée": (
            f"correction FAR : {stage26b['correction_attacked_accuracy_gain_mean_pp']:+.3f} "
            "pp sous attaque"
        ),
        "Verdict": "rejeté",
    },
    {
        "Question": "Gain sous chaque attaque ?",
        "Résultat": "Non",
        "Preuve enregistrée": (
            f"ancre : Bit-Flip "
            f"{stage26b['anchor_attacked_accuracy_gain_by_attack_pp']['bf20_n25_s10']:+.3f} pp ; "
            f"IPM {stage26b['anchor_attacked_accuracy_gain_by_attack_pp']['ipm20_n25']:+.3f} pp"
        ),
        "Verdict": "rejeté",
    },
])

print(f"Runs agrégés : {final_decision['evaluated_runs']}")
print(f"Statut : {final_decision['status']}")
display(decision_table)
decision_table.to_csv(
    EXPORT_ROOT / "dt_ldp_far_final_decision_table.csv", index=False
)
"""
        ),
        code(
            r"""
fig, ax = plt.subplots(figsize=(13, 3.8))
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (0.08, "1. Confidentialité", "Même budget DP\ndans les bras appariés", "#D9F0E3"),
    (1.08, "2. Contrôles", "Cap des poids et\nconfinement respectés", "#D9F0E3"),
    (2.08, "3. Utilité", "Pas de gain reproductible\naccuracy / fairness", "#F7D6D0"),
]
for x, title, body, color in boxes:
    ax.add_patch(plt.Rectangle((x, 0.24), 0.82, 0.52, facecolor=color,
                               edgecolor="#334155", linewidth=1.2))
    ax.text(x + 0.41, 0.62, title, ha="center", va="center",
            fontsize=12, weight="bold")
    ax.text(x + 0.41, 0.42, body, ha="center", va="center", fontsize=10)
for x in (0.93, 1.93):
    ax.annotate("", xy=(x + 0.10, 0.50), xytext=(x - 0.03, 0.50),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"})

ax.text(
    1.5, 0.07,
    "Décision : arrêter la recherche adaptative sur cette instanciation ; "
    "conserver les garanties de sécurité.",
    ha="center", va="center", fontsize=11, weight="bold", color="#8A2D1E"
)
ax.set_title(
    f"Verdict confirmatoire DT-LDP-FAR — {final_decision['evaluated_runs']} runs",
    fontsize=14, pad=12,
)
save_show(fig, "dt_ldp_far_final_negative_decision.png");
"""
        ),
        code(
            r"""
stage_decision_specs = [
    (
        "22", "stage22_lagged_descent_validation_v1/stage22_decision.json",
        "separated_accuracy_gain_mean_pp", "no_attack_accuracy_difference_mean_pp",
        "accuracy_gain_mean_by_attack_pp", "lagged_descent_hypothesis_validated",
    ),
    (
        "23", "stage23_robust_admissibility_validation_v1/stage23_decision.json",
        "integrated_attacked_accuracy_gain_mean_pp", "integrated_clean_accuracy_mean_pp",
        "integrated_attacked_accuracy_gain_by_attack_pp", "stage23_fully_validated",
    ),
    (
        "24", "stage24_multikrum_admissibility_validation_v1/stage24_decision.json",
        "integrated_attacked_accuracy_gain_mean_pp", "integrated_clean_accuracy_mean_pp",
        "integrated_attacked_accuracy_gain_by_attack_pp", "stage24_fully_validated",
    ),
    (
        "25", "stage25_robust_anchor_containment_validation_v1/stage25_decision.json",
        "candidate_attacked_accuracy_gain_mean_pp", "candidate_clean_accuracy_mean_pp",
        "candidate_attacked_accuracy_gain_by_attack_pp", "stage25_fully_validated",
    ),
    (
        "26B", "stage26b_trmean_nnm_anchor_confirmatory_v1/stage26b_decision.json",
        "candidate_attacked_accuracy_gain_mean_pp", "candidate_clean_accuracy_mean_pp",
        "candidate_attacked_accuracy_gain_by_attack_pp", "stage26_fully_validated",
    ),
]

stage_utility_rows = []
for stage, relative_path, attacked_key, clean_key, attacks_key, verdict_key in stage_decision_specs:
    payload = read_json(REPO_ROOT / "results" / "dt_ldp_far" / relative_path)
    observations = payload["observations"]
    by_attack = observations[attacks_key]
    stage_utility_rows.append({
        "Stage": stage,
        "Runs": payload["completed_runs"],
        "Gain attaqué moyen (pp)": observations[attacked_key],
        "Coût propre moyen (pp)": observations[clean_key],
        "Bit-Flip (pp)": by_attack["bf20_n25_s10"],
        "IPM (pp)": by_attack["ipm20_n25"],
        "Hypothèse validée": bool(payload.get(verdict_key, False)),
    })

stage_utility = pd.DataFrame(stage_utility_rows)
display(stage_utility.round(3))
stage_utility.to_csv(
    EXPORT_ROOT / "dt_ldp_far_utility_failures_by_stage.csv", index=False
)

fig, ax = plt.subplots(figsize=(11.5, 5.5))
x = np.arange(len(stage_utility))
width = 0.36
ax.bar(
    x - width / 2, stage_utility["Bit-Flip (pp)"], width,
    label="Bit-Flip ×10", color="#3B82A0",
)
ax.bar(
    x + width / 2, stage_utility["IPM (pp)"], width,
    label="IPM", color="#D97745",
)
ax.axhline(0.0, color="black", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"Stage {value}" for value in stage_utility.Stage])
ax.set_ylabel("Gain de Test Accuracy contre la baseline (pp)")
ax.set_title("Aucune construction ne gagne de façon reproductible sous les deux attaques")
ax.legend()
save_show(fig, "dt_ldp_far_utility_failure_by_attack.png");
"""
        ),
        markdown(
            r"""
### Preuve expérimentale de l'échec d'utilité

Une moyenne globale positive ne suffit pas. Le Stage 23 gagne sous IPM mais
perd sous Bit-Flip et dégrade l'accuracy sans attaque. Le Stage 26B gagne sous
Bit-Flip mais perd sous IPM. Les Stages 22, 24 et 25 ont un gain attaqué moyen
négatif. Aucun stage ne satisfait donc simultanément les gates préenregistrés
sur les deux attaques, le coût propre et la répétabilité entre seeds.

Le résultat négatif ne repose pas sur une unique moyenne : il repose sur
l'absence d'une amélioration de même signe dans les conditions confirmatoires.
"""
        ),
        markdown(
            r"""
### Comment lire le verdict

- **Ce qui fonctionne :** la privacy locale, l'appariement expérimental, le
  contrôle de la concentration et la borne déterministe de correction.
- **Ce qui échoue :** ces propriétés ne se transforment pas en amélioration
  reproductible de l'accuracy ou de la fairness. Au Stage 26B, la correction
  FAR apporte en moyenne environ $-0{,}032$ point sous attaque et
  $-0{,}027$ point sans attaque.
- **Pourquoi arrêter :** les Stages 22 à 26 testent plusieurs constructions sur
  des holdouts indépendants. Continuer à modifier les mêmes seuils après avoir
  observé ces échecs reviendrait à entraîner la méthode sur les données de
  validation.
- **Ce qui reste scientifiquement possible :** une nouvelle piste local-DP doit
  partir d'une hypothèse substantiellement différente, d'un objectif renouvelé
  et d'un nouveau holdout. Ce verdict n'est pas un théorème d'impossibilité de
  la local-DP.

La formulation défendable est donc : **DT-LDP-FAR est sûr au sens des
invariants vérifiés, mais sa supériorité d'utilité est rejetée dans le protocole
Fashion-MNIST étudié.**
"""
        ),
    ]

    cells[summary_index:summary_index] = additions
    notebook.cells = cells

    if args.execute:
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()

    nbformat.write(notebook, NOTEBOOK)
    action = "mis à jour et exécuté" if args.execute else "mis à jour"
    print(f"Notebook {action} : {NOTEBOOK}")


if __name__ == "__main__":
    main()
