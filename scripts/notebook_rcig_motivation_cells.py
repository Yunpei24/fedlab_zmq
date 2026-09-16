#!/usr/bin/env python3
"""Cells explaining the evidence chain that motivates RCIG.

The cells are designed to be inserted immediately after section 2 of
``LDP_Gradient_FAR_Positioning_Analysis.ipynb``.  At that point the notebook
already defines ``ROOT``, ``RESULTS_ROOT``, ``EXPORT_ROOT``, ``final_df``,
``rounds_df``, ``np``, ``pd``, ``plt``, ``Markdown`` and ``display``.

This module deliberately does not edit or execute the notebook.  It only
returns nbformat cells so the surgical updater can preserve user-authored
comments elsewhere in the notebook.
"""

from __future__ import annotations

import hashlib

import nbformat as nbf


def _md(source: str):
    return _cell("markdown", source)


def _code(source: str):
    return _cell("code", source)


def _cell(cell_type: str, source: str):
    clean_source = source.strip()
    digest = hashlib.sha256(clean_source.encode("utf-8")).hexdigest()[:16]
    cell_id = f"rcig-motivation-{digest}"
    metadata = {"tags": ["rcig-motivation-20260912"]}
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(clean_source, metadata=metadata, id=cell_id)
    return nbf.v4.new_code_cell(clean_source, metadata=metadata, id=cell_id)


def _all_cells() -> list:
    """Return source cells, separated below into motivation and evaluation."""

    return [
        _md(r"""
## 2.1 Motivation initiale : partir des entraînements réels, avant d'évaluer RCIG

La question scientifique est : **comment préserver la fairness de FAR sous
local-DP, lorsque le bruit peut différer entre clients, tout en limitant
l'influence Byzantine ?**

Cette partie répond à **pourquoi avons-nous voulu construire un mécanisme
comme RCIG ?** Elle ne prend aucun résultat synthétique de RCIG comme argument
de motivation. Elle part de nos entraînements Fashion-MNIST, puis expose le
raisonnement qui conduit à chercher une information temporelle et un critère
de compatibilité avec le bruit attendu.

La progression est : **problème observé → diagnostic → limite des corrections
simples → hypothèse de construction**. Ces observations motivent une famille
de mécanismes ; elles ne déterminent pas à elles seules la formule de RCIG.

Les tableaux ci-dessous sont reconstruits depuis les artefacts réels. Les
tests synthétiques et le statut d'évaluation de RCIG sont placés à part,
en **section 15**, après la présentation des expériences de positionnement.
"""),
        _code(r"""
from pathlib import Path
import json

RCIG_EXPORT = EXPORT_ROOT / "rcig_motivation"
RCIG_EXPORT.mkdir(parents=True, exist_ok=True)

# Vrais gradients privés Fashion-MNIST, n=10, T=40, trois seeds appariées.
motivation_runs = final_df[
    (final_df["campaign"] == "fmnist_confirmatory_v1")
    & (final_df["n_clients"] == 10)
    & (final_df["horizon"] == 40)
    & (final_df["attack"] == "none")
    & (final_df["noise_profile"].isin(["homogeneous", "heteroscedastic"]))
    & (
        np.isclose(final_df["alpha"], 0.0)
        | (
            np.isclose(final_df["alpha"], 1.0)
            & final_df["reference"].isin(["F_CC", "F_NA-CC"])
        )
    )
].copy()

def rcig_arm(row):
    if np.isclose(row["alpha"], 0.0):
        return "Uniforme (alpha=0)"
    if row["reference"] == "F_CC":
        return "FAR + F_CC (alpha=1)"
    return "FAR + F_NA-CC (alpha=1)"

motivation_runs["arm"] = motivation_runs.apply(rcig_arm, axis=1)
motivation_runs["regime"] = motivation_runs["noise_profile"].map({
    "homogeneous": "Homogène",
    "heteroscedastic": "Hétéroscédastique",
})

protocol_errors = []
if len(motivation_runs) != 18:
    protocol_errors.append(f"18 runs attendus, {len(motivation_runs)} trouvés")
if set(motivation_runs["seed"].astype(int)) != {28, 36, 54}:
    protocol_errors.append("seeds attendues {28, 36, 54} absentes ou supplémentaires")
if set(motivation_runs["noise_profile"]) != {"homogeneous", "heteroscedastic"}:
    protocol_errors.append("les deux régimes de bruit attendus ne sont pas présents")
if set(motivation_runs["arm"]) != {
    "Uniforme (alpha=0)", "FAR + F_CC (alpha=1)", "FAR + F_NA-CC (alpha=1)"
}:
    protocol_errors.append("les trois bras confirmatoires attendus ne sont pas présents")
if set(motivation_runs["dataset"].astype(str).str.lower()) != {"fashionmnist"}:
    protocol_errors.append("dataset différent de Fashion-MNIST")
if protocol_errors:
    display(Markdown("**Artefacts confirmatoires incomplets ou incompatibles :** "
                     + "; ".join(protocol_errors)))
    raise RuntimeError("; ".join(protocol_errors))

metric_labels = {
    "test_accuracy_pct": "Test Acc. (%)",
    "worst20_pct": "Worst-20 (%)",
    "gap_pp": "Gap (pp)",
    "variance_pp2": "Variance (pp²)",
}
summary_rows = []
for (regime, arm), group in motivation_runs.groupby(["regime", "arm"], sort=False):
    row = {"Bruit": regime, "Méthode": arm, "Seeds": group["seed"].nunique()}
    for key, label in metric_labels.items():
        values = pd.to_numeric(group[key], errors="coerce").dropna()
        mean = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        row[label] = f"{mean:.2f} ± {sd:.2f}" if np.isfinite(sd) else f"{mean:.2f}"
    summary_rows.append(row)

motivation_summary = pd.DataFrame(summary_rows)
display(Markdown("### 2.1.1 Vrais gradients privés : tableau final, moyenne ± écart-type"))
display(motivation_summary)

# Deltas strictement appariés à l'agrégation uniforme de la même seed et du
# même régime. Un gap négatif et un Worst-20 positif sont favorables.
paired_rows = []
baseline = motivation_runs[motivation_runs["arm"] == "Uniforme (alpha=0)"]
for arm in ["FAR + F_CC (alpha=1)", "FAR + F_NA-CC (alpha=1)"]:
    treated = motivation_runs[motivation_runs["arm"] == arm]
    paired = treated.merge(
        baseline,
        on=["seed", "regime"],
        suffixes=("_t", "_u"),
        validate="one_to_one",
    )
    for _, row in paired.iterrows():
        paired_rows.append({
            "Bruit": row["regime"],
            "Méthode vs uniforme": arm,
            "Seed": int(row["seed"]),
            "Delta Test Acc. (pp)": row["test_accuracy_pct_t"] - row["test_accuracy_pct_u"],
            "Delta Worst-20 (pp)": row["worst20_pct_t"] - row["worst20_pct_u"],
            "Delta Gap (pp)": row["gap_pp_t"] - row["gap_pp_u"],
        })

paired_seed_deltas = pd.DataFrame(paired_rows)
for col in ["Delta Test Acc. (pp)", "Delta Worst-20 (pp)", "Delta Gap (pp)"]:
    paired_seed_deltas[col] = paired_seed_deltas[col].round(2)
display(Markdown("### Deltas par seed (traitement − uniforme)"))
display(paired_seed_deltas.sort_values(["Bruit", "Méthode vs uniforme", "Seed"]))
display(Markdown(
    "**Ce que cela nous apprend.** Sous bruit homogène, l'accuracy des trois "
    "méthodes est voisine, mais FAR dégrade déjà le Worst-20 et le gap dans ces "
    "runs. Sous bruit hétéroscédastique, l'écart touche aussi fortement "
    "l'accuracy. Les deltas par seed montrent que ce n'est pas uniquement "
    "l'effet d'une moyenne masquant un résultat isolé. Cette expérience "
    "compare des agrégations sous DP ; elle ne contient pas leur témoin "
    "sans DP et ne démontre donc pas à elle seule l'effet causal du bruit. "
    "Elle établit le problème à résoudre dans cette instanciation."
))

motivation_summary.to_csv(RCIG_EXPORT / "fmnist_confirmatory_summary.csv", index=False)
paired_seed_deltas.to_csv(RCIG_EXPORT / "fmnist_confirmatory_paired_seed_deltas.csv", index=False)
print("Source :", RESULTS_ROOT / "fmnist_confirmatory_v1")
"""),
        _code(r"""
# Figure 1 — niveau de performance et bas de distribution des clients.
arm_order = [
    "Uniforme (alpha=0)",
    "FAR + F_CC (alpha=1)",
    "FAR + F_NA-CC (alpha=1)",
]
regime_order = ["Homogène", "Hétéroscédastique"]
colors = ["#4C78A8", "#F58518", "#54A24B"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
for ax, (metric, title) in zip(
    axes,
    [("test_accuracy_pct", "Accuracy test"), ("worst20_pct", "Worst-20")],
):
    x = np.arange(len(regime_order), dtype=float)
    width = 0.24
    for j, arm in enumerate(arm_order):
        means, sds = [], []
        for regime in regime_order:
            values = pd.to_numeric(
                motivation_runs.loc[
                    (motivation_runs["regime"] == regime)
                    & (motivation_runs["arm"] == arm), metric
                ], errors="coerce"
            ).dropna()
            means.append(values.mean())
            sds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.bar(x + (j - 1) * width, means, width, yerr=sds, capsize=4,
               color=colors[j], label=arm)
    ax.set_xticks(x, regime_order)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_title(f"{title} — moyenne ± écart-type (3 seeds)")
    ax.set_ylim(bottom=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
path = RCIG_EXPORT / "real_private_gradients_accuracy_worst20.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
print("Figure :", path)
"""),
        _code(r"""
# Figure 2 — deux mesures de dispersion : gap et écart-type inter-clients.
# La racine carrée de la variance est exprimée en points de pourcentage, ce qui
# la rend comparable visuellement au gap. Pour les deux métriques, plus bas est mieux.
dispersion = motivation_runs.copy()
dispersion["client_sd_pp"] = np.sqrt(dispersion["variance_pp2"].clip(lower=0))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
for ax, (metric, title) in zip(
    axes,
    [("gap_pp", "Gap Best-20 − Worst-20"),
     ("client_sd_pp", "Écart-type inter-clients")],
):
    x = np.arange(len(regime_order), dtype=float)
    width = 0.24
    for j, arm in enumerate(arm_order):
        means, sds = [], []
        for regime in regime_order:
            values = pd.to_numeric(
                dispersion.loc[
                    (dispersion["regime"] == regime)
                    & (dispersion["arm"] == arm), metric
                ], errors="coerce"
            ).dropna()
            means.append(values.mean())
            sds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        ax.bar(x + (j - 1) * width, means, width, yerr=sds, capsize=4,
               color=colors[j], label=arm)
    ax.set_xticks(x, regime_order)
    ax.set_ylabel("Points de pourcentage")
    ax.set_title(f"{title} — moyenne ± écart-type")
    ax.set_ylim(bottom=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
path = RCIG_EXPORT / "real_private_gradients_fairness_dispersion.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
print("Figure :", path)
"""),
        _md(r"""
### 2.1.2 Pourquoi une distance brute confond signal et bruit

Pour une référence $r$ **fixe conditionnellement au passé** et indépendante du
bruit frais, écrivons $Y_i=g_i+Z_i$, avec
$\mathbb E[Z_i]=0$ et $\operatorname{Cov}(Z_i)=\Sigma_i$. Alors

$$
\begin{aligned}
\mathbb E\!\left[\lVert Y_i-r\rVert_2^2\mid r,g_i\right]
&=\mathbb E\!\left[\lVert(g_i-r)+Z_i\rVert_2^2\right]\\
&=\lVert g_i-r\rVert_2^2
  +2(g_i-r)^\top\mathbb E[Z_i]
  +\mathbb E\lVert Z_i\rVert_2^2\\
&=\lVert g_i-r\rVert_2^2+\operatorname{tr}(\Sigma_i).
\end{aligned}
$$

La distance contient donc une **géométrie propre** et un **plancher de bruit**.
Quand les $\Sigma_i$ diffèrent, une grande distance ne signifie pas à elle seule
« client honnête utile » ou « Byzantine ».

La référence intervient ensuite dans l'agrégat FAR. Pour une cohorte fixée avec
$\lVert X_i\rVert_2\le U$, posons

$$
d_i(F)=\lVert X_i-F\rVert_2,
\qquad
\lambda(F)=\operatorname{softmax}(\alpha d(F)),
\qquad
A(F)=\sum_i\lambda_i(F)X_i.
$$

L'inégalité triangulaire inverse donne
$|d_i(F)-d_i(F')|\le\lVert F-F'\rVert_2$. Le contrôle du Jacobien de la
softmax donne alors

$$
\lVert\lambda(F)-\lambda(F')\rVert_1
\le 2|\alpha|\lVert F-F'\rVert_2,
$$

et, pour cette composante où seuls les poids changent,

$$
\lVert A(F)-A(F')\rVert_2
\le U\lVert\lambda(F)-\lambda(F')\rVert_1
\le 2|\alpha|U\lVert F-F'\rVert_2.
$$

Cela explique pourquoi la qualité et la stabilité de la référence comptent
d'autant plus que $|\alpha|$ augmente. Pour $\alpha=0$, les poids sont uniformes
et cette voie d'influence de $F$ sur $A$ est exactement nulle. Cette borne ne
prouve ni fairness, ni optimalité de RCIG, et ne traite pas à elle seule la
dépendance entre poids et bruit du même tour.

L'identité d'espérance du début de cette section a des limites importantes :
elle n'est pas directement valable
si la référence est recalculée sur la même cohorte $F(Y_1,\ldots,Y_n)$, car les
termes deviennent dépendants ; le clipping rend aussi la loi non gaussienne et
modifie covariance et biais. La borne Lipschitz pointwise sur $A(F)$ reste, elle,
valable sous ses hypothèses déterministes. De même, avec des poids du même tour
$\lambda_i(Y_t)$,

$$
\mathbb E\!\left[\sum_i\lambda_i(Y_t)Z_{i,t}\mid\mathcal F_{t-1}\right]
$$

n'est pas automatiquement nul. Cette dépendance est un **risque théorique**,
pas une preuve que la corrélation poids–bruit est positive, ni que le retard ou
RCIG améliorera forcément l'apprentissage. Elle doit être vérifiée empiriquement.
"""),
        _code(r"""
# D2 : diagnostic oracle stratifié de la confusion distance / bruit réalisé.
# Ces variables servent uniquement à l'audit du simulateur ; elles ne sont pas
# disponibles au mécanisme déployé.
d2 = final_df[
    (final_df["campaign"] == "positioning_v3")
    & (final_df["phase"] == "d2_noise_assignment_confirmation")
].copy()

if len(d2) != 48:
    raise RuntimeError(
        f"D2 confirmation incomplète : 48 runs attendus, {len(d2)} trouvés"
    )

oracle_rows = []
for _, row in d2.iterrows():
    payload = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
    all_rounds = payload["rounds"]
    def median_round_metric(key):
        values = pd.to_numeric(
            pd.Series([round_row.get(key, np.nan) for round_row in all_rounds]),
            errors="coerce",
        ).dropna()
        return float(values.median()) if len(values) else np.nan
    assignment = str(row["noise_assignment"])
    severity, order = assignment.split("_", 1)
    oracle_rows.append({
        "n": int(row["n_clients"]),
        "Sévérité": severity,
        "Affectation": order,
        "Référence": row["reference"],
        "Seed": int(row["seed"]),
        "Corr distance-bruit réalisé": median_round_metric(
            "far_distance_dp_noise_corr_oracle"),
        "Corr poids-bruit réalisé": median_round_metric(
            "far_weight_dp_noise_corr_oracle"),
        "Corr score bruité-score propre": median_round_metric(
            "far_honest_noisy_clean_score_corr_oracle"),
        "Rappel top-tail honnête": median_round_metric(
            "far_honest_clean_top_tail_recall_oracle"),
    })

d2_oracle = pd.DataFrame(oracle_rows)
group_keys = ["n", "Sévérité", "Affectation", "Référence"]
diagnostics = [
    "Corr distance-bruit réalisé",
    "Corr poids-bruit réalisé",
    "Corr score bruité-score propre",
    "Rappel top-tail honnête",
]
d2_rows = []
for keys, group in d2_oracle.groupby(group_keys, sort=False):
    out = dict(zip(group_keys, keys))
    for metric in diagnostics:
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        mean = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        out[metric] = (
            f"{mean:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{mean:.3f}"
        )
    d2_rows.append(out)
d2_summary = pd.DataFrame(d2_rows)

display(Markdown(
    "### 2.1.3 D2 — affectations de bruit stratifiées (oracle)\n\n"
    "`identity` et `reverse` permutent les niveaux publics entre identités clientes. "
    "Elles testent si le diagnostic suit le niveau de bruit plutôt qu'une identité "
    "ou une partition particulière. Les trois seeds sont les réplications ; les "
    "deux affectations ne doivent pas être comptées comme six seeds. Chaque valeur "
    "par seed est d'abord la médiane sur les rounds. Ici, « bruit réalisé » désigne "
    "la norme de la réalisation fraîche simulée, connue uniquement de l'oracle, et "
    "non le multiplicateur public sigma."
))
for (n, severity), group in d2_summary.groupby(["n", "Sévérité"]):
    display(Markdown(f"**{n} clients — hétéroscédasticité {severity}**"))
    display(group.drop(columns=["n", "Sévérité"]).sort_values(["Référence", "Affectation"]))
display(Markdown(
    "**Lecture.** Les distances et les poids suivent presque parfaitement "
    "la norme du bruit réalisé, alors que la corrélation score bruité–score "
    "propre est faible, parfois négative. Le rappel mesure la fraction des "
    "20 % d'honnêtes les plus éloignés selon le score propre que retrouve "
    "le classement bruité : 1 signifie tous retrouvés, 0 aucun. Ces clients "
    "sont des outliers géométriques ; leur utilité pour l'apprentissage n'est "
    "pas garantie par cette seule distance.\n\n"
    "Ces corrélations sont calculées **entre clients** à chaque tour. Sous "
    "hétéroscédasticité, elles mélangent l'effet des niveaux de bruit et "
    "celui des fluctuations aléatoires. Elles ne démontrent pas un effet "
    "du tirage frais à niveau de bruit fixé, ni un avantage du retard des poids."
))
d2_oracle.to_csv(RCIG_EXPORT / "d2_oracle_seed_rows.csv", index=False)
d2_summary.to_csv(RCIG_EXPORT / "d2_oracle_stratified_summary.csv", index=False)
print("Source :", RESULTS_ROOT / "positioning_v3" / "d2_noise_assignment_confirmation")
"""),
        _md(r"""
### 2.1.4 Ce que les références statiques n'ont pas résolu

La référence inverse-variance a bien réduit une variance linéaire théorique,
mais le gain ne s'est pas transmis à l'agrégat FAR sur Fashion-MNIST. Les audits
synthétiques suivants (`G0b` à `G0f`) montrent aussi que plusieurs références
Gaussian-aware statiques ont échoué au moins un gate préenregistré de qualité,
de rappel des outliers honnêtes ou de robustesse :

- [G0b](../output/analysis/Gaussian_Aware_Robust_Reference_G0b_MPS.md)
- [G0c](../output/analysis/Gaussian_Aware_Robust_Reference_G0c_MPS.md)
- [G0d-F](../output/analysis/Gaussian_Aware_Robust_Reference_G0d_F_MPS.md)
- [G0e](../output/analysis/Gaussian_Aware_Robust_Reference_G0e_Detailed_Analysis.md)
- [G0f](../output/analysis/Gaussian_Aware_Robust_Reference_G0f_Detailed_Analysis.md)

Ces échecs ne constituent pas un théorème d'impossibilité sur toute référence
robuste. Ils motivent une information supplémentaire : **l'identité temporelle
authentifiée** et la comparaison entre un historique ancien et une fenêtre
récente. RCIG teste alors une rupture, et non une simple grande distance dans
une cohorte instantanée.

### 2.1.5 Construction temporelle RCIG

Soient $A_t$ une fenêtre ancienne et $B_t$ une fenêtre récente, toutes deux
strictement antérieures au tour où la référence est utilisée. Elles produisent
$X_t=F(A_t)$ et $Y_t=F(B_t)$. Les matrices $V_{\mathrm{ancien},t}$ et
$V_{\mathrm{récent},t}$ incorporent déjà les covariances nominales transformées
du bruit DP de leurs fenêtres. Il ne faut donc pas ajouter une seconde fois une
covariance DP. Le protocole K7b utilise :

$$
S_t=V_{\mathrm{ancien},t}+V_{\mathrm{récent},t}
+q_{\mathrm{proc},t}I+\xi I,
\qquad
r_t^2=(Y_t-X_t)^\top S_t^{-1}(Y_t-X_t),
$$

où $q_{\mathrm{proc},t}$ représente la variabilité attendue de l'évolution
honnête, et $\xi>0$ est une régularisation numérique qui permet d'inverser la
matrice. Ces quantités sont des approximations à auditer, pas une preuve que
les sorties clippées sont exactement gaussiennes. Le symbole $\xi$ évite toute confusion avec les poids FAR
$\lambda_i$.

Avec un seuil calibré $c$,

$$
w_t=\min\left\{1,\frac{c}{r_t}\right\},
\qquad
F_{\mathrm{RCIG},t}
=\Pi_{B_G}\!\left[X_t+w_t(Y_t-X_t)\right].
$$

On appelle ici **variation temporelle** la différence $Y_t-X_t$. Si sa norme
standardisée $r_t$ ne dépasse pas le seuil $c$, on a $w_t=1$. Au-delà du seuil,
RCIG réduit son amplitude et rapproche la référence de l'historique : avant
projection, il reste sur le segment entre $X_t$ et $Y_t$. Cela ne classe pas
automatiquement un client comme byzantin. On pose $w_t=1$ lorsque $r_t=0$.
Quand $r_t\le c$, la sortie est exactement $Y_t$
seulement si la projection finale est inactive, c'est-à-dire
$\lVert Y_t\rVert_2\le G$. Elle reste conditionnelle à un historique utile :
si l'attaque persistante contamine déjà $A_t$ et $B_t$, cette formule ne fournit
pas à elle seule une garantie Byzantine universelle.
"""),
        _code(r"""
# K7b : lecture des contrastes verrouillés, des lignes d'évaluation et de
# l'extension nulle. Aucun intervalle n'est recalculé avec une autre règle.
k7_root = RESULTS_ROOT / "gaussian_aware_reference_g0g_k7b_rcig_confirmation_mps_v2"
k7_ext_root = RESULTS_ROOT / "gaussian_aware_reference_g0g_k7b_null_extension_mps_v1"
required_k7 = [
    k7_root / "decision.json",
    k7_root / "evaluation_rows.csv",
    k7_ext_root / "decision.json",
]
missing_k7 = [str(path) for path in required_k7 if not path.exists()]
if missing_k7:
    raise FileNotFoundError("Artefacts K7b absents : " + ", ".join(missing_k7))
k7_decision = json.loads((k7_root / "decision.json").read_text(encoding="utf-8"))
k7_extension = json.loads((k7_ext_root / "decision.json").read_text(encoding="utf-8"))
k7_rows = pd.read_csv(k7_root / "evaluation_rows.csv")

contrast_rows = []
for regime in ["homogeneous", "heteroscedastic"]:
    block = k7_decision["primary_contrasts"][regime]
    for key, label in [
        ("attacked_gain_vs_identity_y", "Sous attaque vs Identity-Y"),
        ("no_attack_gain_vs_identity_y", "Sans attaque vs Identity-Y"),
    ]:
        item = block[key]
        contrast_rows.append({
            "Bruit": regime,
            "Contraste": label,
            "Gain moyen (%)": 100 * item["mean"],
            "IC95 bas (%)": 100 * item["two_sided_low"],
            "IC95 haut (%)": 100 * item["two_sided_high"],
            "n": int(item["n"]),
        })
k7_contrasts = pd.DataFrame(contrast_rows)

specificity_rows = []
for key, label in [
    ("heteroscedastic_full_vs_isotropic", "Covariance complète vs approximation isotrope"),
    ("heteroscedastic_full_vs_euclidean", "Covariance complète vs distance euclidienne"),
]:
    item = k7_decision["gaussian_specificity_contrasts"][key]
    specificity_rows.append({
        "Comparaison, bruit hétéroscédastique": label,
        "Gain MSE (%)": 100 * item["mean"],
        "IC95 bas (%)": 100 * item["two_sided_low"],
        "IC95 haut (%)": 100 * item["two_sided_high"],
        "Seeds": int(item["n"]),
    })
k7_specificity = pd.DataFrame(specificity_rows)

k7_mse = (
    k7_rows.groupby(["noise_regime", "threat"], as_index=False)
    .agg(
        identity_y_mse=("identity_y_mse", "mean"),
        rcig_full_mse=("rcig_full_mse", "mean"),
        rcig_activation=("full_gate_active", "mean"),
        seeds=("seed", "nunique"),
    )
)
k7_mse["gain_rcig_vs_identity_pct"] = 100 * (
    1 - k7_mse["rcig_full_mse"] / k7_mse["identity_y_mse"]
)

null_rows = []
for cell, item in k7_extension["combined_null_audits"].items():
    regime, mode = cell.split("_", 1)
    null_rows.append({
        "Bruit": regime,
        "Mode": mode,
        "Activations": f"{item['combined_activations']}/{item['combined_trials']}",
        "Taux (%)": 100 * item["combined_rate"],
        "Borne CP97,5 (%)": 100 * item["CP97_5_upper"],
        "Passe <= 10 %": bool(item["pass"]),
    })
k7_null = pd.DataFrame(null_rows)

display(Markdown("### 2.1.6 K7b — validation mécanistique synthétique"))
display(Markdown(
    "Ces tests utilisent des vecteurs synthétiques de dimension 8, et non des "
    "images Fashion-MNIST. Identity-Y désigne la référence récente non corrigée. "
    "Le gain est une réduction relative de l'erreur quadratique de la référence "
    "(MSE), **pas un gain d'accuracy**. Les attaques commencent après un historique "
    "propre. Les IC95 ci-dessous portent sur les contrastes appariés entre seeds."
))
display(k7_contrasts.round(2))
display(Markdown("**Apport spécifique de la covariance :**"))
display(k7_specificity.round(2))
display(Markdown(
    "**MSE et activation du gate, calculées depuis `evaluation_rows.csv` :** "
    "le gain du tableau suivant est le rapport des MSE moyennes ; les contrastes "
    "ci-dessus sont la moyenne des gains relatifs appariés. Ces deux statistiques "
    "ne sont pas nécessairement égales."
))
display(k7_mse.round({
    "identity_y_mse": 7,
    "rcig_full_mse": 7,
    "rcig_activation": 3,
    "gain_rcig_vs_identity_pct": 2,
}).rename(columns={
    "noise_regime": "Bruit", "threat": "Attaque",
    "identity_y_mse": "MSE récente Y", "rcig_full_mse": "MSE RCIG",
    "rcig_activation": "Fraction de corrections", "seeds": "Seeds",
    "gain_rcig_vs_identity_pct": "Réduction MSE (%)",
}))
display(Markdown("**Extension nulle indépendante, seuils scientifiques gelés :**"))
display(k7_null.sort_values(["Bruit", "Mode"]).round(2))
display(Markdown(
    "Les bornes nulles de K7b sont **par condition** (360 essais chacune), "
    "et non une borne sur l'union simultanée des six conditions. R1 v2 "
    "ci-dessous teste un critère simultané différent."
))
k7_contrasts.to_csv(RCIG_EXPORT / "k7b_primary_contrasts.csv", index=False)
k7_specificity.to_csv(RCIG_EXPORT / "k7b_covariance_contrasts.csv", index=False)
print("Décision parent K7b :", k7_decision["verdict"])
print("Décision après extension nulle :", k7_extension["verdict"])
print("Limite du claim :", k7_extension["claim_limit"])
"""),
        _code(r"""
# Figure K7b — IC95 lus directement dans decision.json.
plot_df = k7_contrasts.copy()
plot_df["label"] = plot_df.apply(
    lambda r: ("Homogène" if r["Bruit"] == "homogeneous" else "Hétéroscédastique")
    + " — " + ("attaque" if r["Contraste"].startswith("Sous") else "sans attaque"),
    axis=1,
)
plot_df = plot_df.iloc[[0, 2, 1, 3]].reset_index(drop=True)
y = np.arange(len(plot_df))
means = plot_df["Gain moyen (%)"].to_numpy(float)
low = plot_df["IC95 bas (%)"].to_numpy(float)
high = plot_df["IC95 haut (%)"].to_numpy(float)

fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
ax.errorbar(
    means, y,
    xerr=np.vstack([means - low, high - means]),
    fmt="o", color="#2A6F97", ecolor="#61A5C2", capsize=5, markersize=7,
)
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_yticks(y, plot_df["label"])
ax.set_xlabel("Gain relatif de RCIG complet vs Identity-Y (%)")
ax.set_title("K7b synthétique — estimation et IC95 verrouillés")
ax.invert_yaxis()
path = RCIG_EXPORT / "k7b_locked_contrasts_ic95.png"
fig.savefig(path, dpi=180, bbox_inches="tight")
plt.show()
print("Figure :", path)
"""),
        _code(r"""
# Statut du protocole end-to-end v2. R1 audite l'union simultanée de six
# conditions (2 régimes de bruit × 3 modes), pas le seul gate RCIG-full.
v2_root = RESULTS_ROOT / "rcig_end_to_end_v2"
phase_files = {
    "R0 calibration dynamique": v2_root / "_gates" / "r0_dynamic_calibration.json",
    "R1 validation nulle": v2_root / "_gates" / "r1_dynamic_null.json",
    "R2 mécanisme sous attaque": v2_root / "_gates" / "r2_attack_mechanism.json",
    "R3 confirmation end-to-end": v2_root / "_gates" / "r3_e2e_confirmation.json",
}
status_rows, gates = [], {}
for phase, path in phase_files.items():
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        gates[phase] = payload
        status_rows.append({"Phase": phase, "Artefact": "présent", "Décision": payload["decision"]})
    else:
        status_rows.append({"Phase": phase, "Artefact": "absent", "Décision": "non exécutée"})

display(Markdown("### 2.1.7 Statut end-to-end v2 : arrêt au gate nul R1"))
display(pd.DataFrame(status_rows))

r1 = gates["R1 validation nulle"]
e = r1["evidence"]
r1_readout = pd.DataFrame([
    {
        "Quantité": "Union globale des 6 conditions activée",
        "Observé": f"{int(e['global_union_false_activation_count'])}/{int(e['paired_seed_blocks'])}",
        "Seuil": "0/36",
        "Passe": False,
    },
    {
        "Quantité": "Borne CP97,5 de cette union globale",
        "Observé": f"{100 * e['global_union_cp97_5_upper']:.2f} %",
        "Seuil": "<= 10 %",
        "Passe": e["global_union_cp97_5_upper"] <= 0.10,
    },
    {
        "Quantité": "Coût de référence sans attaque, borne unilatérale 97,5 %",
        "Observé": f"{100 * e['no_attack_reference_loss_one_sided_97_5_upper']:.4f} %",
        "Seuil": "<= 2 %",
        "Passe": e["no_attack_reference_loss_one_sided_97_5_upper"] <= 0.02,
    },
])
display(r1_readout)

display(Markdown(
    "**Lecture correcte.** R0 a calibré les six cellules avec une couverture "
    "distribution-free simultanée, puis R1 a observé une activation dans un des "
    "36 blocs appariés. Le taux brut (1/36) est faible, mais sa borne exacte "
    "CP97,5 reste à 14,53 %, au-dessus du seuil préenregistré de 10 % ; le "
    "protocole impose donc `stop`. C'est un échec du critère de validation, "
    "pas une preuve que le vrai taux dépasse 10 %. Cette statistique est l'**union** des modes "
    "full, isotropic et euclidean dans les deux régimes de bruit : elle ne doit "
    "pas être présentée comme le taux de fausse activation du seul RCIG-full. "
    "Le coût sans attaque passe largement son gate, mais R2 et R3 étant absents, "
    "nous n'avons encore aucune confirmation v2 sur attaques persistantes, "
    "accuracy, Worst-20, gap ou robustesse end-to-end."
))
print("Sources :", phase_files["R0 calibration dynamique"], "et", phase_files["R1 validation nulle"])
display(Markdown('''
#### Comprendre « 1 bloc sur 36 » et « borne supérieure de 14,53 % »

Un **bloc apparié** correspond à une seed et à deux entraînements de 40 tours,
l'un à bruit homogène et l'autre à bruit hétéroscédastique. Dans chaque
entraînement, trois règles (covariance complète, isotrope, distance
euclidienne) sont évaluées sur le même transcript privé. On compte le bloc
une seule fois dès qu'au moins une règle s'active à un tour évalué dans l'un
des deux régimes. Ce ne sont donc ni 36 clients, ni 36 tours, ni six
répétitions indépendantes par seed.

Dans R1, **aucune attaque n'est présente**. Une activation est donc une
fausse alarme du détecteur de rupture. La seed **820005** est le seul bloc
positif : en bruit homogène, la règle euclidienne s'active au tour 20, puis
les trois règles au tour 21 ; en bruit hétéroscédastique, les trois règles
s'activent au tour 21. Ces activations liées comptent ensemble comme **un
seul bloc positif**.

- **Fréquence observée : 1 / 36 = 2,78 %.**
- **Borne supérieure unilatérale exacte à 97,5 % : 14,53 %.** C'est une
  marge d'incertitude sur le taux de blocs positifs dans de nouvelles
  répétitions indépendantes du même protocole, pas un taux mesuré de 14,53 %.
  La construction Clopper–Pearson garantit une couverture d'au moins 97,5 %
  lorsque ses hypothèses binomiales sont satisfaites.
- Le protocole exigeait **zéro bloc positif et une borne au plus égale à
  10 %**. Avec zéro sur 36, la borne aurait été 9,74 % ; avec un sur 36,
  elle devient 14,53 %. Le gate ne passe donc pas. Cela ne démontre pas que
  le vrai taux dépasse 10 %, et ne mesure pas une perte d'accuracy.

**R1 n'est pas relancé sur les mêmes seeds** : cela n'ajouterait pas de
répétitions indépendantes. Son échec reste enregistré. R2 et R3 sont demandés
comme une **continuation exploratoire séparée**, sans promotion de R1 et sans
retouche de ses seuils. Leurs sorties sont séparées dans
`results/ldp_gradient_far/rcig_r2_r3_exploratory_v1` ; le tableau v2 ci-dessus
continue de décrire uniquement le protocole confirmatoire initial.
'''))
"""),
        _md(r"""
### 2.1.8 Verdict scientifique actuel

| Niveau | Ce qui est soutenu | Ce qui ne l'est pas encore |
|---|---|---|
| Vrais gradients privés, sans attaque | FAR positif et la référence inverse-variance peuvent dégrader performance et fairness, surtout sous hétéroscédasticité | causalité universelle du bruit, supériorité d'une référence temporelle |
| Oracles D2 | le score brut suit fortement le niveau de bruit dans ces cellules et retrouve mal la géométrie propre | diagnostic déployable ; preuve Byzantine |
| K7b synthétique | RCIG réduit la MSE lors d'une attaque qui commence après un historique propre ; la covariance complète apporte un gain sous bruit hétéroscédastique | accuracy/fairness ; attaque persistante contaminant les deux fenêtres ; haute dimension |
| End-to-end v2 | R0 est valide ; R1 montre un coût de référence négligeable dans le protocole | le gate nul simultané échoue ; R2/R3 ne sont pas exécutés |

**Statut : RCIG est un candidat mécanistiquement motivé, pas un algorithme final
validé.** Le passage à une revendication publiable nécessite un protocole nul
qui passe sans retoucher les seuils, puis les phases verrouillées sur vrais
gradients privés sous attaques persistantes et, enfin, les métriques end-to-end.
"""),
    ]


INITIAL_RATIONALE = r"""
### 2.1.4 Une meilleure variance de référence ne suffit pas

La première correction inverse-variance favorisait les clients dont le bruit
était plus faible. Dans l'exemple de deux groupes de même taille, de variances
respectives v et 4v, les poids internes de référence deviennent 0,16 et 0,04
à n = 10 : les masses de groupe sont 0,8 et 0,2, au lieu de 0,5 et 0,5.
Dans le cas linéaire idéal sans clipping, sa variance est 0,16v au lieu de
0,25v pour l'uniforme, soit une baisse de 36 %.

Mais, si les moyennes propres des groupes diffèrent, cette pondération change
aussi la cible. Elle ne préserve pas automatiquement la représentation d'un
groupe honnête plus bruité. De plus, FAR utilise ensuite les distances à cette
référence pour repondérer les gradients : **optimiser la variance de F n'est
pas optimiser la qualité de l'agrégat FAR**.

C'est précisément ce qu'illustre la campagne réelle ci-dessus : malgré ce gain
de variance théorique, la variante inverse-variance ne restaure ni l'accuracy
ni le Worst-20 sous bruit hétéroscédastique. La dérivation de la cible et de la
covariance résiduelle figure dans
[l'analyse préalable des références](../output/analysis/Gaussian_Aware_Robust_Reference_Beyond_FCC.md).
La réduction de 36 % n'est pas une mesure de variance après toute la chaîne
non linéaire de clipping.

### 2.1.5 Ce que nos premiers essais réels de mémoire nous avaient appris

Il ne faut pas réécrire l'histoire en affirmant que « la mémoire marchait
déjà ». Les essais antérieurs à RCIG étaient exploratoires et peu favorables :

| Essai sur Fashion-MNIST | Observation | Enseignement pour la construction |
|---|---|---|
| Stage 14B : 25 clients, 6 tours, seed 28 | Remplacer FCC courant par une référence passée donne des deltas d'accuracy allant de −0,02 à +0,38 pp selon la menace ; pas de gain cohérent de gap | Un historique utilisé systématiquement n'est pas une solution validée |
| Stage 14C : 25 clients, 20 tours, seed 28, propre → attaque → récupération | FCC temporel : erreur propre de référence 0,0051 contre 0,0025 pour FCC courant ; delta d'accuracy finale moyen −0,17 pp sur les trois menaces | La mémoire peut introduire un décalage nuisible même avant l'attaque |
| Même stage 14C | Les neuf bras reviennent à l'enveloppe propre dès le premier tour de récupération | Ce stress ne démontre pas de bénéfice de mémoire face à un empoisonnement persistant |

Sources : [stage 14B](../output/analysis/DT_LDP_FAR_Stage14B_FashionMNIST_Temporal_Reference.md)
et [stage 14C](../output/analysis/DT_LDP_FAR_Stage14C_Temporal_Poison_Recovery.md).
Ce sont des entraînements avec DP-SGD local à pas Poisson, distincts de la
campagne LDP-Gradient-FAR à gradient unique et batch fixe. Leurs chiffres ne
sont pas fusionnés ; ils apportent un enseignement de conception, pas une
comparaison directe d'algorithmes sous le même protocole.

**Inférence, pas résultat démontré :** si l'on réutilise l'historique, il faut
envisager de le faire sélectivement. Quand la vue récente paraît compatible
avec l'évolution honnête et le bruit attendu, la conserver évite une
correction temporelle inutile. C'est la raison de chercher une **tolérance
statistique au bruit**, plutôt qu'un simple remplacement systématique par le
passé.

### 2.1.6 Pourquoi envisager une référence temporelle sélective ?

**Ce que les données nous ont appris :** une grande distance instantanée peut
refléter le bruit, et une modification de référence peut déplacer les poids
sans améliorer la fairness. **Ce qu'elles n'ont pas prouvé :** qu'un historique
permettrait nécessairement de reconnaître les Byzantins.

Le passage au temporel vient du raisonnement suivant. Dans une seule cohorte,
trois causes peuvent produire un message éloigné : données honnêtes atypiques,
perturbation DP, ou attaque. Si un attaquant reproduit exactement la même loi
observable qu'un honnête atypique, aucune règle utilisant seulement cette
observation ne peut distinguer les deux. Une hypothèse supplémentaire est donc
nécessaire : séparation entre groupes, structure connue, ou continuité
temporelle. **L'historique est l'option que nous proposons d'exploiter ; ce n'est
pas une nécessité universelle démontrée.**

Pour visualiser cette idée, considérons le modèle explicatif avant clipping
serveur, pour une identité cliente i suivie à deux instants :

$$
Y_{i,t}=g_{i,t}+Z_{i,t}+b_{i,t},
\qquad
Y_{i,t}-Y_{i,s}
=(g_{i,t}-g_{i,s})+(Z_{i,t}-Z_{i,s})+(b_{i,t}-b_{i,s}).
$$

Ici g est le gradient propre, Z le bruit DP et b un déplacement d'attaque
(nul pour un client honnête). Ce modèle explicatif ne suppose pas que le
serveur connaît g ou b. Le niveau atypique mais stable de g peut disparaître
dans la différence ; restent son évolution normale, la fluctuation de bruit,
et un éventuel changement d'attaque.

Dans le cas idéal de bruits indépendants à covariances fixées,
Cov(Z au temps t − Z au temps s) = Σ au temps t + Σ au temps s.
On peut donc chercher à comparer une évolution récente à son incertitude
attendue, plutôt que juger uniquement sa distance brute. Les fenêtres de
plusieurs tours peuvent réduire la fluctuation, **au prix d'un retard face à
l'évolution du vrai gradient**. Après clipping et apprentissage adaptatif,
il faut auditer les covariances effectives et le drift ; cette identité simple
ne suffit plus à fournir un test calibré.

### 2.1.7 Du problème observé au cahier des charges — sans résultat synthétique

| Point de départ | Conclusion défendable | Propriété recherchée |
|---|---|---|
| Sous DP, FAR positif détériore Worst-20 et gap dans les comparaisons réelles affichées | L'inclusion par distance ne préserve pas automatiquement la fairness | Vérifier ensemble qualité du centre, poids finaux et fairness |
| Dans D2, distances/poids suivent le bruit tandis que le classement propre est mal retrouvé | Une grande distance ne signifie pas nécessairement signal honnête utile | Tenir compte du niveau et de la géométrie du bruit DP attendu |
| La référence inverse-variance ne transmet pas son gain à l'agrégat | Réduire la variance seule et favoriser systématiquement les clients peu bruités ne suffit pas | Tolérer une variation honnête compatible avec son incertitude |
| Pour α positif, même une bonne référence peut donner un grand poids à un message byzantin éloigné | Référence robuste et agrégat FAR robuste sont deux exigences distinctes | Limiter l'influence d'une rupture anormale, sans exclure tous les honnêtes atypiques |
| Une cohorte instantanée ne contient pas toujours assez d'information pour séparer ces causes | La continuité temporelle peut être une information auxiliaire, sous hypothèses | Comparer historique et évolution récente, avec une tolérance au drift et au bruit |

Le lien avec l'erreur d'agrégation et la convergence est expliqué dans
[l'analyse théorique préalable](../output/analysis/LDP_Gradient_FAR_Convergence_Alpha_Byzantine_DP.md).
Les diagnostics empiriques sous attaques sont détaillés séparément en section 7.

**L'intuition à présenter est donc :** ne pas punir une déviation simplement
parce qu'elle est grande ; examiner si son évolution est compatible avec le
bruit DP et le drift honnête attendus, et limiter son effet lorsqu'elle ne
l'est pas. C'est le cahier des charges qui motive un mécanisme comme RCIG.

Ce n'est pas encore une preuve d'identification des Byzantins. Une évolution
honnête rapide peut être signalée à tort ; une attaque lente ou déjà présente
dans l'historique peut passer inaperçue. Les expériences **après construction**
doivent précisément tester ces limites, sans servir rétrospectivement à
justifier l'intuition de départ. En particulier, cette piste n'est pas fondée
sur un avantage préalablement démontré du retard des poids FAR : exploiter un
historique de référence est une autre proposition.
"""


def build_cells() -> list:
    """Motivation using real training and pre-evaluation reasoning only."""
    return _all_cells()[:6] + [_md(INITIAL_RATIONALE)]


def build_evaluation_cells() -> list:
    """Development/validation evidence, deliberately outside the motivation."""
    cells = _all_cells()[6:]
    replacements = {
        "### 2.1.4 Ce que les références statiques n'ont pas résolu": (
            "## 15. Développement et évaluation de RCIG — distincts de sa motivation initiale\n\n"
            "La section 2.1 expose le problème sur vrais entraînements et le raisonnement "
            "de construction. Cette section rassemble les essais synthétiques de "
            "développement puis la validation du candidat. Ils ne sont pas présentés "
            "comme la preuve qui aurait initialement imposé le choix de RCIG.\n\n"
            "### 15.1 Essais intermédiaires de références statiques"
        ),
        "### 2.1.5 Construction temporelle RCIG": "### 15.2 Construction testée de RCIG",
        "### 2.1.6 K7b": "### 15.3 K7b",
        "### 2.1.7 Statut": "### 15.4 Statut",
        "### 2.1.8 Verdict": "### 15.5 Verdict",
    }
    result = []
    for source_cell in cells:
        source = source_cell.source
        for old, new in replacements.items():
            source = source.replace(old, new)
        result.append(_cell(source_cell.cell_type, source))
    return result


__all__ = ["build_cells", "build_evaluation_cells"]
