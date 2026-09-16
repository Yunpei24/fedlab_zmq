#!/usr/bin/env python3
"""Append the final positioning-v3 and G0d-F audit sections to the notebook.

The updater is deliberately narrow and idempotent:

* the original cells are retained in their original order;
* one obsolete paragraph about the E/F attack phases is replaced in place;
* generated sections 12 and 13 are identified by stable notebook-cell tags;
* an untouched backup is copied to ``/tmp`` before the notebook is written.

The generated cells only read published artifacts.  They never recompute or
modify a scientific gate, and this updater never executes the notebook.
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
SECTION12_TAG = "ldp-positioning-v3-final-392-2026-09"
SECTION13_TAG = "ldp-gaussian-aware-g0d-f-2026-09"
GENERATED_TAGS = {SECTION12_TAG, SECTION13_TAG}


OBSOLETE_ATTACK_TEXT = """Les matrices prévoient des contrôles à 0 % et des attaques à 20 %. Le snapshot
exécuté ci-dessous ne contient encore que 0 % : aucun effet byzantin ne peut
donc être estimé à ce stade, et deux niveaux ne permettraient de toute façon
pas une dose-réponse. De même, le passage de 10 à 25 clients change aussi la
taille locale (N_i), le batch et parfois (C); il s'agit d'un diagnostic de
transfert d'échelle, pas d'un effet causal pur du nombre de clients."""

CURRENT_ATTACK_TEXT = """Les phases E et F sont désormais terminées : E contient 70/70 runs et la
confirmation principale F à 20 rounds contient 180/180 runs. Les cellules
attaquées fixent la fraction byzantine à 20 %, avec des contrôles appariés à
0 %. Ce contraste permet d'estimer l'effet présence/absence des attaques
préenregistrées, mais pas une dose-réponse continue à la fraction byzantine.
L'extension F2 est également complète (36/36 runs) et conserve l'horizon de
40 rounds comme strate séparée. Enfin, le passage de 10 à 25 clients change
aussi la taille locale ($N_i$), le batch et parfois $C$ : il s'agit d'un
diagnostic de transfert d'échelle, pas d'un effet causal pur du nombre de
clients."""


def markdown(source: str, tag: str) -> nbformat.NotebookNode:
    normalized = source.strip()
    cell = nbformat.v4.new_markdown_cell(
        normalized, metadata={"tags": [tag]}
    )
    cell.id = hashlib.sha256(
        f"markdown\0{tag}\0{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return cell


def code(source: str, tag: str) -> nbformat.NotebookNode:
    normalized = source.strip()
    cell = nbformat.v4.new_code_cell(normalized, metadata={"tags": [tag]})
    cell.id = hashlib.sha256(
        f"code\0{tag}\0{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    return cell


def remove_generated_cells(
    cells: list[nbformat.NotebookNode],
) -> list[nbformat.NotebookNode]:
    return [
        cell
        for cell in cells
        if GENERATED_TAGS.isdisjoint(
            set(cell.get("metadata", {}).get("tags", []))
        )
    ]


def update_attack_text(cells: list[nbformat.NotebookNode]) -> None:
    matches = [
        cell
        for cell in cells
        if cell.cell_type == "markdown"
        and str(cell.source).lstrip().startswith(
            "## 7. Attaques byzantines et références robustes"
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one section-7 attack markdown cell, found "
            f"{len(matches)}"
        )
    cell = matches[0]
    source = str(cell.source)
    if OBSOLETE_ATTACK_TEXT in source:
        cell.source = source.replace(OBSOLETE_ATTACK_TEXT, CURRENT_ATTACK_TEXT)
    elif CURRENT_ATTACK_TEXT not in source:
        raise RuntimeError(
            "Section 7 no longer contains either the registered obsolete text "
            "or its updated replacement; refusing a broad rewrite."
        )


def positioning_cells() -> list[nbformat.NotebookNode]:
    tag = SECTION12_TAG
    return [
        markdown(
            r"""
## 12. Positioning v3 — bilan final préenregistré

Cette section lit les artefacts finaux déjà produits par l'analyseur. Elle ne
recalcule aucun gate. La campagne est complète à **392/392 runs valides** :
$180/180$ pour la confirmation principale F à $T=20$ et $36/36$ pour
l'extension F2 à $T=40$. Les deux horizons restent des strates distinctes.

Les fichiers `*_gate_evidence.json` ci-dessous sont affichés tels quels. Leur
présence documente les contrôles préenregistrés disponibles ; l'absence d'un
fichier spécifique à F2 ne doit pas être remplacée par un gate improvisé.
""",
            tag,
        ),
        code(
            r"""
POSITIONING_V3_ANALYSIS = (
    ROOT / "output" / "analysis" / "ldp_gradient_far_positioning_v3"
)
POSITIONING_PHASE_STATUS = POSITIONING_V3_ANALYSIS / "phase_status.csv"
POSITIONING_VARIANTS = POSITIONING_V3_ANALYSIS / "variant_summary.csv"

if not POSITIONING_PHASE_STATUS.exists():
    raise FileNotFoundError(POSITIONING_PHASE_STATUS)
if not POSITIONING_VARIANTS.exists():
    raise FileNotFoundError(POSITIONING_VARIANTS)

positioning_phase_status = pd.read_csv(POSITIONING_PHASE_STATUS)
positioning_gate_files = sorted(
    POSITIONING_V3_ANALYSIS.glob("*gate_evidence.json")
)
if not positioning_gate_files:
    raise FileNotFoundError(
        f"Aucun *gate_evidence.json dans {POSITIONING_V3_ANALYSIS}"
    )
positioning_gate_evidence = {
    path.stem.removesuffix("_gate_evidence"): json.loads(
        path.read_text(encoding="utf-8")
    )
    for path in positioning_gate_files
}

expected_total = int(positioning_phase_status["expected"].sum())
complete_total = int(positioning_phase_status["complete"].sum())
invalid_total = int(positioning_phase_status["invalid"].sum())
missing_total = int(positioning_phase_status["missing"].sum())
if (expected_total, complete_total, invalid_total, missing_total) != (392, 392, 0, 0):
    raise AssertionError(
        "Le snapshot positioning-v3 n'est pas le bilan final attendu : "
        f"{complete_total}/{expected_total}, invalid={invalid_total}, "
        f"missing={missing_total}."
    )

display(Markdown(
    f"**Complétude vérifiée : {complete_total}/{expected_total}; "
    f"invalides : {invalid_total}; manquants : {missing_total}.**"
))
display(positioning_phase_status)

gate_rows = []
for phase, evidence in positioning_gate_evidence.items():
    row = {"phase": phase}
    row.update(evidence)
    gate_rows.append(row)
positioning_gate_table = pd.DataFrame(gate_rows).sort_values("phase")
display(positioning_gate_table)
""",
            tag,
        ),
        code(
            r"""
positioning_variant_summary = pd.read_csv(POSITIONING_VARIANTS)
final_phase_names = ["f_confirmation_t20", "f2_horizon_extension_t40"]
positioning_f = positioning_variant_summary[
    positioning_variant_summary["phase"].isin(final_phase_names)
].copy()

f_columns = [
    "phase", "variant_id", "num_clients", "num_rounds", "seeds_complete",
    "dp_enabled", "target_epsilon", "attack", "far_alpha",
    "robust_reference", "test_accuracy_final_pct_mean",
    "test_accuracy_final_pct_sd", "worst20_final_pct_mean",
    "worst20_final_pct_sd", "gap_final_pp_mean", "gap_final_pp_sd",
    "variance_final_pp2_mean", "variance_final_pp2_sd",
]
display(
    positioning_f[f_columns]
    .sort_values(["phase", "num_clients", "attack", "variant_id"])
    .reset_index(drop=True)
)

t20 = positioning_f[positioning_f.phase == "f_confirmation_t20"].set_index(
    "variant_id"
)
t40 = positioning_f[
    positioning_f.phase == "f2_horizon_extension_t40"
].set_index("variant_id")
paired_ids = t20.index.intersection(t40.index)
paired_horizons = pd.DataFrame(index=paired_ids)
for metric in (
    "test_accuracy_final_pct_mean",
    "worst20_final_pct_mean",
    "gap_final_pp_mean",
    "variance_final_pp2_mean",
):
    short = metric.removesuffix("_final_pct_mean").removesuffix("_final_pp_mean")
    short = short.removesuffix("_final_pp2_mean")
    paired_horizons[f"{short}_T20"] = t20.loc[paired_ids, metric]
    paired_horizons[f"{short}_T40"] = t40.loc[paired_ids, metric]
    paired_horizons[f"delta_{short}_T40_minus_T20"] = (
        t40.loc[paired_ids, metric] - t20.loc[paired_ids, metric]
    )
paired_horizons = paired_horizons.reset_index(names="variant_id")
display(Markdown(
    "### Comparaison strictement appariée des variantes communes à T=20 et T=40"
))
display(paired_horizons.round(4))

delta_columns = [
    column for column in paired_horizons if column.startswith("delta_")
]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, column in zip(axes.ravel(), delta_columns):
    ordered = paired_horizons.sort_values(column)
    ax.barh(ordered["variant_id"], ordered[column])
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle=":")
    ax.set_title(column.replace("delta_", "Δ ").replace("_", " "))
fig.suptitle("F2 (T=40) − F (T=20), variantes strictement appariées")
plt.tight_layout()
plt.show()
""",
            tag,
        ),
        markdown(
            r"""
**Lecture.** F fournit l'estimation multi-seeds principale au même horizon que
le développement. F2 teste uniquement le transfert à un horizon plus long.
Une différence F2−F ne doit donc pas être mélangée à l'estimation primaire et
ne crée pas rétroactivement un nouveau critère de sélection.
""",
            tag,
        ),
    ]


def g0d_cells() -> list[nbformat.NotebookNode]:
    tag = SECTION13_TAG
    return [
        markdown(
            r"""
## 13. G0d-F — référence Gaussian-aware calibrée sous le nul

G0d-F est un audit **référence-seulement** : il n'utilise ni poids FAR, ni
accuracy, ni entraînement du réseau. Les rayons de Huber sont calibrés sur une
simulation nulle indépendante et poolée, avec une référence pilote
leave-one-out. Douze candidats sont comparés en développement ; un seul est
verrouillé avant l'évaluation sur de nouvelles seeds holdout.

Les tables suivantes chargent tous les CSV et JSON dédiés. Elles affichent les
gates déjà enregistrés sans les recalculer ni les modifier.
""",
            tag,
        ),
        code(
            r"""
G0D_ROOT = (
    ROOT / "results" / "ldp_gradient_far"
    / "gaussian_aware_reference_g0d_f_mps_v1"
)
if not G0D_ROOT.exists():
    raise FileNotFoundError(G0D_ROOT)

g0d_csv = {
    path.stem: pd.read_csv(path)
    for path in sorted(G0D_ROOT.glob("*.csv"))
}
g0d_json = {
    path.stem: json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(G0D_ROOT.glob("*.json"))
}

required_csv = {
    "null_calibration", "development_detail", "development_stability",
    "development_summary", "holdout_detail", "holdout_stability",
    "holdout_summary", "holdout_comparators",
}
required_json = {"development_lock", "decision", "run_manifest"}
missing_csv = required_csv - set(g0d_csv)
missing_json = required_json - set(g0d_json)
resolved_config_path = G0D_ROOT / "resolved_config.yaml"
if missing_csv or missing_json or not resolved_config_path.exists():
    raise FileNotFoundError(
        f"Artefacts G0d-F manquants — CSV={sorted(missing_csv)}, "
        f"JSON={sorted(missing_json)}, "
        f"resolved_config={resolved_config_path.exists()}"
    )
g0d_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))

expected_counts = {
    "development_detail": 7020,
    "development_stability": 1080,
    "holdout_detail": 9996,
    "holdout_stability": 588,
}
observed_counts = {name: len(g0d_csv[name]) for name in expected_counts}
if observed_counts != expected_counts:
    raise AssertionError(
        f"Complétude G0d-F inattendue : {observed_counts} != {expected_counts}"
    )

print("CSV chargés :", sorted(g0d_csv))
print("JSON chargés :", sorted(g0d_json))
display(pd.DataFrame([{"table": key, "lignes": value} for key, value in observed_counts.items()]))
display(pd.DataFrame([g0d_json["run_manifest"]]))
""",
            tag,
        ),
        markdown(
            r"""
### 13.1 Calibration nulle poolée et leave-one-out

Pour chaque bloc et quantile, la même valeur publique est appliquée aux cellules
homogène, hétéroscédastique-identity et hétéroscédastique-reverse. Les taux de
queue par cellule peuvent différer : cette différence est un diagnostic, pas
une recalibration conditionnelle au groupe observé.
""",
            tag,
        ),
        code(
            r"""
g0d_calibration = g0d_csv["null_calibration"].copy()
calibration_table = (
    g0d_calibration.groupby(["quantile", "block", "threshold"], as_index=False)
    .agg(
        cells=("calibration_pool", "size"),
        tail_rate_min=("empirical_tail_rate", "min"),
        tail_rate_max=("empirical_tail_rate", "max"),
        null_residuals_pool=("num_null_residuals_in_pool", "max"),
    )
    .sort_values(["quantile", "block"])
)
display(calibration_table.round(6))

fig, ax = plt.subplots(figsize=(9, 5))
for block, group in calibration_table.groupby("block"):
    ax.plot(group["quantile"], group["threshold"], marker="o", label=f"bloc {block}")
ax.set_xlabel("Quantile nul préenregistré")
ax.set_ylabel("Seuil standardisé")
ax.set_title("G0d-F — seuils issus de la calibration nulle poolée")
ax.legend(ncol=2)
plt.tight_layout()
plt.show()
""",
            tag,
        ),
        markdown(
            r"""
### 13.2 Les 12 candidats de développement et le verrou

Le verrou minimise d'abord le nombre de gates échoués, puis la pénalité
normalisée, selon la règle préenregistrée. Le holdout n'intervient jamais dans
ce classement.
""",
            tag,
        ),
        code(
            r"""
g0d_dev = g0d_csv["development_summary"].copy()
gate_columns = [column for column in g0d_dev if column.startswith("gate_")]
dev_columns = [
    "candidate", "observations", "complete_fraction",
    "clean_error_ratio_worst_group",
    "attacked_error_ratio_to_uniform_worst_group",
    "attacked_error_ratio_to_fcc_worst_group",
    "evasive_error_ratio_to_uniform_worst_group",
    "evasive_error_ratio_to_fcc_worst_group",
    "regular_honest_tail_rate", "identity_reverse_ci95_low",
    "identity_reverse_ci95_high", "replace_one_bound_max",
    "replace_one_observed_max", "solver_gradient_residual_max",
    "covariance_limited_and_tail_fraction", "gate_fail_count",
    "passes_all_gates", "normalized_gate_penalty",
]
g0d_dev_ranked = g0d_dev.sort_values(
    ["gate_fail_count", "normalized_gate_penalty", "candidate"]
)
display(g0d_dev_ranked[dev_columns].reset_index(drop=True))

g0d_lock = g0d_json["development_lock"]
display(Markdown(f"**Candidat verrouillé : `{g0d_lock['candidate']}`.**"))
display(pd.DataFrame([{
    key: value for key, value in g0d_lock.items()
    if key in dev_columns or key.startswith("selected_on_")
}]))

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["tab:green" if value else "tab:red" for value in g0d_dev_ranked["passes_all_gates"]]
ax.bar(g0d_dev_ranked["candidate"], g0d_dev_ranked["gate_fail_count"], color=colors)
ax.set_ylabel("Nombre de gates échoués")
ax.set_title("G0d-F — classement développement des 12 candidats")
ax.tick_params(axis="x", rotation=65)
plt.tight_layout()
plt.show()
""",
            tag,
        ),
        markdown(
            r"""
### 13.3 Gates enregistrés : développement et holdout

Le tableau suivant transpose uniquement les booléens déjà sauvegardés. Un gate
à `False` reste un échec : le notebook n'en change ni le seuil ni le sens.
""",
            tag,
        ),
        code(
            r"""
g0d_holdout = g0d_csv["holdout_summary"].iloc[0].to_dict()
stored_gate_names = sorted(
    key for key in set(g0d_lock) | set(g0d_holdout) if key.startswith("gate_")
)
stored_gate_table = pd.DataFrame({
    "gate": stored_gate_names,
    "development": [g0d_lock.get(key) for key in stored_gate_names],
    "holdout": [g0d_holdout.get(key) for key in stored_gate_names],
})
display(stored_gate_table)

decision = g0d_json["decision"]
decision_view = pd.DataFrame([decision])
display(Markdown(
    "**Décision enregistrée : "
    + ("PROMOTION" if decision["promote"] else "AUCUNE PROMOTION")
    + ".**"
))
display(decision_view)

gates = g0d_config["gates"]
gate_measurements = [
    ("Erreur propre / uniforme", "clean_error_ratio_worst_group", "≤ 1.00"),
    ("Attaques séparées / uniforme", "attacked_error_ratio_to_uniform_worst_group", "≤ 0.90"),
    ("Attaques séparées / FCC", "attacked_error_ratio_to_fcc_worst_group", "≤ 1.05"),
    ("ALIE / uniforme", "evasive_error_ratio_to_uniform_worst_group", "≤ 1.05"),
    ("ALIE / FCC", "evasive_error_ratio_to_fcc_worst_group", "≤ 1.05"),
    ("Queue honnête régulière", "regular_honest_tail_rate", "[0.06, 0.20]"),
    ("Borne replace-one", "replace_one_bound_max", "≤ 0.12"),
    ("Violations replace-one", "replace_one_violation_count", "= 0"),
    ("Résidu solveur", "solver_gradient_residual_max", "≤ 1e-6"),
    ("Covariance active dans la queue", "covariance_limited_and_tail_fraction", "≥ 0.05"),
]
gate_boolean_by_measure = {
    "clean_error_ratio_worst_group": "gate_clean_error",
    "attacked_error_ratio_to_uniform_worst_group": "gate_attacked_vs_uniform",
    "attacked_error_ratio_to_fcc_worst_group": "gate_attacked_vs_fcc",
    "evasive_error_ratio_to_uniform_worst_group": "gate_evasive_vs_uniform",
    "evasive_error_ratio_to_fcc_worst_group": "gate_evasive_vs_fcc",
    "regular_honest_tail_rate": "gate_honest_tail_rate",
    "replace_one_bound_max": "gate_replace_one_bound",
    "replace_one_violation_count": "gate_replace_one_empirical",
    "solver_gradient_residual_max": "gate_solver_residual",
    "covariance_limited_and_tail_fraction": "gate_covariance_effective",
}
gate_value_table = pd.DataFrame([
    {
        "gate": label,
        "seuil préenregistré": threshold,
        "développement": g0d_lock[field],
        "holdout": g0d_holdout[field],
        "verdict holdout": bool(g0d_holdout[gate_boolean_by_measure[field]]),
    }
    for label, field, threshold in gate_measurements
])
display(Markdown("#### Valeurs, seuils et verdicts — candidat verrouillé"))
display(gate_value_table)
""",
            tag,
        ),
        markdown(
            r"""
### 13.4 Comparateurs appariés et stabilité replace-one

FCC, RFA, moyenne tronquée et médiane coordonnée sont évaluées sur exactement
les mêmes cohortes, bruits, attaques et seeds que le candidat verrouillé. Une
bonne erreur empirique ne crée pas un certificat : les champs `N/A` de RFA et
de la médiane coordonnée doivent rester `N/A`.
""",
            tag,
        ),
        code(
            r"""
g0d_comparators = g0d_csv["holdout_comparators"].copy()
comparator_columns = [
    "candidate", "observations",
    "clean_error_ratio_to_uniform_worst_group",
    "attacked_error_ratio_to_uniform_worst_group",
    "attacked_error_ratio_to_fcc_worst_group",
    "evasive_error_ratio_to_uniform_worst_group",
    "evasive_error_ratio_to_fcc_worst_group",
    "clean_population_error_mean", "identity_reverse_ci95_low",
    "identity_reverse_ci95_high", "empirical_replace_one_max",
    "global_o_1_over_n_certificate", "theoretical_bound",
]
display(g0d_comparators[comparator_columns])

g0d_holdout_detail = g0d_csv["holdout_detail"].copy()
g0d_holdout_detail["attack_group"] = np.select(
    [
        g0d_holdout_detail["threat"].eq("none"),
        g0d_holdout_detail["threat"].eq("alie"),
    ],
    ["propre", "ALIE"],
    default="attaques séparées",
)
raw_error_table = (
    g0d_holdout_detail.groupby(["candidate", "attack_group"])["reference_error"]
    .mean().unstack("attack_group")
    .reindex(columns=["propre", "attaques séparées", "ALIE"])
    .sort_values("attaques séparées")
)
display(Markdown(
    "#### Erreur brute moyenne sur toutes les cellules appariées du holdout\n\n"
    "Cette moyenne descriptive complète les ratios de pire groupe utilisés par les gates."
))
display(raw_error_table)

locked_candidate = decision["locked_candidate"]
paired_left = g0d_holdout_detail[
    g0d_holdout_detail.candidate.eq(locked_candidate)
][["pairing_id", "seed", "attack_group", "reference_error"]].rename(
    columns={"reference_error": "error_g0d"}
)
paired_right = g0d_holdout_detail[
    g0d_holdout_detail.candidate.eq("fcc")
][["pairing_id", "reference_error"]].rename(
    columns={"reference_error": "error_fcc"}
)
paired_g0d_fcc = paired_left.merge(
    paired_right, on="pairing_id", how="inner", validate="one_to_one"
)
paired_g0d_fcc["difference_g0d_minus_fcc"] = (
    paired_g0d_fcc.error_g0d - paired_g0d_fcc.error_fcc
)
seed_differences = (
    paired_g0d_fcc.groupby(["attack_group", "seed"])["difference_g0d_minus_fcc"]
    .mean().reset_index()
)

def seven_seed_t_interval(values):
    values = pd.to_numeric(values, errors="raise")
    if len(values) != 7:
        raise AssertionError(f"Sept seeds holdout attendues, reçu {len(values)}")
    mean = float(values.mean())
    half_width = 2.446912 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return pd.Series({"différence moyenne": mean, "IC95 bas": mean-half_width, "IC95 haut": mean+half_width})

paired_seed_ci = (
    seed_differences.groupby("attack_group")["difference_g0d_minus_fcc"]
    .apply(seven_seed_t_interval).unstack()
    .reindex(["propre", "attaques séparées", "ALIE"])
)
display(Markdown(
    "#### Différence appariée G0d-F − FCC, agrégée par seed\n\n"
    "Une valeur positive signifie que G0d-F a une erreur plus grande. "
    "Les IC95 t sont descriptifs et non corrigés pour comparaisons multiples."
))
display(paired_seed_ci)

ratio_columns = [
    "clean_error_ratio_to_uniform_worst_group",
    "attacked_error_ratio_to_uniform_worst_group",
    "evasive_error_ratio_to_uniform_worst_group",
]
plot_comparators = g0d_comparators.set_index("candidate")[ratio_columns]
ax = plot_comparators.plot(kind="bar", figsize=(13, 5))
ax.axhline(1.0, color="black", linewidth=0.9, linestyle=":")
ax.set_ylabel("Ratio d'erreur")
ax.set_title("G0d-F — comparateurs appariés sur le holdout")
ax.legend([
    "propre / uniforme", "attaques séparées / uniforme", "ALIE / uniforme"
])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

stability_frames = []
for phase_name in ("development", "holdout"):
    frame = g0d_csv[f"{phase_name}_stability"].copy()
    frame["phase"] = phase_name
    stability_frames.append(frame)
g0d_stability = pd.concat(stability_frames, ignore_index=True)

def finite_max(series):
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.max()) if len(numeric) else np.nan

stability_table = (
    g0d_stability.groupby(["phase", "candidate"], as_index=False)
    .agg(
        observed_delta_max=("observed_delta", "max"),
        theoretical_bound_max=("theoretical_bound", finite_max),
        empirical_violations=(
            "violation",
            lambda values: int(
                sum(bool(value) for value in values if pd.notna(value))
            ),
        ),
        certificate=("certificate", lambda values: "; ".join(sorted(set(values)))),
        device=("resolved_device", lambda values: ", ".join(sorted(set(values)))),
    )
)
display(stability_table)

locked_holdout = g0d_csv["holdout_summary"].iloc[0]
tail_decomposition = pd.DataFrame([{
    "queue honnête régulière": locked_holdout["regular_honest_tail_rate"],
    "covariance limitée ET queue": locked_holdout["covariance_limited_and_tail_fraction"],
    "note": "dénominateurs voisins mais non identiques; ne pas soustraire comme une décomposition exacte",
}])
display(Markdown(
    "#### La covariance intervient, mais n'explique pas la queue excessive"
))
display(tail_decomposition)

fig, ax = plt.subplots(figsize=(12, 5))
holdout_stability_plot = stability_table[
    stability_table.phase == "holdout"
].sort_values("observed_delta_max")
ax.bar(
    holdout_stability_plot["candidate"],
    holdout_stability_plot["observed_delta_max"],
    label="maximum observé",
)
finite_bounds = holdout_stability_plot[
    holdout_stability_plot["theoretical_bound_max"].notna()
]
ax.scatter(
    finite_bounds["candidate"], finite_bounds["theoretical_bound_max"],
    color="black", marker="_", s=320, linewidths=2.5,
    label="borne théorique disponible",
)
ax.set_ylabel(r"Variation replace-one $\ell_2$")
ax.set_title("G0d-F — stabilité holdout observée et certificats disponibles")
ax.tick_params(axis="x", rotation=55)
ax.legend()
plt.tight_layout()
plt.show()
""",
            tag,
        ),
        markdown(
            r"""
### 13.5 Conclusion enregistrée

Le candidat verrouillé est `g0d_q80_g022_r025`. Il ne passe ni tous les gates
de développement ni ceux du holdout ; la décision sauvegardée est donc
`promote=false`. La calibration nulle, l'invariance identity/reverse et les
audits de stabilité restent informatifs, mais ils ne suffisent pas à transformer
ce résultat en validation de la référence Gaussian-aware.
""",
            tag,
        ),
    ]


def backup_notebook(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("/tmp") / f"{path.name}.{stamp}.bak"
    shutil.copy2(path, destination)
    return destination


def update(notebook_path: Path) -> tuple[Path, int]:
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = remove_generated_cells(list(notebook.cells))
    update_attack_text(cells)

    original_cell_count = len(cells)
    if original_cell_count < 30:
        raise RuntimeError(
            f"Expected at least the 30 original cells, found {original_cell_count}"
        )
    cells.extend(positioning_cells())
    cells.extend(g0d_cells())
    notebook.cells = cells

    backup = backup_notebook(notebook_path)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    return backup, len(notebook.cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    backup, cell_count = update(args.notebook.resolve())
    print(f"Backup: {backup}")
    print(f"Notebook updated without execution: {args.notebook.resolve()}")
    print(f"Cells: {cell_count}")


if __name__ == "__main__":
    main()
