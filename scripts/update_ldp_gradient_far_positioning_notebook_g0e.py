#!/usr/bin/env python3
"""Append the G0e confirmatory audit to the positioning notebook.

This updater is intentionally narrow and idempotent.  It only owns cells
tagged ``ldp-gaussian-aware-g0e-2026-09`` and preserves every other notebook
cell.  The generated cells read published G0e artifacts; they do not rerun the
experiment, recompute a scientific gate, or open a blocked holdout.
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
SECTION_TAG = "ldp-gaussian-aware-g0e-2026-09"


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


def g0e_cells() -> list[nbformat.NotebookNode]:
    return [
        markdown(
            r"""
## 14. G0e — correction Gaussian-aware bornée autour de FCC

G0e est une expérience **référence-seulement** : elle n'utilise ni poids FAR,
ni accuracy, ni entraînement du réseau. Elle teste un unique candidat figé,
sans grille post-hoc : un pilote FCC online, puis une correction Huber par
blocs, Gaussian-aware, à influence bornée. Les références FCC leave-one-out
servent uniquement à la calibration offline et aux diagnostics.

$$
F_{\mathrm{G0e}}(X)
=(1-\beta)F_{\mathrm{CC}}(X)+\beta p_K(X).
$$

Le protocole est séquentiel : le holdout ne peut être créé ou lu que si
**tous** les gates de développement passent. Les cellules ci-dessous rendent
donc l'absence attendue du holdout explicite, sans la traiter comme une erreur.
"""
        ),
        code(
            r"""
G0E_ROOT = (
    ROOT / "results" / "ldp_gradient_far"
    / "gaussian_aware_reference_g0e_mps_v1"
)
G0E_AVAILABLE = G0E_ROOT.exists() and (G0E_ROOT / "decision.json").exists()

g0e_json = {}
g0e_csv = {}
g0e_config = {}
g0e_holdout_csv = {}

if not G0E_AVAILABLE:
    display(Markdown(
        "**G0e n'est pas encore publié dans le dossier de résultats.** "
        "La section reste exécutable et n'invente aucune valeur."
    ))
else:
    required_json = {
        "decision", "derived_parameters", "calibration_artifact",
        "run_manifest",
    }
    required_csv = {
        "calibration_probes", "development_summary",
        "development_comparators", "development_diagnostic_groups",
        "development_regular_tier_block_diagnostics",
        "development_stability",
    }
    g0e_json = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(G0E_ROOT.glob("*.json"))
    }
    g0e_csv = {
        path.stem: pd.read_csv(path)
        for path in sorted(G0E_ROOT.glob("*.csv"))
        if not path.stem.startswith("holdout")
    }
    missing_json = required_json - set(g0e_json)
    missing_csv = required_csv - set(g0e_csv)
    config_path = G0E_ROOT / "resolved_config.yaml"
    if missing_json or missing_csv or not config_path.exists():
        raise FileNotFoundError(
            f"Artefacts G0e manquants — JSON={sorted(missing_json)}, "
            f"CSV={sorted(missing_csv)}, config={config_path.exists()}"
        )
    g0e_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    decision = g0e_json["decision"]
    holdout_paths = sorted(G0E_ROOT.rglob("*holdout*"))
    holdout_files = [path for path in holdout_paths if path.is_file()]
    if decision["holdout_status"] == "blocked_by_development_gate":
        if holdout_files:
            raise AssertionError(
                "Le protocole déclare le holdout bloqué, mais des artefacts "
                f"holdout existent : {[str(p) for p in holdout_files]}"
            )
    elif decision["holdout_status"] == "executed_after_development_pass":
        g0e_holdout_csv = {
            path.stem: pd.read_csv(path)
            for path in sorted(G0E_ROOT.glob("holdout_*.csv"))
        }
        required_holdout = {
            "holdout_summary", "holdout_comparators",
            "holdout_diagnostic_groups", "holdout_stability",
        }
        missing_holdout = required_holdout - set(g0e_holdout_csv)
        if missing_holdout:
            raise FileNotFoundError(
                f"Holdout déclaré exécuté mais incomplet : {sorted(missing_holdout)}"
            )
    else:
        raise ValueError(
            f"Statut holdout G0e inconnu : {decision['holdout_status']}"
        )

    display(pd.DataFrame([g0e_json["run_manifest"]]))
    display(Markdown(
        f"**Décision enregistrée :** `promote={decision['promote']}`; "
        f"développement=`{decision['development_passes']}`; "
        f"holdout=`{decision['holdout_status']}`."
    ))
"""
        ),
        markdown(
            r"""
### 14.1 Construction figée et calibration indépendante

Les paramètres ne sont pas choisis après observation de l'erreur. Ils sont
dérivés de budgets publics. Pour $q=(1+2\gamma)^{-1}$, le mélange finite-$K$
est

$$
\beta=\min\!\left\{1,
\frac{\gamma B_{\mathrm{corr}}}{G(1-q^K)}\right\},
$$

et le certificat replace-one utilisé par le gate est

$$
\Delta_2(F_{\mathrm{G0e}})
\leq \Delta_{\mathrm{FCC}}
+\beta\frac{2G(1-q^K)}{\gamma n}.
$$

La calibration split-conformal emploie **un probe public par cohorte
indépendante et par strate**. Le rayon statistique sert à mesurer la queue ;
le rayon déployé ajoute la marge de transport FCC leave-one-out vers FCC
online. Ainsi, « statistical-tail », « cap-active » et « outlier cap-limited »
restent trois événements distincts.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    derived = g0e_json["derived_parameters"]
    calibration = g0e_json["calibration_artifact"]
    derived_fields = [
        "regularization", "influence_cap_total", "num_steps",
        "beta_finite_solver", "beta_asymptotic_control",
        "correction_budget", "finite_correction_norm_bound",
        "pilot_replace_one_bound", "finite_solver_replace_one_bound",
        "replace_one_bound_budget", "solver_gradient_residual_bound",
        "fcc_replacement_contamination_bound",
        "raw_correction_replacement_contamination_bound",
        "blended_correction_replacement_contamination_bound",
        "total_reference_replacement_contamination_bound",
    ]
    display(pd.DataFrame([
        {field: derived.get(field) for field in derived_fields}
    ]).T.rename(columns={0: "valeur"}))

    calibration_summary = pd.DataFrame({
        "bloc": np.arange(len(calibration["standardized_thresholds"])),
        "seuil standardisé public": calibration["standardized_thresholds"],
        "taux de queue calibration": calibration["calibration_tail_rate_by_block"],
    })
    display(calibration_summary)
    display(pd.DataFrame(
        sorted(calibration["calibration_tail_rate_by_context"].items()),
        columns=["contexte de calibration", "taux de queue"],
    ))
"""
        ),
        markdown(
            r"""
### 14.2 Trois gates de qualité de référence, puis registre complet

Les trois comparaisons centrales demandent que G0e : (1) ne dégrade pas la
moyenne uniforme en situation propre ; (2) ne dégrade pas FCC de plus de 5 %
sous attaques séparées ; (3) ne dégrade pas FCC de plus de 5 % sous ALIE.
Ce ne sont pas les seuls certificats : le second tableau affiche **tous** les
booléens préenregistrés, sans en changer le seuil.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    dev = g0e_csv["development_summary"].iloc[0]
    core_gates = pd.DataFrame([
        {
            "gate": "Propre / uniforme",
            "valeur": dev["clean_error_ratio_to_uniform_worst_group"],
            "seuil": "≤ 1.00",
            "passe": bool(dev["gate_clean_vs_uniform"]),
        },
        {
            "gate": "Attaques séparées / FCC",
            "valeur": dev["attacked_error_ratio_to_fcc_worst_group"],
            "seuil": "≤ 1.05",
            "passe": bool(dev["gate_attacked_vs_fcc"]),
        },
        {
            "gate": "ALIE / FCC",
            "valeur": dev["evasive_error_ratio_to_fcc_worst_group"],
            "seuil": "≤ 1.05",
            "passe": bool(dev["gate_evasive_vs_fcc"]),
        },
    ])
    display(core_gates)

    gate_columns = sorted(
        column
        for column in dev.index
        if column.startswith("gate_") and column != "gate_fail_count"
    )
    all_gates = pd.DataFrame({
        "gate enregistré": gate_columns,
        "passe": [bool(dev[column]) for column in gate_columns],
    })
    display(Markdown(
        f"**Bilan développement : {int(dev['gate_fail_count'])} échec(s); "
        f"passes_all_gates={bool(dev['passes_all_gates'])}.**"
    ))
    display(all_gates)
"""
        ),
        markdown(
            r"""
### 14.3 Comparaison appariée : G0e, FCC, uniforme, RFA et trMean

Toutes les références ci-dessous voient les mêmes cohortes, seeds, bruits et
attaques. Les ratios de pire groupe sont les statistiques des gates ; les
moyennes d'erreur donnent une lecture descriptive complémentaire. Une faible
erreur empirique ne constitue pas à elle seule un certificat de stabilité.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    comparators = g0e_csv["development_comparators"].copy()
    labels = {
        "g0e": "G0e", "fcc": "FCC", "uniform_mean": "Uniforme",
        "rfa": "RFA", "trimmed_mean": "trMean",
    }
    comparators["référence"] = comparators["candidate"].map(labels).fillna(
        comparators["candidate"]
    )
    comparator_columns = [
        "référence", "observations", "clean_reference_error_mean",
        "clean_error_ratio_to_uniform_worst_group",
        "attacked_reference_error_mean",
        "attacked_error_ratio_to_fcc_worst_group",
        "evasive_reference_error_mean",
        "evasive_error_ratio_to_fcc_worst_group",
        "replace_one_observed_max", "replace_one_theoretical_bound",
        "certificate",
    ]
    display(comparators[comparator_columns].sort_values("référence"))

    plot_columns = [
        "clean_error_ratio_to_uniform_worst_group",
        "attacked_error_ratio_to_fcc_worst_group",
        "evasive_error_ratio_to_fcc_worst_group",
    ]
    plot_data = comparators.set_index("référence")[plot_columns]
    ax = plot_data.plot(kind="bar", figsize=(12, 5))
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Ratio d'erreur de référence (pire groupe)")
    ax.set_title("G0e — comparateurs strictement appariés en développement")
    ax.legend(["propre / uniforme", "attaques / FCC", "ALIE / FCC"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
"""
        ),
        markdown(
            r"""
### 14.4 Rayon Gaussian-aware avant cap : branche déployée dégénérée

Le rayon Gaussian-aware déployé du client $i$ et du bloc $b$ est reconstruit
uniquement à partir des artefacts publics :

$$
R^{\mathrm{deploy}}_{i,b}
=z_b\sqrt{v^{\mathrm{DP}}_{i,b}+h_b^2+v^{\mathrm{ref}}_{i,b}+v_0}
+m_b.
$$

L'algorithme utilise ensuite le rayon effectif

$$
R^{\mathrm{eff}}_{i,b}=\min\{R^{\mathrm{deploy}}_{i,b},G_b\}.
$$

La table suivante vérifie, par régime, permutation, niveau de bruit et bloc,
si le rayon Gaussian-aware **avant cap** dépasse le cap public
$G_b=0{,}065$. Une fraction `cap-limited` égale à 1 signifie que tous les
clients de cette cellule utilisent finalement $R^{\mathrm{eff}}_{i,b}=G_b$.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    calibration = g0e_json["calibration_artifact"]
    derived = g0e_json["derived_parameters"]
    n_clients = int(g0e_config["cohort"]["num_clients"])
    base_std = float(g0e_config["privacy_noise"]["base_std"])
    block_multipliers = np.asarray(
        g0e_config["privacy_noise"]["block_std_multipliers"], dtype=float
    )
    heterogeneity_variances = np.square(np.asarray(
        g0e_config["cohort"]["heterogeneity_std_by_block"], dtype=float
    ))
    thresholds = np.asarray(calibration["standardized_thresholds"], dtype=float)
    margin = float(calibration["crossfit_to_full_fcc_margin_per_block"])
    variance_floor = float(
        g0e_config["references"]["g0e_public_budgets"]["variance_floor"]
    )
    influence_caps = np.asarray(derived["influence_cap_per_block"], dtype=float)

    radius_rows = []
    for regime in g0e_config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        public_tiers = np.asarray([
            float(regime["client_std_multipliers"][
                client % len(regime["client_std_multipliers"])
            ])
            for client in range(n_clients)
        ])
        for permutation in regime["permutations"]:
            permutation = str(permutation)
            if permutation == "identity":
                tiers = public_tiers.copy()
            elif permutation == "reverse":
                tiers = public_tiers[::-1].copy()
            elif permutation.startswith("rotate"):
                tiers = np.roll(
                    public_tiers, int(permutation.removeprefix("rotate"))
                )
            else:
                raise ValueError(f"Permutation de tiers inconnue : {permutation}")

            noise_variances = np.square(
                base_std * tiers[:, None] * block_multipliers[None, :]
            )
            reference_variances = np.asarray(
                calibration["reference_variance_by_cell"][
                    f"{regime_name}|{permutation}"
                ],
                dtype=float,
            )
            statistical_radii = thresholds[None, :] * np.sqrt(
                noise_variances
                + heterogeneity_variances[None, :]
                + reference_variances
                + variance_floor
            )
            deployed_radii = statistical_radii + margin

            for tier in sorted(set(tiers.tolist())):
                client_mask = np.isclose(tiers, tier)
                for block in range(len(thresholds)):
                    cell = deployed_radii[client_mask, block]
                    cap = float(influence_caps[block])
                    radius_rows.append({
                        "régime": regime_name,
                        "permutation": permutation,
                        "tier de bruit": float(tier),
                        "bloc": block,
                        "clients": int(client_mask.sum()),
                        "rayon avant cap min": float(cell.min()),
                        "rayon avant cap max": float(cell.max()),
                        "cap G_b": cap,
                        "fraction cap-limited": float(np.mean(cell > cap)),
                        "rayon effectif max": float(np.minimum(cell, cap).max()),
                    })

    g0e_radius_cap_table = pd.DataFrame(radius_rows)
    display(g0e_radius_cap_table)
    global_cap_limited_fraction = float(
        np.average(
            g0e_radius_cap_table["fraction cap-limited"],
            weights=g0e_radius_cap_table["clients"],
        )
    )
    if not np.isclose(global_cap_limited_fraction, 1.0):
        raise AssertionError(
            "Le diagnostic enregistré attend une géométrie entièrement "
            f"cap-limited, observé={global_cap_limited_fraction:.6f}."
        )
    display(Markdown(
        "**Résultat : fraction cap-limited globale = "
        f"{global_cap_limited_fraction:.3f}.** Le plus petit rayon déployé "
        f"({g0e_radius_cap_table['rayon avant cap min'].min():.4f}) dépasse "
        f"déjà le cap ({influence_caps.min():.3f}). La branche online utilise "
        "donc le cap dans chaque bloc : les différences de rayons liées au "
        "niveau de bruit n'affectent plus la correction Huber. La calibration "
        "Gaussian-aware reste visible dans le diagnostic de queue, mais elle "
        "est neutralisée dans l'estimateur déployé."
    ))
"""
        ),
        markdown(
            r"""
### 14.5 Queue statistique, cap d'influence et honest outliers

Pour éviter qu'une moyenne masque une cellule défavorable, on affiche le
groupe exact qui réalise chaque pire cas :

- **queue statistique** : résidu régulier au-delà du rayon statistique ;
- **cap actif** : correction régulière limitée par le cap d'influence ;
- **outlier retenu** : honest outlier dont aucun bloc n'est limité ;
- **masse Byzantine** : part de l'influence de correction portée par les
  Byzantins dans une attaque séparée.

Ces diagnostics répondent à des questions différentes. En particulier, un
honest outlier peut appartenir à la queue statistique sans être écrasé par le
cap déployé.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    diagnostic_groups = g0e_csv["development_diagnostic_groups"].copy()
    tier_groups = g0e_csv[
        "development_regular_tier_block_diagnostics"
    ].copy()
    separated = set(g0e_config["threats"]["separated_for_gates"])

    def diagnostic_row(frame, field, mode, label):
        values = pd.to_numeric(frame[field], errors="raise")
        index = values.idxmax() if mode == "max" else values.idxmin()
        row = frame.loc[index]
        identity = [
            "noise_regime", "noise_permutation", "outlier_geometry",
            "threat", "severity", "block",
        ]
        if "noise_tier" in frame:
            identity.insert(2, "noise_tier")
        return {
            "diagnostic": label,
            "valeur pire groupe": float(row[field]),
            **{column: row[column] for column in identity},
            "observations": int(row["observations"]),
            "seeds": int(row["seed_clusters"]),
        }

    clean_tiers = tier_groups[tier_groups["threat"].eq("none")]
    clean_groups = diagnostic_groups[diagnostic_groups["threat"].eq("none")]
    attacked_groups = diagnostic_groups[
        diagnostic_groups["threat"].isin(separated)
    ]
    worst_diagnostics = pd.DataFrame([
        diagnostic_row(
            clean_tiers, "regular_honest_statistical_tail_rate", "max",
            "Queue statistique régulière (max)",
        ),
        diagnostic_row(
            clean_tiers, "regular_honest_influence_cap_activation_rate", "max",
            "Cap régulier actif (max)",
        ),
        diagnostic_row(
            clean_groups,
            "honest_outlier_all_blocks_not_cap_limited_rate", "min",
            "Outliers sans aucun bloc capé (min)",
        ),
        diagnostic_row(
            attacked_groups, "byzantine_influence_share", "max",
            "Masse d'influence Byzantine (max)",
        ),
    ])
    display(worst_diagnostics)

    thresholds = {
        "Queue statistique régulière (max)": (
            g0e_json["derived_parameters"]["false_tail_probability"]
            + g0e_config["gates"][
                "regular_honest_statistical_tail_rate_upper_excess_max"
            ]
        ),
        "Cap régulier actif (max)": g0e_config["gates"][
            "regular_honest_influence_cap_activation_rate_max"
        ],
        "Outliers sans aucun bloc capé (min)": g0e_config["gates"][
            "honest_outlier_not_cap_limited_rate_min"
        ],
        "Masse d'influence Byzantine (max)": g0e_config["gates"][
            "separated_byzantine_influence_share_max"
        ],
    }
    plot_diag = worst_diagnostics.set_index("diagnostic")["valeur pire groupe"]
    ax = plot_diag.plot(kind="bar", figsize=(12, 5), color="tab:blue")
    for position, label in enumerate(plot_diag.index):
        ax.scatter(position, thresholds[label], color="black", marker="_", s=280)
    ax.set_ylabel("Taux ou part")
    ax.set_title("G0e — pire groupe; tiret noir = seuil préenregistré")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.show()
"""
        ),
        markdown(
            r"""
### 14.6 Stabilité replace-one : observations et bornes

Le tableau compare le maximum empirique au certificat disponible. Pour RFA,
l'absence de borne dimension-free est conservée comme `N/A` : elle ne doit pas
être remplacée par le maximum observé. Le test empirique vérifie la cohérence
de l'implémentation sur les paires tirées ; il ne remplace pas la preuve
universelle.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    stability = g0e_csv["development_stability"].copy()
    stability["theoretical_numeric"] = pd.to_numeric(
        stability["theoretical_bound"], errors="coerce"
    )
    stability["observed_numeric"] = pd.to_numeric(
        stability["observed_delta"], errors="raise"
    )

    def finite_max(values):
        values = pd.to_numeric(values, errors="coerce")
        values = values[np.isfinite(values)]
        return float(values.max()) if len(values) else np.nan

    stability_summary = (
        stability.groupby("candidate", as_index=False)
        .agg(
            paires=("observed_numeric", "size"),
            maximum_observe=("observed_numeric", "max"),
            borne_theorique=("theoretical_numeric", finite_max),
            violations=(
                "violation",
                lambda values: int(sum(
                    bool(value) for value in values if pd.notna(value)
                )),
            ),
            certificat=(
                "certificate", lambda values: "; ".join(sorted(set(values)))
            ),
        )
    )
    stability_summary["référence"] = stability_summary["candidate"].map({
        "g0e": "G0e", "fcc": "FCC", "uniform_mean": "Uniforme",
        "rfa": "RFA", "trimmed_mean": "trMean",
    }).fillna(stability_summary["candidate"])
    display(stability_summary[[
        "référence", "paires", "maximum_observe", "borne_theorique",
        "violations", "certificat",
    ]])

    ordered = stability_summary.sort_values("maximum_observe")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ordered["référence"], ordered["maximum_observe"], label="observé")
    bounded = ordered[ordered["borne_theorique"].notna()]
    ax.scatter(
        bounded["référence"], bounded["borne_theorique"],
        color="black", marker="_", s=320, linewidths=2.5,
        label="borne théorique",
    )
    ax.set_ylabel(r"Variation replace-one $\ell_2$")
    ax.set_title("G0e — stabilité observée versus certificats")
    ax.legend()
    plt.tight_layout()
    plt.show()
"""
        ),
        markdown(
            r"""
### 14.7 Décision et statut du holdout

Cette cellule est la clôture anti-fuite du protocole. Un échec en développement
implique exactement : **holdout non ouvert, non généré et non utilisé**. Cela
ne signifie pas que le holdout vaut zéro ; cela signifie qu'aucune donnée de
holdout ne peut influencer une nouvelle décision sur G0e.
"""
        ),
        code(
            r"""
if G0E_AVAILABLE:
    decision = g0e_json["decision"]
    if decision["holdout_status"] == "blocked_by_development_gate":
        display(Markdown(
            "## Verdict G0e : arrêt au développement\n\n"
            "**Le holdout n'a été ni ouvert, ni généré, ni lu.** "
            "La décision enregistrée est `promote=false`."
        ))
    else:
        holdout_summary = g0e_holdout_csv["holdout_summary"].iloc[0]
        display(Markdown(
            "## Verdict G0e : holdout exécuté après passage du développement\n\n"
            f"`passes_all_gates={bool(holdout_summary['passes_all_gates'])}`; "
            f"`promote={decision['promote']}`."
        ))
        display(g0e_holdout_csv["holdout_summary"])
"""
        ),
    ]


def backup_notebook(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path("/tmp") / f"{path.name}.{stamp}.g0e.bak"
    shutil.copy2(path, destination)
    return destination


def update(notebook_path: Path) -> tuple[Path, int]:
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = [
        cell
        for cell in notebook.cells
        if SECTION_TAG
        not in set(cell.get("metadata", {}).get("tags", []))
    ]
    if len(cells) < 30:
        raise RuntimeError(
            f"Expected at least 30 pre-existing cells, found {len(cells)}"
        )
    cells.extend(g0e_cells())
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
