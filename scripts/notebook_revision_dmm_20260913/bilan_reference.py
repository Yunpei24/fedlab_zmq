# Bilan figé de la campagne RCIG N10 terminée : 120 comparaisons + 12 calibrations.
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

RCIG_BILAN_ROOT = ROOT / "output/analysis/notebook_dmm_revision_20260913"
RCIG_BILAN_ROOT.mkdir(parents=True, exist_ok=True)
evidence = json.loads((ROOT / "output/analysis/rcig_n10_policy_ablation_final/Evidence.json").read_text())
rcig_bilan_rows = []
for path in sorted((RESULTS_ROOT / "rcig_n10_policy_ablation_v1/comparison").rglob("metrics.json")):
    rel = str(path.relative_to(ROOT))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["source_sha256"][rel]
    data = json.loads(path.read_text()); cfg = data["config"]; rows = data["rounds"]
    status = json.loads(path.parent.parent.joinpath("orchestration_status.json").read_text())
    assert status["status"] == "completed" and len(rows) == 40
    task = status["task"]
    rec = {k: task[k] for k in ["arm", "scenario", "noise", "seed"]}
    rec["path"] = path
    rec["acc"] = 100 * rows[-1]["test_accuracy"]
    rec["worst20"] = rows[-1]["worst20_accuracy_pct"]
    rec["gap"] = rows[-1]["best20_worst20_gap_pct"]
    rcig_bilan_rows.append(rec)
rcig_bilan = pd.DataFrame(rcig_bilan_rows)
assert len(rcig_bilan) == 120
# Les résumés de fenêtres sont lus dans l’analyse vérifiée, pas reconstruits par
# sélection de tours favorable ; chaque observation statistique reste une seed.
record_map = {(r["noise"], r["seed"], r["scenario"], r["arm"]): r
              for r in evidence["records"] if r["phase"] == "comparison"}
for idx, row in rcig_bilan.iterrows():
    r = record_map[(row.noise, row.seed, row.scenario, row.arm)]
    window = "ready_all" if row.scenario == "none" else "attacked_all"
    # Le bras FAR + RFA n'instrumente pas cette erreur : manquant, pas zéro.
    ref_error = r["windows"][window].get("error_reference")
    assert ref_error is not None or row.arm == "rfa"
    rcig_bilan.loc[idx, "reference_error"] = ref_error
    rcig_bilan.loc[idx, "byzantine_mass"] = r["windows"][window]["byzantine_mass"]
display(Markdown(
    "## 8.2 Bilan scientifique à présenter : référence, agrégat, modèle\n\n"
    "**Périmètre :** Fashion-MNIST, 10 clients, 40 tours, α=0,1, trois seeds, "
    "deux bruits, sans attaque puis BF/IPM/ALIE. 132 runs terminés : 12 calibrations "
    "et 120 comparaisons. Les résultats ci-dessous ne sont pas ceux de la nouvelle "
    "campagne uniforme/RFA directe.\n\n"
    "### Résultat 1 — Une référence meilleure ne suffit pas"
))
baseline = rcig_bilan[rcig_bilan.arm == "recent"]
mean_ref = rcig_bilan[rcig_bilan.arm == "midpoint"].merge(
    baseline, on=["noise", "seed", "scenario"], suffixes=("_m", "_r"), validate="one_to_one")
mean_ref["Réduction erreur référence (%)"] = 100 * (1 - mean_ref.reference_error_m / mean_ref.reference_error_r)
mean_ref["Δ accuracy (pp)"] = mean_ref.acc_m - mean_ref.acc_r
mean_ref["Δ Worst-20 (pp)"] = mean_ref.worst20_m - mean_ref.worst20_r
mean_ref["Δ gap (pp)"] = mean_ref.gap_m - mean_ref.gap_r
labels = {"none": "Sans attaque", "bf_persistent": "BF ×10", "ipm_persistent": "IPM", "alie_persistent": "ALIE"}
rows_display = []
for (noise, scenario), group in mean_ref.groupby(["noise", "scenario"]):
    r = {"Bruit": "Homogène" if noise == "homogeneous" else "Hétéroscédastique", "Scénario": labels[scenario]}
    for field in ["Réduction erreur référence (%)", "Δ accuracy (pp)", "Δ Worst-20 (pp)", "Δ gap (pp)"]:
        r[field] = f"{group[field].mean():.3f} ± {group[field].std(ddof=1):.3f}"
    rows_display.append(r)
display(pd.DataFrame(rows_display))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for noise, color in [("homogeneous", "#247ba0"), ("heteroscedastic", "#d47721")]:
    part = mean_ref[mean_ref.noise == noise]
    label = "Homogène" if noise == "homogeneous" else "Hétéroscédastique"
    # Une barre par scénario ; moyenne ± écart-type entre les trois seeds.
    order = list(labels)
    x = np.arange(4) + (-.13 if noise == "homogeneous" else .13)
    for ax, field in zip(axes, ["Réduction erreur référence (%)", "Δ accuracy (pp)"]):
        stat = part.groupby("scenario")[field].agg(["mean", "std"]).reindex(order)
        ax.errorbar(x, stat["mean"], yerr=stat["std"], fmt="o", capsize=4, label=label, color=color)
        ax.set_xticks(range(4), [labels[s] for s in order]); ax.set_ylabel(field)
        ax.axhline(0, color="black", lw=.8)
axes[0].set_title("Moyenne temporelle vs vue récente : référence")
axes[1].set_title("Même comparaison : accuracy finale")
axes[0].legend(); fig.tight_layout()
fig.savefig(RCIG_BILAN_ROOT / "reference_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig)
display(Markdown(
    "**Lecture :** l’erreur quadratique de référence baisse d’environ 43–50 %, "
    "mais les gains finaux restent minuscules. On ne doit pas déclarer un algorithme "
    "meilleur parce que seule sa référence est plus proche du centre honnête. "
    "La baisse de l’erreur de référence est favorable ; pour le modèle, Δ accuracy et "
    "Δ Worst-20 positifs, Δ gap négatif seraient favorables. ± désigne l’écart-type, pas un IC95."
))
mean_ref.drop(columns=[c for c in mean_ref if c.startswith("path")]).to_csv(RCIG_BILAN_ROOT / "midpoint_vs_recent_paired.csv", index=False)
