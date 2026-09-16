# Relecture des 18 runs réellement utilisés dans les courbes de confirmation F.
# Aucune nouvelle expérience, aucune modification des métriques sauvegardées.
import json
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import Markdown, display

AUDIT_ROOT = ROOT / "output/analysis/notebook_dmm_revision_20260913"
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
f_audit_rows = []
for path in sorted((RESULTS_ROOT / "positioning_v3/f_confirmation_t20").rglob("metrics.json")):
    if "__none__" not in str(path):
        continue
    data = json.loads(path.read_text())
    cfg, summary, rows = data["config"], data["summary"], data["rounds"]
    if cfg.get("robust_reference") != "centered_clipping":
        continue
    assert len(rows) == summary["num_rounds"] == 20
    assert cfg["fixed_steps_per_round"] == 1
    assert cfg["far_update_mode"] == "single_step_gradient"
    assert cfg["sampling_scheme"] == "fixed_without_replacement"
    local_rates = [r["privacy_clip_rate_mean"] for r in rows
                   if r.get("privacy_clip_rate_mean") is not None]
    last = rows[-1]
    f_audit_rows.append({
        "Clients": summary["num_clients"], "DP": bool(cfg["enable_dp"]),
        "α": cfg["far_alpha"], "seed": summary["seed"],
        "C": cfg["clip_norm"], "U": cfg["far_server_clip_norm"],
        "B": cfg["fixed_batch_size"], "N local": cfg["privacy_public_dataset_size"],
        "local_epochs (champ)": cfg["local_epochs"],
        "gradients/batchs par tour": cfg["fixed_steps_per_round"],
        "Pas serveur": cfg["far_server_lr"],
        "Clip serveur max (%)": 100 * max(r["far_server_clip_rate"] for r in rows),
        "Clip local moyen (%)": 100 * np.mean(local_rates) if local_rates else np.nan,
        "Accuracy (%)": 100 * last["test_accuracy"],
        "Concentration finale": last["far_noise_amplification_vs_uniform"],
        "Loss finale": last["test_loss"], "source": str(path.relative_to(ROOT)),
    })
f_protocol_runs = pd.DataFrame(f_audit_rows)
assert len(f_protocol_runs) == 18 and set(f_protocol_runs.seed) == {28, 36, 54}
protocol_display = []
for (n, dp, alpha), group in f_protocol_runs.groupby(["Clients", "DP", "α"]):
    assert len(group) == 3
    for name in ["C", "U", "B", "N local", "local_epochs (champ)", "gradients/batchs par tour", "Pas serveur"]:
        assert group[name].nunique() == 1
    protocol_display.append({
        "Clients": n, "Condition": f"{'DP' if dp else 'Sans DP'} · α={alpha:g}",
        "C local": group.C.iloc[0], "U serveur": group.U.iloc[0],
        "Batch / taille locale": f"{group.B.iloc[0]:.0f} / {group['N local'].iloc[0]:.0f}",
        "Clip serveur max (20 tours)": f"{group['Clip serveur max (%)'].max():.1f} %",
        "Clip local moyen (20 tours)": "non publié sous DP" if dp else f"{group['Clip local moyen (%)'].mean():.2f} %",
        "Accuracy finale (%)": f"{group['Accuracy (%)'].mean():.2f} ± {group['Accuracy (%)'].std(ddof=1):.2f}",
        "nΣλ² final": round(group['Concentration finale'].mean(), 3),
    })
display(Markdown("### Audit des courbes : qu’est-ce qui est réellement comparé ?"))
display(f_protocol_runs[["Clients", "C", "U", "B", "N local", "gradients/batchs par tour"]]
        .drop_duplicates().sort_values("Clients").reset_index(drop=True)
        .rename(columns={"C": "C local", "U": "U serveur", "B": "Batch"}))
display(Markdown("**Mesures : moyenne ± écart-type entre trois seeds.** Le taux local est moyenné sur 20 tours ; le taux serveur ci-dessous est le maximum."))
display(pd.DataFrame(protocol_display)[[
    "Clients", "Condition", "Clip serveur max (20 tours)", "Clip local moyen (20 tours)",
    "Accuracy finale (%)", "nΣλ² final",
]])
display(Markdown(
    "**Clipping serveur :** U=16 est configuré dans les deux branches, mais il ne tronque "
    "aucun upload dans ces 18 runs : taux maximal **0 % sur les 20 tours**. Il ne peut donc "
    "pas expliquer directement la baisse sans DP. Le clipping **local par exemple**, lui, "
    "reste actif ; il touche en moyenne 0,91 % des exemples à n=10 et 51,09 % à n=25 "
    "dans les bras sans DP. Son effet causal sur l’accuracy n’est pas identifié sans témoin "
    "sans clipping local. Une valeur absente sous DP ne signifie jamais zéro clipping.\n\n"
    "**Entraînement local :** le champ historique `local_epochs=1` ne lance pas une époque. "
    "Chaque client calcule **un gradient moyen sur un batch**, l’envoie, et ne fait aucune "
    "mise à jour locale de paramètres. Le serveur fait un pas de taille 0,2. "
    "B/N=0,05 : 20 tours traitent N exemples en comptant les répétitions, mais ne "
    "garantissent pas de visiter chaque exemple une fois. Le tirage est sans remise "
    "**dans chaque batch**, pas sur l’ensemble des tours.\n\n"
    "**Pourquoi DP/α=2 finit plus haut ?** Les poids finaux sont observés beaucoup moins "
    "concentrés avec DP (nΣλ²≈1,03–1,06) que sans DP (≈1,97–2,61). C’est une explication "
    "compatible avec les résultats, pas une démonstration que le bruit améliore l’apprentissage. "
    "La phase F manque de sans-DP/α=0 et ne fournit pas de traces certifiant un appariement "
    "strict des tirages. Cette comparaison n’est pas une reproduction du protocole FAR natif.\n\n"
    "**Held-out** signifie données réservées à l’évaluation, non utilisées pour calculer "
    "les gradients de ces runs. Les anciennes cellules exécutées [7] et [8] affichent respectivement "
    "l’accuracy et la loss globales ; ce ne sont pas des métriques d’entraînement."
))
f_protocol_runs.to_csv(AUDIT_ROOT / "F_protocol_per_seed.csv", index=False)
pd.DataFrame(protocol_display).to_csv(AUDIT_ROOT / "F_protocol_readable.csv", index=False)
