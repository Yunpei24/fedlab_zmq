#!/usr/bin/env python3
"""Offline, reproducible analysis of the completed RCIG precursor screen.

No training, selection gate or launch is performed. Input result files remain
read-only. Statistics use four seed blocks, not rounds as independent replicates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rcig_batch_screen import (
    load_campaign, provenance, verify_lock, status_report, file_hash,
    read_json, epsilon_at, calibrated_sigma,
)

OUT = ROOT / "output/analysis/RCIG_Batch_Screen_216_Results.md"
EVIDENCE = ROOT / "output/analysis/RCIG_Batch_Screen_216_Evidence.json"
SEEDS = (24, 42, 72, 121)
BATCHES = (120, 240, 480)
NOISES = ("homogeneous", "heteroscedastic")
SCENARIOS = ("none", "bf_x10_persistent", "ipm_persistent")
METHODS = ("uniform", "rfa", "recent")
LABEL = {"uniform": "Uniforme", "rfa": "FAR + RFA", "recent": "FAR + récent",
         "homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique",
         "none": "Sans attaque", "bf_x10_persistent": "BF ×10", "ipm_persistent": "IPM"}


def contrast_label(name):
    return " − ".join(LABEL.get(part, part) for part in name.split("-"))


METRICS = {
    "test_acc": ("test_accuracy", 100), "client_acc": ("client_accuracy_mean", 100),
    "loss": ("test_loss", 1), "client_loss": ("client_loss_mean", 1),
    "worst20": ("worst20_accuracy_pct", 1), "gap": ("best20_worst20_gap_pct", 1),
    "variance": ("client_accuracy_variance_pct2", 1),
}
ORACLE = "rcig_reference_squared_l2_error_to_clean_honest_center_oracle"


def t_cdf_df3(x):
    u = x / math.sqrt(3)
    return .5 + (math.atan(u) + u / (1 + u * u)) / math.pi


def t95():
    low, high = 0., 10.
    for _ in range(80):
        mid = (low + high) / 2
        if t_cdf_df3(mid) < .975:
            low = mid
        else:
            high = mid
    return (low + high) / 2


T95 = t95()
assert abs(T95 - 3.182446305284263) < 1e-10


def summarize(values):
    values = list(values)
    assert len(values) == 4 and all(math.isfinite(x) for x in values)
    mean, sd = st.mean(values), st.stdev(values)
    half = T95 * sd / 2
    return {"mean": mean, "sd": sd, "ci95_low": mean-half, "ci95_high": mean+half,
            "values": values, "n_seeds": 4, "positive_seeds": sum(v > 0 for v in values)}


def values_at(row):
    return {name: float(row[key]) * scale for name, (key, scale) in METRICS.items()}


def client_stats(row, ids=None):
    source = dict(zip(row["evaluated_client_ids_oracle"], row["client_accuracy_values_oracle"]))
    if ids is not None:
        source = {cid: source[cid] for cid in ids}
    values = sorted(source.values())
    k = math.ceil(.2 * len(values))
    return {"client_acc": 100 * st.mean(values), "worst20": 100 * st.mean(values[:k]),
            "gap": 100 * (st.mean(values[-k:]) - st.mean(values[:k])),
            "variance": 10000 * st.pvariance(values)}


def diagnostics(rows):
    # Summarize within a run first; four resulting seed statistics are compared.
    segment = [r for r in rows if r["round_num"] >= 17]
    mapping = {"clip_pct": ("far_server_clip_rate", 100),
               "n_lambda_max": ("max_client_weight", 25),
               "concentration": ("far_weight_l2_squared", 25),
               "entropy_fraction": ("weight_entropy", 1 / math.log(25)),
               "logit_span": ("far_logit_range", 1),
               "byzantine_mass": ("byzantine_weight_mass_oracle", 1),
               "upload_norm_median": ("received_update_norm_p50", 1),
               "reference_norm": ("far_reference_norm", 1),
               "reference_sq_error": (ORACLE, 1)}
    result = {}
    for name, (key, scale) in mapping.items():
        vals = [r[key] * scale for r in segment if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
        result[name] = st.median(vals) if vals else None
    return result


def build_evidence():
    campaign = load_campaign()
    stamp = provenance(campaign)
    verify_lock(campaign, stamp)
    validation = status_report(campaign, stamp)
    assert validation["valid_complete"] == 216 and not validation["active"] and not validation["invalid"]
    assert read_json(campaign.output_root / "_status.json")["status"] == "completed"
    records, all_rounds, hashes = [], {}, {}
    oracle_counts = defaultdict(int)
    checked_client_rows = 0
    for task in campaign.tasks:
        directory = campaign.output_root / "runs" / task.run_id
        paths = list(directory.glob("**/metrics.json"))
        assert len(paths) == 1
        path = paths[0]
        payload = read_json(path)
        rows = payload["rounds"]
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
        key = (task.batch_size, task.noise_regime, task.scenario, task.reference, task.seed)
        assert key not in all_rounds
        all_rounds[key] = rows
        expected_ids = list(range(25)) if task.scenario == "none" else list(range(5, 25))
        for row in rows:
            assert row["evaluated_client_ids_oracle"] == expected_ids
            derived = client_stats(row)
            for name, value in derived.items():
                assert math.isclose(value, values_at(row)[name], rel_tol=1e-8, abs_tol=1e-7), (task.run_id, row["round_num"], name)
            if task.scenario == "none":
                assert abs(row["test_accuracy"] - row["client_accuracy_mean"]) < 1e-10
            checked_client_rows += 1
            if isinstance(row.get(ORACLE), (int, float)):
                assert task.reference == "recent" and row["round_num"] >= 13
                assert row[ORACLE] == row["rcig_identity_new_squared_l2_error_to_clean_honest_center_oracle"]
                oracle_counts[task.reference] += 1
        records.append({"B": task.batch_size, "noise": task.noise_regime, "scenario": task.scenario,
                        "method": task.reference, "seed": task.seed, "run_id": task.run_id,
                        "path": str(path.relative_to(ROOT)), "final": values_at(rows[-1]),
                        "snapshots": {str(t): values_at(rows[t-1]) for t in (1, 12, 16, 24, 40)},
                        "diagnostics_t17_t40_medians": diagnostics(rows),
                        "epsilon_max_final": rows[-1]["privacy_epsilon_max"],
                        "epsilon_mean_final": rows[-1]["privacy_epsilon_mean"]})
    assert checked_client_rows == 8640 and oracle_counts == {"recent": 2016}
    lookup = {(r["B"], r["noise"], r["scenario"], r["method"], r["seed"]): r for r in records}
    groups = []
    for B, noise, scenario, method in itertools.product(BATCHES, NOISES, SCENARIOS, METHODS):
        rr = [lookup[B, noise, scenario, method, seed] for seed in SEEDS]
        stats = {metric: summarize(r["final"][metric] for r in rr) for metric in METRICS}
        dg = {}
        for metric in rr[0]["diagnostics_t17_t40_medians"]:
            vals = [r["diagnostics_t17_t40_medians"][metric] for r in rr]
            dg[metric] = summarize(vals) if all(v is not None for v in vals) else None
        groups.append({"B": B, "noise": noise, "scenario": scenario, "method": method, "stats": stats, "diagnostics": dg})
    contrasts = []
    for B, noise, scenario in itertools.product(BATCHES, NOISES, SCENARIOS):
        for lhs, rhs in (("recent", "rfa"), ("recent", "uniform"), ("rfa", "uniform")):
            stats = {metric: summarize(lookup[B,noise,scenario,lhs,s]["final"][metric] - lookup[B,noise,scenario,rhs,s]["final"][metric] for s in SEEDS) for metric in METRICS}
            contrasts.append({"B": B, "noise": noise, "scenario": scenario, "contrast": f"{lhs}-{rhs}", "stats": stats})
    batch_contrasts = []
    for noise, scenario, method in itertools.product(NOISES, SCENARIOS, METHODS):
        for lhs, rhs in ((240,120), (480,120), (480,240)):
            stats = {metric: summarize(lookup[lhs,noise,scenario,method,s]["final"][metric] - lookup[rhs,noise,scenario,method,s]["final"][metric] for s in SEEDS) for metric in METRICS}
            batch_contrasts.append({"noise": noise, "scenario": scenario, "method": method, "contrast": f"B{lhs}-B{rhs}", "stats": stats})
    # Secondary analysis: remove the population confound using saved client arrays.
    attack_effects = []
    for B, noise, method, scenario in itertools.product(BATCHES, NOISES, METHODS, SCENARIOS[1:]):
        deltas = defaultdict(list)
        for seed in SEEDS:
            clean = all_rounds[B,noise,"none",method,seed][-1]
            attacked = all_rounds[B,noise,scenario,method,seed][-1]
            clean_metrics, attacked_metrics = client_stats(clean, range(5,25)), client_stats(attacked, range(5,25))
            for metric in clean_metrics:
                deltas[metric].append(attacked_metrics[metric] - clean_metrics[metric])
            deltas["test_acc"].append(100*(attacked["test_accuracy"]-clean["test_accuracy"]))
            deltas["loss"].append(attacked["test_loss"]-clean["test_loss"])
        attack_effects.append({"B": B,"noise": noise,"method": method,"scenario": scenario,
                               "stats": {m:summarize(v) for m,v in deltas.items()}})
    global_descriptive = {}
    for name in ("recent-rfa", "recent-uniform", "rfa-uniform"):
        cells = [c for c in contrasts if c["contrast"] == name]
        global_descriptive[name] = {m: summarize(st.mean(c["stats"][m]["values"][idx] for c in cells) for idx in range(4)) for m in METRICS}
    cells = [c for c in batch_contrasts if c["contrast"] == "B480-B120"]
    global_descriptive["B480-B120"] = {m: summarize(st.mean(c["stats"][m]["values"][idx] for c in cells) for idx in range(4)) for m in METRICS}
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_scientific_hash": read_json(campaign.output_root / "_campaign_lock.json")["scientific_hash"],
            "validation": validation, "input_sha256": hashes,
            "checked_client_rounds": checked_client_rows, "oracle_counts": dict(oracle_counts),
            "t95_df3": T95, "seeds": list(SEEDS), "records": records,
            "groups": groups, "method_contrasts": contrasts, "batch_contrasts": batch_contrasts,
            "attack_effects_common20": attack_effects, "global_descriptive": global_descriptive,
            "privacy": [{"B": B,"q": B/2400,"sigma": calibrated_sigma(B),
                          "noise_sd_base": 4*calibrated_sigma(B)/B,
                          "epsilon_base": epsilon_at(B,40),"epsilon_double_noise": epsilon_at(B,40,2),
                          "epsilon_hetero_mean": (13*epsilon_at(B,40)+12*epsilon_at(B,40,2))/25} for B in BATCHES]}


def fmt(value, digits=2):
    if abs(value) < .5 * 10**(-digits):
        value = 0.
    return f"{value:.{digits}f}".replace(".", ",")


def ms(stat, digits=2):
    return "ND" if stat is None else f"{fmt(stat['mean'],digits)} ± {fmt(stat['sd'],digits)}"


def ci(stat, digits=2):
    return f"{fmt(stat['mean'],digits)} [{fmt(stat['ci95_low'],digits)} ; {fmt(stat['ci95_high'],digits)}]"


def table(headers, rows):
    rows = list(rows)
    return "\n\n" + "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] +
                              ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n\n"


def render_report(e):
    parts = []
    add = parts.append
    add("""# Résultats complets — écran exploratoire RCIG, 216 runs

Analyse du 13 septembre 2026. **216/216 runs complets et valides ; aucun processus parent actif lors du contrôle final.** Aucun entraînement supplémentaire n’est lancé.

## 1. Verdict scientifique

Cet écran ne justifie pas de promouvoir la référence récente comme une amélioration établie de FAR + RFA. Il ne teste pas la correction RCIG. Le résultat le plus instructif concerne l’interaction entre batch, clipping serveur, géométrie des distances et type d’attaque : **un batch plus grand ne garantit pas une meilleure résistance byzantine**.

| Question | Verdict sur cet écran |
| --- | --- |
| Le batch plus grand améliore-t-il l’apprentissage ? | Plusieurs moyennes s’améliorent, surtout sans attaque et sous IPM ; pas de bénéfice uniforme. Sous BF ×10 homogène, FAR se dégrade nettement. |
| La référence récente surpasse-t-elle FAR + RFA ? | Non établi : écarts globaux presque nuls, aucun IC95 de gain Worst-20 strictement positif parmi les 18 conditions. |
| FAR aide-t-il face à l’uniforme ? | Dépend de l’attaque : avantage d’accuracy sous plusieurs conditions IPM, désavantage sous BF ×10 homogène aux grands batchs. |
| L’écran valide-t-il RCIG, sa détection ou sa récupération ? | Non : aucune correction RCIG et aucune phase de récupération ne sont exécutées. |

### 1.1 Résumé équilibré des contrastes

Pour ce résumé descriptif secondaire, on moyenne les 18 conditions à poids égaux **dans chaque seed**, puis on construit l’IC à partir des quatre moyennes par seed. On ne traite ni les 18 conditions ni les 40 tours comme des réplications indépendantes. Cette moyenne de contrastes n’est pas une accuracy de population fusionnée.
""")
    add(table(["Contraste", "Δ Test Acc. (pp), IC95", "Δ Worst-20 (pp), IC95", "Δ gap (pp), IC95"],
              [(contrast_label(name),
                ci(stats["test_acc"],3), ci(stats["worst20"],3), ci(stats["gap"],3)) for name, stats in e["global_descriptive"].items()]))
    add("""**Interprétation.** Un intervalle traversant zéro ne prouve ni égalité ni absence absolue d’effet ; il ne permet pas d’établir son signe avec cette analyse. Les IC sont exploratoires, non corrigés pour les nombreuses comparaisons. La proximité des deux références ne constitue pas un test formel d’équivalence, faute de marge préenregistrée.

### 1.2 Exceptions importantes à ne pas masquer par une moyenne générale
""")
    selected = [(120,"heteroscedastic","ipm_persistent","recent-rfa"),
                (480,"heteroscedastic","ipm_persistent","recent-rfa"),
                (480,"homogeneous","bf_x10_persistent","recent-uniform"),
                (240,"homogeneous","ipm_persistent","recent-uniform")]
    cc = {(c["B"],c["noise"],c["scenario"],c["contrast"]): c for c in e["method_contrasts"]}
    add(table(["B", "Bruit", "Scénario", "Contraste", "Δ Test Acc. (pp), IC95", "Δ Worst-20 (pp), IC95"],
              [(B,LABEL[noise],LABEL[sc],contrast_label(co),ci(cc[B,noise,sc,co]["stats"]["test_acc"],3),ci(cc[B,noise,sc,co]["stats"]["worst20"],3)) for B,noise,sc,co in selected]))
    add("""En particulier, sous IPM hétéroscédastique à B120, le récent perd sur les quatre seeds face à FAR + RFA. À B480, son petit gain d’accuracy ne s’accompagne pas d’un gain Worst-20 établi. Sous BF ×10 homogène à B480, le récent est moins bon que l’uniforme sur l’accuracy **et** le Worst-20.

## 2. Protocole exact et limites de comparaison

| Élément | Configuration réellement exécutée |
| --- | --- |
| Dataset / modèle | Fashion-MNIST / LeNet-5 tanh |
| Clients et partition | 25, N = 2 400 exemples/client ; Dirichlet par client à tailles contrôlées, beta = 0,1 |
| Participation | 25 clients sélectionnés et survivants, tous les tours ; aucun dropout |
| Seeds | 24, 42, 72, 121 |
| Grille | B = 120, 240, 480 × 2 bruits × 3 scénarios × 3 méthodes × 4 seeds = 216 |
| Horizon | T = 40 ; un gradient privé au modèle global courant par client et par round |
| Sampling et adjacence | Batch fixe sans remise dans le pas ; replace-one au niveau d’un exemple local |
| Clipping / pas serveur | C = 4 par exemple, U = 16 au serveur, pas serveur = 0,2 |
| FAR | Distance euclidienne brute ; alpha = 0,1 ; kappa_w = 2 diagnostique, sans cap imposé |
| Uniforme | Moyenne des messages privés après clipping serveur ; alpha = 0 |
| FAR + RFA | RFA utilisée comme référence de distance ; **ce n’est pas RFA en agrégation directe** |
| FAR + récent | Vue récente identity_new, strictement passée, avec gate initial ; pas de correction RCIG |
| Démarrage | Récent : poids uniformes t1–12, puis alpha = 0,1 ; FAR + RFA : alpha = 0,1 dès t1 |
| Attaques | Clients 0–4, soit 20 % ; t1–16 propres, BF ×10 ou IPM ×1 persistants t17–40 |
| Exécution | Gradients privés MPS, fallback désactivé ; post-traitement serveur CPU float64 autorisé |

Le champ historique local_epochs = 1 désigne un pas, pas une époque exhaustive. Chaque client traite B exemples par tour : 4 800 / 9 600 / 19 200 utilisations d’exemples sur 40 tours, soit respectivement 2 / 4 / 8 inclusions attendues par exemple, avec répétitions entre tours. Il ne s’agit pas de 2 / 4 / 8 balayages exhaustifs du dataset. Augmenter B change donc le travail d’apprentissage autant que le bruit normalisé et la variance d’échantillonnage.

La référence récente conserve un gate initial d’acceptation des clients et une mémoire, mais **aucune fusion ancienne/récente, activation statistique de correction, hystérésis ni gel de référence RCIG**. Comparer récent à RFA compare ici deux méthodes complètes, avec des démarrages différents ; ce n’est pas une ablation isolant seulement l’effet de l’historique.

### 2.1 Populations, unités et statistiques

- Test Acc. : modèle global évalué sur le test global, exprimée en %. Client Acc. : moyenne non pondérée des accuracies des clients évalués.
- Sans attaque programmée : 25 clients évalués. Avec attaque programmée : IDs 5–24, soit 20 clients évalués, **même avant t17**. Les partitions de test sont égales ; Client Acc. et Test Acc. coïncident sans attaque, mais pas nécessairement dans les scénarios attaqués.
- Worst-20 : moyenne des cinq accuracies les plus faibles parmi 25, ou des quatre parmi 20. Ce ne sont pas « les 20 pires clients ».
- Gap : Best-20 moins Worst-20, en pp ; **pas le maximum moins le minimum**, ni le gap de balanced accuracy.
- Variance : variance descriptive de population entre clients, multipliée par 10 000 pour les pp². Elle diffère du carré de l’écart-type inter-seeds.
- Les tableaux finaux utilisent le **tour 40 fixé à l’avance**, pas le meilleur tour. « ± » = écart-type échantillonnal entre les quatre seeds (ddof = 1).
- Pour chaque contraste, la différence est calculée par seed avant agrégation. IC95 = moyenne des quatre différences ± 3,182446 × leur écart-type / 2 (Student, 3 degrés de liberté). Normalité approximative des différences et indépendance entre seeds sont les hypothèses statistiques ; quatre seeds restent peu nombreuses.
- Aucune train accuracy/loss fiable n’est fournie par ces runs. Les train_loss/avg_local_loss à zéro sont des valeurs techniques supprimées de l’analyse. Les losses ci-dessous proviennent exclusivement des jeux d’évaluation.

## 3. Résultats finaux : moyenne ± écart-type

Une ligne = quatre runs. Test/Client Acc. et Worst-20 en %, gap en pp, variance en pp², loss sans unité.
""")
    for B,noise in itertools.product(BATCHES,NOISES):
        add(f"\n### B = {B} — bruit {LABEL[noise].lower()}\n")
        gg = [g for g in e["groups"] if g["B"]==B and g["noise"]==noise]
        add(table(["Scénario", "Méthode", "Test Acc.", "Client Acc.", "Test loss", "Worst-20", "Gap", "Variance"],
                  [(LABEL[g["scenario"]],LABEL[g["method"]],ms(g["stats"]["test_acc"]),ms(g["stats"]["client_acc"]),
                    ms(g["stats"]["loss"],3),ms(g["stats"]["worst20"]),ms(g["stats"]["gap"]),ms(g["stats"]["variance"])) for g in gg]))
    add("""## 4. Effet du batch à privacy maximale égale

Les comparaisons suivantes sont appariées par seed, méthode, bruit et scénario. Une cellule est « différence moyenne [IC95] ». Un signe positif est favorable pour accuracy/Worst-20, défavorable pour loss/gap/variance.

**Observation.** B480 donne une accuracy moyenne supérieure à B120 dans 15 des 18 cellules méthode × bruit × scénario. Mais ce n’est pas universel : sous BF ×10 homogène, FAR + RFA perd 3,33 pp et le récent 3,34 pp en moyenne. Les IC95 de ces deux pertes excluent zéro. La moyenne équilibrée des effets de batch ne doit donc pas servir de règle générale de promotion.

**Inférence.** La réduction du bruit normalisé peut améliorer l’apprentissage, mais elle change aussi quelles normes sont clippées. Les attaquants BF restent au rayon U quand les honnêtes passent sous U ; leur distance relative, puis leur poids FAR, peuvent augmenter. Les diagnostics de la section 6 sont compatibles avec cette explication ; ils n’isolent pas une causalité unique.
""")
    for contrast in ("B240-B120","B480-B120","B480-B240"):
        add(f"\n### {contrast_label(contrast)}\n")
        cc = [c for c in e["batch_contrasts"] if c["contrast"]==contrast]
        add(table(["Bruit", "Scénario", "Méthode", "Δ Test Acc. (pp)", "Δ Worst-20 (pp)", "Δ loss"],
                  [(LABEL[c["noise"]],LABEL[c["scenario"]],LABEL[c["method"]],ci(c["stats"]["test_acc"]),ci(c["stats"]["worst20"]),ci(c["stats"]["loss"],3)) for c in cc]))
        add(table(["Bruit", "Scénario", "Méthode", "Δ Client Acc. (pp)", "Δ gap (pp)", "Δ variance (pp²)"],
                  [(LABEL[c["noise"]],LABEL[c["scenario"]],LABEL[c["method"]],ci(c["stats"]["client_acc"]),ci(c["stats"]["gap"]),ci(c["stats"]["variance"])) for c in cc]))
    add("""## 5. Comparaisons appariées entre méthodes

Dans les libellés ci-dessous, recent = FAR + référence récente, rfa = FAR + RFA, uniform = moyenne uniforme. Les mêmes quatre seeds et la même population d’évaluation sont utilisés de chaque côté.

**Bilan récent − RFA.** Sur 18 conditions, deux IC95 d’accuracy excluent zéro positivement et un négativement. Pour Worst-20, aucun n’exclut zéro positivement, un négativement. Il n’y a pas de gain cohérent de la référence récente ; la moyenne globale proche de zéro ne doit pas effacer la perte IPM hétéroscédastique à B120.
""")
    for contrast in ("recent-rfa","recent-uniform","rfa-uniform"):
        add(f"\n### {contrast_label(contrast)}\n")
        cc = [c for c in e["method_contrasts"] if c["contrast"]==contrast]
        add(table(["B", "Bruit", "Scénario", "Δ Test Acc. (pp), IC95", "Δ Client Acc. (pp), IC95", "Δ Test loss, IC95", "Δ Loss cliente, IC95"],
                  [(c["B"],LABEL[c["noise"]],LABEL[c["scenario"]],ci(c["stats"]["test_acc"],3),ci(c["stats"]["client_acc"],3),ci(c["stats"]["loss"],4),ci(c["stats"]["client_loss"],4)) for c in cc]))
        add(table(["B", "Bruit", "Scénario", "Δ Worst-20 (pp), IC95", "Δ gap (pp), IC95", "Δ variance (pp²), IC95"],
                  [(c["B"],LABEL[c["noise"]],LABEL[c["scenario"]],ci(c["stats"]["worst20"],3),ci(c["stats"]["gap"],3),ci(c["stats"]["variance"],3)) for c in cc]))
    add("""## 6. DP, bruit, clipping et poids : ce que les métriques enseignent

### 6.1 Comptabilité finale

Delta = 10⁻⁵, 40 releases du gradient, C = 4, sensibilité replace-one de la somme = 2C. Le bruit ajouté à la somme est sigma_i × C × Z ; l’accountant reçoit sigma_i/2. Le tableau indique des valeurs recalculées, communes aux méthodes et aux seeds.
""")
    add(table(["B", "q = B/N", "sigma de base", "Écart-type upload, facteur 1", "ε facteur 1 = max", "ε facteur 2", "ε moyen hétéroscédastique"],
              [(p["B"],fmt(p["q"]),fmt(p["sigma"],6),fmt(p["noise_sd_base"],6),fmt(p["epsilon_base"],7),fmt(p["epsilon_double_noise"],6),fmt(p["epsilon_hetero_mean"],6)) for p in e["privacy"]]))
    add("""Deux budgets maximaux sont très légèrement supérieurs à 4, à l’intérieur de la tolérance préenregistrée de 10⁻⁴ ; ils ne sont pas réécrits artificiellement en 4,000000. Dans le régime homogène, tous les epsilons valent la colonne facteur 1. Dans l’hétéroscédastique, 13 clients ont le facteur 1 et 12 le facteur 2 ; l’écart-type du bruit des seconds est doublé, sa variance quadruplée.

**Ce que cela enseigne.** Sigma augmente avec B, mais l’écart-type du message, 4 sigma/B, diminue : 0,069121 → 0,058372 → 0,052936. « Sigma plus grand » ne signifie donc pas ici « gradient transmis plus bruité ». Le coût maximal est comparable ; le coût moyen hétéroscédastique ne l’est pas exactement entre B.

L’accountant décrit la protection sample-level d’un run individuel. Il ne protège pas conjointement les 216 modèles et les diagnostics oracle. Les seeds de simulation sont reproductibles, avec aléas appariés : ne pas revendiquer une confidentialité opérationnelle d’artefacts dont le bruit devient reconstructible. En déploiement, les aléas privés doivent rester secrets et les publications supplémentaires doivent être prises en compte. Aucun canal DMD n’appartient à cet écran RCIG.

### 6.2 Lecture des diagnostics

Chaque cellule des tableaux suivants est la **moyenne ± SD inter-seeds des quatre médianes temporelles t17–40**. Ce n’est ni une médiane sur quatre seeds ni un IC utilisant 24 tours indépendants.

- Clip % : fraction des 25 uploads dont la norme dépasse U = 16. Un taux élevé n’est pas une violation DP.
- R : plage des logits FAR, alpha × (distance maximale − distance minimale).
- n λmax : poids maximal divisé par le poids uniforme 1/25 ; 1 correspond à l’uniforme.
- Q = 25 × somme des poids au carré : 1 pour l’uniforme ; quantifie la concentration quadratique, pas à lui seul la qualité du modèle.
- H/log(25) : entropie normalisée ; 1 pour l’uniforme, plus faible quand la répartition est concentrée.
- Masse B : somme des poids des cinq attaquants, pendant leur phase active. Référence uniforme = 0,20. La valeur zéro sans attaque signifie absence d’attaquants actifs.

**Observations.** À B120 sans attaque ou sous BF, 100 % des messages sont clippés et les poids FAR sont presque uniformes. À B480 homogène sous BF, le récent donne une masse byzantine médiane moyenne d’environ 0,2465, contre 0,20 pour l’uniforme. Sous IPM, elle est au contraire entre environ 0,0699 et 0,0853 selon B/bruit. La distinction pertinente n’est donc pas seulement « clipping oui/non », mais **où se situent les attaquants dans la géométrie après clipping**.

**Inférence.** La distance positive récompense ici les messages BF restés à grande norme relativement aux honnêtes, alors qu’elle défavorise les messages IPM proches du centre. Cela explique plausiblement le contraste des performances, sans constituer une preuve universelle du mécanisme sous toute attaque. Les valeurs Q ou H proches de l’uniforme n’excluent pas une différence nuisible de masse byzantine.
""")
    for noise in NOISES:
        add(f"\n### 6 — Diagnostics, bruit {LABEL[noise].lower()}\n")
        gg = [g for g in e["groups"] if g["noise"]==noise]
        add(table(["B", "Scénario", "Méthode", "Clip (%)", "Norme upload médiane", "Norme référence"],
                  [(g["B"],LABEL[g["scenario"]],LABEL[g["method"]],ms(g["diagnostics"]["clip_pct"],1),ms(g["diagnostics"]["upload_norm_median"],3),ms(g["diagnostics"]["reference_norm"],3)) for g in gg]))
        add(table(["B", "Scénario", "Méthode", "R", "n λmax", "Q", "H/log(25)", "Masse B"],
                  [(g["B"],LABEL[g["scenario"]],LABEL[g["method"]],*[ms(g["diagnostics"][key],4) for key in ("logit_span","n_lambda_max","concentration","entropy_fraction","byzantine_mass")]) for g in gg]))
    add("""La référence calculée dans le contrôle uniforme ne détermine pas ses poids puisque alpha = 0 ; sa norme n’est pas une explication d’une repondération. Les valeurs affichées comme 1,0000 ou 0,0000 sont arrondies et ne signifient pas identité bit à bit.

Le taux de clipping des **gradients individuels locaux** est supprimé du transcript : privacy_clip_rate_mean est absent/null. Le taux de clipping serveur ne permet pas de le reconstituer. Aucun oracle de norme de bruit réalisé ni corrélation poids–bruit frais n’est enregistré ici ; ne pas en déduire une validation d’auto-amplification.

## 7. Qualité de la référence : oracle réellement disponible

Une erreur oracle de référence est enregistrée uniquement pour les 72 runs « récent », aux tours 13–40 : **2 016 valeurs**. Les 72 runs FAR + RFA et les 72 uniformes ne fournissent pas cet oracle. On ne peut donc pas conclure que le récent a une erreur de référence plus faible que RFA.

L’erreur est la **norme L2 au carré** entre la référence réellement déployée et la moyenne des gradients propres honnêtes du batch courant, avec le clipping individuel puis la projection serveur correspondants. C’est une somme sur les coordonnées, pas une MSE divisée par la dimension. Les champs identity_new et référence déployée sont égaux sur toutes les observations vérifiées. La cible contient aussi la variabilité de batch et n’est pas le gradient de population exact.

Attention : dans les scénarios attaqués, cette cible oracle utilise 25 clients aux tours 13–16, puis 20 à partir du tour 17 ; une rupture à t17 ne peut pas être interprétée comme le seul effet de l’attaque sur la référence. Les médianes ci-dessous sont restreintes aux tours 17–40 pour conserver une cible de population stable.
""")
    add(table(["B", "Bruit", "Scénario", "Seed 24", "Seed 42", "Seed 72", "Seed 121", "Moyenne ± SD des médianes d’erreur²"],
              [(g["B"],LABEL[g["noise"]],LABEL[g["scenario"]],*[fmt(x,4) for x in g["diagnostics"]["reference_sq_error"]["values"]],ms(g["diagnostics"]["reference_sq_error"],4)) for g in e["groups"] if g["method"]=="recent"]))
    add("""Un oracle plus faible n’est pas en soi une preuve de meilleure accuracy, fairness ou détection. Il n’existe dans cet écran ni test d’activation RCIG, ni taux de fausse activation comparable à R1, ni temps de récupération après attaque. Ces quantités sont **non identifiables**, pas égales à zéro.

## 8. Effet des attaques sur une population commune de 20 clients

Analyse secondaire post-hoc : les accuracies individuelles enregistrées permettent de recalculer le scénario sans attaque sur les seuls IDs 5–24. Les contrastes suivants comparent alors les mêmes 20 clients, pour une même seed/B/bruit/méthode. La Test Accuracy garde le test global complet. Cela élimine le changement de population d’évaluation ; cela ne transforme pas le scénario en essai de détection RCIG.

Exemple récent / B480 / bruit homogène / BF ×10 : la perte de Worst-20 sur les mêmes 20 clients est d’environ **23,16 pp** face au scénario propre. Les attaquants restent donc très nuisibles malgré le clipping serveur. Cette comparaison inclut toute la trajectoire sous attaque à partir de t17 ; elle n’est pas une sensibilité one-step.
""")
    for method in METHODS:
        add(f"\n### 8 — Attaqué moins propre, {LABEL[method]}\n")
        cc = [c for c in e["attack_effects_common20"] if c["method"]==method]
        add(table(["B", "Bruit", "Attaque", "Δ Test Acc. (pp), IC95", "Δ Client Acc. commune (pp), IC95", "Δ Worst-20 commun (pp), IC95"],
                  [(c["B"],LABEL[c["noise"]],LABEL[c["scenario"]],ci(c["stats"]["test_acc"]),ci(c["stats"]["client_acc"]),ci(c["stats"]["worst20"])) for c in cc]))
        add(table(["B", "Bruit", "Attaque", "Δ gap commun (pp), IC95", "Δ variance commune (pp²), IC95", "Δ Test loss, IC95"],
                  [(c["B"],LABEL[c["noise"]],LABEL[c["scenario"]],ci(c["stats"]["gap"]),ci(c["stats"]["variance"]),ci(c["stats"]["loss"],3)) for c in cc]))
    add("""## 9. Trajectoires : avant et pendant les attaques

Les repères t12/t16/t24/t40 sont des observations corrélées d’un même entraînement, pas des seeds supplémentaires. Le tour 12 marque la fin du warmup récent ; le tour 16 est le dernier tour propre ; t24 et t40 sont pendant l’attaque. Sans attaque programmée, tous les tours restent propres. Les valeurs ci-dessous sont les moyennes ± SD entre seeds ; chaque cellule contient Test Acc. / Worst-20.
""")
    for noise in NOISES:
        add(f"\n### 9 — Bruit {LABEL[noise].lower()}\n")
        rows = []
        for B,sc,method in itertools.product(BATCHES,SCENARIOS,METHODS):
            rr = [r for r in e["records"] if r["B"]==B and r["noise"]==noise and r["scenario"]==sc and r["method"]==method]
            cells = [ms(summarize(r["snapshots"][str(t)]["test_acc"] for r in rr)) + " / " + ms(summarize(r["snapshots"][str(t)]["worst20"] for r in rr)) for t in (12,16,24,40)]
            rows.append([B,LABEL[sc],LABEL[method],*cells])
        add(table(["B","Scénario","Méthode","t12","t16","t24","t40"],rows))
    add("""## 10. Décision et expériences éventuellement décisives

### Ce qui est défendable

1. À epsilon maximal comparable, le batch modifie substantiellement l’apprentissage **et** la géométrie après clipping. Il n’est pas un levier universel d’amélioration sous attaques.
2. FAR + récent n’apporte pas de supériorité établie face à FAR + RFA dans cette grille. Des différences de norme de référence ou d’erreur oracle ne suffisent pas à promouvoir sa complexité temporelle.
3. Le bénéfice FAR observé sous IPM coexiste avec une faiblesse sous BF ×10 : il faut raisonner sur la pondération attribuée aux attaquants, pas seulement sur la qualité supposée de la référence robuste.
4. Cet écran ne valide pas RCIG. Son objectif était précisément de vérifier si une vue récente sans correction constituait déjà un contrôle compétitif.

### Ce qu’il ne faut pas conclure

- Pas de « RCIG confirmé », de récupération mesurée, de bornes statistiques d’activation vérifiées ou de gain de retard de poids : ces mécanismes/mesures ne sont pas exécutés ici.
- Pas de « RFA bat/ne bat pas l’uniforme » au sens d’agrégation géométrique directe : il s’agit de FAR dont RFA fournit la référence.
- Pas de preuve que toute hausse de batch, tout alpha positif, ou tout clipping serveur améliore la fairness.
- Pas de confirmation indépendante après avoir choisi une cellule favorable parmi ces quatre seeds.

### Suite à soumettre à une nouvelle autorisation — non lancée

| Question restante | Expérience réellement discriminante |
| --- | --- |
| Pourquoi FAR aide sous IPM mais nuit sous BF ? | Contrôles appariés alpha = 0 / négatif / positif, RFA directe vs FAR + RFA, avec même démarrage et diagnostics de poids/angles/normes. |
| Le récent réduit-il réellement l’erreur de référence ? | Ajouter le **même oracle propre** pour RFA et récent ; ne pas comparer des cibles privées et propres différentes. |
| La correction RCIG ajoute-t-elle quelque chose ? | RCIG vs vue récente, même warmup/gate/historique ; seuils calibrés sur données/seeds séparées et figés avant l’évaluation. |
| Les attaques persistantes sont-elles détectées puis oubliées ? | Séquence propre → attaque → récupération, avec définitions préenregistrées des fausses activations, délais de détection et récupération. |
| Un régime choisi se confirme-t-il ? | Nouvelles seeds indépendantes ; critères conjoints accuracy/Worst-20/gap et compromis d’utilité fixés avant le run. |

**Décision pratique : ne pas déclencher l’extension de 4 302 runs, ni les horizons 115/268 automatiquement.** Le contrôle FAR + RFA reste une référence simple et obligatoire pour toute poursuite temporelle. La priorité scientifique est d’expliquer et traiter la fragilité de la repondération sous BF, puis seulement de tester si une correction temporelle apporte un bénéfice supplémentaire mesurable.

## 11. Résultats par seed — tour 40

Toutes les 216 observations sont affichées ci-dessous, sans sélection de meilleur tour ni suppression de seed. Les chiffres affichés sont arrondis après calcul ; le fichier d’evidence conserve les valeurs non arrondies.
""")
    for B,noise,sc in itertools.product(BATCHES,NOISES,SCENARIOS):
        add(f"\n### B = {B} — {LABEL[noise]} — {LABEL[sc]}\n")
        rr = [r for r in e["records"] if r["B"]==B and r["noise"]==noise and r["scenario"]==sc]
        rr.sort(key=lambda r:(METHODS.index(r["method"]),r["seed"]))
        add(table(["Méthode","Seed","Test Acc. (%)","Client Acc. (%)","Test loss","Loss cliente","Worst-20 (%)","Gap (pp)","Variance (pp²)"],
                  [(LABEL[r["method"]],r["seed"],*[fmt(r["final"][m],4 if "loss" in m else 2) for m in METRICS]) for r in rr]))
    add("""## 12. Vérifications, sources et reproductibilité

- 216 configurations/statuts/metrics validés par le lanceur en lecture seule ; zéro artefact invalide, zéro tâche manquante, aucune tâche active.
- 8 640 lignes de métriques clientes recalculées depuis les accuracies individuelles : moyenne, variance de population, Worst-20 et gap. Identités de population vérifiées à chaque tour.
- 2 016 erreurs oracle récentes vérifiées ; les deux champs référence déployée / identity_new sont identiques. Aucun oracle de référence inventé pour les deux autres bras.
- Calcul indépendant des moyennes, différences appariées et IC95 ; concordance des valeurs non arrondies. Quantile Student t(0,975 ; 3) calculé par inversion de sa CDF, pas approximé par 1,96.
- Les empreintes SHA256 des 216 metrics.json et le verrou scientifique sont conservés dans l’evidence. Aucun fichier de résultats ou source d’entraînement n’a été modifié par l’analyse.

Sources locales :

- [Protocole préenregistré](RCIG_Batch_Screen_216_Protocol.md).
- [Matrice de 216 runs](../../configs/ldp_gradient_far/rcig_batch_screen_v1.yaml).
- [Contrôle de validité du lanceur](../../scripts/run_rcig_batch_screen.py).
- [Implémentation du contrôle récent](../../algorithms/ldp_gradient_far_recent.py).
- [Frontière d’évaluation oracle](../../scripts/run_rcig_batch_screen_experiment.py).
- [Données et calculs non arrondis de cette analyse](RCIG_Batch_Screen_216_Evidence.json).
- [Générateur de l’analyse](../../scripts/analyze_rcig_batch_screen_216.py).
- [Résultats R3 antérieurs, inchangés](RCIG_R3_Exploratory_Results_Complete.md).

Les chemins précis et empreintes de chaque run se trouvent dans l’evidence JSON. Ce rapport remplace une notification de progression par une décision scientifique ; aucune étape suivante n’est lancée automatiquement.
""")
    text = "\n".join(parts)
    # Reject broken local links and legacy math delimiters problematic in preview.
    for target in re.findall(r"\]\(([^)]+)\)",text):
        assert (OUT.parent / target).resolve().exists(), target
    assert "\\[" not in text and "\\(" not in text
    return text


def main():
    evidence = build_evidence()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    OUT.write_text(render_report(evidence))
    print(json.dumps({"records":len(evidence["records"]), "groups":len(evidence["groups"]),
                      "oracle_counts":evidence["oracle_counts"], "privacy":evidence["privacy"],
                      "descriptive":evidence["global_descriptive"]}, indent=2))


if __name__ == "__main__":
    main()
