#!/usr/bin/env python3
"""Read-only analysis of the 254 frozen aggregate-controller experiments.

No training, threshold tuning, or mutation of experiment results. Outputs are
an evidence JSON and a French Markdown report, outside the campaign directory.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/ldp_gradient_far/aggregate_predictor_calibration_v1"
MATRIX = ROOT / "configs/ldp_gradient_far/aggregate_predictor_calibration_v1.yaml"
OUT = ROOT / "output/analysis"
REPORT = OUT / "Aggregate_Predictor_EMA_RCIG_Final_Analysis.md"
EVIDENCE = OUT / "Aggregate_Predictor_EMA_RCIG_Final_Evidence.json"
ARMS = {
    "far_rfa": "FAR(RFA)", "rfa_direct": "RFA directe",
    "ema_smooth": "Lissage EMA permanent", "ema_iso": "EMA isotrope",
    "ema_radial": "EMA radial", "ema_projection": "EMA projection",
    "rcig_iso": "RCIG isotrope", "rcig_radial": "RCIG radial",
    "rcig_projection": "RCIG projection",
}
NOISE = {"homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique"}
SCENARIOS = {"clean": "Sans attaque", "brutal_bf": "Bit-Flip ×10",
             "slow_ipm": "IPM lent", "persistent_alie": "ALIE persistant"}
MODEL = {
    "accuracy": ("test_accuracy", 100), "client_accuracy": ("client_accuracy_mean", 100),
    "loss": ("test_loss", 1), "client_loss": ("client_loss_mean", 1),
    "variance": ("client_accuracy_variance_pct2", 1),
    "worst20": ("worst20_accuracy_pct", 1), "gap": ("best20_worst20_gap_pct", 1),
    "balanced_accuracy": ("mean_client_balanced_accuracy_pct", 1),
}
MECHANISM = ["apc_applied_mse", "apc_raw_far_mse", "apc_correction_norm",
    "apc_removed_alignment_with_clean_gradient", "apc_designated_two_client_mass",
    "apc_designated_current_contribution_norm", "apc_target_norm",
    "far_max_weight", "far_weight_l2_squared", "far_logit_range", "far_server_clip_rate",
    "privacy_clip_rate_mean", "apc_rcig_inner_gate_active", "apc_rcig_inner_trust"]
for pred in ("ema", "rcig"):
    MECHANISM += [f"apc_{pred}_{k}" for k in ("predictor_mse", "residual_l2",
        "residual_mahalanobis", "variance_trace", "variance_condition",
        "target_inside_isotropic", "target_inside_ellipsoid")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def near(a, b, tol=1e-8):
    assert math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol), (a, b)


def stats(values):
    v = list(values)
    assert v and all(math.isfinite(x) for x in v)
    mean = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else None
    ci = 4.302652729911275 * sd / math.sqrt(3) if len(v) == 3 else None
    return {"n": len(v), "mean": mean, "sd": sd, "values": v,
            "descriptive_t95": None if ci is None else [mean-ci, mean+ci]}


def fmt(value, decimals=2):
    if isinstance(value, dict):
        return f"{value['mean']:.{decimals}f} ± {value['sd']:.{decimals}f}".replace(".", ",")
    return f"{value:.{decimals}f}".replace(".", ",")


def window(rows):
    d = {name: st.mean(r[key] * scale for r in rows) for name, (key, scale) in MODEL.items()}
    for key in MECHANISM:
        valid = [r[key] for r in rows if r.get(key) is not None]
        if valid:
            d[key] = st.mean(valid)
    d["rounds"] = [r["round_num"] for r in rows]
    d["correction_rounds"] = [r["round_num"] for r in rows if r["apc_correction_triggered"]]
    d["correction_fraction"] = len(d["correction_rounds"]) / len(rows)
    d["same_path_mse_reduction_pct"] = 100 * (1 - d["apc_applied_mse"] / d["apc_raw_far_mse"])
    d["pointwise_mse_worsened_count"] = sum(r["apc_applied_mse"] > r["apc_raw_far_mse"]+1e-8 for r in rows)
    gammas = [r["apc_gamma"] for r in rows if r["apc_correction_triggered"] and r["apc_gamma"] is not None]
    d["gamma_on_corrected_rounds"] = st.mean(gammas) if gammas else None
    return d


def recover(rows, clean, end, ev):
    # Independent recomputation of the pre-registered first 3-round episode.
    candidates = []
    for idx in range(end, len(rows)-ev["recovery_consecutive_rounds"]+1):
        good = True
        for j in range(idx, idx+ev["recovery_consecutive_rounds"]):
            r, c = rows[j], clean[j]
            good &= (100*(c["test_accuracy"]-r["test_accuracy"]) <= ev["recovery_accuracy_tolerance_pp"]
                and c["worst20_accuracy_pct"]-r["worst20_accuracy_pct"] <= ev["recovery_worst20_tolerance_pp"]
                and r["best20_worst20_gap_pct"]-c["best20_worst20_gap_pct"] <= ev["recovery_gap_tolerance_pp"])
        if good:
            candidates.append(idx+1-end)
    return candidates[0] if candidates else None


def analyze():
    m = yaml.safe_load(MATRIX.read_text())
    locked = read(CAMPAIGN / "campaign_lock.json")
    for kind in ("files", "sources"):
        for path, digest in locked[kind].items():
            assert sha(ROOT / path) == digest, ("source/protocol drift", path)
    artifact = read(CAMPAIGN / "frozen_radii.json")
    completion = read(CAMPAIGN / "completion.json")
    assert completion == {"calibration": 38, "evaluation": 216, "scientific_promotion": False, "validated": 254}
    assert artifact["matrix_hash"] == sha(MATRIX)
    expected = set()
    for phase in ("calibration", "evaluation"):
        grid = ["clean"] if phase == "calibration" else list(SCENARIOS)
        arms = ["far_rfa"] if phase == "calibration" else list(ARMS)
        for n, s, c, a in itertools.product(NOISE, m[phase+"_seeds"], grid, arms):
            expected.add((phase, n, s, c, a))
    data, traces, registry = {}, {}, []
    for status_path in sorted(CAMPAIGN.glob("*/*/orchestration_status.json")):
        status = read(status_path)
        t = status["task"]
        key = (t["phase"], t["noise"], t["seed"], t["scenario"], t["arm"])
        assert key in expected and key not in data and status["status"] == "completed"
        directory = status_path.parent
        paths = list(directory.rglob("metrics.json"))
        assert len(paths) == 1
        path = paths[0]
        assert sha(path) == status["metrics_sha256"]
        assert sha(directory/"resolved_config.yaml") == status["config_sha256"]
        trace_path = directory / "simulator_randomness_private_audit.jsonl"
        assert sha(trace_path) == status["trace_sha256"]
        runtime = read(directory/"runtime_imports.json")
        assert runtime["stage"] == "after_training" and runtime["mps_fallback"] == 0
        assert all(locked['sources'].get(path) == digest for path, digest in runtime['source_sha256'].items())
        assert status["device"] == "mps" and status["mps_fallback"] == 0
        payload = read(path)
        rows = payload["rounds"]
        assert [r["round_num"] for r in rows] == list(range(1, 41))
        for r in rows:
            assert r["apc_target_ids"] == "2,3,4,5,6,7,8,9" and r["num_evaluated_clients"] == 8
            assert r["rcig_private_gradient_mps_fraction"] == 1
            assert r["apc_history_observations"] == r["round_num"]-1
            assert r["apc_predictors_strictly_past"] and not r["apc_current_round_used_for_predictor"]
            assert not r["apc_oracle_used_for_deployment"] and r["apc_same_cohort_verified"]
            for _, (field, _) in MODEL.items():
                assert math.isfinite(r[field]), (key, field)
            acc = r["client_accuracy_values_oracle"]
            if isinstance(acc, dict):
                acc = list(acc.values())
            assert len(acc) == 8
            near(st.mean(acc), r["client_accuracy_mean"])
            near(st.pvariance(acc)*10000, r["client_accuracy_variance_pct2"])
            near(st.mean(sorted(acc)[:2])*100, r["worst20_accuracy_pct"])
            near((st.mean(sorted(acc)[-2:])-st.mean(sorted(acc)[:2]))*100, r["best20_worst20_gap_pct"])
        near(rows[-1]["privacy_epsilon_max"], 4.000036805147497)
        near(rows[-1]["privacy_delta"], 1e-5)
        trace = sorted((json.loads(line) for line in trace_path.read_text().splitlines()),
                       key=lambda z: (z["round"], z["client_id"]))
        assert len(trace) == 400
        assert [(r["round"], r["client_id"]) for r in trace] == list(itertools.product(range(1,41), range(10)))
        traces[key] = trace
        data[key] = payload
        registry.append({**t, "metrics": str(path.relative_to(ROOT)), "sha256": sha(path),
                         "finished_at": status["finished_at"]})
        if t["phase"] == "calibration":
            assert artifact["calibration_metric_hashes"][str(path.relative_to(ROOT))] == sha(path)
        else:
            assert payload["config"]["apc_radius_artifact_sha256"] == sha(CAMPAIGN/"frozen_radii.json")
    assert set(data) == expected and len(data) == 254
    # Actual random draws, not just equality of seed labels.
    paired_records = 0
    for key, trace in traces.items():
        phase, n, s, c, a = key
        partners = [(phase, "heteroscedastic", s, c, a)] if n == "homogeneous" else []
        if phase == "evaluation":
            partners += [(phase,n,s,"clean","far_rfa"), (phase,n,s,"clean",a)]
        for other in set(partners)-{key}:
            for x, y in zip(trace, traces[other], strict=True):
                assert x["permutations"] == y["permutations"] and x["standard_gaussians"] == y["standard_gaussians"]
                paired_records += 1
                if n == other[1]:
                    end = m["scenarios"][c]["start"] if c != "clean" and a == other[4] else 13
                    if x["round"] <= end:
                        assert x["model_before"] == y["model_before"] and x["private_upload"] == y["private_upload"]
    for n, pred, geometry in itertools.product(NOISE, ("ema", "rcig"), ("isotropic", "mahalanobis")):
        field = "residual_l2" if geometry == "isotropic" else "residual_mahalanobis"
        maxima = [max(r[f"apc_{pred}_{field}"] for r in data[("calibration",n,s,"clean","far_rfa")]["rounds"][12:]) for s in m["calibration_seeds"]]
        near(max(maxima), artifact["radii"][n][pred][geometry])
        assert maxima == artifact["per_trajectory_maxima"][n][pred][geometry]
    records = []
    for key in sorted(data):
        phase, n, seed, c, arm = key
        if phase != "evaluation":
            continue
        rows = data[key]["rounds"]
        clean = data[(phase,n,seed,"clean",arm)]["rounds"]
        baseline = data[(phase,n,seed,c,"far_rfa")]["rounds"]
        spec = m["scenarios"][c]
        w = {"postwarmup": rows[12:]}
        if c != "clean":
            w.update(attack=rows[spec["start"]-1:spec["end"]], recovery=rows[spec["end"]:])
            if spec["start"] > 13:
                w["preattack"] = rows[12:spec["start"]-1]
        record = dict(noise=n, seed=seed, scenario=c, arm=arm,
            metrics_path=next(z["metrics"] for z in registry if (z["phase"],z["noise"],z["seed"],z["scenario"],z["arm"])==key),
            final={name:rows[-1][field]*scale for name,(field,scale) in MODEL.items()},
            windows={name:window(rs) for name,rs in w.items()},
            final_delta_vs_far={name:(rows[-1][field]-baseline[-1][field])*scale for name,(field,scale) in MODEL.items()},
            epsilon_max=rows[-1]["privacy_epsilon_max"], epsilon_mean=rows[-1]["privacy_epsilon_mean"],
            cumulative_error_sq=rows[-1]["apc_cumulative_error_norm_sq"],
            time_mean_error_sq=rows[-1]["apc_time_mean_error_norm_sq"])
        record["postwarmup_delta_vs_far"] = {name:st.mean((r[field]-b[field])*scale for r,b in zip(rows[12:],baseline[12:])) for name,(field,scale) in MODEL.items()}
        if c != "clean":
            end=spec["end"]
            record["attack_end"] = {name:rows[end-1][field]*scale for name,(field,scale) in MODEL.items()}
            record["attack_end_delta_vs_far"] = {name:(rows[end-1][field]-baseline[end-1][field])*scale for name,(field,scale) in MODEL.items()}
            record["recovery_delay"] = recover(rows,clean,end,m["evaluation"])
            record["final_damage_vs_own_clean"] = {name:(clean[-1][field]-rows[-1][field])*scale for name,(field,scale) in MODEL.items()}
            record["paired_damage_trajectory"] = {name:[(c0[field]-r[field])*scale for r,c0 in zip(rows,clean)] for name,(field,scale) in MODEL.items()}
        records.append(record)
    grouped=defaultdict(list)
    for r in records:
        grouped[(r["noise"],r["scenario"],r["arm"])].append(r)
    summaries=[]
    for (n,c,a), rs in grouped.items():
        g={"noise":n,"scenario":c,"arm":a,"seeds":[r["seed"] for r in rs]}
        for where in ["final", "final_delta_vs_far", "postwarmup_delta_vs_far"] + ([] if c=="clean" else ["attack_end", "attack_end_delta_vs_far", "final_damage_vs_own_clean"]):
            g[where]={k:stats(r[where][k] for r in rs) for k in MODEL}
        g["windows"]={}
        for name in rs[0]["windows"]:
            fields=[k for k,v in rs[0]["windows"][name].items() if isinstance(v,(float,int))
                    and all(isinstance(r['windows'][name].get(k),(float,int)) for r in rs)]
            g["windows"][name]={k:stats(r["windows"][name][k] for r in rs) for k in fields}
        if c!="clean":
            g["recovery_delays"]=[r["recovery_delay"] for r in rs]
        summaries.append(g)
    # No 18 independent seeds: this checks 3 seeds in 6 scenario/noise strata.
    by_key={(r['noise'],r['scenario'],r['arm'],r['seed']):r for r in records}
    for n,c,a,s in itertools.product(NOISE,list(SCENARIOS)[1:],list(ARMS)[3:],m['evaluation_seeds']):
        assert by_key[n,c,a,s]['recovery_delay'] == by_key[n,c,'far_rfa',s]['recovery_delay']
    existing = read(CAMPAIGN/'results_progress.json')['records']
    for r in existing:
        if r['phase'] != 'evaluation':
            continue
        own=by_key[r['noise'],r['scenario'],r['arm'],r['seed']]
        near(own['final']['accuracy'],r['test_accuracy']*100)
        near(own['windows']['postwarmup']['apc_applied_mse'],r['mean_applied_mse'])
        if r['scenario'] != 'clean':
            assert own['recovery_delay'] == r['recovery_delay_rounds']
    evidence={"campaign":m["campaign_id"],"analysis_time_utc":datetime.now(timezone.utc).isoformat(),
        "audit":{"validated":254,"rounds_checked":10160,"actual_draw_pair_checks":paired_records,
            "metrics_config_trace_hashes_checked":True,"all_private_training_mps":True,
            "fairness_recomputed_on_eight_fixed_clients":True,"radii_recomputed":True,
            "historical_runner_audit_also_passed":True},
        "calibration":artifact,"manifest":registry,"records":records,"groups":summaries,
        "statistics":"sample SD ddof=1 across 3 seeds; t95 intervals descriptive, no multiplicity correction, no promotion",
        "scientific_promotion":False}
    OUT.mkdir(parents=True,exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return m, evidence, data


def report(m,e,data):
    G={(g["noise"],g["scenario"],g["arm"]):g for g in e["groups"]}
    lines=[]
    def put(s=""): lines.append(s)
    def table(headers, rows):
        put("| " + " | ".join(headers) + " |")
        put("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows: put("| " + " | ".join(map(str,row)) + " |")
        put()
    put("# EMA/RCIG : analyse finale du contrôle de l’agrégat FAR sous DP côté client\n")
    put("Campagne `aggregate_predictor_calibration_v1`, analysée le 14 septembre 2026.\n")
    put("## 1. Verdict\n")
    put("**Les 254 runs sont valides : 38 calibrations et 216 évaluations. Cette instanciation ne démontre pas un bénéfice end-to-end pertinent des contrôles occasionnels, ni une supériorité de RCIG sur EMA.** Les erreurs d’agrégat diminuent parfois, mais les gains finaux d’accuracy et de fairness restent minuscules et la récupération n’est pas accélérée. Ce résultat n’est pas un rejet général de la DP côté client ou du contrôle d’agrégat.\n")
    maxacc=max(abs(g["final_delta_vs_far"]["accuracy"]["mean"]) for g in e["groups"] if g["arm"] not in ["far_rfa","rfa_direct","ema_smooth"])
    maxworst=max(abs(g["final_delta_vs_far"]["worst20"]["mean"]) for g in e["groups"] if g["arm"] not in ["far_rfa","rfa_direct","ema_smooth"])
    put(f"Sur les six contrôles à seuil, les différences moyennes finales avec FAR inchangé restent en valeur absolue ≤ **{fmt(maxacc,3)} pp d’accuracy** et ≤ **{fmt(maxworst,3)} pp de Worst-20**, parmi les huit couples régime/scénario. Il s’agit de l’étendue descriptive observée, pas d’une borne universelle ni d’un test de non-infériorité.\n")
    table(["Question", "Observation", "Conclusion limitée à cette campagne"],[
        ["Coût sans attaque", "Corrections absentes ou minuscules ; mêmes accuracies/Worst-20 finaux", "Préservation empirique, pas certification statistique du coût"],
        ["Protection contre Bit-Flip", "Corrections en homogène ; quasi-absence en hétéroscédastique", "Réduction de MSE sans gain modèle substantiel"],
        ["IPM lent", "Aucune correction pendant l’attaque ; corrections ellipsoïdales après son arrêt", "La normalisation réagit au retour au régime propre"],
        ["ALIE persistant", "Corrections isotropes, mais absorption progressive par les états", "Pas d’amélioration utile ni récupération plus rapide"],
        ["RCIG contre EMA", "RCIG ne domine ni les corrections simples ni leur utilité", "Pas de justification empirique de la complexité RCIG ici"],
        ["Lissage permanent", "MSE fortement réduite, accuracy moyenne plus faible dans les huit cas", "La MSE seule ne valide pas une méthode"]])
    put("## 2. Protocole, conventions et audit\n")
    put("Fashion-MNIST, LeNet-5 tanh, 10 clients, partition client-Dirichlet équilibrée β = 0,1. B = 120, T = 40, **un gradient de batch privé par tour**, pas une époque locale complète. C = 4 par exemple ; U = 16 au serveur ; distance brute ; α FAR = 0,1 ; pas serveur = 0,2. Les tours 1–12 sont uniformes et communs. Les règles comparées interviennent aux tours 13–40. RFA courante construit les poids de FAR ; EMA/RCIG sont les prédicteurs du contrôle après agrégation, pas de nouvelles références de calcul des poids.\n")
    put("Le canal est sample-level côté client, batch fixe sans remise, adjacence replace-one d’un exemple. **Ce n’est pas une garantie sur le remplacement de tout le dataset d’un client.** Tous les runs ont ε maximal = 4,000036805 et δ = 10⁻⁵. Le multiplicateur σ de base est 1,465522087. L’écart-type par coordonnée de l’upload vaut Cσ/B = 0,048850736 ; le facteur 2 donne 0,097701472. L’hétéroscédasticité alterne les facteurs publics 1 et 2 : elle augmente aussi l’énergie moyenne du bruit. La comparaison entre régimes n’isole donc pas la dispersion des niveaux à énergie égale.\n")
    put("Les calculs des gradients privés ont été faits sur **MPS, sans fallback**. L’agrégation et les contrôles sont le post-traitement CPU float64 prévu par le protocole. Les métriques oracle et les évaluations held-out sont hors mécanisme privé ; elles ne doivent pas être publiées comme si elles étaient couvertes par le seul budget d’entraînement. Les zéros techniques de train loss ne sont pas utilisés.\n")
    put("Accuracy test : test global complet. Accuracy cliente, Worst-20, gap et variance : **8 identités honnêtes fixes, 2 à 9**, même sans attaque ou après récupération. Worst-20 = moyenne des 2 plus faibles accuracies sur ces 8 clients (arrondi supérieur de 20 % × 8), gap = moyenne des 2 meilleures moins celle des 2 moins bonnes. La variance interclients est calculée avec diviseur 8 et exprimée en pp² ; l’écart-type entre seeds utilise le diviseur 2. L’accuracy globale ne coïncide donc pas nécessairement avec la moyenne des 8 clients retenus.\n")
    put("Seeds d’évaluation : 932001, 932002, 932003. Toutes les moyennes ± écarts-types sont calculées **entre ces trois seeds**, après résumé de chaque trajectoire. Les tours et les neuf politiques appariées ne sont pas des répétitions indépendantes. Les intervalles t95 stockés dans l’evidence sont descriptifs, supposent un modèle de différences approximativement normal et ne corrigent pas les comparaisons multiples. Aucun nouveau seuil de succès n’a été sélectionné après les résultats.\n")
    put(f"Audit indépendant : {e['audit']['validated']} identités complètes ; {e['audit']['rounds_checked']} lignes de tours ; empreintes metrics/configurations/traces vérifiées ; {e['audit']['actual_draw_pair_checks']} comparaisons de tirages effectifs ; huit rayons recalculés ; fairness recalculée à partir des accuracies individuelles. Le validateur du lanceur a également confirmé 254/254 avec les empreintes des sources et l’absence d’accès aux oracles par le mécanisme.\n")
    put("## 3. Calibration hors test\n")
    put("Pour chacun des deux prédicteurs et des deux régimes, 19 trajectoires propres distinctes fournissent leur maximum sur les tours 13–40. Le rayon est le maximum de ces 19 maxima. Le rayon isotrope s’applique à ||A − P||₂ ; le rayon ellipsoïdal à sqrt((A − P)ᵀ V⁻¹(A − P)). V est un **second moment diagonal résiduel**, pas la covariance DP exacte. Les deux rayons ont des unités différentes.\n")
    table(["Bruit", "Prédicteur", "Rayon isotrope", "Rayon ellipsoïdal"],[[NOISE[n],p.upper(),fmt(e['calibration']['radii'][n][p]['isotropic'],6),fmt(e['calibration']['radii'][n][p]['mahalanobis'],6)] for n,p in itertools.product(NOISE,["ema","rcig"])])
    put("L’argument de rang concerne une probabilité marginale de premier déclenchement propre ≤ 1/20, sous échangeabilité des trajectoires. Il ne garantit ni un taux conditionnel au rayon observé, ni une couverture simultanée des politiques, ni un contrôle de la covariance après intervention. Trois seeds d’évaluation ne peuvent pas certifier un taux de fausse correction de 5 %. Les occurrences partagées par plusieurs contrôles ne sont pas indépendantes.\n")
    table(["Bruit", "Contrôle", "Trajectoires avec correction /3", "Tours corrigés /84", "Norme maximale retirée"],[[NOISE[n],ARMS[a],sum(bool(r['windows']['postwarmup']['correction_rounds']) for r in e['records'] if r['noise']==n and r['scenario']=='clean' and r['arm']==a),sum(len(r['windows']['postwarmup']['correction_rounds']) for r in e['records'] if r['noise']==n and r['scenario']=='clean' and r['arm']==a),fmt(max(z['apc_correction_norm'] for key,x in data.items() if key[0]=='evaluation' and key[1]==n and key[3]=='clean' and key[4]==a for z in x['rounds'][12:]),6)] for n,a in itertools.product(NOISE,list(ARMS)[3:])])
    put("## 4. Résultats modèle au tour 40\n")
    put("Plus élevé est meilleur pour les accuracies et Worst-20 ; plus faible est meilleur pour loss, variance et gap. Un gap plus faible peut aussi provenir d’une baisse des meilleurs clients : il faut le lire avec Worst-20 et accuracy.\n")
    for n,c in itertools.product(NOISE,SCENARIOS):
        put(f"### {NOISE[n]} — {SCENARIOS[c]}\n")
        table(["Règle", "Test Acc. (%)", "Client Acc. (%)", "Test loss", "Variance (pp²)", "Worst-20 (%)", "Gap (pp)"],[[ARMS[a]]+[fmt(G[n,c,a]['final'][k],3 if k=='loss' else 2) for k in ['accuracy','client_accuracy','loss','variance','worst20','gap']] for a in ARMS])
    put("## 5. Baisse de MSE : mécanisme contre utilité\n")
    put("La MSE est ||A appliqué − h||₂², avec h moyenne des gradients de batch propres **déjà clippés par exemple**, sur les 8 honnêtes fixes, au modèle propre au run. La comparaison A brut / A appliqué dans une même ligne utilise exactement la même cible et les mêmes messages : c’est la mesure la plus directe de l’action du contrôle. En revanche, comparer des MSE entre trajectoires divergentes ne garde pas le modèle ni h identiques. La réduction présentée ci-dessous est 100 × (1 − moyenne MSE appliquée / moyenne MSE brute), calculée par seed puis résumée.\n")
    table(["Bruit / scénario", "Règle", "MSE brute pendant attaque", "MSE appliquée", "Réduction (%)", "Δ accuracy finale (pp)", "Δ Worst-20 final (pp)"],[[NOISE[n]+' / '+SCENARIOS[c],ARMS[a],fmt(G[n,c,a]['windows']['attack']['apc_raw_far_mse'],3),fmt(G[n,c,a]['windows']['attack']['apc_applied_mse'],3),fmt(G[n,c,a]['windows']['attack']['same_path_mse_reduction_pct']),fmt(G[n,c,a]['final_delta_vs_far']['accuracy'],3),fmt(G[n,c,a]['final_delta_vs_far']['worst20'],3)] for n,c,a in itertools.product(NOISE,list(SCENARIOS)[1:],['ema_iso','ema_radial','rcig_iso','rcig_radial'])])
    put("**Observation :** les contrôles isotropes retirent souvent davantage de MSE que les contrôles ellipsoïdaux ; RCIG n’apporte pas de gain modèle convaincant sur EMA. Le lissage permanent est encore plus efficace sur la MSE mais son accuracy moyenne finale est inférieure à FAR dans les huit couples scénario/régime.\n")
    put("### Ne pas cacher le gain temporaire sous Bit-Flip homogène\n")
    table(["Règle", "Δ accuracy fin d’attaque (t24)", "Δ Worst-20 fin d’attaque", "Δ accuracy finale (t40)", "Δ Worst-20 final"],[[ARMS[a],fmt(G['homogeneous','brutal_bf',a]['attack_end_delta_vs_far']['accuracy']),fmt(G['homogeneous','brutal_bf',a]['attack_end_delta_vs_far']['worst20']),fmt(G['homogeneous','brutal_bf',a]['final_delta_vs_far']['accuracy']),fmt(G['homogeneous','brutal_bf',a]['final_delta_vs_far']['worst20'])] for a in list(ARMS)[3:]])
    put("RCIG isotrope apporte un petit gain transitoire : +0,58 ± 0,27 pp d’accuracy et +0,35 ± 0,40 pp de Worst-20 au tour 24 ; au tour 40 le gain d’accuracy n’est plus que +0,09 pp en moyenne. L’absence de gain final substantiel ne doit pas effacer cette observation, mais elle ne permet pas de promouvoir une amélioration générale.\n")
    table(["Bruit propre", "Règle", "MSE moyenne tours 13–40", "Δ accuracy finale", "Δ Worst-20 final"],[[NOISE[n],ARMS[a],fmt(G[n,'clean',a]['windows']['postwarmup']['apc_applied_mse'],3),fmt(G[n,'clean',a]['final_delta_vs_far']['accuracy']),fmt(G[n,'clean',a]['final_delta_vs_far']['worst20'])] for n,a in itertools.product(NOISE,['far_rfa','ema_smooth'])])
    put("**Inférence :** la correction peut retirer surtout des fluctuations qui contribuent beaucoup à la norme en grande dimension, sans améliorer suffisamment la direction d’apprentissage. La mémoire peut aussi introduire un décalage temporel. Ce ne sont pas des causes identifiées par cette seule campagne. La borne de convergence fondée sur l’erreur d’agrégat est une borne supérieure, pas une équivalence MSE ↔ accuracy/fairness ; l’erreur de sampling/clipping entre h et le gradient de population reste à prendre en compte.\n")
    put("## 6. Déclenchements : bien séparer attaque et récupération\n")
    put("Une correction est une intervention du contrôleur, **pas une identification certifiée d’un client byzantin**. Les pourcentages ci-dessous sont descriptifs : tours corrigés sur la fenêtre, moyennés par seed. Le lissage permanent n’est pas un détecteur et n’est pas inclus.\n")
    table(["Bruit / attaque", "Règle", "Pendant attaque (%)", "Après arrêt (%)"],[[NOISE[n]+' / '+SCENARIOS[c],ARMS[a],fmt({**G[n,c,a]['windows']['attack']['correction_fraction'],'mean':100*G[n,c,a]['windows']['attack']['correction_fraction']['mean'],'sd':100*G[n,c,a]['windows']['attack']['correction_fraction']['sd']}),fmt({**G[n,c,a]['windows']['recovery']['correction_fraction'],'mean':100*G[n,c,a]['windows']['recovery']['correction_fraction']['mean'],'sd':100*G[n,c,a]['windows']['recovery']['correction_fraction']['sd']})] for n,c,a in itertools.product(NOISE,list(SCENARIOS)[1:],['ema_iso','ema_radial','rcig_iso','rcig_radial'])])
    put("### Cas décisif : IPM lent n’est pas détecté pendant son activité\n")
    put("**Observation :** aucun des six contrôles à seuil ne corrige pendant les tours IPM 17–28, dans les deux régimes et les trois seeds. Les contrôles ellipsoïdaux EMA corrigent ensuite tous les tours 29–40. Le taux global de 42,86 % correspond donc à 12 tours de récupération /28, **pas** à la détection de l’attaque.\n")
    put("Exemple transparent : FAR inchangé, bruit hétéroscédastique, seed 932001. Les états ci-dessous sont ceux observés sur la trajectoire non corrigée.\n")
    rs=data[('evaluation','heteroscedastic',932001,'slow_ipm','far_rfa')]['rounds']
    table(["Tour", "Attaque active", "||A − P EMA||₂", "Trace V EMA", "Statistique ellipsoïdale EMA", "Rayon figé"],[[r['round_num'],'oui' if r['apc_attack_active'] else 'non',fmt(r['apc_ema_residual_l2'],3),fmt(r['apc_ema_variance_trace'],3),fmt(r['apc_ema_residual_mahalanobis'],3),fmt(e['calibration']['radii']['heteroscedastic']['ema']['mahalanobis'],3)] for r in rs if r['round_num'] in [16,17,24,28,29,33,40]])
    put("**Interprétation appuyée par les états enregistrés :** IPM réduit ici l’amplitude du résidu, donc le second moment adaptatif décroît. Au retour des uploads honnêtes, le résidu retrouve une amplitude plus grande alors que V reste faible ; la statistique standardisée franchit le rayon. Le contrôleur réagit à une rupture de régime de second moment, pas nécessairement à une attaque. Cette explication ne constitue pas une expérience causale avec V figé.\n")
    put("### Bit-Flip et ALIE\n")
    put("Sous Bit-Flip homogène, la correction isotrope se déclenche sur toute la fenêtre, mais la contraction reste modérée. Sous bruit hétéroscédastique, le bruit propre conduit à des rayons plus larges et seul EMA isotrope intervient, faiblement. Sous ALIE, les contrôles isotropes corrigent pendant toute l’attaque ; le contrôle RCIG radial hétéroscédastique intervient sur très peu de tours. Le prédicteur et V étant actualisés à partir d’observations non corrigées, une contamination persistante peut être progressivement absorbée par leur mémoire.\n")
    put("## 7. Récupération et dommages\n")
    put("Le délai préenregistré est le premier tour après l’arrêt débutant trois tours consécutifs où, par rapport au run propre de **même règle/seed/régime**, accuracy ne perd pas plus de 1 pp, Worst-20 pas plus de 2 pp, et le gap n’augmente pas de plus de 2 pp. ND signifie aucune séquence observée avant T = 40 : résultat censuré, pas délai nul. Les fenêtres disponibles sont 16, 12 et 8 tours pour BF, IPM et ALIE. Un épisode de récupération n’impose pas que l’amélioration se maintienne jusqu’au dernier tour.\n")
    table(["Bruit / attaque", "Règle", "Délais seeds 932001 / 932002 / 932003", "Dommage accuracy final (pp)", "Dommage Worst-20 final (pp)"],[[NOISE[n]+' / '+SCENARIOS[c],ARMS[a], ' / '.join('ND' if d is None else str(d) for d in G[n,c,a]['recovery_delays']),fmt(G[n,c,a]['final_damage_vs_own_clean']['accuracy']),fmt(G[n,c,a]['final_damage_vs_own_clean']['worst20'])] for n,c,a in itertools.product(NOISE,list(SCENARIOS)[1:],['far_rfa','rfa_direct','ema_smooth','ema_iso','rcig_iso','rcig_radial'])])
    put("**Les six contrôles à seuil ont exactement les mêmes délais de récupération que FAR inchangé, pour les 18 trajectoires régime × attaque × seed.** La campagne ne montre donc aucune accélération de récupération par ces contrôles. RFA directe et le lissage ont leurs propres témoins propres : un retour plus rapide à un modèle propre plus faible n’établit pas une supériorité absolue.\n")
    put("Certaines métriques finales sont meilleures sous ALIE que dans le run propre apparié, notamment en hétéroscédastique. Cela ne rend pas ALIE bénéfique en général et ne constitue pas une robustesse certifiée ; la direction et l’amplitude de cette attaque ne maximisent pas notre dommage final. Il faut garder les pertes transitoires et les contrôles appariés, pas sélectionner uniquement le dernier point favorable.\n")
    put("## 8. Comparaison des prédicteurs et diagnostic de covariance\n")
    put("Les deux prédicteurs sont observés sur **les mêmes trajectoires FAR non corrigées** ci-dessous. Le tableau sépare donc leur précision de la divergence end-to-end des modèles.\n")
    table(["Bruit / scénario", "MSE prédicteur EMA", "MSE prédicteur RCIG", "Trace V EMA", "Trace V RCIG", "Gate interne RCIG actif (%)"],[[NOISE[n]+' / '+SCENARIOS[c]]+[fmt(G[n,c,'far_rfa']['windows']['postwarmup'][k],3) for k in ['apc_ema_predictor_mse','apc_rcig_predictor_mse','apc_ema_variance_trace','apc_rcig_variance_trace']]+[fmt(100*G[n,c,'far_rfa']['windows']['postwarmup']['apc_rcig_inner_gate_active']['mean'])] for n,c in itertools.product(NOISE,SCENARIOS)])
    put("Les couvertures oracle de la cible par les domaines, les conditionnements de V, le produit scalaire du vecteur retiré avec h et l’erreur cumulée sont conservés dans l’evidence. Ces mesures ne sont pas utilisées pour recalibrer les rayons. La trace et le conditionnement de V ne prouvent pas une bonne calibration de covariance. Le produit scalaire (A brut − A appliqué)·h est un diagnostic d’alignement, pas une décomposition exacte en signal honnête et byzantin retiré.\n")
    put("La masse des identités 0 et 1 et la norme de leur contribution correspondent aux poids de l’agrégat brut parent. Elles ne décrivent pas à elles seules leur influence finale via le prédicteur historique, et, pour la projection ellipsoïdale euclidienne, il n’existe pas un unique facteur radial gamma à leur appliquer. Sans attaque, ces deux identités sont honnêtes : ne pas appeler leur masse « masse byzantine ».\n")
    put("### Couverture observée et erreur temporelle\n")
    put("La cible oracle h est à l’intérieur des deux domaines, pour les deux prédicteurs, sur tous les tours 13–40 évalués. Aucun des six contrôles à seuil n’augmente la MSE euclidienne instantanée par rapport à son FAR brut au même modèle (tolérance numérique 10⁻⁸). Cela montre la cohérence numérique de ces corrections sur les données observées ; ce n’est ni une preuve de couverture universelle ni une amélioration démontrée de l’accuracy. La projection euclidienne et le radial restent deux opérateurs mathématiquement distincts.\n")
    put("L’erreur moyenne temporelle ci-dessous est ||(1/28) × somme des (A appliqué − h)||₂². Elle ne doit pas être confondue avec la moyenne des ||A appliqué − h||₂² : des fluctuations peuvent se compenser dans la première. Ce n’est pas non plus une estimation non biaisée du carré du biais en espérance. Les trajectoires et cibles diffèrent entre règles.\n")
    table(["Bruit / scénario", "FAR(RFA)", "Lissage EMA", "EMA isotrope", "RCIG isotrope"],[[NOISE[n]+' / '+SCENARIOS[c]]+[fmt(stats(r['time_mean_error_sq'] for r in e['records'] if (r['noise'],r['scenario'],r['arm'])==(n,c,a)),4) for a in ['far_rfa','ema_smooth','ema_iso','rcig_iso']] for n,c in itertools.product(NOISE,SCENARIOS)])
    put("Le lissage peut réduire la MSE instantanée d’environ 86 % tout en diminuant beaucoup moins cette erreur moyenne temporelle. La quantité retirée comporte donc beaucoup de fluctuations qui se compensaient déjà partiellement ; cette observation ne suffit pas à identifier les directions utiles à la classification.\n")
    put("## 9. Ce qui est défendable et décision\n")
    put("**Observations.** La calibration figée préserve ici le modèle propre. Plusieurs contrôles réduisent l’erreur appliquée pendant BF/ALIE. Cette amélioration ne se transmet pas de façon pertinente au modèle. IPM échappe aux contrôles durant l’attaque, tandis que la récupération déclenche des corrections ellipsoïdales. RCIG n’apporte pas de bénéfice end-to-end établi sur EMA.\n")
    put("**Résultat positif distinct :** FAR avec RFA n’est pas équivalent à RFA directe. Sans attaque en bruit hétéroscédastique, FAR atteint 57,08 ± 1,36 % d’accuracy et 22,82 ± 5,39 % de Worst-20, contre 52,39 ± 4,46 % et 14,07 ± 6,29 % pour RFA directe. Ce bénéfice comparatif de la repondération reste un résultat utile, mais ce n’est pas une contribution nouvelle de RCIG et il ne généralise pas à toutes les attaques/régimes. Il n’y a pas de témoin sans DP strictement apparié dans cette campagne : l’attribution causale au bruit n’est pas isolée.\n")
    put("**Inférences.** Une protection calibrée sur une augmentation de norme peut manquer une attaque qui réduit le résidu ou se fond progressivement dans la mémoire. Une partie de la réduction de MSE peut porter sur des directions peu importantes pour l’apprentissage. Ces explications nécessiteraient des interventions spécifiques pour devenir des conclusions causales.\n")
    put("**Non identifiable / non démontré.** Supériorité universelle de RCIG, impossibilité de la local-DP, bénéfice à horizon long, robustesse à un adversaire adaptatif optimal, comportement à α plus fort, covariance effective exacte, non-infériorité statistique propre. Le protocole parle d’un coût acceptable mais ne fixe pas une marge numérique de promotion de l’accuracy/Worst-20 ; nous ne transformons pas après coup les tolérances de récupération en marges de non-infériorité.\n")
    put("**Décision : ne pas promouvoir cette instanciation EMA/RCIG à seuil comme méthode validée et ne pas relancer automatiquement une grille.** Le résultat à présenter est la dissociation entre réduction de MSE, déclenchement et utilité, avec un cas explicite de correction retardée au retour honnête sous IPM. Avant toute nouvelle campagne, il faudrait choisir une hypothèse falsifiable unique, définir le coût honnête admissible et établir un effet utile sur la direction de l’agrégat ; aucune amélioration favorable n’est promise.\n")
    put("## 10. Différences appariées par seed\n")
    put("Différences de valeurs finales avec FAR inchangé, à même seed/bruit/scénario. L’ordre des trois nombres est 932001, 932002, 932003. L’evidence contient également les losses, balanced accuracy, métriques par fenêtre et intervalles descriptifs.\n")
    for n,c in itertools.product(NOISE,SCENARIOS):
        put(f"### {NOISE[n]} — {SCENARIOS[c]}\n")
        table(["Règle", "Δ accuracy (pp) par seed", "Δ Worst-20 (pp) par seed", "Δ gap (pp) par seed"],[[ARMS[a]]+[" / ".join(fmt(v,3) for v in G[n,c,a]['final_delta_vs_far'][k]['values']) for k in ['accuracy','worst20','gap']] for a in list(ARMS)[1:]])
    put("## 11. Traçabilité\n")
    put("- [Protocole préenregistré](Aggregate_Predictor_Calibration_Recovery_Protocol.md)\n- [Matrice figée](../../configs/ldp_gradient_far/aggregate_predictor_calibration_v1.yaml)\n- [Rayons et empreintes de calibration](../../results/ldp_gradient_far/aggregate_predictor_calibration_v1/frozen_radii.json)\n- [Fin de campagne](../../results/ldp_gradient_far/aggregate_predictor_calibration_v1/completion.json)\n- [Evidence : 216 résultats par seed, agrégations et manifeste des 254 runs](Aggregate_Predictor_EMA_RCIG_Final_Evidence.json)\n- [Analyse reproductible](../../scripts/analyze_aggregate_predictor_calibration.py)\n")
    REPORT.write_text("\n".join(lines),encoding="utf-8")
    # Check every local Markdown link. No browser-specific math delimiters used.
    import re
    for link in re.findall(r"\]\(([^)]+)\)", REPORT.read_text()):
        assert (REPORT.parent/link).resolve().exists(), link
    print(json.dumps({"report":str(REPORT),"evidence":str(EVIDENCE),"audit":e['audit'],
        "max_abs_group_mean_delta_accuracy_pp":maxacc,"max_abs_group_mean_delta_worst20_pp":maxworst},indent=2))


if __name__ == "__main__":
    matrix, evidence, payloads = analyze()
    report(matrix,evidence,payloads)
