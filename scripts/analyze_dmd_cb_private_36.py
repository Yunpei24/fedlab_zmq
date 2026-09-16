#!/usr/bin/env python3
"""Read-only analysis of the frozen 36-run DMD-CB screen; optional new reports.

No training, model loading, reevaluation, source mutation or result mutation.
--check validates and computes in memory. --write exclusively creates the two
named analysis artifacts and refuses to overwrite either existing artifact.
The only subprocess is the existing runner's read-only --status interface.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, pvariance, stdev
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/ldp_gradient_far/dmd_cb_private_v1"
MATRIX = ROOT / "configs/ldp_gradient_far/dmd_cb_private_v1.yaml"
PROTOCOL = ROOT / "output/analysis/DMD_CB_Private_Integration_Protocol.md"
REPORT = ROOT / "output/analysis/DMD_CB_Private_36_Results.md"
EVIDENCE = ROOT / "output/analysis/DMD_CB_Private_36_Evidence.json"
CLASS_EVIDENCE = ROOT / "output/analysis/DMD_CB_Private_36_Class_Evidence.json"
SEEDS = (24, 42, 72)
NOISES = ("homogeneous", "heteroscedastic")
MODES = ("uniform", "rfa", "far_rfa")
T95_DF2 = 4.302652729696142
SIGN_TOL = 1e-10
NOISE_LABEL = {"homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique"}
MODE_LABEL = {"uniform": "Uniforme", "rfa": "RFA directe", "far_rfa": "FAR(RFA)"}
# key -> (saved field, multiplier, label, unit, desirable direction, decimals)
METRICS = {
    "test_acc": ("test_accuracy", 100, "TestAcc", "%", "higher", 3),
    "client_acc": ("client_accuracy_mean", 100, "ClientAcc", "%", "higher", 3),
    "test_loss": ("test_loss", 1, "Test CE loss", "loss", "lower", 6),
    "client_loss": ("client_loss_mean", 1, "Client CE loss", "loss", "lower", 6),
    "worst20": ("worst20_accuracy_pct", 1, "Worst20", "%", "higher", 3),
    "best20": ("best20_accuracy_pct", 1, "Best20", "%", "higher", 3),
    "tail_gap": ("best20_worst20_gap_pct", 1, "Gap Best20−Worst20", "pp", "lower", 3),
    "range_gap": (None, 1, "Gap max−min clients", "pp", "lower", 3),
    "acc_variance": ("client_accuracy_variance_pct2", 1, "Variance accuracy", "pp²", "lower", 3),
    "loss_variance": ("client_loss_variance", 1, "Variance loss", "loss²", "lower", 6),
    "balanced_acc": ("mean_client_balanced_accuracy_pct", 1, "BA cliente moyenne", "%", "higher", 3),
    "balanced_worst20": ("worst20_balanced_accuracy_pct", 1, "Worst20 BA", "%", "higher", 3),
    "balanced_best20": ("best20_balanced_accuracy_pct", 1, "Best20 BA", "%", "higher", 3),
    "balanced_tail_gap": ("best_worst_balanced_accuracy_gap_pct", 1, "Gap Best20−Worst20 BA", "pp", "lower", 3),
    "balanced_range_gap": (None, 1, "Gap max−min BA", "pp", "lower", 3),
    "balanced_variance": ("client_balanced_accuracy_variance_pct2", 1, "Variance BA", "pp²", "lower", 3),
    "deficit_mean": ("canonical_cb_deficit_mean", 1, "Déficit CB moyen", "logit²", "lower", 6),
    "deficit_variance": ("canonical_cb_deficit_variance", 1, "Variance déficit CB", "logit⁴", "lower", 6),
    "deficit_upper_semivariance": ("canonical_cb_deficit_upper_semivariance", 1, "Semi-variance supérieure CB", "logit⁴", "lower", 6),
    "deficit_cvar20": ("canonical_cb_deficit_cvar20", 1, "CVaR20 déficit CB", "logit²", "lower", 6),
    "deficit_max": ("canonical_cb_deficit_max", 1, "Déficit CB maximum", "logit²", "lower", 6),
}
BLOCKS = (
    ("Accuracy et pertes d’évaluation", ("test_acc", "client_acc", "test_loss", "client_loss")),
    ("Équité entre clients — accuracy ordinaire", ("worst20", "best20", "tail_gap", "range_gap", "acc_variance", "loss_variance")),
    ("Balanced accuracy cliente — classes présentes", ("balanced_acc", "balanced_worst20", "balanced_best20", "balanced_tail_gap", "balanced_range_gap", "balanced_variance")),
    ("Déficit CB canonique d’évaluation", ("deficit_mean", "deficit_variance", "deficit_upper_semivariance", "deficit_cvar20", "deficit_max")),
)
VECTOR_KEYS = ("client_accuracy_values_oracle", "client_loss_values_oracle", "client_balanced_accuracy_values_oracle", "client_dmd_cb_values_oracle")
COEFFICIENT_FIELDS = ("min_client_weight", "max_client_weight", "weight_entropy", "effective_num_clients", "far_server_clip_rate", "far_weight_l2_squared")
PER_CLASS_KEY = re.compile(r"per_class|class_correct|class_total|class_support|confusion|present_class|class_recall|valid_class_mask")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Non-finite/non-numeric value: {value!r}")
    return float(value)


def close(actual, expected, label):
    require(math.isclose(finite(actual), finite(expected), rel_tol=1e-9, abs_tol=1e-8), f"Mismatch {label}: {actual} != {expected}")


def stats(values):
    values = [finite(x) for x in values]
    require(len(values) == 3, "Exactly three seed-level replicates required")
    avg, sd = mean(values), stdev(values)
    half_width = T95_DF2 * sd / math.sqrt(3)
    return {"n_seeds": 3, "mean": avg, "sd_sample": sd,
            "ci95_low": avg - half_width, "ci95_high": avg + half_width,
            "values_seed_order_24_42_72": values,
            "positive_seeds": sum(v > SIGN_TOL for v in values),
            "negative_seeds": sum(v < -SIGN_TOL for v in values),
            "zero_seeds": sum(abs(v) <= SIGN_TOL for v in values)}


def validate_vectors(row):
    require(row["num_evaluated_clients"] == 25, "Expected 25 evaluated clients")
    require(row["evaluated_client_ids_oracle"] == list(range(25)), "Client ID mismatch")
    require(row["num_excluded_byzantine_clients"] == 0, "Unexpected evaluation exclusion")
    vectors = {}
    for key in VECTOR_KEYS:
        require(len(row[key]) == 25, f"Wrong vector length: {key}")
        vectors[key] = [finite(x) for x in row[key]]
    for key, prefix in ((VECTOR_KEYS[0], ""), (VECTOR_KEYS[2], "balanced_")):
        values = sorted(vectors[key])
        require(all(0 <= v <= 1 for v in values), f"Accuracy range: {key}")
        pairs = (("client_acc", "acc_variance", "worst20", "best20", "tail_gap") if not prefix
                 else ("balanced_acc", "balanced_variance", "balanced_worst20", "balanced_best20", "balanced_tail_gap"))
        expected = (mean(values) * 100, pvariance(values) * 10000,
                    mean(values[:5]) * 100, mean(values[-5:]) * 100,
                    (mean(values[-5:]) - mean(values[:5])) * 100)
        for metric, value in zip(pairs, expected):
            field, scale, *_ = METRICS[metric]
            close(row[field] * scale, value, metric)
    close(row["test_accuracy"], row["client_accuracy_mean"], "TestAcc=ClientAcc")
    close(row["client_loss_mean"], mean(vectors[VECTOR_KEYS[1]]), "ClientLoss")
    close(row["client_loss_variance"], pvariance(vectors[VECTOR_KEYS[1]]), "ClientLoss variance")
    deficits = vectors[VECTOR_KEYS[3]]
    avg = mean(deficits)
    for field, value in {
        "canonical_cb_deficit_mean": avg,
        "canonical_cb_deficit_variance": pvariance(deficits),
        "canonical_cb_deficit_upper_semivariance": mean(max(x - avg, 0) ** 2 for x in deficits),
        "canonical_cb_deficit_cvar20": mean(sorted(deficits)[-5:]),
        "canonical_cb_deficit_max": max(deficits),
    }.items():
        close(row[field], value, field)


def extract(row):
    result = {key: finite(row[field]) * scale for key, (field, scale, *_) in METRICS.items() if field is not None}
    for key, vector in (("range_gap", VECTOR_KEYS[0]), ("balanced_range_gap", VECTOR_KEYS[2])):
        result[key] = 100 * (max(row[vector]) - min(row[vector]))
    return result


def class_analysis(fingerprints):
    """Consume separately authorized, validated post-hoc class results, if any."""
    if not CLASS_EVIDENCE.exists():
        return None
    data = read_json(CLASS_EVIDENCE)
    require(data["validated_final_checkpoints"] == 36 and len(data["records"]) == 36, "Incomplete class reevaluation")
    require(data["device"] == "mps" and data["mps_fallback"] == 0, "Class evaluation was not MPS-only")
    require(not any(data[k] for k in ("training_performed", "downloads_performed", "raw_results_modified")), "Invalid evaluation boundary")
    for key, error in data["max_reproduction_errors"].items():
        tolerance = data["validation_tolerances"]["loss" if "loss" in key else "accuracy_and_balanced_accuracy"]
        require(error <= tolerance, "Class reevaluation failed metric reproduction")
    for path, expected in data["input_and_evaluator_sha256"].items():
        require(digest(ROOT / path) == expected, f"Class reevaluation input changed: {path}")
    fingerprints[relative(CLASS_EVIDENCE)] = digest(CLASS_EVIDENCE)
    index = {(r["noise"], r["arm"], r["seed"]): r for r in data["records"]}
    require(len(index) == 36, "Duplicate class evaluation record")
    cells = []
    for noise in NOISES:
        for mode in MODES:
            for class_id, name in zip(data["class_ids"], data["class_names"]):
                ce = [100 * index[noise, "ce_" + mode, s]["global"]["recall"][class_id] for s in SEEDS]
                dmd = [100 * index[noise, "dmd_" + mode, s]["global"]["recall"][class_id] for s in SEEDS]
                cells.append({"noise": noise, "mode": mode, "class_id": class_id,
                              "class_name": name, "ce": stats(ce), "dmd": stats(dmd),
                              "delta_dmd_minus_ce": stats([d - c for d, c in zip(dmd, ce)])})
    partition_summaries = []
    for seed, partition in data["partitions"].items():
        counts = [sum(mask) for mask in partition["class_present_mask_by_client"]]
        partition_summaries.append({"seed": int(seed), "present_classes_per_client": counts,
                                   "min_present_classes": min(counts), "max_present_classes": max(counts),
                                   "mean_present_classes": mean(counts),
                                   "clients_with_class": [sum(mask[k] for mask in partition["class_present_mask_by_client"]) for k in range(10)]})
    grand = []
    for class_id, name in zip(data["class_ids"], data["class_names"]):
        values = [mean(100 * (index[n, "dmd_" + m, s]["global"]["recall"][class_id] - index[n, "ce_" + m, s]["global"]["recall"][class_id]) for n in NOISES for m in MODES) for s in SEEDS]
        grand.append({"class_id": class_id, "class_name": name, "delta_dmd_minus_ce": stats(values)})
    return {"evidence_path": relative(CLASS_EVIDENCE), "evidence_sha256": digest(CLASS_EVIDENCE),
            "scientific_status": data["scientific_status"], "validated_final_checkpoints": 36,
            "max_reproduction_errors": data["max_reproduction_errors"],
            "validation_tolerances": data["validation_tolerances"],
            "test_class_support": data["test_class_support"], "class_ids": data["class_ids"],
            "class_names": data["class_names"], "partition_support_summaries": partition_summaries,
            "class_recall_grand_descriptive_seed_aggregated": grand,
            "global_class_recall_levels_and_paired_differences_pct": cells}


def build_evidence():
    command = [sys.executable, "-B", str(ROOT / "scripts/run_dmd_cb_private_screen.py"), "--status"]
    proc = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, check=True)
    status = json.loads(proc.stdout)
    require(status["complete_valid"] == status["expected"] == 36 and not any(status[k] for k in ("active", "invalid", "missing")), "Campaign is not complete and valid")
    lock_path = CAMPAIGN / "scientific_lock.json"
    lock = read_json(lock_path)
    fingerprints = {relative(p): digest(p) for p in (MATRIX, PROTOCOL, lock_path, Path(__file__), ROOT / "metrics/client_fairness.py")}
    require(fingerprints["metrics/client_fairness.py"] == lock["provenance"]["source_sha256"]["metrics/client_fairness.py"], "Evaluator source changed since execution")
    records, lookup, available_keys, class_keys = [], {}, set(), set()
    train_placeholders, local_clip_values, all_coefficients = {}, set(), {}
    paths = sorted((CAMPAIGN / "runs").rglob("metrics.json"))
    require(len(paths) == 36, "Expected exactly 36 metrics artifacts")
    for path in paths:
        noise, seed_text, arm = path.relative_to(CAMPAIGN / "runs").parts[0].split("__")
        seed, objective, mode = int(seed_text[4:]), *arm.split("_", 1)
        require(noise in NOISES and seed in SEEDS and objective in ("ce", "dmd") and mode in MODES, "Unexpected run identity")
        payload = read_json(path)
        config, rounds = payload["config"], payload["rounds"]
        require([r["round_num"] for r in rounds] == list(range(1, 41)), "Round sequence is not 1..40")
        require(config["dmd_pairing_seed"] == seed, "Seed mismatch")
        require(config["fixed_batch_size"] == 240 and config["fixed_steps_per_round"] == 1, "Batch/work mismatch")
        require(config["dmd_mu"] == (0.0 if objective == "ce" else 0.1875), "Objective mismatch")
        fingerprints[relative(path)] = digest(path)
        coefficient_key = f"{noise}/{arm}"
        all_coefficients.setdefault(coefficient_key, {field: [] for field in COEFFICIENT_FIELDS})
        coefficient_kinds = set()
        for row in rounds:
            validate_vectors(row)
            available_keys.update(row)
            class_keys.update(k for k in row if PER_CLASS_KEY.search(k))
            coefficient_kinds.add(row["dmd_server_coefficient_kind"])
            for field in ("train_loss", "avg_local_loss"):
                train_placeholders.setdefault(field, set()).add(row.get(field))
            local_clip_values.add(row.get("privacy_clip_rate_mean"))
            for field in COEFFICIENT_FIELDS:
                all_coefficients[coefficient_key][field].append(finite(row[field]))
        row = rounds[-1]
        values = extract(row)
        identity = noise, mode, objective, seed
        require(identity not in lookup, "Duplicate run")
        lookup[identity] = values
        records.append({"noise": noise, "mode": mode, "objective": objective, "seed": seed,
                        "arm": arm, "round_num": 40, "metrics_path": relative(path),
                        "metrics_sha256": fingerprints[relative(path)], "metrics": values,
                        "saved_client_vectors": {k: row[k] for k in ("evaluated_client_ids_oracle", *VECTOR_KEYS)},
                        "privacy_round40": {k: v for k, v in row.items() if k.startswith("privacy_")},
                        "coefficient_kind": sorted(coefficient_kinds),
                        "server_round40": {k: row[k] for k in COEFFICIENT_FIELDS}})
    expected = {(n, m, o, s) for n in NOISES for m in MODES for o in ("ce", "dmd") for s in SEEDS}
    require(set(lookup) == expected, "Design incomplete")
    groups, contrasts, aggregation_contrasts = [], [], []
    for noise in NOISES:
        for mode in MODES:
            for objective in ("ce", "dmd"):
                groups.append({"noise": noise, "mode": mode, "objective": objective,
                               "metrics": {k: stats([lookup[noise, mode, objective, s][k] for s in SEEDS]) for k in METRICS}})
            contrasts.append({"noise": noise, "mode": mode, "contrast": "DMD-CE",
                              "metrics": {k: stats([lookup[noise, mode, "dmd", s][k] - lookup[noise, mode, "ce", s][k] for s in SEEDS]) for k in METRICS}})
        for objective in ("ce", "dmd"):
            aggregation_contrasts.append({"noise": noise, "objective": objective, "contrast": "FAR(RFA)-RFA_direct",
                                          "metrics": {k: stats([lookup[noise, "far_rfa", objective, s][k] - lookup[noise, "rfa", objective, s][k] for s in SEEDS]) for k in METRICS}})
    global_descriptive = {}
    for label, noises in (("all_six_conditions", NOISES), ("homogeneous", ("homogeneous",)), ("heteroscedastic", ("heteroscedastic",))):
        global_descriptive[label] = {k: stats([mean(lookup[n, m, "dmd", s][k] - lookup[n, m, "ce", s][k] for n in noises for m in MODES) for s in SEEDS]) for k in METRICS}
    coefficient_ranges = {key: {field: {"min": min(v), "max": max(v), "n_run_rounds": len(v)} for field, v in fields.items()} for key, fields in all_coefficients.items()}
    classes = class_analysis(fingerprints)
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": "dmd_cb_private_v1", "endpoint": "round40_fixed_not_peak_selected",
            "validation": status, "validation_command": "python3 -B scripts/run_dmd_cb_private_screen.py --status",
            "scientific_hash": lock["scientific_hash"], "input_and_analysis_sha256": fingerprints,
            "metric_definitions": {k: {"saved_field": f, "multiplier": s, "label": l, "unit": u, "better": b} for k, (f, s, l, u, b, _) in METRICS.items()},
            "statistics": {"seed_order": list(SEEDS), "sd_ddof": 1, "client_population_variance_ddof": 0,
                           "ci95": "Student t, df=2, mean ± t*sample_SD/sqrt(3)", "t95_df2": T95_DF2,
                           "multiplicity_correction": "none; exploratory marginal CIs", "sign_tolerance": SIGN_TOL,
                           "unit_of_replication": "seed, not client/round/aggregation",
                           "global_method": "mean contrasts over conditions within each seed, then stats on 3 seed averages"},
            "audit": {"checked_rounds": 1440, "checked_client_evaluations": 36000,
                      "train_placeholders_observed": {k: sorted(v) for k, v in train_placeholders.items()},
                      "private_local_clipping_rate_saved_values": list(local_clip_values),
                      "saved_round_metric_keys": sorted(available_keys), "per_class_metric_keys_found": sorted(class_keys),
                      "per_class_identifiable_from_saved_metrics": bool(class_keys),
                      "class_support_masks_saved": False,
                      "balanced_accuracy_support": "macro recall across classes with positive test support within each client; absent classes omitted",
                      "historical_class_support_masks_saved": False,
                      "posthoc_class_reevaluation_included": classes is not None,
                      "model_checkpoints_loaded_by_primary_analyzer": False,
                      "models_reevaluated_in_separate_authorized_stage": classes is not None,
                      "source_or_result_files_modified": False},
            "records": records, "groups": groups, "paired_dmd_minus_ce": contrasts,
            "paired_far_minus_direct_rfa": aggregation_contrasts, "global_descriptive": global_descriptive,
            "server_coefficient_ranges_all_rounds": coefficient_ranges,
            "class_reevaluation": classes,
            "interpretation_limits": ["36 completed clean exploratory runs; no attack robustness test",
                                      "CE control also pays epsilon_hist=0.25; gradient epsilon=3.75 for both objectives",
                                      "CE using epsilon=4 entirely for gradients is absent from this frozen cohort",
                                      "published benchmark seeds are for research reproducibility, not secret deployment DP randomness",
                                      "privacy guarantee per client/example/per run; not joint release of noise-paired models",
                                      "evaluation metrics are research diagnostics, not privatized confidential evaluation releases"]}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def fmt(value, metric, signed=False):
    decimals = METRICS[metric][5]
    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return format(value, f"{'+' if signed else ''}.{decimals}f")


def avg_sd(s, metric, signed=False):
    return f"{fmt(s['mean'], metric, signed)} ± {fmt(s['sd_sample'], metric)}"


def ci(s, metric):
    return f"[{fmt(s['ci95_low'], metric, True)} ; {fmt(s['ci95_high'], metric, True)}]"


def paired_row(item, key):
    s = item["metrics"][key]
    return [NOISE_LABEL[item["noise"]], MODE_LABEL[item["mode"]], METRICS[key][2],
            avg_sd(s, key, True), ci(s, key),
            "; ".join(fmt(v, key, True) for v in s["values_seed_order_24_42_72"])]


def render(e):
    groups = {(g["noise"], g["mode"], g["objective"]): g["metrics"] for g in e["groups"]}
    classes = e.get("class_reevaluation")
    class_note = ("Une **réévaluation post-hoc explicitement autorisée** des 36 checkpoints finaux sur le seul test public a depuis reconstruit ces informations. Elle est séparée des métriques historiques : section 3 ci-dessous et fichier `DMD_CB_Private_36_Class_Evidence.json`. Aucun entraînement ni téléchargement n’a été effectué et les résultats originaux restent inchangés."
                  if classes else "Aucun checkpoint n’a été chargé et aucun modèle n’a été réévalué pour ce rapport. On ne peut pas attribuer le gain à une classe rare particulière ni vérifier rétrospectivement ses effectifs depuis ces seuls JSON.")
    lines = ["# DMD-CB privé — rapport final des 36 runs",
             f"Généré le {e['created_at_utc']}. Analyse exclusive de `dmd_cb_private_v1`, endpoint fixé au round 40.",
             "## 1. Décision et périmètre",
             "**Campagne terminée : 36/36 runs valides, aucun actif, invalide ou manquant.** DMD-CB donne un petit gain d’accuracy cohérent et un signal d’équité plus net sous bruit hétéroscédastique. Cela justifie une vérification ciblée, pas une promotion définitive ni une revendication de robustesse byzantine.",
             "Le contrôle CE présent paie lui aussi l’histogramme privé (ε=0,25) et ne consacre que ε=3,75 aux gradients. La prochaine comparaison nécessaire est **CE avec ε=4 entièrement consacré aux gradients**, sans histogramme. Ce contrôle n’appartient pas aux 36 runs analysés ici. Toute confirmation ou attaque ultérieure doit être identifiée comme une nouvelle campagne ; rien de cela n’est lancé par cet analyseur.",
             "Différences DMD−CE en points de pourcentage ; moyenne ± écart-type échantillonnal des trois différences appariées :"]
    primary = []
    for c in e["paired_dmd_minus_ce"]:
        a, w = c["metrics"]["test_acc"], c["metrics"]["worst20"]
        primary.append([NOISE_LABEL[c["noise"]], MODE_LABEL[c["mode"]], avg_sd(a, "test_acc", True), ci(a, "test_acc"), avg_sd(w, "worst20", True), ci(w, "worst20")])
    lines.append(table(["Bruit", "Agrégation", "Δ TestAcc = Δ ClientAcc", "IC95", "Δ Worst20", "IC95"], primary))
    counts = {}
    for label, subset in (("toutes", e["paired_dmd_minus_ce"]), ("hétéroscédastiques", [c for c in e["paired_dmd_minus_ce"] if c["noise"] == "heteroscedastic"])):
        counts[label] = {k: sum(c["metrics"][k]["positive_seeds"] for c in subset) for k in ("test_acc", "worst20")}
    lines.extend([f"L’accuracy augmente dans {counts['toutes']['test_acc']}/18 paires ; le Worst20 augmente dans {counts['hétéroscédastiques']['worst20']}/9 paires hétéroscédastiques. Ce sont des signes descriptifs : les agrégations réutilisent les mêmes graines, **pas 18 réplications indépendantes**.",
                  "## 2. Validation, protocole et confidentialité",
                  "La commande de contrôle existante a été exécutée en lecture seule : `python3 -B scripts/run_dmd_cb_private_screen.py --status`. Elle vérifie identités, configurations gelées, comptabilité, MPS, provenance des imports, états completed et empreintes des résultats. L’analyseur recalcule en plus les résumés de 1 440 rounds depuis 36 000 évaluations clientes enregistrées ; chaque round comprend les clients 0–24, sans exclusion.",
                  table(["Paramètre", "Valeur"], [["Données / modèle", "Fashion-MNIST / LeNet-5 tanh"], ["Partition", "Dirichlet équilibrée en taille ; α=0,1"], ["Clients / participation", "25 / 25 à chaque round"], ["Travail", "B=240, T=40, un gradient privé par client et par round ; taille locale publique 2 400"], ["Seeds", "24, 42, 72"], ["Objectif", "CE : μ=0 ; DMD-CB : μ=0,1875"], ["Serveur", "Uniforme ; RFA directe ; FAR avec référence RFA, α=0,1"], ["Clipping / pas serveur", "Gradient individuel combiné C=4 ; clipping serveur U=16 ; pas 0,2"], ["Calcul privé", "MPS, fallback désactivé"], ["Bruit", "Homogène ×1 ; hétéroscédastique : 13 clients ×1, 12 clients ×2"], ["Confidentialité", "ε_hist=0,25 une fois + ε_grad≤3,75 ; ε_total≤4, δ=10⁻⁵, replace-one"], ["Attaques", "Aucune"]]),
                  "La garantie est au niveau de l’exemple chez chaque client, pour un run ; elle ne couvre pas la publication conjointe de modèles appariés avec bruit réutilisé. Les seeds publiques rendent ces benchmarks reproductibles : **elles ne constituent pas un aléa secret approprié à un déploiement DP opérationnel**. Les métriques de test et vecteurs clients sont des diagnostics de recherche non privatisés ; aucun budget de release sur de vrais jeux d’évaluation confidentiels n’est revendiqué."])
    privacy_rows = []
    for noise in NOISES:
        r = next(r for r in e["records"] if r["noise"] == noise)
        p = r["privacy_round40"]
        privacy_rows.append([NOISE_LABEL[noise], f"{p['privacy_model_noise_multiplier_min']:.9f}", f"{p['privacy_model_noise_multiplier_max']:.9f}", f"{p['privacy_gradient_epsilon_max']:.9f}", f"{p['privacy_epsilon_max']:.9f}", f"{p['privacy_epsilon_mean']:.9f}"])
    lines.append(table(["Bruit", "σ min", "σ max", "ε gradient max", "ε total max", "ε total moyen"], privacy_rows))
    lines.extend(["## 3. Définitions et méthode statistique",
                  "Endpoint : round 40, jamais le meilleur round sélectionné a posteriori. Les niveaux sont des moyennes ± SD **échantillonnale entre trois graines** (ddof=1). Pour chaque bruit et agrégation, Δ_s = DMD_s − CE_s ; IC95 = moyenne(Δ) ± 4,302652729696142 × SD(Δ)/√3 (Student, df=2). Les IC sont marginaux, non corrigés pour les nombreuses comparaisons ; ce pilote reste exploratoire. Les moyennes globales sont calculées sur les conditions **à l’intérieur de chaque graine**, puis sur les trois moyennes de graine.",
                  "Les variances entre clients enregistrées utilisent ddof=0 : ce sont les dispersions descriptives des 25 clients évalués, différentes de la SD inter-graines. Accuracy et BA sont affichées en %, différences en pp, variances d’accuracy en pp². Les pertes restent en unités de loss.",
                  "Le champ historique `balanced_performance_fairness` désigne une variance pondérée des losses clientes ; **ce n’est pas une balanced accuracy**. Les tailles locales étant égales ici, cette pondération est uniforme. La variance des losses est conservée dans les tableaux ci-dessous.",
                  "**TestAcc et ClientAcc sont numériquement identiques ici**, vérifié à tous les rounds ; ce ne sont pas deux résultats indépendants. Test loss et client loss sont des cross-entropies d’évaluation du modèle commun, pas la perte d’entraînement CE+DMD. `train_loss=0` et `avg_local_loss=0` sont des placeholders techniques ; aucune vraie train accuracy/loss n’a été sauvegardée. `privacy_clip_rate_mean` est nul au sens valeur manquante (`null`), pas un taux de clipping individuel égal à zéro.",
                  "Worst20 et Best20 sont les moyennes des 5 pires/meilleures accuracies parmi 25 clients, triées séparément dans chaque run. Le gap principal `best20_worst20_gap_pct` est Best20−Worst20. Le gap **max−min** est dérivé séparément du vecteur des 25 accuracies. Les clients d’une queue peuvent changer entre bras : Δ Worst20 compare deux fonctionnelles de distribution, pas nécessairement le même sous-ensemble de cinq clients.",
                  "### Balanced accuracy et limites par classe",
                  "L’évaluateur `metrics/client_fairness.py` définit la BA d’un client comme la moyenne des rappels des classes **présentes dans son jeu de test**. Une classe absente n’est pas imputée à zéro et n’est pas incluse au dénominateur. La moyenne de ces 25 BA n’est donc pas la BA globale à support fixe des dix classes. Les supports et masques de présence peuvent différer entre clients/graines ; la même partition est réutilisée au sein des comparaisons appariées.",
                  "Les 25 valeurs de BA et les identités clientes sont sauvegardées, mais **les rappels par classe, effectifs par classe, masques de support et matrices de confusion ne le sont pas dans les métriques historiques**. Les détails par classe et une BA globale à dix classes sont non identifiables à partir de ces seuls agrégats. " + class_note,
                  "Attention au nom du champ `best_worst_balanced_accuracy_gap_pct` : dans l’implémentation gelée, il représente **Best20−Worst20 BA**, non max−min. Les deux sont distingués ci-dessous.",
                  "Le déficit CB canonique d’évaluation vaut, pour chaque client, la moyenne sur ses classes de test présentes de la moyenne de ½[max(0, meilleur logit concurrent − logit vrai)]². Il utilise le support d’évaluation, **pas les poids d’histogramme DP utilisés pour l’apprentissage**. Sa CVaR20 est la moyenne des cinq déficits clients les plus élevés ; la semi-variance supérieure utilise le seuil de moyenne cliente. Ce sont des diagnostics d’évaluation, non une mesure de train loss ni une démonstration causale du mécanisme DMD.",
                  ])
    if classes:
        lines.extend(["### Réévaluation post-hoc validée — rappels et supports par classe",
                      "Les partitions sont reconstruites à partir de chaque configuration/manifeste et de la graine 24/42/72 avec les sources gelées, sans lire les données d’entraînement. Même Fashion-MNIST test, ToTensor puis normalisation (0,2860 ; 0,3530), ordre des indices et batchs d’évaluation de 256. L’environnement PyTorch 2.11.0 et MPS reproduit les 36 checkpoints finaux. Les tenseurs d’entrée test transformés sont simplement mis en cache sur CPU ; les forwards sont exécutés sur MPS, sans gradients.",
                      table(["Contrôle de reproduction sur 36 checkpoints", "Erreur absolue maximale", "Tolérance"], [[k, f"{v:.17g}", f"{classes['validation_tolerances']['loss' if 'loss' in k else 'accuracy_and_balanced_accuracy']:.2g}"] for k, v in classes["max_reproduction_errors"].items()]),
                      "Les accuracies et losses globales et clientes sont reproduites exactement ; les BA clientes diffèrent au plus à la précision machine. Les nouveaux rappels, effectifs et matrices de confusion sont donc acceptés comme **résultats de cette réévaluation**, et non comme sorties historiquement enregistrées.",
                      "Le test global comporte exactement 1 000 exemples pour chacune des dix classes ; sa BA à support fixe des dix classes est ici égale à TestAcc. Chaque client a 400 exemples de test, mais pas nécessairement toutes les classes. Un rappel de classe absente est `null` avec masque faux, jamais un zéro imputé. Les indices de test, supports, masques, corrects, rappels et confusions de chacun des 25 clients sont conservés dans le JSON de classe."])
        lines.append(table(["Seed", "Classes présentes / client : min–max", "Moyenne", "Nombre de clients possédant chaque classe 0…9"], [[p["seed"], f"{p['min_present_classes']}–{p['max_present_classes']}", f"{p['mean_present_classes']:.2f}", "; ".join(map(str, p["clients_with_class"]))] for p in classes["partition_support_summaries"]]))
        lines.append("Rappels par classe **sur le test global** (1 000 exemples/classe/run). Moyenne ± SD sur trois graines et IC95 apparié DMD−CE, sans correction de multiplicité ; les lignes ne sont pas des expériences indépendantes. Ces différences n’établissent pas qu’une classe soit rare à l’échelle globale du benchmark, qui est équilibré.")
        class_rows = []
        for cell in classes["global_class_recall_levels_and_paired_differences_pct"]:
            a, b, delta = cell["ce"], cell["dmd"], cell["delta_dmd_minus_ce"]
            ca, cb = avg_sd(a, "test_acc"), avg_sd(b, "test_acc")
            if a["mean"] >= b["mean"] - SIGN_TOL:
                ca = f"**{ca}**"
            if b["mean"] >= a["mean"] - SIGN_TOL:
                cb = f"**{cb}**"
            class_rows.append([NOISE_LABEL[cell["noise"]], MODE_LABEL[cell["mode"]], f"{cell['class_id']} {cell['class_name']}", ca, cb, avg_sd(delta, "test_acc", True), ci(delta, "test_acc"), "; ".join(fmt(v, "test_acc", True) for v in delta["values_seed_order_24_42_72"])])
        lines.append(table(["Bruit", "Agrégation", "Classe", "CE recall %", "DMD recall %", "Δ pp ± SD", "IC95 Δ", "Δ seeds 24 ; 42 ; 72"], class_rows))
        lines.append("Synthèse descriptive par classe : moyenne sur les six conditions **au sein de chaque seed**, puis moyenne/SD/IC sur les trois seeds. **Le gain global n’est pas une amélioration uniforme des classes** : les augmentations de rappel de Coat, Bag ou Dress coexistent notamment avec une baisse moyenne de Pullover et Sandal. Sandal baisse dans chacune des trois moyennes de seed ; cela reste un constat de pilote, non un test confirmatoire corrigé pour multiplicité.")
        lines.append(table(["Classe", "Δ rappel moyen ± SD (pp)", "IC95", "Δ moyen seeds 24 ; 42 ; 72"], [[f"{r['class_id']} {r['class_name']}", avg_sd(r["delta_dmd_minus_ce"], "test_acc", True), ci(r["delta_dmd_minus_ce"], "test_acc"), "; ".join(fmt(v, "test_acc", True) for v in r["delta_dmd_minus_ce"]["values_seed_order_24_42_72"])] for r in classes["class_recall_grand_descriptive_seed_aggregated"]]))
    lines.extend(["## 4. Niveaux finaux et contrastes appariés complets",
                  "Le gras désigne le meilleur niveau moyen dans chaque paire CE/DMD selon le sens de la métrique ; il n’affirme pas une significativité. Les vecteurs Δ sont toujours dans l’ordre **24 ; 42 ; 72**."])
    for title, keys in BLOCKS:
        lines.append(f"### {title}")
        for key in keys:
            level_rows = []
            for noise in NOISES:
                for mode in MODES:
                    a, b = groups[noise, mode, "ce"][key], groups[noise, mode, "dmd"][key]
                    ce_text, dmd_text = avg_sd(a, key), avg_sd(b, key)
                    lower = METRICS[key][4] == "lower"
                    if abs(a["mean"] - b["mean"]) <= SIGN_TOL:
                        ce_text, dmd_text = f"**{ce_text}**", f"**{dmd_text}**"
                    elif (a["mean"] < b["mean"]) == lower:
                        ce_text = f"**{ce_text}**"
                    else:
                        dmd_text = f"**{dmd_text}**"
                    level_rows.append([NOISE_LABEL[noise], MODE_LABEL[mode], ce_text, dmd_text])
            lines.append(f"#### {METRICS[key][2]} ({METRICS[key][3]})")
            lines.append(table(["Bruit", "Agrégation", "CE : moyenne ± SD", "DMD : moyenne ± SD"], level_rows))
            lines.append(table(["Bruit", "Agrégation", "Métrique", "Δ moyenne ± SD", "IC95 Δ", "Δ seeds 24 ; 42 ; 72"], [paired_row(c, key) for c in e["paired_dmd_minus_ce"]]))
    lines.extend(["## 5. Agrégation et résumés globaux descriptifs",
                  "RFA directe agrège directement les messages par médiane géométrique ; FAR(RFA) utilise RFA comme référence pour pondérer les messages. Les modes ne sont pas interchangeables. Contrastes appariés FAR(RFA)−RFA directe :"])
    lines.append(table(["Bruit", "Objectif", "Métrique", "Δ moyenne ± SD", "IC95", "Δ seeds 24 ; 42 ; 72"], [[NOISE_LABEL[c["noise"]], c["objective"].upper(), METRICS[k][2], avg_sd(c["metrics"][k], k, True), ci(c["metrics"][k], k), "; ".join(fmt(v, k, True) for v in c["metrics"][k]["values_seed_order_24_42_72"])] for c in e["paired_far_minus_direct_rfa"] for k in ("test_acc", "worst20", "balanced_acc", "test_loss")]))
    lines.append("Moyennes descriptives équilibrées DMD−CE, avec agrégation préalable par graine ; elles ne remplacent pas les six conditions individuelles :")
    lines.append(table(["Conditions", "Métrique", "Δ moyenne ± SD", "IC95", "Moyennes Δ par graine"], [[label, METRICS[k][2], avg_sd(s[k], k, True), ci(s[k], k), "; ".join(fmt(v, k, True) for v in s[k]["values_seed_order_24_42_72"])] for label, s in e["global_descriptive"].items() for k in ("test_acc", "worst20", "tail_gap", "acc_variance", "balanced_acc", "balanced_worst20", "test_loss")]))
    lines.extend(["## 6. Poids serveur et clipping effectivement enregistrés",
                  "Les plages ci-dessous sont descriptives sur les rounds, **pas des répétitions statistiques**. Uniforme emploie des coefficients 1/25 ; RFA directe utilise des coefficients IRLS, FAR(RFA) ses poids FAR. Les vecteurs complets de coefficients ne sont pas sauvegardés dans ces métriques ; seules leurs statistiques et leur nature sont rapportées."])
    coeff_rows = []
    for noise in NOISES:
        for mode in MODES:
            for objective in ("ce", "dmd"):
                arm = f"{objective}_{mode}"
                ranges = e["server_coefficient_ranges_all_rounds"][f"{noise}/{arm}"]
                r = next(r for r in e["records"] if r["noise"] == noise and r["arm"] == arm)
                span = lambda k: f"{ranges[k]['min']:.6g}–{ranges[k]['max']:.6g}"
                coeff_rows.append([NOISE_LABEL[noise], arm, ", ".join(r["coefficient_kind"]), span("min_client_weight"), span("max_client_weight"), span("effective_num_clients"), span("far_server_clip_rate")])
    lines.append(table(["Bruit", "Bras", "Type coefficient", "Poids min : plage", "Poids max : plage", "N effectif : plage", "Clipping serveur : plage"], coeff_rows))
    lines.extend(["Le clipping serveur est un post-traitement ; ne pas le confondre avec le clipping des gradients individuels avant DP, dont le taux n’est pas enregistré. Les valeurs énergétiques et temps simulés ne sont pas une mesure de coût comparatif complet de DMD : ce rapport ne revendique aucun gain énergétique.",
                  "## 7. Lignes individuelles au round 40",
                  "Les valeurs non arrondies, les SD/IC et les vecteurs des 25 clients figurent également dans le JSON de preuve. Les tableaux suivants permettent de contrôler chaque run, sans sélectionner les seules graines favorables."])
    for title, keys in BLOCKS:
        lines.append(f"### {title} — 36 runs")
        rows = [[NOISE_LABEL[r["noise"]], r["arm"], str(r["seed"]), *(fmt(r["metrics"][k], k) for k in keys)] for r in sorted(e["records"], key=lambda r: (NOISES.index(r["noise"]), MODES.index(r["mode"]), r["objective"], r["seed"]))]
        lines.append(table(["Bruit", "Bras", "Seed", *(f"{METRICS[k][2]} ({METRICS[k][3]})" for k in keys)], rows))
    lines.extend(["## 8. Lecture méthodologique et suite prioritaire",
                  "1. **Gain d’accuracy faible mais cohérent** : l’effet est de quelques dixièmes de pp. La faible SD des différences appariées peut donner un IC positif malgré une amplitude pratique petite ; ne pas confondre cohérence du signe et importance pratique.",
                  "2. **Équité hétéroscédastique prometteuse, compromis entre classes** : le Worst20 ordinaire et la variance entre clients s’améliorent plus régulièrement qu’en homogène. Les métriques BA et déficit testent des aspects complémentaires. Les rappels post-hoc montrent des transferts de performance entre classes, notamment Coat/Bag en hausse et Pullover/Sandal en baisse en moyenne ; ils ne justifient ni une amélioration de toutes les classes ni une attribution causale.",
                  "3. **Contrôle de budget indispensable** : CE actuel consomme 0,25 sur un histogramme inutilisé dans son gradient. Comparer DMD (0,25+3,75) à CE sans histogramme (0+4), aux mêmes B/T/graines/partitions et agrégations. Cette nouvelle référence peut absorber le petit avantage observé ; les 36 runs actuels ne répondent pas à cette objection.",
                  "4. **Confirmation ensuite seulement** : geler les endpoints et une tolérance de perte d’accuracy scientifiquement justifiée avant de consulter de nouvelles graines. La sélection d’un régime/agrégateur sur ce pilote exige des graines de confirmation distinctes. Ne pas traiter les 25 clients, 40 rounds ou 3 agrégations comme des graines supplémentaires.",
                  "5. **Attaques dans une campagne distincte conditionnelle** : DMD change l’objectif local et ne constitue pas un détecteur byzantin. Si le contrôle CE plein budget laisse un signal utile, tester ensuite les attaques avec appariement, population d’évaluation explicitée et provenance distincte ; aucune robustesse n’est acquise ici.",
                  "6. **Traçabilité des résultats par classe** : prévoir en amont rappels/confusions, effectifs et masques de support sur le benchmark public, en conservant l’interprétation classes présentes versus support fixe. La réévaluation des checkpoints est une étape séparée explicitement autorisée et documentée ; ses mesures ne doivent pas être décrites comme historiquement enregistrées.",
                  "## 9. Reproduction et fichiers",
                  "`python3 -B scripts/analyze_dmd_cb_private_36.py --check` : revalide le statut et recalcule tout en mémoire, sans modifier les runs ni les rapports. `--write` crée exclusivement les deux rapports ci-dessous et refuse tout écrasement. Aucun appel d’entraînement ou de réévaluation n’existe dans le script.",
                  "- Analyseur : `scripts/analyze_dmd_cb_private_36.py`\n- Rapport : `output/analysis/DMD_CB_Private_36_Results.md`\n- Preuves : `output/analysis/DMD_CB_Private_36_Evidence.json`\n- Réévaluateur distinct : `scripts/reevaluate_dmd_cb_private_classes.py`\n- Preuves de classes : `output/analysis/DMD_CB_Private_36_Class_Evidence.json`\n- Entrées : `results/ldp_gradient_far/dmd_cb_private_v1/runs/**/metrics.json`\n- Protocole historique : `output/analysis/DMD_CB_Private_Integration_Protocol.md`",
                  f"Empreinte scientifique de la campagne : `{e['scientific_hash']}`. Le JSON contient les SHA-256 des 36 métriques, de la matrice, du protocole, de l’évaluateur et du script d’analyse. Les fichiers de résultats, sources gelées et protocole historique n’ont pas été modifiés."])
    return "\n\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Validate and compute without writing (default)")
    group.add_argument("--write", action="store_true", help="Create the two new report artifacts exclusively")
    args = parser.parse_args()
    if args.write:
        require(not REPORT.exists() and not EVIDENCE.exists(), "Refusing to overwrite an existing analysis artifact; use --check")
    check = stats([1, 2, 3])
    close(check["mean"], 2, "stats self-test mean")
    close(check["sd_sample"], 1, "stats self-test sample SD")
    close(check["ci95_high"], 2 + T95_DF2 / math.sqrt(3), "stats self-test CI")
    evidence = build_evidence()
    report = render(evidence)
    require(len(evidence["groups"]) == 12 and len(evidence["paired_dmd_minus_ce"]) == 6, "Summary shape mismatch")
    require(not evidence["audit"]["per_class_metric_keys_found"], "New per-class metrics found: inspect before retaining non-identifiability language")
    # Confirm inputs stayed unchanged while constructing the report.
    for rel, saved in evidence["input_and_analysis_sha256"].items():
        require(digest(ROOT / rel) == saved, f"Input changed during analysis: {rel}")
    if args.write:
        with EVIDENCE.open("x") as handle:
            json.dump(evidence, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with REPORT.open("x") as handle:
            handle.write(report)
    print(json.dumps({"validated": evidence["validation"], "checked_rounds": 1440,
                      "per_class_metrics_saved": False, "balanced_accuracy_saved": True,
                      "wrote_new_artifacts": [relative(REPORT), relative(EVIDENCE)] if args.write else [],
                      "global_descriptive": {k: evidence["global_descriptive"]["all_six_conditions"][k] for k in ("test_acc", "worst20", "balanced_acc", "balanced_worst20")}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
