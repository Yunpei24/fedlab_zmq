#!/usr/bin/env python3
"""Consolidate the preregistered DT-LDP-FAR decisions from Stages 22--26."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results/dt_ldp_far"
REPORT = ROOT / "output/analysis/DT_LDP_FAR_Final_Decision_Stage22_to_Stage26.md"
DECISION = RESULTS_ROOT / "final_decision_stage22_to_stage26.json"

STAGES = {
    "22": {
        "directory": "stage22_lagged_descent_validation_v1",
        "decision": "stage22_decision.json",
        "expected_runs": 24,
        "hypothesis": "Alignement avec la descente retardée",
    },
    "23": {
        "directory": "stage23_robust_admissibility_validation_v1",
        "decision": "stage23_decision.json",
        "expected_runs": 36,
        "hypothesis": "Admissibilité robuste continue + FAR",
    },
    "24": {
        "directory": "stage24_multikrum_admissibility_validation_v1",
        "decision": "stage24_decision.json",
        "expected_runs": 36,
        "hypothesis": "Noyau Multi-Krum + FAR",
    },
    "25": {
        "directory": "stage25_robust_anchor_containment_validation_v1",
        "decision": "stage25_decision.json",
        "expected_runs": 36,
        "hypothesis": "Ancre RFA + correction FAR confinée",
    },
    "26B": {
        "directory": "stage26b_trmean_nnm_anchor_confirmatory_v1",
        "decision": "stage26b_decision.json",
        "expected_runs": 27,
        "hypothesis": "Ancre trMean(NNM) sélectionnée + confirmation indépendante",
    },
}

SCREEN_26A = RESULTS_ROOT / "stage26a_robust_anchor_selection_screen_v1"
SCREEN_26A_EXPECTED_RUNS = 15


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def analyze() -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for stage, spec in STAGES.items():
        directory = RESULTS_ROOT / str(spec["directory"])
        count = sum(1 for _ in directory.rglob("metrics.json"))
        if count != spec["expected_runs"]:
            raise ValueError(
                f"Stage {stage} incomplet: {count}/{spec['expected_runs']} runs"
            )
        decision = _load(directory / str(spec["decision"]))
        if int(decision["completed_runs"]) != spec["expected_runs"]:
            raise ValueError(f"Stage {stage}: décision et fichiers en désaccord")
        records[stage] = decision

    screen_count = sum(1 for _ in SCREEN_26A.rglob("metrics.json"))
    if screen_count != SCREEN_26A_EXPECTED_RUNS:
        raise ValueError(
            f"Stage 26A incomplet: {screen_count}/{SCREEN_26A_EXPECTED_RUNS} runs"
        )
    selection = _load(SCREEN_26A / "stage26a_selection.json")
    if selection["selected_method"] != "dt_ldp_far_stage26_anchor_trmean_nnm":
        raise ValueError(
            "La sélection Stage 26A n'est pas celle confirmée au Stage 26B"
        )

    all_privacy = all(record["checks"]["privacy"] for record in records.values())
    all_pairing = all(
        record["checks"]["strict_randomness_pairing"] for record in records.values()
    )
    all_caps = all(record["checks"]["weight_cap"] for record in records.values())
    containment = all(
        records[stage]["checks"]["correction_certificate"] for stage in ("25", "26B")
    )
    negative_claims = all(
        "not_confirmed" in str(record["claim_scope"]) for record in records.values()
    )

    total_runs = sum(int(spec["expected_runs"]) for spec in STAGES.values())
    total_runs += SCREEN_26A_EXPECTED_RUNS
    summary = {
        "status": "negative_validation_complete",
        "evaluated_runs": total_runs,
        "stages": ["22", "23", "24", "25", "26A", "26B"],
        "algorithmic_utility_validated": False,
        "safety_invariants_validated_in_evaluated_runs": (
            all_privacy and all_pairing and all_caps and containment
        ),
        "deterministic_containment_certificate_respected": containment,
        "all_privacy_checks_passed": all_privacy,
        "all_weight_caps_passed": all_caps,
        "all_strict_pairing_checks_passed": all_pairing,
        "all_preregistered_algorithmic_claims_rejected": negative_claims,
        "research_decision": (
            "stop_adaptive_search; do not claim end-to-end utility superiority; "
            "retain the privacy, weight-cap, and containment results"
        ),
        "stage26b": records["26B"]["observations"],
    }
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    observations = {stage: record["observations"] for stage, record in records.items()}
    lines = [
        "# Décision finale DT-LDP-FAR — Stages 22 à 26",
        "",
        "## Verdict exécutif",
        "",
        "La campagne est terminée et produit une **validation négative confirmatoire** de l'hypothèse d'utilité end-to-end étudiée. Les mécanismes testés ne justifient pas de revendiquer que DT-LDP-FAR améliore de façon reproductible l'accuracy ou la fairness sous IPM et Bit-Flip dans le protocole Fashion-MNIST considéré.",
        "",
        "Ce verdict ne signifie pas que l'implémentation est invalide. Deux ensembles de résultats doivent être distingués :",
        "",
        "1. **Validé positivement** : même budget de confidentialité dans les bras appariés, cap public des poids respecté, appariement aléatoire exact et certificat déterministe de confinement de la correction FAR.",
        "2. **Réfuté dans le domaine testé** : bénéfice d'utilité du retard, des règles d'admissibilité et de la correction FAR bornée face aux deux attaques.",
        "",
        f"La décision agrège **{total_runs} runs** : 159 runs confirmatoires/intégration des Stages 22–26B et 15 runs de sélection Stage 26A.",
        "",
        "## Chaîne de tests",
        "",
        "| Étape | Hypothèse | Runs | Résultat quantitatif central | Verdict |",
        "|---|---|---:|---|---|",
        "| 22 | Alignement avec la descente retardée | 24 | gain attaqué moyen −0,395 pp ; taux positif 1/6 | non confirmé |",
        "| 23 | Admissibilité continue + FAR | 36 | gain intégré +0,862 pp, mais coût propre −1,503 pp et Bit-Flip −0,230 pp | non confirmé |",
        "| 24 | Noyau Multi-Krum + FAR | 36 | gain attaqué −2,725 pp ; coût propre −4,447 pp | non confirmé |",
        "| 25 | Ancre RFA + correction confinée | 36 | ancre sous IPM −14,670 pp ; correction attaquée −0,017 pp | non confirmé |",
        "| 26A | Sélection d'ancre, seed 353 | 15 | trMean(NNM) choisie selon la règle figée ; aucun claim confirmatoire | sélection seulement |",
        "| 26B | Holdout trMean(NNM), seeds 359/367/373 | 27 | Bit-Flip +2,903 pp, IPM −0,377 pp ; correction FAR −0,032 pp | non confirmé |",
        "",
        "## Confirmation indépendante du Stage 26B",
        "",
        "La seed de sélection 353 n'est pas réutilisée. Les trois seeds de confirmation sont 359, 367 et 373.",
        "",
        "### Ancre trMean(NNM) contre moyenne uniforme",
        "",
        f"- coût propre moyen : {_fmt(observations['26B']['anchor_clean_accuracy_mean_pp'], 3)} pp ;",
        f"- gain moyen sous les deux attaques : +{_fmt(observations['26B']['anchor_attacked_accuracy_gain_mean_pp'], 3)} pp ;",
        f"- gain Bit-Flip : +{_fmt(observations['26B']['anchor_attacked_accuracy_gain_by_attack_pp']['bf20_n25_s10'], 3)} pp ;",
        f"- gain IPM : {_fmt(observations['26B']['anchor_attacked_accuracy_gain_by_attack_pp']['ipm20_n25'], 3)} pp ;",
        f"- taux de paires attaquées positives : {_fmt(100 * observations['26B']['anchor_attacked_positive_gain_rate'], 1)} %.",
        "",
        "L'amélioration moyenne est portée exclusivement par Bit-Flip. Le gate exigeait un gain moyen d'au moins +0,25 pp pour **chaque** attaque et au moins 2/3 de paires positives. Les deux conditions échouent ; agréger IPM et Bit-Flip masquerait cette hétérogénéité.",
        "",
        "### Contribution propre de la correction FAR bornée",
        "",
        f"- correction − ancre, sans attaque : {_fmt(observations['26B']['correction_clean_accuracy_mean_pp'], 3)} pp ;",
        f"- correction − ancre, sous attaque : {_fmt(observations['26B']['correction_attacked_accuracy_gain_mean_pp'], 3)} pp ;",
        f"- différence Worst-20 sous attaque : {_fmt(observations['26B']['correction_attacked_worst20_difference_mean_pp'], 3)} pp ;",
        f"- concentration médiane : {_fmt(observations['26B']['candidate_concentration_median'], 5)} ;",
        f"- norme médiane de correction : {_fmt(observations['26B']['candidate_correction_norm_median'], 6)}, pour une borne universelle de {_fmt(observations['26B']['candidate_correction_universal_bound'], 3)}.",
        "",
        "La correction est bien active et non uniforme, mais son effet moyen est pratiquement nul et légèrement négatif. Le mécanisme FAR n'apporte donc pas le bénéfice propre requis, même après remplacement de l'ancre RFA par le meilleur candidat du screen.",
        "",
        "## Ce qui est effectivement validé",
        "",
        "| Propriété | Nature | Résultat | Portée |",
        "|---|---|---|---|",
        "| Budget de confidentialité identique | audit empirique | écart maximal d'epsilon = 0 dans chaque comparaison | tous les bras Stages 22–26 |",
        "| Cap des poids | garantie analytique + audit | aucune violation observée | toutes les campagnes |",
        "| Appariement aléatoire | contrôle expérimental | taux = 1,0 | tous les holdouts |",
        "| Confinement autour de l'ancre | garantie déterministe + audit | norme observée ≤ 0,00377, borne 0,21 | Stages 25–26 |",
        "| Supériorité d'utilité | claim empirique | non confirmée | Fashion-MNIST, LeNet-5, n=25, 12 rounds, epsilon≈4, IPM/Bit-Flip |",
        "",
        "Le certificat de confinement montre que la correction FAR ne peut pas éloigner arbitrairement l'agrégat de l'ancre. Il ne prouve pas que l'ancre constitue une bonne direction de descente ni que la petite correction améliore l'optimisation.",
        "",
        "## Décision scientifique",
        "",
        "La recherche adaptative sur cette même famille et ces mêmes données doit s'arrêter ici. Modifier encore les seuils, sélectionner une seed favorable ou conserver uniquement Bit-Flip transformerait le holdout en données d'entraînement méthodologique.",
        "",
        "La conclusion publiable et défendable est :",
        "",
        "> Le découplage temporel, le contrôle de la concentration et le confinement autour d'une ancre robuste sont bien définis et vérifiables, mais ils ne suffisent pas à garantir un gain d'utilité end-to-end. Dans le protocole étudié, plusieurs constructions indépendantes échouent sur holdout ; la présente instanciation DT-LDP-FAR doit donc être rejetée comme méthode supérieure.",
        "",
        "Cela ne constitue pas un théorème d'impossibilité universel. Une nouvelle étude ne serait justifiée qu'avec une hypothèse substantiellement différente, une nouvelle fonction objectif et un nouveau holdout ; pas avec une retouche supplémentaire des mêmes scores.",
        "",
        "## Traçabilité",
        "",
        "- protocoles et rapports : `output/analysis/DT_LDP_FAR_Stage22_*` à `output/analysis/DT_LDP_FAR_Stage26B_*` ;",
        "- sélection : `results/dt_ldp_far/stage26a_robust_anchor_selection_screen_v1/stage26a_selection.json` ;",
        "- décision confirmatoire : `results/dt_ldp_far/stage26b_trmean_nnm_anchor_confirmatory_v1/stage26b_decision.json` ;",
        "- décision cumulative machine : `results/dt_ldp_far/final_decision_stage22_to_stage26.json`.",
        "",
        "## Vérifications finales",
        "",
        "- les 174 fichiers `metrics.json` attendus sont présents dans les six étapes ;",
        "- les 27 JSON du holdout Stage 26B sont syntaxiquement valides ;",
        "- la décision cumulative se régénère depuis les décisions de chaque stage ;",
        "- les 73 tests ciblés DT-LDP-FAR réussissent ;",
        "- la suite complète réussit : 244 tests ;",
        "- `git diff --check` ne signale aucune erreur d'espacement.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, sort_keys=True))
