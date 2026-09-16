from datetime import datetime, timezone
display(Markdown(
    "### Décision — vérifier l’agrégat réellement appliqué, pas seulement sa référence\n\n"
    "Le modèle reçoit **A = Σ λᵢXᵢ**, pas F. Réduire l’erreur de F ne débruite pas "
    "Xᵢ et n’impose pas une baisse du poids des attaquants.\n\n"
    "**Établi :** les variantes RCIG testées ne démontrent pas de gain final pertinent ; "
    "une moyenne temporelle améliore fortement la référence sans transmettre ce gain au modèle.\n\n"
    "**Non établi :** une nouvelle méthode qui améliore simultanément accuracy, fairness "
    "et robustesse sous la privacy annoncée. Nous avons un diagnostic scientifique, "
    "pas encore un algorithme publiable validé.\n\n"
    "La campagne de diagnostic en cours est **aggregation_role_n10_v1** : "
    "96 runs = 4 règles × 2 bruits × 4 scénarios × 3 seeds. Elle compare uniforme, "
    "RFA directe, FAR + RFA et FAR + moyenne temporelle, avec warmup uniforme 1–12, "
    "40 tours, B=120, C=4, U=16, α=0,1 pour FAR et ε maximal≈4. "
    "Ce n’est **pas** la confirmation F/T20 des courbes de la section 3.\n\n"
    "Les quatre candidats sont aussi évalués sur les **mêmes messages privés** à "
    "chaque tour. Cela localise l’effet référence → poids → agrégat ; l’entraînement "
    "complet mesure ensuite l’effet sur le modèle. Le papier FAR comparait déjà "
    "RFA(buck) à FAR + RFA(buck), sans DP. Notre RFA est sans bucketing et cette "
    "nouvelle campagne n’a pas de témoin sans DP : elle n’identifie pas l’effet causal "
    "du bruit et n’est pas une comparaison nouvelle par sa seule liste de baselines."
))
campaign_path = RESULTS_ROOT / "aggregation_role_n10_v1"
states = [json.loads(p.read_text()) for p in campaign_path.glob("*/orchestration_status.json")]
progress = {k: sum(s.get("status") == k for s in states) for k in ["completed", "running", "failed"]}
display(pd.DataFrame([{
    "Campagne": "Diagnostic des règles d’agrégation N10", "Terminés (statuts)": progress["completed"],
    "En cours (statuts)": progress["running"], "Échecs": progress["failed"], "Total prévu": 96,
    "Lecture UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
}]))
display(Markdown(
    "Cet instantané lit les statuts du superviseur, pas la présence effective d’un processus. "
    "Les résultats partiels ne sont pas utilisés ici pour déclarer une méthode gagnante.\n\n"
    "**Piste à examiner, non exécutée :** corriger ou borner l’agrégat A par rapport à "
    "une prédiction temporelle et une tolérance au bruit, tout en mesurant le signal "
    "honnête retiré. Le déplacement à ce niveau rend l’action effective sur le modèle ; "
    "il ne suffit pas à prouver un gain ou une nouveauté.\n\n"
    "Lecture de DMM et proposition conditionnelle : "
    "[DMM, RCIG et contrôle de l’agrégat](../output/analysis/DMM_RCIG_Agregat_Local_DP_Analyse.md). "
    "DMM est un mécanisme de DP distribuée par agrégation sécurisée ; il n’est pas "
    "directement notre mécanisme de DP locale par message."
))
