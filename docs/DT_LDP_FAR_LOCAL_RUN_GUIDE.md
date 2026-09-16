# Guide local d'exécution de DT-LDP-FAR

Ce guide explique comment valider et lancer DT-LDP-FAR sur le MacBook en attendant que Toubkal soit disponible. Toutes les commandes sont à exécuter depuis la racine du dépôt fedlab_zmq.

## 1. Choisir le niveau d'exécution

| Niveau | Objectif | Valeur scientifique |
|---|---|---|
| Tests unitaires | Vérifier l'algorithme et le lanceur | Validation logicielle |
| Smoke test MNIST | Vérifier l'environnement et l'exécution de bout en bout | Aucune conclusion scientifique |
| Tâche pilote limitée à 2 tours | Vérifier une cellule de la matrice E1–E8 | Test d'infrastructure uniquement |
| Tâche pilote complète | Exécuter une configuration sur les 40 tours prévus | Résultat pilote |
| Campagne pilote E0–E8 | Exécuter les 37 tâches sur une paire de seeds | Go/no-go avant la campagne complète |
| Campagne complète E0–E8 | Exécuter 318 tâches sur trois paires de seeds | Campagne destinée au papier |

Sur le MacBook, l'ordre recommandé est : tests, smoke test, une tâche privée de deux tours, la même tâche sur 40 tours, puis éventuellement les 37 tâches pilotes. La campagne complète doit normalement attendre Toubkal.

## 2. Se placer dans le dépôt

```bash
cd /Users/joshuajusteyunpeinikiema/Documents/PhD_UM6P/Stat_of_art_EnergyEfficientFL/fedlab_zmq
```

## 3. Préparer l'environnement Python

Python 3.10 ou plus récent est requis.

### Première installation

```bash
PYTHON=python3 bash setup_env.sh
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Le script installe PyTorch, les dépendances du framework et celles du dashboard. L'installation éditable permet à Python d'utiliser directement les modifications du dépôt.

### Sessions suivantes

```bash
source venv/bin/activate
python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
```

Si l'environnement virtuel n'est pas activé, utiliser directement venv/bin/python à la place de python.

## 4. Choisir CPU ou MPS

Sur un Mac Apple Silicon, MPS permet à PyTorch d'utiliser le GPU intégré. Vérifier sa disponibilité :

```bash
python -c "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

- Si MPS available vaut True, utiliser --device mps.
- Sinon, utiliser --device cpu.

Les commandes suivantes utilisent MPS. Pour exécuter sur CPU, remplacer mps par cpu. Les tâches MPS doivent être lancées séquentiellement afin de limiter la concurrence sur la mémoire unifiée.

## 5. Valider l'implémentation

```bash
python -m pytest -q tests/test_dt_ldp_far.py tests/test_run_dt_ldp_far.py
```

Résultat attendu avec la version actuelle :

```text
28 passed
```

Ces tests vérifient notamment l'ordre causal retardé, l'alignement par identifiant client, le véritable échantillonnage de Poisson et la cohérence du lanceur matriciel. Ils forment un prérequis logiciel déterministe. Ils sont distincts du bloc expérimental E0 de la matrice YAML, lequel calibre publiquement l'échelle de score `D_score` et tau avant les autres expériences.

## 6. Exécuter le smoke test

Le smoke test MNIST réalise deux tours et confirme le fonctionnement du chargement des données, des clients, du serveur, de DT-LDP-FAR, des métriques et de la sauvegarde.

### Avec MPS

```bash
python run_experiment.py \
  --config configs/dt_ldp_far/smoke_mnist.yaml \
  --device mps \
  --output results/dt_ldp_far/local_smoke
```

### Avec CPU

```bash
python run_experiment.py \
  --config configs/dt_ldp_far/smoke_mnist.yaml \
  --device cpu \
  --output results/dt_ldp_far/local_smoke_cpu
```

Fashion-MNIST sera téléchargé automatiquement au premier lancement d'une tâche qui l'utilise. Une connexion internet est donc nécessaire si les données ne sont pas déjà présentes.

Le smoke test valide seulement l'installation. Il ne doit pas être inclus dans les tableaux scientifiques.

## 7. Valider et inspecter la matrice pilote E0–E8

La matrice pilote v2 contient 37 tâches sur une paire de seeds.

### Validation

```bash
python scripts/run_dt_ldp_far.py \
  --validate \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml
```

Résultat attendu :

```text
VALID campaign=dt_ldp_far_pilot_e0_e8_v2 tasks=37
```

### Liste des tâches et indices

```bash
python scripts/run_dt_ldp_far.py \
  --list \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml
```

| Indices | Bloc |
|---:|---|
| 0–5 | E0, grille publique `D_score` in {0.02, 0.05, 0.10} et tau |
| 6–7 | E1, baselines natives non privées |
| 8–10 | E1, contrôle Poisson/clipping sans bruit |
| 11–12 | E2, découplage bruit–poids |
| 13–16 | E3, frontière confidentialité–utilité |
| 17–19 | E4, clients byzantins |
| 20–24 | E5, choix de la référence robuste |
| 25–27 | E6, coût du retard |
| 28–30 | E7, nombre d'étapes DP locales |
| 31–34 | E8, voie Poisson commune |
| 35 | E8, FedFDP natif |
| 36 | E8, FedFDP limité à cinq minibatches |

Les filtres ne renumérotent pas les tâches. L'indice reste celui affiché dans la matrice complète.

### Dry-run d'une tâche

L'index 16 correspond à DT-LDP-FAR dans E3 avec un budget cible de 4, cinq étapes Poisson et la référence centered clipping.

```bash
python scripts/run_dt_ldp_far.py \
  --dry-run \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --job-index 16 \
  --device mps \
  --data-root data \
  --output-root results/dt_ldp_far/local_pilot_v2
```

Le dry-run écrit resolved_config.yaml et affiche la commande finale sans lancer l'entraînement.

## 8. Lancer d'abord une tâche privée courte

```bash
python -u scripts/run_dt_ldp_far.py \
  --run \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --job-index 16 \
  --pilot-rounds 2 \
  --device mps \
  --data-root data \
  --output-root results/dt_ldp_far/local_check_2rounds
```

Le lanceur recalcule l'horizon de l'accountant pour deux tours et place cette sortie dans un espace de short run. Elle est explicitement non utilisable comme preuve pour le papier.

## 9. Lancer une tâche pilote complète

Une fois le test court réussi :

```bash
python -u scripts/run_dt_ldp_far.py \
  --run \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --job-index 16 \
  --device mps \
  --data-root data \
  --output-root results/dt_ldp_far/local_pilot_v2 \
  --resume
```

Dans la voie Poisson principale, le champ local_epochs désigne le nombre d'étapes DP de Poisson indépendantes par tour de communication, et non le nombre de passages complets sur les données locales. E7 étudie précisément 1, 2 et 5 étapes.

## 10. Lancer les 37 tâches pilotes séquentiellement

```bash
set -o pipefail
mkdir -p logs/dt_ldp_far

for dt_job_index in $(seq 0 36); do
  python -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
    --job-index "$dt_job_index" \
    --device mps \
    --data-root data \
    --output-root results/dt_ldp_far/local_pilot_v2 \
    --resume 2>&1 | tee -a logs/dt_ldp_far/pilot_v2_mps.log || break
done
```

Pour CPU, remplacer --device mps par --device cpu et éventuellement renommer le fichier de log.

L'option --resume saute une tâche uniquement si son metrics.json est valide et contient tous les tours déclarés. Elle ne reprend pas au milieu d'un entraînement. Une tâche incomplète ou en échec est relancée depuis le début.

### Lancer seulement le bloc E3

Afficher les indices concernés :

```bash
python scripts/run_dt_ldp_far.py \
  --list \
  --matrix configs/dt_ldp_far/pilot_e1_e8.yaml \
  --experiment E3_privacy_utility \
  --method dt_ldp_far
```

Puis lancer séparément chacun des indices affichés. Le runner n'offre actuellement ni ordonnanceur local parallèle, ni option --jobs. Une exécution séquentielle est la voie recommandée sur MPS.

### E7 étendu : nombre de pas Poisson et horizon de communication

La matrice dédiée étend E7 à

```bash
\[
K\in\{1,2,5,10,20\},
\]
```

où \(K\) est le nombre de pas DP avec échantillonnage de Poisson exécutés par client et par tour de communication. Elle peut être croisée avec les horizons publics

```bash
\[
T\in\{40,80,120\}.
\]
```

Valider et inspecter les cinq tâches de la matrice :

```bash
python scripts/run_dt_ldp_far.py \
  --validate \
  --matrix configs/dt_ldp_far/pilot_e7_extended.yaml

python scripts/run_dt_ldp_far.py \
  --list \
  --matrix configs/dt_ldp_far/pilot_e7_extended.yaml
```

Exécuter la grille complète de 15 configurations sur MPS :

```bash
set -o pipefail
mkdir -p logs/dt_ldp_far

for dt_rounds in 40 80 120; do
  for dt_job_index in $(seq 0 4); do
    python -u scripts/run_dt_ldp_far.py \
      --run \
      --matrix configs/dt_ldp_far/pilot_e7_extended.yaml \
      --job-index "$dt_job_index" \
      --pilot-rounds "$dt_rounds" \
      --device mps \
      --data-root data \
      --output-root results/dt_ldp_far/e7_extended_v1 \
      --resume 2>&1 | \
      tee -a "logs/dt_ldp_far/e7_extended_T${dt_rounds}_mps.log" || exit 1
  done
done
```

Chaque horizon est automatiquement placé dans un sous-répertoire distinct `short_runs/rounds_T`. Le lanceur recalcule aussi l'horizon de composition de l'accountant pour les \(T\times K\) mécanismes locaux. À la fin de la grille, 15 fichiers `metrics.json` sont attendus :

```bash
find results/dt_ldp_far/e7_extended_v1 -name metrics.json | wc -l
```

Cette grille utilise une seule paire de seeds et sert à sélectionner un régime \((T,K)\). L'option `--pilot-rounds` marque explicitement les sorties comme calibration pilote, et non comme preuve finale pour le papier. Après sélection, la configuration retenue doit être réexécutée avec plusieurs seeds et partitions dans une matrice finale figée.

Pour la grille scientifique complète, utiliser la matrice qui encode directement l'horizon dans les scénarios. Elle couvre deux niveaux d'hétérogénéité, trois horizons, cinq valeurs de \(K\) et trois paires de seeds, soit 90 tâches. Il ne faut pas lui passer `--pilot-rounds` :

```bash
python scripts/run_dt_ldp_far.py \
  --validate \
  --matrix configs/dt_ldp_far/e7_extended_paper.yaml

python scripts/run_dt_ldp_far.py \
  --list \
  --matrix configs/dt_ldp_far/e7_extended_paper.yaml
```

Exécution séquentielle sur MPS :

```bash
set -o pipefail
mkdir -p logs/dt_ldp_far

for dt_job_index in $(seq 0 89); do
  python -u scripts/run_dt_ldp_far.py \
    --run \
    --matrix configs/dt_ldp_far/e7_extended_paper.yaml \
    --job-index "$dt_job_index" \
    --device mps \
    --data-root data \
    --output-root results/dt_ldp_far \
    --resume 2>&1 | \
    tee -a logs/dt_ldp_far/e7_extended_paper_v1_mps.log || exit 1
done
```

Dans cette matrice, le scénario fait partie de l'identifiant de tâche. Les horizons \(T=40,80,120\) ont donc des sorties distinctes sans ambiguïté, et le multiplicateur de bruit est recalibré pour conserver le même budget cible \((\varepsilon,\delta)=(4,10^{-5})\) malgré la variation du nombre total de mécanismes \(T K\).

## 11. Résultats et états des tâches

Chaque tâche possède son propre répertoire sous l'output-root. Le lanceur produit :

- resolved_config.yaml : configuration entièrement résolue ;
- dt_ldp_far_task_manifest.json : identité et état de la tâche ;
- metrics.json : métriques par tour ;
- manifest.json : métadonnées du framework ;
- survival.csv : diagnostics de participation et d'énergie ;
- final_model.pt : modèle final.

Compter les tâches ayant produit des métriques :

```bash
find results/dt_ldp_far/local_pilot_v2 -name metrics.json | wc -l
```

Inspecter les états :

```bash
rg -n '"status": "(complete|complete_reused|failed|incomplete|running)"' \
  results/dt_ldp_far/local_pilot_v2 \
  -g 'dt_ldp_far_task_manifest.json'
```

Un processus interrompu peut laisser un manifeste à running. Si les métriques ne sont pas complètes, --resume relance néanmoins la tâche.

## 12. Visualiser les résultats dans le dashboard

Dans un second terminal :

```bash
cd /Users/joshuajusteyunpeinikiema/Documents/PhD_UM6P/Stat_of_art_EnergyEfficientFL/fedlab_zmq
source venv/bin/activate
streamlit run dashboard/app.py
```

Le dashboard parcourt récursivement results. Les sorties dans results/dt_ldp_far/local_pilot_v2 sont donc détectées automatiquement. Pour un autre output-root, indiquer son chemin absolu dans le champ Results folder de la barre latérale.

## 13. Pipeline privé réellement exécuté

La voie principale est une DP locale au niveau des exemples avec adjacence add/remove :

1. chaque exemple local est inclus indépendamment avec la probabilité publique q = 0,05 ;
2. chaque gradient sélectionné est clippé ;
3. les gradients clippés sont sommés ;
4. un vecteur gaussien est ajouté à cette somme ;
5. la somme bruitée est divisée par la taille de batch attendue publique q fois N_i public ;
6. le client effectue une étape d'optimisation et transmet son update privé ;
7. le serveur clippe cet update privé puis l'agrège avec les poids calculés au tour précédent ;
8. la référence et les scores du tour courant servent uniquement à construire les poids du tour suivant.

Le retard est la contrainte causale essentielle : le poids appliqué à un update ne doit pas être calculé à partir du bruit privé contenu dans ce même update.

Le runner local utilise le simulateur mono-processus run_experiment.py. Il n'exécute pas le backend ZMQ distribué.

## 14. Règles à respecter

- Ne pas remplacer l'échantillonnage de Poisson par un minibatch fixe en conservant l'accountant Poisson.
- Ne pas utiliser les diagnostics privés internes du client comme sorties d'une expérience de confidentialité. Ils sont activés uniquement dans E2 à des fins instrumentales ; E2 n'est pas une preuve DP.
- Ne pas revendiquer d'amplification par sélection de clients : la voie théorique utilise la participation complète et zéro dropout.
- Ne pas interpréter le smoke test ou --pilot-rounds 2 comme une expérience scientifique.
- Ne pas modifier manuellement resolved_config.yaml. Toute modification scientifique doit être faite dans les fichiers de protocole puis validée.
- Conserver les mêmes paires de seeds pour les comparaisons appariées.
- Enregistrer l'epsilon réalisé, delta, le multiplicateur de bruit, le nombre d'étapes, l'adjacence et le sampler avec chaque résultat.

## 15. Campagne complète lorsque Toubkal sera disponible

La matrice complète contient 318 tâches et trois paires de seeds :

```bash
python scripts/run_dt_ldp_far.py \
  --validate \
  --matrix configs/dt_ldp_far/full_e1_e8.yaml

python scripts/run_dt_ldp_far.py \
  --list \
  --matrix configs/dt_ldp_far/full_e1_e8.yaml
```

Elle ne doit être lancée qu'après acceptation des 37 tâches pilotes et des critères go/no-go. Une tâche complète donnée peut être testée localement avec la même commande --run et un job-index précis, mais lancer les 318 tâches sur le MacBook est déconseillé en raison du temps, de l'énergie et du risque d'interruption.

## Checklist avant campagne

- [ ] environnement Python activé ;
- [ ] MPS ou CPU détecté correctement ;
- [ ] suite ciblée `test_dt_ldp_far`, `test_run_dt_ldp_far` et références FAR/FedFDP réussie ;
- [ ] smoke test terminé ;
- [ ] matrice validée avec le nombre attendu de tâches ;
- [ ] liste des indices archivée ;
- [ ] dry-run de la première tâche contrôlé ;
- [ ] tâche privée de deux tours terminée ;
- [ ] tâche pilote complète terminée avec metrics.json valide ;
- [ ] epsilon réalisé et diagnostics causaux inspectés ;
- [ ] dashboard capable de charger les sorties ;
- [ ] campagne pilote autorisée avant le passage aux 318 tâches.

## 16. Ablation à batch fixe sans remise (Stage 9)

Cette ablation ne réutilise pas l'accountant Poisson. À chaque pas local, le
client tire uniformément un sous-ensemble de taille publique exactement fixée
à $m=120$ parmi $N_i^{\mathrm{pub}}=2400$ exemples. Les exemples sont donc
distincts à l'intérieur d'un pas, avec

$$
q=\frac{m}{N_i^{\mathrm{pub}}}=0{,}05.
$$

Deux pas successifs effectuent deux tirages indépendants et peuvent contenir
des exemples communs. Le contrat de confidentialité est l'adjacence
**replace-one**, appropriée aux jeux voisins de même cardinalité. Après un
clipping individuel à $C$, remplacer un exemple peut modifier la somme des
gradients de $2C$. Le bruit implémenté est donc calibré avec cette
sensibilité et la borne RDP du sous-échantillonnage uniforme sans remise ; il
ne faut pas comparer son multiplicateur de bruit brut à celui de la voie
Poisson add/remove comme s'ils protégeaient exactement la même relation de
voisinage.

Valider puis lancer les 36 tâches sur MPS :

```bash
python scripts/run_dt_ldp_far.py \
  --validate \
  --matrix configs/dt_ldp_far/decisive_stage9_fixed_without_replacement_n25.yaml

./scripts/run_dt_ldp_far_stage9_fixed_wor_local.sh
```

Produire les tables appariées après la fin des 36 tâches :

```bash
python scripts/analyze_dt_ldp_far_stage9_fixed_wor.py
```

La comparaison **courant contre retardé** à l'intérieur de Stage 9 est la
comparaison causale principale : les deux bras partagent leurs seeds et leurs
clés de randomness. La comparaison **batch fixe replace-one contre Poisson
add/remove** est une analyse de protocole secondaire. Elle ne permet pas
d'attribuer à elle seule une différence d'accuracy au sampler, puisque
l'adjacence et la sensibilité changent simultanément.
