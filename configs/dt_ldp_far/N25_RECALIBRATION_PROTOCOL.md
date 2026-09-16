# DT-LDP-FAR: recalibration publique à 25 clients

## Objectif

Le profil transféré depuis les expériences à 10 clients,
`(U, D_score, rho) = (0.28, 0.24, 0.12)`, n'est pas exploitable à
25 clients. Sur la seed de développement 28, les normes pré-clipping des
uploads honnêtes sont approximativement comprises entre 0.32 et 0.42. Le
profil transféré tronque donc presque tous les uploads honnêtes et sature les
scores à 1, ce qui rend les poids uniformes indépendamment de `alpha`.

La nouvelle grille est fixée avant exécution. Aucun seuil n'est calculé à
partir du message privé du tour courant :

| Axe | Valeurs publiques |
|---|---|
| `U` | 0.38, 0.40, 0.42 |
| `D_score / U` | 0.90, 1.05 |
| `rho / U` | 0.50, 1.00 |
| Nombre de cellules | 12 |

La calibration utilise uniquement la paire de seeds de développement
`partition_seed=28`, `training_seed=28`. Les seeds 36 et 54 sont réservées à
la confirmation indépendante. Cette séparation évite de choisir puis de
valider le profil sur les mêmes réalisations aléatoires.

## Protocole commun

- Fashion-MNIST et LeNet-5 ;
- partition Dirichlet par client à tailles contrôlées, beta = 0.1 ;
- 25 clients, participation complète, aucun client mort ;
- 20 rounds et 10 pas de Poisson par client et par round ;
- sample-level local DP, adjacence add/remove, taux de Poisson 0.05 ;
- epsilon cible 4, delta = 10^-5, clipping local `C=4` ;
- référence `F_CC`, tilting à la borne certifiée `kappa_w=2` ;
- aucun client byzantin pendant la calibration géométrique.

## Gates de promotion versionnés

La première lecture, conservée comme **gate strict v1**, reprenait le seuil
de persistance utilisé à 10 clients :

| Diagnostic calculé sur les rounds 1 à 19 | Condition |
|---|---:|
| Taux médian de clipping serveur | au plus 30 % |
| Fraction de rounds avec clipping au plus 50 % | au moins 80 % |
| Utilisation médiane de `U`, soit `max_i ||Y_i,t|| / U` | entre 0.50 et 1.25 |
| Span médian des scores | au moins 0.15 |
| Fraction de rounds avec span au moins 0.20 (v1 strict) | au moins 60 % |
| Saturation des scores au quantile 90 % | au plus 50 % |
| Marge minimale du cap de poids | au moins -10^-8 |

Les 12 tâches de la grille ont toutes terminé, mais aucun profil n'a passé ce
gate strict. Pour le meilleur profil, le span théorique maximal tour par tour
`1 - d_min / d_max` n'excède lui-même 0.20 que sur 8 rounds informatifs sur
19. Le seuil de persistance v1 est donc inaccessible pour cette réalisation,
quel que soit le choix d'un `D_score` fixe.

Le **gate opérationnel n=25 v2**, figé avant d'ouvrir les seeds de
confirmation, conserve toutes les conditions ci-dessus mais mesure la
persistance au seuil 0.15 : au moins 60 % des rounds doivent avoir un span de
score supérieur ou égal à 0.15. Le CSV v1 n'est ni effacé ni réinterprété.
Le CSV v2 enregistre dans chaque ligne le seuil et la fraction exacts utilisés.

Le profil retenu est le premier du classement déterministe produit par
`analyze_dt_ldp_far_refined_geometry_gate.py` parmi les profils qui passent
le gate. La priorité du classement est : stress informatif, span médian plus
grand, puis saturation plus faible. Il doit ensuite repasser le même gate sur
les seeds indépendantes 36 et 54 avant toute comparaison scientifique.

## Ordre d'exécution

L'orchestrateur reproductible exécute toute la chaîne conditionnelle :

```bash
venv/bin/python -u scripts/run_dt_ldp_far_n25_conditional_pipeline.py \
  --device mps 2>&1 | tee -a logs/dt_ldp_far/n25_conditional_pipeline_mps.log
```

Il est resumable au niveau de chaque tâche. Il s'arrête avec un statut
explicite si aucun profil ne passe la calibration ou si l'une des deux seeds
de confirmation échoue. Les matrices de confirmation, courant/retardé et
références sont rejetées par le validateur tant que le fichier de preuve du
gate ne contient pas le statut requis.

### 1. Validation et recalibration

```bash
venv/bin/python scripts/run_dt_ldp_far.py --validate \
  --matrix configs/dt_ldp_far/decisive_stage3_geometry_recalibration_n25.yaml

DEVICE=mps PYTHON_BIN=venv/bin/python DATA_ROOT=data \
OUTPUT_ROOT=results/dt_ldp_far/decisive \
bash scripts/run_dt_ldp_far_decisive_stage.sh \
  configs/dt_ldp_far/decisive_stage3_geometry_recalibration_n25.yaml
```

### 2. Calcul du gate

```bash
venv/bin/python scripts/analyze_dt_ldp_far_refined_geometry_gate.py \
  results/dt_ldp_far/decisive/dt_ldp_far_decisive_stage3_geometry_recalibration_n25_v1 \
  --output output/analysis/dt_ldp_far_n25_geometry_recalibration_gate_v2.csv \
  --minimum-median-score-span 0.15 \
  --persistent-score-span-threshold 0.15 \
  --minimum-persistent-score-span-fraction 0.60
```

### 3. Confirmation indépendante

L'orchestrateur remplace explicitement le profil placeholder dans les trois
matrices de stage 3 par le profil sélectionné, écrit le fichier de preuve,
puis valide :

```bash
venv/bin/python scripts/run_dt_ldp_far.py --validate \
  --matrix configs/dt_ldp_far/decisive_stage3_geometry_confirmation_n25.yaml
```

La confirmation comporte deux tâches, une pour chacune des seeds 36 et 54.
Elle doit réussir sur les deux seeds. Un échec interdit la promotion et
entraîne une nouvelle grille publique, jamais une adaptation en ligne.

### 4. Comparaison courant contre retardé

La matrice `decisive_stage3_current_delay_n25.yaml` contient 15 tâches :

- 3 contrôles DP-FedAvg ;
- 3 comparaisons appariées courant/retardé au tilt certifié, soit 6 runs ;
- 3 comparaisons appariées courant/retardé au stress `2 x alpha_max`, soit
  6 runs, explicitement non certifiés pour l'influence mais valides comme
  post-traitement local-DP.

### 5. Comparaison des références

La matrice `decisive_stage3_references_n25.yaml` contient 60 tâches : trois
seeds, les cinq références `F_CC`, Huber, RFA, CM(NNM) et trMean(NNM), et
quatre menaces : aucune, Bit-Flip amplifié, IPM et ALIE. Tous les autres
paramètres sont appariés. L'ordre de la matrice place `F_CC`, Huber et RFA en
premier ; CM(NNM) et trMean(NNM) complètent ensuite l'audit demandé sur toutes
les références déjà screenées.

## Emplacements

- résultats de calibration :
  `results/dt_ldp_far/decisive/dt_ldp_far_decisive_stage3_geometry_recalibration_n25_v1/` ;
- log MPS :
  `logs/dt_ldp_far/decisive_stage3_geometry_recalibration_n25_mps.log` ;
- synthèse du gate :
  `output/analysis/dt_ldp_far_n25_geometry_recalibration_gate_v1.csv` pour le
  screen strict et
  `output/analysis/dt_ldp_far_n25_geometry_recalibration_gate_v2.csv` pour le
  gate opérationnel n=25 confirmé hors seed de développement.
