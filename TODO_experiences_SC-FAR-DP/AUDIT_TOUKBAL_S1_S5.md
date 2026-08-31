# Audit d'exécutabilité Toubkal des matrices S1 à S5

Date de l'audit : 2026-08-31.

## Verdict global

Les matrices **S1 à S4 sont maintenant préspécifiées, validées et
soumissibles** avec le protocole full-update du papier 1. Elles emploient le
moteur `scripts/run_scfar_paper1.py`, les manifestes sous
`configs/scpfar/paper1/` et les workers Toubkal
`hpc/run_scfar_paper1_{cpu,gpu}.slurm`.

Cette readiness signifie que les facteurs, seeds, sorties et invariants sont
verrouillés et contrôlés avant exécution. Elle ne signifie pas que les
résultats existent déjà. S5 reste à construire à partir des décisions prises
après S1 à S4. Les anciens lanceurs `run_level_s_scfar_validation*` concernent
l'extension partial et ne doivent pas être utilisés pour le papier 1.

| Matrice | Verdict | Ce qui est exécutable | Bloqueur principal |
|---|---|---|---|
| S1, robustesse-biais de F_CC | **Prête, non exécutée** | 1 908 tâches ; `tau/C`, ancre, cadence, attaques et fractions figés | Exécuter d'abord les sous-matrices de screening, puis sélectionner le régime primaire sans regarder S2/S4 |
| S2, ablations full-update | **Prête, non exécutée** | 360 tâches ; neuf bras non privés/DP, F_CC principal et Huber en ablation | Effectuer un pilote court avant la campagne de 100 rounds |
| S3, inclusion/fairness/attaques | **Prête, non exécutée** | 1 224 tâches ; grille `(alpha/alpha_max,kappa_w)`, honest outliers et intensités préenregistrés | Les identités d'outliers sont oracle offline et ne doivent pas intervenir dans l'entraînement |
| S4, user-level central DP | **Prête, non exécutée** | 720 tâches ; epsilon `{infini,1,3,6,10}`, q=1, `2C` contre sensibilité certifiée | Cross-check avec une seconde bibliothèque toujours requis avant publication finale |
| S5, campagne principale | **À construire après S1–S4** | Runner, seeds et protocole full-update réutilisables | Réduire les grilles exploratoires en un petit nombre de bras confirmatoires |

## Contrôles d'infrastructure

### Chemins et stockage

Les lanceurs utilisent par défaut :

```text
REPO_DIR=$HOME/fedlab_zmq
PROJECT_DIR=$HOME/lustre/manapy-um6p-st-msda-1wabcjwe938
WORK_ROOT=$PROJECT_DIR/users/$USER/fedlab_zmq
DATA_ROOT=$WORK_ROOT/datasets
OUTPUT_ROOT=$WORK_ROOT/results/...
```

Ces chemins correspondent à l'arborescence Toubkal vérifiée pour le projet.
Ils restent surchargeables avec `sbatch --export=ALL,NAME=value`. Les journaux
Slurm doivent être écrits depuis `$HOME/fedlab_zmq/slurm_logs`, tandis que les
datasets, checkpoints et résultats volumineux restent sur Lustre.

### Environnement

- La voie CPU documentée utilise `Anaconda3/2025.06-1`, l'environnement
  `fedlab-zmq` et les roues CPU de PyTorch.
- Le lanceur GPU exige un environnement distinct contenant une roue CUDA de
  PyTorch. Un environnement créé selon le guide CPU ne peut pas exécuter
  `--device cuda`.
- Les lanceurs ne codent pas le compte Slurm en dur : le compte CPU ou GPU doit
  être fourni à `sbatch`.
- Tous les fichiers `.slurm` présents passent `bash -n`, y compris le lanceur
  S0.5.

### Configurations actuellement résolues

| Fichier | Algorithme | Portée réelle | État |
|---|---|---|---|
| `configs/scpfar/smoke.yaml` | `sc_partial_far_dp` | Smoke partiel, RFA/CM-NNM et bruit numérique non calibré pour un claim | Exécutable, non scientifique |
| `configs/scpfar/cifar10_resnet18_main.yaml` | `sc_partial_far_dp` | Extension partial, F_CC, attaque Min-Sum, epsilon 6 | Exécutable isolément, hors papier 1 full-update |
| `configs/scpfar/cifar10_resnet18_huber_ablation.yaml` | `sc_partial_far_dp` | Même extension avec Huber à 10 itérations | Exécutable isolément, hors matrice S1–S5 |
| `configs/scpfar/paper1/s1_reference_tradeoff.yaml` | `scfar_dp` | S1 full-update, F_CC et Huber ablation | Prête, 1 908 tâches |
| `configs/scpfar/paper1/s2_full_update_ablations.yaml` | Plusieurs bras | S2 full-update | Prête, 360 tâches |
| `configs/scpfar/paper1/s3_inclusion_attacks.yaml` | `scfar_dp` | S3 full-update | Prête, 1 224 tâches |
| `configs/scpfar/paper1/s4_central_dp.yaml` | `scfar_dp` | S4 full-update central-DP | Prête, 720 tâches |

Les modèles `LeNet5`, `AlexNet` et `ResNet18-GN4`, ainsi que les datasets
FashionMNIST et CIFAR-10, sont enregistrés. Cette disponibilité logicielle ne
remplace pas la préspécification des bras expérimentaux.

## Audit par matrice

### S1 — robustesse-biais de F_CC

Le manifeste développe de façon déterministe les facteurs suivants :

```text
tau/C × qualité de l'ancre × anchor_update_rate
× fraction byzantine × attaque × seed de partition × seed d'entraînement
```

Les choix publics sont `tau/C` dans `{0.25,0.5,1,2}`, ancre nulle fixe,
EMA de la publication avec taux `{0.1,0.5}`, publication précédente, et une
ablation de cadence `{1,5}` rounds. Avec `n=25`, les nombres de Byzantins
`b={0,2,5}` représentent exactement les fractions `{0,0.08,0.20}`. La matrice
contient 1 908 tâches ; elle mesure erreur de référence, biais de queue
honnête, dérive d'ancre, masse byzantine et métriques de performance.

### S2 — chaîne d'ablation full-update

Le manifeste exécute FedAvg, F_CC seul, FAR original, FAR user-clippé,
SC-FAR sans DP, DP-FedAvg, DP-FAR à `2C`, SC-FAR-DP à `2C`, SC-FAR-DP
certifié et Huber certifié en ablation. Toutes les variantes du papier 1
utilisent `algorithm: scfar_dp` et aucune clé de partial training. Le bras
certifié est exécutable, mais son interprétation utilitaire doit attendre le
choix robustesse-biais issu de S1.

### S3 — inclusion, fairness et attaques

Les cinq attaques requises sont implémentées après collecte des updates, avec
identités byzantines persistantes recommandées. Les métriques Worst-20, gap,
variance en points de pourcentage carrés, entropie des poids et masse
byzantine sont enregistrables.

La grille est `kappa_w` dans `{1.25,2,5}` et
`alpha/alpha_max` dans `{0,0.25,0.5,0.75,1}`. Un honest outlier est défini
avant training comme l'un des 20 % des vingt clients toujours honnêtes (soit
quatre clients) ayant la plus forte
divergence Jensen-Shannon entre sa distribution de labels et la distribution
globale. Les identités sont préenregistrées par scénario et seed de partition.

BF, IPM et ALIE utilisent des intensités `{0.5,1,2}`. Min-Max et Min-Sum
utilisent `{0.5,1}` : `1` est le point furtif optimisé ; dépasser `1` serait
un stress plus fort, mais ne conserverait plus sa contrainte de furtivité.

### S4 — mécanisme DP et accountant

La lane principale `q=1` est correctement séparée de l'amplification par
sampling : l'accountant compose des mécanismes gaussiens ordinaires. Le bruit
est calibré avec un multiplicateur tel que l'écart-type publié soit
`sigma * Delta_2`.

La matrice epsilon `{1,3,6,10,infini}` est figée. Le bras `infini` désactive
réellement le bruit. Les bras finis utilisent 100 compositions gaussiennes
ordinaires, sans amplification par sampling, et comparent de manière appariée
le calibrage `2C` à la sensibilité certifiée. Les tests contrôlent la formule
gaussienne, la calibration et le ledger. Une comparaison externe avec une
seconde bibliothèque reste un contrôle publication-grade, pas un changement
de protocole.

### S5 — matrice principale et statistiques

S5.2 impose cinq seeds d'entraînement et trois seeds de partition, soit
**15 jobs par bras expérimental**. Le nombre de bras n'étant pas figé, le total
de jobs, le temps GPU et le stockage ne sont pas encore estimables. Les pilotes
S5.1 ne doivent commencer qu'après l'existence d'une matrice S2 exécutable et
la validation S1/S4 des constantes utilisées.

## S0.5 — audit des références comparatrices

### Protocole exécuté

S0.5 est un audit synthétique indépendant du training neural. Il compare la
moyenne arithmétique, la médiane coordonnée (`CM`), le trimmed mean (`trMean`)
et la médiane géométrique (`RFA`) sur des cohortes voisines au sens
replace-one.

Les paramètres publics sont :

```text
n dans {5, 10, 20, 40}
dimension d = 64
borne user-level C = 1
borne des distances D_max = 2
alpha demandé = 1
kappa_w = 2
fraction byzantine nominale = 0.2
ancre publique r_0 = 0
seed synthétique = 42
```

Pour chaque taille `n`, le programme génère d'abord 200 paires `(U,U')` de
cohortes gaussiennes clippées. Une paire partage exactement `n-1` updates ; un
seul update d'indice public est remplacé par un nouveau tirage, lui aussi
projeté dans la boule L2 de rayon `C`. Les quatre références reçoivent les
**mêmes 200 paires** pour permettre une comparaison appariée. Cela représente
`4 tailles × 4 méthodes × 200 = 3 200` observations aléatoires.

Deux paires déterministes sont ensuite ajoutées pour chaque couple
`(n, méthode)` :

1. une cohorte alignée sur un axe et placée près d'un changement de majorité,
   destinée à stresser CM et RFA ;
2. une cohorte placée aux frontières conservées/rejetées du trimmed mean.

Ces 32 observations supplémentaires donnent **3 232 observations** au total.
Les cohortes de stress sont importantes : un échantillonnage gaussien typique
peut suggérer une décroissance empirique avec `n` tout en manquant une cohorte
voisine défavorable.

Pour chaque paire, S0.5 mesure :

```text
||F(U) - F(U')||_2
||A(U) - A(U')||_2
```

où `A` est la chaîne déterministe SC-FAR complète après clipping user-level,
calcul de la référence, scores bornés, tilting contrôlé et combinaison
pondérée des updates. Aucun bruit DP n'est ajouté dans cet audit.

### Résultats complets

Dans le tableau, **F random** est le déplacement maximal de la référence parmi
les 200 paires aléatoires. **F stress** est le maximum des deux cohortes
construites. **Agrégat** est le déplacement maximal de `A`, toutes cohortes
confondues.

| n | Référence | F random, max | F stress, max | Agrégat, max |
|---:|---|---:|---:|---:|
| 5  | moyenne | 0.3302 | 0.4000 | 0.3366 |
| 5  | CM      | 0.5771 | 2.0000 | 0.5600 |
| 5  | trMean  | 0.3952 | 0.6667 | 0.3390 |
| 5  | RFA     | 0.3558 | 2.0000 | 0.5600 |
| 10 | moyenne | 0.1642 | 0.2000 | 0.1892 |
| 10 | CM      | 0.2545 | 1.0000 | 0.2400 |
| 10 | trMean  | 0.1933 | 0.3333 | 0.1646 |
| 10 | RFA     | 0.1654 | 1.0000 | 0.2400 |
| 20 | moyenne | 0.0809 | 0.1000 | 0.0997 |
| 20 | CM      | 0.1378 | 1.0000 | 0.2667 |
| 20 | trMean  | 0.0899 | 0.1667 | 0.0858 |
| 20 | RFA     | 0.0822 | 1.0000 | 0.2667 |
| 40 | moyenne | 0.0402 | 0.0500 | 0.0511 |
| 40 | CM      | 0.0784 | 1.0000 | 0.3000 |
| 40 | trMean  | 0.0449 | 0.0833 | 0.0448 |
| 40 | RFA     | 0.0408 | 0.9999 | 0.3000 |

### À quoi comparer ces valeurs ?

La quantité `2C/n` sert ici de **cible de stabilité en `1/n`**, et non de
borne théorique déjà démontrée pour toutes les méthodes. Avec `C = 1`, elle
vaut successivement `0.4`, `0.2`, `0.1` et `0.05`. Le ratio

```text
déplacement de référence sous stress / (2C/n)
```

permet de voir si le déplacement reste du même ordre que `C/n` lorsque `n`
augmente. Un ratio borné par une constante indépendante de `n` est compatible
avec une loi `O(C/n)` ; un ratio qui croît avec `n` ne l'est pas.

| Référence | Ratio n=5 | Ratio n=10 | Ratio n=20 | Ratio n=40 | Lecture |
|---|---:|---:|---:|---:|---|
| moyenne | 1.00 | 1.00 | 1.00 | 1.00 | Certificat exact `2C/n` |
| CM | 5.00 | 5.00 | 10.00 | 20.00 | Le ratio croît ; le stress ne décroît pas en `1/n` |
| trMean | 1.67 | 1.67 | 1.67 | 1.67 | Compatible empiriquement avec `O(C/n)` dans ce stress |
| RFA | 5.00 | 5.00 | 10.00 | 20.00 | Le ratio croît ; le stress ne décroît pas en `1/n` |

Cette comparaison ne transforme pas `2C/n` en borne de CM, trMean ou RFA.
Elle sert à diagnostiquer l'échelle. Pour `trMean`, les valeurs de stress
suivent ici

```text
2C / (n - 2f),
```

qui est bien de l'ordre de `C/n` lorsque `f/n` reste strictement inférieur à
`1/2`. `trMean` ne doit donc pas être déclaré mauvais candidat sur la seule
base de S0.5. Il reste un candidat théorique secondaire, mais il lui manque
encore dans notre cadre une preuve vectorielle globale, avec la dépendance en
dimension explicitée, puis le transport de sa stabilité à toute la chaîne
SC-FAR. Tant que cette preuve n'est pas fermée, le mécanisme conserve la borne
prudente `2C` lorsqu'il utilise `trMean`.

### Certificats réellement évalués

Pour la moyenne arithmétique,

```text
F_mean(U) = (1/n) sum_i u_i,
```

les termes communs s'annulent sous replace-one, d'où

```text
||F_mean(U) - F_mean(U')||_2
  = ||u_k - u'_k||_2 / n
  <= 2C/n.
```

Les cohortes de stress atteignent exactement `2C/n` : respectivement 0.4,
0.2, 0.1 et 0.05 pour les quatre valeurs de `n`. Les 800 vérifications de la
moyenne respectent cette borne.

Pour l'agrégat final, chaque sortie est une combinaison convexe d'updates dont
la norme est au plus `C`. Par convexité de la boule L2,

```text
||A(U)||_2 <= C,
||A(U')||_2 <= C,
```

puis l'inégalité triangulaire donne la borne globale

```text
||A(U) - A(U')||_2 <= 2C.
```

Cette borne a été respectée dans les 3 232 cas. Elle est volontairement
conservatrice et ne constitue pas un certificat `O(C/n)`.

CM, trMean et RFA ne reçoivent aucun certificat de référence dans S0.5. Les
valeurs observées pour ces méthodes sont des diagnostics de falsification, pas
des bornes universelles.

### Pourquoi robustesse byzantine et stabilité replace-one sont distinctes

Une garantie de robustesse byzantine affirme typiquement que, sous une borne
sur la fraction d'attaquants et des hypothèses sur la dispersion des clients
honnêtes, la sortie reste proche d'un centre honnête. Elle répond à la
question : « l'estimateur résiste-t-il à plusieurs contributions adverses dans
ce régime ? »

La stabilité replace-one demande au contraire une borne uniforme sur

```text
sup_{U ~ U'} ||F(U) - F(U')||_2
```

pour **toutes** les cohortes voisines autorisées. Elle répond à la question :
« quelle variation maximale un utilisateur entier peut-il produire ? » Les
hypothèses, les quantificateurs et l'objet borné ne sont donc pas les mêmes.

CM et RFA illustrent concrètement la différence. Près d'une égalité de
majorité entre deux amas situés à `-C e_1` et `+C e_1`, le remplacement d'un
seul client peut faire changer le centre sélectionné. Pour `n=40`, le
déplacement de leur référence reste proche de `C`, alors que `2C/n = 0.05`.
Ces méthodes peuvent néanmoins être utiles contre des Byzantins dans leurs
régimes théoriques respectifs. Leur robustesse ne fournit simplement pas, à
elle seule, le certificat de stabilité requis pour calibrer le bruit de
SC-FAR-DP à l'échelle `C/n`.

S0.5 ne prouve pas non plus, à lui seul, que `F_CC` est une bonne référence
robuste. Son rôle est plus limité : il falsifie l'hypothèse selon laquelle une
référence robuste standard serait automatiquement replace-one stable en
`O(C/n)`. `F_CC` reste le candidat principal parce que sa construction avec
ancre publique possède séparément le certificat analytique
`2 min(C,tau)/n`. Il reste encore à valider son compromis robustesse-biais dans
S1.

### Exécution Toubkal et artefacts

S0.5 peut être soumis immédiatement sur CPU :

```bash
cd "$HOME/fedlab_zmq/slurm_logs"
sbatch \
  --account=MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU \
  ../hpc/run_s0_5_reference_comparators_cpu.slurm
```

Le job écrit sous
`$WORK_ROOT/results/scfar/sensitivity_s0_5`. L'exécution locale détaillée est
disponible dans :

```text
results/scpfar/sensitivity_s0_5/replace_one_comparators.csv
results/scpfar/sensitivity_s0_5/summary.json
```

## Soumission des matrices prêtes

Les invariants sont contrôlés par `scripts/run_scfar_paper1.py --validate` :
full participation, aucun dropout, aucune clé partial, `scfar_dp` pour les bras
SC-FAR, clipping user-level positif, `alpha_bound_policy=error`, seeds séparés
et absence de chevauchement entre honest outliers et Byzantins.

| Matrice | Nombre de tâches | Indices Slurm |
|---|---:|---:|
| S1 | 1 908 | `0-1907` |
| S2 | 360 | `0-359` |
| S3 | 1 224 | `0-1223` |
| S4 | 720 | `0-719` |

Exemple CPU pour S2 :

```bash
cd "$HOME/fedlab_zmq/slurm_logs"
sbatch \
  --account=MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU \
  --array=0-359%8 \
  --export=ALL,SCFAR_MATRIX=s2_full_update_ablations.yaml \
  ../hpc/run_scfar_paper1_cpu.slurm
```

Avant les campagnes complètes, soumettre un pilote court en ajoutant
`SCFAR_PILOT_ROUNDS=1`. S5 sera ensuite réduit aux régimes retenus par S1 à
S4 ; il ne doit pas recopier toutes les grilles exploratoires.
