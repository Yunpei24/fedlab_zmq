# Exécuter DMD sur Toubkal

Ce workflow exécute la confirmation **non privée** de DMD-CB et de
DMD-CB+USV-0.25 avec l'infrastructure native de FedLab. Toutes les méthodes
sont chargées depuis le registre `algorithms/`, puis exécutées par
`run_experiment.py`. Aucun fichier placé sous `research/` n'est nécessaire sur
Toubkal.

`run_experiment.py` simule les clients séquentiellement dans un processus. Il
réutilise néanmoins les mêmes contrats `FLAlgorithm`, le même registre, les
mêmes modèles, les mêmes partitions, les mêmes métriques et le même format
`metrics.json` que le framework. Il sert donc à confirmer proprement la
contribution algorithmique. Le déploiement multi-processus avec les sockets ZMQ
constitue une validation système ultérieure.

Il ne faut pas présenter Phase A comme une expérience DMD-DP. Le passage à la
DP reste une campagne séparée qui nécessite le clipping, la protection du
rapport de marge et un accountant commun aux baselines.

## Organisation scientifique

Chaque tâche SLURM entraîne une seule méthode et écrit dans un dossier unique.
Les méthodes d'un même couple `(dataset, alpha, seed, participation)` partagent
les mêmes seeds de partition, d'initialisation et de participation, mais aucun
job n'écrit dans le dossier d'un autre job.

| Stage | Données et architectures | Alpha | Seeds | Tâches |
|---|---|---|---|---:|
| `smoke` | CIFAR-10/ResNet-18-GN et EMNIST/ByClass/CNN-GN | 0.1 | 101 | 24 |
| `phase_a_full` | les deux scénarios, participation 10/10 | 0.1, 0.3, 1.0 | 101-105 | 180 |
| `phase_a_partial_no_dropout` | 5/10 sélectionnés, aucun abandon | 0.1, 0.3 | 101-105 | 120 |
| `phase_a_partial_dropout` | 5/10 sélectionnés, 1 abandon, 4 survivants | 0.1, 0.3 | 101-105 | 120 |

Les seeds 101 à 105 sont indépendantes des seeds 91, 92, 93 et 24 utilisées
pendant la sélection exploratoire d'USV-0.25.

Les six bras sont FedAvg, FedFair-loss non privé, l'approximation fédérée de
TERM, Margin-Mean pondéré par exemples, DMD-CB et DMD-CB+USV-0.25. Tous les
bras entraînent sur le même sous-ensemble de train. Le jeu d'ancrage local fixe
est séparé avant le premier round. TERM et les méthodes DMD l'utilisent pour
mesurer leur signal pré-round. TERM n'est pas le gradient central exact de
l'objectif log-sum-exp; c'est une approximation fédérée par pondération des
updates complets.

## 1. Préparer le dépôt et les données

Le runner utilise `download=False`. Le répertoire désigné par `DATA_ROOT` doit
donc déjà contenir :

```text
DATA_ROOT/cifar-10-batches-py/data_batch_1
DATA_ROOT/EMNIST/raw/emnist-byclass-train-images-idx3-ubyte
```

Le lanceur reste indépendant de l'installation du cluster. Les variables
suivantes peuvent être fournies à `sbatch` :

```text
REPO_DIR, PYTHON_BIN, DATA_ROOT, OUTPUT_ROOT,
MODULE_SETUP, VENV_ACTIVATE, DEVICE
```

Ne pas coder en dur les modules CUDA ou le chemin du virtualenv dans le dépôt.

## 2. Valider la matrice avant la soumission

Depuis la racine du dépôt :

```bash
python3 scripts/run_dmd_toubkal.py \
  --validate \
  --stage smoke \
  --data-root /chemin/vers/data

python3 scripts/run_dmd_toubkal.py \
  --list \
  --stage phase_a_full
```

La première commande doit afficher `VALID: 24 unique tasks`; la seconde doit
énumérer 180 tâches compactes de 0 à 179.

## 3. Exécuter le smoke CUDA

```bash
sbatch --array=0-23 \
  --export=ALL,STAGE=smoke,REPO_DIR=/chemin/fedlab_zmq,DATA_ROOT=/chemin/data,OUTPUT_ROOT=/chemin/results/toubkal_dmd,VENV_ACTIVATE=/chemin/venv/bin/activate \
  hpc/run_dmd_phase_a.slurm
```

Les 24 tâches doivent produire des losses finies, un modèle, un `metrics.json`
natif et un `manifest.json`. La reprise est actuellement faite au niveau de la
tâche : une tâche complète est ignorée, tandis qu'une tâche interrompue est
relancée depuis son début. Le checkpoint round par round pourra être ajouté
séparément sans changer l'algorithme.

## 4. Lancer Phase A, puis les extensions

Après validation du smoke :

```bash
sbatch --array=0-179 \
  --export=ALL,STAGE=phase_a_full,REPO_DIR=/chemin/fedlab_zmq,DATA_ROOT=/chemin/data,OUTPUT_ROOT=/chemin/results/toubkal_dmd,VENV_ACTIVATE=/chemin/venv/bin/activate \
  hpc/run_dmd_phase_a.slurm

sbatch --array=0-119 \
  --export=ALL,STAGE=phase_a_partial_no_dropout,REPO_DIR=/chemin/fedlab_zmq,DATA_ROOT=/chemin/data,OUTPUT_ROOT=/chemin/results/toubkal_dmd,VENV_ACTIVATE=/chemin/venv/bin/activate \
  hpc/run_dmd_phase_a.slurm

sbatch --array=0-119 \
  --export=ALL,STAGE=phase_a_partial_dropout,REPO_DIR=/chemin/fedlab_zmq,DATA_ROOT=/chemin/data,OUTPUT_ROOT=/chemin/results/toubkal_dmd,VENV_ACTIVATE=/chemin/venv/bin/activate \
  hpc/run_dmd_phase_a.slurm
```

Il est préférable d'achever et d'analyser `phase_a_full` avant de lancer les
extensions partielles. Le drift temporel n'est pas encore implémenté dans ce
runner et n'est donc pas annoncé comme prêt.

## 5. Dashboard

À la fin de chaque tâche, `run_experiment.py` produit déjà le format natif du
dashboard. Le lanceur copie une vue légère de `metrics.json` et du manifeste
dans `OUTPUT_ROOT/dashboard_exports`. Le résultat scientifique original n'est
jamais modifié.

Le script ci-dessous reste utile uniquement pour convertir les anciens runs
CSV produits avant la migration native :

```bash
python3 scripts/export_dmd_to_dashboard.py \
  --input-root /chemin/vers/les/resultats \
  --output-root results/dashboard_dmd \
  --overwrite
```

Le dashboard affiche séparément accuracy, balanced accuracy, Worst-20 BA,
variance et gap de BA, déficit DMD-CB, CVaR-20, upper-semivariance et
diagnostics USV. Les distributions par client sont des métriques oracle de
recherche, pas de la télémétrie destinée à un déploiement privé.

## 6. Comment DMD est maintenant exécuté dans l'infrastructure native

### 6.1 Le contexte retardé dont DMD-CB+USV a besoin

Au round `t`, DMD-CB+USV utilise une statistique calculée au round précédent :

```text
round t-1 : profils de marge des clients
            -> déficits DMD-CB
            -> moyenne inter-clients bar_D_(t-1)

round t   : le client minimise
            CE + mu_M D_i(w) + mu_V [D_i(w)-bar_D_(t-1)]_+^2
```

La quantité `bar_D_(t-1)` est appelée **contexte DMD retardé**. Elle doit être :

1. calculée par le serveur après réception des rapports du round `t-1`;
2. conservée dans l'état serveur;
3. envoyée aux clients dans le `TRAIN_REQ` du round `t`;
4. lue par les clients avant leur optimisation locale.

Sans ce contexte, le terme d'upper-semivariance ne peut pas être calculé. Le
client retombe alors sur la cross-entropy seule.

### 6.2 Propagation causale dans `run_experiment.py`

La chaîne est maintenant explicite :

```text
client au round t
  1. reçoit w_t et le contexte construit à t-1
  2. évalue le profil de marge de w_t sur son ancre locale
  3. optimise localement CE + DMD avec le contexte retardé
  4. envoie update + profil pré-entraînement

serveur à la fin du round t
  5. agrège les updates
  6. construit le contexte DMD à partir des profils reçus
  7. retourne _server_state_updates = {dmd_round_context: ...}

run_experiment.py au round t+1
  8. fusionne cet état dans la configuration envoyée aux clients
```

L'assertion `source_round=t-1` est vérifiée côté client. Une erreur de décalage
temporel arrête le run au lieu d'exécuter silencieusement une autre méthode.
Pendant le warm-up, l'absence de contexte active uniquement la CE.

### 6.3 Jeu d'ancrage et comparabilité des baselines

`run_experiment.py` construit une séparation déterministe, approximativement
stratifiée, à l'intérieur de chaque partition client. Le train et l'ancre sont
disjoints. Tous les bras apprennent sur le même train réduit. Le profil DMD est
calculé sur l'ancre avec le modèle global avant SGD. Les seeds de séparation,
de partition et de participation sont appariées entre les méthodes.

Cette séparation évite deux erreurs : utiliser le test local pour piloter
l'entraînement, ou profiler le modèle après personnalisation locale alors que
la question DMD porte sur la qualité du modèle collaboratif reçu.

### 6.4 Ce qui reste pour le déploiement ZMQ multi-processus

Le contexte lui-même est maintenant compatible avec le hook générique d'état
serveur. La validation Toubkal demandée ici passe toutefois par
`run_experiment.py`, qui est le runner natif mono-processus. Pour un déploiement
avec de vrais workers ZMQ, il restera à construire de façon persistante le même
split train/ancre sur chaque worker et à ajouter un test de roundtrip complet
`TRAIN_REQ -> update+profil -> agrégation -> TRAIN_REQ suivant`.

Ce travail système ne change ni la fonction objectif, ni le format du contexte,
ni le registre de l'algorithme. Il vérifie que la réalisation distribuée
respecte la même sémantique.

## 7. Disponibilité du code après un clone Git

Le dossier `research/` peut rester entièrement dans `.gitignore`. Phase A
utilise seulement des fichiers versionnés du framework :

```text
algorithms/dmd/
algorithms/fedfair_loss.py
datasets/anchor_split.py
run_experiment.py
configs/dmd/toubkal_phase_a.yaml
scripts/run_dmd_toubkal.py
hpc/run_dmd_phase_a.slurm
```

Après un commit, un push et un `git pull` sur Toubkal, on contrôle le chemin
natif avec :

```bash
python3 run_experiment.py --list-algos | grep dmd
python3 scripts/run_dmd_toubkal.py --validate --stage smoke --data-root "$DATA_ROOT"
python3 scripts/run_dmd_toubkal.py --dry-run --stage smoke --job-index 0 --pilot-rounds 2 --skip-data-check
```

## 8. Passage ultérieur à DMD-DP

La confidentialité différentielle est un problème distinct de la propagation
du contexte. Même après avoir rendu ZMQ fidèle, la future Phase C doit rester
bloquée jusqu'à ce que les quatre éléments suivants soient disponibles :

1. un mécanisme DP explicite pour l'update;
2. un mécanisme DP explicite pour le profil de marge DMD;
3. un accountant composant les coûts privacy de ces deux canaux;
4. des comparaisons DP-FedAvg, FedFDP et DMD-DP au même niveau d'adjacence et au
   même budget `(epsilon, delta)`.

La raison est simple : cacher ou transporter correctement le contexte DMD ne
rend pas le mécanisme différentiellement privé. ZMQ traite le transport et
l'orchestration; la DP exige clipping, randomisation et accounting.
