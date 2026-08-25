# Exécuter DMD sur Toubkal

Ce workflow exécute la confirmation **non privée** de DMD-CB et de
DMD-CB+USV-0.25 avec l'infrastructure native de FedLab. Toutes les méthodes
sont chargées depuis le registre `algorithms/`, puis exécutées par
`run_experiment.py`. Aucun fichier placé sous `research/` n'est nécessaire sur
Toubkal.

La campagne commence volontairement sur **CPU** avec
`hpc/run_dmd_phase_a_cpu.slurm`. Ce choix permet de valider l'environnement,
les données, la matrice scientifique, la reprise et les artefacts sans dépendre
d'une allocation GPU. Le script GPU `hpc/run_dmd_phase_a.slurm` reste une
option ultérieure; il ne doit pas être utilisé en parallèle pour écrire dans le
même `OUTPUT_ROOT`.

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

## 1. Prérequis communs avec le framework

Commencer par suivre [le guide CPU général de Toubkal](../hpc/SETUP_TOUKBAL_CPU.md)
jusqu'à l'obtention d'un nœud CPU interactif. Le guide DMD réutilise sans les
renommer les quatre variables du framework :

```bash
export FEDLAB_REPO="$HOME/fedlab_zmq"
export FEDLAB_PROJECT="$HOME/lustre/manapy-um6p-st-msda-1wabcjwe938"
export FEDLAB_WORK="$FEDLAB_PROJECT/users/$USER/fedlab_zmq"
export FEDLAB_ACCOUNT="MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU"
```

Les commandes de création de l'environnement, d'acceptation des conditions
Conda, d'installation des roues CPU de PyTorch et d'installation éditable du
dépôt restent centralisées dans `hpc/SETUP_TOUKBAL_CPU.md`. Elles ne doivent pas
être dupliquées ou exécutées dans un job de campagne.

Mettre ensuite le dépôt à jour depuis le nœud de connexion :

```bash
cd "$FEDLAB_REPO"
git status --short
git fetch origin
git switch fedlab_zmq/toubkal-framework-checkpoint
git pull --ff-only
git log -1 --oneline
```

Si `git status --short` n'est pas vide, examiner les modifications avant le
`pull`; ne pas les effacer automatiquement.

Créer les répertoires conformément au guide général :

```bash
mkdir -p "$FEDLAB_WORK"/{datasets,results,checkpoints,cache}
mkdir -p "$FEDLAB_REPO/slurm_logs"

export DMD_DATA="$FEDLAB_WORK/datasets"
export DMD_OUTPUT="$FEDLAB_WORK/results/toubkal_dmd_cpu"
mkdir -p "$DMD_OUTPUT"
```

Le runner utilise `download=False`. Le répertoire désigné par `DATA_ROOT` doit
donc déjà contenir :

```text
DATA_ROOT/cifar-10-batches-py/data_batch_1
DATA_ROOT/EMNIST/raw/emnist-byclass-train-images-idx3-ubyte
```

Le script CPU accepte les variables suivantes via `sbatch --export`. Les
quatre premières sont directement dérivées des variables `FEDLAB_*` :

```text
REPO_DIR              clone Git du framework, par défaut $HOME/fedlab_zmq
PROJECT_DIR           racine Lustre du projet
WORK_ROOT             espace utilisateur sous PROJECT_DIR
DATA_ROOT             CIFAR-10 et EMNIST déjà téléchargés
OUTPUT_ROOT           résultats scientifiques CPU
DASHBOARD_OUTPUT_ROOT vues dérivées pour le dashboard
DMD_MATRIX            matrice YAML, normalement configs/dmd/toubkal_phase_a.yaml
STAGE                 smoke, phase_a_full ou une extension partielle
CONDA_ENV             environnement Conda, par défaut fedlab-zmq
PYTHON_BIN            interpréteur, par défaut python
PILOT_ROUNDS          surcharge facultative du nombre de rounds
```

Dans les commandes de soumission, utiliser donc les correspondances :

```text
REPO_DIR=$FEDLAB_REPO
PROJECT_DIR=$FEDLAB_PROJECT
WORK_ROOT=$FEDLAB_WORK
DATA_ROOT=$DMD_DATA
OUTPUT_ROOT=$DMD_OUTPUT
```

Le script force `--device cpu`; une variable `DEVICE=cuda` n'a donc aucun effet
sur ce fichier SLURM. Pour utiliser un GPU plus tard, employer le script GPU
dédié.

## 2. Valider DMD sur un nœud CPU interactif

Ne pas lancer les tests, la lecture des datasets ou un entraînement sur le
nœud de connexion. Demander un nœud interactif comme dans le guide général :

```bash
srun \
  --account="$FEDLAB_ACCOUNT" \
  --partition=compute \
  --qos=intr \
  --time=01:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=16G \
  --pty bash
```

Sur le nœud alloué :

```bash
module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fedlab-zmq

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export DMD_DATA="$FEDLAB_WORK/datasets"
export DMD_OUTPUT="$FEDLAB_WORK/results/toubkal_dmd_cpu"
cd "$FEDLAB_REPO"

python -c "import torch, yaml, zmq; print(torch.__version__, torch.cuda.is_available())"
python run_experiment.py --list-algos | grep -E 'dmd_mean|dmd_usv|fedfair_loss|term'
python -m pytest -q \
  tests/test_dmd_toubkal_launcher.py \
  tests/test_anchor_split.py \
  tests/dmd/test_contracts_and_adapter.py
```

Dans l'environnement CPU, `torch.cuda.is_available()` doit afficher `False`.
Valider ensuite la matrice :

```bash
python scripts/run_dmd_toubkal.py \
  --validate \
  --stage smoke \
  --device cpu \
  --data-root "$DMD_DATA"

python scripts/run_dmd_toubkal.py \
  --list \
  --stage phase_a_full
```

La première commande doit afficher `VALID: 24 unique tasks`; la seconde doit
énumérer 180 tâches compactes de 0 à 179.

Vérifier ensuite la commande exacte de la première tâche, sans entraînement :

```bash
python scripts/run_dmd_toubkal.py \
  --dry-run \
  --stage smoke \
  --job-index 0 \
  --device cpu \
  --data-root "$DMD_DATA" \
  --output-root "$DMD_OUTPUT"
```

Enfin, exécuter une seule tâche, un seul round, avant tout array SLURM :

```bash
python -u scripts/run_dmd_toubkal.py \
  --run \
  --stage smoke \
  --job-index 0 \
  --device cpu \
  --pilot-rounds 1 \
  --data-root "$DMD_DATA" \
  --output-root "$DMD_OUTPUT" \
  --dashboard-output-root "$DMD_OUTPUT/dashboard_exports" \
  --resume
```

Le pilote est écrit sous un arbre `pilots/rounds_1` distinct. Vérifier qu'il
produit un `metrics.json` et un `manifest.json`, puis quitter le nœud interactif
avec `exit`.

## 3. Exécuter le smoke CPU

Comme dans le guide général, conserver les petits logs SLURM sur le Home et les
résultats volumineux sur Lustre :

```bash
cd "$FEDLAB_REPO/slurm_logs"

sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --array=0-23%4 \
  --export=ALL,STAGE=smoke,REPO_DIR="$FEDLAB_REPO",PROJECT_DIR="$FEDLAB_PROJECT",WORK_ROOT="$FEDLAB_WORK",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$DMD_OUTPUT",CONDA_ENV=fedlab-zmq \
  ../hpc/run_dmd_phase_a_cpu.slurm
```

Les 24 tâches doivent produire des losses finies, un modèle, un `metrics.json`
natif et un `manifest.json`. L'array couvre deux scénarios, deux régimes de
participation et six méthodes. La limite `%4` autorise au maximum quatre tâches
simultanées; elle peut être réduite si le quota CPU ou I/O du projet l'exige.

Contrôler le job avec :

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -f slurm-dmd_phase_a_cpu-<JOB_ID>_0.out
```

Ne lancer Phase A que lorsque les 24 indices ont `COMPLETED` avec un code de
sortie nul. En cas d'échec, corriger d'abord l'environnement ou les données,
puis resoumettre le même array.

La reprise est faite au niveau de la tâche : avec `--resume`, une tâche dont les
artefacts sont complets est ignorée; une tâche interrompue est relancée depuis
le début. Il n'y a pas encore de reprise au milieu d'une trajectoire.

## 4. Lancer Phase A complète sur CPU

Après validation du smoke :

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --array=0-179%4 \
  --export=ALL,STAGE=phase_a_full,REPO_DIR="$FEDLAB_REPO",PROJECT_DIR="$FEDLAB_PROJECT",WORK_ROOT="$FEDLAB_WORK",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$DMD_OUTPUT",CONDA_ENV=fedlab-zmq \
  ../hpc/run_dmd_phase_a_cpu.slurm
```

Le script demande par tâche 8 CPU, 32 Go de mémoire et 12 heures au maximum.
L'array contient 180 tâches :

```text
2 scénarios x 3 niveaux non-IID x 5 seeds x 6 méthodes = 180 tâches
```

Le plafond `%4` représente donc au plus 32 cœurs utilisés simultanément. Il
contrôle la concurrence, pas le nombre total de tâches scientifiques.

Pour un test intermédiaire moins coûteux, `PILOT_ROUNDS` écrit dans un arbre de
pilote séparé et ne contamine pas les résultats à 150 rounds :

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --array=0-5%4 \
  --export=ALL,STAGE=phase_a_full,PILOT_ROUNDS=5,REPO_DIR="$FEDLAB_REPO",PROJECT_DIR="$FEDLAB_PROJECT",WORK_ROOT="$FEDLAB_WORK",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$DMD_OUTPUT",CONDA_ENV=fedlab-zmq \
  ../hpc/run_dmd_phase_a_cpu.slurm
```

## 5. Lancer ensuite les extensions de participation

Il est préférable d'achever et d'analyser `phase_a_full` avant de lancer les
extensions partielles :

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --array=0-119%4 \
  --export=ALL,STAGE=phase_a_partial_no_dropout,REPO_DIR="$FEDLAB_REPO",PROJECT_DIR="$FEDLAB_PROJECT",WORK_ROOT="$FEDLAB_WORK",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$DMD_OUTPUT",CONDA_ENV=fedlab-zmq \
  ../hpc/run_dmd_phase_a_cpu.slurm

sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --array=0-119%4 \
  --export=ALL,STAGE=phase_a_partial_dropout,REPO_DIR="$FEDLAB_REPO",PROJECT_DIR="$FEDLAB_PROJECT",WORK_ROOT="$FEDLAB_WORK",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$DMD_OUTPUT",CONDA_ENV=fedlab-zmq \
  ../hpc/run_dmd_phase_a_cpu.slurm
```

Le premier stage isole l'effet d'une cohorte réduite. Le second ajoute un
abandon avant l'envoi du profil et avant l'entraînement local. Le drift
temporel n'est pas encore implémenté dans ce runner et n'est donc pas annoncé
comme prêt.

## 6. Option GPU ultérieure

Après validation CPU, une réplication GPU peut utiliser
`hpc/run_dmd_phase_a.slurm`. Utiliser un `OUTPUT_ROOT` distinct, par exemple
`toubkal_dmd_gpu`, afin d'éviter toute collision avec les résultats CPU :

```bash
sbatch \
  --array=0-23 \
  --export=ALL,STAGE=smoke,DEVICE=cuda,REPO_DIR="$FEDLAB_REPO",DATA_ROOT="$DMD_DATA",OUTPUT_ROOT="$FEDLAB_WORK/results/toubkal_dmd_gpu",VENV_ACTIVATE=/chemin/venv/bin/activate \
  ../hpc/run_dmd_phase_a.slurm
```

Le changement CPU/GPU ne change pas la matrice scientifique. Il peut produire
de faibles différences numériques; les comparaisons principales doivent donc
rester appariées à l'intérieur d'un même backend.

## 7. Dashboard

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

## 8. Comment DMD est maintenant exécuté dans l'infrastructure native

### 8.1 Le contexte retardé dont DMD-CB+USV a besoin

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

### 8.2 Propagation causale dans `run_experiment.py`

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

### 8.3 Jeu d'ancrage et comparabilité des baselines

`run_experiment.py` construit une séparation déterministe, approximativement
stratifiée, à l'intérieur de chaque partition client. Le train et l'ancre sont
disjoints. Tous les bras apprennent sur le même train réduit. Le profil DMD est
calculé sur l'ancre avec le modèle global avant SGD. Les seeds de séparation,
de partition et de participation sont appariées entre les méthodes.

Cette séparation évite deux erreurs : utiliser le test local pour piloter
l'entraînement, ou profiler le modèle après personnalisation locale alors que
la question DMD porte sur la qualité du modèle collaboratif reçu.

### 8.4 Ce qui reste pour le déploiement ZMQ multi-processus

Le contexte lui-même est maintenant compatible avec le hook générique d'état
serveur. La validation Toubkal demandée ici passe toutefois par
`run_experiment.py`, qui est le runner natif mono-processus. Pour un déploiement
avec de vrais workers ZMQ, il restera à construire de façon persistante le même
split train/ancre sur chaque worker et à ajouter un test de roundtrip complet
`TRAIN_REQ -> update+profil -> agrégation -> TRAIN_REQ suivant`.

Ce travail système ne change ni la fonction objectif, ni le format du contexte,
ni le registre de l'algorithme. Il vérifie que la réalisation distribuée
respecte la même sémantique.

## 9. Disponibilité du code après un clone Git

Le dossier `research/` peut rester entièrement dans `.gitignore`. Phase A
utilise seulement des fichiers versionnés du framework :

```text
algorithms/dmd/
algorithms/fedfair_loss.py
datasets/anchor_split.py
run_experiment.py
configs/dmd/toubkal_phase_a.yaml
scripts/run_dmd_toubkal.py
hpc/run_dmd_phase_a_cpu.slurm
hpc/run_dmd_phase_a.slurm
```

Après un commit, un push et un `git pull` sur Toubkal, on contrôle le chemin
natif avec :

```bash
python run_experiment.py --list-algos | grep dmd
python scripts/run_dmd_toubkal.py --validate --stage smoke --device cpu --data-root "$DATA_ROOT"
python scripts/run_dmd_toubkal.py --dry-run --stage smoke --job-index 0 --device cpu --pilot-rounds 2 --skip-data-check
```

## 10. Passage ultérieur à DMD-DP

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
