# Installation et validation CPU de FedLab ZMQ sur Toubkal

Ce guide configure le dépôt `fedlab_zmq` pour une première campagne sur la
partition CPU de Toubkal.  Les commandes lourdes ne doivent pas être exécutées
sur un nœud de connexion.

## 1. Paramètres de ce projet

```bash
export FEDLAB_REPO="$HOME/fedlab_zmq"
export FEDLAB_PROJECT="$HOME/lustre/manapy-um6p-st-msda-1wabcjwe938"
export FEDLAB_WORK="$FEDLAB_PROJECT/users/$USER/fedlab_zmq"
# Compte recommandé par le support tant que le crédit DEFAULT-CPU est épuisé.
export FEDLAB_ACCOUNT="manapy-um6p-st-msda-1wabcjwe938-low-cpu"
export FEDLAB_QOS="low-cpu"
export PYTHONNOUSERSITE=1
```

Le support a confirmé que le solde `default-cpu` est épuisé et autorise
l'utilisation du compte `low-cpu` avec la QoS `low-cpu` (jobs non décomptés).
Prévenir le PI pour le renouvellement du crédit. Le support signale aussi un
problème avec la nouvelle version de Slurm : l'affichage `mybalance` observé
ne suffit donc pas à confirmer un crédit disponible.

Ne pas combiner le compte `low-cpu` avec `--qos=intr`. Après renouvellement,
le couple pour une session interactive sur le compte standard sera
`default-cpu` / `intr`; vérifier séparément la QoS autorisée pour les campagnes
batch. Les limites de durée et de ressources restent celles de la QoS choisie.

Vérifier les chemins avant toute installation :

```bash
test -d "$FEDLAB_REPO"
test -d "$FEDLAB_PROJECT/users/$USER"
df -hT "$FEDLAB_PROJECT/users/$USER"
mybalance
```

Créer les répertoires de travail volumineux sur Lustre et les petits journaux
sur le Home sauvegardé :

```bash
mkdir -p "$FEDLAB_WORK"/{datasets,results,checkpoints,cache}
mkdir -p "$FEDLAB_REPO/slurm_logs"
```

## 2. Mettre à jour le dépôt

Depuis le nœud de connexion :

```bash
cd "$FEDLAB_REPO"
git status --short
git fetch origin
git switch fedlab_zmq/toubkal-framework-checkpoint
git pull --ff-only
git log -1 --oneline
```

Si `git status --short` affiche des modifications, les examiner avant le
`pull`; ne pas les supprimer automatiquement.

## 3. Créer l'environnement Conda

La version actuelle du framework demande Python 3.10 ou plus. Ce guide crée
un environnement dédié Python 3.12.3 à partir du module Anaconda 2025.06-1.
Charger le module ne suffit pas : son Python de base peut être en version 3.13.
Si `fedlab-zmq` existe déjà (`conda env list`), ne pas le recréer : l'activer.

```bash
module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"

# Si nécessaire, après lecture et acceptation des conditions par l'utilisateur :
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -y -n fedlab-zmq python=3.12.3 pip
conda activate fedlab-zmq
export PYTHONNOUSERSITE=1

python --version
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip setuptools wheel
```

Le chemin de Python doit appartenir à l'environnement `fedlab-zmq`, pas au
Python de base du module. `PYTHONNOUSERSITE=1` empêche de charger les paquets
installés dans `~/.local`, qui peuvent être incompatibles. Ne pas installer
avec `pip --user` dans cet environnement; utiliser `python -m pip`.

Installer d'abord les roues CPU de PyTorch afin de ne pas télécharger les
dépendances CUDA pour cette première phase :

```bash
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Installer ensuite les dépendances du dépôt et le paquet en mode éditable :

```bash
cd "$FEDLAB_REPO"
python -m pip install -r requirements.txt
python -m pip install -e .
```

Le second appel ne doit pas remplacer les roues CPU de PyTorch si les versions
installées satisfont déjà les pins de `requirements.txt`. Le vérifier avec :

```bash
python -m pip show torch torchvision
```

## 4. Vérification légère sur le nœud de connexion

Cette étape ne lance aucun entraînement :

```bash
python - <<'PY'
import torch
import torchvision
import PIL
import algorithms

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("Pillow:", PIL.__version__, PIL.__file__)
print("cuda disponible:", torch.cuda.is_available())
print("threads CPU:", torch.get_num_threads())
print("import algorithms: OK")
PY
```

Pour l'environnement CPU, `torch.cuda.is_available()` doit être `False`.

## 5. Obtenir un nœud CPU interactif

Ne pas lancer les tests ou le téléchargement des datasets sur le login node.

```bash
srun \
  --account="$FEDLAB_ACCOUNT" \
  --partition=compute \
  --qos="$FEDLAB_QOS" \
  --time=01:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=16G \
  --pty bash
```

Pendant l'attente, consulter l'état depuis un autre terminal connecté à Toubkal :

```bash
squeue -u "$USER" \
  -o "%.18i %.9P %.16j %.8T %.10M %.10l %.6D %.40R"
```

```bash
sacctmgr -nP show assoc where user="$USER" \
  format=Account,Partition,QOS,DefaultQOS
```

Pour annuler volontairement une demande, remplacer `<JOB_ID>` par son numéro :

```bash
scancel <JOB_ID>
```

Lorsque l'invite du nœud alloué apparaît :

```bash
hostname
echo "$SLURM_JOB_ID"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fedlab-zmq
export PYTHONNOUSERSITE=1
python --version
python -c "import sys; print(sys.executable)"
python -c "from PIL import Image; import torch, torchvision; print('Imports OK')"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$FEDLAB_REPO"
```

Les variables exportées avant `srun` sont normalement propagées. Si ce n'est
pas le cas, réexécuter le bloc de la section 1 dans la session interactive.

## 6. Valider le framework sur le nœud CPU

Si l'import précédent échoue, arrêter ici et consulter la section 10.
Toujours lancer `python -m pytest`, et non un exécutable `pytest` qui pourrait
appartenir à un autre environnement.

Tests ciblés des références FAR/DP et de SC-Partial-FAR-DP :

```bash
python -m pytest -q \
  tests/test_far_reference_components.py \
  tests/test_sc_partial_far_dp.py
```

Valider les matrices R1/R2/R3 sans entraînement :

```bash
python scripts/run_internship_far_fedfdp.py --validate --lane faithful
```

La sortie attendue contient :

```text
Selected: 135; currently unavailable: 0
```

## 7. Pilote CPU minimal

Exécuter une seule tâche, un seul round et un seul mini-batch. Les résultats du
pilote sont isolés des résultats scientifiques :

```bash
python -u scripts/run_internship_far_fedfdp.py \
  --run \
  --lane faithful \
  --scenario exp1_fairness_no_attack \
  --job-index 0 \
  --device cpu \
  --pilot-rounds 1 \
  --pilot-local-batches 1 \
  --data-root "$FEDLAB_WORK/datasets" \
  --output-root "$FEDLAB_WORK/results/pilots/r1_cpu"
```

Quitter ensuite la session interactive :

```bash
exit
```

## 8. Soumettre les campagnes CPU

Placer les petits logs SLURM dans le dépôt sur le Home :

```bash
cd "$FEDLAB_REPO/slurm_logs"
export PYTHONNOUSERSITE=1
```

Les commandes suivantes utilisent le couple compte/QoS de la section 1.
Les scripts activent leur environnement Conda; l'export `PYTHONNOUSERSITE`
est transmis avec l'environnement de soumission. Une allocation interactive
`RUNNING` ne signifie pas qu'une campagne d'entraînement est déjà lancée.

### Niveau R : expériences R1, R2 et R3

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --qos="$FEDLAB_QOS" \
  ../hpc/run_r1_r2_r3_cpu.slurm
```

### Niveau P : protocoles alignés sur FAR/FedFDP

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --qos="$FEDLAB_QOS" \
  ../hpc/run_level_p_paper_fidelity_cpu.slurm
```

### Niveau S : validation principale de SC-Partial-FAR-DP

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --qos="$FEDLAB_QOS" \
  ../hpc/run_level_s_scfar_validation_cpu.slurm
```

### Niveau S : audit empirique de sensibilité replace-one

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  --qos="$FEDLAB_QOS" \
  ../hpc/run_level_s_sensitivity_audit.slurm
```

## 9. Suivre ou arrêter les tâches

```bash
squeue -u "$USER"
```

Afficher un job :

```bash
scontrol show job <JOB_ID>
```

Arrêter explicitement une campagne :

```bash
scancel <JOB_ID>
```

Les résultats scientifiques sont écrits sous :

```text
$FEDLAB_WORK/results/
```

Les lanceurs R utilisent `--resume`; une tâche déjà terminée et munie de son
`metrics.json` n'est pas recalculée lors d'une nouvelle soumission.

## 10. Diagnostic rapide

### Erreur `GLIBCXX_3.4.29` pendant les imports

L'erreur observée provient de Pillow : sa dépendance `libLerc.so.4` charge
`/lib64/libstdc++.so.6`, qui ne fournit pas le symbole requis. Les traces
montrent alors le Python 3.13 de base d'Anaconda et `torchvision` dans
`~/.local/lib/python3.13`. Ce n'est pas encore un échec des tests algorithmiques :
leur collecte est interrompue par un problème de bibliothèques.

Sur le nœud alloué, commencer par réactiver l'environnement attendu :

```bash
module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
conda activate fedlab-zmq
export PYTHONNOUSERSITE=1
python --version
python -c "import sys, site; print(sys.executable); print('user-site actif:', site.ENABLE_USER_SITE)"
python -c "from PIL import Image; import torch, torchvision; print('Imports OK')"
```

Si l'environnement n'existe pas, reprendre la section 3. Si les imports passent,
relancer les tests de la section 6. Sinon, conserver la trace complète et
recueillir ces informations pour le diagnostic ou le support :

```bash
module list
conda list -n fedlab-zmq
python -m pip show Pillow torch torchvision
echo "$CONDA_PREFIX"
echo "$LD_LIBRARY_PATH"
```

L'activation correcte est une première vérification, pas une garantie de
résolution de toute incompatibilité C++. Ne pas remplacer les bibliothèques
système, supprimer `~/.local` ou modifier globalement `LD_LIBRARY_PATH` sans
identifier la bibliothèque effectivement chargée dans le bon environnement.

### Job en attente et solde du compte

Si `conda activate` échoue dans un job, vérifier que le script charge bien
Anaconda et source `conda.sh`. Si une tâche reste en attente :

```bash
squeue -j <JOB_ID> -o "%.18i %.9P %.16j %.8T %.10M %.30R"
```

La dernière colonne donne habituellement la raison d'attente fournie par
Slurm, par exemple `Resources`, `Priority` ou une erreur de compte/QoS.

`AssocGrpCPUMinutesLimit` indique une limite agrégée de CPU-minutes de
l'association Slurm, éventuellement héritée d'un compte parent. Un usage
personnel nul ne prouve pas que le compte du projet dispose encore de crédit.
Dans notre cas, le support a confirmé l'épuisement du compte `default-cpu`.
Utiliser le couple `low-cpu` / `low-cpu` autorisé en section 1 plutôt que
d'attendre que des nœuds se libèrent sur la même demande bloquée.

Références : [raisons d'attente Slurm](https://slurm.schedmd.com/job_reason_codes.html)
et [QoS Toubkal](https://toubkal.gitlab.io/docs/user-guide/qualities-of-service.html#low-cpu).
