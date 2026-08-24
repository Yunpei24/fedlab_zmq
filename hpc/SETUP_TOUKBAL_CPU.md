# Installation et validation CPU de FedLab ZMQ sur Toubkal

Ce guide configure le dépôt `fedlab_zmq` pour une première campagne sur la
partition CPU de Toubkal.  Les commandes lourdes ne doivent pas être exécutées
sur un nœud de connexion.

## 1. Paramètres de ce projet

```bash
export FEDLAB_REPO="$HOME/fedlab_zmq"
export FEDLAB_PROJECT="$HOME/lustre/manapy-um6p-st-msda-1wabcjwe938"
export FEDLAB_WORK="$FEDLAB_PROJECT/users/$USER/fedlab_zmq"
export FEDLAB_ACCOUNT="MANAPY-UM6P-ST-MSDA-1WABCJWE938-DEFAULT-CPU"
```

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

La version actuelle du framework demande Python 3.10 ou plus. Toubkal fournit
Python 3.12.3 et Anaconda 2025.06-1, qui conviennent au protocole.

```bash
module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n fedlab-zmq python=3.12.3 pip
conda activate fedlab-zmq

python --version
python -m pip install --upgrade pip setuptools wheel
```

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
import algorithms

print("torch:", torch.__version__)
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
  --qos=intr \
  --time=01:00:00 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=16G \
  --pty bash
```

Lorsque l'invite du nœud alloué apparaît :

```bash
hostname
echo "$SLURM_JOB_ID"

module purge
module load Anaconda3/2025.06-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fedlab-zmq

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$FEDLAB_REPO"
```

Les variables exportées avant `srun` sont normalement propagées. Si ce n'est
pas le cas, réexécuter le bloc de la section 1 dans la session interactive.

## 6. Valider le framework sur le nœud CPU

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
```

### Niveau R : expériences R1, R2 et R3

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  ../hpc/run_r1_r2_r3_cpu.slurm
```

### Niveau P : protocoles alignés sur FAR/FedFDP

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  ../hpc/run_level_p_paper_fidelity_cpu.slurm
```

### Niveau S : validation principale de SC-Partial-FAR-DP

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
  ../hpc/run_level_s_scfar_validation_cpu.slurm
```

### Niveau S : audit empirique de sensibilité replace-one

```bash
sbatch \
  --account="$FEDLAB_ACCOUNT" \
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

Si `conda activate` échoue dans un job, vérifier que le script charge bien
Anaconda et source `conda.sh`. Si une tâche reste en attente :

```bash
squeue -j <JOB_ID> -o "%.18i %.9P %.16j %.8T %.10M %.30R"
```

La dernière colonne donne habituellement la raison d'attente fournie par
Slurm, par exemple `Resources`, `Priority` ou une erreur de compte/QoS.
