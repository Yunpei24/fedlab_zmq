# FAR sous DP — lancement sur Toubkal CPU premium

Cette voie est **CPU**, distincte des anciens lanceurs qui imposent MPS. Elle
exécute un gradient de batch par client et par tour, puis FAR côté serveur.
Un job SLURM = un entraînement complet ; les 10 clients sont simulés
séquentiellement dans ce processus. Les jobs sont parallélisés entre eux par
SLURM. Ce n'est pas un déploiement ZMQ à dix processus.

L'association premium a été confirmée le 17 septembre 2026 : compte
`manapy-1wabcjwe938-premium-cpu`, QoS `premium-cpu`, partitions `compute` et
`himem`. Ce guide utilise `compute`. **La mise à jour de ce guide ne soumet aucun job.**

## 1. Fichiers et protocole

- Matrice : `configs/ldp_gradient_far/far_dp_effect_toubkal_cpu.yaml`.
- Lanceur CPU : `scripts/run_far_dp_effect_cpu.py`.
- Job : `hpc/run_far_dp_effect_cpu.slurm`.
- Soumission bornée : `hpc/submit_far_dp_effect_cpu.sh`.
- Choix après calibration : `configs/ldp_gradient_far/far_dp_effect_toubkal_lock.template.yaml`.
- Tests : `tests/test_far_dp_effect_cpu.py`.

Les sources importées, dont `privacy/far_dp_effect_mc.py`, doivent aussi être
présentes sur le cluster. Transférer la version courante du dépôt, pas seulement
le YAML. L'archive autonome minimale `output/FAR_DP_Toubkal_CPU_Bundle.tar.gz`
contient les sources nécessaires, sans datasets ni résultats ni clés de simulation.
La décompresser dans un nouveau répertoire de travail avant de suivre ce guide.
Ne pas écraser un dépôt ou une campagne en cours : utiliser un répertoire distinct.

| Étape | Nombre de jobs par défaut | Rôle |
|:--|--:|:--|
| `smoke` | 18 | 2 tours ; vérifier CPU, données, DP, références et sorties |
| `calibration` | 72 | B = 300/1 200 ; η = 0,05/0,2 ; 600 tours ; 3 seeds de calibration |
| `confirmation` | 88 920 | Produit complet à 10 seeds × 3 répétitions, uniquement après choix explicites |

Le smoke utilise les 100 premiers exemples de chaque partition de validation,
pas les 1 200 utilisés dans la campagne. Ses accuracies ne sont pas des résultats
scientifiques à mélanger aux autres étapes. La calibration n'évalue jamais le
test final. Elle comprend trois contrôles : sans DP sans clip local, sans DP
avec C = 8, DP ε = 4 avec C = 8 ; sans clipping serveur, α = 0. Elle détermine
d'abord une échelle de batch/pas/horizon et fournit des diagnostics de normes.
Elle ne prouve pas la convergence de tous les α ou le meilleur rayon F_CC.

La confirmation conserve les 15 α demandés (zéro inclus), les quatre références,
C = 2/8/16, ε = 1/4/8 et sans DP, avec/sans clipping serveur, plus les contrôles
sans clipping local non privés. À α = 0, une seule trajectoire uniforme est
exécutée plutôt que quatre duplicatas. **F_CC, RFA, CM et trMean sont ici des
références dans FAR, pas quatre agrégations directes.** CM/trMean sont simples,
sans NNM/bucketing. Le paramètre trMean f = 2 n'introduit pas deux attaquants :
aucune attaque n'est active dans cette étude.

Les seeds externes fixent partition, train/validation et modèle initial.
Les répétitions internes changent batches/bruits. Le modèle initial ne change
pas entre MC d'une même seed. Flux séparés par client et tour, indépendants de
α/F/C/ε/U. La reprise conserve les mêmes tirages. L'appariement ne signifie
pas que les modèles restent identiques après des mises à jour différentes.

## 2. Environnement et variables

Exécuter les calculs/tests dans une allocation CPU, pas sur le nœud de connexion.
Utiliser l'environnement PyTorch CPU du projet si déjà installé. Le script batch
ne télécharge ni packages ni datasets. Pour activer un environnement Conda dans
le job, `CONDA_ENV` est optionnel ; sinon `PYTHON_BIN` doit pointer sur le bon Python.

```bash
export FEDLAB_REPO="$HOME/fedlab_zmq"
export FEDLAB_PROJECT="$HOME/lustre/manapy-um6p-st-msda-1wabcjwe938"
export FEDLAB_WORK="$FEDLAB_PROJECT/users/$USER/fedlab_zmq"
export FAR_WORK="$FEDLAB_WORK/results/far_dp_effect_cpu_v1"
export FAR_DATA_ROOT="$FEDLAB_WORK/datasets"
export FEDLAB_ACCOUNT="manapy-1wabcjwe938-premium-cpu"
export FEDLAB_PARTITION="compute"
export FEDLAB_QOS="premium-cpu"

export CONDA_ENV="fedlab-zmq"
export CONDA_MODULE="Anaconda3/2025.06-1"
export PYTHON_BIN="python"
export PYTHONNOUSERSITE=1
cd "$FEDLAB_REPO"
```

Le compte de facturation change, pas automatiquement le chemin Lustre du projet.
Vérifier que ces répertoires sont accessibles. Si l'archive a été extraite dans
un nouveau dossier, adapter seulement `FEDLAB_REPO`. Ne pas réutiliser un root
de campagne dont le manifeste a été préparé avec une autre version du code.
Le wrapper utilise aussi premium/compute par défaut, mais une variable exportée
antérieurement (par exemple `low-cpu`) reste prioritaire : réexécuter ce bloc.

Avec un venv : ne pas définir `CONDA_ENV`, et définir par exemple
`PYTHON_BIN=/chemin/absolu/venv/bin/python`. L'environnement doit être activé
également pour les commandes de préparation, pas seulement dans le job.
Le module Conda indiqué provient des scripts existants ; vérifier sa disponibilité.

Vérifier ses allocations avec les outils autorisés du cluster. Sur une installation
qui permet cette requête, on peut consulter :

```bash
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS
sacctmgr -nP show qos where name=premium-cpu \
  format=Name,Priority,MaxWall,MaxTRESPJ,MaxTRESPU,MaxJobsPU
sinfo -o '%P %a %l %c %m'
```

Ces commandes sont en lecture seule, mais leur disponibilité dépend de Toubkal.
Si refusées, demander les paramètres au support ; ne pas contourner la restriction.

La sortie premium communiquée indique une priorité QoS de 10000, une durée
maximale de **36 h par job**, et des plafonds par utilisateur de **3584 CPU et
64 nœuds**. Ce ne sont pas des ressources réservées : disponibilité, limites
de partition, mémoire et autres associations restent applicables. Les champs
vides ne prouvent pas un accès illimité. `mybalance` ayant présenté un affichage
incohérent sur l'ancien compte, ne pas engager toute la grille sur cette seule base.

### Allocation interactive et activation effective

Depuis le login node, après le bloc de variables :

```bash
srun --account="$FEDLAB_ACCOUNT" --partition="$FEDLAB_PARTITION" \
  --qos="$FEDLAB_QOS" --time=01:00:00 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --pty bash
```

Une fois le nœud alloué, activer aussi Conda pour les tests et la préparation :

```bash
module purge
module load "$CONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
cd "$FEDLAB_REPO"
python --version
python -c "import sys; print(sys.executable)"
python -c "from PIL import Image; import torch, torchvision; print('Imports OK')"
mkdir -p "$FAR_WORK" "$FAR_DATA_ROOT"
```

L'environnement du guide commun est Python **3.12.3**. Ne pas utiliser le
Python 3.13 de base d'Anaconda ou un `pytest` venant de `~/.local`.
Si l'environnement n'existe pas, suivre la section 3 de
[SETUP_TOUKBAL_CPU.md](SETUP_TOUKBAL_CPU.md), en conservant les variables premium
ci-dessus. L'erreur `GLIBCXX_3.4.29` rencontrée à l'import de Pillow est décrite
dans sa section 10. Ne pas lancer la campagne tant que les imports échouent.
Le job batch purge les modules, active Conda si demandé et vérifie les imports
avant l'entraînement. Aucun changement global de bibliothèque système n'est fait.

### Dépendances minimales

Python 3.12 recommandé (version locale testée : 3.12.4) ; PyTorch/torchvision
compatibles avec `torch.func`, NumPy, PyYAML, pytest pour les tests. L'archive
contient `hpc/requirements-far-dp-cpu.txt`, qui fixe les versions testées localement.
Installer les versions CPU compatibles avec l'environnement du cluster, puis
garder **exactement les mêmes versions et le même nombre de threads pour reprendre
un run**. Les versions sont enregistrées ; une reprise incompatible est refusée.
CPU Toubkal et MPS Mac ne sont pas censés être identiques bit à bit.

## 3. Données — une fois, avant les arrays

Si les datasets sont déjà présents, utiliser leur répertoire parent contenant
`MNIST/raw` et `FashionMNIST/raw`. Sinon, depuis un environnement où le
téléchargement est autorisé :

```bash
"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py download --data-root "$FAR_DATA_ROOT"
```

Si Toubkal ne permet pas Internet, transférer ces deux répertoires de datasets.
Les jobs utilisent `download=False` et échouent explicitement si les fichiers
manquent. Aucun téléchargement concurrent par les jobs.

Pour chaque client : 4 800 exemples train, 1 200 validation, 1 000 test.
La partition client-Dirichlet équilibrée utilise β = 0,1. Les profils de labels
sont déterministes et partagés entre train/test via le partitionneur existant.
La taille publique employée par l'accountant est **4 800**, pas 6 000.

## 4. Tests et smoke

Dans une allocation CPU :

```bash
"$PYTHON_BIN" -m pytest tests/test_far_dp_effect_cpu.py -q
```

Préparer le manifeste, sans soumettre de job :

```bash
"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py prepare \
  --stage smoke --output-root "$FAR_WORK"
export FAR_MANIFEST="$FAR_WORK/smoke/manifest.json"
export MAX_PARALLEL=4
bash hpc/submit_far_dp_effect_cpu.sh
```

La dernière commande **affiche seulement** la commande SLURM. Vérifier compte,
partition, QoS et ressources. Puis soumettre les 18 jobs :

```bash
bash hpc/submit_far_dp_effect_cpu.sh --submit
```

Pour un smoke limité à Fashion-MNIST (9 jobs), utiliser dès le premier `prepare`
`--dataset fashionmnist`, dans un root réservé à ce smoke. Ne pas remplacer
ensuite son manifeste par celui des deux datasets.

Suivi :

```bash
squeue -u "$USER"
"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py status --manifest "$FAR_MANIFEST"
```

Exiger 18 sorties valides si les deux datasets ont été préparés. Un job SLURM
« COMPLETED » ne suffit pas : le script `status` contrôle les sorties et leurs
empreintes. Un run arrêté proprement avant sa fin a le statut `paused`.

## 5. Calibration — lancer seulement après le smoke

```bash
"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py prepare \
  --stage calibration --output-root "$FAR_WORK"
export FAR_MANIFEST="$FAR_WORK/calibration/manifest.json"
export MAX_PARALLEL=4
export JOB_OFFSET=0
bash hpc/submit_far_dp_effect_cpu.sh
bash hpc/submit_far_dp_effect_cpu.sh --submit
```

72 jobs : 2 datasets × 3 seeds × 2 batches × 2 pas × 3 contrôles.
Chaque trajectoire a 600 tours. Les préfixes DP ne sont pas recalibrés à ε = 4 :
c'est le budget à 600 tours qui est fixé. Les ε cumulés intermédiaires sont
enregistrés. Pour confirmer un horizon final de 300 tours, il faudra recalibrer
à 300 et refaire l'entraînement ; ne pas réutiliser le préfixe comme un run à ε = 4.

Regarder validation loss, accuracy, Worst-20, gap, oscillations, coût et normes
des uploads. Le rayon serveur et le rayon de F_CC doivent être choisis hors test
et rester identiques dans les contrastes appariés. Les valeurs 16 et 8 du smoke
ne sont pas des choix scientifiques automatiquement promus. Si nécessaire,
faire une calibration supplémentaire explicitement documentée.

## 6. Confirmation — verrouillage puis lancement par morceaux

Copier le modèle de verrouillage, remplir pour chaque dataset : batch, rounds,
learning_rate, server_clip et fcc_radius, ainsi que le chemin vers le rapport
motivant ces choix **à partir de la calibration**. Le pas est constant dans ce
lanceur ; ne pas déclarer un schedule décroissant sans l'implémenter séparément.

```bash
cp configs/ldp_gradient_far/far_dp_effect_toubkal_lock.template.yaml \
  "$FAR_WORK/calibrated_choices.yaml"
# Éditer ce fichier ; les null bloquent volontairement la confirmation.

"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py prepare \
  --stage confirmation --output-root "$FAR_WORK" \
  --lock "$FAR_WORK/calibrated_choices.yaml"
export FAR_MANIFEST="$FAR_WORK/confirmation/manifest.json"
export ARRAY_CHUNK=1000
export JOB_OFFSET=0
export MAX_PARALLEL=4
bash hpc/submit_far_dp_effect_cpu.sh
```

La grille totale contient 88 920 jobs avec les choix par défaut. Le wrapper
ne soumet **qu'un morceau par appel**, par exemple les indices 0–999 :

```bash
bash hpc/submit_far_dp_effect_cpu.sh --submit
```

Pour le morceau suivant : `export JOB_OFFSET=1000`, puis vérifier/soumettre.
Il n'y a pas de boucle automatique qui engage tout le crédit premium. Adapter
`ARRAY_CHUNK` à la limite `MaxArraySize` et aux règles de soumission de Toubkal.
La limite `%4` est **par array** : plusieurs arrays simultanés additionnent leur
concurrence. Vérifier les jobs actifs avant chaque soumission.

On peut préparer la confirmation d'un seul dataset avec `--dataset fashionmnist`
dans un root distinct (44 460 jobs). Ce n'est pas une suppression implicite de
MNIST : sa confirmation sera une étape séparée explicitement annoncée.

## 7. Ressources, interruption et reprise

Défauts : 8 CPU par job, 16 Go, 12 h. Les CPU servent au calcul tensoriel, pas
à huit clients indépendants. Chronométrer les premiers jobs avant d'augmenter
la concurrence. Tester 4/8/16 threads dans des roots de benchmarks distincts
si nécessaire ; plus de threads n'est pas systématiquement plus rapide.

Ces 12 h sont inférieures au plafond premium de 36 h. Quatre jobs simultanés
demandent 32 CPU et jusqu'à 64 Go au total. Garder cette concurrence initiale
avant de mesurer le débit et le coût de la calibration. Ne pas confondre les
3584 CPU autorisés avec une recommandation de les réserver tous.

Surcharges possibles avant soumission :

```bash
export FAR_CPUS=8
export FAR_MEMORY=16G
export FAR_TIME_LIMIT=12:00:00
export MAX_PARALLEL=4
```

Le manifeste fige le code et la matrice. Un changement de source bloque les
runs de ce manifeste : créer une nouvelle version/root pour un nouveau
protocole. Les fichiers `.simulation_entropy` doivent être conservés privés
et sauvegardés avec la campagne, jamais publiés avec les métriques.

Checkpoint tous les 10 tours et à la fin. Le signal SLURM USR1 180 secondes
avant la limite demande un arrêt au prochain round complet, avec sauvegarde.
Si le round est trop long ou SIGKILL survient, reprise depuis le dernier checkpoint.
Le job inclut toujours `--resume`. Il saute les runs déjà terminés ; un verrou
empêche deux processus d'écrire simultanément dans le même dossier.

Resoumettre le même morceau reprend les incomplets, mais les jobs déjà complets
consomment tout de même une petite allocation pour être vérifiés. Ne pas
resoumettre si les tâches sont encore actives. Le wrapper ne consulte pas
automatiquement `squeue` et ne filtre pas les indices terminés.

## 8. Sorties et analyse

Chaque dossier `STAGE/runs/ID/` contient :

- `orchestration_status.json` : progression/erreur/statut et dispositif CPU ;
- `checkpoint.pt` : modèle, ancre F_CC, historique, empreintes et runtime ;
- `metrics.json` après complétion : paramètres, budget, trajectoires et métriques
  clientes/par classe. Les pertes train sont réellement évaluées au dernier
  tour hors smoke, jamais remplacées par des zéros techniques.

Les trajectoires enregistrent aussi poids min/max, concentration, entropie,
distance/logit-span, taux de clipping serveur et normes quantiles. Les
évaluations test sont disponibles seulement en confirmation ; elles ne servent
ni à ajuster le learning rate ni à choisir le checkpoint final.

Résumé hiérarchique des courbes de confirmation :

```bash
"$PYTHON_BIN" scripts/run_far_dp_effect_cpu.py summarize \
  --manifest "$FAR_WORK/confirmation/manifest.json" \
  --output "$FAR_WORK/confirmation/curves_mean_sd.json"
```

Les groupes incomplets sont signalés, pas imputés. Le résumé fournit moyenne,
écart-type des 10 moyennes par seed, dispersion MC intra-seed et dispersion de
toutes les trajectoires. Ce sont des écarts-types, pas des IC95. Le raccordement
des figures au notebook sera effectué depuis ces nouvelles sorties, sans
mélanger anciens runs MPS et nouveaux runs CPU.

## 9. Limites de ce lancement

Le code a des tests numériques CPU et de reprise ; sa validation réelle sur
les nœuds Toubkal reste à faire par le smoke. Aucune promesse de convergence,
de classement DP/sans DP ou de temps total n'est faite avant calibration.
RFA utilise le solveur itératif existant (100 itérations, tolérance 10⁻⁶) ; il
faut contrôler sa sensibilité numérique avant une conclusion publiée.

Garantie étudiée : DP par exemple côté client, adjacency replace-one, batch
fixe sans remise par tour, sensibilité 2C/B, amplitude σ_C C/B. Elle n'est pas
une protection du remplacement de tout le dataset client. La privatisation
n'inclut pas les diagnostics de recherche, évaluations et checkpoints. Des
seeds/clés de simulation permettant de reconstruire le bruit ne doivent pas
être publiées comme un transcript DP. L'accountant est par run : les milliers
de branches ne sont pas conjointement à ε = 4 si publiées sur des données privées.
