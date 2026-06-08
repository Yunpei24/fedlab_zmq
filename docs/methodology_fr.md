# Méthodologie énergie — FedLab ZMQ (FedPartBE)

*Description honnête de l'état RÉEL du dépôt (branche `chore/artifact-cleanup`).
Les valeurs sont citées depuis le code ; « fait vs en cours » est séparé au §8.*

---

## 1. Vue d'ensemble — la question de recherche

On étudie l'**apprentissage fédéré sous contrainte d'énergie** : une flotte
d'appareils edge sur batterie entraîne un modèle partagé. La question est
**quel client entraîne quoi, et quand, pour que la flotte continue de participer
avant que les batteries ne meurent** — et si l'entraînement *partiel*
battery-aware (FedPartBE) aide réellement une fois l'énergie comptée
**honnêtement**.

Le cadre simule, par round et par client, l'énergie dépensée et décharge une
batterie ; quand elle atteint zéro, le client décroche. Le cadre est dans
`hardware/` (`flop_cost.py`, `profiles.py`, `energy_model.py`), piloté par
`run_experiment.py` et les configs YAML de `configs/`.

**Glossaire.** *FLOPs* = nombre d'opérations flottantes d'un calcul (un
décompte). *GFLOP/s* = débit crête d'une puce (une vitesse). *MFU* = fraction du
crête réellement soutenue. *cost_model* = comment on compte les FLOPs compute
d'un round. *alpha* = le facteur d'échelle énergie compute (défini au §4).

---

## 2. Le modèle d'énergie

Par round, par client (`DeviceProfile.round_energy_breakdown`, `profiles.py`) :

```
energie_round = energie_compute + energie_uplink + energie_downlink
```

**Compute** (`compute_energy_j`) :

```
energie_compute = P_compute · (FLOPs / (peak_gflops · 1e9)) · alpha
```

c.-à-d. puissance × temps, où temps = FLOPs / débit-crête. `alpha` ne met à
l'échelle **que le compute** (§4). Les FLOPs viennent du `cost_model` choisi
(§3) ; `peak_gflops` et le décompte de FLOPs partagent la même convention FP32
mul+add, pour que la division soit un vrai temps.

**Communication** (`comm_energy_j`) — le modèle qui décharge réellement la batterie :

```
energie_uplink   = P_tx · (uplink_bytes   / uplink_bytes_per_sec)
energie_downlink = P_rx · (downlink_bytes / downlink_bytes_per_sec)
```

c.-à-d. puissance d'émission/réception × temps de transfert à la bande passante
fixe du lien (par device : `tx_w`, `rx_w`, `uplink_mbps`, `downlink_mbps` dans
`profiles.py`). **alpha ne met PAS à l'échelle la communication** — une
transmission radio n'est pas affectée par l'absence de FPU.

> **Note d'honnêteté.** Un modèle de canal **Shannon–Friis** plus riche existe
> aussi (`energy_model.py` : `friis_path_loss`, `compute_shannon_rate_bps`, avec
> `channel_params` attaché à chaque profil). Il n'est **pas** branché sur le
> chemin qui décharge la batterie — `round_energy_breakdown` utilise le modèle à
> bande passante fixe ci-dessus. Shannon–Friis est disponible comme raffinement
> futur ; les chiffres de cette étude utilisent le modèle comm fixe, pour que le
> découpage se réconcilie exactement avec ce qui décharge la batterie.

Le découpage se resomme au total au bit près (vérifié :
`tests/test_energy_breakdown.py`, et réconciliation par round dans le runner).

---

## 3. Comptabilité FLOP honnête — les trois `cost_model`

Les FLOPs compute d'un round sont la source unique de vérité dans
`hardware/flop_cost.py` (dispatcher `round_compute_flops`, flag `--cost-model`) :

| cost_model | rôle |
|---|---|
| **`phi`** (legacy) | reproduit la formule analytique de chaque algo **au bit près** (baseline de régression, 11 tests) |
| **`corrected`** | modèle *analytique* position-aware pour groupes de couches contigus |
| **`measured`** (défaut) | `torch.utils.flop_counter.FlopCounterMode` exécuté avec le **vrai** masque `requires_grad` de l'algo, mis en cache |

Les formules `phi` legacy (citées du dispatcher) :

- FedAvg / FedResonance : `full = 3 · 2 · N · B · S` (fwd+bwd+update × MAC→FLOP × params × batch × steps)
- FedPart : `full · (1/3 + 2/3 · gflops[g]/Σ gflops)`
- ServerMaskFL : `full · 0.5 · (1 + beta)`
- ccsEF : `2 · B · steps · (N + 2·|primary|)`

**La découverte.** Valider `phi` contre `measured` (FlopCounterMode) a révélé
deux problèmes (cités de `flop_cost.py`) : l'analytique **sous-estime les CNN de
~150–300×** (empiriquement **158×** pour ResNet-8 ici), *et* il **inverse le
classement par groupe**.

**Pourquoi la POSITION, pas la taille, gouverne le coût backward.** N'entraîner
que le groupe *p* (geler le reste). L'autograd doit quand même propager
`grad_input` en arrière à travers **chaque couche en aval de p**. Donc un groupe
*peu profond* (le stem) déclenche un backward quasi complet, alors que la *tête*
(fc) n'a rien en aval et est quasi gratuite. Le naïf `gflops[p]/Σ gflops` fait
paraître le stem le moins cher ; en réalité il est le plus cher. Le modèle
corrigé le facture explicitement (`compute_corrected_group_costs`) :

```
corrected_p = gflops[p] + 0.5 · Σ_{i>p} gflops[i]      # son bwd + moitié de l'aval
round_flops = full · (1/3 + 2/3 · corrected_p / Σ corrected)
```

**La correction de cohérence.** FedPartBE utilisait auparavant `phi` pour la
**comptabilité d'énergie** mais les coûts corrigés pour l'**assignation des
tiers** — incohérent. Désormais assignation **et** comptabilité utilisent le
**même** cost_model. Sous comptabilité honnête, le *vrai* avantage de FedPartBE
apparaît : il était en partie caché sous `phi`, qui mal-tarifait justement les
groupes peu profonds que FedPartBE protège.

> Convention : `calibrate_convention()` mesure ResNet-50 une fois ; PyTorch 2.x
> rapporte de vrais FLOPs → facteur **1.0** (asserté par la suite de tests).

---

## 4. Le facteur alpha (écart d'utilisation compute)

`measured` donne les *vrais* FLOPs, mais une puce ne soutient jamais son crête
nominal sur du conv/BN réel (FPU/SIMD absent ou limité, borné mémoire). `alpha`
convertit le temps à débit-crête en temps soutenu :

```
alpha = 1 / (utilisation soutenue du crête)   ≥ 1
```

`alpha = 1` est l'idéal imbattable ; `alpha < 1` est physiquement impossible.
**alpha ne met à l'échelle que le compute** (flag `--alpha-applies-to`, défaut
`compute` ; `total` n'existe que pour reproduire le sweep legacy déjà commité).
Il **s'annule dans les ratios relatifs (inter-algos)** mais **pas** dans la
survie absolue une fois les batteries mortes (il décide qui meurt et quand).

**Anti-circularité.** alpha n'est **pas** calé sur une cible de survie — ce
serait circulaire (calibrer l'unité d'énergie contre le résultat qu'on mesure).
C'est un paramètre **physique**, fixé indépendamment. On le pose en
**estimations documentées par device** (1/utilisation soutenue), dans
`configs/device_profile_study.yaml` :

| profil | puce | util. soutenue ~ | alpha |
|---|---|---|---|
| esp32_s3 | Xtensa LX7, sans FPU/SIMD | ~10 % | **10** |
| raspberry_pi_zero2w | Cortex-A53 + NEON | ~20 % | **5** |
| raspberry_pi_4 | Cortex-A72 + NEON | ~33 % | **3** |
| smartphone_midrange | CPU/DSP mobile | ~50 % | **2** |
| smartphone_highend | classe NPU | ~65 % | **1.5** |

Ce sont des **estimations, pas des mesures**. La robustesse est montrée par un
**sweep** `alpha ∈ [1, 2, 3, 5, 10, 20]` (`scripts/run_alpha_sensitivity.py`,
marqueur `alpha=5`). Un hook de config (`device_alpha_measured_anchor`) permet
d'injecter plus tard un alpha RPi4 **mesuré** et de rééchelonner les autres par
le ratio d'utilisation.

---

## 5. La découverte d'infaisabilité

Sous comptabilité honnête (`measured`), entraîner un ResNet-8 **complet** sur un
**ESP32-S3** avec E=3 et alpha=12.6 vide la batterie ~13.3 kJ en **~2–4 rounds**
(observé dans `results/CORRECTION/NIID05_E3/`). C'est une **vérité physique**,
pas un artefact de réglage : un MCU sans FPU ne peut tout simplement pas
s'offrir l'entraînement complet. C'est la motivation de l'**entraînement partiel
battery-aware** (FedPartBE) : n'entraîner qu'un sous-ensemble abordable de
couches par round. (Réduire la charge par round à E=1 amène la durée de vie
médiane à **~15–35 rounds** — un régime de survie lisible ; voir §6.)

---

## 6. Étude inter-profils (une projection pilotée par profils)

`scripts/run_device_profile_study.py` rejoue le *même* workload sur les profils
(esp32_s3, raspberry_pi_zero2w, raspberry_pi_4, smartphone mid/high),
`cost_model=measured`, alpha par device (compute-only). Par profil il reporte le
découpage énergie **compute / uplink / downlink** (par round + cumulé) et la
**survie/participation** ; puis une vue inter-profils du basculement de
l'équilibre.

**Cadrage.** C'est une **projection pilotée par profils, pas un déploiement.**
Sur l'ESP32-S3, ResNet-8 (+grads+optimiseur ≈ 15 Mo) ne **tient pas** dans 8 Mo
de RAM (`DeviceProfile.can_run_model` → False) ; cette ligne est donc une
**projection d'énergie** ; RPi/smartphone exécutent réellement le modèle.

**Métrique foregroundée par régime :** survie là où la batterie *mord* (classe
ESP32), énergie totale / rounds-pour-accuracy là où elle ne mord *pas*
(RPi/smartphone).

**Ce que le smoke 2-extrêmes montre à ce stade** (esp32_s3 vs smartphone_highend,
3 clients) : FedPartBE **économise ~57 % d'énergie** vs FedAvg sur ESP32 et
**~35 %** sur le smartphone, et survit plus longtemps là où la batterie mord.
Réserve honnête : pour ResNet-8 (~78 k params) sur données locales complètes, le
**compute domine sur tous les profils** (fraction comm seulement 0.000 → ~0.007).
Un régime réellement comm-bound exigerait un modèle plus gros et/ou des datasets
locaux plus petits. *(Le grid complet 5×5 n'a pas été lancé — voir §8.)*

---

## 7. Métriques

- **Survie** (`metrics/survival.py`) : durée de vie médiane, round de la Nème
  mort (N = 5, 10, 15), aire sous la courbe de survie (Σ clients-vivants sur les
  rounds), fraction de participation.
- **Découpage énergie :** par round et cumulé compute / uplink / downlink
  (se resomme au total).
- **Robustesse alpha :** les claims relatifs (ratios d'énergie inter-algos) sont
  invariants en alpha (alpha s'annule) ; l'*ordre* de survie est robuste en alpha
  sur la grille balayée — la comparaison d'algos ne dépend donc pas de la valeur
  exacte d'alpha.

---

## 8. État : fait vs en cours

**Fait**
- `flop_cost.py` : trois cost_model ; `phi` reproduit le legacy au bit près
  (11 tests de régression verts) ; `corrected` position-aware ; `measured` via
  FlopCounterMode (caché). Facteur de convention 1.0 vérifié.
- Découpage énergie compute/uplink/downlink, par round + cumulé ; se resomme au
  total (5 tests dédiés).
- `alpha_applies_to=compute` est le défaut pour tous les résultats rapportés.
- alpha par device en **estimations** + outillage de sweep ; le diagnostic
  ESP32/E=1 confirme un régime à dizaines de rounds.
- Outillage de l'étude inter-profils ; **smoke 2-extrêmes** validé de bout en bout.
- Reproductibilité : deps épinglées, seeding centralisé, manifest par run, config
  smoke.

**En cours / en attente**
- **Grid alpha complet** (18 runs) pas encore lancé sur vrai matériel / en long.
- **Grid inter-profils 5×5 complet** pas lancé (seulement le smoke 2×2).
- Les valeurs alpha sont des **estimations documentées, pas mesurées** (ancre
  mesurée RPi4 en attente).
- Capacité de **contraste phi** présente mais le run de contraste n'est pas produit.
- L'énergie comm utilise le modèle à **bande passante fixe** ; Shannon–Friis
  existe mais n'est pas branché sur la comptabilité.

---

## 9. Comment lancer (CLI réel)

Flags réels (`run_experiment.py`) : `--config`, `--algo`, `--epochs`, `--output`,
`--cost-model {phi,corrected,measured}`, `--energy-scale-factor`,
`--alpha-applies-to {compute,total}`, `--device`, `--seed`.
**Note :** avec `--config`, `num_clients`/`num_rounds` viennent du YAML —
`--clients`/`--rounds` sont ignorés ; `--epochs`, `--algo`, `--cost-model`,
`--output`, `--device` font bien override.

```bash
# (a) Un run unique
caffeinate -si python run_experiment.py \
    --config configs/fedpartbe_survival_wide_cifar10.yaml \
    --algo fedpart_be \
    --epochs 3 \
    --output results/single/fedpart_be \
    --cost-model measured \
    --device mps

# (a') A/B FedPartBE vs FedPart, comptabilité honnête (les runs CORRECTION)
for ALGO in fedpart_be fedpart; do
    caffeinate -si python run_experiment.py \
        --config configs/fedpartbe_survival_wide_cifar10.yaml \
        --algo $ALGO \
        --epochs 3 \
        --output results/CORRECTION/NIID05_E3/${ALGO}_2 \
        --cost-model measured \
        --device mps
done

# (b) Étude inter-profils / survie (alpha par device, découpage + survie)
caffeinate -si python scripts/run_device_profile_study.py --device mps --jobs 4
#   sanity rapide 2-extrêmes (le smoke validé) :
python scripts/run_device_profile_study.py --smoke \
    --profiles esp32_s3 smartphone_highend --algos fedavg fedpart_be

# (c) Sweep de sensibilité alpha (grille physique [1,2,3,5,10,20], marqueur 5)
caffeinate -si python scripts/run_alpha_sensitivity.py --grid full --device mps --jobs 4
#   ciblé sur l'extrême comm-bound :
python scripts/run_alpha_sensitivity.py --grid full \
    --fleet-device smartphone_highend --device mps --jobs 4

# (d) Contraste phi (annexe : montrer que l'écart est marginal sous le legacy)
python scripts/run_alpha_sensitivity.py --grid full --cost-model phi --device mps --jobs 4

# Sanity d'abord (garde-fou de régression du cost-model, < 5 s) :
make test        # == python -m pytest  → 11 passed (+5 energy-breakdown)
```

Chaque run écrit `metrics.json`, `manifest.json` (config résolue, hash du commit
git, seed, versions des packages, convention FLOP) et `survival.csv` dans son
dossier `results/`.
