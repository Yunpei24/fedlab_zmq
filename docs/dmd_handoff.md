# DMD-CB / USV — passation vers une session locale

État au 16/09/2026. Ce document suffit à reprendre le travail sans le fil de
conversation d'origine. L'audit détaillé est dans `docs/dmd_audit_option_b.md`.

## Où en est le travail

**Audit du code fait**, correctifs appliqués et testés. **Campagne phase 1 en
cours**, 5 runs sur 24 bankés — le reste est à relancer localement.

Le sandbox cloud redémarre toutes les ~30 min et les processus d'arrière-plan
n'y avancent pas de façon fiable entre les tours, d'où la migration en local.

## Ce qui a été corrigé, et pourquoi

Quatre défauts vérifiés numériquement avant correction :

1. **Le seuil CVaR était inerte.** `weighted_upper_cvar` est correct, mais avec
   `|A|` survivants de poids ~égaux et `cvar_tail_mass <= 1/|A|`, la masse de
   queue est épuisée par le seul client le plus déficitaire : `eta = max(D)`.
   Sous participation partielle (4 survivants, `tail_mass=0.2`) le hinge ne
   s'active jamais. Mesuré : fraction de cohorte au-dessus de `eta` = **0.000**
   sur 200 rounds simulés. Toute mesure de DMD-Tail en participation partielle
   mesurait en réalité DMD-CB pur.

2. **La forme Rockafellar–Uryasev n'était pas optimisée.** `eta` étant détaché,
   le `min` sur `eta` — tout l'intérêt de RU — n'avait jamais lieu.

3. **La constante `+ eta` polluait les pertes journalisées** : pour un client
   sous le seuil, 100 % du terme de dispersion loggé était cette constante, ce
   qui rendait les traces incomparables aux bras mean/USV.

4. **`audit["dmd_cvar_eta"]` journalisait toujours la statistique d'ordre**, pas
   le seuil réellement publié aux clients.

Correctif principal : `cvar_eta_mode: dual`, montée duale RU côté serveur,

    eta <- eta + lr * mean(D) * (P(D > eta) / b - 1),  borné dans [0, max D]

Le seuil converge vers le quantile `1-b` en moyennant sur les rounds au lieu de
classer 4 points. Mesuré : fraction de queue **0.000 → ~0.20** pour une cible
`b = 0.25`.

## Phase 2 : marge bornée (prête, pas lancée)

`margin_space` ∈ `{logit, probability, normalized}` dans `DMDConfig`.

Invariance d'échelle mesurée, décisions identiques à 100 % dans tous les cas :

| facteur | logit | probability | normalized |
|---|---|---|---|
| ×1.00 | 29.9643 | 0.1322 | 0.0535 |
| ×0.25 | 1.8728 | 0.0020 | 0.0535 |
| **ratio** | **0.062** | **0.015** | **1.000** |

`logit` : le déficit est divisé par 16, donc l'optimiseur est payé pour
rétrécir les logits, et `weight_decay=1e-4` pousse exactement là.
`probability` : rétrécir coûte, le défaut penche du bon côté.
`normalized` : strictement invariant.

Sensibilité DP du scalaire rapporté, à `B_D = M^2/2` :

| espace | B_D | Delta_D | signal/Delta |
|---|---|---|---|
| logit (M=10) | 50 | 0.393 | **0.27** |
| probability | 0.50 | 0.0119 | **8.81** |
| normalized | 2.00 | 0.0235 | **4.48** |

À 0.27 la sensibilité dépasse la valeur rapportée : après bruit calibré le seuil
est du bruit pur. C'est l'argument pour changer la marge **avant** la phase DP.

Également ajouté : `margin_target` (critère satisficing `[tau - m]_+^2`, défaut
0.0 = comportement publié), et l'algorithme **`cb_ce`** — FedAvg + CE pondérée
inverse-fréquence. C'est le contrôle qui manquait : `Margin-Mean` isolait
« marge sans class-balancing », rien n'isolait « class-balancing sans marge ».

**Attention** : `mean_mu` ne se transpose pas. Le déficit passe de ~30 (logit) à
~1e-5..2e-3 (probability) sur EMNIST. D'où la calibration en deux temps.

## Reprendre en local

### 1. Données

Sur une machine normale, le téléchargement canonique suffit — le registre le
fait tout seul. **Mais** les runs déjà bankés ici utilisent le bundle
TensorFlow Federated, dont les images sont droites alors que les images NIST
canoniques sont transposées (torchvision ne redresse pas). Les deux sources ne
sont **pas** comparables et ne doivent jamais être mélangées dans une table.

Pour garder la continuité avec les 5 runs déjà faits :

    python3 scripts/prepare_emnist_byclass.py --data-root ./data

Pour repartir propre sur la source canonique : supprimer `data/EMNIST/`, laisser
torchvision télécharger, et **relancer les 24 runs** (`rm -rf results/emnist_phase1
logs/emnist_phase1`).

### 2. Phase 1

    ./scripts/run_phase1.sh          # arrière-plan, saute les runs déjà faits

ou, si l'arrière-plan n'est pas fiable :

    scripts/run_until_deadline.sh 'configs/dmd_emnist/*_s[0-9]*.yaml' \
        results/emnist_phase1 emnist_phase1 3600

Sur une machine à 8+ cœurs, du parallélisme est rentable :

    FEDLAB_NUM_WORKERS=0 scripts/run_campaign.sh \
        'configs/dmd_emnist/*_s[0-9]*.yaml' results/emnist_phase1 emnist_phase1 4 2

Les marqueurs `logs/emnist_phase1/*.done` font qu'une relance ne refait rien.
Compter : `ls logs/emnist_phase1/*.done | wc -l` sur 24.

### 3. Phase 2

    python3 scripts/gen_emnist_phase2.py --step calibrate
    FEDLAB_NUM_WORKERS=0 scripts/run_campaign.sh \
        'configs/dmd_emnist_p2/*_mu*_p*.yaml' results/emnist_phase2_cal phase2_cal 4 2
    # puis, avec le mu_M retenu :
    python3 scripts/gen_emnist_phase2.py --step test --mean-mu <gagnant>

## Ce que la phase 1 doit trancher

Six bras × 4 seeds (91/92/93/24), protocole fidèle au cadre conceptuel.

| bras | rôle |
|---|---|
| `fedavg` | ancrage |
| `dmdcb` | DMD-CB fixe, `mu_M=0.1875` |
| `dmdcb_hi` | `mu_M=0.2027` — **contrôle décisif** : c'est l'intensité effective moyenne rapportée pour USV-0.25. S'il reproduit la baisse de VarBA, le terme USV n'est qu'un `mu_M` un peu plus fort. |
| `usv025` | reproduction de USV-0.25 |
| `tail_emp` | CVaR seuil empirique — doit montrer l'inertie du défaut 1 |
| `tail_dual` | **Option B** : CVaR à seuil appris |

Trois questions :

1. `dmdcb_hi` reproduit-il la baisse de VarBA de `usv025` ? Si oui, l'USV est
   mort comme mécanisme distinct.
2. `tail_dual` bat-il `usv025` sur le compromis VarBA / Worst-20 ?
3. `tail_emp` est-il bien inerte (`dmd_cvar_tail_fraction_above_eta` ~ 0) ?

Analyse attendue : deltas appariés par seed sur Acc / MeanBA / Worst-20 / VarBA /
Gap, avec t-tests appariés. Rappel : à n=4 le test de permutation de signes a
p_min = 0.125, donc « 4/4 seeds » **est** p = 0.125.

## Défauts identifiés et délibérément NON corrigés

- **Trois décalages empilés** (défaut C5) : le hinge compare `D_hat` (mini-batch
  d'entraînement / modèle local / round t) à un seuil mesuré sur ancres / modèle
  global / round t-1. Choix de conception, pas un bug ; à trancher séparément.
- **Repli silencieux des ancres** (C7) : sans `anchor_dataloader` ni
  `require_anchor_dataloader`, `client.py` prend le jeu d'entraînement comme
  ancres sans avertir. `configs/dmd/dmd_tail_cifar10.yaml` est dans ce cas.
- **Écart document/code** (C8) : le cadre conceptuel dit que BA/VarBA/Worst-20
  sont mesurés sur les ancres ; `run_experiment.py` les mesure sur des shards de
  test appariés (~7.7k ex/client). Plancher de bruit d'estimation mesuré :
  **0.0005, soit 3.6 % de la VarBA rapportée** — l'objection « bruit de mesure »
  tombe donc largement, mais l'écart doit être réconcilié avant soumission.

## Questions ouvertes soulevées par l'encadrant

- **`sign(m)` dit-il vraiment le bon/mauvais côté ?** Oui, exactement : c'est
  l'argmax réécrit (mesuré 100.0000 % d'accord, 0 égalité). Mais **seul le signe**
  a un sens exact ; la magnitude est en unités logit arbitraires, n'est pas une
  distance, et est re-scalable sans changer une décision.
- **Dépendance aux activations ?** La définition est agnostique, mais ReLU est
  positivement homogène, donc l'échelle du logit est un paramètre libre — raison
  pour laquelle toute la littérature des bornes à base de marge normalise
  (Bartlett–Foster–Telgarsky 2017, Neyshabur et al.). Mesuré sur `cnn_gn` :
  `classifier.4 × 2.0` → marge × 2.000, décisions inchangées ; une conv
  intermédiaire × 2.0 → marge × 1.000 (GroupNorm absorbe). **L'arbitraire
  d'échelle est confiné à la dernière couche, mais il y est exactement.**
  Les activations décident aussi si l'Hypothèse 12.2 (lissité) tient — elle ne
  tient pas avec ReLU.
- **Vraiment multi-classe ?** Empiriquement oui, mais la forme `max` est celle
  de Crammer–Singer, qui n'est Fisher-consistante que s'il existe une classe
  dominante (`max_c p(c|x) > 1/2`) — voir Zhang JMLR 2004, Tewari & Bartlett
  JMLR 2007, Liu 2007. Avec 62 classes cette condition tombe souvent. De plus
  `F + mu*D` peut cesser d'être calibré pour `mu` grand : la pénalité échange
  de la calibration contre de la marge. **Théorème à écrire** : `F + mu*D` est
  classification-calibrated pour `mu < mu*(p)`. À surveiller dans le balayage 2a.
  Ablation bon marché à ajouter : forme somme (Weston–Watkins) vs `max`.
- **Limite de généralisation** : la méthode suppose tête linéaire + argmax. Elle
  ne s'applique pas au multi-label (sigmoïde par classe) : pas d'argmax, pas de
  concurrent, `m` indéfini.

## Verrous avant publication

1. **Nouveauté** vs LDAM (Cao et al. 2019), CB-loss (Cui et al. 2019), FedLC
   (Zhang et al. ICML 2022) — non résolu, c'est le verrou n°1.
2. `cb_ce` doit être battu.
3. Puissance statistique : n=4 ne peut rien établir ; il faut ~10 tirages avec
   partition et init découplés.
4. Niveaux absolus (31–36 % sur EMNIST) : le cap de 500 exemples doit sauter ou
   être justifié.
5. Théorie : le §12 actuel repose sur une hypothèse de lissité fausse pour le
   réseau entraîné. Il faut au moins un vrai théorème — le plus accessible étant
   la borne déficit → erreur → BA sous condition de marge, ou la calibration
   ci-dessus.
