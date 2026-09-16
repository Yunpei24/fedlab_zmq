# Audit DMD-CB / USV et bascule vers l'Option B (CVaR à seuil appris)

Audit du code `algorithms/dmd/` confronté au cadre conceptuel DMD-CB+USV.
Chaque constat a été vérifié numériquement avant correction.

## Constats

### C1 — `eta` CVaR dégénère en maximum de cohorte (bloquant)

`weighted_upper_cvar` calcule le CVaR empirique exact. Avec `|A|` survivants de
poids ~égaux et `tail_mass <= 1/|A|`, la masse de queue est épuisée par le seul
client le plus déficitaire : `eta = max(D)`. Sous participation partielle
(4 survivants, `cvar_tail_mass=0.2`) le hinge `[D - eta]_+` ne s'active donc au
round suivant que pour un client dépassant le maximum précédent.

Mesuré : fraction de cohorte au-dessus de `eta` = **0.000** sur 200 rounds
simulés. Le terme de queue était inerte.

### C2 — la forme Rockafellar–Uryasev n'était pas optimisée

`tail_objective` évalue `eta + [D - eta]_+ / b` à `eta` *détaché*. Le minimum
sur `eta` — qui est tout l'intérêt de RU — n'avait jamais lieu : `eta` était
posé au quantile empirique du round précédent, puis gelé.

### C3 — la constante `+ eta` polluait les pertes journalisées

`eta` étant détaché, `mu_V * eta` n'a aucun gradient mais s'ajoutait à chaque
`local_dmd_addend`, y compris pour les clients strictement sous le seuil.
Mesuré : pour `D=0.05 < eta=0.20`, 100 % du terme de dispersion journalisé était
cette constante. Les traces de perte n'étaient pas comparables aux bras
`mean`/`upper_semivariance`.

### C4 — l'audit journalisait le mauvais `eta`

`audit["dmd_cvar_eta"]` réécrivait systématiquement `tail.eta` (la statistique
d'ordre), même lorsqu'un `eta` différent était publié aux clients. Les
trajectoires publiées ne montraient donc pas le seuil réellement appliqué.

### C5 — trois décalages empilés dans la comparaison centrale

Le seuil et son argument ne sont pas mesurés sur le même objet :

| | argument `D_hat_{i,b}` | seuil `a_{t-1}` / `eta` |
|---|---|---|
| données | mini-batch d'entraînement | ancres `V_i` |
| modèle | local, en cours de SGD | global avant adaptation |
| round | `t` | `t-1` |

Non corrigé ici : c'est un choix de conception, pas un bug, et le corriger
changerait la sémantique de la méthode. À trancher en phase 2.

### C6 — le déficit n'est pas invariant d'échelle

`0.5 * [r - m]_+^2` sur marges de logits bruts. Mesuré : contracter les logits
d'un facteur 4 divise le déficit par ~16 **à décisions strictement inchangées**.
`weight_decay=1e-4` pousse activement dans cette direction. Non corrigé ici
(changer la définition de la marge invaliderait la comparaison aux tables
publiées) — c'est le premier point de la phase 2.

### C7 — repli silencieux des ancres sur le jeu d'entraînement

`client.py` : sans `anchor_dataloader` et sans `require_anchor_dataloader`, les
ancres deviennent le jeu d'entraînement, sans avertissement. `configs/dmd/
dmd_tail_cifar10.yaml` est dans ce cas. Non corrigé ici ; les configs de la
campagne posent `require_anchor_dataloader: true`.

### C8 — écart entre le document et le code sur les métriques d'équité

Le cadre conceptuel indique que BA/VarBA/Worst-20/Gap sont mesurés sur les
ancres. `run_experiment.py` les mesure en réalité sur des shards de test
appariés (~7.7k exemples/client sur EMNIST). Mesuré : plancher de bruit
d'estimation de VarBA = 0.0005, soit 3.6 % de la VarBA rapportée — contre 22 %
si la mesure se faisait bien sur 150 ancres. À réconcilier : soit le document
décrit un pipeline antérieur, soit les CSV publiés viennent de `_legacy_core`.

## Corrections appliquées

- `cvar_eta_mode: {"empirical", "dual"}` et `cvar_eta_lr` (`config.py`).
- `_dual_ascent_eta` (`server.py`) : montée duale RU
  `eta <- eta + lr * mean(D) * (P(D > eta) / b - 1)`, pas normalisé par le
  déficit moyen de cohorte, borné dans `[0, max D]`. Le seuil converge vers le
  quantile `1 - b` en moyennant sur les rounds au lieu de classer 4 points.
- Suppression de la constante `+ eta` de `tail_objective` (C3).
- `dmd_cvar_eta` journalise le seuil publié ; `dmd_cvar_eta_empirical` et
  `dmd_cvar_tail_fraction_above_eta` ajoutés pour l'audit (C4, C1).
- Cache du dataset brut par `(nom, split, racine)` (`datasets/registry.py`) :
  6573 Mo -> 1427 Mo de RSS par run, résultats identiques.
- `global_eval_every` (`run_experiment.py`), défaut 1 = comportement historique.

Validation : `tail_fraction` passe de 0.000 (inerte) à ~0.20 pour une cible
`b = 0.25` sur 200 rounds simulés à 4 survivants.

## Campagne phase 1

`scripts/gen_emnist_campaign.py` — 6 bras x 4 seeds (91/92/93/24), EMNIST/ByClass.

| bras | rôle |
|---|---|
| `fedavg` | ancrage |
| `dmdcb` | DMD-CB fixe, `mu_M=0.1875` |
| `dmdcb_hi` | `mu_M=0.2027` — **contrôle décisif** : c'est l'intensité effective moyenne rapportée pour USV-0.25. S'il reproduit la baisse de VarBA, le terme USV n'est qu'un `mu_M` un peu plus fort. |
| `usv025` | reproduction de USV-0.25 |
| `tail_emp` | CVaR seuil empirique — doit montrer l'inertie de C1 |
| `tail_dual` | **Option B** : CVaR à seuil appris |
