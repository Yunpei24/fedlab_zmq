# DT-LDP-FAR — protocole de stress apparié à 25 clients

## Question scientifique

La campagne sépare trois affirmations qui ne doivent pas être confondues :

1. le retard rend le poids appliqué au tour `t` indépendant du bruit frais de
   ce même tour, conditionnellement à l'historique public ;
2. cette indépendance réduit l'amplification du bruit lorsque le mécanisme
   courant sélectionne effectivement les perturbations les plus fortes ;
3. cette réduction mécanistique se traduit par un gain d'accuracy ou de
   fairness pendant un entraînement réel.

La première affirmation vient de la construction de DT-LDP-FAR. Les deux
autres exigent des expériences. Une comparaison d'accuracy dans un régime où
les poids restent presque uniformes ne peut ni les valider ni les réfuter.

## 1. Stress mécanistique strictement apparié

Le script `scripts/run_dt_ldp_far_delay_mechanistic_stress.py` construit des
vecteurs propres, une perturbation fraîche et une perturbation passée. Pour
chaque tirage, les deux variantes utilisent exactement :

- les mêmes vecteurs propres ;
- la même perturbation fraîche ;
- le même clipping serveur `U` ;
- la même référence `F_CC` et les mêmes constantes `rho` et `D_score` ;
- la même valeur de `alpha`.

La variante courante calcule ses poids avec les scores de la perturbation
fraîche. La variante retardée calcule ses poids avec un tirage passé
indépendant, puis applique ces poids à la même perturbation fraîche que la
variante courante. La différence appariée ne peut donc pas être attribuée à
deux réalisations de bruit différentes.

Il s'agit d'un diagnostic oracle sans entraînement et sans claim de
confidentialité. Les résultats sont dans :

- `results/dt_ldp_far/mechanistic_n25_paired_v2/` ;
- `output/analysis/DT_LDP_FAR_N25_Stress_Mecanistique_Apparie.md`.

## 2. Pourquoi le score borné n'implique pas un logit-span inférieur à 1

Les scores sont définis par

```text
s_i,t = min( ||X_i,t - F_t||_2 / D_score, 1 ).
```

Ils satisfont donc `0 <= s_i,t <= 1`. Le score-span est

```text
S_t = max_i s_i,t - min_i s_i,t,
```

d'où `0 <= S_t <= 1`. La softmax ne reçoit cependant pas `S_t` directement :
elle reçoit les logits `alpha * s_i,t`. Leur plage est

```text
R_t = alpha * S_t.
```

Pour `S_t > 0`, on a donc exactement :

```text
R_t >= 1  si et seulement si  alpha >= 1 / S_t.
```

Conséquences :

- si `alpha < 1`, alors `R_t < 1` pour tout tour ;
- si `alpha = 1`, l'égalité `R_t = 1` n'est atteinte que lorsque les scores
  occupent toute la plage `[0,1]` ;
- si `alpha > 1`, `R_t` peut dépasser 1 sans qu'aucun score sorte de `[0,1]`.

Exemple : si `S_t = 0,20`, il faut `alpha >= 5` pour avoir `R_t >= 1`. Dans la
campagne n=25 précédente, `S_t` était proche de `0,19` et le stress
`2 alpha_max = 1,47` donnait seulement `R_t` proche de `0,28`. Il était donc
normal que la softmax demeure proche de l'uniforme.

`R_t >= 1` n'est pas une condition de validité de DT-LDP-FAR. C'est un seuil
de stress expérimental : le rapport entre le poids maximal et le poids minimal
vaut `exp(R_t)`, donc `R_t = 1` autorise déjà un rapport d'environ `2,72`.

## 3. Deux lanes distinctes pour alpha

### Lane certifiée

Pour des scores dans `[0,1]`, le choix

```text
alpha <= log( kappa_w (n-1) / (n-kappa_w) )
```

garantit `max_i omega_i <= kappa_w/n`. À `n=25` :

| kappa_w | alpha maximal | cap du poids |
|---:|---:|---:|
| 2 | 0,7357 | 0,08 |
| 4 | 1,5198 | 0,16 |
| 10 | 2,7726 | 0,40 |

La lane principale du papier reste `kappa_w=2`. Les valeurs plus grandes sont
des ablations certifiées, mais elles donnent une garantie d'influence plus
faible.

### Lane de falsification

Les profils `stress8_uncertified` et `stress12_uncertified` multiplient la
borne correspondant à `kappa_w=2` par 8 et 12. Ils conservent la local-DP,
car la pondération serveur est un post-traitement des uploads privés. Ils ne
conservent pas le certificat d'influence `2/n` et ne doivent jamais être
présentés comme la méthode certifiée.

## 4. Gate de transfert end-to-end

Le stress synthétique ne suffit pas à conclure sur Fashion-MNIST. La matrice
`decisive_stage4_mechanistic_transfer_screen_n25.yaml` vérifie donc, sur une
seed de développement et 20 rounds, que le régime réel possède simultanément :

- un score-span médian d'au moins `0,15` ;
- un logit-span médian d'au moins `1` ;
- une saturation p90 d'au plus `25 %` ;
- un poids maximal médian d'au moins `1,25/n` ;
- une concentration médiane `n Σ_i omega_i^2` d'au moins `1,03` ;
- une corrélation médiane poids courant–bruit d'au moins `0,10` ;
- une corrélation retardée–bruit en valeur absolue d'au plus `0,15` ;
- une clé de randomness pairing identique et une séquence de normes de bruit
  identique entre les deux bras.

Le transcript de ce gate contient volontairement des oracles de simulation.
Il ne s'agit pas d'une sortie LDP publiable. La campagne longue emploie les
profils `dp_far_current_matched` et `dt_ldp_far`, qui suppriment ces canaux
auxiliaires non comptabilisés.

## 5. Ordre des campagnes

| Ordre | Matrice | Décision |
|---:|---|---|
| 1 | `decisive_stage4_mechanistic_transfer_screen_n25.yaml` | tester si le stress se transfère aux updates réels |
| 2 | analyseur `analyze_dt_ldp_far_n25_mechanistic_transfer.py` | promouvoir ou arrêter la lane longue |
| 3 | `decisive_stage4_server_clip_ablation_n25.yaml` | comparer `U=0,42` à un clipping serveur inactif |
| 4 | `decisive_stage4_current_delay_references_n25.yaml` | comparer F_CC, Huber et RFA sous les mêmes attaques |
| 5 | `decisive_stage4_long_horizon_n25.yaml` | 80/120 rounds, seulement si le gate de transfert est positif |

La matrice longue contient le sur-ensemble pré-enregistré des trois géométries
et des quatre niveaux de tilt, soit 162 tâches possibles. Le lanceur
conditionnel n'exécute que la géométrie et le tilt sélectionnés par le gate,
ainsi que leur contrôle uniforme sur le même chemin serveur : 18 tâches sont
donc lancées après une promotion, et non les 162 tâches du sur-ensemble.

Les comparaisons courant/retardé partagent une clé de randomness pairing qui
exclut uniquement la méthode et le délai. Elle atteste l'intention
expérimentale : mêmes seeds, partition, ordre client, tirages de Poisson, bruit
DP et attaque. Dans un entraînement end-to-end, les modèles peuvent ensuite
diverger causalement ; les gradients propres ne restent donc pas identiques
après le premier tour. L'égalité exacte des vecteurs propres est réservée au
stress mécanistique.

## 6. Interprétation des attaques et du clipping

Les observations antérieures sur Bit-Flip, IPM et ALIE portent sur toutes les
références testées, pas uniquement sur F_CC :

- sous Bit-Flip ×10, tous les uploads byzantins sont clippés pour F_CC, Huber,
  RFA, CM(NNM) et trMean(NNM), mais leur masse totale reste légèrement
  supérieure à la masse uniforme `5/25 = 0,20` ;
- sous IPM et ALIE, aucun upload byzantin n'est clippé pour ces cinq
  références : les attaques restent furtives vis-à-vis d'un test de norme.

Le clipping serveur borne donc les messages de grande norme ; il ne constitue
ni un détecteur universel de Byzantins, ni une preuve que F_CC est
empiriquement supérieur. La comparaison multi-références sert précisément à
séparer ces questions.
