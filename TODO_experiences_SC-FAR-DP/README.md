# Registre des expériences SC-FAR-DP

Dernière mise à jour : 2026-08-31.

Ce fichier est la source de vérité opérationnelle pour les expériences liées à
SC-FAR-DP. Il distingue strictement :

1. la réplication du rapport de stage (`R`) ;
2. la fidélité aux papiers FAR et FedFDP (`P`) ;
3. le papier 1, SC-FAR-DP en full-update (`S-P1`) ;
4. les extensions Partial SC-FAR-DP (`S-EXT1`) et Channel-Aware Adaptive
   Clipping (`S-EXT2`).

Les valeurs de epsilon ne sont comparables que si l'adjacency, l'unité
protégée, le modèle de confiance, le sampling et l'accountant sont identiques.

## Légende

| Symbole | Signification |
|---|---|
| ✅ | Terminé et résultats disponibles |
| 🟡 | En cours ou partiellement réalisé |
| ⬜ | Non commencé |
| ⚠️ | Bloqué par une dépendance théorique ou logicielle |

La colonne **Fait ?** répond littéralement à la question demandée : `Oui`,
`Partiel` ou `Non`.

## Règles communes avant lancement

- Figer le fichier de configuration, le commit Git et la version des
  dépendances.
- Enregistrer séparément `training_seed` et `partition_seed`.
- Conserver les mêmes partitions, initialisations, cohortes et batches dans
  chaque comparaison appariée.
- Pour le papier 1, utiliser `algorithm: scfar_dp`, `sample_fraction: 1.0`,
  aucun `layer_groups` et aucune amplification par sampling.
- L'ordre du pipeline du papier 1 est : attaque éventuelle, clipping
  user-level de confiance, référence F_CC, tilting contrôlé, bruit gaussien
  central, publication.
- Marquer les poids, distances, identités byzantines et erreurs par rapport à
  la moyenne honnête comme télémétrie oracle offline, non publiée.
- Rapporter au minimum moyenne, écart-type, intervalle de confiance, taille
  d'effet et différence appariée.

## R. Réplication et audit du rapport de stage

| ID | Expérience | Fait ? | Statut | Résultats / action restante |
|---|---|---:|---|---|
| R1-v4 | Fairness sans attaque, constructeur `dirichlet`, 27 runs | Oui | ✅ | `results/reproductions/internship_far_fedfdp/faithful/algorithm_fidelity_v4/exp1_fairness_no_attack` |
| R2-v4 | Fairness et robustesse sous IPM/Bit Flip, 54 runs | Oui | ✅ | `results/reproductions/internship_far_fedfdp/faithful/algorithm_fidelity_v4/exp2_fairness_robustness` |
| R3-v4 | Sample-level local DP-SGD, 54 runs | Oui | ✅ | `results/reproductions/internship_far_fedfdp/faithful/algorithm_fidelity_v4/exp3_privacy_fairness` |
| R1-BAL | R1 avec `client_dirichlet_balanced`, 27 runs | Oui | ✅ | `results/reproductions/internship_far_fedfdp/faithful/partition_ablation_client_dirichlet_balanced_v1/exp1_fairness_no_attack_client_dirichlet_balanced` |
| R3-BAL | R3 avec `client_dirichlet_balanced`, 54 runs | Oui | ✅ | 54/54 `metrics.json` valides et 54 statuts `completed` |
| R2-BAL | R2 avec `client_dirichlet_balanced` | Non | ⬜ | Créer la configuration seulement si l'ablation R1/R3 montre un effet de partition important |
| R-PART | Sensibilité à la partition, plusieurs `partition_seed` | Non | ⬜ | Ne pas confondre avec le changement de constructeur ; viser au moins 3 partitions |
| R-ANALYSE | Rapport R1/R3 équilibrés contre v4 et contre le stage | Oui | ✅ | `output/analysis/R1_R3_Client_Dirichlet_Balanced_vs_v4_and_Internship.md` |

### Diagnostics obligatoires pour chaque partition

- coefficient de variation des tailles locales ;
- entropie des labels par client ;
- part de la classe dominante ;
- nombre de classes présentes par client ;
- divergence inter-clients, par exemple Jensen-Shannon ;
- seed et empreinte de la partition.

## P. Fidélité aux papiers FAR et FedFDP

Cette section est un protocole cible. Une cellule n'est `faite` que si la
configuration exécutable, le test de conformité et les résultats existent.

| ID | Expérience | Fait ? | Statut | Action restante |
|---|---|---:|---|---|
| P-FAR-FMNIST | FAR, FashionMNIST/LeNet5, n=25, m=5, beta 0.1/0.5 | Non | ⬜ | Figer les configurations et reproduire les références CM(NNM), trMean(NNM), bucketing, RFA, NBS et CMLS |
| P-FAR-C10 | FAR, CIFAR-10/AlexNet, même protocole | Non | ⬜ | Créer la matrice exécutable et les tests de conformité |
| P-FAR-ATT | NA, BF, IPM, ALIE, Min-Max et Min-Sum | Non | ⬜ | Vérifier pour chaque attaque le point exact d'injection dans le pipeline |
| P-FED-FDP | FedFDP, q=0.05, C=0.1, sigma=2, C_l=2.5, sigma_l=5 | Non | ⬜ | Aligner la configuration sur le papier et enregistrer séparément les ledgers modèle/loss |
| P-ACCOUNT | Cross-validation de l'accountant RDP | Non | ⬜ | Comparer au moins à une bibliothèque indépendante et à un calcul analytique séparé |

## S-P1. Papier 1 : SC-FAR-DP full-update

### S0. Référence et sensibilité synthétique

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S0.1 | Tests unitaires de stabilité de F_CC | Oui | ✅ | Borne `2 min(C,tau)/n` vérifiée sur cas synthétiques |
| S0.2 | Exemple 1D atteignant ou approchant la borne de F_CC | Partiel | 🟡 | Produire un tableau par n, C et tau avec ratio borne/variation observée |
| S0.3 | Transport F_CC vers variation des scores, poids et agrégat | Oui | ✅ | Audit end-to-end exécuté sur 4 800 paires replace-one ; CSV et certificats dans `results/scpfar/sensitivity_s0_3` |
| S0.4 | Huber régularisé à nombre public d'itérations | Partiel | 🟡 | Tests unitaires présents ; ajouter sweep itérations/tolérance/erreur solveur |
| S0.5 | Comparateurs moyenne, CM, trMean et RFA | Oui | ✅ | Audit aléatoire et cohortes de stress exécutés ; résultats dans `results/scpfar/sensitivity_s0_5` |

#### Certificats et résultats de S0.3

S0.3 audite la chaîne déterministe complète sur des cohortes replace-one :

```text
U → F(U) → s(U) → q(U) → A(U) = Σᵢ qᵢ(U) uᵢ
```

L'instanciation exécutée est la suivante :

- borne des updates : `C = 1` ;
- seuil de centered clipping : `tau = 1` ;
- borne de normalisation des distances : `D_max = 2` ;
- facteur maximal des poids : `kappa_w = 2` ;
- tilt demandé : `alpha = 1` ;
- ancre publique : `r_0 = 0` ;
- ablation Huber : `gamma = 1` et `K = 10` itérations publiques fixes.

Le programme utilise
`alpha_eff = min(1, alpha_max(n, kappa_w))`, ce qui garantit
`q_i <= 2/n` pour chaque client.

Pour `F_CC`, les bornes vérifiées sont :

```text
delta_F = 2 min(C, tau) / n

eta_F = min(1, delta_F / D_max)

eta_k = min(1, (2C + delta_F) / D_max)

||s - s'||_1 <= eta_k + (n - 1) eta_F

||q - q'||_1 <= 2 alpha_eff [(kappa_w / n) eta_k + eta_F]

||A - A'||_2 <= min {
    2C,
    [2 C kappa_w / n] (1 + alpha_eff)
    + [2 C alpha_eff / D_max] delta_F
}
```

Dans les tableaux suivants, chaque cellule suit le format
**maximum observé / borne théorique**. Les maxima portent sur 200 paires
replace-one par taille de cohorte et par référence.

| n | Référence F | Score inchangé | Score remplacé | Scores, norme L1 | Poids, norme L1 | Agrégat, norme L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 0.3197 / 0.4000 | 0.0607 / 0.2000 | 0.0876 / 1.0000 | 0.1710 / 1.8000 | 0.0339 / 1.1770 | 0.3274 / 1.9770 |
| 10 | 0.1606 / 0.2000 | 0.0310 / 0.1000 | 0.0816 / 1.0000 | 0.1459 / 1.9000 | 0.0135 / 0.4866 | 0.1612 / 0.8866 |
| 20 | 0.0804 / 0.1000 | 0.0163 / 0.0500 | 0.0607 / 1.0000 | 0.1602 / 1.9500 | 0.0064 / 0.2242 | 0.0803 / 0.4242 |
| 40 | 0.0402 / 0.0500 | 0.0093 / 0.0250 | 0.0431 / 1.0000 | 0.1166 / 1.9750 | 0.0022 / 0.1079 | 0.0400 / 0.2079 |

Pour l'ablation Huber à `K = 10`, la stabilité de la référence utilise :

```text
delta_F_Huber <= [2 min(C, tau) / (gamma n)] (1 - rho^K)

rho = 1 / (1 + 2 gamma) = 1/3
```

| n | Référence Huber observée | Borne de référence | Agrégat observé | Borne d'agrégat |
|---:|---:|---:|---:|---:|
| 5  | 0.1600 | 0.4000 | 0.3247 | 1.9770 |
| 10 | 0.0786 | 0.2000 | 0.1579 | 0.8866 |
| 20 | 0.0411 | 0.1000 | 0.0816 | 0.4242 |
| 40 | 0.0204 | 0.0500 | 0.0407 | 0.2079 |

Le résultat est **zéro violation** sur les certificats évalués. Cela valide la
cohérence de l'implémentation avec les bornes sur les paires testées ; cela ne
remplace pas les preuves universelles. Consulter les
[résultats détaillés CSV](../results/scpfar/sensitivity_s0_3/replace_one_trials.csv)
et la [synthèse JSON](../results/scpfar/sensitivity_s0_3/summary.json).

#### Résultats de S0.5 : comparateurs non certifiés

S0.5 compare la moyenne arithmétique, la médiane coordonnée (`CM`), le
trimmed mean (`trMean`) et la médiane géométrique (`RFA`) sur des cohortes
replace-one. Le protocole utilise `d = 64`, `C = 1`, `D_max = 2`, 200 paires
gaussiennes clippées par taille de cohorte, puis deux paires déterministes de
stress. Les mêmes scores bornés et le même tilting contrôlé que S0.3 sont
ensuite appliqués.

La colonne **F aléatoire** est le déplacement maximal de la référence sur les
200 paires aléatoires. La colonne **F stress** est le maximum sur les deux
cohortes construites pour franchir la majorité médiane ou les frontières du
trimmed mean. La dernière colonne donne le déplacement maximal de l'agrégat
SC-FAR sur l'ensemble des cas.

| n | Référence | F aléatoire, max | F stress, max | Agrégat, max |
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

La paire de stress de la moyenne atteint exactement `2C/n`, ce qui vérifie le
cas test attendu. En revanche, CM et RFA conservent un déplacement de référence
macroscopique quand `n` augmente. Les tirages aléatoires seuls auraient donc
donné une conclusion trop optimiste. S0.5 ne fournit **aucun certificat
`O(C/n)`** à CM, trMean ou RFA. Lorsqu'une de ces références est utilisée dans
SC-FAR-DP, l'implémentation doit conserver le calibrage global conservateur
`Delta_2 <= 2C`. Cette borne concerne l'agrégat final, qui reste une combinaison
convexe d'updates clippés, et non la stabilité propre de la référence.

Les 3 232 mesures détaillées se trouvent dans
[le CSV S0.5](../results/scpfar/sensitivity_s0_5/replace_one_comparators.csv),
avec la [synthèse JSON](../results/scpfar/sensitivity_s0_5/summary.json). Le
lanceur CPU Toubkal est `hpc/run_s0_5_reference_comparators_cpu.slurm`.

### S1. Robustesse-biais de F_CC

Matrice prête : `configs/scpfar/paper1/s1_reference_tradeoff.yaml`, 1 908
tâches full-update. Les expériences ne sont pas encore exécutées.

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S1.1 | Sweep tau/C | Non | 🟡 | Grille prête : `{0.25,0.5,1,2}` |
| S1.2 | Qualité de l'ancre r0 | Non | 🟡 | Ancre nulle fixe, EMA `{0.1,0.5}` et publication précédente |
| S1.3 | Taux/cadence publique de mise à jour | Non | 🟡 | Cadence ciblée `{1,5}` rounds préspécifiée |
| S1.4 | Contamination byzantine de F_CC | Non | 🟡 | `n=25`, `b={0,2,5}`, soit fractions exactes `{0,0.08,0.20}` |
| S1.5 | Biais de queue honnête | Non | 🟡 | Métrique instrumentée dans chaque tâche S1 |

### S2. Chaîne d'ablation full-update

Matrice prête : `configs/scpfar/paper1/s2_full_update_ablations.yaml`, 360
tâches. Elle utilise exclusivement le protocole full-update du papier 1.

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S2.1 | Configuration canonique `scfar_dp` full-update | Non | 🟡 | Manifeste et validation disponibles |
| S2.2 | FedAvg, F_CC seul, FAR, FAR user-clippé | Non | 🟡 | Bras exécutables dans la sous-matrice S2a |
| S2.3 | SC-FAR sans DP | Non | 🟡 | Bras exécutable dans S2a |
| S2.4 | DP-FedAvg et DP-FAR calibré avec 2C | Non | 🟡 | Bras appariés dans S2b |
| S2.5 | SC-FAR-DP avec 2C | Non | 🟡 | Baseline conservatrice dans S2b |
| S2.6 | SC-FAR-DP avec sensibilité certifiée | Non | 🟡 | Bras prêt ; interpréter après sélection S1 |

### S3. Inclusion, fairness et attaques

Matrice prête : `configs/scpfar/paper1/s3_inclusion_attacks.yaml`, 1 224
tâches. `kappa_w={1.25,2,5}` et
`alpha/alpha_max={0,0.25,0.5,0.75,1}`. Les honest outliers sont les 20 % des
vingt clients toujours honnêtes (quatre clients) ayant la plus forte divergence Jensen-Shannon des labels,
définis avant training et utilisés uniquement comme oracle d'évaluation.

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S3.1 | Sweep alpha/alpha_max et kappa_w | Non | 🟡 | Grille analytique prête et validée avant exécution |
| S3.2 | Honest outliers préenregistrés | Non | 🟡 | Identités figées par scénario et partition seed |
| S3.3 | Fairness sans attaque | Non | 🟡 | Worst-20, gap, variance et masse d'inclusion instrumentés |
| S3.4 | Robustesse sous BF, IPM, ALIE, Min-Max, Min-Sum | Non | 🟡 | Intensités préspécifiées ; Min-Max/Min-Sum conservent le point furtif canonique |
| S3.5 | Attaque ciblée éventuelle | Non | ⬜ | Utiliser l'ASR uniquement pour une vraie attaque ciblée |

### S4. Central user-level DP et accountant

Matrice prête : `configs/scpfar/paper1/s4_central_dp.yaml`, 720 tâches. La
voie principale a `q=1`, sans amplification par sampling, et compare
`epsilon={1,3,6,10,infini}` sous une même adjacency replace-one.

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S4.1 | Validation du mécanisme gaussien à q=1 | Partiel | 🟡 | Formule/tests internes validés ; cross-check externe publication-grade restant |
| S4.2 | Sweep epsilon dans {1,3,6,10,infini} | Non | 🟡 | Profils prêts ; `infini` désactive réellement le bruit |
| S4.3 | Borne 2C contre borne certifiée | Non | 🟡 | Bras appariés dans la matrice S4 |
| S4.4 | Composition multi-round RDP | Partiel | 🟡 | Ledger instrumenté ; validation end-to-end sur sorties à exécuter |
| S4.5 | Sampling client | Non | ⬜ | Hors papier 1 principal ; lane secondaire ultérieure |

### S5. Matrice principale et statistiques

| ID | Expérience | Fait ? | Statut | Critère de sortie |
|---|---|---:|---|---|
| S5.1 | Pilote full-update, 3 training seeds | Non | ⚠️ | Dépend de S2.1 et des baselines exécutables |
| S5.2 | Campagne principale, 5 training seeds x 3 partitions | Non | ⬜ | IC, tailles d'effet et tests appariés pré-spécifiés |
| S5.3 | FMNIST/LeNet5 | Non | ⬜ | Matrice clean/attaque et privacy sweep |
| S5.4 | CIFAR-10/AlexNet ou ResNet18-GN | Non | ⬜ | Choisir le modèle avant les résultats et ne plus le modifier |
| S5.5 | Tableau principal du papier | Non | ⬜ | Accuracy, Worst-20, gap, variance, robustesse, epsilon, poids et coûts |

## S-EXT1. Partial SC-FAR-DP

| ID | Expérience | Fait ? | Statut | Action restante |
|---|---|---:|---|---|
| EXT1.1 | Smoke test partial CIFAR-10 | Oui | ✅ | Résultat de smoke disponible, non suffisant pour un claim |
| EXT1.2 | Seuils C_g par groupe | Non | ⬜ | L'implémentation emploie encore principalement un seuil global |
| EXT1.3 | G dans {1,2,4,8} et calendriers publics | Non | ⬜ | Cyclique, sans remise et uniforme |
| EXT1.4 | ResNet18-GN CIFAR-10/100 | Non | ⬜ | Comparer à full-update au même budget DP |
| EXT1.5 | MobileNet | Non | ⬜ | Contrôle face à un modèle déjà efficient |
| EXT1.6 | Coût end-to-end | Non | ⬜ | Joules-, Bytes- et Time-to-Accuracy |

## S-EXT2. Channel-Aware Adaptive Clipping

| ID | Expérience | Fait ? | Statut | Action restante |
|---|---|---:|---|---|
| EXT2.1 | Implémentation C_t^ch = phi(h_t) | Non | ⬜ | h_t doit être public ou correctement privatisé |
| EXT2.2 | Seuil fixe contre distribution-aware contre channel-aware | Non | ⬜ | Isoler l'effet propre du canal |
| EXT2.3 | Sweep SNR/bande passante | Non | ⬜ | Ajouter au moins deux budgets DP et clean/attaque |
| EXT2.4 | Quantification et énergie | Non | ⬜ | Séparer erreur de quantification et énergie E_tx |
| EXT2.5 | Résultats end-to-end | Non | ⬜ | Accuracy, clipping rate, sigma_DP, bits et Joules-to-Accuracy |

## Ordre de lancement recommandé

1. Valider les quatre manifestes avec `scripts/run_scfar_paper1.py --validate`.
2. Exécuter un pilote d'un round des bras S1–S4.
3. Exécuter S1, puis choisir publiquement le régime `tau/ancre` avant de lire
   les résultats confirmatoires S2/S4.
4. Exécuter S2 et S4, puis S3 avec les identités d'outliers préenregistrées.
5. Cross-valider l'accountant avec une seconde implémentation avant le gel des
   tableaux du papier.
6. Construire S5 comme petite matrice confirmatoire à partir des décisions de
   S1–S4, sans recopier toutes les grilles exploratoires.
7. Commencer S-EXT1 seulement après validation du papier 1.
8. Traiter S-EXT2 comme un protocole de papier ultérieur.

## Critères go/no-go du papier 1

| Gate | Go | No-go / repositionnement |
|---|---|---|
| Robustesse-biais | Une plage de tau et une ancre publique conservent une référence utile | Aucun choix ne limite les Byzantins sans écraser la dispersion honnête |
| Sensibilité | La borne end-to-end conserve l'échelle 1/n avec constantes utiles | La meilleure borne valide revient à l'ordre C |
| Inclusion | Au moins certains honest outliers vérifient q_i > 1/n | Les poids deviennent uniformes ou presque one-hot |
| Attaques | Le gain Worst-20 n'augmente pas la masse byzantine de façon incontrôlée | Le tilting favorise systématiquement les attaquants furtifs |
| DP | L'accountant est reproduit indépendamment et toutes les méthodes partagent le même modèle DP | Comparaison fondée sur des epsilon non comparables |
