# DT-LDP-FAR — stress end-to-end de l'auto-sélection du bruit

## Question falsifiable

Le retard d'un tour n'est utile que si la règle courante donne, au même tour,
un poids plus grand aux uploads qui ont reçu une perturbation DP fraîche plus
forte. La campagne teste cette condition avant toute sélection sur l'accuracy.

Le diagnostic principal utilise un update contrefactuel sans bruit construit
avec les mêmes pas de Poisson et les mêmes masques stochastiques que l'update
privé. Après le clipping serveur :

```text
zeta_effectif_i,t = X_prive_i,t - X_sans_bruit_i,t.
```

Cette quantité est un oracle de simulation. Elle n'appartient pas au transcript
LDP publiable.

## Matrice de découverte

La matrice
`decisive_stage5_end_to_end_stress_discovery_n25.yaml` contient 24 runs :

- 25 clients et trois seeds : 28, 36 et 54 ;
- DP-FAR courant et DT-LDP-FAR retardé d'un tour ;
- Fashion-MNIST, LeNet-5, Dirichlet 0,1, six rounds et dix pas de Poisson ;
- clipping par exemple C = 4 et clipping serveur U = 0,42 ;
- un tilt de falsification élevé, explicitement non certifié en influence ;
- quatre cellules ciblées : bruit homogène à epsilon 1, bruit hétéroscédastique
  public à epsilon maximal 4, score de modèle complet, score de dernière couche
  et score sur 256 coordonnées publiques.

L'hétéroscédasticité emploie des multiplicateurs publics supérieurs ou égaux à
un. Aucun client ne dépasse donc le budget epsilon annoncé ; certains clients
reçoivent une confidentialité plus forte.

## Gate préenregistré

Le premier tour est retiré des médianes parce que les poids retardés y sont
nécessairement uniformes. Une cellule ne passe le gate multi-seeds que si, pour
chacune des trois seeds :

```text
median_t(alpha * score_span_t) >= 1
median_t corr_i(poids_courant_i,t, ||zeta_effectif_i,t||) > 0
median_t corr_i(poids_courant_i,t,
                ||zeta_effectif_i,t|| / echelle_publique_i) > 0
median_t(n * somme_i poids_i,t^2) >= 1.05
p90_t(saturation_des_scores) <= 0.25
```

La seconde corrélation empêche une hétéroscédasticité persistante de se faire
passer pour une dépendance à la réalisation fraîche du bruit.

Le gate vérifie également l'identité des clés de randomness pairing et des
séquences de normes du bruit entre les bras courant et retardé. L'accuracy est
enregistrée, mais n'intervient jamais dans la promotion.

## Confirmation si le gate est positif

Une cellule positive n'établit pas encore un avantage d'apprentissage. Elle
autorise une confirmation préenregistrée :

1. au moins cinq nouvelles seeds tenues hors de la découverte ;
2. bras uniformes, courant et retardé sur une randomness appariée ;
3. horizons 40, 80 et 120 rounds ;
4. budgets epsilon 1, 2, 4, 8 et contrôle sans bruit ;
5. Dirichlet 0,1 et 0,5 ;
6. F_CC principal, Huber et RFA en ablation ;
7. aucune attaque, puis Bit-Flip, IPM, ALIE, Min-Max et Min-Sum à plusieurs
   fractions byzantines ;
8. mesure directe de la MSE du bruit agrégé, du coût de retard, de l'accuracy,
   de Worst-20, du gap, de la variance cliente et du temps de convergence.

Le bénéfice mécanistique demandé est une MSE du bruit agrégé plus faible pour
le bras retardé. Le bénéfice final demandé est que cette réduction excède le
coût de vieillissement des scores et améliore au moins une métrique d'utilité ou
de fairness sans dégradation inacceptable des autres.

## Conclusion autorisée en cas de résultat positif

Si le gate est confirmé sur des seeds nouvelles et que la campagne longue
montre une réduction significative de MSE et un gain d'utilité, la conclusion
sera :

> Sous des conditions explicites de géométrie du score et de bruit local, la
> pondération FAR au même tour crée une auto-sélection mesurable du bruit DP.
> Rendre les poids prévisibles à partir du transcript passé supprime cette
> dépendance à la réalisation fraîche, réduit l'amplification du bruit agrégé
> et peut améliorer le compromis confidentialité–utilité–fairness.

Cette conclusion restera conditionnelle au régime observé. Elle ne signifiera
ni que tout bruit LDP est amplifié par FAR, ni que le retard est toujours
meilleur, ni que F_CC fournit à lui seul une robustesse byzantine universelle.

## Lane CIFAR-10/AlexNet

CIFAR-10/AlexNet est une validation externe après la confirmation
Fashion-MNIST. Le score de modèle complet, de dernière couche et de sous-espace
public doivent y être comparés séparément : la grande dimension peut augmenter
la perturbation totale, mais aussi concentrer les normes gaussiennes et masquer
l'auto-sélection. La complexité du dataset n'est donc pas, à elle seule, une
garantie de succès.
