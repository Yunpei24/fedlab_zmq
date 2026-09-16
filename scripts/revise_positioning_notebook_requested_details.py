"""Emit a surgical apply_patch for the current user-edited notebook; no rebuild."""
from pathlib import Path
import ast
import copy
import difflib
import json
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks/LDP_Gradient_FAR_Positioning_Analysis_Editable.ipynb"


def transform(nb):
    cells = {c.id: c for c in nb.cells}
    changed = set()
    def edit(cid, source):
        c = cells[cid]
        if c.source != source:
            changed.add(cid)
            c.source = source
            # Outputs are replaced only after successful execution below;
            # keeping them here avoids a huge image-only source patch.
    def replace(cid, old, new):
        source = cells[cid].source
        assert source.count(old) == 1, (cid, old[:100])
        edit(cid, source.replace(old, new))
    def insert_after(cid, additions):
        at = next(i for i, c in enumerate(nb.cells) if c.id == cid) + 1
        for new_id, source in additions:
            assert new_id not in cells, new_id
            cell = nbformat.v4.new_code_cell(source, id=new_id)
            cell.metadata.update(editable=True, deletable=True)
            nb.cells.insert(at, cell); at += 1
            changed.add(new_id)

    edit('rcig-review-controls-intro', cells['rcig-review-controls-intro'].source + "\n\n"
         "**Oui, sans DP avec α = 0 existe dans les données.** Les courbes suivantes "
         "l'affichent en noir pointillé, avec les trois autres conditions de l'écran E, "
         "pour l'accuracy, les losses, Worst-20 et le gap. Il s'agit de la seed 137, "
         "pas d'une quatrième courbe à trois seeds à ajouter artificiellement à F. "
         "Les panneaux C de la section 3.3 contiennent aussi α = 0 sans DP.")
    replace('rcig-review-controls-table',
        'values = part.groupby("round")[metric].mean()',
        'values = part.groupby("round")[metric].mean().dropna()\n'
        '            if values.empty:\n                continue\n'
        '            # Les segments relient seulement les observations enregistrées.\n'
        '            assert part.seed.nunique() == 1 and set(part.seed) == {137}')
    replace('rcig-review-controls-table',
        'linestyle="-" if dp else "--")',
        'linestyle="-" if dp else "--",\n'
        '                    color="black" if (not dp and alpha == 0) else None)')
    insert_after('rcig-review-controls-table', [
        ('requested-e-test-loss', '_plot_controls_e("test_loss", "Loss test")'),
        ('requested-e-client-loss', '_plot_controls_e("client_loss_heldout", "Loss moyenne des évaluations clientes")'),
        ('requested-e-worst20', '_plot_controls_e("worst20_pct", "Worst-20 (%) · plus élevé = meilleur")'),
        ('requested-e-gap', '_plot_controls_e("gap_pp", "Gap Best-20 − Worst-20 (pp) · plus faible = meilleur")'),
    ])
    replace('228f01f8bee42d78', '## 5. Poids FAR : poids maximal, concentration et entropie',
        '## 5. Poids FAR : poids minimal, poids maximal, concentration et entropie')
    replace('228f01f8bee42d78', '| n × λ max | 1 |',
        '| λ min | 1/n | Plus petit poids du tour. Une valeur proche de zéro indique qu’au moins un client contribue très peu, sans dire s’il est honnête ou malveillant. |\n'
        '| n × λ min | 1 | Comparaison à l’uniforme. À n = 10, 0,2 signifie λ min = 0,02, soit un cinquième du poids uniforme. |\n'
        '| n × λ max | 1 |')
    replace('228f01f8bee42d78', 'Aucun des\ntrois indicateurs', 'Aucun de ces\nindicateurs')
    edit('228f01f8bee42d78', cells['228f01f8bee42d78'].source + r'''

### À quel tour ces quantités sont-elles déterminées ?

**À chaque tour t = 1, …, 20**, avec les poids effectivement appliqués à ce
tour, on calcule :

$$
\lambda_{\min,t}=\min_i\lambda_{i,t},\qquad
M_t=n\max_i\lambda_{i,t},\qquad
Q_t=n\sum_i\lambda_{i,t}^2,\qquad
h_t=\frac{-\sum_i\lambda_{i,t}\log\lambda_{i,t}}{\log n}.
$$

Pour chaque indicateur z, les barres et le premier tableau de cette section
affichent exactement :

$$
\frac{1}{S}\sum_{s=1}^{S}\operatorname{median}_{t=1,\ldots,20} z_t^{(s)},
\qquad S=3.
$$

Ce n'est donc **ni une valeur au seul tour 20, ni une moyenne de tous les
clients et tours mélangés**. Pour 20 valeurs, la médiane est la moyenne des
10e et 11e valeurs après tri : elle ne correspond pas nécessairement à un
tour observé. Le client qui réalise le minimum ou le maximum peut changer
d'un tour à l'autre. Un tableau distinct donne également les valeurs au
**tour 20**, pour ne pas confondre résumé temporel et état final. Les poids
disponibles sont relus dans les metrics.json, y compris λ min.
''')
    original26 = cells['294b03073a3f9c6a'].source
    prefix = original26[:original26.index('# Les métriques de poids')]
    edit('294b03073a3f9c6a', prefix + '''# Recalcul des résumés depuis chaque tour, dont min_client_weight.
weights_rounds = rounds_df[
    (rounds_df.campaign == READABLE_CAMPAIGN)
    & (rounds_df.phase == READABLE_PHASE)
    & rounds_df.dp_enabled & np.isclose(rounds_df.target_epsilon, 4.0)
    & (rounds_df.attack == "none")
].copy()
weights_rounds["lambda_min"] = weights_rounds.min_weight
weights_rounds["n_lambda_min"] = weights_rounds.n_clients * weights_rounds.min_weight
weights_rounds["n_lambda_max"] = weights_rounds.n_clients * weights_rounds.max_weight
weights_rounds["entropy_normalized"] = weights_rounds.weight_entropy / np.log(weights_rounds.n_clients)
weights_rounds["effective_fraction"] = weights_rounds.effective_clients / weights_rounds.n_clients
WEIGHT_COLUMNS = {
    "lambda_min": "λ min", "n_lambda_min": "n × poids min",
    "n_lambda_max": "n × poids max", "weight_concentration": "n × Σ poids²",
    "entropy_normalized": "Entropie / log(n)", "effective_fraction": "Clients effectifs / n",
    "score_span": "Score-span médian", "logit_range": "Logit-range médian",
}
for uid, part in weights_rounds.groupby("run_uid"):
    assert part["round"].tolist() == list(range(1, 21)), uid
    assert np.isfinite(part[list(WEIGHT_COLUMNS)].to_numpy(dtype=float)).all(), uid
    assert (part.min_weight >= 0).all() and (part.min_weight <= part.max_weight).all()
weight_run_keys = ["n_clients", "alpha", "reference", "seed", "run_uid"]
weight_run_medians = weights_rounds.groupby(weight_run_keys)[list(WEIGHT_COLUMNS)].median().reset_index()
weight_summary = weight_run_medians.groupby(["n_clients", "alpha", "reference"]).agg(
    Seeds=("seed", "nunique"), **{key: (key, "mean") for key in WEIGHT_COLUMNS}
).reset_index().rename(columns={
    "n_clients": "Clients", "alpha": "α", "reference": "Référence", **WEIGHT_COLUMNS})
weight_final_summary = weights_rounds[weights_rounds["round"] == 20].groupby(
    ["n_clients", "alpha", "reference"]
).agg(Seeds=("seed", "nunique"), **{key: (key, "mean") for key in WEIGHT_COLUMNS}).reset_index().rename(
    columns={"n_clients": "Clients", "alpha": "α", "reference": "Référence", **WEIGHT_COLUMNS})
for n, part in weight_summary.groupby("Clients"):
    display(Markdown(f"### n={n} — médianes des tours 1–20, puis moyenne sur trois seeds"))
    display(part[["α", "Référence", "Seeds", "λ min", "n × poids min", "n × poids max",
                  "n × Σ poids²", "Entropie / log(n)"]].round(4))
    display(Markdown("**Comparaison : valeurs au seul tour 20**, puis moyenne sur les mêmes seeds."))
    display(weight_final_summary[weight_final_summary.Clients == n][[
        "α", "Référence", "Seeds", "λ min", "n × poids min", "n × poids max",
        "n × Σ poids²", "Entropie / log(n)"
    ]].round(4))
display(Markdown("### Géométrie : médianes des tours 1–20, puis moyenne entre seeds"))
display(weight_summary[["Clients", "α", "Référence", "Seeds", "Score-span médian", "Logit-range médian"]].round(4))
display(Markdown(
    "**Identité par tour :** log(λ max / λ min) = |α| × score-span. "
    "Ne pas calculer le log du rapport des moyennes/médianes pour vérifier cette identité : "
    "le log et les résumés statistiques ne commutent pas."
))
weight_run_medians.to_csv(EXPORT_ROOT / "weights_per_seed_median_rounds_1_20.csv", index=False)
weight_final_summary.to_csv(EXPORT_ROOT / "weights_final_round_20.csv", index=False)
''')
    replace('7a7e19849d9a2299', 'title=f"{ylabel} à α=2, DP ε=4, sans attaque")',
        'title=f"{ylabel} · α=2 · DP ε=4 · sans attaque\\nMédiane tours 1–20 par run, puis moyenne entre seeds")')
    insert_after('7a7e19849d9a2299', [
        ('requested-minimum-weight-plot', '_plot_weight_metric("n × poids min", "n × poids minimal", 1.0)'),
    ])
    replace('rcig-review-weight-bridge', 'weight_bridge_source["n_lambda_max"]',
        'weight_bridge_source["n_lambda_min"] = weight_bridge_source.n_clients * weight_bridge_source.min_weight\n'
        'weight_bridge_source["lambda_min"] = weight_bridge_source.min_weight\n'
        'weight_bridge_source["n_lambda_max"]')
    replace('rcig-review-weight-bridge', '.agg(qmax=("n_lambda_max", "median"),',
        '.agg(lambda_min=("lambda_min", "median"), qmin=("n_lambda_min", "median"), qmax=("n_lambda_max", "median"),')
    replace('rcig-review-weight-bridge', '("qmax", "n × λ max", 3),',
        '("lambda_min", "λ min", 4), ("qmin", "n × λ min", 3), ("qmax", "n × λ max", 3),')
    replace('rcig-review-weight-bridge', 'display(part.drop(columns="Clients"))',
        'display(part[["Condition", "Seeds", "λ min", "n × λ min", "n × λ max", "Q = n × Σ λ²", "H / log(n)"]])\n'
        '    display(part[["Condition", "Seeds", "Accuracy test (%)", "Worst-20 (%)", "Gap (pp)"]])')
    edit('rcig-review-weight-bridge-intro', cells['rcig-review-weight-bridge-intro'].source +
         '\n\n**Tours :** les poids sont des médianes sur t = 1…20, puis moyenne ± écart-type '
         'des trois médianes. Accuracy, Worst-20 et gap sont mesurés au seul tour 20.')
    edit('8fe526cea82974e1', cells['8fe526cea82974e1'].source + r'''

### Comment ε détermine-t-il le bruit dans ces runs ?

**Oui : ici on fixe ε cible, puis on calibre σ avant l'entraînement.** Mais ε
seul ne suffit pas : on fixe aussi δ, le nombre total L de releases, la taille
publique N du dataset et la taille b du batch. L'ordre de Rényi est noté a
ci-dessous pour ne pas le confondre avec α de FAR.

Dans ces campagnes, le batch est uniforme de taille fixe, sans remise **dans
le batch** ; il peut réutiliser des exemples aux tours suivants. Chaque client
envoie un gradient par tour (L = T, sans époque locale d'optimisation) :

$$
\bar g_{i,t,j}=g_{i,t,j}\min\{1,C/\|g_{i,t,j}\|_2\},\qquad
Y_{i,t}=\frac{1}{b}\left(\sum_{j\in B_{i,t}}\bar g_{i,t,j}+\sigma_i C Z_{i,t}\right),
\quad Z_{i,t}\sim\mathcal N(0,I).
$$

On laisse un gradient nul inchangé. L'écart-type du bruit **dans le message
moyen envoyé** est σᵢC/b par coordonnée, pas σᵢC. Pour le remplacement d'un
exemple dans le batch, la sensibilité vaut Δ = 2C/b ; le ratio bruit/sensibilité
transmis au comptable est donc **z = σᵢ/2**, pas σᵢ.

Notons R_WOR(a, q, z) la borne RDP par release du comptable sans remise,
avec q = b/N. Pour des paramètres constants et L releases :

$$
\widehat\varepsilon(\sigma)
=\max\left\{0,\min_{a\in\mathcal A}
\left[L R_{\rm WOR}(a,b/N,\sigma/2)
+\log\frac{a-1}{a}-\frac{\log\delta+\log a}{a-1}\right]\right\},
\qquad \mathcal A=\{2,3,4,5,8,10,16,20,32,64\}.
$$

$$
\sigma_{\rm base}\ \approx\ \inf\{\sigma>0:
\widehat\varepsilon(\sigma)\le\varepsilon_{\rm cible}\}.
$$

Le code résout cette inversion par **dichotomie**, avec une tolérance de
10⁻⁴ sur ε ; ce n'est pas une formule fermée « σ = constante/ε ». La valeur
réalisée doit donc être vérifiée, pas supposée strictement égale à la cible.
Pour un batch complet q = 1, la brique RDP devient a/(2z²). Pour q < 1,
on utilise la borne d'échantillonnage sans remise, **pas la somme du SGM
Poisson**. [Source de l'amplification RDP : Wang, Balle et Kasiviswanathan, AISTATS 2019](https://proceedings.mlr.press/v89/wang19b.html).

À protocole fixé, ε plus petit exige davantage de bruit. Le seuil C fixe
l'amplitude absolue σC/b ; dans ce mécanisme, il se simplifie dans le ratio
bruit/sensibilité. Sous bruit hétéroscédastique, le protocole applique ensuite
des facteurs publics cᵢ : σᵢ = cᵢ σ_base, et comptabilise chaque client.
On rapporte le maximum des ε clients : leurs budgets ne sont pas tous égaux.
Sans DP, σᵢ = 0 et aucune garantie à ε fini n'est revendiquée.

**Périmètre :** ces formules décrivent les anciens runs positioning affichés
ici, sans canal de loss privée supplémentaire. Elles ne doivent pas être
substituées à la composition des deux canaux de la nouvelle campagne V29.
''')
    insert_after('0ec66adcd7272fc7', [('requested-privacy-calibration-check', '''# Vérification scalaire du comptable historique, sans entraîner de modèle.
from privacy.rdp import RDPAccountant
privacy_calibration_rows = []
for row in privacy_screen.itertuples():
    cfg = json.loads(Path(row.metrics_path).read_text())["config"]
    b, N = int(cfg["fixed_batch_size"]), int(cfg["privacy_public_dataset_size"])
    releases = int(cfg["privacy_num_rounds"]) * int(cfg["fixed_steps_per_round"])
    assert cfg["sampling_scheme"] == "fixed_without_replacement" and cfg["privacy_adjacency"] == "replace_one"
    # L'écran D est homogène ; le ratio comptable est sigma / 2.
    assert np.isclose(row.sigma_min, row.sigma_max)
    acc = RDPAccountant()
    acc.add_sampled_without_replacement_gaussian(channel="gradient", sampling_rate=b/N,
        noise_multiplier=row.sigma_mean/2, steps=releases)
    eps, order = acc.epsilon(float(cfg["delta"]))
    assert abs(eps - row.epsilon) < 1e-8
    privacy_calibration_rows.append({
        "Clients": row.n_clients, "ε cible": row.target_epsilon, "L releases": releases,
        "b / N": f"{b}/{N}", "σ enregistré": row.sigma_mean,
        "σ / 2 comptable": row.sigma_mean/2, "Ordre a retenu": order,
        "ε recalculé": eps, "Écart ε enregistré": eps-row.epsilon,
    })
privacy_calibration_check = pd.DataFrame(privacy_calibration_rows)
display(Markdown("### Vérification numérique : σ enregistré → ε recomposé, sans nouveau training"))
display(privacy_calibration_check.sort_values(["Clients", "ε cible"]).round(6))
privacy_calibration_check.to_csv(EXPORT_ROOT / "privacy_sigma_to_epsilon_check.csv", index=False)
''')])
    edit('219710fd17b44ead', cells['219710fd17b44ead'].source + r'''

### Que signifie « oracle » ici ? Quelle erreur est tracée ?

Un **oracle de diagnostic** utilise une information de vérité terrain connue
du simulateur, mais inconnue de la défense en déploiement. Ici le simulateur
sait quels clients sont byzantins : il connaît donc l'ensemble des honnêtes
ℋₜ. Ce n'est ni une prédiction parfaite ni une composante utilisée pour choisir
la référence ou les poids. Les évaluations/oracles de recherche ne constituent
pas des messages publiables couverts par la garantie DP.

Soit Yᵢ,ₜ le message reçu, Xᵢ,ₜ = Clip_U(Yᵢ,ₜ), et P la transformation de
l'espace des scores (identité dans la phase F présentée ici). La valeur
enregistrée sous `far_reference_honest_center_error_oracle` est exactement :

$$
\bar X^{\mathcal H}_{\mathrm{score},t}
=\frac{1}{|\mathcal H_t|}\sum_{i\in\mathcal H_t} P(X_{i,t}),\qquad
E_{\mathrm{ref},t}
=\left\|F_t-\bar X^{\mathcal H}_{\mathrm{score},t}\right\|_2.
$$

**Attention : le centre honnête de cette métrique contient encore le bruit DP.**
On mesure une norme L2, **pas son carré**, et pas l'erreur par rapport au vrai
gradient de population. Son unité est celle du gradient ; ce n'est pas un
pourcentage d'accuracy. Une faible valeur signifie « référence proche de la
moyenne honnête des messages privés », pas « modèle plus juste » ni « bruit
DP retiré ». La métrique d'erreur de l'agrégat au centre honnête *propre* de la
section 8 est une autre quantité et n'a pas la même cible.

Le tableau et la figure d'erreur de référence de cette section prennent
**t = 20, puis la moyenne entre les trois seeds** ; ce n'est pas la médiane
temporelle utilisée pour les poids en section 5. La masse byzantine oracle
est Σᵢ∈ℬₜ λᵢ,ₜ : elle requiert elle aussi les identités malveillantes connues
du simulateur.
''')
    edit('de8df6638be93b94', '_plot_attack_reference("reference_error", "Erreur L2 au centre honnête bruité · t=20", False)')
    for c in nb.cells:
        if c.cell_type == 'code': ast.parse(c.source)
    nbformat.validate(nb)
    return changed


if __name__ == '__main__':
    original = PATH.read_text()
    nb = nbformat.reads(original, as_version=4)
    before = copy.deepcopy(nb)
    changed = transform(nb)
    for c in before.cells:
        if c.id not in changed:
            assert c == next(x for x in nb.cells if x.id == c.id), c.id
    new = nbformat.writes(nb)
    diff = list(difflib.unified_diff(original.splitlines(True), new.splitlines(True), n=3))
    sys.stdout.write('*** Begin Patch\n*** Update File: ' + str(PATH.relative_to(ROOT)) + '\n')
    for line in diff[2:]:
        if line.startswith('@@'): sys.stdout.write('@@\n')
        else: sys.stdout.write(line if line.endswith('\n') else line+'\n')
    sys.stdout.write('*** End Patch\n')
