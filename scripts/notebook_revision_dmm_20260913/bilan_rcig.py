display(Markdown("### Résultat 2 — RCIG actuel n’est pas validé comme solution finale"))
paired_rcig = rcig_bilan[rcig_bilan.arm == "rolling"].merge(
    rcig_bilan[rcig_bilan.arm == "recent"],
    on=["noise", "seed", "scenario"], suffixes=("_c", "_r"), validate="one_to_one")
paired_rcig["delta_acc"] = paired_rcig.acc_c - paired_rcig.acc_r
equal = int(np.isclose(paired_rcig.delta_acc, 0, atol=1e-10, rtol=0).sum())
display(Markdown(
    f"RCIG sans gel et vue récente : **{equal}/{len(paired_rcig)} accuracies finales "
    f"identiques à la précision enregistrée**. Les autres deltas sont compris entre "
    f"{paired_rcig.delta_acc.min():+.2f} et {paired_rcig.delta_acc.max():+.2f} point. "
    "Cela ne signifie pas que les modèles sont identiques."
))
bf = rcig_bilan[(rcig_bilan.scenario == "bf_persistent") & rcig_bilan.arm.isin(["recent", "rolling", "freeze", "midpoint", "rfa"])]
arm_names = {"recent": "Récente", "rolling": "RCIG sans gel", "freeze": "RCIG avec gel", "midpoint": "Moyenne temporelle", "rfa": "FAR + RFA"}
bf_table = bf.groupby(["noise", "arm"]).agg(
    masse=("byzantine_mass", "mean"), accuracy=("acc", "mean"), worst20=("worst20", "mean"), gap=("gap", "mean")).reset_index()
bf_table["Bruit"] = bf_table.noise.map({"homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique"})
bf_table["Référence FAR"] = bf_table.arm.map(arm_names)
display(Markdown("**BF ×10 persistant :** moyennes sur trois seeds. La masse byzantine est moyennée sur les tours attaqués 17–40 ; les métriques du modèle sont celles du tour 40. La masse uniforme de deux clients sur dix serait 0,20."))
display(bf_table[["Bruit", "Référence FAR", "masse", "accuracy", "worst20", "gap"]].rename(columns={
    "masse": "Masse des 2 Byzantins", "accuracy": "Accuracy (%)", "worst20": "Worst-20 (%)", "gap": "Gap (pp)"}).round(4))
ipm_counts = []
for _, r in rcig_bilan[(rcig_bilan.arm == "recent") & (rcig_bilan.scenario == "ipm_persistent")].iterrows():
    window = record_map[(r.noise, r.seed, r.scenario, r.arm)]["windows"]["attacked_all"]
    assert window["n_rounds"] == 24
    count = window["alarm_fraction"] * window["n_rounds"]
    assert abs(count - round(count)) < 1e-9
    ipm_counts.append({"Bruit": r.noise, "seed": r.seed, "Tours en alerte / 24": int(round(count))})
ipm_table = pd.DataFrame(ipm_counts)
ipm_table["Bruit"] = ipm_table.Bruit.map({"homogeneous": "Homogène", "heteroscedastic": "Hétéroscédastique"})
display(ipm_table)
display(Markdown(
    "**Interprétation :** sous BF, toutes ces références laissent plus de 0,20 de "
    "masse aux deux attaquants. Sous IPM, le test ne déclenche pas. Il ne faut donc "
    "pas confondre référence robuste, détection d’une rupture et agrégat robuste. "
    "Le manque de gain concerne cette instanciation ; il ne prouve pas que toute "
    "méthode sous confidentialité locale est impossible."
))
bf_table.to_csv(RCIG_BILAN_ROOT / "BF_reference_weights_and_model.csv", index=False)
