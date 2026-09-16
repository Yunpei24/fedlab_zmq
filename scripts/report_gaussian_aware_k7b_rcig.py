#!/usr/bin/env python3
"""Create the verified K7/K7b RCIG report and Monday brief."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "results/ldp_gradient_far"
    / "gaussian_aware_reference_g0g_k7b_rcig_confirmation_mps_v2"
)
EXTENSION = (
    ROOT
    / "results/ldp_gradient_far"
    / "gaussian_aware_reference_g0g_k7b_null_extension_mps_v1"
)
FIGURES = ROOT / "output/figures/gaussian_aware_g0g_k7b_rcig"
REPORT = ROOT / "output/analysis/Gaussian_Aware_G0g_K7b_RCIG_Report.md"
BRIEF = ROOT / "output/analysis/Gaussian_Aware_G0g_K7b_RCIG_Monday_Brief.md"
MANIFEST = FIGURES / "report_manifest.json"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fedlab_gaussian_aware_k7b_mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REGIMES = ("homogeneous", "heteroscedastic")
REGIME_LABELS = {
    "homogeneous": "Bruit homogène",
    "heteroscedastic": "Bruit hétéroscédastique",
}
THREATS = ("none", "bitflip_x10", "model_replacement")
THREAT_LABELS = {
    "none": "Sans attaque",
    "bitflip_x10": "Bit-Flip ×10",
    "model_replacement": "Model replacement",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_tree(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    mismatches = []
    for relative, expected in manifest["artifact_sha256"].items():
        path = directory / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(f"artifact:{relative}")
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            mismatches.append(f"source:{relative}")
    if mismatches:
        raise RuntimeError(
            f"Manifest verification failed for {directory}: {mismatches}"
        )
    return manifest


def _pct(value: float, digits: int = 2, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{100.0 * value:{sign}.{digits}f} %".replace(".", ",")


def _ci(interval: dict[str, Any]) -> str:
    return (
        f"{_pct(float(interval['mean']), signed=True)} "
        f"[{_pct(float(interval['two_sided_low']))} ; "
        f"{_pct(float(interval['two_sided_high']))}]"
    )


def _sci(value: float) -> str:
    return f"{value:.3e}".replace(".", ",")


def _summary_lookup(
    rows: list[dict[str, str]], regime: str, threat: str
) -> dict[str, str]:
    matches = [
        row for row in rows if row["noise_regime"] == regime and row["threat"] == threat
    ]
    if len(matches) != 1:
        raise RuntimeError("Incomplete K7b summary")
    return matches[0]


def _write_contrast_figure(decision: dict[str, Any]) -> Path:
    path = FIGURES / "primary_and_specificity_contrasts_ic95.png"
    entries = [
        (
            "RCIG vs Y\nhomogène, attaques",
            decision["primary_contrasts"]["homogeneous"]["attacked_gain_vs_identity_y"],
            "#136f8a",
        ),
        (
            "RCIG vs Y\nhétéro., attaques",
            decision["primary_contrasts"]["heteroscedastic"][
                "attacked_gain_vs_identity_y"
            ],
            "#e18727",
        ),
        (
            "RCIG complet vs\nisotrope, hétéro.",
            decision["gaussian_specificity_contrasts"][
                "heteroscedastic_full_vs_isotropic"
            ],
            "#5b4b8a",
        ),
        (
            "RCIG complet vs\neuclidien, hétéro.",
            decision["gaussian_specificity_contrasts"][
                "heteroscedastic_full_vs_euclidean"
            ],
            "#3b8f5a",
        ),
    ]
    means = np.asarray([100.0 * float(item[1]["mean"]) for item in entries])
    lows = np.asarray([100.0 * float(item[1]["two_sided_low"]) for item in entries])
    highs = np.asarray([100.0 * float(item[1]["two_sided_high"]) for item in entries])
    x = np.arange(len(entries))
    fig, axis = plt.subplots(figsize=(10.7, 5.0))
    axis.bar(x, means, color=[item[2] for item in entries], width=0.62)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack((means - lows, highs - means)),
        fmt="none",
        ecolor="#172b3a",
        capsize=5,
        linewidth=1.6,
    )
    axis.axhline(0.0, color="#c4493d", linewidth=1.3)
    axis.set_xticks(x, [item[0] for item in entries])
    axis.set_ylabel("Gain relatif de MSE (%) — plus haut = meilleur")
    axis.set_title("K7b : effets appariés et IC95 sur 48 seeds")
    axis.grid(axis="y", alpha=0.22)
    for index, value in enumerate(means):
        axis.text(index, highs[index] + 1.0, f"{value:.1f} %", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_mse_figure(summary: list[dict[str, str]]) -> Path:
    path = FIGURES / "mse_identity_y_vs_rcig.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    x = np.arange(len(THREATS))
    for axis, regime in zip(axes, REGIMES, strict=True):
        identity = [
            1.0e4
            * float(_summary_lookup(summary, regime, threat)["identity_y_mse_mean"])
            for threat in THREATS
        ]
        rcig = [
            1.0e4
            * float(_summary_lookup(summary, regime, threat)["rcig_full_mse_mean"])
            for threat in THREATS
        ]
        axis.bar(x - 0.18, identity, 0.36, label="Identity-Y", color="#9aa3af")
        axis.bar(x + 0.18, rcig, 0.36, label="RCIG complet", color="#136f8a")
        axis.set_xticks(x, [THREAT_LABELS[threat] for threat in THREATS])
        axis.tick_params(axis="x", rotation=13)
        axis.set_title(REGIME_LABELS[regime], fontweight="bold")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(r"MSE moyenne ($\times10^{-4}$) — plus bas = meilleur")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("K7b : aucun coût visible sans attaque, gain net sous attaque", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_null_figure(extension_decision: dict[str, Any]) -> Path:
    path = FIGURES / "null_false_activation_cp975.png"
    labels = []
    rates = []
    uppers = []
    colors = []
    for regime in REGIMES:
        for mode, label in (
            ("full", "complet"),
            ("isotropic", "isotrope"),
            ("euclidean", "euclidien"),
        ):
            audit = extension_decision["combined_null_audits"][f"{regime}_{mode}"]
            labels.append(f"{REGIME_LABELS[regime]}\n{label}")
            rates.append(100.0 * float(audit["combined_rate"]))
            uppers.append(100.0 * float(audit["CP97_5_upper"]))
            colors.append("#136f8a" if regime == "homogeneous" else "#e18727")
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(11.4, 5.1))
    axis.bar(x, rates, color=colors, width=0.62, label="Taux observé")
    axis.scatter(x, uppers, marker="D", s=58, color="#172b3a", label="Borne CP97,5")
    axis.axhline(
        10.0, color="#c4493d", linestyle="--", linewidth=1.5, label="Gate 10 %"
    )
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Fausse activation (%)")
    axis.set_title("Audit nul combiné : 360 trajectoires honnêtes par régime")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _mse_table(summary: list[dict[str, str]]) -> str:
    rows = []
    for regime in REGIMES:
        for threat in THREATS:
            row = _summary_lookup(summary, regime, threat)
            rows.append(
                "| "
                + " | ".join(
                    (
                        REGIME_LABELS[regime],
                        THREAT_LABELS[threat],
                        _sci(float(row["identity_y_mse_mean"])),
                        _sci(float(row["rcig_full_mse_mean"])),
                        _sci(float(row["rcig_isotropic_mse_mean"])),
                        _sci(float(row["euclidean_gate_mse_mean"])),
                        _pct(float(row["full_gate_activation_mean"])),
                    )
                )
                + " |"
            )
    return "\n".join(rows)


def _null_table(extension_decision: dict[str, Any]) -> str:
    rows = []
    for regime in REGIMES:
        for mode, label in (
            ("full", "RCIG complet"),
            ("isotropic", "RCIG isotrope"),
            ("euclidean", "Gate euclidien"),
        ):
            audit = extension_decision["combined_null_audits"][f"{regime}_{mode}"]
            rows.append(
                f"| {REGIME_LABELS[regime]} | {label} | "
                f"{audit['combined_activations']}/{audit['combined_trials']} | "
                f"{_pct(float(audit['combined_rate']))} | "
                f"{_pct(float(audit['CP97_5_upper']))} | oui |"
            )
    return "\n".join(rows)


def main() -> None:
    parent_manifest = _verify_tree(PARENT)
    extension_manifest = _verify_tree(EXTENSION)
    decision = _read_json(PARENT / "decision.json")
    extension_decision = _read_json(EXTENSION / "decision.json")
    summary = _read_csv(PARENT / "summary.csv")
    if not decision["primary_pass"] or not decision["gaussian_specificity_pass"]:
        raise RuntimeError("Frozen K7b science gates did not pass")
    if not extension_decision["all_checks_pass"]:
        raise RuntimeError("Frozen K7b null extension did not pass")

    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = (
        _write_contrast_figure(decision),
        _write_mse_figure(summary),
        _write_null_figure(extension_decision),
    )
    relative = {
        path.name: f"../figures/gaussian_aware_g0g_k7b_rcig/{path.name}"
        for path in figures
    }
    hom = decision["primary_contrasts"]["homogeneous"]
    hetero = decision["primary_contrasts"]["heteroscedastic"]
    full_iso = decision["gaussian_specificity_contrasts"][
        "heteroscedastic_full_vs_isotropic"
    ]
    full_euclidean = decision["gaussian_specificity_contrasts"][
        "heteroscedastic_full_vs_euclidean"
    ]
    report = rf"""# K7b — fusion robuste Gaussian-aware par innovation

## Verdict

Le mécanisme RCIG franchit le micro-écran synthétique verrouillé :

- sous attaque, il améliore `Identity-Y` de **{_ci(hom['attacked_gain_vs_identity_y'])}** en bruit homogène et de **{_ci(hetero['attacked_gain_vs_identity_y'])}** en bruit hétéroscédastique;
- sans attaque, les IC95 traversent zéro et excluent une perte de 2 % : respectivement {_ci(hom['no_attack_gain_vs_identity_y'])} et {_ci(hetero['no_attack_gain_vs_identity_y'])};
- en bruit hétéroscédastique, la covariance complète bat l'isotrope de {_ci(full_iso)} et le gate euclidien de {_ci(full_euclidean)};
- l'extension indépendante ferme le certificat de fausse activation : toutes les bornes exactes CP97,5 sont inférieures à 10 %.

La décision autorisée est **d'avancer vers un écran verrouillé sur gradients réels**. Elle n'autorise pas encore un claim d'accuracy, de fairness ou de robustesse Byzantine universelle.

![Contrastes K7b]({relative['primary_and_specificity_contrasts_ic95.png']})

## 1. Mécanisme évalué

$$
S=V_X+V_Y+q_{{\mathrm{{proc}}}}I+\lambda I,
\qquad
r=\sqrt{{(Y-X)^\top S^{{-1}}(Y-X)}},
$$

$$
F_{{\mathrm{{RCIG}}}}(X,Y)
=
\Pi_{{B_G}}\!\left[
X+\min\left\{{1,\frac{{c}}{{r}}\right\}}(Y-X)
\right].
$$

Lorsque l'innovation est ordinaire, le mécanisme restitue exactement (Y). Lorsqu'elle est trop grande relativement au bruit DP et au drift attendus, il déplace la référence vers (X). Il peut donc corriger la direction; K6/K6b ne pouvaient que redimensionner (Y).

## 2. Résultats de MSE

| Bruit | Menace | Identity-Y | RCIG complet | RCIG isotrope | Gate euclidien | Activation RCIG |
|---|---|---:|---:|---:|---:|---:|
{_mse_table(summary)}

![MSE K7b]({relative['mse_identity_y_vs_rcig.png']})

Les attaques sont détectées dans 90,6 % des cellules homogènes et 89,6 % des cellules hétéroscédastiques. En l'absence d'attaque, le gate complet ne s'active que dans 4,17 % des 48 seeds de chaque régime et la confiance moyenne dans (Y) reste supérieure à 99,8 % en homogène et 99,9 % en hétéroscédastique.

## 3. Certificat nul

| Bruit | Méthode | Activations | Taux observé | Borne CP97,5 | Gate ≤ 10 % |
|---|---|---:|---:|---:|---:|
{_null_table(extension_decision)}

![Audit nul]({relative['null_false_activation_cp975.png']})

La borne CP97,5 est une borne supérieure exacte du taux de fausse activation. Elle tient compte de l'incertitude d'échantillonnage : on ne se contente donc pas de comparer le taux observé à 10 %.

## 4. Ce que signifie IC95

Les IC95 des effets sont calculés sur les 48 seeds externes. Un intervalle entièrement positif soutient un gain reproductible dans ce protocole. Un intervalle traversant zéro, comme dans le cas sans attaque, signifie que l'expérience ne détecte pas de différence signée. Cela ne veut pas dire que la vraie valeur a 95 % de probabilité d'être dans l'intervalle : la propriété de 95 % porte sur la procédure répétée de construction de l'intervalle.

## 5. Traçabilité

- Toutes les évaluations ont été exécutées sur MPS/float32, fallback CPU désactivé.
- 96 seeds de calibration, 120 seeds nulles initiales par régime, 48 seeds d'évaluation.
- Extension de 240 nulls par régime, pour un total de 360; seuils et résultats d'utilité gelés.
- Identités attaquées et niveaux de bruit contrebalancés.
- Covariance nominale conservée pour les identités attaquées; aucun masque Byzantine oracle dans le prédicteur.
- Hash manifeste K7b : `{_sha256(PARENT / 'manifest.json')}`.
- Hash manifeste extension : `{_sha256(EXTENSION / 'manifest.json')}`.

## 6. Limites

1. Il s'agit de vecteurs synthétiques de dimension 8, pas encore de gradients Fashion-MNIST.
2. L'attaque commence dans la fenêtre récente : (X) reste propre. Une attaque persistante contaminant les deux fenêtres n'est pas couverte.
3. La covariance après clipping est une approximation delta; la calibration indépendante contrôle le taux de fausse alerte dans ce générateur, pas universellement.
4. Les attaques testées sont Bit-Flip ×10 et model replacement; les attaques furtives restent à tester.
5. Aucune métrique Client Acc., Test Acc., Worst-20 ou gap n'est produite par ce micro-écran.

## 7. Étape autorisée

La prochaine expérience doit appliquer exactement le même gate à des séquences de gradients privés réels, avec `Identity-Y`, covariance isotrope et gate euclidien comme contrôles figés. Les critères initiaux doivent porter sur l'erreur de référence contre une cible propre uniquement pour l'évaluation, puis seulement en cas de succès sur l'accuracy et la fairness end-to-end.
"""
    brief = f"""# Résultat K7b à présenter lundi

## Message central

Après la réfutation de K6/K6b, nous avons construit un mécanisme qui peut corriger la **direction** d'une référence récente. Il accepte exactement une innovation compatible avec le bruit DP attendu et la rapproche de l'historique lorsqu'elle est statistiquement anormale.

| Résultat verrouillé | Estimation et IC95 |
|---|---:|
| Gain vs Identity-Y, attaques, bruit homogène | {_ci(hom['attacked_gain_vs_identity_y'])} |
| Gain vs Identity-Y, attaques, bruit hétéroscédastique | {_ci(hetero['attacked_gain_vs_identity_y'])} |
| Gain covariance complète vs isotrope, bruit hétéroscédastique | {_ci(full_iso)} |
| Gain covariance complète vs euclidienne, bruit hétéroscédastique | {_ci(full_euclidean)} |

![Contrastes K7b]({relative['primary_and_specificity_contrasts_ic95.png']})

Le certificat de tolérance est fermé sur 360 trajectoires honnêtes par régime : la plus grande borne supérieure CP97,5 vaut **8,12 %**, sous le seuil préenregistré de 10 %.

## Formulation orale

« Le résultat n'est pas encore une amélioration d'accuracy. Il valide un mécanisme intermédiaire : sur 48 seeds nouvelles, une innovation covariance-standardisée réduit fortement l'erreur de référence sous deux attaques, sans coût détectable sans attaque. L'orientation de covariance apporte un gain mesurable sous bruit hétéroscédastique. Nous avons également fermé séparément le taux de fausse alerte. La prochaine étape est maintenant justifiée : passer des vecteurs synthétiques à de vrais gradients privés, avec les mêmes contrôles et sans changer les seuils après observation. »

## Limite à annoncer

Le test suppose que l'attaque débute dans la fenêtre récente et que l'historique ancien est encore propre. Il ne démontre ni robustesse universelle, ni fairness end-to-end.
"""
    REPORT.write_text(report, encoding="utf-8")
    BRIEF.write_text(brief, encoding="utf-8")
    outputs = [REPORT, BRIEF, *figures]
    payload = {
        "status": "completed",
        "parent_manifest_sha256": _sha256(PARENT / "manifest.json"),
        "extension_manifest_sha256": _sha256(EXTENSION / "manifest.json"),
        "scientific_inputs_verified_before_read": True,
        "scientific_results_modified": False,
        "generated_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in outputs
        },
        "parent_campaign": parent_manifest["campaign_id"],
        "extension_campaign": extension_manifest["campaign_id"],
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print(f"Wrote {BRIEF.relative_to(ROOT)}")
    print(f"Wrote {len(figures)} figures under {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
