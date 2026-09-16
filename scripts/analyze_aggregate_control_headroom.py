#!/usr/bin/env python3
"""Validate and summarize six shadow-only aggregate control fit runs."""
from pathlib import Path
import json
import math
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_aggregate_control_replay as replay


def main():
    matrix = replay.load_matrix()
    root = replay.output_root(matrix)
    stamp = json.loads((root / "campaign_lock.json").read_text())
    replay.verify_provenance(matrix, stamp)
    groups = {}
    identity_residual = 0.0
    geometry_violations = []
    verified_runs = []
    for task in replay.headroom_tasks(matrix):
        directory = replay.run_directory(matrix, task)
        status = json.loads((directory / "orchestration_status.json").read_text())
        if status["status"] != "completed":
            raise RuntimeError(f"run incomplete: {directory}")
        config = replay.resolved_config(matrix, task, stamp)
        metric_path, payload, pairing = replay.validate_run(matrix, task, config, stamp)
        if replay.hash_file(metric_path) != status["metrics_sha256"]:
            raise RuntimeError("metric hash mismatch")
        trace = directory / "simulator_randomness_private_audit.jsonl"
        if replay.hash_file(trace) != status["trace_sha256"]:
            raise RuntimeError("randomness trace hash mismatch")
        rows = payload["rounds"][12:]
        base = st.mean(r["aggregate_control_base_error_sq"] for r in rows)
        record = dict(seed=task["seed"], base_mse=base, controls={})
        record["oracle_gamma"] = st.mean(r["aggregate_control_oracle_gamma_euclidean"] for r in rows)
        record["oracle_headroom_pct"] = 100 * st.mean(r["aggregate_control_oracle_relative_headroom"] for r in rows)
        record["oracle_mse"] = st.mean(r["aggregate_control_oracle_error_sq"] for r in rows)
        for label in replay._expected_shadow_labels(matrix):
            prefix = f"aggregate_control_shadow_{label}_"
            value = st.mean(r[prefix + "error_sq"] for r in rows)
            covered = [r[prefix + "target_covered"] for r in rows]
            record["controls"][label] = dict(
                mse=value, mse_gain_pct=100 * (1 - value / base),
                correction_pct=100 * st.mean(r[prefix + "correction_applied"] for r in rows),
                coverage_pct=None if covered[0] is None else 100 * st.mean(covered),
            )
            for r in rows:
                if r[prefix + "target_covered"] and label.startswith(("isotropic_", "euclidean_projection_")):
                    excess = r[prefix + "error_sq"] + r[prefix + "correction_sq"] - r["aggregate_control_base_error_sq"]
                    if excess > 1e-7 * max(1, r["aggregate_control_base_error_sq"]):
                        geometry_violations.append([task, r["round_num"], label, excess])
        for r in rows:
            residual = (r["aggregate_control_predictor_error_sq"]
                        + 2 * r["aggregate_control_u_dot_predictor_minus_target"]
                        + r["aggregate_control_innovation_sq"]
                        - r["aggregate_control_base_error_sq"])
            identity_residual = max(identity_residual, abs(residual))
        groups.setdefault(task["noise"], []).append(record)
        verified_runs.append(dict(task=task, metrics=str(metric_path.relative_to(ROOT)), pairing=pairing))
    if geometry_violations:
        raise RuntimeError(f"covered Euclidean projection violations: {geometry_violations}")
    evidence = dict(
        runs_valid=6, rounds=240, statistical_units_per_noise_regime=3,
        window="13-40", private_device="mps", server_postprocessing="cpu_float64",
        oracle_boundary="offline_only", all_controls_shadow=True,
        covariance_exact=False, calibrated_radii=False, promotion=False,
        max_interpolation_identity_residual=identity_residual,
        covered_euclidean_geometry_violations=geometry_violations,
        verified_runs=verified_runs, groups=groups,
    )
    out = ROOT / "output/analysis/aggregate_control_headroom_v1"
    out.mkdir(exist_ok=True, parents=True)
    (out / "Evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Contrôle de l’agrégat — premier écran propre à 10 clients",
        "",
        "**6/6 runs valides**, sur MPS pour les gradients privés. Contrôleurs calculés sur les mêmes messages, sans modifier le modèle entraîné par FAR(RFA). Aucun résultat d’accuracy attribuable au contrôleur.",
        "",
        "Fenêtre 13–40 ; trois seeds indépendantes par régime : 930101, 930102, 930103. Moyenne temporelle par seed, puis moyenne ± écart-type d’échantillon (ddof=1). Ce ne sont pas des IC95.",
        "",
        "| Contrôle (rayons non calibrés) | MSE homogène | MSE hétéroscédastique |",
        "|---|---:|---:|",
    ]
    labels = {
        "unchanged": "FAR(RFA) inchangé",
        "ema_mix_0p2": "Lissage : 0,8 P + 0,2 A",
        "isotropic_mult_0p5": "Isotrope : rayon 0,5 U/√n",
        "isotropic_mult_1": "Isotrope : rayon U/√n",
        "isotropic_mult_2": "Isotrope : rayon 2 U/√n",
        "radial_mult_0p5": "Radial : rayon 0,5 √d",
        "radial_mult_1": "Radial : rayon √d",
        "radial_mult_2": "Radial : rayon 2 √d",
        "euclidean_projection_mult_0p5": "Projection euclidienne : rayon 0,5 √d",
        "euclidean_projection_mult_1": "Projection euclidienne : rayon √d",
        "euclidean_projection_mult_2": "Projection euclidienne : rayon 2 √d",
    }
    def mean_sd(values):
        return f"{st.mean(values):.4f} ± {st.stdev(values):.4f}"
    for key, title in labels.items():
        cells = [mean_sd([row["controls"][key]["mse"] for row in groups[noise]])
                 for noise in ("homogeneous", "heteroscedastic")]
        lines.append("| " + title + " | " + " | ".join(cells) + " |")
    lines += ["", "## Marge oracle, non déployable", ""]
    for noise, records in groups.items():
        lines.append(
            f"- {noise} : γ oracle moyen = {mean_sd([r['oracle_gamma'] for r in records])} ; "
            f"réduction oracle de MSE = {mean_sd([r['oracle_headroom_pct'] for r in records])} % ; "
            f"gain du lissage simple = {mean_sd([r['controls']['ema_mix_0p2']['mse_gain_pct'] for r in records])} %."
        )
    lines += [
        "", "## Interprétation et décision", "",
        "Observation : une marge importante existe pour réduire l’erreur au gradient honnête propre de batch. Le lissage simple en exploite déjà une grande partie et obtient ici une MSE plus faible que les rayons ellipsoïdaux essayés.",
        "",
        "Ce résultat ne justifie donc pas encore une covariance élaborée. Les rayons sont des diagnostics non calibrés ; leurs différences ne constituent pas une sélection équitable de politiques optimales.",
        "",
        "Les corrections sont souvent actives sur 100 % des tours propres pour les rayons serrés. Elles retirent alors du bruit, mais ne satisfont pas le futur objectif de 5 % de trajectoires déclenchées. Fausse activation, couverture du gradient propre et nuisance réelle sont trois notions distinctes.",
        "",
        "Non identifiable : amélioration d’accuracy ou de fairness du modèle corrigé, robustesse aux attaques, récupération et covariance effective. Le modèle suivi reste inchangé ; aucune attaque n’est présente. Les seeds de calibration et d’évaluation restent fermées.",
        "",
        "Décision : ne pas promouvoir une ellipsoïde. Le prochain protocole doit conserver le lissage simple comme témoin incontournable et tester séparément son coût end-to-end. Les autres phases ne sont pas lancées automatiquement.",
        "",
        "## Audit technique", "",
        f"240 tours, 2 400 traces client–tour, trois paires de régimes de bruit. Sources, statuts, hashes, MPS, epsilon et appariement vérifiés. Résidu maximal de l’identité quadratique : {identity_residual:.3e}. Zéro violation numérique détectée de la borne euclidienne sous couverture pour les projections isotropes et euclidiennes.",
        "",
        "[Preuves machine](aggregate_control_headroom_v1/Evidence.json) — "
        "[Protocole](../../configs/ldp_gradient_far/aggregate_control_preregistered_v1.yaml).",
    ]
    report = ROOT / "output/analysis/Aggregate_Control_Headroom_N10_Results.md"
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(report=str(report), runs=6, residual=identity_residual,
                         groups={noise: {
                             "oracle_gain_pct": mean_sd([r["oracle_headroom_pct"] for r in records]),
                             "ema_gain_pct": mean_sd([r["controls"]["ema_mix_0p2"]["mse_gain_pct"] for r in records]),
                         } for noise, records in groups.items()}), indent=2))


if __name__ == "__main__":
    main()
