"""Pure, deterministic preregistered gates; no launch and no training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

FIELDS = {
    "accuracy": ("test_accuracy", 100),
    "balanced_accuracy": ("mean_client_balanced_accuracy_pct", 1),
    "worst20": ("worst20_accuracy_pct", 1),
    "gap": ("best20_worst20_gap_pct", 1),
    "loss": ("test_loss", 1),
    "variance": ("client_accuracy_variance_pct2", 1),
}


def final_metrics(run):
    paths = list(Path(run).glob("**/metrics.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one metrics file: {run}")
    payload = json.loads(paths[0].read_text())
    if len(payload["rounds"]) != 40 or payload["rounds"][-1]["round_num"] != 40:
        raise ValueError("A gate never uses an incomplete or best-selected round")
    row = payload["rounds"][-1]
    result = {key: float(row[field]) * scale for key, (field, scale) in FIELDS.items()}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Gate metrics must be finite")
    return result


def statistics_for(values):
    n = len(values)
    if n not in (3, 4) or not all(math.isfinite(x) for x in values):
        raise ValueError("Exactly three or four finite seed differences required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = {3: 4.302652729749462, 4: 3.182446305284263}[n] * sd / math.sqrt(n)
    return {
        "values": values,
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "positive_seeds": sum(x > 0 for x in values),
    }


def check_rules(groups, settings, *, confirmation):
    """Returns every check, not only successes; comparisons are DMD minus CE."""
    checks = []
    modes = settings["required_modes"] if confirmation else [settings["primary_mode"]]

    def add(name, value, threshold, *, greater=False, strict=False):
        passed = (
            value > threshold
            if greater and strict
            else value >= threshold if greater else value <= threshold
        )
        checks.append(
            {
                "criterion": name,
                "value": value,
                "threshold": threshold,
                "relation": ">" if greater and strict else ">=" if greater else "<=",
                "passed": passed,
            }
        )

    for mode in modes:
        for noise in ("homogeneous", "heteroscedastic"):
            stats = groups[f"{noise}/{mode}"]
            prefix = f"{noise}/{mode}"
            endpoint = "ci95_low" if confirmation else "mean"
            for key in ("accuracy", "balanced_accuracy"):
                add(
                    f"{prefix}/{key}/{endpoint}",
                    stats[key][endpoint],
                    -settings[f"{key}_noninferiority_pp"],
                    greater=True,
                    strict=confirmation,
                )
            add(f"{prefix}/gap", stats["gap"]["mean"], settings["gap_max_increase_pp"])
            add(f"{prefix}/loss", stats["loss"]["mean"], settings["loss_max_increase"])
            if noise == "homogeneous":
                add(
                    f"{prefix}/worst20/{endpoint}",
                    stats["worst20"][endpoint],
                    -settings["homogeneous_worst20_noninferiority_pp"],
                    greater=True,
                    strict=confirmation,
                )
            else:
                add(
                    f"{prefix}/worst20/mean",
                    stats["worst20"]["mean"],
                    settings["heteroscedastic_worst20_min_gain_pp"],
                    greater=True,
                )
                add(
                    f"{prefix}/worst20/positive_seeds",
                    stats["worst20"]["positive_seeds"],
                    settings["heteroscedastic_positive_worst20_seeds"],
                    greater=True,
                )
                if confirmation:
                    add(
                        f"{prefix}/worst20/ci95_low",
                        stats["worst20"]["ci95_low"],
                        0,
                        greater=True,
                        strict=True,
                    )
                else:
                    add(
                        f"{prefix}/balanced_accuracy/mean",
                        stats["balanced_accuracy"]["mean"],
                        settings["heteroscedastic_balanced_accuracy_min_gain_pp"],
                        greater=True,
                    )
    return checks


def evaluate_gate(phase, root, matrix, workspace):
    if phase not in ("ce_full_budget_screen", "confirmation"):
        raise ValueError("No promotion gate exists after attacks")
    cfg = matrix["phases"][phase]
    groups = {}
    levels = {}
    old_modes = {"uniform": "uniform", "direct_rfa": "rfa", "far_rfa": "far_rfa"}
    for noise in matrix["noise_regimes"]:
        for mode in cfg["modes"]:
            differences = {key: [] for key in FIELDS}
            values = {obj: {key: [] for key in FIELDS} for obj in ("ce", "dmd")}
            for seed in cfg["seeds"]:
                ce = final_metrics(
                    root / phase / "runs" / f"{noise}__seed{seed}__ce__{mode}__none"
                )
                if phase == "ce_full_budget_screen":
                    run = (
                        workspace
                        / matrix["pilot_root"]
                        / "runs"
                        / f"{noise}__seed{seed}__dmd_{old_modes[mode]}"
                    )
                else:
                    run = (
                        root
                        / phase
                        / "runs"
                        / f"{noise}__seed{seed}__dmd__{mode}__none"
                    )
                dmd = final_metrics(run)
                for key in FIELDS:
                    differences[key].append(dmd[key] - ce[key])
                    values["ce"][key].append(ce[key])
                    values["dmd"][key].append(dmd[key])
            groups[f"{noise}/{mode}"] = {
                k: statistics_for(v) for k, v in differences.items()
            }
            levels[f"{noise}/{mode}"] = {
                obj: {k: statistics_for(v) for k, v in metrics.items()}
                for obj, metrics in values.items()
            }
    confirmation = phase == "confirmation"
    checks = check_rules(
        groups,
        matrix["gates"]["confirmation" if confirmation else "screen"],
        confirmation=confirmation,
    )
    return {
        "phase": phase,
        "seed_order": cfg["seeds"],
        "contrast": "DMD_minus_CE_full_gradient_budget",
        "statistics": groups,
        "levels": levels,
        "checks": checks,
        "decision": "promote" if all(x["passed"] for x in checks) else "stop",
        "failed_criteria": [x for x in checks if not x["passed"]],
        "interpretation": (
            "fixed_practical_screen"
            if not confirmation
            else "new_seed_confirmation_before_attacks"
        ),
    }


def render_gate_report(evidence):
    labels = {
        "accuracy": "Accuracy (%)",
        "balanced_accuracy": "BA cliente (%)",
        "worst20": "Worst20 (%)",
        "gap": "Gap Best20−Worst20 (pp)",
        "variance": "Variance (pp²)",
        "loss": "Test loss",
    }
    lines = [
        "# DMD-CB contre CE au budget complet",
        "",
        f"Phase : {evidence['phase']}. Seeds : {evidence['seed_order']}.",
        "",
        "Comparaisons au round 40 fixé. Δ = DMD − CE ; les IC95 portent sur les différences par seed.",
        "Le pilote réutilise les DMD historiques ; la confirmation emploie de nouvelles seeds. Aucun choix du meilleur bras après observation.",
        "",
        f"Décision automatique selon les critères gelés : **{evidence['decision']}**.",
        "",
    ]
    for group, stats in evidence["statistics"].items():
        lines += [
            "## " + group,
            "",
            "| Métrique | CE moyenne ± SD | DMD moyenne ± SD | Δ moyenne ± SD | IC95 de Δ | Δ par seed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, s in stats.items():
            ce, dmd = (
                evidence["levels"][group]["ce"][key],
                evidence["levels"][group]["dmd"][key],
            )
            lines.append(
                f"| {labels[key]} | {ce['mean']:.4f} ± {ce['sd']:.4f} | {dmd['mean']:.4f} ± {dmd['sd']:.4f} | {s['mean']:.4f} ± {s['sd']:.4f} | [{s['ci95_low']:.4f} ; {s['ci95_high']:.4f}] | "
                + " ; ".join(f"{v:.4f}" for v in s["values"])
                + " |"
            )
        lines.append("")
    lines += [
        "## Critères",
        "",
        "| Critère | Valeur | Condition | Satisfait |",
        "| --- | --- | --- | --- |",
    ]
    for c in evidence["checks"]:
        lines.append(
            f"| {c['criterion']} | {c['value']:.6f} | {c['relation']} {c['threshold']} | {'oui' if c['passed'] else 'non'} |"
        )
    lines += [
        "",
        "Un gate non franchi interdit la phase suivante dans cette chaîne. Il ne prouve pas une impossibilité générale.",
        "La publication conjointe de modèles issus de simulations à bruit partagé ne bénéficie pas automatiquement du budget par run. Aucun de ces écrans propres ne prouve une robustesse byzantine.",
    ]
    return "\n".join(lines) + "\n"
