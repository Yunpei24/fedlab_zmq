#!/usr/bin/env python3
"""Audit and summarize the homogeneous LDP-gradient-FAR T x B ablation.

The four primary cells are compared only at equal realised privacy.  The
fixed-sigma arm is deliberately printed in a separate diagnostic section.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT
    / "configs"
    / "ldp_gradient_far"
    / "fmnist_homogeneous_round_batch_ablation_v1.yaml"
)
DEFAULT_REPORT = (
    ROOT
    / "output"
    / "analysis"
    / "LDP_Gradient_FAR_Homogeneous_Round_Batch_Ablation_Results.md"
)

METRICS = {
    "test_accuracy_pct": ("test_accuracy", 100.0),
    "client_accuracy_pct": ("client_accuracy_mean", 100.0),
    "variance_pp2": ("client_accuracy_variance_pct2", 1.0),
    "worst20_pct": ("worst20_accuracy_pct", 1.0),
    "gap_pp": ("best20_worst20_gap_pct", 1.0),
    "test_loss": ("test_loss", 1.0),
    "epsilon": ("privacy_epsilon_max", 1.0),
    "sigma_impl": ("privacy_model_noise_multiplier_mean", 1.0),
    "local_clip_rate": ("privacy_clip_rate_mean", 1.0),
    "server_clip_rate": ("far_server_clip_rate", 1.0),
}


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    items = list(values)
    if not items:
        return math.nan, math.nan
    return statistics.fmean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_mean_sd(values: Iterable[float], digits: int = 2) -> str:
    mean, sd = _mean_sd(values)
    return f"{_fmt(mean, digits)} ± {_fmt(sd, digits)}"


def _last_at_or_before(rounds: list[dict[str, Any]], target: int) -> dict[str, Any]:
    eligible = [row for row in rounds if int(row.get("round_num", -1)) <= target]
    if not eligible:
        raise ValueError(f"no evaluated round at or before {target}")
    return max(eligible, key=lambda row: int(row["round_num"]))


def _find_metrics(task_dir: Path) -> Path:
    candidates = sorted(task_dir.glob("**/metrics.json"))
    valid: list[Path] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rounds = payload.get("rounds", [])
        expected = int(payload.get("summary", {}).get("num_rounds", -1))
        if len(rounds) == expected and expected > 0:
            valid.append(path)
    if len(valid) != 1:
        raise ValueError(
            f"expected exactly one complete metrics.json below {task_dir}, got {valid}"
        )
    return valid[0]


def _extract(
    path: Path,
    *,
    job: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload["rounds"]
    final = rounds[-1]
    # metrics.json stores the resolved algorithm configuration directly under
    # ``config`` (it is not the original experiment YAML hierarchy).
    algo = payload["config"]
    expected_rounds = int(job["factors"]["rounds"])
    expected_batch = int(job["factors"]["fixed_batch_size"])
    expected_sigma = float(job["factors"]["expected_noise_multiplier"])
    expected_q = expected_batch / 6000.0

    errors: list[str] = []
    checks = {
        "round count": len(rounds) == expected_rounds,
        "seed": int(payload["summary"]["seed"]) == seed,
        "partition seed": int(payload["summary"]["partition_seed"]) == seed,
        "fixed WOR": final.get("privacy_sampling_scheme")
        == "fixed_without_replacement",
        "replace-one": final.get("privacy_adjacency") == "replace_one",
        "sampling rate": math.isclose(
            float(algo["privacy_sampling_rate_override"]), expected_q, abs_tol=1e-12
        ),
        "fixed batch": int(algo["fixed_batch_size"]) == expected_batch,
        "privacy horizon": int(algo["privacy_num_rounds"]) == expected_rounds,
        "C=8": math.isclose(float(algo["clip_norm"]), 8.0),
        "U=16": math.isclose(float(algo["far_server_clip_norm"]), 16.0),
        "server lr=0.2": math.isclose(float(algo["far_server_lr"]), 0.2),
        "uniform alpha=0": math.isclose(float(algo["far_alpha"]), 0.0),
        "ten clients": int(payload["summary"]["num_clients"]) == 10,
    }
    if job["analysis_role"] == "primary_equal_privacy":
        checks["target epsilon=4"] = math.isclose(float(algo["target_epsilon"]), 4.0)
        checks["realised epsilon near 4"] = math.isclose(
            float(final["privacy_epsilon_max"]), 4.0, abs_tol=2e-4
        )
        checks["recalibrated sigma"] = math.isclose(
            float(final["privacy_model_noise_multiplier_mean"]),
            expected_sigma,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    else:
        checks["fixed-sigma target absent"] = algo.get("target_epsilon") is None
        checks["fixed diagnostic sigma"] = math.isclose(
            float(final["privacy_model_noise_multiplier_mean"]),
            expected_sigma,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    for label, passed in checks.items():
        if not passed:
            errors.append(label)
    if errors:
        raise ValueError(f"protocol failure in {path}: {', '.join(errors)}")

    record: dict[str, Any] = {
        "job_id": str(job["id"]),
        "role": str(job["analysis_role"]),
        "seed": seed,
        "rounds": expected_rounds,
        "batch": expected_batch,
        "path": str(path),
    }
    for output_name, (input_name, scale) in METRICS.items():
        value = final.get(input_name)
        record[output_name] = math.nan if value is None else float(value) * scale
    if expected_rounds >= 40:
        at40 = _last_at_or_before(rounds, 40)
        record["test_accuracy_round40_pct"] = 100.0 * float(at40["test_accuracy"])
        record["test_loss_round40"] = float(at40["test_loss"])
    return record


def _paired_delta(
    records: list[dict[str, Any]],
    left_id: str,
    right_id: str,
    metric: str,
) -> list[float]:
    lookup = {(row["job_id"], row["seed"]): row for row in records}
    seeds = sorted(row["seed"] for row in records if row["job_id"] == left_id)
    return [
        float(lookup[(right_id, seed)][metric]) - float(lookup[(left_id, seed)][metric])
        for seed in seeds
    ]


def _summary_table(records: list[dict[str, Any]], job_ids: list[str]) -> list[str]:
    lines = [
        "| Cellule | Test Acc. (%) | Client Acc. (%) | Var. (pp²) | Worst-20 (%) | Gap (pp) | Test loss | ε réalisé | σ impl. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for job_id in job_ids:
        rows = [row for row in records if row["job_id"] == job_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    job_id,
                    _fmt_mean_sd(row["test_accuracy_pct"] for row in rows),
                    _fmt_mean_sd(row["client_accuracy_pct"] for row in rows),
                    _fmt_mean_sd(row["variance_pp2"] for row in rows),
                    _fmt_mean_sd(row["worst20_pct"] for row in rows),
                    _fmt_mean_sd(row["gap_pp"] for row in rows),
                    _fmt_mean_sd((row["test_loss"] for row in rows), 4),
                    _fmt_mean_sd((row["epsilon"] for row in rows), 4),
                    _fmt_mean_sd((row["sigma_impl"] for row in rows), 4),
                ]
            )
            + " |"
        )
    return lines


def _delta_table(
    records: list[dict[str, Any]], comparisons: list[tuple[str, str, str]]
) -> list[str]:
    lines = [
        "| Comparaison appariée (droite − gauche) | Δ Test Acc. (pp) | Δ Worst-20 (pp) | Δ Gap (pp) | Signe Test Acc. par seed |",
        "|---|---:|---:|---:|---|",
    ]
    for label, left_id, right_id in comparisons:
        acc = _paired_delta(records, left_id, right_id, "test_accuracy_pct")
        worst = _paired_delta(records, left_id, right_id, "worst20_pct")
        gap = _paired_delta(records, left_id, right_id, "gap_pp")
        signs = ", ".join("+" if x > 0 else "−" if x < 0 else "0" for x in acc)
        lines.append(
            f"| {label} | {_fmt_mean_sd(acc)} | {_fmt_mean_sd(worst)} | "
            f"{_fmt_mean_sd(gap)} | {signs} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in matrix["seeds"]]
    jobs = matrix["jobs"]
    output_root = (matrix_path.parent / matrix["output_root"]).resolve()
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for job in jobs:
        for seed in seeds:
            task_dir = output_root / f"{job['id']}_seed{seed}"
            try:
                path = _find_metrics(task_dir)
                records.append(_extract(path, job=job, seed=seed))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing.append(f"{job['id']}/seed{seed}: {exc}")
    if missing:
        raise SystemExit(
            "campaign incomplete or invalid:\n  - " + "\n  - ".join(missing)
        )

    primary_ids = [
        str(job["id"])
        for job in jobs
        if job["analysis_role"] == "primary_equal_privacy"
    ]
    diagnostic_ids = [
        str(job["id"])
        for job in jobs
        if job["analysis_role"] == "diagnostic_unequal_privacy"
    ]
    comparisons = [
        ("B: 600 vs 300 à T=40", primary_ids[0], primary_ids[1]),
        ("B: 600 vs 300 à T=80", primary_ids[2], primary_ids[3]),
        ("T: 80 vs 40 à B=300", primary_ids[0], primary_ids[2]),
        ("T: 80 vs 40 à B=600", primary_ids[1], primary_ids[3]),
    ]

    lines = [
        "# Résultats — ablation homogène nombre de rounds × batch fixe",
        "",
        f"Matrice : `{matrix_path}`",
        "",
        "## Validation du protocole",
        "",
        (
            f"Les {len(primary_ids) * len(seeds)} runs primaires et les "
            f"{len(diagnostic_ids) * len(seeds)} runs diagnostiques sont complets. "
            "Le script a validé pour chaque run le sampler fixed-WOR, l'adjacence "
            "replace-one, le taux B/6000, l'horizon de l'accountant, C=8, U=16, "
            "alpha=0, n=10 et les seeds appariées."
        ),
        "",
        "## Comparaison principale à confidentialité constante",
        "",
        *_summary_table(records, primary_ids),
        "",
        "## Effets appariés",
        "",
        *_delta_table(records, comparisons),
        "",
        (
            "Un gain n'est considéré comme répliqué que si sa moyenne a le "
            "bon signe et si ce signe est cohérent sur les trois seeds. Avec trois "
            "seeds, ces statistiques restent descriptives et ne constituent pas un "
            "test de supériorité suffisamment puissant."
        ),
        "",
        "## Contrôle fixed-sigma — non comparable en confidentialité",
        "",
        *_summary_table(records, diagnostic_ids),
        "",
        (
            "Ce contrôle conserve le multiplicateur de bruit du cas T=40, B=300 "
            "mais compose 80 releases. Son epsilon réalisé doit donc être reporté "
            "explicitement et il ne doit jamais être comparé aux cellules epsilon=4 "
            "comme s'il s'agissait du même budget."
        ),
        "",
        "## Règle de conclusion",
        "",
        (
            "- Effet batch soutenu : Test Acc. et Worst-20 augmentent, ou le gap "
            "diminue, avec un signe apparié cohérent dans les deux horizons."
        ),
        (
            "- Effet horizon soutenu : le résultat final à T=80 améliore les "
            "métriques sur les trois seeds malgré la recalibration à epsilon=4."
        ),
        (
            "- Sinon, conclure 'non soutenu dans cette grille' ; ne pas conclure "
            "qu'une augmentation arbitraire de T ou B est universellement inutile."
        ),
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
