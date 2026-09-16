#!/usr/bin/env python3
"""Read-only, post-hoc RCIG mechanism audit. No training, inference or tuning.

Original metrics/config/status files are hashed before and after analysis.
Only new JSON/Markdown/PNG audit artifacts are written under output/.
No algorithm module or torch is imported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/ldp_gradient_far/rcig_end_to_end_v2"
EXPLORATORY = ROOT / "results/ldp_gradient_far/rcig_r2_r3_exploratory_v1"
OUTPUT = ROOT / "output/analysis/rcig_mechanism_posthoc"
SUFFIX = "_squared_l2_error_to_clean_honest_center_oracle"
PHASES = [
    ("R1", PARENT / "r1_dynamic_null", 72),
    ("R2", EXPLORATORY / "r2_attack_mechanism", 144),
    ("R3", EXPLORATORY / "r3_e2e_confirmation", 384),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, sources):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def mean(values):
    values = [float(v) for v in values if finite(v)]
    return st.mean(values) if values else None


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": st.mean(values),
        "sd": st.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "median": st.median(values),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def auc(positive, negative):
    """Descriptive AUC; larger statistic means attack, no fitted threshold."""
    if not positive or not negative:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in positive for b in negative) / (
        len(positive) * len(negative)
    )


def exposure(first_internal, last_internal, start, end):
    if start is None:
        return 0.0
    first, last = int(first_internal) + 1, int(last_internal) + 1
    return max(0, min(last, end) - max(first, start) + 1) / (last - first + 1)


def covariance_parts(norm, isotropic_stat, process, ridge):
    if norm <= 0 or isotropic_stat <= 0:
        return None
    total = (norm / isotropic_stat) ** 2
    floor = process + ridge
    if total < floor - 1e-12:
        raise ValueError("Reconstructed total variance below known floor")
    return {
        "total_isotropic_variance": total,
        "dp_proxy_average_variance": max(0.0, total - floor),
        "floor_fraction": floor / total,
    }


def audit_inputs():
    sources, records, validation = {}, [], {}
    for phase, directory, expected in PHASES:
        paths = sorted(directory.glob("*/*/metrics.json"))
        if len(paths) != expected:
            raise ValueError(f"{phase}: {len(paths)} metrics, expected {expected}")
        detected = 0
        for path in paths:
            run = path.parent.parent
            payload = load(path, sources)
            status_path = run / "orchestration_status.json"
            status = load(status_path, sources)
            config_path = run / "resolved_config.yaml"
            sources[str(config_path.relative_to(ROOT))] = sha(config_path)
            assert status["status"] == "completed", run
            assert status["device"] == "mps" and status["mps_fallback"] == 0, run
            assert status["metrics_sha256"] == sources[str(path.relative_to(ROOT))], run
            assert status["resolved_config_sha256"] == sha(config_path), run
            rows, cfg = payload["rounds"], payload["config"]
            assert len(rows) == 40 and [r["round_num"] for r in rows] == list(
                range(1, 41)
            ), run
            assert cfg["device"] == "mps", run
            assert all(
                finite(r["test_accuracy"]) and finite(r["test_loss"]) for r in rows
            ), run
            assert all(
                r["far_attack_labels_visible_to_server_aggregate"] is False
                and r["far_attack_config_visible_to_server_aggregate"] is False
                for r in rows
            ), run
            if not str(cfg["robust_reference"]).startswith("rcig"):
                continue
            detected += 1
            assert cfg["rcig_persistent_policy"] == "freeze_hysteresis", run
            assert cfg["rcig_public_subspace_dimension"] == 64, run
            assert (
                cfg["rcig_gate_window"]
                == cfg["rcig_old_window"]
                == cfg["rcig_new_window"]
                == 4
            ), run
            attack = cfg.get("attack", {})
            scenario = (
                attack.get("name", "none") if attack.get("enabled", False) else "none"
            )
            start = int(attack["active_round_start"]) if scenario != "none" else None
            end = int(attack["active_round_end"]) if start is not None else None
            noise = (
                "heteroscedastic"
                if run.name.startswith("heteroscedastic")
                else "homogeneous"
            )
            seed = int(payload["summary"]["seed"])
            mode_thresholds = {
                m: float(cfg[k])
                for m, k in (
                    ("full", "rcig_innovation_threshold"),
                    ("isotropic", "rcig_isotropic_innovation_threshold"),
                    ("euclidean", "rcig_euclidean_innovation_threshold"),
                )
            }
            data = []
            for idx, row in enumerate(rows):
                t = int(row["round_num"])
                if not row.get("rcig_history_ready", False):
                    assert t <= 12
                    continue
                assert row["round"] == t - 1
                assert row["rcig_reference_strictly_past"] is True
                assert row["rcig_newer_round_max"] < t - 1
                assert row["rcig_older_round_max"] < row["rcig_newer_round_min"]
                assert row["rcig_private_gradient_mps_fraction"] == 1.0
                point = {
                    "round": t,
                    "attack_current": bool(start is not None and start <= t <= end),
                }
                assert point["attack_current"] == bool(row["attack_window_active"])
                for window in ("gate_source", "older", "newer"):
                    point[window + "_exposed_fraction"] = exposure(
                        row[f"rcig_{window}_round_min"],
                        row[f"rcig_{window}_round_max"],
                        start,
                        end,
                    )
                for mode, threshold in mode_thresholds.items():
                    val = float(row[f"rcig_{mode}_innovation_stat"])
                    assert finite(val) and threshold > 0
                    active = row[f"rcig_{mode}_gate_active"]
                    trust = row[f"rcig_{mode}_newer_trust"]
                    assert active == (val > threshold)
                    assert math.isclose(
                        trust, min(1.0, threshold / max(val, 1e-12)), abs_tol=1e-12
                    )
                    point[mode + "_stat"] = val
                    point[mode + "_ratio"] = val / threshold
                    point[mode + "_active"] = active
                    point[mode + "_trust"] = trust
                point.update(
                    covariance_parts(
                        point["euclidean_stat"],
                        point["isotropic_stat"],
                        cfg["rcig_process_variance"],
                        cfg["rcig_covariance_ridge"],
                    )
                    or {}
                )
                mapping = {
                    "active": "rcig_gate_active",
                    "frozen_before": "rcig_persistent_frozen_before_round",
                    "frozen_after": "rcig_persistent_frozen_after_commit",
                    "action": "rcig_persistent_action",
                    "clip_rate": "far_server_clip_rate",
                    "gate_mass": "rcig_gate_mass",
                    "reference_norm": "rcig_reference_norm",
                    "anisotropy_reported": "rcig_max_covariance_anisotropy_ratio",
                    "byzantine_mass": "byzantine_weight_mass_oracle",
                    "concentration": "far_noise_amplification_vs_uniform",
                    "alpha": "far_alpha",
                    "logit_range": "far_logit_range",
                    "test_loss": "test_loss",
                }
                for new, old in mapping.items():
                    point[new] = row.get(old)
                point["test_accuracy_pct"] = row["test_accuracy"] * 100
                point["test_loss_previous"] = (
                    rows[idx - 1]["test_loss"] if idx else None
                )
                point["test_loss_next"] = (
                    rows[idx + 1]["test_loss"] if idx + 1 < len(rows) else None
                )
                point["full_iso_stat_absolute_difference"] = abs(
                    point["full_stat"] - point["isotropic_stat"]
                )
                point["noise_norm_missing"] = (
                    row.get("privacy_realised_noise_norm_mean_oracle") is None
                )
                for candidate in (
                    "identity_new",
                    "identity_old",
                    "midpoint",
                    "full",
                    "isotropic",
                    "euclidean",
                    "reference",
                ):
                    key = "rcig_" + candidate + SUFFIX
                    if key in row:
                        assert finite(row[key]) and row[key] >= 0
                        point["error_" + candidate] = row[key]
                if phase in {"R1", "R2"}:
                    assert "error_reference" in point and "error_identity_new" in point
                if phase == "R3":
                    assert "error_reference" not in point
                data.append(point)
            assert len(data) == 28 and data[0]["round"] == 13
            records.append(
                {
                    "phase": phase,
                    "noise": noise,
                    "scenario": scenario,
                    "timing": (
                        "clean"
                        if start is None
                        else ("from_round1" if start == 1 else "after_clean")
                    ),
                    "attack_start": start,
                    "attack_end": end,
                    "seed": seed,
                    "thresholds": mode_thresholds,
                    "run": str(run.relative_to(ROOT)),
                    "rows": data,
                    "prefix_metrics": [
                        {
                            "round": r["round_num"],
                            "test_accuracy_pct": 100 * r["test_accuracy"],
                            "test_loss": r["test_loss"],
                        }
                        for r in rows[:16]
                    ],
                }
            )
        validation[phase] = {
            "all_runs_validated": expected,
            "rcig_runs_analyzed": detected,
        }
    assert [validation[p]["rcig_runs_analyzed"] for p in ("R1", "R2", "R3")] == [
        72,
        144,
        96,
    ]
    for path in [
        PARENT / "_gates/r0_dynamic_calibration.json",
        PARENT / "_gates/r1_dynamic_null.json",
        EXPLORATORY / "_diagnostics/r2_attack_mechanism.json",
        EXPLORATORY / "_diagnostics/r3_e2e_confirmation.json",
    ]:
        load(path, sources)
    for rel in [
        "algorithms/rcig_temporal_reference.py",
        "algorithms/gaussian_aware_reference_k7_rcig.py",
        "algorithms/ldp_gradient_far.py",
        "algorithms/far.py",
        "metrics/rcig_evaluation.py",
        "attacks/byzantine.py",
        "run_experiment.py",
        "scripts/analyze_rcig_ldp_gradient_far_v2.py",
        "scripts/run_rcig_ldp_gradient_far_v2.py",
        "privacy/local_dpsgd.py",
        "datasets/registry.py",
        "metrics/client_fairness.py",
    ]:
        sources[rel] = sha(ROOT / rel)
    return records, sources, validation


def summarize(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[(rec["phase"], rec["noise"], rec["scenario"], rec["timing"])].append(
            rec
        )
    summaries = []
    errors = []
    events = []
    for key, runs in sorted(grouped.items()):
        for window in ("ready", "current_attack", "newer_only_exposed", "both_exposed"):

            def select(p):
                return (
                    window == "ready"
                    or (window == "current_attack" and p["attack_current"])
                    or (
                        window == "newer_only_exposed"
                        and p["newer_exposed_fraction"] > 0
                        and p["older_exposed_fraction"] == 0
                    )
                    or (
                        window == "both_exposed"
                        and p["newer_exposed_fraction"]
                        == p["older_exposed_fraction"]
                        == 1
                    )
                )

            per_seed = [(rec, [p for p in rec["rows"] if select(p)]) for rec in runs]
            all_points = [p for _, pts in per_seed for p in pts]
            if not all_points:
                continue
            item = dict(zip(("phase", "noise", "scenario", "timing"), key))
            item.update(
                window=window,
                seeds=len(runs),
                rounds=len(all_points),
                active_rounds=sum(p["active"] for p in all_points),
                frozen_rounds=sum(p["frozen_after"] for p in all_points),
                seeds_with_activation=sum(
                    any(p["active"] for p in pts) for _, pts in per_seed
                ),
            )
            for field in (
                "full_ratio",
                "floor_fraction",
                "dp_proxy_average_variance",
                "clip_rate",
                "byzantine_mass",
                "concentration",
                "logit_range",
                "anisotropy_reported",
                "full_iso_stat_absolute_difference",
                "gate_mass",
            ):
                item[field + "_pooled_descriptive"] = stats(
                    p.get(field) for p in all_points
                )
                item[field + "_per_seed_mean"] = stats(
                    mean(p.get(field) for p in pts) for _, pts in per_seed
                )
            item["max_ratio_per_seed"] = stats(
                max(p["full_ratio"] for p in pts) for _, pts in per_seed if pts
            )
            item["actions"] = dict(Counter(p["action"] for p in all_points))
            summaries.append(item)
        for selection in (
            "all_ready",
            "attack_ready",
            "rolling_active",
            "deployed_frozen",
            "newer_only_exposed",
            "both_exposed",
        ):

            def take(p):
                return (
                    selection == "all_ready"
                    or (selection == "attack_ready" and p["attack_current"])
                    or (selection == "rolling_active" and p["active"])
                    or (selection == "deployed_frozen" and p["frozen_after"])
                    or (
                        selection == "newer_only_exposed"
                        and p["newer_exposed_fraction"] > 0
                        and p["older_exposed_fraction"] == 0
                    )
                    or (
                        selection == "both_exposed"
                        and p["newer_exposed_fraction"]
                        == p["older_exposed_fraction"]
                        == 1
                    )
                )

            per_seed = [
                (rec, [p for p in rec["rows"] if take(p) and "error_identity_new" in p])
                for rec in runs
            ]
            points = [p for _, pts in per_seed for p in pts]
            if not points:
                continue
            item = dict(zip(("phase", "noise", "scenario", "timing"), key))
            item.update(
                selection=selection,
                event_rows=len(points),
                contributing_seeds=sum(bool(pts) for _, pts in per_seed),
            )
            for candidate in (
                "identity_new",
                "identity_old",
                "midpoint",
                "full",
                "reference",
            ):
                error_key = "error_" + candidate
                per_seed_effects = []
                for rec, pts in per_seed:
                    if not pts:
                        continue
                    baseline = mean(p["error_identity_new"] for p in pts)
                    candidate_mean = mean(p[error_key] for p in pts)
                    per_seed_effects.append(
                        {
                            "seed": rec["seed"],
                            "n_rounds": len(pts),
                            "baseline": baseline,
                            "candidate": candidate_mean,
                            "difference": candidate_mean - baseline,
                            "relative_gain_pct": 100
                            * (baseline - candidate_mean)
                            / baseline,
                        }
                    )
                item[candidate] = {
                    "event_weighted_mean_error": mean(p[error_key] for p in points),
                    "event_weighted_mean_difference": mean(
                        p[error_key] - p["error_identity_new"] for p in points
                    ),
                    "rounds_strictly_better": sum(
                        p[error_key] < p["error_identity_new"] - 1e-12 for p in points
                    ),
                    "rounds_strictly_worse": sum(
                        p[error_key] > p["error_identity_new"] + 1e-12 for p in points
                    ),
                    "per_seed_relative_gain_pct": stats(
                        p["relative_gain_pct"] for p in per_seed_effects
                    ),
                    "per_seed_mean_difference": stats(
                        p["difference"] for p in per_seed_effects
                    ),
                    "per_seed": per_seed_effects,
                }
            errors.append(item)
        for rec in runs:
            for p in rec["rows"]:
                if p["active"] or p["frozen_before"] or p["frozen_after"]:
                    events.append(
                        {
                            k: rec[k]
                            for k in (
                                "phase",
                                "noise",
                                "scenario",
                                "timing",
                                "seed",
                                "run",
                            )
                        }
                        | p
                    )
    separation = []
    r3 = [r for r in records if r["phase"] == "R3"]
    clean = {(r["noise"], r["seed"]): r for r in r3 if r["scenario"] == "none"}
    for rec in r3:
        if rec["scenario"] == "none":
            continue
        ctrl = {p["round"]: p for p in clean[(rec["noise"], rec["seed"])]["rows"]}
        for label, first, last in (
            ("entire_history_exposed_period", 18, 40),
            ("onset_transition", 18, 24),
            ("persistent_both_views", 25, 40),
        ):
            pts = [p for p in rec["rows"] if first <= p["round"] <= last]
            null = [ctrl[p["round"]] for p in pts]
            separation.append(
                {
                    "noise": rec["noise"],
                    "scenario": rec["scenario"],
                    "seed": rec["seed"],
                    "window": label,
                    "rounds": len(pts),
                    "mean_ratio_attack": mean(p["full_ratio"] for p in pts),
                    "mean_ratio_clean": mean(p["full_ratio"] for p in null),
                    "paired_mean_ratio_difference": mean(
                        p["full_ratio"] - q["full_ratio"] for p, q in zip(pts, null)
                    ),
                    "auc_high_stat_attack": auc(
                        [p["full_ratio"] for p in pts], [p["full_ratio"] for p in null]
                    ),
                }
            )
    sep_groups = defaultdict(list)
    for row in separation:
        sep_groups[(row["noise"], row["scenario"], row["window"])].append(row)
    sep_summary = [
        dict(zip(("noise", "scenario", "window"), key))
        | {
            field: stats(row[field] for row in rows)
            for field in (
                "mean_ratio_attack",
                "mean_ratio_clean",
                "paired_mean_ratio_difference",
                "auc_high_stat_attack",
            )
        }
        for key, rows in sorted(sep_groups.items())
    ]
    return summaries, errors, events, separation, sep_summary


def prefix_pairing_audit(records):
    """Verify rather than assume equal pre-attack trajectories across scenarios."""
    r3 = [r for r in records if r["phase"] == "R3"]
    clean = {(r["noise"], r["seed"]): r for r in r3 if r["scenario"] == "none"}
    pairs = []
    for rec in r3:
        if rec["scenario"] == "none":
            continue
        baseline = clean[(rec["noise"], rec["seed"])]
        prefix = [
            (p, q) for p, q in zip(rec["prefix_metrics"], baseline["prefix_metrics"])
        ]
        differences = [
            (
                p["round"],
                p["test_accuracy_pct"] - q["test_accuracy_pct"],
                p["test_loss"] - q["test_loss"],
            )
            for p, q in prefix
        ]
        stat_delta = [
            abs(p["full_stat"] - q["full_stat"])
            for p, q in zip(rec["rows"][:5], baseline["rows"][:5])
        ]
        pairs.append(
            {
                "noise": rec["noise"],
                "scenario": rec["scenario"],
                "seed": rec["seed"],
                "max_pre_attack_accuracy_difference_pp": max(
                    abs(d[1]) for d in differences
                ),
                "max_pre_attack_loss_difference": max(abs(d[2]) for d in differences),
                "max_history_unexposed_stat_difference": max(stat_delta),
                "first_global_metric_difference_round": next(
                    (
                        d[0]
                        for d in differences
                        if abs(d[1]) > 1e-12 or abs(d[2]) > 1e-12
                    ),
                    None,
                ),
                "round1_metrics_identical": differences[0][1:] == (0.0, 0.0),
            }
        )
    return {
        "scope": "R3 RCIG attack-versus-clean same-seed comparisons, model metrics rounds1-16 and history stats rounds13-17",
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_with_pre_attack_difference": sum(
            p["first_global_metric_difference_round"] is not None for p in pairs
        ),
        "max_accuracy_difference_pp": max(
            p["max_pre_attack_accuracy_difference_pp"] for p in pairs
        ),
        "max_loss_difference": max(p["max_pre_attack_loss_difference"] for p in pairs),
        "max_stat_difference": max(
            p["max_history_unexposed_stat_difference"] for p in pairs
        ),
        "interpretation": "Same seed is not proof of common realised batches. Evaluation skips5 loaders in attack-configured runs from round1; default DataLoader consumes CPU RNG also used by global randperm. No isolated RNG replay claimed; R2 same-transcript candidate comparisons are unaffected.",
    }


def num(x, digits=4):
    return "NA" if x is None else f"{x:.{digits}f}"


def tables(evidence):
    out = [
        "# RCIG — tableaux mécanistiques recalculés",
        "",
        "Audit descriptif post-hoc, sans nouveau seuil ni entraînement.",
        "",
        "## Statistique normalisée par le seuil figé",
        "",
        "Moyennes des moyennes par seed ; les rounds ne sont pas des réplications indépendantes.",
        "",
        "| Phase | Bruit | Attaque | Début | Fenêtre | Actifs / tours | Ratio moyen | Ratio maximal | Part plancher moyenne | Clipping moyen |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in evidence["summaries"]:
        if s["window"] != ("ready" if s["scenario"] == "none" else "current_attack"):
            continue
        out.append(
            f"| {s['phase']} | {s['noise']} | {s['scenario']} | {s['timing']} | {s['window']} | {s['active_rounds']}/{s['rounds']} | {num(s['full_ratio_per_seed_mean']['mean'])} | {num(s['full_ratio_pooled_descriptive']['max'])} | {num(100*s['floor_fraction_per_seed_mean']['mean'],2)} % | {num(100*s['clip_rate_per_seed_mean']['mean'],2)} % |"
        )
    out += [
        "",
        "## Séparation attaque/propre R3 (12 seeds appariées)",
        "",
        "AUC descriptive : valeur haute prédéfinie comme anormale. Aucune inversion ni sélection de seuil après observation. Appariement par seed seulement : les préfixes propre/attaque ne sont pas strictement identiques (voir Evidence.prefix_pairing).",
        "",
        "| Bruit | Attaque | Fenêtre | Ratio attaque | Ratio propre | Différence appariée moyenne ± SD | AUC moyenne ± SD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for s in evidence["separation_summary"]:
        delta, au = s["paired_mean_ratio_difference"], s["auc_high_stat_attack"]
        out.append(
            f"| {s['noise']} | {s['scenario']} | {s['window']} | {num(s['mean_ratio_attack']['mean'])} | {num(s['mean_ratio_clean']['mean'])} | {num(delta['mean'])} ± {num(delta['sd'])} | {num(au['mean'])} ± {num(au['sd'])} |"
        )
    out += [
        "",
        "## Erreurs de référence sur le même transcript R2",
        "",
        "Δ erreur = candidat − récent, négatif = meilleur ; erreur = norme L2 au carré, pas une loss ni une accuracy. Gain relatif positif = meilleur.",
        "",
        "| Bruit | Attaque | Début | Sélection | Tours | Δ candidat full | Δ référence déployée | Δ milieu | Gain déployé moyen par seed (%) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for e in evidence["errors"]:
        if e["phase"] != "R2" or e["selection"] not in {
            "attack_ready",
            "rolling_active",
            "deployed_frozen",
        }:
            continue
        out.append(
            f"| {e['noise']} | {e['scenario']} | {e['timing']} | {e['selection']} | {e['event_rows']} | {num(e['full']['event_weighted_mean_difference'],6)} | {num(e['reference']['event_weighted_mean_difference'],6)} | {num(e['midpoint']['event_weighted_mean_difference'],6)} | {num(e['reference']['per_seed_relative_gain_pct']['mean'],5)} |"
        )
    out += [
        "",
        "## Tous les événements de gel et d’activation R2/R3",
        "",
        "Les mêmes seeds et tours dans deux bruits ne constituent pas des événements indépendants.",
        "",
        "| Phase | Bruit | Début | Seed | Tour public | Action | Test actif | Ratio | Confiance candidat | Δ erreur candidat | Δ erreur déployée |",
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for e in evidence["events"]:
        if e["phase"] == "R1":
            continue
        base = e.get("error_identity_new")
        df = e["error_full"] - base if base is not None else None
        dr = e["error_reference"] - base if base is not None else None
        out.append(
            f"| {e['phase']} | {e['noise']} | {e['timing']} | {e['seed']} | {e['round']} | {e['action']} | {e['active']} | {num(e['full_ratio'])} | {num(e['full_trust'])} | {num(df,6)} | {num(dr,6)} |"
        )
    return "\n".join(out) + "\n"


def figures(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11})
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, constrained_layout=True
    )
    colors = {"none": "#444444", "bf": "#c44e52", "ipm": "#4c72b0", "alie": "#55a868"}
    for ax, noise in zip(axes, ("homogeneous", "heteroscedastic")):
        for attack in ("none", "bf", "ipm", "alie"):
            rs = [
                r
                for r in records
                if r["phase"] == "R3"
                and r["noise"] == noise
                and r["scenario"] == attack
            ]
            x = list(range(13, 41))
            y = [[r["rows"][j]["full_ratio"] for r in rs] for j in range(28)]
            ax.plot(
                x,
                [st.median(a) for a in y],
                label={"none": "Propre", "bf": "BF ×10", "ipm": "IPM", "alie": "ALIE"}[
                    attack
                ],
                color=colors[attack],
            )
            ax.fill_between(
                x,
                [quantile(a, 0.25) for a in y],
                [quantile(a, 0.75) for a in y],
                color=colors[attack],
                alpha=0.13,
            )
        ax.axhline(1, color="black", linestyle="--", label="Seuil figé")
        ax.axvline(17, color="#777777", linestyle=":")
        ax.axvline(25, color="#999999", linestyle=":")
        ax.set(
            title="Bruit "
            + ("homogène" if noise == "homogeneous" else "hétéroscédastique"),
            ylabel="Statistique / seuil",
        )
        ax.set_ylim(0, 1.12)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=10, loc="lower left")
    axes[1].set_xlabel("Tour public — attaque dès 17 ; deux fenêtres exposées dès 25")
    fig.suptitle(
        "R3 : médiane et intervalle interquartile sur 12 seeds\nBandes descriptives, pas des IC95"
    )
    fig.savefig(figdir / "R3_statistic_threshold.png", dpi=160)
    plt.close(fig)
    # Events are paired oracle diagnostics; scatter shows all 27, no inferential error bars.
    active = [
        (r, p) for r in records if r["phase"] == "R2" for p in r["rows"] if p["active"]
    ]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for label, candidate, color, marker in (
        ("Candidat RCIG calculé", "full", "#4c72b0", "o"),
        ("Référence effectivement utilisée (gel)", "reference", "#c44e52", "x"),
    ):
        ys = [p["error_" + candidate] - p["error_identity_new"] for _, p in active]
        ax.scatter(
            range(1, len(active) + 1), ys, label=label, color=color, marker=marker
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="R2 : la correction calculée et le gel déployé ne sont pas la même opération",
        xlabel="Événement d’activation (ordre de l’inventaire ; 27 lignes, non indépendantes)",
        ylabel="Erreur L2² − erreur de la vue récente\nNégatif = meilleur",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=10)
    fig.savefig(figdir / "R2_candidate_vs_deployed.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "output"):
        raise ValueError("Audit outputs must stay under output/")
    records, sources, validation = audit_inputs()
    summaries, errors, events, sep, sep_summary = summarize(records)
    evidence = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "posthoc_descriptive_no_training_no_threshold_selection",
        "validation": validation,
        "sources_sha256": sources,
        "statistics": "Window averages first within each seed; descriptive SD across seeds; no round-level inferential p-values. Event-only rows explicitly selection-conditioned.",
        "summaries": summaries,
        "errors": errors,
        "events": events,
        "separation_by_seed": sep,
        "separation_summary": sep_summary,
        "prefix_pairing": prefix_pairing_audit(records),
        "records": records,
    }
    # Recheck ALL immutable inputs and analysis sources before publishing any output.
    for relative, digest in sources.items():
        assert sha(ROOT / relative) == digest, f"Input changed during audit: {relative}"
    evidence["source_hashes_unchanged"] = True
    evidence["analyzer_sha256"] = sha(Path(__file__).resolve())
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures(records, out)
    (out / "Evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (out / "Tables.md").write_text(tables(evidence))
    print(
        json.dumps(
            {
                "validation": validation,
                "ready_rows": sum(len(r["rows"]) for r in records),
                "inputs_unchanged": True,
                "R2_active_rows": sum(
                    e["phase"] == "R2" and e["active"] for e in events
                ),
                "R3_active_rows": sum(
                    e["phase"] == "R3" and e["active"] for e in events
                ),
                "output": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
