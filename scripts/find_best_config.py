#!/usr/bin/env python3
"""
Lit tous les metrics.json de results/tuning/ et affiche les hyperparamètres
optimaux par algo (critère : Best Accuracy sur 100 rounds).

Usage:
    python scripts/find_best_config.py results/tuning/
"""
import json
import sys
from pathlib import Path


def load_results(tuning_dir: Path) -> list[dict]:
    records = []
    for metrics_file in tuning_dir.rglob("metrics.json"):
        with open(metrics_file) as f:
            data = json.load(f)
        tag     = metrics_file.parent.parent.name
        summary = data.get("summary", {})
        config  = data.get("config", {})
        records.append({
            "tag":          tag,
            "algo":         data.get("algorithm", "?"),
            "local_epochs": config.get("local_epochs", "?"),
            "T_rot":        config.get("T_rot", "—"),
            "num_tiers":    config.get("num_tiers", "—"),
            "mu":           config.get("mu", "—"),
            "freeze":       config.get("freeze_secondary_backward", False),
            "best_acc":     summary.get("best_accuracy", 0.0),
            "final_acc":    summary.get("final_accuracy", 0.0),
        })
    return records


def print_group(records: list[dict], tag_prefix: str, label: str) -> dict:
    subset = [r for r in records if r["tag"].startswith(tag_prefix)]
    if not subset:
        print(f"  (aucun résultat pour {label})")
        return {}
    subset.sort(key=lambda r: r["best_acc"], reverse=True)

    print(f"\n── {label} ──")
    print(f"  {'Tag':<34} {'E':>3} {'T_rot':>6} {'Tiers':>6} {'mu':>6} "
          f"{'Best Acc':>10} {'Final Acc':>10}")
    print("  " + "─" * 82)
    for r in subset:
        print(f"  {r['tag']:<34} {str(r['local_epochs']):>3} "
              f"{str(r['T_rot']):>6} {str(r['num_tiers']):>6} {str(r['mu']):>6} "
              f"{r['best_acc']*100:>9.2f}% {r['final_acc']*100:>9.2f}%")

    best = subset[0]
    print(f"\n  ★ OPTIMAL : {best['tag']}")
    print(f"    local_epochs = {best['local_epochs']}")
    for k in ["T_rot", "num_tiers", "mu"]:
        if best[k] != "—":
            print(f"    {k:<14} = {best[k]}")
    print(f"    Best Acc     = {best['best_acc']*100:.2f}%")
    return best


def main():
    tuning_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tuning")
    if not tuning_dir.exists():
        print(f"ERROR: {tuning_dir} does not exist.")
        sys.exit(1)

    records = load_results(tuning_dir)
    if not records:
        print("Aucun metrics.json trouvé. Jobs tuning peut-être encore en cours.")
        sys.exit(0)

    print("=" * 90)
    print("  RÉSULTATS TUNING — Hyperparamètres optimaux par algo")
    print(f"  Dossier : {tuning_dir.resolve()}")
    print("=" * 90)

    # ── Baselines ────────────────────────────────────────────
    groups_baselines = [
        ("fedavg_",         "FedAvg       [baseline]"),
        ("fedpart_e",       "FedPart      [baseline]"),
        ("heterofl_",       "HeteroFL     [baseline]"),
        ("fjord_",          "FjORD        [baseline]"),
        ("fedprox_",        "FedProx      [baseline]"),
    ]
    # ── Proposés ─────────────────────────────────────────────
    groups_proposed = [
        ("fedpartbe_",      "FedStep    [proposé]"),
        ("ccsEF_full_",     "ccsEF full   [proposé]"),
        ("ccsEF_frozen_",   "ccsEF frozen [proposé]"),
    ]

    best_configs = {}
    for prefix, label in groups_baselines + groups_proposed:
        key = prefix.rstrip("_")
        best = print_group(records, prefix, label)
        if best:
            best_configs[key] = best

    # ── Commandes benchmark ───────────────────────────────────
    print("\n" + "=" * 90)
    print("  COMMANDES BENCHMARK — copier dans hpc_phase2_benchmark.sh")
    print("=" * 90)

    id_map = {
        "fedavg":         0,
        "fedpart_e":      1,
        "heterofl":       2,
        "fjord":          3,
        "fedprox":        4,
        "fedpartbe":      5,
        "ccsEF_full":     6,
        "ccsEF_frozen":   7,
    }

    for prefix, label in groups_baselines + groups_proposed:
        key  = prefix.rstrip("_")
        best = best_configs.get(key)
        if not best:
            continue
        algo_name  = "ccsEF" if "ccsEF" in key else best["algo"]
        freeze     = "1" if best.get("freeze") else "0"
        extra_key  = "none"
        extra_val  = "none"
        if best["T_rot"] != "—":
            extra_key, extra_val = "T_rot", best["T_rot"]
        elif best["num_tiers"] != "—":
            extra_key, extra_val = "num_tiers", best["num_tiers"]
        elif best["mu"] != "—":
            extra_key, extra_val = "mu", best["mu"]
        tag = f"{key}_opt"
        print(f'  "{algo_name:<12} {best["local_epochs"]}  {extra_key:<10} {extra_val:<6} '
              f'{freeze}  {tag}"   # {label}')

    print()


if __name__ == "__main__":
    main()
