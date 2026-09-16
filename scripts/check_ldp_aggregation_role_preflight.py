#!/usr/bin/env python3
"""Real Fashion-MNIST/MPS integration checks, not scientific comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts import run_ldp_aggregation_role_ablation as runner


def main():
    runner.shared.require_working_mps()
    runner.no_conflicting_training()
    m = runner.matrix()
    stamp = runner.provenance(m)
    parent = ROOT / "output/validation"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="aggregation_role_preflight_", dir=parent))
    print(out, flush=True)
    records = {}
    cases = [("uniform", "none")] + [
        (arm, "bf_persistent") for arm in runner.ARMS
    ]
    for arm, scenario in cases:
        task = dict(noise="homogeneous", seed=909002, scenario=scenario, arm=arm)
        cfg = runner.config_for(m, task, stamp)
        dest = out / f"{arm}_{scenario}"
        dest.mkdir()
        cfg["output_dir"] = str(dest)
        cfg["training"]["num_rounds"] = 18
        cfg["training"]["algo_config"]["aggregation_role_integration_test"] = True
        path = dest / "config.yaml"
        with path.open("x") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"START {arm}/{scenario}", flush=True)
        with (dest / "training.log").open("x") as log:
            proc = subprocess.run(
                [sys.executable, "-B", "-u", str(runner.ENTRY),
                 "--config", str(path), "--output", str(dest), "--device", "mps"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0",
                         PYTHONDONTWRITEBYTECODE="1"),
            )
        if proc.returncode:
            raise RuntimeError(f"preflight failed: {dest/'training.log'}")
        paths = list(dest.rglob("metrics.json"))
        assert len(paths) == 1
        data = json.loads(paths[0].read_text())
        rows = data["rounds"]
        assert len(rows) == 18
        for t, row in enumerate(rows, 1):
            assert row["ldp_gradient_far_private_gradient_mps_fraction"] == 1
            assert row["far_attack_labels_visible_to_server_aggregate"] is False
            assert row["aggregation_role_oracles_visible_to_server"] is False
            assert row["aggregation_role_same_cohort_verified"] is True
            assert row["aggregation_role_effective_arm"] == ("uniform" if t <= 12 else arm)
            for candidate in runner.ARMS:
                pre = "cohort_" + candidate + "_"
                err = row[pre + "aggregate_error_sq"]
                assert row[pre + "decomposition_residual_sq"] <= 1e-18 * max(1, err)
                terms = ["honest_effective_noise_sq", "honest_tilt_sq", "byzantine_centered_sq",
                         "cross_noise_tilt", "cross_noise_byzantine", "cross_tilt_byzantine"]
                assert abs(sum(row[pre + k] for k in terms) - err) <= 1e-9 * max(1, err)
            if t > 12:
                assert row["shadow_rcig_newer_round_max"] < t - 1
        trace = sorted(
            (json.loads(s) for s in (dest / "simulator_randomness_private_audit.jsonl")
             .read_text().splitlines()), key=lambda r: (r["round"], r["client_id"]),
        )
        assert len(trace) == 180
        records[(arm, scenario)] = (rows, trace)
        runtime = json.loads((dest / "runtime_imports.json").read_text())
        assert runtime["stage"] == "after_training" and runtime["mps_fallback"] == 0
        for name, h in runtime["source_sha256"].items():
            assert stamp["sources"].get(name) == h, f"unlocked runtime source {name}"
        runner.verify(stamp, m)
        print(f"PASS {arm}/{scenario}", flush=True)
    clean, clean_trace = records[("uniform", "none")]
    for case, (rows, trace) in records.items():
        for r, s in zip(trace, clean_trace):
            assert r["permutations"] == s["permutations"]
            assert r["standard_gaussians"] == s["standard_gaussians"]
            if r["round"] <= 12 or (case[0] == "uniform" and r["round"] <= 17):
                assert r["model_before"] == s["model_before"]
                assert r["private_upload"] == s["private_upload"]
        if case[0] == "uniform":
            for r, s in zip(rows[:16], clean[:16]):
                assert r["test_accuracy"] == s["test_accuracy"]
                assert r["test_loss"] == s["test_loss"]
    evidence = dict(
        status="passed", scientific_results=False, real_data_mps_runs=5,
        rounds_each=18, seed=909002, same_batches_and_standard_gaussians=True,
        uniform_clean_attack_prefix_identical=True, common_warmup_identical=True,
        all_four_candidate_decompositions_verified=True, oracle_boundary_checked=True,
        private_gradient_device="mps", fallback=0,
        source_hashes_unchanged=True, sources=stamp["sources"], output=str(out),
    )
    runner.write_json(out / "verification.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
