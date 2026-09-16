#!/usr/bin/env python3
"""Source-locked 36-run clean screen. No historical campaign is modified."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import json
import math
import os
import subprocess
import threading
import time
import yaml

from scripts.run_rcig_batch_screen import file_hash, canonical_hash, source_closure, require_working_mps
from scripts.run_rcig_n10_policy_ablation import privacy

CAMPAIGN = "client_momentum_n10_v1"
MATRIX = ROOT / "configs/ldp_gradient_far" / (CAMPAIGN + ".yaml")
OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN
ENTRY = Path(__file__).resolve()
STOP = threading.Event()


def require(test, message):
    if not test:
        raise RuntimeError(message)


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def matrix():
    m = yaml.safe_load(MATRIX.read_text())
    require(m["campaign_id"] == CAMPAIGN and m["expected_runs"] == 36, "campaign mismatch")
    require(len(tasks(m)) == 36 and len(set(m["seeds"])) == 3, "invalid grid")
    return m


def tasks(m, noise=None):
    return [dict(noise=n, seed=s, beta=b, arm=a)
            for n, s, b, a in itertools.product(m["noise_regimes"], m["seeds"], m["betas"], m["arms"])
            if noise is None or n == noise]


def run_id(t):
    return f'{t["noise"]}__seed{t["seed"]}__beta{t["beta"]:g}__{t["arm"]}'


def directory(t):
    return OUTPUT / run_id(t)


def stamp_for(m):
    return dict(campaign=CAMPAIGN, matrix_hash=file_hash(MATRIX),
                protocol_hash=file_hash(ROOT/m["protocol"]), privacy=privacy(m),
                sources=source_closure((ENTRY, ROOT/"run_experiment.py")),
                expected_runs=36, private_device="mps", client_momentum_device="mps",
                aggregation_device="cpu_float64", fallback=0)


def verify(stamp, m):
    require(file_hash(MATRIX) == stamp["matrix_hash"], "matrix changed during campaign")
    require(file_hash(ROOT/m["protocol"]) == stamp["protocol_hash"], "protocol changed during campaign")
    changed = [p for p, h in stamp["sources"].items() if file_hash(ROOT/p) != h]
    require(not changed, "locked sources changed: " + ", ".join(changed))


def config_for(m, t, stamp):
    a = dict(batch_size=m["batch_size"], fixed_batch_size=m["batch_size"],
        local_epochs=1, fixed_steps_per_round=1, sampling_scheme="fixed_without_replacement",
        privacy_adjacency="replace_one", privacy_public_dataset_size=m["public_local_size"],
        privacy_sampling_rate_override=m["batch_size"]/m["public_local_size"],
        per_sample_backend="vectorized", enable_dp=True, target_epsilon=m["epsilon"],
        noise_multiplier=stamp["privacy"]["sigma"], privacy_num_rounds=m["rounds"], delta=m["delta"],
        privacy_noise_multiplier_scale_by_client=m["noise_regimes"][t["noise"]],
        clip_norm=m["local_clip_norm"], far_server_lr=m["server_lr"],
        far_server_clip_norm=m["server_clip_norm"], far_score_mode="raw_distance",
        noise_score_standardization="none", score_subspace_mode="full",
        far_alpha=m["far_alpha"] if t["arm"] == "far_rfa" else 0.0,
        kappa_w=2.0, tilt_bound_policy="diagnostic", robust_reference="rfa", num_byzantine=0,
        rfa_max_iter=100, rfa_tol=1e-6, expected_num_clients=10, client_metrics_every=1,
        fairness_tail_fraction=.2, suppress_private_client_diagnostics=True,
        external_attack_diagnostics=True, enable_oracle_diagnostics=True,
        attack=dict(enabled=False, name="none", num_byzantine=0, client_ids=[]),
        client_momentum_beta=t["beta"], momentum_arm=t["arm"], momentum_pairing_seed=t["seed"],
        momentum_campaign=CAMPAIGN, momentum_scientific_hash=canonical_hash(stamp),
        momentum_runtime_implementation="algorithms.ldp_client_momentum.LDPClientMomentum")
    return dict(seed=t["seed"], device="mps", output_dir=str(directory(t)),
        data=dict(dataset="fashionmnist", model="lenet5_tanh", partition="client_dirichlet_balanced",
                  alpha=.1, partition_seed=t["seed"], data_root=str(ROOT/"data")),
        model=dict(architecture="lenet5_tanh"),
        clients=dict(num_clients=10, min_clients=10, sample_fraction=1.0, sampling_strategy="random",
                     dropout_rate=0.0, fleet=[dict(type="raspberry_pi_4", count=10)],
                     battery_init=dict(distribution="uniform_soc", params=dict(min_soc=1.0, max_soc=1.0))),
        eval=dict(eval_every=1, honest_clients_only=True),
        training=dict(num_rounds=m["rounds"], cost_model="measured", algorithm="ldp_gradient_far", algo_config=a))


def trace_for(t):
    rows = [json.loads(line) for line in (directory(t)/"simulator_randomness_private_audit.jsonl").read_text().splitlines()]
    rows.sort(key=lambda x: (x["round"], x["client_id"]))
    require(len(rows) == 400 and {(r["round"], r["client_id"]) for r in rows} == set(itertools.product(range(1, 41), range(10))), "incomplete client trace")
    last = {}
    for r in rows:
        require(r["momentum_device"].startswith("mps") and r["beta"] == t["beta"], "wrong client device/beta")
        require(r["memory_before"] == last.get(r["client_id"]), "momentum buffer did not persist")
        require(r["memory_after"] == r["private_upload"], "wrong upload")
        if r["round"] == 1 or t["beta"] == 0:
            require(r["private_upload"] == r["raw_private_gradient"], "initial/identity momentum changed gradient")
        last[r["client_id"]] = r["memory_after"]
    return rows


def check_pairing(t, other):
    for r, s in zip(trace_for(t), trace_for(other), strict=True):
        require(r["permutations"] == s["permutations"] and r["standard_gaussians"] == s["standard_gaussians"], "actual batch/Gaussian streams are not paired")
        if r["round"] == 1:
            require(r["model_before"] == s["model_before"], "initial models not identical")
            if t["noise"] == other["noise"]:
                require(r["raw_private_gradient"] == s["raw_private_gradient"], "first-round private gradients not identical")


def validate(m, t, cfg, stamp):
    dest = directory(t)
    paths = list(dest.rglob("metrics.json"))
    require(len(paths) == 1, "missing/ambiguous metrics")
    p = paths[0]; x = json.loads(p.read_text()); rows = x["rounds"]
    require(yaml.safe_load((dest/"resolved_config.yaml").read_text()) == cfg, "config changed")
    require(len(rows) == 40 and x["summary"]["seed"] == t["seed"], "incomplete/wrong training")
    for k, v in cfg["training"]["algo_config"].items():
        require(x["config"].get(k) == v, "runtime config mismatch: " + k)
    for k, r in enumerate(rows, 1):
        require(r["round_num"] == k and r["num_alive_clients"] == 10 and r["num_evaluated_clients"] == 10, "cohort mismatch")
        require(r["momentum_history_length"] == k and r["momentum_arm"] == t["arm"] and r["client_momentum_beta"] == t["beta"], "wrong deployed method")
        require(r["momentum_private_gradient_device"] == "mps" and r["momentum_buffer_device"] == "mps", "CPU training forbidden")
        require(r["momentum_warmup_rounds"] == 0 and r["far_alpha"] == cfg["training"]["algo_config"]["far_alpha"], "undeclared warmup/alpha")
        require(r["momentum_oracle_evaluation_only"] and not r["far_attack_labels_visible_to_server_aggregate"], "oracle boundary mismatch")
        require(r["privacy_adjacency"] == "replace_one" and r["privacy_sampling_scheme"] == "fixed_without_replacement", "DP contract changed")
        for key in ("test_accuracy", "test_loss", "client_loss_mean", "client_accuracy_variance_pct2",
                    "worst20_accuracy_pct", "best20_worst20_gap_pct", "momentum_oracle_applied_mse_current_clean_mean",
                    "momentum_oracle_fresh_noise_energy", "momentum_oracle_filtered_noise_energy", "momentum_oracle_clean_lag_energy"):
            require(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]), "missing/nonfinite " + key)
    require(abs(rows[-1]["privacy_epsilon_max"] - 4) < 1e-4, "incorrect final epsilon")
    runtime = json.loads((dest/"runtime_imports.json").read_text())
    require(runtime["stage"] == "after_training" and runtime["fallback"] == 0, "runtime not complete")
    require(all(stamp["sources"].get(p) == h for p, h in runtime["sources"].items()), "unlocked runtime source")
    trace_for(t)
    baseline = dict(t, beta=0.0, arm="uniform")
    if baseline != t:
        require(json.loads((directory(baseline)/"orchestration_status.json").read_text())["status"] == "completed", "paired control must complete first")
        check_pairing(t, baseline)
    return dict(metrics_path=str(p.relative_to(ROOT)), metrics_sha256=file_hash(p),
                trace_sha256=file_hash(dest/"simulator_randomness_private_audit.jsonl"),
                final_test_accuracy=rows[-1]["test_accuracy"], epsilon_max=rows[-1]["privacy_epsilon_max"],
                paired_control=run_id(baseline), validated_rounds=40, validated_client_records=400)


def worker(cp, dest):
    require_working_mps()
    from algorithms.base import register_algorithm
    from algorithms.ldp_client_momentum import LDPClientMomentum, MomentumEvaluator
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    from scripts.run_rcig_batch_screen_experiment import runtime_source_hashes
    import run_experiment as harness
    register_algorithm("ldp_gradient_far")(LDPClientMomentum)
    old_detach, old_eval, oldargv = harness.detach_rcig_evaluation_oracles, harness.rcig_reference_oracle_metrics, sys.argv
    def detach(updates, **kwargs):
        safe, clean = old_detach(updates, enabled=True, strip_attack_oracles=True)
        for _, metadata, _ in safe:
            metadata.pop("momentum_raw_private_gradient_oracle", None)
        return safe, clean
    harness.detach_rcig_evaluation_oracles = detach
    harness.rcig_reference_oracle_metrics = MomentumEvaluator()
    manifest = dict(stage="before_training", sources=runtime_source_hashes(), private_device="mps",
                    momentum_device="mps", server_device="cpu_float64", fallback=0, algorithm=CAMPAIGN)
    save(dest/"runtime_imports.json", manifest)
    try:
        with (dest/"simulator_randomness_private_audit.jsonl").open("x") as trace:
            def sink(row):
                trace.write(json.dumps(row, sort_keys=True) + "\n"); trace.flush()
            LDPClientMomentum.audit_sink = staticmethod(sink)
            sys.argv = [str(ROOT/"run_experiment.py"), "--config", str(cp), "--output", str(dest), "--device", "mps"]
            with evaluation_rng_isolation(harness):
                harness.main()
        require(all(file_hash(ROOT/p) == h for p, h in manifest["sources"].items()), "source drift during worker")
        manifest.update(stage="after_training", sources=runtime_source_hashes())
        save(dest/"runtime_imports.json", manifest)
    finally:
        LDPClientMomentum.audit_sink = None
        harness.detach_rcig_evaluation_oracles, harness.rcig_reference_oracle_metrics, sys.argv = old_detach, old_eval, oldargv


def run_one(m, t, stamp):
    if STOP.is_set():
        return
    verify(stamp, m)
    dest = directory(t); status_path = dest/"orchestration_status.json"
    cfg = config_for(m, t, stamp)
    if dest.exists():
        require(status_path.exists(), "partial directory: " + str(dest))
        old = json.loads(status_path.read_text())
        require(old["status"] == "completed", "partial/failed run needs inspection: " + run_id(t))
        evidence = validate(m, t, cfg, stamp)
        require(old["metrics_sha256"] == evidence["metrics_sha256"] and old["trace_sha256"] == evidence["trace_sha256"], "completed artifact changed")
        return
    dest.mkdir(parents=True)
    cp = dest/"resolved_config.yaml"; cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    status = dict(status="starting", task=t, supervisor_pid=os.getpid(), started_at=time.time())
    save(status_path, status)
    try:
        with (dest/"training.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-B", "-u", str(ENTRY), "--worker", str(cp), "--output", str(dest)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK":"0", "PYTHONDONTWRITEBYTECODE":"1"})
            status.update(status="running", worker_pid=child.pid); save(status_path, status)
            print("START", run_id(t), "pid", child.pid, flush=True)
            require(child.wait() == 0, "worker failed; inspect " + str(dest/"training.log"))
        verify(stamp, m)
        status.update(validate(m, t, cfg, stamp), status="completed", completed_at=time.time())
        save(status_path, status)
        print("COMPLETE", run_id(t), flush=True)
    except BaseException as exc:
        STOP.set(); status.update(status="failed", error=str(exc), stopped_at=time.time()); save(status_path, status)
        raise


def status(m):
    out = dict(completed=0, total=36, active=[], failed=[], missing=0)
    for t in tasks(m):
        p = directory(t)/"orchestration_status.json"
        if not p.exists():
            out["missing"] += 1; continue
        s = json.loads(p.read_text())
        if s["status"] == "completed":
            out["completed"] += 1
        else:
            out["active" if s["status"] in {"starting", "running"} else "failed"].append(s)
    return out


def no_competitor():
    ps = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    matches = []
    for line in ps.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == os.getpid():
            continue
        command = parts[1]
        if ("python" in command.lower() and
            ("run_experiment.py" in command or "--worker" in command or
             ("scripts/run_" in command and any(flag in command for flag in ("--run", "--launch"))))):
            matches.append(line)
    require(not matches, "other training/supervisor active: " + "\n".join(matches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ("plan", "status", "launch", "run", "audit"):
        group.add_argument("--"+mode, action="store_true")
    group.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        require(args.output is not None, "missing worker output"); worker(args.worker, args.output); return
    m = matrix()
    if args.status:
        print(json.dumps(status(m), indent=2)); return
    if args.plan:
        stamp = stamp_for(m)
        print(json.dumps(dict(tasks=tasks(m), privacy=stamp["privacy"], source_count=len(stamp["sources"]), total=36), indent=2)); return
    require(args.resume or args.audit, "--resume is mandatory")
    if args.launch:
        require_working_mps(); no_competitor()
        logdir = ROOT/"logs/ldp_gradient_far"; logdir.mkdir(parents=True, exist_ok=True)
        with (logdir/(CAMPAIGN+".log")).open("a") as log:
            p = subprocess.Popen([sys.executable, "-B", "-u", str(ENTRY), "--run", "--resume"], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK":"0", "PYTHONDONTWRITEBYTECODE":"1"})
        print(json.dumps(dict(supervisor_pid=p.pid, log=str(logdir/(CAMPAIGN+".log"))))); return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT.parent/("."+CAMPAIGN+".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.audit:
            require_working_mps(); no_competitor()
        OUTPUT.mkdir(exist_ok=True)
        frozen = OUTPUT/"campaign_lock.json"
        if frozen.exists():
            stamp = json.loads(frozen.read_text()); verify(stamp, m)
        else:
            require(not args.audit, "campaign not launched")
            stamp = stamp_for(m); save(frozen, stamp)
        if args.audit:
            for t in tasks(m):
                validate(m, t, config_for(m, t, stamp), stamp)
        else:
            def lane(noise):
                for t in tasks(m, noise):
                    if STOP.is_set(): break
                    run_one(m, t, stamp)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(lane, n) for n in m["noise_regimes"]]
                for f in futures: f.result()
        for t in tasks(m):
            check_pairing(t, dict(t, noise="homogeneous", beta=0.0, arm="uniform"))
        save(OUTPUT/"completion_evidence.json", dict(status="completed", completed=36, total=36,
            finished_at=time.time(), cross_rule_beta_noise_actual_draw_pairing=True,
            scientific_verdict="await_paired_model_quality_analysis_no_automatic_promotion"))
        print(json.dumps(status(m), indent=2))


if __name__ == "__main__":
    main()
