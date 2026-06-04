#!/usr/bin/env python3
"""
hpc/launch_zmq_hpc.py
=====================
FedPartBE ZMQ Distributed Experiment Launcher for UM6P HPC (SLURM).

Each SLURM job allocates ONE node and runs:
  - 1 ZMQ server (tcp://*:ROUTER_PORT, tcp://*:PUB_PORT)
  - N ZMQ workers (one per client), each connecting to 127.0.0.1:ROUTER_PORT

Workers run as parallel background OS processes on the same node.
This is a real distributed FL simulation via ZMQ sockets (DEALER/ROUTER + PUB/SUB).

Usage:
  # Dry-run — print SLURM scripts without submitting
  python hpc/launch_zmq_hpc.py --mode dry-run

  # Submit all enabled experiments
  python hpc/launch_zmq_hpc.py --mode slurm

  # Submit only a specific group
  python hpc/launch_zmq_hpc.py --mode slurm --group benchmark

  # Submit one experiment, one seed
  python hpc/launch_zmq_hpc.py --mode slurm --exp benchmark_fedpartbe --seed 42

  # Run locally (sequential, for testing on laptop)
  python hpc/launch_zmq_hpc.py --mode local --exp benchmark_fedpartbe --seed 42

Requirements:
  pip install pyyaml numpy
  SLURM with sbatch available (for --mode slurm)

IMPORTANT — battery override:
  Pre-sampled SOC values are written to a JSON file (output_dir/batteries.json)
  and passed to the server via FEDLAB_CLIENT_BATTERIES env var.
  server/server.py reads this to override profile-default batteries per client.
  Add the following snippet to FedLabServer.wait_for_registrations() (just before
  the `battery_j = profile.battery.initial_energy_j` line):

    _batteries_env = os.environ.get("FEDLAB_CLIENT_BATTERIES", "{}")
    _batteries_override = json.loads(_batteries_env)
    ...
    # inside the loop, replace:
    battery_j = profile.battery.initial_energy_j if profile else 100000.0
    # with:
    battery_j = float(_batteries_override.get(str(cid),
                    profile.battery.initial_energy_j if profile else 100000.0))
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed.  Run: pip install pyyaml")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not found — battery SOC values will use distribution means.")


PROJ = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = PROJ / "configs" / "fedpartbe_survival_wide_cifar10.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# YAML loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Battery SOC sampling
# ─────────────────────────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    sys.path.insert(0, str(PROJ))
    try:
        from hardware.profiles import DEVICE_PROFILES
        return DEVICE_PROFILES
    except ImportError:
        return {}


def _sample_soc(dist: str, params: dict, rng) -> float:
    if dist == "uniform_soc":
        lo = float(params.get("min_soc", 0.05))
        hi = float(params.get("max_soc", 0.95))
        return float(rng.uniform(lo, hi)) if HAS_NUMPY else (lo + hi) / 2.0
    elif dist == "weibull_soc":
        kappa = float(params.get("kappa", 2.0))
        lam   = float(params.get("lambda_soc", 0.7))
        lo    = float(params.get("min_soc", 0.05))
        hi    = float(params.get("max_soc", 0.95))
        if HAS_NUMPY:
            raw = float(rng.weibull(kappa) * lam)
            return max(lo, min(hi, raw))
        return lam
    else:  # gaussian
        std  = float(params.get("std", 0.1))
        mean = float(params.get("mean_soc", 0.5))
        if HAS_NUMPY:
            return max(0.05, min(0.95, float(rng.normal(mean, std))))
        return mean


def assign_batteries(fleet_def: dict, seed: int) -> list:
    """
    Returns list of (client_id, device_type, battery_j).
    Uses per-device battery distributions when battery_init_by_device is present.
    """
    profiles = _load_profiles()
    rng = np.random.default_rng(seed) if HAS_NUMPY else None

    per_device = fleet_def.get("battery_init_by_device", {})
    global_init = fleet_def.get("battery_init", {
        "distribution": "uniform_soc",
        "params": {"min_soc": 0.05, "max_soc": 0.95},
    })

    assignments = []
    client_id = 0
    for entry in fleet_def["composition"]:
        device_type = entry["device"]
        count       = entry["count"]
        b_init      = per_device.get(device_type, global_init)
        dist        = b_init.get("distribution", "uniform_soc")
        params      = b_init.get("params", {})

        profile     = profiles.get(device_type)
        capacity_j  = profile.battery.capacity_j if profile else 13320.0

        for _ in range(count):
            soc       = _sample_soc(dist, params, rng)
            battery_j = soc * capacity_j
            assignments.append((client_id, device_type, round(battery_j, 2)))
            client_id += 1

    return assignments


# ─────────────────────────────────────────────────────────────────────────────
# Algo config resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_algo_config(cfg: dict, algorithm: str, overrides: dict) -> dict:
    defaults = dict(cfg.get("algo_defaults", {}).get(algorithm, {}))
    return {**defaults, **overrides}


# ─────────────────────────────────────────────────────────────────────────────
# SLURM script generation
# ─────────────────────────────────────────────────────────────────────────────

def _slurm_time(exp: dict, slurm_cfg: dict) -> str:
    group = exp.get("group", "ablation")
    key   = f"time_{group}"
    return slurm_cfg.get(key, slurm_cfg.get("time_ablation", "05:00:00"))


def build_slurm_script(
    exp: dict,
    seed: int,
    cfg: dict,
    client_assignments: list,  # [(client_id, device_type, battery_j), ...]
) -> str:
    global_cfg = cfg.get("global", {})
    hpc_cfg    = cfg.get("hpc", {})
    slurm_cfg  = hpc_cfg.get("slurm", {})
    zmq_cfg    = hpc_cfg.get("zmq", {})

    exp_name    = f"{exp['name']}_s{seed}"
    algorithm   = exp["training"]["algorithm"]
    overrides   = exp.get("algo_config_overrides", {}) or {}
    algo_config = resolve_algo_config(cfg, algorithm, overrides)
    num_clients = len(client_assignments)
    num_rounds  = exp["training"].get("num_rounds", global_cfg.get("num_rounds", 200))

    router_port   = zmq_cfg.get("router_port", 5555)
    pub_port      = zmq_cfg.get("pub_port", 5556)
    round_timeout = zmq_cfg.get("round_timeout_s", 900)
    srv_delay     = zmq_cfg.get("server_start_delay_s", 3)

    output_base = global_cfg.get("output_base", "./results_hpc")
    output_dir  = f"{output_base}/{exp['group']}/{exp_name}"
    device      = global_cfg.get("device", "cpu")
    data_root   = global_cfg.get("data_root", "./data")
    python_bin  = hpc_cfg.get("python", "python3")

    partition  = slurm_cfg.get("partition", "gpu")
    account    = slurm_cfg.get("account", "um6p_fl")
    mem_total  = slurm_cfg.get("mem_total", "64G")
    log_dir    = slurm_cfg.get("log_dir", "./logs")
    wall_time  = _slurm_time(exp, slurm_cfg)

    # Clamp CPUs: 4 per client + 4 for server, capped at 256
    total_cpus = min(4 * (num_clients + 1), 256)

    # Serialize algo_config as JSON (escape for bash double-quotes)
    algo_json = json.dumps(algo_config)

    # Batteries JSON map: "client_id" → battery_j
    batteries_map = {str(cid): batt for cid, _, batt in client_assignments}
    batteries_json = json.dumps(batteries_map)

    # ── Common env vars block (exported once, shared by server + workers) ──
    # We write them into the script as exported shell variables so the long
    # algo_config JSON doesn't need escaping in every worker line.
    common_env_lines = f"""\
export FEDLAB_ALGORITHM="{algorithm}"
export FEDLAB_DATASET="{exp['data']['dataset']}"
export FEDLAB_MODEL="{exp['data']['model']}"
export FEDLAB_NUM_ROUNDS="{num_rounds}"
export FEDLAB_NUM_CLIENTS="{num_clients}"
export FEDLAB_PARTITION="{exp['data']['partition']}"
export FEDLAB_ALPHA="{exp['data']['alpha']}"
export FEDLAB_DEVICE="{device}"
export FEDLAB_OUTPUT_DIR="{output_dir}"
export FEDLAB_EXP_NAME="{exp_name}"
export FEDLAB_LR="{algo_config.get('lr', 0.01)}"
export FEDLAB_LOCAL_EPOCHS="{algo_config.get('local_epochs', 8)}"
export FEDLAB_BATCH_SIZE="{algo_config.get('batch_size', 32)}"
export FEDLAB_ROUTER_PORT="{router_port}"
export FEDLAB_PUB_PORT="{pub_port}"
export FEDLAB_ROUND_TIMEOUT="{round_timeout}"
export FEDLAB_DATA_ROOT="{data_root}"
export FEDLAB_SEED="{seed}"
export FEDLAB_ALGO_CONFIG='{algo_json}'
export FEDLAB_CLIENT_BATTERIES='{batteries_json}'"""

    # ── Per-worker launch lines ────────────────────────────────────────────
    worker_lines = []
    for cid, device_type, battery_j in client_assignments:
        line = (
            f"FEDLAB_CLIENT_ID={cid} "
            f'FEDLAB_DEVICE_PROFILE="{device_type}" '
            f"FEDLAB_SERVER_HOST=127.0.0.1 "
            f"FEDLAB_INITIAL_BATTERY_J={battery_j:.2f} "
            f"{python_bin} worker/worker.py "
            f"> \"{output_dir}/worker_{cid:03d}.log\" 2>&1 &"
        )
        worker_lines.append(f"    {line}")
        worker_lines.append(f"    PIDS+=($!)")
    workers_block = "\n".join(worker_lines)

    # ── Full SLURM script ──────────────────────────────────────────────────
    script = f"""#!/bin/bash
# =====================================================================
# FedPartBE ZMQ Experiment — SLURM job script (auto-generated)
# Experiment : {exp_name}
# Group      : {exp.get('group', 'unknown')}
# Algorithm  : {algorithm}
# Dataset    : {exp['data']['dataset']} / {exp['data']['model']}
# Fleet      : {exp['fleet']} ({num_clients} clients)
# Seed       : {seed}
# Rounds     : {num_rounds}
# =====================================================================
#SBATCH --job-name=fp_{exp['name'][:14]}_s{seed}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={total_cpus}
#SBATCH --mem={mem_total}
#SBATCH --time={wall_time}
#SBATCH --output={PROJ}/{log_dir}/{exp_name}_%j.out
#SBATCH --error={PROJ}/{log_dir}/{exp_name}_%j.err

set -euo pipefail
cd {PROJ}
export PYTHONPATH="{PROJ}:${{PYTHONPATH:-}}"

# ── Setup ─────────────────────────────────────────────────────────
mkdir -p "{output_dir}"
mkdir -p "{PROJ}/{log_dir}"

echo "====================================================="
echo " FedPartBE ZMQ Experiment (SLURM)"
echo "====================================================="
echo " Job        : ${{SLURM_JOB_ID}}"
echo " Experiment : {exp_name}"
echo " Algorithm  : {algorithm}"
echo " Dataset    : {exp['data']['dataset']} / {exp['data']['model']}"
echo " Fleet      : {exp['fleet']} ({num_clients} clients)"
echo " Seed       : {seed}"
echo " Node       : $(hostname)"
echo " Start      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "====================================================="

# ── Common environment variables ──────────────────────────────────
{common_env_lines}

# ── ZMQ Server ────────────────────────────────────────────────────
echo ""
echo "[Launcher] Starting ZMQ server (ROUTER:{router_port}, PUB:{pub_port})..."
{python_bin} server/server.py > "{output_dir}/server.log" 2>&1 &
SERVER_PID=$!
echo "[Launcher] Server PID: $SERVER_PID"

# Give server time to bind sockets before workers connect
sleep {srv_delay}

# ── ZMQ Workers ───────────────────────────────────────────────────
echo "[Launcher] Launching {num_clients} workers in parallel..."
declare -a PIDS=()

{workers_block}

echo "[Launcher] All {num_clients} workers launched."
echo "[Launcher] Worker PIDs: ${{PIDS[*]}}"
echo ""

# ── Wait for all processes ────────────────────────────────────────
# The server broadcasts SHUTDOWN when training is complete.
# Workers exit after receiving SHUTDOWN.
# The server process exits last.
wait
EXIT_CODE=$?

echo ""
echo "[Launcher] Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[Launcher] SUCCESS"
    echo "[Launcher] Results: {output_dir}"
    ls -lh "{output_dir}/" 2>/dev/null || true
else
    echo "[Launcher] FAILED (exit code: $EXIT_CODE)"
    echo "[Launcher] Server log:"
    tail -20 "{output_dir}/server.log" 2>/dev/null || true
    exit $EXIT_CODE
fi
"""
    return script


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FedPartBE ZMQ HPC Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hpc/launch_zmq_hpc.py --mode dry-run
  python hpc/launch_zmq_hpc.py --mode slurm --group benchmark
  python hpc/launch_zmq_hpc.py --mode slurm --exp benchmark_fedpartbe --seed 42
  python hpc/launch_zmq_hpc.py --mode local --exp benchmark_fedpartbe --seed 42
        """,
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Path to master YAML config (default: configs/fedpartbe_survival_wide_cifar10.yaml)",
    )
    parser.add_argument(
        "--mode", choices=["dry-run", "slurm", "local"], default="dry-run",
        help="dry-run=print scripts | slurm=sbatch | local=run sequentially (testing)",
    )
    parser.add_argument("--group", help="Filter: run only this experiment group")
    parser.add_argument("--exp",   help="Filter: run only this experiment name")
    parser.add_argument("--seed",  type=int, help="Filter: run only this seed")
    parser.add_argument(
        "--save-scripts", action="store_true",
        help="Always save SLURM scripts to logs/ (also done automatically for --mode slurm)",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    experiments = cfg.get("experiments", [])

    # ── Apply filters ────────────────────────────────────────────────────────
    if args.exp:
        experiments = [e for e in experiments if e["name"] == args.exp]
    if args.group:
        experiments = [e for e in experiments if e.get("group") == args.group]
    experiments = [e for e in experiments if e.get("enabled", True)]

    if not experiments:
        print("No enabled experiments matched the filters.")
        sys.exit(0)

    all_runs = [
        (exp, s)
        for exp in experiments
        for s in ([args.seed] if args.seed else exp.get("seeds", [42]))
    ]

    print(f"Config      : {args.config}")
    print(f"Mode        : {args.mode}")
    print(f"Experiments : {len(experiments)}")
    print(f"Total jobs  : {len(all_runs)}")
    print()

    (PROJ / cfg.get("hpc", {}).get("slurm", {}).get("log_dir", "logs")).mkdir(
        parents=True, exist_ok=True
    )

    submitted = []

    for exp, seed in all_runs:
        fleet_name  = exp["fleet"]
        fleet_def   = cfg["fleets"][fleet_name]
        assignments = assign_batteries(fleet_def, seed)

        script   = build_slurm_script(exp, seed, cfg, assignments)
        exp_name = f"{exp['name']}_s{seed}"
        log_dir  = cfg.get("hpc", {}).get("slurm", {}).get("log_dir", "logs")
        script_path = PROJ / log_dir / f"slurm_{exp_name}.sh"

        if args.mode == "dry-run":
            print("=" * 60)
            print(f"# JOB: {exp_name}")
            print("=" * 60)
            print(script)

        elif args.mode == "slurm":
            script_path.write_text(script)
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"[OK]  {exp_name:<50} → Job {job_id}")
                submitted.append((exp_name, job_id))
            else:
                print(f"[ERR] {exp_name}: {result.stderr.strip()}")

        elif args.mode == "local":
            script_path.write_text(script)
            script_path.chmod(0o755)
            print(f"[Local] Running {exp_name} ...")
            ret = subprocess.run(["bash", str(script_path)], cwd=str(PROJ))
            if ret.returncode != 0:
                print(f"[Local] FAILED: {exp_name}")

        if args.save_scripts and args.mode != "slurm":
            script_path.write_text(script)
            print(f"[Saved] {script_path}")

    if args.mode == "slurm":
        print()
        print(f"{len(submitted)} jobs submitted.")
        print(f"Monitor  : squeue -u $USER")
        print(f"Outputs  : {cfg.get('global', {}).get('output_base', './results_hpc')}/")
        print(f"Logs     : {PROJ}/{cfg.get('hpc', {}).get('slurm', {}).get('log_dir', 'logs')}/")


if __name__ == "__main__":
    main()
