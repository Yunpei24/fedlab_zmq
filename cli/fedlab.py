#!/usr/bin/env python3
"""
cli/fedlab.py — FedLab ZMQ Command Line Interface

  fedlab run    --config configs/eceffl_cifar10.yaml [--device mps]
  fedlab list   algorithms | devices | datasets | models
  fedlab compare --results results/exp1 results/exp2 [--metric test_accuracy]
  fedlab dashboard
  fedlab new-algo --name my_algo
  fedlab monitor          (live tail of server.log)
"""

import argparse, sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    from core.experiment import ExperimentConfig
    from core.orchestrator import Orchestrator

    if not Path(args.config).exists():
        print(f"[ERROR] Config not found: {args.config}"); sys.exit(1)

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.device:
        cfg.training.algo_config["device"] = args.device
    if args.rounds:
        cfg.training.num_rounds = args.rounds
    if args.output:
        cfg.output_dir = args.output

    Orchestrator(cfg).run()


def cmd_list(args):
    topic = args.topic

    if topic == "algorithms":
        import algorithms.fedavg, algorithms.eceffl, algorithms.leanfed
        import algorithms.fedbacys, algorithms.vaishnav, algorithms.fedsparq  # noqa
        from algorithms.base import list_algorithms
        print(f"\n{'Name':<20} {'Class':<30} Description")
        print("─" * 80)
        for a in list_algorithms():
            print(f"  {a['name']:<18} {a['class']:<30} {a['description'][:45]}")

    elif topic == "devices":
        from hardware.profiles import list_profiles
        print(); list_profiles()

    elif topic == "datasets":
        from datasets.registry import NUM_CLASSES, INPUT_SHAPE
        print(f"\n{'Name':<20} {'Classes':>8}   Input Shape")
        print("─" * 48)
        for name, nc in NUM_CLASSES.items():
            print(f"  {name:<18} {nc:>8}   {str(INPUT_SHAPE.get(name,'?')):<20}")

    elif topic == "models":
        from models.registry import list_models
        print(f"\n{'Name':<20} {'Class':<30}")
        print("─" * 52)
        for m in list_models():
            print(f"  {m['name']:<18} {m['class']:<30}")

    elif topic == "ports":
        print("\nFedLab ZMQ default ports:")
        print("  ROUTER  tcp://*:5555  (worker DEALER → server)")
        print("  PUB     tcp://*:5556  (server → worker SUB)")
    else:
        print(f"Unknown topic: {topic}")


def cmd_compare(args):
    import json, pandas as pd

    metric = args.metric or "test_accuracy"
    data   = {}

    for path in args.results:
        mf = Path(path) / "metrics.json"
        if not mf.exists():
            print(f"[WARN] No metrics.json in {path}"); continue
        with open(mf) as f:
            d = json.load(f)
        algo   = d.get("config", {}).get("algorithm", Path(path).name)
        values = [r.get(metric, 0) for r in d.get("rounds", [])]
        data[algo] = values

    if not data:
        print("No valid results found."); return

    df = pd.DataFrame(data)
    df.index = df.index + 1
    df.index.name = "round"
    print(f"\nLast 10 rounds — {metric}:")
    print(df.tail(10).to_string())
    print("\nBest:")
    for col in df.columns:
        best = df[col].max() if "accuracy" in metric else df[col].min()
        print(f"  {col}: {best:.4f}")


def cmd_dashboard(args):
    import subprocess
    dash = Path(__file__).parent.parent / "dashboard" / "app.py"
    port = args.port or 8501
    print(f"[FedLab] Dashboard → http://localhost:{port}")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(dash), "--server.port", str(port)])


def cmd_monitor(args):
    """Live tail of server.log (like `tail -f`)."""
    import time
    log = Path(args.log or "server.log")
    if not log.exists():
        print(f"[ERROR] Log not found: {log}"); return
    print(f"[FedLab] Monitoring {log} (Ctrl+C to stop)\n")
    with open(log) as f:
        f.seek(0, 2)  # go to end
        while True:
            line = f.readline()
            if line:
                print(line, end="")
            else:
                time.sleep(0.2)


def cmd_new_algo(args):
    name = args.name
    out  = args.output or f"algorithms/{name}.py"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    template = f'''"""
algorithms/{name}.py
{("=" * (len(name) + 14))}
{name}: [Your description — Author, Year]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm


@register_algorithm("{name}")
class {name.replace("_"," ").title().replace(" ","")}(FLAlgorithm):
    """[One-paragraph description of your algorithm.]"""
    name        = "{name}"
    description = "[One-line description]"

    def client_update(self, model, dataloader, state, config):
        device       = config.get("device", "cpu")
        lr           = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)

        w_before = OrderedDict({{k: v.clone() for k, v in model.state_dict().items()}})

        model.train(); model.to(device)
        opt  = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        tloss, steps = 0.0, 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward(); opt.step()
                tloss += loss.item(); steps += 1

        delta = OrderedDict({{
            k: (w_before[k] - model.state_dict()[k].cpu()).float()
            for k in w_before
        }})

        # TODO: apply your compression / modification to delta here

        profile = config.get("device_profile")
        up_bytes   = self.count_bytes(delta, sparse=False)
        down_bytes = self.count_bytes(w_before, sparse=False)
        if profile:
            flops     = profile.flops_for_model(
                sum(p.numel() for p in model.parameters()),
                dataloader.batch_size, local_epochs, len(dataloader.dataset))
            energy_j  = profile.round_energy_j(flops, up_bytes, down_bytes)
        else:
            energy_j = 1.0

        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        return dict(delta), {{
            "client_id": state.client_id, "round_num": state.round_num,
            "beta_actual": 1.0, "battery_j_remaining": state.battery_j,
            "energy_j_consumed": energy_j, "bytes_sent": up_bytes,
            "bytes_received": down_bytes,
            "local_loss": tloss / max(steps, 1), "compression_ratio": 1.0,
        }}

    def server_aggregate(self, global_model, client_updates, round_num, config):
        from server.aggregators import uniform_aggregate
        new_w = uniform_aggregate(
            global_model.state_dict(),
            [(u, 1.0, m) for u, m, _ in client_updates]
        )
        K = len(client_updates)
        return AggregateResult(
            new_weights=new_w,
            metrics={{
                "round": round_num,
                "total_bytes_sent": sum(m.get("bytes_sent",0) for _,m,_ in client_updates),
                "total_energy_j": sum(m.get("energy_j_consumed",0) for _,m,_ in client_updates),
                "avg_battery_j": sum(s.battery_j for _,_,s in client_updates) / K,
                "avg_local_loss": sum(m.get("local_loss",0) for _,m,_ in client_updates) / K,
                "participation_rate": 1.0, "jain_index": 1.0, "num_clients": K,
            }}
        )

    def get_default_config(self):
        return {{
            "lr": 0.01, "local_epochs": 1, "batch_size": 32,
            "device": "cpu", "device_profile": None,
            # TODO: add your hyperparameters here
        }}
'''
    with open(out, "w") as f:
        f.write(template)
    print(f"[FedLab] Template created: {out}")
    print(f"  Add to algorithms/__init__.py: import algorithms.{name}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="fedlab",
        description="FedLab ZMQ — Federated Learning Research Framework (UM6P)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  fedlab run --config configs/eceffl_cifar10.yaml
  fedlab run --config configs/benchmark_suite.yaml --device mps
  fedlab list algorithms
  fedlab list devices
  fedlab monitor
  fedlab compare --results results/exp1 results/exp2
  fedlab dashboard
  fedlab new-algo --name my_algo
""")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run",       help="Run an FL experiment")
    pr.add_argument("--config",  required=True)
    pr.add_argument("--device",  help="cpu | mps | cuda")
    pr.add_argument("--rounds",  type=int)
    pr.add_argument("--output")

    pl = sub.add_parser("list",      help="List components")
    pl.add_argument("topic", choices=["algorithms","devices","datasets","models","ports"])

    pc = sub.add_parser("compare",   help="Compare experiment results")
    pc.add_argument("--results", nargs="+", required=True)
    pc.add_argument("--metric",  default="test_accuracy")

    pd = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    pd.add_argument("--port", type=int, default=8501)

    pm = sub.add_parser("monitor",   help="Live tail of server log")
    pm.add_argument("--log", default="server.log")

    pn = sub.add_parser("new-algo",  help="Generate algorithm template")
    pn.add_argument("--name",   required=True)
    pn.add_argument("--output")

    args = p.parse_args()
    {
        "run":       cmd_run,
        "list":      cmd_list,
        "compare":   cmd_compare,
        "dashboard": cmd_dashboard,
        "monitor":   cmd_monitor,
        "new-algo":  cmd_new_algo,
    }[args.command](args)


if __name__ == "__main__":
    main()
