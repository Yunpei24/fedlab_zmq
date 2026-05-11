"""
server/server.py
================
FedLab ZeroMQ Master Server

Sockets:
  ROUTER tcp://*:5555  ← REGISTER + UPDATE from workers
  PUB    tcp://*:5556  → TRAIN_REQ + EVAL_RESULT + SHUTDOWN to workers

Round loop:
  1. Wait until K workers have registered
  2. For T rounds:
     a. Broadcast TRAIN_REQ via PUB (all workers receive the global model)
     b. Collect K UPDATE messages via ROUTER (with timeout)
     c. Aggregate → new global weights
     d. Evaluate on test set
     e. Broadcast EVAL_RESULT via PUB
     f. Log + checkpoint
  3. Broadcast SHUTDOWN

Environment variables:
  FEDLAB_ALGORITHM   FEDLAB_DATASET    FEDLAB_MODEL
  FEDLAB_NUM_ROUNDS  FEDLAB_NUM_CLIENTS FEDLAB_PARTITION
  FEDLAB_ALPHA       FEDLAB_DEVICE     FEDLAB_OUTPUT_DIR
  FEDLAB_EXP_NAME    FEDLAB_LR         FEDLAB_LOCAL_EPOCHS
  FEDLAB_BATCH_SIZE  FEDLAB_ROUTER_PORT FEDLAB_PUB_PORT
  FEDLAB_ALGO_CONFIG (JSON)

Launch:
  python server/server.py
"""

import os, sys, time, json
from pathlib import Path
from collections import OrderedDict

import zmq
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.protocol import (
    unpack, pack,
    msg_registered, msg_train_req, msg_eval_result, msg_shutdown,
    tensors_to_bytes, bytes_to_tensors,
)
from server.aggregators import fedavg_aggregate
from models.registry import get_model
from datasets.registry import get_dataloader
from algorithms.base import get_algorithm, ClientState

import algorithms.fedavg    # noqa — trigger @register_algorithm
import algorithms.eceffl    # noqa
import algorithms.leanfed   # noqa
import algorithms.fedbacys  # noqa
import algorithms.vaishnav  # noqa
import algorithms.fedsparq  # noqa
import algorithms.fedprox   # noqa
import algorithms.scaffold  # noqa


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "algorithm":    os.environ.get("FEDLAB_ALGORITHM", "fedavg"),
    "dataset":      os.environ.get("FEDLAB_DATASET", "cifar10"),
    "model":        os.environ.get("FEDLAB_MODEL", "resnet18"),
    "num_rounds":   int(os.environ.get("FEDLAB_NUM_ROUNDS", "100")),
    "num_clients":  int(os.environ.get("FEDLAB_NUM_CLIENTS", "10")),
    "partition":    os.environ.get("FEDLAB_PARTITION", "dirichlet"),
    "alpha":        float(os.environ.get("FEDLAB_ALPHA", "0.5")),
    "device":       os.environ.get("FEDLAB_DEVICE", "cpu"),
    "output_dir":   os.environ.get("FEDLAB_OUTPUT_DIR", "./results"),
    "exp_name":     os.environ.get("FEDLAB_EXP_NAME", "experiment"),
    "lr":           float(os.environ.get("FEDLAB_LR", "0.01")),
    "local_epochs": int(os.environ.get("FEDLAB_LOCAL_EPOCHS", "1")),
    "batch_size":   int(os.environ.get("FEDLAB_BATCH_SIZE", "32")),
    "router_port":  int(os.environ.get("FEDLAB_ROUTER_PORT", "5555")),
    "pub_port":     int(os.environ.get("FEDLAB_PUB_PORT", "5556")),
    "algo_config":  json.loads(os.environ.get("FEDLAB_ALGO_CONFIG", "{}")),
    "round_timeout_s": int(os.environ.get("FEDLAB_ROUND_TIMEOUT", "900")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Server class
# ─────────────────────────────────────────────────────────────────────────────

class FedLabServer:

    def __init__(self):
        self.ctx = zmq.Context()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.pub    = self.ctx.socket(zmq.PUB)

        self.router.bind(f"tcp://*:{CFG['router_port']}")
        self.pub.bind(f"tcp://*:{CFG['pub_port']}")

        # Poller for non-blocking receives on ROUTER
        self.poller = zmq.Poller()
        self.poller.register(self.router, zmq.POLLIN)

        # State
        self.global_model  = None
        self.test_loader   = None
        self.registered    = {}    # client_id → {profile_name, dataset_size, battery_j}
        self.client_states = {}    # client_id → {battery_j, round_num, ...}
        self.metrics_log   = []
        self.algo_state    = {}    # algorithm-level server state (e.g. SCAFFOLD c_global)

        # Load algorithm instance for server_aggregate()
        self.algo = get_algorithm(CFG["algorithm"])

        self._init_model_and_data()

        print(f"[Server] ROUTER bound on tcp://*:{CFG['router_port']}")
        print(f"[Server] PUB    bound on tcp://*:{CFG['pub_port']}")
        print(f"[Server] Algorithm : {CFG['algorithm']}")
        print(f"[Server] Dataset   : {CFG['dataset']}")
        print(f"[Server] Rounds    : {CFG['num_rounds']}")
        print(f"[Server] Clients   : {CFG['num_clients']}")

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_model_and_data(self):
        self.global_model = get_model(CFG["model"], CFG["dataset"])
        self.global_model.to(CFG["device"])

        self.test_loader = get_dataloader(
            dataset_name=CFG["dataset"],
            split="test",
            partition="iid",
            client_id=None,
            num_clients=1,
            batch_size=256,
            data_root="./data",
        )

    # ── Registration phase ────────────────────────────────────────────────────

    def wait_for_registrations(self):
        """Block until all K workers have sent a REGISTER message."""
        from hardware.profiles import DEVICE_PROFILES

        print(f"\n[Server] Waiting for {CFG['num_clients']} workers...")

        while len(self.registered) < CFG["num_clients"]:
            # ROUTER receives: [identity, empty, payload]
            frames = self.router.recv_multipart()
            identity, _, payload = frames[0], frames[1], frames[2]
            msg = unpack(payload)

            if msg["type"] != "REGISTER":
                print(f"[Server] Unexpected message type: {msg['type']}")
                continue

            cid          = msg["client_id"]
            profile_name = msg["profile_name"]
            profile      = DEVICE_PROFILES.get(profile_name)
            battery_j    = profile.battery.initial_energy_j if profile else 100000.0

            self.registered[cid] = {
                "identity":    identity,
                "profile_name": profile_name,
                "dataset_size": msg["dataset_size"],
                "battery_j":   battery_j,
            }
            self.client_states[cid] = {
                "battery_j": battery_j,
                "round_num": 0,
                "error_buffer": None,
                "momentum_buffer": None,
                "custom": {},
            }

            # Reply directly via ROUTER (identity-addressed)
            reply = msg_registered(cid, battery_j)
            self.router.send_multipart([identity, b"", reply])

            print(f"[Server] ✓ Worker {cid} registered "
                  f"({profile_name}, {msg['dataset_size']} samples, "
                  f"🔋{battery_j:.0f}J)  "
                  f"[{len(self.registered)}/{CFG['num_clients']}]")

        print(f"[Server] All {CFG['num_clients']} workers ready.\n")

    # ── Training loop ─────────────────────────────────────────────────────────

    def run(self):
        self.wait_for_registrations()

        algo_config = {**CFG["algo_config"], **{
            "lr": CFG["lr"],
            "local_epochs": CFG["local_epochs"],
            "batch_size": CFG["batch_size"],
            "device": CFG["device"],
            "num_rounds": CFG["num_rounds"],  # needed for LR cosine decay schedule
        }}

        for t in range(CFG["num_rounds"]):
            t0 = time.time()
            print(f"[Server] ── Round {t+1}/{CFG['num_rounds']} ──")

            # ── a. Serialize global model ──────────────────────────────────
            weights_bytes = tensors_to_bytes(self.global_model.state_dict())

            # ── b. Broadcast TRAIN_REQ to all workers via PUB ──────────────
            # Small delay to ensure SUB sockets receive (PUB is fire-and-forget)
            time.sleep(0.05)
            train_msg = msg_train_req(
                round_num=t,
                weights_bytes=weights_bytes,
                client_states={
                    str(cid): state
                    for cid, state in self.client_states.items()
                },
                algo_name=CFG["algorithm"],
                algo_config=algo_config,
                hyperparams={
                    "lr": CFG["lr"],
                    "local_epochs": CFG["local_epochs"],
                    "batch_size": CFG["batch_size"],
                },
            )
            self.pub.send(train_msg)
            print(f"[Server]   → TRAIN_REQ broadcast ({len(weights_bytes)/1e6:.1f}MB)")

            # ── c. Collect K updates via ROUTER ───────────────────────────
            updates = self._collect_updates(t, timeout_s=CFG["round_timeout_s"])
            if len(updates) == 0:
                print(f"[Server] ✗ No updates received — skipping round {t+1}")
                continue

            # ── d. Aggregate ──────────────────────────────────────────────
            # Build client_updates tuples for algo.server_aggregate()
            # Format: list of (update_dict, metadata_dict, ClientState)
            algo_client_tuples = []
            for upd in updates:
                update_tensors = bytes_to_tensors(upd["update"])
                cid = upd["client_id"]
                cs  = upd["client_state"]

                # Update server-side client state
                self.client_states[cid]["battery_j"] = cs.get("battery_j",
                    self.client_states[cid]["battery_j"])
                self.client_states[cid]["round_num"] = cs.get("round_num",
                    self.client_states[cid]["round_num"])
                self.client_states[cid]["custom"] = cs.get("custom", {})

                # Restore error buffer if present
                eb_bytes = cs.get("error_buffer")
                if eb_bytes and isinstance(eb_bytes, bytes):
                    self.client_states[cid]["error_buffer"] = eb_bytes

                # Reconstruct ClientState object for algo.server_aggregate()
                c_state = ClientState(
                    client_id=cid,
                    round_num=cs.get("round_num", 0),
                    battery_j=cs.get("battery_j", 0.0),
                    custom=cs.get("custom", {}),
                )

                algo_client_tuples.append((update_tensors, upd["metadata"], c_state))

            # Build algo config including server-level state (e.g. SCAFFOLD c_global)
            agg_config = {**algo_config, **self.algo_state}

            # Use algorithm's server_aggregate() — fixes the critical bug
            # where fedavg_aggregate() was always used regardless of algorithm
            agg_result = self.algo.server_aggregate(
                global_model=self.global_model,
                client_updates=algo_client_tuples,
                round_num=t,
                config=agg_config,
            )
            self.global_model.load_state_dict(
                {k: v.to(CFG["device"]) for k, v in agg_result.new_weights.items()}
            )

            # Update server-level algo state (e.g. SCAFFOLD c_global)
            c_global_update = agg_result.metrics.pop("_scaffold_c_global_update", None)
            if c_global_update:
                self.algo_state["scaffold_c_global"] = c_global_update

            # ── e. Evaluate ───────────────────────────────────────────────
            acc, loss = self._evaluate()

            # ── f. Compute metrics (using algo's aggregate result) ────────
            K = len(updates)
            elapsed = time.time() - t0

            # Use metrics from algo.server_aggregate() — algorithm-specific
            agg_m = agg_result.metrics
            metrics = {
                "round_num":          t + 1,
                "test_accuracy":      acc,
                "test_loss":          loss,
                "train_loss":         agg_m.get("avg_local_loss", 0.0),
                "total_bytes":        agg_m.get("total_bytes_sent", 0),
                "total_energy_j":     agg_m.get("total_energy_j", 0.0),
                "avg_battery_j":      agg_m.get("avg_battery_j", 0.0),
                "avg_beta":           agg_m.get("avg_beta", 1.0),
                "jain_index":         agg_m.get("jain_index", 1.0),
                "participation_rate": agg_m.get("participation_rate",
                                                K / CFG["num_clients"]),
                "num_clients":        K,
                "elapsed_s":          elapsed,
            }
            self.metrics_log.append(metrics)

            print(f"[Server]   Acc={acc*100:.2f}% | Loss={loss:.4f} | "
                  f"Bytes={metrics['total_bytes']/1e6:.1f}MB | "
                  f"E={metrics['total_energy_j']:.0f}J | "
                  f"J={metrics['jain_index']:.3f} | ⏱{elapsed:.1f}s")

            # ── g. Broadcast EVAL_RESULT ──────────────────────────────────
            self.pub.send(msg_eval_result(t, metrics))

        # ── Training done ─────────────────────────────────────────────────
        print("\n[Server] Training complete. Sending SHUTDOWN...")
        self.pub.send(msg_shutdown())
        time.sleep(0.5)

        self._save_results()
        self._close()

    def _collect_updates(self, round_num: int, timeout_s: int) -> list:
        """
        Collect UPDATE messages from all K workers via ROUTER.
        Returns list of message dicts. Times out after timeout_s seconds.
        """
        updates = {}
        deadline = time.time() + timeout_s

        while len(updates) < CFG["num_clients"] and time.time() < deadline:
            remaining_ms = int((deadline - time.time()) * 1000)
            if remaining_ms <= 0:
                break

            ready = dict(self.poller.poll(timeout=min(remaining_ms, 5000)))
            if self.router not in ready:
                continue

            frames = self.router.recv_multipart()
            _, _, payload = frames[0], frames[1], frames[2]
            msg = unpack(payload)

            if msg["type"] == "UPDATE" and msg["round_num"] == round_num:
                cid = msg["client_id"]
                updates[cid] = msg
                print(f"[Server]   ← UPDATE from worker {cid} "
                      f"({len(updates)}/{CFG['num_clients']}) "
                      f"loss={msg['metadata'].get('local_loss',0):.4f} "
                      f"β={msg['metadata'].get('beta_actual',1):.3f}")

            elif msg["type"] == "REGISTER":
                # Late registration — handle gracefully
                cid = msg["client_id"]
                if cid not in self.registered:
                    from hardware.profiles import DEVICE_PROFILES
                    profile = DEVICE_PROFILES.get(msg["profile_name"])
                    battery_j = profile.battery.initial_energy_j if profile else 100000.0
                    self.registered[cid] = {
                        "identity": frames[0],
                        "profile_name": msg["profile_name"],
                        "dataset_size": msg["dataset_size"],
                        "battery_j": battery_j,
                    }
                    self.client_states[cid] = {"battery_j": battery_j, "round_num": 0,
                                                "error_buffer": None, "custom": {}}
                    reply = msg_registered(cid, battery_j)
                    self.router.send_multipart([frames[0], b"", reply])

        if len(updates) < CFG["num_clients"]:
            print(f"[Server] ⚠ Timeout: received {len(updates)}/{CFG['num_clients']} updates")

        return list(updates.values())

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate(self) -> tuple[float, float]:
        model = self.global_model
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct, total, tloss = 0, 0, 0.0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(CFG["device"]), y.to(CFG["device"])
                out = model(x)
                tloss   += criterion(out, y).item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total   += y.size(0)

        model.train()
        return correct / total, tloss / total

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_results(self):
        out_dir = Path(CFG["output_dir"]) / CFG["exp_name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "metrics.json"

        # Per-client registration info (dataset_size, profile, initial battery)
        client_info = {
            str(cid): {
                "profile_name": v["profile_name"],
                "dataset_size": v["dataset_size"],
                "initial_battery_j": v["battery_j"],
            }
            for cid, v in self.registered.items()
        }

        with open(path, "w") as f:
            json.dump({
                "config": CFG,
                "clients": client_info,
                "rounds": self.metrics_log,
            }, f, indent=2)
        print(f"[Server] Results saved → {path}")

        # Save final model checkpoint
        torch.save(
            self.global_model.state_dict(),
            out_dir / "model_final.pt",
        )

    def _close(self):
        self.router.close()
        self.pub.close()
        self.ctx.term()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = FedLabServer()
    server.run()
