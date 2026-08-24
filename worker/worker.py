"""
worker/worker.py
================
FedLab ZeroMQ Worker (Client Node)

Sockets:
  DEALER  tcp://server:5555  ← → REGISTER + UPDATE to/from server ROUTER
  SUB     tcp://server:5556  ← TRAIN_REQ + EVAL_RESULT + SHUTDOWN from server PUB

Lifecycle:
  1. Connect sockets
  2. Send REGISTER → server
  3. Wait for REGISTERED reply
  4. Subscribe to PUB channel
  5. For each TRAIN_REQ:
     a. Load global weights
     b. Run local training via algorithm
     c. Send UPDATE → server
  6. Exit on SHUTDOWN

Environment variables:
  FEDLAB_CLIENT_ID        FEDLAB_SERVER_HOST
  FEDLAB_ROUTER_PORT      FEDLAB_PUB_PORT
  FEDLAB_ALGORITHM        FEDLAB_DATASET
  FEDLAB_MODEL            FEDLAB_DEVICE_PROFILE
  FEDLAB_DEVICE           FEDLAB_NUM_CLIENTS
  FEDLAB_PARTITION        FEDLAB_ALPHA
  FEDLAB_LR               FEDLAB_LOCAL_EPOCHS
  FEDLAB_BATCH_SIZE       FEDLAB_ALGO_CONFIG (JSON)

Launch:
  FEDLAB_CLIENT_ID=0 FEDLAB_DEVICE_PROFILE=raspberry_pi_4 python worker/worker.py
"""

import os, sys, time, json
from pathlib import Path

import zmq
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.protocol import (
    unpack, pack,
    msg_register, msg_update,
    tensors_to_bytes, bytes_to_tensors,
)
from algorithms.base import get_algorithm, ClientState
from models.registry import get_model
from datasets.registry import get_dataloader
from hardware.profiles import DEVICE_PROFILES

import algorithms  # noqa: F401 — load the central registry


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    "client_id":      int(os.environ.get("FEDLAB_CLIENT_ID", "0")),
    "server_host":    os.environ.get("FEDLAB_SERVER_HOST", "127.0.0.1"),
    "router_port":    int(os.environ.get("FEDLAB_ROUTER_PORT", "5555")),
    "pub_port":       int(os.environ.get("FEDLAB_PUB_PORT", "5556")),
    "algorithm":      os.environ.get("FEDLAB_ALGORITHM", "fedavg"),
    "dataset":        os.environ.get("FEDLAB_DATASET", "cifar10"),
    "model":          os.environ.get("FEDLAB_MODEL", "resnet18"),
    "device_profile": os.environ.get("FEDLAB_DEVICE_PROFILE", "raspberry_pi_4"),
    "device":         os.environ.get("FEDLAB_DEVICE", "cpu"),
    "num_clients":    int(os.environ.get("FEDLAB_NUM_CLIENTS", "10")),
    "partition":      os.environ.get("FEDLAB_PARTITION", "dirichlet"),
    "alpha":          float(os.environ.get("FEDLAB_ALPHA", "0.5")),
    "lr":             float(os.environ.get("FEDLAB_LR", "0.01")),
    "local_epochs":   int(os.environ.get("FEDLAB_LOCAL_EPOCHS", "1")),
    "batch_size":     int(os.environ.get("FEDLAB_BATCH_SIZE", "32")),
    "algo_config":    json.loads(os.environ.get("FEDLAB_ALGO_CONFIG", "{}")),
}

CID = CFG["client_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Worker class
# ─────────────────────────────────────────────────────────────────────────────

class FedLabWorker:

    def __init__(self):
        self.ctx    = zmq.Context()
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.sub    = self.ctx.socket(zmq.SUB)

        # Identity for ROUTER addressing
        self.dealer.identity = f"worker_{CID:03d}".encode()

        host = CFG["server_host"]
        self.dealer.connect(f"tcp://{host}:{CFG['router_port']}")
        self.sub.connect(f"tcp://{host}:{CFG['pub_port']}")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all

        # Poller for SUB + DEALER (for REGISTERED reply)
        self.poller = zmq.Poller()
        self.poller.register(self.dealer, zmq.POLLIN)
        self.poller.register(self.sub,    zmq.POLLIN)

        # Hardware profile
        self.profile = DEVICE_PROFILES.get(CFG["device_profile"])
        if self.profile is None:
            raise RuntimeError(f"Unknown device profile: {CFG['device_profile']}")

        # Dataset
        self.dataloader = get_dataloader(
            dataset_name=CFG["dataset"],
            split="train",
            partition=CFG["partition"],
            client_id=CID,
            num_clients=CFG["num_clients"],
            alpha=CFG["alpha"],
            batch_size=CFG["batch_size"],
            data_root="./data",
        )

        # Model skeleton (weights overwritten each round from server)
        self.model = get_model(CFG["model"], CFG["dataset"])
        self.model.to(CFG["device"])

        # Algorithm
        self.algo = get_algorithm(CFG["algorithm"])

        # Client state (persists across rounds)
        self.client_state = ClientState(
            client_id=CID,
            battery_j=self.profile.battery.initial_energy_j,
        )

        print(f"[Worker {CID}] Profile  : {self.profile.name}")
        print(f"[Worker {CID}] Dataset  : {CFG['dataset']} "
              f"({len(self.dataloader.dataset)} samples)")
        print(f"[Worker {CID}] Battery  : {self.client_state.battery_j:.0f}J")
        print(f"[Worker {CID}] Algorithm: {CFG['algorithm']}")

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self):
        """Send REGISTER and wait for REGISTERED reply."""
        reg_msg = msg_register(
            client_id=CID,
            profile_name=CFG["device_profile"],
            dataset_size=len(self.dataloader.dataset),
            model_name=CFG["model"],
        )

        # DEALER sends: [empty, payload]  (ROUTER sees [identity, empty, payload])
        print(f"[Worker {CID}] Registering with server...")
        for attempt in range(20):
            self.dealer.send_multipart([b"", reg_msg])

            # Wait for REGISTERED reply (up to 5s per attempt)
            ready = dict(self.poller.poll(timeout=5000))
            if self.dealer in ready:
                frames = self.dealer.recv_multipart()
                # frames = [empty, payload]  (ROUTER strips identity)
                payload = frames[-1]
                msg = unpack(payload)
                if msg["type"] == "REGISTERED" and msg["client_id"] == CID:
                    self.client_state.battery_j = msg["initial_battery_j"]
                    print(f"[Worker {CID}] ✓ Registered — "
                          f"battery={msg['initial_battery_j']:.0f}J")
                    return

            print(f"[Worker {CID}] Retry {attempt+1}/20...")
            time.sleep(1.0)

        raise RuntimeError(f"[Worker {CID}] Failed to register after 20 attempts")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.register()
        print(f"[Worker {CID}] Listening for TRAIN_REQ...\n")

        while True:
            # Poll both SUB (broadcast) and DEALER (direct messages)
            ready = dict(self.poller.poll(timeout=30000))

            if not ready:
                print(f"[Worker {CID}] No message in 30s — still waiting...")
                continue

            # Check SUB (PUB broadcast from server)
            if self.sub in ready:
                raw = self.sub.recv()
                msg = unpack(raw)

                if msg["type"] == "TRAIN_REQ":
                    self._handle_train_req(msg)

                elif msg["type"] == "EVAL_RESULT":
                    m = msg["metrics"]
                    print(f"[Worker {CID}] Round {m.get('round_num',0)} result — "
                          f"Acc={m.get('test_accuracy',0)*100:.2f}% | "
                          f"Loss={m.get('test_loss',0):.4f}")

                elif msg["type"] == "SHUTDOWN":
                    print(f"[Worker {CID}] SHUTDOWN received. Exiting.")
                    break

            # Check DEALER (direct messages, e.g. late REGISTERED)
            if self.dealer in ready:
                frames = self.dealer.recv_multipart()
                payload = frames[-1]
                msg = unpack(payload)
                print(f"[Worker {CID}] Direct message type={msg.get('type')}")

        self._close()

    # ── Training step ─────────────────────────────────────────────────────────

    def _handle_train_req(self, msg: dict):
        round_num = msg["round_num"]
        t0 = time.time()
        print(f"[Worker {CID}] Round {round_num+1} — training...")

        # ── 1. Load global weights ────────────────────────────────────────
        weights = bytes_to_tensors(msg["weights"])
        self.model.load_state_dict(
            {k: v.to(CFG["device"]) for k, v in weights.items()}
        )

        # ── 2. Restore client state from server ───────────────────────────
        state_from_server = msg["client_states"].get(str(CID), {})
        self.client_state.round_num = state_from_server.get("round_num", self.client_state.round_num)
        self.client_state.battery_j = state_from_server.get("battery_j", self.client_state.battery_j)
        self.client_state.custom    = state_from_server.get("custom", {})

        # Restore error buffer if present
        eb = state_from_server.get("error_buffer")
        if eb and isinstance(eb, bytes):
            self.client_state.error_buffer = bytes_to_tensors(eb)

        # ── 3. Build algorithm config ─────────────────────────────────────
        algo_config = self.algo.get_default_config()
        algo_config.update(msg.get("algo_config", {}))
        algo_config.update(CFG["algo_config"])
        algo_config.update({
            "device": CFG["device"],
            "device_profile": self.profile,
        })

        # ── 4. Run local training ─────────────────────────────────────────
        update, metadata = self.algo.client_update(
            model=self.model,
            dataloader=self.dataloader,
            state=self.client_state,
            config=algo_config,
        )

        elapsed = time.time() - t0
        metadata["elapsed_s"] = elapsed

        print(f"[Worker {CID}]   loss={metadata.get('local_loss',0):.4f} | "
              f"β={metadata.get('beta_actual',1):.3f} | "
              f"↑{metadata.get('bytes_sent',0)/1e6:.2f}MB | "
              f"🔋{self.client_state.battery_j:.0f}J | "
              f"⏱{elapsed:.1f}s")

        # ── 5. Serialize updated state for server ─────────────────────────
        cs_dict = {
            "battery_j": self.client_state.battery_j,
            "round_num": self.client_state.round_num,
            "custom":    self.client_state.custom,
        }
        if self.client_state.error_buffer is not None:
            cs_dict["error_buffer"] = tensors_to_bytes(self.client_state.error_buffer)

        # ── 6. Send UPDATE via DEALER → server ROUTER ─────────────────────
        update_msg = msg_update(
            client_id=CID,
            round_num=round_num,
            update_bytes=tensors_to_bytes(update),
            metadata=metadata,
            client_state=cs_dict,
        )
        self.dealer.send_multipart([b"", update_msg])

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _close(self):
        self.dealer.close()
        self.sub.close()
        self.ctx.term()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    worker = FedLabWorker()
    worker.run()
