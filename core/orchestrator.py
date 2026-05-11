"""
core/orchestrator.py
====================
FedLab ZMQ Orchestrator

Launches the server and K worker processes, waits for completion,
and returns the ExperimentResults. No HTTP — pure subprocess management.

Usage:
  from core.orchestrator import Orchestrator
  from core.experiment import ExperimentConfig

  cfg = ExperimentConfig.from_yaml("configs/eceffl_cifar10.yaml")
  Orchestrator(cfg).run()
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional

from core.experiment import ExperimentConfig


class Orchestrator:

    def __init__(self, config: ExperimentConfig, verbose: bool = True):
        self.config  = config
        self.verbose = verbose
        self._server_proc: Optional[subprocess.Popen] = None
        self._worker_procs: list[subprocess.Popen]   = []
        self._root = Path(__file__).parent.parent

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        cfg = self.config
        self._log("=" * 62)
        self._log(f"  FedLab ZMQ — {cfg.name}")
        self._log(f"  Algorithm : {cfg.training.algorithm}")
        self._log(f"  Dataset   : {cfg.data.dataset} "
                  f"({cfg.data.partition}, α={cfg.data.alpha})")
        self._log(f"  Model     : {cfg.model.architecture}")
        self._log(f"  Rounds    : {cfg.training.num_rounds}")
        self._log(f"  Clients   : {cfg.clients.num_clients}")
        self._log(f"  Transport : ZeroMQ DEALER/ROUTER + PUB/SUB")
        self._log("=" * 62 + "\n")

        try:
            self._launch_server()
            time.sleep(1.5)          # give server time to bind sockets
            self._launch_workers()
            self._server_proc.wait() # block until server finishes all rounds
        except KeyboardInterrupt:
            self._log("\n[Orchestrator] Interrupted.")
        finally:
            self._teardown()

        self._log("[Orchestrator] Done.")

    # ── Environment builders ──────────────────────────────────────────────────

    def _base_env(self) -> dict:
        cfg = self.config
        env = os.environ.copy()
        ac  = cfg.training.algo_config
        env.update({
            "FEDLAB_ALGORITHM":    cfg.training.algorithm,
            "FEDLAB_DATASET":      cfg.data.dataset,
            "FEDLAB_MODEL":        cfg.model.architecture,
            "FEDLAB_NUM_ROUNDS":   str(cfg.training.num_rounds),
            "FEDLAB_NUM_CLIENTS":  str(cfg.clients.num_clients),
            "FEDLAB_PARTITION":    cfg.data.partition,
            "FEDLAB_ALPHA":        str(cfg.data.alpha),
            "FEDLAB_DEVICE":       ac.get("device", "cpu"),
            "FEDLAB_OUTPUT_DIR":   cfg.output_dir,
            "FEDLAB_EXP_NAME":     cfg.name,
            "FEDLAB_LR":           str(ac.get("lr", 0.01)),
            "FEDLAB_LOCAL_EPOCHS": str(ac.get("local_epochs", 1)),
            "FEDLAB_BATCH_SIZE":   str(ac.get("batch_size", 32)),
            "FEDLAB_ALGO_CONFIG":  json.dumps(ac),
            "FEDLAB_ROUTER_PORT":  "5555",
            "FEDLAB_PUB_PORT":     "5556",
            "FEDLAB_SERVER_HOST":  "127.0.0.1",
        })
        return env

    def _worker_env(self, client_id: int, profile_name: str) -> dict:
        env = self._base_env()
        env["FEDLAB_CLIENT_ID"]      = str(client_id)
        env["FEDLAB_DEVICE_PROFILE"] = profile_name
        return env

    # ── Process launchers ─────────────────────────────────────────────────────

    def _launch_server(self):
        env = self._base_env()
        cmd = [sys.executable, "server/server.py"]
        log = open(self._root / "server.log", "w")
        self._server_proc = subprocess.Popen(
            cmd, env=env, cwd=str(self._root),
            stdout=log, stderr=subprocess.STDOUT,
        )
        self._log(f"[Orchestrator] Server launched (PID={self._server_proc.pid}) "
                  f"— log: server.log")

    def _launch_workers(self):
        fleet_spec = self.config.clients.fleet
        profiles   = []
        for entry in fleet_spec:
            profiles.extend([entry["type"]] * entry["count"])

        K = self.config.clients.num_clients
        while len(profiles) < K:
            profiles.append("raspberry_pi_4")
        profiles = profiles[:K]

        for cid, profile_name in enumerate(profiles):
            env = self._worker_env(cid, profile_name)
            cmd = [sys.executable, "worker/worker.py"]
            log = open(self._root / f"worker_{cid}.log", "w")
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(self._root),
                stdout=log, stderr=subprocess.STDOUT,
            )
            self._worker_procs.append(proc)
            self._log(f"[Orchestrator] Worker {cid} launched "
                      f"(PID={proc.pid}, {profile_name}) — log: worker_{cid}.log")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _teardown(self):
        for proc in self._worker_procs:
            proc.terminate()
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()

        for proc in self._worker_procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
