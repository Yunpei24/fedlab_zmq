"""
core/protocol.py
================
FedLab ZeroMQ Message Protocol

All messages are dicts serialized with msgpack (fast binary).
Tensors are serialized with torch.save → bytes → embedded in the dict.

Message types (field "type"):
  REGISTER     Worker → Server : announce presence
  REGISTERED   Server → Worker : confirm + send initial battery
  TRAIN_REQ    Server → Worker : start local training (via PUB)
  UPDATE       Worker → Server : compressed gradient + metadata
  EVAL_RESULT  Server → Worker : round metrics (via PUB)
  SHUTDOWN     Server → Worker : stop all workers (via PUB)

ZMQ Sockets:
  Server ROUTER  tcp://*:5555   ← receives REGISTER + UPDATE from workers
  Server PUB     tcp://*:5556   → broadcasts TRAIN_REQ + EVAL_RESULT + SHUTDOWN
  Worker DEALER  tcp://host:5555 → sends REGISTER + UPDATE
  Worker SUB     tcp://host:5556 ← receives TRAIN_REQ + EVAL_RESULT + SHUTDOWN

Worker identity = b"worker_{client_id:03d}"  (e.g. b"worker_002")
"""

import io
import struct
import time
import torch
import msgpack


# ─────────────────────────────────────────────────────────────────────────────
# Tensor serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def tensors_to_bytes(state_dict: dict) -> bytes:
    """Serialize a dict of tensors to bytes (torch.save format)."""
    buf = io.BytesIO()
    cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
    torch.save(cpu_sd, buf)
    return buf.getvalue()


def bytes_to_tensors(data: bytes) -> dict:
    """Deserialize bytes back to a dict of CPU tensors."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# Message pack / unpack
# ─────────────────────────────────────────────────────────────────────────────

def pack(msg: dict) -> bytes:
    """Serialize message dict to msgpack bytes."""
    return msgpack.packb(msg, use_bin_type=True)


def unpack(data: bytes) -> dict:
    """Deserialize msgpack bytes to message dict."""
    return msgpack.unpackb(data, raw=False)


# ─────────────────────────────────────────────────────────────────────────────
# Message constructors
# ─────────────────────────────────────────────────────────────────────────────

def msg_register(
    client_id: int,
    profile_name: str,
    dataset_size: int,
    model_name: str,
) -> bytes:
    return pack({
        "type": "REGISTER",
        "client_id": client_id,
        "profile_name": profile_name,
        "dataset_size": dataset_size,
        "model_name": model_name,
        "timestamp": time.time(),
    })


def msg_registered(
    client_id: int,
    initial_battery_j: float,
    status: str = "ok",
) -> bytes:
    return pack({
        "type": "REGISTERED",
        "client_id": client_id,
        "initial_battery_j": initial_battery_j,
        "status": status,
        "timestamp": time.time(),
    })


def msg_train_req(
    round_num: int,
    weights_bytes: bytes,
    client_states: dict,   # {client_id: {battery_j, round_num, ...}}
    algo_name: str,
    algo_config: dict,
    hyperparams: dict,
) -> bytes:
    """Broadcast from server to all workers via PUB."""
    return pack({
        "type": "TRAIN_REQ",
        "round_num": round_num,
        "weights": weights_bytes,      # raw bytes embedded in msgpack
        "client_states": client_states,
        "algo_name": algo_name,
        "algo_config": algo_config,
        "hyperparams": hyperparams,
        "timestamp": time.time(),
    })


def msg_update(
    client_id: int,
    round_num: int,
    update_bytes: bytes,
    metadata: dict,
    client_state: dict,
) -> bytes:
    """Sent from worker to server via DEALER."""
    return pack({
        "type": "UPDATE",
        "client_id": client_id,
        "round_num": round_num,
        "update": update_bytes,
        "metadata": metadata,
        "client_state": client_state,
        "timestamp": time.time(),
    })


def msg_eval_result(
    round_num: int,
    metrics: dict,
) -> bytes:
    """Broadcast from server to all workers via PUB (after aggregation)."""
    return pack({
        "type": "EVAL_RESULT",
        "round_num": round_num,
        "metrics": metrics,
        "timestamp": time.time(),
    })


def msg_shutdown() -> bytes:
    return pack({"type": "SHUTDOWN", "timestamp": time.time()})
