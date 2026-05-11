"""
algorithms/base.py
==================
Abstract base class that every FL algorithm must implement.

To add a new algorithm to FedLab:
  1. Create algorithms/my_algo.py
  2. Subclass FLAlgorithm
  3. Implement the 3 abstract methods
  4. Register with @register_algorithm("my_algo")

The framework handles everything else: scheduling, energy tracking,
serialization, logging, visualization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Data transfer objects (shared between server and workers via FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClientState:
    """Mutable state held by each worker across rounds."""
    client_id: int
    round_num: int = 0
    battery_j: float = 0.0          # current battery in Joules
    error_buffer: Optional[dict] = None   # for EF-based algorithms
    momentum_buffer: Optional[dict] = None
    local_steps: int = 0
    custom: dict = field(default_factory=dict)  # algo-specific state


@dataclass
class TrainRequest:
    """Server → Worker: what to do this round."""
    round_num: int
    global_weights: bytes        # serialized model state_dict
    client_state: ClientState
    hyperparams: dict            # lr, epochs, batch_size, etc.
    algo_config: dict            # algorithm-specific params (beta_min, etc.)


@dataclass
class TrainResponse:
    """Worker → Server: result of local training."""
    client_id: int
    round_num: int
    update: bytes                # serialized compressed gradient/weights
    metadata: dict               # bytes_sent, energy_j, beta_actual, etc.
    client_state: ClientState    # updated state for next round


@dataclass
class AggregateResult:
    """Result of server-side aggregation."""
    new_weights: dict[str, torch.Tensor]
    metrics: dict                # loss, accuracy, comm_bytes, etc.


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm interface
# ─────────────────────────────────────────────────────────────────────────────

class FLAlgorithm(ABC):
    """
    Abstract base class for federated learning algorithms.

    Every algorithm in FedLab implements exactly three methods:
      - client_update(): local training + compression
      - server_aggregate(): global model update
      - get_default_config(): hyperparameter defaults

    The framework handles: scheduling, energy tracking, serialization,
    logging, checkpointing, and visualization.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        Local training step at the client.

        Args:
            model:      Current global model (already loaded with global weights)
            dataloader: Local dataset loader
            state:      Persistent client state (battery, error buffer, etc.)
            config:     Algorithm hyperparameters + device profile

        Returns:
            update:   Dict of tensors to send to server (compressed gradient,
                      model delta, or quantized weights — algo choice)
            metadata: Dict with keys:
                        'bytes_sent'    : float — uplink bytes
                        'energy_j'      : float — energy consumed this round
                        'compression_ratio': float — actual compression ratio
                        'local_loss'    : float — final local training loss
                        [algo-specific keys]

        Note: state is mutated in-place (error buffer, battery, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Server-side aggregation of client updates.

        Args:
            global_model:   Current global model
            client_updates: List of (update_dict, metadata, client_state)
                            from all participating clients
            round_num:      Current round index
            config:         Server-side config

        Returns:
            AggregateResult with new_weights and round metrics
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_config(self) -> dict:
        """
        Return default hyperparameter configuration for this algorithm.

        Keys should be fully documented. Example:
            {
                "lr": 0.01,
                "local_epochs": 1,
                "batch_size": 32,
                "beta_min": 0.01,   # minimum sparsification ratio
                "beta_max": 1.0,    # maximum sparsification ratio
            }
        """
        raise NotImplementedError

    # ── Convenience helpers (not abstract, can be overridden) ─────────────

    def compute_model_size_bytes(self, model: nn.Module) -> int:
        """Size of full model in bytes (FP32)."""
        return sum(p.numel() * 4 for p in model.parameters())

    def serialize_update(self, update: dict[str, torch.Tensor]) -> bytes:
        """Serialize sparse/dense update dict to bytes for HTTP transport."""
        import io, pickle
        buf = io.BytesIO()
        # Convert tensors to CPU before serialization
        cpu_update = {k: v.cpu() for k, v in update.items()}
        torch.save(cpu_update, buf)
        return buf.getvalue()

    def deserialize_update(self, data: bytes) -> dict[str, torch.Tensor]:
        """Deserialize bytes back to tensor dict."""
        import io
        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu")

    def count_bytes(self, update: dict[str, torch.Tensor],
                    sparse: bool = False,
                    bytes_per_value: int = 4) -> int:
        """
        Count actual transmitted bytes for a sparse or dense update.

        Sparse format: (index, value) pairs → 4 bytes index + bytes_per_value
          bytes_per_value=4 → float32 (default)
          bytes_per_value=2 → float16 (quantized uplink)
        Dense format: always float32 (4 bytes per element).
        """
        total = 0
        for tensor in update.values():
            if sparse:
                nnz = tensor.count_nonzero().item()
                total += nnz * (4 + bytes_per_value)  # index + value
            else:
                total += tensor.numel() * 4  # dense always float32
        return total

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm registry
# ─────────────────────────────────────────────────────────────────────────────

_ALGORITHM_REGISTRY: dict[str, type[FLAlgorithm]] = {}


def register_algorithm(name: str):
    """
    Decorator to register an algorithm.

    Usage:
        @register_algorithm("my_algo")
        class MyAlgo(FLAlgorithm):
            ...
    """
    def decorator(cls: type[FLAlgorithm]):
        cls.name = name
        _ALGORITHM_REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm(name: str) -> FLAlgorithm:
    """Instantiate a registered algorithm by name."""
    if name not in _ALGORITHM_REGISTRY:
        available = list(_ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Algorithm '{name}' not found. Available: {available}\n"
            f"To add a new algorithm, see algorithms/base.py"
        )
    return _ALGORITHM_REGISTRY[name]()


def list_algorithms() -> list[dict]:
    """Return info about all registered algorithms."""
    result = []
    for name, cls in _ALGORITHM_REGISTRY.items():
        result.append({
            "name": name,
            "class": cls.__name__,
            "description": cls.description,
            "config": cls().get_default_config(),
        })
    return result
