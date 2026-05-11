"""
core/experiment.py
==================
Experiment configuration, runner, and result manager.

A complete experiment is defined by a YAML file:
  fedlab run --config configs/eceffl_cifar10.yaml

The framework handles: server/worker launch, round loop,
evaluation, checkpointing, and result export.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
import json
import time
import pathlib
import uuid


# ─────────────────────────────────────────────────────────────────────────────
# Configuration schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    dataset: str                  # "cifar10", "cifar100", "femnist", "tiny_imagenet"
    partition: str                # "iid", "dirichlet", "pathological", "femnist_natural"
    alpha: float = 0.5            # Dirichlet concentration (lower = more non-IID)
    num_shards: int = 200         # for pathological partition
    val_fraction: float = 0.1
    data_root: str = "./data"


@dataclass
class ModelConfig:
    architecture: str             # "mlp", "lenet5", "resnet18", "resnet50", "vit_tiny"
    pretrained: bool = False
    num_classes: Optional[int] = None   # inferred from dataset if None


@dataclass
class ClientConfig:
    num_clients: int              # total clients in the federation
    clients_per_round: int        # clients selected per round (= num_clients for full participation)
    # Device fleet definition: list of {type: ..., count: ..., battery_noise: ...}
    fleet: list = field(default_factory=lambda: [
        {"type": "raspberry_pi_4", "count": 5},
        {"type": "smartphone_midrange", "count": 3},
        {"type": "esp32_s3", "count": 2},
    ])
    battery_noise_std: float = 0.1


@dataclass
class TopologyConfig:
    type: str = "star"            # "star", "hierarchical", "ring", "custom"
    # Hierarchical: define edge aggregators
    num_edge_servers: int = 0
    clients_per_edge: int = 0


@dataclass
class TrainingConfig:
    algorithm: str                # registered algorithm name
    num_rounds: int               # communication rounds T
    algo_config: dict = field(default_factory=dict)   # algorithm-specific params


@dataclass
class EvalConfig:
    eval_every: int = 1           # evaluate global model every N rounds
    metrics: list = field(default_factory=lambda: [
        "accuracy", "loss", "comm_bytes", "energy_j", "jain_index", "battery_j"
    ])
    compare_with: list = field(default_factory=list)  # other algorithms to compare


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    name: str
    description: str = ""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    clients: ClientConfig = field(default_factory=ClientConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./results"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load experiment config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        data = DataConfig(**raw.get("data", {}))
        model = ModelConfig(**raw.get("model", {}))
        clients = ClientConfig(**raw.get("clients", {}))
        topology = TopologyConfig(**raw.get("topology", {}))
        training = TrainingConfig(**raw.get("training", {}))
        eval_cfg = EvalConfig(**raw.get("eval", {}))

        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            seed=raw.get("seed", 42),
            data=data, model=model, clients=clients,
            topology=topology, training=training, eval=eval_cfg,
            output_dir=raw.get("output_dir", "./results"),
        )

    def to_yaml(self, path: str):
        """Save config to YAML."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)


# ─────────────────────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round_num: int
    test_accuracy: float
    test_loss: float
    train_loss: float
    total_bytes: float
    total_energy_j: float
    avg_battery_j: float
    jain_index: float
    participation_rate: float
    algo_metrics: dict = field(default_factory=dict)   # algorithm-specific extras
    timestamp: float = field(default_factory=time.time)


class ExperimentResults:
    """Container for all results from an experiment run."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_id = str(uuid.uuid4())[:8]
        self.rounds: list[RoundResult] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def add_round(self, result: RoundResult):
        self.rounds.append(result)

    def finalize(self):
        self.end_time = time.time()

    def save(self, output_dir: Optional[str] = None):
        """Save results to JSON + generate summary."""
        import dataclasses
        base = pathlib.Path(output_dir or self.config.output_dir)
        exp_dir = base / f"{self.config.name}_{self.exp_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Round-by-round metrics
        records = [dataclasses.asdict(r) for r in self.rounds]
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump({
                "experiment": dataclasses.asdict(self.config),
                "exp_id": self.exp_id,
                "duration_s": (self.end_time or time.time()) - self.start_time,
                "rounds": records,
            }, f, indent=2)

        # Summary statistics
        if self.rounds:
            best_acc = max(r.test_accuracy for r in self.rounds)
            final = self.rounds[-1]
            summary = {
                "algorithm": self.config.training.algorithm,
                "dataset": self.config.data.dataset,
                "partition": self.config.data.partition,
                "num_rounds": len(self.rounds),
                "best_accuracy": best_acc,
                "final_accuracy": final.test_accuracy,
                "total_bytes_gb": sum(r.total_bytes for r in self.rounds) / 1e9,
                "total_energy_j": sum(r.total_energy_j for r in self.rounds),
                "final_jain_index": final.jain_index,
                "final_avg_battery_j": final.avg_battery_j,
            }
            with open(exp_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        print(f"Results saved to: {exp_dir}")
        return exp_dir

    def to_dataframe(self):
        """Convert round results to pandas DataFrame for analysis."""
        import pandas as pd
        import dataclasses
        return pd.DataFrame([dataclasses.asdict(r) for r in self.rounds])
