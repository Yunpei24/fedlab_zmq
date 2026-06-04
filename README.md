# FedLab ZMQ

**Federated Learning Research Framework — ZeroMQ transport**

> Mohammed VI Polytechnic University (UM6P) — College of Computing  
> **Author:** J. Nikiema  
> **Supervisors:** E. Amhoud · H. Elhammouti · I. Kissami

[![Code](https://img.shields.io/badge/GitHub-fedlab__zmq-blue?logo=github)](https://github.com/Yunpei24/fedlab_zmq)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)](https://fedlabzmq-dashboard.streamlit.app)

---

## Overview

FedLab ZMQ is a research framework for federated learning on energy-constrained IoT devices. It implements a realistic battery simulation, heterogeneous device profiles, and a suite of FL algorithms including the proposed **FedPartBE** — a battery-aware partial network update algorithm that adapts which layers each client trains based on its remaining battery.

**Key results (FedPartBE vs FedPart, 30 ESP32-S3, CIFAR-10):**
- +1.1% accuracy on CIFAR-10, +3.9% on CIFAR-100
- −60% FLOPs per round (partial backward pass)
- Jain fairness index = 0.633 (energy drain equity)

---

## Communication Architecture

```
Server  ROUTER tcp://*:5555  ← REGISTER + UPDATE from workers
Server  PUB    tcp://*:5556  → TRAIN_REQ + EVAL_RESULT + SHUTDOWN

Worker  DEALER tcp://host:5555  → REGISTER + UPDATE
Worker  SUB    tcp://host:5556  ← TRAIN_REQ + EVAL_RESULT + SHUTDOWN
```

All messages are serialized with **msgpack**. PyTorch tensors are embedded as `torch.save → bytes` inside the msgpack dict.

**Why ZeroMQ instead of FastAPI?**

| Criterion | FastAPI/uvicorn | ZeroMQ |
|---|---|---|
| Latency | ~5–20 ms (HTTP overhead) | ~0.1–1 ms (direct TCP) |
| Serialization | JSON/base64 | msgpack (binary) |
| Broadcast | Non-native (HTTP loop) | Native PUB/SUB |
| Model throughput | Limited by HTTP framing | Limited by TCP only |
| Multi-machine | HTTP routing | Direct TCP |
| Overhead | REST verbs + headers | Message frames only |

---

## Quick Start

```bash
# 1. Install (exact pinned versions — see Reproducibility below)
pip install -r requirements.txt        # runtime deps
pip install -e .                       # the fedlab-zmq package
#   or, for the full dev/eval toolchain (pytest, black, isort, ruff):
#   pip install -r requirements-dev.txt && pip install -e .
#   or with conda:  conda env create -f environment.yml && conda activate fedlab-zmq

# 2. Sanity-check the install (canonical cost-model regression guardrail)
make test            # == python -m pytest  → 11 passed

# 3. Fast end-to-end smoke run (< 1 min, CPU only) — confirms the pipeline works
make smoke           # == python run_experiment.py --config configs/smoke.yaml

# 4. Run a full experiment
python run_experiment.py --config configs/fedpartbe_survival_wide.yaml --device mps
```

Every run writes, into its `results/<run>/` directory:
`metrics.json` (per-round metrics + summary), `manifest.json` (resolved config,
git commit, seed, package versions, FLOP convention, timestamp), and
`survival.csv` (per-client lifetimes for the survival/Pareto figures).

---

## Reproducibility

This repository is set up for artifact evaluation:

- **Pinned dependencies.** `requirements.txt` pins exact runtime versions
  (`torch==2.11.0` first); `requirements-lock.txt` is a full `pip freeze`
  snapshot; `environment.yml` is the conda equivalent. Tested with
  **Python 3.12.4** on macOS arm64 (Apple Silicon, MPS).
- **Centralized seeding.** `core/seeding.py:seed_everything(seed)` seeds
  python / numpy / torch (CPU+CUDA; MPS via `torch.manual_seed`) and is called
  by the experiment runner. Residual non-determinism (cuDNN, MPS reduction
  order, FlopCounter absolute energy vs. torch version) is documented in that
  module; strict deterministic mode is opt-in and **off by default** so the
  `cost_model="phi"` numbers stay bit-exact.
- **Per-run manifest.** `core/manifest.py` records exactly what produced each
  result (see above).
- **Config-driven.** All experiment parameters (K, SoC range, α, E, rounds, lr,
  seed, `cost_model`, `energy_scale_factor`, …) live in versioned YAML configs
  in `configs/` — one per paper table/figure. No magic constants in code.

### Reproducing the paper

Paper: *"FedPartBE: Battery-Energy Aware Partial Network Training for Federated
Learning"* (Nikiema & Amhoud, 2026). Two execution modes:
**single-process** (`run_experiment.py`, laptop/dev) and **ZMQ distributed**
(`hpc/launch_zmq_hpc.py --group <group>`, HPC/SLURM). The master config
`configs/fedpartbe_survival_wide_cifar10.yaml` defines the experiment *groups*;
the standalone configs below run one experiment each.

| Paper artifact | Config / command | What it produces |
|---|---|---|
| **Smoke** (not a result) | `python run_experiment.py --config configs/smoke.yaml` | < 1 min sanity check |
| **Table 1** — FedPartBE vs baselines | `configs/fedpartbe_benchmark.yaml` · or `--group benchmark` | accuracy / energy / Jain vs FedAvg, FedPart, … |
| **Table 3** — component ablation | `configs/fedpartbe_ablation_no_repr.yaml`, `…_no_staleness.yaml`, `…_m1.yaml` · or `--group ablation` | effect of repr-prox, staleness, single-tier |
| **Figure 4** — #tiers M sweep | `configs/fedpartbe_sensitivity_tiers.yaml` · or `--group sensitivity` | accuracy vs number of energy tiers |
| **Figure 5** — Dirichlet α sweep | `configs/fedpartbe_sensitivity_alpha.yaml` · or `--group sensitivity` | accuracy vs heterogeneity α |
| **Table 4** — other datasets/models | `--group dataset` | CIFAR-100, TinyImageNet, MobileNetV2 |
| **Experiment A** — survival curve | `configs/fedpartbe_survival.yaml` | % clients alive vs rounds |
| **Experiment B** — wide battery spread | `configs/fedpartbe_survival_wide.yaml` (CIFAR-10: `…_cifar10.yaml`; 60-client: `…_fleet60.yaml`) | survival + accuracy under SoC ∈ [5%, 95%] |
| **Cost-model methodology** — {algo}×{phi,corrected,measured} ablation | `python scripts/run_costmodel_ablation.py` | does the FedPartBE-vs-FedPart gap widen under the corrected/measured cost model? → `results/costmodel_ablation/comparison.csv` |

> Confirm the exact Table/Figure numbers against the paper draft — the mapping
> above is taken from the headers inside each config file.

**Expected runtime / hardware.** Configs default to CIFAR-10 / ResNet-8 /
30 ESP32-S3 / 200 rounds. On an Apple M-series laptop (`--device mps`) a single
200-round run is on the order of tens of minutes; the full Table 1 sweep is
intended for the UM6P HPC (SLURM, see `hpc/` and the `hpc:` block in the master
config). The smoke config runs in well under a minute on a laptop CPU.

### Compute-cost model

`hardware/flop_cost.py` is the **single source of truth** for the per-round
compute FLOPs of every algorithm. It exposes three models via the `cost_model`
config key (CLI `--cost-model`, else YAML, else `"phi"`):

- **`phi`** (default, legacy) — reproduces the pre-refactor analytic per-algo
  formulas bit-for-bit; this is the path the paper numbers use and the one the
  regression test `tests/test_flop_cost.py` pins.
- **`corrected`** — position-aware analytic model for contiguous layer groups.
- **`measured`** — `torch.utils.flop_counter.FlopCounterMode` with the
  algorithm's actual `requires_grad` mask (cached).

The MAC-vs-FLOP convention is a property of the installed torch version;
`calibrate_convention()` detects it at runtime (returns `1.0` or `2.0`) and the
test suite asserts the detected value. Pin `torch` (see `requirements.txt`) to
reproduce absolute FLOP/energy figures. Communication energy lives separately
in `hardware/energy_model.py` (Shannon channel model).

---

## Message Protocol

| Message | Direction | Socket | Content |
|---|---|---|---|
| `REGISTER` | Worker → Server | DEALER→ROUTER | client_id, profile, dataset_size |
| `REGISTERED` | Server → Worker | ROUTER→DEALER | initial_battery_j |
| `TRAIN_REQ` | Server → Workers | PUB | round, weights_bytes, client_states, config |
| `UPDATE` | Worker → Server | DEALER→ROUTER | update_bytes, metadata, client_state |
| `EVAL_RESULT` | Server → Workers | PUB | round metrics |
| `SHUTDOWN` | Server → Workers | PUB | — |

---

## Implemented Algorithms

| Algorithm | Description |
|---|---|
| `fedavg` | FedAvg (McMahan et al., 2017) — full-model baseline |
| `fedprox` | FedProx (Li et al., 2020) — proximal regularization |
| `fedpart` | FedPart (Wang et al., NeurIPS 2024) — partial network updates |
| `fedpart_be` | **FedPartBE** (ours) — battery-aware multi-tier partial updates |
| `heterofl` | HeteroFL (Diao et al., 2021) — model heterogeneity |
| `fjord` | FjORD (Horvath et al., 2021) — ordered dropout |
| `scaffold` | SCAFFOLD (Karimireddy et al., 2020) — variance reduction |
| `leanfed` | LeanFed (Pereira et al., 2025) — data reduction |

---

## Device Profiles

| Profile | CPU | RAM | ↑ Mbps | Battery |
|---|---|---|---|---|
| `raspberry_pi_4` | ARM Cortex-A72 @ 1.5 GHz | 3.9 GB | 50 | 183 kJ |
| `raspberry_pi_zero2w` | ARM Cortex-A53 @ 1.0 GHz | 480 MB | 20 | 54 kJ |
| `jetson_nano` | Maxwell GPU 472 GFLOPS | 3.8 GB | 80 | 360 kJ |
| `esp32_s3` | Xtensa LX7 @ 240 MHz | 8 MB | 5 | 13.3 kJ |
| `smartphone_midrange` | Octa-core @ 2.0 GHz | 5.5 GB | 25 | 69 kJ |
| `smartphone_highend` | NPU @ 3.0 GHz | 11 GB | 100 | 65 kJ |

Energy is computed via the **T_Correction model**: $E_\text{comp} = P_c \times 3F_\text{eff} / \text{GFLOPS}$, with partial backward FLOPs measured per layer group via PyTorch forward hooks.

---

## Project Structure

```
fedlab_zmq/
├── run_experiment.py       ← main experiment entry point
├── algorithms/             ← pluggable FL algorithm implementations
│   ├── fedavg.py
│   ├── fedpart.py
│   ├── fedpart_be.py       ← FedPartBE (proposed)
│   └── ...
├── hardware/
│   ├── profiles.py         ← device profiles (ESP32, RPi, Jetson, ...)
│   └── energy_model.py     ← T_Correction + Shannon energy models
├── models/                 ← ResNet-8, ResNet-18, MLP, ViT-Tiny
├── datasets/               ← CIFAR-10/100, FEMNIST, TinyImageNet
├── dashboard/app.py        ← Streamlit monitoring dashboard
├── configs/                ← YAML experiment configurations
│   ├── fedpartbe_survival_wide.yaml
│   └── ...
├── docs/                   ← energy model derivation, analysis docs
└── papers_fedpartbe/       ← presentation and paper drafts
```

---

## Configuration

Experiments are fully defined by YAML configs:

```yaml
seed: 42
output_dir: "./results/my_experiment"
device: mps   # or cuda, cpu

data:
  dataset: cifar10
  model: resnet8
  partition: dirichlet
  alpha: 0.5

training:
  num_rounds: 200
  algorithm: fedpart_be
  algo_config:
    lr: 0.01
    local_epochs: 8
    num_tiers: 2
    server_lr: 1.0
    rotation_period: 5
    energy_scale_factor: 12.6

clients:
  num_clients: 30
  fleet:
    - type: esp32_s3
      count: 30
  battery_init:
    distribution: uniform_soc
    params:
      min_soc: 0.05
      max_soc: 0.95
```

---

## Gradient Variance Estimator (σ²)

Measures the inter-client gradient variance σ² empirically, used to validate the optimal tier formula:

$$M^* = \left(\frac{2G^3 K}{\sigma^2}\right)^{1/6}$$

```bash
# From the project root
python3 -m diagnostics.gradient_variance --dataset cifar10  --model resnet8 --clients 30 --alpha 0.5 --batches 3
python3 -m diagnostics.gradient_variance --dataset cifar100 --model resnet8 --clients 30 --alpha 0.5 --batches 3

# Faster estimate (1 batch/client)
python3 -m diagnostics.gradient_variance --dataset cifar10 --clients 30 --batches 1

# On Apple Silicon
python3 -m diagnostics.gradient_variance --dataset cifar10 --clients 30 --batches 3 --device mps
```

The result is also available interactively in the dashboard under the **σ² Estimator** tab.

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | Dataset name |
| `--model` | `resnet8` | Model architecture |
| `--clients` | `30` | Number of FL clients K |
| `--alpha` | `0.5` | Dirichlet α (heterogeneity) |
| `--batches` | `3` | Mini-batches per client (more = more accurate) |
| `--device` | `cpu` | `cpu`, `mps`, or `cuda` |

---

## Links

- **Code**: [github.com/Yunpei24/fedlab_zmq](https://github.com/Yunpei24/fedlab_zmq)
- **Dashboard**: [fedlabzmq-dashboard.streamlit.app](https://fedlabzmq-dashboard.streamlit.app)

---

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this software or the FedPartBE algorithm, please cite it using the
metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{nikiema_fedlab_zmq_2026,
  title  = {FedLab ZMQ: Energy-Efficient Federated Learning on Constrained IoT Devices},
  author = {Nikiema, Joshua Juste Yunpei and Amhoud, El Mehdi and
            Elhammouti, Hajar and Kissami, Imad},
  year   = {2026},
  url    = {https://github.com/Yunpei24/fedlab_zmq},
  note   = {Mohammed VI Polytechnic University (UM6P)}
}
```
