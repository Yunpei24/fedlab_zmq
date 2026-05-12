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
pip install -r requirements.txt
pip install -e .

# Run a full benchmark
python run_experiment.py --config configs/fedpartbe_survival_wide.yaml

# Launch the dashboard
streamlit run dashboard/app.py

# Compare experiment results
python scripts/compare_results.py --results results/exp1 results/exp2
```

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

## Links

- **Code**: [github.com/Yunpei24/fedlab_zmq](https://github.com/Yunpei24/fedlab_zmq)
- **Dashboard**: [fedlabzmq-dashboard.streamlit.app](https://fedlabzmq-dashboard.streamlit.app)
