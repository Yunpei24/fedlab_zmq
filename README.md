# FedLab ZMQ 🔬
### Federated Learning Research Framework — ZeroMQ transport

> Mohammed VI Polytechnic University (UM6P)  
> J. Nikiema & E. Amhoud — College of Computing

---

## Architecture de communication

```
Server  ROUTER tcp://*:5555  ← REGISTER + UPDATE from workers
Server  PUB    tcp://*:5556  → TRAIN_REQ + EVAL_RESULT + SHUTDOWN

Worker  DEALER tcp://host:5555  → REGISTER + UPDATE
Worker  SUB    tcp://host:5556  ← TRAIN_REQ + EVAL_RESULT + SHUTDOWN
```

**Pourquoi ZeroMQ vs FastAPI ?**

| Critère | FastAPI/uvicorn | ZeroMQ |
|---|---|---|
| Latence | ~5–20ms (HTTP overhead) | ~0.1–1ms (direct TCP) |
| Sérialisation | JSON/base64 | msgpack (binaire) |
| Broadcast | Non natif (loop HTTP) | PUB/SUB natif |
| Débit modèle | Limité par HTTP framing | Limité par TCP seul |
| Multi-machine | HTTP routing | TCP direct |
| Overhead | REST verbs + headers | Message frames uniquement |

---

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

# Lancer un benchmark complet
fedlab run --config configs/benchmark_suite.yaml

# Dashboard
fedlab dashboard

# Créer un nouvel algorithme
fedlab new-algo --name mon_algo

# Voir les logs en temps réel
fedlab monitor --log server.log

# Comparer des résultats
fedlab compare --results results/exp1 results/exp2
```

---

## Protocole de messages

| Message | Direction | Socket | Contenu |
|---|---|---|---|
| `REGISTER` | Worker → Server | DEALER→ROUTER | client_id, profile, dataset_size |
| `REGISTERED` | Server → Worker | ROUTER→DEALER | initial_battery_j |
| `TRAIN_REQ` | Server → Workers | PUB | round, weights_bytes, client_states, config |
| `UPDATE` | Worker → Server | DEALER→ROUTER | update_bytes, metadata, client_state |
| `EVAL_RESULT` | Server → Workers | PUB | round metrics |
| `SHUTDOWN` | Server → Workers | PUB | — |

Tous les messages sont sérialisés en **msgpack**. Les tenseurs PyTorch sont sérialisés en **torch.save → bytes** embarqués dans le dict msgpack.

---

## Ajouter votre algorithme

```bash
fedlab new-algo --name mon_algo
# → crée algorithms/mon_algo.py avec le template complet

# Ajouter l'import dans algorithms/__init__.py :
# import algorithms.mon_algo
```

---

## Profils matériels (DeviceProfile)

| Profil | CPU | RAM | ↑Mbps | Batterie |
|---|---|---|---|---|
| `raspberry_pi_4` | ARM A72 @ 1.5GHz | 3.9GB | 50 | 185kJ |
| `raspberry_pi_zero2w` | ARM A53 @ 1.0GHz | 480MB | 20 | 55kJ |
| `jetson_nano` | Maxwell GPU 472GF | 3.8GB | 80 | 370kJ |
| `esp32_s3` | Xtensa LX7 @ 240MHz | 8MB | 5 | 13.5kJ |
| `smartphone_midrange` | Octa @ 2.0GHz | 5.5GB | 25 | 74kJ |
| `smartphone_highend` | NPU @ 3.0GHz | 11GB | 100 | 70kJ |

---

## Structure

```
fedlab_zmq/
├── server/server.py        ← ZMQ ROUTER + PUB
├── worker/worker.py        ← ZMQ DEALER + SUB
├── core/
│   ├── protocol.py         ← types de messages + sérialisation
│   ├── orchestrator.py     ← lance server + workers
│   └── experiment.py       ← config YAML + résultats
├── algorithms/             ← 6 algos pluggables
├── hardware/profiles.py    ← profils matériels réalistes
├── models/registry.py      ← MLP, LeNet5, ResNet18/50, ViT-Tiny
├── datasets/               ← CIFAR-10/100, FEMNIST, TinyImageNet
├── dashboard/app.py        ← Streamlit
├── cli/fedlab.py           ← interface CLI
└── configs/                ← YAML d'expériences
```
