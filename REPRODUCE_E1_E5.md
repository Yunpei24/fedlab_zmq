# Reproducing FedSTEP experiments E1–E5

Commands to reproduce every experiment in the FedSTEP IoT-J paper. Each experiment
has **one config YAML per condition** and **one `scripts/run_eX.sh`** that loops
algorithms × seeds and aggregates to mean ± std. The experiments are contiguous
**E1–E5** (E4 = severe non-IID stress test, E5 = cross-dataset generalisation).

> Standardized harness committed in `1816a5e` (branch `chore/artifact-cleanup`).
> Honest accounting throughout: `cost_model: measured` (FlopCounterMode).

---

## 0. Prerequisites

```bash
# Use the project venv for every command (torch 2.11 / torchvision 0.26):
PY=venv/bin/python          # or: source venv/bin/activate

# CIFAR-10 / CIFAR-100 / EMNIST download automatically on first run (torchvision).
# REAL FEMNIST (LEAF per-writer split) for E5 must be generated once:
bash scripts/prepare_femnist.sh          # SF=0.05 (~36 writers, fast)
#   SF=1.0 bash scripts/prepare_femnist.sh   # full ~3500 writers (large)
```

- **Device**: configs use `device: mps` (Apple Silicon). Override with `--device cpu`
  or `--device cuda` on the command line if needed.
- **Seeds**: every `run_eX.sh` defaults to `SEEDS="42 43 44"` (≥3 for mean ± std).
- **Outputs**: `results/E<x>/.../seed<S>/metrics.json` (gitignored — `results/` and
  `data/` are not tracked).

---

## 1. One command per experiment

```bash
bash scripts/run_e1.sh      # E1 — Main comparison (iso-setup)
bash scripts/run_e2.sh      # E2 — Device heterogeneity (compute<->comm balance)
bash scripts/run_e3.sh      # E3 — Ablations of FedSTEP
bash scripts/run_e4.sh      # E4 — Severe non-IID stress test (alpha=0.1)
bash scripts/run_e5.sh      # E5 — Generalisation across datasets
```

Each script loops its conditions × seeds, writes per-seed `metrics.json`, then prints
mean ± std (and `--latex` rows for E1/E4/E5). They use `caffeinate -si` (no sleep) and
the venv python. Live logs go to `/tmp/e<x>_*.log`.

| Exp | what | algorithms | datasets / device | E (local epochs) |
|-----|------|-----------|-------------------|------------------|
| **E1** | main comparison, iso-setup | fedavg, fedpart, fedstep | CIFAR-10, 40×RPi-4, α=1 | 8 |
| **E2** | device heterogeneity | + fed_resonance | CIFAR-10, swept fleet¹ | 8 (fed_resonance: 3) |
| **E3** | ablations (rotation, repr, M, T_r) | fedstep | CIFAR-10, 40×RPi-4, α=1 | 8 |
| **E4** | severe non-IID | fedavg, fedpart, fedstep | CIFAR-10, 40×RPi-4, **α=0.1** | 8 |
| **E5** | cross-dataset | fedavg, fedpart, fedstep | CIFAR-100, FEMNIST(LEAF), EMNIST² | 8 (EMNIST: 1) |

¹ E2 fleets: `esp32_s3, raspberry_pi_4, raspberry_pi_zero2w, smartphone_midrange,
smartphone_highend`. ESP32-S3 is expected to die in warmup (the physical
infeasibility finding, not a bug).
² EMNIST-ByClass (the Dirichlet stand-in when LEAF is unavailable) is data-heavy
(~17k samples/client), so it runs on a **Jetson-Nano fleet at E=1**; true FEMNIST
(LEAF, ~100–300 samples/writer) runs on RPi-4 at the iso E=8.

---

## 2. Customizing a run (env overrides)

```bash
# fewer seeds / one algorithm
SEEDS="42" ALGOS="fedstep" bash scripts/run_e1.sh

# only some E3 ablations
VARIANTS="no_rotation M2" bash scripts/run_e3.sh

# only one device class for E2
DEVICES="raspberry_pi_4" bash scripts/run_e2.sh

# only one dataset for E5
DATASETS="femnist_natural" bash scripts/run_e5.sh

# run a single config directly (full control)
$PY run_experiment.py --config configs/e1_main.yaml --algo fedstep \
    --seed 42 --output results/E1/fedstep/seed42 --cost-model measured --device mps
```

---

## 3. Aggregating seeds → mean ± std

```bash
# groups by (algorithm, dataset); add --latex for paste-ready table rows
$PY scripts/aggregate_seeds.py results/E1 --latex
$PY scripts/aggregate_seeds.py results/E4 --latex
$PY scripts/aggregate_seeds.py results/E4/femnist_natural --latex

# E2 (device) and E3 (variant) vary by PATH, not by the summary -> aggregate per-subdir:
$PY scripts/aggregate_seeds.py results/E3/no_rotation
$PY scripts/aggregate_seeds.py results/E2/raspberry_pi_4__fedstep
```

Each cell becomes e.g. `69.8 +/- 0.4`. Convention: report **mean ± std over ≥3 seeds**;
accuracy and survival vary with the seed (Dirichlet partition), energy/uplink are
near-deterministic. Ablations (E3) may be reported single-seed.

---

## 4. Config map

| experiment | config(s) |
|---|---|
| E1 | `configs/e1_main.yaml` (≡ frozen `configs/fedstep_e1_alpha1.yaml`) |
| E2 | `configs/e2_base.yaml` (fleet type swapped per device by `run_e2.sh`) |
| E3 | `configs/e3_{no_rotation,no_repr,M2,M3,Tr2,Tr4,Tr5}.yaml` |
| E4 | `configs/e5_severe_niid.yaml` (α=0.1) |
| E5 | `configs/e5_{cifar100,femnist_natural,emnist}.yaml` (`e6_tinyimagenet.yaml` deferred) |

Shared E1/E3/E4 setup: ResNet-8, 40×RPi-4, Dirichlet, T=200, warmup=5, M=4, T_r=3,
μ_repr=0.01, adam lr=1e-3, measured cost, α_energy=3. E3/E4 differ by exactly the one
line named in the variant; E5 swaps dataset (and EMNIST swaps device+E).

---

## 5. Notes

- **Numbering**: contiguous E1–E5; E4 = severe non-IID, E5 = cross-dataset. (The
  $\alpha$-utilisation-gap robustness check is in App.~D, via `run_alpha_robustness.sh`.)
- The full multi-seed grid is heavy (E2 alone = 5 devices × 4 algos × 3 seeds = 60
  runs). Start with `SEEDS="42"` to smoke-test, then scale up.
- `results/` and `data/` are gitignored; the paper `.tex` (`papers_fedpartbe/`) and
  `docs/` are gitignored on this branch.
- Per-experiment definitions and the filled tables live in
  `papers_fedpartbe/fedstep_iotj_sections.tex` (sections E1–E5).
