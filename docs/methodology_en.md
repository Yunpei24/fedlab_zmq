# Energy Methodology — FedLab ZMQ (FedStep)

*Honest description of the real repository state (branch `chore/artifact-cleanup`).
Values are quoted from the code; "done vs in-progress" is separated in §8.*

---

## 1. Overview — the research question

We study **energy-constrained federated learning**: a fleet of battery-powered
edge devices trains a shared model. The question is **which client trains what,
and when, so the fleet keeps participating before batteries die** — and whether
battery-aware *partial* training (FedStep) actually helps once energy is
accounted for **honestly**.

The energy framework simulates, per round and per client, the energy spent and
drains a battery; when a battery hits zero the client drops out. The framework
is in `hardware/` (`flop_cost.py`, `profiles.py`, `energy_model.py`), driven by
`run_experiment.py` and YAML configs in `configs/`.

**Glossary.** *FLOPs* = number of floating-point operations of a computation
(a count). *GFLOP/s* = a chip's peak rate (a speed). *MFU* = fraction of peak a
chip actually sustains. *cost_model* = how we count a round's compute FLOPs.
*alpha* = the compute energy_scale_factor (defined in §4).

---

## 2. The energy model

Per round, per client (`DeviceProfile.round_energy_breakdown`, `profiles.py`):

```
energy_round = energy_compute + energy_uplink + energy_downlink
```

**Compute** (`compute_energy_j`):

```
energy_compute = P_compute · (FLOPs / (peak_gflops · 1e9)) · alpha
```

i.e. power × time, where time = FLOPs / peak-rate. `alpha` scales **compute
only** (§4). FLOPs come from the chosen `cost_model` (§3); `peak_gflops` and the
FLOP count share the same FP32 mul+add convention so the division is a real time.

**Communication** (`comm_energy_j`) — the model that actually drains the battery:

```
energy_uplink   = P_tx · (uplink_bytes   / uplink_bytes_per_sec)
energy_downlink = P_rx · (downlink_bytes / downlink_bytes_per_sec)
```

i.e. transmit/receive power × transfer time at the device's fixed link
bandwidth (per-device `tx_w`, `rx_w`, `uplink_mbps`, `downlink_mbps` in
`profiles.py`). **alpha does NOT scale communication** — a radio transmission is
unaffected by the lack of an FPU.

> **Honesty note.** A richer **Shannon–Friis** channel model also exists
> (`energy_model.py`: `friis_path_loss`, `compute_shannon_rate_bps`, with
> `channel_params` attached to each profile). It is **not** currently wired into
> the battery-draining path — `round_energy_breakdown` uses the fixed-bandwidth
> model above. Shannon–Friis is available as a future refinement; the numbers in
> this study use the fixed-bandwidth comm model so the breakdown reconciles
> exactly with what drains the battery.

The breakdown sums back to the total bit-for-bit (verified:
`tests/test_energy_breakdown.py`, and round-level reconciliation in the runner).

---

## 3. Honest FLOP accounting — the three `cost_model`s

The compute FLOPs of a round are the single source of truth in
`hardware/flop_cost.py` (dispatcher `round_compute_flops`, flag `--cost-model`):

| cost_model | what it does |
|---|---|
| **`phi`** (legacy) | reproduces each algorithm's old analytic formula **bit-for-bit** (regression baseline, 11 tests) |
| **`corrected`** | position-aware *analytic* model for contiguous layer groups |
| **`measured`** (default) | `torch.utils.flop_counter.FlopCounterMode` run with the algorithm's **actual** `requires_grad` mask, cached |

The legacy `phi` formulas (quoted from the dispatcher):

- FedAvg / FedResonance: `full = 3 · 2 · N · B · S` (fwd+bwd+update × MAC→FLOP × params × batch × steps)
- FedPart: `full · (1/3 + 2/3 · gflops[g]/Σ gflops)`
- ServerMaskFL: `full · 0.5 · (1 + beta)`
- ccsEF: `2 · B · steps · (N + 2·|primary|)`

**The discovery.** Validating `phi` against `measured` (FlopCounterMode) showed
two problems (quoted from `flop_cost.py`): the analytic model **under-counts
CNNs by ~150–300×** (empirically **158×** for ResNet-8 here), *and* it
**inverts the per-group ranking**.

**Why position, not size, drives backward cost.** Train only group *p* (freeze
the rest). Autograd must still propagate `grad_input` backward through **every
layer downstream of p**. So a *shallow* group (the stem) triggers an almost-full
backward pass, while the *head* (fc) has nothing downstream and is nearly free.
The naive `gflops[p]/Σ gflops` makes the stem look cheapest; in reality it is the
most expensive. The corrected model charges this explicitly
(`compute_corrected_group_costs`):

```
corrected_p = gflops[p] + 0.5 · Σ_{i>p} gflops[i]      # own bwd + half of downstream
round_flops = full · (1/3 + 2/3 · corrected_p / Σ corrected)
```

**The consistency fix.** FedStep previously used `phi` for **energy
accounting** but corrected costs for **tier assignment** — internally
inconsistent. Now assignment **and** accounting use the **same** cost model. Under
honest costing the *real* FedStep advantage appears: it was partly hidden under
`phi`, which mis-priced exactly the shallow groups FedStep protects.

> Convention: `calibrate_convention()` measures ResNet-50 once; PyTorch 2.x
> reports true FLOPs → factor **1.0** (asserted by the test suite).

---

## 4. The alpha factor (compute utilization gap)

`measured` gives the *real* FLOPs, but a chip never sustains its rated peak on
real conv/BN (no/limited FPU/SIMD, memory-bound). `alpha` converts peak-rate
time into sustained time:

```
alpha = 1 / (sustained utilization of peak)   ≥ 1
```

`alpha = 1` is the unbeatable ideal; `alpha < 1` is physically impossible.
**alpha scales compute only** (flag `--alpha-applies-to`, default `compute`;
`total` exists solely to reproduce the legacy committed sweep). It **cancels in
relative (cross-algorithm) energy ratios** but **not** in absolute survival once
batteries die (it decides who dies and when).

**Anti-circularity.** alpha is **not** tuned to hit a survival target — that would
be circular (calibrating the energy unit against the outcome we measure). It is a
**physical** parameter, fixed independently. We set it as **documented per-device
estimates** (1/sustained-utilization), in `configs/device_profile_study.yaml`:

| profile | chip | ~sustained util | alpha |
|---|---|---|---|
| esp32_s3 | Xtensa LX7, no FPU/SIMD | ~10% | **10** |
| raspberry_pi_zero2w | Cortex-A53 + NEON | ~20% | **5** |
| raspberry_pi_4 | Cortex-A72 + NEON | ~33% | **3** |
| smartphone_midrange | mobile CPU/DSP | ~50% | **2** |
| smartphone_highend | NPU-class | ~65% | **1.5** |

These are **estimates, not measurements**. Robustness is shown by a **sweep**
`alpha ∈ [1, 2, 3, 5, 10, 20]` (`scripts/run_alpha_sensitivity.py`, marker
`alpha=5`). A config hook (`device_alpha_measured_anchor`) lets us drop in a
**measured** RPi4 alpha later and rescale the others by the utilization ratio.

---

## 5. The infeasibility finding

Under honest accounting (`measured`), training a **full** ResNet-8 on an
**ESP32-S3** with E=3 and alpha=12.6 drains the ~13.3 kJ battery in **~2–4
rounds** (observed in `results/CORRECTION/NIID05_E3/`). This is a **physical
truth**, not a tuning artifact: a no-FPU MCU simply cannot afford full-model
training. It is the motivation for **battery-aware partial training**
(FedStep): only train an affordable subset of layers per round. (Reducing the
per-round workload to E=1 moves median lifetime to **~15–35 rounds** — a readable
survival regime; see §6.)

---

## 6. Inter-profile study (a profile-driven projection)

`scripts/run_device_profile_study.py` replays the *same* workload across device
profiles (esp32_s3, raspberry_pi_zero2w, raspberry_pi_4, smartphone mid/high),
`cost_model=measured`, per-device alpha (compute-only). Per profile it reports
the **compute / uplink / downlink** energy split (per-round + cumulative) and
**survival/participation**; then a cross-profile view of how the balance shifts.

**Framing.** This is a **projection driven by profiles, not a deployment.** On
the ESP32-S3 ResNet-8 (+grads+optimizer ≈ 15 MB) does **not** fit in 8 MB RAM
(`DeviceProfile.can_run_model` → False), so that row is an **energy projection**;
RPi/smartphone genuinely run the model.

**Foregrounded metric by regime:** survival where the battery *bites* (ESP32
class), total energy / rounds-to-accuracy where it does *not* (RPi/smartphone).

**What the 2-extreme smoke shows so far** (esp32_s3 vs smartphone_highend,
3 clients): FedStep **saves ~57 % energy** vs FedAvg on ESP32 and **~35 %** on
the smartphone, and survives longer where the battery bites. Honest caveat: for
ResNet-8 (~78 k params) on full local data, **compute dominates on every
profile** (comm fraction only 0.000 → ~0.007). A genuinely comm-bound regime
would need a larger model and/or smaller local datasets. *(The full 5×5 grid has
not been run — see §8.)*

---

## 7. Metrics

- **Survival** (`metrics/survival.py`): median lifetime, round of the Nth death
  (N = 5, 10, 15), survival AUC (Σ clients-alive over rounds), participation
  fraction.
- **Energy breakdown:** per-round and cumulative compute / uplink / downlink
  (reconciles to the total).
- **alpha robustness:** relative claims (cross-algorithm energy ratios) are
  alpha-invariant (alpha cancels); the survival *order* is alpha-robust across
  the swept grid — so the algorithm comparison does not hinge on the exact alpha.

---

## 8. Status: done vs in-progress

**Done**
- `flop_cost.py`: three cost models; `phi` reproduces legacy bit-for-bit
  (11 regression tests green); position-aware `corrected`; `measured` via
  FlopCounterMode (cached). Convention factor 1.0 verified.
- Energy breakdown compute/uplink/downlink, per-round + cumulative; reconciles
  to total (5 dedicated tests).
- `alpha_applies_to=compute` is the default for all reported results.
- Per-device alpha **estimates** + sweep tooling; ESP32/E=1 diagnostic confirms
  a tens-of-rounds regime.
- Device-profile study tooling; **2-extreme smoke** validated end-to-end.
- Reproducibility: pinned deps, centralized seeding, per-run manifest, smoke
  config.

**In progress / pending**
- **Full alpha grid** (18 runs) not yet run on real hardware / at length.
- **Full 5×5 device-profile grid** not run (only the 2×2 smoke).
- alpha values are **documented estimates, not measured** (RPi4 measured anchor
  pending).
- **phi-contrast** sweep capability exists but the contrast run is not produced.
- Comm energy uses the **fixed-bandwidth** model; Shannon–Friis exists but is not
  wired into accounting.

---

## 9. How to run (real CLI)

Real flags (`run_experiment.py`): `--config`, `--algo`, `--epochs`, `--output`,
`--cost-model {phi,corrected,measured}`, `--energy-scale-factor`,
`--alpha-applies-to {compute,total}`, `--device`, `--seed`.
**Note:** with `--config`, `num_clients`/`num_rounds` come from the YAML —
`--clients`/`--rounds` are ignored; `--epochs`, `--algo`, `--cost-model`,
`--output`, `--device` do override.

```bash
# (a) Single run
caffeinate -si python run_experiment.py \
    --config configs/fedpartbe_survival_wide_cifar10.yaml \
    --algo fedstep \
    --epochs 3 \
    --output results/single/fedstep \
    --cost-model measured \
    --device mps

# (a') A/B FedStep vs FedPart, honest costing (the CORRECTION runs)
for ALGO in fedstep fedpart; do
    caffeinate -si python run_experiment.py \
        --config configs/fedpartbe_survival_wide_cifar10.yaml \
        --algo $ALGO \
        --epochs 3 \
        --output results/CORRECTION/NIID05_E3/${ALGO}_2 \
        --cost-model measured \
        --device mps
done

# (b) Inter-profile / survival study (per-device alpha, breakdown + survival)
caffeinate -si python scripts/run_device_profile_study.py --device mps --jobs 4
#   fast 2-extreme sanity (the validated smoke):
python scripts/run_device_profile_study.py --smoke \
    --profiles esp32_s3 smartphone_highend --algos fedavg fedstep

# (c) alpha sensitivity sweep (physical grid [1,2,3,5,10,20], marker 5)
caffeinate -si python scripts/run_alpha_sensitivity.py --grid full --device mps --jobs 4
#   targeted on the comm-bound extreme:
python scripts/run_alpha_sensitivity.py --grid full \
    --fleet-device smartphone_highend --device mps --jobs 4

# (d) phi contrast (appendix: show the gap is marginal under the legacy model)
python scripts/run_alpha_sensitivity.py --grid full --cost-model phi --device mps --jobs 4

# Sanity first (cost-model regression guardrail, < 5 s):
make test        # == python -m pytest  → 11 passed (+5 energy-breakdown)
```

Each run writes `metrics.json`, `manifest.json` (resolved config, git commit,
seed, package versions, FLOP convention), and `survival.csv` into its `results/`
dir.
