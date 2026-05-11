# ROLE

You are helping me implement and experimentally validate a **new research variant of FedPart** inside an existing federated learning codebase.

Your job is to help me **extend that framework** with a new algorithm, while keeping the implementation **modular, clean, reproducible, and easy to ablate**.

You must think like a **research engineer + ML systems engineer + FL algorithm designer**.

---

# HIGH-LEVEL RESEARCH IDEA

I want to implement a **new variant of FedPart** that addresses a key limitation:

> FedPart implicitly assumes that the computational cost across layers is roughly uniform, but in reality different layers have very different costs.

My proposed idea is to make FedPart:

# **Clustered Cost-Aware FedPart with Layer Coverage Constraints**

(working title; can also be renamed internally)

The key intuition is:

* **Different layers have different computational costs**
* **Different clients have different resource capabilities**
* Therefore, **not all clients should train the same subset of layers**

Instead, I want to:

1. **Group model layers by computational cost**

   * low-cost layers
   * medium-cost layers
   * high-cost layers

2. **Cluster clients by system/resource capability**
   based on:

   * battery level
   * compute frequency / speed
   * memory (optional)
   * available training time / budget
   * optionally bandwidth / latency

3. **Optionally improve client clustering using statistical balancing**
   so that each cluster is not only resource-compatible, but also less statistically skewed when possible.

4. **Assign different subsets of layers to different client clusters**
   so weaker clients train cheaper layers, and stronger clients can train more expensive layers.

5. Add **stabilization mechanisms** so the method does not collapse under non-IID data:

   * global anchor layers
   * periodic full synchronization / full training rounds
   * optional latent alignment / distillation
   * layer coverage constraints

This should be implemented as a **research prototype** suitable for experimentation and ablation.

---

# CORE GOAL

Implement a new FL algorithm that extends FedPart and can answer experimentally:

* Does cost-aware clustered partial training reduce local compute cost?
* Does it preserve / improve accuracy?
* Does it improve training time / energy / wall-clock?
* How does it behave under IID vs non-IID data?
* How sensitive is it to client clustering and layer assignment?

---

# IMPORTANT IMPLEMENTATION REQUIREMENTS

## 1) DO NOT rewrite the whole framework

Instead:

* inspect the existing FedPart which is an implementation
* reuse as much as possible
* add the new method in a modular way

## 2) Keep compatibility with the current codebase

The new implementation should:

* reuse existing dataset loaders
* reuse existing model definitions
* reuse existing client/server training loops when possible
* reuse FedPart masks / layer-group logic if already present

## 3) Everything should be configurable

I want to be able to turn ON/OFF each component for ablation.

So create a config-driven design.

---

# NAME OF THE NEW ALGORITHM

Internally, implement it under a clean name such as:

* `cc_fedpart`
* or `clustered_cost_fedpart`

Pick one and use it consistently.

---

# ALGORITHM DESIGN

---

# PART A — LAYER COST MODELING

We define the model as having (M) trainable parameter groups (or layer groups):

[
w = [w_1, w_2, \dots, w_M]
]

Each layer/group ( \ell \in {1,\dots,M} ) has a computational cost:

[
c_\ell > 0
]

We want to estimate or approximate this cost.

## IMPLEMENTATION REQUIREMENT

Implement a **layer cost profiler**.

For each trainable layer or layer group, estimate a scalar cost (c_\ell).

Support at least these modes:

### Mode 1 — FLOPs-based proxy

Estimate cost using approximate FLOPs.

### Mode 2 — Parameter-count proxy

Use number of parameters as a simpler proxy.

### Mode 3 — Measured runtime proxy

If possible, estimate actual per-layer runtime using a small profiling pass on dummy input or one minibatch.

### Mode 4 — Hybrid weighted proxy

Allow:
[
c_\ell = \alpha \cdot \text{FLOPs}*\ell + \beta \cdot \text{Params}*\ell + \gamma \cdot \text{Runtime}_\ell
]

Implement this cleanly.

## OUTPUT

The profiler should output:

* list of layer names / group names
* their cost values
* normalized costs
* assignment into cost bins:

  * low
  * medium
  * high

Use either:

* quantiles
* or configurable thresholds

Config options:

* `cost_mode`
* `num_cost_bins`
* `cost_bin_strategy`
* `cost_bin_thresholds`

---

# PART B — CLIENT RESOURCE MODELING

Each client (i) at round (t) has a resource capability score:

[
r_i^{(t)}
]

This score can be based on:

* battery level
* compute speed / frequency
* memory
* time budget
* bandwidth / latency (optional)

For now, if real hardware signals are not available, simulate them.

## IMPLEMENTATION REQUIREMENT

Implement a **client resource simulator / profile generator**.

Each client should have a resource vector:

[
u_i^{(t)} = [b_i^{(t)}, f_i^{(t)}, m_i^{(t)}, \tau_i^{(t)}, n_i^{(t)}]
]

Where possible define:

* (b_i^{(t)}): battery level
* (f_i^{(t)}): compute speed
* (m_i^{(t)}): memory capacity
* (\tau_i^{(t)}): available training budget / time budget
* (n_i^{(t)}): optional network condition

Then define a scalar capability score:

[
r_i^{(t)} = \lambda_1 b_i^{(t)} + \lambda_2 f_i^{(t)} + \lambda_3 m_i^{(t)} + \lambda_4 \tau_i^{(t)} + \lambda_5 n_i^{(t)}
]

Configurable weights.

## IMPORTANT

Support:

* static resource profiles
* dynamic per-round resource updates

---

# PART C — OPTIONAL STATISTICAL CLIENT BALANCING

This is important under non-IID data.

Even if each client is individually non-IID, I want client clusters to be more balanced **collectively**.

Each client (i) has a local empirical label distribution:

[
q_i = (q_i^{(1)}, \dots, q_i^{(K)})
]

where (K) is the number of classes.

I want the clustering to optionally consider both:

* system/resource similarity
* statistical balancing

## IMPLEMENTATION REQUIREMENT

Implement optional **hybrid client clustering**.

Support at least:

### Mode A — Resource-only clustering

Cluster using only resource vectors.

### Mode B — Resource + label histogram clustering

Cluster using:

* resource vector
* label histogram summary

### Mode C — Resource + label balancing assignment

Instead of pure clustering, assign clients to clusters to improve cluster-level label diversity.

## Practical simplification

Start with:

* KMeans or simple threshold-based clustering for resources
* optional balancing refinement based on label histograms

## Objective intuition

Each cluster should be:

* internally coherent in resource capability
* not too statistically skewed

---

# PART D — CLIENT CLUSTERS AND LAYER ASSIGNMENT

Suppose we have 3 client clusters:

* weak clients
* medium clients
* strong clients

and 3 layer-cost groups:

* low-cost layers
* medium-cost layers
* high-cost layers

We want an assignment policy.

Define for each client (i) at round (t) a selected set of trainable layers:

[
S_i^{(t)} \subseteq {1,\dots,M}
]

with binary mask:

[
x_{i\ell}^{(t)} =
\begin{cases}
1 & \text{if client } i \text{ trains layer } \ell \text{ at round } t \
0 & \text{otherwise}
\end{cases}
]

## REQUIRED INITIAL POLICY

Implement a simple default policy:

### Weak cluster:

train only low-cost layers

### Medium cluster:

train low-cost + medium-cost layers

### Strong cluster:

train low + medium + high-cost layers

Also support variants:

* strong cluster trains all layers
* medium cluster probabilistically trains some high-cost layers
* weak cluster occasionally rotates into medium layers

This must be configurable.

---

# PART E — LAYER COVERAGE CONSTRAINTS

This is critical.

A major failure mode is that expensive layers might be trained by too few clients.

For each layer (\ell), define the active set:

[
\mathcal{A}_\ell^{(t)} = { i : \ell \in S_i^{(t)} }
]

We want a minimum coverage constraint:

[
|\mathcal{A}*\ell^{(t)}| \ge m*{\min}
]

## IMPLEMENTATION REQUIREMENT

Implement a **layer coverage controller**.

At each round:

* check how many clients are assigned to each layer
* if a layer is under-covered, expand its assignment to additional eligible clients

Possible strategies:

* random fill
* highest-resource fill
* under-trained layer priority fill

Make this configurable.

This is one of the most important modules.

---

# PART F — GLOBAL ANCHOR LAYERS

To reduce layer mismatch, I want some layers to remain globally shared / synchronized more consistently.

Let:

[
\mathcal{L}_{anchor} \subseteq {1,\dots,M}
]

These are anchor layers.

## IMPLEMENTATION REQUIREMENT

Support optional anchor layers.

Examples:

* classifier head always trained by all clients
* some middle block always shared
* user-configurable anchor indices

Behavior:

* anchor layers should be included in every client’s trainable mask
  OR
* at least always synchronized globally depending on configuration

Add config:

* `use_anchor_layers`
* `anchor_layer_indices`
* `anchor_mode`

---

# PART G — PERIODIC FULL TRAINING / REALIGNMENT

To prevent accumulated mismatch drift, implement periodic full rounds.

Every (K) rounds, do:

[
S_i^{(t)} = {1,\dots,M}
]

for all selected clients (or at least all eligible clients).

## IMPLEMENTATION REQUIREMENT

Implement:

* `full_sync_period`
* `full_sync_warmup_rounds`
* `full_sync_mode`

Support:

* warm-up full rounds at the beginning
* periodic full rounds later

This is very important for stabilization.

---

# PART H — OPTIONAL LATENT ALIGNMENT / DISTILLATION

This is optional but should be implemented in a modular way.

The goal is to reduce representational mismatch across clusters.

Let (h_i(x)) be the latent representation of client (i) at some chosen layer.

We can define a latent alignment loss:

[
\mathcal{L}*{align}^{(i)} = | h_i(x) - h*{ref}(x) |^2
]

or class-prototype alignment.

## IMPLEMENTATION REQUIREMENT

Implement a modular latent alignment module.

Support at least these modes:

### Mode 1 — OFF

No latent alignment.

### Mode 2 — Global teacher latent alignment

Use the global model’s latent representation as reference.

### Mode 3 — Class prototype alignment

Each client computes class-wise latent prototypes, server aggregates them, and clients align to global prototypes.

### Mode 4 — Cluster-level prototype alignment (optional if feasible)

## Loss

Client local objective becomes:

[
\mathcal{L}*i =
\mathcal{L}*{task}
+
\lambda_{align}\mathcal{L}_{align}
]

Implement this in a way that can be easily turned on/off.

---

# PART I — LOCAL TRAINING OBJECTIVE

Each client (i) should optimize only over the selected layers (S_i^{(t)}).

Formally:

[
\min_{w_i} F_i(w_i)
]

but only selected coordinates / layers are updated.

Equivalent masked gradient view:

[
g_i^{(t)} = m_i^{(t)} \odot \nabla F_i(w^{(t)})
]

where:

* (m_i^{(t)}) is the binary layer mask expanded to parameter tensors
* only selected layers are trainable

## IMPLEMENTATION REQUIREMENT

Reuse FedPart’s masking / partial-update machinery if it already exists.

This is extremely important:

* DO NOT implement partial updates in a hacky way
* respect optimizer states where possible
* frozen layers should not be updated

---

# PART J — SERVER AGGREGATION

Aggregation should be **layer-wise**.

For each layer (\ell), aggregate only over clients that updated that layer:

[
w_\ell^{(t+1)} =
\sum_{i \in \mathcal{A}*\ell^{(t)}} \alpha*{i,\ell}^{(t)} , w_{i,\ell}^{(t+1)}
]

where weights may depend on:

* local dataset size
* uniform weighting
* optional trust/resource weighting

## IMPLEMENTATION REQUIREMENT

Implement clean layer-wise aggregation.

Important corner cases:

* if a layer is not updated by any client at round (t), keep previous global value
* if only a few clients updated a layer, log that clearly

Add logging for:

* per-layer number of contributors
* per-layer aggregation weights
* under-covered layers

---

# MATHEMATICAL OPTIMIZATION VIEW

I want this algorithm to be implemented in a way consistent with the following optimization interpretation.

We define binary assignment variables:

[
x_{i\ell}^{(t)} \in {0,1}
]

where (x_{i\ell}^{(t)} = 1) if client (i) trains layer (\ell) at round (t).

## Main constrained optimization intuition

We want to approximately solve:

[
\min_{{x_{i\ell}^{(t)}},, w}
\quad
\mathcal{L}*{global}(w)
+
\lambda_1 \cdot \mathcal{C}*{comp}
+
\lambda_2 \cdot \mathcal{R}*{mismatch}
+
\lambda_3 \cdot \mathcal{C}*{energy}
+
\lambda_4 \cdot \mathcal{R}_{imbalance}
]

subject to:

### (1) Client budget constraints

[
\sum_{\ell=1}^{M} x_{i\ell}^{(t)} c_\ell \le B_i^{(t)}
]

### (2) Layer coverage constraints

[
\sum_{i=1}^{N} x_{i\ell}^{(t)} \ge m_{\min}
]

### (3) Anchor constraints

[
x_{i\ell}^{(t)} = 1
\quad \forall \ell \in \mathcal{L}_{anchor}
]

### (4) Periodic full-sync override

At special rounds:
[
x_{i\ell}^{(t)} = 1 \quad \forall i,\ell
]

## IMPORTANT

You do NOT need to solve this optimization exactly.
Instead, implement the algorithm as a **heuristic constrained scheduler** inspired by this formulation.

But structure the code and comments so this optimization interpretation is preserved.

---

# OBJECTIVE FUNCTION TO TRACK EXPERIMENTALLY

Please help me implement logging / metrics corresponding to these quantities.

## 1) Task performance

* train loss
* test accuracy / test loss
* convergence curves

## 2) Computation cost

Track per-client and per-round:

* number of trainable parameters
* estimated FLOPs
* selected layer cost sum:
  [
  \sum_{\ell} x_{i\ell}^{(t)} c_\ell
  ]
* optional local wall-clock training time

## 3) Coverage metrics

Track:

* how many clients updated each layer
* average layer coverage
* min layer coverage
* coverage imbalance

## 4) Drift / mismatch proxy

We need a measurable proxy for layer mismatch.

Implement at least one or more of:

### Proxy A — Layer divergence

For each layer:
[
\frac{1}{|\mathcal{A}*\ell^{(t)}|}
\sum*{i \in \mathcal{A}*\ell^{(t)}}
| w*{i,\ell}^{(t+1)} - w_\ell^{(t)} |^2
]

### Proxy B — Cross-client layer variance

### Proxy C — Latent representation divergence (if latent alignment is enabled)

## 5) Fairness / client burden metrics

Track:

* average cost per client cluster
* variance of client training cost
* average assigned cost vs capability score

---

# PSEUDOCODE TO IMPLEMENT

Please implement the algorithm based on the following pseudocode.

---

## **Algorithm: Clustered Cost-Aware FedPart**

**Input:**

* Global model (w^{(0)})
* Clients ({1,\dots,N})
* Number of rounds (T)
* Layer costs ({c_\ell}_{\ell=1}^M)
* Client resource profiles ({u_i^{(t)}})
* Minimum layer coverage (m_{\min})
* Full-sync period (K)

---

### **Initialization**

1. Partition layers into:

   * low-cost
   * medium-cost
   * high-cost

2. Initialize client resource profiles.

3. Optionally initialize anchor layers and latent prototype buffers.

---

### **For each round (t = 0,1,\dots,T-1):**

1. **Sample participating clients** ( \mathcal{P}^{(t)} )

2. **Update / read client resource profiles** for participating clients

3. **Cluster participating clients**

   * based on resources only
   * or resources + statistical balancing

4. **Assign trainable layer subsets**
   For each client (i \in \mathcal{P}^{(t)}):

   * determine cluster membership
   * assign layer subset (S_i^{(t)})

5. **Apply anchor layer rules**

   * ensure anchor layers are included if enabled

6. **Apply periodic full-sync override**

   * if (t) is a full-sync round, assign all layers

7. **Apply layer coverage correction**

   * ensure every layer has at least (m_{\min}) assigned clients

8. **Local training**
   For each client (i \in \mathcal{P}^{(t)}):

   * receive global model
   * freeze non-selected layers
   * train selected layers locally
   * optionally compute latent alignment loss
   * return updated selected parameters and logging stats

9. **Server aggregation**
   For each layer (\ell):

   * aggregate only over clients that updated (\ell)

10. **Update global model**
    [
    w^{(t+1)} \leftarrow \text{layerwise aggregate}
    ]

11. **Log metrics**

    * accuracy
    * cost
    * coverage
    * mismatch proxies
    * cluster statistics

---

# ABLATION SUPPORT (VERY IMPORTANT)

Please implement the code so I can run clean ablations:

## Ablation switches:

* baseline FedAvg
* baseline FedPart
* cost-aware only
* cost-aware + clustering
* cost-aware + clustering + coverage
* * anchor layers
* * periodic full-sync
* * latent alignment
* * statistical balancing

This is extremely important.

---

# EXPECTED FILE / MODULE ORGANIZATION

Please inspect the current codebase first and adapt to its structure, but I would like a modular design similar to:

* `algorithms/clustered_cost_fedpart.py`
* `schedulers/layer_cost_profiler.py`
* `schedulers/client_clusterer.py`
* `schedulers/layer_assignment.py`
* `schedulers/coverage_controller.py`
* `losses/latent_alignment.py`
* `utils/metrics_cost.py`
* `utils/metrics_mismatch.py`

If the existing framework has a different structure, adapt cleanly rather than forcing this exact layout.

---

# EXPERIMENTAL PLAN TO SUPPORT

After implementing, help me run the following experiments.

## Datasets

Reuse whatever datasets are already in the framework, but prioritize:

* CIFAR-10
* CIFAR-100
* FashionMNIST / MNIST if available

## Models

Reuse models already implemented, such as:

* small CNN
* ResNet-8 / ResNet-18 if available

## Data regimes

Support:

* IID
* non-IID label skew
* non-IID Dirichlet partition

## Comparisons

Compare:

* FedAvg
* FedPart
* my new method

## Main plots I want

1. Test accuracy vs communication rounds
2. Test accuracy vs estimated compute cost
3. Test accuracy vs wall-clock time
4. Layer coverage over rounds
5. Mismatch proxy over rounds
6. Cluster-wise training burden

## Ablations

Test effect of:

* no coverage constraints
* no full-sync rounds
* no anchor layers
* no latent alignment
* no statistical balancing

---

# IMPLEMENTATION STYLE

Very important:

* Write clean, well-documented, research-friendly code
* Avoid hacks
* Add comments explaining how each module maps to the algorithm idea
* Make debugging easy
* Add clear logging
* Make it easy to inspect:

  * which clients trained which layers
  * layer costs
  * cluster assignments
  * per-round metrics

---

# WHAT I WANT FROM YOU FIRST

Before coding:

1. Inspect my existing FedPart code structure
2. Tell me exactly:

   * which files should be modified
   * which new files should be created
   * what the implementation plan is
3. Then implement incrementally in small, verifiable steps
4. After each step, explain what was added and how to test it

Do not jump into a giant uncontrolled rewrite.

---

# FINAL GOAL

The implementation should be good enough for:

* experimentation
* ablation studies
* possibly becoming the basis of a workshop / conference paper

Please be rigorous, modular, and practical.
