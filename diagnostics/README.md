# Layer Mismatch Diagnostics

Diagnostic tools for measuring structural misalignment in federated learning models after aggregation.

## Overview

In federated learning, after aggregation (e.g., FedAvg), the global model's feature extractor and classifier layers are no longer perfectly aligned because each was trained on different data distributions. This **layer mismatch** can lead to:

- Reduced model accuracy
- Slower convergence
- Poor generalization

This module provides three complementary metrics to quantify layer mismatch:

### 1. Representation Drift

**What it measures:** How much intermediate activations change before vs. after aggregation.

**Formula:** For each layer *l*:

```
drift_l = ||h_l^before - h_l^after||_F / ||h_l^before||_F
```

where *h_l* are the activations at layer *l* on a fixed reference batch.

**Interpretation:**
- `drift ≈ 0`: Layer representations are stable after aggregation
- `drift > 0.5`: Significant drift, indicates structural mismatch

### 2. Post-Aggregation Loss Jump

**What it measures:** How much each client's loss increases immediately after receiving the global model (before any local fine-tuning).

**Formula:** For each client *i*:

```
Δloss_i = Loss_i(w_global) - Loss_i(w_local_i)
```

**Interpretation:**
- `Δloss > 0`: Client's loss increased (expected in non-IID settings)
- `Δloss < 0`: Client's loss decreased (rare, indicates good alignment)
- `|Δloss| > 1.0`: Large jump, severe mismatch

### 3. CKA (Centered Kernel Alignment)

**What it measures:** Similarity between layer representations of different clients.

**Formula:** For layers *l* and clients *(i, j)*:

```
CKA(H_i^l, H_j^l) = HSIC(K_i, K_j) / sqrt(HSIC(K_i, K_i) * HSIC(K_j, K_j))
```

where *K = H @ H^T* (Gram matrix), and HSIC is the Hilbert-Schmidt Independence Criterion.

**Interpretation:**
- `CKA = 1`: Identical representations
- `CKA > 0.7`: High similarity, good alignment
- `CKA < 0.3`: Low similarity, severe mismatch

**Reference:** Kornblith et al. (2019). *Similarity of Neural Network Representations Revisited*. ICML 2019.

---

## Usage

### Standalone Usage

```python
from diagnostics import LayerMismatchDiagnostic
import torch.nn as nn

# Initialize diagnostic
diagnostic = LayerMismatchDiagnostic(
    model=global_model,
    ref_dataloader=test_loader,
    device="cpu",
    layer_names=None  # auto-detect Conv2d and Linear layers
)

# After aggregation, compute all metrics
results = diagnostic.compute(
    model_before_agg=global_model_before,
    model_after_agg=global_model_after,
    client_models=client_models_list,
    client_dataloaders=client_loaders,
    criterion=nn.CrossEntropyLoss()
)

# Get a single summary score (0 = no mismatch, 1 = severe)
mismatch_score = diagnostic.summary_score(results)
print(f"Layer mismatch: {mismatch_score:.4f}")
```

### Integration with run_experiment.py

Enable layer mismatch diagnostics with the `--layer-mismatch` flag:

```bash
python run_experiment.py \
    --algo fedavg \
    --dataset cifar10 \
    --model resnet8 \
    --rounds 100 \
    --clients 10 \
    --layer-mismatch
```

Output will include the `LM=` metric in each round:

```
Round   5/100 | Acc=45.23% | Loss=1.4567 | ... | LM=0.342 | ...
```

Results JSON will contain `layer_mismatch` key in each round's metrics.

---

## API Reference

### `LayerMismatchDiagnostic`

#### Constructor

```python
LayerMismatchDiagnostic(
    model: nn.Module,
    ref_dataloader: DataLoader,
    device: str = "cpu",
    layer_names: Optional[List[str]] = None
)
```

**Parameters:**
- `model`: Global model (used for layer detection)
- `ref_dataloader`: Reference dataloader (one batch is captured)
- `device`: Device to run on ("cpu", "cuda", "mps")
- `layer_names`: Layers to monitor (None = auto-detect Conv2d + Linear)

#### Methods

##### `compute(model_before_agg, model_after_agg, client_models, client_dataloaders, criterion)`

Compute all three metrics.

**Returns:** Dict with keys:
- `"representation_drift"`: Dict[layer_name → drift_value]
- `"loss_jump"`: Dict with "mean", "std", "max", "client_0", "client_1", ...
- `"cka"`: Dict[layer_name → mean_cka_across_pairs]

##### `summary_score(results)`

Compute a single scalar mismatch score from all metrics.

**Returns:** Float in [0, 1] where:
- 0.0 = no mismatch
- 1.0 = severe mismatch

Formula (weighted average):
```
score = (drift_norm + |loss_jump|_norm + (1 - cka)) / 3
```

---

## Implementation Details

### Activation Capture

Uses `register_forward_hook` to capture intermediate activations. Hooks are automatically cleaned up after use.

### Memory Management

- All activation tensors are moved to CPU after capture to avoid OOM on small devices
- CKA computation is done on CPU
- Gram matrices are computed efficiently using `||X^T @ Y||_F^2`

### Edge Cases

- **Single client:** CKA returns 1.0 (no pairs to compare)
- **Empty dataloader:** Graceful failure with warning message
- **Layers with no spatial dimensions:** Handled via `flatten(1)`

---

## Examples

### Demo Script

```bash
python examples/layer_mismatch_demo.py
```

Shows a complete example with simulated FL round and interpretation of results.

### Full Experiment

```bash
# Compare layer mismatch across algorithms
python run_experiment.py --benchmark --layer-mismatch --rounds 50
```

This will generate mismatch scores for all 8 algorithms (FedAvg, E-CEFFL, LeanFed, etc.).

---

## When to Use

Layer mismatch diagnostics are useful for:

1. **Debugging convergence issues:** High mismatch may explain slow convergence
2. **Algorithm comparison:** Compare structural alignment across FL algorithms
3. **Non-IID analysis:** Quantify the effect of data heterogeneity
4. **Hyperparameter tuning:** Find optimal local epochs, learning rate, etc.
5. **Paper experiments:** Report mismatch as a secondary metric

---

## Performance

Diagnostic overhead per round (ResNet-8, 10 clients, CPU):
- Representation drift: ~0.1s
- Loss jump: ~2.0s (dominates, requires full forward passes)
- CKA: ~0.5s

**Total overhead:** ~2.6s per round (~10-20% of round time)

**Recommendation:** Enable only when needed (not for production training).

---

## References

1. **CKA metric:**
   Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019).
   *Similarity of Neural Network Representations Revisited.*
   ICML 2019.

2. **Layer mismatch in FL:**
   Li, Q., Diao, Y., Chen, Q., & He, B. (2021).
   *Federated Learning on Non-IID Data Silos: An Experimental Study.*
   ICDE 2021.

3. **Representation collapse:**
   Chen, X., & He, K. (2021).
   *Exploring Simple Siamese Representation Learning.*
   CVPR 2021.

---

## Future Extensions

Potential additions:
- [ ] Per-layer gradient flow analysis
- [ ] Spectral analysis of weight matrices (SVD)
- [ ] Activation distribution distance (Wasserstein)
- [ ] Layer-wise aggregation weights (adaptive FedAvg)

---

## Contact

For questions or issues, see the main FedLab ZMQ repository README.
