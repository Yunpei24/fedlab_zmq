"""
diagnostics/gradient_variance.py
=================================
Mesure empirique de la variance des gradients pour valider la formule M*.

Deux métriques distinctes :

  σ²_intra  — variance INTRA-client (Assumption 2 du papier Wang et al.) :
               E[||∇F_k(w;ξ) − ∇f_k(w)||²]
               Bruit dû au sampling mini-batch au sein d'un même client.
               C'est le σ² de la formule M* = (2G³K/σ²)^{1/6}.

  σ²_inter  — variance INTER-client (hétérogénéité des données) :
               (1/K) Σ_k ||∇f_k(w) − ∇f(w)||²
               À quel point les gradients clients divergent entre eux.

Usage CLI:
    python3 -m diagnostics.gradient_variance \\
        --dataset cifar10 --model resnet8 \\
        --clients 30 --alpha 0.5 --batches 5

Usage Python:
    from diagnostics.gradient_variance import measure_gradient_variance
    result = measure_gradient_variance("cifar10", "resnet8", num_clients=30)
    print(result.sigma2_intra, result.sigma2_inter)
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from datasets.registry import _load_raw_dataset, NUM_CLASSES
    from datasets.partitioner import partition_dataset
    from models.registry import get_model
except ModuleNotFoundError:
    import pathlib, sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from datasets.registry import _load_raw_dataset, NUM_CLASSES
    from datasets.partitioner import partition_dataset
    from models.registry import get_model


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradVarianceResult:
    # ── σ² intra-client (paper's Assumption 2, used for M* formula) ──────────
    sigma2_intra: float
    sigma2_intra_per_client: list[float]   # per-client intra variance

    # ── σ² inter-client (data heterogeneity, informational) ──────────────────
    sigma2_inter: float
    sigma2_inter_per_client: list[float]   # ||∇f_k − ∇f̄||² per client

    # ── gradient norms ────────────────────────────────────────────────────────
    grad_norm_per_client: list[float]      # ||ḡ_k|| (mean gradient per client)
    mean_grad_norm: float                  # ||∇f̄|| global mean

    # ── metadata ──────────────────────────────────────────────────────────────
    dataset: str
    model: str
    num_clients: int
    alpha: float
    batches_per_client: int
    device: str
    elapsed_s: float
    m_star_table: list[dict] = field(default_factory=list)

    # σ² used for M* is always intra-client
    @property
    def sigma2(self) -> float:
        return self.sigma2_intra

    def optimal_m(self, G: int, K: int | None = None) -> float:
        K = K or self.num_clients
        return (2 * G**3 * K / self.sigma2_intra) ** (1 / 6)

    def rounded_m(self, G: int, K: int | None = None) -> int:
        return max(1, round(self.optimal_m(G, K)))


# ─────────────────────────────────────────────────────────────────────────────

def _client_gradients(
    model: torch.nn.Module,
    subset,
    batch_size: int,
    batches_per_client: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Collect `batches_per_client` separate mini-batch gradient vectors for one client.
    Returns list of flat gradient tensors (one per mini-batch).
    Requires batches_per_client >= 2 to estimate intra variance.
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch_grads = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        g = torch.cat([
            p.grad.view(-1)
            for p in model.parameters()
            if p.grad is not None
        ]).detach().cpu()
        batch_grads.append(g)
        if len(batch_grads) >= batches_per_client:
            break
    model.zero_grad()
    return batch_grads


def measure_gradient_variance(
    dataset: str = "cifar10",
    model_name: str = "resnet8",
    num_clients: int = 30,
    alpha: float = 0.5,
    batches_per_client: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cpu",
    data_root: str = "./data",
    progress_cb: Callable[[int, int], None] | None = None,
) -> GradVarianceResult:
    """
    Measure intra-client and inter-client gradient variance.

    σ²_intra (paper's σ²) : variance of mini-batch gradients within each client,
                             averaged across clients. Used in M* formula.
    σ²_inter               : variance of client mean-gradients around global mean.
                             Reflects data heterogeneity.

    Args:
        batches_per_client: Number of mini-batches per client. Must be >= 2 to
                            estimate intra-client variance. 5 recommended.
    """
    if batches_per_client < 2:
        raise ValueError("batches_per_client must be >= 2 to estimate intra-client variance.")

    t0 = time.time()
    torch.manual_seed(seed)
    dev = torch.device(device)

    # ── 1. Load and partition dataset ─────────────────────────────────────────
    raw = _load_raw_dataset(dataset, "train", data_root)
    subsets = partition_dataset(
        dataset=raw,
        partition="dirichlet",
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    # ── 2. Load model (random init) ───────────────────────────────────────────
    model = get_model(model_name, dataset).to(dev)
    model.train()

    # ── 3. Collect per-client batch gradients ─────────────────────────────────
    # client_batch_grads[k] = list of B gradient vectors for client k
    client_batch_grads: list[list[torch.Tensor]] = []
    client_mean_grads: list[torch.Tensor] = []

    for k, subset in enumerate(subsets):
        if progress_cb:
            progress_cb(k, num_clients)
        if len(subset) == 0:
            continue

        batch_grads = _client_gradients(model, subset, batch_size, batches_per_client, dev)
        if len(batch_grads) < 2:
            continue

        client_batch_grads.append(batch_grads)
        g_stack = torch.stack(batch_grads)            # [B, d]
        client_mean_grads.append(g_stack.mean(dim=0)) # [d]

    if progress_cb:
        progress_cb(num_clients, num_clients)

    # ── 4. σ²_intra — variance of mini-batch gradients within each client ─────
    # For client k: σ²_k = (1/B) Σ_b ||g_{k,b} − ḡ_k||²
    sigma2_intra_per_client = []
    for batch_grads in client_batch_grads:
        g_stack = torch.stack(batch_grads)           # [B, d]
        g_mean  = g_stack.mean(dim=0, keepdim=True)  # [1, d]
        var_k   = ((g_stack - g_mean) ** 2).sum(dim=1).mean().item()  # scalar
        sigma2_intra_per_client.append(var_k)
    sigma2_intra = float(np.mean(sigma2_intra_per_client))

    # ── 5. σ²_inter — variance of client means around global mean ────────────
    mean_grads = torch.stack(client_mean_grads)      # [K_eff, d]
    global_mean = mean_grads.mean(dim=0)             # [d]
    diff = mean_grads - global_mean.unsqueeze(0)
    sigma2_inter_per_client = (diff ** 2).sum(dim=1).tolist()
    sigma2_inter = float(np.mean(sigma2_inter_per_client))

    grad_norms = mean_grads.norm(dim=1).tolist()
    mean_grad_norm = float(global_mean.norm())

    # ── 6. Build result ───────────────────────────────────────────────────────
    result = GradVarianceResult(
        sigma2_intra=sigma2_intra,
        sigma2_intra_per_client=sigma2_intra_per_client,
        sigma2_inter=sigma2_inter,
        sigma2_inter_per_client=sigma2_inter_per_client,
        grad_norm_per_client=grad_norms,
        mean_grad_norm=mean_grad_norm,
        dataset=dataset,
        model=model_name,
        num_clients=num_clients,
        alpha=alpha,
        batches_per_client=batches_per_client,
        device=device,
        elapsed_s=time.time() - t0,
    )
    result.m_star_table = _build_m_star_table(result)
    return result


def _build_m_star_table(r: GradVarianceResult) -> list[dict]:
    K_values = sorted(set([10, 30, 100, r.num_clients]))
    rows = []
    for G in [5, 10, 18]:
        for K in K_values:
            m_cont = r.optimal_m(G, K)
            m_int  = r.rounded_m(G, K)
            rows.append({
                "G (layer groups)": G,
                "K (clients)": K,
                "σ²_intra": round(r.sigma2_intra, 2),
                "M* (continuous)": round(m_cont, 3),
                "M* (rounded)": m_int,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Measure gradient variance σ² (intra + inter client) for FedStep M* formula."
    )
    parser.add_argument("--dataset", default="cifar10", choices=list(NUM_CLASSES.keys()))
    parser.add_argument("--model", default="resnet8")
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batches", type=int, default=5,
                        help="Mini-batches per client. Min 2 required for intra variance.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    print(f"[σ²] {args.dataset} / {args.model} / K={args.clients} / α={args.alpha} "
          f"/ {args.batches} batches/client")
    print(f"     σ²_intra = variance mini-batch au sein de chaque client (formule M*)")
    print(f"     σ²_inter = divergence entre clients (hétérogénéité des données)")

    def _progress(k, total):
        bar = "█" * (k * 20 // total) + "░" * (20 - k * 20 // total)
        print(f"\r  [{bar}] client {k}/{total}", end="", flush=True)

    result = measure_gradient_variance(
        dataset=args.dataset,
        model_name=args.model,
        num_clients=args.clients,
        alpha=args.alpha,
        batches_per_client=args.batches,
        batch_size=args.batch_size,
        seed=int(args.seed),
        device=args.device,
        data_root=args.data_root,
        progress_cb=_progress,
    )
    print()

    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  σ²_intra  = {result.sigma2_intra:>10.2f}   ← formule M* (paper's σ²)")
    print(f"  σ²_inter  = {result.sigma2_inter:>10.2f}   ← hétérogénéité données")
    print(f"  ratio     = {result.sigma2_intra / max(result.sigma2_inter, 1e-9):>10.2f}×  (intra/inter)")
    print(f"  ||∇f̄||   = {result.mean_grad_norm:>10.4f}")
    print(f"  Temps     = {result.elapsed_s:>10.1f}s")
    print(f"\n  M* = (2G³K/σ²_intra)^{{1/6}}  avec K={args.clients} :")
    print(f"  {'G':>4}  {'M* (cont)':>10}  {'M* (int)':>8}")
    print(f"  {'-'*28}")
    for row in result.m_star_table:
        if row["K (clients)"] == args.clients:
            print(f"  {row['G (layer groups)']:>4}  {row['M* (continuous)']:>10.3f}  {row['M* (rounded)']:>8}")
    print(sep)


if __name__ == "__main__":
    _cli()
