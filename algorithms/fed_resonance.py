"""
algorithms/fed_resonance.py
============================
Fed-Resonance: Adaptive Spectral Compression for Federated Learning
J. Nikiema, EL Amhoud, H. ELHAMMOUTI & I. KISSAMI — UM6P, 2026

Algorithm overview
------------------
Fed-Resonance performs per-layer hybrid compression by switching between
multiple compression methods depending on layer type and round-level conditions:

  1. SVD-truncated (FedSVD mode):
       G_l ≈ U_r S_r V_r^T  (rank-r approximation)
       Cost: r(m + n + 1) floats transmitted per layer
       Applied to: Conv and Linear layers

  2. Subspace projection (FedSubspace mode):
       g_l ≈ B_l alpha_l  (projection on shared basis)
       Cost: r floats per layer (only coefficients, basis already shared)
       Applied to: Conv and Linear layers with stable bases

  3. Sparse TopK compression:
       Retains top-k values by absolute magnitude
       Cost: k * 8 bytes (4B index + 4B value)
       Applied to: Embedding layers (large vocabulary matrices)

  4. Dense transmission:
       Full gradient transmitted without compression
       Applied to: Norm layers, bias vectors, small layers

Switch rule (per layer, per round):
  - SPARSE (TopK) if layer type is "embedding"
  - DENSE if layer type is "norm" or "bias"
  - DENSE if layer too small (< rank_min * 2 elements)
  - SVD(rank_min) if battery critically low (beta < 0.1)
  - SUBSPACE if subspace_drift < tau_drift AND r_eff >= rank_coherence_thresh * rank_min
  - SVD(adaptive rank) otherwise

Error feedback is maintained across mode switches, ensuring that no
gradient information is permanently discarded regardless of which
compression mode is active.

Basis Memory Management (for large models):
  - LRU eviction: evicts least recently used bases when memory exceeds budget
  - Age-based expiry: evicts bases unused for > max_age_rounds
  - Critical for ResNet-50+ and ImageNet-scale models

Battery model inherited from E-CEFFL:
  beta_t^k = beta_min + (beta_max - beta_min) * B_t^k / B_max
  beta in [beta_min, beta_max] always

Convergence:
  Under standard smoothness, bounded variance, and the contraction
  condition alpha_c = spectral_energy_thresh uniformly over rounds,
  Fed-Resonance achieves O(1/sqrt(T)) non-convex convergence.

Computational efficiency (Randomized SVD):
  - Full SVD: O(m·n·min(m,n)) — prohibitive for large layers
  - Randomized SVD: O(m·n·(r+p)) — practical for r << min(m,n)
  - Observed speedups: 17-37x for typical FL scenarios (rank 20-50)
  - Algorithm: Halko, Martinsson, Tropp (2011)

Reference algorithms:
  - PowerSGD:   Vogels et al., NeurIPS 2019 (fixed-rank, no battery)
  - FedSubspace: gradient subspace learning for FL
  - E-CEFFL:    Nikiema & Amhoud, 2025 (battery-adaptive TopK + EF)
"""

import gc
import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .base import FLAlgorithm, ClientState, AggregateResult, register_algorithm
from hardware.flop_cost import round_compute_flops


# ─────────────────────────────────────────────────────────────────────────────
# Spectral compression helpers
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_rank(
    tensor: torch.Tensor,
    energy_thresh: float,
    use_rsvd: bool = True,
    rsvd_oversample: int = 10,
    rsvd_power_iter: int = 2,
    rank_criterion: str = "energy",
    rank_budget_bytes: int = 0,
    rank_min: int = 1,
) -> int:
    """
    Compute the rank for SVD compression using one of three criteria.

    Criteria (controlled by rank_criterion):

      "energy" (default — existing behavior, unchanged):
          Smallest r such that top-r singular values capture >= energy_thresh
          of total spectral energy.
          r* = argmin_r { cumsum(sigma_i^2)_r / sum(sigma_i^2) >= energy_thresh }

      "elbow":
          Rank at the elbow (knee) of the cumulative energy curve, found via
          the Kneedle algorithm (maximum perpendicular distance from the
          line connecting the first and last points of the cumsum curve).
          Better than "energy" when energy_thresh is hard to tune across layers,
          but can return large ranks on flat spectra (clip externally with rank_max).
          Failure modes: flat spectra → overestimates rank; rank-1 spectra → rank=1.

      "budget":
          Maximum rank that fits within rank_budget_bytes bytes for SVD
          transmission of a (m, n) matrix: r = floor(budget / (4*(m+1+n))).
          Relevant when per-layer communication budget is fixed independently of
          the battery-level beta. Note: this is orthogonal to the battery-aware
          compression already handled by beta_min/beta_max — use "budget" only
          when you want a hard byte cap per layer independent of battery state.

    Args:
        tensor:           2-D tensor (the layer gradient reshaped as a matrix).
        energy_thresh:    Fraction of energy to capture (used only for "energy" criterion).
        use_rsvd:         If True, use randomized SVD for large matrices (default True).
        rsvd_oversample:  Oversampling parameter for rSVD (default 10).
        rsvd_power_iter:  Power iterations for rSVD (default 2).
        rank_criterion:   One of "energy" (default), "elbow", "budget".
        rank_budget_bytes: Byte budget per layer (used only for "budget" criterion).
        rank_min:         Minimum rank to return for "budget" and "elbow" (default 1).

    Returns:
        r* in [1, min(m, n)].
    """
    # Use float32 for numerical stability
    t = tensor.float()
    m, n = t.shape
    max_rank = min(m, n)

    # ── "budget" criterion does not need singular values ──────────────────────
    if rank_criterion == "budget":
        bytes_per_rank = 4 * (m + 1 + n)
        if bytes_per_rank <= 0 or rank_budget_bytes <= 0:
            return max(1, rank_min)
        r = rank_budget_bytes // bytes_per_rank
        return max(rank_min, min(int(r), max_rank))

    # ── All other criteria need singular values ───────────────────────────────
    # torch.linalg.svdvals returns singular values in descending order
    try:
        # For adaptive rank, we need all singular values to determine the cutoff.
        # For large matrices where rSVD would be beneficial, we use a heuristic:
        # estimate rank using full svdvals (which is still faster than full SVD)
        # OR use a conservative upper bound and iterate.
        #
        # Strategy: For small matrices (max_rank <= 128), always use full svdvals.
        # For large matrices, use full svdvals anyway since it's O(m*n*min(m,n)) but
        # doesn't require computing full U, V matrices — only singular values.
        #
        # Note: torch.linalg.svdvals is already optimized and faster than full SVD.
        # Randomized methods for computing singular values require iterative refinement
        # which may not be faster in practice for this use case.
        sv = torch.linalg.svdvals(t)
    except Exception:
        # Fallback: return rank 1 if SVD fails (e.g., singular matrix)
        return 1

    energy = sv.pow(2)
    total = energy.sum().item()
    if total < 1e-12:
        return 1

    # ── "elbow" criterion ─────────────────────────────────────────────────────
    if rank_criterion == "elbow":
        # Kneedle algorithm: find index of maximum perpendicular distance
        # from the line connecting the first and last points of the
        # normalized cumulative energy curve.
        cumsum = energy.cumsum(0)
        E = cumsum / total         # normalized cumsum, shape (max_rank,)
        nr = E.numel()
        if nr <= 1:
            return max(1, rank_min)
        t_axis = torch.linspace(
            0.0, 1.0, nr, dtype=torch.float32, device=sv.device
        )
        E_start = E[0].item()
        E_end = E[-1].item()
        L = E_start + t_axis * (E_end - E_start)
        diff = (E - L).abs()
        knee_idx = int(diff.argmax().item())
        return max(rank_min, knee_idx + 1)

    # ── "energy" criterion (default, existing behavior) ───────────────────────
    cumsum = energy.cumsum(0)
    # Find first index where cumsum / total >= energy_thresh
    mask = (cumsum / total) >= energy_thresh
    if mask.any():
        r = int(mask.nonzero(as_tuple=False)[0].item()) + 1
    else:
        r = int(sv.numel())

    return max(1, r)


def randomized_svd(
    tensor: torch.Tensor,
    rank: int,
    oversampling: int = 10,
    power_iter: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD for fast low-rank approximation.

    Algorithm from Halko, Martinsson, and Tropp (2011):
    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions."

    Complexity: O(m·n·(r+p) + (r+p)^2·(m+n)) vs O(m·n·min(m,n)) for full SVD.

    Args:
        tensor:      2-D float tensor of shape (m, n).
        rank:        Target rank r.
        oversampling: Extra samples p for accuracy (default 10).
        power_iter:   Number of power iterations q for improved accuracy (default 2).

    Returns:
        (U, S, Vt) where U.shape=(m,r), S.shape=(r,), Vt.shape=(r,n).
    """
    m, n = tensor.shape
    r = max(1, min(rank, min(m, n)))

    # If rank is close to matrix dimensions, fall back to full SVD
    if r + oversampling >= min(m, n):
        U_full, S_full, Vt_full = torch.linalg.svd(tensor, full_matrices=False)
        return U_full[:, :r].contiguous(), S_full[:r].contiguous(), Vt_full[:r, :].contiguous()

    # Step 1: Draw random Gaussian matrix Ω of shape (n, r+p)
    l = r + oversampling
    Omega = torch.randn(n, l, dtype=tensor.dtype, device=tensor.device)

    # Step 2: Y = G @ Ω  (m, r+p)
    Y = tensor @ Omega

    # Step 3: Power iterations to improve accuracy (optional, q times)
    # Y = (G @ G^T)^q @ Y
    for _ in range(power_iter):
        Y = tensor @ (tensor.t() @ Y)

    # Step 4: QR decomposition of Y to get orthonormal basis Q (m, r+p)
    Q, _ = torch.linalg.qr(Y)

    # Step 5: Project G into low-dimensional space: B = Q^T @ G  (r+p, n)
    B = Q.t() @ tensor

    # Step 6: Compute SVD of small matrix B
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)

    # Step 7: Recover left singular vectors: U = Q @ U_hat
    U = Q @ U_hat

    # Step 8: Truncate to rank r
    U = U[:, :r].contiguous()
    S = S[:r].contiguous()
    Vt = Vt[:r, :].contiguous()

    return U, S, Vt


def svd_compress(
    tensor: torch.Tensor,
    rank: int,
    use_rsvd: bool = True,
    rsvd_oversample: int = 10,
    rsvd_power_iter: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncated SVD compression with optional randomized SVD.

    Decomposes the input matrix as G ≈ U[:,:r] diag(S[:r]) V^T[:r,:].

    Args:
        tensor:          2-D float tensor of shape (m, n).
        rank:            Number of singular components to retain.
        use_rsvd:        If True, use randomized SVD; otherwise use full SVD (default True).
        rsvd_oversample: Oversampling parameter for rSVD (default 10).
        rsvd_power_iter: Power iterations for rSVD accuracy (default 2).

    Returns:
        (U, S, Vt) where U.shape=(m,r), S.shape=(r,), Vt.shape=(r,n).
    """
    t = tensor.float()
    rank = min(rank, min(t.shape))
    rank = max(1, rank)

    try:
        if use_rsvd:
            # Use randomized SVD for efficiency
            U, S, Vt = randomized_svd(t, rank, rsvd_oversample, rsvd_power_iter)
        else:
            # Use full SVD (backward compatibility)
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            U = U[:, :rank].contiguous()
            S = S[:rank].contiguous()
            Vt = Vh[:rank, :].contiguous()
    except Exception:
        # Degenerate fallback: return a rank-1 approximation
        U = torch.zeros(t.shape[0], 1, dtype=t.dtype, device=t.device)
        S = torch.zeros(1, dtype=t.dtype, device=t.device)
        Vt = torch.zeros(1, t.shape[1], dtype=t.dtype, device=t.device)

    return U, S, Vt


def svd_decompress(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a matrix from its truncated SVD.

    G_approx = U diag(S) V^T

    Args:
        U:  (m, r) left singular vectors.
        S:  (r,)   singular values.
        Vt: (r, n) right singular vectors (transposed).

    Returns:
        Reconstructed tensor of shape (m, n).
    """
    return (U * S.unsqueeze(0)) @ Vt


def subspace_project(
    tensor: torch.Tensor, basis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project a vector onto a shared orthonormal basis.

    For a flattened gradient g ∈ R^d and basis B ∈ R^(d, r) (B^T B = I_r):
        alpha = B^T g      (coordinates, r-dimensional)
        residual = g - B alpha

    Args:
        tensor: 1-D float tensor of length d.
        basis:  (d, r) orthonormal basis matrix.

    Returns:
        (alpha, residual) where alpha.shape=(r,) and residual.shape=(d,).
    """
    g = tensor.float().flatten()
    B = basis.float()
    alpha = B.t() @ g           # (r,)
    g_approx = B @ alpha        # (d,)
    residual = g - g_approx     # (d,)
    return alpha, residual


def subspace_reconstruct(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a gradient from subspace coefficients.

    g_approx = B alpha

    Args:
        coefficients: (r,) coordinate vector.
        basis:        (d, r) orthonormal basis.

    Returns:
        Reconstructed vector of shape (d,).
    """
    return basis.float() @ coefficients.float()


def subspace_drift(basis_old: torch.Tensor, basis_new: torch.Tensor) -> float:
    """
    Compute normalized Frobenius distance between two projection operators.

    drift = ||P_old - P_new||_F / sqrt(r)
          = ||B_old B_old^T - B_new B_new^T||_F / sqrt(r)

    This quantity lies in [0, 2] (theoretical bounds via ||P-Q||_F <= 2sqrt(r)
    for rank-r projectors, divided by sqrt(r)).

    Args:
        basis_old: (d, r) previous orthonormal basis.
        basis_new: (d, r) new orthonormal basis.

    Returns:
        Scalar drift value in [0, 2].
    """
    if basis_old is None or basis_new is None:
        return 2.0   # Maximum drift — treat as fully outdated

    B_old = basis_old.float()
    B_new = basis_new.float()

    # Efficient computation: ||P_old - P_new||_F^2
    # = Tr(P_old) + Tr(P_new) - 2 Tr(P_old P_new)
    # = r + r - 2 ||B_old^T B_new||_F^2
    r = B_old.shape[1]
    cross = B_old.t() @ B_new          # (r, r)
    frob_sq = 2.0 * r - 2.0 * (cross * cross).sum().item()
    frob_sq = max(0.0, frob_sq)        # Numerical safety

    return math.sqrt(frob_sq) / math.sqrt(max(r, 1))


def _matrix_form(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape an arbitrary gradient tensor into a 2-D matrix for SVD.

    Convolution weights (C_out, C_in, H, W) → (C_out, C_in * H * W).
    Vectors (d,) → (d, 1).
    """
    if tensor.dim() == 1:
        return tensor.float().unsqueeze(1)
    elif tensor.dim() == 2:
        return tensor.float()
    else:
        # Flatten all dims except the first
        return tensor.float().reshape(tensor.shape[0], -1)


def _count_svd_bytes(m: int, n: int, r: int) -> int:
    """
    Byte cost of transmitting a truncated SVD.
    Format: U (m×r), S (r,), Vt (r×n) in float32.
    """
    return (m * r + r + r * n) * 4


def _count_subspace_bytes(r: int) -> int:
    """
    Byte cost of transmitting subspace coefficients (r floats, float32).
    The basis B is shared and not re-transmitted every round.
    """
    return r * 4


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION 1: Hybrid Compression (Layer-Type-Aware)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_layer_type(name: str, tensor: torch.Tensor) -> str:
    """
    Detect layer type based on name and tensor shape for hybrid compression.

    Returns:
      - "embedding": large vocabulary matrices (vocab embeddings)
      - "norm":      normalization layers (BN, LN) with < 512 params
      - "conv":      convolutional layers (3D+ tensors)
      - "linear":    fully-connected layers (2D weight matrices)
      - "bias":      bias vectors (1D)
      - "other":     fallback category

    Args:
        name:   Layer parameter name (e.g., "model.embed.weight").
        tensor: Parameter gradient tensor.

    Returns:
        String label indicating layer type.
    """
    name_lower = name.lower()

    # Embedding detection: explicit name check or large 2D matrix (vocab size > 5000)
    if "embed" in name_lower:
        return "embedding"
    if tensor.dim() == 2 and tensor.shape[0] > 5000:
        return "embedding"

    # Normalization layers: small parameter count
    norm_keywords = ["norm", "bn", "ln", "batch_norm", "layer_norm"]
    if any(kw in name_lower for kw in norm_keywords) and tensor.numel() < 512:
        return "norm"

    # Bias vectors
    if "bias" in name_lower and tensor.dim() == 1:
        return "bias"

    # Convolutional layers (3D or 4D tensors)
    if tensor.dim() >= 3:
        return "conv"

    # Linear layers (2D weight matrices)
    if tensor.dim() == 2:
        return "linear"

    return "other"


def sparse_topk_compress(
    tensor: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a tensor by retaining only the top-k values by absolute magnitude.

    Args:
        tensor: Input tensor (arbitrary shape).
        ratio:  Fraction of elements to retain (e.g., 0.01 for 1%).

    Returns:
        (values, indices) where:
          - values: float32 tensor of shape (k,) — the top-k values
          - indices: int32 tensor of shape (k,) — their flattened indices
    """
    flat = tensor.flatten().float()
    k = max(1, int(ratio * flat.numel()))
    vals, idx = torch.topk(flat.abs(), k)
    # Keep original signs
    topk_values = flat[idx]
    return topk_values, idx.int()


def sparse_topk_decompress(
    values: torch.Tensor, indices: torch.Tensor, shape: list[int]
) -> torch.Tensor:
    """
    Reconstruct a tensor from sparse TopK representation.

    Args:
        values:  (k,) float32 — the non-zero values
        indices: (k,) int32 — their flattened indices
        shape:   Original tensor shape

    Returns:
        Reconstructed dense tensor of given shape.
    """
    numel = int(torch.prod(torch.tensor(shape)).item())
    flat = torch.zeros(numel, dtype=values.dtype, device=values.device)
    flat[indices.long()] = values.float()
    return flat.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION 2: Basis Memory Management
# ─────────────────────────────────────────────────────────────────────────────

class BasisMemoryManager:
    """
    Manages memory usage of stored subspace bases across layers.

    Policy:
      1. Age-based expiry: bases older than `max_age_rounds` rounds are evicted
      2. LRU eviction: when total memory exceeds `max_memory_bytes`, evict the
         least recently used (oldest basis_round) until within budget

    This is critical for large models (ResNet-50+, ImageNet) where storing
    full-rank bases for all layers can exceed client memory constraints.

    Usage:
        mgr = BasisMemoryManager(max_memory_mb=128, max_age_rounds=10)
        mgr.update(bases, basis_rounds, current_round)
        # bases and basis_rounds dicts are modified in place (evicted entries removed)
    """

    def __init__(self, max_memory_mb: float = 128.0, max_age_rounds: int = 20):
        """
        Args:
            max_memory_mb:   Maximum total memory (MB) for stored bases.
            max_age_rounds:  Maximum age in rounds before eviction.
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.max_age_rounds = max_age_rounds

    def _basis_bytes(self, basis: torch.Tensor) -> int:
        """Compute memory footprint of a single basis (float32)."""
        return basis.numel() * 4

    def total_memory_bytes(self, bases: dict) -> int:
        """Compute total memory used by all stored bases."""
        return sum(self._basis_bytes(b) for b in bases.values() if b is not None)

    def update(
        self, bases: dict, basis_rounds: dict, current_round: int
    ) -> int:
        """
        Evict stale and excess bases in-place.

        Args:
            bases:         dict[layer_name -> Tensor(m_dim, r)] — modified in place
            basis_rounds:  dict[layer_name -> int] — modified in place
            current_round: Current training round number

        Returns:
            Number of bases evicted.
        """
        evicted = 0

        # Step 1: Age-based eviction — remove bases older than max_age_rounds
        stale = [
            name
            for name, rnd in basis_rounds.items()
            if current_round - rnd > self.max_age_rounds
        ]
        for name in stale:
            bases.pop(name, None)
            basis_rounds.pop(name, None)
            evicted += 1

        # Step 2: LRU eviction — if still over memory budget, evict oldest first
        while self.total_memory_bytes(bases) > self.max_memory_bytes and bases:
            # Find least recently used (smallest basis_round value)
            oldest_name = min(basis_rounds, key=lambda n: basis_rounds[n])
            bases.pop(oldest_name, None)
            basis_rounds.pop(oldest_name, None)
            evicted += 1

        return evicted


# ─────────────────────────────────────────────────────────────────────────────
# Main algorithm
# ─────────────────────────────────────────────────────────────────────────────

@register_algorithm("fed_resonance")
class FedResonance(FLAlgorithm):
    """
    Fed-Resonance: Adaptive Spectral Compression with Battery-Aware Switching.

    Key properties:
      - Per-layer mode selection: SVD-truncated vs Subspace projection vs dense
      - Battery-proportional compression budget (inherited from E-CEFFL)
      - Subspace drift detection: switches to SVD when basis becomes stale
      - Adaptive rank selection: spectral energy threshold per layer
      - Error feedback across mode switches: no information permanently lost
      - Convergence: O(1/sqrt(T)) under standard FL assumptions

    State stored in ClientState.custom:
      - "subspace_bases": dict[layer_name -> Tensor(d_l, r)] — per-layer bases
      - "basis_round":    dict[layer_name -> int] — round when basis was last set

    Metadata returned per client:
      - mode_per_layer:   dict[layer_name -> "svd" | "subspace" | "dense"]
      - avg_rank:         mean rank used across compressible layers
      - avg_drift:        mean subspace drift across layers with stored bases
      - compression_ratio: bytes_sent / bytes_full_dense
    """

    name = "fed_resonance"
    description = (
        "Battery-aware per-layer hybrid spectral compression (SVD-truncated / "
        "Subspace projection) with error feedback. "
        "Nikiema & EL Amhoud, H. ELHAMMOUTI & I. KISSAMI,  UM6P 2026."
    )

    # ------------------------------------------------------------------
    # client_update
    # ------------------------------------------------------------------

    def client_update(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        state: ClientState,
        config: dict,
    ) -> tuple[dict, dict]:
        """
        Local training + per-layer hybrid spectral compression.

        Steps:
          1. Compute beta from battery (E-CEFFL formula).
          2. Local SGD for local_epochs.
          3. Compute delta = w_before - w_after.
          4. Add error feedback to delta.
          5. Per-layer hybrid compression (SVD vs Subspace vs dense).
          6. Store residual in error buffer.
          7. Energy accounting and battery update.
        """

        device     = config.get("device", "cpu")
        lr         = config.get("lr", 0.01)
        local_epochs = config.get("local_epochs", 1)
        warmup_rounds = int(config.get("warmup_rounds", 0))
        is_warmup  = state.round_num < warmup_rounds
        beta_min   = config.get("beta_min", 0.01)
        beta_max   = config.get("beta_max", 1.0)
        battery_max_j = config.get("battery_max_j", 185400.0)
        rank_min   = int(config.get("rank_min", 4))
        rank_max   = int(config.get("rank_max", 32))
        eps        = float(config.get("spectral_energy_thresh", 0.95))
        tau_drift  = float(config.get("subspace_drift_thresh", 0.1))
        tau_rank   = float(config.get("rank_coherence_thresh", 0.7))
        use_ef     = bool(config.get("use_error_feedback", True))
        use_rsvd   = bool(config.get("use_rsvd", True))
        rsvd_oversample = int(config.get("rsvd_oversample", 10))
        rsvd_power_iter = int(config.get("rsvd_power_iter", 2))
        profile    = config.get("device_profile")

        # ── Q3: Rank selection criterion ───────────────────────────────────
        rank_criterion     = str(config.get("rank_criterion", "energy"))
        rank_budget_bytes  = int(config.get("rank_budget_bytes", 0))

        # ── Q4: Soft projection ────────────────────────────────────────────
        use_soft_proj      = bool(config.get("use_soft_projection", False))
        soft_proj_adaptive = bool(config.get("soft_projection_adaptive", False))
        soft_proj_alpha    = float(config.get("soft_projection_alpha", 0.8))
        soft_proj_alpha_start = float(config.get("soft_projection_alpha_start", 0.5))
        soft_proj_alpha_end   = float(config.get("soft_projection_alpha_end", 0.95))
        total_rounds       = int(config.get("num_rounds", 100))

        # Compute alpha for this round (used only when use_soft_proj=True)
        if use_soft_proj and soft_proj_adaptive:
            # Linear schedule: alpha_start at round 0, alpha_end at total_rounds
            t_frac = min(1.0, state.round_num / max(total_rounds - 1, 1))
            soft_proj_alpha = (
                soft_proj_alpha_start
                + t_frac * (soft_proj_alpha_end - soft_proj_alpha_start)
            )

        # ── Dead-client guard ──────────────────────────────────────────────
        # A client with zero battery cannot train. Return immediately with a
        # skipped=True marker so the server excludes this client from the
        # aggregation weights and participation metrics.
        if state.battery_j <= 0:
            dataset_size = len(dataloader.dataset) if dataloader is not None else 1
            return {}, {
                "client_id":            state.client_id,
                "round_num":            state.round_num,
                "beta_actual":          0.0,
                "battery_j_remaining":  0.0,
                "energy_j_consumed":    0.0,
                "bytes_sent":           0,
                "bytes_received":       0,
                "local_loss":           0.0,
                "compression_ratio":    0.0,
                "avg_rank":             0.0,
                "avg_drift":            0.0,
                "mode_per_layer":       {},
                "dataset_size":         dataset_size,
                "battery_critical":     False,
                "basis_evictions":      0,
                "skipped":              True,
            }

        # ── Step 1: Battery → beta ─────────────────────────────────────────
        battery_j = state.battery_j
        beta = beta_min + (beta_max - beta_min) * (battery_j / battery_max_j)
        beta = max(beta_min, min(beta_max, beta))
        battery_critical = beta < 0.1

        # ── Step 2: Save weights before training ───────────────────────────
        w_before = OrderedDict(
            {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        )

        # ── Step 3: Local training ─────────────────────────────────────────
        model.train()
        model.to(device)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        optimizer_type = config.get("optimizer", "sgd").lower()
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        num_batches = 0

        for _ in range(local_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        # ── Step 4: Compute delta = w_before - w_after ─────────────────────
        w_after = model.state_dict()
        delta = OrderedDict({
            k: (w_before[k] - w_after[k].cpu()).float()
            for k in w_before
        })
        del w_before
        del w_after
        gc.collect()

        # ── Step 5: Initialize persistent state ───────────────────────────
        if state.error_buffer is None and use_ef:
            state.error_buffer = {k: torch.zeros_like(v) for k, v in delta.items()}

        if "subspace_bases" not in state.custom:
            state.custom["subspace_bases"] = {}   # layer_name -> Tensor(d_l, r)
        if "basis_round" not in state.custom:
            state.custom["basis_round"] = {}      # layer_name -> int

        bases    = state.custom["subspace_bases"]
        b_rounds = state.custom["basis_round"]
        current_round = state.round_num

        # ── Step 6: Per-layer hybrid compression (or warmup dense pass) ───────
        update = OrderedDict()      # what is transmitted to server
        new_error = OrderedDict()
        mode_per_layer: dict[str, str] = {}
        total_bytes_sent = 0
        total_bytes_dense = 0
        ranks_used: list[int] = []
        drifts: list[float] = []

        for name, g in delta.items():
            # Warmup: transmit full delta without any compression (= FedAvg).
            # Bases and error buffers stay uninitialised until warmup ends.
            if is_warmup:
                n_bytes = g.numel() * 4
                total_bytes_dense += n_bytes
                total_bytes_sent  += n_bytes
                mode_per_layer[name] = "dense"
                update[name] = {"mode": "dense", "data": g}
                if use_ef:
                    new_error[name] = torch.zeros_like(g)
                continue

            # Add error feedback
            if use_ef and state.error_buffer is not None:
                u = g + state.error_buffer[name]
            else:
                u = g

            # ── Layer type detection for hybrid compression ────────────────
            layer_type = _detect_layer_type(name, u)

            # ── Decide mode for this layer ─────────────────────────────────
            numel = u.numel()
            dense_bytes = numel * 4
            total_bytes_dense += dense_bytes

            # EXTENSION 1: Handle embedding layers with sparse TopK compression
            if layer_type == "embedding":
                topk_ratio = config.get("embedding_topk_ratio", 0.01)
                values, indices = sparse_topk_compress(u, topk_ratio)
                k = values.numel()
                sparse_bytes = k * 8  # 4 bytes index + 4 bytes value
                total_bytes_sent += sparse_bytes
                residual = u.flatten().clone()
                residual[indices.long()] = 0.0
                residual = residual.reshape(u.shape)
                update[name] = {
                    "mode": "sparse",
                    "values": values,
                    "indices": indices,
                    "shape": list(u.shape),
                }
                mode_per_layer[name] = "sparse"
                if use_ef:
                    new_error[name] = residual.clone()
                continue  # skip the existing SVD/subspace logic

            # EXTENSION 1: Handle norm/bias layers with dense transmission
            elif layer_type in ("norm", "bias"):
                # Always dense for tiny layers — overhead of SVD is not worth it
                update[name] = {"mode": "dense", "data": u.clone()}
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                if use_ef:
                    new_error[name] = torch.zeros_like(u)
                continue

            # Existing logic for conv/linear layers
            if numel < rank_min * 2:
                # Layer too small — always dense
                mode = "dense"
            elif battery_critical:
                # Battery critically low — maximum compression via SVD rank_min
                mode = "svd_forced"
            else:
                # Check subspace viability
                if name in bases and bases[name] is not None:
                    basis_old = bases[name]
                    # Proxy: how well does the stored basis explain this gradient?
                    # The basis B has shape (m_dim, r) — it lives in the row space
                    # of _matrix_form(u), which has shape (m_dim, n_dim).
                    # We must project in the same (m_dim, n_dim) space, NOT in
                    # the fully-flattened (d,) space, to avoid the dimension
                    # mismatch that occurs when numel(u) != m_dim.
                    G_proxy = _matrix_form(u)           # (m_dim, n_dim)
                    B = basis_old.float()               # (m_dim, r)
                    # Guard: basis rows must match current layer's m_dim.
                    # If the model was re-initialized or the layer shape changed
                    # between rounds the stored basis is stale — fall back to SVD.
                    if B.shape[0] != G_proxy.shape[0]:
                        # Basis is stale (shape mismatch) — fall back to SVD
                        mode = "svd"
                    else:
                        g_norm_sq = G_proxy.pow(2).sum().item()
                        if g_norm_sq < 1e-10:
                            # Risk 3 fix: gradient near zero (convergence plateau) →
                            # proj_energy numerically unstable → force SVD to refresh basis.
                            mode = "svd"
                        else:
                            # Project each column of G onto the column space of B:
                            # alpha_cols = B^T G  (r, n_dim)
                            # G_approx   = B alpha_cols  (m_dim, n_dim)
                            alpha_cols = B.t() @ G_proxy            # (r, n_dim)
                            G_approx_proxy = B @ alpha_cols         # (m_dim, n_dim)
                            proj_energy = (
                                G_approx_proxy.pow(2).sum() / g_norm_sq
                            ).item()
                            # proj_energy >= tau_rank → basis explains gradient well
                            if proj_energy >= tau_rank:
                                mode = "subspace"
                                drifts.append(1.0 - proj_energy)
                            else:
                                mode = "svd"
                else:
                    # No basis stored yet — use SVD to build the basis
                    mode = "svd"

            # ── Compress according to mode ─────────────────────────────────
            if mode == "dense":
                v = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                update[name] = {"mode": "dense", "data": v}

            elif mode in ("svd", "svd_forced"):
                G = _matrix_form(u)
                m_dim, n_dim = G.shape

                if mode == "svd_forced":
                    r = rank_min
                else:
                    # Q3: pass rank_criterion and rank_budget_bytes through
                    r = adaptive_rank(
                        G, eps, use_rsvd, rsvd_oversample, rsvd_power_iter,
                        rank_criterion=rank_criterion,
                        rank_budget_bytes=rank_budget_bytes,
                        rank_min=rank_min,
                    )
                    r = max(rank_min, min(rank_max, r))

                ranks_used.append(r)
                U, S, Vt = svd_compress(G, r, use_rsvd, rsvd_oversample, rsvd_power_iter)
                G_approx = svd_decompress(U, S, Vt)

                # Q4: soft projection in SVD mode
                # Hard projection: v = G_approx (= U S Vt, the low-rank approx)
                # Soft projection: v = alpha * G_approx + (1 - alpha) * G
                # EF residual is always: original u - v  (what we DID NOT send)
                # With hard projection: residual = u - G_approx  (captures everything outside subspace)
                # With soft projection: residual = u - (alpha*G_approx + (1-alpha)*G)
                #                               = (1 - (1-alpha)) * (u - G_approx) * ... no:
                #                               = u - alpha*G_approx - (1-alpha)*G
                #                               = alpha * (u - G_approx)
                #                               = alpha * (G - G_approx)  [since u reshaped = G in same space]
                # So soft projection shrinks the EF residual by alpha.
                # The bias of hard projection P(E[g]) != E[g] is reduced because
                # the update v = alpha*proj(u) + (1-alpha)*u always keeps a (1-alpha)
                # fraction of the full gradient, limiting worst-case bias.
                if use_soft_proj:
                    G_soft = soft_proj_alpha * G_approx + (1.0 - soft_proj_alpha) * G
                    v = G_soft.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape) * soft_proj_alpha
                else:
                    v = G_approx.reshape(u.shape)
                    residual = u - v

                # Update the shared subspace basis for this layer.
                # Store U (m_dim, r) — left singular vectors — as the basis
                # for subsequent subspace-mode rounds.
                new_basis = U.contiguous()   # (m_dim, r)
                bases[name] = new_basis
                b_rounds[name] = current_round

                svd_bytes = _count_svd_bytes(m_dim, n_dim, r)
                total_bytes_sent += svd_bytes
                mode_tag = "svd" if mode == "svd" else "svd_forced"
                mode_per_layer[name] = mode_tag
                update[name] = {
                    "mode": mode_tag,
                    "U": U, "S": S, "Vt": Vt,
                    "shape": list(u.shape),
                }

            elif mode == "subspace":
                B = bases[name].float()   # (m_dim, r)
                G = _matrix_form(u)       # (m_dim, n_dim)

                # Project in matrix space: alpha_cols = B^T G  shape (r, n_dim)
                # Reconstruct:             G_approx   = B alpha_cols  (m_dim, n_dim)
                # Residual:                G - G_approx
                #
                # This is consistent with the stored basis shape (m_dim, r) and
                # avoids the dimension mismatch that arises when projecting the
                # fully-flattened gradient (length d = m_dim * n_dim) against B.
                alpha_cols = B.t() @ G                          # (r, n_dim)
                G_approx = B @ alpha_cols                       # (m_dim, n_dim)

                # Q4: soft projection in subspace mode (same logic as SVD mode)
                if use_soft_proj:
                    G_soft = soft_proj_alpha * G_approx + (1.0 - soft_proj_alpha) * G
                    v = G_soft.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape) * soft_proj_alpha
                else:
                    v = G_approx.reshape(u.shape)
                    residual = (G - G_approx).reshape(u.shape)

                r_used = B.shape[1]
                n_dim_sub = G.shape[1]
                ranks_used.append(r_used)
                # Byte cost: r * n_dim floats (still << m_dim * n_dim for r << m_dim)
                sub_bytes = r_used * n_dim_sub * 4
                total_bytes_sent += sub_bytes
                mode_per_layer[name] = "subspace"
                update[name] = {
                    "mode": "subspace",
                    "alpha_cols": alpha_cols,       # (r, n_dim) — renamed from alpha
                    "basis_round": b_rounds[name],
                    "shape": list(u.shape),
                    "layer_name": name,
                }
            else:
                # Fallback (should not be reached)
                v = u.clone()
                residual = torch.zeros_like(u)
                mode_per_layer[name] = "dense"
                total_bytes_sent += dense_bytes
                update[name] = {"mode": "dense", "data": v}

            # Store residual in new error buffer
            if use_ef:
                new_error[name] = residual.clone()

        if use_ef:
            state.error_buffer = new_error

        # EXTENSION 2: Apply basis memory policy (important for large models)
        max_basis_mb = float(config.get("max_basis_memory_mb", 128.0))
        basis_max_age = int(config.get("basis_max_age_rounds", 20))
        basis_mgr = BasisMemoryManager(max_basis_mb, basis_max_age)
        n_evicted = basis_mgr.update(bases, b_rounds, current_round)

        state.custom["subspace_bases"] = bases
        state.custom["basis_round"] = b_rounds

        # ── Step 7: Energy accounting ──────────────────────────────────────
        num_params    = sum(p.numel() for p in model.parameters())
        dataset_size  = len(dataloader.dataset)
        batch_size    = dataloader.batch_size
        downlink_bytes = total_bytes_dense   # server sends full weights

        if profile is not None:
            # fed_resonance trains the full model — compression lives on the
            # upload, not on the backward. trainable_names = all parameters.
            trainable_names = [n for n, _ in model.named_parameters()]
            flops = round_compute_flops(
                model, trainable_names, config,
                profile, dataloader, local_epochs,
            )
            _bd = profile.round_energy_breakdown(
                flops,
                total_bytes_sent,
                downlink_bytes,
                config.get("energy_scale_factor", 1.0),
                config.get("alpha_applies_to", "compute"),
            )
        else:
            actual_beta_proxy = total_bytes_sent / max(total_bytes_dense, 1)
            _e = (0.5 + 2.0 * actual_beta_proxy) * config.get(
                "energy_scale_factor", 1.0
            )
            _bd = {"compute": _e, "uplink": 0.0, "downlink": 0.0, "total": _e}

        energy_j = _bd["total"]
        state.battery_j = max(0.0, state.battery_j - energy_j)
        state.round_num += 1

        del optimizer
        compression_ratio = total_bytes_sent / max(total_bytes_dense, 1)
        avg_rank = float(sum(ranks_used) / len(ranks_used)) if ranks_used else 0.0
        avg_drift = float(sum(drifts) / len(drifts)) if drifts else 0.0

        metadata = {
            "client_id":            state.client_id,
            "round_num":            state.round_num,
            "beta_actual":          beta,
            "battery_j_remaining":  state.battery_j,
            "energy_j_consumed":    energy_j,
            "energy_compute_j":     _bd["compute"],
            "energy_uplink_j":      _bd["uplink"],
            "energy_downlink_j":    _bd["downlink"],
            "bytes_sent":           total_bytes_sent,
            "bytes_received":       downlink_bytes,
            "local_loss":           total_loss / max(num_batches, 1),
            "compression_ratio":    compression_ratio,
            "mode_per_layer":       mode_per_layer,
            "avg_rank":             avg_rank,
            "avg_drift":            avg_drift,
            "dataset_size":         dataset_size,
            "battery_critical":     battery_critical,
            "basis_evictions":      n_evicted,
        }

        return dict(update), metadata

    # ------------------------------------------------------------------
    # server_aggregate
    # ------------------------------------------------------------------

    def server_aggregate(
        self,
        global_model: nn.Module,
        client_updates: list[tuple[dict, dict, ClientState]],
        round_num: int,
        config: dict,
    ) -> AggregateResult:
        """
        Server-side aggregation with multi-mode reconstruction.

        For each client and each layer:
          - mode "dense":    update already has the dense gradient vector.
          - mode "svd" / "svd_forced": reconstruct from (U, S, Vt).
          - mode "subspace": reconstruct from (alpha, basis stored in client state).

        After reconstruction, apply weighted FedAvg:
          w_{t+1} = w_t - sum_k (n_k / n) * v_k

        The global subspace is updated by SVD on the aggregated dense gradient
        (optional, used by clients in subsequent rounds if transmitted).
        """

        K = len(client_updates)
        global_sd = global_model.state_dict()

        # ── Server-side basis cache (Fix risk 1) ──────────────────────────
        # The server maintains its own copy of each client's subspace basis,
        # extracted from SVD updates (U is already transmitted — zero extra bytes).
        # This makes subspace mode FL-real: no dependency on cstate.custom.
        # {client_id: {layer_name: (U_tensor, last_round)}}
        if not hasattr(self, "_server_state"):
            self._server_state = {}
        if "basis_cache" not in self._server_state:
            self._server_state["basis_cache"] = {}
        basis_cache = self._server_state["basis_cache"]

        # Age-based eviction (fix risk 4): remove cached bases older than
        # basis_max_age_rounds to bound server memory at O(K × active_layers × m × r).
        basis_max_age = config.get("basis_max_age_rounds", 20)
        for cid in list(basis_cache.keys()):
            basis_cache[cid] = {
                layer: (B, rnd)
                for layer, (B, rnd) in basis_cache[cid].items()
                if round_num - rnd <= basis_max_age
            }

        # ── Separate alive (active) from dead (skipped) clients ───────────
        active_updates = [
            (u, m, s) for u, m, s in client_updates
            if not m.get("skipped", False) and u
        ]
        K_act = max(len(active_updates), 1)

        # Dataset-size weights only over active clients (FedAvg-style)
        sizes = [m.get("dataset_size", 1) for _, m, _ in active_updates]
        total_n = max(sum(sizes), 1)
        weights = [n_k / total_n for n_k in sizes]

        # Accumulate weighted reconstructed gradients
        agg: Optional[OrderedDict] = None

        for (update, metadata, cstate), w_k in zip(active_updates, weights):
            # Reconstruct dense gradients from compressed updates
            dense = OrderedDict()

            client_id = metadata.get("client_id", id(cstate))
            if client_id not in basis_cache:
                basis_cache[client_id] = {}
            client_bases = basis_cache[client_id]  # server-side cache (FL-real)

            for name, packed in update.items():
                mode = packed["mode"]

                if mode == "dense":
                    dense[name] = packed["data"].float()

                elif mode in ("svd", "svd_forced"):
                    U   = packed["U"].float()
                    S   = packed["S"].float()
                    Vt  = packed["Vt"].float()
                    shape = packed["shape"]
                    G_approx = svd_decompress(U, S, Vt)
                    dense[name] = G_approx.reshape(shape).float()
                    # Cache U as basis for future subspace reconstruction.
                    # U is already transmitted — zero extra bytes needed.
                    client_bases[name] = (U.contiguous(), round_num)

                elif mode == "sparse":
                    values = packed["values"].float()
                    indices = packed["indices"].long()
                    shape = packed["shape"]
                    dense[name] = sparse_topk_decompress(values, indices, shape)

                elif mode == "subspace":
                    # alpha_cols: (r, n_dim) — only coefficients, no basis transmitted.
                    # Server reconstructs using its cached basis (extracted from last
                    # SVD update for this client/layer).
                    alpha_cols = packed["alpha_cols"].float()
                    shape = packed["shape"]
                    if name in client_bases:
                        B, _ = client_bases[name]
                        B = B.float()                           # (m_dim, r)
                        G_approx = B @ alpha_cols               # (m_dim, n_dim)
                        dense[name] = G_approx.reshape(shape).float()
                    else:
                        # Basis not yet cached (first subspace attempt without prior SVD)
                        # — treat as zero update; client will fall back to SVD next round.
                        dense[name] = torch.zeros(shape, dtype=torch.float32)

                else:
                    ref = global_sd.get(name)
                    if ref is not None:
                        dense[name] = torch.zeros_like(ref, dtype=torch.float32)

            # Weighted accumulation
            if agg is None:
                agg = OrderedDict({k: v.clone() * w_k for k, v in dense.items()})
            else:
                for k in dense:
                    if k in agg:
                        agg[k] += dense[k] * w_k
                    else:
                        agg[k] = dense[k] * w_k

        # Apply aggregated gradient: w_{t+1} = w_t - agg
        new_weights = OrderedDict()
        for k in global_sd:
            if agg is not None and k in agg:
                new_weights[k] = global_sd[k].float() - agg[k].to(global_sd[k].device)
            else:
                new_weights[k] = global_sd[k].float()
        del agg
        gc.collect()

        # ── Round-level metrics ────────────────────────────────────────────
        total_bytes  = sum(m["bytes_sent"]        for _, m, _ in client_updates)
        total_energy = sum(m["energy_j_consumed"] for _, m, _ in client_updates)
        avg_battery  = sum(s.battery_j            for _, _, s in client_updates) / K
        batt_min     = min(s.battery_j            for _, _, s in client_updates)
        # Averages over active clients only (dead clients contribute zeros — biases)
        avg_beta     = sum(m["beta_actual"]               for _, m, _ in active_updates) / K_act
        avg_loss     = sum(m["local_loss"]                for _, m, _ in active_updates) / K_act
        avg_rank     = sum(m.get("avg_rank", 0.0)         for _, m, _ in active_updates) / K_act
        avg_drift    = sum(m.get("avg_drift", 0.0)        for _, m, _ in active_updates) / K_act
        avg_cr       = sum(m.get("compression_ratio", 1.0) for _, m, _ in active_updates) / K_act

        # Mode distribution across active clients and layers
        mode_counts: dict[str, int] = {
            "svd": 0, "svd_forced": 0, "subspace": 0, "dense": 0, "sparse": 0
        }
        for _, m, _ in active_updates:
            for mode in m.get("mode_per_layer", {}).values():
                mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Jain fairness index on bytes_sent (identical formula to fedpart_be)
        # Dead clients contribute 0 bytes → penalised correctly.
        bytes_sent = [m["bytes_sent"] for _, m, _ in client_updates]
        if sum(bytes_sent) > 0:
            jain = (sum(bytes_sent) ** 2) / (K * sum(b ** 2 for b in bytes_sent))
        else:
            jain = 1.0

        # Total basis evictions across all active clients (EXTENSION 2)
        total_evictions = sum(m.get("basis_evictions", 0) for _, m, _ in active_updates)

        metrics = {
            "round":                 round_num,
            "total_bytes_sent":      total_bytes,
            "total_energy_j":        total_energy,
            "avg_beta":              avg_beta,
            "avg_battery_j":         avg_battery,
            "batt_min_j":            batt_min,
            "avg_local_loss":        avg_loss,
            "avg_rank":              avg_rank,
            "avg_drift":             avg_drift,
            "avg_compression_ratio": avg_cr,
            "compression_ratio":     avg_cr,
            "mode_counts":           mode_counts,
            "participation_rate":    len(active_updates) / K,
            "jain_index":            jain,
            "num_clients":           K,
            "skipped_clients":       K - len(active_updates),
            "basis_evictions":       total_evictions,
        }

        return AggregateResult(new_weights=new_weights, metrics=metrics)

    # ------------------------------------------------------------------
    # get_default_config
    # ------------------------------------------------------------------

    def get_default_config(self) -> dict:
        return {
            # Training
            "lr":                   0.01,
            "local_epochs":         1,
            "batch_size":           32,
            "optimizer":            "sgd",
            "momentum":             0.9,
            "weight_decay":         1e-4,
            # SVD parameters
            "rank_min":             4,      # minimum truncated SVD rank
            "rank_max":             32,     # maximum truncated SVD rank
            "spectral_energy_thresh": 0.95, # fraction of spectral energy for adaptive rank
            "use_rsvd":             True,   # use randomized SVD (faster for large matrices)
            "rsvd_oversample":      10,     # oversampling parameter p for rSVD accuracy
            "rsvd_power_iter":      2,      # power iterations q for rSVD accuracy
            # Switch criteria
            "subspace_drift_thresh": 0.1,   # tau_drift: max normalized Frobenius drift
            "rank_coherence_thresh": 0.7,   # tau_rank: min subspace alignment quality
            # Battery model (E-CEFFL formula)
            "beta_min":             0.01,
            "beta_max":             1.0,
            "battery_max_j":        185400.0,   # 10Ah @ 5.1V (RPi4 powerbank)
            # Error feedback
            "use_error_feedback":   True,
            # Device
            "device":               "cpu",
            "device_profile":       None,
            # EXTENSION 1: Hybrid compression parameters
            "embedding_topk_ratio": 0.01,   # fraction of embedding params to keep (1% default)
            # EXTENSION 2: Basis memory management (for large models like ResNet-50+)
            "max_basis_memory_mb":  128.0,  # max RAM for stored bases (per client)
            "basis_max_age_rounds": 20,     # evict bases unused for > N rounds
            # Q3: Rank selection criterion
            # "energy"  — existing behavior: smallest r capturing >= spectral_energy_thresh
            # "elbow"   — Kneedle elbow on cumulative energy curve (no threshold to tune)
            # "budget"  — maximum r that fits in rank_budget_bytes for this layer
            "rank_criterion":       "energy",
            "rank_budget_bytes":    0,       # byte budget per layer (used only for "budget")
            # Q4: Soft projection (reduces hard-projection bias in non-IID FL)
            # Off by default for backward compatibility.
            # Only applied to SVD and subspace modes (not dense, not sparse).
            "use_soft_projection":          False,
            "soft_projection_alpha":        0.8,    # fixed alpha when adaptive=False
            "soft_projection_adaptive":     False,  # if True, alpha is scheduled over rounds
            "soft_projection_alpha_start":  0.5,    # alpha at round 0 (adaptive schedule)
            "soft_projection_alpha_end":    0.95,   # alpha at num_rounds (adaptive schedule)
            "num_rounds":                   100,    # total rounds (needed for adaptive schedule)
        }
