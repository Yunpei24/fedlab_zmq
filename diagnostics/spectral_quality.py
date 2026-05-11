"""
diagnostics/spectral_quality.py
================================
Spectral quality metrics for Fed-Resonance gradient analysis.
J. Nikiema, EL Amhoud — UM6P, 2026

Public API
----------
  spectral_entropy(singular_values) -> float
      Shannon entropy of the normalized singular value spectrum.

  layer_spectral_entropy(tensor, use_rsvd, rsvd_oversample, rsvd_power_iter) -> float
      Convenience wrapper: computes SVD of a raw 2-D tensor and returns entropy.

  round_spectral_entropy(delta, use_rsvd, rsvd_oversample, rsvd_power_iter) -> dict
      Compute per-layer spectral entropy for a full gradient dict (OrderedDict of tensors).

Design notes
------------
spectral_alignment was considered but NOT implemented here because the existing
`proj_energy` metric (lines 771-774 of algorithms/fed_resonance.py) already
measures one-sided subspace energy capture (||B B^T G||_F^2 / ||G||_F^2), which
is the correct quantity for the stored left-singular-vector basis. The two-sided
version proposed by the user would require also storing Vt per layer, doubling
basis memory cost, for a marginal diagnostic improvement.

spectral_entropy is genuinely new: it measures HOW CONCENTRATED the singular
value spectrum is, not how well a specific basis captures it. These are
orthogonal diagnostics:
  - proj_energy: is THIS stored basis still valid?
  - spectral_entropy: is ANY low-rank basis valid for this layer?

Interpretation
--------------
  Low entropy (near 0):
    - Spectrum dominated by 1-2 singular values
    - Gradient is highly low-rank
    - SVD compression loses almost nothing
    - The chosen rank is well-justified

  High entropy (max = log(min(m,n))):
    - Energy spread evenly across many singular values
    - No low-rank structure exists
    - Compression will incur significant bias
    - Should use dense transmission or higher rank

  In non-IID FL: high entropy often correlates with high client heterogeneity
  for that layer — useful for diagnosing which layers are hardest to compress.
"""

import math
import torch

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def spectral_entropy(singular_values: torch.Tensor) -> float:
    """
    Shannon entropy of the normalized squared singular value distribution.

    Formally, given singular values sigma_1 >= sigma_2 >= ... >= sigma_r > 0:

        p_i = sigma_i^2 / sum_j sigma_j^2        (normalized spectral energy)
        H   = -sum_i p_i * log(p_i + epsilon)    (entropy, nats)

    Range:
        H = 0         when exactly one sigma is non-zero (rank-1 matrix)
        H = log(r)    when all sigmas are equal (maximally diffuse spectrum)

    Args:
        singular_values: 1-D tensor of non-negative singular values, in any order.
                         Zero values are handled numerically via the epsilon term.

    Returns:
        Entropy in nats as a Python float. Returns 0.0 if all values are zero.
    """
    sv = singular_values.float()

    # Filter out numerical zeros (SVD can produce tiny negatives due to float errors)
    sv = sv[sv > 0.0]
    if sv.numel() == 0:
        return 0.0

    energy = sv.pow(2)
    total = energy.sum()
    if total.item() < 1e-12:
        return 0.0

    p = energy / total                                        # normalized, sum=1
    entropy = -(p * torch.log(p + 1e-10)).sum()
    return float(entropy.item())


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def layer_spectral_entropy(
    tensor: torch.Tensor,
    use_rsvd: bool = False,
    rsvd_oversample: int = 10,
    rsvd_power_iter: int = 2,
) -> float:
    """
    Compute spectral entropy of a raw 2-D gradient tensor.

    Reshapes the tensor to 2-D (same convention as _matrix_form in fed_resonance),
    computes singular values, and returns spectral_entropy.

    Args:
        tensor:          Gradient tensor of any shape (1-D, 2-D, or higher).
        use_rsvd:        If True, use randomized SVD to estimate singular values
                         (faster for large matrices but approximate). Default False
                         because for diagnostics we prefer exact values.
        rsvd_oversample: Oversampling for rSVD (ignored if use_rsvd=False).
        rsvd_power_iter: Power iterations for rSVD (ignored if use_rsvd=False).

    Returns:
        Spectral entropy (nats) as a float. Returns 0.0 on failure.
    """
    try:
        t = tensor.float()
        # Reshape to 2-D (same as _matrix_form)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.dim() > 2:
            t = t.reshape(t.shape[0], -1)

        if use_rsvd:
            # Use randomized SVD for large matrices; only need singular values
            # We compute a fixed-rank approximation at min(m,n)/4 to cover most energy
            m, n = t.shape
            r_approx = max(4, min(m, n) // 4)
            r_approx = min(r_approx, min(m, n) - 1)
            # Inline rSVD (avoid circular import from algorithms.fed_resonance)
            l = r_approx + rsvd_oversample
            if l >= min(m, n):
                sv = torch.linalg.svdvals(t)
            else:
                Omega = torch.randn(n, l, dtype=t.dtype, device=t.device)
                Y = t @ Omega
                for _ in range(rsvd_power_iter):
                    Y = t @ (t.t() @ Y)
                Q, _ = torch.linalg.qr(Y)
                B = Q.t() @ t
                sv = torch.linalg.svdvals(B)
        else:
            sv = torch.linalg.svdvals(t)

        return spectral_entropy(sv)

    except Exception:
        return 0.0


def round_spectral_entropy(
    delta: dict,
    use_rsvd: bool = False,
    rsvd_oversample: int = 10,
    rsvd_power_iter: int = 2,
    layer_types_to_skip: Optional[tuple] = None,
) -> dict:
    """
    Compute per-layer spectral entropy for a full gradient dict.

    Typically called once per round with the client's delta dict (w_before - w_after)
    for diagnostic purposes. This is intentionally NOT called in the hot path of
    client_update — it should be called separately when diagnostics are needed.

    Args:
        delta:                   OrderedDict[name -> Tensor] of gradient tensors.
        use_rsvd:                Use rSVD for large layers (faster, approximate).
        rsvd_oversample:         rSVD oversampling parameter.
        rsvd_power_iter:         rSVD power iterations.
        layer_types_to_skip:     Tuple of layer name substrings to skip (e.g.,
                                 ("bias", "norm", "bn", "ln")). If None, processes
                                 all layers.

    Returns:
        dict[layer_name -> float]: per-layer spectral entropy values.
        Also contains "__mean__" and "__max__" aggregate keys.

    Example:
        entropies = round_spectral_entropy(delta)
        print(entropies["__mean__"])  # average entropy across all layers
    """
    if layer_types_to_skip is None:
        layer_types_to_skip = ("bias", "norm", "bn", "ln", "batch_norm", "layer_norm")

    result = {}

    for name, tensor in delta.items():
        # Skip tiny or norm/bias layers (same logic as _detect_layer_type)
        name_lower = name.lower()
        if any(kw in name_lower for kw in layer_types_to_skip):
            continue
        if tensor.numel() < 8:
            continue

        h = layer_spectral_entropy(
            tensor, use_rsvd, rsvd_oversample, rsvd_power_iter
        )
        result[name] = h

    if result:
        vals = list(result.values())
        result["__mean__"] = sum(vals) / len(vals)
        result["__max__"] = max(vals)
        result["__min__"] = min(vals)
    else:
        result["__mean__"] = 0.0
        result["__max__"] = 0.0
        result["__min__"] = 0.0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rank selection helpers (used by algorithms/fed_resonance.py Q3 additions)
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_elbow(singular_values: torch.Tensor, rank_min: int = 1) -> int:
    """
    Select rank via the elbow (knee) method on the singular value curve.

    The elbow is the index where the marginal gain in captured energy drops
    most sharply — i.e., the point of maximum curvature of the cumulative
    energy curve. We approximate this by finding the index that maximizes the
    perpendicular distance from the line connecting the first and last points
    of the cumulative energy curve (Kneedle algorithm).

    Formally:
        Let E_i = cumsum(sigma_j^2)[i] / total_energy   (normalized cumsum)
        Let t_i = i / (r-1)                             (normalized x-axis)
        Line from (0, E_0) to (1, E_{r-1}) parameterized as L(t) = E_0 + t*(E_{r-1} - E_0)
        Knee index k* = argmax_i |E_i - L(t_i)|

    Failure modes in FL:
      - Very flat spectra (many equal singular values): elbow is ill-defined;
        the method may select a very large rank. Clip to rank_max externally.
      - Very steep spectra (rank-1 concentration): elbow is at i=0, method
        correctly returns rank 1.
      - Noise floor: small trailing singular values can create a second elbow.
        The Kneedle approach finds the FIRST and most prominent knee.

    Args:
        singular_values: 1-D tensor of non-negative singular values (descending).
        rank_min:        Minimum rank to return (default 1).

    Returns:
        Elbow rank as an int, always >= rank_min.
    """
    sv = singular_values.float()
    sv = sv[sv > 0.0]
    n = sv.numel()

    if n <= 1:
        return max(rank_min, 1)

    energy = sv.pow(2)
    cumsum = energy.cumsum(0)
    total = cumsum[-1].item()
    if total < 1e-12:
        return max(rank_min, 1)

    # Normalized cumulative energy curve: values in [0, 1]
    E = cumsum / total   # shape (n,)

    # Normalized x-axis: values in [0, 1]
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=sv.device)

    # Line from first to last point of E curve
    E_start = E[0].item()
    E_end = E[-1].item()   # should be 1.0
    # Line: L(t) = E_start + t * (E_end - E_start)
    L = E_start + t * (E_end - E_start)

    # Perpendicular distance from each point to the line
    # For a curve vs line in [0,1]x[0,1], the signed difference is sufficient
    # (Kneedle uses the normalized difference)
    diff = (E - L).abs()

    knee_idx = int(diff.argmax().item())

    # Return 1-indexed rank (knee_idx is 0-indexed)
    return max(rank_min, knee_idx + 1)


def rank_by_budget(
    singular_values: torch.Tensor,
    m: int,
    n: int,
    budget_bytes: int,
    rank_min: int = 1,
) -> int:
    """
    Select the maximum rank that fits within a byte budget for SVD transmission.

    SVD of a (m, n) matrix at rank r costs: (m*r + r + r*n) * 4 bytes
    = r * (m + 1 + n) * 4 bytes.

    Solving for r:
        r = floor(budget_bytes / (4 * (m + 1 + n)))

    Args:
        singular_values: Not used for the computation, but passed for interface
                         consistency. May be used in future for energy-clipping.
        m:               Number of rows of the gradient matrix.
        n:               Number of columns.
        budget_bytes:    Byte budget for this layer's SVD transmission.
        rank_min:        Minimum rank to return regardless of budget.

    Returns:
        Budget-derived rank, clipped to [rank_min, min(m, n)].
    """
    if budget_bytes <= 0:
        return rank_min

    bytes_per_rank_unit = 4 * (m + 1 + n)
    if bytes_per_rank_unit <= 0:
        return rank_min

    r = budget_bytes // bytes_per_rank_unit
    r = max(rank_min, min(int(r), min(m, n)))
    return r
