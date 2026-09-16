"""Paired private loss contrasts: certificate, NOT an end-to-end FL algorithm.

Server inputs contain noisy contrasts and public constants only. The certificate
upper-bounds a change of mean/CVaR risk; it does not equate that change to the
mean/CVaR of changes. No clipping of contrasts is performed to claim smaller DP.
"""
import math

from algorithms.private_risk_selection import risk


def paired_sensitivity(validation_size, candidates):
    """Replace-one sensitivity of K averaged differences of [0,1] losses."""
    if not isinstance(validation_size, int) or validation_size < 1:
        raise ValueError("public validation size must be a positive integer")
    if not isinstance(candidates, int) or candidates < 1:
        raise ValueError("candidate count must be a positive integer")
    return 2.0 * math.sqrt(candidates) / validation_size


def worst_honest_risk(values, byzantine_bound, fairness_mix=.5, tail_fraction=.2):
    """Exact max of risk on all subsets of size >= n-b (equal client weights).

    The worst subset retains the n-b largest values. This shortcut is valid for
    risk OF contrasts, not for a difference of two independently ranked risks.
    """
    n = len(values)
    if not isinstance(byzantine_bound, int) or not 0 <= byzantine_bound < n:
        raise ValueError("invalid public Byzantine bound")
    return risk(sorted(values, reverse=True)[:n-byzantine_bound], fairness_mix, tail_fraction)


def choose_under_bound(upper_bounds, damage_tolerance=0.):
    """Index 0 = no-op; positive tolerance certifies damage <= tolerance only.

    Strict descent mode keeps the no-op on equality. A positive tolerance can
    permit a harmful action and is deliberately not described as progress.
    """
    if not math.isfinite(damage_tolerance) or damage_tolerance < 0:
        raise ValueError("invalid public damage tolerance")
    if not upper_bounds or not all(math.isfinite(x) for x in upper_bounds):
        raise ValueError("invalid bounds")
    k = min(range(len(upper_bounds)), key=upper_bounds.__getitem__)
    if damage_tolerance == 0 and upper_bounds[k] >= 0:
        return 0
    return k+1 if upper_bounds[k] <= damage_tolerance else 0


def certify_paired_reports(reports, *, noise_stds, max_releases,
                           failure_probability, byzantine_bound,
                           fairness_mix=.5, tail_fraction=.2):
    """Simultaneous Gaussian intervals + worst-possible-honest-set certificate.

    Three columns describe differences vs one shared baseline. We retain K+1
    coordinates in the union bound for a like-for-like comparison with the old
    four-level release. It is conservative, since no baseline contrast is sent.
    Malformed coordinates yield [-1,1], never a fabricated precise report.
    """
    n = len(reports); k = len(reports[0]) if n else 0
    if k < 1 or len(noise_stds) != n or any(len(row) != k for row in reports):
        raise ValueError("invalid report dimensions")
    if not isinstance(max_releases, int) or max_releases < 1 or not 0 < failure_probability < 1:
        raise ValueError("invalid confidence constants")
    if any(not math.isfinite(s) or s < 0 for s in noise_stds):
        raise ValueError("invalid public standard deviation")
    multiplier = math.sqrt(2*math.log(2*n*(k+1)*max_releases/failure_probability))
    widths = [multiplier*s for s in noise_stds]
    lower, upper = [], []
    for row, width in zip(reports, widths):
        lo, hi = [], []
        for value in row:
            a, z = max(-1., value-width), min(1., value+width)
            if not math.isfinite(value) or a > z:
                a, z = -1., 1.
            lo.append(a); hi.append(z)
        lower.append(lo); upper.append(hi)
    bounds = [worst_honest_risk([upper[i][j] for i in range(n)], byzantine_bound,
                               fairness_mix, tail_fraction) for j in range(k)]
    return dict(selected=choose_under_bound(bounds), upper_bounds=bounds,
                lower=lower, upper=upper, halfwidths=widths,
                guarantee="upper_bound_on_change_of_empirical_honest_mean_CVaR_risk",
                input_boundary="noisy_paired_reports_and_public_constants_only")


def private_paired_release_mps(losses, noise_multiplier):
    """Client-side primitive, shape N x (K+1); losses lie in [0,1].

    Column zero is baseline loss on EACH SAME EXAMPLE. No minibatch
    renormalization, no clipping of the mean, and no per-example truncation.
    Caller owns secure RNG, accounting, model construction and local datasets.
    Returns only the noisy K-vector (not the unprotected empirical contrasts).
    """
    import torch
    if losses.device.type != "mps" or losses.ndim != 2 or losses.shape[1] < 2 or losses.shape[0] < 1:
        raise ValueError("requires nonempty MPS matrix of paired bounded losses")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("positive Gaussian multiplier required")
    # These checks audit the caller's mathematical domain, not a private filter
    # that can reject selected records. A production caller must guarantee it.
    if not bool(torch.isfinite(losses).all()) or bool(((losses < 0) | (losses > 1)).any()):
        raise ValueError("caller violated the bounded-loss domain")
    n, columns = losses.shape
    std = paired_sensitivity(n, columns-1)*noise_multiplier
    contrast = (losses[:, 1:]-losses[:, :1]).mean(0)
    return contrast + std*torch.randn(columns-1, device="mps", dtype=losses.dtype)
