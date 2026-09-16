"""Private-report-only risk selector; no datasets, gradients or oracle labels.

The certificate concerns a fixed empirical client-risk vector, not test accuracy
or population risk. Enumeration is exact at n=10 and deliberately conservative.
"""
from functools import lru_cache
from itertools import combinations
import math


def cvar_upper(values, fraction=0.2):
    values = sorted(map(float, values), reverse=True)
    if not values or not 0 < fraction <= 1 or not all(map(math.isfinite, values)):
        raise ValueError("invalid CVaR inputs")
    mass = fraction * len(values)
    whole = int(math.floor(mass))
    residual = mass - whole
    return (sum(values[:whole]) + (residual * values[whole] if residual > 0 else 0)) / mass


def risk(values, fairness_mix=0.5, tail_fraction=0.2):
    values = list(map(float, values))
    if not 0 <= fairness_mix <= 1:
        raise ValueError("invalid fairness coefficient")
    return (1-fairness_mix)*sum(values)/len(values) + fairness_mix*cvar_upper(values, tail_fraction)


@lru_cache(maxsize=32)
def possible_honest_sets(n, b):
    if not 0 <= b < n:
        raise ValueError("invalid Byzantine bound")
    return tuple(s for size in range(n-b, n+1) for s in combinations(range(n), size))


def select_private_risk(reports, *, noise_stds, max_releases, failure_probability,
                        byzantine_bound, fairness_mix=0.5, tail_fraction=0.2):
    """Return a candidate index using ONLY private reports and public constants.

    Column 0 is the identical-current-model no-op. Invalid report coordinates
    produce [0,1], never a spurious narrow interval. All candidate queries are
    covered by one per-run union bound, including all releases and clients.
    """
    n = len(reports)
    k = len(reports[0]) if n else 0
    if k < 2 or len(noise_stds) != n or any(len(row) != k for row in reports):
        raise ValueError("invalid report dimensions")
    if max_releases < 1 or not 0 < failure_probability < 1:
        raise ValueError("invalid confidence parameters")
    if any(not math.isfinite(s) or s < 0 for s in noise_stds):
        raise ValueError("invalid public standard deviation")
    multiplier = math.sqrt(2*math.log(2*n*k*max_releases/failure_probability))
    halfwidths = [s*multiplier for s in noise_stds]
    lower, upper = [], []
    for row, width in zip(reports, halfwidths):
        lo, hi = [], []
        for value in row:
            a, z = max(0., value-width), min(1., value+width)
            if not math.isfinite(value) or a > z:
                a, z = 0., 1.
            lo.append(a); hi.append(z)
        lower.append(lo); upper.append(hi)
    sets = possible_honest_sets(n, byzantine_bound)
    bounds = [0.]
    for column in range(1, k):
        bounds.append(max(
            risk([upper[i][column] for i in s], fairness_mix, tail_fraction)
            - risk([lower[i][0] for i in s], fairness_mix, tail_fraction)
            for s in sets))
    chosen = min(range(k), key=lambda column: bounds[column])
    clipped = [[min(1., max(0., v)) if math.isfinite(v) else .5 for v in row] for row in reports]
    naive_risks = [risk([row[j] for row in clipped], fairness_mix, tail_fraction) for j in range(k)]
    naive = min(range(k), key=lambda j: naive_risks[j])
    return dict(selected=chosen, upper_bounds=bounds, halfwidths=halfwidths,
                possible_honest_sets=len(sets), naive_selected=naive,
                lower=lower, upper=upper, input_boundary="private_reports_and_public_parameters_only")


def forge_reports(reports, *, identities, mode):
    """Fixed report attacks use no true risk or unknown honest identities."""
    output = [list(row) for row in reports]
    if mode not in {"truthful", "veto", "endorse_far"}:
        raise ValueError("unknown report attack")
    if mode != "truthful":
        for i in identities:
            output[i] = [0.] + [1.]*(len(output[i])-1) if mode == "veto" else [1.]*(len(output[i])-1)+[0.]
    return output
