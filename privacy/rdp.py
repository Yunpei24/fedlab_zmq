"""Small, explicit RDP accountant for Gaussian sampling mechanisms.

This module implements the finite-sum expression used in FedFDP for integer
Renyi orders.  It is intentionally transparent and dependency-free; it is not
a replacement for a production accountant such as PRV or PLD accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _log_add(log_x: float, log_y: float) -> float:
    if log_x == -math.inf:
        return log_y
    if log_y == -math.inf:
        return log_x
    larger = max(log_x, log_y)
    return larger + math.log(math.exp(log_x - larger) + math.exp(log_y - larger))


def sampled_gaussian_rdp(
    order: int, sampling_rate: float, noise_multiplier: float
) -> float:
    """RDP cost of one Poisson-sampled Gaussian step at integer ``order``.

    The expression is

    ``log(sum_k C(a,k)(1-q)^(a-k)q^k exp((k^2-k)/(2 sigma^2)))/(a-1)``.
    """

    if order < 2 or int(order) != order:
        raise ValueError("This pedagogical accountant supports integer orders >= 2")
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError("sampling_rate must lie in [0,1]")
    if noise_multiplier <= 0:
        return math.inf
    q = float(sampling_rate)
    if q == 0:
        return 0.0
    log_a = -math.inf
    for k in range(order + 1):
        log_binomial = (
            math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1)
        )
        log_probability = 0.0
        if order - k:
            if q == 1.0:
                continue
            log_probability += (order - k) * math.log1p(-q)
        if k:
            log_probability += k * math.log(q)
        gaussian = (k * k - k) / (2.0 * noise_multiplier**2)
        log_a = _log_add(log_a, log_binomial + log_probability + gaussian)
    return log_a / (order - 1)


def _log_sub_sign(log_x: float, log_y: float) -> tuple[bool, float]:
    """Return the sign and log-magnitude of ``exp(log_x) - exp(log_y)``."""

    if log_x == log_y:
        return True, -math.inf
    if log_x > log_y:
        return True, log_x + math.log1p(-math.exp(log_y - log_x))
    return False, log_y + math.log1p(-math.exp(log_x - log_y))


def _forward_difference_log_magnitudes(
    function, order: int
) -> tuple[list[float], list[bool]]:
    """Compute forward differences in signed log space.

    This is the dependency-free equivalent of the stable construction used
    by TensorFlow Privacy for Theorem 27 of Wang, Balle and Kasiviswanathan
    (AISTATS 2019).  Directly differencing ``exp(function(x))`` overflows at
    the Renyi orders used by the experiments.
    """

    values = [0.0] * (order + 3)
    signs = [True] * (order + 3)
    differences = [0.0] * (order + 2)
    difference_signs = [True] * (order + 2)
    for index in range(1, order + 3):
        values[index] = float(function(float(index - 1)))
    for level in range(order + 2):
        count = order + 2 - level
        for index in range(count):
            if signs[index] == signs[index + 1]:
                new_sign, magnitude = _log_sub_sign(values[index + 1], values[index])
                if not signs[index + 1]:
                    new_sign = not new_sign
                signs[index] = new_sign
                values[index] = magnitude
            else:
                values[index] = _log_add(values[index], values[index + 1])
                signs[index] = signs[index + 1]
        differences[level] = values[0]
        difference_signs[level] = signs[0]
    return differences, difference_signs


def sampled_without_replacement_gaussian_rdp(
    order: int, sampling_rate: float, noise_multiplier: float
) -> float:
    """RDP upper bound for one fixed-size sample without replacement.

    The sample is a uniformly random subset of public size ``m`` from a
    public dataset of size ``N``, with ``sampling_rate = m / N``.  The
    neighboring relation is replace-one.  ``noise_multiplier`` is the
    Gaussian standard deviation divided by the *replace-one sensitivity* of
    the sum query.  For per-example gradients clipped at ``C``, the caller
    must therefore pass ``sigma / 2`` when the implementation adds noise with
    standard deviation ``sigma * C``.

    The expression is Theorem 27 of Wang, Balle and Kasiviswanathan,
    "Subsampled Renyi Differential Privacy and Analytical Moments
    Accountant", AISTATS 2019.  Only integer orders up to 256 are needed by
    this repository and are evaluated with the stable forward-difference
    construction used by TensorFlow Privacy.
    """

    if order < 2 or int(order) != order:
        raise ValueError("Fixed-size WOR accounting supports integer orders >= 2")
    if order > 256:
        raise ValueError("Fixed-size WOR accounting supports orders up to 256")
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError("sampling_rate must lie in [0,1]")
    if noise_multiplier <= 0:
        return math.inf
    q = float(sampling_rate)
    sigma = float(noise_multiplier)
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return gaussian_rdp(order, sigma)

    def cgf(x: float) -> float:
        return x * (x + 1.0) / (2.0 * sigma**2)

    def base_rdp(x: float) -> float:
        return x / (2.0 * sigma**2)

    log_a = 0.0  # The i=0 and i=1 contribution equals one.
    base_order_two = base_rdp(2.0)
    log_f2_minus_one = base_order_two + math.log1p(-math.exp(-base_order_two))
    differences, _ = _forward_difference_log_magnitudes(cgf, int(order))
    for index in range(2, int(order) + 1):
        log_binomial = (
            math.lgamma(order + 1)
            - math.lgamma(index + 1)
            - math.lgamma(order - index + 1)
        )
        if index == 2:
            term = min(
                math.log(4.0) + log_f2_minus_one,
                base_order_two + math.log(2.0),
            )
        else:
            lower = differences[2 * math.floor(index / 2.0) - 1]
            upper = differences[2 * math.ceil(index / 2.0) - 1]
            term = min(
                math.log(4.0) + 0.5 * (lower + upper),
                math.log(2.0) + cgf(index - 1.0),
            )
        term += index * math.log(q) + log_binomial
        log_a = _log_add(log_a, term)
    return float(log_a) / (order - 1)


def gaussian_rdp(order: float, noise_multiplier: float) -> float:
    """Exact RDP cost of an ordinary Gaussian mechanism.

    The released statistic has L2 sensitivity ``Delta`` and Gaussian standard
    deviation ``noise_multiplier * Delta``.  Therefore the sensitivity cancels
    from the privacy cost and the mechanism is

    ``(order, order / (2 * noise_multiplier**2))-RDP``.

    This explicit helper is the primary accountant primitive for full-
    participation central SC-FAR-DP.  It avoids labelling the ``q=1`` case as
    sampled or claiming privacy amplification that is not part of the theorem.
    """

    if order <= 1:
        raise ValueError("Renyi order must be greater than one")
    if noise_multiplier <= 0:
        return math.inf
    return float(order) / (2.0 * float(noise_multiplier) ** 2)


def calibrate_sampled_gaussian_noise(
    *,
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    orders: tuple[int, ...] = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64),
    lower: float = 0.05,
    upper: float = 100.0,
    tolerance: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Calibrate a noise multiplier by monotone binary search.

    This helper follows the same accountant implemented in this module.  It
    does not repair a mismatch between Poisson sampling assumed by the
    accountant and ordinary shuffled fixed-size mini-batches used by a data
    loader; callers must record that sampling assumption explicitly.
    """

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    def epsilon_at(noise: float) -> float:
        accountant = RDPAccountant(orders=orders)
        accountant.add_sampled_gaussian(
            channel="model",
            sampling_rate=sampling_rate,
            noise_multiplier=noise,
            steps=steps,
        )
        return accountant.epsilon(delta)[0]

    while epsilon_at(upper) > target_epsilon:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Could not bracket a noise multiplier")
    if epsilon_at(lower) < target_epsilon:
        return lower
    for _ in range(max_iter):
        middle = (lower + upper) / 2.0
        epsilon = epsilon_at(middle)
        if abs(epsilon - target_epsilon) <= tolerance:
            return middle
        if epsilon > target_epsilon:
            lower = middle
        else:
            upper = middle
    return upper


def calibrate_sampled_without_replacement_gaussian_noise(
    *,
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    sensitivity_multiplier: float = 2.0,
    orders: tuple[int, ...] = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64),
    lower: float = 0.05,
    upper: float = 100.0,
    tolerance: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Calibrate the implementation multiplier for fixed-size WOR sampling.

    The returned value multiplies the per-example clipping bound ``C`` in the
    training code.  Under replace-one adjacency the sensitivity of the sum is
    ``2C``; hence the accountant receives ``returned / 2`` by default.
    """

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sensitivity_multiplier <= 0:
        raise ValueError("sensitivity_multiplier must be positive")

    def epsilon_at(implementation_noise: float) -> float:
        accountant = RDPAccountant(orders=orders)
        accountant.add_sampled_without_replacement_gaussian(
            channel="model",
            sampling_rate=sampling_rate,
            noise_multiplier=implementation_noise / sensitivity_multiplier,
            steps=steps,
        )
        return accountant.epsilon(delta)[0]

    while epsilon_at(upper) > target_epsilon:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Could not bracket a fixed-size WOR multiplier")
    if epsilon_at(lower) < target_epsilon:
        return lower
    for _ in range(max_iter):
        middle = (lower + upper) / 2.0
        epsilon = epsilon_at(middle)
        if abs(epsilon - target_epsilon) <= tolerance:
            return middle
        if epsilon > target_epsilon:
            lower = middle
        else:
            upper = middle
    return upper


def calibrate_gaussian_noise(
    *,
    target_epsilon: float,
    delta: float,
    steps: int,
    orders: tuple[int, ...] = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64),
    lower: float = 0.05,
    upper: float = 100.0,
    tolerance: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Calibrate one multiplier for ``steps`` ordinary Gaussian releases."""

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    def epsilon_at(noise: float) -> float:
        accountant = RDPAccountant(orders=orders)
        accountant.add_gaussian(
            channel="central_model",
            noise_multiplier=noise,
            steps=steps,
        )
        return accountant.epsilon(delta)[0]

    while epsilon_at(upper) > target_epsilon:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Could not bracket a Gaussian noise multiplier")
    if epsilon_at(lower) < target_epsilon:
        return lower
    for _ in range(max_iter):
        middle = (lower + upper) / 2.0
        epsilon = epsilon_at(middle)
        if abs(epsilon - target_epsilon) <= tolerance:
            return middle
        if epsilon > target_epsilon:
            lower = middle
        else:
            upper = middle
    return upper


def calibrate_composed_sampled_gaussian_noise(
    *,
    target_epsilon: float,
    delta: float,
    channels: tuple[tuple[float, int, float], ...],
    orders: tuple[int, ...] = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64),
    lower: float = 0.05,
    upper: float = 100.0,
    tolerance: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Calibrate one base multiplier for a composition of DP channels.

    A channel is ``(sampling_rate, steps, noise_ratio)`` and its Gaussian
    multiplier is ``base * noise_ratio``.  The binary search therefore
    calibrates the total RDP of, for example, a model channel and a private
    scalar-loss channel rather than calibrating only one of them.
    """

    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    if not channels:
        raise ValueError("channels cannot be empty")
    for sampling_rate, steps, ratio in channels:
        if not 0.0 <= float(sampling_rate) <= 1.0:
            raise ValueError("channel sampling rates must lie in [0,1]")
        if int(steps) <= 0:
            raise ValueError("channel steps must be positive")
        if float(ratio) <= 0:
            raise ValueError("channel noise ratios must be positive")

    def epsilon_at(base: float) -> float:
        accountant = RDPAccountant(orders=orders)
        for index, (sampling_rate, steps, ratio) in enumerate(channels):
            accountant.add_sampled_gaussian(
                channel=f"channel_{index}",
                sampling_rate=float(sampling_rate),
                noise_multiplier=float(base) * float(ratio),
                steps=int(steps),
            )
        return accountant.epsilon(delta)[0]

    while epsilon_at(upper) > target_epsilon:
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("Could not bracket a composed noise multiplier")
    if epsilon_at(lower) < target_epsilon:
        return lower
    for _ in range(max_iter):
        middle = (lower + upper) / 2.0
        epsilon = epsilon_at(middle)
        if abs(epsilon - target_epsilon) <= tolerance:
            return middle
        if epsilon > target_epsilon:
            lower = middle
        else:
            upper = middle
    return upper


@dataclass
class RDPAccountant:
    """Composable ledger for model and auxiliary release channels."""

    orders: tuple[int, ...] = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64)
    channels: dict[str, dict[int, float]] = field(default_factory=dict)

    def add_gaussian(
        self,
        *,
        channel: str,
        noise_multiplier: float,
        steps: int = 1,
    ) -> None:
        """Compose ordinary Gaussian releases without sampling amplification."""

        if steps < 0:
            raise ValueError("steps cannot be negative")
        ledger = self.channels.setdefault(channel, {a: 0.0 for a in self.orders})
        for order in self.orders:
            ledger[order] += steps * gaussian_rdp(order, noise_multiplier)

    def add_sampled_gaussian(
        self,
        *,
        channel: str,
        sampling_rate: float,
        noise_multiplier: float,
        steps: int = 1,
    ) -> None:
        ledger = self.channels.setdefault(channel, {a: 0.0 for a in self.orders})
        for order in self.orders:
            ledger[order] += steps * sampled_gaussian_rdp(
                order, sampling_rate, noise_multiplier
            )

    def add_sampled_without_replacement_gaussian(
        self,
        *,
        channel: str,
        sampling_rate: float,
        noise_multiplier: float,
        steps: int = 1,
    ) -> None:
        """Compose fixed-size uniformly sampled Gaussian releases."""

        if steps < 0:
            raise ValueError("steps cannot be negative")
        ledger = self.channels.setdefault(channel, {a: 0.0 for a in self.orders})
        for order in self.orders:
            ledger[order] += steps * sampled_without_replacement_gaussian_rdp(
                order, sampling_rate, noise_multiplier
            )

    def total_rdp(self) -> dict[int, float]:
        return {
            order: sum(channel.get(order, 0.0) for channel in self.channels.values())
            for order in self.orders
        }

    def epsilon(self, delta: float) -> tuple[float, int]:
        """Convert composed RDP to ``(epsilon, delta)``-DP.

        Uses the tighter conversion quoted by FedFDP.  The best order is
        selected automatically.
        """

        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie in (0,1)")
        candidates = {}
        for order, rdp in self.total_rdp().items():
            candidates[order] = (
                rdp
                + math.log((order - 1) / order)
                - (math.log(delta) + math.log(order)) / (order - 1)
            )
        best_order = min(candidates, key=candidates.get)
        return max(0.0, candidates[best_order]), best_order

    def state_dict(self) -> dict:
        return {
            "orders": list(self.orders),
            "channels": {
                name: {str(k): float(v) for k, v in values.items()}
                for name, values in self.channels.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict | None) -> "RDPAccountant":
        if not state:
            return cls()
        accountant = cls(orders=tuple(int(x) for x in state.get("orders", (2, 3, 4))))
        accountant.channels = {
            name: {int(k): float(v) for k, v in values.items()}
            for name, values in state.get("channels", {}).items()
        }
        return accountant
