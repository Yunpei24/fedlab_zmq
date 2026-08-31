#!/usr/bin/env python3
"""Preflight the q=1 Gaussian RDP accountant used by SC-FAR-DP paper 1.

The script first recomputes every RDP order and the RDP-to-DP conversion from
closed-form expressions outside the accountant object.  If Opacus is
installed, it additionally records an external-library comparison at q=1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy.rdp import RDPAccountant, calibrate_gaussian_noise


DEFAULT_ORDERS = (2, 3, 4, 5, 8, 10, 16, 20, 32, 64)


def direct_epsilon(
    *, noise_multiplier: float, steps: int, delta: float, orders=DEFAULT_ORDERS
) -> tuple[float, int, dict[int, float]]:
    """Independent algebraic evaluation of the finite RDP order grid."""

    total_rdp = {
        int(order): steps * float(order) / (2.0 * noise_multiplier**2)
        for order in orders
    }
    candidates = {
        order: rdp
        + math.log((order - 1) / order)
        - (math.log(delta) + math.log(order)) / (order - 1)
        for order, rdp in total_rdp.items()
    }
    best_order = min(candidates, key=candidates.get)
    return max(0.0, candidates[best_order]), best_order, total_rdp


def opacus_epsilon(
    *, noise_multiplier: float, steps: int, delta: float, orders=DEFAULT_ORDERS
) -> tuple[float, float] | None:
    try:
        from opacus.accountants.analysis import rdp as opacus_rdp
    except ImportError:
        return None
    rdp_values = opacus_rdp.compute_rdp(
        q=1.0,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=list(orders),
    )
    epsilon, best_order = opacus_rdp.get_privacy_spent(
        orders=list(orders), rdp=rdp_values, delta=delta
    )
    return float(epsilon), float(best_order)


def validate_target(*, target: float, steps: int, delta: float) -> dict:
    sigma = calibrate_gaussian_noise(
        target_epsilon=target,
        delta=delta,
        steps=steps,
        orders=DEFAULT_ORDERS,
    )
    accountant = RDPAccountant(orders=DEFAULT_ORDERS)
    accountant.add_gaussian(
        channel="central_model", noise_multiplier=sigma, steps=steps
    )
    internal_epsilon, internal_order = accountant.epsilon(delta)
    direct, direct_order, direct_rdp = direct_epsilon(
        noise_multiplier=sigma, steps=steps, delta=delta
    )
    internal_rdp = accountant.total_rdp()
    maximum_rdp_error = max(
        abs(internal_rdp[order] - direct_rdp[order]) for order in DEFAULT_ORDERS
    )
    external = opacus_epsilon(
        noise_multiplier=sigma, steps=steps, delta=delta
    )
    return {
        "target_epsilon": target,
        "noise_multiplier": sigma,
        "internal_epsilon": internal_epsilon,
        "internal_best_order": internal_order,
        "direct_epsilon": direct,
        "direct_best_order": direct_order,
        "absolute_internal_direct_error": abs(internal_epsilon - direct),
        "maximum_rdp_order_error": maximum_rdp_error,
        "opacus": (
            None
            if external is None
            else {"epsilon": external[0], "best_order": external[1]}
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="1,3,6,10")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--require-opacus", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or not 0.0 < args.delta < 1.0:
        raise ValueError("Need positive steps and delta in (0,1)")
    targets = [float(value) for value in args.targets.split(",")]
    rows = [
        validate_target(target=target, steps=args.steps, delta=args.delta)
        for target in targets
    ]
    if any(row["absolute_internal_direct_error"] > 1e-10 for row in rows):
        raise RuntimeError("Internal accountant disagrees with direct q=1 formula")
    if args.require_opacus and any(row["opacus"] is None for row in rows):
        raise RuntimeError("Opacus is required but is not installed")
    payload = {
        "mechanism": "ordinary_gaussian_q_equals_1",
        "steps": args.steps,
        "delta": args.delta,
        "orders": list(DEFAULT_ORDERS),
        "opacus_available": all(row["opacus"] is not None for row in rows),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
