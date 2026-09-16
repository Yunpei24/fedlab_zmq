#!/usr/bin/env python3
"""MPS-only geometry preflight for K7b; never computes or saves MSE."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_gaussian_aware_reference_g0g_k7b_rcig_confirmation as k7b,
)


def main() -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must equal 0")
    k7b._validate_protocol()
    k7b._configure_k7_dependency()
    device = k7b.k7._require_mps()
    failures: list[dict[str, object]] = []
    checked = 0
    registries = [
        (
            "calibration",
            k7b.CALIBRATION_SEEDS,
            k7b.k7.REGIMES,
            ("none",),
        ),
        (
            "null_validation",
            k7b.NULL_VALIDATION_SEEDS_BY_REGIME["homogeneous"],
            ("homogeneous",),
            ("none",),
        ),
        (
            "null_validation",
            k7b.NULL_VALIDATION_SEEDS_BY_REGIME["heteroscedastic"],
            ("heteroscedastic",),
            ("none",),
        ),
        (
            "evaluation",
            k7b.EVALUATION_SEEDS,
            k7b.k7.REGIMES,
            k7b.k7.THREATS,
        ),
    ]
    for phase, seeds, regimes, threats in registries:
        for seed in seeds:
            for regime in regimes:
                for threat in threats:
                    checked += 1
                    try:
                        k7b.k7._base_context(
                            seed,
                            regime,
                            threat,
                            phase=phase,
                            device=device,
                        )
                    except RuntimeError as exc:
                        failures.append(
                            {
                                "phase": phase,
                                "seed": seed,
                                "noise_regime": regime,
                                "threat": threat,
                                "error": str(exc),
                            }
                        )
    print(
        json.dumps(
            {
                "status": "pass" if not failures else "fail",
                "device": str(device),
                "mps_fallback_disabled": True,
                "checked_contexts": checked,
                "failure_count": len(failures),
                "failures": failures,
                "selection_information_used": "technical_geometry_only_no_MSE_no_target",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
