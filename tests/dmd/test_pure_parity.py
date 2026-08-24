import json
from pathlib import Path

import torch

from algorithms.dmd import _legacy_core as legacy
from algorithms.dmd.objectives import quadratic_margin_deficit
from algorithms.dmd.profiles import class_margin_profile, true_class_margin
from algorithms.dmd.references import (
    robust_margin_reference,
    temporal_robust_margin_reference,
)
from algorithms.dmd.tail_risk import weighted_upper_cvar

FIXTURE = Path(__file__).parent / "fixtures" / "golden_pre_modularization.json"


def test_golden_profile_deficit_and_tail() -> None:
    golden = json.loads(FIXTURE.read_text())
    logits = torch.tensor(golden["logits"])
    targets = torch.tensor(golden["targets"])
    margins = true_class_margin(logits, targets)
    profile = class_margin_profile(logits, targets, 3)
    assert torch.allclose(margins, torch.tensor(golden["margins"]))
    assert torch.allclose(profile.values, torch.tensor(golden["profile_values"]))
    assert profile.counts.tolist() == golden["profile_counts"]
    deficit = quadratic_margin_deficit(profile, torch.tensor(golden["reference"]))
    assert torch.allclose(deficit, torch.tensor(golden["quadratic_deficit"]))
    tail = weighted_upper_cvar(
        torch.tensor(golden["tail_values"]),
        torch.tensor(golden["tail_weights_input"]),
        tail_mass=golden["tail_mass"],
    )
    assert float(tail.eta) == golden["tail_eta"]
    assert float(tail.cvar) == golden["tail_cvar"]
    assert torch.allclose(tail.tail_weights, torch.tensor(golden["tail_weights"]))


def test_new_pure_functions_match_frozen_legacy_core() -> None:
    generator = torch.Generator().manual_seed(20260820)
    logits = torch.randn(24, 4, generator=generator)
    targets = torch.randint(0, 4, (24,), generator=generator)
    new_profile = class_margin_profile(logits, targets, 4)
    old_profile = legacy.class_margin_profile(logits, targets, 4)
    assert torch.allclose(new_profile.values, old_profile.values, equal_nan=True)
    assert torch.equal(new_profile.counts, old_profile.counts)
    profiles = torch.randn(7, 4, generator=generator)
    profiles[0, 2] = torch.nan
    new_reference = robust_margin_reference(profiles, method="median", min_clients=2)
    old_reference = legacy.robust_margin_reference(
        profiles, method="median", min_clients=2
    )
    assert torch.allclose(new_reference[0], old_reference[0], equal_nan=True)
    assert torch.equal(new_reference[1], old_reference[1])


def test_temporal_reference_matches_frozen_legacy_core() -> None:
    values = torch.tensor([[1.0, 4.0], [2.0, 5.0], [9.0, float("nan")]])
    rounds = torch.tensor([[4, 4], [3, 3], [4, -1]])
    kwargs = dict(
        current_round=5,
        window=3,
        decay_gamma=0.2,
        min_effective_clients=2,
        target_effective_clients=3,
        publication_support="raw",
    )
    new = temporal_robust_margin_reference(values, rounds, **kwargs)
    old = legacy.temporal_robust_margin_reference(values, rounds, **kwargs)
    for name in (
        "values",
        "scale",
        "raw_support",
        "effective_support",
        "reliability",
        "mean_age",
        "max_age",
    ):
        assert torch.allclose(getattr(new, name), getattr(old, name), equal_nan=True)
