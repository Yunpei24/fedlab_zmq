#!/usr/bin/env python3
"""Paired synthetic oracle audit for Gaussian-aware robust references.

This experiment is intentionally upstream of an end-to-end vision benchmark.
The latent honest gradients, honest outliers, public DP covariance and
Byzantine identities are known, so each mechanism can be audited against
quantities that cannot be observed on Fashion-MNIST.  Every reference sees the
same cohort in every cell; reference choice is the only changed factor.

The script writes raw paired rows, grouped summaries and a machine-readable
decision.  Its pre-registered gates are necessary, not sufficient, for an
end-to-end promotion.  Test accuracy is deliberately absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.gaussian_aware_reference import (  # noqa: E402
    gaussian_aware_huber_leave_one_out,
    gaussian_aware_huber_reference,
    standardized_quadratic_scores,
)
from robustness.aggregators import (  # noqa: E402
    centered_clipping,
    centered_clipping_leave_one_out,
    clip_l2,
    geometric_median,
    noise_aware_centered_clipping,
)

REFERENCE_NAMES = {
    "uniform_mean",
    "fcc",
    "fna_cc",
    "rfa",
    "f_sigma_huber",
}
THREAT_NAMES = {"none", "alie", "ipm", "bitflip_x10", "model_replacement"}

# The synthetic audit used to construct every tensor implicitly on CPU in
# float64.  Keep an explicit runtime so a requested MPS execution cannot
# silently fall back to CPU.  CPU remains available only for unit tests and
# deliberate reproducibility checks.
_RUNTIME_DEVICE = torch.device("cpu")
_RUNTIME_DTYPE = torch.float64


def _configure_runtime(device: str | torch.device) -> tuple[torch.device, torch.dtype]:
    global _RUNTIME_DEVICE, _RUNTIME_DTYPE
    requested = torch.device(device)
    if requested.type not in {"cpu", "mps"}:
        raise ValueError(
            "The oracle audit supports only explicit 'cpu' or 'mps' devices"
        )
    if requested.type == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is unavailable. Refusing to fall back silently to CPU."
        )
    _RUNTIME_DEVICE = requested
    _RUNTIME_DTYPE = torch.float32 if requested.type == "mps" else torch.float64
    return _RUNTIME_DEVICE, _RUNTIME_DTYPE


def _seed(*parts: object) -> int:
    """Stable seed independent of Python's randomized ``hash`` function."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _generator(*parts: object) -> torch.Generator:
    return torch.Generator(device=_RUNTIME_DEVICE).manual_seed(_seed(*parts))


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-15)


def _orthogonal(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return _unit(vector - (vector @ direction) * direction)


def _corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(dtype=_RUNTIME_DTYPE)
    right = right.to(dtype=_RUNTIME_DTYPE)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-15:
        return float("nan")
    return float((left @ right / denominator).item())


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(finite)) if finite else float("nan")


def _finite_std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _validate_config(config: dict[str, Any]) -> None:
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    f = int(cohort["num_byzantine"])
    dimension = int(cohort["dimension"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    if n < 3 or not 0 <= f < n / 2:
        raise ValueError("Need n >= 3 and 0 <= num_byzantine < n/2")
    if dimension < 2 or sum(blocks) != dimension or any(value < 1 for value in blocks):
        raise ValueError("block_sizes must be positive and sum to dimension")
    if len(cohort["heterogeneity_std_by_block"]) != len(blocks):
        raise ValueError("One heterogeneity standard deviation is needed per block")
    outlier_count = int(cohort["honest_outliers"]["count"])
    if not 0 <= outlier_count <= n - f:
        raise ValueError("Honest outliers must fit among clients never replaced")
    block_noise = config["privacy_noise"]["block_std_multipliers"]
    if len(block_noise) != len(blocks):
        raise ValueError("One privacy-noise multiplier is needed per block")
    candidates = set(config["references"]["candidates"])
    unknown = candidates - REFERENCE_NAMES
    if unknown:
        raise ValueError(f"Unknown references: {sorted(unknown)}")
    threats = set(config["threats"]["names"])
    unknown_threats = threats - THREAT_NAMES
    if unknown_threats or "none" not in threats:
        raise ValueError(f"Invalid threat set: {sorted(unknown_threats)}")
    score = config["score"]
    if not (
        float(score["novelty_start_z"])
        < float(score["novelty_full_z"])
        <= float(score["rejection_start_z"])
        < float(score["rejection_full_z"])
    ):
        raise ValueError("Score novelty/rejection thresholds must be ordered")
    if not 0.0 < float(score["trust_floor"]) <= 1.0:
        raise ValueError("trust_floor must lie in (0,1]")


def _block_slices(block_sizes: Sequence[int]) -> tuple[slice, ...]:
    start = 0
    result = []
    for width in block_sizes:
        result.append(slice(start, start + int(width)))
        start += int(width)
    return tuple(result)


def _expand_block_values(
    values: torch.Tensor, block_sizes: Sequence[int]
) -> torch.Tensor:
    """Expand ``[..., B]`` block values to ``[..., d]`` coordinates."""

    pieces = [
        values[..., block : block + 1].expand(*values.shape[:-1], int(width))
        for block, width in enumerate(block_sizes)
    ]
    return torch.cat(pieces, dim=-1)


def _tier_multipliers(
    multipliers: Sequence[float],
    n: int,
    permutation: str,
    *,
    num_byzantine: int = 0,
) -> torch.Tensor:
    values = torch.tensor(
        [float(multipliers[index % len(multipliers)]) for index in range(n)],
        dtype=_RUNTIME_DTYPE,
        device=_RUNTIME_DEVICE,
    )
    if permutation == "identity":
        return values
    if permutation.startswith("rotate"):
        try:
            shift = int(permutation.removeprefix("rotate"))
        except ValueError as error:
            raise ValueError(f"Invalid tier permutation {permutation!r}") from error
        return torch.roll(values, shifts=shift)
    if permutation == "reverse":
        return values.flip(0)
    if permutation in {"byzantine_high", "byzantine_low"}:
        b = int(num_byzantine)
        if not 0 < b < n:
            raise ValueError(
                f"{permutation} requires num_byzantine strictly between 0 and n"
            )
        ordered = values.sort().values
        if permutation == "byzantine_high":
            return torch.cat((ordered[:-b], ordered[-b:]))
        return torch.cat((ordered[b:], ordered[:b]))
    raise ValueError(f"Unknown tier permutation {permutation!r}")


def _noise_variances(
    config: dict[str, Any], regime: dict[str, Any], permutation: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(config["cohort"]["num_clients"])
    tiers = _tier_multipliers(
        regime["client_std_multipliers"],
        n,
        permutation,
        num_byzantine=int(config["cohort"].get("num_byzantine", 0)),
    )
    base = float(config["privacy_noise"]["base_std"])
    block = torch.tensor(
        config["privacy_noise"]["block_std_multipliers"],
        dtype=_RUNTIME_DTYPE,
        device=_RUNTIME_DEVICE,
    )
    std = base * tiers[:, None] * block[None, :]
    return std.square(), tiers


def _population_geometry(
    config: dict[str, Any], *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return population centre, descent direction and public lagged anchor."""

    dimension = int(config["cohort"]["dimension"])
    descent = _unit(
        torch.randn(
            dimension,
            generator=_generator("descent", seed),
            dtype=_RUNTIME_DTYPE,
            device=_RUNTIME_DEVICE,
        )
    )
    centre = float(config["cohort"]["honest_mean_norm"]) * descent
    anchor_mode = str(config["references"].get("public_anchor", "zero"))
    if anchor_mode == "zero":
        anchor = torch.zeros_like(centre)
    elif anchor_mode == "lagged_public_proxy":
        offset = _orthogonal(
            torch.randn(
                dimension,
                generator=_generator("anchor", seed),
                dtype=_RUNTIME_DTYPE,
                device=_RUNTIME_DEVICE,
            ),
            descent,
        )
        anchor = (
            centre + float(config["references"]["public_anchor_error_norm"]) * offset
        )
    else:
        raise ValueError(f"Unknown public anchor mode {anchor_mode!r}")
    return centre, descent, anchor


def _honest_clean_vectors(
    config: dict[str, Any],
    *,
    seed: int,
    draw: int,
    geometry: str,
    include_outliers: bool,
    centre_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cohort = config["cohort"]
    n = int(cohort["num_clients"])
    f = int(cohort["num_byzantine"])
    blocks = [int(value) for value in cohort["block_sizes"]]
    centre, descent, anchor = _population_geometry(config, seed=seed)
    if centre_override is not None:
        anchor = anchor - centre + centre_override
        centre = centre_override
    block_std = torch.tensor(
        cohort["heterogeneity_std_by_block"],
        dtype=_RUNTIME_DTYPE,
        device=_RUNTIME_DEVICE,
    )
    coordinate_std = _expand_block_values(block_std, blocks)
    generator = _generator("honest", seed, draw, geometry, include_outliers)
    clean = (
        centre
        + torch.randn(
            n,
            int(cohort["dimension"]),
            generator=generator,
            dtype=_RUNTIME_DTYPE,
            device=_RUNTIME_DEVICE,
        )
        * coordinate_std[None, :]
    )
    outlier_mask = torch.zeros(n, dtype=torch.bool, device=_RUNTIME_DEVICE)
    if include_outliers:
        count = int(cohort["honest_outliers"]["count"])
        # Byzantine attacks replace the final f clients; honest outliers are
        # selected only among the clients that remain honest in every cell.
        ids = (
            torch.linspace(0, n - f - 1, steps=count, device=_RUNTIME_DEVICE)
            .round()
            .long()
            .unique()
        )
        if ids.numel() != count:
            raise RuntimeError("Honest-outlier ID construction lost an identity")
        shifts = []
        for index in range(count):
            direction = _orthogonal(
                torch.randn(
                    int(cohort["dimension"]),
                    generator=generator,
                    dtype=_RUNTIME_DTYPE,
                    device=_RUNTIME_DEVICE,
                ),
                descent,
            )
            if geometry == "aligned":
                direction = _unit(0.60 * descent + 0.80 * direction)
            elif geometry != "orthogonal":
                raise ValueError(f"Unknown honest-outlier geometry {geometry!r}")
            shifts.append(direction)
        clean[ids] += float(cohort["honest_outliers"]["shift_norm"]) * torch.stack(
            shifts
        )
        outlier_mask[ids] = True
    return clean, outlier_mask, centre, anchor


def _add_private_noise(
    clean: torch.Tensor,
    noise_variances: torch.Tensor,
    block_sizes: Sequence[int],
    *,
    seed: int,
    draw: int,
    regime: str,
    permutation: str,
    geometry: str,
    pair_permutations: bool = False,
) -> torch.Tensor:
    coordinate_std = _expand_block_values(noise_variances.sqrt(), block_sizes)
    noise = torch.randn(
        clean.shape,
        generator=_generator(
            "private-noise",
            seed,
            draw,
            regime,
            "paired-tier-assignment" if pair_permutations else permutation,
            geometry,
        ),
        dtype=clean.dtype,
        device=clean.device,
    )
    return clean + coordinate_std * noise


def _replace_with_attack(
    observed: torch.Tensor,
    config: dict[str, Any],
    *,
    threat: str,
    severity: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = observed.shape[0]
    f = int(config["cohort"]["num_byzantine"])
    mask = torch.zeros(n, dtype=torch.bool, device=observed.device)
    if threat == "none":
        return observed.clone(), mask
    mask[n - f :] = True
    honest = observed[~mask]
    mean = honest.mean(dim=0)
    coordinate_std = honest.std(dim=0, unbiased=False).clamp_min(1e-12)
    settings = config["threats"]
    if threat == "alie":
        attack = mean - (float(severity) * float(settings["alie_z"]) * coordinate_std)
    elif threat == "ipm":
        attack = -float(severity) * float(settings["ipm_scale"]) * mean
    elif threat == "bitflip_x10":
        attack = -float(severity) * float(settings["bitflip_scale"]) * mean
    elif threat == "model_replacement":
        adversarial = _orthogonal(
            torch.randn(
                observed.shape[1],
                generator=_generator("model-replacement", seed),
                dtype=observed.dtype,
                device=observed.device,
            ),
            _unit(mean),
        )
        scale = (
            float(severity)
            * float(settings["model_replacement_scale"])
            * float(config["aggregation"]["server_clip_norm"])
        )
        attack = mean + scale * adversarial
    else:
        raise ValueError(f"Unknown threat {threat!r}")
    attacked = observed.clone()
    attacked[mask] = attack[None, :]
    return attacked, mask


def _reference_kwargs(
    config: dict[str, Any],
    *,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
) -> dict[str, Any]:
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    heterogeneity_variances = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=_RUNTIME_DTYPE,
        device=_RUNTIME_DEVICE,
    ).square()
    settings = config["references"]["f_sigma_huber"]
    standardized_threshold = settings.get("standardized_threshold")
    if isinstance(standardized_threshold, list):
        standardized_threshold = [float(value) for value in standardized_threshold]
    elif standardized_threshold is not None:
        standardized_threshold = float(standardized_threshold)
    influence_cap = settings["influence_cap"]
    if isinstance(influence_cap, list):
        influence_cap = [float(value) for value in influence_cap]
    else:
        influence_cap = float(influence_cap)
    return {
        "anchor": anchor,
        "noise_variances": noise_variances,
        "block_sizes": block_sizes,
        "heterogeneity_variances": heterogeneity_variances,
        # Reference covariance is relevant for scoring, but including an
        # estimated covariance of the unknown optimiser in its own IRLS
        # geometry would create a circular definition.
        "reference_variances": None,
        "standardized_threshold": standardized_threshold,
        "null_tail_probability": float(settings["null_tail_probability"]),
        "influence_cap": influence_cap,
        "regularization": float(settings["regularization"]),
        "num_steps": int(settings["num_steps"]),
        "variance_floor": float(settings["variance_floor"]),
        "output_radius": settings.get("output_radius"),
    }


def _reference(
    method: str,
    vectors: torch.Tensor,
    config: dict[str, Any],
    *,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if method == "uniform_mean":
        result = vectors.mean(dim=0)
        return (result, {}) if return_diagnostics else result
    if method == "fcc":
        result = centered_clipping(
            vectors,
            anchor=anchor,
            tau=float(config["references"]["fcc"]["radius"]),
        )
        return (result, {}) if return_diagnostics else result
    if method == "fna_cc":
        # F_NA-CC supports one scalar isotropic variance per client.  Taking
        # the trace-average is the declared reduction of the block covariance.
        widths = torch.tensor(
            config["cohort"]["block_sizes"],
            dtype=noise_variances.dtype,
            device=noise_variances.device,
        )
        scalar_variances = (noise_variances * widths[None, :]).sum(dim=1) / widths.sum()
        settings = config["references"]["fna_cc"]
        result = noise_aware_centered_clipping(
            vectors,
            anchor=anchor,
            tau=float(settings["radius"]),
            noise_variances=scalar_variances,
            max_weight_ratio=float(settings["max_weight_ratio"]),
            variance_floor=float(settings["variance_floor"]),
        )
        return (result, {}) if return_diagnostics else result
    if method == "rfa":
        settings = config["references"]["rfa"]
        result = geometric_median(
            vectors,
            max_iter=int(settings["max_iter"]),
            tol=float(settings["tolerance"]),
            smoothing=float(settings["smoothing"]),
        )
        return (result, {}) if return_diagnostics else result
    if method == "f_sigma_huber":
        return gaussian_aware_huber_reference(
            vectors,
            **_reference_kwargs(config, anchor=anchor, noise_variances=noise_variances),
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(f"Unknown reference {method!r}")


def _leave_one_out_references(
    method: str,
    vectors: torch.Tensor,
    config: dict[str, Any],
    *,
    anchor: torch.Tensor,
    noise_variances: torch.Tensor,
) -> torch.Tensor:
    n = vectors.shape[0]
    if n < 3:
        raise ValueError("Leave-one-out scoring requires at least three clients")
    if method == "uniform_mean":
        return (vectors.sum(dim=0, keepdim=True) - vectors) / float(n - 1)
    if method == "fcc":
        return centered_clipping_leave_one_out(
            vectors,
            anchor=anchor,
            tau=float(config["references"]["fcc"]["radius"]),
        )
    if method == "f_sigma_huber":
        return gaussian_aware_huber_leave_one_out(
            vectors,
            **_reference_kwargs(config, anchor=anchor, noise_variances=noise_variances),
        )

    rows = []
    for index in range(n):
        keep = torch.ones(n, dtype=torch.bool, device=vectors.device)
        keep[index] = False
        rows.append(
            _reference(
                method,
                vectors[keep],
                config,
                anchor=anchor,
                noise_variances=noise_variances[keep],
            )
        )
    return torch.stack(rows)


def _calibrate_reference_variances(
    config: dict[str, Any],
    *,
    regime: dict[str, Any],
    permutation: str,
    draws: int,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Estimate only the covariance of each LOO reference under the null.

    These draws are independent of every evaluation seed.  The estimated
    block variances are fixed before evaluating outliers or attacks.
    """

    methods = tuple(str(value) for value in config["references"]["candidates"])
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    noise_variances, tiers = _noise_variances(config, regime, permutation)
    samples: dict[str, list[torch.Tensor]] = {method: [] for method in methods}
    observed_samples: list[torch.Tensor] = []
    zero = torch.zeros(
        int(config["cohort"]["dimension"]),
        dtype=_RUNTIME_DTYPE,
        device=_RUNTIME_DEVICE,
    )
    for draw in range(draws):
        clean, _, _, anchor = _honest_clean_vectors(
            config,
            seed=seed,
            draw=draw,
            geometry="orthogonal",
            include_outliers=False,
            centre_override=zero,
        )
        observed = _add_private_noise(
            clean,
            noise_variances,
            block_sizes,
            seed=seed,
            draw=draw,
            regime=str(regime["name"]),
            permutation=permutation,
            geometry="null",
            pair_permutations=bool(
                config["randomness"].get("pair_noise_across_tier_permutations", False)
            ),
        )
        if bool(config["aggregation"].get("clip_before_reference", False)):
            observed = clip_l2(
                observed, float(config["aggregation"]["server_clip_norm"])
            )
        observed_samples.append(observed)
        for method in methods:
            samples[method].append(
                _leave_one_out_references(
                    method,
                    observed,
                    config,
                    anchor=anchor,
                    noise_variances=noise_variances,
                )
            )

    calibrated: dict[str, dict[str, torch.Tensor]] = {}
    slices = _block_slices(block_sizes)
    for method, method_samples in samples.items():
        stacked = torch.stack(method_samples)  # draws x clients x dimension
        coordinate_variances = stacked.var(dim=0, unbiased=True)
        reference_variances = torch.stack(
            [coordinate_variances[:, block].mean(dim=1) for block in slices],
            dim=1,
        ).clamp_min(float(config["score"]["variance_floor"]))
        # Pool by public privacy tier.  This uses only the authenticated
        # covariance class, never a client label or evaluation outcome, and
        # avoids an identity-specific calibration that would force the
        # empirical score/noise correlation to zero by construction.
        for tier in torch.unique(tiers):
            selected = tiers.eq(tier)
            reference_variances[selected] = reference_variances[selected].mean(
                dim=0, keepdim=True
            )
        raw_scores = []
        for observed, references in zip(observed_samples, method_samples, strict=True):
            score, _ = _standardized_scores(
                observed,
                references,
                config,
                noise_variances=noise_variances,
                reference_variances=reference_variances,
            )
            raw_scores.append(score)
        stacked_scores = torch.stack(raw_scores)
        null_abs_correlations: list[float] = []
        null_tier_ranges: list[float] = []
        null_mask = torch.ones(
            int(config["cohort"]["num_clients"]),
            dtype=torch.bool,
            device=_RUNTIME_DEVICE,
        )
        if bool(config["randomness"].get("null_match_regular_honest_subset", False)):
            count = int(config["cohort"]["honest_outliers"]["count"])
            n = int(config["cohort"]["num_clients"])
            f = int(config["cohort"]["num_byzantine"])
            excluded_ids = (
                torch.linspace(0, n - f - 1, steps=count, device=_RUNTIME_DEVICE)
                .round()
                .long()
                .unique()
            )
            null_mask[excluded_ids] = False
        null_tiers = tiers[null_mask]
        unique_tiers = torch.unique(null_tiers)
        for draw_scores in stacked_scores:
            null_scores = draw_scores[null_mask]
            correlation = _corr(null_scores, null_tiers)
            if math.isfinite(correlation):
                null_abs_correlations.append(abs(correlation))
            if unique_tiers.numel() >= 2:
                tier_means = [
                    float(null_scores[null_tiers.eq(tier)].mean().item())
                    for tier in unique_tiers
                ]
                null_tier_ranges.append(max(tier_means) - min(tier_means))
        calibrated[method] = {
            "reference_variances": reference_variances,
            # These are diagnostics only.  Evaluation scores are *not*
            # re-centred per client, because doing so would make noise-tier
            # invariance true by construction rather than test it.
            "score_mean": stacked_scores.mean(dim=0),
            "score_std": stacked_scores.std(dim=0, unbiased=True).clamp_min(1e-8),
            # One pooled method/regime threshold preserves noise-tier
            # differences instead of cancelling them client by client.
            "score_threshold": torch.quantile(
                stacked_scores[:, null_mask].flatten(),
                float(config["score"]["false_outlier_quantile"]),
            ),
            # G0b centres finite-cohort invariance statistics on these null
            # expectations.  Absolute correlation and a max-minus-min range
            # are positive even under a perfectly pivotal score when only a
            # few clients are observed, so an absolute zero-centred gate is
            # not statistically meaningful.
            "null_abs_score_noise_correlation_mean": _finite_mean(
                null_abs_correlations
            ),
            "null_noise_tier_score_range_mean": _finite_mean(null_tier_ranges),
        }
    return calibrated


def _standardized_scores(
    vectors: torch.Tensor,
    references: torch.Tensor,
    config: dict[str, Any],
    *,
    noise_variances: torch.Tensor,
    reference_variances: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    heterogeneity_variances = torch.tensor(
        config["cohort"]["heterogeneity_std_by_block"],
        dtype=vectors.dtype,
        device=vectors.device,
    ).square()
    scores, diagnostics = standardized_quadratic_scores(
        vectors,
        references=references,
        noise_variances=noise_variances,
        block_sizes=tuple(int(value) for value in config["cohort"]["block_sizes"]),
        heterogeneity_variances=heterogeneity_variances,
        reference_variances=reference_variances,
        variance_floor=float(config["score"]["variance_floor"]),
        return_diagnostics=True,
    )
    return scores, diagnostics


def _bandpass_far_weights(
    scores: torch.Tensor, config: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reward moderate novelty while redescending on extreme anomalies."""

    settings = config["score"]
    novelty = (
        (scores - float(settings["novelty_start_z"]))
        / (float(settings["novelty_full_z"]) - float(settings["novelty_start_z"]))
    ).clamp(0.0, 1.0)
    trust = (
        (float(settings["rejection_full_z"]) - scores)
        / (float(settings["rejection_full_z"]) - float(settings["rejection_start_z"]))
    ).clamp(0.0, 1.0)
    trust = trust.clamp_min(float(settings["trust_floor"]))
    logits = float(settings["alpha"]) * novelty + trust.log()
    weights = torch.softmax(logits, dim=0)
    return weights, {"novelty": novelty, "trust": trust, "logits": logits}


def _oracle_weights(
    scores: torch.Tensor,
    config: dict[str, Any],
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one pre-registered weighting ablation for the oracle audit."""

    settings = config["score"]
    novelty = (
        (scores - float(settings["novelty_start_z"]))
        / (float(settings["novelty_full_z"]) - float(settings["novelty_start_z"]))
    ).clamp(0.0, 1.0)
    trust = (
        (float(settings["rejection_full_z"]) - scores)
        / (float(settings["rejection_full_z"]) - float(settings["rejection_start_z"]))
    ).clamp(0.0, 1.0)
    trust = trust.clamp_min(float(settings["trust_floor"]))
    if mode == "reference_only":
        logits = torch.zeros_like(scores)
        weights = torch.full_like(scores, 1.0 / float(scores.numel()))
    elif mode == "novelty_only":
        logits = float(settings["alpha"]) * novelty
        weights = torch.softmax(logits, dim=0)
    elif mode == "novelty_confidence":
        logits = float(settings["alpha"]) * novelty + trust.log()
        weights = torch.softmax(logits, dim=0)
    else:
        raise ValueError(f"Unknown oracle weight mode {mode!r}")
    return weights, {"novelty": novelty, "trust": trust, "logits": logits}


def _tier_metrics(
    *,
    clean: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
    tiers: torch.Tensor,
    regular_honest_mask: torch.Tensor,
) -> tuple[float, float, float]:
    unique = torch.unique(tiers[regular_honest_mask])
    if unique.numel() < 2:
        return 0.0, 0.0, float("nan")
    score_means = []
    clean_means = []
    for tier in unique:
        selected = regular_honest_mask & tiers.eq(tier)
        score_means.append(float(scores[selected].mean().item()))
        clean_means.append(clean[selected].mean(dim=0))
    score_range = max(score_means) - min(score_means)
    distance_to_tiers = [
        float(torch.linalg.vector_norm(reference - mean).item()) for mean in clean_means
    ]
    distance_range = max(distance_to_tiers) - min(distance_to_tiers)
    contrast = clean_means[-1] - clean_means[0]
    if float(torch.linalg.vector_norm(contrast).item()) <= 1e-15:
        projection = 0.0
    else:
        honest_target = clean[regular_honest_mask].mean(dim=0)
        projection = abs(float(((reference - honest_target) @ _unit(contrast)).item()))
    return score_range, distance_range, projection


def _evaluate_reference(
    *,
    method: str,
    vectors: torch.Tensor,
    clean: torch.Tensor,
    outlier_mask: torch.Tensor,
    byzantine_mask: torch.Tensor,
    noise_variances: torch.Tensor,
    tiers: torch.Tensor,
    anchor: torch.Tensor,
    centre: torch.Tensor,
    calibration: dict[str, torch.Tensor],
    config: dict[str, Any],
    weight_mode: str | None = None,
) -> dict[str, Any]:
    reference_result = _reference(
        method,
        vectors,
        config,
        anchor=anchor,
        noise_variances=noise_variances,
        return_diagnostics=True,
    )
    reference, reference_diagnostics = reference_result
    loo_references = _leave_one_out_references(
        method,
        vectors,
        config,
        anchor=anchor,
        noise_variances=noise_variances,
    )
    raw_scores, score_diagnostics = _standardized_scores(
        vectors,
        loo_references,
        config,
        noise_variances=noise_variances,
        reference_variances=calibration["reference_variances"],
    )
    scores = raw_scores
    if weight_mode is not None:
        weights, weight_diagnostics = _oracle_weights(scores, config, mode=weight_mode)
        weight_rule = weight_mode
    elif method == "uniform_mean":
        weights = torch.full_like(scores, 1.0 / float(scores.numel()))
        weight_diagnostics = {
            "novelty": torch.zeros_like(scores),
            "trust": torch.ones_like(scores),
            "logits": torch.zeros_like(scores),
        }
        weight_rule = "uniform"
    else:
        weights, weight_diagnostics = _bandpass_far_weights(scores, config)
        weight_rule = "gaussian_aware_bandpass_far"

    server_clip = float(config["aggregation"]["server_clip_norm"])
    contributions = clip_l2(vectors, server_clip)
    aggregate = (weights[:, None] * contributions).sum(dim=0)
    honest_mask = ~byzantine_mask
    regular_honest_mask = honest_mask & ~outlier_mask
    honest_target = clean[honest_mask].mean(dim=0)
    outliers = honest_mask & outlier_mask
    threshold = float(calibration["score_threshold"].item())
    false_outlier_rate = float(
        (scores[regular_honest_mask] > threshold).to(scores.dtype).mean().item()
    )
    outlier_recall = (
        float((scores[outliers] > threshold).to(scores.dtype).mean().item())
        if bool(outliers.any())
        else float("nan")
    )
    tier_score_range, tier_reference_distance_range, tier_projection = _tier_metrics(
        clean=clean,
        reference=reference,
        scores=scores,
        tiers=tiers,
        regular_honest_mask=regular_honest_mask,
    )
    entropy = float((-(weights * weights.clamp_min(1e-15).log()).sum()).item())
    result: dict[str, Any] = {
        "reference": method,
        "weight_rule": weight_rule,
        "reference_error_to_honest_clean_mean": float(
            torch.linalg.vector_norm(reference - honest_target).item()
        ),
        "reference_error_to_population_centre": float(
            torch.linalg.vector_norm(reference - centre).item()
        ),
        "aggregate_error_to_honest_clean_mean": float(
            torch.linalg.vector_norm(aggregate - honest_target).item()
        ),
        "false_outlier_rate": false_outlier_rate,
        "honest_outlier_recall": outlier_recall,
        "score_calibration_mean_regular_honest": float(
            scores[regular_honest_mask].mean().item()
        ),
        "score_calibration_std_regular_honest": float(
            scores[regular_honest_mask].std(unbiased=False).item()
        ),
        "score_noise_std_correlation": _corr(
            scores[regular_honest_mask], tiers[regular_honest_mask]
        ),
        "noise_tier_mean_score_range": tier_score_range,
        "noise_tier_reference_distance_range": tier_reference_distance_range,
        "noise_tier_reference_bias_projection_abs": tier_projection,
        "honest_outlier_weight_mass": float(weights[outliers].sum().item()),
        "byzantine_weight_mass": float(weights[byzantine_mask].sum().item()),
        "max_individual_weight": float(weights.max().item()),
        "weight_concentration_n_sum_q2": float(
            weights.numel() * weights.square().sum().item()
        ),
        "weight_entropy": entropy,
        "effective_num_clients": float(1.0 / weights.square().sum().item()),
        "score_min": float(scores.min().item()),
        "score_max": float(scores.max().item()),
        "score_span": float((scores.max() - scores.min()).item()),
        "mean_trust_honest": float(
            weight_diagnostics["trust"][honest_mask].mean().item()
        ),
        "mean_trust_byzantine": (
            float(weight_diagnostics["trust"][byzantine_mask].mean().item())
            if bool(byzantine_mask.any())
            else float("nan")
        ),
        "server_clip_rate_honest": float(
            (torch.linalg.vector_norm(vectors[honest_mask], dim=1) > server_clip)
            .to(vectors.dtype)
            .mean()
            .item()
        ),
        "server_clip_rate_byzantine": (
            float(
                (torch.linalg.vector_norm(vectors[byzantine_mask], dim=1) > server_clip)
                .to(vectors.dtype)
                .mean()
                .item()
            )
            if bool(byzantine_mask.any())
            else float("nan")
        ),
        "quadratic_energy_mean": float(
            score_diagnostics["quadratic_energy"].mean().item()
        ),
        "reference_tail_fraction_client_blocks": float(
            reference_diagnostics.get("tail_fraction_client_blocks", float("nan"))
        ),
        "reference_finite_solver_replace_one_bound": float(
            reference_diagnostics.get("finite_solver_replace_one_bound", float("nan"))
        ),
        "reference_exact_minimizer_replace_one_bound": float(
            reference_diagnostics.get("exact_minimizer_replace_one_bound", float("nan"))
        ),
        "reference_fraction_covariance_limited_client_blocks": float(
            reference_diagnostics.get(
                "fraction_covariance_limited_client_blocks", float("nan")
            )
        ),
        "reference_covariance_branch_active": bool(
            reference_diagnostics.get("covariance_branch_active", False)
        ),
        "reference_gradient_residual_norm": float(
            reference_diagnostics.get("gradient_residual_norm", float("nan"))
        ),
    }
    return result


def _attach_paired_ratios(rows: list[dict[str, Any]]) -> None:
    baselines = {
        row["pairing_id"]: row for row in rows if row["reference"] == "uniform_mean"
    }
    for row in rows:
        baseline = baselines[row["pairing_id"]]
        ref_denominator = float(baseline["reference_error_to_honest_clean_mean"])
        agg_denominator = float(baseline["aggregate_error_to_honest_clean_mean"])
        row["reference_error_ratio_to_uniform"] = (
            float(row["reference_error_to_honest_clean_mean"]) / ref_denominator
            if ref_denominator > 0.0
            else float("nan")
        )
        row["aggregate_error_ratio_to_uniform"] = (
            float(row["aggregate_error_to_honest_clean_mean"]) / agg_denominator
            if agg_denominator > 0.0
            else float("nan")
        )


def _group_means(
    rows: list[dict[str, Any]], field: str
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["noise_regime"],
            row["noise_permutation"],
            row["outlier_geometry"],
            row["threat"],
            row["severity"],
        )
        grouped[key].append(float(row[field]))
    return {key: _finite_mean(values) for key, values in grouped.items()}


def _summarize(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    separated = set(str(value) for value in config["threats"]["separated_for_gates"])
    gates = config["gates"]
    summaries: list[dict[str, Any]] = []
    for method in [str(value) for value in config["references"]["candidates"]]:
        selected = [row for row in rows if row["reference"] == method]
        clean = [row for row in selected if row["threat"] == "none"]
        heteroscedastic_clean = [
            row for row in clean if row["noise_regime"] == "heteroscedastic"
        ]
        attacked = [row for row in selected if row["threat"] in separated]
        evasive = [
            row
            for row in selected
            if row["threat"] in set(config["threats"]["evasive_controls"])
        ]
        attacked_ref_groups = _group_means(attacked, "reference_error_ratio_to_uniform")
        attacked_agg_groups = _group_means(attacked, "aggregate_error_ratio_to_uniform")
        attacked_mass_groups = _group_means(attacked, "byzantine_weight_mass")
        clean_reference_ratio = _finite_mean(
            row["reference_error_ratio_to_uniform"] for row in clean
        )
        clean_aggregate_ratio = _finite_mean(
            row["aggregate_error_ratio_to_uniform"] for row in clean
        )
        attacked_reference_ratio = max(attacked_ref_groups.values())
        false_outlier_rate = _finite_mean(row["false_outlier_rate"] for row in clean)
        noise_correlation = _finite_mean(
            abs(float(row["score_noise_std_correlation"]))
            for row in heteroscedastic_clean
        )
        signed_noise_correlation = _finite_mean(
            row["score_noise_std_correlation"] for row in heteroscedastic_clean
        )
        tier_range = _finite_mean(
            row["noise_tier_mean_score_range"] for row in heteroscedastic_clean
        )
        outlier_recall = _finite_mean(row["honest_outlier_recall"] for row in clean)
        uniform_outlier_mass = float(
            config["cohort"]["honest_outliers"]["count"]
        ) / float(config["cohort"]["num_clients"])
        outlier_weight_mass = _finite_mean(
            row["honest_outlier_weight_mass"] for row in clean
        )
        outlier_weight_gain = outlier_weight_mass - uniform_outlier_mass
        byzantine_mass = max(attacked_mass_groups.values())
        aggregate_ratio = max(attacked_agg_groups.values())
        checks = {
            "clean_reference_error": clean_reference_ratio
            <= float(gates["clean_reference_error_ratio_to_uniform_max"]),
            "attacked_reference_error": attacked_reference_ratio
            <= float(gates["attacked_reference_error_ratio_to_uniform_max"]),
            "false_outlier_rate": float(gates["false_outlier_rate_min"])
            <= false_outlier_rate
            <= float(gates["false_outlier_rate_max"]),
            "noise_scale_invariance": noise_correlation
            <= float(gates["abs_score_noise_std_correlation_max"]),
            "noise_tier_invariance": tier_range
            <= float(gates["noise_tier_mean_score_range_max"]),
            "honest_outlier_recall": outlier_recall
            >= float(gates["honest_outlier_recall_min"]),
            "honest_outlier_weight_mass": outlier_weight_gain
            >= float(gates["honest_outlier_weight_mass_gain_over_uniform_min"]),
            "byzantine_weight_mass": byzantine_mass
            <= float(gates["separated_byzantine_weight_mass_max"]),
            "aggregate_error": aggregate_ratio
            <= float(gates["separated_aggregate_error_ratio_to_uniform_max"]),
        }
        summary: dict[str, Any] = {
            "reference": method,
            "observations": len(selected),
            "clean_reference_error_mean": _finite_mean(
                row["reference_error_to_honest_clean_mean"] for row in clean
            ),
            "clean_reference_error_std": _finite_std(
                row["reference_error_to_honest_clean_mean"] for row in clean
            ),
            "clean_reference_error_ratio_to_uniform_mean": clean_reference_ratio,
            "clean_aggregate_error_ratio_to_uniform_mean": clean_aggregate_ratio,
            "separated_attacked_reference_error_ratio_to_uniform_worst_group": (
                attacked_reference_ratio
            ),
            "false_outlier_rate_mean": false_outlier_rate,
            "abs_score_noise_std_correlation_mean": noise_correlation,
            "signed_score_noise_std_correlation_mean": signed_noise_correlation,
            "noise_tier_mean_score_range_mean": tier_range,
            "noise_tier_reference_bias_projection_abs_mean": _finite_mean(
                row["noise_tier_reference_bias_projection_abs"]
                for row in heteroscedastic_clean
            ),
            "honest_outlier_recall_mean": outlier_recall,
            "honest_outlier_weight_mass_mean": outlier_weight_mass,
            "honest_outlier_weight_mass_gain_over_uniform": outlier_weight_gain,
            "separated_byzantine_weight_mass_worst_group": byzantine_mass,
            "separated_aggregate_error_ratio_to_uniform_worst_group": aggregate_ratio,
            "evasive_byzantine_weight_mass_mean": _finite_mean(
                row["byzantine_weight_mass"] for row in evasive
            ),
            "evasive_aggregate_error_ratio_to_uniform_mean": _finite_mean(
                row["aggregate_error_ratio_to_uniform"] for row in evasive
            ),
            "reference_tail_fraction_mean": _finite_mean(
                row["reference_tail_fraction_client_blocks"] for row in selected
            ),
            "clean_reference_tail_fraction_mean": _finite_mean(
                row["reference_tail_fraction_client_blocks"] for row in clean
            ),
            "max_individual_weight_observed": max(
                float(row["max_individual_weight"]) for row in selected
            ),
        }
        for name, passed in checks.items():
            summary[f"gate_{name}"] = bool(passed)
        summary["passes_all_gates"] = all(checks.values())
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("All CSV rows must have exactly the same fields")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def _write_report(
    path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# Audit oracle des références Gaussian-aware",
        "",
        "## Verdict",
        "",
        (
            "Le candidat principal `f_sigma_huber` satisfait tous les gates "
            "synthétiques pré-enregistrés. Une confirmation Fashion-MNIST sur "
            "seeds tenues à l'écart est autorisée."
            if decision["primary_passes_all_gates"]
            else "Le candidat principal `f_sigma_huber` ne satisfait pas tous "
            "les gates synthétiques pré-enregistrés. Il ne doit pas être ajusté "
            "sur les tirages d'évaluation."
        ),
        "",
        "Ce résultat ne constitue ni une preuve Byzantine universelle, ni une "
        "validation d'utilité end-to-end. ALIE est un contrôle évasif rapporté, "
        "mais ne définit pas le claim conditionnel des attaques séparées.",
        "",
        "## Question isolée",
        "",
        "Le banc mesure séparément : (1) l'erreur de la référence par rapport "
        "à la moyenne honnête propre connue, (2) la calibration du score "
        "leave-one-out sous le nul DP, et (3) l'allocation des poids lorsqu'un "
        "honest outlier ou un message byzantin est présent. Aucune accuracy "
        "n'entre dans la sélection.",
        "",
        "## Randomness appariée",
        "",
        "Dans chaque `pairing_id`, les références reçoivent exactement les "
        "mêmes gradients propres, honest outliers, bruits gaussiens et messages "
        "d'attaque. Les simulations nulles utilisées pour calibrer les "
        "covariances des références sont indépendantes des seeds d'évaluation. "
        "Elles sont regroupées uniquement par tier de covariance DP public. "
        "Le score final n'est pas recentré client par client : la corrélation "
        "score–bruit reste donc une véritable métrique d'audit.",
        "",
        "## Résultats synthétiques",
        "",
        "| Référence | erreur propre / uniforme | erreur réf. attaquée, "
        "pire groupe | FOR | corrélation absolue score–bruit | rappel outliers | gain masse "
        "outliers | masse byz., pire groupe | erreur agrégat / uniforme | "
        "Tous gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            (
                "| {name} | {clean} | {attack} | {forate} | {corr} | "
                "{recall} | {gain} | {mass} | {aggregate} | {passed} |"
            ).format(
                name=row["reference"],
                clean=_fmt(row["clean_reference_error_ratio_to_uniform_mean"]),
                attack=_fmt(
                    row[
                        "separated_attacked_reference_error_ratio_to_uniform_worst_group"
                    ]
                ),
                forate=_fmt(row["false_outlier_rate_mean"]),
                corr=_fmt(row["abs_score_noise_std_correlation_mean"]),
                recall=_fmt(row["honest_outlier_recall_mean"]),
                gain=_fmt(row["honest_outlier_weight_mass_gain_over_uniform"]),
                mass=_fmt(row["separated_byzantine_weight_mass_worst_group"]),
                aggregate=_fmt(
                    row["separated_aggregate_error_ratio_to_uniform_worst_group"]
                ),
                passed="oui" if row["passes_all_gates"] else "non",
            )
        )
    primary = next(
        row for row in summaries if row["reference"] == decision["primary_candidate"]
    )
    fcc = next(row for row in summaries if row["reference"] == "fcc")
    lines.extend(
        [
            "",
            "## Lecture quantitative du candidat principal",
            "",
            "Observations :",
            "",
            f"- sans attaque, la référence Huber a une erreur moyenne de "
            f"`{_fmt(primary['clean_reference_error_ratio_to_uniform_mean'])}×` "
            "celle de la moyenne uniforme ;",
            f"- son agrégat pondé sans attaque a toutefois une erreur de "
            f"`{_fmt(primary['clean_aggregate_error_ratio_to_uniform_mean'])}×` "
            "celle de l'uniforme : la référence s'améliore, mais la règle de "
            "poids réintroduit ici un coût d'utilité ;",
            f"- sous les attaques séparées, son pire ratio d'erreur de référence "
            f"est `{_fmt(primary['separated_attacked_reference_error_ratio_to_uniform_worst_group'])}`, "
            f"contre `{_fmt(fcc['separated_attacked_reference_error_ratio_to_uniform_worst_group'])}` "
            "pour FCC ;",
            f"- la masse byzantine maximale tombe à "
            f"`{_fmt(primary['separated_byzantine_weight_mass_worst_group'])}` "
            "et l'erreur d'agrégat attaqué reste à "
            f"`{_fmt(primary['separated_aggregate_error_ratio_to_uniform_worst_group'])}×` "
            "l'uniforme ; ces deux gardes passent ;",
            f"- le rappel des honest outliers n'est que "
            f"`{_fmt(primary['honest_outlier_recall_mean'])}`, même si leur masse "
            f"totale gagne `{_fmt(primary['honest_outlier_weight_mass_gain_over_uniform'])}` "
            "au-dessus de la masse uniforme ;",
            f"- `{_fmt(primary['clean_reference_tail_fraction_mean'])}` des couples "
            "client–bloc honnêtes sont déjà dans la queue Huber sans attaque. "
            "La zone quadratique equal-client est donc peu active dans cette "
            "instanciation.",
            "",
            "Inférence : le principe de la référence est prometteur pour estimer "
            "le centre propre, mais cette instanciation ne domine pas FCC sous "
            "attaque et le couplage score–poids ne préserve pas assez les honest "
            "outliers. Elle n'est donc pas promue vers Fashion-MNIST.",
            "",
            "## Audit post-exécution de deux gates d'invariance",
            "",
            "Deux échecs booléens doivent être interprétés avec prudence. Le gate "
            "de corrélation utilise la moyenne de la valeur absolue d'une "
            "corrélation calculée sur seulement vingt clients honnêtes ordinaires "
            "par cohorte. Même sous indépendance, cette quantité est strictement "
            "positive par bruit d'échantillonnage. Ici elle vaut "
            f"`{_fmt(primary['abs_score_noise_std_correlation_mean'])}` pour Huber "
            f"et `{_fmt(next(row for row in summaries if row['reference'] == 'uniform_mean')['abs_score_noise_std_correlation_mean'])}` "
            "pour l'uniforme. La corrélation signée moyenne de Huber est seulement "
            f"`{_fmt(primary['signed_score_noise_std_correlation_mean'])}`.",
            "",
            "De même, la plage inter-tiers est calculée séparément dans chaque "
            "petite cohorte ; toutes les références donnent environ `0.64`. Le seuil "
            "absolu `0.25` n'a pas été centré par sa distribution nulle finie. Ces "
            "deux gates restent enregistrés comme échecs dans `decision.json`, mais "
            "ils ne prouvent pas, à eux seuls, que le score suit le niveau de bruit. "
            "Une version G0b devra préenregistrer une statistique agrégée ou un "
            "excès par rapport au nul et employer de nouvelles seeds tenues à l'écart.",
            "",
            "## Interprétation des métriques",
            "",
            "- `FOR` est le taux de faux outliers parmi les clients honnêtes "
            "ordinaires. Son seuil est le quantile à 95 % des simulations "
            "nulles indépendantes, regroupé par méthode et régime de bruit, "
            "mais jamais par identité cliente. Une valeur proche de 5 % "
            "indique une calibration cohérente.",
            "- `corr(score, σ)` et la plage inter-tiers vérifient que le score "
            "ne classe pas simplement les clients selon leur niveau de bruit.",
            "- Le rappel mesure la fraction des honest outliers réellement "
            "retrouvés ; il ne doit pas être obtenu en donnant aussi une masse "
            "forte aux Byzantins.",
            "- L'erreur de référence est calculée avant le clipping serveur. "
            "L'erreur d'agrégat utilise les contributions bornées à "
            f"`U={config['aggregation']['server_clip_norm']}`.",
            "",
            "## Construction de `f_sigma_huber`",
            "",
            "Le seuil radial est déterminé par la borne gaussienne de "
            "Laurent–Massart pour chaque bloc de dimension 16, avec une "
            "probabilité de queue publique de 1 %. Il ne s'agit donc pas d'un "
            "seuil scalaire 2,5 appliqué à tort à une norme de dimension 16. "
            "Le cap euclidien par bloc vaut `0.065 = 0.13/sqrt(4)`. La norme "
            "du gradient d'influence d'un client est donc bornée par `0.13`, "
            "comme le rayon total du comparateur FCC. Les certificats ne sont "
            "malgré cela pas identiques : la régularisation et les 20 étapes "
            "du solveur Huber interviennent aussi dans sa stabilité.",
            "",
            "## Fichiers",
            "",
            f"- Configuration : `{config_path.resolve()}`",
            f"- Tirages appariés : `{(output_dir / 'paired_detail.csv').resolve()}`",
            f"- Calibration publique : `{(output_dir / 'null_calibration.csv').resolve()}`",
            f"- Synthèse : `{(output_dir / 'summary.csv').resolve()}`",
            f"- Décision : `{(output_dir / 'decision.json').resolve()}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    device: str | torch.device = "cpu",
    calibration_draws_override: int | None = None,
    draws_per_seed_override: int | None = None,
    seeds_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    runtime_device, runtime_dtype = _configure_runtime(device)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    config["execution"] = {
        "requested_device": str(device),
        "resolved_device": str(runtime_device),
        "tensor_dtype": str(runtime_dtype).removeprefix("torch."),
        "silent_cpu_fallback_allowed": False,
    }
    calibration_draws = int(
        calibration_draws_override
        if calibration_draws_override is not None
        else config["randomness"]["calibration_draws"]
    )
    draws_per_seed = int(
        draws_per_seed_override
        if draws_per_seed_override is not None
        else config["randomness"]["draws_per_seed"]
    )
    seeds = tuple(
        int(value)
        for value in (
            seeds_override
            if seeds_override is not None
            else config["randomness"]["evaluation_seeds"]
        )
    )
    if calibration_draws < 2 or draws_per_seed < 1 or not seeds:
        raise ValueError("Need >=2 calibration draws, >=1 evaluation draw and seeds")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    methods = tuple(str(value) for value in config["references"]["candidates"])
    block_sizes = tuple(int(value) for value in config["cohort"]["block_sizes"])
    calibration_seed = int(config["randomness"]["calibration_seed"])
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for regime in config["privacy_noise"]["regimes"]:
        regime_name = str(regime["name"])
        for permutation in regime["permutations"]:
            permutation = str(permutation)
            noise_variances, tiers = _noise_variances(config, regime, permutation)
            calibration = _calibrate_reference_variances(
                config,
                regime=regime,
                permutation=permutation,
                draws=calibration_draws,
                seed=calibration_seed,
            )
            for method in methods:
                for client in range(int(config["cohort"]["num_clients"])):
                    for block in range(len(block_sizes)):
                        calibration_rows.append(
                            {
                                "noise_regime": regime_name,
                                "noise_permutation": permutation,
                                "reference": method,
                                "client": client,
                                "block": block,
                                "public_noise_variance": float(
                                    noise_variances[client, block].item()
                                ),
                                "reference_variance": float(
                                    calibration[method]["reference_variances"][
                                        client, block
                                    ].item()
                                ),
                                "raw_score_null_mean": float(
                                    calibration[method]["score_mean"][client].item()
                                ),
                                "raw_score_null_std": float(
                                    calibration[method]["score_std"][client].item()
                                ),
                                "pooled_null_score_threshold": float(
                                    calibration[method]["score_threshold"].item()
                                ),
                            }
                        )

            for seed in seeds:
                for draw in range(draws_per_seed):
                    for geometry in config["cohort"]["honest_outliers"]["geometries"]:
                        clean, outliers, centre, anchor = _honest_clean_vectors(
                            config,
                            seed=seed,
                            draw=draw,
                            geometry=str(geometry),
                            include_outliers=True,
                        )
                        observed = _add_private_noise(
                            clean,
                            noise_variances,
                            block_sizes,
                            seed=seed,
                            draw=draw,
                            regime=regime_name,
                            permutation=permutation,
                            geometry=str(geometry),
                        )
                        for threat in config["threats"]["names"]:
                            severities = (
                                [1.0]
                                if threat == "none"
                                else config["threats"]["severities"]
                            )
                            for severity in severities:
                                attacked, byzantine_mask = _replace_with_attack(
                                    observed,
                                    config,
                                    threat=str(threat),
                                    severity=float(severity),
                                    seed=_seed(seed, draw, geometry, threat),
                                )
                                pairing_id = (
                                    f"{regime_name}:{permutation}:{geometry}:"
                                    f"{threat}:{float(severity):.3f}:{seed}:{draw}"
                                )
                                for method in methods:
                                    metrics = _evaluate_reference(
                                        method=method,
                                        vectors=attacked,
                                        clean=clean,
                                        outlier_mask=outliers,
                                        byzantine_mask=byzantine_mask,
                                        noise_variances=noise_variances,
                                        tiers=tiers,
                                        anchor=anchor,
                                        centre=centre,
                                        calibration=calibration[method],
                                        config=config,
                                    )
                                    rows.append(
                                        {
                                            "campaign_id": config["campaign_id"],
                                            "pairing_id": pairing_id,
                                            "noise_regime": regime_name,
                                            "noise_permutation": permutation,
                                            "outlier_geometry": str(geometry),
                                            "threat": str(threat),
                                            "severity": float(severity),
                                            "seed": seed,
                                            "draw": draw,
                                            **metrics,
                                        }
                                    )

    _attach_paired_ratios(rows)
    summaries = _summarize(rows, config)
    primary_name = str(config["decision"]["primary_candidate"])
    primary = next(row for row in summaries if row["reference"] == primary_name)
    decision = {
        "campaign_id": config["campaign_id"],
        "config": str(config_path.resolve()),
        "requested_device": str(device),
        "resolved_device": str(runtime_device),
        "tensor_dtype": str(runtime_dtype).removeprefix("torch."),
        "silent_cpu_fallback_allowed": False,
        "primary_candidate": primary_name,
        "primary_passes_all_gates": bool(primary["passes_all_gates"]),
        "primary_gate_checks": {
            key.removeprefix("gate_"): value
            for key, value in primary.items()
            if key.startswith("gate_")
        },
        "accuracy_used_for_selection": False,
        "calibration_draws": calibration_draws,
        "draws_per_seed": draws_per_seed,
        "evaluation_seeds": list(seeds),
        "paired_rows": len(rows),
    }
    _write_csv(output_dir / "paired_detail.csv", rows)
    _write_csv(output_dir / "null_calibration.csv", calibration_rows)
    _write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        config_path=config_path,
        output_dir=output_dir,
        config=config,
        summaries=summaries,
        decision=decision,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ldp_gradient_far/gaussian_aware_reference_oracle.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/ldp_gradient_far/gaussian_aware_reference_oracle_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "output/analysis/Gaussian_Aware_Robust_Reference_Oracle_Audit.md",
    )
    parser.add_argument("--calibration-draws", type=int)
    parser.add_argument("--draws-per-seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help=(
            "Tensor execution device. The CLI defaults to MPS and refuses an "
            "unavailable MPS device instead of silently falling back to CPU."
        ),
    )
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.report.resolve(),
        device=args.device,
        calibration_draws_override=args.calibration_draws,
        draws_per_seed_override=args.draws_per_seed,
        seeds_override=args.seeds,
    )


if __name__ == "__main__":
    main()
