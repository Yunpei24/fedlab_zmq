#!/usr/bin/env python3
"""Frozen-update Monte-Carlo audit for the Stage-7 FAR score hypothesis.

This is a mechanism-identification experiment, not an FL benchmark.  The
clean client vectors are frozen while only Gaussian DP noise is resampled.
The first draws estimate the effective residual covariance after server
clipping and the selected reference.  Disjoint draws evaluate whether a
noise-floor-subtracted energy score preserves honest geometric outliers while
discarding differences caused only by heterogeneous privacy noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.noise_aware_scores import (
    debiased_distance_scores,
    excess_energy_scores,
    standardize_distances,
)
from robustness.aggregators import aggregate_vectors, centered_clipping, clip_l2


def _corr(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x = x.double() - x.double().mean()
    y = y.double() - y.double().mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-15:
        return None
    return float((x @ y / denominator).item())


def _reference(vectors: torch.Tensor, name: str, rho: float) -> torch.Tensor:
    if name == "fcc":
        return centered_clipping(
            vectors,
            anchor=torch.zeros(vectors.shape[1], dtype=vectors.dtype),
            tau=rho,
        )
    if name == "rfa":
        return aggregate_vectors(
            vectors,
            "rfa",
            num_byzantine=0,
            max_iter=100,
            tol=1e-7,
            smoothing=1e-8,
        )
    raise ValueError(name)


def _frozen_clean_vectors(
    seed: int, *, n: int, dimension: int, honest_outliers: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    centre = torch.randn(dimension, generator=generator, dtype=torch.float64)
    centre = 0.04 * centre / torch.linalg.vector_norm(centre)
    deviations = torch.randn(n, dimension, generator=generator, dtype=torch.float64)
    deviations = 0.025 * deviations / torch.linalg.vector_norm(
        deviations, dim=1, keepdim=True
    )
    vectors = centre + deviations
    mask = torch.zeros(n, dtype=torch.bool)
    if honest_outliers:
        ids = torch.tensor([5, 9, 13, 17, 21])
        outlier_directions = torch.randn(
            len(ids), dimension, generator=generator, dtype=torch.float64
        )
        outlier_directions = outlier_directions / torch.linalg.vector_norm(
            outlier_directions, dim=1, keepdim=True
        )
        vectors[ids] += 0.13 * outlier_directions
        mask[ids] = True
    return vectors, mask


def _noise_scales(permutation: str, n: int) -> torch.Tensor:
    base = torch.tensor(([1.0, 1.5, 2.0] * 9)[:n], dtype=torch.float64)
    if permutation == "identity":
        return base
    if permutation == "rotate7":
        return torch.roll(base, shifts=7)
    if permutation == "reverse":
        return torch.flip(base, dims=(0,))
    raise ValueError(permutation)


def _draw_residuals(
    clean: torch.Tensor,
    scales: torch.Tensor,
    *,
    draws: int,
    seed: int,
    noise_std: float,
    server_clip: float,
    reference: str,
    rho: float,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(draws):
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        noisy = clip_l2(clean + noise_std * scales[:, None] * noise, server_clip)
        ref = _reference(noisy, reference, rho)
        rows.append(noisy - ref)
    return torch.stack(rows)


def _score_rows(
    residuals: torch.Tensor,
    *,
    mode: str,
    scales: torch.Tensor,
    residual_variances: torch.Tensor,
    distance_scale: float,
    z_clip: float,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(residuals, dim=2)
    if mode == "raw":
        return (distances / distance_scale).clamp(max=1.0)
    if mode == "stage6_covariance_proxy":
        rows = []
        for row in distances:
            corrected, _, _ = standardize_distances(
                row,
                scales,
                mode="isotropic_dp_covariance_proxy",
            )
            rows.append((corrected / distance_scale).clamp(max=1.0))
        return torch.stack(rows)
    if mode == "excess_energy_empirical_covariance":
        rows = []
        for row in distances:
            score, _ = excess_energy_scores(
                row,
                residual_variances,
                score_dimension=residuals.shape[2],
                z_clip=z_clip,
            )
            rows.append(score)
        return torch.stack(rows)
    if mode == "debiased_distance_empirical_covariance":
        rows = []
        for row in distances:
            score, _ = debiased_distance_scores(
                row,
                residual_variances,
                score_dimension=residuals.shape[2],
                distance_clip=distance_scale,
            )
            rows.append(score)
        return torch.stack(rows)
    raise ValueError(mode)


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True))


def run(
    output_dir: Path,
    calibration_draws: int,
    evaluation_draws: int,
    *,
    include_debiased_distance: bool = False,
) -> None:
    n = 25
    dimension = 256
    server_clip = 0.42
    distance_scale = 0.441
    rho = 0.21
    noise_std = 0.015
    z_clip = 5.0
    alpha = math.log(2.0 * (n - 1) / (n - 2.0))
    modes = ["raw", "stage6_covariance_proxy", "excess_energy_empirical_covariance"]
    if include_debiased_distance:
        modes.append("debiased_distance_empirical_covariance")
    detail_rows: list[dict] = []

    for seed in (28, 36, 54):
        for honest_outliers in (False, True):
            clean, outlier_mask = _frozen_clean_vectors(
                seed, n=n, dimension=dimension, honest_outliers=honest_outliers
            )
            for permutation in ("identity", "rotate7", "reverse"):
                scales = _noise_scales(permutation, n)
                for reference in ("fcc", "rfa"):
                    calibration = _draw_residuals(
                        clean,
                        scales,
                        draws=calibration_draws,
                        seed=seed * 1000 + 17,
                        noise_std=noise_std,
                        server_clip=server_clip,
                        reference=reference,
                        rho=rho,
                    )
                    calibration_mean = calibration.mean(dim=0)
                    residual_variances = (
                        (calibration - calibration_mean[None, :, :])
                        .square()
                        .mean(dim=(0, 2))
                        .clamp_min(1e-12)
                    )
                    evaluation = _draw_residuals(
                        clean,
                        scales,
                        draws=evaluation_draws,
                        seed=seed * 1000 + 97,
                        noise_std=noise_std,
                        server_clip=server_clip,
                        reference=reference,
                        rho=rho,
                    )
                    clean_reference = _reference(clean, reference, rho)
                    clean_distances = torch.linalg.vector_norm(
                        clean - clean_reference, dim=1
                    )
                    target_scores = (clean_distances / distance_scale).clamp(max=1.0)
                    target_weights = torch.softmax(alpha * target_scores, dim=0)
                    for mode in modes:
                        scores = _score_rows(
                            evaluation,
                            mode=mode,
                            scales=scales,
                            residual_variances=residual_variances,
                            distance_scale=distance_scale,
                            z_clip=z_clip,
                        )
                        for draw_index, score in enumerate(scores):
                            weights = torch.softmax(alpha * score, dim=0)
                            top_ids = torch.topk(score, k=5).indices
                            top_outlier_recall = (
                                float(outlier_mask[top_ids].float().mean())
                                if honest_outliers
                                else float("nan")
                            )
                            detail_rows.append(
                                {
                                    "seed": seed,
                                    "honest_outliers": honest_outliers,
                                    "noise_permutation": permutation,
                                    "reference": reference,
                                    "score_mode": mode,
                                    "draw": draw_index,
                                    "score_clean_geometry_corr": _corr(
                                        score, clean_distances
                                    ),
                                    "score_noise_scale_corr": _corr(score, scales),
                                    "weight_l1_to_clean_target": float(
                                        torch.linalg.vector_norm(
                                            weights - target_weights, ord=1
                                        )
                                    ),
                                    "honest_outlier_weight_mass": (
                                        float(weights[outlier_mask].sum())
                                        if honest_outliers
                                        else None
                                    ),
                                    "honest_outlier_top5_recall": (
                                        top_outlier_recall
                                        if honest_outliers
                                        else None
                                    ),
                                    "score_span": float(score.max() - score.min()),
                                    "weight_concentration": float(
                                        n * weights.square().sum()
                                    ),
                                }
                            )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "frozen_noise_trials.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    group_keys = ("honest_outliers", "reference", "score_mode")
    grouped: dict[tuple, list[dict]] = {}
    for row in detail_rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    summary = []
    metrics = (
        "score_clean_geometry_corr",
        "score_noise_scale_corr",
        "weight_l1_to_clean_target",
        "honest_outlier_weight_mass",
        "honest_outlier_top5_recall",
        "score_span",
        "weight_concentration",
    )
    for key, rows in sorted(grouped.items()):
        item = dict(zip(group_keys, key))
        for metric in metrics:
            values = [
                float(row[metric])
                for row in rows
                if row[metric] is not None and math.isfinite(float(row[metric]))
            ]
            if values:
                item[f"{metric}_mean"], item[f"{metric}_std"] = _mean_std(values)
        summary.append(item)

    summary_path = output_dir / "frozen_noise_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(group_keys)
        fieldnames.extend(
            sorted({name for row in summary for name in row if name not in group_keys})
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "status": "completed",
        "calibration_draws": calibration_draws,
        "evaluation_draws": evaluation_draws,
        "num_trials": len(detail_rows),
        "n": n,
        "dimension": dimension,
        "server_clip": server_clip,
        "distance_scale": distance_scale,
        "rho": rho,
        "noise_std": noise_std,
        "z_clip": z_clip,
        "alpha": alpha,
        "score_modes": modes,
        "includes_debiased_distance": include_debiased_distance,
        "references": ["fcc", "rfa"],
        "noise_permutations": ["identity", "rotate7", "reverse"],
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dt_ldp_far/decisive/stage7_frozen_noise"),
    )
    parser.add_argument("--calibration-draws", type=int, default=200)
    parser.add_argument("--evaluation-draws", type=int, default=300)
    parser.add_argument(
        "--include-debiased-distance",
        action="store_true",
        help="Add the unstandardized noise-floor-subtracted distance score.",
    )
    args = parser.parse_args()
    run(
        args.output_dir,
        args.calibration_draws,
        args.evaluation_draws,
        include_debiased_distance=args.include_debiased_distance,
    )


if __name__ == "__main__":
    main()
