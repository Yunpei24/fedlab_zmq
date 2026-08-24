#!/usr/bin/env python3
"""Export historical DMD prototype CSVs to the FedLab dashboard schema.

The scientific DMD runner predates the framework-wide ``metrics.json``
contract.  This adapter is intentionally lossless: every round column is kept,
while a small set of unit-explicit aliases is added for the native dashboard
panels.  Source CSVs are never modified.

Examples
--------
Export one campaign into the derived dashboard tree::

    python scripts/export_dmd_to_dashboard.py \
      --input-root results/dmd_cb_confirmatory_150_mps

Choose another derived dashboard root::

    python scripts/export_dmd_to_dashboard.py \
      --input-root results \
      --output-root results/dashboard_dmd \
      --campaign-glob 'dmd_*'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUND_RE = re.compile(r"^round_metrics_(?P<method>.+)_seed(?P<seed>\d+)\.csv$")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DMDArtifact:
    round_csv: Path
    client_csv: Path | None
    method: str
    seed: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON values (no NaN/Inf)."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def _infer_metadata(path: Path) -> dict[str, Any]:
    lowered = "/".join(part.lower() for part in path.parts)
    dataset = "emnist" if "emnist" in lowered else "cifar10" if "cifar10" in lowered else "unknown"
    if "resnet18" in lowered:
        model = "resnet18_gn4"
    elif "cnn" in lowered and "gn" in lowered:
        model = "cnn_gn"
    elif dataset == "cifar10":
        model = "resnet8_gn4"
    else:
        model = "unknown"

    alpha = None
    for part in path.parts:
        match = re.fullmatch(r"alpha_([0-9]+)p([0-9]+)", part.lower())
        if match:
            alpha = float(f"{match.group(1)}.{match.group(2)}")
            break
    return {"dataset": dataset, "model": model, "alpha": alpha}


def discover_artifacts(input_root: Path, campaign_glob: str | None = None) -> list[DMDArtifact]:
    roots: Iterable[Path]
    if campaign_glob:
        roots = sorted(path for path in input_root.glob(campaign_glob) if path.is_dir())
    else:
        roots = (input_root,)

    artifacts: list[DMDArtifact] = []
    seen: set[Path] = set()
    for root in roots:
        for round_csv in sorted(root.rglob("round_metrics_*_seed*.csv")):
            if "dashboard_exports" in round_csv.parts:
                continue
            resolved = round_csv.resolve()
            if resolved in seen:
                continue
            match = ROUND_RE.match(round_csv.name)
            if not match:
                continue
            method = match.group("method")
            seed = int(match.group("seed"))
            client_csv = round_csv.with_name(f"client_metrics_{method}_seed{seed}.csv")
            artifacts.append(
                DMDArtifact(
                    round_csv=round_csv,
                    client_csv=client_csv if client_csv.exists() else None,
                    method=method,
                    seed=seed,
                )
            )
            seen.add(resolved)
    return artifacts


def _attach_client_distributions(rounds: pd.DataFrame, clients: pd.DataFrame | None) -> pd.DataFrame:
    if clients is None or clients.empty or "round" not in clients:
        return rounds
    output = rounds.copy()
    fields = {
        "client_id": "evaluated_client_ids_oracle",
        "accuracy": "client_accuracy_values_oracle",
        "balanced_accuracy": "client_balanced_accuracy_values_oracle",
        "loss": "client_loss_values_oracle",
        "canonical_cb_deficit": "client_dmd_cb_values_oracle",
    }
    grouped = clients.groupby("round", sort=False)
    for source, target in fields.items():
        if source not in clients:
            continue
        mapping = grouped[source].apply(
            lambda values: [_safe(value) for value in values.tolist()]
        )
        output[target] = output["round"].map(mapping)
    return output


def _source_sidecar(path: Path) -> dict[str, Any] | None:
    """Read a small scientific sidecar without mutating or interpreting it."""

    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        content = None
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "content": _safe(content),
    }


def _add_dashboard_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "round" in output and "round_num" not in output:
        output["round_num"] = output["round"]

    aliases = {
        # Standard accuracy-space panel (ordinary client accuracy).
        "worst20_accuracy": ("worst20_accuracy_pct", 100.0),
        "client_accuracy_variance": ("client_accuracy_variance_pct2", 10_000.0),
        "weighted_client_loss_variance": ("balanced_performance_fairness", 1.0),
        # DMD-specific class-balanced performance panel.
        "mean_client_balanced_accuracy": ("mean_client_balanced_accuracy_pct", 100.0),
        "worst20_client_balanced_accuracy": ("worst20_balanced_accuracy_pct", 100.0),
        "client_balanced_accuracy_variance": (
            "client_balanced_accuracy_variance_pct2",
            10_000.0,
        ),
        "best_worst_client_balanced_accuracy_gap": (
            "best_worst_balanced_accuracy_gap_pct",
            100.0,
        ),
    }
    for source, (target, scale) in aliases.items():
        if source in output and target not in output:
            output[target] = pd.to_numeric(output[source], errors="coerce") * scale
    return output


def _summary(frame: pd.DataFrame, metadata: dict[str, Any], artifact: DMDArtifact) -> dict[str, Any]:
    accuracy = pd.to_numeric(frame.get("test_accuracy", pd.Series(dtype=float)), errors="coerce")
    loss = pd.to_numeric(frame.get("test_loss", pd.Series(dtype=float)), errors="coerce")
    valid_accuracy = accuracy.dropna()
    valid_loss = loss.dropna()
    return {
        "algorithm": artifact.method,
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "alpha": metadata["alpha"],
        "seed": artifact.seed,
        "num_rounds": int(pd.to_numeric(frame["round"], errors="coerce").max()),
        "best_accuracy": float(valid_accuracy.max()) if not valid_accuracy.empty else None,
        "final_accuracy": float(valid_accuracy.iloc[-1]) if not valid_accuracy.empty else None,
        "final_test_loss": float(valid_loss.iloc[-1]) if not valid_loss.empty else None,
    }


def output_directory(artifact: DMDArtifact, input_root: Path, output_root: Path | None) -> Path:
    leaf = f"{artifact.method}_seed{artifact.seed}"
    if output_root is None:
        return artifact.round_csv.parent / "dashboard_exports" / leaf
    try:
        relative_parent = artifact.round_csv.parent.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative_parent = Path(artifact.round_csv.parent.name)
    return output_root / relative_parent / leaf


def export_artifact(
    artifact: DMDArtifact,
    *,
    input_root: Path,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    destination = output_directory(artifact, input_root, output_root)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not overwrite:
        return metrics_path

    rounds = pd.read_csv(artifact.round_csv)
    if rounds.empty or "round" not in rounds:
        raise ValueError(f"{artifact.round_csv}: expected non-empty CSV with a round column")
    if rounds["round"].duplicated().any():
        raise ValueError(f"{artifact.round_csv}: duplicate round numbers")
    clients = pd.read_csv(artifact.client_csv) if artifact.client_csv else None
    rounds = _attach_client_distributions(rounds, clients)
    rounds = _add_dashboard_aliases(rounds)

    metadata = _infer_metadata(artifact.round_csv)
    summary = _summary(rounds, metadata, artifact)
    config = {
        "algorithm": artifact.method,
        "seed": artifact.seed,
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "alpha": metadata["alpha"],
        "source_format": "dmd_prototype_csv",
        "algo_config": {},
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": artifact.method,
        "dataset": metadata["dataset"],
        "config": config,
        "rounds": [_safe(row) for row in rounds.to_dict(orient="records")],
        "summary": _safe(summary),
        "export": {
            "adapter": "scripts/export_dmd_to_dashboard.py",
            "source_round_csv": str(artifact.round_csv.resolve()),
            "source_round_sha256": _sha256(artifact.round_csv),
            "source_client_csv": (
                str(artifact.client_csv.resolve()) if artifact.client_csv else None
            ),
            "source_client_sha256": (
                _sha256(artifact.client_csv) if artifact.client_csv else None
            ),
            "source_manifest": _source_sidecar(
                artifact.round_csv.parent / "manifest.json"
            ),
            "source_summary": _source_sidecar(
                artifact.round_csv.parent / "summary.json"
            ),
            "semantic_warning": (
                "best_worst_balanced_accuracy_gap_pct is best-client minus "
                "worst-client, not Best-20 minus Worst-20"
            ),
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destination / "manifest.json").write_text(
        json.dumps(payload["export"], indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "results",
        help="Campaign or results root scanned recursively for DMD round CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "dashboard_dmd",
        help="Derived dashboard root. Source result directories are never modified.",
    )
    parser.add_argument(
        "--campaign-glob",
        default=None,
        help="Optional glob relative to input-root, for example 'dmd_*'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = discover_artifacts(args.input_root, args.campaign_glob)
    if not artifacts:
        raise SystemExit(f"No DMD round_metrics CSV found under {args.input_root}")
    print(f"Discovered {len(artifacts)} DMD trajectories")
    for artifact in artifacts:
        target = output_directory(artifact, args.input_root, args.output_root) / "metrics.json"
        if args.dry_run:
            print(f"[dry-run] {artifact.round_csv} -> {target}")
            continue
        exported = export_artifact(
            artifact,
            input_root=args.input_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        print(exported)


if __name__ == "__main__":
    main()
