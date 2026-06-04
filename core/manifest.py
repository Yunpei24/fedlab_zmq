"""
core/manifest.py
================
Per-run reproducibility manifest. Every experiment writes a ``manifest.json``
next to its ``metrics.json`` capturing exactly what produced the numbers:
the resolved config, the git commit, the seed, the FLOP-counter convention,
package versions, platform, and a UTC timestamp.

This is the single artifact a reviewer needs to answer "what was this run?"
without re-deriving it from the directory name.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Packages whose exact version can change numerical results or the cost model.
_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "pyzmq",
    "msgpack",
    "PyYAML",
)


def _git_commit(repo_root: Path) -> dict[str, Any]:
    """Best-effort git provenance. Never raises."""
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        # dirty == uncommitted changes present
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["dirty"] = bool(status)
    except Exception:
        pass
    return info


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _flop_convention() -> float | None:
    """The FlopCounterMode MAC/FLOP factor (1.0 or 2.0), or None if unavailable."""
    try:
        from hardware.flop_cost import calibrate_convention

        return float(calibrate_convention())
    except Exception:
        return None


def write_manifest(
    out_dir: str | Path,
    resolved_config: dict[str, Any],
    seed: int,
    repo_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Write ``manifest.json`` into ``out_dir`` and return its path.

    Args:
        out_dir:         the run's results directory.
        resolved_config: the fully merged config actually used for the run.
        seed:            the master seed passed to seed_everything.
        repo_root:       repo root for git provenance (defaults to two levels
                         up from this file).
        extra:           optional extra key/values to fold in (e.g. cost_model,
                         device).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git": _git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "flop_convention_factor": _flop_convention(),
        "packages": _package_versions(),
        "resolved_config": resolved_config,
    }
    if extra:
        manifest.update(extra)

    path = out_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
