"""
core/seeding.py
===============
Single, centralized seeding utility for FedLab ZMQ experiments.

Every entry point (run_experiment.py and the cost-model scripts) calls
``seed_everything(seed)`` so that a given seed reproduces the same run.
It seeds, in one place, the four RNG sources that influence an experiment:

    python ``random``        — fleet/battery sampling fallbacks
    numpy  ``np.random``     — data partitioning, fleet generation
    torch  CPU generator     — model init, dropout, DataLoader shuffling
    torch  CUDA generator(s) — same, on GPU (no-op on CPU/MPS)

Note on MPS / Apple Silicon: ``torch.manual_seed`` already seeds the MPS
generator, so no separate call is needed.

Residual non-determinism (documented, NOT silently forced)
----------------------------------------------------------
This utility does **not** flip ``torch.use_deterministic_algorithms(True)``
or ``torch.backends.cudnn.deterministic`` by default, because doing so would
change kernel selection and therefore the numerical results — which would
break the bit-for-bit ``cost_model="phi"`` regression guarantee. Sources of
residual run-to-run variation a reviewer should be aware of:

  * cuDNN may pick non-deterministic convolution algorithms (CUDA only).
  * MPS (Apple Silicon) reduction order is not guaranteed bit-stable across
    torch versions.
  * Multi-worker DataLoader ordering depends on OS thread scheduling unless
    ``num_workers=0`` (the default in this repo's in-process runner).
  * FlopCounterMode FLOP *counts* are deterministic, but the absolute energy
    figures depend on the pinned torch version (see requirements.txt).

Pass ``deterministic=True`` to opt into the strict (slower, kernel-changing)
mode for a clean-room determinism check; leave it False to reproduce the
paper numbers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """
    Seed python/numpy/torch (CPU+CUDA) from a single integer.

    Args:
        seed:          the master seed.
        deterministic: if True, additionally force deterministic torch
                       algorithms and cuDNN. This CHANGES numerical results
                       vs. the default path and must NOT be used when
                       reproducing the paper's phi-model numbers. Default
                       False.

    Returns:
        The seed (for convenience / logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Opt-in strict mode — slower and may alter results. Not the default.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
