"""
diagnostics/__init__.py
=======================
Diagnostics and analysis tools for FL training.

Available:
  - LayerMismatchDiagnostic: Measures feature extractor vs classifier misalignment
  - spectral_entropy:        Shannon entropy of a singular value distribution
  - layer_spectral_entropy:  Per-tensor spectral entropy (handles reshape)
  - round_spectral_entropy:  Per-layer entropy for a full gradient dict
  - rank_by_elbow:           Elbow/Kneedle rank selection
  - rank_by_budget:          Byte-budget rank selection
"""

from diagnostics.layer_mismatch import LayerMismatchDiagnostic
from diagnostics.spectral_quality import (
    spectral_entropy,
    layer_spectral_entropy,
    round_spectral_entropy,
    rank_by_elbow,
    rank_by_budget,
)

__all__ = [
    "LayerMismatchDiagnostic",
    "spectral_entropy",
    "layer_spectral_entropy",
    "round_spectral_entropy",
    "rank_by_elbow",
    "rank_by_budget",
]
