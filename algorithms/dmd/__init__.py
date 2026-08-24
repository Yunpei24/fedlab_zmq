"""Decision-Margin Deficit algorithms and reusable mathematical primitives."""

from .algorithm import DMDMeanAlgorithm, DMDTailAlgorithm, DMDUSVAlgorithm
from .config import DMDConfig
from .contracts import DMDClientReport, DMDRoundContext, MarginProfile
from .metrics import *  # noqa: F401,F403
from .objectives import *  # noqa: F401,F403
from .profiles import *  # noqa: F401,F403
from .references import *  # noqa: F401,F403
from .tail_risk import weighted_upper_cvar

__all__ = [
    "DMDConfig",
    "DMDClientReport",
    "DMDRoundContext",
    "MarginProfile",
    "DMDMeanAlgorithm",
    "DMDUSVAlgorithm",
    "DMDTailAlgorithm",
    "weighted_upper_cvar",
]
