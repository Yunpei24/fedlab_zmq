"""
algorithms/__init__.py
======================
Auto-imports all algorithm modules so their @register_algorithm decorators fire.
Add your algorithm here after creating its module.
"""

from algorithms.base import (
    FLAlgorithm,
    ClientState,
    AggregateResult,
    register_algorithm,
    get_algorithm,
    list_algorithms,
)

# ── Built-in algorithms ───────────────────────────────────────────────────────
from algorithms import fedavg       # FedAvg  (McMahan et al., 2017)
from algorithms import eceffl       # E-CEFFL (Nikiema & Amhoud, 2025)
from algorithms import leanfed      # LeanFed (Pereira et al., 2025)
from algorithms import fedbacys     # FedBacys (Jeong et al., 2025)
from algorithms import vaishnav     # Vaishnav (Vaishnav et al., 2024)
from algorithms import fedsparq     # FedSparQ (Medjadji et al., 2025)
from algorithms import fedprox      # FedProx  (Li et al., MLSys 2020)
from algorithms import scaffold     # SCAFFOLD (Karimireddy et al., ICML 2020)
from algorithms import fed_resonance          # Fed-Resonance        (Nikiema & Amhoud, UM6P 2026)
from algorithms import fed_osmosis            # Fed-Osmosis          (Nikiema & Amhoud, UM6P 2026)
from algorithms import fed_resonance_osmosis  # Fed-Resonance+Osmosis (Nikiema & Amhoud, UM6P 2026)
from algorithms import fed_grad_align         # FedGradAlign         (Nikiema & Amhoud, UM6P 2026)
from algorithms import fed_resonance_plus     # Fed-Resonance+       (Nikiema & Amhoud, UM6P 2026)
from algorithms import fedpart                # FedPart              (Wang et al., NeurIPS 2024)
from algorithms import fedpart_be             # FedPartBE            (Battery-Energy-aware FedPart)
from algorithms import fedpart_universal      # FedPartUniversal     (Nikiema & Amhoud, UM6P 2026)
from algorithms import fedpart_be_lora_gs    # FedPartBE-LoRA-GS    (ASSC + LoRA, Nikiema & Amhoud, UM6P 2026)
from algorithms import ccsEF                 # CCS-EF               (Nikiema et al., UM6P 2026)
from algorithms import heterofl               # HeteroFL             (Diao et al., ICLR 2021)
from algorithms import fjord                  # FjORD                (Samuel et al., NeurIPS 2021)

__all__ = [
    "FLAlgorithm", "ClientState", "AggregateResult",
    "register_algorithm", "get_algorithm", "list_algorithms",
]
