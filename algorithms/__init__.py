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
from algorithms import fedpart_be             # FedStep            (Battery-Energy-aware FedPart)
from algorithms import fedpart_universal      # FedPartUniversal     (Nikiema & Amhoud, UM6P 2026)
from algorithms import fedpart_be_lora_gs    # FedStep-LoRA-GS    (ASSC + LoRA, Nikiema & Amhoud, UM6P 2026)
from algorithms import ccsEF                 # CCS-EF               (Nikiema et al., UM6P 2026)
from algorithms import heterofl               # HeteroFL             (Diao et al., ICLR 2021)
from algorithms import fjord                  # FjORD                (Samuel et al., NeurIPS 2021)
from algorithms import depthfl                # DepthFL              (Kim et al., ICLR 2023)
from algorithms import scalefl                # ScaleFL              (Ilhan et al., CVPR 2023)
from algorithms import fedle                  # FedLE                (Yan et al., lifespan-extension selection)
from algorithms import fedqgate               # FedQGate             (Nikiema & Amhoud, UM6P 2026)
from algorithms import server_mask_fl         # Server-Mask FL       (Nikiema & Amhoud, UM6P 2026)
from algorithms import fed_od                 # FedOD                (Nikiema, UM6P 2026)
from algorithms import dmd                    # DMD-Mean/USV/Tail    (research adapters)
from algorithms import qffl                   # q-FedAvg / q-FFL     (Li et al., ICLR 2020)
from algorithms import term                   # Client-level TERM    (Li et al., ICLR 2021)
from algorithms import far                    # FAR robust-reference reweighting
from algorithms import fedfdp                 # FedFair / FedFDP
from algorithms import robust_fedavg          # Standalone Byzantine-robust baselines
from algorithms import dp_references          # Shared DP-SGD FedAvg / q-FFL / FAR
from algorithms import sc_partial_far_dp      # Sensitivity-Controlled (Partial) FAR-DP

__all__ = [
    "FLAlgorithm", "ClientState", "AggregateResult",
    "register_algorithm", "get_algorithm", "list_algorithms",
]
