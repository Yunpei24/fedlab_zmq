#!/usr/bin/env python3
"""Standalone constructed-message audit, not a training/DP performance run."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from privacy.fair_objective import require_mps
from privacy.split_risk_gradient import weighted_rfa
from scripts.run_fair_objective_screen import save


def main():
    require_mps();torch.mps.manual_seed(171015)
    honest=.01*torch.randn(8,16,device='mps');target=honest.mean(0)
    lam=torch.tensor([1.]*8+[3.]*2,device='mps');lam/=lam.sum()
    rows=[]
    for scale in (1.,1e3,1e8,1e18,1e30):
        vectors=torch.cat([honest,torch.full((2,16),scale,device='mps')])
        center,diag=weighted_rfa(vectors,lam)
        finite=bool(torch.isfinite(center).all())
        error=float(torch.linalg.vector_norm(center-target)) if finite else None
        scalar_finite=all(torch.isfinite(torch.tensor(v,device='mps')).item() for k,v in diag.items() if isinstance(v,float))
        rows.append(dict(byzantine_coordinate=scale,byzantine_mass=float(lam[-2:].sum()),
            output_finite=finite,certificate_finite=scalar_finite,error_to_honest_mean=error if error is None or error<float('inf') else None,
            certificate=diag if scalar_finite else None))
    payload=dict(device='mps',constructed_messages_only=True,training=False,
        source_sha256=__import__('hashlib').sha256((ROOT/'privacy/split_risk_gradient.py').read_bytes()).hexdigest(),
        rows=rows,warning='Fixed 40 iterations and weighted-mean initialization are not exact RFA under arbitrary scales.')
    destination=ROOT/'output/analysis/Weighted_RFA_Numerical_Stress_Audit.json'
    save(destination,payload);print(json.dumps(payload,indent=2,allow_nan=False))


if __name__=='__main__':main()
