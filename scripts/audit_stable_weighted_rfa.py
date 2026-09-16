#!/usr/bin/env python3
"""Paired numerical audit only; no dataset, training, or utility claim."""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import weighted_rfa, stable_norm
from scripts.run_fair_objective_screen import save, digest


def main():
    require_mps()
    torch.mps.manual_seed(171015)
    honest = .01 * torch.randn(8, 16, device='mps')
    lam = torch.tensor([1.] * 8 + [3.] * 2, device='mps')
    rows = []
    for scale in (1., 1e3, 1e8, 1e18, 1e30):
        x = torch.cat([honest, torch.full((2, 16), scale, device='mps')])
        center, diagnostics = weighted_rfa(x, lam)
        rows.append(dict(byzantine_coordinate=scale,
                         error_to_honest_mean=float(stable_norm(center-honest.mean(0))),
                         diagnostics=diagnostics))
    payload = dict(device='mps', constructed_messages_only=True, training=False,
                   rows=rows, source_sha256=digest(ROOT/'privacy/stable_weighted_rfa.py'),
                   no_fairness_privacy_or_model_performance_validation=True)
    save(ROOT/'output/analysis/Stable_Weighted_RFA_Numerical_Audit.json', payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
