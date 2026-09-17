#!/usr/bin/env python3
"""One-state real-data numerical probe; NOT a training/privacy validation run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')

import torch
from torchvision.datasets import FashionMNIST
from models.registry import get_model
from privacy.far_dp_effect_mc import gradient_release, sampled_indices, far_aggregate, require_mps


def main():
    require_mps()
    data = FashionMNIST(ROOT/'data', train=True, download=False)
    # Public probe RNG: no privacy claim for these vectors or this diagnostic.
    torch.manual_seed(2026091601)
    torch.mps.manual_seed(2026091601)
    model = get_model('lenet5_tanh', 'fashionmnist').to('mps').eval()
    ids = sampled_indices(len(data), 1200, 2026091602)
    x = (data.data.to('mps').float()[ids].unsqueeze(1)/255-.286)/.353
    y = data.targets.to('mps')[ids]
    original = [p.detach().clone() for p in model.parameters()]
    records = []
    for b, c in [(300, None), (300, 8.), (1200, 8.)]:
        torch.mps.synchronize(); started = time.perf_counter()
        result, diagnostics = gradient_release(model, x[:b], y[:b], local_clip=c,
                noise_std=0, noise_seed=100, microbatch=32)
        torch.mps.synchronize(); elapsed = time.perf_counter()-started
        row = dict(batch=b, clip=c, seconds=elapsed, gradient_norm=float(result.norm()),
                   local_clip_fraction=diagnostics['clip_fraction_oracle'])
        if c is None:
            expected = torch.autograd.grad(torch.nn.functional.cross_entropy(model(x[:b]), y[:b]),
                                           tuple(model.parameters()))
            expected = torch.cat([g.flatten() for g in expected])
            row['max_abs_error_vs_batch_autograd'] = float((result-expected).abs().max())
            assert torch.allclose(result, expected, atol=3e-6, rtol=5e-4)
        records.append(row)
    # Ten distinct small real batches, one frozen model: aggregation contracts.
    messages = []
    for client in range(10):
        subset = slice(12*client, 12*(client+1))
        message, _ = gradient_release(model, x[subset], y[subset], local_clip=8.,
                                     noise_std=0, noise_seed=client)
        messages.append(message)
    messages = torch.stack(messages)
    refs = []
    for reference in ['rfa', 'coordinate_median', 'trimmed_mean', 'centered_clipping']:
        a, diag = far_aggregate(messages, alpha=0., reference=reference, server_clip=None,
                                reference_radius=8.)
        error = float((a-messages.mean(0)).abs().max())
        assert torch.allclose(a, messages.mean(0), atol=3e-6, rtol=5e-4)
        refs.append(dict(reference=reference, max_abs_error_alpha0_vs_mean=error))
    assert all(torch.equal(p, old) for p, old in zip(model.parameters(), original))
    report = dict(device='mps', dataset='fashionmnist', model='lenet5_tanh',
                  examples_in_probe=1200, parameters=sum(p.numel() for p in model.parameters()),
                  model_optimizer_steps=0, training_runs=0, privacy_claim=False,
                  calibration_decision=False, timings_are_not_full_training_estimates=True,
                  gradient_checks=records, reference_checks=refs,
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                       ['privacy/far_dp_effect_mc.py', 'scripts/probe_far_dp_effect_mc.py',
                        'privacy/local_dpsgd.py', 'robustness/aggregators.py', 'models/registry.py']})
    output = ROOT/'output/analysis/FAR_DP_Effect_MPS_Probe_20260917.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
