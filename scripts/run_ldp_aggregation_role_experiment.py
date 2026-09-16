#!/usr/bin/env python3
"""Isolated MPS-only entrypoint and offline same-cohort oracle boundary."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',required=True,choices=['mps'])
    args=parser.parse_args()
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')!='0':
        raise RuntimeError('CPU fallback forbidden')
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
    import torch
    import yaml
    from algorithms.base import register_algorithm
    from algorithms.ldp_aggregation_role_ablation import LDPAggregationRoleAblation,ARMS
    from metrics.aggregation_role_evaluation import evaluate_aggregation_roles
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    from scripts.run_rcig_batch_screen_experiment import write_runtime_manifest
    import run_experiment as harness
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable: no training performed')
    cfg=yaml.safe_load(args.config.read_text());a=cfg['training']['algo_config']
    if cfg['clients']['num_clients']!=10 or a['aggregation_role_arm'] not in ARMS or not a['external_attack_diagnostics'] or not a['enable_oracle_diagnostics']:
        raise RuntimeError('campaign/oracle boundary mismatch')
    register_algorithm('ldp_gradient_far')(LDPAggregationRoleAblation)
    args.output.mkdir(parents=True,exist_ok=True)
    before=write_runtime_manifest(args.output,algorithm='ldp_aggregation_role_ablation',stage='before_training')
    original_eval=harness.rcig_reference_oracle_metrics
    original_argv=sys.argv
    with (args.output/'simulator_randomness_private_audit.jsonl').open('x') as trace:
        def sink(row):
            trace.write(json.dumps(row,sort_keys=True)+'\n');trace.flush()
        LDPAggregationRoleAblation.audit_sink=staticmethod(sink)
        harness.rcig_reference_oracle_metrics=evaluate_aggregation_roles
        sys.argv=[str(ROOT/'run_experiment.py'),'--config',str(args.config),'--output',str(args.output),'--device','mps']
        try:
            with evaluation_rng_isolation(harness):
                harness.main()
        finally:
            harness.rcig_reference_oracle_metrics=original_eval
            LDPAggregationRoleAblation.audit_sink=None
            sys.argv=original_argv
    write_runtime_manifest(args.output,algorithm='ldp_aggregation_role_ablation',stage='after_training',previous=before)


if __name__=='__main__':
    main()
