#!/usr/bin/env python3
"""Exercise the V30 endpoint auditor on two already-audited V29 clean controls.

In-memory schema adaptation only. No V30 result is created, no gradient is
published, no attack is injected, no method is selected, and no training runs.
This covers clean round120 replay, not attacked endpoints or whole-run auditing.
"""
import copy
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import audit_full_population_private_risk_attacks_v30 as audit
from scripts import run_full_population_private_risk_attacks_v30 as runner
from privacy.fair_objective import require_mps
from privacy.stable_weighted_rfa import stable_norm

PARENT = ROOT/'results/ldp_gradient_far/full_population_private_risk_confirmation_v29'
DEST = ROOT/'output/analysis/V30_Endpoint_Audit_V29_Control_Smoke'


def main():
    require_mps()
    manifest = json.loads((PARENT/'manifest.json').read_text())
    stamp = dict(manifest['source_stamp']); base.verify_stamp(stamp)
    inputs = [Path(__file__), Path(audit.__file__), Path(runner.__file__),
        ROOT/'tests/test_full_population_private_risk_attacks_v30_audit.py']
    stamp.update({str(p.relative_to(ROOT)): base.digest(p) for p in inputs})
    key = json.loads((PARENT/'simulator_secret.json').read_text())['key']
    data = base.prepare(manifest['profile'], 180701)
    records = []
    for method in ('erm_mean', 'erm_rfa'):
        folder = PARENT/f'seed180701__b4800__{method}'
        status = json.loads((folder/'orchestration_status.json').read_text())
        assert status['status'] == 'completed' and status['device'] == 'mps'
        files = {n: base.digest(folder/n) for n in ('metrics.json', 'checkpoint.pt', 'simulator_oracle.json')}
        assert files['metrics.json'] == status['metrics_sha256']
        assert files['checkpoint.pt'] == status['checkpoint_sha256']
        assert files['simulator_oracle.json'] == status['oracle_sha256']
        source = json.loads((folder/'metrics.json').read_text())
        original = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        raw = json.loads((folder/'simulator_oracle.json').read_text())
        assert original['round'] == 120 and source['source_stamp'] == manifest['source_stamp']
        # All conversions below are derived exactly from already-stored tensors.
        cp = dict(original, last_generated_messages=original['last_private_messages'],
            last_sent_messages=original['last_private_messages'],
            last_generated_reports=original['last_reports'], last_sent_reports=original['last_reports'])
        record = copy.deepcopy(source); row = record['rounds'][-1]
        X = original['last_private_messages'].to('mps'); step = original['last_step'].to('mps')
        queries = original['last_clean_means'].to('mps'); A = step/.5; target = queries[2:].mean(0)
        agg = row['aggregation']; effective = agg['objective_weights'] if agg['solver'] is None else agg['solver']['stationary_weights']
        nu = torch.tensor(effective, device='mps', dtype=torch.float32)
        row.update(pre_model_sha256={k: base.ids_hash(v.to('mps')) for k, v in original['pre_round_model'].items()},
            model_sha256={k: base.ids_hash(v.to('mps')) for k, v in original['model'].items()},
            private_messages_sha256=base.ids_hash(X), private_reports_sha256=None,
            step_sha256=base.ids_hash(step), test_honest=runner.honest_metrics(row['test']),
            validation_honest=runner.honest_metrics(row['validation']))
        raw['rounds'][-1]['aggregate'] = dict(
            squared_error_to_unnoised_clipped_honest_mean=float(stable_norm(A-target).square()),
            honest_target_norm=float(stable_norm(target)), aggregate_norm=float(stable_norm(A)),
            designated_objective_mass=sum(agg['objective_weights'][:2]),
            designated_effective_mass=sum(effective[:2]), active_byzantine_contribution_norm=0.,
            reconstruction_error=float(stable_norm(A-(nu[:, None]*X).sum(0))))
        j = dict(seed=180701, method=method, attack='none')
        replay = audit.replay_endpoint(j, cp, record, raw, data, manifest['profile'], key)
        records.append(dict(source_job=source['job'], source_files=files, replay=replay))
    base.verify_stamp(stamp)
    payload = dict(device='mps', tests_passed=True, real_clean_control_endpoints=2,
        source_stamp=stamp, records=records, in_memory_schema_adapter=True,
        v30_campaign_created=False, training_launched=False, attacks_tested=False,
        round90_tested=False, risk_report_replay_tested=False, whole_run_audit_tested=False,
        empirical_candidate_validation=False)
    base.save(DEST.with_suffix('.json'), payload)
    print(json.dumps(dict(tests_passed=True, device='mps', real_clean_control_endpoints=2,
        replays=[dict(method=r['source_job']['method'], **r['replay']) for r in records],
        training_launched=False, attacks_tested=False, report=str(DEST.with_suffix('.json'))),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
