#!/usr/bin/env python3
"""Exercise V30's risk-report endpoint audit on two already-audited V29 runs.

No V30 campaign, training, tuning, or attack. The existing two-ERM smoke report
is preserved. All schema adaptation is in memory; the source runs are read-only.
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
VERIFIED = ROOT/'output/analysis/Full_Population_Private_Risk_Confirmation_V29_Analyse_Partial.json'
DEST = ROOT/'output/analysis/V30_Risk_Endpoint_Audit_V29_Smoke.json'


def main():
    require_mps()
    manifest = json.loads((PARENT/'manifest.json').read_text())
    assert manifest['torch_version'] == str(torch.__version__)
    stamp = dict(manifest['source_stamp']); base.verify_stamp(stamp)
    proof_hash = base.digest(VERIFIED)
    proof = json.loads(VERIFIED.read_text())
    assert proof['audit_passed']
    # The partial audit evolves. Preserve its relevant evidence inside this
    # report rather than adding its mutable path to the immutable source stamp.
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in
        (Path(__file__), Path(audit.__file__), Path(runner.__file__),
         ROOT/'tests/test_full_population_private_risk_attacks_v30_audit.py')})
    key = json.loads((PARENT/'simulator_secret.json').read_text())['key']
    data = base.prepare(manifest['profile'], 180701)
    records, prior_records = [], []
    for method in ('risk_mean', 'risk_rfa'):
        folder = PARENT/f'seed180701__b4800__{method}'
        status = json.loads((folder/'orchestration_status.json').read_text())
        assert status['status'] == 'completed' and status['device'] == 'mps'
        checked = next(r for r in proof['records'] if r['job'] == dict(seed=180701,batch=4800,method=method))
        prior_records.append(checked)
        files = {n:base.digest(folder/n) for n in checked['files']}
        assert files == checked['files']
        assert files['metrics.json'] == status['metrics_sha256']
        assert files['checkpoint.pt'] == status['checkpoint_sha256']
        assert files['simulator_oracle.json'] == status['oracle_sha256']
        source = json.loads((folder/'metrics.json').read_text())
        original = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        raw = json.loads((folder/'simulator_oracle.json').read_text())
        assert original['round'] == 120 and source['source_stamp'] == manifest['source_stamp']
        assert original['last_reports'] is not None
        cp = dict(original, last_generated_messages=original['last_private_messages'],
            last_sent_messages=original['last_private_messages'],
            last_generated_reports=original['last_reports'], last_sent_reports=original['last_reports'])
        record = copy.deepcopy(source); row = record['rounds'][-1]
        X = original['last_private_messages'].to('mps')
        reports = original['last_reports'].to('mps')
        step = original['last_step'].to('mps')
        queries = original['last_clean_means'].to('mps'); A = step/.5
        target = queries[2:].mean(0)
        agg = row['aggregation']
        effective = agg['objective_weights'] if agg['solver'] is None else agg['solver']['stationary_weights']
        nu = torch.tensor(effective, device='mps', dtype=torch.float32)
        row.update(pre_model_sha256={k:base.ids_hash(v.to('mps')) for k,v in original['pre_round_model'].items()},
            model_sha256={k:base.ids_hash(v.to('mps')) for k,v in original['model'].items()},
            private_messages_sha256=base.ids_hash(X), private_reports_sha256=base.ids_hash(reports),
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
    assert base.digest(VERIFIED) == proof_hash, 'Partial parent audit changed during reconstruction'
    base.save(DEST, dict(device='mps', tests_passed=True, real_clean_risk_endpoints=2,
        source_stamp=stamp, records=records, in_memory_schema_adapter=True,
        parent_audit_evidence=dict(source_file=str(VERIFIED.relative_to(ROOT)),
            snapshot_sha256=proof_hash, audit_passed=proof['audit_passed'],
            valid_runs_at_read=proof['valid_runs'], selected_records=prior_records,
            source_stamp=proof['source_stamp']),
        v30_campaign_created=False, training_launched=False, attacks_tested=False,
        round90_tested=False, risk_report_replay_tested=True, whole_run_audit_tested=False,
        empirical_candidate_validation=False))
    print(json.dumps(dict(tests_passed=True, device='mps', real_clean_risk_endpoints=2,
        replays=[dict(method=r['source_job']['method'], **r['replay']) for r in records],
        training_launched=False, attacks_tested=False, report=str(DEST)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
