#!/usr/bin/env python3
"""Read-only replay of saved real loss means; Gaussian diagnostic draws on MPS.

No model training, no change to historical runs, no oracle enters a selector.
The audit itself exports nonprivate diagnostics and uses public simulation RNG.
It is NOT a new DP release to deploy or compose as if it cost epsilon zero.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import itertools
import json
import math
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
import torch
import yaml
from algorithms.private_risk_selection import risk, select_private_risk, forge_reports
from algorithms.private_paired_risk import paired_sensitivity, certify_paired_reports, choose_under_bound
from privacy.rdp import calibrate_gaussian_noise, RDPAccountant
from scripts.run_private_risk_feasibility import require_mps, reseed, file_hash, save, source_closure

MATRIX=ROOT/'configs/ldp_gradient_far/private_paired_risk_audit_v1.yaml'
OUTPUT=ROOT/'results/ldp_gradient_far/private_paired_risk_audit_v1'


def paired_attack(reports,mode):
    out=[row[:] for row in reports]
    if mode=='veto':
        out[0]=[1.,1.,1.]; out[1]=[1.,1.,1.]
    elif mode=='endorse_far':
        out[0]=[0.,0.,-1.]; out[1]=[0.,0.,-1.]
    elif mode!='truthful': raise ValueError(mode)
    return out


def input_records(c):
    base=ROOT/c['source_campaign']
    paths=sorted(base.glob('*/metrics.json'))
    assert len(paths)==c['expected_source_runs']
    result=[]; hashes={}
    for path in paths:
        run=json.loads(path.read_text()); state=json.loads((path.parent/'orchestration_status.json').read_text())
        assert state['status']=='completed' and file_hash(path)==state['metrics_sha256']
        assert len(run['rounds'])==40 and all(r['private_gradient_device']=='mps' for r in run['rounds'])
        assert run['manifest']['device']=='mps' and run['manifest']['fallback']==0
        hashes[str(path.relative_to(ROOT))]=file_hash(path)
        for relative,h in run['artifact_sha256'].items():
            assert file_hash(path.parent/relative)==h
            hashes[str((path.parent/relative).relative_to(ROOT))]=h
        for probe in run['probes']:
            oracle_path=path.parent/'simulator_oracle'/f"round{probe['round']:02d}_{probe['scenario']}.json"
            oracle=json.loads(oracle_path.read_text())
            # mean(loss_k-loss_0) == mean(loss_k)-mean(loss_0), EXACTLY.
            # This identity would not permit reconstructing per-example clipping.
            levels=[[oracle['validation'][k][i]['brier'] for k in range(4)] for i in range(10)]
            result.append(dict(noise=run['job']['noise'],seed=run['job']['seed'],round=probe['round'],
                scenario=probe['scenario'],levels=levels,
                deltas=[[v-row[0] for v in row[1:]] for row in levels],
                factors=[1.]*10 if run['job']['noise']=='homogeneous' else [1.,2.]*5))
    assert len(result)==c['expected_source_probes']
    return result,hashes


def run():
    require_mps(); c=yaml.safe_load(MATRIX.read_text())
    assert c['release_unclipped_contrasts_only'] and c['automatic_training'] is False
    records,hashes=input_records(c)
    sigma=calibrate_gaussian_noise(target_epsilon=c['component_epsilon_target'],
                                  delta=c['component_delta'],steps=c['max_releases_per_hypothetical_run'])
    old_std=sigma*2/c['validation_size']
    new_std=sigma*paired_sensitivity(c['validation_size'],3)
    old_width=old_std*math.sqrt(2*math.log(2*10*4*9/c['confidence_failure_probability']))
    new_width=new_std*math.sqrt(2*math.log(2*10*4*9/c['confidence_failure_probability']))
    ledger=RDPAccountant(); ledger.add_gaussian(channel='paired_validation',noise_multiplier=sigma,steps=9)
    calibration=dict(gaussian_multiplier=sigma,old_coordinate_std=old_std,paired_coordinate_std=new_std,
        old_difference_interval_halfwidth=2*old_width,paired_interval_halfwidth=new_width,
        interval_ratio=new_width/(2*old_width),sensitivity=paired_sensitivity(c['validation_size'],3),
        alternative_evaluation_epsilon=ledger.epsilon(c['component_delta'])[0],
        alternative_component_delta=c['component_delta'],releases_per_alternative_run=9,
        hypothetical_replacement_only=True,not_additional_free_queries=True)
    summary=defaultdict(lambda:dict(n=0,accept=0,harm=0,chosen_change_sum=0.,oracle_progress_sum=0.,
                                  coverage_count=0,required_tolerances=[]))
    noiseless=[]; observed_probes=[]
    count=c['diagnostic_noise_replicates']
    for index,r in enumerate(records):
        reseed(c['diagnostic_rng_seed']+index)
        # All simulation Gaussians on MPS. Same z for old/new COMPARISON only;
        # these would be mutually exclusive protocols, not joint private releases.
        z=torch.randn(count,10,4,device='mps')
        levels=torch.tensor(r['levels'],device='mps')
        deltas=torch.tensor(r['deltas'],device='mps')
        factors=torch.tensor(r['factors'],device='mps')
        old=(levels[None,:,:]+old_std*factors[None,:,None]*z).cpu().tolist()
        paired=(deltas[None,:,:]+new_std*factors[None,:,None]*z[:,:,:3]).cpu().tolist()
        for mode in c['report_modes']:
            h=list(range(10)) if r['scenario']=='none' and mode=='truthful' else list(range(2,10))
            true_risks=[risk([r['levels'][i][k] for i in h]) for k in range(4)]
            changes=[v-true_risks[0] for v in true_risks]
            progress=max(0.,-min(changes))
            contrast_risks=[risk([r['deltas'][i][k] for i in h]) for k in range(3)]
            assert all(changes[k+1]<=contrast_risks[k]+1e-12 for k in range(3))
            if mode=='truthful':
                observed_probes.append(dict(**{key:r[key] for key in ['noise','seed','round','scenario']},
                    true_risk_changes=changes,oracle_progress=progress,
                    risk_of_contrasts=contrast_risks,
                    subadditivity_slack=[contrast_risks[k]-changes[k+1] for k in range(3)]))
            for b in [0,2]:
                kw=dict(max_releases=9,failure_probability=c['confidence_failure_probability'],byzantine_bound=b)
                a=select_private_risk(forge_reports(r['levels'],identities=[0,1],mode=mode),noise_stds=[0.]*10,**kw)
                d=certify_paired_reports(paired_attack(r['deltas'],mode),noise_stds=[0.]*10,**kw)
                noiseless.append(dict(**{key:r[key] for key in ['noise','seed','round','scenario']},mode=mode,b=b,
                    assumption_valid=(b==2 or (mode=='truthful' and r['scenario']=='none')),
                    levels_selected=a['selected'],paired_selected=d['selected'],
                    paired_upper_bounds=d['upper_bounds'],paired_selected_change=changes[d['selected']]))
            for j in range(count):
                for b in [0,2]:
                    kw=dict(max_releases=9,failure_probability=c['confidence_failure_probability'],byzantine_bound=b)
                    a=select_private_risk(forge_reports(old[j],identities=[0,1],mode=mode),
                        noise_stds=[old_std*f for f in r['factors']],**kw)
                    d=certify_paired_reports(paired_attack(paired[j],mode),
                        noise_stds=[new_std*f for f in r['factors']],**kw)
                    valid=(b==2 or (mode=='truthful' and r['scenario']=='none'))
                    coverage=all(d['lower'][i][k]-1e-6<=r['deltas'][i][k]<=d['upper'][i][k]+1e-6 for i in h for k in range(3))
                    if valid and coverage:
                        assert all(changes[k+1]<=d['upper_bounds'][k]+1e-6 for k in range(3))
                    policies=[('levels_descent',a['selected'],0.)]
                    policies.extend(('paired',choose_under_bound(d['upper_bounds'],tol),tol) for tol in c['damage_tolerances'])
                    for mechanism,choice,tolerance in policies:
                        key=(r['noise'],r['seed'],r['scenario'],mode,b,valid,mechanism,tolerance)
                        cell=summary[key]; cell['n']+=1; cell['accept']+=int(choice!=0)
                        cell['harm']+=int(choice!=0 and changes[choice]>1e-6)
                        cell['chosen_change_sum']+=changes[choice]; cell['oracle_progress_sum']+=progress
                        cell['coverage_count']+=int(coverage)
                        if mechanism=='paired':
                            cell['required_tolerances'].append(max(0.,min(d['upper_bounds'])))
                            if valid and coverage and choice:
                                assert changes[choice]<=tolerance+1e-6
        print(f"MPS report replay {index+1}/{len(records)}: {r['noise']} seed={r['seed']} round={r['round']} {r['scenario']}",flush=True)
    rows=[]
    for key,v in summary.items():
        width=v.pop('required_tolerances')
        rows.append(dict(zip(['noise','seed','scenario','report_mode','b','certificate_assumption_valid','mechanism','damage_tolerance'],key)),)
        rows[-1].update(v)
        rows[-1].update(acceptance_rate=v['accept']/v['n'],harm_rate=v['harm']/v['n'],
            retained_progress=-v['chosen_change_sum']/v['oracle_progress_sum'] if v['oracle_progress_sum']>1e-12 else None,
            required_tolerance_median=st.median(width) if width else None)
    assert all(file_hash(ROOT/p)==h for p,h in hashes.items()),'historical data changed during audit'
    outcome=dict(completed=True,device='mps',no_training=True,privacy_protected_output=False,
        scope='offline_distributional_audit_on_54_saved_real_probes_not_new_end_to_end_results',
        diagnostic_rng_public=True,replicates_per_probe=count,replicates_are_not_independent_training_seeds=True,
        config_sha256=file_hash(MATRIX),source_closure=source_closure((Path(__file__),)),
        source_artifact_hashes=hashes,calibration=calibration,probes=observed_probes,
        zero_evaluation_noise_diagnostics=noiseless,by_seed=rows,automatic_training=False)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'audit.json').exists():
        raise RuntimeError('audit exists; preserve it, use a new version rather than overwrite')
    save(OUTPUT/'audit.json',outcome,private=True)
    save(OUTPUT/'status.json',dict(status='completed',probes=54,model_training_runs=0,device='mps',
                                  audit_sha256=file_hash(OUTPUT/'audit.json')))
    print(json.dumps(calibration,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); args=parser.parse_args()
    if not args.run: parser.error('requires --run')
    run()
