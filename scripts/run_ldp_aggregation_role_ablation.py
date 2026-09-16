#!/usr/bin/env python3
"""Resume-only 96-run MPS screen: uniform/direct RFA/FAR references."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import statistics as st
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import yaml
from scripts import run_rcig_n10_policy_ablation as legacy
from scripts import run_rcig_batch_screen as shared
from scripts.run_rcig_ldp_gradient_far_v2 import _metrics_path,_post_run_device_audit

MATRIX=ROOT/'configs/ldp_gradient_far/aggregation_role_n10_v1.yaml'
OUTPUT=ROOT/'results/ldp_gradient_far/aggregation_role_n10_v1'
ENTRY=ROOT/'scripts/run_ldp_aggregation_role_experiment.py'
LOG=ROOT/'logs/aggregation_role_n10_v1.log'
ARMS=('uniform','rfa_direct','far_rfa','far_midpoint')
hash_file=shared.file_hash
write_json=shared.write_json


def matrix():
    m=yaml.safe_load(MATRIX.read_text())
    assert m['arms']==list(ARMS) and m['rounds']==40 and m['num_clients']==10
    assert m['comparison_seeds']==[920001,920002,920003]
    assert m['warmup_uniform_rounds']==12 and not m['new_threshold_calibration']
    return m


def tasks(m):
    ts=[dict(noise=noise,seed=seed,scenario=scenario,arm=arm)
        for noise,seed,scenario,arm in itertools.product(m['noise_regimes'],m['comparison_seeds'],m['scenarios'],m['arms'])]
    assert len(ts)==96
    return ts


def run_id(task):
    return f"{task['noise']}__seed{task['seed']}__{task['scenario']}__{task['arm']}"


def directory(task):
    return OUTPUT/run_id(task)


def provenance(m):
    return dict(campaign=m['campaign_id'],matrix_hash=hash_file(MATRIX),
                protocol_hash=hash_file(ROOT/m['protocol']),
                base_hash=hash_file(MATRIX.parent/m['base_config']),
                shadow_threshold_hash=hash_file(legacy.ARTIFACT),
                privacy=legacy.privacy(m),expected_runs=96,
                sources=shared.source_closure((Path(__file__),ENTRY,ROOT/'run_experiment.py')),
                private_device='mps',fallback=0)


def verify(stamp,m):
    for name,h in stamp['sources'].items():
        if hash_file(ROOT/name)!=h:raise RuntimeError('source drift: '+name)
    for p,key in [(MATRIX,'matrix_hash'),(ROOT/m['protocol'],'protocol_hash'),
                  (MATRIX.parent/m['base_config'],'base_hash'),(legacy.ARTIFACT,'shadow_threshold_hash')]:
        if hash_file(p)!=stamp[key]:raise RuntimeError('configuration/protocol drift: '+str(p))


def config_for(m,task,stamp):
    # Reuse construction in memory only, then replace all identity/output fields.
    thresholds=json.loads(legacy.ARTIFACT.read_text())['thresholds'][task['noise']]
    cfg=legacy.config_for(m,dict(task,phase='comparison',arm='midpoint'),stamp,thresholds)
    cfg['output_dir']=str(directory(task))
    a=cfg['training']['algo_config']
    a.update(aggregation_role_arm=task['arm'],aggregation_role_campaign=m['campaign_id'],
             aggregation_role_actual_rfa_preprocessing='none',
             aggregation_role_all_candidates_offline_audited=True,
             rcig_campaign_id=m['campaign_id'],
             rcig_n10_runtime_implementation='algorithms.ldp_aggregation_role_ablation.LDPAggregationRoleAblation')
    return cfg


def load_trace(task):
    p=directory(task)/'simulator_randomness_private_audit.jsonl'
    return sorted((json.loads(line) for line in p.read_text().splitlines()),key=lambda r:(r['round'],r['client_id']))


def check_pairing(task,rows):
    trace=load_trace(task)
    assert len(trace)==400 and {(r['round'],r['client_id']) for r in trace}==set(itertools.product(range(1,41),range(10)))
    canonical=dict(task,arm='uniform',scenario='none')
    result=dict(actual_draws=400,same_arm_clean_prefix=None)
    if task!=canonical:
        for r,s in zip(trace,load_trace(canonical)):
            assert r['permutations']==s['permutations'] and r['standard_gaussians']==s['standard_gaussians'],'unpaired draws'
            if r['round']<=12:
                assert r['model_before']==s['model_before'] and r['private_upload']==s['private_upload'],'warmup mismatch'
        result['warmup_identical']=True
    if task['scenario']!='none':
        clean=dict(task,scenario='none')
        for r,s in zip(trace,load_trace(clean)):
            if r['round']<=17:
                assert r['model_before']==s['model_before'] and r['private_upload']==s['private_upload'],'pre-attack mismatch'
        clean_rows=json.loads(_metrics_path(directory(clean)).read_text())['rounds']
        for r,s in zip(rows[:16],clean_rows[:16]):
            for key in ['test_accuracy','test_loss']:assert abs(r[key]-s[key])<1e-12
        result['same_arm_clean_prefix']=True
    return result


def validate(task,cfg,stamp):
    p=_metrics_path(directory(task))
    if p is None:raise RuntimeError('missing/incomplete/nonunique metrics: '+run_id(task))
    x=json.loads(p.read_text());rows=x['rounds'];summary=x['summary']
    assert len(rows)==40
    assert (summary['num_clients'],summary['num_rounds'],summary['seed'],summary['model'],summary['dataset'])==(10,40,task['seed'],'lenet5_tanh','fashionmnist')
    for key,val in cfg['training']['algo_config'].items():assert x['config'].get(key)==val,(key,p)
    _post_run_device_audit(p,cfg)
    for t,r in enumerate(rows,1):
        assert r['round_num']==t and r['num_alive_clients']==10
        assert r['aggregation_role_arm']==task['arm']
        assert r['aggregation_role_effective_arm']==('uniform' if t<=12 else task['arm'])
        assert r['aggregation_role_only_declared_rule_applied']
        assert r['aggregation_role_same_cohort_verified'] and not r['aggregation_role_oracles_visible_to_server']
        assert r['aggregation_role_rfa_preprocessing']=='none_no_bucketing'
        alpha=.1 if t>12 and task['arm'].startswith('far_') else 0
        assert abs(r['far_alpha']-alpha)<1e-12
        for key in ['test_accuracy','test_loss','client_accuracy_mean','client_accuracy_variance_pct2','worst20_accuracy_pct','best20_worst20_gap_pct','deployed_aggregate_error_sq']:
            assert isinstance(r[key],(float,int)) and math.isfinite(r[key]),(key,p)
        for arm in ARMS:
            pre='cohort_'+arm+'_'
            assert r[pre+'decomposition_residual_sq']<=1e-18*max(1,r[pre+'aggregate_error_sq'])
            decomposition=sum(r[pre+k] for k in ['honest_effective_noise_sq','honest_tilt_sq','byzantine_centered_sq','cross_noise_tilt','cross_noise_byzantine','cross_tilt_byzantine'])
            assert abs(decomposition-r[pre+'aggregate_error_sq'])<=1e-9*max(1,r[pre+'aggregate_error_sq'])
        if t>12:assert r['shadow_rcig_newer_round_max']<t-1
    assert abs(rows[-1]['privacy_epsilon_max']-4)<1e-4
    runtime=json.loads((directory(task)/'runtime_imports.json').read_text())
    assert runtime['stage']=='after_training' and runtime['mps_fallback']==0
    assert all(stamp['sources'].get(name)==h for name,h in runtime['source_sha256'].items()),'unlocked runtime source'
    return p,x,check_pairing(task,rows)


def summarize(completed):
    groups={}
    for task,x in completed:
        key='/'.join(task[k] for k in ['noise','scenario','arm'])
        last=x['rounds'][-1]
        entry=dict(seed=task['seed'])
        for name in ['test_accuracy','test_loss','client_accuracy_mean','client_accuracy_variance_pct2','worst20_accuracy_pct','best20_worst20_gap_pct','mean_client_balanced_accuracy_pct']:
            entry[name]=last[name]
        ready=x['rounds'][12:] if task['scenario']=='none' else x['rounds'][16:]
        for name in ['deployed_aggregate_error_sq','deployed_byzantine_centered_sq','deployed_byzantine_mass']:
            entry[name]=st.mean(r[name] for r in ready)
        groups.setdefault(key,[]).append(entry)
    write_json(OUTPUT/'summary_progress.json',dict(completed=len(completed),total=96,groups=groups,scientific_status='exploratory_no_promotion'))
    lines=['# Agrégation uniforme / RFA directe / FAR : résultats progressifs','',
           f'{len(completed)}/96 runs terminés. Données partielles : aucun gagnant déclaré. Les valeurs ± sont des écarts-types entre seeds, pas des IC95.','',
           '| Bruit / scénario / méthode | Seeds | Test Acc. (%) | Test loss | Worst-20 (%) | Gap (pp) | Erreur agrégat | Masse byzantine |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,values in groups.items():
        fields=[]
        for metric,scale in [('test_accuracy',100),('test_loss',1),('worst20_accuracy_pct',1),('best20_worst20_gap_pct',1),('deployed_aggregate_error_sq',1),('deployed_byzantine_mass',1)]:
            nums=[v[metric]*scale for v in values];sd=f'{st.stdev(nums):.4f}' if len(nums)>1 else '—'
            fields.append(f'{st.mean(nums):.4f} ± {sd}')
        lines.append(f'| {key} | {len(values)} | '+' | '.join(fields)+' |')
    lines+=['','Le warmup 1–12 est uniforme pour les quatre méthodes, y compris RFA directe. Erreurs d’agrégat et masses : moyenne des tours 17–40 (13–40 sans attaque), puis entre seeds. Les diagnostics par candidat sur le même transcript restent dans metrics.json.','']
    (OUTPUT/'Results_Progress.md').write_text('\n'.join(lines))


@contextmanager
def execution_lock():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT.parent/'.aggregation_role_n10_v1.execution.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def no_conflicting_training():
    shared.check_no_other_training_process()
    lines=subprocess.run(['/bin/ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True).stdout.splitlines()
    guarded={'run_rcig_n10_experiment.py',ENTRY.name,'run_ldp_aggregation_role_ablation.py'}
    for line in lines:
        parts=line.strip().split(None,2)
        if len(parts)!=3 or int(parts[0])==os.getpid():continue
        try:args=shlex.split(parts[2])
        except ValueError:continue
        if any(Path(a).name in guarded for a in args) and '--launch' not in args:
            raise RuntimeError('another active training/supervisor: '+line)


def status(m):
    result=dict(completed=0,total=96,active=[],failed=[],missing=0)
    for task in tasks(m):
        p=directory(task)/'orchestration_status.json'
        if not p.exists():result['missing']+=1;continue
        s=json.loads(p.read_text())
        if s['status']=='completed':result['completed']+=1
        else:result['active' if s['status']=='running' else 'failed'].append(dict(run=run_id(task),**s))
    return result


def execute(m,stamp,max_new=None):
    completed=[];new=0
    for task in tasks(m):
        verify(stamp,m);dest=directory(task);sp=dest/'orchestration_status.json'
        cfg=config_for(m,task,stamp)
        if sp.exists():
            s=json.loads(sp.read_text())
            if s['status']!='completed':raise RuntimeError('incomplete run requires inspection: '+str(dest))
            assert yaml.safe_load((dest/'resolved_config.yaml').read_text())==cfg
            p,x,pair=validate(task,cfg,stamp)
            assert hash_file(p)==s['metrics_sha256'] and hash_file(dest/'simulator_randomness_private_audit.jsonl')==s['trace_sha256']
        else:
            if max_new is not None and new>=max_new:break
            if dest.exists() and any(dest.iterdir()):raise RuntimeError('unlocked partial output: '+str(dest))
            dest.mkdir(parents=True,exist_ok=True)
            with (dest/'resolved_config.yaml').open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
            base=dict(task=task,config_sha256=hash_file(dest/'resolved_config.yaml'),device='mps',fallback=0,started_at=datetime.now(timezone.utc).isoformat())
            write_json(sp,dict(base,status='running'))
            print(f'START {len(completed)+1}/96 {run_id(task)}',flush=True)
            try:
                with (dest/'training.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B','-u',str(ENTRY),'--config',str(dest/'resolved_config.yaml'),'--output',str(dest),'--device','mps'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONDONTWRITEBYTECODE='1'))
                    write_json(sp,dict(base,status='running',pid=child.pid));code=child.wait()
                if code:raise RuntimeError(f'training exit={code}: {dest / "training.log"}')
                verify(stamp,m);p,x,pair=validate(task,cfg,stamp)
                write_json(sp,dict(base,status='completed',finished_at=datetime.now(timezone.utc).isoformat(),metrics_sha256=hash_file(p),trace_sha256=hash_file(dest/'simulator_randomness_private_audit.jsonl'),pairing=pair))
                new+=1
                print(f'DONE {len(completed)+1}/96 test_acc={x["rounds"][-1]["test_accuracy"]:.6f}',flush=True)
            except BaseException as exc:
                write_json(sp,dict(base,status='failed',error=str(exc)));raise
        completed.append((task,x));summarize(completed)
        write_json(OUTPUT/'progress.json',dict(completed=len(completed),total=96,status='completed' if len(completed)==96 else 'active',last_completed=task))


def main():
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group(required=True)
    for name in ['plan','status','launch','run']:actions.add_argument('--'+name,action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-new',type=int)
    args=p.parse_args();m=matrix()
    if args.status:print(json.dumps(status(m),indent=2));return
    stamp=provenance(m)
    if args.plan:print(json.dumps(dict(total=96,arms=ARMS,privacy=stamp['privacy'],sources=len(stamp['sources']),output=str(OUTPUT)),indent=2));return
    if not args.resume:raise ValueError('--resume required')
    shared.require_working_mps()
    with execution_lock():
        no_conflicting_training()
        if args.launch:
            LOG.parent.mkdir(parents=True,exist_ok=True)
            with LOG.open('a') as log:
                cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run','--resume']
                if args.max_new is not None:cmd+=['--max-new',str(args.max_new)]
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK='0',PYTHONDONTWRITEBYTECODE='1'))
            print(json.dumps(dict(pid=child.pid,log=str(LOG))));return
        lock=OUTPUT/'campaign_lock.json'
        if lock.exists():assert json.loads(lock.read_text())==stamp,'campaign lock mismatch'
        else:
            if OUTPUT.exists() and any(OUTPUT.iterdir()):raise RuntimeError('unlocked existing campaign')
            OUTPUT.mkdir(parents=True,exist_ok=True);write_json(lock,stamp,exclusive=True)
        try:execute(m,stamp,max_new=args.max_new)
        except BaseException as exc:
            write_json(OUTPUT/'failure.json',dict(error=str(exc),at=datetime.now(timezone.utc).isoformat()));raise


if __name__=='__main__':main()
