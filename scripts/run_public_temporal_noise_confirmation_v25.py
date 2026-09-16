#!/usr/bin/env python3
"""Four fresh seeds; frozen candidate and constant/scheduled strong controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
import yaml
from scripts import run_fair_objective_screen as base
from scripts import run_public_temporal_noise_training_v24 as prior
from scripts.analyze_private_risk_confirmation_v12 import verify
from privacy.fair_objective import require_mps, per_example, release
from privacy.split_risk_gradient import private_risk
from privacy.scheduled_private_risk import aggregate
from privacy.private_risk_message_safety import sanitize
from privacy.public_temporal_noise_training_v24 import METHODS, round_parameters, prefix_ledger
from privacy.public_temporal_noise_confirmation_v25 import plan, compare

NAME = 'public_temporal_noise_confirmation_v25'
OUT = ROOT/'results/ldp_gradient_far'/NAME
MATRIX = ROOT/'configs/ldp_gradient_far'/f'{NAME}.yaml'
PROTOCOL = ROOT/'output/analysis/Public_Temporal_Noise_Confirmation_V25_Protocol.md'
REPORT = ROOT/'output/analysis/Public_Temporal_Noise_Confirmation_V25_Status.md'
TEST = ROOT/'tests/test_public_temporal_noise_confirmation_v25.py'
AUDIT = ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Analyse.json'


def inputs():
    c = yaml.safe_load(MATRIX.read_text())
    assert c['campaign_id'] == NAME and c['device'] == 'mps'
    assert c['confirmation_seeds'] == [180601,180602,180603,180604]
    assert c['grid_indices'] == [0,13] and c['selected_grid_index'] == 13 and c['selected_method'] == 'risk_rfa'
    assert c['methods'] == list(METHODS) and c['expected_runs'] == 32
    assert c['primary_control_methods'] == ['erm_mean','erm_rfa'] and c['primary_control_grid_indices'] == [0,13]
    expected = dict(dataset='fashionmnist',model='lenet5_tanh',num_clients=10,partition='client_dirichlet_balanced',
        dirichlet_beta=.1,public_train_size=4800,validation_size=1200,batch_size=240,rounds=120,clip=2.,risk_scale=.5,
        epsilon=4.,delta=1e-5,risk_epsilon=.25,local_optimizer_steps=0,server_clip=None,recursive_memory=False,
        adjacency='replace_one',sampling='fixed_without_replacement',primary_endpoint='final_test_round_120',
        minimum_worst20_advantage_pp_per_seed=1.,maximum_accuracy_loss_pp_per_seed=1.,confidence_level=.95,
        paired_t_critical_df3=3.182446305284263,prospective_confirmation_wave=0,one_sided_alpha=.025,
        attacks='none',automatic_attacks=False,automatic_retry=False)
    assert all(c[k] == v for k,v in expected.items())
    assert c['learning_rates'] == [2.,.5] and c['test_evaluation_rounds'] == [120]
    assert c['future_wave_alpha_rule'] == '0.025*2**(-wave)'
    assert all(c[k] for k in ('require_positive_worst20_ci_lower','require_accuracy_ci_lower_ge_minus_one',
        'require_nonpositive_mean_gap_change','require_nonpositive_mean_variance_change'))
    audit = json.loads(AUDIT.read_text())
    assert audit['audit_passed'] and audit['decision']['selected'] == 13 and audit['decision']['eligible_for_independent_confirmation']
    old = json.loads((prior.OUT/'manifest.json').read_text())
    assert set(c['confirmation_seeds']).isdisjoint(old['config']['seeds'])
    stamp = dict(old['source_stamp']); base.verify_stamp(stamp)
    files = [Path(__file__),MATRIX,PROTOCOL,TEST,AUDIT,
        ROOT/'privacy/public_temporal_noise_confirmation_v25.py',ROOT/'scripts/analyze_public_temporal_noise_training_v24.py']
    files += [prior.OUT/prior.identifier(j)/name for j in prior.jobs(old['config'])
              for name in ('metrics.json','simulator_oracle.json','orchestration_status.json')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in files})
    profile = {k:c[k] for k in ('dataset','model','num_clients','partition','dirichlet_beta','public_train_size','validation_size','rounds')}
    profile['evaluation_rounds'] = [1]+list(range(10,121,10))
    return c,profile,stamp


def jobs(c):
    return [dict(seed=s,grid_index=k,method=m) for s in c['confirmation_seeds'] for k in c['grid_indices'] for m in METHODS]


def identifier(j):
    return f'seed{j["seed"]}__k{j["grid_index"]}__{j["method"]}'


def completed(j,stamp):
    path = OUT/identifier(j); status = path/'orchestration_status.json'
    if not status.exists() or json.loads(status.read_text())['status'] != 'completed':
        return False
    s = json.loads(status.read_text()); r = json.loads((path/'metrics.json').read_text())
    assert s['metrics_sha256'] == base.digest(path/'metrics.json') and s['oracle_sha256'] == base.digest(path/'simulator_oracle.json')
    assert r['job'] == j and r['source_stamp'] == stamp and r['device'] == 'mps'
    assert r['privacy'] == json.loads(json.dumps(plan(j['grid_index'],j['method'])))
    assert r['privacy']['epsilon_realized'] <= 4 and r['test_evaluated'] and r['test_evaluation_rounds'] == [120]
    assert [t['round'] for t in r['rounds']] == list(range(1,121))
    assert [t['round'] for t in r['rounds'] if t['test'] is not None] == [120]
    assert r['final'] == r['rounds'][-1]
    verify(r['final']['test']); verify(r['final']['validation'])
    return True


def train(j,c,profile,stamp,data,key):
    if completed(j,stamp):
        return
    path = OUT/identifier(j); path.mkdir(parents=True,exist_ok=True)
    p = plan(j['grid_index'],j['method']); model = base.new_model(profile,j['seed'])
    baseline = OUT/f'seed{j["seed"]}__k0__erm_mean'
    reference,expected = None,None
    if path != baseline:
        assert completed(dict(seed=j['seed'],grid_index=0,method='erm_mean'),stamp)
        reference = json.loads((baseline/'metrics.json').read_text())
        expected = json.loads((baseline/'simulator_oracle.json').read_text())['rounds']
    rows,oracles,start,elapsed = [],[],0,0.
    cp = path/'checkpoint.pt'; key_sha = base.digest(OUT/'simulator_secret.json')
    if cp.exists():
        state = torch.load(cp,map_location='cpu',weights_only=True)
        assert state['source_stamp'] == stamp and state['job'] == j and state['key_sha'] == key_sha
        model.load_state_dict(state['model'])
        rows,oracles,start,elapsed,initial = (state[k] for k in ('rows','oracles','round','elapsed_seconds','initial'))
    else:
        initial = base.evaluate(model,data,'val')
    if reference is not None:
        assert reference['initial'] == initial and reference['splits'] == data['splits']
    base.save(path/'public_protocol.json',dict(config=c,profile=profile,job=j,privacy=p,source_stamp=stamp))
    code = {k:v for k,v in stamp.items() if Path(k).suffix in ('.py','.yaml','.md')}
    for t in range(start,120):
        tick = time.monotonic(); require_mps(); base.verify_stamp(code)
        rp = round_parameters(p,t+1)
        status = dict(status='running',device='mps',active=identifier(j),round=t+1,total_rounds=120,
                      pid=os.getpid(),updated_unix=time.time())
        base.save(OUT/'status.json',status); base.save(path/'orchestration_status.json',status)
        before = {name:v.detach().cpu().clone() for name,v in model.state_dict().items()}
        sent,reports,local = [],[],[]
        for cid,ids in enumerate(data['train']):
            rr,raw = None,None
            if j['method'].startswith('risk_'):
                rr,raw = private_risk(model,data['x'][ids],data['y'][ids],noise_std=p['risk_std'],
                    seed=base.seed_for(key,j['seed'],t,cid,'risk'),N=4800)
                reports.append(rr)
            ix = base.draw_indices(4800,240,base.seed_for(key,j['seed'],t,cid,'batch'))
            ih = base.ids_hash(ix)
            if expected is not None:
                assert ih == expected[t]['clients'][cid]['batch_hash']
            _,g,norms,_ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            sent.append(release(g.mean(0),noise_std=rp['sigma'],seed=base.seed_for(key,j['seed'],t,cid,'gaussian')))
            local.append(dict(client=cid,batch_hash=ih,gradient_clipped_count=int((norms>2).sum()),batch_size=240,
                private_risk=None if rr is None else float(rr),raw_risk=None if raw is None else float(raw)))
        X,r,safety = sanitize(torch.stack(sent),None if not reports else torch.stack(reports))
        assert safety['invalid_message_rows'] == safety['nonfinite_risk_reports'] == 0
        step,agg = aggregate(X,r,kind=j['method'],round_number=t+1,horizon=120)
        assert agg['eta'] == rp['eta']; agg['message_safety'] = safety
        base.apply_gradient(model,step,1.)
        val = base.evaluate(model,data,'val') if t+1 in profile['evaluation_rounds'] else None
        test = base.evaluate(model,data,'test') if t+1 == 120 else None
        if val is not None: verify(val)
        if test is not None: verify(test)
        prefix = prefix_ledger(p,t+1); assert prefix['epsilon'] <= 4
        rows.append(dict(round=t+1,device='mps',epsilon_realized=prefix['epsilon'],privacy_order=prefix['order'],
                         noise_schedule=rp,aggregation=agg,validation=val,test=test))
        oracles.append(dict(round=t+1,clients=local)); elapsed += time.monotonic()-tick
        base.checkpoint(cp,dict(model={name:v.detach().cpu() for name,v in model.state_dict().items()},previous_model=before,
            rows=rows,oracles=oracles,round=t+1,elapsed_seconds=elapsed,initial=initial,source_stamp=stamp,job=j,key_sha=key_sha))
        if val is not None:
            print(f'{identifier(j)} {t+1}/120 val_acc={val["accuracy_pct"]:.3f} W20={val["worst20_pct"]:.3f}',flush=True)
    base.verify_stamp(stamp)
    base.save(path/'metrics.json',dict(job=j,source_stamp=stamp,device='mps',privacy=p,initial=initial,rounds=rows,final=rows[-1],
        splits=data['splits'],test_evaluated=True,test_evaluation_rounds=[120],per_client_batch_gradient_evaluations=120,
        validation_test_and_oracles_not_private=True,elapsed_seconds=elapsed))
    base.save(path/'simulator_oracle.json',dict(privacy_protected=False,feeds_mechanism=False,rounds=oracles))
    base.save(path/'orchestration_status.json',dict(status='completed',device='mps',job=j,round=120,
        metrics_sha256=base.digest(path/'metrics.json'),oracle_sha256=base.digest(path/'simulator_oracle.json')))
    del model; torch.mps.empty_cache()


def decide(c,rows):
    expected = {(j['seed'],j['grid_index'],j['method']) for j in jobs(c)}
    index = {(r['job']['seed'],r['job']['grid_index'],r['job']['method']):r['final']['test'] for r in rows}
    if len(rows) != 32 or len(index) != 32 or set(index) != expected:
        raise ValueError('All 32 distinct preregistered final-test results required')
    contrasts = []
    for mode in c['primary_control_grid_indices']:
        for method in c['primary_control_methods']:
            pairs = []
            for seed in c['confirmation_seeds']:
                a,b = index[seed,13,'risk_rfa'],index[seed,mode,method]
                delta = {k:a[k]-b[k] for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2')}
                pairs.append(dict(seed=seed,delta=delta,seed_gate=delta['accuracy_pct']>=-1. and delta['worst20_pct']>=1.))
            contrast = compare([p['delta'] for p in pairs])
            contrasts.append(dict(control_grid_index=mode,control_method=method,pairs=pairs,**contrast))
    return dict(clean_confirmation_passed=all(x['passed'] for x in contrasts),contrasts=contrasts,
        selected_grid_index=13,selected_method='risk_rfa',endpoint='fixed final test round 120',
        prospective_confirmation_wave=0,one_sided_alpha=.025,attacks_evaluated=False,
        joint_privacy_fairness_robustness_validated=False)


def report(c,stamp):
    rows = [json.loads((OUT/identifier(j)/'metrics.json').read_text()) for j in jobs(c) if completed(j,stamp)]
    lines = ['# V25 — confirmation sur quatre nouvelles seeds','',f'**{len(rows)}/32 runs MPS valides**.', '',
        'Test final120, critères fixés avant les résultats. Candidate unique : risque-RFA au ratio1,756252 ; bruit constant conservé comme contrôle. Aucun résultat de robustesse à ce stade.', '',
        '| Seed | Ratio | Règle | Test accuracy % | Worst-20 % | Gap pp | Variance pp² | Brier |',
        '|--:|--:|:--|--:|--:|--:|--:|--:|']
    for r in rows:
        j,v = r['job'],r['final']['test']
        lines.append(f'| {j["seed"]} | {r["privacy"]["ratio"]:.6f} | {j["method"]} | {v["accuracy_pct"]:.4f} | {v["worst20_pct"]:.4f} | {v["gap_best20_worst20_pp"]:.4f} | {v["variance_pp2"]:.4f} | {v["brier_loss"]:.5f} |')
    decision = None
    if len(rows) == 32:
        decision = decide(c,rows); base.save(OUT/'evidence.json',dict(source_stamp=stamp,decision=decision))
        lines += ['',f'Confirmation propre : **{"PASS" if decision["clean_confirmation_passed"] else "FAIL"}**. Audit indépendant requis ; aucune validation conjointe et aucune attaque automatique.']
    lines += ['','[Protocole préenregistré](Public_Temporal_Noise_Confirmation_V25_Protocol.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    return len(rows),decision


def worker():
    require_mps(); OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            c,profile,stamp = inputs(); manifest = dict(config=c,profile=profile,source_stamp=stamp)
            if (OUT/'manifest.json').exists():
                assert json.loads((OUT/'manifest.json').read_text()) == manifest
            else:
                result = subprocess.run(['rg','--files','--hidden','--no-ignore','results'],cwd=ROOT,capture_output=True,text=True,check=True)
                pattern = re.compile(r'(?<![0-9])18060[1-4](?![0-9])')
                own = str(OUT.relative_to(ROOT))+'/'
                conflicts = [s for s in result.stdout.splitlines() if pattern.search(s) and not s.startswith(own)]
                assert not conflicts, 'Reserved seeds found in prior output paths'
                base.save(OUT/'seed_reservation.json',dict(seeds=c['confirmation_seeds'],prior_output_path_conflicts=conflicts,
                    search_scope='All result file paths including hidden/ignored files before launch; declarations previously searched in configs/scripts/protocols',reserved_before_training=True))
                base.save(OUT/'manifest.json',manifest)
            if not (OUT/'tests.json').exists():
                tests = subprocess.run([sys.executable,'-m','pytest',str(TEST),'-q'],cwd=ROOT,capture_output=True,text=True)
                base.save(OUT/'tests.json',dict(passed=tests.returncode==0,source_stamp=stamp,output=tests.stdout+tests.stderr))
            tests = json.loads((OUT/'tests.json').read_text()); assert tests['passed'] and tests['source_stamp'] == stamp
            if not (OUT/'simulator_secret.json').exists():
                assert not list(OUT.glob('seed*/checkpoint.pt'))
                base.save(OUT/'simulator_secret.json',dict(key=secrets.token_hex(32),private_release=False))
            key = json.loads((OUT/'simulator_secret.json').read_text())['key']
            for seed in c['confirmation_seeds']:
                data = base.prepare(profile,seed)
                for j in [j for j in jobs(c) if j['seed'] == seed]:
                    train(j,c,profile,stamp,data,key)
                    n,_ = report(c,stamp); print(f'{n}/32 completed',flush=True)
                del data; torch.mps.empty_cache()
            n,decision = report(c,stamp); assert n == 32
            base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=32,
                clean_confirmation_passed=decision['clean_confirmation_passed'],next_campaign_launched=False))
        except Exception as exc:
            base.save(OUT/'status.json',dict(status='failed',device='mps',error=repr(exc),pid=os.getpid()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume',action='store_true',required=True)
    parser.parse_args(); worker()
