"""Frozen 24-branch calibration continuation, not a new confirmation."""
import argparse
import fcntl
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from scripts.screen_private_risk_winsor_v32 import evaluation_subset
from privacy.fair_objective import require_mps
from privacy.full_population_private_risk_v28 import ledger, private_population_gradient
from privacy.split_risk_gradient import private_risk
from privacy.capped_private_risk import weights
from privacy.stable_weighted_rfa import weighted_rfa,stable_norm
from privacy.private_risk_winsor_v32 import winsor

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
OUT=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
PROTOCOL=ROOT/'output/analysis/Private_Risk_V33_Continuation_Protocol.md'
TEST=ROOT/'tests/test_private_risk_continuation_v33.py'
SEEDS=(170501,170502)
METHODS=('risk_mean','risk_rfa','risk_winsor')
ATTACKS=('none','abrupt_bf','persistent_alie','slow_ipm')

def epsilon(plan,k):
    if type(k)!=int or not 0<=k<=12:raise ValueError('Extension prefix must be 0..12')
    return min(((120+k)*a*(1/(2*plan['gradient_z']**2)+1/(2*plan['risk_z']**2))
                +math.log(1/plan['delta'])/(a-1),a) for a in range(2,65))

def attack(messages,reports,kind,k):
    if kind not in ATTACKS or type(k)!=int or not 1<=k<=12:raise ValueError('Invalid public schedule')
    active=kind!='none' and 5<=k<=(12 if kind=='persistent_alie' else 8)
    x,r=messages.clone(),reports.clone();factor=None
    if active:
        mu=messages[2:].mean(0)
        if kind=='abrupt_bf':factor=10.;x[:2]=-factor*mu
        elif kind=='persistent_alie':factor=1.5;x[:2]=mu+factor*messages[2:].std(0,correction=0)
        else:factor=.5*(k-4);x[:2]=-factor*mu
        r[:2]=1.
    return x,r,dict(active=active,multiplier=factor,byzantine_ids=[0,1],oracle_used=False)

def aggregate(x,r,method):
    lam,_=weights(r,.5)
    if method=='risk_mean':return (lam[:,None]*x).sum(0),{},None
    if method=='risk_rfa':
        A,info=weighted_rfa(x,lam)
        assert info['unsmoothed_objective_gap_upper']<=.001
        return A,dict(solver=info),None
    if method!='risk_winsor':raise ValueError('Unknown method')
    A,info,v=winsor(x,r,C=2.,f_budget=2)
    assert info['pilot_solver']['unsmoothed_objective_gap_upper']<=.001
    assert info['clipped_count']<=2 and info['center_norm']<=2+1e-5
    hm=float(stable_norm(x[2:],dim=1).max())
    assert info['radius']<=max(4.,2+hm)+1e-4
    assert float(stable_norm(A))<=max(6.,4+hm)+1e-4
    return A,info,v

def name(seed,kind,method):return f'seed{seed}__{kind}__{method}'
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def model_hash(model):return base.ids_hash(torch.cat([p.detach().flatten() for p in model.parameters()]))

def decision(results):
    index={(r['seed'],r['attack'],r['method']):r for r in results}
    expected={(s,a,m) for s in SEEDS for a in ATTACKS for m in METHODS}
    if len(results)!=24 or set(index)!=expected:raise ValueError('24 unique complete branches required')
    def metric(s,a,m,k,ids):
        record=index[s,a,m]
        if [r['step'] for r in record['rows']]!=list(range(1,13)):raise ValueError('Incomplete trajectory')
        return evaluation_subset(record['rows'][k-1]['validation'],ids)
    checks=[]
    for s in SEEDS:
        for k in (4,8,12):
            v=metric(s,'none','risk_winsor',k,range(10));b=metric(s,'none','risk_mean',k,range(10))
            da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
            checks.append(dict(seed=s,attack='none',step=k,control='risk_mean',accuracy_delta_pp=float(da),
                worst20_delta_pp=float(dw),passed=da>=Q(-1,10) and dw>=Q(-1,4)))
        for a in ATTACKS[1:]:
            for k in (8,12):
                v=metric(s,a,'risk_winsor',k,range(2,10))
                for ca,cm in (('none','risk_winsor'),(a,'risk_rfa')):
                    b=metric(s,ca,cm,k,range(2,10));da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
                    checks.append(dict(seed=s,attack=a,step=k,control=f'{ca}/{cm}',accuracy_delta_pp=float(da),
                        worst20_delta_pp=float(dw),passed=da>=-1 and dw>=-1))
        for m in METHODS:
            clean=index[s,'none',m]
            for a in ATTACKS[1:]:
                for k in range(4):
                    if index[s,a,m]['rows'][k]['model_hash']!=clean['rows'][k]['model_hash']:
                        raise ValueError('Pre-attack pairing failed')
    return dict(local_gate_passed=all(c['passed'] for c in checks),comparisons=checks,
        global_validation=False,automatic_promotion=False,V29_changed=False,V30_opened=False)

def train(seed,kind,method,profile,stamp,key,plan,data):
    folder=OUT/name(seed,kind,method);folder.mkdir(parents=True,exist_ok=True)
    mp,cp=folder/'metrics.json',folder/'checkpoint.pt'
    sig=dict(seed=seed,attack=kind,method=method,source_stamp=stamp)
    if mp.exists():
        result=json.loads(mp.read_text());status=json.loads((folder/'orchestration_status.json').read_text())
        assert result['signature']==sig and status['status']=='completed' and status['metrics_sha256']==base.digest(mp)
        assert status['checkpoint_sha256']==base.digest(cp)
        assert result['device']=='mps' and result['test_evaluated'] is False and result['privacy']==json.loads(json.dumps(plan))
        assert [row['step'] for row in result['rows']]==list(range(1,13))
        return result
    parent=torch.load(SOURCE/f'seed{seed}__risk_rfa'/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert parent['round']==120 and parent['privacy']==plan
    model=base.new_model(profile,seed);model.load_state_dict(parent['model'])
    rows=[];initial=None;error_sum=torch.zeros(sum(p.numel() for p in model.parameters()),device='mps');attack_sum=error_sum.clone()
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True);assert saved['signature']==sig
        model.load_state_dict(saved['model']);rows=saved['rows'];initial=saved['initial']
        assert [row['step'] for row in rows]==list(range(1,len(rows)+1)) and len(rows)<=12
        error_sum=saved['error_sum'].to('mps');attack_sum=saved['attack_sum'].to('mps')
    else:initial=base.evaluate(model,data,'val');assert initial==parent['rows'][-1]['validation']
    del parent
    for k in range(len(rows)+1,13):
        require_mps();base.verify_stamp(stamp)
        state=dict(status='running',device='mps',pid=os.getpid(),active=folder.name,step=k,total_steps=12,updated_unix=time.time())
        base.save(OUT/'status.json',state);base.save(folder/'orchestration_status.json',state)
        before=model_hash(model);pre_state=cpu_state(model);sent=[];reports=[];clean=[];local=[]
        for cid,ids in enumerate(data['train']):
            rr,_=private_risk(model,data['x'][ids],data['y'][ids],noise_std=plan['risk_std'],
                seed=base.seed_for(key,'v33',seed,k,cid,'risk'),N=4800)
            ix=base.draw_indices(4800,4800,base.seed_for(key,'v33',seed,k,cid,'batch'))
            msg,query,diag=private_population_gradient(model,data['x'][ids[ix]],data['y'][ids[ix]],C=2.,block_size=240,
                noise_std=plan['gradient_std'],seed=base.seed_for(key,'v33',seed,k,cid,'gaussian'))
            sent.append(msg);reports.append(rr);clean.append(query)
            local.append(dict(client=cid,permutation_hash=base.ids_hash(ix),private_message_hash=base.ids_hash(msg),
                gradient=diag,private_risk=float(rr)))
        assert model_hash(model)==before
        raw,rr=torch.stack(sent),torch.stack(reports)
        x,r,info=attack(raw,rr,kind,k);assert torch.equal(x[2:],raw[2:]) and torch.equal(r[2:],rr[2:])
        A,diag,v=aggregate(x,r,method);A0,_,_=aggregate(raw,rr,method)
        lam,_=weights(r,.5);beta=lam[:2].sum();assert float(beta)<=6/14+1e-6
        target=(lam[2:,None]*torch.stack(clean)[2:]).sum(0)/(1-beta)
        err=A-target;pert=A-A0;error_sum+=err;attack_sum+=pert
        oracle=dict(target='risk-weighted honest clipped gradient at own current model, private report weights',
            applied_error_squared=float(err.square().sum()),cumulative_error_norm=float(stable_norm(error_sum)),
            same_state_attack_perturbation_norm=float(stable_norm(pert)),
            cumulative_attack_perturbation_norm=float(stable_norm(attack_sum)),reserved_pair_mass=float(beta),
            active_byzantine_mass=float(beta) if info['active'] else 0.,feeds_mechanism=False)
        if v is not None:
            removed=(lam[2:,None]*v['removed'][2:]).sum(0)
            oracle.update(honest_removed_norm=float(stable_norm(removed)),
                honest_clipped_count=sum(f<1 for f in diag['factors'][2:]),
                reserved_pair_clipped_count=sum(f<1 for f in diag['factors'][:2]))
        base.apply_gradient(model,A,.5)
        val=base.evaluate(model,data,'val') if k in (4,8,12) else None
        ep,order=epsilon(plan,k)
        rows.append(dict(step=k,parent_round=120,validation=val,model_hash=model_hash(model),device='mps',
            epsilon_total=ep,epsilon_order=order,aggregation=diag,attack=info,oracle=oracle,clients=local))
        base.checkpoint(cp,dict(signature=sig,model=cpu_state(model),rows=rows,initial=initial,
            pre_round_model=pre_state,last_private_messages=raw.detach().cpu(),last_reports=rr.detach().cpu(),
            last_clean_means=torch.stack(clean).detach().cpu(),last_step=(.5*A).detach().cpu(),
            error_sum=error_sum.cpu(),attack_sum=attack_sum.cpu(),privacy_protected=False))
        print(f'V33 {folder.name} {k}/12'+(f" acc={val['accuracy_pct']:.3f} W20={val['worst20_pct']:.3f}" if val else ''),flush=True)
    result=dict(**{k:sig[k] for k in ('seed','attack','method')},signature=sig,rows=rows,initial=initial,
        device='mps',test_evaluated=False,global_validation=False,privacy=plan,epsilon_total_132=epsilon(plan,12)[0],
        artifacts_private=False,shared_parent=True,independent_confirmation=False)
    base.verify_stamp(stamp);base.save(mp,result)
    base.save(folder/'orchestration_status.json',dict(status='completed',device='mps',steps=12,
        metrics_sha256=base.digest(mp),checkpoint_sha256=base.digest(cp)))
    del model;torch.mps.empty_cache();return result

def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=json.loads((SOURCE/'manifest.json').read_text());stamp=dict(parent['source_stamp'])
        for s in SEEDS:stamp.update(source_evidence(dict(seed=s,method='risk_rfa')))
        v32=json.loads((ROOT/'output/analysis/Private_Risk_V32_Independent_Replay.json').read_text())
        assert v32['audit_passed'] and v32['independent_gate_passed'];stamp.update(v32['source_stamp'])
        secret=ROOT/'results/ldp_gradient_far/recursive_private_risk_calibration_v19/simulator_secret.json'
        extra=[Path(__file__),PROTOCOL,TEST,secret,ROOT/'output/analysis/Private_Risk_V32_Independent_Replay.json',
               ROOT/'scripts/screen_private_risk_winsor_v32.py']
        stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in extra});base.verify_stamp(stamp)
        plan=ledger('risk_rfa');manifest=dict(stamp=stamp,seeds=SEEDS,methods=METHODS,attacks=ATTACKS,steps=12,
            expected_runs=24,privacy_parent=plan,epsilon_total_132=epsilon(plan,12)[0],device='mps',global_validation=False)
        frozen=json.loads(json.dumps(manifest));mp=OUT/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==frozen
        else:base.save(mp,manifest)
        tests=subprocess.run([sys.executable,'-m','pytest',str(TEST),'tests/test_private_risk_winsor_v32.py','-q'],cwd=ROOT,capture_output=True,text=True)
        base.save(OUT/'tests.json',dict(passed=tests.returncode==0,output=tests.stdout+tests.stderr,source_stamp=stamp))
        assert tests.returncode==0,tests.stdout+tests.stderr
        key=json.loads(secret.read_text())['key'];results=[]
        for seed in SEEDS:
            data=base.prepare(parent['profile'],seed)
            for kind in ATTACKS:
                for method in METHODS:
                    results.append(train(seed,kind,method,parent['profile'],stamp,key,plan,data))
                    print(f'V33 completed {len(results)}/24',flush=True)
            del data;torch.mps.empty_cache()
        verdict=decision(results)
        base.save(OUT/'decision.json',dict(**verdict,source_stamp=stamp,independent_audit_pending=True))
        base.save(OUT/'status.json',dict(status='completed',device='mps',valid_runs=24,global_validation=False,
            local_gate_passed=verdict['local_gate_passed'],independent_audit_pending=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',required=True);p.parse_args()
    try:main()
    except BlockingIOError:raise
    except Exception as exc:
        base.save(OUT/'failure.json',dict(error=repr(exc),pid=os.getpid(),device='mps',time=time.time()))
        base.save(OUT/'status.json',dict(status='failed',error=repr(exc),device='mps',pid=os.getpid()))
        raise
