"""24 fixed-state MPS rejection probes; no end-to-end training or promotion."""
import argparse
import fcntl
from fractions import Fraction as Q
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts.analyze_full_population_error_attribution_v28b import source_evidence
from privacy.fair_objective import require_mps
from privacy.capped_private_risk import weights
from privacy.private_risk_attacks import inject
from privacy.private_risk_winsor_v32 import winsor
from privacy.stable_weighted_rfa import weighted_rfa,stable_norm

SOURCE=ROOT/'results/ldp_gradient_far/full_population_private_risk_calibration_v28'
OUT=ROOT/'results/ldp_gradient_far/private_risk_winsor_screen_v32'
DEST=ROOT/'output/analysis/Private_Risk_V32_Fixed_State_Screen'
PROTOCOL=ROOT/'output/analysis/Private_Risk_V32_Construction_and_Protocol.md'
TESTS=(ROOT/'tests/test_private_risk_winsor_v32.py',ROOT/'tests/test_private_risk_winsor_v32_screen.py')
SEEDS=(170501,170502)
ATTACKS=('none','abrupt_bf','persistent_alie','slow_ipm')
METHODS=('risk_mean','risk_rfa','risk_winsor')


def evaluation_subset(evaluation,ids):
    clients=[evaluation['clients'][i] for i in ids]
    acc=[]
    for c in clients:
        n,h=c['class_count'],c['class_hits']
        if (c['N']!=1200 or len(n)!=10 or len(h)!=10 or sum(n)!=1200
                or any(not math.isfinite(x) or x<0 or x!=int(x) for x in n+h)
                or any(y>x for x,y in zip(n,h))):
            raise ValueError('Invalid exact validation counts')
        acc.append(Q(100*int(sum(h)),1200))
    tail=math.ceil(.2*len(clients));ordered=sorted(acc);mean=sum(acc)/len(acc)
    return dict(accuracy_pct=mean,worst20_pct=sum(ordered[:tail])/tail,
        gap_best20_worst20_pp=(sum(ordered[-tail:])-sum(ordered[:tail]))/tail,
        variance_pp2=sum((x-mean)**2 for x in acc)/len(acc))


def decision(rows):
    expected={(s,a,m) for s in SEEDS for a in ATTACKS for m in METHODS}
    index={(r['seed'],r['attack'],r['method']):r for r in rows}
    if len(rows)!=24 or set(index)!=expected:raise ValueError('All 24 distinct probes required')
    comparisons=[]
    for seed in SEEDS:
        c=evaluation_subset(index[seed,'none','risk_winsor']['evaluation'],range(10))
        b=evaluation_subset(index[seed,'none','risk_mean']['evaluation'],range(10))
        da,dw=c['accuracy_pct']-b['accuracy_pct'],c['worst20_pct']-b['worst20_pct']
        comparisons.append(dict(seed=seed,attack='none',comparison='vs risk mean, all ten',
            accuracy_delta_pp=float(da),worst20_delta_pp=float(dw),passed=da>=Q(-1,10) and dw>=Q(-1,4)))
        clean=evaluation_subset(index[seed,'none','risk_winsor']['evaluation'],range(2,10))
        for attack in ATTACKS[1:]:
            v=evaluation_subset(index[seed,attack,'risk_winsor']['evaluation'],range(2,10))
            rfa=evaluation_subset(index[seed,attack,'risk_rfa']['evaluation'],range(2,10))
            for label,control in (('vs own clean, eight honest',clean),('vs RFA, eight honest',rfa)):
                da,dw=v['accuracy_pct']-control['accuracy_pct'],v['worst20_pct']-control['worst20_pct']
                comparisons.append(dict(seed=seed,attack=attack,comparison=label,
                    accuracy_delta_pp=float(da),worst20_delta_pp=float(dw),passed=da>=-1 and dw>=-1))
    numerical=all(r['numerical_checks_passed'] for r in rows)
    return dict(local_feasibility_passed=numerical and all(c['passed'] for c in comparisons),
        comparisons=comparisons,numerical_checks_passed=numerical,global_validation=False,
        automatic_training=False,V30_opened=False,V29_changed=False)


def run_seed(seed,profile,stamp):
    name=f'seed{seed}__risk_rfa';inputs=source_evidence(dict(seed=seed,method='risk_rfa'))
    assert inputs is not None
    signature=dict(stamp=stamp,inputs=inputs)
    path=OUT/f'seed{seed}.json'
    if path.exists():
        cached=json.loads(path.read_text());assert cached['signature']==signature
        return cached['rows']
    cp=torch.load(SOURCE/name/'checkpoint.pt',map_location='cpu',weights_only=True)
    data=base.prepare(profile,seed);model=base.new_model(profile,seed)
    messages=cp['last_private_messages'].to('mps');reports=cp['last_reports'].to('mps')
    rows=[]
    for attack in ATTACKS:
        x,r,attack_info=inject(messages,reports,attack=attack,round_number=90)
        assert torch.equal(x[2:],messages[2:]) and torch.equal(r[2:],reports[2:])
        lam,_=weights(r,.5)
        mean=(lam[:,None]*x).sum(0)
        rfa,solver=weighted_rfa(x,lam)
        candidate,diag,v=winsor(x,r,C=2.,f_budget=2)
        assert diag['clipped_count']<=2
        assert diag['center_norm']<=2+1e-5
        hmax=float(stable_norm(x[2:],dim=1).max())
        assert diag['radius']<=max(4.,2+hmax)+1e-4
        assert float(stable_norm(candidate))<=max(6.,4+hmax)+1e-4
        beta=lam[:2].sum();target=(lam[2:,None]*x[2:]).sum(0)/(1-beta)
        removed=(lam[2:,None]*v['removed'][2:]).sum(0)
        byz=(lam[:2,None]*v['clipped_residuals'][:2]).sum(0)
        reconstruction=-removed+beta*(v['center']-target)+byz
        torch.testing.assert_close(candidate-target,reconstruction,rtol=3e-5,atol=2e-6)
        bound=float(stable_norm(removed)+beta*(stable_norm(v['center']-target)+diag['radius']))
        assert float(stable_norm(candidate-target))<=bound+1e-4
        assert float(beta)<=6/14+1e-6
        assert solver['unsmoothed_objective_gap_upper']<=.001
        assert diag['pilot_solver']['unsmoothed_objective_gap_upper']<=.001
        identity=dict(reserved_honest_ids=list(range(2,10)),beta=float(beta),
            honest_removed_norm=float(stable_norm(removed)),
            byzantine_centered_contribution_norm=float(stable_norm(byz)),
            byzantine_centered_upper=float(beta)*diag['radius'],
            applied_minus_honest_private_mean_norm=float(stable_norm(candidate-target)),
            error_upper=bound,actual_attack_active=attack_info['active'],
            note='Identities 0/1 reserved even for clean pairing; clean beta is not malicious mass')
        for kind,A in (('risk_mean',mean),('risk_rfa',rfa),('risk_winsor',candidate)):
            model.load_state_dict(cp['pre_round_model']);base.apply_gradient(model,A,.5)
            evaluated=base.evaluate(model,data,'val')
            if attack=='none' and kind=='risk_rfa':
                assert torch.equal(.5*A,cp['last_step'].to('mps'))
                assert all(torch.equal(value,cp['model'][k].to('mps')) for k,value in model.state_dict().items())
                assert evaluated==cp['rows'][-1]['validation']
            rows.append(dict(seed=seed,attack=attack,method=kind,evaluation=evaluated,
                honest_metrics={k:float(val) for k,val in evaluation_subset(evaluated,range(2,10)).items()},
                aggregate_norm=float(stable_norm(A)),
                squared_error_to_private_honest_mean=float((A-target).square().sum()),
                candidate_diagnostics=diag if kind=='risk_winsor' else None,
                bound_identity=identity if kind=='risk_winsor' else None,
                numerical_checks_passed=True,device='mps',attack_info=attack_info))
        print(f'V32 seed {seed} {attack}: 3/3 virtual steps complete',flush=True)
    base.verify_stamp(inputs);base.verify_stamp(stamp)
    base.save(path,dict(rows=rows,signature=signature,source_original_reproduced=True,
        private_release=False,test_evaluated=False,training=False,device='mps'))
    del model,data,cp
    torch.mps.empty_cache()
    return rows


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'screen.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=json.loads((SOURCE/'manifest.json').read_text());stamp=dict(parent['source_stamp'])
        extra=(Path(__file__),PROTOCOL,*TESTS,SOURCE/'manifest.json',
               ROOT/'privacy/private_risk_winsor_v32.py',ROOT/'privacy/private_risk_attacks.py',
               ROOT/'scripts/analyze_full_population_error_attribution_v28b.py')
        stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in extra});base.verify_stamp(stamp)
        manifest=dict(stamp=stamp,seeds=SEEDS,attacks=ATTACKS,methods=METHODS,expected_steps=24,
            C=2.,f=2,eta=.5,no_training=True,no_automatic_promotion=True)
        if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==json.loads(json.dumps(manifest))
        else:base.save(OUT/'manifest.json',manifest)
        test=subprocess.run([sys.executable,'-m','pytest',*[str(t) for t in TESTS],'-q'],cwd=ROOT,capture_output=True,text=True)
        base.save(OUT/'tests.json',dict(passed=test.returncode==0,output=test.stdout+test.stderr,stamp=stamp))
        assert test.returncode==0,test.stdout+test.stderr
        rows=[]
        for seed in SEEDS:rows.extend(run_seed(seed,parent['profile'],stamp))
        verdict=decision(rows);base.verify_stamp(stamp)
        base.save(DEST.with_suffix('.json'),dict(rows=rows,decision=verdict,source_stamp=stamp,
            global_validation=False,device='mps',training=False,test_evaluated=False))
        lines=['# V32 — écran local de conservation et de résistance','',
            '**24/24 pas virtuels sur MPS, deux anciennes seeds de calibration.**', '',
            'Faisabilité locale : **'+('PASS' if verdict['local_feasibility_passed'] else 'FAIL')+'**. '
            'Ce verdict ne change pas V29 et n’ouvre ni V30 ni une nouvelle campagne. '
            'Un pas terminal n’est pas une attaque persistante ni une preuve de convergence.', '',
            '| Seed | Condition | Méthode | Accuracy honnête (%) | Worst-20 honnête (%) | Gap honnête (pp) | Variance honnête (pp²) |',
            '|--:|:--|:--|--:|--:|--:|--:|']
        for r in rows:
            h=r['honest_metrics']
            lines.append(f"| {r['seed']} | {r['attack']} | {r['method']} | "+' | '.join(f'{h[k]:.5f}' for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2'))+' |')
        lines+=['','Les huit identités 2…9 sont identiques pour les lignes propres et attaquées. '
            'Le contrôle propre principal compare toutefois les dix clients comme fixé au protocole.', '',
            '| Seed | Condition | Comparaison | Δ accuracy (pp) | Δ Worst-20 (pp) | Critère |',
            '|--:|:--|:--|--:|--:|:--|']
        for c in verdict['comparisons']:
            lines.append(f"| {c['seed']} | {c['attack']} | {c['comparison']} | {c['accuracy_delta_pp']:+.5f} | {c['worst20_delta_pp']:+.5f} | {'PASS' if c['passed'] else 'FAIL'} |")
        lines+=['','Les oracles et évaluations restent hors transcript privé. Les attaquants '
            'ne lisent que les messages privés, pas les gradients propres. L’erreur diagnostique '
            'vise la moyenne pondérée des huit messages honnêtes privés, non le gradient de population. '
            'Les bornes de contribution ne garantissent pas la performance du modèle.', '',
            '[Construction, preuves et protocole](Private_Risk_V32_Construction_and_Protocol.md) · '
            '[Résultats complets](Private_Risk_V32_Fixed_State_Screen.json)', '']
        DEST.with_suffix('.md').write_text('\n'.join(lines))
        base.save(OUT/'status.json',dict(status='completed',device='mps',virtual_steps=24,
            local_feasibility_passed=verdict['local_feasibility_passed'],global_validation=False))
        print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true',required=True);p.parse_args();main()
