#!/usr/bin/env python3
"""MPS inference-only diagnostic of known/local bounded-loss gradient controls."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps,losses
from privacy.stable_weighted_rfa import stable_norm


def gradients(z,y):
    p=z.softmax(-1);one=torch.nn.functional.one_hot(y,num_classes=z.shape[1]).to(p.dtype)
    ce=p-one
    br=p*(ce-(p*ce).sum(1,keepdim=True))
    loss=torch.nn.functional.cross_entropy(z,y,reduction='none')
    py=p.gather(1,y[:,None]).squeeze(1)
    return dict(ce=ce,brier=br,bounded_log=ce/(1+loss[:,None]).square(),gce=.5*py.sqrt()[:,None]*ce),p,loss,py


def tests():
    torch.mps.manual_seed(171010)
    z=torch.randn(17,10,device='mps',requires_grad=True)
    y=torch.arange(17,device='mps')%10
    gg,p,ll,py=gradients(z,y)
    vals=dict(ce=ll,brier=losses(z,y),bounded_log=ll/(1+ll),gce=1-py.sqrt())
    for name,v in vals.items():
        actual=torch.autograd.grad(v.sum(),z,retain_graph=True)[0]
        torch.testing.assert_close(actual,gg[name],atol=2e-7,rtol=1e-5)
    extreme=torch.tensor([[0.,10.],[10.,0.]],device='mps');lab=torch.tensor([0,1],device='mps')
    g,_,_,_=gradients(extreme,lab)
    assert bool((stable_norm(g['bounded_log'],1)>50*stable_norm(g['brier'],1)).all())


def describe(x):
    if not len(x):return None
    ordered=x.sort().values;n=len(x)
    return dict(mean=float(x.mean()),median=float(ordered[(n-1)//2]),p10=float(ordered[int(.1*(n-1))]),p90=float(ordered[int(.9*(n-1))]))


def main():
    require_mps();tests()
    oldroot=ROOT/'results/ldp_gradient_far/capped_private_risk_calibration_v7'
    manifest=json.loads((oldroot/'manifest.json').read_text());base.verify_stamp(manifest['source_stamp'])
    m=manifest['config'];rows=[];groups=[];sources={}
    for seed in (170501,170502):
        d=oldroot/f'seed{seed}__erm_mean_C2';rec=json.loads((d/'metrics.json').read_text())
        sources[str(seed)]={k:base.digest(d/k) for k in ('metrics.json','checkpoint.pt')}
        hard=sorted(range(10),key=lambda i:rec['final']['validation']['clients'][i]['accuracy'])[:2]
        cp=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert cp['round']==60
        model=base.new_model(m,seed);model.load_state_dict(cp['model']);data=base.prepare(m,seed)
        assert data['splits']==rec['splits']
        pooled=[]
        for cid,ids in enumerate(data['train']):
            fragments=[]
            with torch.no_grad():
                for batch in ids.split(256):
                    z=model(data['x'][batch]);y=data['y'][batch]
                    gg,p,ll,py=gradients(z,y)
                    norms={name:stable_norm(g,1) for name,g in gg.items()}
                    fragment=torch.stack([py,ll,(p.argmax(1)!=y).float(),
                        norms['brier']/norms['ce'].clamp_min(1e-30),norms['bounded_log']/norms['ce'].clamp_min(1e-30),
                        norms['gce']/norms['ce'].clamp_min(1e-30),norms['ce'],norms['brier'],norms['bounded_log']],1)
                    assert bool(torch.isfinite(fragment).all());fragments.append(fragment)
            a=torch.cat(fragments);pooled.append(a)
            err=a[:,2]>0
            rows.append(dict(seed=seed,client=cid,hard=cid in hard,N=len(a),error_count=int(err.sum()),
                misclassified=dict(p_true=describe(a[err,0]),brier_over_ce=describe(a[err,3]),bounded_log_over_ce=describe(a[err,4]),
                    fraction_brier_over_ce_le_01=float((a[err,3]<=.1).float().mean()) if bool(err.any()) else None,
                    fraction_brier_over_ce_le_001=float((a[err,3]<=.01).float().mean()) if bool(err.any()) else None)))
        for which,ids in [('hard',hard),('other',[i for i in range(10) if i not in hard]),('all',list(range(10)))]:
            a=torch.cat([pooled[i] for i in ids]);err=a[:,2]>0
            for correctness,mask in [('incorrect',err),('correct',~err)]:
                aa=a[mask]
                groups.append(dict(seed=seed,group=which,hard_ids=hard,correctness=correctness,N=len(aa),
                    p_true=describe(aa[:,0]),brier_over_ce=describe(aa[:,3]),bounded_log_over_ce=describe(aa[:,4]),gce_over_ce=describe(aa[:,5]),
                    brier_below_01_count=int((aa[:,3]<=.1).sum()),brier_below_001_count=int((aa[:,3]<=.01).sum()),
                    fraction_bounded_log_over_2brier=float((aa[:,8]>2*aa[:,7]).float().mean()) if len(aa) else None))
        del model,data,cp,pooled
        torch.mps.empty_cache()
    signal=all(g['brier_below_01_count']/g['N']>=.1 for g in groups if g['group']=='hard' and g['correctness']=='incorrect')
    dest=ROOT/'output/analysis/Brier_Logit_Saturation_V10_Analyse'
    source_stamp={str(p.relative_to(ROOT)):base.digest(p) for p in [Path(__file__),ROOT/'output/analysis/Brier_Logit_Saturation_V10_Protocol.md']}
    base.save(dest.with_suffix('.json'),dict(device='mps',tests_passed=True,rows=rows,groups=groups,sources=sources,
        source_stamp=source_stamp,inference_only=True,oracle_only=True,test_evaluated=False,training=False,
        followup_parameter_gradient_diagnostic_supported=signal,global_validation=False))
    lines=['# V10 — saturation du gradient de loss au niveau des logits','',
        '**Audit MPS terminé, identités vérifiées par autograd.** Deux modèles ERM C=2 de v7, 96 000 exemples d’entraînement évalués au total. '
        'Aucun entraînement supplémentaire ni évaluation test.', '',
        '| Seed | Groupe | Exemples incorrects | Médiane p_y | Médiane ||g_Brier||/||g_CE|| | Part ≤ 0,1 | Part ≤ 0,01 | Part g_log borné > 2 g_Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for g in groups:
        if g['correctness']!='incorrect':continue
        lines.append(f"| {g['seed']} | {g['group']} | {g['N']} | {g['p_true']['median']:.4f} | {g['brier_over_ce']['median']:.4f} | {100*g['brier_below_01_count']/g['N']:.2f} % | {100*g['brier_below_001_count']/g['N']:.2f} % | {100*g['fraction_bounded_log_over_2brier']:.2f} % |")
    lines+=['','Critère descriptif pour approfondir la piste sur des gradients de paramètres : '+('atteint' if signal else 'non atteint')+'.', '',
        'Cette mesure décrit un effet possible au niveau de la loss. Le gradient des paramètres comporte aussi le Jacobien du réseau ; '
        'le clipping peut ensuite effacer l’amplification. La plus grande norme d’une autre loss ne prouve ni meilleure descente ni meilleure fairness.', '',
        'Les losses robustifiées, dont GCE, ne sont pas des inventions de cet audit. La transformation rationnelle est ici un contrôle explicite, '
        'sans revendication de nouveauté ni de calibration probabiliste stricte. Aucune loss n’est promue et aucun nouveau budget DP n’est déclaré par ces oracles.', '',
        '[Protocole et sources bibliographiques](Brier_Logit_Saturation_V10_Protocol.md). '
        '[Evidence par client et groupe](Brier_Logit_Saturation_V10_Analyse.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(device='mps',groups=[g for g in groups if g['correctness']=='incorrect'],followup_supported=signal),indent=2))


if __name__=='__main__':main()
