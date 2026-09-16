#!/usr/bin/env python3
"""Independent metric, privacy, source and comparison audit for horizon 120."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_fair_direction_curvature_v8 import verify,close
from privacy.fair_objective import wor_rdp


def label(key):
    # The shared v9 gate uses a legacy 'v7/' label for every historical entry.
    # Here we resolve the actual source, preserving the original evidence key.
    if key.startswith('v7/v7_'):return 'v7/'+key[len('v7/v7_'):]
    if key.startswith('v7/v9_'):return 'v9/'+key[len('v7/v9_'):]
    return 'v11/'+key


def main():
    out=ROOT/'results/ldp_gradient_far/public_horizon_private_risk_v11'
    status=json.loads((out/'status.json').read_text());assert status['status']=='completed'
    manifest=json.loads((out/'manifest.json').read_text());m=manifest['config']
    for path,sha in manifest['source_stamp'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    evidence=json.loads((out/'evidence.json').read_text());records=[]
    for j in [dict(seed=s,method=k) for s in m['calibration_seeds'] for k in m['methods']]:
        d=out/f"seed{j['seed']}__{j['method']}";file=d/'metrics.json';r=json.loads(file.read_text())
        state=json.loads((d/'orchestration_status.json').read_text())
        assert state['status']=='completed' and state['metrics_sha256']==hashlib.sha256(file.read_bytes()).hexdigest()
        assert r['job']==j and r['source_stamp']==manifest['source_stamp'] and r['device']=='mps' and not r['test_evaluated']
        verify(r['initial']);assert len(r['rounds'])==120
        for t in r['rounds']:
            assert t['device']=='mps' and t['aggregation']['eta']==(2. if t['round']<=60 else .5)
            if t['validation'] is not None:verify(t['validation'])
        assert [t['round'] for t in r['rounds']]==list(range(1,121))
        p=r['privacy']
        if j['method'].startswith('risk_'):
            eps=min(120*wor_rdp(a,.05,p['gradient_z'])+120*a/(2*p['risk_z']**2)+math.log(1e5)/(a-1) for a in range(2,65))
            close(p['gradient_std'],p['gradient_z']*4/240);close(p['risk_std'],p['risk_z']/4800)
            assert p['risk_releases']==120 and p['gradient_releases']==120
        else:
            eps=min(120*wor_rdp(a,.05,p['z'])+math.log(1e5)/(a-1) for a in range(2,65))
            close(p['gradient_std'],p['z']*4/240);assert p['steps']==120
        close(eps,p['epsilon_realized']);assert eps<=4
        records.append(dict(job=j,validation=r['final']['validation'],privacy=p))
    hist=evidence['historical_controls']
    for h in hist:
        file=Path(h['source']);r=json.loads(file.read_text());state=json.loads((file.parent/'orchestration_status.json').read_text())
        assert state['status']=='completed' and state['metrics_sha256']==hashlib.sha256(file.read_bytes()).hexdigest()
        assert r['final']==h['final'];verify(r['final']['validation'])
    recomputed=[]
    for candidate in ('risk_mean','risk_rfa'):
        for seed in m['calibration_seeds']:
            a=next(r['validation'] for r in records if r['job']==dict(seed=seed,method=candidate))
            controls=[(k,next(r['validation'] for r in records if r['job']==dict(seed=seed,method=k))) for k in m['new_controls']]
            controls += [('v7/'+k,next(r['final']['validation'] for r in hist if r['job']==dict(seed=seed,arm=k))) for k in m['historical_controls']]
            for key,b in controls:
                da=a['accuracy_pct']-b['accuracy_pct'];dw=a['worst20_pct']-b['worst20_pct']
                got=dict(seed=seed,control=key,accuracy_delta_pp=da,worst20_delta_pp=dw,passed=da>=-1 and dw>=1)
                assert got in evidence['decision']['gates'][candidate]['comparisons']
                recomputed.append(dict(candidate=candidate,original_control=key,control_label=label(key),**{k:v for k,v in got.items() if k!='control'}))
    assert len(recomputed)==32
    selected='risk_rfa' if all(c['passed'] for c in recomputed if c['candidate']=='risk_rfa') else None
    assert selected==evidence['decision']['selected_robust_candidate']==status['selected_robust_candidate']
    assert json.loads((out/'pairing_audit.json').read_text())['passed']
    payload=dict(audit_passed=True,valid_runs=8,comparisons=recomputed,records=records,selected_robust_candidate=selected,
        confirmatory=False,attack_evidence=False,test_evaluated=False,manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest())
    dest=ROOT/'output/analysis/Public_Horizon_Private_Risk_V11_Analyse'
    dest.with_suffix('.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# V11 — horizon 120, budget privé identique','',
        '**8/8 runs MPS terminés et vérifiés indépendamment.** Toutes les évaluations sont sur validation, sur deux seeds de calibration.', '',
        '## Décision','',
        ('**Risque-RFA passe les 16 comparaisons propres préenregistrées et devient admissible pour une confirmation indépendante.** '
         'Ce n’est pas encore une validation de la combinaison DP + fairness + robustesse : il reste les seeds réservées et les attaques.' if selected else
         '**Aucune candidate robuste admissible : au moins une des 16 comparaisons propres préenregistrées échoue.** '
         'Aucun résultat moyen ne remplace ces contrôles ; ni confirmation ni attaque lancée.'), '',
        '## Protocole et coût privé','',
        'Fashion-MNIST / LeNet-5 tanh, n=10, C=2, batch 240 sans remise, N=4800 ; un gradient privé par client et par tour, sans epoch locale. '
        'η=2 aux tours 1–60 puis 0,5 aux tours 61–120. Coefficients de risque a_i=1+2 min(R̃_i/0,5,1). '
        'Les mêmes modèles initiaux, partitions et Gaussiens standardisés sont utilisés entre conditions.', '',
        '| Canal/règle | Multiplicateur gradient z | Écart-type gradient | Écart-type rapport de risque | ε réalisé |',
        '|:--|--:|--:|--:|--:|']
    for method in ('erm_mean','risk_rfa'):
        p=next(r['privacy'] for r in records if r['job']['method']==method)
        lines.append(f"| {method} | {p.get('gradient_z',p.get('z')):.6f} | {p['gradient_std']:.6f} | {p['risk_std']:.6f} | {p['epsilon_realized']:.8f} |")
    lines+=['','δ=10⁻⁵. Les coûts des deux canaux sont composés pour 120 tours. '
        'La garantie est sample-level côté client, replace-one, par exécution idéale ; '
        'ni les oracles ni le transcript conjoint des simulations à bruit partagé ne sont certifiés par cet ε.', '',
        '## Moyenne ± écart-type, deux seeds','',
        '| Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
        '|:--|--:|--:|--:|--:|--:|']
    for method in m['methods']:
        rr=[r['validation'] for r in records if r['job']['method']==method]
        vals=[]
        for k in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','brier_loss'):
            vv=[v[k] for v in rr];vals.append(f'{st.mean(vv):.3f} ± {st.stdev(vv):.3f}')
        lines.append('| '+method+' | '+' | '.join(vals)+' |')
    lines+=['','## Comparaisons qui décident de l’admissibilité','',
        '| Seed | Contrôle exact | Δ accuracy risque-RFA (pp) | Δ Worst-20 risque-RFA (pp) | Passe |',
        '|--:|:--|--:|--:|:--|']
    for c in recomputed:
        if c['candidate']=='risk_rfa':lines.append(f"| {c['seed']} | {c['control_label']} | {c['accuracy_delta_pp']:+.3f} | {c['worst20_delta_pp']:+.3f} | {'oui' if c['passed'] else 'non'} |")
    lines+=['','Les anciens contrôles v7/v9 et les nouveaux contrôles T=120 sont tous conservés. '
        'L’evidence brute hérite d’un préfixe historique générique du calculateur v9 ; les noms ci-dessus sont résolus à partir des véritables chemins de fichiers, '
        'sans modifier les valeurs ni les décisions.', '',
        '## Limites','',
        'L’horizon plus long coûte plus de calcul et impose un bruit par publication supérieur pour maintenir ε. '
        'L’écart à v9 ne peut donc pas être attribué au seul nombre de tours à bruit constant. '
        'Les deux seeds ont servi à plusieurs calibrations : leur moyenne ± écart-type ne constitue pas une estimation confirmatoire indépendante.', '',
        'Une admissibilité propre n’est ni une preuve de nouveauté ni une mesure de robustesse sous attaques. '
        'Le critère final continue d’exiger une confirmation indépendante et des attaques à protocole figé. Aucun seuil n’est modifié après observation.', '',
        '## Sources','',
        '- [Protocole préenregistré](Public_Horizon_Private_Risk_V11_Protocol.md).',
        '- [Résultats par seed](Public_Horizon_Private_Risk_V11_Status.md).',
        '- [Audit indépendant et toutes les comparaisons](Public_Horizon_Private_Risk_V11_Analyse.json).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,valid_runs=8,selected=selected,matched_comparisons=[r for r in recomputed if r['candidate']=='risk_rfa' and r['control_label'].startswith('v11/')]),indent=2))


if __name__=='__main__':main()
