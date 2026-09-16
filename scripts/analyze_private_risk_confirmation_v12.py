#!/usr/bin/env python3
"""Independent audit of the fixed final test endpoint; scalar evidence only."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode = True
from privacy.fair_objective import wor_rdp
from scripts.analyze_fair_direction_curvature_v8 import verify as verify_core, close

OUT = ROOT/'results/ldp_gradient_far/private_risk_confirmation_v12'
DEST = ROOT/'output/analysis/Private_Risk_Confirmation_V12_Analyse'
METRICS = ('accuracy_pct','balanced_accuracy_pct','worst20_pct','gap_best20_worst20_pp',
           'gap_best_worst_pp','variance_pp2','ce_loss','brier_loss')
LABELS = ('Test accuracy (%)','Balanced accuracy (%)','Worst-20 (%)','Gap Best-20–Worst-20 (pp)',
          'Gap best–worst (pp)','Variance (pp²)','Loss CE','Loss demi-Brier')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(v):
    verify_core(v)
    close(v['ce_loss'],st.mean(c['ce_loss'] for c in v['clients']))
    close(v['client_accuracy_pct'],v['accuracy_pct'])
    acc = [c['accuracy'] for c in v['clients']]
    close(v['gap_best_worst_pp'],100*(max(acc)-min(acc)))
    balanced = []
    for c in v['clients']:
        assert all(int(x)==x for x in c['class_count']+c['class_hits'])
        present = [(h,n) for h,n in zip(c['class_hits'],c['class_count']) if n>0]
        b = st.mean(h/n for h,n in present)
        assert abs(c['balanced_accuracy']-b)<2e-7
        balanced.append(b)
    assert abs(v['balanced_accuracy_pct']-100*st.mean(balanced))<2e-5


def audit():
    state = json.loads((OUT/'status.json').read_text())
    assert state['status']=='completed' and state['valid_runs']==16
    manifest = json.loads((OUT/'manifest.json').read_text())
    m,profile,stamp = manifest['config'],manifest['profile'],manifest['source_stamp']
    for path,sha in stamp.items():
        assert digest(ROOT/path)==sha,path
    assert m['confirmation_seeds']==[170601,170602,170603,170604]
    evidence = json.loads((OUT/'evidence.json').read_text())
    records = []
    batches = {}
    initial = {}
    splits = {}
    for seed in m['confirmation_seeds']:
        for method in m['methods']:
            d = OUT/f'seed{seed}__{method}'
            r = json.loads((d/'metrics.json').read_text())
            s = json.loads((d/'orchestration_status.json').read_text())
            assert s['status']=='completed' and s['metrics_sha256']==digest(d/'metrics.json')
            assert s['oracle_sha256']==digest(d/'simulator_oracle.json')
            assert r['source_stamp']==stamp and r['job']==dict(seed=seed,method=method)
            assert r['device']=='mps' and r['test_evaluation_rounds']==[120]
            assert r['final']==r['rounds'][-1] and len(r['rounds'])==120
            assert [t['round'] for t in r['rounds']]==list(range(1,121))
            assert [t['round'] for t in r['rounds'] if t['test'] is not None]==[120]
            assert [t['round'] for t in r['rounds'] if t['validation'] is not None]==profile['evaluation_rounds']
            p = r['privacy']
            zg = p['gradient_z'] if method.startswith('risk_') else p['z']
            def epsilon_at(t):
                return min(t*wor_rdp(a,.05,zg)+(t*a/(2*p['risk_z']**2) if method.startswith('risk_') else 0)
                           +math.log(1e5)/(a-1) for a in range(2,65))
            close(p['gradient_std'],zg*4/240)
            close(epsilon_at(120),p['epsilon_realized'])
            assert p['epsilon_realized']<=4 and p['delta']==1e-5
            if method.startswith('risk_'):
                close(p['risk_std'],p['risk_z']/4800)
                assert p['risk_releases']==p['gradient_releases']==120
                for a in range(2,65):
                    close(p['rdp'][str(a)],120*wor_rdp(a,.05,zg)+120*a/(2*p['risk_z']**2))
            else:
                assert p['risk_releases']==0 and p['steps']==120 and p['risk_std']==0
            verify(r['initial'])
            for t in r['rounds']:
                assert t['device']=='mps' and t['aggregation']['eta']==(2. if t['round']<=60 else .5)
                close(t['epsilon_realized'],epsilon_at(t['round']))
                safety = t['aggregation']['message_safety']
                assert safety['invalid_message_rows']==safety['nonfinite_risk_reports']==0
                w = t['aggregation']['objective_weights']
                close(sum(w),1.)
                assert len(w)==10 and all(0<w_i<=.25+2e-6 for w_i in w)
                if t['validation'] is not None:
                    verify(t['validation'])
            verify(r['final']['test'])
            oracle = json.loads((d/'simulator_oracle.json').read_text())
            assert not oracle['privacy_protected'] and not oracle['feeds_mechanism']
            pattern = [[c['batch_hash'] for c in t['clients']] for t in oracle['rounds']]
            assert len(pattern)==120 and all(len(q)==10 for q in pattern)
            if seed in batches:
                assert batches[seed]==pattern and initial[seed]==r['initial'] and splits[seed]==r['splits']
            else:
                batches[seed],initial[seed],splits[seed] = pattern,r['initial'],r['splits']
            records.append(dict(job=r['job'],test=r['final']['test'],validation=r['final']['validation'],privacy=p,
                metrics_sha256=digest(d/'metrics.json'),source=str(d/'metrics.json'),
                elapsed_seconds=r['elapsed_seconds']))
    assert json.loads((OUT/'pairing_audit.json').read_text())['passed']
    assert json.loads((OUT/'tests.json').read_text())['passed']
    index = {(r['job']['seed'],r['job']['method']):r for r in records}
    comparisons = {}
    for control in [*m['primary_controls'],'risk_mean']:
        raw = {k:[] for k in METRICS}
        per_seed = []
        for seed in m['confirmation_seeds']:
            a,b = index[seed,'risk_rfa']['test'],index[seed,control]['test']
            delta = {k:a[k]-b[k] for k in METRICS}
            for k in METRICS:
                raw[k].append(delta[k])
            per_seed.append(dict(seed=seed,delta=delta,passed=delta['accuracy_pct']>=-1 and delta['worst20_pct']>=1))
        summary = {}
        for k,values in raw.items():
            mean,sd = st.mean(values),st.stdev(values)
            radius = 3.182446305284263*sd/math.sqrt(4)
            summary[k] = dict(mean=mean,sd=sd,ci95=[mean-radius,mean+radius])
        gates = dict(all_seed_gates=all(p['passed'] for p in per_seed),
            worst20_ci_lower_positive=summary['worst20_pct']['ci95'][0]>0,
            accuracy_ci_lower_noninferior=summary['accuracy_pct']['ci95'][0]>=-1,
            mean_gap_nonincreasing=summary['gap_best20_worst20_pp']['mean']<=0,
            mean_variance_nonincreasing=summary['variance_pp2']['mean']<=0)
        old = evidence['decision']['comparisons'][control]
        assert gates==old['gates']
        for k,s in old['summaries'].items():
            close(summary[k]['mean'],s['mean'])
            close(summary[k]['sd'],s['sd'])
            for u,v in zip(summary[k]['ci95'],s['ci95']):
                close(u,v)
        comparisons[control] = dict(primary=control in m['primary_controls'],per_seed=per_seed,
                                   summaries=summary,gates=gates,passed=all(gates.values()))
    passed = all(comparisons[c]['passed'] for c in m['primary_controls'])
    assert passed==state['clean_confirmation_passed']==evidence['decision']['clean_confirmation_passed']
    return dict(audit_passed=True,clean_confirmation_passed=passed,valid_runs=16,records=records,
        comparisons=comparisons,manifest_sha256=digest(OUT/'manifest.json'),
        analysis_source_sha256=digest(__file__),joint_objective_validated=False,attacks_evaluated=False)


def write(result):
    DEST.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    verdict = 'PASS' if result['clean_confirmation_passed'] else 'FAIL'
    lines = ['# Risque privé + RFA : confirmation indépendante V12','',
        f'**16/16 runs MPS terminés et audités. Gate propre : {verdict}.**','',
        'La décision porte sur quatre seeds réservées, avec test au tour 120 et paramètres figés avant ces runs. '
        'Elle ne valide pas encore la robustesse sous attaques, ni une revendication de nouveauté.', '',
        '## 1. Résultats par seed','',
        '| Seed | Règle | Accuracy (%) | Balanced acc. (%) | Worst-20 (%) | Gap B20–W20 (pp) | Variance (pp²) | CE | Demi-Brier |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|--:|']
    selected = [k for k in METRICS if k!='gap_best_worst_pp']
    for r in result['records']:
        v,j = r['test'],r['job']
        lines.append(f"| {j['seed']} | {j['method']} | "+' | '.join(f'{v[k]:.4f}' for k in selected)+' |')
    lines += ['', '## 2. Moyenne ± écart-type échantillonnal','',
        '| Métrique | ERM moyenne | ERM RFA | Risque moyenne | Risque RFA |',
        '|:--|--:|--:|--:|--:|']
    for k,label in zip(METRICS,LABELS):
        cells = []
        for method in ('erm_mean','erm_rfa','risk_mean','risk_rfa'):
            values = [r['test'][k] for r in result['records'] if r['job']['method']==method]
            cells.append(f'{st.mean(values):.4f} ± {st.stdev(values):.4f}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    for control,c in result['comparisons'].items():
        lines += ['',f'## 3. Comparaison appariée à {control}', '',
            ('Comparateur primaire.' if c['primary'] else 'Contrôle de mécanisme, non substituable aux comparateurs primaires.'), '',
            '| Seed | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ gap B20–W20 (pp) | Δ variance (pp²) |',
            '|--:|--:|--:|--:|--:|']
        for p in c['per_seed']:
            d = p['delta']
            lines.append(f"| {p['seed']} | {d['accuracy_pct']:+.4f} | {d['worst20_pct']:+.4f} | {d['gap_best20_worst20_pp']:+.4f} | {d['variance_pp2']:+.4f} |")
        lines += ['', '| Différence risque-RFA − contrôle | Moyenne | IC95 t apparié |', '|:--|--:|:--|']
        for k,label in zip(METRICS,LABELS):
            s = c['summaries'][k]
            lines.append(f"| {label} | {s['mean']:+.4f} | [{s['ci95'][0]:+.4f} ; {s['ci95'][1]:+.4f}] |")
        lines += ['', '| Critère préenregistré | Passe |','|:--|:--|']
        translations = dict(all_seed_gates='Sur chaque seed : ΔW20 ≥ 1 pp et Δacc ≥ −1 pp',
            worst20_ci_lower_positive='Borne basse IC95 du gain W20 > 0',
            accuracy_ci_lower_noninferior='Borne basse IC95 de Δacc ≥ −1 pp',
            mean_gap_nonincreasing='Variation moyenne du gap ≤ 0',
            mean_variance_nonincreasing='Variation moyenne de la variance ≤ 0')
        for k,p in c['gates'].items():
            lines.append(f"| {translations[k]} | {'oui' if p else 'non'} |")
    lines += ['', '## 4. Privacy et limites','',
        'ε ≤ 4, δ = 10⁻⁵, sensibilité replace-one 2C/B = 4/240 pour le gradient, 1/4800 pour le rapport. '
        'Les 120 rapports et 120 gradients sont composés en RDP ; ERM n’envoie aucun rapport et utilise tout son budget pour le gradient. '
        'Le modèle emploie un batch fixe sans remise par tour, sans epoch locale, et le même planning η=2 puis 0,5. '
        'La politique numérique des messages est restée inactive sur tous les messages honnêtes.', '',
        'La garantie est sample-level, côté client, pour le mécanisme idéal par exécution : pas client-level, '
        'pas pour le transcript conjoint des quatre règles à bruit partagé, ni pour les métriques de test/oracles de ce benchmark public. '
        'Les IC t utilisent quatre différences indépendantes et trois degrés de liberté. Les hypothèses de ces IC restent approximatives avec seulement quatre seeds ; '
        'les intervalles présentés sont marginaux, pas des bandes simultanées sur toutes les métriques.', '',
        'Une baisse de loss ou de variance ne remplace pas un échec du critère primaire. '
        'Un résultat propre favorable ne prouve pas qu’un risque falsifié ne pourra pas favoriser un attaquant. '
        'Aucune attaque n’est lancée automatiquement par ce programme.', '',
        '## Sources vérifiées','',
        '- [Protocole figé](Private_Risk_Confirmation_V12_Protocol.md).',
        '- [Calculs et audit complet](Private_Risk_Confirmation_V12_Analyse.json).',
        '- [Résultats de calibration antérieurs](Public_Horizon_Private_Risk_V11_Analyse.md).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,valid_runs=16,clean_confirmation_passed=result['clean_confirmation_passed'],
        comparison_gates={k:c['gates'] for k,c in result['comparisons'].items()},report=str(DEST.with_suffix('.md'))),indent=2))


if __name__=='__main__':
    write(audit())
