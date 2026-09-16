#!/usr/bin/env python3
"""Independent host-scalar audit of recorded evaluations; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ldp_gradient_far/fair_direction_curvature_v8'
DEST=ROOT/'output/analysis/Fair_Direction_Curvature_V8_Interpretation'


def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6),(a,b)


def phi(r,c=.5):return r+r*r/c if r<=c else 3*r-c


def verify(v):
    a=[]
    for c in v['clients']:
        close(sum(c['class_count']),c['N'])
        close(sum(c['class_hits'])/c['N'],c['accuracy'])
        assert all(0<=hit<=count for hit,count in zip(c['class_hits'],c['class_count']))
        a.append(c['accuracy'])
    close(100*st.mean(a),v['accuracy_pct'])
    tail=max(1,math.ceil(.2*len(a)));a.sort()
    close(100*st.mean(a[:tail]),v['worst20_pct'])
    close(100*(st.mean(a[-tail:])-st.mean(a[:tail])),v['gap_best20_worst20_pp'])
    close(10000*st.pvariance(a),v['variance_pp2'])
    close(st.mean(c['brier_loss'] for c in v['clients']),v['brier_loss'])


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p,sha in manifest['source_stamp'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
    assert json.loads((OUT/'status.json').read_text())['status']=='completed'
    blocks=[json.loads(p.read_text()) for p in sorted(OUT.glob('seed*_replay*.json'))]
    assert len(blocks)==8 and all(b['device']=='mps' and b['oracle_only'] and not b['test_evaluated'] for b in blocks)
    details=[]
    for b in blocks:
        v=b['before'];verify(v)
        hard=b['hard_clients_fixed']
        assert hard==sorted(range(10),key=lambda i:v['clients'][i]['accuracy'])[:2]
        for e in b['evaluations']:
            a=e['after'];verify(a)
            gain=st.mean(phi(c['brier_loss'])-phi(d['brier_loss']) for c,d in zip(v['clients'],a['clients']))
            close(gain,e['actual']['Jc_gain'])
            close(e['predicted_Jc_gain']-gain,e['actual']['Jc_taylor_remainder'])
            for metric,key in [('accuracy_pct','accuracy_gain_pp'),('worst20_pct','worst20_gain_pp'),('gap_best20_worst20_pp','gap_change_pp'),('variance_pp2','variance_change_pp2')]:
                close(a[metric]-v[metric],e['actual'][key])
            hard_loss=st.mean(v['clients'][i]['brier_loss']-a['clients'][i]['brier_loss'] for i in hard)
            close(hard_loss,e['actual']['fixed_hard_loss_gain'])
            newhard=sorted(range(10),key=lambda i:a['clients'][i]['accuracy'])[:2]
            details.append(dict(seed=b['seed'],replay=b['replay'],condition=e['condition'],eta=e['eta'],
                actual=e['actual'],predicted_Jc=e['predicted_Jc_gain'],hard_before=hard,hard_after=newhard,
                hard_changed=set(hard)!=set(newhard)))
    assert len(details)==192
    grouped=[]
    for seed in (170501,170502):
        for condition in manifest['config']['conditions']:
            for eta in (.5,1.,2.):
                q=[d for d in details if d['seed']==seed and d['condition']==condition and d['eta']==eta]
                assert len(q)==4
                grouped.append(dict(seed=seed,condition=condition,eta=eta,
                    actual={k:st.mean(d['actual'][k] for d in q) for k in ('Jc_gain','Jc_taylor_remainder','accuracy_gain_pp','worst20_gain_pp','fixed_hard_loss_gain')},
                    predicted_Jc=st.mean(d['predicted_Jc'] for d in q),hard_changed_count=sum(d['hard_changed'] for d in q),
                    curvature_proxy=st.mean(d['actual']['Jc_taylor_remainder']/eta**2 for d in q)))
    def row(seed,condition,eta):return next(r for r in grouped if r['seed']==seed and r['condition']==condition and r['eta']==eta)
    contrasts=[]
    for seed in (170501,170502):
        for eta in (.5,1.,2.):
            a,b=row(seed,'dp_risk_rfa',eta),row(seed,'dp_mean_full',eta)
            contrasts.append(dict(seed=seed,eta=eta,advantage={k:a['actual'][k]-b['actual'][k] for k in a['actual']}))
    payload=dict(audit_passed=True,model_evaluations_recomputed=192,independent_seeds=2,replays_per_seed=4,
        grouped=grouped,contrasts=contrasts,details=details,global_validation=False,
        max_legacy_stable_rfa_distance=max(b['legacy_solver_direction_distance'] for b in blocks),
        manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
    DEST.with_suffix('.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# V8 — ce que le diagnostic explique, et ce qu’il ne valide pas','',
        '**192 évaluations vérifiées indépendamment à partir des comptes par classe et des losses enregistrées.** '
        'Deux seeds, quatre tirages par seed ; validation seulement, sans entraînement cumulatif.', '',
        '## 1. Le pas change le signe du bénéfice, sans changer la direction', '',
        'Pour la RFA privée pondérée, toutes les conditions ci-dessous partent du même checkpoint ERM C=2 par seed. '
        'Un gain Jc positif est une baisse du risque équitable ; Δ accuracy/Worst-20 sont des variations par rapport à ce checkpoint.', '',
        '| Seed | η | Gain Jc prédit | Gain Jc réel | Δ accuracy (pp) | Δ Worst-20 (pp) | Reste / η² |',
        '|--:|--:|--:|--:|--:|--:|--:|']
    for seed in (170501,170502):
        for eta in (.5,1.,2.):
            r=row(seed,'dp_risk_rfa',eta);a=r['actual']
            lines.append(f"| {seed} | {eta:g} | {r['predicted_Jc']:+.6f} | {a['Jc_gain']:+.6f} | {a['accuracy_gain_pp']:+.3f} | {a['worst20_gain_pp']:+.3f} | {r['curvature_proxy']:.6f} |")
    lines+=['','Le reste de Taylor augmente approximativement comme η² sur ces pas. '
            'Il est assez grand pour renverser l’amélioration prédite sur la seed 170501 à η=2. '
            'Sur la seed 170502, Jc reste légèrement amélioré mais l’accuracy baisse. '
            'C’est un résultat local sur le pas appliqué, pas une démonstration que le bruit seul explique l’échec global.', '',
            '## 2. La comparaison pertinente garde le contrôle au même pas','',
            'Avantage de la RFA équitable privée sur ERM privé utilisant tout le budget gradient :', '',
            '| Seed | η | Avantage Jc | Avantage accuracy (pp) | Avantage Worst-20 (pp) |',
            '|--:|--:|--:|--:|--:|']
    for r in contrasts:
        a=r['advantage']
        lines.append(f"| {r['seed']} | {r['eta']:g} | {a['Jc_gain']:+.6f} | {a['accuracy_gain_pp']:+.3f} | {a['worst20_gain_pp']:+.3f} |")
    lines+=['','À η=0,5 ou 1, le gain de loss équitable et le gain de Worst-20 sont positifs en moyenne sur chacune des deux seeds, '
            'mais leur amplitude n’est pas un résultat end-to-end. À η=2, l’accuracy relative est moins bonne sur les deux seeds. '
            'On ne compare donc pas une candidate à petit pas avec seulement un contrôle à grand pas.', '',
            '## 3. Clipping et bruit : effets distincts','',
            'Sans clipping, η=2 dégrade fortement les deux objectifs, y compris sans bruit. '
            'Le clipping C=2 amortit ces directions. L’ajout du bruit peut ensuite réduire le gain, mais n’est pas la source unique du problème. '
            'La pondération des risques renforce la descente prédite, et augmente aussi le coût de courbure : ces deux effets doivent être contrôlés ensemble.', '',
            'Les deux clients initialement les moins précis ne restent pas nécessairement les deux moins précis après le pas. '
            'Une baisse de leur loss Brier peut donc coexister avec une baisse du Worst-20 recalculé, en plus du fait que loss et accuracy sont des métriques différentes. '
            'Les identités avant/après sont conservées dans l’evidence.', '',
            '## 4. RFA numérique et hypothèse restante','',
            f"L’écart maximal ancien/nouveau solveur sur ces messages est {payload['max_legacy_stable_rfa_distance']:.8g}. "
            'L’échec propre de v7 ne peut donc pas être attribué ici à l’instabilité numérique observée face aux messages immenses. '
            'Cette dernière reste un problème réel pour une future campagne byzantine.', '',
            'La seule expérience nouvelle motivée par ce diagnostic est une calibration finie avec planning public du pas, identique pour les contrôles et candidats. '
            'Elle repart de l’initialisation et conserve le budget complet. Elle doit encore battre les contrôles forts et appariés ; '
            'un passage resterait à confirmer sur les seeds réservées et sous attaques.', '',
            '## Sources','',
            '- [Protocole et limites avant exécution](Fair_Direction_Curvature_V8_Protocol.md).',
            '- [Tableau complet](Fair_Direction_Curvature_V8_Analyse.md).',
            '- [Recalcul indépendant](Fair_Direction_Curvature_V8_Interpretation.json).',
            '- [Protocole v9](Public_Decay_Private_Risk_V9_Protocol.md).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,evaluations=192,contrasts=contrasts),indent=2))


if __name__=='__main__':main()
