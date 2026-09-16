#!/usr/bin/env python3
"""Independent recomputation of the unchanged clean screen and privacy ledger."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_fair_direction_curvature_v8 import verify,close
from privacy.fair_objective import wor_rdp


def main():
    out=ROOT/'results/ldp_gradient_far/public_decay_private_risk_v9'
    manifest=json.loads((out/'manifest.json').read_text())
    m=manifest['config']
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    for p,sha in manifest['source_stamp'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,p
    records=[];checks=[];pairing=[]
    for seed in m['calibration_seeds']:
        for method in m['methods']:
            d=out/f'seed{seed}__{method}';r=json.loads((d/'metrics.json').read_text())
            status=json.loads((d/'orchestration_status.json').read_text())
            assert status['status']=='completed' and status['metrics_sha256']==hashlib.sha256((d/'metrics.json').read_bytes()).hexdigest()
            assert r['device']=='mps' and not r['test_evaluated'] and r['source_stamp']==manifest['source_stamp']
            verify(r['initial'])
            for step in r['rounds']:
                assert step['device']=='mps' and step['aggregation']['eta']==(2 if step['round']<=30 else .5)
                if step['validation'] is not None:verify(step['validation'])
            assert [x['round'] for x in r['rounds']]==list(range(1,61))
            p=r['privacy']
            if method.startswith('risk_'):
                eps=min(60*wor_rdp(a,.05,p['gradient_z'])+60*a/(2*p['risk_z']**2)+math.log(1e5)/(a-1) for a in range(2,65))
                close(p['gradient_std'],p['gradient_z']*4/240);close(p['risk_std'],p['risk_z']/4800)
            else:
                eps=min(60*wor_rdp(a,.05,p['z'])+math.log(1e5)/(a-1) for a in range(2,65))
                close(p['gradient_std'],p['z']*4/240)
            close(eps,p['epsilon_realized']);assert eps<=4
            v=r['final']['validation'];records.append(dict(seed=seed,method=method,validation=v,epsilon=eps,privacy=p))
            oldarm=method+'_C2'+('_scale0.5' if method.startswith('risk_') else '')
            olddir=ROOT/f'results/ldp_gradient_far/capped_private_risk_calibration_v7/seed{seed}__{oldarm}'
            old=json.loads((olddir/'metrics.json').read_text())
            assert r['splits']==old['splits']
            oracle=json.loads((d/'simulator_oracle.json').read_text())
            oldoracle=json.loads((olddir/'simulator_oracle.json').read_text())
            assert [[c['batch_hash'] for c in t['clients']] for t in oracle['rounds']]==[[c['batch_hash'] for c in t['clients']] for t in oldoracle['rounds']]
            delta_before=max(abs(a['validation']['accuracy_pct']-b['validation']['accuracy_pct']) for a,b in zip(r['rounds'][:30],old['rounds'][:30]) if a['validation'])
            pairing.append(dict(seed=seed,method=method,identical_splits_batches=True,maximum_pre_decay_accuracy_difference_pp=delta_before,
                                includes_solver_change=method.endswith('rfa')))
    for candidate in ('risk_mean','risk_rfa'):
        for seed in m['calibration_seeds']:
            a=next(r['validation'] for r in records if r['seed']==seed and r['method']==candidate)
            controls=[(kind,next(r['validation'] for r in records if r['seed']==seed and r['method']==kind)) for kind in m['new_controls']]
            for kind in m['historical_controls']:
                f=ROOT/f'results/ldp_gradient_far/capped_private_risk_calibration_v7/seed{seed}__{kind}/metrics.json'
                controls.append(('v7/'+kind,json.loads(f.read_text())['final']['validation']))
            for kind,b in controls:
                da=a['accuracy_pct']-b['accuracy_pct'];dw=a['worst20_pct']-b['worst20_pct']
                checks.append(dict(candidate=candidate,seed=seed,control=kind,accuracy_delta_pp=da,worst20_delta_pp=dw,
                                    passed=da>=-1 and dw>=1))
    assert len(checks)==24
    selected='risk_rfa' if all(c['passed'] for c in checks if c['candidate']=='risk_rfa') else None
    original=json.loads((out/'evidence.json').read_text())['decision']
    assert selected==original['selected_robust_candidate']
    for candidate in ('risk_mean','risk_rfa'):
        expect=original['gates'][candidate]['comparisons']
        got=[{k:v for k,v in r.items() if k!='candidate'} for r in checks if r['candidate']==candidate]
        assert got==expect
    dest=ROOT/'output/analysis/Public_Decay_Private_Risk_V9_Analyse'
    payload=dict(audit_passed=True,records=records,comparisons=checks,pairing=pairing,selected_robust_candidate=selected,
                 confirmatory=False,no_test_or_attack=True)
    dest.with_suffix('.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# V9 — effet du planning du pas : analyse complète','',
           '**8/8 runs MPS valides.** Comptes par classe, accuracy, Worst-20, variance, gap, budget composé et critères recomputés indépendamment.', '',
           '## Décision','',
           '**Aucune candidate ne passe le critère global fixé.** Il existe un gain de fairness face aux contrôles au même planning, '
           'mais le contrôle ERM C=2 à pas constant reste supérieur sur la première seed. Aucun test ni attaque ni confirmation indépendante n’a été lancé.', '',
           '## Protocole','',
           'Fashion-MNIST / LeNet-5 tanh, 10 clients équilibrés Dirichlet 0,1 ; 60 tours ; un batch de 240 exemples sans remise par client/tour ; '
           'C=2 ; aucune epoch locale ; η=2 aux tours 1–30 puis 0,5 aux tours 31–60. Risque privé plafonné à c=0,5. '
           'Seeds de calibration 170501/170502 ; seules les données de validation sont évaluées.', '',
           'La privacy sample-level côté client est calibrée à ε≤4, δ=10⁻⁵, replace-one. '
           'Les candidats composent les canaux risque/gradient ; ERM consacre tout le budget aux gradients. '
           'Les rapports oracle et les simulations appariées ne constituent pas un transcript conjoint couvert par cette garantie.', '',
           '## Résultats : moyenne ± écart-type des deux seeds','',
           'Ces statistiques sont descriptives sur des seeds de calibration réutilisées, pas des intervalles de confirmation.', '',
           '| Méthode | Accuracy (%) | Worst-20 (%) | Gap (pp) | Variance (pp²) | Brier |',
           '|:--|--:|--:|--:|--:|--:|']
    for method in m['methods']:
        rr=[r['validation'] for r in records if r['method']==method]
        vals=[]
        for key in ('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2','brier_loss'):
            numbers=[v[key] for v in rr];vals.append(f'{st.mean(numbers):.3f} ± {st.stdev(numbers):.3f}')
        lines.append('| '+method+' | '+' | '.join(vals)+' |')
    lines+=['','## Contrastes qui déterminent la décision','',
            '| Seed | Contrôle | Δ accuracy risque-RFA (pp) | Δ Worst-20 risque-RFA (pp) | Passe |',
            '|--:|:--|--:|--:|:--|']
    for c in checks:
        if c['candidate']=='risk_rfa':
            lines.append(f"| {c['seed']} | {c['control']} | {c['accuracy_delta_pp']:+.3f} | {c['worst20_delta_pp']:+.3f} | {'oui' if c['passed'] else 'non'} |")
    lines+=['','## Ce que cela nous apprend','',
            '1. **Composante fairness mesurée.** Au même planning, risque-RFA gagne 1,292 et 2,792 points de Worst-20 face à ERM moyen, '
            'avec 0,242 et 0,692 point d’accuracy perdu. Le gap et la variance diminuent. Ce résultat propre est cohérent avec une pondération des risques utile.',
            '2. **Performance globale insuffisante.** Face à ERM C=2 constant, la première seed perd 2,625 points d’accuracy et 3,792 points de Worst-20. '
            'On ne peut pas masquer ce contrôle au motif qu’il utilise un autre planning : il fait partie des contrôles forts préenregistrés.',
            '3. **Transfert partiel du diagnostic v8.** Réduire le pas stabilise une direction locale, mais réduire le pas dès le tour 31 peut aussi ralentir la progression. '
            'V8 ne permettait pas de fixer universellement le moment optimal de réduction. Cette explication est une inférence ; elle ne change pas le verdict.',
            '4. **Robustesse non évaluée ici.** Le solveur est numériquement amélioré et la masse de risque reste bornée ; aucune résistance du modèle à une attaque n’a encore été mesurée pour cette nouvelle campagne.',
            '5. **Pas de revendication de nouveauté sur le planning.** La décroissance du pas est un contrôle d’optimisation. Une contribution publiable devrait encore porter sur le mécanisme équitable privé robuste, avec baselines pertinentes et confirmation indépendante.', '',
            '## Appariement et réserves','',
            f"Écart maximal d’accuracy avant la réduction de pas par rapport au run historique correspondant : {max(p['maximum_pre_decay_accuracy_difference_pp'] for p in pairing):.8g} pp. "
            'Batches et partitions ont été vérifiés identiques. Les versions RFA incluent aussi la correction numérique du solveur : une attribution précise au seul planning doit conserver cette réserve.', '',
            'Les quatre seeds 170601–170604 restent réservées. Les échecs précédents sont conservés. '
            'La décision ne peut devenir positive en abaissant après coup le seuil de Worst-20 ou en retirant le contrôle défavorable.', '',
            '## Sources','',
            '- [Protocole avant exécution](Public_Decay_Private_Risk_V9_Protocol.md).',
            '- [Tableau par seed](Public_Decay_Private_Risk_V9_Status.md).',
            '- [Audit indépendant et toutes les différences](Public_Decay_Private_Risk_V9_Analyse.json).',
            '- [Diagnostic directionnel v8](Fair_Direction_Curvature_V8_Interpretation.md).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audit_passed=True,valid_runs=8,selected=selected,pairing=pairing),indent=2))


if __name__=='__main__':main()
