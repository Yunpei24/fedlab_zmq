#!/usr/bin/env python3
"""Offline scalar oracle audit: private risk weights versus noiseless risk weights.

No model, threshold, coefficient or run is selected by this diagnostic. All
rounds and all sixteen risk arms are retained after the complete V25 audit.
Public descriptive arithmetic only; no tensor training or CPU fallback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/ldp_gradient_far/public_temporal_noise_confirmation_v25'
AUDIT = ROOT/'output/analysis/Public_Temporal_Noise_Confirmation_V25_Analyse.json'
DEST = ROOT/'output/analysis/Public_Temporal_Noise_Confirmation_V25_Weight_Diagnostic'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(risks):
    a = [1+2*min(max(r,0.)/.5,1.) for r in risks]
    return [x/sum(a) for x in a]


def pearson(x,y):
    mx,my = st.mean(x),st.mean(y)
    numerator = sum((a-mx)*(b-my) for a,b in zip(x,y))
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return None if denominator <= 1e-14 else numerator/denominator


def top_two_membership(risks):
    """Fractional membership at ties avoids selecting arbitrary tied client IDs."""
    cut = sorted(risks,reverse=True)[1]
    above = sum(r>cut for r in risks); tied = sum(r==cut for r in risks)
    return [1. if r>cut else (2-above)/tied if r==cut else 0. for r in risks]


def round_diagnostic(raw_risks, private_risks, applied_weights, solver=None):
    assert len(raw_risks) == len(private_risks) == len(applied_weights) == 10
    assert all(math.isfinite(r) and 0<=r<=1 for r in raw_risks+private_risks)
    ideal, actual = weights(raw_risks), weights(private_risks)
    assert all(math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-6) for a,b in zip(actual,applied_weights))
    center = st.mean(raw_risks); high = top_two_membership(raw_risks)
    ideal_tilt = sum(w*(r-center) for w,r in zip(ideal,raw_risks))
    assert ideal_tilt >= -1e-14, 'Monotone noiseless coefficients must not invert mean risk targeting'
    result = dict(
        risk_standard_deviation_clients=st.pstdev(raw_risks),
        weight_total_variation_noise=.5*sum(abs(a-b) for a,b in zip(actual,ideal)),
        weight_total_variation_ideal_vs_uniform=.5*sum(abs(a-.1) for a in ideal),
        weight_concentration_ideal=10*sum(w*w for w in ideal),
        weight_concentration_private=10*sum(w*w for w in actual),
        correlation_private_weight_true_risk=pearson(actual,raw_risks),
        risk_targeting_ideal=ideal_tilt,
        risk_targeting_private=sum(w*(r-center) for w,r in zip(actual,raw_risks)),
        top_two_true_risk_mass_ideal=sum(w*h for w,h in zip(ideal,high)),
        top_two_true_risk_mass_private=sum(w*h for w,h in zip(actual,high)),
        report_projection_boundary_fraction=sum(r in (0.,1.) for r in private_risks)/10,
        coefficient_upper_saturation_fraction=sum(r>=.5 for r in private_risks)/10,
    )
    if solver is not None:
        nu = solver['stationary_weights']
        assert len(nu)==10 and all(math.isfinite(v) and v>0 for v in nu)
        assert math.isclose(sum(nu),1.,rel_tol=2e-6,abs_tol=2e-6)
        # These reconstruct a fixed-point candidate, not necessarily the exact
        # finite-iteration output; always retain its reconstruction error.
        result.update(
            stationary_reconstruction_error=solver['stationary_reconstruction_error'],
            effective_weight_total_variation_vs_private=.5*sum(abs(a-b) for a,b in zip(nu,actual)),
            effective_weight_concentration=10*sum(w*w for w in nu),
            effective_risk_targeting=sum(w*(r-center) for w,r in zip(nu,raw_risks)),
            effective_top_two_true_risk_mass=sum(w*h for w,h in zip(nu,high)),
        )
    return result


def summarize(rows):
    keys = set().union(*(r.keys() for r in rows))-{'round'}
    result = {}
    for k in sorted(keys):
        values = [r[k] for r in rows if r.get(k) is not None]
        result[k] = dict(median=st.median(values) if values else None,
                         min=min(values) if values else None,max=max(values) if values else None,
                         defined_rounds=len(values))
    result['negative_true_risk_targeting_round_fraction'] = sum(r['risk_targeting_private'] < -1e-12 for r in rows)/len(rows)
    # Diagnostic only: no assertion that a large ratio causes worse learning.
    result['round_fraction_noise_tv_exceeds_ideal_tilt_tv'] = sum(
        r['weight_total_variation_noise'] > r['weight_total_variation_ideal_vs_uniform'] for r in rows)/len(rows)
    return result


def analyze(audit_path=AUDIT,out=OUT,dest=DEST,*,label='V25',expected_runs=32):
    audited = json.loads(audit_path.read_text())
    assert audited['audit_passed'] and audited['runs']==expected_runs and not audited.get('snapshot_partial',False)
    details = []
    for verified in audited['details']:
        job = verified['job']
        if not job['method'].startswith('risk_'):
            continue
        directory = out/f'seed{job["seed"]}__k{job["grid_index"]}__{job["method"]}'
        assert digest(directory/'metrics.json') == verified['metrics_sha256']
        status = json.loads((directory/'orchestration_status.json').read_text())
        assert status['status']=='completed' and status['oracle_sha256']==digest(directory/'simulator_oracle.json')
        metrics = json.loads((directory/'metrics.json').read_text())
        oracle = json.loads((directory/'simulator_oracle.json').read_text())
        assert not oracle['feeds_mechanism'] and not oracle['privacy_protected']
        rows=[]
        for t,raw in zip(metrics['rounds'],oracle['rounds']):
            assert t['round']==raw['round']
            values = round_diagnostic([c['raw_risk'] for c in raw['clients']],
                [c['private_risk'] for c in raw['clients']],t['aggregation']['objective_weights'],
                t['aggregation']['solver'])
            rows.append(dict(round=t['round'],**values))
        assert [r['round'] for r in rows] == list(range(1,121))
        details.append(dict(job=job,risk_report_noise_std=metrics['privacy']['risk_std'],rounds=rows,
            early=summarize(rows[:60]),late=summarize(rows[60:]),
            metrics_sha256=verified['metrics_sha256'],oracle_sha256=status['oracle_sha256']))
    assert len(details)==expected_runs//2
    result=dict(descriptive_diagnostic=True,private_release=False,feeds_mechanism=False,
        candidate_selected=False,campaign=label,source_sha256=digest(Path(__file__)),audit_sha256=digest(audit_path),
        scope='Scalar oracle diagnostics on all risk arms/all rounds, no new training; no causal model verdict',details=details)
    dest.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=[f'# {label} — le bruit du rapport privé perturbe-t-il les poids de risque ?', '',
        f'Diagnostic descriptif hors mécanisme, calculé après audit des {expected_runs} runs. Il ne sélectionne ni coefficient, ni filtre, ni autre candidate. Les rapports bruts d’entraînement sont des oracles de benchmark, pas des releases DP.', '',
        '## Ce qui est comparé', '',
        '- Poids oracle : application de la même fonction 1 + 2 min(R/0,5 ; 1), puis normalisation, au risque pré-tour non bruité R.',
        '- Poids privés : application de cette fonction au rapport effectivement privatisé puis projeté dans [0,1].',
        '- TV bruit = ½ Σ |λ privée − λ oracle| : masse totale de pondération déplacée par la privatisation du rapport.',
        '- TV signal = ½ Σ |λ oracle − 0,1| : ampleur du changement que voudrait produire le risque non bruité face à la moyenne uniforme.',
        '- Ciblage du risque = Σ λᵢ(Rᵢ − moyenne(R)) : positif signifie que la règle privilégie les clients à plus fort risque d’entraînement. Négatif signifie une inversion de ce ciblage. Ce n’est pas une erreur d’agrégat ni une variation d’accuracy.',
        '- Top-2 : masse des deux clients à plus fort risque d’entraînement ; 0,20 est la masse uniforme. Ce ne sont pas forcément les deux moins bonnes accuracies de test. Les égalités sont réparties fractionnellement.', '',
        '## Tous les runs et les deux phases', '',
        'Médianes temporelles sur 1–60 ou 61–120. Les fractions portent sur 60 tours ; ces tours corrélés ne sont pas des réplications statistiques indépendantes. Aucune omission de phase initiale.', '',
        '| Seed | k | Règle | Tours | TV bruit médiane | TV signal médiane | Corr. poids–risque médiane | Masse top-2 médiane | Ciblage inversé (%) | Tours TV bruit > TV signal (%) |',
        '|--:|--:|:--|:--|--:|--:|--:|--:|--:|--:|']
    for d in details:
        j=d['job']
        for phase,label in (('early','1–60'),('late','61–120')):
            v=d[phase]
            def med(k):
                value=v[k]['median']
                return 'non défini' if value is None else f'{value:.5f}'
            cells=[str(j['seed']),str(j['grid_index']),j['method'],label,
                med('weight_total_variation_noise'),med('weight_total_variation_ideal_vs_uniform'),
                med('correlation_private_weight_true_risk'),med('top_two_true_risk_mass_private'),
                f'{100*v["negative_true_risk_targeting_round_fraction"]:.2f}',
                f'{100*v["round_fraction_noise_tv_exceeds_ideal_tilt_tv"]:.2f}']
            lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Rôle de RFA', '',
        'Le JSON conserve aussi les poids stationnaires ν de la médiane, leur distance aux λ d’objectif et leur erreur de reconstruction. Ils ne sont pas confondus : ν dépend de la géométrie des messages privés, tandis que λ dépend des rapports privés de risque. Avec un solveur fini, ν ne représente l’agrégat obtenu qu’à son erreur de reconstruction près.', '',
        '## Interprétation autorisée et limites', '',
        'Un déplacement TV bruit du même ordre que TV signal indique que le canal de risque peut brouiller le ciblage désiré. Cela ne démontre pas que cette variabilité cause une baisse d’accuracy, ni qu’un lissage améliorera le modèle : le retard et le biais d’un tel lissage devraient être payés et testés séparément.', '',
        'Les risques d’entraînement et les positions des clients ne sont pas fixes au cours du temps. Les oracles de différentes règles proviennent de trajectoires différentes. Leur différence ne constitue pas un contrefactuel à messages ou modèles identiques. Le bruit du gradient et la géométrie de RFA ne sont pas isolés par cette seule analyse du canal risque.', '',
        ('Le verdict V24 porte sur la validation de calibration au tour final ; le test final indépendant appartient à V25. '
         if expected_runs==16 else 'Le verdict V25 porte sur le test final indépendant et les critères préenregistrés. ')
        +'Une bonne corrélation poids–risque ne valide pas la fairness du modèle ; une mauvaise corrélation ne suffit pas à condamner toutes les pondérations privées.', '',
        f'[Audit de la campagne et verdict]({audit_path.with_suffix(".md").name}).']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(audited_risk_runs=len(details),diagnostic_only=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration-v24',action='store_true',help='Apply the same descriptive checks to the already audited calibration, not the independent confirmation')
    args=parser.parse_args()
    if args.calibration_v24:
        analyze(ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Analyse.json',
            ROOT/'results/ldp_gradient_far/public_temporal_noise_training_v24',
            ROOT/'output/analysis/Public_Temporal_Noise_Training_V24_Weight_Diagnostic',
            label='V24 (calibration, pas confirmation indépendante)',expected_runs=16)
    else:
        analyze()
