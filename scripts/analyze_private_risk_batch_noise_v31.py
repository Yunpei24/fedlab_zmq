"""Read-only scalar/count audit and descriptive paired V31 contrasts."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/analysis'
INPUT=OUT/'Private_Risk_V31_Local_Batch_Noise_Diagnostic.json'
SOURCE=ROOT/'results/ldp_gradient_far/private_risk_batch_noise_diagnostic_v31'
DEST=OUT/'Private_Risk_V31_Local_Batch_Noise_Contrasts'
SEEDS=(170501,170502)
BATCHES=(4800,240)
NOISES=('zero_oracle','full_plan','small_plan')
AGGS=('mean','rfa')
KEYS=('accuracy_pct','worst20_pct','gap_best20_worst20_pp','variance_pp2',
      'balanced_accuracy_pct','ce_loss','brier_loss')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b,tol=1e-8):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>tol*max(1,abs(a),abs(b)):
        raise ValueError(f'Inconsistent values {a} / {b}')


def check_row(r):
    ev=r['metrics'];clients=ev['clients'];matrix=r['confusion']
    if len(clients)!=10 or len(matrix)!=10 or any(len(line)!=10 for line in matrix):
        raise ValueError('Wrong population')
    if any(type(x)!=int or x<0 for line in matrix for x in line):
        raise ValueError('Invalid confusion counts')
    acc=[];balanced=[]
    for c in clients:
        n,h=c['class_count'],c['class_hits']
        if len(n)!=10 or len(h)!=10 or c['N']!=1200 or sum(n)!=1200:
            raise ValueError('Wrong client population')
        if any(not math.isfinite(x) or x<0 or x!=int(x) for x in n+h) or any(y>x for x,y in zip(n,h)):
            raise ValueError('Invalid client counts')
        acc.append(100*sum(h)/1200)
        balanced.append(st.mean(100*y/x for x,y in zip(n,h) if x))
    s=sorted(acc)
    for k,v in dict(accuracy_pct=st.mean(acc),worst20_pct=st.mean(s[:2]),
            gap_best20_worst20_pp=st.mean(s[-2:])-st.mean(s[:2]),
            variance_pp2=st.pvariance(acc),balanced_accuracy_pct=st.mean(balanced)).items():
        close(v,ev[k],tol=2e-6 if k=='balanced_accuracy_pct' else 1e-8)
    for k in range(10):
        assert sum(matrix[k])==sum(c['class_count'][k] for c in clients)
        assert matrix[k][k]==sum(c['class_hits'][k] for c in clients)
    d=r['decomposition']
    close(d['error_squared'],d['clean_error_squared']+d['noise_displacement_squared']+d['cross_term'],3e-6)
    if r['noise']=='zero_oracle':
        assert r['std']==0
        close(d['noise_displacement_squared'],0,1e-12)
        close(d['cross_term'],0,1e-12)


def summarize(values):
    if len(values)!=4:raise ValueError('Four paired repetitions required')
    return dict(values=values,mean=st.mean(values),sd=st.stdev(values))


def paired(index,seed,left,right,label):
    pairs=[]
    for repeat in range(4):
        a,b=index[(seed,repeat,*left)],index[(seed,repeat,*right)]
        d={k:a['metrics'][k]-b['metrics'][k] for k in KEYS}
        d.update(error_squared=a['decomposition']['error_squared']-b['decomposition']['error_squared'],
                 projected_gain=a['projected_objective_gain']-b['projected_objective_gain'])
        cm=[]
        for truth in range(10):
            n=sum(a['confusion'][truth]);assert n==sum(b['confusion'][truth])
            cm.append([100*(x-y)/n if n else None for x,y in zip(a['confusion'][truth],b['confusion'][truth])])
        pairs.append(dict(repetition=repeat,delta=d,confusion_row_rate_delta_pp=cm))
    return dict(seed=seed,label=label,left=left,right=right,pairs=pairs,
        summary={k:summarize([p['delta'][k] for p in pairs]) for k in pairs[0]['delta']},
        class_recall_delta=[summarize([p['confusion_row_rate_delta_pp'][k][k] for p in pairs]) for k in range(10)])


def main():
    evidence=json.loads(INPUT.read_text());stamp=evidence['source_stamp']
    for p,h in stamp.items():
        if digest(ROOT/p)!=h:raise ValueError('Changed frozen source '+p)
    status=json.loads((SOURCE/'status.json').read_text())
    assert status['status']=='completed' and status['virtual_steps']==96 and status['device']=='mps'
    assert evidence['virtual_steps']==96 and evidence['independent_seeds']==2
    assert not evidence['test_evaluated'] and not evidence['global_validation'] and not evidence['private_release']
    rows=evidence['rows'];index={}
    for r in rows:
        key=(r['seed'],r['repetition'],r['batch'],r['noise'],r['aggregation'])
        if key in index:raise ValueError('Duplicate intervention')
        check_row(r);index[key]=r
    expected=set(itertools.product(SEEDS,range(4),BATCHES,NOISES,AGGS))
    if set(index)!=expected:raise ValueError('Missing intervention')
    file_hashes={str(INPUT.relative_to(ROOT)):digest(INPUT)}
    for seed in SEEDS:
        for rep in range(4):
            p=SOURCE/f'seed{seed}__risk_rfa'/f'repetition{rep}.json'
            saved=json.loads(p.read_text());file_hashes[str(p.relative_to(ROOT))]=digest(p)
            assert saved['device']=='mps' and saved['original_branch_bitwise'] and saved['fixed_weights']
            assert not saved['test_evaluated'] and not saved['training'] and not saved['private_release']
            assert len(saved['rows'])==12 and all(index[(seed,rep,r['batch'],r['noise'],r['aggregation'])]==r for r in saved['rows'])
            for source,h in saved['signature']['inputs'].items():
                assert digest(ROOT/source)==h
        for agg in AGGS:
            ref=index[(seed,0,4800,'zero_oracle',agg)]
            for rep in range(1,4):
                r=index[(seed,rep,4800,'zero_oracle',agg)]
                assert r['metrics']==ref['metrics'] and r['confusion']==ref['confusion']
        for rep in range(4):
            for noise in ('full_plan','small_plan'):
                # With frozen mean weights and paired Z, batch does not change the noise vector.
                a=index[(seed,rep,4800,noise,'mean')]
                b=index[(seed,rep,240,noise,'mean')]
                close(a['decomposition']['noise_displacement_squared'],b['decomposition']['noise_displacement_squared'],3e-6)
            a=index[(seed,rep,4800,'full_plan','mean')]
            b=index[(seed,rep,4800,'small_plan','mean')]
            close(b['decomposition']['noise_displacement_squared'],a['decomposition']['noise_displacement_squared']*(b['std']/a['std'])**2,3e-6)
    contrasts=[]
    for seed in SEEDS:
        for batch,agg in itertools.product(BATCHES,AGGS):
            contrasts.append(paired(index,seed,(batch,'small_plan',agg),(batch,'full_plan',agg),'bruit haut − bas'))
        for noise,agg in itertools.product(NOISES,AGGS):
            contrasts.append(paired(index,seed,(240,noise,agg),(4800,noise,agg),'petit batch − complet'))
        for batch,noise in itertools.product(BATCHES,NOISES):
            contrasts.append(paired(index,seed,(batch,noise,'rfa'),(batch,noise,'mean'),'RFA − moyenne'))
    geometry=[]
    for seed in SEEDS:
        p=ROOT/f'results/ldp_gradient_far/full_population_private_risk_calibration_v28/seed{seed}__risk_rfa/metrics.json'
        source=json.loads(p.read_text());file_hashes[str(p.relative_to(ROOT))]=digest(p)
        lam=source['rounds'][-1]['aggregation']['objective_weights']
        for noise in NOISES:
            shifts=[]
            for rep in range(4):
                row=index[(seed,rep,4800,noise,'rfa')]
                nu=row['solver']['stationary_weights']
                close(sum(lam),1,2e-6);close(sum(nu),1,2e-6)
                shifts.append(sum(abs(a-b) for a,b in zip(lam,nu)))
            geometry.append(dict(seed=seed,noise=noise,effective_minus_objective_l1=summarize(shifts)))
    output=dict(contrasts=contrasts,verified_steps=96,independent_seeds=2,
        full_batch_rfa_weight_geometry=geometry,
        count_and_scalar_audit_passed=True,full_independent_gradient_replay=False,
        sources=file_hashes,analysis_sha256=digest(Path(__file__)),global_validation=False,
        new_confirmation_gate=False,V29_changed=False,V30_opened=False)
    DEST.with_suffix('.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def fmt(s):return f"{s['mean']:+.5f} ± {s['sd']:.5f}"
    lines=['# V31 — contrastes appariés et vérification des comptages','',
        '**96/96 interventions contrôlées ; 2 seeds de calibration ; aucune validation globale.**','',
        'Chaque cellule est moyenne ± écart-type de quatre différences strictement appariées '
        'au même modèle, batch/tirage et tenseur Z lorsque le contraste l’exige. '
        'Les deux modèles proviennent de la candidate risque-RFA : ils ne représentent pas '
        'tous les états d’entraînement. Aucun test statistique confirmatoire ni choix de seuil.', '',
        '| Seed | Contraste | Branche gauche (batch, bruit, agrégateur) | Δ accuracy (pp) | Δ Worst-20 (pp) | Δ CE | Δ erreur² |',
        '|--:|:--|:--|--:|--:|--:|--:|']
    for c in contrasts:
        lines.append(f"| {c['seed']} | {c['label']} | {c['left']} | "+' | '.join(fmt(c['summary'][k]) for k in ('accuracy_pct','worst20_pct','ce_loss','error_squared'))+' |')
    lines+=['','## Poids effectifs de RFA : diagnostic post-hoc','',
        'λ sont les poids de risque privés reçus ; ν sont les coefficients de reconstruction '
        'de la médiane géométrique pondérée, calculés par le solveur. Un faible écart L1 '
        'indique que RFA modifie peu les poids nominaux dans ces messages, pas une preuve '
        'd’équivalence des trajectoires ou de résistance byzantine. Ce complément est descriptif.', '',
        '| Seed | Bruit, batch complet | Somme des écarts absolus entre ν et λ |',
        '|--:|:--|--:|']
    for g in geometry:
        lines.append(f"| {g['seed']} | {g['noise']} | {fmt(g['effective_minus_objective_l1'])} |")
    lines+=['','## Lecture et limites','',
        '- Δ accuracy/Worst-20 positif favorise la branche gauche ; Δ CE/erreur² négatif la favorise.',
        '- « Bruit haut − bas » conserve la requête et Z ; seule son amplitude change. '
        'Cela isole un effet local du bruit, pas un avantage end-to-end.',
        '- « Petit batch − complet » conserve l’amplitude et Z ; la requête clippée change. '
        'Cela mesure localement le sampling à modèle et poids fixes.',
        '- « RFA − moyenne » conserve les messages et poids reçus ; seule la règle change.',
        '- Les passages croisés petit batch/bruit complet et les contrôles sans bruit '
        'ne sont pas revendiqués comme mécanismes epsilon=4. Oracles et validation sont hors transcript.',
        '- Les matrices de confusion de toutes les classes et chaque réalisation sont disponibles '
        'dans les fichiers. Il ne faut pas assimiler le rappel Shirt au Worst-20 des clients.', '',
        '## Audit réalisé','',
        'Empreintes des sources/entrées, 96 clés distinctes, validation composée de 10×1200 exemples, '
        'accuracy/Worst-20/gap/variance/balanced accuracy recalculés, marges et diagonales des '
        'matrices de confusion, identité avec terme croisé, invariance sans bruit et scaling '
        'quadratique du bruit pour la moyenne. Les pas originaux avaient été reproduits bit à bit '
        'sur MPS par le diagnostic. Ce contrôle scalaire n’est pas un second replay indépendant '
        'des 96 gradients et prédictions. Aucun entraînement CPU.', '',
        '[Protocole](Private_Risk_V31_Local_Batch_Noise_Diagnostic_Protocol.md) · '
        '[Contrastes et changements de confusion par classe](Private_Risk_V31_Local_Batch_Noise_Contrasts.json) · '
        '[Interventions originales](Private_Risk_V31_Local_Batch_Noise_Diagnostic.md)', '']
    DEST.with_suffix('.md').write_text('\n'.join(lines))
    assert all(digest(ROOT/p)==h for p,h in file_hashes.items())
    print(json.dumps(dict(verified_steps=96,contrasts=len(contrasts),global_validation=False)))


if __name__=='__main__':main()
