"""Preserve the complete first-seed Bit-Flip block, without claiming full audit."""
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from scripts import run_fair_objective_screen as base
from scripts.audit_private_risk_continuation_v33 import exact

OUT=ROOT/'results/ldp_gradient_far/private_risk_continuation_v33'
DEST=ROOT/'output/analysis/Private_Risk_V33_First_BitFlip_Block'

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());base.verify_stamp(manifest['stamp'])
    hashes={str(Path(__file__).relative_to(ROOT)):base.digest(Path(__file__))}
    records={};seed=170501
    for condition in ('none','abrupt_bf'):
        for method in ('risk_mean','risk_rfa','risk_winsor'):
            folder=OUT/f'seed{seed}__{condition}__{method}'
            status=json.loads((folder/'orchestration_status.json').read_text())
            rec=json.loads((folder/'metrics.json').read_text())
            assert status['status']=='completed' and rec['device']==status['device']=='mps'
            assert rec['test_evaluated'] is False and rec['signature']['source_stamp']==manifest['stamp']
            assert status['metrics_sha256']==base.digest(folder/'metrics.json')
            assert status['checkpoint_sha256']==base.digest(folder/'checkpoint.pt')
            assert [r['step'] for r in rec['rows']]==list(range(1,13))
            for fn in ('metrics.json','checkpoint.pt','orchestration_status.json'):
                p=folder/fn;hashes[str(p.relative_to(ROOT))]=base.digest(p)
            for k in (4,8,12):
                ev=rec['rows'][k-1]['validation']
                for field,num in exact(ev,range(10)).items():
                    assert math.isclose(float(num),ev[field],rel_tol=1e-9,abs_tol=1e-8)
            records[condition,method]=rec
    for method in ('risk_mean','risk_rfa','risk_winsor'):
        for k in range(4):
            assert records['none',method]['rows'][k]['model_hash']==records['abrupt_bf',method]['rows'][k]['model_hash']
            assert records['none',method]['rows'][k]['clients']==records['abrupt_bf',method]['rows'][k]['clients']
    table=[];comparisons=[]
    for method in ('risk_mean','risk_rfa','risk_winsor'):
        for k in (4,8,12):
            clean=exact(records['none',method]['rows'][k-1]['validation'],range(2,10))
            v=exact(records['abrupt_bf',method]['rows'][k-1]['validation'],range(2,10))
            table.append(dict(method=method,step=k,accuracy=float(v['accuracy_pct']),worst20=float(v['worst20_pct']),
                accuracy_delta_from_clean=float(v['accuracy_pct']-clean['accuracy_pct']),
                worst20_delta_from_clean=float(v['worst20_pct']-clean['worst20_pct'])))
    for k in (8,12):
        v=exact(records['abrupt_bf','risk_winsor']['rows'][k-1]['validation'],range(2,10))
        for a,m in (('none','risk_winsor'),('abrupt_bf','risk_rfa')):
            b=exact(records[a,m]['rows'][k-1]['validation'],range(2,10))
            da,dw=v['accuracy_pct']-b['accuracy_pct'],v['worst20_pct']-b['worst20_pct']
            comparisons.append(dict(step=k,control=f'{a}/{m}',accuracy_delta_pp=float(da),worst20_delta_pp=float(dw),
                accuracy_delta_exact=str(da),worst20_delta_exact=str(dw),passed=da>=Q(-1) and dw>=Q(-1)))
    diagnostics=[dict(step=r['step'],attack=r['attack'],aggregation=r['aggregation'],oracle=r['oracle'])
        for r in records['abrupt_bf','risk_winsor']['rows']]
    base.verify_stamp(manifest['stamp'])
    for p,h in hashes.items():assert base.digest(ROOT/p)==h
    failed=any(not c['passed'] for c in comparisons)
    result=dict(seed=seed,completed_branches_checked=6,exact_counts_checked=True,pre_attack_pairing_passed=True,
        necessary_predeclared_criterion_failed=failed,full_campaign_completed=False,full_audit=False,
        candidate_promotable=False,global_validation=False,table=table,comparisons=comparisons,
        winsor_diagnostics=diagnostics,source_stamp=manifest['stamp'],input_hashes=hashes,
        scope='First calibration seed only; last-step numerical replay remains pending')
    base.save(DEST.with_suffix('.json'),result)
    lines=['# V33 — premier bloc Bit-Flip complet : critère nécessaire non satisfait','',
        'Six trajectoires terminées, seed de calibration170501. Comptages exacts, empreintes et '
        'appariement avant attaque vérifiés. La campagne entière et son rejeu numérique ne sont pas terminés.', '',
        '**La règle V32 évite le grand effondrement de la moyenne, mais échoue à un critère de '
        'protection fixé avant les runs.** Comme les critères sont conjoints et exigés pour chaque '
        'seed, les résultats restants ne peuvent compenser cet échec. Ils restent nécessaires '
        'pour compléter le diagnostic ; on ne modifie pas les marges.', '',
        'Population comparée : les mêmes huit honnêtes2…9. Attaque aux pas5–8, récupération9–12.', '',
        '| Règle | Pas | Accuracy honnête (%) | Worst-20 (%) | Δ Acc contre propre (pp) | Δ Worst-20 contre propre (pp) |',
        '|:--|--:|--:|--:|--:|--:|']
    for t in table:
        lines.append(f"| {t['method']} | {t['step']} | {t['accuracy']:.4f} | {t['worst20']:.4f} | {t['accuracy_delta_from_clean']:+.4f} | {t['worst20_delta_from_clean']:+.4f} |")
    lines+=['','## Comparaisons de V32 qui déterminent le rejet local','',
        '| Pas | Contrôle | Δ Acc (pp) | Δ Worst-20 (pp) | Les deux ≥−1 pp ? |',
        '|--:|:--|--:|--:|:--|']
    for c in comparisons:
        lines.append(f"| {c['step']} | {c['control']} | {c['accuracy_delta_pp']:+.6f} | {c['worst20_delta_pp']:+.6f} | {'oui' if c['passed'] else 'non'} |")
    lines+=['','## Diagnostics de la règle V32','',
        '| Pas | Attaque active | Honnêtes clippés | Paire0/1 clippée | Rayon | Masse active byzantine | Norme perturbation attaque, état identique | Norme somme des perturbations |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in diagnostics:
        d,o=r['aggregation'],r['oracle']
        lines.append(f"| {r['step']} | {r['attack']['active']} | {o['honest_clipped_count']} | {o['reserved_pair_clipped_count']} | {d['radius']:.4f} | {o['active_byzantine_mass']:.4f} | {o['same_state_attack_perturbation_norm']:.5f} | {o['cumulative_attack_perturbation_norm']:.5f} |")
    lines+=['','## Interprétation limitée','',
        'Le bornage d’amplitude réduit le dommage, mais ne garantit pas une petite erreur de '
        'direction pendant plusieurs pas. La perturbation de même état compare la règle sur '
        'les messages falsifiés et sur les messages originaux produits par SON modèle au même tour. '
        'Ce diagnostic n’est pas une décomposition causale complète de la différence de modèles '
        'entre deux trajectoires qui divergent. La récupération ne doit pas effacer le coût '
        'pendant l’attaque. Les attaques ALIE/IPM et la seconde seed restent à analyser.', '',
        'Aucune validation globale, aucune modification de V29/V30, aucune nouvelle candidate '
        'lancée sur la base de ce bloc partiel.', '',
        '[Protocole figé](Private_Risk_V33_Continuation_Protocol.md) · '
        '[Comptages, écarts exacts et empreintes](Private_Risk_V33_First_BitFlip_Block.json)']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(necessary_criterion_failed=failed,comparisons=comparisons,table=table)))

if __name__=='__main__':main()
