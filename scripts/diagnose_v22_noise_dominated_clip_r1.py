#!/usr/bin/env python3
"""Two genuine pre-round V22 states; V19 fresh previous_model is inapplicable."""
import json
import math
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fresh_private_risk_step_v22_r1 as run
from scripts import run_fair_objective_screen as base
from privacy.fair_objective import require_mps, per_example, release
from privacy.aggregate_radial_step_v21 import controlled_step

DEST = ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Noise_Dominated_Clip'
AMENDMENT = ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Noise_Diagnostic_Amendment.md'


def main():
    require_mps(); assert not DEST.with_suffix('.json').exists(), 'Do not replace a completed diagnostic'
    audit_path = ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Analyse.json'
    assert json.loads(audit_path.read_text())['audit_passed']
    manifest = json.loads((run.OUT/'manifest.json').read_text())
    profile,stamp = manifest['profile'],dict(manifest['source_stamp'])
    paths = [Path(__file__),audit_path,AMENDMENT,ROOT/'scripts/diagnose_v22_noise_dominated_clip.py']
    paths += [run.OUT/f'seed{s}__global_clip__erm_mean'/name for s in (170501,170502)
              for name in ('checkpoint.pt','metrics.json','simulator_oracle.json','orchestration_status.json')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths}); base.verify_stamp(stamp)
    key = json.loads((run.prior.OUT/'simulator_secret.json').read_text())['key']; records = []
    for seed in (170501,170502):
        path = run.OUT/f'seed{seed}__global_clip__erm_mean'
        metric = json.loads((path/'metrics.json').read_text()); status = json.loads((path/'orchestration_status.json').read_text())
        assert status['status'] == 'completed' and status['metrics_sha256'] == base.digest(path/'metrics.json')
        assert metric['device'] == 'mps' and not metric['test_evaluated']
        data = base.prepare(profile,seed); model = base.new_model(profile,seed)
        cp = torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert cp['round'] == 120
        model.load_state_dict(cp['previous_model'])
        oracle = json.loads((path/'simulator_oracle.json').read_text())['rounds'][-1]
        assert oracle['round'] == 120
        clean,sent = [],[]; p = metric['privacy']
        for cid,ids in enumerate(data['train']):
            ix = base.draw_indices(4800,240,base.seed_for(key,seed,119,cid,'batch'))
            assert base.ids_hash(ix) == oracle['clients'][cid]['batch_hash']
            _,g,norms,_ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
            assert int((norms>2).sum()) == oracle['clients'][cid]['gradient_clipped_count']
            h = g.mean(0); clean.append(h)
            sent.append(release(h,noise_std=p['gradient_std'],seed=base.seed_for(key,seed,119,cid,'gaussian')))
        lam = torch.ones(10,device='mps')/10
        H = (lam[:,None]*torch.stack(clean)).sum(0)
        A = (lam[:,None]*torch.stack(sent)).sum(0); Z = A-H
        h2,z2,a2,cross = float(H.square().sum()),float(Z.square().sum()),float(A.square().sum()),float(2*torch.dot(H,Z))
        assert math.isclose(a2,h2+z2+cross,rel_tol=2e-6,abs_tol=2e-6)
        step,diag = controlled_step(A,.5,'global_clip',radius=1.)
        assert diag == metric['final']['aggregation']['step_control']
        base.apply_gradient(model,step,1.)
        for name,tensor in model.state_dict().items():
            torch.testing.assert_close(tensor,cp['model'][name].to('mps'),atol=0,rtol=0)
        d = A.numel(); expected_z2 = d*p['gradient_std']**2/10
        records.append(dict(seed=seed,round=120,dimension=d,
            clean_batch_aggregate_norm=math.sqrt(h2),noise_component_norm=math.sqrt(z2),aggregate_norm=math.sqrt(a2),
            clean_squared_norm=h2,noise_squared_norm=z2,cross_term=cross,total_squared_norm=a2,
            theoretical_noise_squared_norm=expected_z2,theoretical_noise_rms=math.sqrt(expected_z2),
            control_factor=diag['factor'],clean_component_norm_retained=diag['factor']*math.sqrt(h2),
            last_update_bitwise_verified=True))
        print(json.dumps(records[-1]),flush=True)
        del model,data; torch.mps.empty_cache()
    base.verify_stamp(stamp)
    base.save(DEST.with_suffix('.json'),dict(audit_passed=True,source_stamp=stamp,device='mps',
        interpretation='Post-outcome diagnostic of two V22 ERM-mean final updates, not a new training result; research oracles, not DP outputs.',
        rows=records,new_training_runs=0,global_validation=False))
    lines = ['# V22 — le rayon global était sous l’échelle du bruit honnête','',
        'Diagnostic postérieur aux résultats V22 ; **deux derniers pas ERM-moyenne V22 reconstitués exactement sur MPS**, sans nouvel entraînement. Aucun critère ni rayon n’est modifié. Les états V19 frais sont exclus pour la raison de stockage documentée dans l’amendement, pas selon leurs résultats.', '',
        '## Décomposition exacte contrôlée','',
        'H est la moyenne des gradients de batch clippés avant bruit, Z=A−H la composante injectée reconstituée (avec arrondis float32), A la moyenne privée.', '',
        '```text\n||A||² = ||H||² + ||Z||² + 2〈H,Z〉.\n```','',
        'Le terme croisé est conservé : les deux premières énergies ne sont pas présentées comme des pourcentages additifs de ||A||². H n’est pas le gradient de population sans clipping.', '',
        '| Seed | Norme H | Norme Z | Norme A | Terme croisé | Facteur du pas | Norme H retenue |',
        '|--:|--:|--:|--:|--:|--:|--:|']
    for r in records:
        cells = [f'{r[k]:.7f}' for k in ('clean_batch_aggregate_norm','noise_component_norm','aggregate_norm','cross_term','control_factor','clean_component_norm_retained')]
        lines.append('| '+' | '.join([str(r['seed']),*cells])+' |')
    lines += ['','## Échelle attendue et interprétation','',
        'Pour dix messages à bruits gaussiens indépendants de covariance σ²I, conditionnellement au modèle et aux batches :', '',
        '```text\nE[||Z||² | modèle,batches] = d σ² / 10,\nE[||A||² | modèle,batches] = ||H||² + d σ² / 10.\n```','',
        f'Avec d={records[0]["dimension"]}, la norme RMS du bruit seul vaut **{records[0]["theoretical_noise_rms"]:.6f}**, supérieure au rayon public1. RMS signifie sqrt(E||Z||²), pas exactement E||Z||.', '',
        'Le clipping radial multiplie simultanément H et Z par le même facteur. Il réduit donc aussi la direction de batch honnête sur ces réalisations. Cela explique pourquoi le facteur appliqué peut rester presque constant, voisin de0,44, même sans aucun attaquant. Ce diagnostic local ne mesure pas toute sa contribution causale à la perte finale.', '',
        'La covariance σ²I/10 concerne seulement la moyenne uniforme, pas RFA. Le résultat ne prouve pas que le clipping serveur est inutile sous attaque ; il interdit de lire le rayon1 comme un filtre sélectif de seules anomalies byzantines dans ces deux états propres.', '',
        'Les deux modèles finaux sont reproduits bit à bit après le pas reconstruit. Ces oracles ne sont pas des sorties privées publiables. Aucun modèle pré-tour manquant n’a été inventé.', '',
        '[Résultats V22](Fresh_Private_Risk_Step_V22_Analyse.md) · [Décompositions numériques](Fresh_Private_Risk_Step_V22_Noise_Dominated_Clip.json) · [Amendement de disponibilité des états](Fresh_Private_Risk_Step_V22_Noise_Diagnostic_Amendment.md).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
