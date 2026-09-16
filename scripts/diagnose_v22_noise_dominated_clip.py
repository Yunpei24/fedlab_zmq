#!/usr/bin/env python3
"""Post-V22 diagnostic, four saved ERM-mean states; no new training/selection."""
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


def main():
    require_mps()
    audit_path = ROOT/'output/analysis/Fresh_Private_Risk_Step_V22_Analyse.json'
    audit = json.loads(audit_path.read_text()); assert audit['audit_passed']
    manifest = json.loads((run.OUT/'manifest.json').read_text())
    profile,stamp = manifest['profile'],dict(manifest['source_stamp'])
    paths = [Path(__file__),audit_path]
    for seed in (170501,170502):
        paths += [root/folder/name for root,folder in (
            (run.OUT,f'seed{seed}__global_clip__erm_mean'),
            (run.prior.OUT,f'seed{seed}__fresh__erm_mean'))
            for name in ('checkpoint.pt','metrics.json','simulator_oracle.json','orchestration_status.json')]
    stamp.update({str(p.relative_to(ROOT)):base.digest(p) for p in paths}); base.verify_stamp(stamp)
    key = json.loads((run.prior.OUT/'simulator_secret.json').read_text())['key']
    records = []
    for seed in (170501,170502):
        data = base.prepare(profile,seed); model = base.new_model(profile,seed)
        for mode in ('unchanged','global_clip'):
            path = run.OUT/f'seed{seed}__global_clip__erm_mean' if mode == 'global_clip' else run.prior.OUT/f'seed{seed}__fresh__erm_mean'
            metric = json.loads((path/'metrics.json').read_text()); status = json.loads((path/'orchestration_status.json').read_text())
            assert status['status'] == 'completed' and status['metrics_sha256'] == base.digest(path/'metrics.json')
            assert metric['device'] == 'mps' and not metric['test_evaluated']
            cp = torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
            # V19 stores the same pre-round model under previous_model.
            model.load_state_dict(cp['previous_model'])
            oracle = json.loads((path/'simulator_oracle.json').read_text())['rounds'][-1]
            assert oracle['round'] == 120
            clean,sent = [],[]; p = metric['privacy']
            for cid,ids in enumerate(data['train']):
                ix = base.draw_indices(4800,240,base.seed_for(key,seed,119,cid,'batch'))
                assert base.ids_hash(ix) == oracle['clients'][cid]['batch_hash']
                _,g,_,_ = per_example(model,data['x'][ids[ix]],data['y'][ids[ix]],clip_norm=2.)
                h = g.mean(0); clean.append(h)
                sent.append(release(h,noise_std=p['gradient_std'],seed=base.seed_for(key,seed,119,cid,'gaussian')))
            lam = torch.ones(10,device='mps')/10
            H = (lam[:,None]*torch.stack(clean)).sum(0)
            A = (lam[:,None]*torch.stack(sent)).sum(0)
            Z = A-H
            h2,z2,a2,cross = float(H.square().sum()),float(Z.square().sum()),float(A.square().sum()),float(2*torch.dot(H,Z))
            assert math.isclose(a2,h2+z2+cross,rel_tol=2e-6,abs_tol=2e-6)
            step,diag = controlled_step(A,.5,mode,radius=1.)
            base.apply_gradient(model,step,1.)
            for name,tensor in model.state_dict().items():
                torch.testing.assert_close(tensor,cp['model'][name].to('mps'),atol=0,rtol=0)
            if mode == 'global_clip':
                expected = metric['final']['aggregation']['step_control']
                assert diag == expected
            d = A.numel(); expected_z2 = d*p['gradient_std']**2/10
            records.append(dict(seed=seed,mode=mode,round=120,dimension=d,
                clean_batch_aggregate_norm=math.sqrt(h2),noise_component_norm=math.sqrt(z2),aggregate_norm=math.sqrt(a2),
                clean_squared_norm=h2,noise_squared_norm=z2,cross_term=cross,total_squared_norm=a2,
                theoretical_noise_squared_norm=expected_z2,theoretical_noise_rms=math.sqrt(expected_z2),
                control_factor=diag['factor'],clean_component_norm_retained=diag['factor']*math.sqrt(h2),
                last_update_bitwise_verified=True))
            print(json.dumps(records[-1]),flush=True)
        del model,data; torch.mps.empty_cache()
    base.verify_stamp(stamp)
    base.save(DEST.with_suffix('.json'),dict(audit_passed=True,source_stamp=stamp,device='mps',
        interpretation='Post-outcome diagnostic of four ERM-mean final updates, not a new training result; research oracles, not DP outputs.',
        rows=records,new_training_runs=0,global_validation=False))
    lines = ['# V22 — le rayon global était sous l’échelle du bruit honnête','',
        'Diagnostic postérieur aux résultats V22 ; quatre derniers pas ERM-moyenne reconstitués sur MPS, sans nouvel entraînement. Aucun critère ni rayon n’est modifié.', '',
        '## Décomposition exacte contrôlée','',
        'H est la moyenne des gradients de batch clippés **avant bruit**, Z=A−H la composante injectée reconstituée (avec les arrondis float32), A la moyenne privée. Pour chaque état :', '',
        '```text\n||A||² = ||H||² + ||Z||² + 2〈H,Z〉.\n```','',
        'Le terme croisé est conservé : on ne présente pas les deux premières énergies comme des pourcentages additifs de ||A||². H n’est pas le gradient de population sans clipping.', '',
        '| Seed | Contrôle | Norme H | Norme Z | Norme A | Terme croisé | Facteur du pas | Norme H retenue |',
        '|--:|:--|--:|--:|--:|--:|--:|--:|']
    for r in records:
        cells = [f'{r[k]:.7f}' for k in ('clean_batch_aggregate_norm','noise_component_norm','aggregate_norm','cross_term','control_factor','clean_component_norm_retained')]
        lines.append('| '+' | '.join([str(r['seed']),r['mode'],*cells])+' |')
    lines += ['','## Pourquoi cette échelle est attendue','',
        'Pour les dix messages à bruits gaussiens indépendants de covariance σ²I, conditionnellement au modèle et aux batches :', '',
        '```text\nE[||Z||² | modèle,batches] = d σ² / 10,\nE[||A||² | modèle,batches] = ||H||² + d σ² / 10.\n```','',
        f'Avec d={records[0]["dimension"]}, la norme RMS du bruit seul vaut **{records[0]["theoretical_noise_rms"]:.6f}**, supérieure au rayon public 1. La RMS est sqrt(E||Z||²), pas exactement E||Z||.', '',
        'Le clipping radial multiplie simultanément H et Z par le même facteur. Sur ces réalisations, il réduit donc aussi la direction de batch honnête. Ce constat explique le facteur presque constant voisin de0,44 sur les trajectoires ERM V22 ; il n’isole pas à lui seul toute la causalité de la perte finale.', '',
        'Ce résultat concerne la moyenne uniforme. Il ne faut pas remplacer la covariance de RFA par σ²I/10. Il ne prouve pas non plus que le clipping serveur est inutile sous attaque : ces quatre états sont propres. Il réfute une lecture du rayon1 comme filtre sélectif de seules anomalies byzantines dans ce régime.', '',
        'Les quatre modèles finaux sont reproduits bit à bit après le pas reconstruit. Les états propres diffèrent entre les deux trajectoires ; il ne s’agit pas d’une intervention à un modèle hôte commun.', '',
        '[Résultats V22](Fresh_Private_Risk_Step_V22_Analyse.md) · [Décompositions numériques](Fresh_Private_Risk_Step_V22_Noise_Dominated_Clip.json).']
    DEST.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
