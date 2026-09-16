#!/usr/bin/env python3
"""Public temporal-noise covariance calculation plus paired MPS reconstruction."""
import json
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','0')
import torch
from scripts import run_fair_objective_screen as base
from scripts import run_recursive_private_risk_calibration_v19 as prior
from privacy.fair_objective import require_mps,release

OUT=ROOT/'results/ldp_gradient_far/recursive_temporal_noise_diagnostic_v20'
REPORT=ROOT/'output/analysis/Recursive_Temporal_Noise_V20_Analyse.md'


def main():
    require_mps();OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((prior.OUT/'manifest.json').read_text());stamp=dict(manifest['source_stamp']);base.verify_stamp(stamp)
    stamp[str(Path(__file__).relative_to(ROOT))]=base.digest(Path(__file__))
    assert json.loads((prior.OUT/'status.json').read_text())['status']=='completed'
    key=json.loads((prior.OUT/'simulator_secret.json').read_text())['key']
    theta=1-math.sqrt(.5+1e-8);a=1-theta;r=theta+a*theta
    eta=[2.]*60+[.5]*60
    coeff=[(1 if s==0 else r)*sum(eta[t]*a**(t-s) for t in range(s,120)) for s in range(120)]
    fresh_sum=sum(x*x for x in eta);recursive_sum=sum(x*x for x in coeff)
    theory=dict(theta=theta,a=a,noise_std_ratio=r,stationary_message_variance_ratio=r*r/(1-a*a),
        stationary_lag_one_correlation=a,long_constant_step_cumulative_variance_ratio=(r/theta)**2,
        actual_schedule_fresh_sum_sq=fresh_sum,actual_schedule_recursive_sum_sq=recursive_sum,
        actual_schedule_cumulative_variance_ratio=recursive_sum/fresh_sum,rounds=120,
        scope='Explicit Gaussian component only, fixed uniform client weights; not total model covariance nor a FAR/RFA covariance formula')
    rows=[]
    for seed in manifest['config']['seeds']:
        paths={mode:prior.OUT/f'seed{seed}__{mode}__erm_mean' for mode in ('fresh','recursive')}
        metrics={mode:json.loads((folder/'metrics.json').read_text()) for mode,folder in paths.items()}
        sigma=metrics['fresh']['privacy']['gradient_std'];assert sigma==metrics['recursive']['privacy']['gradient_std']
        model=base.new_model(manifest['profile'],seed)
        initial=torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad]);d=initial.numel()
        fresh=torch.zeros_like(initial);recursive=torch.zeros_like(initial);direct=torch.zeros_like(initial)
        memory=torch.zeros((10,d),device='mps')
        for t in range(120):
            noise=torch.stack([release(torch.zeros_like(initial),noise_std=sigma,seed=base.seed_for(key,seed,t,cid,'gaussian')) for cid in range(10)])
            memory=noise if t==0 else a*memory+r*noise
            fresh-=eta[t]*noise.mean(0);recursive-=eta[t]*memory.mean(0);direct-=coeff[t]*noise.mean(0)
        torch.testing.assert_close(recursive,direct,rtol=1e-4,atol=4e-6)
        for mode,N in [('fresh',fresh),('recursive',recursive)]:
            cp=torch.load(paths[mode]/'checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model'])
            final=torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad]);drift=final-initial
            response=drift-N
            # This algebraic residual includes gradients responding to past noise.
            actual_sq=float(drift.square().sum());noise_sq=float(N.square().sum());resp_sq=float(response.square().sum());cross=float(2*torch.dot(N,response))
            assert math.isclose(actual_sq,noise_sq+resp_sq+cross,rel_tol=2e-5,abs_tol=2e-4)
            expected=d*sigma*sigma/10*(fresh_sum if mode=='fresh' else recursive_sum)
            row=dict(seed=seed,mode=mode,dimension=d,sigma=sigma,explicit_noise_displacement_sq=noise_sq,
                theoretical_expected_noise_displacement_sq=expected,actual_model_displacement_sq=actual_sq,
                residual_query_response_sq=resp_sq,noise_response_cross=cross,
                endpoint_validation=metrics[mode]['final']['validation'])
            rows.append(row)
        base.checkpoint(OUT/f'seed{seed}_components.pt',dict(fresh_noise=fresh.detach().cpu(),recursive_noise=recursive.detach().cpu(),
            direct_recursive_noise=direct.detach().cpu(),privacy_protected=False,source_stamp=stamp))
        print(f'seed {seed}: explicit cumulative noise ratio {rows[-1]["explicit_noise_displacement_sq"]/rows[-2]["explicit_noise_displacement_sq"]:.6f}',flush=True)
        del model;torch.mps.empty_cache()
    base.save(OUT/'evidence.json',dict(device='mps',source_stamp=stamp,theory=theory,rows=rows,
        algebraic_audits_passed=True,global_validation=False,oracle_only=True))
    lines=['# V20 — le bruit réduit par tour peut s’accumuler davantage','',
        'Calcul public exact des coefficients, puis reconstruction sur MPS des Gaussiennes des deux témoins ERM-moyenne de V19. Aucun nouveau modèle entraîné.', '',
        '## Résultat','',
        f"Variance stationnaire par message : **{theory['stationary_message_variance_ratio']:.6f}×** le bruit frais. Corrélation entre tours consécutifs : **{a:.6f}**. Variance de la composante gaussienne cumulée avec le calendrier réel : **{theory['actual_schedule_cumulative_variance_ratio']:.6f}×** celle du mécanisme frais.", '',
        '## Dérivation, par coordonnée et client','',
        'Notons Z_t des Gaussiennes indépendantes de variance σ², E_1=Z_1, puis E_t=aE_(t−1)+rZ_t. La composante gaussienne explicite du déplacement final est −Σ_t η_t E_t. En échangeant les sommes :', '',
        '```text\nb_1 = Σ_(t=1)^T η_t a^(t−1),\nb_s = r Σ_(t=s)^T η_t a^(t−s), pour s≥2,\nVar(Σ_t η_t E_t) = σ² Σ_s b_s².\n```', '',
        f"Dans V19, Ση²={fresh_sum:.6f} et Σb²={recursive_sum:.6f}. La moyenne uniforme de dix bruits clients indépendants divise les deux variances par dix. Les d coordonnées donnent une espérance de norme carrée dσ²Σb²/10.", '',
        'À pas constant et horizon long, Cov(E_t,E_(t+k)) tend vers [r²σ²/(1−a²)]a^k. La somme des covariances donne le facteur asymptotique r²/(1−a)²=r²/θ², soit environ2,9142 ici. Les bruits sont moins dispersés à un tour, mais persistent dans la même direction aux tours suivants.', '',
        '| Seed | Requête | Norme² du déplacement gaussien explicite | Espérance théorique | Norme² du déplacement réel du modèle | Terme croisé bruit/réponse |',
        '|--:|:--|--:|--:|--:|--:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['mode']} | {row['explicit_noise_displacement_sq']:.4f} | {row['theoretical_expected_noise_displacement_sq']:.4f} | {row['actual_model_displacement_sq']:.4f} | {row['noise_response_cross']:+.4f} |")
    lines+=['','## Limites importantes','',
        'Le modèle répond au bruit via les gradients et le clipping. Sa variance totale n’est donc pas la variance de cette seule composante. Le tableau conserve le terme croisé avec la réponse des requêtes : il ne suppose pas leur indépendance. Ces deux réalisations et le calcul public ne constituent pas une preuve que ce phénomène explique à lui seul toute la perte d’accuracy.', '',
        'La formule d’agrégation concerne les poids clients uniformes et fixés. On ne la transpose pas à RFA ou aux poids de risque adaptatifs. La récursion gaussienne de chaque message reste pertinente pour comprendre le mécanisme, mais il faudrait alors traiter la dépendance des coefficients aux messages.', '',
        'Conclusion défendable : optimiser uniquement la variance marginale d’un message, comme le critère mécanistique V18, omet sa covariance temporelle. Ce critère ne peut pas justifier seul la promotion d’un optimiseur privé.', '',
        '[Résultats V19](Recursive_Private_Risk_V19_Analyse.md). [Diagnostic de population V20](Recursive_Complete_Query_V20_Population_Analyse.md).']
    REPORT.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
