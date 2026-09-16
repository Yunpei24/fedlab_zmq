"""Offline oracle audit of four aggregates on the exact same private cohort.

The deployed server exports only private-transcript-derived candidates.
This function sees clean simulator gradients and attack truth after aggregation.
It returns scalars, not vectors, and cannot influence a deployed choice.
"""
import math
import torch
from robustness.tensor_ops import stack_updates
from algorithms.ldp_aggregation_role_ablation import ARMS


def squared(x):
    return float(x.square().sum())


def corr(x,y):
    a,b=x-x.mean(),y-y.mean()
    den=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return None if den<=1e-15 else float(a@b/den)


def evaluate_aggregation_roles(payload, clean_by_client, client_updates):
    if not payload or payload.get('kind')!='aggregation_role_v1' or clean_by_client is None:
        raise ValueError('missing same-cohort audit inputs')
    if any(payload[k] is not False for k in ['contains_clean_data','contains_attack_labels','contains_realised_noise']):
        raise ValueError('forbidden server oracle data')
    ids=[int(m['client_id']) for _,m,_ in client_updates]
    if ids!=payload['client_ids'] or set(ids)!=set(clean_by_client):
        raise ValueError('client alignment mismatch')
    mask=torch.tensor([not bool(m.get('is_byzantine',False)) for _,m,_ in client_updates])
    if not mask.any():
        raise ValueError('no honest target')
    clean,_=stack_updates([clean_by_client[i] for i in ids])
    factors=(payload['server_clip_norm']/torch.linalg.vector_norm(clean,dim=1).clamp_min(1e-12)).clamp(max=1)
    g=clean*factors[:,None]
    target=g[mask].mean(0)
    x=payload['vectors']
    actual,_=stack_updates([u for u,_,_ in client_updates])
    f=(payload['server_clip_norm']/torch.linalg.vector_norm(actual,dim=1).clamp_min(1e-12)).clamp(max=1)
    if not torch.equal(x,actual*f[:,None]):
        raise ValueError('counterfactual cohort differs from actual private inputs')
    rules=payload['rules']
    if set(rules['aggregates'])!=set(ARMS) or set(rules['weights'])!=set(ARMS):
        raise ValueError('missing candidate')
    result=dict(aggregation_role_oracles_visible_to_server=False,
                aggregation_role_same_cohort_verified=True,
                aggregation_role_oracle_boundary='offline_simulator_only',
                aggregation_role_oracle_target='mean_honest_clean_per_example_clipped_batch_gradients_then_server_clipped',
                aggregation_role_honest_evaluation_clients=int(mask.sum()))
    noise_norm=torch.linalg.vector_norm(x[mask]-g[mask],dim=1)
    public_scales=torch.tensor(payload['public_noise_scales'],dtype=x.dtype)
    for name in ARMS:
        a,w=rules['aggregates'][name],rules['weights'][name]
        if not torch.isfinite(a).all() or not torch.isfinite(w).all() or (w<0).any() or abs(float(w.sum())-1)>1e-12:
            raise ValueError('invalid candidate or coefficients')
        if not torch.allclose(a,(w[:,None]*x).sum(0),atol=1e-11,rtol=1e-11):
            raise ValueError('weights do not reconstruct aggregate')
        # Exact algebraic decomposition, AFTER server clipping:
        # A-gH = honest effective noise + honest reweighting + centred Byzantines.
        noise=(w[mask,None]*(x[mask]-g[mask])).sum(0)
        tilt=(w[mask,None]*(g[mask]-target)).sum(0)
        byz=(w[~mask,None]*(x[~mask]-target)).sum(0)
        error=a-target
        reconstruction=squared(error-noise-tilt-byz)
        if reconstruction>1e-18*max(1,squared(error)):
            raise ValueError('aggregate-error decomposition failed')
        prefix='cohort_'+name+'_'
        entries=dict(aggregate_error_sq=squared(error),aggregate_norm=float(torch.linalg.vector_norm(a)),
                     honest_effective_noise_sq=squared(noise),honest_tilt_sq=squared(tilt),
                     byzantine_centered_sq=squared(byz),byzantine_raw_contribution_sq=squared((w[~mask,None]*x[~mask]).sum(0)),
                     cross_noise_tilt=2*float(noise@tilt),cross_noise_byzantine=2*float(noise@byz),cross_tilt_byzantine=2*float(tilt@byz),
                     decomposition_residual_sq=reconstruction,byzantine_mass=float(w[~mask].sum()),
                     max_weight=float(w.max()),concentration=float(len(w)*w.square().sum()),
                     entropy=float(-(w*w.clamp_min(1e-15).log()).sum()),
                     weight_l1_vs_uniform=float((w-rules['weights']['uniform']).abs().sum()),
                     weight_effective_noise_corr_honest=corr(w[mask],noise_norm),
                     weight_public_noise_scale_corr_honest=corr(w[mask],public_scales[mask]))
        result.update({prefix+k:v for k,v in entries.items()})
    for name,ref in rules['references'].items():
        result['cohort_reference_'+name+'_error_sq']=squared(ref-target) if name=='rfa' or payload['ready'] else None
    result['cohort_far_weights_l1_rfa_vs_midpoint']=float((rules['weights']['far_rfa']-rules['weights']['far_midpoint']).abs().sum())
    result['cohort_far_aggregates_sq_rfa_vs_midpoint']=squared(rules['aggregates']['far_rfa']-rules['aggregates']['far_midpoint'])
    deployed=payload['deployed']
    for metric in ['aggregate_error_sq','honest_effective_noise_sq','honest_tilt_sq','byzantine_centered_sq','byzantine_mass']:
        result['deployed_'+metric]=result['cohort_'+deployed+'_'+metric]
    if not all(v is None or not isinstance(v,float) or math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite oracle diagnostic')
    return result
