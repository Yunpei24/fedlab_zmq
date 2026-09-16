"""Four private-gradient aggregation rules on one cohort, one deployed rule.

This scoped subclass preserves the audited private client and temporal-state
machinery. The parent computes/commits a shadow midpoint; its proposed model
is never applied. Only the predeclared rule below determines the returned model.
Clean gradients and attack labels remain exclusively in the evaluator.
"""
import math
import torch

from algorithms.rcig_n10_ablation import N10PolicyAblation
from algorithms.reference_utils import apply_delta
from robustness.tensor_ops import stack_updates, unflatten_update
from metrics.robustness import weight_diagnostics

ARMS = ("uniform", "rfa_direct", "far_rfa", "far_midpoint")


def rfa_with_weights(vectors, *, max_iter=100, tol=1e-6, smoothing=1e-8):
    """Exactly the existing smoothed Weiszfeld solver, with last weights.

    These are effective convex-combination coefficients, not FAR weights.
    No bucketing/NNM preprocessing is added.
    """
    if max_iter < 1 or tol <= 0 or smoothing <= 0:
        raise ValueError("invalid RFA numerical settings")
    point = vectors.mean(dim=0)
    for iteration in range(1, max_iter + 1):
        inv = torch.linalg.vector_norm(vectors-point, dim=1).clamp_min(smoothing).reciprocal()
        candidate = (inv[:, None]*vectors).sum(0)/inv.sum()
        displacement = float(torch.linalg.vector_norm(candidate-point))
        weights = inv/inv.sum()
        point = candidate
        if displacement <= tol:
            break
    assert torch.allclose(point, (weights[:, None]*vectors).sum(0), atol=1e-12, rtol=1e-12)
    return point, weights, iteration, displacement


def cohort_rules(vectors, midpoint, *, alpha, max_iter=100, tol=1e-6):
    if vectors.ndim != 2 or vectors.shape[0] < 2 or not torch.isfinite(vectors).all():
        raise ValueError("invalid private cohort")
    if midpoint.shape != vectors.shape[1:] or not torch.isfinite(midpoint).all():
        raise ValueError("invalid midpoint")
    if not math.isfinite(alpha):
        raise ValueError("nonfinite alpha")
    rfa, rfa_weights, iterations, residual = rfa_with_weights(vectors, max_iter=max_iter, tol=tol)
    uniform = torch.full((len(vectors),), 1/len(vectors), dtype=vectors.dtype, device=vectors.device)
    distances = {"far_rfa": torch.linalg.vector_norm(vectors-rfa, dim=1),
                 "far_midpoint": torch.linalg.vector_norm(vectors-midpoint, dim=1)}
    weights = {"uniform": uniform, "rfa_direct": rfa_weights,
               **{name:torch.softmax(alpha*d, dim=0) for name,d in distances.items()}}
    aggregates = {name:(w[:, None]*vectors).sum(0) for name,w in weights.items()}
    aggregates['rfa_direct'] = rfa
    return dict(aggregates=aggregates, weights=weights, references={'rfa':rfa,'midpoint':midpoint},
                distances=distances, rfa_iterations=iterations, rfa_last_displacement=residual)


class LDPAggregationRoleAblation(N10PolicyAblation):
    """Same local DP channel, four fixed alternatives, no oracle selection."""

    def server_aggregate(self, global_model, client_updates, round_num, config):
        arm = config['aggregation_role_arm']
        if arm not in ARMS or config['rcig_n10_arm'] != 'midpoint':
            raise ValueError("undeclared deployment or shadow-state rule")
        if config.get('attack') is not None:
            raise ValueError("attack config crossed the aggregation boundary")
        for _, metadata, _ in client_updates:
            if any(k == 'is_byzantine' or k.startswith('attack_') or 'oracle' in k for k in metadata):
                raise ValueError("oracle crossed the aggregation boundary")
        # Does not mutate global_model. This also validates the privacy channel,
        # public covariance registry, MPS provenance, and commits one snapshot.
        result = super().server_aggregate(global_model, client_updates, round_num, config)
        parent_payload = result.metrics.pop('_rcig_evaluation_payload', None)
        ready = round_num >= 12
        if ready != (parent_payload is not None):
            raise RuntimeError("unexpected temporal chronology")
        vectors, layout = stack_updates([u for u,_,_ in client_updates])
        radius = float(config['far_server_clip_norm'])
        factors = (radius/torch.linalg.vector_norm(vectors, dim=1).clamp_min(1e-12)).clamp(max=1)
        x = vectors*factors[:, None]
        midpoint = parent_payload['deployed_reference'] if ready else torch.zeros_like(x[0])
        rules = cohort_rules(x, midpoint, alpha=float(config['far_alpha']) if ready else 0,
                             max_iter=int(config['rfa_max_iter']), tol=float(config['rfa_tol']))
        deployed = arm if ready else 'uniform'
        aggregate = rules['aggregates'][deployed]
        weights = rules['weights'][deployed]
        lr = float(config['far_server_lr'])
        expected_midpoint = apply_delta(global_model, {k:lr*v for k,v in unflatten_update(rules['aggregates']['far_midpoint'],layout).items()})
        if not all(torch.equal(expected_midpoint[k],result.new_weights[k]) for k in expected_midpoint):
            raise RuntimeError("shadow midpoint is inconsistent with historical FAR")
        result.new_weights = apply_delta(global_model, {k:lr*v for k,v in unflatten_update(aggregate,layout).items()})
        # Never leave midpoint diagnostics labelled as the deployed rule.
        for key in list(result.metrics):
            if key.startswith('far_') or key.startswith('_far_'):
                result.metrics.pop(key)
            elif key.startswith('rcig_'):
                # Device names are retained for the common device auditor only.
                if key not in {'rcig_private_gradient_mps_fraction','rcig_private_gradient_compute_device',
                               'rcig_server_aggregation_device','rcig_server_aggregation_dtype'}:
                    result.metrics['shadow_'+key] = result.metrics.pop(key)
        ids = [int(m['client_id']) for _,m,_ in client_updates]
        actual_alpha = float(config['far_alpha']) if ready and arm.startswith('far_') else 0.0
        diagnostics = weight_diagnostics(weights)
        diagnostics.update(
            aggregation_role_arm=arm, aggregation_role_effective_arm=deployed,
            aggregation_role_warmup=not ready,
            aggregation_role_reference_is_final_aggregate=(deployed=='rfa_direct'),
            aggregation_role_rfa_preprocessing='none_no_bucketing',
            aggregation_role_shadow_midpoint_drives_update=(deployed=='far_midpoint'),
            aggregation_role_only_declared_rule_applied=True,
            aggregation_role_rfa_iterations=rules['rfa_iterations'],
            aggregation_role_rfa_last_displacement=rules['rfa_last_displacement'],
            aggregation_role_weight_semantics='weiszfeld_effective_coefficients' if deployed=='rfa_direct' else 'far_softmax' if deployed.startswith('far_') else 'uniform',
            far_alpha=actual_alpha, far_server_lr=lr, far_update_mode='single_step_gradient',
            far_server_clip_norm=radius, far_server_clip_rate=float((factors<1).float().mean()),
            far_score_mode='raw_distance' if deployed.startswith('far_') else 'not_applicable',
            far_max_weight=float(weights.max()), far_weight_l2_squared=float(weights.square().sum()),
            far_noise_amplification_vs_uniform=float(len(weights)*weights.square().sum()),
            far_weight_cap=float(config['kappa_w'])/len(weights),
            far_weight_cap_respected=bool(weights.max()<=float(config['kappa_w'])/len(weights)+1e-12),
            far_tilt_influence_certificate_claimed=False,
            far_attack_labels_visible_to_server_aggregate=False,
            far_attack_config_visible_to_server_aggregate=False,
            far_external_attack_diagnostics=True,
            far_external_attack_diagnostics_boundary='posthoc_simulator_only',
            ldp_gradient_far_effective_alpha=actual_alpha,
            ldp_gradient_far_reference_noise_aware=(deployed=='far_midpoint'),
            ldp_gradient_far_alpha_cap_exceeded=actual_alpha>float(result.metrics['ldp_gradient_far_alpha_cap'])+1e-12,
        )
        if deployed.startswith('far_'):
            distances=rules['distances'][deployed]
            diagnostics.update(far_reference_norm=float(torch.linalg.vector_norm(rules['references']['rfa' if deployed=='far_rfa' else 'midpoint'])),
                               far_score_span=float(distances.max()-distances.min()),far_logit_range=actual_alpha*float(distances.max()-distances.min()))
        else:
            diagnostics.update(far_reference_norm=None,far_score_span=None,far_logit_range=None)
        result.metrics.update(diagnostics)
        result.metrics['_far_external_weight_diagnostics_payload'] = dict(client_ids=ids,weights=weights.tolist(),contains_attack_labels=False)
        # Transient data flow out to an evaluator, never back to this instance.
        result.metrics['_rcig_evaluation_payload'] = dict(
            kind='aggregation_role_v1', round=round_num+1, client_ids=ids, server_clip_norm=radius,
            deployed=deployed, requested_arm=arm, ready=ready, vectors=x,
            public_noise_scales=[self._registered_noise_scale(i,config) for i in ids],
            rules=rules, contains_clean_data=False, contains_attack_labels=False,
            contains_realised_noise=False)
        return result
