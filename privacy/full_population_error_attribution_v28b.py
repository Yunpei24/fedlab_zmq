"""Post-hoc MPS diagnostic only: never a training or private-release mechanism."""
import itertools
import math
import torch
from privacy.fair_objective import require_mps

NAMES=('clipping','private_risk_weights','rfa_geometry','effective_noise','numerical_solver')


def decompose(raw,clipped,messages,applied,oracle_weights,received_weights,effective_weights,*,objective_scale,eta):
    require_mps()
    tensors=(raw,clipped,messages,applied,oracle_weights,received_weights,effective_weights)
    if any(t.device.type!='mps' or t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError('Finite float32 MPS tensors required')
    if (raw.ndim!=2 or min(raw.shape)<1 or clipped.shape!=raw.shape or messages.shape!=raw.shape
            or applied.shape!=(raw.shape[1],) or any(w.shape!=(len(raw),) for w in tensors[-3:])):
        raise ValueError('Inconsistent cohort dimensions')
    if not all(math.isfinite(v) and v>0 for v in (objective_scale,eta)):
        raise ValueError('Positive finite objective scaling and public step required')
    for w in tensors[-3:]:
        if float(w.min())<=0 or abs(float(w.sum())-1)>2e-6:
            raise ValueError('Strictly positive normalized weights required')
    target=(oracle_weights[:,None]*raw).sum(0)
    gradient=objective_scale*target
    components=dict(zip(NAMES,(
        (oracle_weights[:,None]*(clipped-raw)).sum(0),
        ((received_weights-oracle_weights)[:,None]*clipped).sum(0),
        ((effective_weights-received_weights)[:,None]*clipped).sum(0),
        (effective_weights[:,None]*(messages-clipped)).sum(0),
        applied-(effective_weights[:,None]*messages).sum(0))))
    error=applied-target
    total=sum(components.values(),torch.zeros_like(applied))
    torch.testing.assert_close(error,total,rtol=6e-5,atol=4e-7)
    squares={name:float(v.square().sum()) for name,v in components.items()}
    cross={f'{a}__{b}':float(2*torch.dot(components[a],components[b])) for a,b in itertools.combinations(NAMES,2)}
    error_sq=float(error.square().sum())
    if not math.isclose(error_sq,sum(squares.values())+sum(cross.values()),rel_tol=6e-5,abs_tol=4e-7):
        raise AssertionError('Squared decomposition does not close')
    penalties={name:float(-eta*torch.dot(gradient,v)) for name,v in components.items()}
    ideal_gain=float(eta*torch.dot(gradient,target))
    applied_gain=float(eta*torch.dot(gradient,applied))
    if not math.isclose(ideal_gain-applied_gain,sum(penalties.values()),rel_tol=6e-5,abs_tol=4e-7):
        raise AssertionError('First-order decomposition does not close')
    return dict(squared_error=error_sq,squared_components=squares,doubled_cross_products=cross,
                target_norm=float(torch.linalg.vector_norm(target)),objective_gradient_norm=float(torch.linalg.vector_norm(gradient)),
                objective_scale=objective_scale,eta=eta,ideal_predicted_gain=ideal_gain,
                applied_predicted_gain=applied_gain,first_order_gain_loss=penalties,
                vector_identity_max_error=float((error-total).abs().max()),
                squared_identity_error=abs(error_sq-sum(squares.values())-sum(cross.values())),
                directional_identity_error=abs(ideal_gain-applied_gain-sum(penalties.values())),
                independent_or_centered_errors_assumed=False,privacy_protected=False,feeds_mechanism=False),target,components
