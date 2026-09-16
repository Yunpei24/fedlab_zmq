"""MPS-only research oracles. None of these validation gradients enter a server."""
import math
import statistics as st
import torch
from privacy.fair_objective import require_mps, losses


def require_vector(x):
    if x.device.type != 'mps' or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite MPS tensor required')


def flat_parameters(model):
    p = torch.cat([x.detach().flatten() for x in model.parameters() if x.requires_grad])
    require_vector(p)
    return p.clone()


@torch.no_grad()
def set_parameters(model, flat):
    require_vector(flat)
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.device.type != 'mps':
                raise ValueError('MPS model required')
            n = p.numel()
            p.copy_(flat[offset:offset+n].reshape_as(p))
            offset += n
    assert offset == len(flat)


def mean_brier_gradient(model, x, y, batch_size=256):
    require_mps()
    if x.device.type != 'mps' or y.device.type != 'mps' or not len(y):
        raise ValueError('Nonempty MPS examples required')
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if any(p.device.type != 'mps' for p in params):
        raise ValueError('MPS model required')
    g = torch.zeros(sum(p.numel() for p in params), device='mps')
    risk = 0.
    for start in range(0, len(y), batch_size):
        value = losses(model(x[start:start+batch_size]), y[start:start+batch_size], 'brier').sum()/len(y)
        grads = torch.autograd.grad(value, params)
        g += torch.cat([a.detach().flatten() for a in grads])
        risk += float(value.detach())
    require_vector(g)
    return risk, g


def clip_rows(g, C):
    require_vector(g)
    if g.ndim != 2 or C <= 0 or not math.isfinite(C):
        raise ValueError('Matrix and finite positive radius required')
    return g * (C/torch.linalg.vector_norm(g, dim=1).clamp_min(1e-20)).clamp(max=1)[:, None]


def directional_metrics(u, client_gradients, risks, hard_ids):
    require_vector(u)
    require_vector(client_gradients)
    rr = torch.tensor(risks, device='mps')
    dot = client_gradients @ u
    norm = float(torch.linalg.vector_norm(u))
    gbar = client_gradients.mean(0)
    gj = ((1+2*rr)[:, None]*client_gradients).mean(0)
    gh = client_gradients[hard_ids].mean(0)
    def score(g):
        gn = float(torch.linalg.vector_norm(g))
        d = float(torch.dot(g, u))
        return dict(predicted_loss_gain=d, gain_per_unit_step=d/norm if norm > 1e-12 else None,
                    cosine=max(-1., min(1., d/(gn*norm))) if gn*norm > 1e-20 else None)
    return dict(step_norm=norm, all_clients=score(gbar), equitable_J2=score(gj),
                fixed_hard_clients=score(gh), per_client_predicted_gain=dot.cpu().tolist())


def actual_metrics(before, after, hard_ids, predicted):
    r0 = [x['brier_loss'] for x in before['clients']]
    r1 = [x['brier_loss'] for x in after['clients']]
    gain = [a-b for a, b in zip(r0, r1)]
    return dict(
        per_client_loss_gain=gain,
        per_client_first_order_remainder=[p-g for p, g in zip(predicted['per_client_predicted_gain'], gain)],
        mean_loss_gain=st.mean(gain), fixed_hard_loss_gain=st.mean(gain[i] for i in hard_ids),
        J2_gain=st.mean(a+a*a-b-b*b for a, b in zip(r0, r1)),
        accuracy_gain_pp=after['accuracy_pct']-before['accuracy_pct'],
        worst20_gain_pp=after['worst20_pct']-before['worst20_pct'],
        fixed_hard_accuracy_gain_pp=100*st.mean(after['clients'][i]['accuracy']-before['clients'][i]['accuracy'] for i in hard_ids),
        gap_change_pp=after['gap_best20_worst20_pp']-before['gap_best20_worst20_pp'],
        variance_change_pp2=after['variance_pp2']-before['variance_pp2'])
