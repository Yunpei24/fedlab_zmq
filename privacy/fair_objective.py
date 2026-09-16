"""Private gradients of a client-level quadratic risk, isolated research lane.

No existing FAR/DP implementation is changed. Losses are half-Brier in [0,1];
only the ERM baseline may use CE. Per-example gradients are clipped jointly.
The complete query is noised; treating its dependent weights as public would
underestimate sensitivity. Privacy: fixed-cardinality replace-one, fixed-WOR,
Wang/Balle/Kasiviswanathan (2019), generic Theorem 9, not Poisson accounting.
All tensor work is MPS-only. Scalar accounting uses ordinary host arithmetic.
"""
from collections import OrderedDict
from functools import lru_cache
import math
import os
import torch
from torch.func import functional_call, grad_and_value, vmap

ORDERS = tuple(range(2,65))
MODES = ('erm','naive','unbiased','per_example')


def require_mps():
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') != '0':
        raise RuntimeError('CPU fallback is forbidden')
    if not torch.backends.mps.is_available():
        raise RuntimeError('Local MPS is required; no CPU fallback')


def losses(logits, labels, kind='brier'):
    if kind == 'ce':
        return torch.nn.functional.cross_entropy(logits,labels,reduction='none')
    if kind != 'brier':
        raise ValueError(kind)
    p=logits.softmax(-1)
    target=(torch.arange(p.shape[-1],device=p.device)[None,:] == labels[:,None]).to(p.dtype)
    return .5*(p-target).square().sum(-1)


def per_example(model,x,y,*,kind='brier',clip_norm=.5,chunk_size=24):
    require_mps()
    if x.device.type!='mps' or y.device.type!='mps':
        raise ValueError('Private tensors must be on MPS')
    if not math.isfinite(clip_norm) or clip_norm<=0 or chunk_size<1 or len(x)==0:
        raise ValueError('Invalid clipping/chunk/input')
    if any(isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)) for m in model.modules()):
        raise ValueError('Batch-independent, deterministic model required')
    params=OrderedDict((k,v) for k,v in model.named_parameters() if v.requires_grad)
    buffers=OrderedDict(model.named_buffers())
    if any(v.device.type!='mps' for v in params.values()):
        raise ValueError('Model must be on MPS')
    def one(parameters,buffers_,xx,yy):
        output=functional_call(model,(parameters,buffers_),(xx.unsqueeze(0),))
        return losses(output,yy.unsqueeze(0),kind).sum()
    fn=vmap(grad_and_value(one),in_dims=(None,None,0,0),randomness='error')
    rs,gs,norms=[],[],[]
    raw_sum=torch.zeros(sum(p.numel() for p in params.values()),device='mps')
    for start in range(0,len(x),chunk_size):
        gd,rr=fn(params,buffers,x[start:start+chunk_size],y[start:start+chunk_size])
        flat=torch.cat([g.detach().flatten(1) for g in gd.values()],1)
        rr=rr.detach()
        nn=torch.linalg.vector_norm(flat,dim=1)
        if not bool(torch.isfinite(flat).all()) or not bool(torch.isfinite(rr).all()):
            raise FloatingPointError('Non-finite private gradient/loss')
        raw_sum+=flat.sum(0)
        gs.append(flat*(clip_norm/nn.clamp_min(1e-20)).clamp(max=1)[:,None])
        rs.append(rr);norms.append(nn)
    return torch.cat(rs),torch.cat(gs),torch.cat(norms),raw_sum/len(x)


def query(r,g,*,population_size,beta,mode):
    if mode not in MODES or beta<0 or not math.isfinite(beta):
        raise ValueError('Invalid objective')
    b=len(r)
    if g.ndim!=2 or r.ndim!=1 or len(g)!=b or not 2<=b<=population_size:
        raise ValueError('Require 2 <= batch <= public N, with matching tensors')
    if r.device.type!='mps' or g.device.type!='mps':
        raise ValueError('MPS-only query')
    if not bool(torch.isfinite(r).all()) or not bool(torch.isfinite(g).all()):
        raise FloatingPointError('Non-finite query input')
    if beta and (float(r.min()) < -1e-6 or float(r.max()) > 1+1e-6):
        raise ValueError('Fairness query requires a loss in [0,1], not raw CE')
    gbar=g.mean(0)
    if mode=='erm':
        if beta!=0:
            raise ValueError('ERM must have beta=0')
        return gbar
    if mode=='naive':
        return (1+beta*r.mean())*gbar
    if mode=='per_example':
        return ((1+beta*r)[:,None]*g).mean(0)
    diagonal=(r[:,None]*g).mean(0)
    cross=(r.sum()*g.sum(0)-b*diagonal)/(b*(b-1))
    return gbar+beta*(diagonal/population_size+(population_size-1)/population_size*cross)


def sensitivity(*,population_size,batch_size,clip_norm,beta,mode):
    if (mode not in MODES or not 2<=batch_size<=population_size or
        not math.isfinite(clip_norm) or clip_norm<=0 or not math.isfinite(beta) or beta<0):
        raise ValueError('Invalid sensitivity parameters')
    if mode=='erm':
        if beta!=0: raise ValueError('ERM beta must be zero')
        factor=2.
    elif mode=='naive': factor=2+3*beta
    elif mode=='per_example': factor=2*(1+beta)
    else: factor=2+beta*(3-1/population_size)
    return clip_norm/batch_size*factor


def _logadd(a,b):
    high=max(a,b)
    return high+math.log1p(math.exp(min(a,b)-high))


def wor_rdp(order,q,z):
    """Generic Theorem 9 with Gaussian epsilon(j)=j/(2*z**2).

    z = actual std / full replace-one query sensitivity. The potentially
    non-additive query is allowed. We do not use the sharper forward-difference
    formula, nor extrapolate a Poisson accountant to fixed batch sampling.
    """
    if int(order)!=order or order<2 or not 0<=q<=1 or not math.isfinite(z) or z<=0:
        raise ValueError('Invalid RDP arguments')
    if q==0:return 0.
    base=order/(2*z*z)
    if q==1:return base
    e2=1/(z*z)
    log_expm1=e2+math.log(-math.expm1(-e2))
    second=min(math.log(4)+log_expm1,math.log(2)+e2)
    total=0.
    for j in range(2,order+1):
        term=(math.lgamma(order+1)-math.lgamma(j+1)-math.lgamma(order-j+1)+j*math.log(q))
        term+=second if j==2 else math.log(2)+j*(j-1)/(2*z*z)
        total=_logadd(total,term)
    return min(base,total/(order-1))


def epsilon_bound(*,q,z,steps,delta,orders=ORDERS):
    if steps<0 or int(steps)!=steps or not 0<delta<1:raise ValueError('Invalid composition')
    if steps==0:return 0.,orders[0]
    choices=[(steps*wor_rdp(a,q,z)+math.log(1/delta)/(a-1),a) for a in orders]
    return min(choices)


@lru_cache(maxsize=128)
def calibrate(*,q,steps,epsilon,delta):
    if epsilon<=0 or steps<1 or not 0<q<=1:raise ValueError('Invalid calibration')
    low,high=.01,1.
    while epsilon_bound(q=q,z=high,steps=steps,delta=delta)[0]>epsilon:
        high*=2
        if high>1e6:raise RuntimeError('Calibration failed')
    for _ in range(70):
        mid=(low+high)/2
        if epsilon_bound(q=q,z=mid,steps=steps,delta=delta)[0]>epsilon:low=mid
        else:high=mid
    return high*(1+1e-8)


def release(qvec,*,noise_std,seed):
    """Research RNG. The seed is secret per paired simulation, not transmitted."""
    require_mps()
    if qvec.device.type!='mps' or noise_std<0 or not math.isfinite(noise_std):
        raise ValueError('Invalid release')
    if not bool(torch.isfinite(qvec).all()):raise FloatingPointError('Invalid query')
    if noise_std==0:return qvec.clone()
    state=torch.mps.get_rng_state()
    try:
        torch.mps.manual_seed(seed)
        return qvec+noise_std*torch.randn_like(qvec)
    finally:
        torch.mps.set_rng_state(state)
