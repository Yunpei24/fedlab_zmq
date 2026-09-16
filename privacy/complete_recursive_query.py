"""Per-example projection of the entire recursive query, with explicit radius."""
import math
import torch


def queries(current, previous, *, C, D, theta):
    """Return exact, difference-clipped, and whole-query-clipped MPS rows.

    This function emits nonprivate rows for a diagnostic. A deployment must add
    Gaussian noise to their batch mean, not release any of these raw matrices.
    """
    if not all(math.isfinite(v) for v in (C,D,theta)) or C<=0 or D<=0 or not 0<theta<=1:
        raise ValueError('Invalid public parameters')
    if current.device.type!='mps' or previous.device.type!='mps' or current.ndim!=2 or current.shape!=previous.shape or len(current)==0:
        raise ValueError('Matching nonempty per-example MPS matrices required')
    for rows in (current,previous):
        if not bool(torch.isfinite(rows).all()):raise ValueError('Nonfinite rows')
        if float(torch.linalg.vector_norm(rows,dim=1).max())>C*(1+2e-6):raise ValueError('Rows must have local per-example clipping')
    a=1-theta;S=theta*C+a*D
    diff=current-previous;dn=torch.linalg.vector_norm(diff,dim=1)
    clipped_diff=diff*(D/dn.clamp_min(1e-20)).clamp(max=1)[:,None]
    old=theta*current+a*clipped_diff
    # Use the exact recursion identity, not an aggregate/batch clipping.
    exact=current-a*previous;en=torch.linalg.vector_norm(exact,dim=1)
    projected=exact*(S/en.clamp_min(1e-20)).clamp(max=1)[:,None]
    return exact,old,projected,dict(effective_C=S,query_sensitivity=2*S/len(current),
        difference_clipped_count=int((dn>D).sum()),complete_query_clipped_count=int((en>S).sum()),batch_size=len(current))
