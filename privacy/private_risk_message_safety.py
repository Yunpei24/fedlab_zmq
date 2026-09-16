"""Total numerical post-processing for already-parsed private message rows.

This is not gradient clipping and not a privacy mechanism. It prevents one
non-finite/out-of-domain vector or scalar report from crashing the aggregation.
Network authentication, shape validation and missing-client handling precede it.
"""
import torch


def sanitize(messages, reports=None):
    if messages.device.type!='mps' or messages.dtype!=torch.float32 or messages.ndim!=2 or min(messages.shape)<1:
        raise ValueError('Parsed fixed-shape MPS float32 message matrix required')
    valid=torch.isfinite(messages).all(1)&(messages.abs().amax(1)<=1e30*(1+1e-6))
    out=torch.where(valid[:,None],messages,torch.zeros_like(messages))
    normalized=None;bad_reports=0
    if reports is not None:
        if reports.device.type!='mps' or reports.dtype!=torch.float32 or reports.shape!=(len(messages),):
            raise ValueError('Parsed fixed-shape MPS report vector required')
        finite=torch.isfinite(reports)
        normalized=torch.where(finite&valid,reports.clamp(0,1),torch.zeros_like(reports))
        bad_reports=int((~finite).sum())
    return out,normalized,dict(invalid_message_rows=int((~valid).sum()),nonfinite_risk_reports=bad_reports,
        policy='invalid vector -> zero placeholder and risk zero; invalid report -> risk zero; finite reports -> [0,1]',
        numerical_coordinate_limit=1e30,cohort_size_preserved=True)
