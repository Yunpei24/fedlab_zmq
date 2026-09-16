"""Simulator-only gradual attack wrapper; never passed to an aggregator."""
from attacks.byzantine import apply_configured_attack, scheduled_attack_phase


def scheduled_attack(client_updates,config,*,round_num=None):
    output=apply_configured_attack(client_updates,config,round_num=round_num)
    if not config or scheduled_attack_phase(config,round_num)!="attack":return output
    ramp_end=config.get("ramp_end")
    if ramp_end is None:return output
    start=int(config["active_round_start"])
    if int(ramp_end)<start:raise ValueError("invalid ramp")
    fraction=min(1.,(int(round_num)+2-start)/(int(ramp_end)-start+1))
    if not 0<fraction<=1:raise ValueError("invalid ramp fraction")
    result=[]
    for (original,_,_),(attacked,meta,state) in zip(client_updates,output,strict=True):
        if meta.get("is_byzantine",False):
            vector={k:original[k]+fraction*(attacked[k]-original[k]) for k in original}
        else:vector=attacked
        result.append((vector,dict(meta,attack_ramp_fraction=fraction),state))
    return result
