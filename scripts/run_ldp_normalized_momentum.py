#!/usr/bin/env python3
"""18 independent bias-corrected momentum runs, historical controls read-only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import itertools
import json
import math
import os
import subprocess
import threading
import time
import yaml
from scripts import run_ldp_client_momentum as historical
from scripts.run_rcig_batch_screen import file_hash, canonical_hash, source_closure, require_working_mps
from scripts.run_rcig_n10_policy_ablation import privacy

CAMPAIGN="normalized_momentum_n10_v1"
MATRIX=ROOT/"configs/ldp_gradient_far"/(CAMPAIGN+".yaml")
OUTPUT=ROOT/"results/ldp_gradient_far"/CAMPAIGN
ENTRY=Path(__file__).resolve()
STOP=threading.Event()
require=historical.require
save=historical.save


def matrix():
    m=yaml.safe_load(MATRIX.read_text())
    require(m["campaign_id"]==CAMPAIGN and m["expected_runs"]==18 and m["betas"]==[.9],"grid mismatch")
    require(len(tasks(m))==18,"wrong number of tasks")
    return m


def tasks(m,noise=None):
    return historical.tasks(m,noise)


def directory(t):return OUTPUT/historical.run_id(t)


def audit_controls():
    m=historical.matrix(); stamp=json.loads((historical.OUTPUT/"campaign_lock.json").read_text())
    historical.verify(stamp,m)
    hashes={}
    for t in historical.tasks(m):
        evidence=historical.validate(m,t,historical.config_for(m,t,stamp),stamp)
        dest=historical.directory(t); status=json.loads((dest/"orchestration_status.json").read_text())
        require(status["status"]=="completed" and status["metrics_sha256"]==evidence["metrics_sha256"] and status["trace_sha256"]==evidence["trace_sha256"],"invalid historical control")
        for p in [ROOT/evidence["metrics_path"],dest/"simulator_randomness_private_audit.jsonl",dest/"resolved_config.yaml",dest/"orchestration_status.json",dest/"runtime_imports.json"]:
            hashes[str(p.relative_to(ROOT))]=file_hash(p)
    return hashes


def stamp_for(m):
    return dict(campaign=CAMPAIGN,matrix_hash=file_hash(MATRIX),protocol_hash=file_hash(ROOT/m["protocol"]),
        privacy=privacy(m),sources=source_closure((ENTRY,ROOT/"run_experiment.py")),
        historical_artifacts=audit_controls(),expected_runs=18,private_device="mps",momentum_device="mps",server_device="cpu_float64",fallback=0)


def verify(stamp,m):
    require(file_hash(MATRIX)==stamp["matrix_hash"] and file_hash(ROOT/m["protocol"])==stamp["protocol_hash"],"protocol drift")
    changed=[p for p,h in {**stamp["sources"],**stamp["historical_artifacts"]}.items() if file_hash(ROOT/p)!=h]
    require(not changed,"locked source/control changed: "+", ".join(changed))


def config_for(m,t,stamp):
    cfg=historical.config_for(m,t,stamp)
    cfg["output_dir"]=str(directory(t))
    cfg["training"]["algo_config"].update(momentum_campaign=CAMPAIGN,
        momentum_initialization="zero_buffer_with_bias_correction",
        momentum_runtime_implementation="algorithms.ldp_normalized_momentum.LDPNormalizedMomentum")
    return cfg


def trace_for(t):
    rows=[json.loads(line) for line in (directory(t)/"simulator_randomness_private_audit.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r:(r["round"],r["client_id"]))
    require(len(rows)==400 and {(r["round"],r["client_id"]) for r in rows}==set(itertools.product(range(1,41),range(10))),"incomplete trace")
    previous={}
    for r in rows:
        require(r["momentum_device"].startswith("mps") and r["beta"]==.9 and r["initialization"]=="zero_buffer_with_bias_correction","wrong client path")
        require(r["memory_kind"]=="unnormalized_z" and r["memory_before"]==previous.get(r["client_id"]),"lost normalized EMA memory")
        require(abs(r["normalization_denominator"]-(1-.9**r["round"]))<1e-14,"incorrect denominator")
        if r["round"]==1:require(r["private_upload"]==r["raw_private_gradient"],"first upload not identical")
        previous[r["client_id"]]=r["memory_after"]
    return rows


def check_pairing(t):
    rows=trace_for(t)
    for beta in (0.,.9):
        for a,b in zip(rows,historical.trace_for(dict(t,beta=beta)),strict=True):
            require(a["permutations"]==b["permutations"] and a["standard_gaussians"]==b["standard_gaussians"],"actual draws differ from historical control")
            if a["round"]==1:
                require(a["model_before"]==b["model_before"] and a["private_upload"]==b["private_upload"],"first-round control mismatch")


def validate(m,t,cfg,stamp):
    dest=directory(t);paths=list(dest.rglob("metrics.json"));require(len(paths)==1,"missing/ambiguous metrics")
    p=paths[0];x=json.loads(p.read_text());rows=x["rounds"]
    require(yaml.safe_load((dest/"resolved_config.yaml").read_text())==cfg,"saved config changed")
    require(len(rows)==40 and x["summary"]["seed"]==t["seed"],"wrong/incomplete run")
    for k,v in cfg["training"]["algo_config"].items():require(x["config"].get(k)==v,"runtime config mismatch: "+k)
    for k,r in enumerate(rows,1):
        require(r["round_num"]==k and r["num_alive_clients"]==r["num_evaluated_clients"]==10,"wrong cohort")
        require(r["momentum_arm"]==t["arm"] and r["client_momentum_beta"]==.9 and r["momentum_history_length"]==k,"wrong rule/state")
        require(r["momentum_private_gradient_device"]==r["momentum_buffer_device"]=="mps","CPU private step")
        require(r["momentum_initialization"]=="zero_buffer_with_bias_correction" and r["momentum_warmup_rounds"]==0,"wrong initialization/warmup")
        factor=(1-.9)/(1+.9)*(1+.9**k)/(1-.9**k)
        require(abs(r["momentum_linear_noise_factor"]-factor)<1e-12,"incorrect nominal noise factor")
        require(r["privacy_sampling_scheme"]=="fixed_without_replacement" and r["privacy_adjacency"]=="replace_one","DP mismatch")
        require(r["far_server_clip_norm"]==16 and r["far_alpha"]==cfg["training"]["algo_config"]["far_alpha"],"clipping/alpha changed")
        require(r["momentum_oracle_evaluation_only"] and not r["momentum_no_clip_counterfactual_is_end_to_end"],"oracle boundary declaration mismatch")
        for key in ("test_accuracy","test_loss","client_loss_mean","client_accuracy_variance_pct2","worst20_accuracy_pct","best20_worst20_gap_pct","mean_client_balanced_accuracy_pct",
                    "momentum_oracle_applied_mse_current_clean_mean","momentum_oracle_filtered_noise_energy","momentum_oracle_clean_lag_energy",
                    "momentum_oracle_no_clip_same_messages_mse","momentum_oracle_clip_aggregate_displacement","momentum_oracle_cumulative_applied_error_squared"):
            require(isinstance(r.get(key),(float,int)) and math.isfinite(r[key]),"missing/nonfinite "+key)
        if r["far_server_clip_rate"]==0:
            require(r["momentum_oracle_clip_aggregate_displacement"]<1e-20,"inactive clipping changed aggregate")
    require(abs(rows[-1]["privacy_epsilon_max"]-4)<1e-4,"wrong epsilon")
    runtime=json.loads((dest/"runtime_imports.json").read_text())
    require(runtime["stage"]=="after_training" and runtime["fallback"]==0,"runtime incomplete")
    require(all(stamp["sources"].get(p)==h for p,h in runtime["sources"].items()),"unlocked imported source")
    check_pairing(t)
    return dict(metrics_path=str(p.relative_to(ROOT)),metrics_sha256=file_hash(p),trace_sha256=file_hash(dest/"simulator_randomness_private_audit.jsonl"),
        validated_rounds=40,validated_client_records=400,paired_historical_betas=[0.,.9],epsilon_max=rows[-1]["privacy_epsilon_max"])


def worker(cp,dest):
    require_working_mps()
    from algorithms.base import register_algorithm
    from algorithms.ldp_normalized_momentum import LDPNormalizedMomentum,NormalizedMomentumEvaluator
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    from scripts.run_rcig_batch_screen_experiment import runtime_source_hashes
    import run_experiment as harness
    register_algorithm("ldp_gradient_far")(LDPNormalizedMomentum)
    old_detach,old_eval,oldargv=harness.detach_rcig_evaluation_oracles,harness.rcig_reference_oracle_metrics,sys.argv
    def detach(updates,**kwargs):
        safe,clean=old_detach(updates,enabled=True,strip_attack_oracles=True)
        for _,metadata,_ in safe:metadata.pop("momentum_raw_private_gradient_oracle",None)
        return safe,clean
    harness.detach_rcig_evaluation_oracles=detach;harness.rcig_reference_oracle_metrics=NormalizedMomentumEvaluator()
    manifest=dict(stage="before_training",sources=runtime_source_hashes(),private_device="mps",momentum_device="mps",server_device="cpu_float64",fallback=0,algorithm=CAMPAIGN)
    save(dest/"runtime_imports.json",manifest)
    try:
        with (dest/"simulator_randomness_private_audit.jsonl").open("x") as trace:
            def sink(row):trace.write(json.dumps(row,sort_keys=True)+"\n");trace.flush()
            LDPNormalizedMomentum.audit_sink=staticmethod(sink)
            sys.argv=[str(ROOT/"run_experiment.py"),"--config",str(cp),"--output",str(dest),"--device","mps"]
            with evaluation_rng_isolation(harness):harness.main()
        require(all(file_hash(ROOT/p)==h for p,h in manifest["sources"].items()),"runtime source drift")
        manifest.update(stage="after_training",sources=runtime_source_hashes());save(dest/"runtime_imports.json",manifest)
    finally:
        LDPNormalizedMomentum.audit_sink=None
        harness.detach_rcig_evaluation_oracles,harness.rcig_reference_oracle_metrics,sys.argv=old_detach,old_eval,oldargv


def run_one(m,t,stamp):
    if STOP.is_set():return
    verify(stamp,m);dest=directory(t);cfg=config_for(m,t,stamp);sp=dest/"orchestration_status.json"
    if dest.exists():
        require(sp.exists(),"partial directory requires inspection")
        old=json.loads(sp.read_text());require(old["status"]=="completed","partial/failed run requires inspection")
        ev=validate(m,t,cfg,stamp)
        require(old["metrics_sha256"]==ev["metrics_sha256"] and old["trace_sha256"]==ev["trace_sha256"],"completed artifact changed")
        return
    dest.mkdir(parents=True);cp=dest/"resolved_config.yaml";cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
    state=dict(status="starting",task=t,supervisor_pid=os.getpid(),started_at=time.time());save(sp,state)
    try:
        with (dest/"training.log").open("x") as log:
            p=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--worker",str(cp),"--output",str(dest)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
            state.update(status="running",worker_pid=p.pid);save(sp,state);print("START",historical.run_id(t),p.pid,flush=True)
            require(p.wait()==0,"worker failed: "+str(dest/"training.log"))
        verify(stamp,m);state.update(validate(m,t,cfg,stamp),status="completed",completed_at=time.time());save(sp,state)
        print("COMPLETE",historical.run_id(t),flush=True)
    except BaseException as exc:
        STOP.set();state.update(status="failed",error=str(exc),stopped_at=time.time());save(sp,state);raise


def status(m):
    out=dict(completed=0,total=18,active=[],failed=[],missing=0)
    for t in tasks(m):
        p=directory(t)/"orchestration_status.json"
        if not p.exists():out["missing"]+=1;continue
        st=json.loads(p.read_text())
        if st["status"]=="completed":out["completed"]+=1
        else:out["active" if st["status"] in {"running","starting"} else "failed"].append(st)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for mode in ("plan","status","launch","run","audit"):g.add_argument("--"+mode,action="store_true")
    g.add_argument("--worker",type=Path);p.add_argument("--output",type=Path);p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.worker:
        require(args.output is not None,"worker output required");worker(args.worker,args.output);return
    m=matrix()
    if args.status:print(json.dumps(status(m),indent=2));return
    if args.plan:
        st=stamp_for(m);print(json.dumps(dict(total=18,tasks=tasks(m),privacy=st["privacy"],historical_artifacts=len(st["historical_artifacts"]),sources=len(st["sources"])),indent=2));return
    require(args.resume or args.audit,"--resume required")
    if args.launch:
        require_working_mps();historical.no_competitor()
        log=ROOT/"logs/ldp_gradient_far"/(CAMPAIGN+".log");log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a") as stream:
            child=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--run","--resume"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
        print(json.dumps(dict(supervisor_pid=child.pid,log=str(log))));return
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT.parent/("."+CAMPAIGN+".lock")).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.audit:require_working_mps();historical.no_competitor()
        OUTPUT.mkdir(exist_ok=True);frozen=OUTPUT/"campaign_lock.json"
        if frozen.exists():stamp=json.loads(frozen.read_text());verify(stamp,m)
        else:
            require(not args.audit,"campaign not yet launched");stamp=stamp_for(m);save(frozen,stamp)
        if not args.audit:
            def lane(noise):
                for t in tasks(m,noise):
                    if STOP.is_set():break
                    run_one(m,t,stamp)
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(lane,noise) for noise in m["noise_regimes"]]
                for f in fs:f.result()
        verify(stamp,m)
        for t in tasks(m):validate(m,t,config_for(m,t,stamp),stamp)
        save(OUTPUT/"completion_evidence.json",dict(status="completed",completed=18,total=18,finished_at=time.time(),
            historical_controls_unchanged=True,paired_to_both_historical_initializations=True,
            scientific_verdict="await_model_quality_analysis_no_automatic_promotion"))
        print(json.dumps(status(m),indent=2))


if __name__=="__main__":main()
