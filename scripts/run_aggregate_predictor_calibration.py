#!/usr/bin/env python3
"""Two MPS lanes: 38 clean calibrations, frozen radii, 216 recovery evaluations."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import yaml
from scripts import run_aggregate_control_e2e as pilot
from scripts import run_ldp_aggregation_role_ablation as historical
from scripts import run_rcig_batch_screen as shared
from scripts.run_rcig_ldp_gradient_far_v2 import _metrics_path,_post_run_device_audit

CAMPAIGN="aggregate_predictor_calibration_v1"
ENTRY=Path(__file__).resolve()
MATRIX=ROOT/f"configs/ldp_gradient_far/{CAMPAIGN}.yaml"
OUTPUT=ROOT/f"results/ldp_gradient_far/{CAMPAIGN}"
LOG=ROOT/f"logs/{CAMPAIGN}_mps.log"
ARTIFACT=OUTPUT/"frozen_radii.json"
require=pilot.require;hash_file=shared.file_hash
WRITE_LOCK=threading.Lock()
STOP=threading.Event()


def write_json(path,data):
    """Atomic visibility to concurrent lanes/status readers."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:json.dump(data,stream,indent=2,sort_keys=True,allow_nan=False)
    os.replace(temporary,path)


def matrix():
    m=yaml.safe_load(MATRIX.read_text())
    from algorithms.aggregate_predictor_control import POLICIES
    require(m["campaign_id"]==CAMPAIGN and m["arms"]==list(POLICIES),"campaign identity changed")
    require(m["calibration_seeds"]==list(range(931001,931020)) and m["evaluation_seeds"]==[932001,932002,932003],"frozen seeds changed")
    require(m["rounds"]==40 and m["warmup"]==12 and m["max_parallel_workers"]==2,"execution contract changed")
    require(m["noise_regimes"]==["homogeneous","heteroscedastic"],"noise regime changed")
    require(m["calibration"]["target_trajectory_exceedance"]==.05 and m["calibration"]["order_statistic"]==19,"calibration rank changed")
    require(list(m["scenarios"])==["clean","brutal_bf","slow_ipm","persistent_alie"],"scenario grid changed")
    return m


def tasks(m,phase,noise=None):
    noises=m["noise_regimes"] if noise is None else [noise]
    if phase=="calibration":
        return [dict(phase=phase,seed=s,noise=n,scenario="clean",arm="far_rfa")
                for n,s in itertools.product(noises,m["calibration_seeds"])]
    return [dict(phase=phase,seed=s,noise=n,scenario=c,arm=a)
            for n,s,c,a in itertools.product(noises,m["evaluation_seeds"],m["scenarios"],m["arms"])]


def run_id(t):return f"{t['noise']}__seed{t['seed']}__{t['scenario']}__{t['arm']}"
def directory(t):return OUTPUT/t["phase"]/run_id(t)


def provenance(m):
    inherited=pilot.provenance(pilot.matrix())
    return dict(campaign=CAMPAIGN,inherited=inherited,privacy=inherited["privacy"],
        files={str(p.relative_to(ROOT)):hash_file(p) for p in (MATRIX,ROOT/m["protocol"])},
        sources=shared.source_closure((ENTRY,ROOT/"algorithms/aggregate_predictor_control.py",ROOT/"run_experiment.py")),
        counts=dict(calibration=38,evaluation=216),max_parallel_mps_workers=2)


def verify(stamp):
    pilot.verify(stamp["inherited"])
    for key in ("files","sources"):
        for path,digest in stamp[key].items():require(hash_file(ROOT/path)==digest,"source drift: "+path)


def load_radii(stamp):
    require(ARTIFACT.exists(),"calibration incomplete: no frozen radii")
    artifact=json.loads(ARTIFACT.read_text())
    require(artifact["matrix_hash"]==stamp["files"][str(MATRIX.relative_to(ROOT))],"radii matrix mismatch")
    for name,h in artifact["calibration_metric_hashes"].items():require(hash_file(ROOT/name)==h,"calibration data drift")
    return artifact


def config_for(m,t,stamp,artifact=None):
    require(t in tasks(m,t["phase"]),"undeclared task")
    a0="rfa_direct" if t["arm"]=="rfa_direct" else "far_rfa"
    cfg=historical.config_for(historical.matrix(),dict(t,scenario="none",arm=a0),stamp)
    cfg.update(seed=t["seed"],device="mps",output_dir=str(directory(t)))
    cfg["data"]["partition_seed"]=t["seed"]
    a=cfg["training"]["algo_config"]
    a.update(aggregation_role_campaign=CAMPAIGN,rcig_campaign_id=CAMPAIGN,
        rcig_n10_pairing_seed=t["seed"],rcig_n10_phase=t["phase"],rcig_n10_scenario=t["scenario"],
        rcig_n10_runtime_implementation="algorithms.aggregate_predictor_control.AggregatePredictorControl",
        apc_phase=t["phase"],apc_arm=t["arm"],apc_matrix_hash=stamp["files"][str(MATRIX.relative_to(ROOT))],
        apc_frozen_radii=None if artifact is None else artifact["radii"][t["noise"]],
        apc_radius_artifact_sha256=None if artifact is None else hash_file(ARTIFACT),
        num_byzantine=2)
    scenario=m["scenarios"][t["scenario"]]
    a["attack"]=dict(enabled=t["scenario"]!="clean",name=scenario["name"],scale=scenario["scale"],
        num_byzantine=2,client_ids=[0,1])
    if t["scenario"]!="clean":a["attack"].update(active_round_start=scenario["start"],active_round_end=scenario["end"])
    if "ramp_end" in scenario:a["attack"]["ramp_end"]=scenario["ramp_end"]
    require(a["far_alpha"]==.1 and a["far_server_clip_norm"]==16 and a["clip_norm"]==4 and a["fixed_steps_per_round"]==1,"gradient channel changed")
    require(a["fixed_batch_size"]==120 and a["privacy_num_rounds"]==40 and a["privacy_public_dataset_size"]==6000,"private sampling changed")
    require(a["privacy_adjacency"]=="replace_one" and a["sampling_scheme"]=="fixed_without_replacement","accountant mismatch")
    if t["phase"]=="evaluation":require(artifact is not None,"evaluation cannot open before calibration")
    return cfg


def trace_for(t):
    data=sorted((json.loads(line) for line in (directory(t)/"simulator_randomness_private_audit.jsonl").read_text().splitlines()),key=lambda r:(r["round"],r["client_id"]))
    require(len(data)==400 and {(r["round"],r["client_id"]) for r in data}==set(itertools.product(range(1,41),range(10))),"incomplete actual-draw trace")
    return data


def validate(m,t,cfg,stamp):
    dest=directory(t);p=_metrics_path(dest)
    require(p is not None,"no unique completed metrics: "+run_id(t))
    x=json.loads(p.read_text());rows=x["rounds"];s=x["summary"]
    require(len(rows)==40 and (s["seed"],s["num_clients"],s["num_rounds"],s["dataset"],s["model"])==(t["seed"],10,40,"fashionmnist","lenet5_tanh"),"trajectory identity mismatch")
    require(yaml.safe_load((dest/"resolved_config.yaml").read_text())==cfg,"saved config drift")
    for key,value in cfg["training"]["algo_config"].items():require(x["config"].get(key)==value,"runtime config mismatch: "+key)
    _post_run_device_audit(p,cfg)
    spec=m["scenarios"][t["scenario"]]
    for k,r in enumerate(rows,1):
        require(r["round_num"]==k and r["num_alive_clients"]==10,"cohort/round changed")
        require(r["apc_phase"]==t["phase"] and r["apc_arm"]==t["arm"],"wrong applied rule")
        require(r["apc_history_observations"]==k-1 and r["apc_predictors_strictly_past"] and not r["apc_current_round_used_for_predictor"],"future leakage")
        require(not r["apc_oracle_used_for_deployment"] and r["apc_same_cohort_verified"],"oracle/current-cohort boundary failed")
        require(r["apc_fixed_honest_count"]==8 and r["num_evaluated_clients"]==8,"honest target/evaluation cohort changed")
        active=t["scenario"]!="clean" and spec["start"]<=k<=spec["end"]
        require(r["apc_attack_active"]==active,"attack chronology mismatch")
        for key in ("test_accuracy","test_loss","client_loss_mean","client_accuracy_variance_pct2","worst20_accuracy_pct","best20_worst20_gap_pct","apc_applied_mse","apc_raw_far_mse"):
            require(isinstance(r.get(key),(float,int)) and math.isfinite(r[key]),"nonfinite/missing "+key)
        if k>12:
            for pred in ("ema","rcig"):
                for metric in ("residual_l2","residual_mahalanobis","variance_trace","predictor_mse"):
                    value=r.get(f"apc_{pred}_{metric}");require(isinstance(value,(float,int)) and math.isfinite(value),"missing calibration statistic")
        if k<=12 or t["arm"]=="far_rfa":require(not r["apc_correction_triggered"],"identity path modified")
    require(abs(rows[-1]["privacy_epsilon_max"]-4)<1e-4,"wrong privacy budget")
    runtime=json.loads((dest/"runtime_imports.json").read_text())
    require(runtime["stage"]=="after_training" and runtime["mps_fallback"]==0,"incomplete runtime audit")
    require(all(stamp["sources"].get(name)==h for name,h in runtime["source_sha256"].items()),"unlocked imported source")
    trace=trace_for(t);paired=[]
    if t["phase"]=="evaluation":
        clean=dict(t,scenario="clean")
        baseline=dict(t,scenario="clean",arm="far_rfa")
        for other in (baseline,clean):
            if other==t or other in paired:continue
            sp=directory(other)/"orchestration_status.json"
            require(sp.exists() and json.loads(sp.read_text())["status"]=="completed","counterpart must complete first")
            for r,q in zip(trace,trace_for(other),strict=True):
                require(r["permutations"]==q["permutations"] and r["standard_gaussians"]==q["standard_gaussians"],"unpaired actual draws")
                if other["arm"]==t["arm"]:
                    end=spec["start"] if t["scenario"]!="clean" else 13
                else:end=13
                if r["round"]<=end:require(r["model_before"]==q["model_before"] and r["private_upload"]==q["private_upload"],"pre-intervention model/upload mismatch")
            paired.append(other)
    return p,x,dict(actual_trace_records=400,paired_to=paired)


def calibration_threshold(values,alpha=.05):
    require(len(values)==19 and all(math.isfinite(v) and v>0 for v in values),"nineteen finite positive trajectory scores required")
    k=math.ceil((len(values)+1)*(1-alpha))
    require(k<=len(values),"no finite rank threshold")
    return sorted(values)[k-1]


def calibrate(m,stamp):
    hashes={};radii={};scores={}
    for noise in m["noise_regimes"]:
        radii[noise]={};scores[noise]={}
        records=[]
        for t in tasks(m,"calibration",noise):
            cfg=config_for(m,t,stamp);p,x,_=validate(m,t,cfg,stamp)
            status=json.loads((directory(t)/"orchestration_status.json").read_text())
            require(status["status"]=="completed" and status["metrics_sha256"]==hash_file(p),"unvalidated calibration run")
            hashes[str(p.relative_to(ROOT))]=hash_file(p);records.append((t,x))
        for pred in m["predictors"]:
            radii[noise][pred]={};scores[noise][pred]={}
            for geometry,metric in (("isotropic","residual_l2"),("mahalanobis","residual_mahalanobis")):
                # Only private-transcript nonconformity scores. No oracle/MSE/accuracy.
                values=[max(row[f"apc_{pred}_{metric}"] for row in x["rounds"][12:]) for _,x in records]
                radii[noise][pred][geometry]=calibration_threshold(values)
                scores[noise][pred][geometry]=values
    # Actual same standard normals and batch permutations across noise regimes.
    for seed in m["calibration_seeds"]:
        a=dict(phase="calibration",seed=seed,noise="homogeneous",scenario="clean",arm="far_rfa")
        b=dict(a,noise="heteroscedastic")
        for x,y in zip(trace_for(a),trace_for(b),strict=True):
            require(x["permutations"]==y["permutations"] and x["standard_gaussians"]==y["standard_gaussians"],"regime calibration pairing failed")
    artifact=dict(campaign=CAMPAIGN,matrix_hash=stamp["files"][str(MATRIX.relative_to(ROOT))],radii=radii,
        per_trajectory_maxima=scores,calibration_metric_hashes=hashes,n_trajectories_per_regime=19,
        quantile_rank=19,target_exceedance=.05,oracle_consumed=False,test_accuracy_consumed=False,
        guarantee_scope="marginal_first_trigger_per_policy_under_exchangeable_trajectories_not_conditional_95CI",
        evaluation_seeds_opened=False)
    if ARTIFACT.exists():require(json.loads(ARTIFACT.read_text())==artifact,"frozen radii changed")
    else:write_json(ARTIFACT,artifact)
    print("CALIBRATION FROZEN: "+json.dumps(radii),flush=True)
    return artifact


def recovery_delay(rows,clean_rows,end,m):
    """First 3-round joint utility recovery, relative to same-arm clean path."""
    ev=m["evaluation"];patience=ev["recovery_consecutive_rounds"]
    good=[]
    for r,c in zip(rows,clean_rows,strict=True):
        good.append(100*(c["test_accuracy"]-r["test_accuracy"])<=ev["recovery_accuracy_tolerance_pp"]
            and c["worst20_accuracy_pct"]-r["worst20_accuracy_pct"]<=ev["recovery_worst20_tolerance_pp"]
            and r["best20_worst20_gap_pct"]-c["best20_worst20_gap_pct"]<=ev["recovery_gap_tolerance_pp"])
    for offset in range(end,len(rows)-patience+1):
        if all(good[offset:offset+patience]):return offset+1-end
    return None


def summarize(m):
    rows=[]
    for phase in ("calibration","evaluation"):
        for t in tasks(m,phase):
            sp=directory(t)/"orchestration_status.json"
            if not sp.exists() or json.loads(sp.read_text()).get("status")!="completed":continue
            payload=json.loads(_metrics_path(directory(t)).read_text());rs=payload["rounds"];last=rs[-1]
            r=dict(t,**{k:last[k] for k in ("test_accuracy","test_loss","client_loss_mean","client_accuracy_variance_pct2","worst20_accuracy_pct","best20_worst20_gap_pct")})
            r["mean_applied_mse"]=statistics.mean(z["apc_applied_mse"] for z in rs[12:])
            r["any_correction"]=any(z["apc_correction_triggered"] for z in rs[12:])
            r["correction_fraction"]=statistics.mean(z["apc_correction_triggered"] for z in rs[12:])
            if t["scenario"]!="clean":
                clean=json.loads(_metrics_path(directory(dict(t,scenario="clean"))).read_text())["rounds"]
                spec=m["scenarios"][t["scenario"]]
                delay=recovery_delay(rs,clean,spec["end"],m)
                r.update(recovery_delay_rounds=delay,recovery_right_censored=delay is None,
                    final_accuracy_damage_pp=100*(clean[-1]["test_accuracy"]-last["test_accuracy"]),
                    final_worst20_damage_pp=clean[-1]["worst20_accuracy_pct"]-last["worst20_accuracy_pct"])
            rows.append(r)
    write_json(OUTPUT/"results_progress.json",dict(records=rows,completed=len(rows),total=254,
        status="completed" if len(rows)==254 else "active",scientific_promotion=False))


def execute_one(m,t,stamp,artifact):
    if STOP.is_set():raise RuntimeError("chain stopped after another lane failed")
    verify(stamp);cfg=config_for(m,t,stamp,artifact);dest=directory(t);sp=dest/"orchestration_status.json"
    if sp.exists():
        s=json.loads(sp.read_text());require(s["status"]=="completed","partial/failed run requires inspection: "+str(dest))
        p,x,pair=validate(m,t,cfg,stamp)
        require(s["metrics_sha256"]==hash_file(p) and s["trace_sha256"]==hash_file(dest/"simulator_randomness_private_audit.jsonl"),"completed artifact changed")
        return
    require(not dest.exists() or not any(dest.iterdir()),"unlocked partial directory")
    dest.mkdir(parents=True,exist_ok=True);cp=dest/"resolved_config.yaml"
    with cp.open("x") as stream:yaml.safe_dump(cfg,stream,sort_keys=False)
    base=dict(task=t,status="running",device="mps",mps_fallback=0,config_sha256=hash_file(cp),started_at=datetime.now(timezone.utc).isoformat())
    write_json(sp,base);print("START "+t["phase"]+" "+run_id(t),flush=True)
    try:
        with (dest/"training.log").open("x") as log:
            child=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--worker","--config",str(cp),"--output",str(dest)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
            base["pid"]=child.pid;write_json(sp,base);code=child.wait()
        require(code==0,"worker failed: "+str(dest/"training.log"))
        verify(stamp);p,x,pair=validate(m,t,cfg,stamp)
        write_json(sp,dict(base,status="completed",metrics_sha256=hash_file(p),trace_sha256=hash_file(dest/"simulator_randomness_private_audit.jsonl"),pairing=pair,finished_at=datetime.now(timezone.utc).isoformat()))
        print("DONE "+t["phase"]+" "+run_id(t),flush=True)
        with WRITE_LOCK:summarize(m)
    except BaseException as exc:
        STOP.set();write_json(sp,dict(base,status="failed",error=str(exc)));raise


def lane(m,noise,phase,stamp,artifact):
    for t in tasks(m,phase,noise):execute_one(m,t,stamp,artifact)


def parallel_phase(m,phase,stamp,artifact=None):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,m,n,phase,stamp,artifact) for n in m["noise_regimes"]]
        for future in as_completed(futures):future.result()


def worker(cp,dest):
    shared.require_working_mps()
    from algorithms.aggregate_predictor_control import AggregatePredictorControl,PredictorControlEvaluator
    from algorithms.base import register_algorithm
    from scripts.aggregate_control_attack_schedule import scheduled_attack
    from scripts.run_rcig_batch_screen_experiment import write_runtime_manifest
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    import run_experiment as harness
    register_algorithm("ldp_gradient_far")(AggregatePredictorControl)
    old_eval=harness.rcig_reference_oracle_metrics;old_attack=harness.apply_configured_attack;oldargv=sys.argv
    before=write_runtime_manifest(dest,algorithm=CAMPAIGN,stage="before_training")
    with (dest/"simulator_randomness_private_audit.jsonl").open("x") as trace:
        def sink(r):trace.write(json.dumps(r,sort_keys=True)+"\n");trace.flush()
        AggregatePredictorControl.audit_sink=staticmethod(sink)
        harness.rcig_reference_oracle_metrics=PredictorControlEvaluator();harness.apply_configured_attack=scheduled_attack
        sys.argv=[str(ROOT/"run_experiment.py"),"--config",str(cp),"--output",str(dest),"--device","mps"]
        try:
            with evaluation_rng_isolation(harness):harness.main()
            write_runtime_manifest(dest,algorithm=CAMPAIGN,stage="after_training",previous=before)
        finally:
            AggregatePredictorControl.audit_sink=None
            harness.rcig_reference_oracle_metrics=old_eval;harness.apply_configured_attack=old_attack;sys.argv=oldargv


def status(m):
    out={phase:dict(completed=0,total=len(tasks(m,phase)),active=[],failed=[],missing=0) for phase in ("calibration","evaluation")}
    for phase,s in out.items():
        for t in tasks(m,phase):
            sp=directory(t)/"orchestration_status.json"
            if not sp.exists():s["missing"]+=1;continue
            x=json.loads(sp.read_text())
            if x["status"]=="completed":s["completed"]+=1
            else:s["active" if x["status"]=="running" else "failed"].append(x)
    out["radii_frozen"]=ARTIFACT.exists();return out


@contextmanager
def lock():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT.parent/("."+CAMPAIGN+".lock")).open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for x in ("plan","status","launch","run","worker","audit"):g.add_argument("--"+x,action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--config",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.worker:
        require(args.config is not None and args.output is not None,"worker paths required");worker(args.config,args.output);return
    m=matrix()
    if args.status:print(json.dumps(status(m),indent=2));return
    stamp=provenance(m)
    if args.plan:
        for t in tasks(m,"calibration"):config_for(m,t,stamp)
        print(json.dumps(dict(calibration_runs=38,evaluation_runs=216,total=254,parallel_mps_lanes=2,
            arms=m["arms"],privacy=stamp["privacy"],source_count=len(stamp["sources"]),output=str(OUTPUT)),indent=2));return
    if args.audit:
        locked=json.loads((OUTPUT/"campaign_lock.json").read_text());verify(locked);count=0
        artifact=load_radii(locked) if ARTIFACT.exists() else None
        for phase in ("calibration","evaluation"):
            for t in tasks(m,phase):
                sp=directory(t)/"orchestration_status.json"
                if not sp.exists() or json.loads(sp.read_text())["status"]!="completed":continue
                cfg=config_for(m,t,locked,artifact if phase=="evaluation" else None)
                p0,_,_=validate(m,t,cfg,locked)
                require(json.loads(sp.read_text())["metrics_sha256"]==hash_file(p0),"completed metrics hash changed")
                count+=1
        summarize(m);print(json.dumps(dict(validated=count,total=254)));return
    require(args.resume,"--resume mandatory");shared.require_working_mps()
    with lock():
        pilot.no_duplicates()
        if args.launch:
            LOG.parent.mkdir(parents=True,exist_ok=True)
            with LOG.open("a") as log:
                child=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--run","--resume"],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
            print(json.dumps(dict(pid=child.pid,log=str(LOG),calibration=38,evaluation=216,parallel_lanes=2)));return
        OUTPUT.mkdir(parents=True,exist_ok=True);cp=OUTPUT/"campaign_lock.json"
        if cp.exists():require(json.loads(cp.read_text())==stamp,"campaign stamp changed")
        else:write_json(cp,stamp)
        try:
            parallel_phase(m,"calibration",stamp)
            artifact=calibrate(m,stamp)
            parallel_phase(m,"evaluation",stamp,artifact)
            # Final cross-regime pairing, whose counterpart may finish later.
            for t in tasks(m,"evaluation","homogeneous"):
                for a,b in zip(trace_for(t),trace_for(dict(t,noise="heteroscedastic")),strict=True):
                    require(a["permutations"]==b["permutations"] and a["standard_gaussians"]==b["standard_gaussians"],"evaluation regime pairing failed")
            summarize(m);write_json(OUTPUT/"completion.json",dict(validated=254,calibration=38,evaluation=216,scientific_promotion=False))
        except BaseException as exc:
            STOP.set();write_json(OUTPUT/"failure.json",dict(error=str(exc),time=datetime.now(timezone.utc).isoformat()));raise


if __name__=="__main__":main()
