#!/usr/bin/env python3
"""Source-locked 24-run clean EMA-utility pilot, MPS private gradients only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import yaml
from scripts import run_aggregate_control_replay as replay
from scripts import run_ldp_aggregation_role_ablation as historical
from scripts import run_rcig_batch_screen as shared
from scripts.run_rcig_ldp_gradient_far_v2 import _metrics_path, _post_run_device_audit

CAMPAIGN = "aggregate_control_e2e_pilot_v1"
MATRIX = ROOT / f"configs/ldp_gradient_far/{CAMPAIGN}.yaml"
OUTPUT = ROOT / f"results/ldp_gradient_far/{CAMPAIGN}"
ENTRY = Path(__file__).resolve()
LOG = ROOT / f"logs/{CAMPAIGN}_mps.log"
ARMS = ["far_rfa", "ema_far_rfa", "uniform", "rfa_direct"]
hash_file, write_json = shared.file_hash, shared.write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def matrix():
    m = yaml.safe_load(MATRIX.read_text())
    fixed = dict(campaign_id=CAMPAIGN, required_device="mps", mps_fallback=0,
        seeds=[930401,930402,930403], arms=ARMS, scenario="none", rounds=40,
        warmup_uniform_rounds=12, predictor_rate=.25, ema_current_mix=.2, expected_runs=24,
        noise_regimes=["homogeneous", "heteroscedastic"])
    require(all(m.get(k)==v for k,v in fixed.items()), "frozen pilot identity changed")
    old = replay.load_matrix()
    closed = {x for name in ("headroom_fit", "radius_calibration", "untouched_evaluation")
              for x in old["seed_registry"][name]}
    require(not set(m["seeds"]) & closed, "historical/closed seeds used by utility pilot")
    return m


def tasks(m):
    return [dict(seed=s,noise=n,arm=a) for s,n,a in itertools.product(m["seeds"],m["noise_regimes"],m["arms"])]


def run_id(task):
    return f"{task['noise']}__seed{task['seed']}__{task['arm']}"


def directory(task):
    return OUTPUT / run_id(task)


def provenance(m):
    inherited = replay.provenance(replay.load_matrix())
    files = [MATRIX, ROOT/m["protocol"], replay.MATRIX, historical.MATRIX,
             historical.MATRIX.parent/historical.matrix()["base_config"], historical.legacy.ARTIFACT]
    return dict(campaign=CAMPAIGN, sources=shared.source_closure((ENTRY,
        ROOT/"algorithms/aggregate_control_e2e.py", ROOT/"run_experiment.py")),
        files={str(p.relative_to(ROOT)):hash_file(p) for p in files}, inherited=inherited,
        privacy=historical.legacy.privacy(historical.matrix()), total=24, device="mps", fallback=0)


def verify(stamp):
    replay.verify_provenance(replay.load_matrix(), stamp["inherited"])
    for key in ("files", "sources"):
        for name, digest in stamp[key].items():
            require(hash_file(ROOT/name)==digest, "source/config drift: "+name)


def config_for(m, task, stamp):
    require(task in tasks(m), "undeclared pilot task")
    parent_arm = "far_rfa" if task["arm"]=="ema_far_rfa" else task["arm"]
    cfg = historical.config_for(historical.matrix(), dict(task,scenario="none",arm=parent_arm), stamp)
    cfg.update(output_dir=str(directory(task)),seed=task["seed"],device="mps")
    cfg["data"]["partition_seed"] = task["seed"]
    a = cfg["training"]["algo_config"]
    a.update(aggregation_role_campaign=CAMPAIGN, rcig_campaign_id=CAMPAIGN,
        rcig_n10_phase="clean_utility_pilot", rcig_n10_pairing_seed=task["seed"],
        rcig_n10_scenario="none", num_byzantine=0,
        rcig_n10_runtime_implementation="algorithms.aggregate_control_e2e.AggregateControlEndToEnd",
        aggregate_e2e_arm=task["arm"],aggregate_e2e_scenario="none",
        aggregate_e2e_predictor_rate=m["predictor_rate"],aggregate_e2e_current_mix=m["ema_current_mix"],
        aggregate_e2e_matrix_hash=stamp["files"][str(MATRIX.relative_to(ROOT))],
        aggregate_e2e_oracle_never_selects_deployment=True,
        aggregate_e2e_scientific_status=m["scientific_status"])
    p = m["private_training_contract"]
    expected = dict(fixed_batch_size=p["batch_size"],clip_norm=p["local_clip_norm"],
        far_server_clip_norm=p["server_clip_norm"],far_alpha=p["alpha"],far_server_lr=p["server_learning_rate"],
        fixed_steps_per_round=p["local_steps"],local_epochs=1,privacy_public_dataset_size=p["public_local_size"],
        privacy_num_rounds=m["rounds"],sampling_scheme=p["sampling"],privacy_adjacency="replace_one")
    require(all(a.get(k)==v for k,v in expected.items()), "resolved private training contract mismatch")
    require(not a["attack"]["enabled"] and cfg["clients"]["num_clients"]==10, "clean n10 protocol mismatch")
    require(a["privacy_noise_multiplier_scale_by_client"]==replay.load_matrix()["noise_regimes"][task["noise"]], "noise profile drift")
    return cfg


def trace_for(task):
    p = directory(task)/"simulator_randomness_private_audit.jsonl"
    records = sorted((json.loads(line) for line in p.read_text().splitlines()),key=lambda r:(r["round"],r["client_id"]))
    require(len(records)==400 and {(r["round"],r["client_id"]) for r in records}==set(itertools.product(range(1,41),range(10))), "incomplete private trace")
    return records


def validate(task, cfg, stamp):
    dest = directory(task)
    p = _metrics_path(dest)
    require(p is not None, "missing or ambiguous complete metrics")
    x = json.loads(p.read_text()); rows=x["rounds"]; summary=x["summary"]
    require(len(rows)==40, "incomplete trajectory")
    require((summary["num_clients"],summary["num_rounds"],summary["seed"],summary["model"],summary["dataset"])
            ==(10,40,task["seed"],"lenet5_tanh","fashionmnist"), "summary identity mismatch")
    for k,v in cfg["training"]["algo_config"].items():
        require(x["config"].get(k)==v, "metrics config mismatch: "+k)
    require(yaml.safe_load((dest/"resolved_config.yaml").read_text())==cfg, "saved config changed")
    _post_run_device_audit(p,cfg)
    numeric = ["test_accuracy","test_loss","client_accuracy_mean","client_accuracy_variance_pct2",
        "worst20_accuracy_pct","best20_worst20_gap_pct","mean_client_balanced_accuracy_pct",
        "aggregate_e2e_applied_mse","aggregate_e2e_uncorrected_far_mse"]
    for t,r in enumerate(rows,1):
        require(r["round_num"]==t and r["num_alive_clients"]==10, "round/client count drift")
        require(r["aggregate_e2e_arm"]==task["arm"] and r["aggregate_e2e_effective_arm"]==("uniform" if t<=12 else task["arm"]), "wrong model driver")
        require(r["aggregate_e2e_correction_applied_to_model"]==(t>12 and task["arm"]=="ema_far_rfa"), "correction was not actually deployed")
        require(r["aggregate_e2e_history_observations"]==t-1 and r["aggregate_e2e_predictor_strictly_past"], "predictor chronology violated")
        require(not r["aggregate_e2e_oracle_used_for_deployment"] and not r["aggregate_e2e_rcig_predictor_used"], "unexpected predictor/oracle input")
        require(r["aggregation_role_same_cohort_verified"] and not r["aggregation_role_oracles_visible_to_server"], "oracle boundary mismatch")
        for k in numeric:
            require(isinstance(r.get(k),(float,int)) and math.isfinite(r[k]), "nonfinite/missing metric: "+k)
        if t>12:
            require(r["aggregate_e2e_predictor_mse"] is not None, "missing past predictor")
            require(r["shadow_rcig_newer_round_max"]<t-1, "shadow read current private cohort")
        if task["arm"]=="far_rfa" or t<=12:
            require(abs(r["aggregate_e2e_applied_mse"]-r["aggregate_e2e_uncorrected_far_mse"])<1e-10, "unchanged baseline differs")
    require(abs(rows[-1]["privacy_epsilon_max"]-4)<1e-4, "privacy budget mismatch")
    runtime=json.loads((dest/"runtime_imports.json").read_text())
    require(runtime["stage"]=="after_training" and runtime["mps_fallback"]==0, "runtime incomplete/CPU fallback")
    require(all(stamp["sources"].get(name)==h for name,h in runtime["source_sha256"].items()), "runtime import outside source lock")
    records=trace_for(task); paired=[]
    for other in (dict(task,arm="far_rfa"),dict(task,arm="far_rfa",noise="homogeneous")):
        if other==task or other in paired:
            continue
        status_path=directory(other)/"orchestration_status.json"
        require(status_path.exists() and json.loads(status_path.read_text())["status"]=="completed", "canonical counterpart must complete first")
        other_records=trace_for(other)
        for a,b in zip(records,other_records,strict=True):
            require(a["permutations"]==b["permutations"] and a["standard_gaussians"]==b["standard_gaussians"], "random draws not strictly paired")
            if task["noise"]==other["noise"] and a["round"]<=13:
                require(a["model_before"]==b["model_before"] and a["private_upload"]==b["private_upload"], "shared pre-policy prefix changed")
        if task["noise"]==other["noise"]:
            other_rows=json.loads(_metrics_path(directory(other)).read_text())["rounds"]
            for a,b in zip(rows[:12],other_rows[:12]):
                require(a["test_accuracy"]==b["test_accuracy"] and a["test_loss"]==b["test_loss"], "warmup model utilities differ")
        paired.append(other)
    return p,x,dict(records=400,paired_to=paired,strict_batch_and_gaussian_pairing=True)


def summarize(completed):
    by_task={run_id(t):(t,x) for t,x in completed}
    items=[]
    for task,x in completed:
        last=x["rounds"][-1]
        entry=dict(task,**{k:last[k] for k in ("test_accuracy","test_loss","client_accuracy_mean",
            "client_accuracy_variance_pct2","worst20_accuracy_pct","best20_worst20_gap_pct","mean_client_balanced_accuracy_pct")})
        for k in ("applied_mse","uncorrected_far_mse","predictor_mse"):
            entry[k]=statistics.mean(r["aggregate_e2e_"+k] for r in x["rounds"][12:])
        base=by_task[run_id(dict(task,arm="far_rfa"))][1]["rounds"][-1]
        entry["paired_delta_accuracy_pp"]=100*(last["test_accuracy"]-base["test_accuracy"])
        entry["paired_delta_worst20_pp"]=last["worst20_accuracy_pct"]-base["worst20_accuracy_pct"]
        entry["paired_delta_gap_pp"]=last["best20_worst20_gap_pct"]-base["best20_worst20_gap_pct"]
        entry["within_descriptive_clean_margins"]=(entry["paired_delta_accuracy_pp"]>=-1 and
            entry["paired_delta_worst20_pp"]>=-2 and entry["paired_delta_gap_pp"]<=2)
        items.append(entry)
    write_json(OUTPUT/"summary_progress.json",dict(completed=len(completed),total=24,records=items,
        classification="exploratory_no_automatic_promotion",std_semantics="sample_std_across_seeds_not_CI95"))


def status():
    out=dict(completed=0,total=24,active=[],failed=[],missing=0)
    for task in tasks(matrix()):
        p=directory(task)/"orchestration_status.json"
        if not p.exists():out["missing"]+=1;continue
        s=json.loads(p.read_text());state=s.get("status")
        if state=="completed":out["completed"]+=1
        else:out["active" if state=="running" else "failed"].append(s)
    return out


@contextmanager
def execution_lock():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with (OUTPUT.parent/("."+CAMPAIGN+".lock")).open("a+") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def no_duplicates():
    shared.check_no_other_training_process()
    guarded={ENTRY.name,"run_aggregate_control_replay.py","run_ldp_aggregation_role_ablation.py",
        "run_ldp_aggregation_role_experiment.py","run_rcig_n10_experiment.py"}
    for line in subprocess.run(["/bin/ps","-axo","pid=,ppid=,command="],capture_output=True,text=True,check=True).stdout.splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3 or int(parts[0])==os.getpid():continue
        try:args=shlex.split(parts[2])
        except ValueError:continue
        require(not(any(Path(a).name in guarded for a in args) and ("--run" in args or "--worker" in args or "--config" in args)), "active experiment/supervisor: "+line)


def execute(m,stamp):
    completed=[]
    for index,task in enumerate(tasks(m),1):
        verify(stamp);cfg=config_for(m,task,stamp);dest=directory(task)
        sp=dest/"orchestration_status.json";cp=dest/"resolved_config.yaml"
        if sp.exists():
            s=json.loads(sp.read_text())
            require(s.get("status")=="completed", "partial or failed run requires inspection: "+run_id(task))
            p,x,pair=validate(task,cfg,stamp)
            require(s["metrics_sha256"]==hash_file(p) and s["trace_sha256"]==hash_file(dest/"simulator_randomness_private_audit.jsonl"), "completed artifact hash changed")
        else:
            require(not dest.exists() or not any(dest.iterdir()), "unlocked partial directory: "+str(dest))
            dest.mkdir(parents=True,exist_ok=True)
            with cp.open("x") as f:yaml.safe_dump(cfg,f,sort_keys=False)
            s=dict(campaign=CAMPAIGN,task=task,status="running",device="mps",mps_fallback=0,
                config_sha256=hash_file(cp),started_at=datetime.now(timezone.utc).isoformat())
            write_json(sp,s);print(f"START {index}/24 {run_id(task)}",flush=True)
            try:
                with (dest/"training.log").open("x") as log:
                    child=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--worker","--config",str(cp),"--output",str(dest)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
                    s["pid"]=child.pid;write_json(sp,s);code=child.wait()
                require(code==0,"training failed: "+str(dest/"training.log"))
                verify(stamp);p,x,pair=validate(task,cfg,stamp)
                write_json(sp,dict(s,status="completed",finished_at=datetime.now(timezone.utc).isoformat(),
                    metrics_sha256=hash_file(p),trace_sha256=hash_file(dest/"simulator_randomness_private_audit.jsonl"),pairing=pair))
                print(f"DONE {index}/24 test_acc={x['rounds'][-1]['test_accuracy']:.6f}",flush=True)
            except BaseException as exc:
                write_json(sp,dict(s,status="failed",error=str(exc)));raise
        completed.append((task,x));summarize(completed)
        write_json(OUTPUT/"progress.json",dict(completed=len(completed),total=24,
            status="completed" if len(completed)==24 else "active",last_completed=run_id(task),next_phases_launched=False))


def worker(cp,dest):
    shared.require_working_mps()
    cfg=yaml.safe_load(cp.read_text())
    from algorithms.aggregate_control_e2e import AggregateControlEndToEnd, EndToEndEvaluator
    from algorithms.base import register_algorithm
    from scripts.run_rcig_batch_screen_experiment import write_runtime_manifest
    from scripts.run_rcig_n10_experiment import evaluation_rng_isolation
    import run_experiment as harness
    register_algorithm("ldp_gradient_far")(AggregateControlEndToEnd)
    before=write_runtime_manifest(dest,algorithm=CAMPAIGN,stage="before_training")
    original=harness.rcig_reference_oracle_metrics;oldargv=sys.argv
    with (dest/"simulator_randomness_private_audit.jsonl").open("x") as trace:
        def sink(row):
            trace.write(json.dumps(row,sort_keys=True)+"\n");trace.flush()
        AggregateControlEndToEnd.audit_sink=staticmethod(sink)
        harness.rcig_reference_oracle_metrics=EndToEndEvaluator()
        sys.argv=[str(ROOT/"run_experiment.py"),"--config",str(cp),"--output",str(dest),"--device","mps"]
        try:
            with evaluation_rng_isolation(harness):harness.main()
            write_runtime_manifest(dest,algorithm=CAMPAIGN,stage="after_training",previous=before)
        finally:
            harness.rcig_reference_oracle_metrics=original;sys.argv=oldargv
            AggregateControlEndToEnd.audit_sink=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    for name in ("plan","status","audit","launch","run","worker"):g.add_argument("--"+name,action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--config",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.worker:
        require(args.config is not None and args.output is not None,"worker paths required")
        worker(args.config,args.output);return
    m=matrix()
    if args.status:print(json.dumps(status(),indent=2));return
    stamp=provenance(m)
    if args.plan:
        for task in tasks(m):config_for(m,task,stamp)
        print(json.dumps(dict(campaign=CAMPAIGN,total=24,privacy=stamp["privacy"],
            arms=ARMS,seeds=m["seeds"],source_count=len(stamp["sources"]),output=str(OUTPUT),
            actual_model_correction=True,rcig_predictor=False,untouched_old_seeds_opened=False),indent=2));return
    if args.audit:
        locked=json.loads((OUTPUT/"campaign_lock.json").read_text());verify(locked)
        done=[]
        for task in tasks(m):
            sp=directory(task)/"orchestration_status.json"
            if not sp.exists() or json.loads(sp.read_text())["status"]!="completed":continue
            q,x,pair=validate(task,config_for(m,task,locked),locked)
            s=json.loads(sp.read_text())
            require(s["metrics_sha256"]==hash_file(q) and s["trace_sha256"]==hash_file(directory(task)/"simulator_randomness_private_audit.jsonl"),"artifact hashes changed")
            done.append((task,x))
        summarize(done);print(json.dumps(dict(validated=len(done),total=24)));return
    require(args.resume,"--resume mandatory")
    shared.require_working_mps()
    with execution_lock():
        no_duplicates()
        if args.launch:
            LOG.parent.mkdir(parents=True,exist_ok=True)
            with LOG.open("a") as log:
                child=subprocess.Popen([sys.executable,"-B","-u",str(ENTRY),"--run","--resume"],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,"PYTORCH_ENABLE_MPS_FALLBACK":"0","PYTHONDONTWRITEBYTECODE":"1"})
            print(json.dumps(dict(pid=child.pid,log=str(LOG),total=24)));return
        OUTPUT.mkdir(parents=True,exist_ok=True);lock=OUTPUT/"campaign_lock.json"
        if lock.exists():require(json.loads(lock.read_text())==stamp,"campaign provenance drift")
        else:write_json(lock,stamp)
        try:execute(m,stamp)
        except BaseException as exc:
            write_json(OUTPUT/"failure.json",dict(error=str(exc),time=datetime.now(timezone.utc).isoformat()));raise


if __name__=="__main__":
    main()
