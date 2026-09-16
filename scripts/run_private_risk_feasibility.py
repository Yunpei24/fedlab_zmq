#!/usr/bin/env python3
"""MPS-only, six-run headroom/private-validation screen; no online selector.

Use --launch --resume --device mps. Historical implementations stay untouched.
Only the clean uniform host determines future model states. Oracle diagnostics
are never arguments to the private selector and are not DP transcript outputs.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
import secrets
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from algorithms.private_risk_selection import risk, select_private_risk, forge_reports
from datasets.registry import TRANSFORMS
from datasets.partitioner import partition_dataset
from models.registry import get_model
from privacy.local_dpsgd import private_gradient_release_fixed_without_replacement
from privacy.rdp import RDPAccountant, calibrate_sampled_without_replacement_gaussian_noise, calibrate_gaussian_noise
from scripts.run_rcig_batch_screen import source_closure, file_hash

CAMPAIGN = "private_risk_feasibility_n10_v1"
MATRIX = ROOT / "configs/ldp_gradient_far" / (CAMPAIGN + ".yaml")
OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN


def save(path, value, *, private=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        if private: os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def config():
    m = yaml.safe_load(MATRIX.read_text())
    assert m["campaign_id"] == CAMPAIGN and m["expected_runs"] == 6
    assert m["num_clients"] == 10 and m["train_size"] + m["validation_size"] == 6000
    assert m["candidate_names"] == ["no_update", "uniform", "rfa_direct", "far_rfa"]
    assert m["probe_rounds"] == [8, 20, 40] and m["rounds"] == 40
    assert m["gradient_scenarios"] == ["none", "bf_x10", "ipm_x5"]
    assert m["privacy"]["adjacency"] == "replace_one"
    return m


def jobs(m):
    return [dict(noise=noise, seed=seed) for noise, seed in itertools.product(m["noise_regimes"], m["seeds"])]


def job_id(job): return f"{job['noise']}__seed{job['seed']}"


def privacy_plan(m):
    p = m["privacy"]; steps = len(m["probe_rounds"])*len(m["gradient_scenarios"])
    sigma = calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=p["gradient_epsilon"]-.001, delta=p["component_delta"],
        sampling_rate=m["batch_size"]/m["train_size"], steps=m["rounds"], sensitivity_multiplier=2.)
    multiplier = calibrate_gaussian_noise(target_epsilon=p["evaluation_epsilon"]-.001,
                                         delta=p["component_delta"], steps=steps)
    sensitivity = math.sqrt(len(m["candidate_names"]))/m["validation_size"]
    ledgers = {}
    for factor in (1., 2.):
        g = RDPAccountant(); e = RDPAccountant(); joint = RDPAccountant()
        for ledger in (g, joint):
            ledger.add_sampled_without_replacement_gaussian(channel="gradient",
                sampling_rate=m["batch_size"]/m["train_size"], noise_multiplier=sigma*factor/2, steps=m["rounds"])
        for ledger in (e, joint):
            ledger.add_gaussian(channel="validation", noise_multiplier=multiplier*factor, steps=steps)
        eg = g.epsilon(p["component_delta"])[0]; ee = e.epsilon(p["component_delta"])[0]
        assert eg <= p["gradient_epsilon"] and ee <= p["evaluation_epsilon"]
        assert eg+ee <= p["total_epsilon_max"]
        ledgers[str(int(factor))] = dict(gradient_epsilon=eg, evaluation_epsilon=ee,
            sequential_epsilon=eg+ee, composed_rdp_epsilon=joint.epsilon(p["total_delta"])[0],
            joint_ledger=joint.state_dict())
    return dict(gradient_sigma=sigma, evaluation_multiplier=multiplier,
                evaluation_sensitivity=sensitivity, evaluation_std=multiplier*sensitivity,
                evaluation_releases=steps, ledgers=ledgers,
                privacy_scope="per_client_per_run_protocol_only_excludes_oracles_and_cross_run_composition")


def require_mps():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable/fallback enabled: no CPU training permitted")
    assert float(torch.ones(1, device="mps").sum()) == 1.


def reseed(seed):
    torch.manual_seed(seed); torch.mps.manual_seed(seed)


def derived_seed(key, *parts):
    return int.from_bytes(hashlib.sha256((key+"/"+"/".join(map(str, parts))).encode()).digest()[:8], "big") % (2**63-1)


def datasets_for(m, seed):
    # No network download or data-dependent validation augmentation.
    aug = FashionMNIST(ROOT/"data", train=True, transform=TRANSFORMS["fashionmnist"]["train"], download=False)
    deterministic = FashionMNIST(ROOT/"data", train=True, transform=TRANSFORMS["fashionmnist"]["test"], download=False)
    test = FashionMNIST(ROOT/"data", train=False, transform=TRANSFORMS["fashionmnist"]["test"], download=False)
    kwargs = dict(num_clients=10, partition=m["partition"], alpha=m["dirichlet_beta"], seed=seed)
    train_parts = partition_dataset(deterministic, **kwargs)
    test_parts = partition_dataset(test, **kwargs)
    train, val, test_loaders, split_audit = [], [], [], []
    for i, part in enumerate(train_parts):
        assert len(part) == 6000
        generator = torch.Generator().manual_seed(seed*1009+i)
        indices = [int(part.indices[j]) for j in torch.randperm(len(part), generator=generator).tolist()]
        tr, va = indices[:m["train_size"]], indices[m["train_size"]:]
        assert set(tr).isdisjoint(va) and len(va) == m["validation_size"]
        train.append(DataLoader(Subset(aug, tr), batch_size=m["batch_size"], num_workers=0))
        val.append(DataLoader(Subset(deterministic, va), batch_size=256, num_workers=0))
        test_loaders.append(DataLoader(test_parts[i], batch_size=256, num_workers=0))
        split_audit.append(dict(client=i, training_count=len(tr), validation_count=len(va),
            test_count=len(test_parts[i]), disjoint=True,
            train_sha256=hashlib.sha256(json.dumps(tr).encode()).hexdigest(),
            val_sha256=hashlib.sha256(json.dumps(va).encode()).hexdigest()))
    return train, val, test_loaders, split_audit


def flat_state(state):
    return torch.cat([value.detach().to("mps").float().reshape(-1) for value in state.values()])


def load_vector(model, vector):
    offset = 0
    with torch.no_grad():
        for value in model.state_dict().values():
            size = value.numel()
            value.copy_(vector[offset:offset+size].view_as(value)); offset += size
    assert offset == vector.numel()


def clip_cohort(vectors, radius):
    return vectors*(radius/torch.linalg.vector_norm(vectors, dim=1).clamp_min(1e-12)).clamp(max=1)[:, None]


def candidate_vectors(vectors, *, radius, alpha):
    """Only the already private messages enter this public server function."""
    assert vectors.device.type == "mps"
    x = clip_cohort(vectors, radius)
    point = x.mean(0)
    for _ in range(100):
        inv = torch.linalg.vector_norm(x-point, dim=1).clamp_min(1e-8).reciprocal()
        proposed = (inv[:, None]*x).sum(0)/inv.sum()
        displacement = float(torch.linalg.vector_norm(proposed-point))
        point = proposed
        if displacement <= 1e-6: break
    weights = torch.softmax(alpha*torch.linalg.vector_norm(x-point, dim=1), dim=0)
    values = [torch.zeros_like(point), x.mean(0), point, (weights[:, None]*x).sum(0)]
    assert all(bool(torch.isfinite(value).all()) for value in values)
    return values


def attacked_cohort(vectors, scenario, ids):
    """Simulator only; no clean gradients used even in these attack inputs."""
    out = vectors.clone()
    if scenario == "bf_x10": out[ids] = -10*out[ids]
    elif scenario == "ipm_x5":
        other = [i for i in range(len(out)) if i not in ids]
        out[ids] = -5*out[other].mean(0)
    elif scenario != "none": raise ValueError(scenario)
    return out


@torch.no_grad()
def local_evaluate(model, loaders):
    assert next(model.parameters()).device.type == "mps"
    model.eval(); rows = []
    for loader in loaders:
        total, correct, ce, brier = 0, 0, 0., 0.
        hits = torch.zeros(10, device="mps"); counts = torch.zeros(10, device="mps")
        for x, y in loader:
            x, y = x.to("mps"), y.to("mps")
            logits = model(x); predicted = logits.argmax(1)
            total += len(y); correct += int((predicted == y).sum())
            ce += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            probabilities = torch.softmax(logits, dim=1)
            one_hot = torch.nn.functional.one_hot(y, 10).float()
            brier += float(.5*(probabilities-one_hot).square().sum())
            for label in range(10):
                counts[label] += (y == label).sum()
                hits[label] += ((y == label) & (predicted == y)).sum()
        assert total == len(loader.dataset) and 0 <= brier/total <= 1+1e-6
        supported = counts > 0
        rows.append(dict(count=total, accuracy=correct/total, loss=ce/total, brier=brier/total,
            balanced_accuracy=float((hits[supported]/counts[supported]).mean()),
            class_counts=counts.cpu().tolist(), class_hits=hits.cpu().tolist()))
    return rows


def model_summary(rows, honest_ids):
    values = [rows[i]["accuracy"] for i in honest_ids]
    ordered = sorted(values); tail = math.ceil(.2*len(values)); mean = statistics.mean(values)
    return dict(test_accuracy=sum(r["count"]*r["accuracy"] for r in rows)/sum(r["count"] for r in rows),
                honest_client_accuracy=mean, honest_loss=statistics.mean(rows[i]["loss"] for i in honest_ids),
                worst20_pct=100*statistics.mean(ordered[:tail]),
                gap_pp=100*(statistics.mean(ordered[-tail:])-statistics.mean(ordered[:tail])),
                variance_pp2=10000*statistics.mean((v-mean)**2 for v in values),
                balanced_accuracy=statistics.mean(rows[i]["balanced_accuracy"] for i in honest_ids))


def private_release(levels, stds, key, *context):
    # Each full risk vector is privatized locally; fresh noise per query/client.
    output = []
    for cid, (row, scale) in enumerate(zip(levels, stds)):
        reseed(derived_seed(key, "validation", *context, cid))
        value = torch.tensor(row, device="mps") + scale*torch.randn(len(row), device="mps")
        output.append(value.cpu().tolist())
    return output


def probe(model, vectors, val, test, m, plan, job, key, round_num, dest):
    base = flat_state(model.state_dict()).clone()
    stds = [plan["evaluation_std"]*scale for scale in m["noise_regimes"][job["noise"]]]
    out = []
    for scenario in m["gradient_scenarios"]:
        save(dest/"orchestration_status.json", dict(status="running", round=round_num,
             stage="candidate_validation_"+scenario, **job, pid=os.getpid()))
        messages = attacked_cohort(vectors, scenario, m["attacker_ids"])
        aggregates = candidate_vectors(messages, radius=m["server_clip"], alpha=m["far_alpha"])
        validation_rows, test_rows = [], []
        for vector in aggregates:
            load_vector(model, base-m["server_lr"]*vector)
            validation_rows.append(local_evaluate(model, val))
            test_rows.append(local_evaluate(model, test))
        load_vector(model, base)
        levels = [[validation_rows[k][cid]["brier"] for k in range(4)] for cid in range(10)]
        reports = private_release(levels, stds, key, job["noise"], round_num, scenario)
        prefix = f"round{round_num:02d}_{scenario}"
        # Plain risks are diagnostic-only, isolated from the private transcript.
        save(dest/"simulator_oracle"/(prefix+".json"), dict(validation=validation_rows, test=test_rows,
             privacy_protected=False, feeds_host_or_selector=False), private=True)
        policies = []; safe_outputs = []
        for mode in m["report_modes"]:
            submitted = forge_reports(reports, identities=m["attacker_ids"], mode=mode)
            is_clean = scenario == "none" and mode == "truthful"
            h = list(range(10)) if is_clean else list(range(2, 10))
            true_risks = [risk([levels[i][k] for i in h], m["fairness"]["mean_tail_mix"], m["fairness"]["tail_fraction"]) for k in range(4)]
            changes = [v-true_risks[0] for v in true_risks]
            for b in (0, m["byzantine_bound"]):
                # This call has no access to levels, H, test metrics, or attack IDs.
                decision = select_private_risk(submitted, noise_stds=stds,
                    max_releases=plan["evaluation_releases"], failure_probability=m["confidence_failure_probability"],
                    byzantine_bound=b, fairness_mix=m["fairness"]["mean_tail_mix"], tail_fraction=m["fairness"]["tail_fraction"])
                chosen = decision["selected"]; naive = decision["naive_selected"]
                coverage = all(decision["lower"][i][k]-1e-6 <= levels[i][k] <= decision["upper"][i][k]+1e-6 for i in h for k in range(4))
                meaningful_certificate = is_clean or b >= 2
                if coverage and meaningful_certificate:
                    assert all(changes[k] <= decision["upper_bounds"][k]+1e-6 for k in range(4))
                candidates = [model_summary(rows, h) for rows in test_rows]
                policies.append(dict(report_mode=mode, byzantine_bound=b, actual_honest_ids=h,
                    certificate_assumption_valid=meaningful_certificate,
                    true_risk_changes=changes, oracle_best_candidate=min(range(4), key=lambda k: true_risks[k]),
                    oracle_progress=max(0., -min(changes)), oracle_gap_vs_uniform=changes[1]-min(changes[1:]),
                    selected=chosen, selected_change=changes[chosen], naive_selected=naive, naive_change=changes[naive],
                    coverage=coverage, upper_bounds=decision["upper_bounds"], halfwidths=decision["halfwidths"],
                    candidate_test_metrics=candidates))
                safe_outputs.append(dict(report_mode=mode, byzantine_bound=b, submitted_reports=submitted, decision=decision))
        save(dest/"private_transcript"/(prefix+".json"), dict(round=round_num, scenario=scenario,
             reports=reports, public_noise_stds=stds, policies=safe_outputs, contains_oracles=False), private=True)
        out.append(dict(round=round_num, scenario=scenario, policies=policies,
                        messages_shared_across_candidates=True, selections_applied_to_host=False))
        print(f"[{job_id(job)}] probe round={round_num} {scenario}: completed (diagnostic only)", flush=True)
    load_vector(model, base)
    return out


def manifest(m, plan):
    return dict(matrix_sha256=file_hash(MATRIX), protocol_sha256=file_hash(ROOT/m["protocol"]),
                sources=source_closure((Path(__file__),)), privacy=plan, runtime_torch=torch.__version__, device="mps", fallback=0)


def verify_manifest(stamp, m):
    assert stamp["matrix_sha256"] == file_hash(MATRIX) and stamp["protocol_sha256"] == file_hash(ROOT/m["protocol"]), "protocol drift"
    assert all(file_hash(ROOT/p) == h for p, h in stamp["sources"].items()), "source drift: stop, do not overwrite results"


def validate(dest, m, stamp):
    x = json.loads((dest/"metrics.json").read_text())
    assert x["manifest"] == stamp and len(x["rounds"]) == m["rounds"]
    assert len(x["probes"]) == 9 and len(list((dest/"private_transcript").glob("*.json"))) == 9
    assert len(list((dest/"simulator_oracle").glob("*.json"))) == 9
    assert all(r["round"] == t and r["private_gradient_device"] == "mps" for t, r in enumerate(x["rounds"], 1))
    assert all(len(p["policies"]) == 6 and not p["selections_applied_to_host"] for p in x["probes"])
    assert x["completed"] and x["epsilon_upper_bound_max"] <= 4
    assert x["contains_unprivatized_simulator_diagnostics"]
    assert all(math.isfinite(q["selected_change"]) for p in x["probes"] for q in p["policies"])
    expected = {(t, s) for t in m["probe_rounds"] for s in m["gradient_scenarios"]}
    assert {(p["round"], p["scenario"]) for p in x["probes"]} == expected
    assert all(file_hash(dest/relative) == digest for relative, digest in x["artifact_sha256"].items())
    assert len(x["artifact_sha256"]) == 19  # nine private reports, nine oracle files, split audit
    for t, scenario in expected:
        transcript = json.loads((dest/"private_transcript"/f"round{t:02d}_{scenario}.json").read_text())
        assert transcript["contains_oracles"] is False
        assert len(transcript["reports"]) == 10 and all(len(row) == 4 for row in transcript["reports"])
        assert all(math.isfinite(v) for row in transcript["reports"] for v in row)
    return x


def run_job(m, plan, stamp, job):
    require_mps(); dest = OUTPUT/job_id(job); dest.mkdir(parents=True, exist_ok=True)
    if (dest/"metrics.json").exists():
        validate(dest, m, stamp)
        save(dest/"orchestration_status.json", dict(status="completed", rounds=40, **job, metrics_sha256=file_hash(dest/"metrics.json")))
        return
    checkpoint = dest/"checkpoint.pt"
    private_key = dest/".private_randomness.json"
    if private_key.exists(): key = json.loads(private_key.read_text())["key"]
    else:
        if checkpoint.exists(): raise RuntimeError("cannot resume without the private randomness key")
        key = secrets.token_hex(32); save(private_key, dict(key=key), private=True)
    reseed(job["seed"])
    model = get_model(m["model"], m["dataset"]).to("mps")
    train, val, test, splits = datasets_for(m, job["seed"])
    save(dest/"split_audit.json", splits, private=True)
    rows, probes, start = [], [], 0
    if checkpoint.exists():
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert restored["matrix_sha256"] == stamp["matrix_sha256"]
        model.load_state_dict(restored["model"])
        rows, probes, start = restored["rounds"], restored["probes"], restored["round"]
    try:
        for t in range(start+1, m["rounds"]+1):
            verify_manifest(stamp, m)
            save(dest/"orchestration_status.json", dict(status="running", stage="private_gradients", round=t, **job, pid=os.getpid()))
            base = flat_state(model.state_dict()).clone(); messages = []
            for cid, loader in enumerate(train):
                reseed(derived_seed(key, "gradient", t, cid))
                gradient, _ = private_gradient_release_fixed_without_replacement(model, loader,
                    device="mps", batch_size=m["batch_size"], clip_norm=m["local_clip"],
                    noise_multiplier=plan["gradient_sigma"]*m["noise_regimes"][job["noise"]][cid],
                    backend="vectorized", return_noise_free_oracle=False)
                assert next(model.parameters()).device.type == "mps"
                messages.append(flat_state(gradient))
            vectors = torch.stack(messages)
            assert torch.equal(base, flat_state(model.state_dict())), "client unexpectedly mutated model"
            if t in m["probe_rounds"]:
                probes.extend(probe(model, vectors, val, test, m, plan, job, key, t, dest))
            applied = clip_cohort(vectors, m["server_clip"]).mean(0)
            load_vector(model, base-m["server_lr"]*applied)
            rows.append(dict(round=t, private_gradient_device="mps", host_rule="clean_uniform",
                             aggregate_norm=float(torch.linalg.vector_norm(applied)), validation_choice_used=False))
            state = dict(round=t, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                         rounds=rows, probes=probes, matrix_sha256=stamp["matrix_sha256"])
            temp = checkpoint.with_suffix(".tmp")
            with temp.open("wb") as stream:
                os.fchmod(stream.fileno(), 0o600); torch.save(state, stream)
            os.replace(temp, checkpoint)
            print(f"[{job_id(job)}] round {t}/40 saved", flush=True)
        save(dest/"metrics.json", dict(completed=True, job=job, manifest=stamp,
             rounds=rows, probes=probes, epsilon_upper_bound_max=max(v["sequential_epsilon"] for v in plan["ledgers"].values()),
             delta=m["privacy"]["total_delta"], contains_unprivatized_simulator_diagnostics=True,
             artifact_sha256={str(p.relative_to(dest)): file_hash(p) for p in
                 [dest/"split_audit.json", *sorted((dest/"private_transcript").glob("*.json")),
                  *sorted((dest/"simulator_oracle").glob("*.json"))]},
             interpretation="offline_one_step_not_selector_end_to_end"), private=True)
        validate(dest, m, stamp)
        save(dest/"orchestration_status.json", dict(status="completed", rounds=40, **job, metrics_sha256=file_hash(dest/"metrics.json")))
    except BaseException as exc:
        save(dest/"orchestration_status.json", dict(status="failed", **job, error=str(exc), resumable_checkpoint=checkpoint.exists()))
        raise


def status(m):
    output = dict(completed=0, total=6, active=[], failed=[], missing=[])
    for job in jobs(m):
        path = OUTPUT/job_id(job)/"orchestration_status.json"
        if not path.exists(): output["missing"].append(job_id(job)); continue
        value = json.loads(path.read_text())
        if value["status"] == "completed":
            try:
                stamp = json.loads((OUTPUT/"campaign_lock.json").read_text())
                validate(path.parent, m, stamp)
                assert file_hash(path.parent/"metrics.json") == value["metrics_sha256"]
                output["completed"] += 1
            except (AssertionError, OSError, ValueError, KeyError) as exc:
                output["failed"].append(dict(**job, status="invalid_completed_run", error=str(exc)))
        elif value["status"] == "running": output["active"].append(value)
        else: output["failed"].append(value)
    return output


def summarize(m, stamp):
    records = []
    for job in jobs(m):
        dest = OUTPUT/job_id(job)
        if not (dest/"metrics.json").exists(): continue
        x = validate(dest, m, stamp)
        for p in x["probes"]:
            for q in p["policies"]:
                if q["byzantine_bound"] == 2:
                    records.append(dict(**job, round=p["round"], scenario=p["scenario"], **q))
    grouped = []
    for noise, scenario, mode, seed in itertools.product(m["noise_regimes"], m["gradient_scenarios"], m["report_modes"], m["seeds"]):
        rs = [r for r in records if (r["noise"],r["scenario"],r["report_mode"],r["seed"]) == (noise,scenario,mode,seed)]
        if not rs: continue
        available = sum(r["oracle_progress"] for r in rs)
        retention = -sum(r["selected_change"] for r in rs)/available if available > 1e-12 else None
        accepted = [r for r in rs if r["selected"] != 0]
        harmful = sum(r["selected_change"] > 1e-6 for r in accepted)
        cfg = m["screen_criteria"]
        informative = statistics.mean(min(r["true_risk_changes"][1:]) <= -cfg["minimum_oracle_progress"]
            and r["oracle_gap_vs_uniform"] >= cfg["minimum_gap_vs_uniform"] for r in rs)
        grouped.append(dict(noise=noise, scenario=scenario, report_mode=mode, seed=seed,
             observations=len(rs), informative_fraction=informative, progress_retention=retention,
             accepted=len(accepted), harmful_accepted=harmful, abstention=1-len(accepted)/len(rs),
             oracle_available_progress=available, minimum_upper_bound_nonzero=min(min(r["upper_bounds"][1:]) for r in rs),
             headroom_gate=informative >= cfg["minimum_informative_fraction"],
             private_gate=retention is not None and retention >= cfg["minimum_progress_retention"]
                 and harmful/max(1,len(accepted)) <= cfg["maximum_harmful_acceptance_fraction"]))
    regime_gates = []
    for noise, scenario, mode in itertools.product(m["noise_regimes"], m["gradient_scenarios"], m["report_modes"]):
        ss = [r for r in grouped if (r["noise"], r["scenario"], r["report_mode"]) == (noise, scenario, mode)]
        complete = len(ss) == len(m["seeds"]) and all(r["observations"] == len(m["probe_rounds"]) for r in ss)
        joint_passes = sum(r["headroom_gate"] and r["private_gate"] for r in ss)
        regime_gates.append(dict(noise=noise, scenario=scenario, report_mode=mode,
             completed_seeds=len(ss), headroom_seeds_passed=sum(r["headroom_gate"] for r in ss),
             joint_seeds_passed=joint_passes, required_seeds=m["screen_criteria"]["seeds_required_per_regime"],
             status="pending" if not complete else (
                 "pass" if joint_passes >= m["screen_criteria"]["seeds_required_per_regime"] else "fail")))
    save(OUTPUT/"summary_progress.json", dict(status=status(m), seed_summaries=grouped, regime_gates=regime_gates,
         automatic_promotion=False, scientific_scope="feasibility_only_no_end_to_end_claim"), private=True)


@contextmanager
def execution_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/"execution.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def launch(m):
    require_mps()
    with execution_lock():
        plan = privacy_plan(m)
        lock = OUTPUT/"campaign_lock.json"
        if lock.exists():
            stamp = json.loads(lock.read_text()); verify_manifest(stamp, m)
            assert stamp["privacy"] == plan and stamp["runtime_torch"] == torch.__version__
        else:
            stamp = manifest(m, plan); save(lock, stamp)
        save(OUTPUT/"privacy_plan.json", plan)
        for job in jobs(m):
            verify_manifest(stamp, m); run_job(m, plan, stamp, job); summarize(m, stamp)
        save(OUTPUT/"campaign_status.json", dict(status="completed", runs=6, automatic_promotion=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--device", choices=["mps"], default="mps")
    args = parser.parse_args(); m = config()
    if args.status: print(json.dumps(status(m), indent=2)); return
    if args.plan: print(json.dumps(privacy_plan(m), indent=2)); return
    if not args.launch or not args.resume: parser.error("launch requires --launch --resume")
    launch(m)


if __name__ == "__main__": main()
