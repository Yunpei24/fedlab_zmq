#!/usr/bin/env python3
"""Source-locked, resume-only N10 pilot: 12 clean calibration + 120 policy runs.

--plan/--status do not load torch. --launch detaches a sequential MPS-only
supervisor; --run runs it in the current process. Technical failures stop the
chain. Scientific evidence is exploratory and never promotes old R1/R2/R3.
"""

from contextlib import contextmanager
import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import yaml
from scripts import run_rcig_batch_screen as shared
from scripts.run_rcig_ldp_gradient_far_v2 import _metrics_path, _post_run_device_audit

MATRIX = ROOT / "configs/ldp_gradient_far/rcig_n10_policy_ablation_v1.yaml"
CAMPAIGN = "rcig_n10_policy_ablation_v1"
OUTPUT = ROOT / "results/ldp_gradient_far" / CAMPAIGN
ENTRY = ROOT / "scripts/run_rcig_n10_experiment.py"
ARTIFACT = OUTPUT / "thresholds_n10.json"
MODES = ("full", "isotropic", "euclidean")
hash_file = shared.file_hash
hash_json = shared.canonical_hash
write_json = shared.write_json


def matrix():
    m = yaml.safe_load(MATRIX.read_text())
    assert m["campaign_id"] == CAMPAIGN
    assert (m["num_clients"], m["rounds"], m["batch_size"], m["public_local_size"]) == (
        10,
        40,
        120,
        6000,
    )
    assert not set(m["calibration_seeds"]) & set(m["comparison_seeds"])
    assert len(m["calibration_seeds"]) == 6 and len(m["comparison_seeds"]) == 3
    assert m["arms"] == ["recent", "rolling", "freeze", "midpoint", "rfa"]
    assert m["noise_regimes"] == {
        "homogeneous": [1] * 10,
        "heteroscedastic": [1, 2] * 5,
    }
    return m


def tasks(m):
    result = []
    for noise, seed in itertools.product(m["noise_regimes"], m["calibration_seeds"]):
        result.append(
            dict(
                phase="calibration",
                noise=noise,
                seed=seed,
                scenario="none",
                arm="recent",
            )
        )
    # Each same-arm clean counterpart finishes before its attacked runs.
    for noise, seed, scenario, arm in itertools.product(
        m["noise_regimes"], m["comparison_seeds"], m["scenarios"], m["arms"]
    ):
        result.append(
            dict(phase="comparison", noise=noise, seed=seed, scenario=scenario, arm=arm)
        )
    assert len(result) == 132
    return result


def run_id(task):
    return f"{task['noise']}__seed{task['seed']}__{task['scenario']}__{task['arm']}"


def directory(task):
    return OUTPUT / task["phase"] / run_id(task)


def privacy(m):
    rdp = shared.rdp_module()
    q = m["batch_size"] / m["public_local_size"]
    sigma = rdp.calibrate_sampled_without_replacement_gaussian_noise(
        target_epsilon=m["epsilon"],
        delta=m["delta"],
        sampling_rate=q,
        steps=m["rounds"],
        sensitivity_multiplier=2.0,
    )
    ledger = {}
    for scale in (1, 2):
        a = rdp.RDPAccountant()
        a.add_sampled_without_replacement_gaussian(
            channel="gradient",
            sampling_rate=q,
            noise_multiplier=sigma * scale / 2,
            steps=m["rounds"],
        )
        epsilon, order = a.epsilon(m["delta"])
        ledger[str(scale)] = dict(
            epsilon=epsilon,
            order=order,
            sigma=sigma * scale,
            std=sigma * scale * m["local_clip_norm"] / m["batch_size"],
        )
    assert abs(ledger["1"]["epsilon"] - 4) < 1e-4
    return dict(q=q, sigma=sigma, per_scale=ledger)


def provenance(m):
    base_path = MATRIX.parent / m["base_config"]
    protocol = ROOT / m["protocol"]
    return dict(
        campaign=CAMPAIGN,
        matrix_hash=hash_file(MATRIX),
        base_hash=hash_file(base_path),
        protocol_hash=hash_file(protocol),
        privacy=privacy(m),
        sources=shared.source_closure(
            (Path(__file__), ENTRY, ROOT / "run_experiment.py")
        ),
        private_device="mps",
        fallback=0,
        expected_runs=132,
    )


def config_for(m, task, stamp, thresholds=None):
    cfg = copy.deepcopy(
        yaml.safe_load((MATRIX.parent / m["base_config"]).read_text())["base"]
    )
    cfg.update(seed=task["seed"], device="mps", output_dir=str(directory(task)))
    cfg["data"]["partition_seed"] = task["seed"]
    cfg["clients"].update(
        num_clients=10, min_clients=10, fleet=[dict(type="raspberry_pi_4", count=10)]
    )
    cfg["training"]["algorithm"] = "ldp_gradient_far"
    a = cfg["training"]["algo_config"]
    temporal = task["arm"] != "rfa"
    thresholds = thresholds or {mode: 1.0 for mode in MODES}
    a.update(
        rcig_n10_runtime_implementation="algorithms.rcig_n10_ablation.N10PolicyAblation",
        rcig_n10_arm=task["arm"],
        rcig_n10_pairing_seed=task["seed"],
        rcig_n10_phase=task["phase"],
        rcig_n10_scenario=task["scenario"],
        rcig_campaign_id=CAMPAIGN,
        rcig_campaign_scientific_hash=hash_json(stamp),
        expected_num_clients=10,
        num_byzantine=2,
        privacy_public_dataset_size=6000,
        privacy_sampling_rate_override=0.02,
        noise_multiplier=stamp["privacy"]["sigma"],
        privacy_noise_multiplier_scale_by_client=m["noise_regimes"][task["noise"]],
        robust_reference="rcig_temporal" if temporal else "rfa",
        rcig_persistent_policy=(
            "freeze_hysteresis" if task["arm"] == "freeze" else "rolling"
        ),
        rcig_innovation_threshold=thresholds["full"],
        rcig_isotropic_innovation_threshold=thresholds["isotropic"],
        rcig_euclidean_innovation_threshold=thresholds["euclidean"],
        rcig_recovery_threshold_ratio=0.8,
        rcig_recovery_patience=2,
        rcig_calibration_mode=task["phase"] == "calibration",
        enable_oracle_diagnostics=temporal,
        rcig_oracle_evaluation_only=temporal,
        rcig_oracle_metrics_are_not_release=temporal,
        rcig_oracle_separation_required=temporal,
    )
    if task["phase"] == "comparison":
        a["rcig_threshold_artifact_sha256"] = hash_file(ARTIFACT)
    attacked = task["scenario"] != "none"
    name = task["scenario"].split("_")[0] if attacked else "none"
    a["attack"] = dict(
        enabled=attacked,
        name=name,
        scale=10.0 if name == "bf" else 1.0,
        num_byzantine=2 if attacked else 0,
        client_ids=[0, 1] if attacked else [],
        active_round_start=17,
        active_round_end=40,
    )
    return cfg


def verify_sources(stamp):
    changed = [
        name for name, h in stamp["sources"].items() if hash_file(ROOT / name) != h
    ]
    if changed:
        raise RuntimeError(f"source drift: {changed}")
    if hash_file(MATRIX) != stamp["matrix_hash"]:
        raise RuntimeError("matrix drift")
    m = matrix()
    if (
        hash_file(ROOT / m["protocol"]) != stamp["protocol_hash"]
        or hash_file(MATRIX.parent / m["base_config"]) != stamp["base_hash"]
    ):
        raise RuntimeError("protocol/base drift")


def load_trace(task):
    return [
        json.loads(line)
        for line in (directory(task) / "simulator_randomness_private_audit.jsonl")
        .read_text()
        .splitlines()
    ]


def pairing_audit(task, rows):
    trace = load_trace(task)
    keys = {(r["round"], r["client_id"]) for r in trace}
    assert keys == set(itertools.product(range(1, 41), range(10))) and len(trace) == 400
    trace = sorted(trace, key=lambda r: (r["round"], r["client_id"]))
    # Canonical recent/no-attack path anchors all methods to the SAME actual
    # batch/noise draws. Models may differ after policies start at round13.
    canonical = dict(task, arm="recent", scenario="none")
    if task == canonical:
        return dict(actual_draws_verified=400, same_arm_clean_prefix=None)
    canonical_rows = sorted(
        load_trace(canonical), key=lambda r: (r["round"], r["client_id"])
    )
    for r, s in zip(trace, canonical_rows):
        assert r["permutations"] == s["permutations"], "realized batches unpaired"
        assert (
            r["standard_gaussians"] == s["standard_gaussians"]
        ), "realized Gaussians unpaired"
        if r["round"] <= 12:
            assert r["model_before"] == s["model_before"], "warmup model mismatch"
            assert r["private_upload"] == s["private_upload"], "warmup upload mismatch"
    prefix = None
    if task["scenario"] != "none":
        clean_task = dict(task, scenario="none")
        clean_trace = {(r["round"], r["client_id"]): r for r in load_trace(clean_task)}
        for r in trace:
            s = clean_trace[(r["round"], r["client_id"])]
            # Pre-round17 model/upload still clean; evaluation after17 is not.
            if r["round"] <= 17:
                assert (
                    r["model_before"] == s["model_before"]
                ), "pre-attack model mismatch"
                assert (
                    r["private_upload"] == s["private_upload"]
                ), "pre-attack private upload mismatch"
        clean_payload = json.loads(_metrics_path(directory(clean_task)).read_text())
        for r, s in zip(rows[:16], clean_payload["rounds"][:16]):
            for key in ("test_accuracy", "test_loss"):
                assert abs(r[key] - s[key]) <= 1e-12, "pre-attack evaluation mismatch"
        prefix = True
    return dict(
        actual_draws_verified=400, warmup_equal=True, same_arm_clean_prefix=prefix
    )


def validate(task, cfg, stamp):
    path = _metrics_path(directory(task))
    if path is None:
        raise RuntimeError("missing/nonunique/incomplete metrics")
    payload = json.loads(path.read_text())
    summary, rows = payload["summary"], payload["rounds"]
    for key, value in dict(
        num_clients=10,
        num_rounds=40,
        seed=task["seed"],
        dataset="fashionmnist",
        model="lenet5_tanh",
    ).items():
        assert summary[key] == value, (key, summary[key], value)
    for key, value in cfg["training"]["algo_config"].items():
        assert payload["config"].get(key) == value, f"resolved mismatch {key}"
    _post_run_device_audit(path, cfg)
    for t, r in enumerate(rows, 1):
        assert r["round_num"] == t and r["num_alive_clients"] == 10
        for key in (
            "test_accuracy",
            "test_loss",
            "client_accuracy_mean",
            "client_accuracy_variance_pct2",
            "worst20_accuracy_pct",
            "best20_worst20_gap_pct",
        ):
            assert isinstance(r[key], (int, float)) and math.isfinite(r[key]), key
        if task["arm"] != "rfa" and t >= 13:
            assert r["rcig_newer_round_max"] < t - 1
            assert r["rcig_n10_arm"] == task["arm"]
            assert r["rcig_oracle_was_visible_to_server_aggregate"] is False
            assert math.isfinite(
                r["rcig_reference_squared_l2_error_to_clean_honest_center_oracle"]
            )
        assert abs(r["far_alpha"] - (0 if t <= 12 else 0.1)) < 1e-12
    assert abs(rows[-1]["privacy_epsilon_max"] - 4) < 1e-4
    runtime = json.loads((directory(task) / "runtime_imports.json").read_text())
    assert runtime["stage"] == "after_training" and runtime["mps_fallback"] == 0
    for name, h in runtime["source_sha256"].items():
        assert stamp["sources"].get(name) == h, f"unlocked runtime source {name}"
    return path, payload, pairing_audit(task, rows)


def thresholds_from_calibration(m, all_tasks, stamp):
    maxima = {noise: {mode: [] for mode in MODES} for noise in m["noise_regimes"]}
    sources = {}
    for task in all_tasks:
        if task["phase"] != "calibration":
            continue
        cfg = config_for(m, task, stamp)
        path, payload, _ = validate(task, cfg, stamp)
        sources[str(path.relative_to(ROOT))] = hash_file(path)
        for mode in MODES:
            maxima[task["noise"]][mode].append(
                max(
                    float(r[f"rcig_{mode}_innovation_stat"])
                    for r in payload["rounds"][12:]
                )
            )
    result = dict(
        campaign=CAMPAIGN,
        interpretation="exploratory_not_a_false_alarm_certificate",
        source_reference="recent",
        seed_maxima=maxima,
        sources=sources,
        thresholds={
            noise: {mode: max(values) for mode, values in modes.items()}
            for noise, modes in maxima.items()
        },
    )
    if ARTIFACT.exists():
        assert json.loads(ARTIFACT.read_text()) == result, "calibration drift"
    else:
        write_json(ARTIFACT, result, exclusive=True)
    return result["thresholds"]


def compact_summary(completed):
    groups = {}
    for task, payload in completed:
        if task["phase"] != "comparison":
            continue
        key = f"{task['noise']}/{task['scenario']}/{task['arm']}"
        last = payload["rounds"][-1]
        entry = {"seed": task["seed"]}
        for metric in (
            "test_accuracy",
            "test_loss",
            "client_accuracy_variance_pct2",
            "worst20_accuracy_pct",
            "best20_worst20_gap_pct",
        ):
            entry[metric] = last[metric]
        if task["arm"] != "rfa":
            ready = (
                payload["rounds"][16:]
                if task["scenario"] != "none"
                else payload["rounds"][12:]
            )
            entry["deployed_reference_error"] = statistics.mean(
                r["rcig_reference_squared_l2_error_to_clean_honest_center_oracle"]
                for r in ready
            )
            entry["rolling_alarm_fraction"] = statistics.mean(
                r["rcig_gate_active"] for r in ready
            )
            entry["freeze_fraction"] = statistics.mean(
                r["rcig_persistent_frozen_after_commit"] for r in ready
            )
        groups.setdefault(key, []).append(entry)
    out = {}
    for key, values in groups.items():
        out[key] = {"n_seeds": len(values), "seeds": values, "mean_sd": {}}
        for metric in values[0]:
            if metric == "seed":
                continue
            nums = [v[metric] for v in values]
            out[key]["mean_sd"][metric] = dict(
                mean=statistics.mean(nums),
                sd=statistics.stdev(nums) if len(nums) > 1 else None,
            )
    write_json(
        OUTPUT / "summary_progress.json",
        dict(interpretation="exploratory_not_promoted", groups=out),
    )
    # Human-readable numeric delivery alongside the complete JSON. Never
    # declare a scientific winner from partial cells or selected activations.
    n = sum(len(v) for v in groups.values())
    lines = [
        "# RCIG N10 — résultats descriptifs",
        "",
        f"Comparaisons terminées et vérifiées : {n}/120. Calibration distincte : 12 runs.",
        "",
        "Écran exploratoire, sans promotion des anciens R1/R2/R3. Moyenne ± écart-type entre seeds ; un tiret signifie que l’écart-type n’est pas estimable avec une seule seed. Les cellules incomplètes ne permettent pas une conclusion finale.",
        "",
        "## Performance au dernier tour",
        "",
        "| Régime / scénario / bras | Seeds | Test Acc. (%) | Test loss | Variance (pp²) | Worst-20 (%) | Gap (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def display(row, metric, scale=1):
        item = row["mean_sd"].get(metric)
        if item is None:
            return "non mesuré"
        sd = "—" if item["sd"] is None else f"{item['sd']*scale:.4f}"
        return f"{item['mean']*scale:.4f} ± {sd}"

    for key, row in out.items():
        values = [display(row, "test_accuracy", 100)] + [
            display(row, metric)
            for metric in (
                "test_loss",
                "client_accuracy_variance_pct2",
                "worst20_accuracy_pct",
                "best20_worst20_gap_pct",
            )
        ]
        lines.append(f"| {key} | {row['n_seeds']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Mécanisme déployé",
        "",
        "Moyennes temporelles calculées d’abord par seed : tours 17–40 sous attaque, 13–40 sans attaque. La référence au tour 17 n’a pas encore vu les messages attaqués. Les fenêtres 18–24 et 25–40 restent séparables dans les métriques originales pour l’analyse finale.",
        "",
        "L’alerte roulante est un diagnostic, pas une correction exécutée dans les bras recent/midpoint. L’erreur est la norme L2 au carré à un centre honnête propre oracle de simulation, pas une loss de classification.",
        "",
        "| Régime / scénario / bras | Seeds | Erreur de référence déployée | Tours avec alerte (%) | Tours avec gel (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in out.items():
        if "deployed_reference_error" not in row["mean_sd"]:
            continue
        lines.append(
            f"| {key} | {row['n_seeds']} | {display(row,'deployed_reference_error')} | {display(row,'rolling_alarm_fraction',100)} | {display(row,'freeze_fraction',100)} |"
        )
    lines += [
        "",
        "Les résultats par seed non arrondis sont dans `summary_progress.json`. Les artefacts par tour restent dans chaque run. Aucun chiffre manquant n’est remplacé par zéro.",
        "",
    ]
    (OUTPUT / "Results_Progress.md").write_text("\n".join(lines), encoding="utf-8")


@contextmanager
def execution_lock():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT.parent / f".{CAMPAIGN}.execution.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(m, stamp, *, max_new=None):
    completed = []
    new = 0
    thresholds = None
    for task in tasks(m):
        verify_sources(stamp)
        if task["phase"] == "comparison" and thresholds is None:
            thresholds = thresholds_from_calibration(m, tasks(m), stamp)
        cfg = config_for(
            m, task, stamp, None if thresholds is None else thresholds[task["noise"]]
        )
        dest = directory(task)
        status_path = dest / "orchestration_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["status"] != "completed":
                raise RuntimeError(
                    f"incomplete/failed run requires explicit inspection: {dest}"
                )
            assert yaml.safe_load((dest / "resolved_config.yaml").read_text()) == cfg
            path, payload, pairing = validate(task, cfg, stamp)
            assert status["metrics_sha256"] == hash_file(path)
            assert status["trace_sha256"] == hash_file(
                dest / "simulator_randomness_private_audit.jsonl"
            )
        else:
            if max_new is not None and new >= max_new:
                break
            if dest.exists() and any(dest.iterdir()):
                raise RuntimeError(f"unlocked partial directory: {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "resolved_config.yaml").open("x") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            base = dict(
                task=task,
                config_sha256=hash_file(dest / "resolved_config.yaml"),
                device="mps",
                fallback=0,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            write_json(status_path, dict(base, status="running"))
            print(
                f"START {len(completed)+1}/132 {task['phase']}/{run_id(task)}",
                flush=True,
            )
            try:
                env = dict(
                    os.environ,
                    PYTORCH_ENABLE_MPS_FALLBACK="0",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                with (dest / "training.log").open("x") as log:
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-B",
                            "-u",
                            str(ENTRY),
                            "--config",
                            str(dest / "resolved_config.yaml"),
                            "--output",
                            str(dest),
                            "--device",
                            "mps",
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    write_json(status_path, dict(base, status="running", pid=p.pid))
                    code = p.wait()
                if code:
                    raise RuntimeError(
                        f"training process exit={code}; see {dest/'training.log'}"
                    )
                verify_sources(stamp)
                path, payload, pairing = validate(task, cfg, stamp)
                write_json(
                    status_path,
                    dict(
                        base,
                        status="completed",
                        metrics_sha256=hash_file(path),
                        trace_sha256=hash_file(
                            dest / "simulator_randomness_private_audit.jsonl"
                        ),
                        pairing=pairing,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    ),
                )
                print(
                    f"DONE {len(completed)+1}/132 test_acc={payload['rounds'][-1]['test_accuracy']:.6f}",
                    flush=True,
                )
                new += 1
            except BaseException as exc:
                write_json(status_path, dict(base, status="failed", error=str(exc)))
                raise
        completed.append((task, payload))
        compact_summary(completed)
        write_json(
            OUTPUT / "progress.json",
            dict(
                completed=len(completed),
                total=132,
                current=task,
                status="completed" if len(completed) == 132 else "active",
            ),
        )
    return len(completed)


def status(m):
    result = {
        phase: dict(
            completed=0,
            total=12 if phase == "calibration" else 120,
            active=[],
            failed=[],
            missing=0,
        )
        for phase in ("calibration", "comparison")
    }
    for task in tasks(m):
        p = directory(task) / "orchestration_status.json"
        group = result[task["phase"]]
        if not p.exists():
            group["missing"] += 1
            continue
        s = json.loads(p.read_text())
        if s["status"] == "completed":
            group["completed"] += 1
        else:
            group["active" if s["status"] == "running" else "failed"].append(
                dict(run=run_id(task), **s)
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "status", "run", "launch"):
        action.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    m = matrix()
    if args.status:
        print(json.dumps(status(m), indent=2))
        return
    stamp = provenance(m)
    if args.plan:
        print(
            json.dumps(
                dict(
                    total=132,
                    calibration=12,
                    comparison=120,
                    privacy=stamp["privacy"],
                    sources=len(stamp["sources"]),
                    output=str(OUTPUT),
                    arms=m["arms"],
                    seeds=m["comparison_seeds"],
                ),
                indent=2,
            )
        )
        return
    if not args.resume:
        raise ValueError("--resume required; no overwrite")
    shared.require_working_mps()
    if args.launch:
        with execution_lock():
            shared.check_no_other_training_process()
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            log_path = ROOT / "logs/rcig_n10_policy_ablation_v1.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as log:
                cmd = [
                    sys.executable,
                    "-B",
                    "-u",
                    str(Path(__file__).resolve()),
                    "--run",
                    "--resume",
                ]
                if args.max_new is not None:
                    cmd += ["--max-new", str(args.max_new)]
                child = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=dict(
                        os.environ,
                        PYTORCH_ENABLE_MPS_FALLBACK="0",
                        PYTHONDONTWRITEBYTECODE="1",
                    ),
                )
            print(json.dumps(dict(pid=child.pid, log=str(log_path))))
            return
    with execution_lock():
        shared.check_no_other_training_process()
        lock = OUTPUT / "campaign_lock.json"
        if lock.exists():
            assert json.loads(lock.read_text()) == stamp, "source/protocol drift"
        else:
            if OUTPUT.exists() and any(OUTPUT.iterdir()):
                raise RuntimeError("preexisting unlocked campaign")
            OUTPUT.mkdir(parents=True, exist_ok=True)
            write_json(lock, stamp, exclusive=True)
        try:
            execute(m, stamp, max_new=args.max_new)
        except BaseException as exc:
            write_json(
                OUTPUT / "failure.json",
                dict(error=str(exc), at=datetime.now(timezone.utc).isoformat()),
            )
            raise


if __name__ == "__main__":
    main()
