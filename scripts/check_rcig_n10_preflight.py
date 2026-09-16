#!/usr/bin/env python3
"""Three short real-data MPS integration checks, outside scientific results.

18 rounds exercise warmup, temporal deployment and the first two attacked
rounds. Privacy sigma stays calibrated for the future 40-round horizon.
No preflight outcome is used to tune a parameter or choose a candidate.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts import run_rcig_n10_policy_ablation as runner


def main():
    runner.shared.require_working_mps()
    runner.shared.check_no_other_training_process()
    m = runner.matrix()
    stamp = runner.provenance(m)
    parent = ROOT / "output/validation"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="rcig_n10_preflight_", dir=parent))
    print(out, flush=True)
    records = {}
    for arm, scenario in (
        ("recent", "none"),
        ("recent", "bf_persistent"),
        ("freeze", "bf_persistent"),
    ):
        task = dict(
            phase="calibration",
            noise="homogeneous",
            seed=909001,
            scenario=scenario,
            arm=arm,
        )
        cfg = runner.config_for(m, task, stamp)
        dest = out / f"{arm}_{scenario}"
        dest.mkdir()
        cfg["output_dir"] = str(dest)
        cfg["training"]["num_rounds"] = 18
        a = cfg["training"]["algo_config"]
        a["rcig_n10_integration_test_not_scientific_result"] = True
        path = dest / "config.yaml"
        with path.open("x") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"START {arm}/{scenario}", flush=True)
        with (dest / "training.log").open("x") as log:
            p = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-u",
                    str(runner.ENTRY),
                    "--config",
                    str(path),
                    "--output",
                    str(dest),
                    "--device",
                    "mps",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    PYTORCH_ENABLE_MPS_FALLBACK="0",
                    PYTHONDONTWRITEBYTECODE="1",
                ),
            )
        if p.returncode:
            raise RuntimeError(f"preflight failed: {dest/'training.log'}")
        paths = list(dest.rglob("metrics.json"))
        assert len(paths) == 1
        payload = json.loads(paths[0].read_text())
        assert len(payload["rounds"]) == 18
        for row in payload["rounds"]:
            assert row["ldp_gradient_far_private_gradient_mps_fraction"] == 1
            assert row["far_attack_labels_visible_to_server_aggregate"] is False
            if row["round_num"] >= 13:
                assert row["rcig_oracle_was_visible_to_server_aggregate"] is False
                assert row["rcig_n10_arm"] == arm
        trace = [
            json.loads(s)
            for s in (dest / "simulator_randomness_private_audit.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(trace) == 180
        records[(arm, scenario)] = (payload, trace)
        runtime = json.loads((dest / "runtime_imports.json").read_text())
        for name, h in runtime["source_sha256"].items():
            assert stamp["sources"].get(name) == h, f"unlocked runtime source {name}"
        print(f"PASS {arm}/{scenario}", flush=True)
    clean, clean_trace = records[("recent", "none")]
    attacked, attacked_trace = records[("recent", "bf_persistent")]
    frozen, frozen_trace = records[("freeze", "bf_persistent")]
    for pairs in (zip(clean_trace, attacked_trace), zip(clean_trace, frozen_trace)):
        for a, b in pairs:
            assert a["permutations"] == b["permutations"]
            assert a["standard_gaussians"] == b["standard_gaussians"]
    for a, b in zip(clean_trace, attacked_trace):
        if a["round"] <= 17:
            assert a["model_before"] == b["model_before"]
            assert a["private_upload"] == b["private_upload"]
    for a, b in zip(clean["rounds"][:16], attacked["rounds"][:16]):
        assert (
            a["test_accuracy"] == b["test_accuracy"]
            and a["test_loss"] == b["test_loss"]
        )
    for a, b in zip(clean_trace, frozen_trace):
        if a["round"] <= 12:
            assert a["private_upload"] == b["private_upload"]
    assert any(
        row.get("rcig_persistent_frozen_after_commit", False)
        for row in frozen["rounds"]
    )
    runner.verify_sources(stamp)
    result = dict(
        status="passed",
        real_data_mps_runs=3,
        rounds_each=18,
        exact_clean_attack_prefix=True,
        exact_standard_gaussians_and_batches=True,
        frozen_policy_exercised=True,
        source_hashes_unchanged=True,
        output=str(out),
    )
    runner.write_json(out / "verification.json", result, exclusive=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
