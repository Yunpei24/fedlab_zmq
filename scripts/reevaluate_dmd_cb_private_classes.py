#!/usr/bin/env python3
"""Authorized post-hoc MPS evaluation of 36 FINAL checkpoints on public test.

Never trains, downloads, changes raw results or loads private training examples.
Reconstructs the exact saved test partition and validates recorded accuracies
and client-present balanced accuracies before accepting newly computed classes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAMPAIGN = ROOT / "results/ldp_gradient_far/dmd_cb_private_v1"
DESTINATION = ROOT / "output/analysis/DMD_CB_Private_36_Class_Evidence.json"
ACCURACY_TOL = 1e-12
LOSS_TOL = 5e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Evaluate all 36 and exclusively create class evidence")
    parser.add_argument("--smoke", action="store_true", help="Evaluate first checkpoint only, write nothing")
    args = parser.parse_args()
    require(not (args.run and args.smoke), "Choose run or smoke")
    require(not args.run or not DESTINATION.exists(), "Refusing to overwrite class evidence")
    require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "MPS fallback must be disabled")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    from torchvision.datasets import FashionMNIST
    import yaml
    from datasets.registry import TRANSFORMS
    from datasets.partitioner import partition_dataset
    from models.registry import get_model

    require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback authorized")
    paths = sorted((CAMPAIGN / "runs").rglob("metrics.json"))
    require(len(paths) == 36, "Expected 36 frozen runs")
    lock_path = CAMPAIGN / "scientific_lock.json"
    lock = read(lock_path)
    sources = ("datasets/registry.py", "datasets/partitioner.py", "models/registry.py", "metrics/client_fairness.py", "run_experiment.py")
    source_hashes = {name: sha(ROOT / name) for name in sources}
    for name, digest in source_hashes.items():
        require(lock["provenance"]["source_sha256"][name] == digest, f"Frozen evaluation source differs: {name}")
    fingerprints = {**source_hashes, rel(lock_path): sha(lock_path), rel(Path(__file__)): sha(Path(__file__))}
    # Read exact data_root from each saved resolved YAML; never ask torchvision
    # to download or reconstruct training datasets.
    configs = {}
    for path in paths:
        config_path = path.parent.parent / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        require(config["data"]["dataset"] == "fashionmnist" and config["data"]["model"] == "lenet5_tanh", "Unexpected dataset/model")
        require(config["data"]["partition"] == "client_dirichlet_balanced" and config["data"]["alpha"] == 0.1, "Unexpected partition")
        require(config["training"]["num_rounds"] == 40 and config["clients"]["num_clients"] == 25, "Unexpected T/N")
        configs[rel(path)] = config
        fingerprints[rel(config_path)] = sha(config_path)
        fingerprints[rel(path)] = sha(path)
        manifest = path.with_name("manifest.json")
        manifest_data = read(manifest)
        require(manifest_data["partition_seed"] == config["data"]["partition_seed"], "Manifest seed mismatch")
        require(manifest_data["packages"]["torch"] == torch.__version__, "Use the exact training PyTorch version for reevaluation")
        fingerprints[rel(manifest)] = sha(manifest)
        checkpoint = path.with_name("final_model.pt")
        require(checkpoint.is_file(), f"Missing final checkpoint {checkpoint}")
        fingerprints[rel(checkpoint)] = sha(checkpoint)
    data_roots = {(ROOT / c["data"]["data_root"]).resolve() for c in configs.values()}
    require(len(data_roots) == 1, "Different dataset roots")
    data_root = next(iter(data_roots))
    raw_paths = [data_root / "FashionMNIST/raw" / name for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")]
    require(all(p.is_file() for p in raw_paths), "Test dataset absent: download forbidden; ask user/root")
    for p in raw_paths:
        fingerprints[rel(p)] = sha(p)
    raw = FashionMNIST(str(data_root), train=False, transform=TRANSFORMS["fashionmnist"]["test"], download=False)
    require(len(raw) == 10000, "Expected 10000 public test examples")
    labels = np.asarray(raw.targets, dtype=np.int64)
    require(np.bincount(labels, minlength=10).tolist() == [1000] * 10, "Test class support mismatch")
    partitions = {}
    for seed in (24, 42, 72):
        subsets = partition_dataset(raw, num_clients=25, partition="client_dirichlet_balanced", alpha=0.1, seed=seed, matched_dirichlet=True)
        indices = [list(map(int, subset.indices)) for subset in subsets]
        require(all(len(x) == 400 for x in indices), "Expected 400 test examples per client")
        require(sorted(i for x in indices for i in x) == list(range(10000)), "Partition overlaps or omits test examples")
        supports = [np.bincount(labels[x], minlength=10).tolist() for x in indices]
        partitions[str(seed)] = {"partition_seed": seed, "client_ids": list(range(25)),
                                 "test_indices_by_client": indices, "class_support_by_client": supports,
                                 "class_present_mask_by_client": [[n > 0 for n in row] for row in supports]}
    print(json.dumps({"preflight": "ok", "mps": True, "checkpoints": 36, "download": False,
                      "test_size": 10000, "support_classes_per_client": {s: sorted({sum(row) for row in p['class_present_mask_by_client']}) for s, p in partitions.items()}}), flush=True)
    if not (args.run or args.smoke):
        return
    # Cache ONLY the exact deterministic test transform on CPU. Preserving the
    # 256-example boundaries and subset order matches both historical loaders.
    images = torch.stack([raw[i][0] for i in range(len(raw))])
    targets = torch.as_tensor(labels.copy(), dtype=torch.long)
    model = get_model("lenet5_tanh", "fashionmnist").to("mps")
    model.eval()
    criterion_mean = torch.nn.CrossEntropyLoss(reduction="mean")
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")

    def evaluate(indices, batch_size, global_loader=False):
        confusion = np.zeros((10, 10), dtype=np.int64)
        loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                ix = indices[start:start + batch_size]
                x, y = images[ix].to("mps"), targets[ix].to("mps")
                logits = model(x)
                pred = logits.argmax(1).to("cpu").numpy()
                truth = labels[ix]
                confusion += np.bincount(truth * 10 + pred, minlength=100).reshape(10, 10)
                loss_sum += float(criterion_mean(logits, y).item()) * len(ix) if global_loader else float(criterion_sum(logits, y).item())
        support = confusion.sum(axis=1)
        correct = confusion.diagonal()
        recalls = [int(c) / int(n) if n else None for c, n in zip(correct, support)]
        balanced = sum(v for v in recalls if v is not None) / sum(n > 0 for n in support)
        return {"support": support.tolist(), "correct": correct.tolist(), "recall": recalls,
                "present_mask": (support > 0).tolist(), "confusion_true_rows_pred_columns": confusion.tolist(),
                "accuracy": int(correct.sum()) / len(indices), "balanced_accuracy_present_classes": float(balanced),
                "cross_entropy_loss": loss_sum / len(indices)}

    records = []
    started = time.monotonic()
    for position, path in enumerate(paths[:1] if args.smoke else paths, 1):
        config = configs[rel(path)]
        payload = read(path)
        stored = payload["rounds"][-1]
        require(stored["round_num"] == 40, "Checkpoint endpoint metrics missing")
        seed = config["data"]["partition_seed"]
        require(stored["evaluated_client_ids_oracle"] == list(range(25)), "Stored evaluation mask differs")
        state = torch.load(path.with_name("final_model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        global_result = evaluate(list(range(10000)), 256, global_loader=True)
        batch = int(payload["config"].get("client_eval_batch_size", 256))
        clients = [evaluate(ix, batch) for ix in partitions[str(seed)]["test_indices_by_client"]]
        for cid, client in enumerate(clients):
            require(client["support"] == partitions[str(seed)]["class_support_by_client"][cid], "Reconstructed support mismatch")
        errors = {
            "test_accuracy_abs_error": abs(global_result["accuracy"] - stored["test_accuracy"]),
            "test_loss_abs_error": abs(global_result["cross_entropy_loss"] - stored["test_loss"]),
            "client_accuracy_max_abs_error": max(abs(v["accuracy"] - saved) for v, saved in zip(clients, stored["client_accuracy_values_oracle"])),
            "client_balanced_accuracy_max_abs_error": max(abs(v["balanced_accuracy_present_classes"] - saved) for v, saved in zip(clients, stored["client_balanced_accuracy_values_oracle"])),
            "client_loss_max_abs_error": max(abs(v["cross_entropy_loss"] - saved) for v, saved in zip(clients, stored["client_loss_values_oracle"])),
        }
        for name, error in errors.items():
            tolerance = LOSS_TOL if "loss" in name else ACCURACY_TOL
            require(error <= tolerance, f"Run {path.parent.parent.name}: {name}={error} exceeds {tolerance}")
        noise, seed_text, arm = path.parent.parent.name.split("__")
        records.append({"noise": noise, "seed": int(seed_text[4:]), "arm": arm,
                        "metrics_path": rel(path), "checkpoint_path": rel(path.with_name("final_model.pt")),
                        "round": 40, "validation": errors, "global": global_result,
                        "clients": [{"client_id": cid, **v} for cid, v in enumerate(clients)]})
        print(json.dumps({"validated_checkpoint": position, "of": 1 if args.smoke else 36,
                          "arm": path.parent.parent.name, "elapsed_s": round(time.monotonic() - started, 2),
                          "reproduction_errors": errors}), flush=True)
    for name, digest in fingerprints.items():
        require(sha(ROOT / name) == digest, f"Input changed during reevaluation: {name}")
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_status": "authorized_posthoc_public_test_reevaluation_not_historically_saved_class_metrics",
              "campaign_id": "dmd_cb_private_v1", "scientific_hash": lock["scientific_hash"],
              "validated_final_checkpoints": len(records), "device": "mps", "mps_fallback": 0,
              "training_performed": False, "downloads_performed": False, "raw_results_modified": False,
              "torch_version": torch.__version__, "evaluation_elapsed_s": time.monotonic() - started,
              "class_ids": list(range(10)), "class_names": raw.classes,
              "preprocessing": repr(TRANSFORMS["fashionmnist"]["test"]),
              "test_class_support": [1000] * 10, "global_batch_size": 256,
              "support_policy": "client BA averages only classes with positive test support; absent recalls null, never zero-imputed",
              "validation_tolerances": {"accuracy_and_balanced_accuracy": ACCURACY_TOL, "loss": LOSS_TOL},
              "max_reproduction_errors": {k: max(r["validation"][k] for r in records) for k in records[0]["validation"]},
              "input_and_evaluator_sha256": fingerprints, "partitions": partitions, "records": records}
    if args.run:
        require(len(records) == 36, "Only all 36 validated checkpoints may be published")
        with DESTINATION.open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"complete": len(records), "written": rel(DESTINATION) if args.run else None,
                      "max_reproduction_errors": result["max_reproduction_errors"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
