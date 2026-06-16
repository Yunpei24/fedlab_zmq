"""
datasets/femnist_loader.py
==========================
Reader for LEAF-preprocessed FEMNIST (Federated EMNIST, Caldas et al. 2018).

LEAF emits per-writer JSON files under
    <femnist_dir>/train/*.json
    <femnist_dir>/test/*.json
each of the form
    {"users": ["f0000_14", ...],
     "num_samples": [n, ...],
     "user_data": {"f0000_14": {"x": [[784 floats in [0,1]], ...],
                                "y": [int label 0..61, ...]}}}

This loader concatenates all writers into one pooled dataset while recording
which writer produced each sample (`writer_ids`). The registry uses that to
build the NATURAL per-writer federated partition (1 writer -> 1 client, or a
contiguous block of writers per client when num_clients < num_writers).

Generate the JSON with scripts/prepare_femnist.sh (LEAF preprocess.sh).
Images are returned as float tensors in [0,1] of shape (1, 28, 28); LeNet5
trains directly on these, so no PIL transform is applied here.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FEMNISTDataset(Dataset):
    def __init__(self, femnist_dir, split: str = "train", transform=None):
        split_dir = Path(femnist_dir) / split
        files = sorted(glob.glob(str(split_dir / "*.json")))
        if not files:
            raise FileNotFoundError(
                f"No LEAF FEMNIST JSON found in {split_dir}. "
                f"Run scripts/prepare_femnist.sh first."
            )

        xs, ys, wids, users = [], [], [], []
        for fpath in files:
            with open(fpath) as fh:
                blob = json.load(fh)
            for u in blob["users"]:
                ud = blob["user_data"][u]
                w = len(users)            # integer writer index
                users.append(u)
                xs.append(np.asarray(ud["x"], dtype=np.float32))
                yu = np.asarray(ud["y"], dtype=np.int64)
                ys.append(yu)
                wids.append(np.full(len(yu), w, dtype=np.int64))

        X = np.concatenate(xs, axis=0).reshape(-1, 1, 28, 28)
        self.data = torch.from_numpy(X)                          # [N,1,28,28] in [0,1]
        self.targets = torch.from_numpy(np.concatenate(ys, axis=0))
        self.writer_ids = torch.from_numpy(np.concatenate(wids, axis=0))
        self.users = users                                       # writer string ids
        self.transform = transform
        self.natural_partition = True                            # tag for registry

    @property
    def num_writers(self) -> int:
        return len(self.users)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, i):
        x = self.data[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.targets[i])
