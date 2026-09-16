"""Build EMNIST/ByClass idx files from the TensorFlow Federated HDF5 bundle.

Only needed where the canonical NIST download is unreachable (the cloud sandbox
blocks biometrics.nist.gov with a 403). On a normal machine
``torchvision.datasets.EMNIST(root, split="byclass", download=True)`` works and
this script is unnecessary.

IMPORTANT -- the two sources are not interchangeable. The canonical NIST raw
images are stored transposed relative to the MNIST convention and torchvision
does not undo that; the TFF bundle is already upright. A model trained on one
sees the other's images rotated, so accuracies are broadly similar but runs from
the two sources are NOT comparable and must never be pooled into the same table.
Pick one source per campaign.

Usage:
    python3 scripts/prepare_emnist_byclass.py --data-root ./data
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import tarfile
from pathlib import Path

import numpy as np

TFF_URL = "https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2"


def _write_idx(images: np.ndarray, labels: np.ndarray, stem: Path) -> None:
    with open(f"{stem}-images-idx3-ubyte", "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, images.shape[0], 28, 28))
        handle.write(images.tobytes())
    with open(f"{stem}-labels-idx1-ubyte", "wb") as handle:
        handle.write(struct.pack(">II", 2049, labels.shape[0]))
        handle.write(labels.tobytes())


def _convert(h5_path: Path, stem: Path) -> tuple[int, int]:
    import h5py

    group = h5py.File(h5_path, "r")["examples"]
    pixel_blocks, label_blocks = [], []
    for client in group.keys():
        pixel_blocks.append(np.array(group[client]["pixels"]))
        label_blocks.append(np.array(group[client]["label"]))
    pixels = np.concatenate(pixel_blocks)
    labels = np.concatenate(label_blocks).astype(np.uint8)
    # TFF stores light background / dark ink; MNIST-style loaders expect the
    # opposite, and the normalisation constants downstream assume it.
    if pixels.mean() > 0.5:
        pixels = 1.0 - pixels
    _write_idx((pixels * 255).round().clip(0, 255).astype(np.uint8), labels, stem)
    return pixels.shape[0], len(np.unique(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    root = Path(args.data_root)
    raw = root / "EMNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = root / "fed_emnist.tar.bz2"

    if not archive.exists():
        print(f"telechargement {TFF_URL} (~170 Mo)")
        subprocess.run(["curl", "-sS", "-o", str(archive), TFF_URL], check=True)
    if not (root / "fed_emnist_train.h5").exists():
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(root)

    for split, stem in (("train", "train"), ("test", "test")):
        count, classes = _convert(
            root / f"fed_emnist_{split}.h5", raw / f"emnist-byclass-{stem}"
        )
        print(f"  {split}: {count} exemples, {classes} classes")
    print(f"\nEcrit dans {raw}/ ; torchvision le lira avec download=False.")


if __name__ == "__main__":
    main()
