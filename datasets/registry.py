"""
datasets/registry.py
====================
Unified dataset registry for FedLab.

Supported datasets:
  mnist, fashionmnist, cifar10, cifar100,
  femnist, tiny_imagenet

All datasets are auto-downloaded on first use.
"""

import warnings
from numpy.exceptions import VisibleDeprecationWarning as _NpVisibleDeprecationWarning
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional

from datasets.partitioner import partition_dataset

# ---------------------------------------------------------------------------
# Suppress the NumPy 2.4 VisibleDeprecationWarning emitted by torchvision's
# CIFAR pickle loader.  The warning fires because torchvision passes
# `align=0` (an integer) to numpy.dtype(), which NumPy 2.4 now requires to
# be a Python/NumPy boolean.  This is a torchvision upstream bug; the
# workaround here keeps our logs clean until a fixed torchvision release is
# available.
# Note: in NumPy 2.x this class moved to numpy.exceptions.
warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean",
    category=_NpVisibleDeprecationWarning,
    module=r"torchvision\.datasets\.cifar",
)

# ---------------------------------------------------------------------------
# pin_memory is only beneficial (and supported) when the training device is
# CUDA.  On MPS (Apple Silicon) PyTorch emits a UserWarning and disables it
# anyway; on CPU it has no effect.  We therefore set it True only for CUDA.
_PIN_MEMORY: bool = torch.cuda.is_available()
# On macOS, forked DataLoader workers consume enormous virtual memory (each worker
# inherits the full Python address space via copy-on-write).  Use 0 workers on macOS
# to avoid 60+ forked processes with 30 clients.
import platform as _platform
_NUM_WORKERS: int = 0 if _platform.system() == "Darwin" else 2


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "mnist": {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
    },
    "fashionmnist": {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
    },
    "cifar10": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ]),
    },
    "cifar100": {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ]),
    },
    "tiny_imagenet": {
        "train": transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ]),
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ]),
    },
}

# Number of classes per dataset
NUM_CLASSES = {
    "mnist": 10,
    "fashionmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "femnist": 62,
    "tiny_imagenet": 200,
}

# Input shape (C, H, W)
INPUT_SHAPE = {
    "mnist":        (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "cifar10":      (3, 32, 32),
    "cifar100":     (3, 32, 32),
    "femnist":      (1, 28, 28),
    "tiny_imagenet":(3, 64, 64),
}


# ─────────────────────────────────────────────────────────────────────────────
# Raw dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_raw_dataset(name: str, split: str, data_root: str):
    """Load raw torchvision dataset (no partitioning)."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train = (split == "train")
    tf = TRANSFORMS.get(name, {}).get(split, transforms.ToTensor())

    if name == "mnist":
        return datasets.MNIST(root, train=train, transform=tf, download=True)
    elif name == "fashionmnist":
        return datasets.FashionMNIST(root, train=train, transform=tf, download=True)
    elif name == "cifar10":
        return datasets.CIFAR10(root, train=train, transform=tf, download=True)
    elif name == "cifar100":
        return datasets.CIFAR100(root, train=train, transform=tf, download=True)
    elif name == "tiny_imagenet":
        return _load_tiny_imagenet(root, split, tf)
    elif name == "femnist":
        return _load_femnist(root, split, tf)
    else:
        raise ValueError(f"Unknown dataset: {name}. "
                         f"Available: {list(NUM_CLASSES.keys())}")


def _load_tiny_imagenet(root: Path, split: str, tf):
    """Load TinyImageNet from ImageFolder (auto-download if missing)."""
    data_dir = root / "tiny-imagenet-200"
    if not data_dir.exists():
        _download_tiny_imagenet(root)
    folder = data_dir / ("train" if split == "train" else "val")
    return datasets.ImageFolder(str(folder), transform=tf)


def _download_tiny_imagenet(root: Path):
    """Download and extract TinyImageNet."""
    import urllib.request, zipfile, shutil
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = root / "tiny-imagenet-200.zip"
    print(f"[datasets] Downloading TinyImageNet to {root} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)
    zip_path.unlink()

    # Fix val folder structure (ImageFolder expects class subfolders)
    val_dir = root / "tiny-imagenet-200" / "val"
    annotations = val_dir / "val_annotations.txt"
    with open(annotations) as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split("\t")
        img, cls = parts[0], parts[1]
        cls_dir = val_dir / cls
        cls_dir.mkdir(exist_ok=True)
        src = val_dir / "images" / img
        dst = cls_dir / img
        if src.exists():
            shutil.move(str(src), str(dst))

    print("[datasets] TinyImageNet ready.")


def _load_femnist(root: Path, split: str, tf):
    """
    Load FEMNIST from LEAF project.
    Falls back to EMNIST if LEAF data not available.
    """
    femnist_dir = root / "femnist"
    if femnist_dir.exists():
        from datasets.femnist_loader import FEMNISTDataset
        return FEMNISTDataset(femnist_dir, split=split, transform=tf)
    else:
        print("[datasets] FEMNIST (LEAF) not found, falling back to EMNIST/ByClass")
        return datasets.EMNIST(root, split="byclass",
                               train=(split == "train"),
                               transform=tf, download=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    dataset_name: str,
    split: str,
    partition: str,
    client_id: Optional[int],
    num_clients: int,
    alpha: float = 0.5,
    num_shards: int = 200,
    batch_size: int = 32,
    num_workers: int = _NUM_WORKERS,
    seed: int = 42,
    data_root: str = "./data",
) -> DataLoader:
    """
    Get a DataLoader for a specific FL client.

    Args:
        dataset_name: "cifar10", "cifar100", "mnist", "fashionmnist",
                      "femnist", "tiny_imagenet"
        split:        "train" or "test"
        partition:    "iid", "dirichlet", "pathological", or "natural"
        client_id:    Which client's shard to load. None = full dataset (for server eval)
        num_clients:  Total number of clients
        alpha:        Dirichlet alpha (non-IID degree)
        batch_size:   Mini-batch size
        seed:         Random seed for reproducibility

    Returns:
        DataLoader for the specified client's local dataset.
    """
    raw = _load_raw_dataset(dataset_name, split, data_root)

    # Server evaluation: use full test set
    if client_id is None or split == "test":
        return DataLoader(
            raw,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_PIN_MEMORY,
        )

    # Client training: partition the training set
    if partition == "natural" or dataset_name == "femnist":
        # FEMNIST natural split: each client = one writer
        # The dataset itself handles this if loaded from LEAF
        subset = raw  # already client-specific if using FEMNISTDataset
    else:
        subsets = partition_dataset(
            dataset=raw,
            num_clients=num_clients,
            partition=partition,
            alpha=alpha,
            num_shards=num_shards,
            seed=seed,
        )
        if client_id >= len(subsets):
            raise ValueError(f"client_id={client_id} >= num_clients={num_clients}")
        subset = subsets[client_id]

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
        drop_last=True,  # avoid incomplete last batch
    )


def get_dataset_info(name: str) -> dict:
    return {
        "name": name,
        "num_classes": NUM_CLASSES.get(name, "?"),
        "input_shape": INPUT_SHAPE.get(name, "?"),
    }
