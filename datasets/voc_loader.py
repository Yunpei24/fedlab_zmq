"""
datasets/voc_loader.py
======================
PASCAL VOC 2012 dataloader with FL partitioning.

Supports:
  - Dirichlet partitioning by dominant class (non-IID, controlled by alpha)
  - IID partitioning (uniform random split)
  - SSDLite320-compatible transforms (resize 300×300, ImageNet normalize)
  - Proper collate for variable-size bounding box targets

Usage:
    from datasets.voc_loader import get_voc_dataloader, get_voc_test_loader

    # Get client 3 of 20 with Dirichlet(alpha=0.5)
    loader = get_voc_dataloader(
        data_root="./data", split="train",
        client_id=3, num_clients=20,
        alpha=0.5, partition="dirichlet",
        batch_size=8, seed=42,
    )
    # Get global test set
    test_loader = get_voc_test_loader(data_root="./data", batch_size=16)

VOC classes (20 object classes, index 0 = background):
  1: aeroplane  2: bicycle    3: bird       4: boat       5: bottle
  6: bus        7: car        8: cat        9: chair      10: cow
  11: diningtable 12: dog     13: horse     14: motorbike 15: person
  16: pottedplant 17: sheep   18: sofa      19: train     20: tvmonitor
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import VOCDetection

# ─────────────────────────────────────────────────────────────────────────────
# VOC class mapping
# ─────────────────────────────────────────────────────────────────────────────

VOC_CLASSES = [
    "__background__",  # 0
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 21 (including background)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def _voc_transforms():
    """Image-only transform (SSDLite handles resize internally via model.transform)."""
    return T.Compose([
        T.ToTensor(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Target parsing (VOC XML → {boxes, labels} dict)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_voc_target(annotation: dict, image_id: str) -> dict:
    """
    Convert VOCDetection annotation dict to SSDLite-compatible target.

    Returns:
        {
          "boxes":    FloatTensor[N, 4]  (xyxy format, absolute pixel coords)
          "labels":   Int64Tensor[N]     (1-based class index)
          "image_id": Int64Tensor[1]
        }
    """
    boxes, labels = [], []
    objs = annotation["annotation"].get("object", [])
    if isinstance(objs, dict):  # single object → wrap in list
        objs = [objs]

    for obj in objs:
        name = obj["name"]
        if name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[name]
        if label == 0:
            continue  # skip background annotations
        bb = obj["bndbox"]
        xmin = float(bb["xmin"])
        ymin = float(bb["ymin"])
        xmax = float(bb["xmax"])
        ymax = float(bb["ymax"])
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    if not boxes:
        boxes  = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes  = torch.tensor(boxes,  dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

    return {
        "boxes":    boxes,
        "labels":   labels,
        "image_id": torch.tensor([hash(image_id) % (2**31)], dtype=torch.int64),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOC Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VOCDetectionFL(VOCDetection):
    """VOCDetection wrapped for FL: returns (image_tensor, target_dict)."""

    def __getitem__(self, idx):
        img, annotation = super().__getitem__(idx)
        image_id = annotation["annotation"]["filename"]
        target = _parse_voc_target(annotation, image_id)
        return img, target


def _dominant_class(dataset: VOCDetectionFL, idx: int) -> int:
    """Return the most frequent class label in image idx (used for partitioning)."""
    _, target = dataset[idx]
    labels = target["labels"]
    if len(labels) == 0:
        return 1  # default to aeroplane if empty
    return int(labels.bincount().argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning
# ─────────────────────────────────────────────────────────────────────────────

def _dirichlet_partition(
    dataset: VOCDetectionFL,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """
    Partition VOC images across clients using Dirichlet(alpha) on dominant class.

    Each image is assigned to a client with probability proportional to the
    Dirichlet draw for that image's dominant class.

    Returns: list of index lists (one per client).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)

    # Compute dominant class for each image (expensive but done once)
    dominant = []
    for i in range(n):
        _, tgt = dataset[i]
        labs = tgt["labels"]
        dominant.append(int(labs.bincount().argmax()) if len(labs) > 0 else 1)

    dominant = np.array(dominant)
    num_classes_obj = NUM_CLASSES - 1  # exclude background (classes 1-20)

    # Group indices by dominant class
    class_indices = {
        c: np.where(dominant == c)[0].tolist()
        for c in range(1, NUM_CLASSES)
    }

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls_idx, img_indices in class_indices.items():
        if not img_indices:
            continue
        rng.shuffle(img_indices)
        # Dirichlet draw for this class
        proportions = rng.dirichlet([alpha] * num_clients)
        # Assign images to clients proportionally
        splits = (proportions * len(img_indices)).astype(int)
        # Fix rounding errors: assign residual to the client with the most
        residual = len(img_indices) - splits.sum()
        splits[splits.argmax()] += residual

        ptr = 0
        for cid, count in enumerate(splits):
            client_indices[cid].extend(img_indices[ptr: ptr + count])
            ptr += count

    return client_indices


def _iid_partition(
    dataset: VOCDetectionFL,
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    """Randomly split images uniformly across clients."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [s.tolist() for s in splits]


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (required for variable-size boxes)
# ─────────────────────────────────────────────────────────────────────────────

def voc_collate_fn(batch):
    """
    Collate for object detection:
      images  → list of Tensors  (NOT stacked — SSDLite expects list)
      targets → list of dicts
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache for partition indices (avoid recomputing Dirichlet every call)
_PARTITION_CACHE: dict = {}


def get_voc_dataloader(
    data_root: str = "./data",
    split: str = "train",
    client_id: Optional[int] = None,
    num_clients: int = 20,
    alpha: float = 0.5,
    partition: str = "dirichlet",
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 2,
) -> DataLoader:
    """
    Return a DataLoader for one FL client's partition of PASCAL VOC 2012.

    Args:
        data_root:   Path where VOC data will be downloaded (./data/VOCdevkit).
        split:       "train" or "val".
        client_id:   Client index [0, num_clients). None = entire dataset.
        num_clients: Total number of FL clients.
        alpha:       Dirichlet concentration (lower = more non-IID).
        partition:   "dirichlet" | "iid".
        batch_size:  Mini-batch size.
        seed:        Random seed (also controls partition).
        num_workers: DataLoader workers.
    """
    data_root = str(Path(data_root) / "voc")
    transform = _voc_transforms()

    dataset = VOCDetectionFL(
        root=data_root,
        year="2012",
        image_set="train" if split == "train" else "val",
        download=True,
        transform=transform,
    )

    if client_id is None:
        # Return full dataset (e.g. for global test eval)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=voc_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── Partition ────────────────────────────────────────────────────────────
    cache_key = (data_root, split, num_clients, alpha, partition, seed)
    if cache_key not in _PARTITION_CACHE:
        if partition == "dirichlet":
            _PARTITION_CACHE[cache_key] = _dirichlet_partition(
                dataset, num_clients, alpha, seed
            )
        else:
            _PARTITION_CACHE[cache_key] = _iid_partition(dataset, num_clients, seed)

    client_idx = _PARTITION_CACHE[cache_key][client_id]
    subset = Subset(dataset, client_idx)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=voc_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(len(subset) >= batch_size),
    )


def get_voc_test_loader(
    data_root: str = "./data",
    batch_size: int = 8,
    num_workers: int = 2,
) -> DataLoader:
    """Return the full VOC 2012 val set for global evaluation."""
    return get_voc_dataloader(
        data_root=data_root,
        split="val",
        client_id=None,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def voc_dataset_sizes(
    data_root: str,
    num_clients: int,
    alpha: float,
    partition: str,
    seed: int,
) -> list[int]:
    """Return dataset size for each client (used for weighted aggregation)."""
    data_root_voc = str(Path(data_root) / "voc")
    transform = _voc_transforms()
    dataset = VOCDetectionFL(
        root=data_root_voc, year="2012", image_set="train",
        download=False, transform=transform,
    )
    cache_key = (data_root_voc, "train", num_clients, alpha, partition, seed)
    if cache_key not in _PARTITION_CACHE:
        if partition == "dirichlet":
            _PARTITION_CACHE[cache_key] = _dirichlet_partition(
                dataset, num_clients, alpha, seed
            )
        else:
            _PARTITION_CACHE[cache_key] = _iid_partition(dataset, num_clients, seed)
    return [len(_PARTITION_CACHE[cache_key][cid]) for cid in range(num_clients)]
