"""
models/registry.py
==================
Model zoo for FedLab.

Available:
  mlp          — Multi-Layer Perceptron (MNIST, FashionMNIST)
  lenet5       — LeNet-5 (MNIST, FashionMNIST, CIFAR-10)
  resnet8      — ResNet-8 (CIFAR-10, constrained devices, ~78K params)
  resnet18     — ResNet-18 (CIFAR-10/100, TinyImageNet)
  resnet50     — ResNet-50 (CIFAR-100, TinyImageNet)
  vit_tiny     — Vision Transformer Tiny (CIFAR-10/100)
  mobilenet_v3 — MobileNetV3-Small (mobile device simulation)

Usage:
  model = get_model("resnet8", "cifar10")
  model = get_model("resnet18", "cifar10")
  model = get_model("lenet5", "fashionmnist")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from datasets.registry import NUM_CLASSES, INPUT_SHAPE


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Simple MLP for MNIST / FashionMNIST.
    Architecture: 784 → 512 → 256 → num_classes
    """
    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 hidden: list[int] = None):
        super().__init__()
        hidden = hidden or [512, 256]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


# ─────────────────────────────────────────────────────────────────────────────
# LeNet-5
# ─────────────────────────────────────────────────────────────────────────────

class LeNet5(nn.Module):
    """
    LeNet-5 adapted for 28×28 (grayscale) and 32×32 (RGB) inputs.
    LeCun et al. (1998)

    The classifier input size is computed dynamically from img_size so the
    same class works correctly for both MNIST/FashionMNIST (28×28 → 400) and
    CIFAR-10 (32×32 → 576) without any hardcoded magic numbers.

    Feature map derivation for img_size H:
      Conv1 (k=5, p=2): H  → H          (same-padding keeps spatial size)
      AvgPool2d(2,2):    H  → H//2
      Conv2 (k=5, p=0):  H//2 → H//2 - 4   (no padding, shrinks by 4)
      AvgPool2d(2,2):    H//2 - 4 → (H//2 - 4)//2
      flat_dim = 16 * ((H//2 - 4)//2) ** 2
    Examples:
      H=28 → flat_dim = 16 * 5 * 5 = 400
      H=32 → flat_dim = 16 * 6 * 6 = 576
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10,
                 img_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
        )
        # Compute the flattened feature size analytically from img_size.
        # After Conv1+Pool: H//2; after Conv2+Pool: (H//2 - 4)//2
        after_pool2 = (img_size // 2 - 4) // 2
        flat_dim = 16 * after_pool2 * after_pool2
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-8 (lightweight, ~78K params, constrained-device FL)
# ─────────────────────────────────────────────────────────────────────────────

class _BasicBlock8(nn.Module):
    """
    Standard BasicBlock for ResNet-8.

    Two 3×3 conv layers with BN + ReLU.  A projection shortcut (1×1 conv + BN)
    is added automatically when in_channels != out_channels or stride != 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut: needed when spatial dims or channel count change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet8(nn.Module):
    """
    ResNet-8 for CIFAR-10 on constrained FL devices (~78K parameters).

    Architecture (He et al. 2016, CIFAR variant):
      Conv1:   3  → 16, 3×3, stride 1, padding 1, BN, ReLU
      Block 1: 16 → 16, stride 1  (identity shortcut)
      Block 2: 16 → 32, stride 2  (projection shortcut)
      Block 3: 32 → 64, stride 2  (projection shortcut)
      AvgPool: adaptive 1×1
      FC:      64 → num_classes

    Parameter count for CIFAR-10 (num_classes=10): ~78 K
    Comparison: LeNet-5 ~83 K, ResNet-18 ~11 M

    Suitable for RPi4, smartphones, and other edge devices where ResNet-18
    is too heavy but LeNet-5 accuracy on RGB images is insufficient.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Stem: single conv, no maxpool. in_channels=3 for CIFAR, 1 for
        # MNIST/EMNIST/FEMNIST. AdaptiveAvgPool handles any spatial size (28 or 32).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _BasicBlock8(16, 16, stride=1)   # 32×32 → 32×32
        self.layer2 = _BasicBlock8(16, 32, stride=2)   # 32×32 → 16×16
        self.layer3 = _BasicBlock8(32, 64, stride=2)   # 16×16 →  8× 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     #  8× 8 →  1× 1
        self.fc = nn.Linear(64, num_classes)

        # Weight initialisation (Kaiming, same as torchvision ResNets)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 adapted for small inputs (CIFAR-10/100)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32×32 CIFAR inputs.
    Changes vs standard: first conv 3×3 stride=1, no maxpool.
    He et al. (2016)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        # Adapt for 32×32 input
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class ResNet18TinyImageNet(nn.Module):
    """ResNet-18 for TinyImageNet 64×64 inputs."""
    def __init__(self, num_classes: int = 200):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


class ResNet50CIFAR(nn.Module):
    """ResNet-50 adapted for CIFAR-100 / TinyImageNet."""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        base = tv_models.resnet50(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(2048, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# MobileNetV3-Small (for mobile device simulation)
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small: low-latency model for smartphone simulation."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = tv_models.mobilenet_v3_small(weights=None)
        base.classifier[-1] = nn.Linear(1024, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# ViT-Tiny (lightweight Vision Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class ViTTiny(nn.Module):
    """
    Vision Transformer Tiny — 192 dim, 3 heads, 12 layers.
    Inspired by DeiT-Tiny (Touvron et al., 2021).
    Suitable for CIFAR-10/100.
    """
    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 192, depth: int = 12,
                 num_heads: int = 3, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "mlp":          MLP,
    "lenet5":       LeNet5,
    "resnet8":      ResNet8,
    "resnet18":     ResNet18CIFAR,
    "resnet50":     ResNet50CIFAR,
    "mobilenet_v3": MobileNetV3Small,
    "vit_tiny":     ViTTiny,
}

# Recommended (model, dataset) pairings
_DEFAULTS: dict[str, dict] = {
    "mnist":         {"model": "mlp",      "in_channels": 1},
    "fashionmnist":  {"model": "lenet5",   "in_channels": 1},
    "cifar10":       {"model": "resnet18", "in_channels": 3},
    "cifar100":      {"model": "resnet18", "in_channels": 3},
    "femnist":       {"model": "lenet5",   "in_channels": 1},
    "emnist":        {"model": "lenet5",   "in_channels": 1},
    "tiny_imagenet": {"model": "resnet18", "in_channels": 3},
}


def get_model(model_name: str, dataset_name: str) -> nn.Module:
    """
    Instantiate a model configured for the given dataset.

    Args:
        model_name:   "mlp", "lenet5", "resnet8", "resnet18", "resnet50",
                      "mobilenet_v3", "vit_tiny"
        dataset_name: "mnist", "cifar10", etc.

    Returns:
        Initialized nn.Module

    Example:
        model = get_model("resnet8",  "cifar10")    # → ResNet8(num_classes=10)
        model = get_model("resnet18", "cifar10")    # → ResNet18CIFAR(num_classes=10)
        model = get_model("lenet5", "fashionmnist") # → LeNet5(in_channels=1)
    """
    nc = NUM_CLASSES.get(dataset_name, 10)
    shape = INPUT_SHAPE.get(dataset_name, (3, 32, 32))
    in_c = shape[0]
    img_size = shape[1]

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    if model_name == "mlp":
        return MLP(input_dim=in_c * img_size * img_size, num_classes=nc)
    elif model_name == "lenet5":
        return LeNet5(in_channels=in_c, num_classes=nc, img_size=img_size)
    elif model_name == "resnet8":
        return ResNet8(num_classes=nc, in_channels=in_c)
    elif model_name == "resnet18":
        if img_size >= 64:
            return ResNet18TinyImageNet(num_classes=nc)
        return ResNet18CIFAR(num_classes=nc)
    elif model_name == "resnet50":
        return ResNet50CIFAR(num_classes=nc)
    elif model_name == "mobilenet_v3":
        return MobileNetV3Small(num_classes=nc)
    elif model_name == "vit_tiny":
        return ViTTiny(img_size=img_size, in_channels=in_c, num_classes=nc)

    raise ValueError(f"Unhandled model: {model_name}")


def get_default_model(dataset_name: str) -> nn.Module:
    """Return the recommended default model for a dataset."""
    default = _DEFAULTS.get(dataset_name, {"model": "resnet18"})
    return get_model(default["model"], dataset_name)


def list_models() -> list[dict]:
    """Return info about all registered models."""
    return [
        {"name": name, "class": cls.__name__, "description": cls.__doc__ or ""}
        for name, cls in _MODEL_REGISTRY.items()
    ]
