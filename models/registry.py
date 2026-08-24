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
  alexnet      — AlexNet adapted to 28x28/32x32 federated benchmarks

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
# AlexNet for small-image Byzantine/fairness benchmarks
# ─────────────────────────────────────────────────────────────────────────────

class AlexNetCIFAR(nn.Module):
    """AlexNet-style CNN adapted to MNIST/FashionMNIST/CIFAR inputs.

    The original ImageNet network expects 224x224 images and a very large
    classifier.  FAR-style experiments use 28x28 or 32x32 images, so this
    version keeps the five convolutional stages, uses adaptive pooling, and a
    smaller classifier. ``norm='gn'`` is available for small, non-IID batches
    and DP experiments where BatchNorm running statistics are undesirable.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        norm: str = "none",
    ):
        super().__init__()

        def block(cin: int, cout: int, *, pool: bool = False):
            layers: list[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1)
            ]
            if norm == "gn":
                groups = 8
                while groups > 1 and cout % groups:
                    groups //= 2
                layers.append(nn.GroupNorm(groups, cout))
            layers.append(nn.ReLU(inplace=True))
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return layers

        self.features = nn.Sequential(
            *block(in_channels, 64, pool=True),
            *block(64, 192, pool=True),
            *block(192, 384),
            *block(384, 256),
            *block(256, 256, pool=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


class EMNISTCNNGN(nn.Module):
    """Compact GroupNorm CNN for 28x28 grayscale character recognition.

    The network deliberately contains no BatchNorm buffers, so every state
    entry can be averaged across non-IID clients without mixing client-local
    running statistics.  Adaptive pooling keeps the classifier independent of
    small changes to the input resolution.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        # Global pooling is supported by MPS for the 7x7 feature map, unlike
        # non-divisible adaptive pooling targets such as 3x3.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=5 ** 0.5, nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-8 (lightweight, ~78K params, constrained-device FL)
# ─────────────────────────────────────────────────────────────────────────────

def _make_norm(kind: str, c: int) -> nn.Module:
    """Normalization layer factory. kind='bn' → BatchNorm2d (running stats,
    aggregated across clients — fragile under severe non-IID); kind='gnN' →
    GroupNorm with at most N groups (NO running stats → nothing to aggregate
    → robust to class skew; the standard FL fix, Hsieh et al. 2020 'Non-IID
    Quagmire'). Bare 'gn' means gn8. The group count halves until it divides
    the channel count. γ/β stay per-channel, so gn4/gn8 checkpoints are
    state_dict-compatible."""
    if kind.startswith("gn"):
        g = int(kind[2:]) if kind[2:] else 8
        while g > 1 and c % g:
            g //= 2
        return nn.GroupNorm(g, c)
    return nn.BatchNorm2d(c)


class _BasicBlock8(nn.Module):
    """
    Standard BasicBlock for ResNet-8.

    Two 3×3 conv layers with norm + ReLU.  A projection shortcut (1×1 conv +
    norm) is added automatically when in_channels != out_channels or stride != 1.
    `norm` selects BatchNorm ('bn', default) or GroupNorm ('gn').
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = _make_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = _make_norm(norm, out_channels)

        # Projection shortcut: needed when spatial dims or channel count change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                _make_norm(norm, out_channels),
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

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 width_mult: float = 1.0, norm: str = "bn"):
        super().__init__()
        # width_mult < 1 slims every hidden layer (HeteroFL/ScaleFL-style
        # upper-left channel slicing): channels = ceil(base × width_mult).
        # Used by the ScaleFL baseline to build width-scaled submodels; the
        # global model always uses width_mult=1.0.
        # norm='bn' (default) | 'gn' (GroupNorm, no running stats — FL non-IID fix).
        import math as _math
        c1 = max(1, int(_math.ceil(16 * width_mult)))
        c2 = max(1, int(_math.ceil(32 * width_mult)))
        c3 = max(1, int(_math.ceil(64 * width_mult)))
        self._widths = (c1, c2, c3)
        self._norm = norm
        # Stem: single conv, no maxpool. in_channels=3 for CIFAR, 1 for
        # MNIST/EMNIST/FEMNIST. AdaptiveAvgPool handles any spatial size (28 or 32).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm(norm, c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _BasicBlock8(c1, c1, stride=1, norm=norm)   # 32×32 → 32×32
        self.layer2 = _BasicBlock8(c1, c2, stride=2, norm=norm)   # 32×32 → 16×16
        self.layer3 = _BasicBlock8(c2, c3, stride=2, norm=norm)   # 16×16 →  8× 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     #  8× 8 →  1× 1
        self.fc = nn.Linear(c3, num_classes)

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


class EarlyExitResNet8(ResNet8):
    """
    ResNet-8 with early-exit auxiliary heads (for FedStep exit_mode="early_exit").

    Exit points (depth k):
      k=1 : stem + layer1 → aux head (GAP + Linear 16 → num_classes)
      k=2 : + layer2      → aux head (GAP + Linear 32 → num_classes)
      k=3 : + layer3      → final head (existing fc, 64 → num_classes)

    A client whose battery affords only depth k runs forward AND backward
    through blocks 1..k only — the deeper blocks are never touched, so the
    round cost is genuinely proportional to the prefix FLOPs (this is the
    property that gives the battery a real grip on energy spend; see
    algorithms/fedpart_be.py docstring, "early_exit" mode).

    Aux heads add ~ (16+32) × num_classes + 2·num_classes params (< 1% of the
    backbone) and are aggregated per-depth like any other parameter.

    Related work: BranchyNet (early exits), DepthFL (ICLR 2023 — depth chosen
    by static device capacity). FedStep-EE differs by choosing the depth from
    the *residual battery* each round (dynamic β-rule).
    """

    # exit depth → parameter-name prefixes trained at that depth (cumulative)
    EXIT_PREFIXES: list[list[str]] = [
        ["stem", "layer1", "aux_heads.1"],
        ["layer2", "aux_heads.2"],
        ["layer3", "fc"],
    ]

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 width_mult: float = 1.0, norm: str = "bn"):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         width_mult=width_mult, norm=norm)
        c1, c2, _ = self._widths
        self.aux_heads = nn.ModuleDict(
            {
                "1": nn.Linear(c1, num_classes),
                "2": nn.Linear(c2, num_classes),
            }
        )

    @property
    def num_exits(self) -> int:
        return len(self.EXIT_PREFIXES)  # 3 (k ∈ {1, 2, 3})

    def _head(self, x: torch.Tensor, k: int) -> torch.Tensor:
        pooled = self.avgpool(x).flatten(1)
        return self.aux_heads[str(k)](pooled)

    def forward(self, x, exit_k: int | None = None, return_boundary: bool = False):
        """Standard forward (exit_k=None → final head, identical to ResNet8).

        exit_k=k truncates the computation graph after block k: deeper blocks
        are neither executed nor recorded for autograd.

        return_boundary=True additionally returns the GAP-pooled feature at
        the exit boundary — the interface the deeper blocks consume. Used by
        the FedStep-EE boundary-feature proximal anchor (repr_anchor=
        "boundary"); costs nothing (the pooled tensor feeds the head anyway).
        """
        x = self.stem(x)
        x = self.layer1(x)
        if exit_k == 1:
            pooled = self.avgpool(x).flatten(1)
            out = self.aux_heads["1"](pooled)
            return (out, pooled) if return_boundary else out
        x = self.layer2(x)
        if exit_k == 2:
            pooled = self.avgpool(x).flatten(1)
            out = self.aux_heads["2"](pooled)
            return (out, pooled) if return_boundary else out
        x = self.layer3(x)
        pooled = self.avgpool(x).flatten(1)
        out = self.fc(pooled)
        return (out, pooled) if return_boundary else out

    def forward_all_exits(self, x) -> dict[int, torch.Tensor]:
        """One full forward returning logits at every exit (warmup joint loss)."""
        out: dict[int, torch.Tensor] = {}
        x = self.stem(x)
        x = self.layer1(x)
        out[1] = self._head(x, 1)
        x = self.layer2(x)
        out[2] = self._head(x, 2)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        out[3] = self.fc(x)
        return out

    def forward_exits_upto(self, x, k: int, return_boundary: bool = False):
        """Truncated multi-exit forward: prefix up to depth k, logits at every
        exit ≤ k (deep→shallow self-distillation, ee_self_distill).

        Same energy contract as forward(exit_k=k) — blocks beyond k are
        neither executed nor recorded for autograd; the extra shallow heads
        are GAP+Linear on features already computed (negligible FLOPs).
        forward_exits_upto(x, num_exits) computes what forward_all_exits does.

        return_boundary=True additionally returns the GAP-pooled feature at
        the DEEPEST computed exit (the repr_anchor="boundary" interface).
        """
        out: dict[int, torch.Tensor] = {}
        x = self.stem(x)
        x = self.layer1(x)
        out[1] = self._head(x, 1)
        if k >= 2:
            x = self.layer2(x)
            out[2] = self._head(x, 2)
        if k >= 3:
            x = self.layer3(x)
            pooled = self.avgpool(x).flatten(1)
            out[3] = self.fc(pooled)
        if not return_boundary:
            return out
        if k < 3:
            pooled = self.avgpool(x).flatten(1)
        return out, pooled


class EarlyExitResNet8Ensemble(EarlyExitResNet8):
    """EarlyExitResNet8 whose default forward returns the ENSEMBLE prediction.

    DepthFL (ICLR 2023) evaluates the global model as the ensemble of all
    internal classifiers (mean softmax over exits). Registering this variant
    lets the standard runner evaluation measure exactly what the paper
    reports, with the same parameters/keys as resnet8_ee (state dicts are
    interchangeable). forward(x) → log of the mean softmax (argmax-equivalent
    to the ensemble vote); the truncated/exit paths of the parent remain
    available for training.
    """

    def forward(self, x, exit_k=None, return_boundary: bool = False):
        if exit_k is not None or return_boundary:
            return super().forward(x, exit_k=exit_k, return_boundary=return_boundary)
        outs = self.forward_all_exits(x)
        probs = torch.stack([torch.softmax(z, dim=1) for z in outs.values()])
        return torch.log(probs.mean(dim=0).clamp_min(1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 adapted for small inputs (CIFAR-10/100)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32×32 CIFAR inputs.
    Changes vs standard: first conv 3×3 stride=1, no maxpool.
    He et al. (2016)
    """
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        norm: str = "bn",
    ):
        super().__init__()
        norm_layer = lambda channels: _make_norm(norm, channels)
        base = tv_models.resnet18(weights=None, norm_layer=norm_layer)
        # Adapt for 32×32 input
        base.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
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


class EarlyExitViTTiny(ViTTiny):
    """
    ViT-Tiny with early-exit heads for FedStep-EE (exit_mode="early_exit").

    Exit points (depth k), 12 encoder blocks:
      k=1 : patch_embed + blocks 0-2   → aux head (LN + Linear on CLS)
      k=2 : + blocks 3-5               → aux head
      k=3 : + blocks 6-8               → aux head
      k=4 : + blocks 9-11              → final norm + head

    Transformers are the ideal early-exit substrate: all blocks cost the
    same, so the round cost is LINEAR in k (EXIT_FRACTIONS below) and the
    battery→depth β/deadline rule gets a nearly continuous dial. Exits after
    encoder blocks follow DeeBERT/PABEE/LayerSkip practice.

    EXIT_FRACTIONS overrides the group-FLOPs estimation (identical blocks →
    analytic fractions are exact up to the small patch-embed/head terms).
    """

    EXIT_BLOCKS = [3, 6, 9, 12]  # cumulative encoder blocks per depth

    # patch_embed + attention head ≈ 4% of a full forward; 12 equal blocks
    # carry the remaining 96% → fraction(k) = 0.04 + 0.96 · (blocks_k / 12).
    EXIT_FRACTIONS = [0.28, 0.52, 0.76, 1.0]

    EXIT_PREFIXES: list[list[str]] = [
        ["patch_embed", "cls_token", "pos_embed",
         "transformer.layers.0", "transformer.layers.1", "transformer.layers.2",
         "aux_heads.1"],
        ["transformer.layers.3", "transformer.layers.4", "transformer.layers.5",
         "aux_heads.2"],
        ["transformer.layers.6", "transformer.layers.7", "transformer.layers.8",
         "aux_heads.3"],
        ["transformer.layers.9", "transformer.layers.10", "transformer.layers.11",
         "norm", "head"],
    ]

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 10, **kw):
        super().__init__(img_size=img_size, patch_size=patch_size,
                         in_channels=in_channels, num_classes=num_classes, **kw)
        d = self.pos_embed.shape[-1]
        self.aux_heads = nn.ModuleDict(
            {
                str(k): nn.Sequential(nn.LayerNorm(d), nn.Linear(d, num_classes))
                for k in (1, 2, 3)
            }
        )

    @property
    def num_exits(self) -> int:
        return len(self.EXIT_PREFIXES)  # 4

    def _embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.pos_drop(x + self.pos_embed)

    def forward(self, x, exit_k: int | None = None, return_boundary: bool = False):
        """Truncated forward: only blocks 0..EXIT_BLOCKS[k-1] execute.

        Boundary feature = the CLS token after the exit's LayerNorm (192-d) —
        what the deeper blocks / final head would consume.
        """
        x = self._embed(x)
        n_blocks = (
            self.EXIT_BLOCKS[exit_k - 1] if exit_k else self.EXIT_BLOCKS[-1]
        )
        for i, blk in enumerate(self.transformer.layers):
            if i >= n_blocks:
                break
            x = blk(x)
        if exit_k and exit_k < self.num_exits:
            ln, lin = self.aux_heads[str(exit_k)]
            pooled = ln(x[:, 0])
            out = lin(pooled)
        else:
            pooled = self.norm(x[:, 0])
            out = self.head(pooled)
        return (out, pooled) if return_boundary else out

    def forward_all_exits(self, x) -> dict[int, torch.Tensor]:
        """One full forward returning logits at every exit (warmup joint loss)."""
        x = self._embed(x)
        out: dict[int, torch.Tensor] = {}
        prev = 0
        for k, n_blocks in enumerate(self.EXIT_BLOCKS, start=1):
            for i in range(prev, n_blocks):
                x = self.transformer.layers[i](x)
            prev = n_blocks
            if k < self.num_exits:
                ln, lin = self.aux_heads[str(k)]
                out[k] = lin(ln(x[:, 0]))
            else:
                out[k] = self.head(self.norm(x[:, 0]))
        return out

    def forward_exits_upto(self, x, k: int, return_boundary: bool = False):
        """Truncated multi-exit forward: blocks up to EXIT_BLOCKS[k-1], logits
        at every exit ≤ k (deep→shallow self-distillation, ee_self_distill).
        Same energy contract as forward(exit_k=k); extra shallow heads are
        LN+Linear on the CLS token (negligible). Boundary = post-LN CLS at
        the deepest computed exit.
        """
        x = self._embed(x)
        out: dict[int, torch.Tensor] = {}
        pooled = None
        prev = 0
        for d in range(1, k + 1):
            n_blocks = self.EXIT_BLOCKS[d - 1]
            for i in range(prev, n_blocks):
                x = self.transformer.layers[i](x)
            prev = n_blocks
            if d < self.num_exits:
                ln, lin = self.aux_heads[str(d)]
                pooled = ln(x[:, 0])
                out[d] = lin(pooled)
            else:
                pooled = self.norm(x[:, 0])
                out[d] = self.head(pooled)
        return (out, pooled) if return_boundary else out


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "mlp":          MLP,
    "lenet5":       LeNet5,
    "resnet8":      ResNet8,
    "resnet8_ee":   EarlyExitResNet8,   # FedStep early-exit variant
    "resnet8_ee_gn": EarlyExitResNet8,  # early-exit + GroupNorm (BN-free, non-IID fix)
    "resnet8_ee_gn4": EarlyExitResNet8,  # idem, 4 groups (wider groups for narrow nets)
    "resnet8_ee_ens": EarlyExitResNet8Ensemble,  # DepthFL ensemble inference
    "resnet18":     ResNet18CIFAR,
    "resnet18_gn4": ResNet18CIFAR,
    "resnet50":     ResNet50CIFAR,
    "mobilenet_v3": MobileNetV3Small,
    "vit_tiny":     ViTTiny,
    "vit_tiny_ee":  EarlyExitViTTiny,   # FedStep early-exit transformer
    "alexnet":      AlexNetCIFAR,
    "alexnet_gn":   AlexNetCIFAR,
    "cnn_gn":       EMNISTCNNGN,
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
        model_name:   "mlp", "lenet5", "alexnet", "resnet8", "resnet18",
                      "resnet50", "mobilenet_v3", "vit_tiny"
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
    elif model_name == "alexnet":
        return AlexNetCIFAR(in_channels=in_c, num_classes=nc)
    elif model_name == "alexnet_gn":
        return AlexNetCIFAR(in_channels=in_c, num_classes=nc, norm="gn")
    elif model_name == "cnn_gn":
        return EMNISTCNNGN(in_channels=in_c, num_classes=nc)
    elif model_name == "resnet8":
        return ResNet8(num_classes=nc, in_channels=in_c)
    elif model_name == "resnet8_ee":
        return EarlyExitResNet8(num_classes=nc, in_channels=in_c)
    elif model_name == "resnet8_ee_gn":
        return EarlyExitResNet8(num_classes=nc, in_channels=in_c, norm="gn")
    elif model_name == "resnet8_ee_gn4":
        return EarlyExitResNet8(num_classes=nc, in_channels=in_c, norm="gn4")
    elif model_name == "resnet8_ee_ens":
        return EarlyExitResNet8Ensemble(num_classes=nc, in_channels=in_c)
    elif model_name == "resnet18":
        if img_size >= 64:
            return ResNet18TinyImageNet(num_classes=nc)
        return ResNet18CIFAR(num_classes=nc, in_channels=in_c)
    elif model_name == "resnet18_gn4":
        if img_size >= 64:
            raise ValueError("resnet18_gn4 currently targets 28x28/32x32 inputs")
        return ResNet18CIFAR(
            num_classes=nc,
            in_channels=in_c,
            norm="gn4",
        )
    elif model_name == "resnet50":
        return ResNet50CIFAR(num_classes=nc)
    elif model_name == "mobilenet_v3":
        return MobileNetV3Small(num_classes=nc)
    elif model_name == "vit_tiny":
        return ViTTiny(img_size=img_size, in_channels=in_c, num_classes=nc)
    elif model_name == "vit_tiny_ee":
        return EarlyExitViTTiny(img_size=img_size, in_channels=in_c, num_classes=nc)

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
