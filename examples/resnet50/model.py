"""ResNet-50 image classifier from torchvision (needs ``pip install torchvision``).

``build`` returns the architecture with deterministic random weights: enough to exercise export,
engine building, benchmarking and serving, where accuracy does not matter. Numerical agreement
between precisions on random weights says little about a trained model, so for real use point the
config at your trained weights (``kind: checkpoint``); see ``configs/examples/resnet50.yaml``.
"""

from __future__ import annotations

import torch
from torch import nn


def build(seed: int = 0, num_classes: int = 1000) -> nn.Module:
    from torchvision.models import resnet50  # noqa: PLC0415 - optional dependency

    torch.manual_seed(seed)
    model: nn.Module = resnet50(weights=None, num_classes=num_classes)
    return model
