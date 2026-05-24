"""TinyConvNet: conv-bn-relu x2, global average pooling, linear head. About 5k parameters."""

from __future__ import annotations

import torch
from torch import nn


class TinyConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(2 * width, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = self.features(image).flatten(1)
        logits: torch.Tensor = self.head(pooled)
        return logits


def build(seed: int = 0, num_classes: int = 10, width: int = 16) -> nn.Module:
    """Deterministically initialized model (no weights file needed)."""
    torch.manual_seed(seed)
    model = TinyConvNet(num_classes=num_classes, width=width)
    # Non-trivial batch-norm statistics, so folding/validation exercises real arithmetic.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
    return model
