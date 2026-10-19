from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from constants import CLASSES


class ExpressionModel(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        self.name = name
        if name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights, progress=False)
            self.embedding_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights, progress=False)
            self.embedding_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights, progress=False)
            self.embedding_dim = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        self.head = nn.Linear(self.embedding_dim, len(CLASSES))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(images))


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[ExpressionModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class ordering does not match the application")
    model = ExpressionModel(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["config"]
