from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .base import IC50Model


class MLPModel(IC50Model):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()

        layer_dims = [in_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layer_dims[-1], 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)  # shape: [N]

    def unpack_batch(self, batch: Any, device: torch.device) -> tuple[Tensor, Tensor]:
        x, y = batch
        return x.to(device), y.to(device)


def build_mlp(training_cfg: dict, fingerprint_cfg: dict) -> MLPModel:
    return MLPModel(
        in_dim=fingerprint_cfg["N_BITS"],
        hidden_dims=training_cfg["HIDDEN_DIMS"],
        dropout=training_cfg["DROPOUT"],
    )
