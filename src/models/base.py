from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn


class IC50Model(nn.Module, ABC):
    @abstractmethod
    def unpack_batch(self, batch: Any, device: torch.device) -> tuple[Any, Tensor]:
        pass
