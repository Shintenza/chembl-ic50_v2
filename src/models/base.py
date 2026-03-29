"""Abstract base class shared by all IC50 prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn


class IC50Model(nn.Module, ABC):
    """Base class for all pIC50 regression models.

    Subclasses must implement:
    - ``forward(inputs)`` — standard PyTorch forward pass.
    - ``unpack_batch(batch, device)`` — extract ``(model_input, labels)``
      from a raw DataLoader batch and move both to *device*.

    The training loop calls::

        inputs, labels = model.unpack_batch(batch, device)
        preds = model(inputs)

    so the trainer is fully model-agnostic.
    """

    @abstractmethod
    def unpack_batch(self, batch: Any, device: torch.device) -> tuple[Any, Tensor]:
        """Unpack a DataLoader batch into ``(model_input, labels)``.

        Parameters
        ----------
        batch:
            Raw batch as produced by the DataLoader (e.g. a PyG ``Batch``
            object for graph models, or a ``(X, y)`` tensor tuple for
            tabular models).
        device:
            Target device.  Implementations must move all tensors here.

        Returns
        -------
        tuple[Any, Tensor]
            ``(model_input, labels)`` where *model_input* is whatever
            ``forward()`` expects and *labels* has shape ``(N,)``.
        """
        ...
