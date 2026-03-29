"""
Model-agnostic training loop with early stopping.

Works with any :class:`~src.models.base.IC50Model` subclass. The trainer
never inspects the batch directly — it always delegates to
``model.unpack_batch(batch, device)`` which returns ``(inputs, labels)``.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import IC50Model
from .metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def resolve_device(device_override: str | None = None) -> torch.device:
    """Return the target device, auto-detecting CUDA if no override is given."""
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------


def train_epoch(
    model: IC50Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one full pass over *loader* and return the mean training loss."""
    model.train()
    total_loss: float = 0.0
    total_samples: int = 0

    for batch in tqdm(loader, desc="  train", leave=False, unit="batch"):
        inputs, labels = model.unpack_batch(batch, device)
        optimizer.zero_grad()

        preds = model(inputs)
        loss = loss_fn(preds, labels.view_as(preds))
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        del inputs, labels, preds, loss

    gc.collect()
    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: IC50Model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate *model* on *loader* and return regression metrics."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
            inputs, labels = model.unpack_batch(batch, device)
            preds = model(inputs)

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            del inputs, labels, preds

    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    del all_preds, all_labels
    gc.collect()
    return compute_metrics(preds_arr, labels_arr)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def train(
    model: IC50Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    model_save_path: Path,
) -> dict:
    """Train *model* with early stopping, saving the best checkpoint."""
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    best_val_rmse: float = float("inf")
    best_epoch: int = 0
    epochs_no_improve: int = 0

    train_losses: list[float] = []
    val_metrics_history: list[dict] = []

    for epoch in range(max_epochs):
        train_loss  = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)

        val_rmse = val_metrics["rmse"]
        improved = val_rmse < best_val_rmse

        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            checkpoint_marker = " *"
        else:
            epochs_no_improve += 1
            checkpoint_marker = ""

        logger.info(
            "Epoch %03d | train_loss=%.4f | val %s%s",
            epoch + 1,
            train_loss,
            format_metrics(val_metrics),
            checkpoint_marker,
        )

        if epochs_no_improve >= patience:
            logger.info("Early stopping after %d epochs without improvement.", patience)
            break

    logger.info(
        "Training complete. Best val RMSE=%.4f at epoch %d.",
        best_val_rmse,
        best_epoch + 1,
    )

    return {
        "train_losses": train_losses,
        "val_metrics": val_metrics_history,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------


def run_training(
    model: IC50Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    max_epochs: int,
    patience: int,
    model_save_path: Path,
    device: torch.device,
) -> dict:
    """End-to-end training: runs the loop, then evaluates the best checkpoint on test.

    Parameters
    ----------
    model:
        Initialised :class:`~src.models.base.IC50Model`.
    train_loader, val_loader, test_loader:
        Pre-built DataLoaders.
    optimizer:
        Configured optimiser (e.g. ``torch.optim.Adam``).
    loss_fn:
        Loss function (e.g. ``nn.MSELoss()``).
    max_epochs:
        Hard upper bound on training epochs.
    patience:
        Early-stopping patience (epochs without val-RMSE improvement).
    model_save_path:
        Path at which to save and later reload the best checkpoint.
    device:
        Target device.

    Returns
    -------
    dict
        Test-set metrics (``rmse``, ``r2``, ``mae``) plus a ``history`` sub-dict.
    """
    logger.info("Training on device: %s", device)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        model_save_path=model_save_path,
    )

    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test metrics: %s", format_metrics(test_metrics))

    return {**test_metrics, "history": history}
