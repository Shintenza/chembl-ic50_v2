"""
Model-agnostic training loop with early stopping.

This module works with any ``nn.Module`` whose ``forward()`` accepts a PyG
:class:`~torch_geometric.data.Batch` and returns a ``(N,)`` or ``(N, 1)``
prediction tensor.  It is completely decoupled from the graph construction
and featurisation code.
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

from .dataset import create_dataloaders
from .metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-epoch training
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one full pass over *loader* and return the mean training loss.

    Parameters
    ----------
    model:
        PyTorch model in training mode.
    loader:
        Training DataLoader.
    optimizer:
        Optimiser instance (already configured with model parameters).
    loss_fn:
        Loss function, e.g. ``nn.MSELoss()``.
    device:
        Device to move batches to before the forward pass.

    Returns
    -------
    float
        Average loss over all batches in *loader*.
    """
    model.train()
    total_loss: float = 0.0
    total_samples: int = 0

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False, unit="batch")):
        batch = batch.to(device)
        optimizer.zero_grad()

        preds = model(batch)
        labels = batch.y.view_as(preds)

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        del batch, preds, labels, loss

    gc.collect()
    return total_loss / max(total_samples, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate *model* on *loader* and return regression metrics.

    Parameters
    ----------
    model:
        PyTorch model (will be set to eval mode internally).
    loader:
        Validation or test DataLoader.
    device:
        Device to move batches to.

    Returns
    -------
    dict
        Metrics dict with keys ``'rmse'``, ``'r2'``, ``'mae'`` as returned
        by :func:`~src.training.metrics.compute_metrics`.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="  eval ", leave=False, unit="batch")):
            batch = batch.to(device)
            preds = model(batch)
            labels = batch.y

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())
            del batch, preds, labels

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    del all_preds, all_labels
    gc.collect()
    return compute_metrics(preds_arr, labels_arr)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    model_save_path: Path,
) -> dict:
    """Train *model* with early stopping, saving the best checkpoint.

    Training stops when the validation RMSE has not improved for *patience*
    consecutive epochs.  The best model state (lowest val RMSE) is saved to
    *model_save_path*.

    Parameters
    ----------
    model:
        Untrained (or partially trained) ``nn.Module``.
    train_loader:
        Training DataLoader.
    val_loader:
        Validation DataLoader.
    optimizer:
        Configured optimiser.
    loss_fn:
        Loss function.
    device:
        Target device.
    max_epochs:
        Hard upper bound on the number of training epochs.
    patience:
        Number of epochs without val-RMSE improvement before stopping.
    model_save_path:
        Path at which to save the best model weights (``torch.save``).

    Returns
    -------
    dict
        History dict with keys:

        - ``'train_losses'``: list of per-epoch average training losses.
        - ``'val_metrics'``: list of per-epoch val-metric dicts.
        - ``'best_epoch'``: epoch index (0-based) of the best checkpoint.
        - ``'best_val_rmse'``: validation RMSE at the best checkpoint.
    """
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    best_val_rmse: float = float("inf")
    best_epoch: int = 0
    epochs_no_improve: int = 0

    train_losses: list[float] = []
    val_metrics_history: list[dict] = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
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
            logger.info(
                "Early stopping after %d epochs without improvement.", patience
            )
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
    model: nn.Module,
    split_map_path: Path,
    graphs_dir: Path,
    training_config: dict,
    model_save_path: Path,
    device_override: str | None = None,
) -> dict:
    """End-to-end training convenience wrapper.

    Creates DataLoaders from the split map, configures the optimiser,
    runs the full training loop, then loads the best checkpoint and
    evaluates on the test set.

    Parameters
    ----------
    model:
        Initialised ``nn.Module`` ready for training.
    split_map_path:
        Path to the split-map Parquet file.
    graphs_dir:
        Directory containing graph chunk files and the chunk index.
    training_config:
        Dict with keys matching ``config.TRAINING`` (``BATCH_SIZE``,
        ``EVAL_BATCH_SIZE``, ``LEARNING_RATE``, ``WEIGHT_DECAY``,
        ``MAX_EPOCHS``, ``PATIENCE``).
    model_save_path:
        Where to save (and later load) the best model checkpoint.

    Returns
    -------
    dict
        Test-set metrics dict with keys ``'rmse'``, ``'r2'``, ``'mae'``,
        plus the training ``'history'`` sub-dict.
    """
    if device_override is not None:
        device = torch.device(device_override)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Training on device: %s", device)

    # Split-specific graphs live in graphs_dir/<split_name>/train|val|test/
    split_graphs_dir = Path(graphs_dir) / Path(split_map_path).stem

    if not split_graphs_dir.exists():
        raise FileNotFoundError(
            f"Split graphs not found: {split_graphs_dir}\n"
            "Run 04b_reorganize_split.py first."
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        split_graphs_dir=split_graphs_dir,
        batch_size=training_config["BATCH_SIZE"],
        eval_batch_size=training_config["EVAL_BATCH_SIZE"],
        num_workers=training_config.get("NUM_WORKERS", 0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["LEARNING_RATE"],
        weight_decay=training_config["WEIGHT_DECAY"],
    )
    loss_fn = nn.MSELoss()

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_epochs=training_config["MAX_EPOCHS"],
        patience=training_config["PATIENCE"],
        model_save_path=model_save_path,
    )

    # Load best checkpoint for test evaluation
    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)

    logger.info("Test metrics: %s", format_metrics(test_metrics))

    return {**test_metrics, "history": history}
