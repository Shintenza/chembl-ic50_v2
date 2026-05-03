from sympy.physics.units import W
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


def resolve_device(device_override: str | None = None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: IC50Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
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


def evaluate(
    model: IC50Model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
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

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    del all_preds, all_labels
    gc.collect()
    return compute_metrics(preds_arr, labels_arr)


def train(
    model: IC50Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: ReduceLROnPlateau,
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
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)

        lr_scheduler.step(val_metrics["mse"])

        train_losses.append(train_loss)
        val_metrics_history.append(val_metrics)

        val_rmse = val_metrics["rmse"]
        improved = val_rmse < best_val_rmse

        current_lr = optimizer.param_groups[0]['lr']

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
            "Epoch %03d | train_loss=%.4f | val %s | LR: %.6f %s",
            epoch + 1,
            train_loss,
            format_metrics(val_metrics),
            current_lr,
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


def run_training(
    model: IC50Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: ReduceLROnPlateau,
    loss_fn: nn.Module,
    max_epochs: int,
    patience: int,
    model_save_path: Path,
    device: torch.device,
) -> dict:
    logger.info("Training on device: %s", device)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
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
