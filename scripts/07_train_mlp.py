#!/usr/bin/env python3
"""
07_train_mlp.py — Train an MLP baseline model for IC50 (pIC50) regression.

Usage
-----
    python scripts/07_train_mlp.py
    python scripts/07_train_mlp.py --split-map scaffold_split.parquet \\
                                   --run-name mlp_baseline_01

The model uses Morgan fingerprints (2048-bit) as input, produced by
06_build_fingerprints.py.  Hyperparameters come from ``config.MLP_TRAINING``.
The best checkpoint is saved to ``config.PATHS['MODELS_DIR'] / <run_name>.pt``.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import config
from src.models import build_mlp
from src.training.dataset import create_fp_dataloaders
from src.training.trainer import run_training, resolve_device
from src.training.metrics import format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    default_run_name = "mlp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Train an MLP baseline for pIC50 regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-map",
        type=str,
        default="scaffold_split.parquet",
        metavar="FILENAME",
        help="Filename of the split-map Parquet file inside SPLITS_DIR.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=default_run_name,
        metavar="NAME",
        help="Identifier for this run. Model saved as MODELS_DIR/<run-name>.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force a specific device (overrides auto-detection).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_map_path = config.PATHS["SPLITS_DIR"] / args.split_map
    if not split_map_path.exists():
        raise FileNotFoundError(
            f"Split map not found: {split_map_path}\nRun 03_split.py first."
        )

    split_fp_dir = config.PATHS["FINGERPRINTS_DIR"] / split_map_path.stem
    if not split_fp_dir.exists():
        raise FileNotFoundError(
            f"Fingerprint data not found: {split_fp_dir}\n"
            "Run 06_build_fingerprints.py first."
        )

    model_save_path = config.PATHS["MODELS_DIR"] / f"{args.run_name}.pt"
    device = resolve_device(args.device)

    logger.info("=== MLP Training ===")
    logger.info("Split map   : %s", split_map_path)
    logger.info("FP dir      : %s", split_fp_dir)
    logger.info("Run name    : %s", args.run_name)
    logger.info("Model save  : %s", model_save_path)
    logger.info("Device      : %s", device)
    logger.info("Hidden dims : %s", config.MLP_TRAINING["HIDDEN_DIMS"])
    logger.info("Dropout     : %s", config.MLP_TRAINING["DROPOUT"])

    model = build_mlp(
        training_cfg=config.MLP_TRAINING,
        fingerprint_cfg=config.FINGERPRINT,
    )
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_loader, val_loader, test_loader = create_fp_dataloaders(
        split_fp_dir=split_fp_dir,
        batch_size=config.MLP_TRAINING["BATCH_SIZE"],
        eval_batch_size=config.MLP_TRAINING["EVAL_BATCH_SIZE"],
        num_workers=config.MLP_TRAINING.get("NUM_WORKERS", 0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.MLP_TRAINING["LEARNING_RATE"],
        weight_decay=config.MLP_TRAINING["WEIGHT_DECAY"],
    )
    loss_fn = nn.MSELoss()

    results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=config.MLP_TRAINING["MAX_EPOCHS"],
        patience=config.MLP_TRAINING["PATIENCE"],
        model_save_path=model_save_path,
        device=device,
    )

    history = results.pop("history")

    print()
    print("=" * 55)
    print(f"Training Complete — Run: {args.run_name}")
    print("=" * 55)
    print(f"  Best epoch (0-based): {history['best_epoch']}")
    print(f"  Best val RMSE       : {history['best_val_rmse']:.4f}")
    print()
    print("  Test-set metrics:")
    print(f"    {format_metrics(results)}")
    print()
    print(f"  Model checkpoint    : {model_save_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
