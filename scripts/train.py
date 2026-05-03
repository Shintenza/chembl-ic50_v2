#!/usr/bin/env python3
"""
05_train.py — Train a GNN or MLP model for IC50 (pIC50) regression.

Usage
-----
    python scripts/05_train.py --model gcn
    python scripts/05_train.py --model mlp
    python scripts/05_train.py --model gcn --loss mae --split random
    python scripts/05_train.py --model mlp --loss mse --split scaffold --run-name mlp_scaffold_01
"""

import argparse
import logging
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*The usage of `scatter.*")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn

import config
from src.enums import LossFunction, ModelType, SplitStrategy
from src.models import build_model, build_mlp
from src.training.dataset import create_graph_dataloaders, create_fp_dataloaders
from src.training.trainer import run_training, resolve_device
from src.training.metrics import format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_LOSS_FNS = {
    LossFunction.MSE: nn.MSELoss,
    LossFunction.MAE: nn.L1Loss,
    LossFunction.HUBER: nn.HuberLoss,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GNN or MLP model for pIC50 regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=ModelType,
        required=True,
        choices=list(ModelType),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--loss",
        type=LossFunction,
        default=LossFunction.MSE,
        choices=list(LossFunction),
        help="Loss function.",
    )
    parser.add_argument(
        "--split",
        type=SplitStrategy,
        default=SplitStrategy.SCAFFOLD,
        choices=list(SplitStrategy),
        help="Splitting strategy to use for training.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Identifier for this run. Model saved as MODELS_DIR/<run-name>.pt. "
            "Defaults to <model>_<split>_<loss>."
        ),
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

    split_map_path = config.PATHS["SPLITS_DIR"] / f"{args.split}_split.parquet"
    if not split_map_path.exists():
        raise FileNotFoundError(
            f"Split map not found: {split_map_path}\n"
            f"Run: 03_split.py --split {args.split}"
        )

    split_map_df = pd.read_parquet(split_map_path, engine="pyarrow")
    split_map: dict[int, str] = dict(
        zip(split_map_df["activity_id"].tolist(), split_map_df["split"].tolist())
    )

    run_name = args.run_name or f"{args.model}_{args.split}_{args.loss}"
    model_save_path = config.PATHS["MODELS_DIR"] / f"{run_name}.pt"
    device = resolve_device(args.device)
    loss_fn = _LOSS_FNS[args.loss]()

    logger.info(
        "=== Training [model=%s  loss=%s  split=%s] ===",
        args.model,
        args.loss,
        args.split,
    )
    logger.info("Split map   : %s", split_map_path)
    logger.info("Run name    : %s", run_name)
    logger.info("Model save  : %s", model_save_path)
    logger.info("Device      : %s", device)

    if args.model == ModelType.GNN:
        chunks_dir = config.PATHS["GRAPHS_DIR"]
        if not any(chunks_dir.glob("chunk_*.pt")):
            raise FileNotFoundError(
                f"No graph chunks found in {chunks_dir}\n"
                "Run 04_build_features.py --features graphs first."
            )
        model = build_model()
        train_cfg = config.TRAINING
        train_loader, val_loader, test_loader = create_graph_dataloaders(
            chunks_dir=chunks_dir,
            split_map=split_map,
            batch_size=train_cfg["BATCH_SIZE"],
            eval_batch_size=train_cfg["EVAL_BATCH_SIZE"],
            num_workers=train_cfg.get("NUM_WORKERS", 0),
        )
    else:
        chunks_dir = config.PATHS["FINGERPRINTS_DIR"]
        if not any(chunks_dir.glob("chunk_*.pt")):
            raise FileNotFoundError(
                f"No fingerprint chunks found in {chunks_dir}\n"
                "Run 04_build_features.py --features fingerprints first."
            )
        model = build_mlp()
        train_cfg = config.MLP_TRAINING
        train_loader, val_loader, test_loader = create_fp_dataloaders(
            chunks_dir=chunks_dir,
            split_map=split_map,
            batch_size=train_cfg["BATCH_SIZE"],
            eval_batch_size=train_cfg["EVAL_BATCH_SIZE"],
            num_workers=train_cfg.get("NUM_WORKERS", 0),
        )

    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["LEARNING_RATE"],
        weight_decay=train_cfg["WEIGHT_DECAY"],
    )

    results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        max_epochs=train_cfg["MAX_EPOCHS"],
        patience=train_cfg["PATIENCE"],
        model_save_path=model_save_path,
        device=device,
    )

    history = results.pop("history")

    print()
    print("=" * 55)
    print(f"Training Complete — Run: {run_name}")
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
