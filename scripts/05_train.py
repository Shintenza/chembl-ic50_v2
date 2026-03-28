#!/usr/bin/env python3
"""
05_train.py — Train a GNN model for IC50 (pIC50) regression.

Usage
-----
    python scripts/05_train.py
    python scripts/05_train.py --split-map scaffold_split.parquet \\
                               --model gcn --run-name experiment_01

A GCN model is built using hyperparameters from ``config.TRAINING``.  The
best checkpoint is saved to ``config.PATHS['MODELS_DIR'] / <run_name>.pt``.
Test-set metrics are printed at the end.

Extending with new model types
-------------------------------
Add a branch to ``_build_model()`` and pass ``--model <your_key>``.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

import config
from src.training.trainer import run_training
from src.training.metrics import format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class GCNModel(nn.Module):
    """A simple GCN with global mean pooling and a 2-layer MLP readout.

    Architecture
    ------------
    - ``num_gcn_layers`` GCNConv layers with ReLU activations.
    - Global mean pooling to produce a fixed-size graph embedding.
    - Two-layer MLP: ``hidden_dim → head_dim → 1``.

    Parameters
    ----------
    in_channels:
        Number of input atom features (should equal ``config.GRAPH['NUM_ATOM_FEATURES']``).
    hidden_dim:
        Width of each GCN hidden layer.
    head_dim:
        Width of the intermediate MLP layer.
    num_layers:
        Number of GCN message-passing layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        head_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # MLP readout
        self.lin1 = nn.Linear(hidden_dim, head_dim)
        self.lin2 = nn.Linear(head_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Global mean pooling: [num_atoms, hidden_dim] → [num_graphs, hidden_dim]
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)  # shape: [num_graphs, 1]
        return x.squeeze(-1)  # shape: [num_graphs]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model(model_key: str, training_cfg: dict, graph_cfg: dict) -> nn.Module:
    """Instantiate and return the requested model architecture.

    Parameters
    ----------
    model_key:
        Identifier string (e.g. ``'gcn'``).
    training_cfg:
        ``config.TRAINING`` dict.
    graph_cfg:
        ``config.GRAPH`` dict.

    Returns
    -------
    nn.Module
        Initialised model (weights randomly initialised).
    """
    key = model_key.lower()
    if key == "gcn":
        return GCNModel(
            in_channels=graph_cfg["NUM_ATOM_FEATURES"],
            hidden_dim=training_cfg["HIDDEN_DIM"],
            head_dim=training_cfg["HEAD_DIM"],
            num_layers=training_cfg["NUM_GCN_LAYERS"],
        )
    else:
        raise ValueError(
            f"Unknown model key: '{model_key}'. Supported models: ['gcn']"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    default_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Train a GNN model for pIC50 regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-map",
        type=str,
        default="scaffold_split.parquet",
        metavar="FILENAME",
        help=(
            "Filename of the split-map Parquet file inside SPLITS_DIR. "
            "Generate it with 03_split.py."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=default_run_name,
        metavar="NAME",
        help=(
            "Identifier for this training run. "
            "Model is saved as MODELS_DIR/<run-name>.pt"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force a specific device (overrides auto-detection).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    splits_dir: Path = config.PATHS["SPLITS_DIR"]
    graphs_dir: Path = config.PATHS["GRAPHS_DIR"]
    models_dir: Path = config.PATHS["MODELS_DIR"]

    split_map_path = splits_dir / args.split_map
    if not split_map_path.exists():
        raise FileNotFoundError(
            f"Split map not found: {split_map_path}\n"
            "Run 03_split.py first."
        )

    model_save_path = models_dir / f"{args.run_name}.pt"

    logger.info("=== GNN Training ===")
    logger.info("Split map   : %s", split_map_path)
    logger.info("Model       : %s", args.model)
    logger.info("Run name    : %s", args.run_name)
    logger.info("Model save  : %s", model_save_path)
    logger.info("Device      : %s", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = _build_model(
        model_key=args.model,
        training_cfg=config.TRAINING,
        graph_cfg=config.GRAPH,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", n_params)

    results = run_training(
        model=model,
        split_map_path=split_map_path,
        graphs_dir=graphs_dir,
        training_config=config.TRAINING,
        model_save_path=model_save_path,
        device_override=args.device,
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
