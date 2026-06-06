from src.models import IC50Model
from typing import Any, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import GINEConv, global_add_pool, BatchNorm
from config import GRAPH, TRAINING


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return Sequential(
        Linear(in_dim, out_dim),
        BatchNorm1d(out_dim),
        ReLU(),
        Linear(out_dim, out_dim),
    )


class GNN(IC50Model):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        num_global_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.node_encoder = Linear(node_in_dim, hidden_dim)
        self.edge_encoder = Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINEConv(nn=_make_mlp(hidden_dim, hidden_dim), edge_dim=hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        self.linear1 = Linear(hidden_dim + num_global_features, hidden_dim // 2)
        self.linear2 = Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr, batch, global_features = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.global_features,
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x = global_add_pool(x, batch)

        x = torch.cat([x, global_features], dim=1)

        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        out = self.linear2(x)

        return out.squeeze(-1)

    def unpack_batch(
        self, batch: Any, device: torch.device
    ) -> tuple[Any, torch.Tensor]:
        b = batch.to(device)
        return b, b.y.squeeze(-1)


def build_model() -> IC50Model:
    node_in_dim = cast(int, GRAPH.get("NUM_ATOM_FEATURES"))
    edge_in_dim = cast(int, GRAPH.get("NUM_BOND_FEATURES"))
    num_global_features = cast(int, GRAPH.get("NUM_GLOBAL_FEATURES"))
    hidden_dim = cast(int, TRAINING.get("HIDDEN_DIM"))
    num_layers = cast(int, TRAINING.get("NUM_GIN_LAYERS"))

    return GNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        num_global_features=num_global_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
