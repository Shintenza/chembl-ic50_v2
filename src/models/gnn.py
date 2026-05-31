from src.models import IC50Model
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from typing import cast
from config import GRAPH


class GNN(IC50Model):
    def __init__(
        self,
        node_in_dim: int,
        num_global_features: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        self.linear1 = Linear(hidden_dim + num_global_features, hidden_dim // 2)
        self.linear2 = Linear(hidden_dim // 2, 1)
        self.dropout = Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, _edge_attr, batch, global_features = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.global_features,
        )

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = torch.cat([x, global_features], dim=1)

        x = self.dropout(x)
        x = F.relu(self.linear1(x))

        out = self.linear2(x)

        return out.squeeze(-1)

    def unpack_batch(
        self, batch: Any, device: torch.device
    ) -> tuple[Any, torch.Tensor]:
        b = batch.to(device)
        # print(b.x.size(1))
        # print(b.global_features.size(1))
        return b, b.y.squeeze(-1)


def build_model() -> IC50Model:
    node_in_dim = cast(int, GRAPH.get("NUM_ATOM_FEATURES"))
    num_global_features = cast(int, GRAPH.get("NUM_GLOBAL_FEATURES"))

    return GNN(node_in_dim=node_in_dim, num_global_features=num_global_features)
