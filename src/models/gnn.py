from src.models import IC50Model
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool, BatchNorm, Sequential, GINEConv
from typing import cast
from config import GRAPH


class GCN(IC50Model):
    def __init__(
        self,
        node_in_dim: int,
        num_global_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout_rate: float = 0.15,
    ):
        super().__init__()

        self.node_encoder = Linear(node_in_dim, hidden_dim)

        gnn_layers = []

        for _ in range(num_layers):
            gnn_layers.extend([
                (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                BatchNorm(hidden_dim),
                ReLU(inplace=True),
                Dropout(p=dropout_rate),
            ])

        self.gnn_blocks = Sequential('x, edge_index', gnn_layers)

        self.mlp_head = torch.nn.Sequential(
            Linear(hidden_dim + num_global_features, hidden_dim // 2),
            ReLU(inplace=True),
            Dropout(p=dropout_rate),
            Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch, global_features = (
            data.x,
            data.edge_index,
            data.batch,
            data.global_features,
        )

        x = self.node_encoder(x)
        x = self.gnn_blocks(x, edge_index)
        x = global_mean_pool(x, batch)
        x = torch.cat([x, global_features], dim=1)

        out = self.mlp_head(x)

        return out.squeeze(-1)

    def unpack_batch(
        self, batch: Any, device: torch.device
    ) -> tuple[Any, torch.Tensor]:
        b = batch.to(device)
        return b, b.y.squeeze(-1)

class GINE(IC50Model):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

        self.lin1 = nn.Linear(hidden_dim * 2 + global_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch, global_feat = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.global_features
        )

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        x_pool = torch.cat([x_mean, x_max, global_feat], dim=1)

        x_pool = self.dropout(x_pool)
        x_out = F.relu(self.lin1(x_pool))

        x_out = self.dropout(x_out)
        x_out = self.lin2(x_out)

        return x_out.squeeze(-1)

    def unpack_batch(
        self, batch: Any, device: torch.device
    ) -> tuple[Any, torch.Tensor]:
        b = batch.to(device)
        return b, b.y.squeeze(-1)

def build_gine_model() -> IC50Model:
    return GINE(34, 7, 3)

def build_gcn_model() -> IC50Model:
    return GCN(34, 3)

