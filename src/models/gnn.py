from src.models import IC50Model
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from typing import cast
from config import GRAPH

class GNN(IC50Model):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout_rate: float = 0.3,  # <-- DODANE: Parametr do sterowania siłą dropoutu
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

        # <-- DODANE: Inicjalizacja warstwy Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        # <-- DODANE: Dropout na zgrupowanym wektorze z całego grafu
        x_pool = self.dropout(x_pool)

        x_out = F.relu(self.lin1(x_pool))
        
        # <-- DODANE: Drugi Dropout przed ostateczną predykcją (mocno ogranicza zapamiętywanie)
        x_out = self.dropout(x_out)

        x_out = self.lin2(x_out)

        return x_out.squeeze(-1)

    def unpack_batch(
        self, batch: Any, device: torch.device
    ) -> tuple[Any, torch.Tensor]:
        b = batch.to(device)
        return b, b.y.squeeze(-1)


def build_model() -> IC50Model:
    node_in_dim = cast(int, GRAPH.get("NUM_ATOM_FEATURES"))
    edge_in_dim = cast(int, GRAPH.get("NUM_BOND_FEATURES"))

    return GNN(
        node_in_dim,
        edge_in_dim,
        hidden_dim=64,
        num_layers=3,
    )