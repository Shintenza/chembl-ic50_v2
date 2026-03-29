"""GCN model definition and factory for pIC50 regression."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


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


def build_model(model_key: str, training_cfg: dict, graph_cfg: dict) -> nn.Module:
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
