from rdkit import Chem
import torch
from torch_geometric.data import Data


def get_node_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalDegree(),
    ]


def smiles_to_graph(smiles: str, pic50: float, id: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_node_features(atom))
    x = torch.tensor(node_features, dtype=torch.float)

    edges_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_list.append((i, j))
        edges_list.append((j, i))

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()

    y = torch.tensor([[pic50]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.activity_id = id
    return data
