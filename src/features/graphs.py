import torch
from rdkit import Chem
from torch_geometric.data import Data

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]
ATOMS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Unknown"]
DEGREE = [0, 1, 2, 3, 4, "MoreThan4"]
HYBRIDIZATION = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
NUMBER_OF_HS = [0, 1, 2, 3, 4, "MoreThan4"]

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [
        int(boolean_value)
        for boolean_value in list(map(lambda s: x == s, permitted_list))
    ]
    return binary_encoding


def get_node_features(atom):
    atom_type = atom.GetSymbol()
    atom_type_enc = one_hot_encoding(
        atom_type, ATOMS
    )

    degree_enc = one_hot_encoding(atom.GetTotalDegree(), DEGREE)

    hybridization_enc = one_hot_encoding(
        str(atom.GetHybridization()), HYBRIDIZATION 
    )

    num_implicit_h_enc = one_hot_encoding(
        atom.GetTotalNumHs(), NUMBER_OF_HS
    )

    is_aromatic = [int(atom.GetIsAromatic())]
    is_in_ring = [int(atom.IsInRing())]
    formal_charge = [atom.GetFormalCharge()]

    return (
        atom_type_enc
        + degree_enc
        + hybridization_enc
        + num_implicit_h_enc
        + is_aromatic
        + is_in_ring
        + formal_charge
    )


def get_edge_features(bond):
    bond_type = str(bond.GetBondType())
    bond_type_enc = one_hot_encoding(
        bond_type, BOND_TYPES 
    )

    is_conjugated = [int(bond.GetIsConjugated())]
    is_in_ring = [int(bond.IsInRing())]

    return bond_type_enc + is_conjugated + is_in_ring


def smiles_to_graph(smiles: str, pic50: float, id: int):
    mol = Chem.MolFromSmiles(smiles)

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_node_features(atom))
    x = torch.tensor(node_features, dtype=torch.float)

    edges_list = []
    edge_features_list = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feats = get_edge_features(bond)

        edges_list.append((i, j))
        edge_features_list.append(bond_feats)

        edges_list.append((j, i))
        edge_features_list.append(bond_feats)

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

    y = torch.tensor([[pic50]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.activity_id = id

    return data