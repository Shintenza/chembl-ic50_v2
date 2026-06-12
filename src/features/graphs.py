from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
from rdkit.Chem import Lipinski

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
ATOMS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
DEGREE = [0, 1, 2, 3, 4]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUMBER_OF_HS = [0, 1, 2, 3, 4]
CHIRAL_TAG = [0, 1, 2]


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def get_node_features(atom):
    atom_type = atom.GetSymbol()
    atom_type_enc = one_hot_encoding(atom_type, ATOMS)

    degree_enc = one_hot_encoding(atom.GetTotalDegree(), DEGREE)

    hybridization_enc = one_hot_encoding(str(atom.GetHybridization()), HYBRIDIZATION)

    num_implicit_h_enc = one_hot_encoding(atom.GetTotalNumHs(), NUMBER_OF_HS)

    is_aromatic = [int(atom.GetIsAromatic())]
    is_in_ring = [int(atom.IsInRing())]
    chiral_tag = one_hot_encoding(atom.GetChiralTag(), choices=CHIRAL_TAG)

    return (
        atom_type_enc
        + degree_enc
        + hybridization_enc
        + num_implicit_h_enc
        + is_aromatic
        + is_in_ring
        + chiral_tag
    )


# TODO remove these as switched from GINE to GCN
def get_edge_features(bond):
    bond_type = str(bond.GetBondType())
    bond_type_enc = one_hot_encoding(bond_type, BOND_TYPES)

    is_conjugated = [int(bond.GetIsConjugated())]
    is_in_ring = [int(bond.IsInRing())]

    return bond_type_enc + is_conjugated + is_in_ring


def smiles_to_graph_input(smiles: str, global_features_scaler: StandardScaler) -> Data:
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

    if edges_list:
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_edge_features(mol.GetDummyAtoms()[0].GetBonds()[0])) if mol.GetNumBonds() > 0 else 0), dtype=torch.float)

    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    raw_global_features = np.array([[molwt, logp, tpsa]])
    scaled_global_features = global_features_scaler.transform(raw_global_features)

    global_features = torch.tensor(scaled_global_features, dtype=torch.float)
    
    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        global_features=global_features,
    )



def smiles_to_graph(smiles: str, pic50: float, id: int) -> Data:
    data = smiles_to_graph_input(smiles)

    data.y = torch.tensor([[pic50]], dtype=torch.float)
    data.activity_id = id

    return data
