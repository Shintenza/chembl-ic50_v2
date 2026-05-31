import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]
ATOMS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
DEGREE = [0, 1, 2, 3, 4]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUMBER_OF_HS = [0, 1, 2, 3, 4, "MoreThan4"]


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

    mass = [atom.GetMass() / 100.0]

    degree_enc = one_hot_encoding(atom.GetTotalDegree(), DEGREE)

    hybridization_enc = one_hot_encoding(str(atom.GetHybridization()), HYBRIDIZATION)

    num_implicit_h_enc = one_hot_encoding(atom.GetTotalNumHs(), NUMBER_OF_HS)

    is_aromatic = [int(atom.GetIsAromatic())]
    formal_charge = [atom.GetFormalCharge()]
    radical_electrons = [atom.GetNumRadicalElectrons()]
    is_in_ring = [int(atom.IsInRing())]

    return (
        atom_type_enc
        + mass
        + degree_enc
        + hybridization_enc
        + num_implicit_h_enc
        + is_aromatic
        + is_in_ring
        + formal_charge
        + radical_electrons
    )


# TODO remove these as switched from GINE to GCN
def get_edge_features(bond):
    bond_type = str(bond.GetBondType())
    bond_type_enc = one_hot_encoding(bond_type, BOND_TYPES)

    is_conjugated = [int(bond.GetIsConjugated())]
    is_in_ring = [int(bond.IsInRing())]

    return bond_type_enc + is_conjugated + is_in_ring


def smiles_to_graph_input(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

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

    mol_wt = Descriptors.MolWt(mol) / 100.0
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol) / 100.0
    rot_bonds = Descriptors.NumRotatableBonds(mol) / 10.0

    h_donors = Descriptors.NumHDonors(mol) / 10.0
    h_acceptors = Descriptors.NumHAcceptors(mol) / 10.0

    fraction_csp3 = Descriptors.FractionCSP3(mol)
    mol_mr = Descriptors.MolMR(mol) / 100.0

    aromatic_rings = Descriptors.NumAromaticRings(mol) / 5.0
    aliphatic_rings = Descriptors.NumAliphaticRings(mol) / 5.0
    bertz_ct = Descriptors.BertzCT(mol) / 1000.0

    global_features = torch.tensor(
        [
            [
                mol_wt,
                logp,
                tpsa,
                rot_bonds,
                h_donors,
                h_acceptors,
                fraction_csp3,
                mol_mr,
                aromatic_rings,
                aliphatic_rings,
                bertz_ct,
            ]
        ],
        dtype=torch.float,
    )

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features
    )


def smiles_to_graph(smiles: str, pic50: float, id: int) -> Data | None:
    data = smiles_to_graph_input(smiles)
    if data is None:
        return None

    data.y = torch.tensor([[pic50]], dtype=torch.float)
    data.activity_id = id

    return data
