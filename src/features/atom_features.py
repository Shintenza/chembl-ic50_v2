import numpy as np
from rdkit.Chem import rdchem

ATOM_FEATURE_DIM: int = 38

_ATOM_TYPES: list[str] = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "other",
]  # 13 entries (last entry is the explicit "other" bucket — add_other=False)

_DEGREES: list[int] = [0, 1, 2, 3, 4]  # 5 entries + "other" → 6
_FORMAL_CHARGES: list[int] = [-2, -1, 0, 1, 2]  # 5 entries + "other" → 6
_HYBRIDISATIONS: list = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]  # 5 entries + "other" → 6
_NUM_HS: list[int] = [0, 1, 2, 3]  # 4 entries + "other" → 5


def one_hot(val, allowed: list, add_other: bool = True) -> list[int]:
    """One-hot encode *val* against *allowed*.

    Parameters
    ----------
    val:
        The value to encode.
    allowed:
        Ordered list of recognised values.
    add_other:
        If ``True``, append an extra dimension that fires when *val* is not
        in *allowed*.  The resulting vector length is
        ``len(allowed) + 1``.

    Returns
    -------
    list[int]
        Binary list of length ``len(allowed) + int(add_other)``.
    """
    encoding = [int(val == v) for v in allowed]
    if add_other:
        encoding.append(int(val not in allowed))
    return encoding


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    """Compute the 38-dimensional feature vector for *atom*.

    Parameters
    ----------
    atom:
        An RDKit :class:`rdkit.Chem.rdchem.Atom` instance.

    Returns
    -------
    np.ndarray
        Shape ``(ATOM_FEATURE_DIM,)`` float32 array.
    """
    # Atom type: 13 dims (12 explicit + "other")
    # _ATOM_TYPES already includes "other" as last element, so add_other=False.
    feat_atom_type = one_hot(atom.GetSymbol(), _ATOM_TYPES[:-1], add_other=True)
    # → 13 dims

    # Degree: 6 dims (5 values + "other")
    feat_degree = one_hot(atom.GetDegree(), _DEGREES, add_other=True)
    # → 6 dims

    # Formal charge: 6 dims (5 values + "other")
    feat_charge = one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES, add_other=True)
    # → 6 dims

    # Hybridisation: 6 dims (5 values + "other")
    feat_hybrid = one_hot(atom.GetHybridization(), _HYBRIDISATIONS, add_other=True)
    # → 6 dims

    # Is aromatic: 1 dim
    feat_aromatic = [int(atom.GetIsAromatic())]
    # → 1 dim

    # Is in ring: 1 dim
    feat_in_ring = [int(atom.IsInRing())]
    # → 1 dim

    # Total Hs (implicit + explicit): 5 dims (4 values + "other")
    feat_hs = one_hot(atom.GetTotalNumHs(), _NUM_HS, add_other=True)
    # → 5 dims

    features = (
        feat_atom_type  # 13
        + feat_degree  #  6
        + feat_charge  #  6
        + feat_hybrid  #  6
        + feat_aromatic  #  1
        + feat_in_ring  #  1
        + feat_hs  #  5
    )  # = 38

    assert (
        len(features) == ATOM_FEATURE_DIM
    ), f"Expected {ATOM_FEATURE_DIM} features, got {len(features)}"

    return np.array(features, dtype=np.float32)
