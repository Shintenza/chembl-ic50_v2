"""
Bond featuriser for molecular graph construction.

Each bond is described by a 6-dimensional binary vector.

Dimension breakdown
-------------------
Bond type    4   SINGLE, DOUBLE, TRIPLE, AROMATIC
Is conjugated 1  bool
Is in ring   1   bool
             ---
Total        6
"""

from __future__ import annotations

import numpy as np
from rdkit.Chem import rdchem

# ---------------------------------------------------------------------------
# Public constant — keeps config.py in sync
# ---------------------------------------------------------------------------

BOND_FEATURE_DIM: int = 6

# ---------------------------------------------------------------------------
# Allowed bond types
# ---------------------------------------------------------------------------

_BOND_TYPES: list = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]  # 4 entries, no "other" category needed for bonds


def bond_features(bond: rdchem.Bond) -> np.ndarray:
    """Compute the 6-dimensional feature vector for *bond*.

    Parameters
    ----------
    bond:
        An RDKit :class:`rdkit.Chem.rdchem.Bond` instance.

    Returns
    -------
    np.ndarray
        Shape ``(BOND_FEATURE_DIM,)`` float32 array.
    """
    # Bond type: 4 dims (one-hot, no "other" — only standard bond types appear)
    bond_type = bond.GetBondType()
    feat_type = [int(bond_type == bt) for bt in _BOND_TYPES]
    # → 4 dims

    # Is conjugated: 1 dim
    feat_conjugated = [int(bond.GetIsConjugated())]
    # → 1 dim

    # Is in ring: 1 dim
    feat_in_ring = [int(bond.IsInRing())]
    # → 1 dim

    features = feat_type + feat_conjugated + feat_in_ring  # 6 dims total

    assert len(features) == BOND_FEATURE_DIM, (
        f"Expected {BOND_FEATURE_DIM} bond features, got {len(features)}"
    )

    return np.array(features, dtype=np.float32)
