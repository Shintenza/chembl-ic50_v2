import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Draw
from langchain_core.tools import tool
import streamlit as st

from src.features.graphs import smiles_to_graph_input
from src.features.fingerprints import smiles_to_fingerprint
from src.models import build_model, build_mlp


def _is_gnn(model_path: str) -> bool:
    name = Path(model_path).stem.lower()
    return "gnn" in name or "gine" in name or "graph" in name


@st.cache_resource
def _load_model(model_path: str):
    model = build_model() if _is_gnn(model_path) else build_mlp()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@tool
def predict_ic50(smiles: str) -> float:
    """Predict the pIC50 value for a molecule. Use when the user asks to predict, estimate, or calculate IC50 or biological activity. Returns a numerical pIC50 score."""
    model_path = st.session_state.get("selected_pt_model")
    if not model_path:
        raise RuntimeError("No model selected.")

    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    model = _load_model(model_path)
    device = torch.device("cpu")

    if _is_gnn(model_path):
        graph = smiles_to_graph_input(smiles)
        with torch.no_grad():
            pred = model(Batch.from_data_list([graph]).to(device))
    else:
        fp = smiles_to_fingerprint(smiles)
        with torch.no_grad():
            pred = model(fp.unsqueeze(0).to(device))

    return pred.item()


@tool
def draw_molecule(smiles: str) -> str:
    """Render a 2D structural image of a molecule. Use when the user asks to draw, show, visualize, or display the structure. Do NOT use for IC50 prediction."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    st.session_state["pending_image"] = Draw.MolToImage(mol, size=(400, 300))
    return "Molecule rendered."
