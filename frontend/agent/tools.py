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
def predict_ic50(smiles: str) -> str:
    """Use this tool to predict the biological activity (IC50) for a molecule given its SMILES string."""
    selected_model = st.session_state.get("selected_pt_model")
    if not selected_model:
        return "No model selected. Please choose a model from the sidebar."

    model = _load_model(selected_model)
    device = torch.device("cpu")

    if _is_gnn(selected_model):
        graph = smiles_to_graph_input(smiles)
        if graph is None:
            return f"Invalid SMILES string: {smiles}"
        batch = Batch.from_data_list([graph]).to(device)
        with torch.no_grad():
            pred = model(batch)
    else:
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            return f"Invalid SMILES string: {smiles}"
        with torch.no_grad():
            pred = model(fp.unsqueeze(0).to(device))

    pic50 = pred.item()
    ic50_nm = 10 ** (9 - pic50)
    return f"Predicted pIC50: {pic50:.3f} (IC50 ≈ {ic50_nm:.2f} nM)"


@tool
def draw_molecule(smiles: str) -> str:
    """Use this tool to draw a 2D structure of a molecule from its SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"Invalid SMILES string: {smiles}"

    st.session_state["pending_image"] = Draw.MolToImage(mol, size=(400, 300))
    return f"2D structure of {smiles} has been drawn."
