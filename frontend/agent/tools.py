import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Draw
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import streamlit as st

from src.features.graphs import smiles_to_graph_input
from src.features.fingerprints import smiles_to_fingerprint
from src.models import build_model, build_mlp


def is_gnn(model_path: str) -> bool:
    name = Path(model_path).stem.lower()
    return "gnn" in name


@st.cache_resource
def load_model(model_path: str):
    model = build_model() if is_gnn(model_path) else build_mlp()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@tool
def predict_ic50(smiles: str, config: RunnableConfig) -> str:
    """
    CALL THIS TOOL IMMEDIATELY IF THE USER WANTS TO PREDICT IC50 OR ACTIVITY.
    INPUT MUST BE A SMILES STRING.
    """
    print("I WAS CALLED WITH SMILES: ", smiles)
    model_path = config.get("configurable", {}).get("selected_model")

    if not model_path:
        return "Error: No model selected. Tell the user to select a model from the sidebar."

    try:
        model = load_model(model_path)
        device = torch.device("cpu")

        if is_gnn(model_path):
            graph = smiles_to_graph_input(smiles)
            with torch.no_grad():
                pred = model(Batch.from_data_list([graph]).to(device))
        else:
            fp = smiles_to_fingerprint(smiles)
            with torch.no_grad():
                pred = model(fp.unsqueeze(0).to(device))
        pic50 = pred.item()
        ic50_nm = 10 ** (9 - pic50)
        return f"Predicted pIC50 is {pic50:.2f}, which equals {ic50_nm:.2f} nM."

    except Exception as e:
        return f"Error during model inference: {str(e)}"


@tool
def draw_molecule(smiles: str, config: RunnableConfig) -> str:
    """
    CALL THIS TOOL IMMEDIATELY IF THE USER WANTS TO SEE, DRAW, OR VISUALIZE THE 2D STRUCTURE OF A MOLECULE.
    Input MUST be a valid SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)

    try:
        img = Draw.MolToImage(mol, size=(400, 300))
        shared_state = config.get("configurable", {}).get("shared_state")

        if shared_state is not None:
            shared_state["pending_image"] = img

        return "Molecule successfully rendered and sent to the UI. Tell the user: 'Here is the 2D structure of the molecule.'"

    except Exception as e:
        return f"Error during drawing molecule: {str(e)}"
