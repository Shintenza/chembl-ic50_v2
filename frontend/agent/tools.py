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


def load_model(model_path: str):
    model = build_model() if is_gnn(model_path) else build_mlp()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@tool
def predict_ic50(smiles: str, config: RunnableConfig):
    """
    Predicts pIC50 and IC50 values for a molecule represented as a SMILES string.

    Use this tool whenever the user:
    - asks for IC50 prediction
    - asks for pIC50 prediction
    - asks about predicted activity of a molecule
    - provides a SMILES and asks how potent or active the molecule is

    Input:
    - smiles (str): valid SMILES representation of a molecule

    Returns:
    {
        "status": "success",
        "pic50": float,
        "ic50": float
    }

    where:
    - pic50 is the predicted pIC50 value
    - ic50 is the predicted IC50 value in nM

    If prediction fails:

    {
        "status": "error"
    }

    Important:
    - Always use this tool instead of estimating activity yourself.
    - Never invent IC50 values.
    - The input must be a SMILES string.

    Examples:

    User:
    Predict IC50 for CCO

    Action:
    predict_ic50("CCO")

    User:
    How active is this molecule? CCO

    Action:
    predict_ic50("CCO")
    """
    print("I WAS CALLED WITH SMILES: ", smiles)
    model_path = config.get("configurable", {}).get("selected_model")
    global_features_scaler = config.get("configurable", {}).get("global_features_scaler")
    print("SCALER: ", global_features_scaler)

    if not model_path or not global_features_scaler:
        return {
            "status": "error",
        }

    try:
        model = load_model(model_path)
        device = torch.device("cpu")

        if is_gnn(model_path):
            graph = smiles_to_graph_input(smiles, global_features_scaler)
            print("GRAPH: ", graph)

            with torch.no_grad():
                pred = model(Batch.from_data_list([graph]).to(device))  # ty:ignore[unresolved-attribute]
                print("PREDICTION: ", pred)
        else:
            fp = smiles_to_fingerprint(smiles)
            with torch.no_grad():
                pred = model(fp.unsqueeze(0).to(device))  # ty:ignore[unresolved-attribute]
        pic50 = pred.item()
        ic50_nm = 10 ** (9 - pic50)

        return {
            "status": "success",
            "pic50": pic50,
            "ic50" : ic50_nm
        }

    except Exception as e:
        print("ERROR: ", e)
        return {
            "status": "error",
        }


@tool
def draw_molecule(smiles: str, config: RunnableConfig):
    """
    Generates a 2D depiction of a molecule from its SMILES representation.

    Use this tool whenever the user:
    - wants to see a molecule
    - asks to draw a molecule
    - asks for a molecular structure image
    - asks to visualize a SMILES string
    - asks for the 2D structure of a compound
    - agrees to render a molecule after receiving an IC50 prediction

    Input:
    - smiles (str): valid SMILES representation of a molecule

    Returns on success:
    {
        "status": "success"
    }

    Returns on failure:
    {
        "status": "failure"
    }

    Important:
    - The input must be a valid SMILES string.
    - Use this tool only when the user explicitly requests visualization or agrees to view a structure.
    - Do not call this tool when the user only requests activity prediction, IC50 prediction, or other numerical properties.
    - If the user asks for both activity prediction and visualization, first predict activity, then render the molecule if requested.

    Examples:

    User:
    Draw this molecule: CCO

    Action:
    draw_molecule("CCO")

    User:
    Show me the structure of aspirin

    Action:
    Obtain the aspirin SMILES if available, then call:
    draw_molecule(smiles)

    User:
    Yes, show me the structure

    Action:
    Use the most recently referenced valid SMILES and call:
    draw_molecule(smiles)
    """
    mol = Chem.MolFromSmiles(smiles)

    try:
        img = Draw.MolToImage(mol, size=(400, 300))
        shared_state = config.get("configurable", {}).get("shared_state")

        if shared_state is not None:
            shared_state["pending_image"] = img

        return {
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "failure"
        }
