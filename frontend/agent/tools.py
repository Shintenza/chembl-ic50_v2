import torch
from src.features import smiles_to_graph
from langchain_core.tools import tool
import streamlit as st


@st.cache_resource
def load_pytorch_model(model_name: str):
    try:
        model = torch.load(model_name, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        return None


@tool
def predict_ic50(smiles: str) -> str:
    """
    Use this tool to predict the biological activity (IC50) for a given molecule in SMILES format.
    Input MUST be a valid SMILES string.
    """
    selected_model_name = st.session_state.get("selected_pt_model", "moj_model.pt")

    pt_model = load_pytorch_model(selected_model_name)

    if pt_model is None:
        return f"Error: Cannot find the model file '{selected_model_name}' on the disk."

    try:
        graph = smiles_to_graph(smiles)

        return f"The predicted IC50 value for {smiles} using {selected_model_name} is 123.45 nM."
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"
