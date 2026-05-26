import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

import config
from frontend.agent.tools import predict_ic50, draw_molecule

st.set_page_config(page_title="IC50 Agent", page_icon="🧪")

with st.sidebar:
    st.header("Settings")

    models_dir = config.PATHS["MODELS_DIR"]
    model_files = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
    available_models = [str(p) for p in model_files]

    if not available_models:
        st.warning("No models found in data/models/")
        st.session_state.selected_pt_model = None
    else:
        selected = st.selectbox(
            "Choose model:",
            options=available_models,
            format_func=lambda p: Path(p).stem,
        )
        st.session_state.selected_pt_model = selected
        st.info(f"Active: **{Path(selected).stem}**")

LLM_MODEL = "llama3.2"
llm = ChatOllama(model=LLM_MODEL, temperature=0)

tools = [predict_ic50, draw_molecule]

SYSTEM_PROMPT = (
    "You are an expert chemoinformatics AI assistant. "
    "You help users predict IC50 values and visualize molecules using provided tools. "
    "Rules: "
    "1. When a user provides a SMILES string and asks for IC50, use the predict_ic50 tool. "
    "2. After successfully predicting IC50, ALWAYS politely ask the user if they would like to see the 2D structure of the molecule. "
    "3. If the user asks to draw or see the molecule, use the draw_molecule tool."
)

agent = create_agent(llm, tools)

st.title("Chemoinformatics AI Agent")
st.markdown("Enter a SMILES string to predict its IC50 biological activity.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

if user_input := st.chat_input("Enter SMILES (e.g., CCO) or chat with the agent..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = agent.invoke(
                    {"messages": [("system", SYSTEM_PROMPT), ("human", user_input)]}
                )

                output = response["messages"][-1].content
                st.markdown(output)

                msg_data = {"role": "assistant", "content": output}

                pending = st.session_state.pop("pending_image", None)
                if pending is not None:
                    st.image(pending)
                    msg_data["image"] = pending

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Agent encountered an error: {str(e)}")
