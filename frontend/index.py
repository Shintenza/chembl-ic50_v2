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
    else:
        st.selectbox(
            "Choose model:",
            options=available_models,
            format_func=lambda p: Path(p).stem,
            key="selected_pt_model",
        )
        st.info(f"Active: **{Path(st.session_state.selected_pt_model).stem}**")

@st.cache_resource
def get_agent():
    llm = ChatOllama(model="llama3.2", temperature=0)
    return create_agent(llm, [predict_ic50, draw_molecule])

SYSTEM_PROMPT = (
    "You are a chemoinformatics assistant. You have two tools: predict_ic50 and draw_molecule. "
    "Use predict_ic50 when the user wants to know the IC50 or biological activity of a molecule. "
    "Use draw_molecule when the user wants to see the 2D structure of a molecule. "
    "predict_ic50 returns a pIC50 value. Present it as pIC50 and also convert to IC50 in nM using IC50 = 10^(9 - pIC50). "
    "After predicting IC50, ask if the user would like to see the 2D structure."
)

st.title("Chemoinformatics AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"])

if user_input := st.chat_input("Enter SMILES or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            selected_model = st.session_state.get("selected_pt_model")
            if not selected_model:
                st.error("Please select a model from the sidebar first.")
            else:
                try:
                    agent = get_agent()
                    response = agent.invoke(
                        {"messages": [("system", SYSTEM_PROMPT), ("human", user_input)]},
                        config={"configurable": {"model_path": selected_model}},
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
                    st.error(f"Error: {e}")
