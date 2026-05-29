import sys
from pathlib import Path
from langchain.agents import create_agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_ollama import ChatOllama
from frontend.agent.tools import predict_ic50, draw_molecule
from frontend.agent.prompt import AGENT_PROMPT

import config

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
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)
    return create_agent(llm, [predict_ic50, draw_molecule])


st.title("Chemoinformatics AI Agent")

agent = get_agent()

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
                    shared_state = {}
                    response = agent.invoke(
                        {
                            "messages": [
                                ("system", AGENT_PROMPT),
                                ("human", user_input),
                            ]
                        },
                        config={
                            "configurable": {
                                "shared_state": shared_state,
                                "selected_model": selected_model,
                            }
                        },
                    )

                    output = response["messages"][-1].content
                    st.markdown(output)

                    msg_data = {"role": "assistant", "content": output}

                    if "pending_image" in shared_state:
                        st.image(shared_state["pending_image"])
                        msg_data["image"] = shared_state["pending_image"]

                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    st.error(f"Error: {e}")
