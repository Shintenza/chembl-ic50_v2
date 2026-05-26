import streamlit as st
import os
import glob
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="IC50 Agent", page_icon="🧪")

# --- Sidebar: Dropdown z wyborem modelu ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Automatycznie szuka wszystkich plików .pt w folderze z projektem
    available_models = glob.glob("*.pt")

    # Jeśli nie ma modeli w folderze, dajemy zaślepkę
    if not available_models:
        available_models = ["moj_model.pt"]

    # Tworzymy dropdown. Zapisujemy wynik do st.session_state.selected_pt_model
    st.session_state.selected_pt_model = st.selectbox(
        "Choose PyTorch Model:", options=available_models
    )

    st.info(f"Currently active model: **{st.session_state.selected_pt_model}**")

# --- Reszta konfiguracji Agenta (bez zmian) ---
LLM_MODEL = "llama3.2"
llm = ChatOllama(model=LLM_MODEL, temperature=0)

tools = [predict_ic50, draw_molecule]
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert chemoinformatics AI assistant. "
            "You help users predict IC50 values and visualize molecules using provided tools. "
            "Rules: "
            "1. When a user provides a SMILES string and asks for IC50, use the predict_ic50 tool. "
            "2. After successfully predicting IC50, ALWAYS politely ask the user if they would like to see the 2D structure of the molecule. "
            "3. If the user asks to draw or see the molecule, use the draw_molecule tool.",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Główny interfejs UI ---
st.title("🧪 Chemoinformatics AI Agent")
st.markdown("Enter a SMILES string to predict its IC50 biological activity.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg and os.path.exists(msg["image"]):
            st.image(msg["image"])

if user_input := st.chat_input("Enter SMILES (e.g., CCO) or chat with the agent..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                output = response["output"]
                st.markdown(output)

                msg_data = {"role": "assistant", "content": output}

                image_path = "molecule.png"
                if os.path.exists(image_path):
                    st.image(image_path)
                    import time

                    new_image_path = f"molecule_{int(time.time())}.png"
                    os.rename(image_path, new_image_path)
                    msg_data["image"] = new_image_path

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Agent encountered an error: {str(e)}")
