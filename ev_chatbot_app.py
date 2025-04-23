import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Load cleaned CSV or JSON
df = pd.read_json("cleaned_ev_data.json")

# Set page config
st.set_page_config(page_title="EV Station Chatbot", layout="wide")

# Sidebar
st.sidebar.title("⚡ EV Station Insight ChatBot")
st.sidebar.markdown("Ask questions like:")
st.sidebar.markdown("- Stations in San Jose by vendor?")
st.sidebar.markdown("- Most reviewed station?")
st.sidebar.markdown("- Average rating per vendor?")
st.sidebar.markdown("- Show chart of vendor station count")

if not os.path.exists("cleaned_ev_data.json"):
    st.error("🚨 File 'cleaned_ev_data.json' is missing. Please upload it to the app folder.")
    st.stop()

# OpenAI Key from env or input
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Initialize SmartDataframe
if st.session_state["OPENAI_API_KEY"]:
    llm = OpenAI(api_token=st.session_state["OPENAI_API_KEY"], model="gpt-4o")
    sdf = SmartDataframe(df, config={"llm": llm})
else:
    st.warning("🔑 Please provide a valid OpenAI API key")
    st.stop()

# Chat UI
st.title("🔌 EV Charging Station Chatbot")
query = st.text_input("Ask something about the EV stations dataset:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    try:
        response = sdf.chat(query)

        # Save to history
        st.session_state.chat_history.append({"question": query, "response": response})

        # Display response
        if isinstance(response, str):
            st.write(response)
        elif hasattr(response, "figure"):
            st.pyplot(response.figure)
        elif isinstance(response, plt.Figure):
            st.pyplot(response)
        else:
            st.write(response)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Display chat history
if st.session_state.chat_history:
    with st.expander("🧠 Chat History"):
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['response']}")
