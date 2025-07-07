import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
import os

# Load OpenAI key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI Medical Assistant", page_icon="🧬", layout="wide")
st.title("🧬 AI Medical Assistant")
st.markdown("This AI agent helps doctors interpret lab results, predict diseases, and answer follow-up questions.")

# Session state to store conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload patient lab results
uploaded_file = st.file_uploader("Upload patient results (CSV or TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        result_text = df.to_string()
    else:
        result_text = uploaded_file.read().decode()

    st.success("File uploaded successfully!")

    extra_context = st.text_area("Any additional context (e.g., symptoms)?")

    if st.button("Interpret & Predict Disease"):
        with st.spinner("Consulting AI Doctor..."):
            prompt = f"""
You are a senior medical expert AI. The following are lab results for a patient:

{result_text}

Context provided by the doctor: {extra_context}

Please:
1. Interpret the lab results clearly.
2. Predict any likely diseases or medical conditions.
3. Explain any abnormalities.
4. Recommend further steps for diagnosis or treatment.
"""
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional AI medical diagnostic assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_reply = response['choices'][0]['message']['content']
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
            st.markdown("### 🧠 AI Interpretation & Prediction")
            st.write(ai_reply)

# Follow-up chat section
st.markdown("### 💬 Ask Follow-up Questions")
follow_up = st.text_input("Ask a follow-up question based on these results...")

if follow_up:
    st.session_state.chat_history.append({"role": "user", "content": follow_up})
    with st.spinner("AI is thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional AI medical assistant helping a doctor with patient diagnosis."},
                *st.session_state.chat_history
            ]
        )
        ai_reply = response['choices'][0]['message']['content']
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
        st.markdown("**AI Answer:**")
        st.write(ai_reply)

# Show chat history
with st.expander("📝 Conversation History"):
    for msg in st.session_state.chat_history:
        speaker = "👨‍⚕️ Doctor" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{speaker}:** {msg['content']}")
