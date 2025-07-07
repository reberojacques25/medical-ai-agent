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
st.markdown("This AI agent helps doctors interpret patients' lab results or reports.")

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

    if st.button("Interpret Results"):
        with st.spinner("Analyzing with AI..."):
            prompt = f"You are a medical expert. Interpret the following patient results and provide potential diagnoses, abnormalities, or recommendations:\n\n{result_text}\n\nAdditional context: {extra_context}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional medical diagnostic assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            st.markdown("### 🧠 AI Interpretation")
            st.write(response['choices'][0]['message']['content'])