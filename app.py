# app.py - Nutrition Genius with Dataset Upload Support

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
from io import StringIO
import os

# --- Config ---
st.set_page_config(page_title="🍏 Nutrition Genius", page_icon="🍏", layout="wide")
st.title("🍏 Nutrition Genius")
st.markdown("> *Upload your own dataset & get AI-powered nutrition insights!*")

# --- Session State ---
if "df" not in st.session_state:
    st.session_state.df = None
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
if "data_type" not in st.session_state:
    st.session_state.data_type = "supervised"

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    data_type = st.radio("Dataset Type", ["supervised", "unsupervised"])
    st.session_state.data_type = data_type

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅ Dataset loaded successfully!")

            # Auto-process
            with st.spinner("Processing dataset..."):
                sentences = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
                embeddings = st.session_state.model.encode(sentences)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings.astype('float32'))
                st.session_state.index = index
                st.session_state.sentences = sentences
                st.success("✅ Dataset processed & indexed!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    # Show preview if exists
    if st.session_state.df is not None:
        st.subheader("📊 Dataset Preview")
        st.dataframe(st.session_state.df.head(5))

# --- RAG Functions ---
def retrieve(query, k=3):
    if st.session_state.index is None:
        return pd.DataFrame()
    query_vec = st.session_state.model.encode([query])
    D, I = st.session_state.index.search(np.array(query_vec).astype('float32'), k)
    results = []
    for i in I[0]:
        if i != -1 and i < len(st.session_state.df):
            results.append(st.session_state.df.iloc[i])
    return pd.DataFrame(results).drop_duplicates()

# --- Response Generator ---
def generate_response(query):
    if st.session_state.df is None:
        return "⚠️ Please upload a dataset first.", None

    results = retrieve(query, k=3)
    chart = None

    # Nutrition breakdown
    if "breakdown" in query.lower():
        if not results.empty:
            row = results.iloc[0]
            nutrients = ['Protein (g)', 'Fat (g)', 'Carbs (g)']
            if all(nut in row for nut in nutrients):
                fig = px.pie(
                    values=[row[nut] for nut in nutrients],
                    names=nutrients,
                    title=f"{row['Food']} - Nutrition"
                )
                chart = fig
                return f"📊 Nutrition breakdown for {row['Food']} (per 100g)", chart

    # Show retrieved data
    if not results.empty:
        return "🔍 Retrieved data:\n\n" + results.to_markdown(index=False), None
    else:
        return "❌ No matching data found in your dataset.", None

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg and msg["chart"]:
            st.plotly_chart(msg["chart"], use_container_width=True)

if prompt := st.chat_input("Ask anything about your dataset..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response, chart = generate_response(prompt)
        st.markdown(response)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "chart": chart
        })

# --- Tips ---
st.info("💡 Try: 'Show nutrition breakdown of chicken' or 'Find high protein foods'")
