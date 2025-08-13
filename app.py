# app.py - Nutrition Genius with Auto Dataset Type Detection

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
st.markdown("> *Smart Dataset Auto-Detector & AI Nutrition Assistant*")

# --- Session State ---
if "df" not in st.session_state:
    st.session_state.df = None
if "index" not in st.session_state:
    st.session_state.index = None
if "model" not in st.session_state:
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
if "data_type" not in st.session_state:
    st.session_state.data_type = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# --- Auto Dataset Type Detection ---
def detect_dataset_type(df):
    """Automatically detect if dataset is supervised or unsupervised"""
    
    # Common supervised target columns
    target_columns = ['target', 'label', 'class', 'outcome', 'result', 'category']
    
    # Check for target columns (supervised indicators)
    has_target = any(col.lower() in target_columns for col in df.columns)
    
    # Check if most columns are numeric (unsupervised clustering data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns)
    
    # Heuristics for detection
    if has_target or 'category' in df.columns.str.lower().tolist():
        return "supervised"
    elif numeric_ratio > 0.6:  # Mostly numeric columns
        return "unsupervised"
    else:
        return "supervised"  # Default to supervised if uncertain

# --- Data Processing Based on Type ---
def process_dataset(df, data_type):
    """Process dataset based on its type"""
    
    with st.spinner(f"Processing {data_type} dataset..."):
        try:
            # Convert all data to text for embedding
            sentences = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
            embeddings = st.session_state.model.encode(sentences)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            return index, sentences, True, None
        except Exception as e:
            return None, None, False, str(e)

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Auto-detect dataset type
            detected_type = detect_dataset_type(df)
            st.session_state.data_type = detected_type
            
            st.success(f"✅ Dataset loaded! Detected as: **{detected_type.upper()}**")
            
            # Process dataset
            index, sentences, success, error = process_dataset(df, detected_type)
            
            if success:
                st.session_state.index = index
                st.session_state.sentences = sentences
                st.session_state.processed = True
                st.success("✅ Dataset processed & indexed successfully!")
            else:
                st.error(f"❌ Processing failed: {error}")
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    
    # Show dataset info
    if st.session_state.df is not None:
        st.subheader("📊 Dataset Info")
        st.write(f"**Type:** {st.session_state.data_type}")
        st.write(f"**Rows:** {len(st.session_state.df)}")
        st.write(f"**Columns:** {list(st.session_state.df.columns)}")
        st.dataframe(st.session_state.df.head(5))

# --- RAG Functions ---
def retrieve(query, k=3):
    if st.session_state.index is None or not st.session_state.processed:
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
    if st.session_state.df is None or not st.session_state.processed:
        return "⚠️ Please upload and process a dataset first.", None

    results = retrieve(query, k=3)
    chart = None

    # Nutrition breakdown (for food datasets)
    if "breakdown" in query.lower() or "composition" in query.lower():
        if not results.empty:
            row = results.iloc[0]
            # Try to find nutrition columns
            nutrition_cols = [col for col in ['Protein (g)', 'Fat (g)', 'Carbs (g)', 'Calories'] 
                            if col in row.index]
            if len(nutrition_cols) >= 2:
                values = [row[col] for col in nutrition_cols]
                fig = px.pie(
                    values=values,
                    names=nutrition_cols,
                    title=f"Nutrition Breakdown"
                )
                chart = fig
                return f"📊 Nutrition breakdown from your dataset", chart

    # Show retrieved data
    if not results.empty:
        return "🔍 Retrieved matching records:\n\n" + results.to_markdown(index=False), None
    else:
        return "❌ No matching records found in your dataset.", None

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
st.info("💡 Try: 'Show nutrition breakdown' or 'Find records with high values'")
st.info("💡 The system automatically detects if your dataset is supervised or unsupervised!")
