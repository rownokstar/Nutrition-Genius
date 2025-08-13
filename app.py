# app.py - Nutrition Genius with Optimized Performance
# Developed by: DM Shahriar Hossain (https://github.com/rownokstar/)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
from io import StringIO
import time
import hashlib

# --- Config ---
st.set_page_config(page_title="🍏 Nutrition Genius", page_icon="🍏", layout="wide")
st.title("🍏 Nutrition Genius")
st.markdown("> *Smart Dataset Auto-Detector & AI Assistant*")
st.markdown("**Developed by: [DM Shahriar Hossain](https://linkedin.com/in/dm-shahriar-hossain/)**")

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
if "sentences" not in st.session_state:
    st.session_state.sentences = None

# --- Auto Dataset Type Detection ---
def detect_dataset_type(df):
    """Automatically detect if dataset is supervised or unsupervised"""
    
    # Common supervised target columns
    target_columns = ['target', 'label', 'class', 'outcome', 'result', 'category']
    
    # Check for target columns (supervised indicators)
    has_target = any(col.lower() in target_columns for col in df.columns)
    
    # Check if most columns are numeric (unsupervised clustering data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    
    # Heuristics for detection
    if has_target or ('category' in [col.lower() for col in df.columns]):
        return "supervised"
    elif numeric_ratio > 0.6 and len(numeric_cols) > 2:  # Mostly numeric columns
        return "unsupervised"
    else:
        return "supervised"  # Default to supervised if uncertain

# --- Optimized Smart Processing with Progress Tracking ---
def smart_process_dataset(df, data_type):
    """Smart processing with automatic optimization and progress tracking"""
    
    # Auto-optimize based on dataset size
    total_rows = len(df)
    
    if total_rows > 10000:
        # For very large datasets, sample intelligently
        sample_size = 2000
        df_process = df.sample(n=sample_size, random_state=42)
        st.warning(f"📊 Large dataset detected ({total_rows} rows). Processing {sample_size} representative samples for optimal performance.")
    elif total_rows > 1000:
        # Medium dataset
        sample_size = min(1000, total_rows)
        df_process = df.sample(n=sample_size, random_state=42)
        st.info(f"⚡ Processing {sample_size} rows for better performance.")
    else:
        # Small dataset - process all
        df_process = df
        st.info(f"🚀 Processing all {total_rows} rows.")
    
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status = st.empty()
        
        # Step 1: Convert to sentences
        status.text("📝 Converting data to text format...")
        sentences = []
        df_process_rows = list(df_process.iterrows())
        
        for i, (idx, row) in enumerate(df_process_rows):
            sentences.append(' '.join(str(val) for val in row.values))
            if i % max(1, len(df_process_rows)//10) == 0:
                progress_bar.progress(int(20 * i / len(df_process_rows)))
        
        # Step 2: Generate embeddings (batch processing)
        status.text("🧠 Generating AI embeddings...")
        batch_size = 50  # Smaller batches for better memory management
        embeddings_list = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = st.session_state.model.encode(
                batch, 
                show_progress_bar=False,
                batch_size=min(32, batch_size)
            )
            embeddings_list.extend(batch_embeddings)
            progress_bar.progress(20 + int(60 * i / len(sentences)))
        
        embeddings = np.array(embeddings_list)
        
        # Step 3: Create FAISS index
        status.text("🔍 Creating search index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        progress_bar.progress(100)
        
        status.text("✅ Processing complete!")
        time.sleep(1)
        progress_bar.empty()
        status.empty()
        
        return index, sentences, True, None
        
    except Exception as e:
        # Clean up UI elements on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status' in locals():
            status.empty()
        return None, None, False, str(e)

# --- RAG Functions ---
def retrieve(query, k=3):
    if st.session_state.index is None or not st.session_state.processed:
        return pd.DataFrame()
    
    try:
        query_vec = st.session_state.model.encode([query])
        D, I = st.session_state.index.search(np.array(query_vec).astype('float32'), k)
        
        results = []
        for i in I[0]:
            if i != -1 and st.session_state.sentences and i < len(st.session_state.sentences):
                # Try to reconstruct the row from stored sentences
                # This is a simplified approach - in production, store the actual dataframe rows
                results.append({"Matched Data": st.session_state.sentences[i][:200] + "..."})
        
        return pd.DataFrame(results).drop_duplicates()
    except Exception as e:
        st.warning(f"Search error: {str(e)}")
        return pd.DataFrame()

# --- Response Generator ---
def generate_response(query):
    if not st.session_state.processed:
        return "⚠️ Please upload and process a dataset first.", None

    results = retrieve(query, k=3)
    chart = None

    # Show retrieved data
    if not results.empty:
        response_text = "🔍 Retrieved matching records:\n\n"
        response_text += results.to_markdown(index=False) if len(results) > 0 else "No detailed results available."
        return response_text, None
    else:
        return "❌ No matching records found in your dataset.", None

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    st.markdown("Developed by: [DM Shahriar Hossain](https://github.com/rownokstar/)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file)
            
            # Auto-optimize for performance
            if len(df) > 5000:
                df_display = df.sample(10)  # Show sample
                df_process = df.sample(2000)  # Process sample
                st.warning(f"📊 Large dataset detected ({len(df)} rows). Processing 2000 samples for optimal performance.")
            else:
                df_display = df.head(10)
                df_process = df
            
            st.session_state.df = df_display  # Show smaller preview
            
            # Auto-detect dataset type
            detected_type = detect_dataset_type(df_process)
            st.session_state.data_type = detected_type
            
            st.success(f"✅ Dataset loaded! Detected as: **{detected_type.upper()}**")
            
            # Quick processing with timeout simulation
            with st.spinner("🚀 Processing dataset... (This should take < 30 seconds)"):
                index, sentences, success, error = smart_process_dataset(df_process, detected_type)
                
            if success:
                st.session_state.index = index
                st.session_state.sentences = sentences
                st.session_state.processed = True
                st.success("✅ Dataset processed & ready for queries!")
                st.balloons()  # Celebration!
            else:
                st.error(f"❌ Processing failed: {error}")
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    # Show dataset info
    if st.session_state.df is not None:
        st.subheader("📊 Dataset Info")
        st.write(f"**Type:** {st.session_state.data_type or 'Not processed'}")
        st.write(f"**Preview Rows:** {len(st.session_state.df)}")
        if st.session_state.df is not None:
            st.write(f"**Columns:** {list(st.session_state.df.columns)}")
            st.dataframe(st.session_state.df.head(5))

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg and msg["chart"]:
            st.plotly_chart(msg["chart"], use_container_width=True)

# Chat input
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
st.info("💡 Try: 'Show records with high values' or 'Find similar patterns'")
st.info("💡 The system automatically detects if your dataset is supervised or unsupervised!")
st.markdown("---")
st.markdown("👨‍💻 Developed by: [DM Shahriar Hossain](https://linkedin.com/in/dm-shahriar-hossain/) | [GitHub](https://github.com/rownokstar/)")
