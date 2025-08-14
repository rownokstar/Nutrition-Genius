# app.py - Nutrition Genius with Local Llama 3 Integration
# Developed by: DM Shahriar Hossain (https://github.com/rownokstar/)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
# from io import StringIO # Not used in current code
import time
import re
import random

# --- Import for Hugging Face Transformers ---
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("Transformers library not found. Install with `pip install transformers torch accelerate bitsandbytes`.")

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
if "model" not in st.session_state: # This is the SentenceTransformer model for embeddings
    st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
if "data_type" not in st.session_state:
    st.session_state.data_type = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "sentences" not in st.session_state:
    st.session_state.sentences = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}
# --- New Session State for Llama 3 Pipeline ---
if "llm_pipeline" not in st.session_state:
    st.session_state.llm_pipeline = None

# --- Auto Dataset Type Detection ---
def detect_dataset_type(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    if numeric_ratio > 0.5:
        return "unsupervised"
    else:
        return "supervised"

# --- Optimized Smart Processing with Progress Tracking ---
def smart_process_dataset(df, data_type):
    total_rows = len(df)

    if total_rows > 10000:
        sample_size = 2000
        df_process = df.sample(n=sample_size, random_state=42)
    elif total_rows > 1000:
        sample_size = min(1000, total_rows)
        df_process = df.sample(n=sample_size, random_state=42)
    else:
        df_process = df

    try:
        progress_bar = st.progress(0)
        status = st.empty()

        status.text("📝 Converting data to text format...")
        sentences = []
        df_process_rows = list(df_process.iterrows())

        for i, (idx, row) in enumerate(df_process_rows):
            sentence_parts = []
            for val in row.values:
                if pd.notna(val):
                    sentence_parts.append(str(val))
            sentences.append(' '.join(sentence_parts))
            if i % max(1, len(df_process_rows)//10) == 0:
                progress_bar.progress(int(20 * i / len(df_process_rows)))

        status.text("🧠 Generating AI embeddings...")
        batch_size = 50
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

        status.text("🔍 Creating search index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        progress_bar.progress(100)

        status.text("✅ Processing complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status.empty()

        return index, sentences, True, None

    except Exception as e:
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status' in locals():
            status.empty()
        return None, None, False, str(e)

# --- Column Mapping Helper (Specific to nutrition_table.csv structure) ---
def identify_nutrition_columns(df):
    """
    Identify key nutrition columns based on the known structure of nutrition_table.csv.
    Columns seem to be unnamed integers, so we map by position.
    Expected order (based on sample):
    0: ID, 1: fat_100g, 2: carbohydrates_100g, 3: sugars_100g, 4: proteins_100g,
    5: salt_100g, 6: sodium_100g, 7: energy_kcal, 8: fiber_100g, 9: g_sum_or_exceeded, 10: product
    """
    mapping = {}
    num_cols = len(df.columns)
    if num_cols >= 11: # Need at least 11 columns
        try:
            # Map by integer column index
            mapping['id'] = df.columns[0]
            mapping['fat_100g'] = df.columns[1]
            mapping['carbohydrates_100g'] = df.columns[2]
            mapping['sugars_100g'] = df.columns[3]
            mapping['proteins_100g'] = df.columns[4]
            mapping['salt_100g'] = df.columns[5]
            mapping['sodium_100g'] = df.columns[6]
            mapping['energy_kcal'] = df.columns[7] # Assuming column 7 is kcal
            # mapping['energy_kj'] = df.columns[8] # If column 8 were kj, but it's fiber in sample
            mapping['fiber_100g'] = df.columns[8]
            # mapping['g_sum_or_exceeded'] = df.columns[9] # Often not needed
            mapping['product'] = df.columns[-1] # Last column is the product name
            st.success("✅ Successfully identified nutrition data columns by position.")
        except IndexError as e:
            st.warning(f"⚠️ Could not map all columns by position: {e}")
            mapping = {}
    else:
        st.warning("⚠️ Dataset does not seem to have the expected number of columns for nutrition data.")
    return mapping

# --- RAG Retriever using FAISS ---
def retrieve_context(query, index, sentences, k=3):
    """Retrieve top k most similar sentences from the FAISS index."""
    if index is None or sentences is None:
        return []
    try:
        query_vec = st.session_state.model.encode([query])
        D, I = index.search(np.array(query_vec).astype('float32'), k)
        results = []
        for i in I[0]:
            if i != -1 and i < len(sentences):
                results.append(sentences[i])
        return results
    except Exception as e:
        st.warning(f"Search error: {e}")
        return []

# --- Load Llama 3 Pipeline (Run once) ---
@st.cache_resource # Cache this to avoid reloading the model on every rerun
def load_llm_pipeline(model_name="meta-llama/Meta-Llama-3-8B-Instruct", use_quantization=True):
    """Load the Llama 3 model and tokenizer pipeline."""
    if not TRANSFORMERS_AVAILABLE:
        return None

    try:
        # --- Model Configuration ---
        # You might need to adjust these based on your hardware (RAM/GPU VRAM)
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        st.info(f"Using compute dtype: {compute_dtype}")

        model_kwargs = {
            "torch_dtype": compute_dtype,
            "device_map": "auto", # Automatically place layers on available devices (CPU/GPU)
        }

        # Optional: Use 4-bit quantization to reduce memory footprint (requires bitsandbytes)
        if use_quantization:
            try:
                # from transformers import BitsAndBytesConfig # Already imported
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model_kwargs["quantization_config"] = quantization_config
                st.info("🔧 Using 4-bit quantization for Llama 3 model (requires bitsandbytes).")
            except Exception as e_quant:
                st.warning(f"bitsandbytes not available or failed to configure quantization: {e_quant}. Running without quantization (may require more memory).")
                # Continue without quantization if bnb is not available or fails

        # --- Load Model and Tokenizer ---
        # If using a gated model like meta-llama/Meta-Llama-3-8B-Instruct,
        # you'll need to provide your Hugging Face token.
        # You can pass it via `use_auth_token=True` if you've logged in via `huggingface-cli login`
        # Or pass the token string directly: `use_auth_token="your_hf_token_here"`
        st.info(f"📥 Loading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        # Ensure there's a pad token for batched inference
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            st.info(".pad_token set to eos_token for tokenizer.")

        st.info(f"📥 Loading model '{model_name}' (this will take a while)...")
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, **model_kwargs)

        # --- Create Pipeline ---
        # Using a pipeline simplifies text generation
        st.info("🔧 Creating text generation pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500, # Adjust as needed
            temperature=0.6,     # Adjust creativity vs. determinism
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False, # Only return the generated text, not the prompt
            # pad_token_id=tokenizer.eos_token_id # Explicitly set if needed
        )
        st.success(f"✅ Llama 3 model '{model_name}' loaded successfully!")
        return pipe

    except Exception as e:
        st.error(f"❌ Failed to load Llama 3 model '{model_name}': {e}")
        return None

# --- Enhanced Response Generator with Local Llama 3 ---
def generate_response_with_llama3(query, original_df, column_mapping, index, sentences):
    """
    Generate a response using a local Llama 3 model.
    """
    thinking_steps = []
    response_parts = []
    chart = None # Chart generation logic not integrated with Llama 3 yet

    if original_df is None or column_mapping is None or len(column_mapping) == 0:
        thinking_steps.append("⚠️ **Thinking:** The dataset needs to be processed correctly first.")
        response_parts.append("⚠️ The dataset needs to be processed correctly first. Please re-upload and wait for processing to complete.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    if not TRANSFORMERS_AVAILABLE:
        thinking_steps.append("❌ **Thinking:** Transformers library is not installed.")
        response_parts.append("❌ Transformers library is not installed. Please install it with `pip
