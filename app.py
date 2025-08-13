# app.py - Nutrition Genius with Optimized Performance and Improved Responses
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
import re

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
if "original_df" not in st.session_state:
    st.session_state.original_df = None

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

# --- Response Generator with Natural Language and Formatting ---
def generate_response(query, original_df):
    """
    Generate a natural language response with potential table/chart.
    This is a rule-based approach for specific queries until full RAG is implemented.
    """
    chart = None
    response_parts = []
    
    # --- Rule-based Responses for Specific Queries ---
    # 1. List foods with saturated fat content above Xg
    # Note: The provided dataset doesn't seem to have a specific 'saturated fat' column.
    # We'll assume 'fat_100g' is total fat. Let's find foods with high total fat.
    saturated_fat_match = re.search(r"saturated fat.*above\s*(\d+\.?\d*)\s*g", query.lower())
    fat_match = re.search(r"(?:total\s*)?fat.*above\s*(\d+\.?\d*)\s*g", query.lower()) or saturated_fat_match
    
    if fat_match:
        try:
            threshold = float(fat_match.group(1))
            # Assume 'fat_100g' column exists
            if 'fat_100g' in original_df.columns:
                high_fat_foods = original_df[original_df['fat_100g'] > threshold]
                if not high_fat_foods.empty:
                    response_parts.append(f"Here are foods with fat content above {threshold}g per 100g:")
                    # Select relevant columns for display
                    cols_to_show = [col for col in ['product', 'fat_100g'] if col in original_df.columns]
                    if not cols_to_show:
                         cols_to_show = original_df.columns[:min(5, len(original_df.columns))] # Show first 5 if specific cols not found
                    # Limit results for display
                    high_fat_foods_display = high_fat_foods[cols_to_show].head(10) 
                    # Format as markdown table string
                    table_md = high_fat_foods_display.to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                     response_parts.append(f"I couldn't find any foods with fat content above {threshold}g per 100g in the dataset.")
            else:
                 response_parts.append("The dataset doesn't seem to have a 'fat_100g' column to filter by fat content.")
        except (ValueError, IndexError):
             response_parts.append("I couldn't understand the fat threshold value from your query.")
        return "\n".join(response_parts), chart

    # 2. Show me the top N foods highest in protein
    top_protein_match = re.search(r"top\s*(\d+)\s*.*protein", query.lower())
    if top_protein_match:
        try:
            n = int(top_protein_match.group(1))
            if 'proteins_100g' in original_df.columns and 'product' in original_df.columns:
                top_proteins = original_df.nlargest(n, 'proteins_100g')[['product', 'proteins_100g']]
                response_parts.append(f"Here are the top {n} foods highest in protein (per 100g):")
                table_md = top_proteins.to_markdown(index=False)
                response_parts.append(f"\n{table_md}\n")
                
                # Offer to visualize
                if st.button("Show Protein Chart"):
                    fig = px.bar(top_proteins, x='product', y='proteins_100g', title=f"Top {n} High-Protein Foods")
                    st.plotly_chart(fig)
            else:
                response_parts.append("Required columns ('product', 'proteins_100g') not found for this query.")
        except (ValueError, IndexError):
            response_parts.append("I couldn't understand the number 'N' from your query.")
        return "\n".join(response_parts), chart

    # 3. Find foods with more than Xg carbs and less than Yg fat
    carb_fat_match = re.search(r"more than\s*(\d+\.?\d*)\s*g.*carbohydrates.*less than\s*(\d+\.?\d*)\s*g.*fat", query.lower())
    if carb_fat_match:
         try:
             carb_threshold = float(carb_fat_match.group(1))
             fat_threshold = float(carb_fat_match.group(2))
             if 'carbohydrates_100g' in original_df.columns and 'fat_100g' in original_df.columns and 'product' in original_df.columns:
                 filtered_foods = original_df[
                     (original_df['carbohydrates_100g'] > carb_threshold) &
                     (original_df['fat_100g'] < fat_threshold)
                 ][['product', 'carbohydrates_100g', 'fat_100g']]
                 if not filtered_foods.empty:
                     response_parts.append(f"Foods with more than {carb_threshold}g carbs and less than {fat_threshold}g fat per 100g:")
                     table_md = filtered_foods.head(10).to_markdown(index=False)
                     response_parts.append(f"\n{table_md}\n")
                 else:
                      response_parts.append(f"No foods found matching those criteria.")
             else:
                  response_parts.append("Required columns not found for this query.")
         except (ValueError, IndexError):
              response_parts.append("I couldn't parse the carbohydrate or fat thresholds.")
         return "\n".join(response_parts), chart

    # 4. Nutritional breakdown of a specific food (pie chart)
    breakdown_match = re.search(r"nutritional breakdown.*(?:of|for)\s*(.*)", query.lower())
    if breakdown_match:
        food_name = breakdown_match.group(1).strip()
        # Find matching food (simple substring match)
        matching_rows = original_df[original_df['product'].str.contains(food_name, case=False, na=False)]
        if not matching_rows.empty:
            food_data = matching_rows.iloc[0]
            # Assume standard nutrition columns exist
            nutrition_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g']
            # Check for sugar in carbohydrates
            if 'sugars_100g' in original_df.columns and 'carbohydrates_100g' in original_df.columns:
                 # Approximate fiber or other carbs if needed, but let's keep it simple
                 # Pie chart needs positive values, ensure they exist or are calculated sensibly
                 values = [food_data.get(col, 0) for col in nutrition_cols]
                 names = ['Fat', 'Carbohydrates', 'Protein']
                 
                 # Filter out non-existent or negative values for charting
                 valid_data = [(name, val) for name, val in zip(names, values) if pd.notna(val) and val >= 0]
                 if valid_data:
                     names_filtered, values_filtered = zip(*valid_data)
                     if sum(values_filtered) > 0: # Avoid empty chart
                         fig = px.pie(values=values_filtered, names=names_filtered, title=f"Nutritional Breakdown: {food_data['product']}")
                         chart = fig
                         response_parts.append(f"Here is the nutritional breakdown for {food_data['product']} (per 100g):")
                         # List the values
                         for name, val in valid_data:
                             response_parts.append(f"- {name}: {val:.2f}g")
                     else:
                         response_parts.append(f"Nutritional data for {food_data['product']} is not suitable for a pie chart.")
                 else:
                      response_parts.append(f"Could not extract valid nutritional data for {food_data['product']}.")
            else:
                response_parts.append("Standard nutrition columns not found for breakdown.")
        else:
            response_parts.append(f"Food '{food_name}' not found in the dataset.")
        return "\n".join(response_parts), chart

    # --- Default/Fallback Response (Simplified RAG-like) ---
    # This part would ideally be replaced by a full RAG implementation.
    # For now, it returns a generic message.
    response_parts.append("I'm still learning to understand all your questions perfectly!")
    response_parts.append("Try asking specific questions like:")
    response_parts.append("- 'Show top 5 foods highest in protein'")
    response_parts.append("- 'Find foods with fat content above 20g'")
    response_parts.append("- 'Nutritional breakdown of Greek Yogurt'")
    return "\n".join(response_parts), chart

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    st.markdown("Developed by: [DM Shahriar Hossain](https://github.com/rownokstar/)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file)
            
            # Store original dataframe for rule-based queries
            st.session_state.original_df = df.copy()
            
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
if prompt := st.chat_input("Ask anything about your dataset (e.g., 'Top 5 high protein foods')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Simulate streaming by displaying response word by word (basic simulation)
        full_response, chart = generate_response(prompt, st.session_state.original_df if st.session_state.original_df is not None else pd.DataFrame())
        
        # --- Streaming Display ---
        message_placeholder = st.empty()
        displayed_text = ""
        for word in full_response.split(" "):
            displayed_text += word + " "
            message_placeholder.markdown(displayed_text + "▌") # Cursor effect
            time.sleep(0.01) # Adjust delay for streaming speed
        message_placeholder.markdown(displayed_text) # Final display without cursor
        
        # Display chart if generated
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
            
        # Store complete message in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "chart": chart
        })

# --- Tips ---
st.info("💡 Try: 'Show top 5 foods highest in protein' or 'Find foods with fat content above 20g'")
st.info("💡 Ask for 'nutritional breakdown of [food name]' for pie charts!")
st.markdown("---")
st.markdown("👨‍💻 Developed by: [DM Shahriar Hossain](https://linkedin.com/in/dm-shahriar-hossain/) | [GitHub](https://github.com/rownokstar/)")
