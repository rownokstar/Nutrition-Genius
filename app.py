# app.py - Nutrition Genius with Optimized Performance and Improved Responses (For nutrition_table.csv)
# Developed by: DM Shahriar Hossain (https://github.com/rownokstar/)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
from io import StringIO
import time
import re
import random

# --- Config ---
st.set_page_config(page_title="🍏 Nutrition Genius", page_icon="🍏", layout="wide")
st.title("🍏 Nutrition Genius")
st.markdown("> *Smart Dataset Auto-Detector & AI Assistant*")
st.markdown("**Developed by: [DM Shahriar Hossain](https://linkedin.com/in/dm-shahriar-hossain/)**")

# --- Session State ---
if "df" not in st.session_state:
    st.session_state.df = None # Processed/sampled df for display/search
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
    st.session_state.original_df = None # Full original df for rule-based queries
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {} # Map generic names to actual column indices/names

# --- Auto Dataset Type Detection ---
def detect_dataset_type(df):
    """Automatically detect if dataset is supervised or unsupervised"""
    # Very basic heuristic based on numeric density for this specific case
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
    if numeric_ratio > 0.5:
        return "unsupervised" # Nutrition data is mostly numeric
    else:
        return "supervised"

# --- Optimized Smart Processing with Progress Tracking ---
def smart_process_dataset(df, data_type):
    """Smart processing with automatic optimization and progress tracking"""
    total_rows = len(df)

    if total_rows > 10000:
        sample_size = 2000
        df_process = df.sample(n=sample_size, random_state=42)
        # st.warning(f"📊 Large dataset detected ({total_rows} rows). Processing {sample_size} representative samples for optimal performance.")
    elif total_rows > 1000:
        sample_size = min(1000, total_rows)
        df_process = df.sample(n=sample_size, random_state=42)
        # st.info(f"⚡ Processing {sample_size} rows for better performance.")
    else:
        df_process = df
        # st.info(f"🚀 Processing all {total_rows} rows.")

    try:
        progress_bar = st.progress(0)
        status = st.empty()

        status.text("📝 Converting data to text format...")
        sentences = []
        df_process_rows = list(df_process.iterrows())

        for i, (idx, row) in enumerate(df_process_rows):
            sentences.append(' '.join(str(val) for val in row.values if pd.notna(val)))
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

# --- Column Mapping Helper ---
def identify_nutrition_columns(df):
    """
    Attempt to identify key nutrition columns based on position or data characteristics.
    This is specific to the structure seen in nutrition_table.csv.
    Assumes columns are roughly in this order:
    [ID, Fat, Carbs, Sugars, Protein, ... , Product Name]
    """
    mapping = {}
    num_cols = len(df.columns)
    if num_cols >= 10: # Sanity check
        # Based on the sample, these seem to be consistent positions
        # Adjust indices if your data structure differs slightly
        try:
            mapping['id'] = df.columns[0]
            mapping['fat_100g'] = df.columns[1] # 11.11, 16.67 etc.
            mapping['carbohydrates_100g'] = df.columns[2] # 44.44, 33.33 etc.
            mapping['sugars_100g'] = df.columns[3] # 33.33, 33.33 etc.
            mapping['proteins_100g'] = df.columns[4] # 22.22, 22.22 etc.
            # mapping['salt_100g'] = df.columns[5] # 20.883879999999998 etc.
            # mapping['sodium_100g'] = df.columns[6] # 1393.0 etc.
            mapping['energy_kcal'] = df.columns[7] # 1566.51 etc.
            # mapping['energy_kj'] = df.columns[7] # 1566.51 etc. (Seems kcal is col 7)
            mapping['fiber_100g'] = df.columns[8] # 77.77, 72.22 etc.
            # mapping['fruits_veg_nuts_100g'] = df.columns[9] # 0, 0 etc.
            mapping['product'] = df.columns[-1] # Last column is usually the name
        except IndexError:
            pass # If columns don't match expected structure
    return mapping

# --- Enhanced Response Generator with Natural Language and Formatting ---
def generate_response(query, original_df, column_mapping):
    """
    Generate a natural language response with potential table/chart.
    """
    chart = None
    response_parts = []

    if original_df is None or column_mapping is None or len(column_mapping) == 0:
        response_parts.append("⚠️ The dataset needs to be processed correctly first. Please re-upload and wait for processing to complete.")
        return "\n".join(response_parts), chart

    # Ensure columns exist
    fat_col = column_mapping.get('fat_100g')
    carb_col = column_mapping.get('carbohydrates_100g')
    sugar_col = column_mapping.get('sugars_100g')
    protein_col = column_mapping.get('proteins_100g')
    kcal_col = column_mapping.get('energy_kcal')
    fiber_col = column_mapping.get('fiber_100g')
    product_col = column_mapping.get('product')

    # 1. Show top N foods highest in protein
    top_protein_match = re.search(r"top\s*(\d+)\s*.*(?:high.*|highest.*|rich.*|most.*|greatest.*)\s*protein", query.lower())
    if top_protein_match and protein_col and product_col:
        try:
            n = int(top_protein_match.group(1))
            if protein_col in original_df.columns and product_col in original_df.columns:
                # Filter out rows where protein content might be missing/invalid for sorting
                filtered_df = original_df.dropna(subset=[protein_col])
                top_proteins = filtered_df.nlargest(n, protein_col)[[product_col, protein_col]]
                if not top_proteins.empty:
                    response_parts.append(f"🏆 Here are the top {n} foods highest in protein (per 100g):")
                    table_md = top_proteins.to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                    response_parts.append(f"😕 Couldn't find enough foods with valid protein data for the top {n} list.")
            else:
                response_parts.append("❌ Required columns for protein data not found correctly.")
        except (ValueError, IndexError):
            response_parts.append("🔢 I couldn't understand the number 'N' from your query. Please specify like 'top 5'.")
        return "\n".join(response_parts), chart

    # 2. Find foods with more than Xg carbs and less than Yg fat
    carb_fat_match = re.search(r"more than\s*(\d+\.?\d*)\s*g.*(?:carbohydrates?|carbs?|carbs).*less than\s*(\d+\.?\d*)\s*g.*(?:fat|fats)", query.lower())
    if carb_fat_match and carb_col and fat_col and product_col:
         try:
             carb_threshold = float(carb_fat_match.group(1))
             fat_threshold = float(carb_fat_match.group(2))
             if carb_col in original_df.columns and fat_col in original_df.columns and product_col in original_df.columns:
                 # Filter out rows with missing data
                 filtered_df = original_df.dropna(subset=[carb_col, fat_col])
                 mask = (filtered_df[carb_col] > carb_threshold) & (filtered_df[fat_col] < fat_threshold)
                 filtered_foods = filtered_df[mask][[product_col, carb_col, fat_col]]
                 if not filtered_foods.empty:
                     response_parts.append(f"✅ Found foods with more than {carb_threshold}g carbs and less than {fat_threshold}g fat per 100g:")
                     table_md = filtered_foods.head(10).to_markdown(index=False) # Limit display
                     response_parts.append(f"\n{table_md}\n")
                 else:
                      response_parts.append(f"📭 No foods found matching the criteria (carbs > {carb_threshold}g AND fat < {fat_threshold}g).")
             else:
                  response_parts.append("❌ Required columns for carbs/fat filtering not found correctly.")
         except (ValueError, IndexError):
              response_parts.append("🔢 I couldn't parse the carbohydrate or fat thresholds. Please use format like 'more than 20g carbs and less than 10g fat'.")
         return "\n".join(response_parts), chart

    # 3. Find foods with fat content above Xg
    fat_above_match = re.search(r"(?:fat.*content|total fat|fats?).*above\s*(\d+\.?\d*)\s*g", query.lower()) or \
                      re.search(r"find.*foods.*(?:fat|fats).*above\s*(\d+\.?\d*)\s*g", query.lower())
    if fat_above_match and fat_col and product_col:
        try:
            threshold = float(fat_above_match.group(1))
            if fat_col in original_df.columns and product_col in original_df.columns:
                # Filter out rows with missing data
                filtered_df = original_df.dropna(subset=[fat_col])
                high_fat_foods = filtered_df[filtered_df[fat_col] > threshold][[product_col, fat_col]]
                if not high_fat_foods.empty:
                    response_parts.append(f"🥓 Here are foods with fat content above {threshold}g per 100g:")
                    table_md = high_fat_foods.head(10).to_markdown(index=False) # Limit display
                    response_parts.append(f"\n{table_md}\n")
                else:
                     response_parts.append(f"📭 No foods found with fat content above {threshold}g.")
            else:
                 response_parts.append("❌ Required columns for fat filtering not found correctly.")
        except (ValueError, IndexError):
             response_parts.append("🔢 I couldn't understand the fat threshold value. Please specify like 'fat content above 20g'.")
        return "\n".join(response_parts), chart

    # 4. Nutritional breakdown of a specific food (pie chart)
    breakdown_match = re.search(r"nutritional breakdown.*(?:of|for)\s*(.*)", query.lower()) or \
                      re.search(r"nutrition.*(?:of|for)\s*(.*)", query.lower())
    if breakdown_match and product_col:
        food_name = breakdown_match.group(1).strip()
        if food_name:
            # Find matching food (simple substring match)
            if product_col in original_df.columns:
                matching_rows = original_df[original_df[product_col].str.contains(food_name, case=False, na=False)]
                if not matching_rows.empty:
                    food_data = matching_rows.iloc[0]
                    # Define nutrients for pie chart
                    nutrients_for_chart = [
                        (fat_col, 'Fat'),
                        (carb_col, 'Carbohydrates'),
                        (protein_col, 'Protein')
                        # Add sugar if needed: (sugar_col, 'Sugar')
                    ]
                    # Collect valid data
                    values = []
                    names = []
                    for col_key, name in nutrients_for_chart:
                        col = column_mapping.get(col_key)
                        if col and col in food_data.index and pd.notna(food_data[col]) and food_data[col] >= 0:
                            values.append(food_data[col])
                            names.append(name)

                    # Display nutrient list
                    response_parts.append(f"📊 Nutritional breakdown for **{food_data[product_col]}** (per 100g):")
                    nutrient_list = [
                        (fat_col, 'Fat'),
                        (carb_col, 'Carbohydrates'),
                        (sugar_col, 'Sugars'),
                        (protein_col, 'Protein'),
                        (kcal_col, 'Calories (kcal)'),
                        (fiber_col, 'Fiber')
                    ]
                    for col_key, name in nutrient_list:
                        col = column_mapping.get(col_key)
                        if col and col in food_data.index and pd.notna(food_data[col]):
                            response_parts.append(f"- **{name}**: {food_data[col]:.2f}g" if not 'kcal' in name else f"- **{name}**: {food_data[col]:.0f} kcal")

                    # Create pie chart if valid data exists
                    if values and sum(values) > 0:
                        # Only show chart if there are meaningful values
                        if any(v > 0.1 for v in values): # Avoid chart for negligible values
                             fig = px.pie(values=values, names=names, title=f"Nutrient Distribution: {food_data[product_col]}")
                             chart = fig
                             response_parts.append("\n_(See pie chart below for visual breakdown)_")
                        else:
                             response_parts.append("\n_(Nutrient values are too low to display in a chart)_")
                    else:
                        response_parts.append("\n_(Could not generate a chart due to missing or invalid data)_")

                else:
                    response_parts.append(f"📭 Food '{food_name}' not found in the dataset.")
            else:
                 response_parts.append("❌ Product name column not found correctly.")
        else:
             response_parts.append("🔤 Please specify the food name for the breakdown.")
        return "\n".join(response_parts), chart

    # 5. Find foods with protein content above Xg
    protein_above_match = re.search(r"(?:protein.*content|proteins?).*above\s*(\d+\.?\d*)\s*g", query.lower()) or \
                          re.search(r"find.*foods.*(?:protein|proteins).*above\s*(\d+\.?\d*)\s*g", query.lower())
    if protein_above_match and protein_col and product_col:
        try:
            threshold = float(protein_above_match.group(1))
            if protein_col in original_df.columns and product_col in original_df.columns:
                filtered_df = original_df.dropna(subset=[protein_col])
                high_protein_foods = filtered_df[filtered_df[protein_col] > threshold][[product_col, protein_col]]
                if not high_protein_foods.empty:
                    response_parts.append(f"🍗 Here are foods with protein content above {threshold}g per 100g:")
                    table_md = high_protein_foods.head(10).to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                     response_parts.append(f"📭 No foods found with protein content above {threshold}g.")
            else:
                 response_parts.append("❌ Required columns for protein filtering not found correctly.")
        except (ValueError, IndexError):
             response_parts.append("🔢 I couldn't understand the protein threshold value. Please specify like 'protein content above 10g'.")
        return "\n".join(response_parts), chart


    # --- Default/Fallback Response ---
    response_parts.append("🤔 I'm still learning to understand all your questions perfectly!")
    response_parts.append("\n💡 Try asking specific questions like:")
    response_parts.append("- 'Show top 5 foods highest in protein'")
    response_parts.append("- 'Find foods with fat content above 20g'")
    response_parts.append("- 'List foods with more than 30g carbs and less than 5g fat'")
    response_parts.append("- 'Nutritional breakdown of [food name]'")
    return "\n".join(response_parts), chart

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    st.markdown("Developed by: [DM Shahriar Hossain](https://github.com/rownokstar/)")
    uploaded_file = st.file_uploader("Upload CSV (e.g., nutrition_table.csv)", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("📥 Loading dataset..."):
            try:
                # Load dataset
                df = pd.read_csv(uploaded_file, header=None) # Load without assuming header

                # Store original dataframe for rule-based queries
                st.session_state.original_df = df.copy()

                # Identify columns
                st.session_state.column_mapping = identify_nutrition_columns(df)
                # st.write("Identified Columns:", st.session_state.column_mapping) # Debug

                # Auto-optimize for performance (always sample for large datasets like this one)
                if len(df) > 1000:
                    df_display = df.sample(min(5, len(df)), random_state=42) # Show tiny sample
                    df_process = df.sample(min(2000, len(df)), random_state=42) # Process sample
                    st.warning(f"📊 Large dataset detected ({len(df)} rows). Processing a sample for optimal performance.")
                else:
                    df_display = df.head(5)
                    df_process = df

                st.session_state.df = df_display # Show smaller preview

                # Auto-detect dataset type
                detected_type = detect_dataset_type(df_process)
                st.session_state.data_type = detected_type

                st.success(f"✅ Dataset loaded! Detected as: **{detected_type.upper()}**")

                # Quick processing
                with st.spinner("🚀 Processing dataset... (This should be quick now)"):
                    index, sentences, success, error = smart_process_dataset(df_process, detected_type)

                if success:
                    st.session_state.index = index
                    st.session_state.sentences = sentences
                    st.session_state.processed = True
                    st.success("✅ Dataset processed & ready for queries!")
                    st.balloons()
                else:
                    st.error(f"❌ Processing failed: {error}")

            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                # Reset state on error
                st.session_state.original_df = None
                st.session_state.column_mapping = {}

    # Show dataset info
    if st.session_state.df is not None and st.session_state.column_mapping:
        st.subheader("📊 Dataset Info")
        st.write(f"**Type:** {st.session_state.data_type or 'Not processed'}")
        st.write(f"**Total Rows:** {len(st.session_state.original_df) if st.session_state.original_df is not None else 'Unknown'}")
        st.write(f"**Sample Rows Displayed:** {len(st.session_state.df)}")
        # Show mapped column names
        # st.write(f"**Identified Columns:** {st.session_state.column_mapping}")
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head(3))

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg and msg["chart"] is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about foods, nutrients, or diets... (e.g., 'Top 5 high protein foods')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Simulate streaming by displaying response word by word
        full_response, chart = generate_response(
            prompt,
            st.session_state.original_df,
            st.session_state.column_mapping
        )

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
st.info("💡 **Pro Tip:** The more specific your question, the better the answer!")
st.info("📋 **Try these examples:**\n"
        "- 'Show top 5 foods highest in protein'\n"
        "- 'Find foods with fat content above 30g'\n"
        "- 'List foods with more than 25g carbs and less than 3g fat'\n"
        "- 'Nutritional breakdown of Greek Yogurt'")
st.markdown("---")
st.markdown("👨‍💻 Developed by: [DM Shahriar Hossain](https://linkedin.com/in/dm-shahriar-hossain/) | [GitHub](https://github.com/rownokstar/)")
