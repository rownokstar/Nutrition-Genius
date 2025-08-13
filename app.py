# app.py - Nutrition Genius with Thinking Process Display
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
    Attempt to identify key nutrition columns based on position.
    Assumes columns are roughly in this order:
    [ID, Fat, Carbs, Sugars, Protein, Salt, Sodium, Energy_kcal, Fiber, G_sum/Exceeded, Product Name]
    """
    mapping = {}
    num_cols = len(df.columns)
    if num_cols >= 10:
        try:
            mapping['id'] = df.columns[0]
            mapping['fat_100g'] = df.columns[1]
            mapping['carbohydrates_100g'] = df.columns[2]
            mapping['sugars_100g'] = df.columns[3]
            mapping['proteins_100g'] = df.columns[4]
            mapping['salt_100g'] = df.columns[5]
            mapping['sodium_100g'] = df.columns[6]
            mapping['energy_kcal'] = df.columns[7]
            mapping['fiber_100g'] = df.columns[8]
            # mapping['g_sum_or_exceeded'] = df.columns[9] # Often not needed
            mapping['product'] = df.columns[-1] # Last column is usually the name
        except IndexError:
            pass
    return mapping

# --- Enhanced Response Generator with Natural Language, Formatting, and Thinking ---
def generate_response_with_thinking(query, original_df, column_mapping):
    """
    Generate a natural language response with potential table/chart and thinking process.
    """
    chart = None
    thinking_steps = []
    response_parts = []

    if original_df is None or column_mapping is None or len(column_mapping) == 0:
        thinking_steps.append("⚠️ **Thinking:** The dataset needs to be processed correctly first.")
        response_parts.append("⚠️ The dataset needs to be processed correctly first. Please re-upload and wait for processing to complete.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # Ensure columns exist
    fat_col = column_mapping.get('fat_100g')
    carb_col = column_mapping.get('carbohydrates_100g')
    sugar_col = column_mapping.get('sugars_100g')
    protein_col = column_mapping.get('proteins_100g')
    kcal_col = column_mapping.get('energy_kcal')
    fiber_col = column_mapping.get('fiber_100g')
    product_col = column_mapping.get('product')

    # --- Helper Function for Safe Numerical Filtering ---
    def safe_numeric_filter(df, column_name, condition_func, step_desc=""):
        """Safely filter a DataFrame column by converting to numeric and applying a condition."""
        if step_desc:
             thinking_steps.append(f"🧠 **Thinking:** {step_desc}")
        if column_name not in df.columns:
            thinking_steps.append(f"❌ **Thinking:** Column '{column_name}' not found in dataset.")
            return pd.DataFrame()
        df_copy = df.copy()
        initial_count = len(df_copy)
        thinking_steps.append(f"...Converting column '{column_name}' to numeric (errors become NaN)...")
        df_copy[column_name] = pd.to_numeric(df_copy[column_name], errors='coerce')
        after_conversion_count = len(df_copy.dropna(subset=[column_name]))
        thinking_steps.append(f"...Dropping rows with invalid '{column_name}' (NaN). Went from {initial_count} to {after_conversion_count} rows...")
        df_filtered = df_copy.dropna(subset=[column_name])
        try:
            mask = condition_func(df_filtered[column_name])
            final_df = df_filtered[mask]
            thinking_steps.append(f"...Applied condition. Final count: {len(final_df)} rows.")
            return final_df
        except Exception as e:
            thinking_steps.append(f"❌ **Thinking:** Error applying condition to column {column_name}: {e}")
            return pd.DataFrame()

    # --- 1. Show top N foods highest in protein ---
    top_protein_match = re.search(r"top\s*(\d+)\s*.*(?:high.*|highest.*|rich.*|most.*|greatest.*)\s*protein", query.lower())
    if top_protein_match and protein_col and product_col:
        thinking_steps.append("🔍 **Thinking:** Query matches pattern for 'Top N foods highest in protein'.")
        try:
            n = int(top_protein_match.group(1))
            thinking_steps.append(f"...Looking for top {n} foods by protein content.")
            if protein_col in original_df.columns and product_col in original_df.columns:
                thinking_steps.append("...Required columns found. Filtering data...")
                filtered_df = safe_numeric_filter(original_df, protein_col, lambda x: x >= 0, "Filtering for valid (non-negative) protein values.")
                if not filtered_df.empty:
                    thinking_steps.append("...Sorting filtered data by protein content (descending)...")
                    top_proteins = filtered_df.nlargest(n, protein_col)[[product_col, protein_col]]
                    if not top_proteins.empty:
                        thinking_steps.append("✅ **Thinking:** Found results. Formatting response...")
                        response_parts.append(f"🏆 Here are the top {n} foods highest in protein (per 100g):")
                        top_proteins_display = top_proteins.copy()
                        top_proteins_display[protein_col] = top_proteins_display[protein_col].apply(lambda x: f"{x:.2f}g")
                        table_md = top_proteins_display.to_markdown(index=False)
                        response_parts.append(f"\n{table_md}\n")
                    else:
                        thinking_steps.append("📭 **Thinking:** Couldn't find enough foods with valid protein data for the top list.")
                        response_parts.append(f"😕 Couldn't find enough foods with valid protein data for the top {n} list.")
                else:
                    thinking_steps.append("📭 **Thinking:** No foods with valid protein data found after filtering.")
                    response_parts.append(f"😕 No foods with valid protein data found.")
            else:
                thinking_steps.append("❌ **Thinking:** Required columns for protein data not found correctly.")
                response_parts.append("❌ Required columns for protein data not found correctly.")
        except (ValueError, IndexError):
            thinking_steps.append("🔢 **Thinking:** Couldn't parse the number 'N'.")
            response_parts.append("🔢 I couldn't understand the number 'N' from your query. Please specify like 'top 5'.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- 2. Find foods with more than Xg carbs and less than Yg fat ---
    carb_fat_match = re.search(r"more than\s*(\d+\.?\d*)\s*g.*(?:carbohydrates?|carbs?|carbs).*less than\s*(\d+\.?\d*)\s*g.*(?:fat|fats)", query.lower())
    if carb_fat_match and carb_col and fat_col and product_col:
        thinking_steps.append("🔍 **Thinking:** Query matches pattern for 'Foods with carbs > Xg AND fat < Yg'.")
        try:
            carb_threshold = float(carb_fat_match.group(1))
            fat_threshold = float(carb_fat_match.group(2))
            thinking_steps.append(f"...Carb threshold: > {carb_threshold}g, Fat threshold: < {fat_threshold}g.")
            if carb_col in original_df.columns and fat_col in original_df.columns and product_col in original_df.columns:
                thinking_steps.append("...Required columns found. Filtering data...")
                df_carbs_ok = safe_numeric_filter(original_df, carb_col, lambda x: x > carb_threshold, f"Filtering for carbs > {carb_threshold}g.")
                if df_carbs_ok.empty:
                    thinking_steps.append("📭 **Thinking:** No foods found meeting the carb criterion.")
                    response_parts.append(f"📭 No foods found with carbohydrates > {carb_threshold}g.")
                    return "\n".join(thinking_steps), "\n".join(response_parts), chart

                df_final = safe_numeric_filter(df_carbs_ok, fat_col, lambda x: x < fat_threshold, f"Filtering for fat < {fat_threshold}g (from carb-filtered set).")
                if not df_final.empty:
                    thinking_steps.append("✅ **Thinking:** Found matching foods. Formatting response...")
                    response_parts.append(f"✅ Found foods with more than {carb_threshold}g carbs and less than {fat_threshold}g fat per 100g:")
                    result_df = df_final[[product_col, carb_col, fat_col]].head(10)
                    result_df_display = result_df.copy()
                    result_df_display[carb_col] = result_df_display[carb_col].apply(lambda x: f"{x:.2f}g")
                    result_df_display[fat_col] = result_df_display[fat_col].apply(lambda x: f"{x:.2f}g")
                    table_md = result_df_display.to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                    thinking_steps.append("📭 **Thinking:** No foods found meeting both criteria.")
                    response_parts.append(f"📭 No foods found matching both criteria (carbs > {carb_threshold}g AND fat < {fat_threshold}g).")
            else:
                thinking_steps.append("❌ **Thinking:** Required columns for filtering not found.")
                response_parts.append("❌ Required columns for carbs/fat filtering not found correctly.")
        except (ValueError, IndexError):
            thinking_steps.append("🔢 **Thinking:** Couldn't parse the thresholds.")
            response_parts.append("🔢 I couldn't parse the carbohydrate or fat thresholds. Please use format like 'more than 20g carbs and less than 10g fat'.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- 3. Find foods with fat content above Xg ---
    fat_above_match = re.search(r"(?:fat.*content|total fat|fats?).*above\s*(\d+\.?\d*)\s*g", query.lower()) or \
                      re.search(r"find.*foods.*(?:fat|fats).*above\s*(\d+\.?\d*)\s*g", query.lower())
    if fat_above_match and fat_col and product_col:
        thinking_steps.append("🔍 **Thinking:** Query matches pattern for 'Foods with fat > Xg'.")
        try:
            threshold = float(fat_above_match.group(1))
            thinking_steps.append(f"...Fat threshold: > {threshold}g.")
            if fat_col in original_df.columns and product_col in original_df.columns:
                thinking_steps.append("...Required columns found. Filtering data...")
                high_fat_foods_df = safe_numeric_filter(original_df, fat_col, lambda x: x > threshold, f"Filtering for fat > {threshold}g.")
                if not high_fat_foods_df.empty:
                    thinking_steps.append("✅ **Thinking:** Found matching foods. Formatting response...")
                    response_parts.append(f"🥓 Here are foods with fat content above {threshold}g per 100g:")
                    result_df = high_fat_foods_df[[product_col, fat_col]].head(10)
                    result_df_display = result_df.copy()
                    result_df_display[fat_col] = result_df_display[fat_col].apply(lambda x: f"{x:.2f}g")
                    table_md = result_df_display.to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                    thinking_steps.append("📭 **Thinking:** No foods found meeting the fat criterion.")
                    response_parts.append(f"📭 No foods found with fat content above {threshold}g.")
            else:
                 thinking_steps.append("❌ **Thinking:** Required columns for fat filtering not found.")
                 response_parts.append("❌ Required columns for fat filtering not found correctly.")
        except (ValueError, IndexError):
             thinking_steps.append("🔢 **Thinking:** Couldn't parse the fat threshold.")
             response_parts.append("🔢 I couldn't understand the fat threshold value. Please specify like 'fat content above 20g'.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- 4. Nutritional breakdown of a specific food (pie chart) ---
    breakdown_match = re.search(r"nutritional breakdown.*(?:of|for)\s*(.*)", query.lower()) or \
                      re.search(r"nutrition.*(?:of|for)\s*(.*)", query.lower())
    if breakdown_match and product_col:
        thinking_steps.append("🔍 **Thinking:** Query matches pattern for 'Nutritional breakdown of [food]'.")
        food_name = breakdown_match.group(1).strip()
        if food_name:
            thinking_steps.append(f"...Searching for food: '{food_name}'.")
            if product_col in original_df.columns:
                thinking_steps.append("...Product column found. Searching...")
                matching_rows = original_df[original_df[product_col].str.contains(food_name, case=False, na=False)]
                if not matching_rows.empty:
                    food_data = matching_rows.iloc[0]
                    thinking_steps.append(f"...Found '{food_data[product_col]}'. Extracting nutrient data...")
                    nutrients_for_chart = [(fat_col, 'Fat'), (carb_col, 'Carbohydrates'), (protein_col, 'Protein')]
                    values = []
                    names = []
                    details = []
                    for col_key, name in nutrients_for_chart:
                        col = column_mapping.get(col_key)
                        if col and col in food_data.index:
                            val_str = food_data[col]
                            try:
                                val_num = float(val_str)
                                if val_num >= 0:
                                    values.append(val_num)
                                    names.append(name)
                                    details.append(f"- **{name}**: {val_num:.2f}g")
                                else:
                                    details.append(f"- **{name}**: Invalid data ({val_str})")
                            except (ValueError, TypeError):
                                details.append(f"- **{name}**: Not available ({val_str})")
                        else:
                             details.append(f"- **{name}**: Column not found")

                    response_parts.append(f"📊 Nutritional breakdown for **{food_data[product_col]}** (per 100g):")
                    response_parts.extend(details)

                    if kcal_col and kcal_col in food_data.index:
                        try:
                            kcal_val = float(food_data[kcal_col])
                            response_parts.append(f"- **Energy**: {kcal_val:.0f} kcal")
                        except (ValueError, TypeError):
                            response_parts.append(f"- **Energy**: Not available ({food_data[kcal_col]})")

                    if values and sum(values) > 0:
                        if any(v > 0.1 for v in values):
                             thinking_steps.append("✅ **Thinking:** Valid data found. Creating pie chart...")
                             fig = px.pie(values=values, names=names, title=f"Nutrient Distribution: {food_data[product_col]}")
                             chart = fig
                             response_parts.append("\n_(See pie chart below for visual breakdown)_")
                        else:
                             thinking_steps.append("📉 **Thinking:** Nutrient values are too low for a meaningful chart.")
                             response_parts.append("\n_(Nutrient values are too low to display in a chart)_")
                    else:
                        thinking_steps.append("📉 **Thinking:** Could not generate chart due to missing/invalid data.")
                        response_parts.append("\n_(Could not generate a chart due to missing or invalid data)_")

                else:
                    thinking_steps.append(" HttpNotFound **Thinking:** Food not found in dataset.")
                    response_parts.append(f" HttpNotFound Food '{food_name}' not found in the dataset.")
            else:
                 thinking_steps.append("❌ **Thinking:** Product name column not found.")
                 response_parts.append("❌ Product name column not found correctly.")
        else:
             thinking_steps.append("🔤 **Thinking:** No food name specified in query.")
             response_parts.append("🔤 Please specify the food name for the breakdown.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- 5. Find foods with protein content above Xg ---
    protein_above_match = re.search(r"(?:protein.*content|proteins?).*above\s*(\d+\.?\d*)\s*g", query.lower()) or \
                          re.search(r"find.*foods.*(?:protein|proteins).*above\s*(\d+\.?\d*)\s*g", query.lower())
    if protein_above_match and protein_col and product_col:
        thinking_steps.append("🔍 **Thinking:** Query matches pattern for 'Foods with protein > Xg'.")
        try:
            threshold = float(protein_above_match.group(1))
            thinking_steps.append(f"...Protein threshold: > {threshold}g.")
            if protein_col in original_df.columns and product_col in original_df.columns:
                thinking_steps.append("...Required columns found. Filtering data...")
                high_protein_foods_df = safe_numeric_filter(original_df, protein_col, lambda x: x > threshold, f"Filtering for protein > {threshold}g.")
                if not high_protein_foods_df.empty:
                    thinking_steps.append("✅ **Thinking:** Found matching foods. Formatting response...")
                    response_parts.append(f"🍗 Here are foods with protein content above {threshold}g per 100g:")
                    result_df = high_protein_foods_df[[product_col, protein_col]].head(10)
                    result_df_display = result_df.copy()
                    result_df_display[protein_col] = result_df_display[protein_col].apply(lambda x: f"{x:.2f}g")
                    table_md = result_df_display.to_markdown(index=False)
                    response_parts.append(f"\n{table_md}\n")
                else:
                    thinking_steps.append(" HttpNotFound **Thinking:** No foods found meeting the protein criterion.")
                    response_parts.append(f" HttpNotFound No foods found with protein content above {threshold}g.")
            else:
                 thinking_steps.append("❌ **Thinking:** Required columns for protein filtering not found.")
                 response_parts.append("❌ Required columns for protein filtering not found correctly.")
        except (ValueError, IndexError):
             thinking_steps.append("🔢 **Thinking:** Couldn't parse the protein threshold.")
             response_parts.append("🔢 I couldn't understand the protein threshold value. Please specify like 'protein content above 10g'.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart


    # --- Default/Fallback Response ---
    thinking_steps.append("🤔 **Thinking:** Query didn't match any specific rule-based pattern.")
    thinking_steps.append("💡 **Thinking:** Providing fallback guidance.")
    response_parts.append("🤔 I'm still learning to understand all your questions perfectly!")
    response_parts.append("\n💡 Try asking specific questions like:")
    response_parts.append("- 'Show top 5 foods highest in protein'")
    response_parts.append("- 'Find foods with fat content above 20g'")
    response_parts.append("- 'List foods with more than 30g carbs and less than 5g fat'")
    response_parts.append("- 'Nutritional breakdown of [food name]'")
    return "\n".join(thinking_steps), "\n".join(response_parts), chart

# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    st.markdown("Developed by: [DM Shahriar Hossain](https://github.com/rownokstar/)")
    uploaded_file = st.file_uploader("Upload CSV (e.g., nutrition_table.csv)", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("📥 Loading dataset..."):
            try:
                # Load dataset - Crucially, specify header=None as the first row is data
                df = pd.read_csv(uploaded_file, header=None)

                # Store original dataframe for rule-based queries
                st.session_state.original_df = df.copy()

                # Identify columns based on known structure
                st.session_state.column_mapping = identify_nutrition_columns(df)

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
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df.head(3))

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Check if the message content is a tuple (thinking, response) or just a string
        if isinstance(msg["content"], tuple):
            thinking_content, response_content = msg["content"]
            if thinking_content:
                with st.expander("🧠 My Thinking Process", expanded=False):
                    st.markdown(thinking_content)
            st.markdown(response_content)
        else:
            st.markdown(msg["content"])
            
        if "chart" in msg and msg["chart"] is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about foods, nutrients, or diets... (e.g., 'Top 5 high protein foods')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Generate response with thinking
        thinking_process, full_response, chart = generate_response_with_thinking(
            prompt,
            st.session_state.original_df,
            st.session_state.column_mapping
        )

        # --- Streaming Display for Thinking ---
        if thinking_process:
            with st.expander("🧠 My Thinking Process", expanded=True): # Start expanded
                thinking_placeholder = st.empty()
                displayed_thinking = ""
                thinking_lines = thinking_process.splitlines()
                for i, line in enumerate(thinking_lines):
                    displayed_thinking += line + "\n"
                    thinking_placeholder.markdown(displayed_thinking)
                    # Simulate processing delay for thinking steps
                    if "✅" not in line and "❌" not in line and " HttpNotFound" not in line and "📉" not in line and "💡" not in line and "🔍" not in line and "🔢" not in line and "🔤" not in line: # Don't delay final outcomes
                         time.sleep(0.1) # Adjust for speed of thinking display

        # --- Streaming Display for Response ---
        if full_response:
            message_placeholder = st.empty()
            displayed_text = ""
            words = full_response.split(" ")
            for i, word in enumerate(words):
                displayed_text += word + " "
                message_placeholder.markdown(displayed_text + "▌") # Cursor effect
                time.sleep(0.01) # Adjust delay for streaming speed
            message_placeholder.markdown(displayed_text) # Final display without cursor
        else:
            st.markdown("Sorry, I couldn't generate a response for that query.")

        # Display chart if generated
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

        # Store complete message in history (store thinking and response separately)
        st.session_state.messages.append({
            "role": "assistant",
            "content": (thinking_process, full_response),
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
