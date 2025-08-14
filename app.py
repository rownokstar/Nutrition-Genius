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
        response_parts.append("❌ Transformers library is not installed. Please install it with `pip install transformers torch accelerate bitsandbytes`.")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- Ensure LLM Pipeline is Loaded ---
    if st.session_state.llm_pipeline is None:
        thinking_steps.append("🧠 **Thinking:** Loading the Llama 3 language model... This might take a while.")
        # You can make the model name configurable via sidebar input
        # model_name = st.sidebar.selectbox("Select Llama 3 Model", ["meta-llama/Meta-Llama-3-8B-Instruct", "unsloth/llama-3-8b-Instruct-bnb-4bit"], index=0)
        # For this specific task, Llama 3 8B Instruct is a good choice.
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # Default or configurable
        with st.spinner("Loading Llama 3 model (this can take several minutes)..."):
            st.session_state.llm_pipeline = load_llm_pipeline(model_name=model_name, use_quantization=True)
        if st.session_state.llm_pipeline is None:
            thinking_steps.append("❌ **Thinking:** Failed to load the Llama 3 model.")
            response_parts.append("❌ Failed to load the Llama 3 model. Check the logs for details. Ensure you have access to the model on Hugging Face and have provided your token if required.")
            return "\n".join(thinking_steps), "\n".join(response_parts), chart
        thinking_steps.append("✅ **Thinking:** Llama 3 model loaded.")

    # --- Prepare Context for the LLM ---
    thinking_steps.append("🧠 **Thinking:** Searching the dataset for relevant information...")
    try:
        # Use FAISS to find relevant data rows/sentences based on the query
        retrieved_sentences = retrieve_context(query, index, sentences, k=3)
        if retrieved_sentences:
            # Format the retrieved context for the LLM prompt
            # Joining sentences with newlines for clarity
            context_str = "\n".join([f"- {sent}" for sent in retrieved_sentences])
            thinking_steps.append(f"...Found {len(retrieved_sentences)} relevant data entries using similarity search.")
        else:
             # Fallback: If FAISS retrieval fails or returns nothing, use a simple sample
             # This is less ideal but ensures some context is provided.
             thinking_steps.append("...No context found via similarity search. Using a sample of the dataset as context.")
             sample_rows = original_df.sample(min(2, len(original_df)), random_state=42)
             # Use the column mapping to select relevant columns for the sample context
             cols_to_show = [col for col in [column_mapping.get('product'), column_mapping.get('fat_100g'),
                                            column_mapping.get('carbohydrates_100g'), column_mapping.get('proteins_100g'),
                                            column_mapping.get('energy_kcal'), column_mapping.get('fiber_100g')]
                             if col in original_df.columns]
             if cols_to_show and column_mapping.get('product'):
                 context_str = "Sample Data Context:\n" + sample_rows[cols_to_show].to_string(index=False)
             else:
                 # Ultimate fallback if column mapping is incomplete
                 context_str = "Sample Data Context (All Columns):\n" + sample_rows.to_string(index=False)

    except Exception as e:
        thinking_steps.append(f"❌ **Thinking:** Error during data retrieval/preparation: {e}")
        response_parts.append(f"Sorry, I encountered an error preparing the data context: {e}")
        return "\n".join(thinking_steps), "\n".join(response_parts), chart

    # --- Construct the Prompt for the LLM ---
    # Llama 3 Instruct models expect specific prompt formatting for best results.
    # Using the chat template if available, or a structured prompt otherwise.
    system_message = (
        "You are a helpful and knowledgeable Nutrition Assistant. A user has uploaded a food nutrition dataset (nutrition_table.csv). "
        "You have access to specific data snippets from that dataset relevant to the user's query. "
        "Your task is to answer the user's query using the provided data context. "
        "Your response should be in two parts: "
        "1.  **Thinking Process:** A numbered list detailing your reasoning steps. Start with \"1. Analyzing the query...\", \"2. Referring to the data context...\", etc. Be concise. "
        "2.  **Final Response:** A clear, natural language answer to the user's query. If the query asks for a list or table, format it nicely. If the query asks for a breakdown suitable for a chart, state the values clearly. "
    )
    user_message = (
        f"Here is the user's query: "
        f"<query> "
        f"{query} "
        f"</query> "
        f"Here is the relevant data context from the uploaded dataset: "
        f"<context> "
        f"{context_str} "
        f"</context> "
        f"Please provide your response in the following format: "
        f"<thinking> "
        f"1. [First step of your reasoning] "
        f"2. [Second step of your reasoning] "
        f"... "
        f"N. [Final step, leading to the conclusion] "
        f"</thinking> "
        f"<response> "
        f"[Your clear, natural language response to the user based on the query and context.] "
        f"</response> "
    )

    # Format messages according to Llama 3 Instruct expectations using the tokenizer's chat template
    # This often produces better results than a simple string prompt.
    try:
        # Ensure the tokenizer is available from the pipeline
        if hasattr(st.session_state.llm_pipeline, 'tokenizer'):
            tokenizer = st.session_state.llm_pipeline.tokenizer
        else:
            # Fallback if pipeline doesn't expose tokenizer directly (less likely)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=True)

        messages = [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()},
        ]
        # Apply the chat template to format the prompt correctly for Llama 3
        # The `add_generation_prompt` argument is important for instruct models.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # st.write(f"Debug: Formatted Prompt Length: {len(prompt)} chars") # Optional debug

    except Exception as e_template:
        st.warning(f"Could not apply chat template, falling back to string prompt: {e_template}")
        # Fallback to a simple concatenated prompt if template fails
        prompt = f"{system_message}\n\n{user_message}"

    thinking_steps.append("🤖 **Thinking:** Sending query and context to the local Llama 3 language model...")
    try:
        # --- Call the Llama 3 Pipeline ---
        # The pipeline handles tokenization, generation, and decoding
        # We pass the formatted prompt string.
        outputs = st.session_state.llm_pipeline(prompt)
        # Extract the generated text from the output list
        # The output is usually a list of dicts, e.g., [{'generated_text': '...'}]
        raw_llm_output = outputs[0]["generated_text"]

        thinking_steps.append("✅ **Thinking:** Received response from the Llama 3 model.")
        # --- Parse the LLM Output ---
        # Attempt to extract the <thinking> and <response> sections using regex
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_llm_output, re.DOTALL)
        response_match = re.search(r"<response>(.*?)</response>", raw_llm_output, re.DOTALL)

        if thinking_match:
            llm_thinking = thinking_match.group(1).strip()
            # Add LLM's thinking steps to our display
            thinking_steps.append("--- Model's Own Reasoning Steps ---")
            thinking_steps.append(llm_thinking)
        else:
            # If the model didn't follow the exact format, show the raw output or a note
            thinking_steps.append("--- Model Output (Format Parsing Failed) ---")
            # Limit the length of raw output shown
            thinking_steps.append(raw_llm_output[:1000] + "..." if len(raw_llm_output) > 1000 else raw_llm_output)

        if response_match:
            final_response = response_match.group(1).strip()
            response_parts.append(final_response)
        else:
            # Handle case where <response> tag is missing
            response_parts.append("I couldn't generate a clear final answer based on the model's output.")
            response_parts.append("\n--- Raw Model Output ---\n")
            response_parts.append(raw_llm_output[:1000] + "..." if len(raw_llm_output) > 1000 else raw_llm_output)

    except Exception as e:
        # Catch any errors during the LLM call or output processing
        thinking_steps.append(f"❌ **Thinking:** Error calling the Llama 3 model or processing its output: {e}")
        response_parts.append(f"Sorry, I encountered an error while trying to think about your query with the local Llama 3 model: {e}")

    return "\n".join(thinking_steps), "\n".join(response_parts), chart # Return chart as None for now


# --- Sidebar: Upload Dataset ---
with st.sidebar:
    st.header("📁 Upload Your Dataset")
    st.markdown("Developed by: [DM Shahriar Hossain](https://github.com/rownokstar/)")
    uploaded_file = st.file_uploader("Upload CSV (e.g., nutrition_table.csv)", type=["csv"])

    # --- Optional: Model Selection (Advanced) ---
    # if TRANSFORMERS_AVAILABLE:
    #     st.subheader("🤖 LLM Settings (Advanced)")
    #     model_options = {
    #         "Llama 3 8B (Quantized - Recommended)": "unsloth/llama-3-8b-Instruct-bnb-4bit",
    #         "Llama 3 8B (Full - Needs HF Token & Resources)": "meta-llama/Meta-Llama-3-8B-Instruct",
    #         # Add other models as needed
    #     }
    #     selected_model = st.selectbox("Select Model", list(model_options.keys()), index=0)
    #     selected_model_name = model_options[selected_model]
    #     # Store selected model in session state if needed for dynamic loading

    if uploaded_file is not None:
        with st.spinner("📥 Loading dataset..."):
            try:
                # Load dataset - Crucially, specify header=None as the first row is data in nutrition_table.csv
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state.original_df = df.copy()
                # Identify columns based on the known structure of nutrition_table.csv
                st.session_state.column_mapping = identify_nutrition_columns(df)

                if len(df) > 1000:
                    df_display = df.sample(min(5, len(df)), random_state=42)
                    df_process = df.sample(min(2000, len(df)), random_state=42)
                    st.warning(f"📊 Large dataset detected ({len(df)} rows). Processing a sample for optimal performance.")
                else:
                    df_display = df.head(5)
                    df_process = df

                st.session_state.df = df_display
                detected_type = detect_dataset_type(df_process)
                st.session_state.data_type = detected_type

                st.success(f"✅ Dataset loaded! Detected as: **{detected_type.upper()}**")

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
                st.session_state.index = None
                st.session_state.sentences = None
                st.session_state.processed = False

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
                # Display the "Thinking Process" in an expander
                with st.expander("🧠 My Thinking Process", expanded=False):
                    st.markdown(thinking_content)
            # Display the main response
            st.markdown(response_content)
        else:
            # Fallback for older messages or errors
            st.markdown(msg["content"])

        # Display chart if it exists in the message
        if "chart" in msg and msg["chart"] is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about foods, nutrients, or diets... (e.g., 'Top 5 high protein foods')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        # --- Call the NEW Llama 3-based function ---
        thinking_process, full_response, chart = generate_response_with_llama3(
            prompt,
            st.session_state.original_df,
            st.session_state.column_mapping,
            st.session_state.index,
            st.session_state.sentences
        )

        # --- Streaming Display for Thinking ---
        if thinking_process:
            with st.expander("🧠 My Thinking Process", expanded=True): # Start expanded for visibility
                thinking_placeholder = st.empty()
                displayed_thinking = ""
                thinking_lines = thinking_process.splitlines()
                for i, line in enumerate(thinking_lines):
                    displayed_thinking += line + "\n"
                    thinking_placeholder.markdown(displayed_thinking)
                    # Simulate processing delay for thinking steps for better readability
                    # Reduced frequency of delay to make it feel smoother but still slow
                    # Avoid delaying final outcome messages (marked with emojis)
                    if i % 2 == 0 and not any(marker in line for marker in ["✅", "❌", " HttpNotFound", "📉", "💡", "🔍", "🔢", "🔤", "---"]):
                         time.sleep(0.2) # Slower delay for thinking steps

        # --- Streaming Display for Response (SLOWED DOWN) ---
        if full_response:
            message_placeholder = st.empty()
            displayed_text = ""
            words = full_response.split(" ")
            # --- ADJUSTED STREAMING SPEED ---
            # New delay is 0.05 seconds per word. Adjust the 0.05 value to make it faster/slower.
            word_delay = 0.05 # Slower speed for eye comfort

            for i, word in enumerate(words):
                displayed_text += word + " "
                message_placeholder.markdown(displayed_text + "▌") # Cursor effect
                # --- SLOWED DOWN DELAY ---
                time.sleep(word_delay) # Increased delay for a more eye-soothing pace
            message_placeholder.markdown(displayed_text) # Final display without cursor
        else:
            # Handle case where no response was generated
            st.markdown("Sorry, I couldn't generate a response for that query.")

        # Display chart if generated (Note: Chart generation logic not integrated with Llama 3 yet)
        # You would need to parse the LLM response to identify if a chart is requested and what data to use.
        # For now, we keep the chart variable but it won't be populated by the Llama 3 function.
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)

        # Store complete message in history (store thinking and response separately)
        st.session_state.messages.append({
            "role": "assistant",
            "content": (thinking_process, full_response), # Store as tuple
            "chart": chart # Store chart
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
