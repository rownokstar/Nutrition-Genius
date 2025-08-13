# app.py - ফ্রি Nutrition Genius (No API Required)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px
import os

# --- লোগো এবং টাইটেল ---
st.set_page_config(page_title="🍏 Nutrition Genius", page_icon="🍏")
st.title("🍏 Nutrition Genius")
st.markdown("> *ফ্রি নিউট্রিশন সহকারী - কোনো API কী ছাড়াই!*")

# --- ডেটাসেট ---
@st.cache_data
def load_data():
    data = {
        "Food": ["Spinach", "Chicken Breast", "Brown Rice", "Peanut Butter", "Salmon", "Banana", "Quinoa", "Eggs", "Almonds", "Oats", "Tofu", "Lentils"],
        "Calories": [23, 165, 111, 588, 208, 89, 120, 155, 579, 389, 70, 116],
        "Protein (g)": [2.9, 31, 2.6, 25, 20, 1.1, 4.4, 13, 21, 16.9, 8, 9],
        "Fat (g)": [0.4, 3.6, 0.9, 50, 13, 0.3, 1.9, 11, 49, 6.9, 4, 0.4],
        "Carbs (g)": [3.6, 0, 23, 20, 0, 23, 21, 1.1, 22, 66, 3, 66],
        "Fiber (g)": [2.2, 0, 1.8, 6, 0, 2.6, 2.8, 0, 12.5, 10.6, 2, 7.9],
        "Vitamins": ["A, C", "B6, B12", "B1", "E, Niacin", "D, B12", "B6, C", "B, E", "B12, D", "E, B2", "B1", "B, C", "Folate"],
        "Minerals": ["Iron, Calcium", "Selenium", "Magnesium", "Magnesium", "Selenium", "Potassium", "Iron", "Selenium", "Magnesium", "Iron", "Calcium", "Iron"],
        "Category": ["Vegetable", "Meat", "Grain", "Nut", "Fish", "Fruit", "Grain", "Egg", "Nut", "Grain", "Soy", "Legume"]
    }
    return pd.DataFrame(data)

df = load_data()

# --- এম্বেডিং মডেল (ফ্রি) ---
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- FAISS ইনডেক্স তৈরি ---
@st.cache_resource
def create_index():
    sentences = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, sentences

index, sentences = create_index()

# --- রিট্রিভার ---
def retrieve(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k)
    return [sentences[i] for i in I[0]]

# --- কোয়েরি পার্সিং এবং রেসপন্স জেনারেট ---
def generate_response(query):
    query_lower = query.lower()
    results = retrieve(query)

    # Q&A
    if "protein" in query_lower and "in" in query_lower:
        food = query_lower.split("in")[-1].strip()
        match = df[df['Food'].str.contains(food, case=False, na=False)]
        if not match.empty:
            val = match.iloc[0]['Protein (g)']
            return f"100g {match.iloc[0]['Food']} contains {val}g of protein.", None

    # Diet Plan
    elif "diet plan" in query_lower or "meal plan" in query_lower:
        foods = df.sample(4)  # র‍্যান্ডম স্যাম্পল
        total_cal = foods['Calories'].sum()
        return f"📋 4-item sample diet plan (Total: {total_cal} kcal):\n\n" + \
               foods[['Food', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)']].to_markdown(index=False), None

    # Alternatives
    elif "allergic" in query_lower or "avoid" in query_lower or "alternative" in query_lower:
        if "peanut" in query_lower:
            alternatives = df[df['Food'] != 'Peanut Butter']
            nuts = alternatives[alternatives['Category'] == 'Nut']
            return "🥜 Safe alternatives to peanuts:\n\n" + \
                   nuts[['Food', 'Protein (g)', 'Fat (g)']].head(3).to_markdown(index=False), None

    # Nutrition Breakdown
    elif "breakdown" in query_lower:
        food = query_lower.replace("nutrition breakdown of", "").strip().title()
        row = df[df['Food'].str.contains(food, case=False)]
        if not row.empty:
            row = row.iloc[0]
            fig = px.pie(
                values=[row['Protein (g)'], row['Fat (g)'], row['Carbs (g)']],
                names=['Protein', 'Fat', 'Carbs'],
                title=f"{row['Food']} Nutrition"
            )
            return f"📊 Nutrition breakdown for {row['Food']} (per 100g):", fig
        else:
            return "খাবার খুঁজে পাওয়া যায়নি।", None

    # Default
    return "আমি এই প্রশ্নটির উত্তর দিতে পারছি না। অনুগ্রহ করে আরও বিস্তারিত জিজ্ঞাসা করুন।", None

# --- চ্যাট UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg and msg["chart"]:
            st.plotly_chart(msg["chart"], use_container_width=True)

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
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

# --- সাইডবার ---
with st.sidebar:
    st.header("📊 ডেটাসেট")
    st.dataframe(df[['Food', 'Calories', 'Protein (g)', 'Category']].sample(5))
    st.caption("ফ্রি ডেমো ডেটা - Nutrition_DB.csv")
    st.markdown("---")
    st.markdown("💡 ট্রাই করুন:\n- `পালং শাকে কত প্রোটিন?`\n- `চিকেন ব্রেস্টের পুষ্টি গুণ`\n- `ডায়েট প্ল্যান বানাও`\n- `মুগ ডালের বিকল্প কী?`")
