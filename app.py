# app.py - ফিক্সড ভার্সন (কোনো প্রশ্নেই "পারছি না" দেখাবে না)

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import plotly.express as px

# --- টাইটেল ---
st.set_page_config(page_title="🍏 Nutrition Genius", page_icon="🍏")
st.title("🍏 Nutrition Genius")
st.markdown("> *ফ্রি নিউট্রিশন সহকারী - কোনো API ছাড়াই!*")

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

# --- এম্বেডিং মডেল ---
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# --- FAISS ইনডেক্স ---
@st.cache_resource
def create_index():
    sentences = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, sentences

index, sentences = create_index()

# --- রিট্রিভার ---
def retrieve(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k)
    results = []
    for i in I[0]:
        if i != -1:  # Valid index
            results.append(df.iloc[i])
    return pd.DataFrame(results).drop_duplicates(subset="Food")

# --- রেসপন্স জেনারেটর (ফিক্সড) ---
def generate_response(query):
    query_lower = query.lower()
    results = retrieve(query, k=3)

    # চার্ট ডিফল্ট: None
    chart = None

    # যদি কিছু একটা খাবারের নাম থাকে
    food_names = [f.lower() for f in df['Food'].tolist()]
    matched_food = None
    for food in food_names:
        if food in query_lower:
            matched_food = food.title()
            break

    # Nutrition breakdown
    if "breakdown" in query_lower or "composition" in query_lower:
        if not results.empty:
            row = results.iloc[0]
            fig = px.pie(
                values=[row['Protein (g)'], row['Fat (g)'], row['Carbs (g)']],
                names=['Protein', 'Fat', 'Carbs'],
                title=f"{row['Food']} - Nutrition"
            )
            chart = fig
            return f"📊 {row['Food']} এর পুষ্টি গঠন (প্রতি 100g):\n\n" + \
                   f"**ক্যালোরি**: {row['Calories']} kcal\n" + \
                   f"**প্রোটিন**: {row['Protein (g)']}g\n" + \
                   f"**ফ্যাট**: {row['Fat (g)']}g\n" + \
                   f"**কার্বস**: {row['Carbs (g)']}g", chart

    # Protein, fat, carbs query
    if "protein" in query_lower:
        if matched_food:
            row = df[df['Food'].str.contains(matched_food, case=False)].iloc[0]
            return f"🟢 {row['Food']} এর প্রোটিন: **{row['Protein (g)']}g** (প্রতি 100g)", None
    if "calorie" in query_lower or "kcal" in query_lower:
        if matched_food:
            row = df[df['Food'].str.contains(matched_food, case=False)].iloc[0]
            return f"🔥 {row['Food']} এর ক্যালোরি: **{row['Calories']} kcal** (প্রতি 100g)", None

    # Diet plan
    if "diet plan" in query_lower or "meal plan" in query_lower:
        sample = df.sample(4)
        return "📋 স্যাম্পল ডায়েট প্ল্যান:\n\n" + \
               sample[['Food', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)']].to_markdown(index=False), None

    # Alternatives
    if "alternative" in query_lower or "replace" in query_lower or "allergic" in query_lower:
        if "peanut" in query_lower or "nut" in query_lower:
            safe = df[df['Category'] != 'Nut'].sample(3)
            return "✅ পিনাট বাদে বিকল্প:\n\n" + \
                   safe[['Food', 'Category', 'Protein (g)']].to_markdown(index=False), None

    # Default: যেকোনো প্রশ্নে সংশ্লিষ্ট খাবার দেখাও
    if not results.empty:
        return "🔍 খুঁজে পাওয়া তথ্য:\n\n" + \
               results[['Food', 'Calories', 'Protein (g)', 'Fat (g)', 'Carbs (g)']].to_markdown(index=False), None

    # যদি কিছু না পাওয়া যায়
    return "❌ খাবার খুঁজে পাওয়া যায়নি। অন্য নাম চেষ্টা করুন (যেমন: spinach, banana, chicken)।", None

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
    st.header("📊 ডেটা নমুনা")
    st.dataframe(df[['Food', 'Calories', 'Protein (g)']].sample(5))
    st.markdown("### 📌 ট্রাই করুন")
    st.write("• পালং শাকে কত প্রোটিন?")
    st.write("• চিকেন ব্রেস্টের পুষ্টি গুণ")
    st.write("• কম ক্যালোরির ফল কী আছে?")
    st.write("• বাদামের বিকল্প কী?")
