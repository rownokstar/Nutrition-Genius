# 🍏 Nutrition Genius – Free AI Nutrition Assistant (RAG-Powered)

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Nutrition Genius** is a free, open-source web app that answers nutrition questions, generates diet plans, suggests food alternatives, and visualizes nutrient breakdown — **without any paid API**. It runs on a **Retrieval-Augmented Generation (RAG)** system using entirely free and local tools.

> ✅ No OpenAI or paid API required  
> ✅ Runs on Google Colab & Streamlit Cloud  
> ✅ Supports English & Bengali queries  
> ✅ Shows charts, tables, and smart responses

🔗 **Live Demo**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(replace with your link)*

---

## 🎯 Features

- **Q&A**: "How much protein is in spinach?"
- **Diet Plans**: "Create a 2000-calorie vegetarian meal plan"
- **Food Alternatives**: "I'm allergic to peanuts, suggest alternatives"
- **Nutrition Charts**: "Show nutrition breakdown of chicken breast" → Pie chart
- **100% Free**: No API keys, no cost, no cloud dependency

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|--------|
| **Python** | Backend logic |
| **Streamlit** | User interface |
| **FAISS** | Fast similarity search |
| **Sentence Transformers** | Free embeddings (`all-MiniLM-L6-v2`) |
| **Pandas & NumPy** | Data handling |
| **Plotly** | Interactive visualizations |
| **Tabulate** | Markdown table generation |

---

## 📦 How to Run

### 1. Locally or on Google Colab

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
