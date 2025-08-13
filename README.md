# 🍏 Nutrition Genius – ফ্রি নিউট্রিশন অ্যাসিসট্যান্ট (RAG পাওয়ার্ড)

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Nutrition Genius** একটি ফ্রি, ওপেন-সোর্স ওয়েব অ্যাপ যা কোনো পেইড API ছাড়াই খাদ্য, পুষ্টি, ডায়েট প্ল্যান এবং বিকল্প খাবার সম্পর্কে তথ্য দেয়। এটি **RAG (Retrieval-Augmented Generation)** মডেল ব্যবহার করে এবং সম্পূর্ণ ফ্রি টুলস দিয়ে তৈরি।

> ✅ কোনো OpenAI API কী লাগে না  
> ✅ গুগল কল্যাব ও Streamlit Cloud-এ চালানো যায়  
> ✅ বাংলা ও ইংরেজি প্রশ্ন সাপোর্ট  
> ✅ পুষ্টি চার্ট, ডায়েট প্ল্যান, বিকল্প খাবার সাজেস্ট

🔗 **লাইভ ডেমো**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(তোমার লিঙ্কটি দাও)*

---

## 🎯 ফিচারসমূহ

- **প্রশ্ন-উত্তর**: "পালং শাকে কত প্রোটিন?"
- **ডায়েট প্ল্যান**: "২০০০ ক্যালোরির ভেজ ডায়েট প্ল্যান বানাও"
- **খাবার বিকল্প**: "বাদাম এলার্জি আছে, বিকল্প কী?"
- **পুষ্টি চার্ট**: "চিকেন ব্রেস্টের পুষ্টি গুণ দেখাও" → পাই চার্ট
- **ফ্রি এবং লোকাল**: কোনো পেইড এপিআই নেই

---

## 🛠️ টেক স্ট্যাক

| টুল | ব্যবহার |
|------|--------|
| **Python** | ব্যাকএন্ড লজিক |
| **Streamlit** | ইউজার ইন্টারফেস |
| **FAISS** | ভেক্টর সিমিলারিটি সার্চ |
| **Sentence Transformers** | ফ্রি এম্বেডিং (`all-MiniLM-L6-v2`) |
| **Pandas & NumPy** | ডেটা হ্যান্ডলিং |
| **Plotly** | ইন্টারঅ্যাকটিভ চার্ট |
| **Tabulate** | মার্কডাউন টেবিল জেনারেশন |

---

## 📦 কীভাবে রান করবে?

### 1. লোকালি বা Google Colab-এ

```bash
# ডিপেন্ডেন্সি ইনস্টল
pip install -r requirements.txt

# অ্যাপ চালাও
streamlit run app.py
