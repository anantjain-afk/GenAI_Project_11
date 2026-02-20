import streamlit as st
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intelligent News Credibility Analyzer",
    page_icon="📰",
    layout="centered"
)

# ---------------- HELPER: EXTRACT TEXT FROM URL ----------------
def extract_text(url):
    """Fetch and extract article text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    return text.strip()

# ---------------- HELPER: CLEAN TEXT ----------------
def clean_text(text):
    """Clean input text to match training preprocessing."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("ml/model.pkl")

model = load_model()

# ---------------- UI ----------------
st.title("📰 Intelligent News Credibility Analyzer")
st.caption("ML-based News Credibility Analysis (No LLMs)")

st.markdown(
    """
This system analyzes **news articles** using **classical NLP & Machine Learning**
to assess **credibility risk** based on textual patterns.
"""
)

# ---------------- INPUT ----------------
input_type = st.radio(
    "Choose input method:",
    ["Paste Article Text", "Enter Article URL"]
)

article_text = ""

if input_type == "Paste Article Text":
    article_text = st.text_area(
        "Paste news article text here",
        height=250,
        placeholder="Paste the full article content..."
    )

else:
    url = st.text_input("Enter news article URL")
    if st.button("Fetch Article"):
        if url.strip():
            with st.spinner("Extracting article..."):
                try:
                    article_text = extract_text(url)
                    st.success("Article extracted successfully!")
                    st.text_area("Extracted Article", article_text, height=250)
                except Exception as e:
                    st.error("Failed to extract article")

# ---------------- ANALYSIS ----------------
if st.button("Analyze Credibility"):
    if not article_text.strip():
        st.warning("Please provide article text or URL.")
    else:
        with st.spinner("Analyzing credibility..."):
            cleaned = clean_text(article_text)
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            confidence = np.max(probabilities)

        st.divider()

        # -------- RESULT --------
        if prediction == 1:
            st.error("⚠️ Low Credibility Detected")
        else:
            st.success("✅ High Credibility Detected")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}"
        )

        # -------- EXPLANATION --------
        with st.expander("📊 How was this decision made?"):
            st.write(
                """
- Text was cleaned and vectorized using **TF-IDF**
- Model used: **Logistic Regression**
- Decision based on learned linguistic & credibility patterns
- No external APIs or LLMs were used
                """
            )

        # -------- DISCLAIMER --------
        with st.expander("⚖️ Ethical Disclaimer"):
            st.write(
                """
This tool provides **probabilistic analysis**, not absolute truth.
It should be used as a **decision-support system**, not a final authority.
                """
            )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Academic Project | NLP & ML | Deployed via Streamlit")