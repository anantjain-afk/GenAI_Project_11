import streamlit as st
import joblib
import numpy as np
import re
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Intelligent News Credibility Analyzer",
    page_icon="📰",
    layout="centered"
)

# extract text from url
def extract_text(url):
    """Fetch and extract article text from a URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    return text.strip()




# loading the model and vectorizer . 
@st.cache_resource
def load_model():
    model = joblib.load("ml/model.pkl")
    vectorizer = joblib.load("ml/vectorizer.pkl")
    return model, vectorizer

model , vectorizer = load_model()


# making the ui of the app . 
st.title("📰 Intelligent News Credibility Analyzer")
st.caption("ML-based News Credibility Analysis (No LLMs)")

st.markdown(
    """
This system analyzes **news articles** using **classical NLP & Machine Learning**
to assess **credibility risk** based on textual patterns.
"""
)

# taking input from the user .
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

# anahlyzing the credibility of the article .
if st.button("Analyze Credibility"):
    if not article_text.strip():
        st.warning("Please provide article text or URL.")
    else:
        with st.spinner("Analyzing credibility..."):
            
            transformed = vectorizer.transform([article_text])
            prediction = model.predict(transformed)[0]
            probabilities = model.predict_proba(transformed)[0]

            confidence = np.max(probabilities)

        st.divider()

        # Result 
        if prediction == 1:
            st.success("✅ High Credibility Detected")
        else:
            st.error("⚠️ Low Credibility Detected")

        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}"
        )

        # Explanation 
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