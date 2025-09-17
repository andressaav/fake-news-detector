# app.py
import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("models/tfidf_logreg.joblib")

st.title("📰 Fake News Detector")
st.write("Paste a full news **article** and click **Predict**.")

model = load_model()
text = st.text_area("Input text", height=180, placeholder="Type or paste a headline/article here...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([text])[0]
        proba = model.predict_proba([text])[0].max()
        st.markdown(f"### Prediction: **{pred}**")
        st.markdown(f"Confidence: **{proba:.2f}**")

st.caption("Model: TF-IDF + Logistic Regression")

