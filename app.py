import streamlit as st
import joblib
import re

# Load components
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("rf.pkl")
label_mapping = joblib.load("label_mapping.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d+", '', text)
    return text.strip()

st.title("🧠 Mental Health Classifier (ML)")

input_text = st.text_area("Enter a mental health-related statement:")

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.markdown(f"### 🧾 Predicted Label: `{label_mapping[prediction]}`")
