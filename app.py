import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("data.csv")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# Train model automatically
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# App UI
st.title("Cyberbullying Detection App")
st.write("This app detects cyberbullying using TF-IDF + Logistic Regression.")

user_input = st.text_area("Enter a comment:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized)
    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        st.error(f"🚨 Cyberbullying Detected ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ Not Cyberbullying ({confidence:.2f}% confidence)")