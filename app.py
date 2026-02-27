import streamlit as st
import pickle
import re
import string

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# App UI
st.title("Cyberbullying Detection App")
st.write("This app detects whether a comment contains cyberbullying content.")

user_input = st.text_area("Enter a comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)

        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.error(f"🚨 Cyberbullying Detected ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ Not Cyberbullying ({confidence:.2f}% confidence)")