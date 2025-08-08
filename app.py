import streamlit as st
import pickle
import os
import numpy as np

MODEL_PATH = os.path.join('app_files', 'model.pkl')
VECTORIZER_PATH = os.path.join('app_files','vectorizer.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

st.header('Email Spam Classifier System')
st.write("Input email text and receive a real-time Classification (spam or not spam)!")

email_text = st.text_area("Enter the email text here:")

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        try:
            X = vectorizer.transform([email_text])
            prediction = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X)) * 100
            label = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("---")
st.subheader("How does it work?")
st.info("""
**Logistic Regression** is a statistical model used for binary classification.  
It estimates the probability that an input belongs to a particular category (e.g., spam or not spam).  
**Spam filtering** uses machine learning to detect unwanted emails by analyzing their content and patterns.
""")