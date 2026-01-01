import streamlit as st
import numpy as np
import pickle
from scipy.sparse import hstack

tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
clf_scaler = pickle.load(open("models/classification_scaler.pkl", "rb"))
classifier = pickle.load(open("models/classification_model.pkl", "rb"))
reg_scaler = pickle.load(open("models/regression_scaler.pkl", "rb"))
regressor = pickle.load(open("models/regression_model.pkl", "rb"))

st.set_page_config(page_title="AutoJudge", layout="centered")
st.title(" AutoJudge")
st.subheader("Predict Programming Problem Difficulty")
st.write("Paste the problem details below:")
title = st.text_area("Problem Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

#Feature Engineering
def extract_features(text):
    text = text.lower()
    text_length = len(text)
    symbol_count = sum(text.count(s) for s in "+-*/=<>&^%")

    keywords = ["graph", "dp", "dynamic programming", "tree", "recursion"]
    keyword_counts = [text.count(k) for k in keywords]

    numeric_features = np.array(
        [text_length, symbol_count] + keyword_counts
    ).reshape(1, -1)

    return numeric_features

if st.button("Predict Difficulty"):
    combined_text = f"{title} {description} {input_desc} {output_desc}"

    if combined_text.strip() == "":
        st.warning("Please enter problem details.")
    else:
       
        text_tfidf = tfidf.transform([combined_text])
        numeric = extract_features(combined_text)

        num_scaled_clf = clf_scaler.transform(numeric)
        X_clf = hstack([text_tfidf, num_scaled_clf])
        class_pred = classifier.predict(X_clf)[0]

        num_scaled_reg = reg_scaler.transform(numeric)
        X_reg = hstack([text_tfidf, num_scaled_reg])
        score_pred = regressor.predict(X_reg)[0]
        st.success("Prediction Complete")

        st.write("Predicted Difficulty Class")
        st.write(f"**{class_pred.upper()}**")

        st.write("Predicted Difficulty Score")
        st.write(f"**{round(score_pred, 2)}**")