import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack

from data_preprocessing import preprocess_data
from feature_engineering import build_features_for_inference

# Load models
tfidf = joblib.load("models_saved/tfidf.pkl")
scaler = joblib.load("models_saved/scaler.pkl")
classifier = joblib.load("models_saved/logistic_classifier.pkl")
gb_regressor = joblib.load("models_saved/gb_regressor.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge")
st.write("Predict programming problem difficulty")

title = st.text_input("Problem Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict"):
    if not all([title, description, input_desc, output_desc]):
        st.warning("Please fill all fields")
    else:
        df = pd.DataFrame([{
            "title": title,
            "description": description,
            "input_description": input_desc,
            "output_description": output_desc
        }])

        # df = preprocess_data(df)
        df["combined_text"] = (
            df["title"] + " " +
            df["description"] + " " +
            df["input_description"] + " " +
            df["output_description"]
        )

        df["combined_text"] = (
            df["combined_text"]
            .str.lower()
            .str.replace("\n", " ", regex=False)
            .str.replace("\t", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )

        X_text, X_num = build_features_for_inference(df, tfidf, scaler)
        X = hstack([X_text, X_num])

        class_pred = classifier.predict(X)[0]
        score_pred = gb_regressor.predict(X)[0]

        st.success("Prediction Complete ✅")
        st.metric("Difficulty Class", class_pred.capitalize())
        st.metric("Difficulty Score", round(score_pred, 2))