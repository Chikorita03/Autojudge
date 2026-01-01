import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error

df = pd.read_json("problems_dataset.jsonl" , lines = True)
#print(df.isnull().sum())
df["combined_text"] = ( df["title"] + " " + df["description"] + " " + df["input_description"]
                        + " " + df["output_description"])
df["combined_text"] = (df["combined_text"].str.lower().str.replace("\n", " ", regex=False).str.replace("\t", " ", regex=False))
df["combined_text"] = df["combined_text"].str.replace(r"\s+", " ", regex=True)

##Feature Engineering
df["text_length"] = df["combined_text"].str.len()
#print(df["text_length"])
math_symbols = r"[+\-*/=<>\^%]"
df["symbol_count"] = df["combined_text"].str.count(math_symbols)
#print(df["symbol_count"])
keyword = ["graph", "dp", "dynamic programming", "tree", "recursion"]
for kw in keyword:
    pattern = r"\b" + kw + r"\b"
    col_name = f"freq_{kw.replace(' ', '_')}"
    df[col_name] = df["combined_text"].str.count(pattern)

tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X_tfidf = tfidf.fit_transform(df["combined_text"])
numeric_features = df[["text_length", "symbol_count", "freq_graph", "freq_dp", "freq_dynamic_programming", "freq_tree",
                        "freq_recursion"]].values

###MODEL 1 - CLASSIFICATION

y_class = df["problem_class"]
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, y_class, test_size=0.2, random_state=42, stratify=y_class)

num_train, num_test = train_test_split(
    numeric_features, test_size=0.2, random_state=42, stratify=y_class)

#Scale numeric features 
scaler = StandardScaler()
num_train_scaled = scaler.fit_transform(num_train)
num_test_scaled = scaler.transform(num_test)

X_train = hstack([X_train_tfidf, num_train_scaled])
X_test = hstack([X_test_tfidf, num_test_scaled])

classifier = LogisticRegression(
    max_iter=3000,
    n_jobs=-1
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred[:10])

##model accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred, labels=["easy", "medium", "hard"])
#print(cm)

###MODEL 2 - REGRESSION

y_reg = df["problem_score"]

X_train_tfidf, X_test_tfidf, y_train_reg, y_test_reg = train_test_split(
    X_tfidf, y_reg, test_size=0.2, random_state=42)

num_train, num_test = train_test_split(
    numeric_features, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
num_train_scaled = scaler_reg.fit_transform(num_train)
num_test_scaled = scaler_reg.transform(num_test)

X_train_reg = hstack([X_train_tfidf, num_train_scaled])
X_test_reg = hstack([X_test_tfidf, num_test_scaled])

regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = regressor.predict(X_test_reg)
print(y_pred_reg[:10])

mae = mean_absolute_error(y_test_reg, y_pred_reg)
print("MAE:", mae)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print("RMSE:", rmse)

baseline_pred = np.full_like(y_test_reg, y_train_reg.mean())
baseline_mae = mean_absolute_error(y_test_reg, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test_reg, baseline_pred))

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

import pickle
import os

os.makedirs("models", exist_ok=True)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("models/classification_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/classification_model.pkl", "wb") as f:
    pickle.dump(classifier, f)
with open("models/regression_scaler.pkl", "wb") as f:
    pickle.dump(scaler_reg, f)
with open("models/regression_model.pkl", "wb") as f:
    pickle.dump(regressor, f) 
