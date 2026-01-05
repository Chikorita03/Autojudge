from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.classification.logistic import (
    train_logistic_classifier,
    evaluate_logistic_classifier
)
from models.regression.linear import (
    train_linear_regressor,
    evaluate_linear_regressor
)

# 1. Load + preprocess data
df = preprocess_data("problems_dataset.jsonl")

# 2. Build features
X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

# 3. Targets (classification)
y_train = df.loc[idx_train, "problem_class"]
y_test = df.loc[idx_test, "problem_class"]

# 4. Train model
classifier = train_logistic_classifier(X_train, y_train)

# 5. Evaluate
results = evaluate_logistic_classifier(classifier, X_test, y_test)

print("Accuracy:", results["accuracy"])
print("Confusion Matrix:\n", results["confusion_matrix"])
print("Classification Report:\n", results["classification_report"])

###MODEL 2 - REGRESSION
y_train_score = df.loc[idx_train, "problem_score"]
y_test_score = df.loc[idx_test, "problem_score"]

# Train
regressor = train_linear_regressor(X_train, y_train_score)

# Evaluate
reg_results = evaluate_linear_regressor(
    regressor,
    X_test,
    y_test_score,
    y_train_mean=y_train_score.mean()
)

print("MAE:", reg_results["mae"])
print("RMSE:", reg_results["rmse"])
print("Baseline MAE:", reg_results["baseline_mae"])
print("Baseline RMSE:", reg_results["baseline_rmse"])

import pickle
import os

os.makedirs("models_saved", exist_ok=True)

with open("models_saved/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models_saved/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models_saved/classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("models_saved/regressor.pkl", "wb") as f:
    pickle.dump(regressor, f) 