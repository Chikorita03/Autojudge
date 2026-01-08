from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.classification.logistic import (
    train_logistic_classifier,
    evaluate_logistic_classifier
)
from models.regression.gradient_boosting import (
    train_gb_regressor,
    evaluate_gb_regressor
)

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

y_train = df.loc[idx_train, "problem_class"]
y_test = df.loc[idx_test, "problem_class"]

classifier = train_logistic_classifier(X_train, y_train)

results = evaluate_logistic_classifier(classifier, X_test, y_test)

print("Accuracy:", results["accuracy"])
print("Confusion Matrix:\n", results["confusion_matrix"])
# print("Classification Report:\n", results["classification_report"])

y_train_score = df.loc[idx_train, "problem_score"]
y_test_score = df.loc[idx_test, "problem_score"]

gb_model = train_gb_regressor(
    X_train,
    y_train_score,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3
)

gb_results = evaluate_gb_regressor(gb_model, X_test, y_test_score)

print("MAE:", gb_results["mae"])
print("RMSE:", gb_results["rmse"])

import joblib
import os

os.makedirs("models_saved", exist_ok=True)

joblib.dump(tfidf, "models_saved/tfidf.pkl")
joblib.dump(scaler, "models_saved/scaler.pkl")
joblib.dump(classifier, "models_saved/logistic_classifier.pkl")
joblib.dump(gb_model, "models_saved/gb_regressor.pkl")
