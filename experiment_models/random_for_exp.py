from data_preprocessing import preprocess_data
from feature_engineering import build_features

from models.classification.random_forest import (
    train_rf_classifier,
    evaluate_rf_classifier
)

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

y_train = df.loc[idx_train, "problem_class"]
y_test = df.loc[idx_test, "problem_class"]

rf_model = train_rf_classifier(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None
)

rf_results = evaluate_rf_classifier(rf_model, X_test, y_test)

print("Accuracy:", rf_results["accuracy"])
print("Confusion Matrix:\n", rf_results["confusion_matrix"])
print("Classification Report:\n", rf_results["classification_report"])