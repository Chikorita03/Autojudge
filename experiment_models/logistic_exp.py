from data_preprocessing import preprocess_data
from feature_engineering import build_features
from models.classification.logistic import (
    train_logistic_classifier,
    evaluate_logistic_classifier
)

df = preprocess_data("problems_dataset.jsonl")

X_train, X_test, idx_train, idx_test, tfidf, scaler = build_features(df)

y_train = df.loc[idx_train, "problem_class"]
y_test = df.loc[idx_test, "problem_class"]

classifier = train_logistic_classifier(X_train, y_train)

results = evaluate_logistic_classifier(classifier, X_test, y_test)

print("Accuracy:", results["accuracy"])
print("Confusion Matrix:\n", results["confusion_matrix"])
print("Classification Report:\n", results["classification_report"])